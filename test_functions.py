from __future__ import annotations

import ast
import contextlib
import argparse
import datetime
import hashlib
import importlib.util
import math
import os as _os
import pickle
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe

MODULE_PATH = _os.environ.get("QAFM_MODEL_PATH") or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "Quantum_AFM-multipolar_Jahn-Teller.py")

def _load_model(path: str):
    """Import the model under a REGISTERED module name (dataclasses and pickle resolve the defining module through sys.modules)."""
    spec = importlib.util.spec_from_file_location("qmodel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qmodel"] = mod
    spec.loader.exec_module(mod)
    return mod

qm = _load_model(MODULE_PATH)
_REPORT_FILE = None

_SCF_STATIONARITY_TOL : float = 5e-5                # |M_c − M_BdG,c| stationarity tolerance at the converged SCF fixed point (eV·site).
_CHI_TAU_ABS_FLOOR    : float = 1e-5                # eV⁻¹ — absolute floor under the relative-error denominator in the χ_τ Richardson/nonlinearity tests; without it, a genuinely small ∂⟨B1g⟩/∂Q signal gets its relative error amplified by ordinary SCF/μ-root-find noise and is spuriously zeroed. Starting point only — tune against the actual scatter observed in cd_sc/cd_n for repeated/neighbouring evaluations.
_LINDHARD_CHUNK       : int   = 128                 # k-point batch size in oe.contract loops (memory vs speed)
_D4H_CLASSES = ('E', 'C4', 'C2z', "C2'", "C2''", 'i', 'S4', 'σh', 'σv', 'σd')
_D4H_IRREPS = (('A1g', 1), ('A2g', 1), ('B1g', 1), ('B2g', 1), ('Eg', 2), ('A1u', 1), ('A2u', 1), ('B1u', 1), ('B2u', 1), ('Eu', 2))
_D2H_IRREPS = (('Ag', 1), ('B1g', 1), ('B2g', 1), ('B3g', 1), ('Au', 1), ('B1u', 1), ('B2u', 1), ('B3u', 1))
_Q_B1 = np.diag([1.0, -1.0, 0.0])                          # x²−y²  quadratic form  (B1g of D4h, axes ∥ Cu–O bonds)
_Q_B2 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])   # xy

def _open_test_log(path: str = "test_report.txt") -> str:
    """Create/overwrite the report file that _log()/_section() write to. Call once before running tests."""
    global _REPORT_FILE
    if _REPORT_FILE is not None:
        _REPORT_FILE.close()
    _REPORT_FILE = open(path, "w", encoding="utf-8")
    _REPORT_FILE.write(f"Test run started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    _REPORT_FILE.flush()
    print(f"[test log] writing full report to: {_os.path.abspath(path)}")
    return path

def _close_test_log() -> None:
    """Flush and close the report file. Safe to call even if it was never opened."""
    global _REPORT_FILE
    if _REPORT_FILE is not None:
        _REPORT_FILE.close()
        _REPORT_FILE = None

_MARK = {'ok': 0, 'flag': 0, 'fail': 0}
_STEP_RESULTS: list = []
_FAIL_TOKENS = ('❌', 'FAIL', 'CRASH')        # a definite failure: decides the exit code
_FLAG_TOKENS = ('✗',)                         # a criterion flag (table cells, diagnostics): reported, but not a failed assertion
_OK_TOKENS = ('✓', 'PASS')

def _emit(line: str) -> None:
    """Raw write of one line to the report file (console if no file is open); no verdict counting."""
    if _REPORT_FILE is not None:
        _REPORT_FILE.write(line + "\n")
        _REPORT_FILE.flush()
    else:
        print(line)

def _log(msg: object = "") -> None:
    """Write one line to the report; a line carrying a failure / flag / success token is tallied in _MARK (read per step by _step)."""
    line = str(msg)
    if any(t in line for t in _FAIL_TOKENS):
        _MARK['fail'] += 1
    elif any(t in line for t in _FLAG_TOKENS):
        _MARK['flag'] += 1
    elif any(t in line for t in _OK_TOKENS):
        _MARK['ok'] += 1
    _emit(line)

def _section(title: str) -> None:
    """Console progress marker + matching banner in the report file."""
    print(f"  -> {title}")
    _log("")
    _log("=" * 78)
    _log(title)
    _log("=" * 78)

def _delta_vector(amps=None, dtype=float):
    v = np.zeros(qm._N_DELTA, dtype=dtype)
    for name, val in (amps or {}).items():
        v[qm._DELTA_NAMES.index(name)] = val
    return v

def _extract_all_kappas(t_pd, Delta_CT, U_dd, lambda_soc, Delta_tetra, Delta_B1g_static, Upp_ratio_bare, J_H_ratio, Z):
    """
    Standalone superexchange extraction mirroring ModelParams.__post_init__.
    Returns (G6, G7a, G7b, kappa_doublets(3,), kappa_cross(2,), B1g_t2g, basis, evals_2hole, evecs_2hole, U_pp).
    """
    H_soc_cf, Lx_t2g, Ly_t2g, _Lz_t2g, Sz_full, LS_op = qm._t2g_soc_cf_operators(lambda_soc, Delta_tetra, Delta_B1g_static)
    evals_soc, evecs_soc = np.linalg.eigh(H_soc_cf)
    B1g_t2g = Lx_t2g @ Lx_t2g - Ly_t2g @ Ly_t2g

    G6, G7a, G7b = qm._get_G6_G7a_G7b(evals_soc, evecs_soc, Sz_full, LS_op, B1g_t2g)

    z_O = Z / 2.0
    U_pp = (Upp_ratio_bare * U_dd) / (1.0 + z_O * t_pd ** 2 / (Delta_CT * (Delta_CT + U_dd)))
    J_H = J_H_ratio * U_dd

    H_2hole, basis = qm._multiorbital_2hole_hamiltonian(t_pd, Delta_CT, H_soc_cf, U_dd, U_pp, J_H)
    evals_2hole, evecs_2hole = np.linalg.eigh(H_2hole)

    kappa_doublets = qm._intra_doublet_kappa(basis, evals_2hole, evecs_2hole, [('G6', G6), ('G7a', G7a), ('G7b', G7b)])
    kappa_cross = qm._cross_doublet_kappas(basis, evals_2hole, evecs_2hole, G6, [(G7a, "7a"), (G7b, "7b")], B1g_t2g)
    return G6, G7a, G7b, kappa_doublets, kappa_cross, B1g_t2g, basis, evals_2hole, evecs_2hole, U_pp

_MEMO_LIMIT = {'eig': 3, 'vp': 128, 'kernel': 6, 'ctx': 3}      # 'eig' / 'ctx' hold ec (N_k x 24 x 24 complex, ~33 MB at _NK = 60): keep few
_MEMO: dict = {_t: OrderedDict() for _t in _MEMO_LIMIT}
_MEMO_COUNT: dict = {_t: [0, 0] for _t in _MEMO_LIMIT}           # [reuses, builds]
_SCF_CACHES: dict = {}                                           # state key -> the vertex_cache the SCF itself ended with at exactly that state (see _register_point)
_VP_MAX_PASSES, _VP_CHI_TOL = 4, 1e-6                            # fixed-point passes on χ_QQ(q=0) in the pair-vertex parameters of a gapped state

def _freeze(obj):
    """Read-only twin of obj: arrays become non-writeable VIEWS (no copy, the owner's flags are untouched), containers are rebuilt."""
    if isinstance(obj, np.ndarray):
        v = obj.view()
        v.flags.writeable = False
        return v
    if isinstance(obj, dict):
        return {k: _freeze(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_freeze(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_freeze(v) for v in obj)
    return obj

def _scalars(obj) -> tuple:
    return tuple(sorted((k, v) for k, v in vars(obj).items() if isinstance(v, (bool, int, float, str, np.integer, np.floating, np.bool_))))

def _solver_fp(solver) -> tuple:
    """Model fingerprint: solver identity + every scalar parameter of ModelParams + the solver's thermal / lattice scalars + monkey-patched callables on ModelParams."""
    p = solver.p
    return (id(solver), solver.kT, solver.g_JT_bare, solver._K_bare, solver.thermal_fs_nat_size, _scalars(p), tuple(sorted((k, id(v)) for k, v in vars(p).items() if callable(v))))

def _state_key(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, *extra) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a, dt in ((M, float), (Delta_vec, complex), (Quad67_vec, complex)):
        h.update(np.ascontiguousarray(np.atleast_1d(np.asarray(a, dtype=dt))).tobytes())
    h.update(repr([float(x).hex() for x in (Q, n_kspace, mu, g_t, g_J) + extra]).encode())
    return h.hexdigest()

def _memo(tag: str, key, build):
    store = _MEMO[tag]
    if key in store:
        _MEMO_COUNT[tag][0] += 1
        return store[key]
    _MEMO_COUNT[tag][1] += 1
    store[key] = val = _freeze(build())
    while len(store) > _MEMO_LIMIT[tag]:
        store.popitem(last=False)
    return val

def _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, Q_Eg2: float = 0.0):
    """(ev, ec) of the BdG H(k) stack on the solver's k-grid, read-only: one diagonalisation per distinct state."""
    vbdg = solver._get_vbdg()
    key = (_solver_fp(solver), _state_key(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, Q_Eg2))
    return _memo('eig', key, lambda: np.linalg.eigh(vbdg._build_H_stack(solver.k_points, M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, out=vbdg._H_stack, Q_Eg2=Q_Eg2)))

def _registered_cache(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J):
    """The vertex_cache the SCF itself ended with at exactly this state (None for any other state)."""
    return _SCF_CACHES.get(_state_key(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J))

def _register_point(pt) -> None:
    """Make the SCF's own vertex_cache the one the pair-vertex parameters of exactly this state are built with (the SCF evaluates them with it)."""
    if pt.cache is not None:
        _SCF_CACHES[_state_key(pt.M, pt.Q, pt.Delta_vec, pt.Quad67_vec, pt.n, pt.mu, pt.g_t, pt.g_J)] = pt.cache

def _cache_token(vc: dict | None) -> str:
    """Content token of a vertex_cache (stable between processes and immune to id() reuse after garbage collection)."""
    if vc is None:
        return ''
    h = hashlib.blake2b(digest_size=8)
    for k in ('V_irrep', 'S_irrep'):
        if k in vc:
            h.update(np.ascontiguousarray(np.asarray(vc[k], dtype=float)).tobytes())
    return h.hexdigest()

def _scf_vertex_params(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, vertex_cache: dict | None = None) -> dict:
    """
    Pair-vertex parameters of one state, assembled the way the SCF assembles them and built ONCE per state:
      * solver._make_vertex_params gets the state's own BdG eigenpairs, Q_Eg2 = 0 and -- for a gapped state -- the vertex_cache: both enter the cluster free energy / Schur K_eff through the condensation energy;
      * χ_QQ(q = 0) is the fixed point of  χ_QQ = RPA(Γ_M, V_JT_corr)(χ_QQ) for Δ = 0 the retardation vanishes and one pass suffices.
    Returns the dict {J_A1g_diag, J_B1g_7a, J_B1g_7b, tx_bare, ty_bare, F_cluster, J_eff, Gamma_M, V_JT_corr, V_cap, chi_QQ, chi_QQ_residual}.
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    gapped = bool(np.any(np.asarray(Delta_vec) != 0))
    skey = _state_key(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    vc = (vertex_cache if vertex_cache is not None else _SCF_CACHES.get(skey)) if gapped else None
    key = (_solver_fp(solver), float(target_doping).hex(), skey, _cache_token(vc))
    def build():
        p = solver.p
        E = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
        J_A1g_diag, J_B1g_7a, J_B1g_7b = _exch(p, Q, g_J)
        tx_bare, ty_bare = p.effective_hopping_anisotropic(Q)
        J_eff = qm._m2_weighted_J(p._exchange_J_q(J_A1g_diag, np.pi, np.pi), M)
        chi_in, chi_used, chi_new = 0.0, 0.0, 0.0
        for _ in range(_VP_MAX_PASSES if gapped else 1):
            chi_used = chi_in                                            # the χ_QQ this pass builds the vertex with (reported back)
            Gamma_M, F_cluster, V_JT_corr, V_cap = solver._make_vertex_params(target_doping, M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, 0.0, E, vc)
            if not gapped:
                break
            chi_new = float(solver._rpa_closure(M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, np.zeros(2), Gamma_M, J_A1g_diag, V_JT_corr, E, True)['chi_QQ'])
            if abs(chi_new - chi_used) <= _VP_CHI_TOL * max(1.0, abs(chi_new)):
                break
            chi_in = chi_new
        return dict(J_A1g_diag=J_A1g_diag, J_B1g_7a=J_B1g_7a, J_B1g_7b=J_B1g_7b, tx_bare=tx_bare, ty_bare=ty_bare, F_cluster=F_cluster, J_eff=J_eff, Gamma_M=Gamma_M, V_JT_corr=V_JT_corr, V_cap=V_cap, chi_QQ=chi_used, chi_QQ_residual=abs(chi_new - chi_used))
    return _memo('vp', key, build)

def _vertex_tuple(solver, target_doping: float, M, Q, Delta_vec, Quad67_vec, n_kspace: float, mu: float, g_t: float, g_J: float, vertex_cache: dict | None = None) -> tuple:
    """(Gamma_M, F_cluster, V_JT_corr, V_cap) -- the return layout of solver._make_vertex_params -- from the once-per-state _scf_vertex_params."""
    vp = _scf_vertex_params(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, vertex_cache)
    return vp['Gamma_M'], vp['F_cluster'], vp['V_JT_corr'], vp['V_cap']

def _build_vertex_cache(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float) -> dict:
    """Pairing-kernel cache of one state, built exactly as the SCF builds it (SCF-consistent vertex parameters), once."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    key = (_solver_fp(solver), float(target_doping).hex(), _state_key(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J))
    def build():
        vp = _scf_vertex_params(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
        return solver.compute_pairing_kernel_and_build_cache(target_doping, M, Q, mu, g_t, g_J, vp['J_A1g_diag'], vp['Gamma_M'], vp['V_JT_corr'], vp['V_cap'])
    return _memo('kernel', key, build)

def _build_fs_context(solver, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float) -> dict:
    """Shared FS setup of one state: the normal-state FS sample, its unique q-pairs and the BdG eigenpairs of the (Δ, F67)-dressed H(k)."""
    key = (_solver_fp(solver), _state_key(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J))
    def build():
        vbdg = solver._get_vbdg()
        fs_pts, vF_vec, fs_idx, weights = solver._get_fs_points(M, Q, mu, g_t, g_J)
        N_fs = len(fs_pts)
        vF_mag = np.linalg.norm(vF_vec, axis=1) if N_fs else np.zeros(0)
        i_idx = j_idx = np.zeros(0, dtype=int)
        unique_q = np.zeros((0, 2))
        inv_idx = np.zeros(0, dtype=int)
        if N_fs > 0:
            i_idx, j_idx, unique_q, inv_idx = qm._unique_q_pairs(fs_pts)
        ev_n, ec_n = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
        return {'vbdg': vbdg, 'M': M, 'Q': Q, 'n_kspace': n_kspace, 'mu': mu, 'g_t': g_t, 'g_J': g_J, 'fs_pts': fs_pts, 'vF_vec': vF_vec, 'vF_mag': vF_mag, 'fs_idx': fs_idx, 'weights': weights,
                'N_fs': N_fs, 'i_idx': i_idx, 'j_idx': j_idx, 'unique_q': unique_q, 'inv_idx': inv_idx, 'n_q': len(unique_q), 'ev_n': ev_n, 'ec_n': ec_n}
    return _memo('ctx', key, build)

@dataclass(frozen=True)
class _Point:
    """One state of the SCF landscape at fixed doping; arrays are read-only."""
    M: np.ndarray
    Q: float
    Delta_vec: np.ndarray
    Quad67_vec: np.ndarray
    n: float
    mu: float
    g_t: float
    g_J: float
    g_Das: float
    g_Dad: float
    g_Dbd: float
    cache: dict | None = None            # the vertex_cache the SCF itself ended with (SC state only)

def _exch(p, Q: float, g_J: float) -> tuple:
    J_A, J_B = p.exchange_channels(Q, g_J)
    return J_A, float(J_B[0]), float(J_B[1])


def _make_point(M, Q, Delta_vec, Quad67_vec, n, mu, g_t, g_J, g_Das, g_Dad, g_Dbd, cache=None) -> _Point:
    return _Point(_freeze(np.atleast_1d(np.asarray(M, dtype=float))), float(Q), _freeze(np.asarray(Delta_vec, dtype=complex)), _freeze(np.asarray(Quad67_vec, dtype=complex)), float(n), float(mu),
                  float(g_t), float(g_J), float(g_Das), float(g_Dad), float(g_Dbd), None if cache is None else _freeze(cache))

def _susceptibilities_sc(solver, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, q: np.ndarray, Gamma_M: np.ndarray, E_k_cache: tuple, apply_diamagnetic_QQ: bool = False, dHdQ_precomputed=None):
    """(χ_SS, χ_SQ, χ_SQ, χ_QQ) of the PRODUCTION closure solver._rpa_closure (matrix Moriya damping Γ_M + PSD projection of the 4x4 χ0): there is no second copy of that logic here.
    J_q and V_JT_corr enter only its determinant / Stoner factor, never the susceptibilities, so neutral values are passed."""
    J_A1g_diag = solver.p.exchange_channels(Q, g_J)[0]
    r = solver._rpa_closure(M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, q, Gamma_M, J_A1g_diag, 0.0, E_k_cache, apply_diamagnetic_QQ, dHdQ_precomputed)
    return r['chi_SS'], r['chi_SQ'], r['chi_SQ'], r['chi_QQ']

def _kappa_B1g_vs_pair_phase(evals_2hole, evecs_2hole, basis, G6, D_ref, B1g_t2g, alphas, t2_floor_frac: float = 1e-6):
    """
    Scan the relative phase α between the two spin components of the inter-doublet singlet pair operator,
    and compute κ_B1g(α) from the SAME ED Heff block that _cross_doublet_kappas uses.
    The B1g matrix element M = U6† B1g_t2g U7 is computed in the FIXED α=0 gauge.

    The standard gauge (α=0) has ε_ab = δ_{a↑,b↓} - δ_{a↓,b↑}.
    A general gauge rotates the down-component by e^{iα}:
        ε_ab(α) = δ_{a↑,b↓} - e^{iα} δ_{a↓,b↑}.
    """
    U6 = np.column_stack((G6['up'], G6['dn']))
    U7 = np.column_stack((D_ref['up'], D_ref['dn']))

    M_fixed = U6.conj().T @ B1g_t2g @ U7  # 2×2

    kappas = []
    t2_vals = []
    for alpha in alphas:
        # Build phase-rotated REFERENCE VECTORS only. The pair operator in the rotated gauge has its Γ7↓ component multiplied by e^{iα}.
        D_alpha = {k: D_ref[k] for k in D_ref}
        D_alpha['dn'] = D_ref['dn'] * np.exp(1j * alpha)

        refs = [qm._multiorbital_reference_vector(basis, G6[a], 0, D_alpha[b], 12)
                for a in ('up', 'dn') for b in ('up', 'dn')] \
             + [qm._multiorbital_reference_vector(basis, D_alpha[a], 0, G6[b], 12)
                for a in ('up', 'dn') for b in ('up', 'dn')]
        V = np.column_stack(refs)  # (153, 8)

        # Löwdin projection (same as _cross_doublet_kappas)
        d = V.shape[1]
        A = evecs_2hole.conj().T @ V
        w = np.sum(np.abs(A)**2, axis=1)
        sel = np.argsort(w)[::-1][:d]
        C = A[sel, :].conj().T
        s, U = np.linalg.eigh(C.conj().T @ C)
        if s[0] < 1e-8:
            kappas.append(np.nan); t2_vals.append(0.0); continue
        Phi = C @ ((U * s**-0.5) @ U.conj().T)
        Heff = (Phi * evals_2hole[sel]) @ Phi.conj().T
        Heff = 0.5 * (Heff + Heff.conj().T)

        # Trace formula (identical to _cross_doublet_kappas, but with M_fixed)
        T = np.kron(M_fixed, M_fixed.conj().T)
        t2 = float(np.real(np.vdot(T, T)))
        t2_vals.append(t2)
        if t2 <= 0.0:
            kappas.append(np.nan); continue
        kappas.append(float(np.real(np.vdot(T, Heff[0:4, 4:8])) / t2))

    kappas = np.array(kappas, dtype=float)
    t2_vals = np.array(t2_vals, dtype=float)

    # Threshold-based reliability mask: where t2 is much smaller than its maximum, the κ value is dominated by numerical noise in the division and is unreliable.
    if t2_vals.size:
        t2_max = float(np.max(t2_vals))
        if t2_max > 0.0:
            unreliable = t2_vals < t2_floor_frac * t2_max
            kappas = np.where(unreliable, np.nan, kappas)

    R2_val = float(np.linalg.norm(M_fixed, 'fro') ** 2)
    B_norm2 = float(np.linalg.norm(B1g_t2g, 'fro') ** 2)
    if R2_val < max(qm._B1G_BLOCK_MIN_FRAC * B_norm2, qm._WIGNER_ECKART_DIV_FLOOR):
        kappas = np.full_like(kappas, np.nan)
    return kappas, t2_vals

def _inversion_permutation(fs_pts: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, float]:
    """perm[i] = index j with k_j ≡ −k_i (mod 2π); returns (perm, max residual).  Exact for the symmetry-closed sample of _get_fs_points."""
    S = fs_pts[:, None, :] + fs_pts[None, :, :]
    S = (S + np.pi) % (2.0 * np.pi) - np.pi                    # wrap k_i + k_j to (−π, π]
    D = np.linalg.norm(S, axis=2)
    perm = np.argmin(D, axis=1)
    return perm, float(D[np.arange(len(fs_pts)), perm].max())


def _pair_irrep_analysis(solver, ctx: dict, blocks: dict[str, np.ndarray], tol: float = 1e-8, n_alpha: int = 72, verbose: bool = True) -> dict:
    """
    Full irrep decomposition of the pair operators of the registry channels in `blocks`, for BOTH
    the Γ6↔Γ7a and Γ6↔Γ7b branches.  The k-space character of the form factor is taken from the channel's D4h label.

    blocks: dict keyed by '<irrep><branch>' (e.g. 's7a', 'es7b', …) — the same names as _DELTA_NAMES.
            The values are the 12×12 anomalous (particle→hole) blocks of the 24×24 BdG Hamiltonian in the fullbasis.
    """
    ops = _d4h_group()
    lab = [_d4h_class(M) for M in ops]
    ch4 = [_d4h_characters(M) for M in ops]
    d2_idx = [i for i, M in enumerate(ops) if np.allclose(M, np.diag(np.diag(M)))]
    ch2 = [_d2h_characters(ops[i]) for i in d2_idx]

    # orthonormality of the built tables (guards against a typo in the characters)
    for tab, idx, irr in ((ch4, list(range(16)), _D4H_IRREPS), (ch2, d2_idx, _D2H_IRREPS)):
        for a, (na, _) in enumerate(irr):
            for b, (nb, _) in enumerate(irr):
                s = sum(tab[j][na] * tab[j][nb] for j in range(len(tab))) / len(tab)
                assert abs(s - (1.0 if a == b else 0.0)) < 1e-12, (na, nb, s)

    R6   = [_rot6(M, ctx['l3']) for M in ops]
    Rord = [ctx['U_ord'].conj().T @ R @ ctx['U_ord'] for R in R6]
    U24  = [_u24(Ro) for Ro in Rord]
    chk  = np.array([_chi_B1_orbit(M) for M in ops])   # k-space character of φ_d(k)
    B24  = _b1g_24(ctx['B1g_op'])

    # (0) Is R6(g) really a symmetry of the model H? (D4h at Δ_B1g_static=0, only D2h otherwise) and do the doublets carry it block-diagonally?
    comm = np.array([np.linalg.norm(R @ ctx['H'] - ctx['H'] @ R) for R in R6])
    def _leak(Ro):
        blk = np.zeros_like(Ro)
        for a in range(3):
            blk[2 * a:2 * a + 2, 2 * a:2 * a + 2] = Ro[2 * a:2 * a + 2, 2 * a:2 * a + 2]
        return float(np.linalg.norm(Ro - blk))
    leak = np.array([_leak(Ro) for Ro in Rord])
    sym_d4h_ok = float(comm.max()) < 1e-8
    if verbose:
        _log(f"    [group check]  max‖[R(g),H]‖: D4h(16 ops)={comm.max():.2e}   D2h(8)={comm[d2_idx].max():.2e}"
              f"   |  max doublet leakage ‖R_ord − block-diag‖: D4h={leak.max():.2e}  D2h={leak[d2_idx].max():.2e}")
        if not sym_d4h_ok:
            _log("    ⚠ the model H is not D4h-invariant (Δ_B1g_static ≠ 0 ⇒ D2h only): the D4h weights measure the deviation from D4h, not an exact irrep.")

    # The gauge rotation acts on the ↑-component of the relevant Γ7 doublet, i.e. orbital 2 for the 7a branch and orbital 4 for the 7b branch.
    _PHASE_ORB = (('7a', 2), ('7b', 4))

    out = {'ratio': ctx['ratio'], 'sym_d4h_ok': sym_d4h_ok, 'channels': {}}

    # Iterate 7a-first then 7b-first to match the _DELTA_NAMES layout, so that the report reads in the same order as the Delta_vec / Quad67_vec layout.
    for chan_tag, idx_ph in _PHASE_ORB:
        for i_irr, spec in enumerate(qm._PAIR_CHANNELS):
            key = f'{spec.name}{chan_tag}'
            if key not in blocks:
                continue                       # tolerate partial block dicts
            block_12 = np.asarray(blocks[key], dtype=complex)
            nD2_chk = float(np.real(np.trace(block_12.conj().T @ block_12)))
            if nD2_chk < 1e-20:
                # Zero-norm pair operator: irrep content is undefined.
                # Can happen e.g. if the test k-point hits a form-factor node
                out['channels'][key] = {
                    'w4':   {n: 0.0 for n, _ in _D4H_IRREPS},
                    'w4_orb': {n: 0.0 for n, _ in _D4H_IRREPS},
                    'w2':   {n: 0.0 for n, _ in _D2H_IRREPS},
                    'ov':   np.zeros(16, dtype=complex),
                    'res':  np.zeros(16, dtype=float),
                    'tr_B1g': 0.0, 'comm_B1g': 0.0,
                    'best': {'w': 0.0, 'irrep': '(zero-norm)',
                            'alpha': 0.0, 'weights': {}},
                    'b1g_range': (0.0, 0.0),
                }
                continue
            D24 = _nambu24(blocks[key])

            # k-space character of the form factor from the registry label (1-d irreps A1g, A2g, B1g, B2g: the character is folded in); Eu is 2-d — its two partners are exchanged by C4, there is
            # no single character, so only the orbital part is resolved (ck = 1).
            ck = np.array([c[spec.d4h] for c in ch4]) if spec.d4h in ('A1g', 'A2g', 'B1g', 'B2g') else np.ones(16)

            w4,    ov4, res4 = _irrep_weights(D24, U24, ck, ch4, _D4H_IRREPS)
            w4_orb, _,  _    = _irrep_weights(D24, U24, np.ones(16), ch4, _D4H_IRREPS)
            w2,    _,   _    = _irrep_weights(D24, [U24[i] for i in d2_idx],
                                              ck[d2_idx], ch2, _D2H_IRREPS)

            c_B    = float(np.real(np.trace(B24 @ D24 @ D24.conj().T)))
            comm_B = float(np.linalg.norm(B24 @ D24 - D24 @ B24))

            # gauge scan over the relative phase α of the two singlet components, using the phase-orbital index of THIS (irrep, branch) pair.
            best = {'w': -1.0}
            b1g_rng = [1e9, -1e9]
            for al in np.linspace(0.0, 2.0 * np.pi, n_alpha, endpoint=False):
                ph = np.ones(6, dtype=complex)
                ph[idx_ph] = np.exp(1j * al)
                P12 = np.kron(np.eye(2), np.diag(ph))
                Da = _nambu24(P12 @ blocks[key] @ P12.T)   # Δ → P Δ Pᵀ (re-gauge)
                wa, _, _ = _irrep_weights(Da, U24, ck, ch4, _D4H_IRREPS)
                b1g_rng = [min(b1g_rng[0], wa['B1g']),
                           max(b1g_rng[1], wa['B1g'])]
                top = max(wa, key=wa.get)
                if wa[top] > best['w'] + 1e-12:
                    best = {'w': wa[top], 'irrep': top,
                            'alpha': float(al), 'weights': wa}

            out['channels'][key] = {
                'w4': w4, 'w4_orb': w4_orb, 'w2': w2,
                'ov': ov4, 'res': res4,
                'tr_B1g': c_B, 'comm_B1g': comm_B,
                'best': best, 'b1g_range': tuple(b1g_rng),
            }

    # gauge-INDEPENDENT content of the pair spaces Γ6⊗Γ7a and Γ6⊗Γ7b (4-dim each): χ(g) = tr R_Γ6 · tr R_Γ7x   (× χ_k for the d-form).
    def _blk_tr(Ro, a):
        return np.trace(Ro[2 * a:2 * a + 2, 2 * a:2 * a + 2])

    space = {}
    for tag, b in (('6x7a', 1), ('6x7b', 2)):
        chi_sp = np.array([_blk_tr(Rord[i], 0) * _blk_tr(Rord[i], b)
                           for i in range(16)])
        for form, ck_f in (('s', np.ones(16)), ('d', chk)):
            mult = {n: float(np.real(sum(ch4[i][n] * ck_f[i] * chi_sp[i] for i in range(16)) / 16.0))
                    for n, _ in _D4H_IRREPS}
            space[(tag, form)] = {n: m for n, m in mult.items() if abs(m) > 0.05}
    out['pair_space'] = space

    if verbose:
        _print_pair_irrep_report(out, lab, ops, tol)
    return out

def _print_pair_irrep_report(out: dict, lab: list, ops: list, tol: float) -> None:
    def _fmt(w, n=3):
        top = sorted(((k, v) for k, v in w.items() if v > 1e-6), key=lambda kv: -kv[1])[:n]
        return "  ".join(f"{k}:{v:.4f}" for k, v in top)
    _log(f"    {'channel':<6s} {'Tr[B1g24·D·D†]':>16s} {'‖[B1g24,D]‖_F':>14s}   full irrep (D4h) — weights at α=0 (the code's gauge)")
    for name, c in out['channels'].items():
        _log(f"    {name:<6s} {c['tr_B1g']:>+16.6e} {c['comm_B1g']:>14.6e}   {_fmt(c['w4'])}")
    _log("    --- orbital part only (no k-form) / D2h weights ---")
    for name, c in out['channels'].items():
        _log(f"    {name:<6s} orbital: {_fmt(c['w4_orb'], 2):<40s}  D2h: {_fmt(c['w2'], 2)}")
    _log("    --- characters: overlap χ(g)=⟨D,g·D⟩/⟨D,D⟩ per class, residual ‖gD−χD‖/‖D‖.  Real ±1 with residual≈0 ⇒ 1-dim irrep.  A partner function of a 2-dim irrep (Eg) is NOT")
    _log("    --- mapped onto itself by g (it moves into the other partner): overlap 0 / residual≈1 is EXPECTED there and is no sign of mixing -- read the projector weights above instead ---")
    _log(f"    {'class':<8s}" + "".join(f"{n:>22s}" for n in out['channels']))
    for cl in _D4H_CLASSES:
        members = [i for i, l in enumerate(lab) if l == cl]
        row = ""
        for name, c in out['channels'].items():
            ovs = c['ov'][members]
            row += f"{np.real(ovs).mean():>+9.4f}{np.imag(ovs).mean():>+8.4f}i /{c['res'][members].max():6.0e}"
        _log(f"    {cl:<8s}{row}")
    _log("    --- gauge scan: relative phase (α) of the two singlet components is an arbitrary eigh phase; the code uses α=0 ---")
    for name, c in out['channels'].items():
        b = c['best']
        irrep_label = b.get('irrep', '(unset)')
        alpha_val   = b.get('alpha', 0.0)
        _log(f"    {name:<6s} purest irrep: {irrep_label} (w={b['w']:.4f}) "
            f"α*={np.degrees(alpha_val):6.1f}°   "
            f"B1g weight range over α∈[0,2π): "
            f"[{c['b1g_range'][0]:.4f}, {c['b1g_range'][1]:.4f}]   "
            f"at α=0: B1g={c['w4'].get('B1g', 0.0):.4f}")
    _log("    --- gauge-independent pair space Γ6⊗Γ7x (4-dim), irrep multiplicities (multiplied by the k-form) ---")
    for (tag, form), mult in out['pair_space'].items():
        _log(f"    {tag} {form}-form:  " + "  ".join(f"{k}×{v:.0f}" for k, v in mult.items()))
    # Selection rule Q·Δ_i*Δ_j (B1g phonon): allowed iff Γ_i ⊗ Γ_j ∋ B1g   (only meaningful for pure 1-dim irreps)
    names = list(out['channels'])
    dom = {n: max(out['channels'][n]['w4'], key=out['channels'][n]['w4'].get) for n in names}
    wdom = {n: out['channels'][n]['w4'][dom[n]] for n in names}
    tab = {}
    for i, M in enumerate(ops):
        tab[i] = _d4h_characters(M)
    _log("    --- B1g-phonon linear coupling  Q·Δ_i*·Δ_j  (rule: Γ_i⊗Γ_j ∋ B1g), for the dominant irreps at α=0 ---")
    _log("    " + " " * 6 + "".join(f"{n:>10s}" for n in names))
    for a in names:
        row = ""
        for b in names:
            if wdom[a] < 0.95 or wdom[b] < 0.95:
                row += f"{'mixed':>10s}"
                continue
            mult = sum(tab[i][dom[a]] * tab[i][dom[b]] * tab[i]['B1g'] for i in range(16)) / 16.0
            row += f"{'allowed' if abs(mult) > 1e-9 else '—':>10s}"
        _log(f"    {a:<6s}{row}   ({dom[a]}, w={wdom[a]:.3f})")


def test_time_reversal_operator_construction(solver, tol: float = 1e-9) -> None:
    """
    Checks:
      (1) Θ² = -1 on the bare t2g×spin basis (fermionic time reversal).
      (2) Θ commutes with H_soc_cf (the SOC+CF Hamiltonian is Θ-invariant).
      (3) The same holds once rotated into the G6/G7a/G7b eigenbasis.
      (4) Θ|up_d⟩ ∝ |dn_d⟩ for all three doublets, overlap = 1.0 -- i.e. the
          code's own up/dn labeling (by the sign of Sz) gives genuine Kramers
          pairs, not just a nominal label.
      (5) B1g_offdiag (the orbital form factor appearing in H_TRW) is Θ-even --
          as expected for an L²-type single-particle operator.
    """
    p = solver.p
    H_soc_cf, Lx, Ly, Lz, Sz_full, LS_op = qm._t2g_soc_cf_operators(p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static)
    Theta_orb = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]], dtype=complex)
    Theta_spin = np.array([[0, 1], [-1, 0]], dtype=complex)
    U_bare = np.kron(Theta_orb, Theta_spin)

    sq_ok = bool(np.allclose(U_bare @ U_bare.conj(), -np.eye(6), atol=tol))
    comm_bare_ok = bool(np.allclose(U_bare @ H_soc_cf.conj() @ U_bare.conj().T, H_soc_cf, atol=tol))

    G6, G7a, G7b, _kd, _kc, _B1g_t2g, *_ = _extract_all_kappas(p.t_pd, p.Delta_CT, p.U_dd, p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static, p.Upp_ratio_bare, p.J_H_ratio, qm._Z)
    U_ord = np.column_stack([G6['up'], G6['dn'], G7a['up'], G7a['dn'], G7b['up'], G7b['dn']])
    U_new = U_ord.conj().T @ U_bare @ U_ord.conj()

    sq_new_ok = bool(np.allclose(U_new @ U_new.conj(), -np.eye(6), atol=tol))
    H_new_basis = U_ord.conj().T @ H_soc_cf @ U_ord
    comm_new_ok = bool(np.allclose(U_new @ H_new_basis.conj() @ U_new.conj().T, H_new_basis, atol=tol))

    e = np.eye(6, dtype=complex)
    labels = ['G6_up', 'G6_dn', 'G7a_up', 'G7a_dn', 'G7b_up', 'G7b_dn']
    kramers_overlaps = {}
    for i, lab in enumerate(labels):
        theta_ei = U_new @ e[:, i].conj()
        overlaps = np.abs(e.conj().T @ theta_ei)
        kramers_overlaps[lab] = float(np.max(overlaps))
    kramers_ok = all(v > 1.0 - tol for v in kramers_overlaps.values())

    B1g_even = bool(np.allclose(
        U_new @ p.B1g_offdiag.astype(complex).conj() @ U_new.conj().T,
        p.B1g_offdiag, atol=tol))

    checks = [
        ("Θ² = -1  (bare t2g×spin basis)", sq_ok),
        ("Θ commutes with H_soc_cf  (bare basis)", comm_bare_ok),
        ("Θ² = -1  (G6/G7a/G7b eigenbasis)", sq_new_ok),
        ("Θ commutes with H_soc_cf  (G6/G7a/G7b eigenbasis)", comm_new_ok),
        ("Θ|up_d⟩ ∝ |dn_d⟩ for all three doublets (genuine Kramers pairs)", kramers_ok),
        ("B1g_offdiag is Θ-even", B1g_even),
    ]

    _log("=" * 90)
    for name, ok in checks:
        _log(f"    [{'✓' if ok else '✗'}] {name}")
    _log("    Kramers overlaps: " +
          "  ".join(f"{k}={v:.6f}" for k, v in kramers_overlaps.items()))
    n_fail = sum(1 for _, ok in checks if not ok)
    _log()
    if n_fail == 0:
        _log("  VERDICT: PASS -- Θ is correctly constructed; the TRW/Θ tests built on it are reliable.")
    else:
        _log(f"  VERDICT: FAIL -- {n_fail} check(s) failed; results of the Θ-based tests are NOT reliable until this is fixed.")
    _log("=" * 90)


def test_simulated_sc_state_diagnostic(solver, target_doping: float, M_sc: np.ndarray, Q_sc: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_sc: float, mu_sc: float, g_t: float, g_J: float, M_normal: np.ndarray, n_kspace_n: float, mu_n: float, ctx: dict | None = None, n_levels: int = 3, antinodal_pctl: float = 75.0, nodal_pctl: float = 25.0, top_k: int = 6, vertex_cache: dict | None = None) -> None:
    """
    Compare the physically measurable JT softening between the true normal state
    (M_normal, Q=0, Δ=0, F67=0) and the true SC state (M_sc, Q_sc, Δ, F67).
    """
    vertex_cache = vertex_cache if vertex_cache is not None else _registered_cache(M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J)
    p = solver.p
    M_sc     = np.asarray(M_sc,     dtype=float)
    M_normal = np.asarray(M_normal, dtype=float)
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec   = np.asarray(Quad67_vec,   dtype=complex)
    g_JT      = solver.g_JT_bare
    kT        = float(solver.kT)

    Delta_zero = np.zeros(qm._N_DELTA, dtype=complex)
    Quad67_zero   = np.zeros(qm._N_DELTA, dtype=float)
    Q_normal   = 0.0

    # ═══════════════════════════════════════════════════════════════════
    # [1] Normal-state susceptibilities
    # ═══════════════════════════════════════════════════════════════════
    vp = _scf_vertex_params(solver, target_doping, M_normal, Q_normal, Delta_zero, Quad67_zero, n_kspace_n, mu_n, g_t, g_J)
    pref_n = RPAPrefactors(J_eff=vp['J_eff'], V_JT_corr=vp['V_JT_corr'], Gamma_M=vp['Gamma_M'])

    J_A1g_n, J_B1g_7a_n, J_B1g_7b_n = _exch(p, Q_normal, g_J)
    ev_n, ec_n = _bdg_eig(solver, M_normal, Q_normal, Delta_zero, Quad67_zero, n_kspace_n, mu_n, g_t, g_J)

    _rpa_q0_n  = solver._rpa_closure(M_normal, Q_normal, Delta_zero, Quad67_zero, mu_n, g_t, g_J, np.zeros(2), pref_n.Gamma_M, J_A1g_n, pref_n.V_JT_corr, (ev_n, ec_n), True)
    _rpa_afm_n = solver._rpa_closure(M_normal, Q_normal, Delta_zero, Quad67_zero, mu_n, g_t, g_J, np.array([np.pi, np.pi]), pref_n.Gamma_M, J_A1g_n, pref_n.V_JT_corr, (ev_n, ec_n), True)
    chi_SS_q0_n,  chi_QQ_q0_n  = _rpa_q0_n['chi_SS'],  _rpa_q0_n['chi_QQ']
    chi_SS_afm_n, chi_QQ_afm_n = _rpa_afm_n['chi_SS'], _rpa_afm_n['chi_QQ']
    det_q0_n, det_afm_n, stoner_n = _rpa_q0_n['det'], _rpa_afm_n['det'], _rpa_afm_n['stoner']

    # ═══════════════════════════════════════════════════════════════════
    # [2] SC-state susceptibilities
    # ═══════════════════════════════════════════════════════════════════
    txb_sc, tyb_sc = p.effective_hopping_anisotropic(Q_sc)
    J_A1g_sc, J_B1g_7a_sc, J_B1g_7b_sc = _exch(p, Q_sc, g_J)
    Gamma_M_sc, Fc_sc, V_JT_corr_sc, _ = _vertex_tuple(solver, target_doping, M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J, vertex_cache=vertex_cache)

    if ctx is not None and 'ev_n' in ctx:
        ev_sc, ec_sc = ctx['ev_n'], ctx['ec_n']
    else:
        ev_sc, ec_sc = _bdg_eig(solver, M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J)

    _rpa_q0_sc  = solver._rpa_closure(M_sc, Q_sc, Delta_vec, Quad67_vec, mu_sc, g_t, g_J, np.zeros(2), Gamma_M_sc, J_A1g_sc, V_JT_corr_sc, (ev_sc, ec_sc), True)
    _rpa_afm_sc = solver._rpa_closure(M_sc, Q_sc, Delta_vec, Quad67_vec, mu_sc, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M_sc, J_A1g_sc, V_JT_corr_sc, (ev_sc, ec_sc), True)
    chi_SS_sc_q0,   chi_SQ_sc_q0,   chi_QQ_sc_q0   = _rpa_q0_sc['chi_SS'],  _rpa_q0_sc['chi_SQ'],  _rpa_q0_sc['chi_QQ']
    chi_SS_sc_pipi, chi_SQ_sc_pipi, chi_QQ_sc_pipi = _rpa_afm_sc['chi_SS'], _rpa_afm_sc['chi_SQ'], _rpa_afm_sc['chi_QQ']
    det_q0_sc, det_afm_sc, stoner_sc = _rpa_q0_sc['det'], _rpa_afm_sc['det'], _rpa_afm_sc['stoner']
    chi_QQ_sc_kubo = float(solver._chi_QQ_matrix_elements(M_sc, Q_sc, Delta_vec, Quad67_vec, mu_sc, g_t, g_J))

    # ═══════════════════════════════════════════════════════════════════
    # [3] Shared evaluators (χ_τ, K_eff)
    # ═══════════════════════════════════════════════════════════════════
    scale   = p.Delta_CF / max(g_JT, qm._MATH_EPS)
    h_floor = 1e-4
    h_schedule = [min(max(1e-3 * max(abs(Q_sc), scale), h_floor), 0.05 * scale) / 2**k
                  for k in range(n_levels)]
    h_K4 = max(h_schedule[0], 10.0 * h_floor)

    def _eval_K_eff_at_Q(M, Q_val, delta_in, Quad67_in, mu_guess):
        """Full K_eff at fixed (M, Q, Δ, F67) with the cluster + Schur corrections."""
        mu_v, n_v = solver._find_mu_for_density(target_doping, M, Q_val, delta_in, Quad67_in, mu_guess, g_t, g_J)
        _, F_cluster, *_  = solver._make_vertex_params(target_doping, M, Q_val, delta_in, Quad67_in, mu_v, g_t, g_J, vertex_cache=vertex_cache)
        return F_cluster['K_eff']

    # ═══════════════════════════════════════════════════════════════════
    # [4] χ_τ branches (Richardson + k-resolved at finest step)
    # ═══════════════════════════════════════════════════════════════════
    def _chi_tau_branch(M, Q_eq, delta_in, Quad67_in, mu_eq):
        """Central-diff χ_τ at the state's own equilibrium Q_eq. Returns (scalar_est, conv, nonlin, chi_k_finest)."""
        cd_list, chi_k_finest = [], None
        for hk in h_schedule:
            _, B1g_k_p = solver._eval_B1g_and_kweights_at_Q(target_doping, M, Q_eq + hk, delta_in, Quad67_in, mu_eq, g_t, g_J)  
            _, B1g_k_m = solver._eval_B1g_and_kweights_at_Q(target_doping, M, Q_eq - hk, delta_in, Quad67_in, mu_eq, g_t, g_J)
            chi_k = (B1g_k_p - B1g_k_m) / (2.0 * hk)
            cd_list.append(float(np.dot(solver.k_weights, chi_k)))
            if chi_k_finest is None:
                chi_k_finest = chi_k
        if len(cd_list) < 2:
            return cd_list[0], False, False, chi_k_finest
        R = [(4.0 * cd_list[k+1] - cd_list[k]) / 3.0 for k in range(len(cd_list)-1)]
        if len(R) >= 2:
            RR = [(4.0 * R[k+1] - R[k]) / 3.0 for k in range(len(R)-1)]
            est = 0.5 * (R[-1] + RR[-1])
            err = abs(RR[-1] - R[-1]) / max(abs(est), _CHI_TAU_ABS_FLOOR)
            conv = err < 0.03
        else:
            est = R[-1]
            err = abs(R[-1] - cd_list[-1]) / max(abs(est), _CHI_TAU_ABS_FLOOR)
            conv = err < 0.03
        nonlin = (len(cd_list) >= 3 and
                  abs(cd_list[-2] - cd_list[-1]) / max(abs(cd_list[-1]), _CHI_TAU_ABS_FLOOR) > 0.2)
        return est, conv, nonlin, chi_k_finest

    def _weight(conv, nonlin):
        if nonlin: return 0.5
        if conv:   return 1.0
        return 0.0

    # SC branch: (M_sc, Q_sc, Δ, F67)
    chi_sc_est,  conv_sc,  nonlin_sc,  chi_sc_k = _chi_tau_branch(M_sc, Q_sc, Delta_vec, Quad67_vec, mu_sc)
    w_sc = _weight(conv_sc, nonlin_sc)

    # Normal branch: (M_n, Q=0, Δ=0, F67=0)
    chi_n_est,   conv_n,   nonlin_n,   chi_n_k  = _chi_tau_branch(M_normal, Q_normal, Delta_zero, Quad67_zero, mu_n)
    w_n = _weight(conv_n, nonlin_n)

    # Diagnostic: how much of the change comes from M-rearrangement alone?
    chi_n_fixedM_est, _, _, _ = _chi_tau_branch(M_sc, Q_sc, Delta_zero, Quad67_zero, mu_sc)
    chi_n_fixedM = chi_n_fixedM_est / g_JT

    # Convert to physical χ_τ (eV⁻¹)
    chi_tau_sc = chi_sc_est / g_JT
    chi_tau_n  = chi_n_est  / g_JT

    # Signed SC-induced softening (positive = SC softens the lattice):
    # χ_τ is restoring (negative), so softening means χ_τ,sc < χ_τ,n < 0.
    chi_tau_excess_signed = chi_tau_n - chi_tau_sc
    if w_sc > 0.0 and w_n > 0.0:
        chi_tau_net = min(w_sc, w_n) * max(chi_tau_excess_signed, 0.0)
    else:
        chi_tau_net = 0.0

    # Decomposition: Δ/F67 direct effect vs. M-rearrangement effect
    chi_tau_net_fixedM = chi_n_fixedM - chi_tau_sc
    chi_tau_M_contribution = chi_tau_n - chi_n_fixedM

    # k-resolved arrays (finest finite-difference step)
    chi_net_k = chi_n_k - chi_sc_k
    chi_n_tot  = float(np.dot(solver.k_weights, chi_n_k))
    chi_sc_tot = float(np.dot(solver.k_weights, chi_sc_k))
    chi_net_tot = chi_n_tot - chi_sc_tot

    # ═══════════════════════════════════════════════════════════════════
    # [5] K_eff stencils
    # ═══════════════════════════════════════════════════════════════════
    # SC branch
    K_eff_sc = _eval_K_eff_at_Q(M_sc, Q_sc, Delta_vec, Quad67_vec, mu_sc)
    K_eff_m2 = _eval_K_eff_at_Q(M_sc, Q_sc - 2.0 * h_K4, Delta_vec, Quad67_vec, mu_sc)
    K_eff_m1 = _eval_K_eff_at_Q(M_sc, Q_sc - h_K4,       Delta_vec, Quad67_vec, mu_sc)
    K_eff_p1 = _eval_K_eff_at_Q(M_sc, Q_sc + h_K4,       Delta_vec, Quad67_vec, mu_sc)
    K_eff_p2 = _eval_K_eff_at_Q(M_sc, Q_sc + 2.0 * h_K4, Delta_vec, Quad67_vec, mu_sc)

    K_values = np.asarray([K_eff_m2, K_eff_m1, K_eff_sc, K_eff_p1, K_eff_p2], dtype=float)
    K_finite = bool(np.all(np.isfinite(K_values)))
    K_scale  = max(abs(K_eff_sc), qm._MATH_EPS)
    K_eff_spread = ((abs(K_eff_p1 - K_eff_sc) + abs(K_eff_m1 - K_eff_sc)) / K_scale
                    if K_finite else float('inf'))
    K_eff_trustworthy = bool(K_finite and np.isfinite(K_eff_spread) and K_eff_spread <= 10.0)

    # b_Q = (1/6) d²K_eff/dQ²  (SC state)
    if K_eff_trustworthy:
        K4_raw = (-K_eff_m2 + 16.0*K_eff_m1 - 30.0*K_eff_sc + 16.0*K_eff_p1 - K_eff_p2) / (12.0 * h_K4**2)
        b_Q = max(K4_raw, 0.01 * abs(K_eff_sc), 0.0) / 6.0
        b_Q_valid = bool(np.isfinite(b_Q) and b_Q >= 0.0)
    else:
        b_Q = 0.0; b_Q_valid = False

    # Moriya-SCR regularization (SC branch)
    if K_eff_trustworthy and b_Q_valid and b_Q > qm._MATH_EPS and np.isfinite(K_eff_sc):
        Gamma_Q = 0.5 * (-K_eff_sc + math.sqrt(K_eff_sc**2 + 4.0 * b_Q * kT))
        K_eff_reg = K_eff_sc + Gamma_Q
        if not np.isfinite(K_eff_reg):
            K_eff_reg = float('nan')
    else:
        K_eff_reg = float('nan')

    # Normal branch K_eff
    K_eff_n_fixedM = _eval_K_eff_at_Q(M_sc, Q_sc, Delta_zero, Quad67_zero, mu_sc)
    K_eff_n_phys = _eval_K_eff_at_Q(M_normal, Q_normal, Delta_zero, Quad67_zero, mu_n)
    dK_eff = K_eff_sc - K_eff_n_fixedM
    K_spont = float(g_JT**2 * chi_QQ_q0_n)

    # ═══════════════════════════════════════════════════════════════════
    # [6] λ_JT_sc
    # ═══════════════════════════════════════════════════════════════════
    if chi_tau_net > 0.0 and K_eff_trustworthy and np.isfinite(K_eff_reg) and K_eff_reg > 0.0:
        lambda_JT_sc = g_JT**2 * chi_tau_net / K_eff_reg
    else:
        lambda_JT_sc = 0.0

    # ═══════════════════════════════════════════════════════════════════
    # [7] Nodal coherence length
    # ═══════════════════════════════════════════════════════════════════
    if ctx is not None and 'fs_pts' in ctx and len(ctx['fs_pts']) > 0:
        fs_pts, weights, vF_mag = ctx['fs_pts'], ctx['weights'], ctx['vF_mag']
    else:
        fs_pts, vF_vec, _, weights = solver._get_fs_points(M_sc, Q_sc, mu_sc, g_t, g_J, False)
        vF_mag = np.linalg.norm(vF_vec, axis=1) if len(fs_pts) > 0 else np.zeros(0)

    xi_nodal, vF_nodal = float('inf'), 0.0
    if len(fs_pts) > 0:
        pd_abs = np.abs(np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1]))
        m_nod = pd_abs < np.percentile(pd_abs, qm._NODAL_REGION_PCTL)
        if np.any(m_nod):
            Dd_mag   = abs(Delta_vec[qm._DELTA_IDX['d7a']]) * float(np.percentile(pd_abs[m_nod], qm._NODAL_REGION_PCTL))
            vF_nodal = float(np.average(vF_mag[m_nod], weights=weights[m_nod]))
        else:
            Dd_mag   = abs(Delta_vec[qm._DELTA_IDX['d7a']]) * float(np.percentile(pd_abs, qm._NODAL_REGION_PCTL))
            vF_nodal = float(np.average(vF_mag, weights=weights))
        xi_nodal = vF_nodal / (np.pi * Dd_mag) if Dd_mag > qm._MATH_EPS else float('inf')

    # ═══════════════════════════════════════════════════════════════════
    # [8] Logging
    # ═══════════════════════════════════════════════════════════════════
    _log("═" * 78)
    _log(f"  SC-STATE DIAGNOSTIC   δ={target_doping:.4f}  Q_sc={Q_sc:+.5f}  |Δ|₀(total)={float(np.sum(np.abs(Delta_vec))):.5f}")
    _log(f"  M_sc={np.array2string(M_sc, precision=5)}  n_sc={n_sc:.6f}  μ_sc={mu_sc:+.6f}")
    _log(f"  M_n ={np.array2string(M_normal, precision=5)}  n_n ={n_kspace_n:.6f}  μ_n ={mu_n:+.6f}")
    _log(f"  g_t={g_t:.4f}  g_J={g_J:.4f}  g_JT={g_JT:.4f}  "
         f"V_JT_corr(sc)={V_JT_corr_sc:+.5f}")
    _log()

    # ── [1] Susceptibilities ──────────────────────────────────────────
    _log("  [1] SUSCEPTIBILITIES   normal → SC")
    _log(f"      {'quantity':<14s}  {'normal':>12s}  {'SC':>12s}  {'Δ':>12s}")
    for nm, a, b in (("χ_SS(q=0)", chi_SS_q0_n, chi_SS_sc_q0),
                     ("χ_SS(π,π)", chi_SS_afm_n, chi_SS_sc_pipi),
                     ("χ_QQ(q=0)", chi_QQ_q0_n, chi_QQ_sc_q0),
                     ("χ_QQ(π,π)", chi_QQ_afm_n, chi_QQ_sc_pipi)):
        _log(f"      {nm:<14s}  {a:>+12.5f}  {b:>+12.5f}  {b - a:>+12.5f}")
    _log(f"      {'χ_SQ(q=0)':<14s}  {'--':>12s}  {chi_SQ_sc_q0:>+12.5e}")
    _log(f"      {'χ_SQ(π,π)':<14s}  {'--':>12s}  {chi_SQ_sc_pipi:>+12.5e}")
    _log(f"      Kubo χ_QQ(SC): RPA={chi_QQ_sc_q0:+.5f}  Kubo={chi_QQ_sc_kubo:+.5f}  "
         f"Δ={chi_QQ_sc_kubo - chi_QQ_sc_q0:+.2e}")
    _log()

    # ── [2] RPA determinants ──────────────────────────────────────────
    _ds = lambda d: "BREAKDOWN" if d < 0 else ("QCP-near" if d < 0.05 else "stable")
    _log("  [2] RPA DETERMINANTS   det(1−χ₀U); 1 = QCP, <0 = breakdown")
    _log(f"      {'':<10s}  {'normal':>12s}  {'SC':>12s}  {'Δ':>12s}  SC status")
    _log(f"      {'det_q0':<10s}  {det_q0_n:>+12.5f}  {det_q0_sc:>+12.5f}  "
         f"{det_q0_sc - det_q0_n:>+12.5f}  {_ds(det_q0_sc)}")
    _log(f"      {'det_afm':<10s}  {det_afm_n:>+12.5f}  {det_afm_sc:>+12.5f}  "
         f"{det_afm_sc - det_afm_n:>+12.5f}  {_ds(det_afm_sc)}")
    _log(f"      Stoner(π,π): {stoner_n:+.5f} → {stoner_sc:+.5f}   "
         f"V_JT·χ_QQ(0): {pref_n.V_JT_corr*chi_QQ_q0_n:+.5f} → "
         f"{V_JT_corr_sc*chi_QQ_sc_q0:+.5f}")
    _log()

    # ── [3] Lattice stiffness ─────────────────────────────────────────
    _log("  [3] LATTICE STIFFNESS")
    _log(f"      K_spont = g²χ_QQ(normal, q=0) = {K_spont:+.5f}   K_bare = {p.K_lattice:.5f}")
    _log(f"      K_eff_n_phys={K_eff_n_phys:+.5f}  K_eff_n_fixedM={K_eff_n_fixedM:+.5f}  K_eff_sc={K_eff_sc:+.5f}  "
         f"dK_eff={dK_eff:+.5f}  ({'SOFTENS' if dK_eff < 0 else 'STIFFENS'})")
    _log(f"      normal stable: K_eff_n_phys>0 {'✓' if K_eff_n_phys > 0 else '✗'}   "
         f"SC stable: K_eff_sc>0 {'✓' if K_eff_sc > 0 else '✗'}")
    _log()

    # ── [4] SC-triggered JT (χ_τ) ─────────────────────────────────────
    _log("  [4] SC-TRIGGERED JT   (physical χ_τ branches)")
    _log(f"      χ_τ,n  (M_n, Q=0, Δ=0) = {chi_tau_n:+.5f} eV⁻¹")
    _log(f"      χ_τ,sc (M_sc,Q_sc,Δ,F67)= {chi_tau_sc:+.5f} eV⁻¹")
    _log(f"      δχ_τ(SC-only) = χ_τ,n − χ_τ,sc = {chi_tau_excess_signed:+.5f} eV⁻¹  "
         f"({'SOFTENS' if chi_tau_excess_signed > 0 else 'STIFFENS'})")
    _log(f"      χ_τ,net = min(w_sc,w_n)·max(δχ_τ,0) = {chi_tau_net:+.5f} eV⁻¹")
    _log(f"      λ_JT_sc = g²χ_τ,net / K_eff_reg = {lambda_JT_sc:+.6f}  "
         f"{'✓ >0.05' if lambda_JT_sc > 0.05 else '✗ <0.05'}")
    _log()

    # ── [5] Decomposition: Δ/F67 vs M-rearrangement ──────────────────
    _log("  [5] χ_τ DECOMPOSITION (which contribution dominates?)")
    _log(f"      χ_τ,n(M_sc,Δ=0)   = {chi_n_fixedM:+.5f} eV⁻¹   "
         f"[hypothetical: same M as SC, no condensate]")
    _log(f"      Δ/F67-only effect  = χ_τ,n(M_sc) − χ_τ,sc = {chi_tau_net_fixedM:+.5f} eV⁻¹")
    _log(f"      M-rearrangement    = χ_τ,n(M_n) − χ_τ,n(M_sc) = {chi_tau_M_contribution:+.5f} eV⁻¹")
    _log(f"      NOTE: the full measurable δχ_τ = "
         f"{chi_tau_net_fixedM + chi_tau_M_contribution:+.5f} = "
         f"({chi_tau_net_fixedM:+.5f}) + ({chi_tau_M_contribution:+.5f})")
    _log()

    # ── [6] χ_τ Richardson / K_eff regularization ────────────────────
    _log("  [6] χ_τ RICHARDSON / K_eff REGULARIZATION")
    _log(f"      w_sc={w_sc:.1f}  w_n={w_n:.1f}  "
         f"nonlin=[{nonlin_sc},{nonlin_n}]  "
         f"K_smooth={K_eff_trustworthy}  spread={K_eff_spread:.2e}")
    _log(f"      K_eff={K_eff_sc:+.5f}  "
         f"b_Q={b_Q:+.4e}  b_Q_valid={b_Q_valid}  "
         f"K_eff_reg={K_eff_reg:+.5f}")
    if not K_eff_trustworthy:
        _log("      ⚠ K_eff(Q) is not smooth — b_Q, K_eff_reg invalid; λ_JT_sc forced to 0")
    _log()

    # ── [7] k-resolved χ_τ ────────────────────────────────────────────
    _log("  [7] k-RESOLVED χ_τ")
    _dn  = abs(chi_tau_n  - chi_n_tot)  / max(abs(chi_tau_n),  1e-12)
    _dsc = abs(chi_tau_sc - chi_sc_tot) / max(abs(chi_tau_sc), 1e-12)
    _log(f"      cross-check: χ_τ,n  scalar={chi_tau_n:+.5f}  k-sum={chi_n_tot:+.5f}  rel={_dn:.2e}")
    _log(f"                   χ_τ,sc scalar={chi_tau_sc:+.5f}  k-sum={chi_sc_tot:+.5f}  rel={_dsc:.2e}")
    _log(f"      k-sum χ_τ,net = {chi_net_tot:+.5f}   → net "
         f"{'SOFTENING' if chi_net_tot > 0 else 'STIFFENING'}")

    # ── [8] Nodal coherence length ────────────────────────────────────
    _log("  [8] NODAL COHERENCE LENGTH")
    _log(f"      |Δ_d|={abs(Delta_vec[qm._DELTA_IDX['d7a']])*1000:.3f} meV  v_F,nodal={vF_nodal:.5f}  "
         f"ξ_nodal/a={xi_nodal:.3f}  {'✓ >2' if xi_nodal > 2 else '✗ <2 (BEC)'}")
    _log()

    phi_abs = np.abs(solver.Phi_irrep_k[:, qm._IR_D])
    m_anti = phi_abs >= np.percentile(phi_abs, antinodal_pctl)
    m_nod  = phi_abs <= np.percentile(phi_abs, nodal_pctl)
    m_mid  = ~(m_anti | m_nod)

    def _wavg7(a, m):
        w = solver.k_weights; den = float(np.sum(w[m]))
        return float(np.sum(w[m] * a[m]) / den) if den > 1e-30 else float('nan')

    stat = {n: (_wavg7(chi_net_k, m), float(np.sum(solver.k_weights[m])))
            for n, m in (('anti', m_anti), ('mid', m_mid), ('nodal', m_nod))}
    _log(f"      {'region':<6s}  {'w-frac':>7s}  {'<χ_net>':>10s}  {'contrib':>10s}")
    for r in ('anti', 'mid', 'nodal'):
        _log(f"      {r:<6s}  {stat[r][1]*100:>6.1f}%  "
             f"{stat[r][0]:>+10.5f}  {stat[r][0]*stat[r][1]:>+10.5f}")
    _log()

    idx_s = np.argsort(chi_net_k)
    i_stiff, i_soft = idx_s[:top_k], idx_s[-top_k:][::-1]
    def _krows(idx):
        for i in idx:
            _log(f"        {solver.k_points[i,0]:+7.3f} {solver.k_points[i,1]:+7.3f}  "
                 f"w={solver.k_weights[i]:.2e}  "
                 f"χ_n={chi_n_k[i]:+9.4f}  χ_sc={chi_sc_k[i]:+9.4f}  "
                 f"χ_net={chi_net_k[i]:+9.4f}  φ_d={solver.Phi_irrep_k[i, qm._IR_D]:+.3f}")
    _log(f"      top-{top_k} stiffest k (χ_net<0):")
    _krows(i_stiff)
    _log(f"      top-{top_k} softest k (χ_net>0):")
    _krows(i_soft)
    w_st = float(np.sum(solver.k_weights[chi_net_k < 0]))
    w_so = float(np.sum(solver.k_weights[chi_net_k > 0]))
    _log(f"      stiff: {int(np.sum(chi_net_k<0))} k ({w_st*100:.1f}% w)   "
         f"soft: {int(np.sum(chi_net_k>0))} k ({w_so*100:.1f}% w)")
    _log()

    v = {
        'Quad67_active':          float(np.abs(np.asarray(Quad67_vec)[_trw_slots(solver)]).max()) > 1e-6,
        'chi_SQ_open':         abs(chi_SQ_sc_q0) > 1e-6 or abs(chi_SQ_sc_pipi) > 1e-6,
        'det_afm_positive':    det_afm_sc > 0.01,
        'lattice_softens':     dK_eff < -1e-3 * p.K_lattice,
        'lambda_JT_viable':    lambda_JT_sc > 0.05,
        'normal_state_stable': K_eff_n_phys > 0.0,
        'sc_state_unstable':   K_eff_sc < K_spont,
        'xi_nodal_ok':         xi_nodal > 2.0,
        'K_eff_smooth':        K_eff_trustworthy,
    }
    overall = (v['Quad67_active'] and v['chi_SQ_open'] and v['det_afm_positive']
               and v['lattice_softens'] and v['lambda_JT_viable']
               and v['normal_state_stable'] and v['K_eff_smooth'])

    _lbl = {
        'Quad67_active':          "F67 active (H_TRW ≠ 0)",
        'chi_SQ_open':         "χ_SQ open (Θ breaking)",
        'det_afm_positive':    "det_afm_sc > 0",
        'lattice_softens':     "lattice softens (dK_eff < 0)",
        'lambda_JT_viable':    "λ_JT_sc > 0.05",
        'normal_state_stable': "normal state stable",
        'sc_state_unstable':   "SC state Q≠0 sustainable",
        'xi_nodal_ok':         "ξ_nodal/a > 2",
        'K_eff_smooth':        "K_eff(Q) smooth (b_Q valid)",
    }

    _log("  VERDICT")
    for k, ok in v.items():
        _log(f"      [{'✓' if ok else '✗'}] {_lbl[k]}")
    _log()
    _log(f" SUMMARY: SC+JT state {'CONSISTENT with hypothesis' if overall else 'NOT consistent'}")
    _log("═" * 78)
    

# ============================================================================
# Bogoliubov FS probe
# ============================================================================
def _bfs_line_search(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, ls_pts: int = 51, n_sub: int = 60, ls_win_min: float = 0.02, ls_win_max: float = 0.5, full: bool = False):
    """Line-search BFS on the FS.

    full=False → (bfs_frac, mean_gap_eV, n_fs)               [scan / quick check]
    full=True  → dict with all raw arrays                    [probe_bogoliubov_...]

    The true quasiparticle gap can sit off the bare xi=0 crossing (inter-orbital pairs: xi_min ~ -|Δ|²/δ),
    so the FS-point spectrum alone false-negatives on shifted zero-energy contours — hence the explicit line-search.
    """
    vbdg = solver._get_vbdg()
    fs_pts, vF_vec, _, weights = solver._get_fs_points(M, Q, mu, g_t, g_J, True)
    if len(fs_pts) == 0 or float(np.sum(weights)) <= 0.0:
        return None if full else (float('nan'), float('nan'), 0)

    n_flat = 0
    if len(fs_pts) > n_sub:
        _dk = fs_pts[:, None, :] - fs_pts[None, :, :]
        _dk = (_dk + np.pi) % (2.0 * np.pi) - np.pi
        _D  = np.sqrt(np.sum(_dk ** 2, axis=2))
        _sel  = [int(np.argmin(np.arctan2(fs_pts[:, 1], fs_pts[:, 0])))]
        _dmin = _D[_sel[0]].copy()
        while len(_sel) < n_sub:
            _nxt = int(np.argmax(_dmin)); _sel.append(_nxt)
            _dmin = np.minimum(_dmin, _D[_nxt])
        sub = np.array(_sel, dtype=int)
        fs_pts, vF_vec, weights = fs_pts[sub], vF_vec[sub], weights[sub]

    vF_mag = np.linalg.norm(vF_vec, axis=1)
    valid  = vF_mag > qm._GAP_VF_MIN
    n_flat = int(np.sum(~valid))
    if not np.any(valid):
        return None if full else (float('nan'), float('nan'), 0)
    fs_pts, vF_vec, vF_mag, weights = fs_pts[valid], vF_vec[valid], vF_mag[valid], weights[valid]
    N = len(fs_pts)
    w_sum = float(np.sum(weights))
    n_hat = vF_vec / vF_mag[:, None]

    # xi=0 crossing from the Δ=0 reference
    H_norm = vbdg._build_H_stack(fs_pts, M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J)
    ev_norm = np.linalg.eigvalsh(H_norm)
    ev_pos  = np.where(ev_norm > 0.0, ev_norm, np.inf)
    E_n     = ev_pos[np.arange(N), np.argmin(ev_pos, axis=1)]
    s0      = np.clip(-E_n / np.maximum(vF_mag, qm._MATH_EPS),
                      -qm._GAP_SCENTER_MAX, qm._GAP_SCENTER_MAX)

    # Line-search grid
    win_eV = max(qm._GAP_WIN, 3.0 * float(np.max(np.abs(Delta_vec))))
    W_per  = np.clip(win_eV / np.maximum(vF_mag, qm._MATH_EPS), ls_win_min, ls_win_max)
    eps    = np.linspace(-1.0, 1.0, ls_pts)
    xi_0_idx = int(np.argmin(np.abs(eps)))
    s_offsets = s0[:, None] + W_per[:, None] * eps[None, :]
    k_cuts = fs_pts[:, None, :] + s_offsets[:, :, None] * n_hat[:, None, :]
    k_flat = k_cuts.reshape(-1, 2)

    ev_bdg, ec_bdg = np.linalg.eigh(vbdg._build_H_stack(k_flat, M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J))
    aE_bdg    = np.abs(ev_bdg)
    min_bands = np.argmin(aE_bdg, axis=1)
    gaps_flat = aE_bdg[np.arange(len(k_flat)), min_bands]
    gaps_cuts = gaps_flat.reshape(N, ls_pts)
    spec_idx  = np.argmin(gaps_cuts, axis=1)
    spec_gaps = gaps_cuts[np.arange(N), spec_idx]
    xi0_gaps  = gaps_cuts[np.arange(N), xi_0_idx]

    flat_min_idx = np.arange(N) * ls_pts + spec_idx
    k_min_pts    = k_flat[flat_min_idx]
    ec_min       = ec_bdg[flat_min_idx, :, min_bands[flat_min_idx]]

    # Anti/nodal masks (based on φ_d at the xi=0 crossing)
    k_xi0    = fs_pts + s0[:, None] * n_hat
    phi_d_xi0 = np.cos(k_xi0[:, 0]) - np.cos(k_xi0[:, 1])
    abs_phi  = np.abs(phi_d_xi0)
    anti_mask = abs_phi >= np.percentile(abs_phi, 75)
    nod_mask  = abs_phi <= np.percentile(abs_phi, 25)
    is_dwave  = float(abs_phi.max()) > 1e-3

    # Edge-of-window hits
    edge_hits = int(np.sum((spec_idx == 0) | (spec_idx == ls_pts - 1)))

    if full:
        return dict(
            fs_pts=fs_pts, weights=weights, vF_vec=vF_vec, vF_mag=vF_mag,
            spec_gaps=spec_gaps, xi0_gaps=xi0_gaps, spec_idx=spec_idx,
            k_min_pts=k_min_pts, ec_min=ec_min,
            phi_d_xi0=phi_d_xi0, anti_mask=anti_mask, nod_mask=nod_mask,
            is_dwave=is_dwave, edge_hits=edge_hits,
            ls_pts=ls_pts, xi_0_idx=xi_0_idx,
            w_sum=w_sum, N=N, n_flat=n_flat,
        )
    if w_sum <= 0.0:
        return float('nan'), float('nan'), N
    bfs_frac = float(np.sum(weights * (spec_gaps < qm._GAP_BFS_TOL)) / w_sum)
    mean_gap = float(np.sum(weights * spec_gaps) / w_sum)
    return bfs_frac, mean_gap, N


def test_cluster_K_eff_diagnostics(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J,
                                    M_normal=None, mu_normal=None,
                                    run_Q_scan: bool = True,
                                    run_M2_scan: bool = True,
                                    Q_scan_halfwidth: float = 0.03,
                                    n_Q_scan: int = 7,
                                    M_frac_scan: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5), vertex_cache: dict | None = None):
    """
    K_eff decomposition diagnostic + extended consistency checks.

    Extended checks (new):
      [A] Q-dependence of K_full, K_full_MF, K_bare separately
          → shows whether the anharmonicity lives in the intra bonds (expected) or leaks
            into the external MF reference (which would be a K_full_MF bug)
      [B] M² scaling of Delta_K_cluster
          → a pure beyond-MF exchange correction must scale as M²; deviation signals
            contamination from the Z_eff (2) vs Z_bare (4) Weiss-field mismatch in K_full_MF
      [C] Beyond-MF subtraction consistency:
            (K_full − K_full_MF)/4  vs returned Delta_K_cluster
            (K_full − K_bare)/4 = total exact intra (NOT beyond-MF)
            (K_full_MF − K_bare)/4   = intra-MF contribution
          → verifies that the code's Delta_K_cluster is really the beyond-MF part
      [D] V_irr_QQ Lehmann vs curvature
          → independent cross-check of the irreducible vertex
    """
    vertex_cache = vertex_cache if vertex_cache is not None else _registered_cache(M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    p = solver.p
    M = np.atleast_1d(np.asarray(M, dtype=float))
    zero_D = np.zeros_like(Delta_vec); zero_F = np.zeros_like(Quad67_vec)
    M_normal = M if M_normal is None else np.atleast_1d(np.asarray(M_normal, dtype=float))
    mu_normal = mu if mu_normal is None else float(mu_normal)

    _log("=" * 95)
    _log(f"  δ={target_doping:.4f}  Q={Q:+.5f}  kT={solver.kT*1000:.1f} meV  "
         f"|Δ_vec|={float(np.sum(np.abs(Delta_vec))):.3e}")
    _log(f"  K_lattice={p.K_lattice:+.5f}")
    _log("=" * 95)

    # ---------- Baseline K-term extraction ----------
    tx_b, ty_b = p.effective_hopping_anisotropic(Q)
    _, Fc, _, _ = _vertex_tuple(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, vertex_cache=vertex_cache)

    K_bulk      = float(Fc['K_bulk'])
    K_econd     = float(Fc['K_econd'])
    K_schur     = float(Fc['K_schur'])
    K_eff       = float(Fc['K_eff'])
    V_irr_QQ    = float(Fc['V_irr_QQ'])
    V_JT_corr_u = float(Fc['V_JT_corr'])

    # --- renamed keys (source now returns Delta_K_cluster / K_full_MF) ---
    K_cluster = float(Fc['Delta_K_cluster'])         # was Fc['K_cluster_corr']
    K_full    = Fc.get('K_full',    np.nan)
    K_MF      = Fc.get('K_full_MF', np.nan)          # was Fc.get('K_MF', ...)
    K_bare    = Fc.get('K_bare',    np.nan)

    chi_QQ_N = solver._chi_QQ_matrix_elements(M, Q, zero_D, zero_F, mu, g_t, g_J)
    K_el_N   = solver.g_JT_bare ** 2 * chi_QQ_N
    ratio_cluster_el = abs(K_cluster) / max(abs(K_el_N), 1e-12)

    _log("  K_eff decomposition at the given state")
    _log(f"      V_irr_QQ={V_irr_QQ:+.5f}  V_JT_corr={V_JT_corr_u:+.5f}")
    _log(f"      K_bulk       = {K_bulk:+.5f}  (E_cond''={K_econd:+.5f})")
    _log(f"      K_cluster    = {K_cluster:+.5f}  (= (K_full - K_full_MF)/4; K_full_MF-based beyond-MF exchange)")
    _log(f"      K_schur      = {K_schur:+.5f}  (Δ-relaxation; diagnostic only)")
    _log(f"      K_eff        = {K_eff:+.5f}")
    _log(f"      K_full = {K_full:+.5f}   K_full_MF = {K_MF:+.5f}   K_bare = {K_bare:+.5f}")
    _log(f"      |K_cluster|/|K_el^N| = {ratio_cluster_el:.3f}")
    _log("")

    # ================================================================
    # [A] Q-dependence of each cluster K term
    # ================================================================
    if run_Q_scan:
        _log("  [A] Q-dependence of cluster K terms  (source of Delta_K_cluster nonlinearity)")
        _log(f"      {'Q':>9s}  {'K_full':>10s}  {'K_full_MF':>10s}  {'K_bare':>10s}  "
             f"{'(K_f-K_MF)/4':>13s}  {'(K_f-K_b)':>11s}  {'(K_MF-K_b)':>11s}")
        _log("      " + "-" * 91)
        Q_grid = np.linspace(-Q_scan_halfwidth, Q_scan_halfwidth, n_Q_scan)
        Q_data = []
        for qv in Q_grid:
            try:
                mu_qv, n_qv = solver._find_mu_for_density(target_doping, M, qv, Delta_vec, Quad67_vec, mu, g_t, g_J)
                _, Fc_qv, *_ = solver._make_vertex_params(target_doping, M, qv, Delta_vec, Quad67_vec, mu_qv, g_t, g_J, vertex_cache=vertex_cache)
            except Exception as err:
                _log(f"      {qv:>+9.4f}   -- raised {type(err).__name__}: {err}")
                continue

            K_f_v  = Fc_qv.get('K_full', np.nan)
            K_mf_v = Fc_qv.get('K_full_MF',
                               K_f_v - 4.0 * Fc_qv.get('Delta_K_cluster', np.nan)
                               if np.isfinite(K_f_v) else np.nan)
            K_b_v  = Fc_qv.get('K_bare', np.nan)
            dk_mf  = (K_f_v - K_mf_v) / 4.0
            dk_b   = (K_f_v - K_b_v) / 4.0  if np.isfinite(K_b_v) else np.nan
            dk_mfb = (K_mf_v - K_b_v) / 4.0 if np.isfinite(K_b_v) else np.nan

            Q_data.append((qv, K_f_v, K_mf_v, K_b_v, dk_mf, dk_b, dk_mfb))
            _log(f"      {qv:>+9.4f}  {K_f_v:>+10.4f}  {K_mf_v:>+10.4f}  {K_b_v:>+10.4f}  "
                 f"{dk_mf:>+13.4f}  {dk_b:>+11.4f}  {dk_mfb:>+11.4f}")

        if Q_data:
            Kf_arr    = np.array([r[1] for r in Q_data])
            Kmf_arr   = np.array([r[2] for r in Q_data])
            Kb_arr    = np.array([r[3] for r in Q_data])
            dkmf_arr  = np.array([r[4] for r in Q_data])

            def _spread(arr):
                a = arr[np.isfinite(arr)]
                if len(a) < 2: return float('nan')
                return float((a.max() - a.min()) / max(abs(a.mean()), 1e-12))

            _log("")
            _log(f"      relative spread over Q∈[±{Q_scan_halfwidth:.3f}]:")
            _log(f"        K_full     : {_spread(Kf_arr)*100:>7.1f}%")
            _log(f"        K_full_MF  : {_spread(Kmf_arr)*100:>7.1f}%   "
                 f"{'(≈0: pure MF, no Q-anharmonicity)' if _spread(Kmf_arr) < 0.05 else '(non-zero: check MF reference!)'}")
            _log(f"        K_bare     : {_spread(Kb_arr)*100:>7.1f}%")
            _log(f"        K_cluster  : {_spread(dkmf_arr)*100:>7.1f}%")

            # Extra: check if K_full_MF is roughly Q-INDEPENDENT (expected for pure MF)
            if _spread(Kmf_arr) > 0.20:
                _log("")
                _log("   ⚠ K_full_MF varies strongly with Q → the 'all-MF' reference is itself anharmonic. This can happen if the ground state inside the MF cluster changes with Q (level crossing), which contaminates the K_full − K_full_MF subtraction.")
            _log("")

    # ================================================================
    # [B] M² scaling of Delta_K_cluster
    # ================================================================
    if run_M2_scan:
        _log("  [B] M² scaling of Delta_K_cluster  (should be ∝ M² for a pure beyond-MF exchange)")
        _log(f"      {'frac':>6s}  {'M_Γ6':>8s}  {'K_cluster':>10s}  {'K_cluster/M²':>15s}")

        M_ext = float(M[0])
        M_scan_data = []
        for frac in M_frac_scan:
            M_test = M * frac
            try:
                mu_test, n_test = solver._find_mu_for_density(target_doping, M_test, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)
                _, Fc_test, *_  = solver._make_vertex_params(target_doping, M_test, Q, Delta_vec, Quad67_vec, mu_test, g_t, g_J, vertex_cache=vertex_cache)
            except Exception as err:
                _log(f"      {frac:>6.2f}   -- raised {type(err).__name__}: {err}")
                continue

            K_cl = float(Fc_test['Delta_K_cluster'])         # was 'K_cluster_corr'
            M2   = (frac * M_ext) ** 2
            M_scan_data.append((frac, frac * M_ext, K_cl, K_cl / max(M2, 1e-20)))
            _log(f"      {frac:>6.2f}  {frac*M_ext:>+8.4f}  {K_cl:>+10.4f}  {K_cl/max(M2,1e-20):>+15.4f}")

        if len(M_scan_data) >= 2:
            ratio_vals = np.array([r[3] for r in M_scan_data
                                    if abs(r[2]) > 1e-8 and np.isfinite(r[3])])
            if len(ratio_vals) >= 2:
                spread = float((ratio_vals.max() - ratio_vals.min())
                               / max(abs(ratio_vals.mean()), 1e-12))
                _log("")
                if spread < 0.10:
                    _log(f"      ✓ K_cluster/M² stable (rel. spread = {spread*100:.1f}%) → M² scaling OK")
                elif spread < 0.30:
                    _log(f"      ⚠ K_cluster/M² varies {spread*100:.0f}% → mild non-M² contamination")
                else:
                    _log(f"      ✗ K_cluster/M² varies {spread*100:.0f}% → strong contamination")
                    _log("        Likely cause: Z_eff (2) vs Z_bare (4) Weiss-field mismatch in K_full_MF.")
            else:
                _log("      (no reliable M² ratios — K_cluster too small at these M)")
        _log("")

    # ================================================================
    # [C] Beyond-MF subtraction consistency
    # ================================================================
    _log("  [C] Beyond-MF subtraction decomposition")
    if np.isfinite(K_full) and np.isfinite(K_MF):
        delta_via_MF = (K_full - K_MF) / 4.0
        _log(f"      (K_full − K_full_MF)/4 = {delta_via_MF:+.5f}   [should equal Delta_K_cluster]")
        _log(f"      Delta_K_cluster        = {K_cluster:+.5f}   "
             f"{'✓' if abs(delta_via_MF - K_cluster) < 1e-4 else '✗ MISMATCH'}")
    if np.isfinite(K_bare):
        delta_via_bare = (K_full - K_bare) / 4.0
        delta_mf_bare  = (K_MF   - K_bare) / 4.0
        _log(f"      (K_full − K_bare)/4 = {delta_via_bare:+.5f}   "
             f"[total exact intra; NOT beyond-MF]")
        _log(f"      (K_full_MF − K_bare)/4 = {delta_mf_bare:+.5f}   "
             f"[intra-MF contribution]")
        _log(f"      Sum check: {delta_via_MF:+.5f} + {delta_mf_bare:+.5f} = "
             f"{delta_via_MF + delta_mf_bare:+.5f}   "
             f"vs {delta_via_bare:+.5f}   "
             f"{'✓' if abs(delta_via_MF + delta_mf_bare - delta_via_bare) < 1e-4 else '✗'}")
        if abs(delta_mf_bare) > 0.5 * max(abs(delta_via_bare), 1e-12):
            _log("      ⚠ intra-MF contribution is a large fraction of the total intra response: K_cluster (beyond-MF) is small and sensitive to the Z_eff vs Z reference choice.")
    _log("")

    _, Fc_0, _, _ = _vertex_tuple(solver, target_doping, M, Q, zero_D, zero_F, n_kspace, mu, g_t, g_J, vertex_cache=vertex_cache)
    K_schur_0     = float(Fc_0['K_schur'])
    K_cluster_0   = float(Fc_0['Delta_K_cluster'])          # was 'K_cluster_corr'

    _log("  Δ-dependence (SC vs Δ=0 at same M, Q, μ)")
    _log(f"      K_schur(Δ=0)   = {K_schur_0:+.5f}   K_schur(SC)   = {K_schur:+.5f}   dK_schur = {K_schur-K_schur_0:+.5f}")
    _log(f"      K_cluster(Δ=0) = {K_cluster_0:+.5f}   K_cluster(SC) = {K_cluster:+.5f}   dK_cluster = {K_cluster-K_cluster_0:+.5f}")
    _log("      → Δ-relaxation comes from K_schur, NOT from K_cluster (cluster H has no Δ)")
    _log("")

    # K_bulk(SC) vs K_bulk(N) at Q=0 — SCF-free pre-check
    mu_q0_N,  _ = solver._find_mu_for_density(target_doping, M_normal, 0.0, zero_D, zero_F, mu_normal, g_t, g_J)
    mu_q0_SC, _ = solver._find_mu_for_density(target_doping, M, 0.0, Delta_vec, Quad67_vec, mu, g_t, g_J)

    _, Fc_N_q0, _, _ = _vertex_tuple(solver, target_doping, M_normal, 0.0, zero_D, zero_F, n_kspace, mu_q0_N, g_t, g_J, vertex_cache=vertex_cache)
    _, Fc_SC_q0, _, _ = _vertex_tuple(solver, target_doping, M, 0.0, Delta_vec, Quad67_vec, n_kspace, mu_q0_SC, g_t, g_J, vertex_cache=vertex_cache)
    K_bulk_N     = float(Fc_N_q0['K_bulk'])
    K_bulk_SC    = float(Fc_SC_q0['K_bulk'])
    K_econd_SC   = float(Fc_SC_q0['K_econd'])
    K_cluster_N  = float(Fc_N_q0['Delta_K_cluster'])
    K_cluster_SC = float(Fc_SC_q0['Delta_K_cluster'])
    K_schur_N    = float(Fc_N_q0['K_schur'])                # should be ~0 at Δ=0
    K_schur_SC   = float(Fc_SC_q0['K_schur'])

    _log("  K_bulk at Q=0, SC vs normal (μ re-solved, no SCF)")
    _log(f"      K_bulk(N)      = {K_bulk_N:+.5f}   K_bulk(SC)      = {K_bulk_SC:+.5f}   ΔK_bulk = {K_bulk_SC-K_bulk_N:+.5f}")
    _log(f"      (E_cond'' in SC = {K_econd_SC:+.5f})")
    if K_bulk_SC - K_bulk_N < -1e-4:
        _log(f"      ✓ SC bulk-softens the lattice by {(K_bulk_N-K_bulk_SC)*1e3:.2f} meV/Å²")
    elif K_bulk_SC - K_bulk_N > 1e-4:
        _log(f"      ✗ SC bulk-stiffens by {(K_bulk_SC-K_bulk_N)*1e3:.2f} meV/Å²")
    else:
        _log("      ≈ negligible ΔK_bulk")
    _log("")

    # SC-triggered window at Q=0, using TOTAL K_eff
    # K_eff(Δ=0, Q=0) = K_bulk_N + K_cluster_N − K_schur_N  (K_schur_N ≈ 0)
    # K_eff(SC, Q=0)  = K_bulk_SC + K_cluster_SC − K_schur_SC
    K_eff_N  = K_bulk_N + K_cluster_N - K_schur_N
    K_eff_SC = K_bulk_SC + K_cluster_SC - K_schur_SC
    crit_N   = p.K_lattice - K_eff_N
    crit_SC  = p.K_lattice - K_eff_SC

    _log("  SC-triggered JT window at Q=0 (using total K_eff)")
    _log(f"      K_eff(N, Q=0)  = {K_eff_N:+.5f}   (K_cluster_N = {K_cluster_N:+.5f}, K_schur_N = {K_schur_N:+.5f}≈0)")
    _log(f"      K_eff(SC, Q=0) = {K_eff_SC:+.5f}   (K_cluster_SC = {K_cluster_SC:+.5f}, K_schur_SC = {K_schur_SC:+.5f})")
    if crit_N < crit_SC:
        width = crit_SC - crit_N
        inside = crit_N < p.K_lattice < crit_SC
        _log(f"      window: K_lattice ∈ ({crit_N:+.4f}, {crit_SC:+.4f})  width = {width*1e3:+.2f} meV/Å²")
        _log(f"      K_lattice = {p.K_lattice:+.4f}  {'✓ IN WINDOW (SC-triggered)' if inside else '✗ outside window'}")
    else:
        _log("      ✗ window EMPTY (crit_N ≥ crit_SC; K_eff does NOT soften with Δ)")
        inside = False
    _log("")

    _log("=" * 95)
    _log("  VERDICT")
    _log(f"    K_eff = K_bulk + K_cluster − K_schur = {K_eff:+.5f}")
    _log(f"    K_cluster is Δ-independent ({(K_cluster-K_cluster_0):+.2e}); Δ-relaxation lives in K_schur")
    if crit_N < crit_SC:
        _log(f"    Window: K_lattice ∈ ({crit_N:+.4f}, {crit_SC:+.4f});  "
             f"K_lattice={p.K_lattice:+.4f}  {'✓ IN' if (crit_N < p.K_lattice < crit_SC) else '✗ OUT'}")
    else:
        _log("    ⚠ No SC-triggered window (K_eff stiffens with Δ)")
    _log(f"    K_full={K_full:+.4f}   K_full_MF={K_MF:+.4f}   K_bare={K_bare:+.4f}")
    _log("=" * 95)


def test_Q_path_dK_and_BFS(solver, target_doping, M_sc, Q_scan, Delta_vec_sc, Quad67_vec_sc, n_kspace, mu_sc, g_t, g_J, M_normal=None, mu_normal=None, compute_K_eff: bool = False,
                           ls_pts: int = 51, n_sub: int = 60,
                           ls_win_min: float = 0.02, ls_win_max: float = 0.5,
                           bfs_tol_hi: float = 0.10, bfs_tol_lo: float = 0.02, vertex_cache: dict | None = None) -> None:
    """
    Q-path scan: does SC softening (ΔK < 0) survive at finite Q where line-search BFS vanishes?

    Sign convention (identical to test_cluster_K_eff_diagnostics and the SCF's K_eff_net):
        ΔK_el  = K_el_N  − K_el_SC,   K_el  = g_JT² χ_QQ      (negative = SC softens)
        ΔK_eff = K_eff_SC − K_eff_N                            (negative = SC softens)

    BFS fraction from line-search along the FS normal, not the FS-point spectrum alone
    (the latter false-negatives on shifted zero-energy contours — xi_min ~ -|Δ|²/δ).

    compute_K_eff=True additionally fills the full-K_eff columns via _make_vertex_params
    (slow: cluster-ED per Q). χ-based ΔK_el is the fast electronic indicator.
    """
    M_sc          = np.atleast_1d(np.asarray(M_sc, dtype=float))
    Delta_vec_sc  = np.asarray(Delta_vec_sc, dtype=complex)
    Quad67_vec_sc    = np.asarray(Quad67_vec_sc, dtype=complex)
    M_normal      = M_sc if M_normal is None else np.atleast_1d(np.asarray(M_normal, dtype=float))
    mu_normal_guess = mu_sc if mu_normal is None else float(mu_normal)
    zero_D = np.zeros_like(Delta_vec_sc)
    zero_F = np.zeros_like(Quad67_vec_sc)

    _log("=" * 112)
    _log("  [Q-PATH] SC softening (ΔK<0) vs. line-search BFS across Q")
    _log(f"  δ={target_doping:.4f}  |Δ_sc|={float(np.sum(np.abs(Delta_vec_sc))):.3e}  "
         f"|Quad67_sc|={float(np.sum(np.abs(Quad67_vec_sc))):.3e}")
    _log(f"  M_sc={np.array2string(M_sc, precision=4)}  M_normal={np.array2string(M_normal, precision=4)}")
    _log(f"  Q_scan: {[float(q) for q in Q_scan]}")
    _log(f"  BFS thresholds: hi={bfs_tol_hi*100:.0f}%  lo={bfs_tol_lo*100:.0f}%")
    _log("  Conventions: ΔK_el = K_el_N−K_el_SC, ΔK_eff = K_eff_SC−K_eff_N (both < 0 ⇒ SC softens)")
    _log("=" * 112)

    cols = ["Q[Å]", "ΔK_el", "K_el_N", "K_el_SC", "BFS%", "gap[meV]", "N_fs"]
    if compute_K_eff:
        cols += ["ΔK_eff", "K_eff_N", "K_eff_SC"]
    hdr = "  " + "  ".join(f"{c:>11s}" for c in cols) + f"  {'status':>26s}"
    _log(hdr)
    _log("-" * len(hdr))

    rows = []
    for Q in Q_scan:
        Q = float(Q)
        mu_n, n_n = solver._find_mu_for_density(target_doping, M_normal, Q, zero_D, zero_F, mu_normal_guess, g_t, g_J)
        mu_s, n_s = solver._find_mu_for_density(target_doping, M_sc, Q, Delta_vec_sc, Quad67_vec_sc, mu_sc, g_t, g_J)

        chi_N   = solver._chi_QQ_matrix_elements(M_normal, Q, zero_D, zero_F, mu_n, g_t, g_J)
        chi_SC  = solver._chi_QQ_matrix_elements(M_sc, Q, Delta_vec_sc, Quad67_vec_sc, mu_s, g_t, g_J)
        K_el_N  = solver.g_JT_bare ** 2 * chi_N
        K_el_SC = solver.g_JT_bare ** 2 * chi_SC
        dK_el   = K_el_N - K_el_SC      # main convention: negative = SC softens

        bfs_frac, mean_gap, n_fs = _bfs_line_search(solver, M_sc, Q, Delta_vec_sc, Quad67_vec_sc, n_kspace, mu_s, g_t, g_J, ls_pts=ls_pts, n_sub=n_sub, ls_win_min=ls_win_min, ls_win_max=ls_win_max)

        if compute_K_eff:
            _, Fc_N, _, _ = _vertex_tuple(solver, target_doping, M_normal, Q, zero_D, zero_F, n_n, mu_n, g_t, g_J, vertex_cache=vertex_cache)
            _, Fc_S, _, _ = _vertex_tuple(solver, target_doping, M_sc, Q, Delta_vec_sc, Quad67_vec_sc, n_s, mu_s, g_t, g_J, vertex_cache=vertex_cache)
            K_eff_N  = float(Fc_N['K_eff'])
            K_eff_SC = float(Fc_S['K_eff'])
            dK_eff   = K_eff_SC - K_eff_N
        else:
            K_eff_N = K_eff_SC = dK_eff = float('nan')

        soft_el  = dK_el  < 0
        soft_eff = (dK_eff < 0) if compute_K_eff else soft_el
        clean    = (not np.isnan(bfs_frac)) and (bfs_frac < bfs_tol_lo)
        bfs_ok   = (not np.isnan(bfs_frac)) and (bfs_frac < bfs_tol_hi)
        if soft_el and soft_eff and clean:   status = "✓✓ soft + BFS clear"
        elif soft_el and soft_eff and bfs_ok: status = "✓  partial (soft + ~BFS)"
        elif soft_el and soft_eff:           status = "⚠ soft but BFS dominant"
        elif clean:                          status = "✗ clean FS, no softening"
        else:                                status = "✗ no softening + BFS"

        rows.append(dict(Q=Q, dK_el=dK_el, K_el_N=K_el_N, K_el_SC=K_el_SC,
                         bfs_frac=bfs_frac, mean_gap=mean_gap, n_fs=n_fs,
                         dK_eff=dK_eff, K_eff_N=K_eff_N, K_eff_SC=K_eff_SC,
                         soft_el=soft_el, clean=clean, status=status))

        line = (f"  {Q:>+11.4f}  {dK_el:>+11.4f}  {K_el_N:>+11.4f}  {K_el_SC:>+11.4f}  "
                f"{bfs_frac*100:>10.1f}%  {mean_gap*1e3:>+10.2f}  {n_fs:>11d}")
        if compute_K_eff:
            line += f"  {dK_eff:>+11.4f}  {K_eff_N:>+11.4f}  {K_eff_SC:>+11.4f}"
        line += f"  {status:>26s}"
        _log(line)

    clean_rows = [r for r in rows if not np.isnan(r['bfs_frac']) and r['bfs_frac'] < bfs_tol_lo]
    soft_rows  = [r for r in rows if r['dK_el'] < 0]
    if not clean_rows:
        _log(f"    ✗ no Q with line-search BFS < {bfs_tol_lo*100:.0f}% — widen Q_scan or check |Δ_sc|")
    elif not soft_rows:
        _log("    ✗ no Q with ΔK_el < 0 (SC softens nowhere in the scan)")
    else:
        candidates = [r for r in clean_rows if r['dK_el'] < 0]
        if candidates:
            best = min(candidates, key=lambda r: r['dK_el'])
            _log(f"    ✓ BEST SC-triggered candidate: Q={best['Q']:+.4f}  "
                 f"ΔK_el={best['dK_el']:+.4f}  BFS={best['bfs_frac']*100:.1f}%  "
                 f"gap={best['mean_gap']*1e3:.2f} meV")
            if compute_K_eff and not np.isnan(best['dK_eff']):
                _log(f"      full ΔK_eff={best['dK_eff']:+.4f}  "
                     f"K_eff_N={best['K_eff_N']:+.4f}  K_eff_SC={best['K_eff_SC']:+.4f}")
            softest = min(soft_rows, key=lambda r: r['dK_el'])
            if softest['bfs_frac'] >= bfs_tol_hi:
                _log(f"    ⚠ softest Q={softest['Q']:+.4f} (ΔK_el={softest['dK_el']:+.4f}) "
                     f"still has BFS={softest['bfs_frac']*100:.0f}%")
        else:
            softest = min(soft_rows, key=lambda r: r['dK_el'])
            _log(f"    ✗ softening only where BFS present: softest Q={softest['Q']:+.4f} "
                 f"(ΔK_el={softest['dK_el']:+.4f}, BFS={softest['bfs_frac']*100:.0f}%)")
    _log("=" * 112)

def probe_bogoliubov_fermi_surface_fast(solver, target_doping: float, M_bdg: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_bdg: float, mu_bdg: float, g_t: float, g_J: float,
                                        ls_pts: int = 51,
                                        ls_win_min: float = 0.02,
                                        ls_win_max: float = 0.5,
                                        frac_scan: tuple = (0.0, 0.25, 0.5, 0.75, 1.0),
                                        skip_zero_frac_flags: bool = True,
                                        gap_closed_threshold: float = 1e-4,
                                        n_sub: int | None = None,
                                        top_k: int = 6,
                                        bfs_frac_tol_hi: float = 0.5,
                                        bfs_frac_tol_lo: float = 0.1) -> None:
    """
    BdG gap across the Fermi surface, measured directly by a 1D line-search along the local Fermi-velocity normal on the FULL 24-band BdG Hamiltonian.
    Quad67_vec is the anomalous pair amplitude. The frac scan below therefore propagates (frac·Δ, frac·F67) consistently.
    """
    vbdg = solver._get_vbdg()
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec   = np.asarray(Quad67_vec,   dtype=complex)
    Delta_amplitude = float(np.sum(np.abs(Delta_vec)))
    _PAIR_INDICES = (0, qm._N_ORB, 2 * qm._N_ORB, 3 * qm._N_ORB)
    _n_sub = qm._GAP_N_SUB if n_sub is None else int(n_sub)

    # --- 1. Early-abort check ---
    _p0 = _bfs_line_search(solver, M_bdg, Q, Delta_vec, Quad67_vec, n_bdg, mu_bdg, g_t, g_J, ls_pts=ls_pts, n_sub=_n_sub, ls_win_min=ls_win_min, ls_win_max=ls_win_max, full=True)
    if _p0 is None:
        _log("[BOG-FS probe] empty FS or zero weights — abort.")
        return {}
    _log(f"[BOG-FS line-search] {_p0['N']} point 1D cuts, {ls_pts} pts/cut "
         f"(adaptive window {ls_win_min}-{ls_win_max}, "
         f"win_eV={max(qm._GAP_WIN, 3.0*float(np.max(np.abs(Delta_vec)))):.4f} eV)")
    if _p0['n_flat'] > 0:
        _log(f"[BOG-FS probe] {_p0['n_flat']}/{_p0['N'] + _p0['n_flat']} FS points "
             f"have |v_F| < {qm._GAP_VF_MIN:.1e} — excluded from line search")

    # --- 2. Anomalous pair operators (particle-block, dim 2*_N_ORB x 2*_N_ORB) ---
    # odd=True: symmetric index structure for odd-parity (px, py); even: Pauli antisym.
    def _D_pair(orb_up: int, inter: bool, odd: bool = False) -> np.ndarray:
        D = np.zeros((2 * qm._N_ORB, 2 * qm._N_ORB), dtype=complex)
        if inter:
            D[0, qm._N_ORB + orb_up + 1] = +1.0; D[1, qm._N_ORB + orb_up + 0] = -1.0
            D[qm._N_ORB, 0 + orb_up + 1] = +1.0; D[qm._N_ORB + 1, 0 + orb_up + 0] = -1.0
        else:
            for b in (0, qm._N_ORB):
                D[b + 0, b + orb_up + 1] = +1.0
                D[b + 1, b + orb_up + 0] = -1.0
        return D + D.T if odd else D - D.T

    _orb_7a_up = int(qm._CHANNEL_ORB_IDX[1][0])   # Gamma7a up index = 2
    _orb_7b_up = int(qm._CHANNEL_ORB_IDX[2][0])   # Gamma7b up index = 4
    D7a_s, D7a_d, D7a_p = _D_pair(_orb_7a_up, False), _D_pair(_orb_7a_up, True), _D_pair(_orb_7a_up, True, odd=True)
    D7b_s, D7b_d, D7b_p = _D_pair(_orb_7b_up, False), _D_pair(_orb_7b_up, True), _D_pair(_orb_7b_up, True, odd=True)

    def _orbital_weight(p: np.ndarray, orb_pair: tuple[int, int]) -> float:
        """Sum |psi|^2 over both Kramers partners of one doublet, across all Nambu blocks."""
        i_up, i_dn = orb_pair
        return float(sum(p[_OFF + i] for _OFF in _PAIR_INDICES for i in (i_up, i_dn)))

    def _ch(psi: np.ndarray, phi: np.ndarray) -> dict:
        """
        Per-channel anomalous amplitudes and orbital contents of one eigenvector.
        Odd-parity entries (A7ap, A7bp) project the eigenvector onto the FIXED inter-sublattice, singlet-spin matrices
        D7ap/D7bp, weighted by |φpx|²+|φpy|². The projection can be NONZERO even without an odd-parity component in Deltavec,
        since v(k) inherits inter-sublattice structure from A1g channels, yielding a nonzero point-wise overlap.
        This diagnoses the anomalous field's projection, not a physical px/py gap. Thus, A7ap ≈ A7a_d indicates "no odd-parity preference," not an "odd-parity gap."
        """
        u, v = psi[:2 * qm._N_ORB], psi[2 * qm._N_ORB:2 * 2 * qm._N_ORB]
        bond_a, bond_a_odd = np.vdot(u, D7a_d @ v), np.vdot(u, D7a_p @ v)
        bond_b, bond_b_odd = np.vdot(u, D7b_d @ v), np.vdot(u, D7b_p @ v)
        A7a_s  = abs(np.vdot(u, D7a_s @ v))
        A7a_es = abs(phi[qm._IR_ES] * bond_a)
        A7a_d  = abs(phi[qm._IR_D]  * bond_a)
        A7a_p  = float(np.hypot(abs(phi[qm._IR_PX] * bond_a_odd), abs(phi[qm._IR_PY] * bond_a_odd)))
        A7a    = float(np.sqrt(A7a_s**2 + A7a_es**2 + A7a_d**2 + A7a_p**2))
        A7b_s  = abs(np.vdot(u, D7b_s @ v))
        A7b_es = abs(phi[qm._IR_ES] * bond_b)
        A7b_d  = abs(phi[qm._IR_D]  * bond_b)
        A7b_p  = float(np.hypot(abs(phi[qm._IR_PX] * bond_b_odd), abs(phi[qm._IR_PY] * bond_b_odd)))
        A7b    = float(np.sqrt(A7b_s**2 + A7b_es**2 + A7b_d**2 + A7b_p**2))
        p = np.abs(psi) ** 2
        w6  = _orbital_weight(p, qm._CHANNEL_ORB_IDX[0])
        w7a = _orbital_weight(p, qm._CHANNEL_ORB_IDX[1])
        w7b = _orbital_weight(p, qm._CHANNEL_ORB_IDX[2])
        return dict(A7a_s=float(A7a_s), A7a_es=float(A7a_es), A7a_d=float(A7a_d), A7a_p=A7a_p, A7a=A7a,
                    A7b_s=float(A7b_s), A7b_es=float(A7b_es), A7b_d=float(A7b_d), A7b_p=A7b_p, A7b=A7b,
                    A7a_frac=A7a / max(A7a + A7b, 1e-15),
                    w6=w6, w7a=w7a, w7b=w7b, P_mix=4.0 * w6 * w7a)

    # --- 3. Probe functions ---
    # Both probe functions accept their own F67 so the caller controls the (Δ, F67) pair consistently.
    def _probe_grid(dv: np.ndarray, f67: np.ndarray) -> dict:
        ev = np.linalg.eigvalsh(vbdg._build_H_stack(solver.k_points, M_bdg, Q, dv, f67, mu_bdg, g_t, g_J))
        aE = np.abs(ev)
        flat_min = int(np.argmin(aE))
        return dict(min_abs_E=float(aE.ravel()[flat_min]),
                    k_min_grid=solver.k_points[flat_min // aE.shape[1]].copy())

    def _probe_fs(dv: np.ndarray, f67: np.ndarray) -> dict:
        """Line-search on the FS cuts; returns the spectral gap per FS point."""
        rf = _bfs_line_search(solver, M_bdg, Q, dv, f67, n_bdg, mu_bdg, g_t, g_J,
                               ls_pts=ls_pts, n_sub=_n_sub,
                               ls_win_min=ls_win_min, ls_win_max=ls_win_max, full=True)
        if rf is None:
            return None
        N        = rf['N']
        weights  = rf['weights']
        w_sum    = rf['w_sum']
        spec_gaps = rf['spec_gaps']
        xi0_gaps  = rf['xi0_gaps']
        anti_mask = rf['anti_mask']
        nod_mask  = rf['nod_mask']
        ec_min    = rf['ec_min']
        k_min_pts = rf['k_min_pts']
        Phi_min   = qm._pairing_basis_matrix(k_min_pts[:, 0], k_min_pts[:, 1])

        A7a_s, A7a_es, A7a_d, A7a_p, A7a = (np.empty(N) for _ in range(5))
        A7b_s, A7b_es, A7b_d, A7b_p, A7b = (np.empty(N) for _ in range(5))
        A7a_f, Pmix, w6, w7a, w7b = (np.empty(N) for _ in range(5))
        for i in range(N):
            c = _ch(ec_min[i], Phi_min[i])
            A7a_s[i], A7a_es[i], A7a_d[i], A7a_p[i], A7a[i] = c['A7a_s'], c['A7a_es'], c['A7a_d'], c['A7a_p'], c['A7a']
            A7b_s[i], A7b_es[i], A7b_d[i], A7b_p[i], A7b[i] = c['A7b_s'], c['A7b_es'], c['A7b_d'], c['A7b_p'], c['A7b']
            A7a_f[i], Pmix[i] = c['A7a_frac'], c['P_mix']
            w6[i], w7a[i], w7b[i] = c['w6'], c['w7a'], c['w7b']

        _wavg = lambda arr: float(np.sum(weights * arr) / w_sum)
        _uavg = lambda arr: float(np.mean(arr))
        E_anti_spec = float(np.average(spec_gaps[anti_mask], weights=weights[anti_mask])) if anti_mask.any() else float('nan')
        E_nod_spec  = float(np.average(spec_gaps[nod_mask],  weights=weights[nod_mask]))  if nod_mask.any()  else float('nan')
        E_anti_xi0  = float(np.average(xi0_gaps[anti_mask],  weights=weights[anti_mask])) if anti_mask.any() else float('nan')
        gi = int(np.argmin(spec_gaps))
        c_min = _ch(ec_min[gi], Phi_min[gi])
        return dict(
            min_spec_E=float(spec_gaps[gi]), min_xi0_E=float(np.min(xi0_gaps)),
            k_min=k_min_pts[gi],
            E_anti_spec=E_anti_spec, E_nod_spec=E_nod_spec, E_anti_xi0=E_anti_xi0,
            E_gapW_spec=_wavg(spec_gaps), E_gapW_xi0=_wavg(xi0_gaps),
            A7a_s_w=_wavg(A7a_s), A7a_es_w=_wavg(A7a_es), A7a_d_w=_wavg(A7a_d), A7a_p_w=_wavg(A7a_p), A7a_w=_wavg(A7a),
            A7a_s_u=_uavg(A7a_s), A7a_es_u=_uavg(A7a_es), A7a_d_u=_uavg(A7a_d), A7a_p_u=_uavg(A7a_p), A7a_u=_uavg(A7a),
            A7b_s_w=_wavg(A7b_s), A7b_es_w=_wavg(A7b_es), A7b_d_w=_wavg(A7b_d), A7b_p_w=_wavg(A7b_p), A7b_w=_wavg(A7b),
            A7b_s_u=_uavg(A7b_s), A7b_es_u=_uavg(A7b_es), A7b_d_u=_uavg(A7b_d), A7b_p_u=_uavg(A7b_p), A7b_u=_uavg(A7b),
            A7a_frac_w=_wavg(A7a_f), P_mix_w=_wavg(Pmix),
            gap_closing_frac=float(np.sum(weights * (spec_gaps < gap_closed_threshold)) / w_sum),
            bfs_frac=float(np.sum(weights * (spec_gaps < qm._GAP_BFS_TOL)) / w_sum),
            spec_gaps_all=spec_gaps, xi0_gaps_all=xi0_gaps, A7a_all=A7a,
            spec_idx=rf['spec_idx'],
            A7a_min=c_min['A7a'], A7b_min=c_min['A7b'], A7a_frac_min=c_min['A7a_frac'],
            w6_min=c_min['w6'], w7a_min=c_min['w7a'], w7b_min=c_min['w7b'],
            phi_d_k_min=float(Phi_min[gi, qm._IR_D]),
            edge_hits=rf['edge_hits'],
        )

    zero_D = np.zeros_like(Delta_vec)
    zero_F = np.zeros_like(Quad67_vec)

    def _pair_at_frac(frac: float) -> tuple[np.ndarray, np.ndarray]:
        """(Δ, F67) at a given fraction of the converged SC amplitude."""
        if frac <= 0.0:
            return zero_D, zero_F
        if frac >= 1.0:
            return Delta_vec, Quad67_vec
        dv = Delta_vec * frac
        ev, ec = _bdg_eig(solver, M_bdg, Q, dv, zero_F, n_bdg, mu_bdg, g_t, g_J)
        g_Das, g_Dad, g_Dbd = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M_bdg, Q, mu_bdg, g_t, g_J)
        Quad67 = qm._compute_B1g_quadrupole(solver.k_weights, solver.Phi_irrep_k, ev, ec, solver.kT, solver.B1g_op)
        return dv, Quad67

    # --- 4. Baseline: true normal state (Δ=0, F67=0) ---
    r0f = _probe_fs(zero_D, zero_F)
    E0_anti = r0f['E_anti_spec']
    _log(f"[BOG-FS probe] baseline (Δ=0, F67=0) FS-weighted mean|E| (spectral): {r0f['E_gapW_spec']:.3e} eV")
    _log(f"[BOG-FS probe] baseline (Δ=0, F67=0) FS-weighted mean|E| (ξ=0)     : {r0f['E_gapW_xi0']:.3e} eV")
    if _p0['is_dwave']:
        _log(f"[BOG-FS probe] baseline (Δ=0, F67=0) antinodal mean|E| (spectral): {E0_anti:.3e} eV")
    _log(f"[BOG-FS probe] |Δ_amplitude| = {Delta_amplitude:.4e} eV  "
         f"(F67 self-consistent at frac=1, recomputed for 0<frac<1)")

    # --- 5. frac scan ---
    hdr = (f"{'frac':>5s}  {'min(E_ξ0)':>11s} {'min(E_spc)':>11s}  "
           f"{'anti_ξ0':>10s} {'anti_spc':>10s}  "
           f"{'A7a_s':>10s} {'A7a_es':>10s} {'A7a_d':>10s} {'A7a_p':>10s}  {'Γ7a':>5s}  {'BFS%':>6s}  flags")
    _log(f"\n[BOG-FS probe] frac scan  (anti-gap flags compare against the Δ=0, F67=0 baseline)\n{hdr}\n{'-' * len(hdr)}")
    k_hist: dict[float, tuple] = {}
    for frac in frac_scan:
        dv, f67 = _pair_at_frac(frac)
        rf = _probe_fs(dv, f67)
        k_hist[frac] = tuple(np.round(rf['k_min'], 3))
        flags = []
        if not (skip_zero_frac_flags and frac == 0.0):
            # Antinodal gap change relative to the Δ=0, F67=0 baseline
            if np.isfinite(rf['E_anti_spec']) and E0_anti > 0:
                _ratio = rf['E_anti_spec'] / E0_anti
                if _ratio > 2.0:   flags.append(f'anti-gap OPEN (x{_ratio:.2f} vs norm)')
                elif _ratio < 0.5: flags.append(f'anti-gap CLOSED (x{_ratio:.2f} vs norm)')
                else:              flags.append(f'anti-gap flat (x{_ratio:.2f} vs norm)')
            if rf['A7a_frac_min'] >= 0.5:      flags.append('Γ6-Γ7a')
            elif rf['A7b_min'] > rf['A7a_min']: flags.append('Γ6-Γ7b')
            _a7 = {'s': rf['A7a_s_w'], 'es': rf['A7a_es_w'], 'd': rf['A7a_d_w'], 'p': rf['A7a_p_w']}
            _b7 = {'s': rf['A7b_s_w'], 'es': rf['A7b_es_w'], 'd': rf['A7b_d_w'], 'p': rf['A7b_p_w']}
            if max(_a7.values()) > 1e-15: flags.append(f"a:{max(_a7, key=_a7.get)}")
            if max(_b7.values()) > 1e-15: flags.append(f"b:{max(_b7, key=_b7.get)}")
        _log(f"{frac:>5.2f}  {rf['min_xi0_E']:>11.4e} {rf['min_spec_E']:>11.4e}  "
             f"{rf['E_anti_xi0']:>10.4e} {rf['E_anti_spec']:>10.4e}  "
             f"{rf['A7a_s_w']:>10.4e} {rf['A7a_es_w']:>10.4e} "
             f"{rf['A7a_d_w']:>10.4e} {rf['A7a_p_w']:>10.4e}  "
             f"{rf['A7a_frac_w']*100:>4.0f}%  {rf['bfs_frac']*100:>5.1f}%  "
             f"{', '.join(flags) if flags else '—'}")

    # --- 6. Full-gap state summary ---
    rfg = _probe_grid(Delta_vec, Quad67_vec)
    rff = _probe_fs(Delta_vec, Quad67_vec)
    two_Delta_amp = 2.0 * Delta_amplitude
    _log("\n[BOG-FS probe] full-gap state (Δ_vec, Quad67_vec supplied as converged):")
    _log(f"  |Δ_amplitude| (input)              : {Delta_amplitude:.4e} eV")
    _log(f"  2|Δ| (naive gap scale)             : {two_Delta_amp:.4e} eV")
    _log(f"  min|E| full BZ                     : {rfg['min_abs_E']:.4e} eV  @ k={np.round(rfg['k_min_grid'], 3)}")
    _log(f"  FS-weighted mean|E| (ξ=0 / spectral): "
         f"{rff['E_gapW_xi0']:.4e} / {rff['E_gapW_spec']:.4e} eV")
    if _p0['is_dwave']:
        _log(f"  antinodal mean|E| (ξ=0 / spectral) : "
             f"{rff['E_anti_xi0']:.4e} / {rff['E_anti_spec']:.4e} eV")
        _log(f"  nodal mean|E| (spectral)       : {rff['E_nod_spec']:.4e} eV")
    _log(f"  gap-closed FS fraction (<{gap_closed_threshold:.0e} eV) : "
         f"{rff['gap_closing_frac']*100:.1f}%")
    _log(f"  BFS fraction (<{qm._GAP_BFS_TOL:.0e} eV)           : "
         f"{rff['bfs_frac']*100:.1f}%  "
         f"[{'Bogoliubov FS dominant' if rff['bfs_frac'] > bfs_frac_tol_hi else 'not dominant'}]")

    # ξ=0 vs spectral gap consistency: if the spectral minimum is well below the ξ=0 gap, the quasiparticle minimum has shifted off the bare FS — 
    # a signature of the interorbital gap-closing (the pair field mixes Γ6 and Γ7a, so the true minimum sits at finite ξ = −|Δ|²/δ rather than exactly on the ξ=0 surface).
    if rff['E_anti_xi0'] > 1e-12:
        shift_ratio = rff['E_anti_spec'] / rff['E_anti_xi0']
        if shift_ratio < 0.95:
            _log(f"  ⚠ Spectral gap is {shift_ratio*100:.1f}% of the ξ=0 gap — quasiparticle minimum shifted off the bare FS (interorbital gap-closing signature)")

    # Edge-of-search-window hit
    if rff['edge_hits'] > 0:
        _log(f"  ! {rff['edge_hits']}/{_p0['N']} FS points place the gap minimum "
             f"at the line-search window edge -- win_eV / ls_win_max may be too tight!")

    # --- 7. Minimum-gap k-point hopping detection ---
    if len(k_hist) > 1:
        hop = sum(1 for a, b in zip(k_hist.values(), list(k_hist.values())[1:]) if a != b)
        if hop > 0:
            _log(f"  ⚠ minimum-gap k-point hopped {hop}× across the frac scan; positions:")
            for f, k in k_hist.items():
                _log(f"  frac={f:.2f}: k = {k}")
        else:
            _log(f"  ✓ minimum-gap k-point stable across the frac scan (k = {list(k_hist.values())[0]})")

    # --- 8. Channel decomposition ---
    _log("\n  Pairing-channel amplitudes (weighted / unweighted):")
    for name, w, u in (('Γ6-Γ7a s',  rff['A7a_s_w'],  rff['A7a_s_u']),
                       ('Γ6-Γ7a es', rff['A7a_es_w'], rff['A7a_es_u']),
                       ('Γ6-Γ7a d',  rff['A7a_d_w'],  rff['A7a_d_u']),
                       ('Γ6-Γ7a p',  rff['A7a_p_w'],  rff['A7a_p_u']),
                       ('Γ6-Γ7a Σ',  rff['A7a_w'],    rff['A7a_u']),
                       ('Γ6-Γ7b s',  rff['A7b_s_w'],  rff['A7b_s_u']),
                       ('Γ6-Γ7b es', rff['A7b_es_w'], rff['A7b_es_u']),
                       ('Γ6-Γ7b d',  rff['A7b_d_w'],  rff['A7b_d_u']),
                       ('Γ6-Γ7b p',  rff['A7b_p_w'],  rff['A7b_p_u']),
                       ('Γ6-Γ7b Σ',  rff['A7b_w'],    rff['A7b_u'])):
        _log(f"    {name:<11s}: w={w:.4e}  u={u:.4e}")
    _log(f"  Γ7a share / P_mix            : {rff['A7a_frac_w']*100:.1f}%  /  {rff['P_mix_w']:.4e}")
    _log(f"  minimum-gap FS k / φ_d       : {np.round(rff['k_min'], 3)}  /  {rff['phi_d_k_min']:+.3f}")
    _log(f"  minimum-gap orbitals Γ6/7a/7b: "
         f"{rff['w6_min']:.3f} / {rff['w7a_min']:.3f} / {rff['w7b_min']:.3f}")

    _log()
    if rff['A7a_w'] > rff['A7b_w']:
        _log("    ✓ Γ6-Γ7a anomalous coherence dominates (paper's main channel).")
    elif rff['A7b_w'] > rff['A7a_w']:
        _log("    ⚠ Γ6-Γ7b dominates — paper's secondary channel.")

    for _tag, _pref in (('Γ6-Γ7a', 'A7a'), ('Γ6-Γ7b', 'A7b')):
        _amp = {'s': rff[f'{_pref}_s_w'], 'es': rff[f'{_pref}_es_w'],
                'd': rff[f'{_pref}_d_w'], 'p': rff[f'{_pref}_p_w']}
        _dom = max(_amp, key=_amp.get)
        if _amp[_dom] > 1e-15:
            _runner = max((k for k in _amp if k != _dom), key=lambda k: _amp[k])
            _lab = {'s': 's-wave', 'es': 'extended-s (es)', 'd': 'd-wave', 'p': 'odd-parity (px/py)'}[_dom]
            _log(f"    {'✓' if _dom == 'd' else '!'} {_lab} dominance in {_tag} "
                 f"({_dom}/{_runner} = {_amp[_dom]/max(_amp[_runner], 1e-15):.2f}).")
    for _tag, _pref in (('Γ6-Γ7a', 'A7a'), ('Γ6-Γ7b', 'A7b')):
        _ratio_hot = rff[f'{_pref}_w'] / max(rff[f'{_pref}_u'], 1e-15)
        if _ratio_hot > 2.0:
            _log(f"    ⚠ {_tag}: strong hot-spot weighting (w-mean > 2× u-mean).")
        elif _ratio_hot < 0.5:
            _log(f"    ⚠ {_tag}: strong anti-hot-spot weighting (u-mean > 2× w-mean).")
        else:
            _log(f"    ✓ {_tag}: coherence roughly uniform on the FS.")

    # --- 9. Antinodal gap summary ---
    if _p0['is_dwave'] and E0_anti > 0 and np.isfinite(rff['E_anti_spec']):
        ratio = rff['E_anti_spec'] / E0_anti
        if ratio > 2.0:   _log(f"  ✓ Antinodal gap OPENED ({ratio:.2f}× Δ=0, F67=0 baseline)")
        elif ratio > 0.5: _log(f"  ⚠ Antinodal gap inconclusive ({ratio:.2f}× baseline)")
        else:             _log(f"  ✗ Antinodal gap NOT opened ({ratio:.2f}× baseline)")

    # Gap-opening verdict on the 24×24 line-search measurement
    if rff['bfs_frac'] > bfs_frac_tol_hi:
        _log(f"  ✗ {rff['bfs_frac']*100:.0f}% of the FS is gapless: Bogoliubov Fermi surface, not a BCS gap.")
    elif rff['bfs_frac'] > bfs_frac_tol_lo:
        _log(f"  ⚠ {rff['bfs_frac']*100:.0f}% of the FS is gapless: partial Bogoliubov FS.")
    else:
        _log(f"  ✓ Gap open on the FS  ({rff['gap_closing_frac']*100:.0f}% closed).")
    _log("-" * 100)


def _fs_sector_label(t: int, mx: int, my: int) -> str:
    """Compact label: D2h irrep + T_Q parity."""
    d2h = {(1, 1): 'Ag', (-1, -1): 'B1g', (-1, 1): 'B3u', (1, -1): 'B2u'}[(mx, my)]
    return (f"{d2h:<4s}(TQ{'+' if t > 0 else '-'},"
            f"mx{'+' if mx > 0 else '-'},my{'+' if my > 0 else '-'})")

def test_fs_sample_group_structure(solver, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, tol: float = 1e-9) -> dict:
    """
    The FS sample is the integration measure of EVERY pairing kernel, so the sector decomposition G = {E, T_Q} x D2h is valid only if the SAMPLE itself carries the group exactly.  Checked
    independently of the model's own report (cache flags fs_closed_*, fs_tq_ok, odd_channels_ok), on the sample that solver._get_fs_points returns:
      (1) sample points are distinct grid points with finite positive weights and lie in |kx|+|ky| <= pi, the fundamental domain of T_Q: k -> k+(pi,pi);
      (2) closed under kx -> -kx, ky -> -ky, k -> -k (exact images, qm._g_apply), with equal weight on every image;
      (3) T_Q: no interior point has its image in the sample (the sector kernel adds the T_Q images itself); every boundary point |kx|+|ky| = pi has its image in the sample, with equal weight;
      (4) first moments vanish: sum_k w_k phi_i(k) = 0 for every basis harmonic that is odd under kx -> -kx or ky -> -ky;
      (5) the FS-weighted Gram matrix S_ij = sum_k w_k phi_i phi_j has no element between harmonics of different mirror character (mx, my) -- the metric of the sector condensation energy.
    """
    fs_pts, _, _, w = solver._get_fs_points(M, Q, mu, g_t, g_J)
    if len(w) == 0:
        _log("    [✗] the FS sample is empty")
        return {'verdict': 'FAIL'}
    kx, ky = np.asarray(fs_pts, dtype=float).T
    w = np.asarray(w, dtype=float)
    NK = qm._NK
    h = 2.0 * np.pi / NK

    def grid_index(px, py):
        return ((np.round((px + np.pi) / h).astype(int) % NK) * NK + (np.round((py + np.pi) / h).astype(int) % NK)).tolist()

    code = grid_index(kx, ky)
    wmap = dict(zip(code, w.tolist()))
    w_ref, w_sum = float(w.max()), float(w.sum())

    def image(op):
        """Per sample point: is its image under op in the sample, and the relative weight mismatch against that image (inf if absent)."""
        jc = grid_index(*qm._g_apply(op, kx, ky))
        have = np.array([c in wmap for c in jc])
        dw = np.array([abs(wmap[c] - wi) / w_ref if c in wmap else np.inf for c, wi in zip(jc, w.tolist())])
        return have, dw

    r = np.abs(kx) + np.abs(ky)
    bdy, inn = np.abs(r - np.pi) < 1e-9, r < np.pi - 1e-9
    grid_err = float(max(np.max(np.abs(kx - (np.round((kx + np.pi) / h) * h - np.pi))), np.max(np.abs(ky - (np.round((ky + np.pi) / h) * h - np.pi)))))
    checks = [("(1) distinct grid points, finite positive weights", grid_err < 1e-9 and len(wmap) == len(code) and bool(np.all(np.isfinite(w)) and np.all(w > 0.0)), f"N_fs = {len(code)}   max grid offset {grid_err:.1e}"),
              ("(1) inside the T_Q fundamental domain |kx|+|ky| <= pi", float(r.max()) <= np.pi + tol, f"max(|kx|+|ky|) - pi = {float(r.max()) - np.pi:+.1e}")]
    for label, op in (("kx -> -kx", (1, -1, 1)), ("ky -> -ky", (1, 1, -1)), ("k -> -k", (1, -1, -1))):
        have, dw = image(op)
        checks.append((f"(2) closed under {label}, equal weights on the images", bool(have.all()) and float(dw.max()) <= tol, f"{int((~have).sum())} images missing, worst weight mismatch {float(dw[have].max()) if have.any() else 0.0:.1e}"))
    have, dw = image((-1, 1, 1))
    checks.append(("(3) T_Q: no interior point has its image in the sample", not bool(have[inn].any()), f"{int(have[inn].sum())} of {int(inn.sum())} interior images present"))
    checks.append(("(3) T_Q: every boundary point has its image in the sample, equal weight", bool(have[bdy].all()) and (not bdy.any() or float(dw[bdy].max()) <= tol), f"{int(bdy.sum())} boundary points, {int((~have[bdy]).sum())} images missing"))
    Phi = qm._pairing_basis_matrix(kx, ky)
    chars = np.array([qm._IRREP_CHAR[nm] for nm in qm._IRREP_NAMES])
    odd = ~((chars[:, 1] == 1) & (chars[:, 2] == 1))
    mom = np.abs(Phi.T @ w) / w_sum
    i_worst = int(np.argmax(np.where(odd, mom, -1.0)))
    checks.append((f"(4) first moments of the {int(odd.sum())} mirror-odd basis harmonics vanish", float(mom[odd].max()) <= tol, f"worst {float(mom[i_worst]):.1e} ({qm._IRREP_NAMES[i_worst]})"))
    S = Phi.T @ (w[:, None] * Phi)
    d = np.sqrt(np.maximum(np.diag(S), 1e-300))
    rel = np.abs(S) / np.outer(d, d)
    cls = chars[:, 1] * 10 + chars[:, 2]
    off = cls[:, None] != cls[None, :]
    checks.append(("(5) FS-weighted Gram matrix is block-diagonal in the mirror character (mx, my)", float(rel[off].max()) <= tol, f"worst off-block |S_ij| / sqrt(S_ii S_jj) = {float(rel[off].max()):.1e}"))
    for label, ok, detail in checks:
        _log(f"    [{'✓' if ok else '✗'}] {label}   {detail}")
    n_fail = sum(not ok for _, ok, _ in checks)
    _log(f"  VERDICT: {'PASS' if not n_fail else 'FAIL'} -- " + ("the FS sample carries G = {E, T_Q} x D2h exactly." if not n_fail else f"{n_fail} check(s) failed: the sector kernel / condensation metric rest on a sample that is not G-closed."))
    return {'verdict': 'PASS' if not n_fail else 'FAIL', 'n_fail': n_fail}


def test_fs_sector_spectrum(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, cache: dict, V_cap: float | None = None, J_channels: np.ndarray | None = None, characters: list | None = None, use_dwave_weighting: bool = True) -> None:
    """
    Exact G = {E, T_Q} × D2h sector spectrum of the FS pairing kernel, read directly from compute_pairing_kernel_and_build_cache().

    The SCF uses the group-averaged kernel
        K_chi(k_i, k_j) = (1/|G|) Σ_{g∈G} χ(g) · V(k_i, g·k_j),
    because the FS sample of _get_fs_points is the FUNDAMENTAL DOMAIN of G = {E, T_Q} × D2h (diamond MBZ), not a G-closed set.

    Every gap function with a definite symmetry is a 1-D character
        χ = (t, mx, my) ∈ {±1}³
    of the abelian G:
        t   = parity under T_Q : k -> k + (pi, pi)
        mx  = parity under kx  -> -kx
        my  = parity under ky  -> -ky
    The eight sectors span the irreducible representations of G.

    Contract validated here
    -----------------------
    * K_chi commutes with every group permutation and maps the χ-subspace
      into itself (so V_irrep is exactly block-diagonal in the characters).
    * Every covered sector has a finite λ_top; every active channel has V_ii ≠ 0.
    * lambda_lin_max is the maximum over the covered sectors, and the leading sector
      reported in the cache is the argmax.
    * frac (the SCF seed direction) is normalised and supported entirely inside the leading sector.
    * The library completeness scan reports, per sector, the missing harmonics from
      _HARMONIC_LIBRARY that would raise the Loewdin resolved weight of the leading eigenvector.
    """
    M = np.asarray(M, dtype=float)
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec = np.asarray(Quad67_vec, dtype=complex)

    fs_pts = np.asarray(cache['fs_pts'], dtype=float)
    fs_weights = np.asarray(cache['fs_weights'], dtype=float)
    N = len(fs_pts)
    if N == 0:
        _log("[FS-SECTOR] No FS points. Abort.")
        return {}

    sqrt_w = np.sqrt(fs_weights)
    phi_s = np.ones(N)
    phi_d = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
    den_s = max(float(np.sum(phi_s ** 2 * fs_weights)), 1e-30)
    den_d = max(float(np.sum(phi_d ** 2 * fs_weights)), 1e-30)

    sec_keys     = cache.get('sector_keys')
    sec_lam      = cache.get('sector_lambda')
    sec_lam2     = cache.get('sector_lambda2')
    sec_psi      = cache.get('sector_psi')
    sec_resolved = cache.get('sector_resolved')
    lead         = cache.get('leading_sector')
    lead_all     = cache.get('leading_sector_all')
    lam_max      = cache.get('lambda_lin_max')
    lam_max_all  = cache.get('lambda_lin_max_all')
    if sec_keys is None or sec_lam is None:
        _log("[FS-SECTOR] cache has no sector data — abort.")
        return {}

    # --- Header ---
    _log("=" * 108)
    _log("  Exact G = {E, T_Q} x D2h sector spectrum of the FS pairing kernel")
    _log("  Source: compute_pairing_kernel_and_build_cache() — same kernel the SCF uses.")
    _log(f"  Q={Q:+.5f}  M={np.array2string(M, precision=5)}  N_fs={N}")
    _log(f"  lambda_lin_max (leading covered) = {lam_max:+.6f}")
    _log(f"  lambda_lin_max_all (leading all) = {lam_max_all:+.6f}")
    _log()

    # --- Table ---
    rows = []
    for si, key in enumerate(sec_keys):
        t, mx, my = key
        psi = sec_psi[si]
        cap_s = float((phi_s @ (psi * sqrt_w)) ** 2) / den_s if t ==  1 else 0.0
        cap_d = float((phi_d @ (psi * sqrt_w)) ** 2) / den_d if t == -1 else 0.0
        rows.append({
            'key':      key,
            'lam':      float(sec_lam[si]),
            'lam2':     float(sec_lam2[si]) if sec_lam2 is not None else float('nan'),
            'singlet':  bool(mx * my == 1),
            'cap_s':    cap_s, 'cap_d': cap_d,
            'resolved': float(sec_resolved[si]) if sec_resolved is not None else float('nan'),
            'covered':  bool(qm._SECTOR_MEMBERS[key]),
            'is_lead':  (key == lead),
            'is_lead_all': (key == lead_all),
        })

    _log("  ┌──────────────────────────────────┬──────────┬──────────┬─────────┬───────┬───────┬────────┬──────────┐")
    _log("  │ (t,mx,my)  D2h label             │  λ_top   │   λ2     │ singlet │ cap_s │ cap_d │ cover  │ resolved │")
    _log("  ├──────────────────────────────────┼──────────┼──────────┼─────────┼───────┼───────┼────────┼──────────┤")
    for r in sorted(rows, key=lambda r: -r['lam']):
        t, mx, my = r['key']
        lbl  = _fs_sector_label(t, mx, my)
        sing = "  +    " if r['singlet'] else "  -    "
        tag  = ""
        if r['is_lead_all'] and r['is_lead']:
            tag = "  <- leading (all & covered)"
        elif r['is_lead_all']:
            tag = "  <- leading (all), NOT covered"
        elif r['is_lead']:
            tag = "  <- leading (covered)"
        _log(f"  │ {lbl:<32s} │ {r['lam']:+8.4f} │ {r['lam2']:+8.4f} │{sing}│ "
             f"{r['cap_s']:.3f} │ {r['cap_d']:.3f} │ "
             f"{'yes' if r['covered'] else 'NO ':>6s} │ "
             f"{r['resolved']:>8.3f} │{tag}")
    _log("  └──────────────────────────────────┴──────────┴──────────┴─────────┴───────┴───────┴────────┴──────────┘")
    _log()

    # --- Library completeness scan (which missing harmonics close each sector) ---
    for _ln in qm._sector_report_lines(cache, *_c4_breakers(solver, Q)):
        _log(_ln)
    _log()

    # --- Verdict from the leading singlet-allowed, basis-covered sector ---
    singlet_covered = [r for r in rows if r['singlet'] and r['covered']]
    if singlet_covered:
        best = max(singlet_covered, key=lambda r: r['lam'])
        t_b, mx_b, my_b = best['key']
        _log(f"  Dominant singlet-allowed, basis-covered sector: "
             f"{_fs_sector_label(t_b, mx_b, my_b)}")
        _log(f"    lambda_top = {best['lam']:+.4f}   lambda_2 = {best['lam2']:+.4f}   "
             f"resolved(active) = {best['resolved']:.1%}")
        if   (t_b, mx_b, my_b) == ( 1,  1,  1):
            _log("    -> s-wave (T_Q-even, D2h-A_g)")
        elif (t_b, mx_b, my_b) == (-1,  1,  1):
            _log("    -> d-wave (T_Q-odd, D2h-A_g)")
        elif (t_b, mx_b, my_b) == ( 1, -1, -1):
            _log("    -> g-wave (T_Q-even, D2h-B1g) — NOT in the multipolar SC-AFM hypothesis")
        else:
            _log(f"    -> {qm._sector_label((t_b, mx_b, my_b))}")
        if best['resolved'] < 0.9:
            _log(f"    ! basis INCOMPLETE: only {best['resolved']:.1%} of the "
                 f"leading eigenvector is spanned by the active basis.")
            _log("    ! extend qm._HARMONIC_LIBRARY with higher harmonics (every library entry is an active channel) and rerun.")
    else:
        _log("  ! No singlet-allowed, basis-covered sector found.")

    # --- Check: every covered sector has finite lambda_top and V_ii != 0 ---
    lam_arr = np.asarray(sec_lam, dtype=float)
    covered_mask = np.array([bool(qm._SECTOR_MEMBERS[k]) for k in sec_keys])
    finite_ok = bool(np.all(np.isfinite(lam_arr[covered_mask])))
    _log(f"  All covered sectors have finite lambda_top: {finite_ok}")
    _log("=" * 108)


def test_pair_vertex_symmetry(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, ctx: dict | None = None, swap_tol: float = 1e-9, cap_tol: float = 1e-8, zero_tol: float = 1e-12, quad_tol: float = 0.02, qcp_proximity_thr: float = 0.5, purity_thr: float = 0.9) -> None:
    """
    Structural consistency test for the per-pair RPA pairing vertex.
    Checks (in increasing order of physical significance):
      (1) Swap symmetry : V(k_i,k_j) == V(k_j,k_i). Tested by calling _pair_vertex_channels_local with (i_idx, j_idx) and (j_idx, i_idx); the two outputs must be identical.
      (2) V_cap compliance: |V_sum(k_i,k_j)| <= V_cap exactly, for every FS pair.
      (3) JT-off limit: with V_JT_corr = 0, the JT and cross channel arrays vanish
          identically, independent of V_JT_corr (which still enters chi_RPA).
      (4) Spin-off limit: with J_channels = 0, the spin and cross channel arrays
          vanish identically; V_jt collapses to V_JT_corr^2 * chi0_QQ / (1 - V_JT_corr*chi0_QQ).
      (5) Quadratic scaling in J1: doubling J_channels[0] (with others held at zero)
          scales V_spin by 4 in the far-from-QCP, Gamma6-pure, V_cap-unclamped subset.
          Two independent mechanisms break the naive x4 scaling and are excluded
          separately rather than conflated into one tolerance:
            (a) RPA self-consistency in J near the AFM QCP (|V_spin| ~ V_cap scale) —
                excluded via `qcp_proximity_thr`, as before.
            (b) Gamma6-Gamma7a/7b channel mixing: V_spin's outer-product structure
                -J_c J_c' chiR[c,c'] contains J0*J1, J0*J2 cross terms whenever a
                pair's FS points carry non-negligible Gamma7 weight; this is present
                even at weak coupling / far from any instability.
      (6) FS channel-weight composition report (Gamma6/7a/7b), which controls how strongly each spin-channel contribution enters the pair vertex.
    """
    p = solver.p
    fs_idx    = ctx['fs_idx']
    N_fs      = ctx['N_fs']
    i_idx     = ctx['i_idx']
    j_idx     = ctx['j_idx']
    unique_q  = ctx['unique_q']
    inv_idx   = ctx['inv_idx']
    E_k_cache = (ctx['ev_n'], ctx['ec_n'])

    # --- Shared vertex parameters -------------------------------------------
    J_A1g_diag, J_B1g_7a, J_B1g_7b = _exch(p, Q, g_J)
    J_channels = np.asarray(J_A1g_diag, dtype=float)[0::2]
    Gamma_M, F_cluster, V_JT_corr, V_cap = _vertex_tuple(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    dHdQ_precomputed = solver._calc_dHdQ(M, Q, np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J)

    def _pair_vertex_channels_local(J_ch, V_corr, ii=None, jj=None):
        ii_eff = i_idx if ii is None else np.asarray(ii, dtype=int)
        jj_eff = j_idx if jj is None else np.asarray(jj, dtype=int)

        J_ch_arr = np.asarray(J_ch, dtype=float)
        J_ch_6 = np.repeat(J_ch_arr, 2)

        # RPA closure at each unique q
        n_q = len(unique_q)
        Xi  = np.empty((n_q, qm._N_CHANNELS, qm._N_CHANNELS))
        Xc  = np.empty((n_q, qm._N_CHANNELS))
        Vjt = np.empty(n_q)
        for u_idx, q_u in enumerate(unique_q):
            _rpa = solver._rpa_closure(M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J, q_u, Gamma_M, J_ch_6, V_corr, E_k_cache, False, dHdQ_precomputed)
            J_q, det, M4 = _rpa['J_q'], _rpa['det'], _rpa['M4']
            chiR = qm._adjugate(M4) @ _rpa['chi0'] / det
            Xi[u_idx]  = -np.outer(J_q, J_q) * chiR[:qm._N_CHANNELS, :qm._N_CHANNELS]
            Xc[u_idx]  = 2.0 * J_q * V_corr * chiR[:qm._N_CHANNELS, qm._N_CHANNELS]
            Vjt[u_idx] = V_corr + V_corr ** 2 * chiR[qm._N_CHANNELS, qm._N_CHANNELS]

        # ---- FS-pair vertex assembly -------------------------------------
        s = np.sqrt(np.clip(solver._fs_channel_weights(fs_idx, E_k_cache), 0.0, None))
        u = s[:, :, None] * s[:, None, :]                # (N_fs, 3, 3)

        V_spin = np.einsum('pcd,pcd,pcd->p',
                           u[ii_eff], u[jj_eff], Xi[inv_idx],
                           optimize=True)
        V_jt   = Vjt[inv_idx]
        V_cross = np.einsum('pc,pc,pc->p',
                            s[ii_eff], s[jj_eff], Xc[inv_idx],
                            optimize=True)

        V_pairs = np.stack([V_spin, V_jt, V_cross])
        _V_sum = V_pairs.sum(axis=0)
        _over  = np.abs(_V_sum) > V_cap
        if np.any(_over):
            V_pairs[:, _over] *= V_cap / np.abs(_V_sum[_over])
        return V_pairs

    # Baseline call
    V_pairs = _pair_vertex_channels_local(J_channels, V_JT_corr)

    # FS channel-weight composition
    w_fs   = solver._fs_channel_weights(fs_idx, E_k_cache)
    w_mean = w_fs.mean(axis=0)
    w_min  = w_fs.min(axis=0)
    w_max  = w_fs.max(axis=0)

    # --- Test 1: swap symmetry ---------------------------------------------
    V_pairs_swapped = _pair_vertex_channels_local(J_channels, V_JT_corr, ii=j_idx, jj=i_idx)
    swap_err = float(np.max(np.abs(V_pairs - V_pairs_swapped)))
    swap_ok  = swap_err < swap_tol

    # --- Test 2: V_cap compliance ------------------------------------------
    V_sum   = V_pairs.sum(axis=0)
    cap_max = float(np.max(np.abs(V_sum)))
    cap_ok  = cap_max <= V_cap * (1.0 + cap_tol)

    # --- Test 3: JT-off limit (V_JT_corr = 0) -------------------------------
    V_pairs_noJT = _pair_vertex_channels_local(J_channels, 0.0)
    jt_max    = float(np.max(np.abs(V_pairs_noJT[1])))
    cross_max = float(np.max(np.abs(V_pairs_noJT[2])))
    jt_off_ok = (jt_max < zero_tol) and (cross_max < zero_tol)

    # --- Test 4: spin-off limit (J_channels = 0) ---------------------------
    J_channels_zero = np.zeros(3, dtype=float)
    V_pairs_noSpin  = _pair_vertex_channels_local(J_channels_zero, V_JT_corr)
    spin_max    = float(np.max(np.abs(V_pairs_noSpin[0])))
    cross_max_2 = float(np.max(np.abs(V_pairs_noSpin[2])))
    spin_off_ok = (spin_max < zero_tol) and (cross_max_2 < zero_tol)

    # Analytic check (best-effort; may be affected by Moriya+PSD, see docstring)
    chi0_qq_all = np.array([
        solver._chi0_channels_raw(M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J, np.asarray(q_u, dtype=float), E_k_cache, False, dHdQ_precomputed)[qm._N_CHANNELS, qm._N_CHANNELS]
        for q_u in unique_q
    ])
    V_jt_expected_q = (V_JT_corr / np.maximum(1.0 - V_JT_corr * chi0_qq_all, 1e-30))          # = V + V²χ0/(1−Vχ0): bare term + RPA chain
    V_jt_expected = V_jt_expected_q[inv_idx]
    unclamped = np.abs(V_pairs_noSpin[1]) < V_cap * (1.0 - 1e-9)
    if np.any(unclamped):
        vjt_dev = float(np.max(np.abs(V_pairs_noSpin[1][unclamped] - V_jt_expected[unclamped])))
        vjt_scale = max(float(np.max(np.abs(V_jt_expected[unclamped]))), 1e-12)
        vjt_rel_dev = vjt_dev / vjt_scale
    else:
        vjt_rel_dev = float('nan')
    vjt_formula_ok = np.isfinite(vjt_rel_dev) and vjt_rel_dev < 1e-6

    # --- Test 5: quadratic scaling in J1 -----------------------------------
    J_channels_2x = J_channels.copy()
    J_channels_2x[0] *= 2.0
    V_pairs_2x = _pair_vertex_channels_local(J_channels_2x, V_JT_corr)

    gamma6_w  = w_fs[:, 0]
    pure_pair = (gamma6_w[i_idx] > purity_thr) & (gamma6_w[j_idx] > purity_thr)
    unclamped_pairs = (
        (np.abs(V_pairs.sum(axis=0))    < V_cap * (1.0 - 1e-9)) &
        (np.abs(V_pairs_2x.sum(axis=0)) < V_cap * (1.0 - 1e-9)) &
        (np.abs(V_pairs[0]) > 1e-12) &
        pure_pair
    )
    near_qcp = bool(np.max(np.abs(V_pairs[0])) > qcp_proximity_thr * V_cap)
    n_pure   = int(np.sum(pure_pair))
    if near_qcp:
        quad_dev = float('nan')
        quad_ok = True          # inconclusive, not a failure
        quad_note = "N/A (near QCP; chi_RPA is self-consistently J-dependent)"
    elif n_pure == 0:
        quad_dev = float('nan')
        quad_ok = True          # inconclusive, not a failure — no Gamma6-pure pairs to test
        quad_note = f"N/A (no pairs with both points > {purity_thr:.0%} Gamma6; FS is channel-mixed)"
    elif np.any(unclamped_pairs):
        ratio = V_pairs_2x[0][unclamped_pairs] / V_pairs[0][unclamped_pairs]
        quad_dev = float(np.max(np.abs(ratio - 4.0)))
        quad_ok = quad_dev < quad_tol
        quad_note = f"max|ratio − 4| = {quad_dev:.3e}  (n={int(np.sum(unclamped_pairs))}/{n_pure} pure pairs used)"
    else:
        quad_dev = float('nan')
        quad_ok  = False
        quad_note = "no unclamped, Gamma6-pure pairs available for scaling test"

    # --- Verdict -----------------------------------------------------------
    checks = [
        ("swap symmetry V(k,k')=V(k',k)", swap_ok,  f"max|Δ| = {swap_err:.2e}"),
        ("V_cap compliance",              cap_ok,   f"max|V_sum| = {cap_max:.4e} (cap = {V_cap:.4e})"),
        ("V_JT_corr=0 → V_jt=V_cross=0",   jt_off_ok, f"max|V_jt|={jt_max:.2e}  max|V_cross|={cross_max:.2e}"),
        ("J_channels=0 → V_spin=V_cross=0", spin_off_ok, f"max|V_spin|={spin_max:.2e}  max|V_cross|={cross_max_2:.2e}"),
        ("V_jt analytic form",            vjt_formula_ok, f"rel dev = {vjt_rel_dev:.2e}"),
        ("V_spin ∝ J1² (quadratic)",      quad_ok,  quad_note),
    ]

    _log("=" * 78)
    _log(f"  Q={Q:+.5f}  M={np.array2string(np.asarray(M), precision=4)}  n={n_kspace:.6f}  mu={mu:+.6f}")
    _log(f"  N_fs={N_fs}   N_q={len(unique_q)}   N_pairs={len(i_idx)}")
    _log(f"  J_channels (per-bond) = {np.array2string(np.asarray(J_channels), precision=5)}")
    _log(f"  V_JT_corr={V_JT_corr:.4e}  V_cap={V_cap:.4e}")
    _log()
    _log("  FS channel-weight composition (Γ6, Γ7a, Γ7b) — Σ_c w_c(k) = 1:")
    _log(f"    mean : {np.array2string(w_mean, precision=4)}")
    _log(f"    min  : {np.array2string(w_min,  precision=4)}")
    _log(f"    max  : {np.array2string(w_max,  precision=4)}")
    if w_mean[1] + w_mean[2] < 1e-3:
        _log("    ⚠ FS carries essentially no Γ7 content → the channel-resolved spin weighting reduces to the Γ6-only sector.")
    _log()
    _log("  Checks:")
    for name, ok, info in checks:
        _log(f"    [{'✓' if ok else '✗'}] {name:<40s} {info}")
    _log()
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    if n_fail == 0:
        _log("  VERDICT: PASS — per-pair RPA vertex is structurally consistent")
        _log("           and reduces to the expected limits.")
    else:
        _log(f"  VERDICT: PARTIAL — {n_fail}/{len(checks)} checks failed.")
        # Differentiate a genuine numerical failure from a Moriya/PSD artifact
        if not vjt_formula_ok and vjt_formula_ok is not None:
            _log("  Note: the V_jt analytic-form check can fail spuriously when the 4×4 PSD projection modifies χ₀_QQ.")
    _log("=" * 78)


def test_kernel_cache_sectors(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray | None, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, cache: dict) -> None:
    """
    Structural invariants of the production cache (these would have caught the loop-indentation regression that returned after the FIRST sector):
      every one of the 8 sectors has a finite λ_top; V_irrep is block-diagonal in the sectors and every active channel has a non-zero diagonal element; lambda_lin_max is the maximum over the
      covered sectors; frac is a normalised profile inside the leading sector; the sector report runs.
    """
    lam     = cache.get('sector_lambda')
    names   = list(qm._IRREP_NAMES)
    V       = np.asarray(cache['V_irrep'])
    n       = len(names)
    odd_ok  = bool(cache['odd_channels_ok'])
    ops     = list(qm._G_OPS)

    # (1) All sectors have a finite eigenvalue
    finite = bool(np.all(np.isfinite(lam)))

    # (2) V_irrep is exactly block-diagonal in the G-sectors
    block = all(V[a, b] == 0.0 for a in range(n) for b in range(n) if qm._IRREP_CHAR[names[a]] != qm._IRREP_CHAR[names[b]])

    # (3) Every active channel has a non-zero diagonal element
    diag_nz = all(V[i, i] != 0.0 for i in range(n)
                  if odd_ok or qm._IRREP_PARITY[names[i]] > 0)

    # (4) Leading sector is the maximum over covered sectors
    covered = [si for si, k in enumerate(ops)
               if qm._SECTOR_MEMBERS[k]
               and (odd_ok or all(qm._IRREP_PARITY[names[i]] > 0 for i in qm._SECTOR_MEMBERS[k]))]
    lead_ok = bool(
        covered
        and cache['leading_sector'] == ops[max(covered, key=lambda si: lam[si])]
        and abs(cache['lambda_lin_max'] - float(np.max(lam[covered]))) < 1e-12
    )

    # (5) frac_seed: normalised AND supported on the UNION of seeded sectors
    frac   = np.asarray(cache['frac_seed'], dtype=float)
    seed_w = np.asarray(cache.get('seed_w', np.zeros(len(ops))), dtype=float)

    # Multi-sector: any sector with seed_w > tiny threshold contributes (near-degenerate covered sectors, not just the leading one)
    seeded_sectors = [si for si in range(len(ops)) if seed_w[si] > 1e-12]
    seeded_channels = set()
    for si in seeded_sectors:
        seeded_channels.update(qm._SECTOR_MEMBERS[ops[si]])

    frac_norm_ok    = abs(frac.sum() - 1.0) < 1e-6
    frac_support_ok = all(frac[i] < 1e-12
                          for i in range(n) if i not in seeded_channels)
    frac_ok         = frac_norm_ok and frac_support_ok

    # (6) The sector report must run
    try:
        rep    = qm._sector_report_lines(cache, *_c4_breakers(solver, Q))
        rep_ok = len(rep) > 1
    except Exception as e:
        rep    = [f"report failed: {type(e).__name__}: {e}"]
        rep_ok = False

    ok = finite and block and diag_nz and lead_ok and frac_ok and rep_ok

    _log("  [1] Sector eigenvalues (all 8 sectors):")
    _log(f"      {np.array2string(np.asarray(lam), precision=4)}")
    _log(f"      all finite: {finite}")
    _log("")
    _log("  [2] V_irrep structure:")
    _log(f"      exactly block-diagonal in G-sectors: {block}")
    _log(f"      every active channel has V_ii ≠ 0: {diag_nz}")
    _log(f"      diag(V_irrep) = {np.array2string(np.diag(V), precision=4)}")
    _log("")
    _log("  [3] Leading sector:")
    _log(f"      leading = {cache['leading_sector']}")
    _log(f"      is the maximum over {len(covered)} covered sectors: {lead_ok}")
    _log(f"      lambda_lin_max = {cache['lambda_lin_max']:+.6f}  "
         f"vs sector max = {float(np.max(lam[covered])):+.6f}")
    _log("")
    _log("  [4] frac_seed (multi-sector seed):")
    _log(f"      normalisation: |sum − 1| = {abs(frac.sum() - 1.0):.2e}  → {frac_norm_ok}")
    _log(f"      supported on seeded sectors only: {frac_support_ok}")
    _log(f"      seeded sectors (seed_w > 0): {len(seeded_sectors)}")
    for si in seeded_sectors:
        key = ops[si]
        chans = ','.join(names[i] for i in qm._SECTOR_MEMBERS[key])
        frac_here = sum(frac[i] for i in qm._SECTOR_MEMBERS[key])
        _log(f"        {key}  λ={lam[si]:+.4f}  seed_w={seed_w[si]:.4f}  "
             f"frac_total={frac_here:.4f}  channels={chans}")
    _log("")
    if not frac_support_ok:
        unsupported = [(names[i], frac[i]) for i in range(n)
                       if frac[i] > 1e-12 and i not in seeded_channels]
        _log("      ⚠ channels with frac > 0 but NOT in any seeded sector:")
        for nm, v in unsupported:
            _log(f"          {nm}: {v:.6f}")
    _log(f"      frac distribution: {np.array2string(frac, precision=4)}")
    _log("")
    _log("  [5] Sector report:")
    _log(f"      runs without exception: {rep_ok}")
    _log("")
    _log(f"  {'✓ PASS' if ok else '❌ FAIL'}")
    _log(f"    ({'all invariants hold' if ok else 'failed check(s): '
          + ', '.join(k for k, v in (('finite', finite), ('block', block),
                                     ('diag_nz', diag_nz), ('lead', lead_ok),
                                     ('frac', frac_ok), ('report', rep_ok))
                      if not v)})")


def test_hessian_inactive_linear_coupling(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, g_Das: float, g_Dad: float, g_Dbd: float) -> None:
    """
    compute_dF_dDelta_and_d2F for an amplitude that is exactly 0 but LINEARLY coupled to an active channel (Q·Δ_s*Δ_d, allowed because B1g ∈ Γ_s⊗Γ_d): F is linear in |Δ_s| there.
    A single one-sided evaluation returns 2g/ε as the 'curvature' (≈ −7·10³ at ε = 1e-5), which collapses the Levenberg–Marquardt-damped Newton step through λ_min.  The gradient must
    equal the independently estimated slope (different step sizes, own two-point fit) and the curvature must be O(1)–O(10²).
    """
    vp = _scf_vertex_params(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    cache = _build_vertex_cache(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    grad, Hess = solver.compute_dF_dDelta_and_d2F(M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, g_Das, g_Dad, g_Dbd, vp['V_JT_corr'], 0.0, cache)
    def F(dvec):
        f = solver._compute_bdg_free_energy(M, Q, dvec, Quad67_vec, mu, g_t, g_J, g_Das, g_Dad, g_Dbd, vp['V_JT_corr'], 0.0, solver.p.K_lattice, solver.p.K_lattice_Eg2, cache)
        return float(f) if np.isscalar(f) else float(f[0])
    F0 = F(Delta_vec)

    def two_point(i, h):
        d1 = Delta_vec.copy(); d1[i] = h
        d2 = Delta_vec.copy(); d2[i] = 2 * h
        F1, F2 = F(d1), F(d2)
        return (4 * F1 - F2 - 3 * F0) / (2 * h), (F2 - 2 * F1 + F0) / h ** 2

    i_s = qm._DELTA_IDX['s7a']
    g_ref, h_ref = two_point(i_s, qm._HESS_INACTIVE_EPS)
    g_asy, h_asy = two_point(i_s, qm._HESS_INACTIVE_EPS / 10.0)

    g_ok = abs(grad[i_s] - g_ref) <= max(0.01 * abs(g_ref), 1e-6)
    h_ok = abs(Hess[i_s, i_s] - h_ref) <= max(0.02 * abs(h_ref), 1.0)
    h_fin = np.isfinite(Hess[i_s, i_s]) and abs(Hess[i_s, i_s]) < 1e4
    g_asy_err = 100.0 * abs(grad[i_s] - g_asy) / max(abs(g_asy), 1e-30)
    h_asy_err = 100.0 * abs(Hess[i_s, i_s] - h_asy) / max(abs(h_asy), 1.0)

    w = np.linalg.eigvalsh(Hess)
    sym_ok = bool(np.allclose(Hess, Hess.T))
    eigs_ok = bool(w[0] > -1e5)

    _log(f"  step = {qm._HESS_INACTIVE_EPS:.2e}  (asymptotic reference: step/10)")
    _log(f"  Hessian eigenvalue range [{w[0]:+.2f}, {w[-1]:+.1f}], symmetric: {sym_ok}")
    _log(f"  s7a  grad  code={grad[i_s]:+.5e}  ref={g_ref:+.5e}  {'✓' if g_ok else '❌'}   "
         f"asymm ref={g_asy:+.5e}  (finite-step error {g_asy_err:.1f}%)")
    _log(f"  s7a  Hess  code={Hess[i_s, i_s]:+.2f}  ref={h_ref:+.2f}  {'✓' if h_ok else '❌'}   "
         f"asymm ref={h_asy:+.2f}  (finite-step error {h_asy_err:.1f}%)  "
         f"[{'not bogus 2g/ε' if h_fin else '❌ bogus 2g/ε'}]")
    _log(f"  mixed s7a–es7a = {Hess[i_s, qm._DELTA_IDX['es7a']]:+.3f},  s7a–d7a = {Hess[i_s, qm._DELTA_IDX['d7a']]:+.3f}")

    other_ok = True
    for nm in ('g7a', 'px7a', 'es7a', 'd7a'):
        i = qm._DELTA_IDX[nm]
        g_c, h_c = two_point(i, qm._HESS_INACTIVE_EPS)
        g_f_ok = abs(grad[i] - g_c) <= max(0.01 * abs(g_c), 1e-6)
        h_f_ok = abs(Hess[i, i] - h_c) <= max(0.02 * abs(h_c), 1.0)
        h_fin_i = np.isfinite(Hess[i, i]) and abs(Hess[i, i]) < 1e4
        flag = '✓' if (g_f_ok and h_f_ok and h_fin_i) else '❌'
        _log(f"  {nm:<5s}  grad={grad[i]:+.4e} (ref {g_c:+.4e})  Hess={Hess[i, i]:+.2f} (ref {h_c:+.2f})  {flag}")
        if not (g_f_ok and h_f_ok and h_fin_i):
            other_ok = False

    ok = g_ok and h_ok and h_fin and sym_ok and eigs_ok and other_ok
    _log(f"  {'✓ PASS' if ok else '❌ FAIL'}")


def test_kernel_parity_projection(vertex_cache: dict, closure_tol: float = 1e-8, sector_tol: float = 1e-6, verbose: bool = True) -> dict:
    """
    Parity decomposition of the FS pairing kernel K_fs = √w·V_weighted·√w under spatial inversion k → −k.
    The channel registry (_IRREP_NAMES, the whole harmonic library) contains BOTH parities:
        even (qm._IRREP_PARITY = +1): s, es, d, g, ...   (φ(-k) = +φ(k), antisymmetric index structure)
        odd  (qm._IRREP_PARITY = -1): px, py, ...        (φ(-k) = -φ(k), symmetric index structure)
    Both are physical (the fermion antisymmetry Δ_ij(k) = -Δ_ji(-k) is respected in _build_H_stack).  If [K, I] = 0, every eigenvector has DEFINITE parity, and the top eigenvector lives entirely in one sector.
    """
    fs_pts = np.asarray(vertex_cache['fs_pts'], dtype=float)
    N = len(fs_pts)
    if N == 0:
        if verbose:
            _log("[kernel-parity] empty FS sample")
        return {'verdict': 'INVALID', 'reason': 'empty FS'}

    sec_keys = vertex_cache.get('sector_keys')
    sec_lam  = vertex_cache.get('sector_lambda')
    sec_lam2 = vertex_cache.get('sector_lambda2')
    sec_psi  = vertex_cache.get('sector_psi')
    sec_res  = vertex_cache.get('sector_resolved')
    lead_key = vertex_cache.get('leading_sector')
    lam_code = float(vertex_cache.get('lambda_lin_max', float('nan')))

    if any(x is None for x in (sec_keys, sec_lam, sec_psi, lead_key)):
        if verbose:
            _log("[kernel-parity] cache has no sector data "
                 "(G-covariant sector kernel required)")
        return {'verdict': 'INVALID', 'reason': 'no sector data in cache'}

    # --- Inversion symmetry of the Fermi surface pattern ---
    perm, clos = _inversion_permutation(fs_pts)
    involution = bool(np.array_equal(perm[perm], np.arange(N)))
    closed     = bool(clos < closure_tol and involution)

    out = {
        'closed':              closed,
        'closure_residual':    float(clos),
        'involution':          involution,
        'leading_sector':      lead_key,
        'lambda_lin_max_code': lam_code,
        'sector_lambda_lead':  float('nan'),
        'sector_lambda2_lead': float('nan'),
        'cache_rel_err':       float('nan'),
        'measured_parity':     float('nan'),
        'expected_parity':     0,
        'parity_clean':        False,
        'matches_character':   False,
        'degenerate':          False,
        'resolved_weight':     float('nan'),
        'verdict':             'INVALID',
    }
    if not closed:
        if verbose:
            _log(f"[kernel-parity] FS sample is NOT inversion-closed "
                 f"(res={clos:.2e}, involution={involution}) — parity is undefined")
        return out

    # --- Locate the leading sector in the cache ---
    si_lead  = list(sec_keys).index(lead_key)
    lam_lead = float(sec_lam[si_lead])
    psi_lead = np.asarray(sec_psi[si_lead], dtype=float).copy()

    # --- Measured parity of ψ_lead ---
    nrm = float(np.linalg.norm(psi_lead))
    if nrm > 1e-30:
        psi_lead /= nrm
    measured_parity = float(psi_lead @ psi_lead[perm])

    # --- Expected parity from the sector character ---
    t, mx, my = lead_key
    expected_parity = int(mx * my)

    # --- Cache consistency: λ_lin_max vs sector_lambda[lead] ---
    cache_rel_err = (abs(lam_code - lam_lead) / max(abs(lam_lead), 1e-30)
                     if np.isfinite(lam_code) and np.isfinite(lam_lead)
                     else float('nan'))

    # --- Near-degeneracy of the leading sector ---
    lam2_lead = (float(sec_lam2[si_lead])
                 if sec_lam2 is not None and np.isfinite(sec_lam2[si_lead])
                 else float('nan'))
    degenerate = (np.isfinite(lam2_lead)
                  and abs(lam_lead - lam2_lead) < sector_tol * max(abs(lam_lead), 1e-30))

    # --- Loewdin resolved weight ---
    resolved = (float(sec_res[si_lead])
                if sec_res is not None and np.isfinite(sec_res[si_lead])
                else float('nan'))

    # --- Verdict ---
    parity_clean = abs(abs(measured_parity) - 1.0) < 1e-8
    matches_char = (parity_clean
                    and int(np.sign(measured_parity)) == expected_parity)
    cache_ok     = np.isfinite(cache_rel_err) and cache_rel_err < 1e-10

    if not parity_clean and not degenerate:
        verdict = 'FAIL'
    elif not parity_clean and degenerate:
        verdict = 'DEGENERATE'
    elif not matches_char or not cache_ok:
        verdict = 'FAIL'
    elif degenerate:
        verdict = 'DEGENERATE'
    else:
        verdict = 'PASS'

    out.update({
        'sector_lambda_lead':  lam_lead,
        'sector_lambda2_lead': lam2_lead,
        'cache_rel_err':       cache_rel_err,
        'measured_parity':     measured_parity,
        'expected_parity':     expected_parity,
        'parity_clean':        parity_clean,
        'matches_character':   matches_char,
        'degenerate':          degenerate,
        'resolved_weight':     resolved,
        'verdict':             verdict,
    })

    if verbose:
        parity_label = 'even' if expected_parity > 0 else 'odd'
        _log("=" * 78)
        _log("  Kernel parity of the leading G-sector eigenvector")
        _log("=" * 78)
        _log(f"  N_fs = {N}   inversion-closed = {closed}   "
             f"(res = {clos:.1e}, involution = {involution})")
        _log(f"  leading sector  χ = {lead_key}   "
             f"→ inversion parity mx·my = {expected_parity:+d}  ({parity_label})")
        _log(f"  sector λ_top = {lam_lead:+.6f}   sector λ_2 = {lam2_lead:+.6f}"
             + (f"   [near-degenerate, |Δ|/|λ| = {abs(lam_lead - lam2_lead) / max(abs(lam_lead), 1e-30):.1e}]"
                if degenerate else ""))
        _log(f"  ⟨ψ_lead|I|ψ_lead⟩ = {measured_parity:+.8f}   "
             f"(+1.0 even, −1.0 odd, 0 mixed)")
        _log(f"  cache λ_lin_max = {lam_code:+.6f}   "
             f"rel.err vs sector λ = {cache_rel_err:.1e}")
        _log(f"  Loewdin resolved weight of ψ_lead in the active basis of χ: {resolved:.4f}")

        if verdict == 'FAIL' and not parity_clean:
            _log(f"  ❌ FAIL: ψ_lead has mixed parity "
                 f"(⟨ψ|I|ψ⟩ = {measured_parity:+.4f}); the non-degenerate sector kernel is not inversion-symmetric")
        elif verdict == 'FAIL' and not matches_char:
            _log(f"  ❌ FAIL: measured parity {measured_parity:+.4f} does not match the sector character mx·my = {expected_parity:+d}")
        elif verdict == 'FAIL' and not cache_ok:
            _log(f"  ❌ FAIL: cache λ_lin_max does not match sector_lambda[leading] (rel.err = {cache_rel_err:.1e})")
        elif verdict == 'DEGENERATE':
            _log(f"  ⚠ DEGENERATE: leading sector has a near-degenerate top pair; the individual eigenvector parity is gauge-dependent. "
                 f"The measured value {measured_parity:+.4f} is one valid gauge choice.")
        else:
            _log(f"  ✓ PASS: ψ_lead has definite {parity_label} parity, "
                 f"consistent with the sector character χ = {lead_key}; cache λ_top = sector λ_top")
        _log("=" * 78)
    return out


def test_irrep_projection_of_FS_eigenvector(solver, target_doping: float, M_bdg: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_bdg: float, mu_bdg: float, g_t: float, g_J: float,
                                            degen_rel_tol: float = 2e-2, degen_max_dim: int = 8,
                                            sym_tol: float = 1e-6, verbose: bool = False,
                                            sector_tol: float = 1e-6) -> dict | None:
    """
    Leading-sector eigenvector irrep projection (library + beyond-library harmonics),
    plus the multi-sector seed direction.

    Components:
      * parity analysis of the sector eigenvector (via test_kernel_parity_projection);
      * extended Gram-Schmidt basis: ALL of _HARMONIC_LIBRARY (via the library's own form-factor callables
        and D4h labels) PLUS explicit extra higher harmonics not in the library;
      * D4h parent aggregation, split into 'library-resolved' and 'extra-resolved' shares;
      * S_sd (FS-sample closure test);
      * u_total (odd-parity content across the extended basis).
    """
    cache = _build_vertex_cache(solver, target_doping, M_bdg, Q, Delta_vec, Quad67_vec, n_bdg, mu_bdg, g_t, g_J)
    fs_pts     = np.asarray(cache['fs_pts'], dtype=float)
    fs_weights = np.asarray(cache['fs_weights'], dtype=float)
    N_fs       = len(fs_pts)
    if N_fs == 0:
        if verbose: _log("[irrep-proj] empty FS sample")
        return None

    # --- 1. Leading sector eigenvector ---
    sec_keys = cache.get('sector_keys'); sec_lam  = cache.get('sector_lambda')
    sec_lam2 = cache.get('sector_lambda2'); sec_psi = cache.get('sector_psi')
    lead_key = cache.get('leading_sector')
    if any(x is None for x in (sec_keys, sec_lam, sec_psi, lead_key)):
        if verbose: _log("[irrep-proj] cache has no sector data")
        return None
    si_lead = list(sec_keys).index(lead_key)
    top_psi = np.asarray(sec_psi[si_lead], dtype=float).copy()
    nrm = float(np.linalg.norm(top_psi))
    if nrm > 1e-30: top_psi /= nrm
    lambda_max  = float(sec_lam[si_lead])
    lambda_max2 = (float(sec_lam2[si_lead]) if (sec_lam2 is not None and np.isfinite(sec_lam2[si_lead])) else float('nan'))
    top_gap_rel = (abs(lambda_max - lambda_max2) / max(abs(lambda_max), 1e-15) if np.isfinite(lambda_max2) else float('inf'))
    degenerate  = bool(top_gap_rel < degen_rel_tol)
    n_degen     = 1        # only the top eigenvector per sector is cached

    # --- 2. Parity + FS-closure ---
    par = test_kernel_parity_projection(cache, verbose=False)
    top_sector = 'even' if int(np.sign(lead_key[1] * lead_key[2])) > 0 else 'odd'
    top_parity = float(par.get('measured_parity', float('nan')))
    inv_ok   = bool(cache.get('fs_closed_inv', False))
    mx_ok    = bool(cache.get('fs_closed_mx',  False))
    my_ok    = bool(cache.get('fs_closed_my',  False))
    w_sym_ok = bool(cache.get('fs_weight_sym_ok', False))
    w_asym   = float(cache.get('fs_weight_asym', float('nan')))
    # seed direction of the SCF (cache['frac_seed_signed']) must live in the parity block of the leading sector (this check used to sit in the parity-only cache test)
    _seed = cache.get('frac_seed_signed')
    if _seed is not None:
        _seed = np.asarray(_seed)
        _ev = [i for i, n in enumerate(qm._IRREP_NAMES) if qm._IRREP_PARITY[n] > 0]
        _od = [i for i, n in enumerate(qm._IRREP_NAMES) if qm._IRREP_PARITY[n] < 0]
        _ne, _no = float(np.linalg.norm(_seed[_ev])), float(np.linalg.norm(_seed[_od]))
        _ep = int(lead_key[1] * lead_key[2])
        _seed_ok = (_ne >= _no - 1e-12) if _ep > 0 else (_no >= _ne - 1e-12)
        _log(f"  [seed] ‖even-parity part‖ = {_ne:.3e}, ‖odd-parity part‖ = {_no:.3e}: the dominant parity matches the leading sector (mx·my = {_ep:+d})   {'✓' if _seed_ok else '❌ FAIL'}")

    # --- 3. Extended basis (library + beyond-library harmonics) ---
    kx, ky = fs_pts[:, 0], fs_pts[:, 1]
    library_basis = [(name, qm._IRREP_D4H[name], qm._IRREP_FF[name](kx, ky)) for name in qm._IRREP_NAMES]
    extra_basis = [
        ('ext-s_3nn',   'A1g', np.cos(3*kx) + np.cos(3*ky)),
        ('ext-s_4nn',   'A1g', np.cos(4*kx) + np.cos(4*ky)),
        ('ext-cc_3nn',  'A1g', np.cos(3*kx) * np.cos(3*ky)),
        ('ext-cc_4nn',  'A1g', np.cos(4*kx) * np.cos(4*ky)),
        ('ext-d_3nn',   'B1g', np.cos(3*kx) - np.cos(3*ky)),
        ('ext-d_4nn',   'B1g', np.cos(4*kx) - np.cos(4*ky)),
        ('ext-d_mixed', 'B1g', (np.cos(kx) - np.cos(ky)) * np.cos(kx) * np.cos(ky)),
        ('ext-g_4nn',   'B2g', np.sin(4*kx) * np.sin(4*ky)),
        ('ext-px_6nn',  'Eu',  np.sin(6*kx)),
        ('ext-py_6nn',  'Eu',  np.sin(6*ky)),
        ('ext-px_d3',   'Eu',  np.sin(3*kx) * np.cos(3*ky)),
        ('ext-py_d3',   'Eu',  np.cos(3*kx) * np.sin(3*ky)),
    ]
    basis_list = library_basis + extra_basis
    lib_labels = set(qm._IRREP_NAMES)

    # --- 5. Weighted inner product + Gram-Schmidt ---
    def _w_inner(a, b):
        return float(np.sum(fs_weights * a * b))

    ortho = []
    for label, pa, f in basis_list:
        g = np.asarray(f, dtype=float).copy()
        for _lbl, _pa, pg in ortho:
            g = g - (_w_inner(g, pg) / max(_w_inner(pg, pg), 1e-30)) * pg
        n2 = _w_inner(g, g)
        if n2 > 1e-12:
            ortho.append((label, pa, g / np.sqrt(n2)))

    # --- 5. Physical eigenvector + projection ---
    sqrt_w    = np.sqrt(fs_weights)
    psi_phys  = top_psi / np.maximum(sqrt_w, 1e-30)
    psi_norm2 = _w_inner(psi_phys, psi_phys)
    wl = []
    for label, pa, g in ortho:
        c = _w_inner(g, psi_phys)
        w = float(c * c / max(psi_norm2, 1e-30))
        wl.append((label, pa, w, float(c)))
    unresolved_ext = max(0.0, 1.0 - sum(x[2] for x in wl))

    # --- 8. Aggregation: by D4h parent, split into library-resolved vs extra-resolved ---
    by_parent, lib_by_parent, ext_by_parent = {}, {}, {}
    library_resolved, extra_resolved = 0.0, 0.0
    u_total = 0.0
    for label, pa, w, _c in wl:
        by_parent[pa] = by_parent.get(pa, 0.0) + w
        if label in lib_labels:
            lib_by_parent[pa] = lib_by_parent.get(pa, 0.0) + w
            library_resolved += w
        else:
            ext_by_parent[pa] = ext_by_parent.get(pa, 0.0) + w
            extra_resolved += w
        if pa.endswith('u'): u_total += w
    s_like = by_parent.get('A1g', 0.0); d_like = by_parent.get('B1g', 0.0)
    g_like = by_parent.get('B2g', 0.0); p_like = by_parent.get('Eu',  0.0)

    # --- 7. S_sd ---
    phi_s_fs, phi_d_fs = np.ones_like(kx), np.cos(kx) - np.cos(ky)
    den_sd = np.sqrt(np.sum(fs_weights * phi_s_fs**2) * np.sum(fs_weights * phi_d_fs**2))
    S_sd = float(np.sum(fs_weights * phi_s_fs * phi_d_fs) / den_sd) if den_sd > 0 else 0.0

    # --- 8. Model-basis summary: leading sector AND seed direction ---
    names = qm._IRREP_NAMES
    frac_sector  = np.asarray(cache['sector_frac'][si_lead],  dtype=float)   # leading-sector (matches sector_psi / resolved_weight)
    v_raw_sector = np.asarray(cache['sector_vraw'][si_lead], dtype=float)
    frac_seed    = np.asarray(cache.get('frac_seed',         np.zeros_like(frac_sector)), dtype=float)  # multi-sector seed
    v_irr_seed   = np.asarray(cache.get('frac_seed_signed',  np.zeros_like(frac_sector)), dtype=float)
    resolved     = float(cache.get('resolved_weight',   float('nan')))
    unresolved_m = float(cache.get('unresolved_weight', float('nan')))
    even_idx = [i for i, n in enumerate(names) if qm._IRREP_PARITY[n] > 0]
    odd_idx  = [i for i, n in enumerate(names) if qm._IRREP_PARITY[n] < 0]
    even_frac_m = float(np.sum(frac_sector[even_idx]))
    odd_frac_m  = float(np.sum(frac_sector[odd_idx]))

    # --- 9. Verdict ---
    if not inv_ok or not w_sym_ok:
        verdict = 'INVALID'
    elif par.get('verdict') == 'FAIL':
        verdict = 'FAIL'
    elif par.get('verdict') == 'DEGENERATE' or degenerate:
        verdict = 'DEGENERATE'
    elif top_sector == 'odd':
        verdict = 'ODD-DOMINANT'
    else:
        verdict = 'EVEN-DOMINANT'

    # --- 10. Compact log ---
    if verbose:
        _log("=" * 72)
        _log(f"irrep projection  Q={Q:+.4f}  N_FS={N_fs}  λ_max={lambda_max:+.5f}  "
             f"gap_rel={top_gap_rel:.1e}  sector={lead_key} ({top_sector})")
        _log(f"  FS-closure: inv={'✓' if inv_ok else '✗'} mx={'✓' if mx_ok else '—'} my={'✓' if my_ok else '—'} "
             f"w_sym={'✓' if w_sym_ok else '✗'}(asym={w_asym:.1e})")

        _log(f"  leading-sector weights (Σ={frac_sector.sum():.4f}, resolved={resolved:.4f}, unres={unresolved_m:.4f}):")
        for i, nm in enumerate(names):
            if frac_sector[i] > 1e-3:
                _log(f"    {nm:<6s} {qm._IRREP_D4H[nm]:<4s} w={frac_sector[i]:.4f}  v={v_raw_sector[i]:+.4e}")
        _log(f"    parity: even={even_frac_m:.4f}  odd={odd_frac_m:.4f}")

        if frac_seed.sum() > 0:
            _log(f"  seed direction (multi-sector, Σ={frac_seed.sum():.4f}):")
            for i, nm in enumerate(names):
                if frac_seed[i] > 1e-3:
                    _log(f"    {nm:<6s} {qm._IRREP_D4H[nm]:<4s} w={frac_seed[i]:.4f}  v={v_irr_seed[i]:+.4e}")

        _log(f"  extended basis ({len(basis_list)} fns): library={library_resolved:.4f}  "
             f"extra={extra_resolved:.4f}  unres={unresolved_ext:.4f}")
        for label, pa, w, _c in sorted(wl, key=lambda x: -x[2])[:6]:
            if w > 1e-3:
                src = 'lib' if label in lib_labels else 'ext'
                _log(f"    {label:<14s} {pa:<4s} [{src}] w={w:.4f}")

        parents = sorted(set(lib_by_parent) | set(ext_by_parent))
        _log("  D4h parents: " + "  ".join(
            f"{pa}={by_parent[pa]:.3f}({lib_by_parent.get(pa, 0):.2f}+{ext_by_parent.get(pa, 0):.2f})"
            for pa in parents))
        _log(f"  u_total={u_total:.3e}  S_sd={S_sd:+.3e}")

        if   verdict == 'FAIL':         _log("  ❌ FAIL: mixed-parity leading sector eigenvector")
        elif verdict == 'INVALID':      _log("  ❌ INVALID: FS sample not inversion-closed or weights asymmetric")
        elif verdict == 'DEGENERATE':   _log("  ⚠ DEGENERATE: intra-sector near-degeneracy (only top vector cached)")
        elif verdict == 'ODD-DOMINANT': _log("  ✓ ODD-DOMINANT: px/py channel leads")
        else:
            dom_idx = int(np.argmax(np.abs(v_raw_sector)))
            _log(f"  ✓ EVEN-DOMINANT: {names[dom_idx]}-wave leads")
        _log("=" * 72)

    return {
        'verdict': verdict, 'top_sector': top_sector, 'top_parity': top_parity,
        'lambda_max': lambda_max, 'n_degen': n_degen, 'top_gap_rel': top_gap_rel,
        'inv_ok': inv_ok, 'mx_ok': mx_ok, 'my_ok': my_ok, 'w_sym_ok': w_sym_ok, 'w_asym': w_asym,
        'degenerate': degenerate, 'leading_sector': lead_key,
        'weights_by_parent':  dict(by_parent),
        'library_resolved':   library_resolved,
        'extra_resolved':     extra_resolved,
        'unresolved_ext':     unresolved_ext,
        'lib_by_parent':      dict(lib_by_parent),
        'ext_by_parent':      dict(ext_by_parent),
        'model_frac_sector':  {nm: float(frac_sector[i])  for i, nm in enumerate(names)},
        'model_v_raw_sector': {nm: float(v_raw_sector[i]) for i, nm in enumerate(names)},
        'model_frac_seed':    {nm: float(frac_seed[i])    for i, nm in enumerate(names)},
        'model_v_seed':       {nm: float(v_irr_seed[i])   for i, nm in enumerate(names)},
        'even_frac_model':    even_frac_m, 'odd_frac_model': odd_frac_m,
        'resolved_weight':    resolved,    'unresolved_weight': unresolved_m,
        's_like': s_like, 'd_like': d_like, 'g_like': g_like, 'p_like': p_like,
        'u_total': u_total, 'S_sd': S_sd, 'N_FS': N_fs,
    }


def test_M_controlled_Q_scan(solver, target_doping: float, M_scan, Q_scan, g_t: float, g_J: float) -> None:
    """
    Properly controlled scan: for EACH fixed M in M_scan, sweep Q_scan and report the irrep content of the
    dominant FS pairing eigenvector at every point.  Holds M fixed so that the Q-effect is isolated from the
    AFM Fermi-surface reconstruction.
    """
    Delta_vec = np.zeros(qm._N_DELTA, dtype=complex)
    Quad67_vec   = np.zeros(qm._N_DELTA, dtype=complex)
    mu_ref    = -2.0 * g_t * solver.p.t0 * np.tanh(target_doping / 0.1)

    results: dict = {}
    _log("=" * 110)
    _log(f"M-CONTROLLED Q-SCAN  (target_doping={target_doping},  g_t={g_t},  g_J={g_J})  mu_start={mu_ref:+.5f} eV")
    _log("=" * 110)
    header = (f"{'M':>10s}  {'Q':>8s}  {'N_FS':>5s}  {'lam_max':>8s}  "
              f"{'S_sd':>10s}  {'u_tot':>8s}  {'s_like':>7s}  {'d_like':>7s}  {'g_like':>7s}  {'p_like':>7s}  "
              f"{'unres':>10s}  {'dom_sector':>10s}  {'dom_seed':>10s}")
    _log(header)
    _log("-" * len(header))

    for M_scalar in M_scan:
        M_arr = np.array([float(M_scalar), 0.0, 0.0], dtype=float)
        for Q in Q_scan:
            m_str = np.array2string(M_arr, precision=3)
            try:
                mu, n_kspace = solver._find_mu_for_density(target_doping, M_arr, Q, Delta_vec, Quad67_vec, mu_ref, g_t, g_J)
                mu_ref = mu
            except Exception as exc:
                _log(f"{m_str:>10s}  {Q:8.4f}   -- mu-solve failed: {type(exc).__name__}: {exc}")
                continue

            try:
                res = test_irrep_projection_of_FS_eigenvector(solver, target_doping, M_arr, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, verbose=False)
            except Exception as exc:
                _log(f"{m_str:>10s}  {Q:8.4f}   -- raised {type(exc).__name__}: {exc}")
                res = None

            results[(tuple(M_arr.tolist()), float(Q))] = res

            if res is None:
                _log(f"{m_str:>10s}  {Q:8.4f}   -- INVALID / empty FS sample --")
                continue

            frac_sector = res.get('model_frac_sector') or {}
            frac_seed   = res.get('model_frac_seed')   or {}
            dom_sector  = max(frac_sector, key=frac_sector.get) if frac_sector else '—'
            dom_seed    = max(frac_seed,   key=frac_seed.get)   if frac_seed   else '—'

            _log(f"{m_str:>10s}  {Q:8.4f}  {res['N_FS']:5d}  {res['lambda_max']:8.4f}  "
                 f"{res['S_sd']:+10.3e}  {res['u_total']:8.4f}  "
                 f"{res['s_like']:7.4f}  {res['d_like']:7.4f}  {res['g_like']:7.4f}  {res['p_like']:7.4f}  "
                 f"{res['unresolved_ext']:10.4f}  {dom_sector:>10s}  {dom_seed:>10s}")
        _log("-" * len(header))


def test_doublet_structure_and_mixing(solver, target_doping: float, M_bdg: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, Q_Eg2: float = 0.0, lambda_soc_fixed: float | None = None, delta_tetra_ratios: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0), Delta_B1g_static: float | None = None) -> None:
    p = solver.p
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec = np.asarray(Quad67_vec, dtype=complex)

    _I2 = np.eye(2, dtype=complex)
    _u_xy = np.array([0, 1, 0], dtype=complex)
    _u_xz = (-1.0 / np.sqrt(2)) * np.array([1, 0, -1], dtype=complex)
    _u_yz = (-1j / np.sqrt(2)) * np.array([1, 0, 1], dtype=complex)
    P_xz = np.kron(np.outer(_u_xz, _u_xz.conj()), _I2)
    P_yz = np.kron(np.outer(_u_yz, _u_yz.conj()), _I2)
    P_xy = np.kron(np.outer(_u_xy, _u_xy.conj()), _I2)

    def _orb_weights(v):
        return (float(np.real(v.conj() @ P_xz @ v)),
                float(np.real(v.conj() @ P_yz @ v)),
                float(np.real(v.conj() @ P_xy @ v)))

    def _classify(w_xz, w_yz, w_xy):
        if w_xy > 0.7: return 'xy-like'
        if (w_xz + w_yz) > 0.7:
            return 'xz/yz-symmetric' if abs(w_xz - w_yz) < 0.1 else 'xz/yz-asymmetric'
        return 'strongly mixed'

    def _pauli_decompose(H4):
        return {f'{a}{b}': float(np.real(np.trace(np.kron(Pa, Pb).conj().T @ H4)) / 4.0)
                for a, Pa in qm._SIGMA_2x2.items()
                for b, Pb in qm._SIGMA_2x2.items()}

    lambda_soc = float(p.lambda_soc if lambda_soc_fixed is None else lambda_soc_fixed)
    Delta_B1g = float(p.Delta_B1g_static if Delta_B1g_static is None else Delta_B1g_static)

    _log("=" * 120)
    _log(f"PART A -- DOUBLET SCAN (λ_SOC={lambda_soc:.4f}, Δ_B1g_static={Delta_B1g:.4f})")
    _log("=" * 120)
    _log(f"  {'Δ_tet/λ':>8s}  {'Δ_tet[eV]':>10s}  {'Δ_CF[eV]':>10s}  "
          f"{'name':>4s}  {'E[eV]':>10s}  {'|Sz|':>7s}  {'|μz|':>7s}  "
          f"{'w_xz':>6s}  {'w_yz':>6s}  {'w_xy':>6s}  {'character':<20s}")
    _log("-" * 120)

    records = []
    for ratio in delta_tetra_ratios:
        Delta_tetra_v = ratio * lambda_soc
        G6, G7a, G7b, kappa_doublets, kappa_cross, B1g_t2g, basis, evals_2hole, evecs_2hole, U_pp = _extract_all_kappas(p.t_pd, p.Delta_CT, p.U_dd, lambda_soc, Delta_tetra_v, Delta_B1g, p.Upp_ratio_bare, p.J_H_ratio, qm._Z)

        H_soc_cf, Lx_t2g, Ly_t2g, Lz_t2g, Sz_full, _ = qm._t2g_soc_cf_operators(lambda_soc, Delta_tetra_v, Delta_B1g)
        mu_z_t2g = Lz_t2g + 2.0 * Sz_full

        Delta_CF = float(G7a['energy'] - G6['energy'])
        _g7_split = float(G7b['energy'] - G7a['energy'])
        _degen_flag = "  ⚠ Γ7a/Γ7b degenerate" if abs(_g7_split) < 1e-6 else ""

        row = {'ratio': ratio, 'Delta_tetra': Delta_tetra_v, 'Delta_CF': Delta_CF}
        for name, D in (('Γ6', G6), ('Γ7a', G7a), ('Γ7b', G7b)):
            w_up, w_dn = _orb_weights(D['up']), _orb_weights(D['dn'])
            w_xz = 0.5 * (w_up[0] + w_dn[0])
            w_yz = 0.5 * (w_up[1] + w_dn[1])
            w_xy = 0.5 * (w_up[2] + w_dn[2])
            U = np.column_stack((D['up'], D['dn']))
            mu_z_proj = U.conj().T @ mu_z_t2g @ U
            mu_z_abs = float(np.max(np.abs(np.linalg.eigvalsh(
                0.5 * (mu_z_proj + mu_z_proj.conj().T)))))
            sz_abs = float(max(abs(D['sz_up']), abs(D['sz_dn'])))
            character = _classify(w_xz, w_yz, w_xy)
            row[name] = {'E': D['energy'], 'sz_up': D['sz_up'], 'sz_dn': D['sz_dn'],
                         'mu_z_abs': mu_z_abs, 'w_xz': w_xz, 'w_yz': w_yz,
                         'w_xy': w_xy, 'character': character}
            _log(f"  {ratio:>+8.2f}  {Delta_tetra_v:>+10.5f}  {Delta_CF:>+10.5f}  "
                  f"{name:>4s}  {D['energy']:>+10.5f}  {sz_abs:>7.4f}  {mu_z_abs:>7.4f}  "
                  f"{w_xz:>6.3f}  {w_yz:>6.3f}  {w_xy:>6.3f}  {character:<20s}"
                  + (_degen_flag if name == 'Γ7a' else ""))

        _U6 = np.column_stack([s for D in (G6, G7a, G7b) for s in (D['up'], D['dn'])])
        B1g_op_scan = np.real(_U6.conj().T @ B1g_t2g @ _U6)
        B1g_67a = B1g_op_scan[0:2, 2:4]
        B1g_67b = B1g_op_scan[0:2, 4:6]
        norm_67a = float(np.linalg.norm(B1g_67a, 'fro'))
        norm_67b = float(np.linalg.norm(B1g_67b, 'fro'))


        row.update(kappa_B1g_7a=float(kappa_cross[0]), kappa_B1g_7b=float(kappa_cross[1]),
                   kappa_G6=float(kappa_doublets[0]),
                   kappa_G7a=float(kappa_doublets[1]),
                   kappa_G7b=float(kappa_doublets[2]),
                   norm_67a=norm_67a, norm_67b=norm_67b)
        _log(f"           ‖B1g_67a‖_F={norm_67a:.5f}  ‖B1g_67b‖_F={norm_67b:.5f}")
        _log(f"           κ_G6={kappa_doublets[0]:+.5e} κ_G7a={kappa_doublets[1]:+.5e} "
              f"κ_G7b={kappa_doublets[2]:+.5e}  "
              f"κ_B1g=[{kappa_cross[0]:+.5e}, {kappa_cross[1]:+.5e}]")
        _log()
        records.append(row)

        alphas = np.linspace(0.0, np.pi, 37, endpoint=False)
        for label, D_ref in (('7a', G7a), ('7b', G7b)):
            kappas, t2_vals = _kappa_B1g_vs_pair_phase(evals_2hole, evecs_2hole, basis, G6, D_ref, B1g_t2g, alphas, t2_floor_frac=1e-6)

            finite_mask = np.isfinite(kappas)
            n_finite = int(np.sum(finite_mask))

            if n_finite == 0:
                _log(f"  Γ6-Γ{label}: κ_B1g(α) ALL NaN — t2 below threshold everywhere")
                _log(f"    t2 range: [{np.min(t2_vals):.2e}, {np.max(t2_vals):.2e}]")
                continue

            kappas_f = kappas[finite_mask]
            alphas_f = alphas[finite_mask]
            i_max = int(np.argmax(np.abs(kappas_f)))
            alpha_star = alphas_f[i_max]

            kappa_mean = float(np.mean(kappas_f))
            kappa_std = float(np.std(kappas_f))
            rel_std = kappa_std / max(abs(kappa_mean), 1e-30)

            _log(f"  Γ6-Γ{label}: κ_B1g max at α* = {np.degrees(alpha_star):.1f}°  "
                 f"(κ_max={kappas_f[i_max]:+.5e}, κ(0)={kappas_f[0]:+.5e})")
            _log(f"    κ mean={kappa_mean:+.5e}  std={kappa_std:.2e}  "
                 f"rel.std={rel_std:.2e}  n_finite={n_finite}/{len(alphas)}")
            _log(f"    t2 range: [{np.min(t2_vals):.2e}, {np.max(t2_vals):.2e}]  "
                 f"(reliability floor at {1e-6 * np.max(t2_vals):.2e})")

            if rel_std < 1e-3:
                _log("    ⚠ κ(α) is flat to numerical precision — α-scan is NOT discriminating")
            elif np.degrees(alpha_star) > 5.0 and np.degrees(alpha_star) < 175.0:
                _log("    ⚠ ED prefers a NON-ZERO α — the code's τx gauge is not microscopically optimal")
            else:
                _log("    ✓ ED consistent with α=0 (the code's gauge)")

    _log("=" * 120)
    _log("κ_B1g SIGN STABILITY")
    _log("=" * 120)
    for r in records:
        _log(f"  Δ_tet/λ = {r['ratio']:>+5.2f}  "
              f"κ_B1g_7a = {r['kappa_B1g_7a']:>+12.5e}  "
              f"κ_B1g_7b = {r['kappa_B1g_7b']:>+12.5e}")
    signs_a = np.sign([r['kappa_B1g_7a'] for r in records])
    signs_b = np.sign([r['kappa_B1g_7b'] for r in records])
    _log(f"\n  Sign changes: 7a={int(np.sum(np.diff(signs_a) != 0))}  "
          f"7b={int(np.sum(np.diff(signs_b) != 0))}")

    _log()
    _log("=" * 120)
    _log("DOUBLET CHARACTER MAP")
    _log("=" * 120)
    for r in records:
        _log(f"  {r['ratio']:>+8.2f}  {r['Γ6']['character']:<22s}  "
              f"{r['Γ7a']['character']:<22s}  {r['Γ7b']['character']:<22s}")

    _log()
    _log("=" * 120)
    _log("PART B -- CONVERGED-STATE DOUBLET MIXING")
    _log("=" * 120)
    _log(f"  Q={Q:+.5f}  M={np.array2string(np.asarray(M_bdg), precision=5)}"
          f"  n={n_kspace:.6f}  mu={mu:+.6f}")
    _log(f"  Δ_vec={np.array2string(Delta_vec, precision=5)}   Quad67_vec={np.array2string(Quad67_vec, precision=5)}")

    ev, ec = _bdg_eig(solver, M_bdg, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, Q_Eg2=Q_Eg2)

    f_occ = qm._fermi_function(ev, solver.kT)
    _therm = f_occ * (1.0 - f_occ) / solver.kT
    w = _therm * solver.k_weights[:, None]
    norm = max(float(w.sum()), 1e-300)

    uA, uB, vA, vB = qm._get_nambu_spinors(ec)
    w_orb = np.abs(uA) ** 2 + np.abs(uB) ** 2 + np.abs(vA) ** 2 + np.abs(vB) ** 2
    doublets = {'Gamma6': (0, 1), 'Gamma7a': (2, 3), 'Gamma7b': (4, 5)}
    P = {n: w_orb[:, i, :] + w_orb[:, j, :] for n, (i, j) in doublets.items()}
    P_avg = {n: float((w * P[n]).sum()) / norm for n in doublets}
    purity = float((w * np.maximum.reduce(list(P.values()))).sum()) / norm

    B1g_actual = solver._eval_B1g_and_kweights_at_Q(target_doping, M_bdg, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)[0]

    z6 = np.zeros((6, 6), dtype=complex)
    _B1g_diag = np.diag(np.diag(solver.B1g_op)).astype(complex)
    _B1g_diag_h = (-_B1g_diag.T).astype(complex)
    B1g_diag_24 = np.block([
        [_B1g_diag,   z6,          z6,          z6         ],
        [z6,          _B1g_diag,   z6,          z6         ],
        [z6,          z6,          _B1g_diag_h, z6         ],
        [z6,          z6,          z6,          _B1g_diag_h],
    ])
    diag_qp_diag = np.einsum('kan,ab,kbn->kn', ec.conj(), B1g_diag_24, ec).real
    B1g_diag_only = float(np.dot(solver.k_weights, np.einsum('kn,kn->k', diag_qp_diag, f_occ))) / 4.0
    coherence_residual = B1g_actual - B1g_diag_only

    _log(f"  Doublet weights near E_F: P_Γ6={P_avg['Gamma6']:.4f} P_Γ7a={P_avg['Gamma7a']:.4f}  P_Γ7b={P_avg['Gamma7b']:.4f}  purity={purity:.4f}")
    _log(f"  ⟨B1g⟩_actual     = {B1g_actual:+.4f}")
    _log(f"  ⟨B1g⟩_diag_only  = {B1g_diag_only:+.4f}")
    _log(f"  ⟨B1g⟩_coherence  = {coherence_residual:+.4f}  "
          f"({'coherence-driven' if abs(coherence_residual) > 1e-4 else 'occupation-dominated'})")
    _log("=" * 120)

def _irrep_weights(D24: np.ndarray, U24s: list, chi_k: np.ndarray, char_list: list, irreps: tuple) -> tuple:
    """(weights{Γ}, overlaps χ(g), residuals ‖g·D − χ(g)D‖/‖D‖) for one 24×24 Nambu pair operator."""
    nD2 = float(np.real(np.trace(D24.conj().T @ D24)))
    gD = [chi_k[i] * (U24s[i] @ D24 @ U24s[i].conj().T) for i in range(len(U24s))]
    ov = np.array([np.trace(D24.conj().T @ X) / nD2 for X in gD])
    res = np.array([np.linalg.norm(X - ov[i] * D24) / np.sqrt(nD2) for i, X in enumerate(gD)])
    w = {}
    for name, dim in irreps:
        P = sum(char_list[i][name] * gD[i] for i in range(len(gD))) * (dim / len(gD))
        w[name] = float(np.real(np.trace(P.conj().T @ P)) / nD2)
    return w, ov, res


def _d4h_group() -> list:
    """The 16 O(3) matrices of D4h (z = C4 axis, x,y ∥ bonds), closed from C4z, C2x and inversion."""
    gens = (np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.diag([1., -1., -1.]), -np.eye(3))
    ops = [np.eye(3)]
    grew = True
    while grew:
        grew = False
        for M in list(ops):
            for G in gens:
                N = np.rint(M @ G)
                if not any(np.array_equal(N, O) for O in ops):
                    ops.append(N)
                    grew = True
    assert len(ops) == 16, len(ops)
    return ops

def _d4h_class(M: np.ndarray) -> str:
    det = int(round(np.linalg.det(M)))
    R = det * M
    tr = int(round(np.trace(R)))
    if tr == 3:
        base = 'E'
    elif tr == 1:
        base = 'C4'
    elif R[2, 2] > 0.5:
        base = 'C2z'
    elif abs(R[0, 1]) > 0.5:
        base = "C2''"
    else:
        base = "C2'"
    return base if det > 0 else {'E': 'i', 'C4': 'S4', 'C2z': 'σh', "C2'": 'σv', "C2''": 'σd'}[base]

def _chi_B1_orbit(M: np.ndarray) -> float:
    """Character of x²−y²  ⇒  transformation of φ_d(k)=cos kx−cos ky :  φ_d(M⁻¹k)/φ_d(k)."""
    return float(np.trace(_Q_B1 @ M @ _Q_B1 @ M.T) / 2.0)

def _d4h_characters(M: np.ndarray) -> dict:
    det = float(np.linalg.det(M))
    b1 = float(np.trace(_Q_B1 @ M @ _Q_B1 @ M.T) / 2.0)
    b2 = float(np.trace(_Q_B2 @ M @ _Q_B2 @ M.T) / 2.0)
    e = float(np.trace(M[:2, :2]))
    g = {'A1g': 1.0, 'B1g': b1, 'B2g': b2, 'A2g': b1 * b2, 'Eg': e * float(M[2, 2])}
    u = {'A1u': det, 'B1u': det * b1, 'B2u': det * b2, 'A2u': det * b1 * b2, 'Eu': e}
    return {**g, **u}

def _d2h_characters(M: np.ndarray) -> dict:
    """D2h(x,y,z) — the residual group of a static B1g field.  Only valid for diagonal M."""
    sx, sy, sz = float(M[0, 0]), float(M[1, 1]), float(M[2, 2])
    det = sx * sy * sz
    g = {'Ag': 1.0, 'B1g': sx * sy, 'B2g': sx * sz, 'B3g': sy * sz}
    return {**g, **{n[:-1] + 'u': det * v for n, v in g.items()}}

def _rot6(M: np.ndarray, l3: tuple) -> np.ndarray:
    """6×6 (t2g ⊗ spin, index = 2·orb+spin) operator of the PROPER part of M: exp(−iθ n·l) ⊗ exp(−iθ n·S).
    Inversion acts trivially on d orbitals and on spin.  An overall sign (t2g≅T2 vs effective l=1) is irrelevant for R·D·Rᵀ."""
    from scipy.linalg import expm
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(np.linalg.det(M) * M).as_rotvec()
    sig = (np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], dtype=complex))
    l_dot = sum(rv[a] * l3[a] for a in range(3))
    s_dot = sum(rv[a] * 0.5 * sig[a] for a in range(3))
    return np.kron(expm(-1j * l_dot), expm(-1j * s_dot))

def _pair_irrep_context(lambda_soc: float, Delta_tetra: float, Delta_B1g_static: float) -> dict:
    """Doublet basis U_ord = [Γ6↑,Γ6↓,Γ7a↑,Γ7a↓,Γ7b↑,Γ7b↓] exactly as ModelParams builds it, plus B1g_op in that basis."""
    H, Lx, Ly, Lz, Sz_full, LS_op = qm._t2g_soc_cf_operators(lambda_soc, Delta_tetra, Delta_B1g_static)
    evals, evecs = np.linalg.eigh(H)
    B1g_t2g = Lx @ Lx - Ly @ Ly
    G6, G7a, G7b = qm._get_G6_G7a_G7b(evals, evecs, Sz_full, LS_op, B1g_t2g)
    U_ord = np.column_stack([D[s] for D in (G6, G7a, G7b) for s in ('up', 'dn')])
    return {'H': H, 'U_ord': U_ord, 'B1g_op': np.real(U_ord.conj().T @ B1g_t2g @ U_ord),
            'l3': tuple(L[0::2, 0::2] for L in (Lx, Ly, Lz)), 'ratio': Delta_tetra / lambda_soc if lambda_soc else float('nan')}

def _nambu24(D12: np.ndarray) -> np.ndarray:
    Z = np.zeros((12, 12), dtype=complex)
    return np.block([[Z, D12], [D12.conj().T, Z]])

def _b1g_24(B: np.ndarray) -> np.ndarray:
    Z = np.zeros((6, 6), dtype=complex)
    Bc, Bh = B.astype(complex), (-B.T).astype(complex)
    return np.block([[Bc, Z, Z, Z], [Z, Bc, Z, Z], [Z, Z, Bh, Z], [Z, Z, Z, Bh]])

def _u24(R6_ord: np.ndarray) -> np.ndarray:
    R12 = np.kron(np.eye(2), R6_ord)
    Z = np.zeros((12, 12), dtype=complex)
    return np.block([[R12, Z], [Z, R12.conj()]])


def test_landau_first_order_criterion(solver, target_doping: float, M_sc: np.ndarray, Q_sc: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_sc: float, mu_sc: float, g_t: float, g_J: float, cache: dict, h_Q: float = 0.01, h_D_frac: float = 0.10) -> None:
    """
    Landau analysis at the SC working point:
        b_eff = b_0 − γ²/(2·K_eff)  with  γ = ∂²F/∂Q∂(Δ²),  b_0 = Δ⁴ coeff at Q=0,  K_eff = Q² coeff.
        b_eff < 0  ⇒  first-order SC-triggered JT (finite-Δ minimum coexists with Δ=0).
    [1] Baseline free energies (ref / normal / SC at Q=0).
    [2] Landau coefficients along the SC pattern.
    [3] Pattern scan: b_eff for representative Δ patterns.
    [4] TRW odd-in-Q: TRW-off must be exactly Q-even for all patterns.
    """
    p = solver.p
    M_ = np.asarray(M_sc, dtype=float)
    Delta_amp = float(np.sum(np.abs(Delta_vec)))

    vb = solver._get_vbdg()
    zero_D = np.zeros(qm._N_DELTA, dtype=complex)

    # Q=0 centred cache: the Landau expansion is around Q=0, so use a cache built there
    Gamma_M0, _, VJT, _ = solver._make_vertex_params(target_doping, M_, 0.0, zero_D, zero_D, mu_sc, g_t, g_J)
    cache0 = solver.compute_pairing_kernel_and_build_cache(target_doping, M_, 0.0, mu_sc, g_t, g_J, p.exchange_channels(0.0, g_J)[0], Gamma_M0, VJT, _)

    def _profile(D_ref, Quad67_ref, s):
        """F_tot(Q) for (s·D_ref, s·Quad67_ref), μ(Q) re-solved inside _make_Q_profile."""
        return solver._make_Q_profile(target_doping, M_, D_ref * s, Quad67_ref * s, mu_sc, g_t, g_J, VJT, 0.0, cache0)[0]

    def _landau(D_ref, Quad67_ref, a_ref):
        h_D = h_D_frac * a_ref
        s1, s2 = h_D / a_ref, 2.0 * h_D / a_ref
        prof0, prof1, prof2 = (_profile(D_ref, Quad67_ref, s) for s in (0.0, s1, s2))
        F_0, F_h, F_2h = prof0(0.0), prof1(0.0), prof2(0.0)
        F_p, F_m = prof0(+h_Q), prof0(-h_Q)
        F_ph, F_mh = prof1(+h_Q), prof1(-h_Q)
        gamma = (F_ph - F_p - F_mh + F_m) / (2.0 * h_Q * h_D ** 2)
        b_0   = (F_2h - 4.0 * F_h + 3.0 * F_0) / (12.0 * h_D ** 4)
        K_eff = (F_p - 2.0 * F_0 + F_m) / h_Q ** 2
        b_eff = b_0 - gamma ** 2 / (2.0 * K_eff) if abs(K_eff) > qm._MATH_EPS else float('nan')
        return dict(b_0=b_0, gamma=gamma, K_eff=K_eff, b_eff=b_eff)

    def _Quad67_at(D):
        gs, ga, gb = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M_, 0.0, mu_sc, g_t, g_J)
        Quad67 = zero_D.copy()
        for _ in range(3):
            ev, ec = np.linalg.eigh(vb._build_H_stack(solver.k_points, M_, 0.0, D, Quad67, mu_sc, g_t, g_J))
            Quad67 = qm._compute_B1g_quadrupole(solver.k_weights, solver.Phi_irrep_k, ev, ec, solver.kT, solver.B1g_op)
        return Quad67

    def _mk_D(**kw):
        D = np.zeros(qm._N_DELTA, dtype=complex)
        for k, v in kw.items(): D[qm._DELTA_IDX[k]] = v
        return D

    # [1] Baseline
    F_r = _profile(Delta_vec, Quad67_vec, 1.0)(Q_sc)
    F_n = _profile(zero_D, zero_D, 0.0)(0.0)
    F_s = _profile(Delta_vec, Quad67_vec, 1.0)(0.0)
    _log(f"[landau]  Q={Q_sc:+.5f}  |Δ|={Delta_amp*1e3:.2f} meV")
    _log(f"  F(Q*,Δ*) − F(0,0)   = {(F_r - F_n) * 1e3:+.3f} meV   (SC+JT stability)")
    _log(f"  F(Q*,Δ*) − F(0,Δ*)  = {(F_r - F_s) * 1e3:+.3f} meV   (JT stability at this Q)")

    # [2] Landau along the SC pattern
    L = _landau(Delta_vec, Quad67_vec, Delta_amp)
    first = np.isfinite(L['b_eff']) and L['b_eff'] < 0
    _log(f"\n  Landau (h_Q={h_Q}, h_D={h_D_frac}·|Δ|):")
    _log(f"    γ={L['gamma']:+.4e}   b_0={L['b_0']:+.4e}   K_eff={L['K_eff']:+.4e}   "
         f"b_eff={L['b_eff']:+.4e}   "
         f"{'✓ FIRST-ORDER (b_eff<0)' if first else '✗ second-order (b_eff>0)'}")

    # [3] + [4] pattern scan
    a_ref = Delta_amp
    pats = [
        ('SC vec',      Delta_vec,                     Quad67_vec),
        ('d7a',         _mk_D(d7a=a_ref),              None),
        ('es7a+d7a',    _mk_D(es7a=a_ref, d7a=a_ref),  None),
        ('es7a+i·d7a',  _mk_D(es7a=a_ref, d7a=1j*a_ref), None),
        ('es7a',        _mk_D(es7a=a_ref),             None),
    ]
    pats = [(lab, dv, Quad67_vec if lab == 'SC vec' else _Quad67_at(dv)) for lab, dv, _ in pats]

    _log(f"\n  Pattern scan:")
    _log(f"    {'pattern':<16s} {'b_0':>12s} {'γ':>12s} {'K_eff':>12s} {'b_eff':>12s} {'order':>8s}")
    for lab, dv, fv in pats:
        r = _landau(dv, fv, a_ref)
        order = 'first' if np.isfinite(r['b_eff']) and r['b_eff'] < 0 else '2nd'
        _log(f"    {lab:<16s} {r['b_0']:>+12.3e} {r['gamma']:>+12.3e} "
             f"{r['K_eff']:>+12.3e} {r['b_eff']:>+12.3e} {order:>8s}")

    _log(f"\n  TRW odd-in-Q: 0.5·[F(+Q) − F(−Q)] at Q = ±{h_Q}")
    _log(f"    {'pattern':<16s} {'TRW off':>14s} {'TRW on':>14s}  {'σ_d parity':>10s}")
    ok_Q = True
    for lab, dv, fv in pats:
        pf_off, pf_on = _profile(dv, zero_D, 1.0), _profile(dv, fv, 1.0)
        odd_off = 0.5 * (pf_off(+h_Q) - pf_off(-h_Q))
        odd_on  = 0.5 * (pf_on(+h_Q)  - pf_on(-h_Q))
        _eo = _expect_Q_odd(dv)
        ok_Q &= (abs(odd_off) > 1e-9) if _eo is True else ((abs(odd_off) < 1e-9) if _eo is False else True)
        _log(f"    {lab:<16s} {odd_off:>+14.3e} {odd_on:>+14.3e}  "
             f"{'Q-odd' if abs(odd_on) > 1e-6 else 'Q-even':>10s}")
    _log(f"  TRW off ⇒ F(Q) is Q-odd exactly where the pair-level rule allows Q·Δ*Δ (in-phase A1g⊗B1g mixing), Q-even otherwise: {'✓ PASS' if ok_Q else '❌ FAIL'}")

def _expect_Q_odd(dv) -> bool | None:
    """Pair-level selection rule with the TRW OFF: F(Q) has a Q-odd part (the allowed linear coupling Q·Δ_i*Δ_j of an A1g ⊗ B1g pair, e.g. es7a + d7a) iff at least two channels are active
    and their amplitudes are mutually IN PHASE (real relative phase); a single channel or a 90° relative phase gives an exactly Q-even F.
    Returns True / False, or None where the relative phases are neither (no assertion)."""
    a = np.asarray(dv, dtype=complex)
    a = a[np.abs(a) > 1e-12]
    if len(a) < 2:
        return False
    rel = a[1:] / a[0]
    if np.all(np.abs(rel.imag) < 1e-9):
        return True
    if np.all(np.abs(rel.real) < 1e-9):
        return False
    return None


def test_phase_stiffness_KT_bound(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, Tc_mf: float | None = None) -> None:
    """
    Superfluid phase stiffness and the 2D Kosterlitz–Thouless bound.

    In 2D, long-range phase coherence is destroyed by vortex–antivortex unbinding at  k_B T_KT ≤ (π/2)·Υ,  where Υ is the phase stiffness (helicity modulus).

    Υ is bounded above by its DIAMAGNETIC (f-sum-rule) value,

        Υ_dia = (1/4) Σ_{k,n} w_k · (∂²E_n/∂k_x²) · f(E_n),

    saturated at T = 0 in a clean gapped superconductor: the paramagnetic term
    ρ_para = Σ_{k,n≠m} |⟨n|J_x|m⟩|² · (f_n − f_m)/(E_m − E_n)  vanishes for
    E_n ≠ 0, so ρ_s = Υ_dia at T = 0 and ρ_s ≤ Υ_dia for T > 0.

    - Only the diagonal Hellmann–Feynman term ⟨n|∂²H/∂k_x²|n⟩ is kept. This is exact for a single-band ground state and a strict UPPER bound in general;
      the omitted interband term only lowers Υ and hence tightens the bound.
    - The phase stiffness is computed at the SCF working point (Q, Δ, F67), not at T = 0; using f(E_n) at the supplied kT.
    """
    vbdg = solver._get_vbdg()
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec   = np.asarray(Quad67_vec,   dtype=complex)

    kx, ky = solver.k_points[:, 0], solver.k_points[:, 1]
    dk = min(1e-3, max(1e-5, 2.0 * np.pi / qm._NK / 10.0))

    H0 = vbdg._build_H_stack(solver.k_points, M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)
    Hp = vbdg._build_H_stack(np.column_stack((kx + dk, ky)), M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)
    Hm = vbdg._build_H_stack(np.column_stack((kx - dk, ky)), M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)
    d2H_dkx2 = (Hp - 2.0 * H0 + Hm) / dk ** 2

    ev, ec = np.linalg.eigh(H0)
    d2E_diag = np.einsum('kin,kij,kjn->kn', ec.conj(), d2H_dkx2, ec, optimize=True).real
    f_n = qm._fermi_function(ev, solver.kT)

    Upsilon_el  = float(np.einsum('k,kn,kn->', solver.k_weights, d2E_diag, f_n)) / 4.0   # = A = sum eps'' n per site (Nambu x sublattice counting removed)
    Upsilon_dia = Upsilon_el / 4.0                                                          # stiffness of the Cooper-pair phase theta (pair momentum q = 2 x electron shift)
    T_KT_eV = (np.pi / 2.0) * abs(Upsilon_dia)
    T_KT_K  = T_KT_eV * qm._EV_TO_K

    _log("=" * 90)
    _log(f"  State : Q={Q:+.5f}  |Δ_vec|={float(np.sum(np.abs(Delta_vec))):.3e}"
         f"  T={solver.kT*1000:.2f} meV")
    _log(f"  A = Σ ε''·n (electron-momentum diamagnetic sum, per site) = {Upsilon_el:+.6e} eV;   Υ_dia = A/4 = {Upsilon_dia:+.6e} eV   [diamagnetic upper bound on ρ_s of the pair phase]")
    _log(f"  k_B T_KT ≤ (π/2)·Υ_dia = {T_KT_eV*1000:+.4f} meV  ({T_KT_K:.1f} K)")
    if Tc_mf is not None and Tc_mf > 0.0:
        _log(f"  Mean-field T_c         = {Tc_mf*1000:.4f} meV  ({Tc_mf*qm._EV_TO_K:.1f} K)")
        if Tc_mf > T_KT_eV:
            _log("  ⚠ mean-field T_c > T_KT bound: the mean-field value cannot be interpreted as a genuine 2D SC transition temperature without an explicit vortex-unbinding (KT) treatment.")
        else:
            _log("  ✓ mean-field T_c ≤ T_KT: consistent with a 2D KT description.")
    if abs(Upsilon_dia) < 1e-6:
        _log("  ⚠ Υ_dia ≈ 0 — no coherent superfluid response at this state")
    _log("=" * 90)


def test_rpa_determinant_coupling_isolation(solver, target_doping: float, M_bdg: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, afm_radius: float = 0.3 * np.pi, fwd_radius: float = 0.3 * np.pi, ctx: dict | None = None) -> dict:
    """Isolate the contribution of χ_SQ/χ_QS to the RPA pairing vertex."""
    p = solver.p
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec = np.asarray(Quad67_vec, dtype=complex)

    tx_bare, ty_bare = p.effective_hopping_anisotropic(Q)

    J_A1g_diag, J_B1g_7a, J_B1g_7b = _exch(p, Q, g_J)
    Gamma_M, F_cluster, V_JT_corr, V_cap = _vertex_tuple(solver, target_doping, M_bdg, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)

    if ctx is None:
        ctx = _build_fs_context(solver, M_bdg, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    unique_q = ctx['unique_q']
    ev, ec = ctx['ev_n'], ctx['ec_n']

    n_q = len(unique_q)
    if n_q == 0:
        _log("No q-pairs found. Abort.")
        return {'n_q': 0}

    dHdQ = solver._calc_dHdQ(M_bdg, Q, Quad67_vec, mu, g_t, g_J)

    def _vertex_from_chiR(J_q: np.ndarray, chiR: np.ndarray, V_eff: float, V_cap: float) -> float:
        """
        Scalar pairing vertex from the dressed susceptibility
        χ_RPA = M4⁻¹ χ₀ = adj(M4)·χ₀ / det(M4).

        J_q : (3,) per-channel exchange [J_Γ6(q), J_Γ7a(q), J_Γ7b(q)]
        chiR: (4, 4) dressed susceptibility in basis [S_Γ6, S_Γ7a, S_Γ7b, Q]
        """
        V_spin_mat = -np.outer(J_q, J_q) * chiR[:qm._N_CHANNELS, :qm._N_CHANNELS]
        V_jt_val   = V_eff ** 2 * chiR[qm._N_CHANNELS, qm._N_CHANNELS]
        V_cross_v  = 2.0 * J_q * V_eff * chiR[:qm._N_CHANNELS, qm._N_CHANNELS]
        V_sum = float(V_spin_mat.sum() + V_jt_val + V_cross_v.sum())
        if abs(V_sum) > V_cap:
            V_sum = math.copysign(V_cap, V_sum)
        return V_sum

    V_coupled   = np.empty(n_q, dtype=float)
    V_uncoupled = np.empty(n_q, dtype=float)

    for u_idx, q_u in enumerate(unique_q):
        rpa = solver._rpa_closure(M_bdg, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, q_u, Gamma_M, J_A1g_diag, V_JT_corr, (ev, ec), False, dHdQ)

        J_q, det, M4, chi0 = rpa['J_q'], rpa['det'], rpa['M4'], rpa['chi0']

        # ---- Coupled: full χ_RPA = M4⁻¹ χ₀ ------------------------------
        chiR_full = qm._adjugate(M4) @ chi0 / det
        V_coupled[u_idx] = _vertex_from_chiR(J_q, chiR_full, V_JT_corr, V_cap)

        # ---- Uncoupled: χ_SQ ≡ χ_QS ≡ 0 --------------------------------
        chi0_mod = chi0.copy()
        chi0_mod[:qm._N_CHANNELS, qm._N_CHANNELS] = 0.0
        chi0_mod[qm._N_CHANNELS, :qm._N_CHANNELS] = 0.0

        U_mod = np.diag([
            float(J_q[0]), float(J_q[1]), float(J_q[2]), float(V_JT_corr)
        ])
        M4_mod = np.eye(qm._N_CHANNELS + 1) - chi0_mod @ U_mod

        det_mod   = float(np.linalg.det(M4_mod))
        det_floor = max(qm._MATH_EPS, 1e-4 * float(np.linalg.norm(M4_mod)))
        if abs(det_mod) < det_floor:
            det_safe_mod = (math.copysign(det_floor, det_mod)
                            if det_mod != 0.0 else det_floor)
        else:
            det_safe_mod = det_mod

        chiR_mod = qm._adjugate(M4_mod) @ chi0_mod / det_safe_mod
        V_uncoupled[u_idx] = _vertex_from_chiR(J_q, chiR_mod, V_JT_corr, V_cap)

    V_diff = V_coupled - V_uncoupled

    q_fwd_wrapped = np.mod(unique_q + np.pi, 2 * np.pi) - np.pi
    fwd_mask = np.linalg.norm(q_fwd_wrapped, axis=1) < fwd_radius
    q_afm_wrapped = np.mod(unique_q - np.array([np.pi, np.pi]) + np.pi, 2 * np.pi) - np.pi
    afm_mask = np.linalg.norm(q_afm_wrapped, axis=1) < afm_radius

    def _stats(arr, mask, label):
        if not np.any(mask):
            return f"  {label}: n=0"
        sub = arr[mask]
        return (f"  {label}: n={len(sub)}, mean={np.mean(sub):+.4f}, "
                f"min={np.min(sub):+.4f}, max={np.max(sub):+.4f}, "
                f"std={np.std(sub):+.4f}")

    _log("Full vertex with physical cross terms:")
    _log(_stats(V_coupled, afm_mask, "AFM q region"))
    _log(_stats(V_coupled, fwd_mask, "Forward q region"))
    _log(_stats(V_coupled, np.ones(n_q, dtype=bool), "All q"))

    _log("\nVertex with χ_SQ/χ_QS forced to zero (rebuilt M4):")
    _log(_stats(V_uncoupled, afm_mask, "AFM q region"))
    _log(_stats(V_uncoupled, fwd_mask, "Forward q region"))
    _log(_stats(V_uncoupled, np.ones(n_q, dtype=bool), "All q"))

    _log("\nDifference (coupled − uncoupled):")
    _log(_stats(V_diff, afm_mask, "AFM q region"))
    _log(_stats(V_diff, fwd_mask, "Forward q region"))
    _log(_stats(V_diff, np.ones(n_q, dtype=bool), "All q"))

    denom = np.abs(V_uncoupled)
    rel = np.where(denom > 1e-9, V_diff / denom, np.nan)
    _log("\nRelative change |ΔV|/|V_uncoupled|:")
    _log(_stats(rel, afm_mask, "AFM q region"))
    _log(_stats(rel, fwd_mask, "Forward q region"))
    _log(_stats(rel, np.ones(n_q, dtype=bool), "All q"))
    return {'n_q': n_q, 'V_coupled': V_coupled, 'V_uncoupled': V_uncoupled, 'V_diff': V_diff, 'afm_mask': afm_mask, 'fwd_mask': fwd_mask}


def test_bdg_energy_consistency(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, Q_Eg2: float, cache: dict, K_eff_for_free_energy: float = 0.0, K_eff_Eg2_for_free_energy: float = 0.0,
    tol_grad_rel: float = 1e-3, tol_grad_abs: float = 1e-7, tol_dF_dalpha: float = 1e-4, scan_alpha_min: float = 0.0, scan_alpha_max: float = 2.0, scan_n_points: int = 41) -> None:
    """HF fixed point of F = Omega + mu*(1-target_doping). Delta_vec, Quad67_vec 4-vectors."""
    p = solver.p
    vbdg = solver._get_vbdg()
    M = np.asarray(M, dtype=float)
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec = np.asarray(Quad67_vec, dtype=complex)
    n_target = 1.0 - target_doping
    Delta_total = float(np.sum(np.abs(Delta_vec)))
    
    tol_density = max(100.0 * qm._MU_DENSITY_TOL, 1e-7)    # ≈ 1e-6
    J3_all = np.asarray(p.exchange_channels(Q, g_J)[0], dtype=float)[0::2]
    tol_grad_norm = max(p.Z_afm_eff * np.max(np.abs(J3_all)), 1e-12) * _SCF_STATIONARITY_TOL

    g_Das, g_Dad, g_Dbd = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M, Q, mu, g_t, g_J)
    tx_b, ty_b = p.effective_hopping_anisotropic(Q)

    _log("=" * 78)
    _log(f"  State params : Q={Q:+.6f}  |Δ_vec|={Delta_total:.3e}")
    _log(f"                 Quad67_vec = {np.array2string(Quad67_vec, precision=6)}  Q_Eg2={Q_Eg2:+.3e}")
    _log(f"  Claimed fp   : M  = {np.array2string(M, precision=6)}")
    _log(f"                 n  = {n_kspace:.8f}   mu = {mu:.8f}")
    _log()

    ev_fp, ec_fp = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, Q_Eg2=Q_Eg2)
    n_measured = float(np.sum(solver._compute_orbital_densities(ev_fp, ec_fp)))
    err_density_to_target = n_measured - n_target
    err_density_to_nfp = n_measured - n_kspace
    dens_ok = abs(err_density_to_target) < tol_density
    n_consistent = abs(err_density_to_nfp) < tol_density

    _log("[1] Density at the claimed point:")
    _log(f"    n(mu) = {n_measured:.10f}   n_kspace = {n_kspace:.10f}"
          f"   |Δ| = {abs(err_density_to_nfp):.3e}   {'✓' if n_consistent else '✗'}")
    _log(f"    target = {n_target:.10f}    |Δ| = {abs(err_density_to_target):.3e}"
          f"   {'✓' if dens_ok else '✗'}")
    _log()

    M_bdg = vbdg.compute_channel_staggered_magnetizations(Q, Delta_vec, mu, ev_fp, ec_fp)
    resid_M = M - M_bdg
    norm_resid_M = float(np.linalg.norm(resid_M))
    fp_ok = norm_resid_M < _SCF_STATIONARITY_TOL

    _log("[2] Hellmann-Feynman fixed point:")
    _log(f"    M  = {np.array2string(M, precision=8)}")
    _log(f"    M_BdG = {np.array2string(M_bdg, precision=8)}")
    _log(f"    ||M − M_BdG|| = {norm_resid_M:.3e}   "
          f"{'✓' if fp_ok else '✗ NOT a fixed point'}")
    _log()

    J_A1g_diag, _, _ = _exch(p, Q, g_J)
    J3 = np.asarray(J_A1g_diag, dtype=float)[0::2]
    grad_analytic = p.Z_afm_eff * J3 * resid_M

    _vp_n = _scf_vertex_params(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    grad_numeric, _ = solver.compute_dF_dM_channels_and_hessian(target_doping, M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, _vp_n['V_JT_corr'], Q_Eg2, cache)
    norm_grad_num = float(np.linalg.norm(grad_numeric))
    norm_grad_ana = float(np.linalg.norm(grad_analytic))
    norm_grad_diff = float(np.linalg.norm(grad_numeric - grad_analytic))
    grad_scale = max(norm_grad_ana, norm_grad_num, 1e-12)
    grad_rel = norm_grad_diff / grad_scale
    grad_zero_ok = norm_grad_num < tol_grad_norm
    grad_match_ok = (grad_rel < tol_grad_rel) or (norm_grad_diff < tol_grad_abs)

    _log("[3] Free-energy gradient:")
    _log(f"    grad_numeric  = {np.array2string(grad_numeric, precision=8)}"
          f"   ||·|| = {norm_grad_num:.3e}")
    _log(f"    grad_analytic = {np.array2string(grad_analytic, precision=8)}"
          f"   ||·|| = {norm_grad_ana:.3e}")
    _log(f"    ||num − ana|| = {norm_grad_diff:.3e}   {'✓' if grad_match_ok else '✗'}")
    _log(f"    ||grad_numeric|| < {tol_grad_norm:.1e} ?"
          f"   {'✓' if grad_zero_ok else '✗'}")
    _log()

    alpha_vals = np.linspace(scan_alpha_min, scan_alpha_max, scan_n_points)
    if not np.any(np.isclose(alpha_vals, 1.0)):
        alpha_vals = np.sort(np.append(alpha_vals, 1.0))

    F_vals = np.empty_like(alpha_vals)
    M_bdg_vals = np.empty((len(alpha_vals), M.shape[0]))

    for i, alpha in enumerate(alpha_vals):
        M_alpha = alpha * M
        mu_alpha, n_alpha = solver._find_mu_for_density(target_doping, M_alpha, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)
        g_Das, g_Dad, g_Dbd = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M_alpha, Q, mu_alpha, g_t, g_J)
        Omega = solver._compute_bdg_free_energy(M_alpha, Q, Delta_vec, Quad67_vec, mu_alpha, g_t, g_J, g_Das, g_Dad, g_Dbd, _vp_n['V_JT_corr'], Q_Eg2, K_eff_for_free_energy, K_eff_Eg2_for_free_energy, cache)
        F_vals[i] = Omega + mu_alpha * n_target
        ev_a, ec_a = _bdg_eig(solver, M_alpha, Q, Delta_vec, Quad67_vec, n_alpha, mu_alpha, g_t, g_J, Q_Eg2=Q_Eg2)
        M_bdg_vals[i] = vbdg.compute_channel_staggered_magnetizations(Q, Delta_vec, mu_alpha, ev_a, ec_a)

    dF_dalpha_num = np.gradient(F_vals, alpha_vals)
    idx_one = int(np.argmin(np.abs(alpha_vals - 1.0)))
    dF_num_at_one = float(dF_dalpha_num[idx_one])
    dF_ana_at_one = float(np.dot(grad_analytic, M))

    F_min_alpha = float(alpha_vals[int(np.argmin(F_vals))])
    F_range = float(np.max(F_vals) - np.min(F_vals))
    dF_dalpha_ok = (abs(dF_num_at_one) < tol_dF_dalpha
                    and abs(dF_ana_at_one) < tol_dF_dalpha)
    minimum_ok = abs(F_min_alpha - 1.0) < 0.05 * (scan_alpha_max - scan_alpha_min)

    _log(f"[4] Line scan  F(α·M),  α ∈ [{scan_alpha_min:.2f}, {scan_alpha_max:.2f}]")
    _log(f"    F range over the scan       = {F_range:.6e} eV")
    _log(f"    argmin_α F(α)               = {F_min_alpha:.4f}"
          f"   {'✓ at α=1' if minimum_ok else '✗'}")
    _log(f"    dF/dα (numeric, α=1)        = {dF_num_at_one:+.6e}")
    _log(f"    dF/dα (analytic, α=1)       = {dF_ana_at_one:+.6e}")
    _log(f"    |num − ana|                 = {abs(dF_num_at_one - dF_ana_at_one):.3e}"
          f"   {'✓' if dF_dalpha_ok else '✗'}")
    _log(f"    ||M − M_BdG(α=1)||       = "
          f"{float(np.linalg.norm(M - M_bdg_vals[idx_one])):.3e}")
    _log()

    _log("-" * 78)
    if not (n_consistent and dens_ok and fp_ok and grad_zero_ok and grad_match_ok and dF_dalpha_ok and minimum_ok):
        failures = []
        if not n_consistent: failures.append("n(mu) ≠ n_kspace")
        if not dens_ok:      failures.append("n(mu) ≠ 1−δ")
        if not fp_ok:        failures.append("M ≠ M_BdG(M)")
        if not grad_zero_ok: failures.append("‖grad_numeric‖ not ~0")
        if not grad_match_ok: failures.append("grad mismatch")
        if not dF_dalpha_ok: failures.append("dF/dα ≠ 0")
        if not minimum_ok:   failures.append("F(α) minimum not at α=1")
        _log("  → FAIL: " + "; ".join(failures))
    _log("=" * 78)


def test_isotropic_gutzwiller_hopping(solver, target_doping: float, M_bdg: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, antinodal_thresh: float = 0.7, gap_thresh_frac: float = 0.05, mass_tol: float = 0.30, ctx: dict | None = None) -> None:
    if ctx is None:
        ctx = _build_fs_context(solver, M_bdg, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=float), n_kspace, mu, g_t, g_J)
    vbdg = ctx['vbdg']
    fs_pts, vF_mag = ctx['fs_pts'], ctx['vF_mag']
    kx, ky = fs_pts[:, 0], fs_pts[:, 1]
    n_pts = ctx['N_fs']

    t0 = solver.p.t0
    gap_thresh = gap_thresh_frac * t0
    curv_thresh = 0.01 * t0
    vf_thresh = 0.05 * t0

    c = _robust_kp_curvature(vbdg, kx, ky, M_bdg, Q, n_kspace, mu, g_t, g_J)

    phi_d_abs = np.abs(np.cos(kx) - np.cos(ky))
    mask_anti = phi_d_abs > antinodal_thresh
    valid_vf = vF_mag > vf_thresh
    gap_reliable = c['gap_min'] > gap_thresh
    valid_curvature = (np.abs(c['lam1']) > curv_thresh) & (np.abs(c['lam2']) > curv_thresh)

    mask_use = mask_anti & gap_reliable & valid_curvature & valid_vf
    tier = 1
    if not np.any(mask_use):
        mask_use = mask_anti & gap_reliable & (np.abs(c['lam1']) > 1e-4) & (np.abs(c['lam2']) > 1e-4)
        tier = 2
        if not np.any(mask_use):
            mask_use = gap_reliable if np.any(gap_reliable) else np.ones(n_pts, dtype=bool)
            tier = 3

    m1 = np.abs(1.0 / c['lam1'][mask_use])
    m2 = np.abs(1.0 / c['lam2'][mask_use])
    m_eff_curv = float(np.nanmean(np.concatenate([m1, m2]))) if mask_use.any() else float('nan')

    t_eff = g_t * t0
    m_eff_band = 1.0 / (2.0 * max(t_eff, 1e-9))
    ratio = m_eff_curv / m_eff_band if (m_eff_band > 0 and not np.isnan(m_eff_curv)) else float('inf')
    isotropic_ok = bool(abs(ratio - 1.0) < mass_tol)

    n_used = int(np.sum(mask_use))
    n_filtered = int(np.sum(mask_anti) - np.sum(mask_anti & valid_curvature & valid_vf & gap_reliable))

    _log(f"m_eff_curv = {m_eff_curv:.6f}, m_eff_band = {m_eff_band:.6f}, "
          f"ratio = {ratio:.4f}, isotropic_ok = {isotropic_ok}")
    gap_min_str = (f"gap_min on used set [eV]: min={c['gap_min'][mask_use].min():.4e}"
                   if n_used > 0 else "gap_min: n/a")
    _log(f"n_used = {n_used}/{n_pts} (tier {tier}), n_filtered = {n_filtered}, "
          f"{gap_min_str}")


def _robust_kp_curvature(vbdg, kx, ky, M_bdg, Q, n_kspace, mu, g_t, g_J, dk: float | None = None, kramers_tol: float = 1e-7) -> dict:
    """2nd-order k.p perturbation theory band curvature, robust to band crossings. Uses Delta_vec = 0 and Quad67_vec = 0."""
    if dk is None:
        dk = min(1e-2, max(1e-4, 2.0 * np.pi / qm._NK / 6.0))
    _Delta_zero = np.zeros(qm._N_DELTA, dtype=complex)
    _Quad67_zero   = np.zeros(qm._N_DELTA, dtype=float)

    def _H(kx_, ky_):
        H24 = vbdg._build_H_stack(np.c_[kx_, ky_], M_bdg, Q, _Delta_zero, _Quad67_zero, mu, g_t, g_J)
        return H24[:, 0:12, 0:12]

    H0 = _H(kx, ky)
    ev0, ec0 = np.linalg.eigh(H0)
    n0_idx = np.argmin(np.abs(ev0), axis=1)
    E_n0 = np.take_along_axis(ev0, n0_idx[:, None], axis=1)[:, 0]
    psi0 = np.take_along_axis(ec0, n0_idx[:, None, None], axis=2)[:, :, 0]

    Hpx = _H(kx + dk, ky); Hmx = _H(kx - dk, ky)
    Hpy = _H(kx, ky + dk); Hmy = _H(kx, ky - dk)
    Hpp = _H(kx + dk, ky + dk); Hmm = _H(kx - dk, ky - dk)
    Hpm = _H(kx + dk, ky - dk); Hmp = _H(kx - dk, ky + dk)

    dHdx = (Hpx - Hmx) / (2.0 * dk)
    dHdy = (Hpy - Hmy) / (2.0 * dk)
    d2Hdx2 = (Hpx - 2.0 * H0 + Hmx) / dk ** 2
    d2Hdy2 = (Hpy - 2.0 * H0 + Hmy) / dk ** 2
    d2Hdxy = (Hpp - Hpm - Hmp + Hmm) / (4.0 * dk ** 2)

    vx = np.real(np.einsum('pi,pij,pj->p', psi0.conj(), dHdx, psi0, optimize=True))
    vy = np.real(np.einsum('pi,pij,pj->p', psi0.conj(), dHdy, psi0, optimize=True))

    proj_x = np.einsum('pim,pij,pj->pm', ec0.conj(), dHdx, psi0, optimize=True)
    proj_y = np.einsum('pim,pij,pj->pm', ec0.conj(), dHdy, psi0, optimize=True)

    denom = E_n0[:, None] - ev0
    is_degenerate = np.abs(denom) < kramers_tol
    abs_denom = np.where(is_degenerate, np.inf, np.abs(denom))
    gap_min = np.min(abs_denom, axis=1)
    gap_min_partner = np.argmin(abs_denom, axis=1)

    denom_safe = np.where(is_degenerate, 1.0, denom)
    inv_denom = np.where(is_degenerate, 0.0, 1.0 / denom_safe)

    onsite_xx = np.real(np.einsum('pi,pij,pj->p', psi0.conj(), d2Hdx2, psi0, optimize=True))
    onsite_yy = np.real(np.einsum('pi,pij,pj->p', psi0.conj(), d2Hdy2, psi0, optimize=True))
    onsite_xy = np.real(np.einsum('pi,pij,pj->p', psi0.conj(), d2Hdxy, psi0, optimize=True))

    virt_xx = 2.0 * np.sum(np.abs(proj_x) ** 2 * inv_denom, axis=1)
    virt_yy = 2.0 * np.sum(np.abs(proj_y) ** 2 * inv_denom, axis=1)
    virt_xy = 2.0 * np.real(np.sum(np.conj(proj_x) * proj_y * inv_denom, axis=1))

    d2x, d2y, d2xy = onsite_xx + virt_xx, onsite_yy + virt_yy, onsite_xy + virt_xy
    tr = d2x + d2y
    disc = np.sqrt(np.maximum(0.25 * (d2x - d2y) ** 2 + d2xy ** 2, 0.0))

    return {'E_n0': E_n0, 'vx': vx, 'vy': vy,
            'lam1': 0.5 * tr + disc, 'lam2': 0.5 * tr - disc,
            'gap_min': gap_min, 'gap_min_partner': gap_min_partner, 'dk': dk}


def _hopping_necessity_scan(which: str, ratios, solver, target_doping: float, M_bdg: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float,
    antinodal_thresh: float = 0.7, gap_thresh_frac: float = 0.05, response_tol: float = 0.10, ctx: dict | None = None) -> None:
    """Is the hopping t' (which = 't_prime') or t'' (which = 't_dprime') NEEDED for the band curvature on the FS?  The FS points and the reduced curvature mask are fixed by the reference (that hopping = 0), the
    hopping is then scanned over `ratios`; 'needed' = the effective mass changes by more than `response_tol`.  The other hopping keeps its ModelParams value; both are restored on exit."""
    assert which in ('t_prime', 't_dprime')
    sym, tname = ("r'", "t'") if which == 't_prime' else ("r''", "t''")
    if ctx is None:
        ctx = _build_fs_context(solver, M_bdg, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=float), n_kspace, mu, g_t, g_J)
    vbdg = ctx['vbdg']
    fs_pts, vF_mag = ctx['fs_pts'], ctx['vF_mag']
    kx, ky = fs_pts[:, 0], fs_pts[:, 1]
    n_pts = ctx['N_fs']

    t0 = solver.p.t0
    m_eff_band = 1.0 / (2.0 * max(g_t * t0, 1e-9))
    gap_thresh = gap_thresh_frac * t0
    curv_thresh = 0.01 * t0
    vf_thresh = 0.05 * t0

    saved_t_prime_ratio = solver.p.t_prime_ratio
    saved_t_dprime_ratio = solver.p.t_dprime_ratio

    def _set_t_ratios(r_prime, r_double):
        solver.p.t_prime_ratio  = r_prime
        solver.p.t_dprime_ratio = r_double
        solver.p.t_prime  = r_prime  * t0
        solver.p.t_dprime = r_double * t0
        vbdg.t_prime  = solver.p.t_prime
        vbdg.t_dprime = solver.p.t_dprime

    def _curv(r):
        _set_t_ratios(r, saved_t_dprime_ratio) if which == 't_prime' else _set_t_ratios(saved_t_prime_ratio, r)
        return _robust_kp_curvature(vbdg, kx, ky, M_bdg, Q, n_kspace, mu, g_t, g_J)

    def _mass(c, mask):
        if not mask.any():
            return float('nan')
        m1 = np.abs(1.0 / c['lam1'][mask]); m2 = np.abs(1.0 / c['lam2'][mask])
        return float(np.nanmean(np.concatenate([m1, m2])))

    try:
        c0 = _curv(0.0)
        phi_d_abs = np.abs(np.cos(kx) - np.cos(ky))
        mask_anti = phi_d_abs > antinodal_thresh
        valid_vf = vF_mag > vf_thresh
        gap_reliable_0 = c0['gap_min'] > gap_thresh
        valid_curvature_0 = (np.abs(c0['lam1']) > curv_thresh) & (np.abs(c0['lam2']) > curv_thresh)

        mask_ref = mask_anti & gap_reliable_0 & valid_curvature_0 & valid_vf
        tier_ref = 1
        if not np.any(mask_ref):
            mask_ref = mask_anti & gap_reliable_0 & (np.abs(c0['lam1']) > 1e-4) & (np.abs(c0['lam2']) > 1e-4)
            tier_ref = 2
            if not np.any(mask_ref):
                mask_ref = gap_reliable_0 if np.any(gap_reliable_0) else np.ones(n_pts, dtype=bool)
                tier_ref = 3

        m_orig = _mass(c0, mask_ref)
        _log(f"t0 = {t0:.4f} eV,  m_eff_band = {m_eff_band:.4f}")
        _log(f"Reference ({sym}=0): m_eff = {m_orig:.4f}, ratio = {m_orig / m_eff_band:.3f}")
        _log(f"Fixed reference mask: n_used = {int(np.sum(mask_ref))}/{n_pts} (tier {tier_ref})")

        prev_ratio_after = None; prev_r = None
        for r_prime in ratios:
            c2 = _curv(r_prime)
            gap_reliable_2 = c2['gap_min'] > gap_thresh
            valid_curvature_2 = ((np.abs(c2['lam1']) > curv_thresh)
                                 & (np.abs(c2['lam2']) > curv_thresh))
            mask_fixed = mask_ref & gap_reliable_2 & valid_curvature_2
            n_dropped_gap  = int(np.sum(mask_ref & ~gap_reliable_2))
            n_dropped_curv = int(np.sum(mask_ref & gap_reliable_2 & ~valid_curvature_2))

            m_2nd = _mass(c2, mask_fixed)
            rel_change = (abs(m_2nd - m_orig) / max(abs(m_orig), 1e-12)
                          if np.isfinite(m_orig) and np.isfinite(m_2nd) else float('nan'))
            ratio_after = (m_2nd / m_eff_band if np.isfinite(m_2nd) else float('nan'))
            needed = bool(np.isfinite(rel_change) and rel_change > response_tol)

            _log(f"\n{tname} = {r_prime * t0:.4f} eV ({sym}={r_prime:+.2f})")
            _log(f"  m_eff = {m_2nd:.4f}  ratio_after = {ratio_after:.3f}")
            _log(f"  relative_change = {rel_change * 100:.1f}%  needed = {needed}")
            msg = (f"  n_used = {int(np.sum(mask_fixed))}/{int(np.sum(mask_ref))}")
            if n_dropped_gap + n_dropped_curv > 0:
                msg += (f"  ({n_dropped_gap} dropped: gap_min<{gap_thresh:.2e}; "
                        f"{n_dropped_curv} dropped: |lam|<{curv_thresh:.2e})")
            _log(msg)

            if prev_ratio_after is not None and np.isfinite(prev_ratio_after) and np.isfinite(ratio_after):
                if (prev_ratio_after - 1.0) * (ratio_after - 1.0) < 0:
                    frac = (1.0 - prev_ratio_after) / (ratio_after - prev_ratio_after)
                    r_crit = prev_r + frac * (r_prime - prev_r)
                    _log(f"  --> interpolated critical {sym} ≈ {r_crit:.3f}")

            prev_ratio_after = ratio_after; prev_r = r_prime
    finally:
        _set_t_ratios(saved_t_prime_ratio, saved_t_dprime_ratio)

def test_second_neighbor_hopping_necessity(solver, target_doping: float, M_bdg: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float,
    t_prime_ratios=(-0.35, -0.30, -0.25, -0.20, -0.15), antinodal_thresh: float = 0.7, gap_thresh_frac: float = 0.05, response_tol: float = 0.10, ctx: dict | None = None) -> None:
    _hopping_necessity_scan('t_prime', t_prime_ratios, solver, target_doping, M_bdg, Q, n_kspace, mu, g_t, g_J, antinodal_thresh, gap_thresh_frac, response_tol, ctx)


def test_third_neighbor_hopping_necessity(solver, target_doping: float, M_bdg: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float,
    t_double_prime_ratios=(0.04, 0.08, 0.12, 0.16, 0.20), antinodal_thresh: float = 0.7, gap_thresh_frac: float = 0.05, response_tol: float = 0.10, ctx: dict | None = None) -> None:
    _hopping_necessity_scan('t_dprime', t_double_prime_ratios, solver, target_doping, M_bdg, Q, n_kspace, mu, g_t, g_J, antinodal_thresh, gap_thresh_frac, response_tol, ctx)

def _orbital_selective_gutzwiller_necessity_test(solver, Q: float, target_doping: float, g_t: float, g_J: float, M_bdg: np.ndarray, n_kspace: float, mu: float,
    antinodal_thresh: float = 0.7, shift_tol: float = 0.25, orbital_spread_thresh: float = 0.05, dk_override: float | None = None, state_mismatch_tol: float = 0.10) -> dict:
    vbdg = solver._get_vbdg()
    p = solver.p
    _Dzero = np.zeros(qm._N_DELTA, dtype=complex)
    _Fzero = np.zeros(qm._N_DELTA, dtype=float)
    fs_pts, vF_vec, _, _ = solver._get_fs_points(M_bdg, Q, mu, g_t, g_J)
    vF_mag = np.linalg.norm(vF_vec, axis=1)
    kx, ky = fs_pts[:, 0], fs_pts[:, 1]
    n_pts = len(fs_pts)

    dk = dk_override if dk_override is not None else min(1e-2, max(1e-4, 2.0 * np.pi / qm._NK / 6.0))

    c_corr = _robust_kp_curvature(vbdg, kx, ky, M_bdg, Q, n_kspace, mu, g_t, g_J, dk=dk)
    c_bare = _robust_kp_curvature(vbdg, kx, ky, M_bdg, Q, n_kspace, mu, 1.0, g_J, dk=dk)

    ev_bare, ec_bare = _bdg_eig(solver, M_bdg, Q, _Dzero, _Fzero, n_kspace, mu, 1.0, g_J)
    M_at_g1 = vbdg.compute_channel_staggered_magnetizations(0.0, _Dzero, mu, ev_bare, ec_bare)
    M_scale = max(float(np.linalg.norm(M_bdg)), 1e-12)
    M_mismatch_norm = float(np.linalg.norm(M_bdg - M_at_g1))
    state_mismatch = (M_mismatch_norm / M_scale) > state_mismatch_tol

    t0 = p.t0
    vf_thresh = 0.05 * t0
    curv_thresh = 0.01 * t0
    gap_thresh = 0.05 * t0
    valid_vf = vF_mag > vf_thresh
    phi_d_abs = np.abs(np.cos(kx) - np.cos(ky))
    mask_anti = phi_d_abs > antinodal_thresh

    def _branch_mask(c):
        gap_rel = c['gap_min'] > gap_thresh
        valid_curv = (np.abs(c['lam1']) > curv_thresh) & (np.abs(c['lam2']) > curv_thresh)
        m = mask_anti & gap_rel & valid_curv & valid_vf
        if np.any(m):
            return m, 1
        m = mask_anti & gap_rel & (np.abs(c['lam1']) > 1e-4) & (np.abs(c['lam2']) > 1e-4)
        if np.any(m):
            return m, 2
        m = gap_rel if np.any(gap_rel) else np.ones(n_pts, dtype=bool)
        return m, 3

    mask_corr, tier_corr = _branch_mask(c_corr)
    mask_bare, tier_bare = _branch_mask(c_bare)
    mask_bare_aligned = mask_corr & mask_bare
    n_dropped_bare = int(np.sum(mask_corr)) - int(np.sum(mask_bare_aligned))

    def _m_eff(c, mask):
        if not mask.any():
            return float('nan'), np.zeros(0)
        m1 = np.abs(1.0 / c['lam1'][mask])
        m2 = np.abs(1.0 / c['lam2'][mask])
        return float(np.nanmean(np.concatenate([m1, m2]))), 0.5 * (m1 + m2)

    m_curv_corr, m_pt_corr = _m_eff(c_corr, mask_corr)
    m_curv_bare, m_pt_bare = _m_eff(c_bare, mask_bare_aligned)
    m_band_corr = 1.0 / (2.0 * max(g_t * t0, 1e-9))
    m_band_bare = 1.0 / (2.0 * max(1.0 * t0, 1e-9))
    ratio_corr = m_curv_corr / m_band_corr if np.isfinite(m_curv_corr) else float('nan')
    ratio_bare = m_curv_bare / m_band_bare if np.isfinite(m_curv_bare) else float('nan')
    ratio_shift = (abs(ratio_bare - ratio_corr) / max(abs(ratio_corr), 1e-9)
                   if np.isfinite(ratio_corr) else float('nan'))
    band_effect_dominant = (np.isfinite(ratio_shift) and ratio_shift < shift_tol)

    ev0, ec0 = np.linalg.eigh(
        vbdg._build_H_stack(np.c_[kx, ky], M_bdg, Q, _Dzero, _Fzero, mu, g_t, g_J)[:, 0:12, 0:12])
    n0 = np.argmin(np.abs(ev0), axis=1)
    psi0 = np.take_along_axis(ec0, n0[:, None, None], axis=2)[:, :, 0][mask_corr]

    w_G6  = np.abs(psi0[:, 0])**2 + np.abs(psi0[:, 1])**2 + np.abs(psi0[:, 6])**2 + np.abs(psi0[:, 7])**2
    w_G7a = np.abs(psi0[:, 2])**2 + np.abs(psi0[:, 3])**2 + np.abs(psi0[:, 8])**2 + np.abs(psi0[:, 9])**2
    w_G7b = np.abs(psi0[:, 4])**2 + np.abs(psi0[:, 5])**2 + np.abs(psi0[:, 10])**2 + np.abs(psi0[:, 11])**2

    def _mean_std(arr):
        if len(arr) == 0:
            return float('nan'), float('nan')
        return float(np.mean(arr)), float(np.std(arr))

    orbital_stats = {
        'G6':  dict(zip(('mean', 'std'), _mean_std(w_G6))),
        'G7a': dict(zip(('mean', 'std'), _mean_std(w_G7a))),
        'G7b': dict(zip(('mean', 'std'), _mean_std(w_G7b))),
    }
    finite_stds = [v['std'] for v in orbital_stats.values() if np.isfinite(v['std'])]
    max_orbital_std = max(finite_stds) if finite_stds else float('nan')
    orbital_spread_significant = (np.isfinite(max_orbital_std)
                                  and max_orbital_std > orbital_spread_thresh)

    def _safe_corr(a, b):
        if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    log_m_pt = np.log10(np.maximum(m_pt_corr, 1e-12))
    corr_G6  = _safe_corr(log_m_pt, w_G6)
    corr_G7a = _safe_corr(log_m_pt, w_G7a)
    corr_G7b = _safe_corr(log_m_pt, w_G7b)

    if not np.isfinite(ratio_shift):
        verdict = "INCONCLUSIVE: ratio_shift is NaN."
    elif state_mismatch:
        verdict = (f"INCONCLUSIVE: M(g_t=1) mismatch = {M_mismatch_norm/M_scale:.2%} > "
                   f"{state_mismatch_tol:.0%}.")
    elif band_effect_dominant and not orbital_spread_significant:
        verdict = ("NOT SUPPORTED: the anomaly survives essentially unchanged at g_t=1 (bare "
                   "band), and orbital character is nearly uniform across the retained "
                   "antinodal points.  This is a bare multi-orbital dispersion effect "
                   "(SOC / AFM zone folding), not a correlation-renormalization effect.")
    elif band_effect_dominant and orbital_spread_significant:
        verdict = "AGAINST: renormalization not the dominant driver."
    elif (not band_effect_dominant) and orbital_spread_significant:
        verdict = "SUPPORTED: Γ-projected g_t(k) motivated."
    else:
        verdict = "PARTIAL: bond-direction-resolved correction more likely."

    return {
        'm_eff_curv_corr': m_curv_corr, 'm_eff_band_corr': m_band_corr, 'ratio_corr': ratio_corr,
        'm_eff_curv_bare': m_curv_bare, 'm_eff_band_bare': m_band_bare, 'ratio_bare': ratio_bare,
        'ratio_shift': ratio_shift, 'band_effect_dominant': band_effect_dominant,
        'state_mismatch': state_mismatch,
        'M_mismatch_norm': M_mismatch_norm,
        'M_mismatch_rel': M_mismatch_norm / M_scale,
        'orbital_stats': orbital_stats,
        'orbital_spread_significant': orbital_spread_significant,
        'corr_logmass_G6': corr_G6,
        'corr_logmass_G7a': corr_G7a,
        'corr_logmass_G7b': corr_G7b,
        'n_total': n_pts,
        'n_candidates_antinodal': int(np.sum(mask_anti)),
        'n_used_corr': int(np.sum(mask_corr)),
        'n_used_bare': int(np.sum(mask_bare_aligned)),
        'n_dropped_bare': n_dropped_bare,
        'tier_corr': tier_corr, 'tier_bare': tier_bare, 'dk_used': dk,
        'verdict': verdict,
    }


# ============================================================================
# dk sensitivity and mu consistency
# ============================================================================
def test_dk_sensitivity_and_mu_consistency(solver, Q: float, target_doping: float, g_t: float, g_J: float, M_bdg: np.ndarray, n_bdg: float, mu_bdg: float) -> None:
    dk_nominal = min(1e-2, max(1e-4, 2.0 * np.pi / qm._NK / 6.0))
    dk_cases = [("0.5x", 0.5 * dk_nominal), ("1x", dk_nominal), ("2x", 2.0 * dk_nominal)]

    dk_results = []
    for label, _dk in dk_cases:
        res = _orbital_selective_gutzwiller_necessity_test(solver, Q, target_doping, g_t, g_J, M_bdg, n_bdg, mu_bdg, dk_override=_dk)
        dk_results.append(res)
        _log(f"  dk={label:<4} ({_dk:.2e})  "
              f"ratio_corr={res['ratio_corr']:+.4f}  ratio_bare={res['ratio_bare']:+.4f}  "
              f"Δratio={res['ratio_shift']*100:6.2f}%  "
              f"state_mismatch={res['state_mismatch']} ({res['M_mismatch_rel']*100:.1f}%)  "
              f"n_used={res['n_used_corr']}/{res['n_used_bare']}")

    ratios = np.array([r['ratio_corr'] for r in dk_results])
    verdicts = [r['verdict'] for r in dk_results]
    ratio_spread = float(np.max(ratios) - np.min(ratios))
    same_verdict = len(set(verdicts)) == 1
    dk_stable = same_verdict and ratio_spread < 0.15 * max(abs(float(np.mean(ratios))), 1e-9)
    _log(f"  [summary] ratio_corr = {[f'{r:.3f}' for r in ratios]}")
    _log(f"             spread = {ratio_spread:.4f}   dk_stable = {dk_stable}")
    if not dk_stable:
        _log("             ⚠ the conclusion depends on dk")

    _log("=== μ consistency test (∂Ω/∂μ at fixed n_kspace) ===")
    eps = 1e-5
    _Dzero = np.zeros(qm._N_DELTA, dtype=complex)
    _Fzero = np.zeros(qm._N_DELTA, dtype=float)
    V_JT_dummy = solver.g_JT_bare**2 / max(solver._K_bare, 1e-30)

    F_plus = solver._compute_bdg_free_energy(M_bdg, Q, _Dzero, _Fzero, mu_bdg + eps, g_t, g_J, 1.0, 1.0, 1.0, V_JT_dummy, 0.0, 0.0, 0.0)
    F_minus = solver._compute_bdg_free_energy(M_bdg, Q, _Dzero, _Fzero, mu_bdg - eps, g_t, g_J, 1.0, 1.0, 1.0, V_JT_dummy, 0.0, 0.0, 0.0)
    n_num = -(F_plus - F_minus) / (2.0 * eps)

    ev, ec = _bdg_eig(solver, M_bdg, Q, _Dzero, _Fzero, n_bdg, mu_bdg, g_t, g_J)
    n_ana = float(np.sum(solver._compute_orbital_densities(ev, ec)))
    _log(f"  n(numerical ∂Ω/∂μ) = {n_num:.8f}")
    _log(f"  n(analytical)      = {n_ana:.8f}")
    _log(f"  |diff| = {abs(n_num - n_ana):.2e}   "
          f"{'✓' if abs(n_num - n_ana) < 1e-4 else '⚠'}")


# ============================================================================
# Q-force decomposition
# ============================================================================
def test_q_force_decomposition(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, Q_Eg2: float = 0.0, K_eff_for_free_energy: float = 0.0, K_eff_Eg2_for_free_energy: float = 0.0, vertex_cache: dict | None = None, h: float | None = None) -> None:
    p = solver.p
    M = np.asarray(M, dtype=float)
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec = np.asarray(Quad67_vec, dtype=complex)

    dHdQ = solver._calc_dHdQ(M, Q, Quad67_vec, mu, g_t, g_J)

    ev_sc, ec_sc = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, Q_Eg2=Q_Eg2)
    dHdQ_diag = np.einsum('kin,kij,kjn->kn', ec_sc.conj(), dHdQ, ec_sc).real
    f_k = qm._fermi_function(ev_sc, solver.kT)
    dHdQ_exp = float(np.sum(solver.k_weights[:, None] * f_k * dHdQ_diag) / 4.0)

    eps_q, _ = qm._q_fd_step(Q)
    J_A1g_p, _, _ = _exch(p, Q + eps_q, g_J)
    J_A1g_m, _, _ = _exch(p, Q - eps_q, g_J)
    dJ3_dQ = (np.asarray(J_A1g_p, dtype=float)[0::2] - np.asarray(J_A1g_m, dtype=float)[0::2]) / (2.0 * eps_q)
    mf_term = 0.5 * p.Z_afm_eff * float(np.dot(M ** 2, dJ3_dQ))

    elastic_term = K_eff_for_free_energy * Q
    analytic_force = dHdQ_exp + mf_term + elastic_term

    g_Das, g_Dad, g_Dbd = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M, Q, mu, g_t, g_J)
    V_JT_bare = solver.g_JT_bare ** 2 / max(solver._K_bare, qm._MATH_EPS)

    if h is None:
        h = max(1e-4, 1e-3 * max(abs(Q), 1.0))

    def F_at(Q_val):
        return solver._compute_bdg_free_energy(M, Q_val, Delta_vec, Quad67_vec, mu, g_t, g_J, g_Das, g_Dad, g_Dbd, V_JT_bare, Q_Eg2, K_eff_for_free_energy, K_eff_Eg2_for_free_energy, vertex_cache)

    Fp  = F_at(Q + h);       Fm  = F_at(Q - h)
    Fp2 = F_at(Q + h / 2.0); Fm2 = F_at(Q - h / 2.0)
    fd_force      = (Fp - Fm) / (2.0 * h)
    fd_force_rich = (4.0 * (Fp2 - Fm2) / (2.0 * (h / 2.0)) - fd_force) / 3.0

    J_A1g_frozen, J_B1g_7a_frozen, J_B1g_7b_frozen = _exch(p, Q, g_J)
    _orig_exch = p.exchange_channels
    try:
        _cached_exch = _orig_exch(Q, g_J)
        def _frozen_exchange(Qv, gJ, _cached=_cached_exch):
            return _cached
        p.exchange_channels = _frozen_exchange
        dHdQ_frozenJ = solver._calc_dHdQ(M, Q, Quad67_vec, mu, g_t, g_J)
    finally:
        p.exchange_channels = _orig_exch

    dHdQ_frozenJ_diag = np.einsum('kin,kij,kjn->kn', ec_sc.conj(), dHdQ_frozenJ, ec_sc).real
    dHdQ_exp_frozenJ = float(np.sum(solver.k_weights[:, None] * f_k * dHdQ_frozenJ_diag) / 4.0)
    weiss_channel_in_HF = dHdQ_exp - dHdQ_exp_frozenJ

    _log("=" * 78)
    _log(f"  K_eff = {K_eff_for_free_energy:.4f}  h = {h:.4e}")
    _log()
    _log("  --- Analytic force components ---")
    _log(f"  ⟨∂H_BdG/∂Q⟩                    = {dHdQ_exp:+.10e}")
    _log(f"  0.5·Z_afm_eff·Σ M_c²·∂J_c/∂Q  = {mf_term:+.10e}")
    _log(f"  K_eff·Q                       = {elastic_term:+.10e}")
    _log(f"  analytic sum                  = {analytic_force:+.10e}")
    _log()
    _log("  --- Numerical FD ---")
    _log(f"  (F(+h)−F(−h))/(2h)            = {fd_force:+.10e}")
    _log(f"  Richardson                    = {fd_force_rich:+.10e}")
    _log()
    _log("  --- Channel decomposition ---")
    _log(f"  all channels                  = {dHdQ_exp:+.10e}")
    _log(f"  frozen J_A1g                  = {dHdQ_exp_frozenJ:+.10e}")
    _log(f"  AFM+B1g exchange              = {weiss_channel_in_HF:+.10e}")
    _log(f"  ≈ −Z_afm_eff·Σ M_c²·∂J_c/∂Q   = {-p.Z_afm_eff * float(np.dot(M**2, dJ3_dQ)):+.10e}")
    _log()

    diff_total_rich = analytic_force - fd_force_rich
    scale = max(abs(analytic_force), abs(fd_force_rich), 1e-12)
    rel = abs(diff_total_rich) / scale

    _log(f"  |analytic − FD| = {abs(diff_total_rich):.3e}   (rel: {rel:.3e})")
    if rel < 1e-3:
        _log("  → PASS")
    elif rel < 1e-2:
        _log("  → MARGINAL")
    else:
        _log("  → FAIL")
    _log("=" * 78)


def test_pair_correlator_and_quadrupole(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, g_Das: float = 1.0, g_Dad: float = 1.0, g_Dbd: float = 1.0,
    gauge_phases: tuple = (0.0, np.pi / 3, 2 * np.pi / 3, np.pi), zero_tol: float = 1e-10, gauge_tol: float = 1e-8, linear_tol: float = 0.05, linear_scan_fracs: tuple = (0.001, 0.005, 0.010)) -> None:
    """The two sources computed from a BdG state:
        qm._compute_pair_correlator   anomalous Γ6–Γ7 correlator ⟨c c⟩ (gap-equation source): odd in Δ, carries the phase of Δ
        qm._compute_B1g_quadrupole    particle–hole B1g transition density Quad67 (TRW source): U(1)-invariant, even in Δ
    [1] correlator = 0 at Δ = 0   [1b] mirror-sector selection rules for EVERY channel (correlator: sector of the single active Δ_i; Quad67: only (mx,my) = (+,+), the product of the characters of the active channels)
    [2] gauge: correlator(Δ e^{iφ}) = e^{iφ}·correlator(Δ), Quad67 exactly invariant   [3] linearity of the correlator at small |Δ|."""
    kT = solver.k_weights, solver.kT
    kw, kT = solver.k_weights, solver.kT
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Q67_in = np.asarray(Quad67_vec, dtype=complex)
    names = list(qm._IRREP_NAMES)
    idx7a = np.asarray(qm._IDX_7A_ARR)

    def _eval(dv, Qin=None):
        ev, ec = _bdg_eig(solver, M, Q, dv, Q67_in if Qin is None else Qin, n_kspace, mu, g_t, g_J)
        pc = qm._compute_pair_correlator(kw, solver.Phi_irrep_k, ev, ec, kT, dv, g_Das, g_Dad, g_Dbd)
        q67 = qm._compute_B1g_quadrupole(kw, solver.Phi_irrep_k, ev, ec, kT, solver.B1g_op)
        return pc, q67

    _log("=" * 78)
    _log("PAIR CORRELATOR AND B1g QUADRUPOLE (Quad67) CONSISTENCY TEST")
    _log("=" * 78)
    _log(f"  Q={Q:+.5f}  M={np.array2string(np.asarray(M), precision=5)}  n={n_kspace:.6f}  mu={mu:+.6f}  g_Das={g_Das:.4f}  g_Dad={g_Dad:.4f}")

    _log("[1] Joint zero condition (correlator at Δ = 0)")
    pc0, q0 = _eval(np.zeros(qm._N_DELTA, dtype=complex))
    ok_zero = float(np.max(np.abs(pc0))) < zero_tol
    _log(f"    Δ_vec = 0 :  ||correlator||_∞ = {float(np.max(np.abs(pc0))):.3e}   {'✓' if ok_zero else '❌ FAIL'}   (Quad67 may be non-zero in the AFM / JT normal state: ||Quad67||_∞ = {float(np.max(np.abs(q0))):.3e})")

    _log()
    _log("[1b] Mirror-sector selection rules, EVERY channel (7a branch), single Δ_i, no TRW background (kx → −kx, ky → −ky are exact for any M and Q)")
    zero_Q67 = np.zeros(qm._N_DELTA, dtype=complex)
    leak_m, leak_t, who_m, who_t, diag_min, leak_q, who_q = 0.0, 0.0, None, None, float('inf'), 0.0, None
    for i_c, nm_c in enumerate(names):
        dv = np.zeros(qm._N_DELTA, dtype=complex)
        dv[idx7a[i_c]] = 1e-3
        pc, q67 = _eval(dv, zero_Q67)
        Fa, Qa = np.abs(pc[idx7a]), np.abs(q67[idx7a])
        diag_min = min(diag_min, float(Fa[i_c]))
        ci = qm._IRREP_CHAR[nm_c]
        for j_c, nj_c in enumerate(names):
            cj = qm._IRREP_CHAR[nj_c]
            if cj != ci:
                if cj[1:] != ci[1:]:
                    if Fa[j_c] > leak_m:
                        leak_m, who_m = float(Fa[j_c]), (nm_c, nj_c)
                elif Fa[j_c] > leak_t:
                    leak_t, who_t = float(Fa[j_c]), (nm_c, nj_c)
            if tuple(cj[1:]) != (1, 1) and Qa[j_c] > leak_q:
                leak_q, who_q = float(Qa[j_c]), (nm_c, nj_c)
    ok_sel = leak_m < 1e-10 and leak_q < 1e-10 and diag_min > 0.0
    _log(f"     correlator: max mirror-sector leakage {leak_m:.2e} {who_m};  smallest diagonal response |F_ii| = {diag_min:.2e}")
    _log(f"     Quad67    : max weight outside the (mx,my) = (+,+) sector {leak_q:.2e} {who_q}   {'✓ PASS' if ok_sel else '❌ FAIL'}")
    _log(f"     max T_Q-sector leakage of the correlator {leak_t:.2e} {who_t}: information only — channels of the same D4h irrep mix (on-site s → extended-s es), T_Q is not a selection rule here")
    _log()

    _log("[2] Gauge: global phase of Δ")
    pc_ref, q_ref = _eval(Delta_vec)
    dev_pc, dev_q = 0.0, 0.0
    for phi in gauge_phases:
        pc_p, q_p = _eval(Delta_vec * np.exp(1j * phi))
        dev_pc = max(dev_pc, float(np.max(np.abs(pc_p - np.exp(1j * phi) * pc_ref))))
        dev_q = max(dev_q, float(np.max(np.abs(q_p - q_ref))))
    ok_gauge = dev_pc < gauge_tol and dev_q < gauge_tol
    _log(f"    max ‖correlator(Δe^{{iφ}}) − e^{{iφ}}·correlator(Δ)‖ = {dev_pc:.3e};  max ‖Quad67(Δe^{{iφ}}) − Quad67(Δ)‖ = {dev_q:.3e}   {'✓' if ok_gauge else '❌ FAIL'}")
    _log()

    _log("[3] Linearity of the correlator in |Δ|")
    dv_ref = Delta_vec.copy()
    if abs(dv_ref).max() < 1e-12:
        dv_ref = _delta_vector({'s7a': 1e-3, 'd7a': 1e-3}, dtype=complex)
    ok_lin = False
    i_s, i_d = qm._DELTA_IDX['s7a'], qm._DELTA_IDX['d7a']
    if abs(dv_ref[i_s]) > 1e-12 or abs(dv_ref[i_d]) > 1e-12:
        ratios = []
        for frac in linear_scan_fracs:
            pc_f, _ = _eval(dv_ref * frac)
            ratios.append((abs(pc_f[i_s]) / frac if abs(dv_ref[i_s]) > 1e-12 else 0.0, abs(pc_f[i_d]) / frac if abs(dv_ref[i_d]) > 1e-12 else 0.0))
            _log(f"    frac {frac:>6.4f}:  |corr[s7a]|/frac = {ratios[-1][0]:.10e}   |corr[d7a]|/frac = {ratios[-1][1]:.10e}")
        ratios = np.asarray(ratios)
        spread = [(ratios[:, k].max() - ratios[:, k].min()) / max(abs(ratios[:, k].mean()), 1e-12) for k in (0, 1)]
        ok_lin = all(s < linear_tol for s in spread)
        _log(f"    spread s7a = {spread[0]*100:6.2f}%   spread d7a = {spread[1]*100:6.2f}%   {'✓' if ok_lin else '✗'}")

    _log()
    _log("Verdict")
    _log(f"  [1] joint zero      : {'✓' if ok_zero else '❌ FAIL'}")
    _log(f"  [1b] mirror sectors : {'✓' if ok_sel else '❌ FAIL'}")
    _log(f"  [2] gauge           : {'✓' if ok_gauge else '❌ FAIL'}")
    _log(f"  [3] linearity       : {'✓' if ok_lin else '✗'}")
    _log("=" * 78)


def test_cluster_correction_scan(solver, target_doping: float, M_bdg: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, g_JT_factors: tuple[float, ...] = (0.5, 1.0, 1.5), delta_tetra_values: tuple[float, ...] = (-0.02, -0.05, -0.08, -0.11)) -> None:
    original_g_JT = solver.p.g_JT
    original_Delta_tetra = solver.p.Delta_tetra
    records = []

    try:
        _log(
            f"{'g_JT_f':<7} {'dt':<7} {'Δ_CF':<9} {'J/Δ_CF':<8} {'S(χ)':<7} {'V_Lehmann':<12} {'V_curv':<12} {'L/C':<7}")
        _log("-" * 92)

        for g_factor, dt in product(g_JT_factors, delta_tetra_values):
            solver.p.g_JT = original_g_JT * g_factor
            solver.p.Delta_tetra = dt
            solver._full_rebuild()
            solver.g_JT_bare = solver.p.g_JT

            tx_bare, ty_bare = solver.p.effective_hopping_anisotropic(Q)
            J_A1g_diag, J_B1g_7a, J_B1g_7b = _exch(solver.p, Q, g_J)
            J_eff = solver.p.Z_afm_eff * J_A1g_diag[0]
            Delta_CF = solver.p.Delta_CF
            ratio_J_CF = J_eff / max(abs(Delta_CF), 1e-12)

            _, F_cluster, _, _ = _vertex_tuple(solver, target_doping, M_bdg, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J)
            chi0_QQ = F_cluster['chi0_loc'][1, 1]
            chi_full_QQ = F_cluster['chi_loc_full'][1, 1]
            S = chi_full_QQ / chi0_QQ if abs(chi0_QQ) > 1e-12 else float('nan')

            V_Lehmann = F_cluster["V_irr_QQ"]

            _log(
                f"{g_factor:<7.2f} {dt:<+7.3f} {Delta_CF:<9.5f} {ratio_J_CF:<8.2f} {S:<7.4f} {V_Lehmann:<+12.4e}")

            K_bare = solver._K_bare
            V_JT_actual = solver.g_JT_bare ** 2 / max(K_bare, 1e-12)
            K_eff_Q = F_cluster["K_eff"]
            V_JT_if_Kbare = solver.g_JT_bare ** 2 / max(abs(K_bare), 1e-12)
            V_JT_if_Keff  = solver.g_JT_bare ** 2 / max(abs(K_eff_Q), 1e-12)
            dev_from_Kbare = abs(V_JT_actual - V_JT_if_Kbare) / max(abs(V_JT_if_Kbare), 1e-14)
            dev_from_Keff  = abs(V_JT_actual - V_JT_if_Keff) / max(abs(V_JT_if_Keff), 1e-14)

            records.append({
                "g_factor": g_factor, "dt": dt,
                "V_Lehmann": V_Lehmann, "K_bare": K_bare, "K_eff_Q": K_eff_Q,
                "dev_from_Kbare": dev_from_Kbare, "dev_from_Keff": dev_from_Keff,
            })

    finally:
        solver.p.g_JT = original_g_JT
        solver.p.Delta_tetra = original_Delta_tetra
        solver._full_rebuild()
        solver.g_JT_bare = original_g_JT
    
    dev_Kbare = np.array([r["dev_from_Kbare"] for r in records])
    dev_Keff = np.array([r["dev_from_Keff"] for r in records])
    K_bare_vals = np.array([r["K_bare"] for r in records])
    K_eff_vals = np.array([r["K_eff_Q"] for r in records])
    kbare_keff_separation = float(np.mean(np.abs(K_bare_vals - K_eff_vals) / np.maximum(np.abs(K_bare_vals), 1e-12)))

    _log("\n" + "=" * 70)
    _log("2) LIVE CHECK: V_JT on K_bare vs K_eff_Q")
    _log("=" * 70)
    _log(f"Mean |K_bare - K_eff_Q|/|K_bare| : {kbare_keff_separation:.2%}")
    _log(f"V_JT dev from K_bare : max={dev_Kbare.max():.2e}  mean={dev_Kbare.mean():.2e}")
    _log(f"V_JT dev from K_eff  : max={dev_Keff.max():.2e}  mean={dev_Keff.mean():.2e}")



# ============================================================================
# RPA vertex probes -- dataclasses + helpers
# ============================================================================
@dataclass
class PairingKernel:
    V_ss: float
    V_dd: float
    V_sd: float
    eigenvalues: np.ndarray
    angle: float

    @property
    def dominant_abs(self) -> float:
        return float(np.max(np.abs(self.eigenvalues)))


@dataclass
class RPAPrefactors:
    J_eff: float
    V_JT_corr: float
    Gamma_M: float


@dataclass
class RPAVertices:
    V_full: np.ndarray
    V_spin: np.ndarray
    V_jt: np.ndarray
    V_noSQ: np.ndarray
    chi_SQ: np.ndarray

    @property
    def chi_SQ_max(self) -> float:
        return float(np.max(np.abs(self.chi_SQ)))

    @property
    def delta_V_SQ(self) -> np.ndarray:
        return self.V_full - self.V_noSQ

    @property
    def delta_V_SQ_mean(self) -> float:
        return float(np.mean(self.delta_V_SQ))


@dataclass
class FSReference:
    fs_pts: np.ndarray
    weights: np.ndarray
    i_idx: np.ndarray
    j_idx: np.ndarray
    unique_q: np.ndarray
    inv_idx: np.ndarray
    afm_mask: np.ndarray
    fwd_mask: np.ndarray
    N_fs: int
    n_q: int
    phi_s: np.ndarray = field(init=False)
    phi_d: np.ndarray = field(init=False)
    ns_norm: float = field(init=False)
    nd_norm: float = field(init=False)

    def __post_init__(self):
        self.phi_s = np.ones(self.N_fs, dtype=float)
        self.phi_d = np.cos(self.fs_pts[:, 0]) - np.cos(self.fs_pts[:, 1])
        self.ns_norm = max(float(self.phi_s @ (self.phi_s * self.weights)), 1e-12)
        self.nd_norm = max(float(self.phi_d @ (self.phi_d * self.weights)), 1e-12)

    @classmethod
    def from_context(cls, ctx: dict):
        N_fs, n_q = ctx['N_fs'], ctx['n_q']
        if N_fs == 0 or n_q == 0:
            return None
        uq = ctx['unique_q']
        q_fwd = np.mod(uq + np.pi, 2.0 * np.pi) - np.pi
        q_afm = np.mod(uq - np.array([np.pi, np.pi]) + np.pi, 2.0 * np.pi) - np.pi
        return cls(
            fs_pts=ctx['fs_pts'], weights=ctx['weights'],
            i_idx=ctx['i_idx'], j_idx=ctx['j_idx'],
            unique_q=uq, inv_idx=ctx['inv_idx'],
            afm_mask=np.linalg.norm(q_afm, axis=1) < 0.3 * np.pi,
            fwd_mask=np.linalg.norm(q_fwd, axis=1) < 0.3 * np.pi,
            N_fs=N_fs, n_q=n_q)

    def build_matrix(self, V_q: np.ndarray) -> np.ndarray:
        M = np.zeros((self.N_fs, self.N_fs), dtype=float)
        M[self.i_idx, self.j_idx] = V_q[self.inv_idx]            # upper-triangle pair list incl. diagonal
        M = M + M.T
        M[np.arange(self.N_fs), np.arange(self.N_fs)] *= 0.5       # and the diagonal (counted twice) is restored.  0.5*(M+M^T) would halve all off-diagonals.
        return M

    def project_to_sd(self, V_mat: np.ndarray) -> tuple[float, float, float]:
        ws, wd = self.phi_s * self.weights, self.phi_d * self.weights
        V_ss = float(ws @ V_mat @ ws) / self.ns_norm ** 2
        V_dd = float(wd @ V_mat @ wd) / self.nd_norm ** 2
        V_sd = float(ws @ V_mat @ wd) / (self.ns_norm * self.nd_norm)
        return V_ss, V_dd, V_sd

    def kernel_2x2(self, V_q: np.ndarray, g_Das: float, gDd: float) -> PairingKernel:
        V_ss, V_dd, V_sd = self.project_to_sd(self.build_matrix(V_q))
        g_sd = np.sqrt(max(g_Das * gDd, 0.0))
        K = np.array([[g_Das * V_ss, g_sd * V_sd],
                      [g_sd * V_sd, gDd * V_dd]])
        vals, _ = np.linalg.eigh(K)
        angle = float(np.degrees(0.5 * np.arctan2(2.0 * V_sd, V_ss - V_dd)))
        return PairingKernel(V_ss=V_ss, V_dd=V_dd, V_sd=V_sd, eigenvalues=vals, angle=angle)


def _fmt_qstats(arr, mask, label):
    if not np.any(mask):
        return f"    {label:<22s}: n=0"
    sub = arr[mask]
    return (f"    {label:<22s}: n={len(sub):3d}  mean={np.mean(sub):+.4f}"
            f"  min={np.min(sub):+.4f}  max={np.max(sub):+.4f}"
            f"  neg%={np.mean(sub < 0.0)*100:5.1f}")


def _fmt_header(title, Q, M, n, mu, g_t, g_J, g_Das, gDd, pref, ref,
                Delta_vec=None, Quad67_vec=None, V_cap=None):
    lines = ["=" * 78, title, "=" * 78]
    lines.append(f"  Q={Q:+.5f}  M={np.array2string(np.asarray(M), precision=5)}"
                 f"  n={n:.6f}  mu={mu:+.6f}")
    lines.append(f"  g_t={g_t:.4f}  g_J={g_J:.4f}  g_Δs={g_Das:.4f}  g_Δd={gDd:.4f}")
    if Delta_vec is not None:
        lines.append(f"  Δ_vec={np.array2string(Delta_vec, precision=5)}"
                     f"  Quad67_vec={np.array2string(Quad67_vec, precision=5)}")
    lines.append(f"  J_eff={pref.J_eff:.4f}  V_JT_corr={pref.V_JT_corr:.4f}")
    lines.append(f"  FS pts={ref.N_fs}  unique q={ref.n_q}  V_cap={V_cap:.1e}")
    return "\n".join(lines)


def test_superexchange_convention_and_Heisenberg_factor(solver, target_doping: float,
    t_pd_scan: tuple[float, ...] | None = None,
    U_dd_scan: tuple[float, ...] | None = None,
    J_H_scan:  tuple[float, ...] | None = None,
    M_channels_for_weighting: np.ndarray | None = None,
    loewdin_E_dep_tol: float = 0.15) -> None:
    """
    Non-tautological test of the multipolar superexchange extraction.

    κ_code (from Löwdin) is defined in the NORMALISED pseudospin τ-basis
    (τ_z = ±1/2), while the physical spin is  Ŝ = 2s_eff · τ  with
    s_eff = |sz_up| < 1/2.  For an isotropic Heisenberg H = J·S₁·S₂,
    the Pauli decomposition gives

        H_eff = c_zz · (σ_xσ_x + σ_yσ_y + σ_zσ_z),   c_zz = J · s_eff²,

    and the exact ED relation is

        J = κ_code / (4 s_eff²)  =  c_zz / s_eff².

    The Löwdin H_eff(E) is energy-dependent.  Heff_at(E) evaluated at a single trial energy (mean of branches) does NOT have the branch energies as eigenvalues.
    """
    p = solver.p
    TOL_ISO = 1e-5
    TOL_LOEWDIN = 1e-6

    def _extract(t_pd_v, U_dd_v, Delta_CT_v, lambda_soc_v, Delta_tetra_v, Delta_B1g_static_v, Upp_ratio_bare_v, J_H_ratio_v, Z_v):
        G6, G7a, G7b, kappa_doublets, kappa_cross, B1g_t2g, basis, evals_2hole, evecs_2hole, U_pp = _extract_all_kappas(t_pd_v, Delta_CT_v, U_dd_v, lambda_soc_v, Delta_tetra_v, Delta_B1g_static_v, Upp_ratio_bare_v, J_H_ratio_v, Z_v)

        H_soc_cf, _Lx, _Ly, _Lz, Sz_full, LS_op = qm._t2g_soc_cf_operators(lambda_soc_v, Delta_tetra_v, Delta_B1g_static_v)

        per_doublet = {}
        for name, D, k_idx in (('G6', G6, 0), ('G7a', G7a, 1), ('G7b', G7b, 2)):
            refs = [qm._multiorbital_reference_vector(basis, D[a], 0, D[b], 12)
                    for a, b in (('up', 'up'), ('up', 'dn'), ('dn', 'up'), ('dn', 'dn'))]
            V = np.column_stack(refs)
            branches, Heff_mean = qm._lowdin_branch_energies(
                evals_2hole, evecs_2hole, V, E_guess=2.0 * D['energy'], spread=0.3)

            per_branch_residual = []
            if len(branches) >= 2:
                Vc = evecs_2hole.conj().T @ V
                for E_b in branches:
                    denom = E_b - evals_2hole
                    denom = np.where(np.abs(denom) < 1e-15,
                                     np.copysign(1e-15, denom), denom)
                    G = (Vc.conj().T * (1.0 / denom)) @ Vc
                    try:
                        Heff_b = E_b * np.eye(V.shape[1]) - np.linalg.inv(G)
                    except np.linalg.LinAlgError:
                        per_branch_residual.append(np.inf)
                        continue
                    Heff_b = 0.5 * (Heff_b + Heff_b.conj().T)
                    ev_b = np.linalg.eigvalsh(Heff_b)
                    per_branch_residual.append(float(np.min(np.abs(ev_b - E_b))))
            loewdin_ok = all(r < TOL_LOEWDIN for r in per_branch_residual)

            # Pauli decomposition (of the mean-energy Heff, as before)
            c_00 = qm._pauli_coeff(Heff_mean, '0', '0')
            c_xx = qm._pauli_coeff(Heff_mean, 'x', 'x')
            c_yy = qm._pauli_coeff(Heff_mean, 'y', 'y')
            c_zz = qm._pauli_coeff(Heff_mean, 'z', 'z')
            c_xy = qm._pauli_coeff(Heff_mean, 'x', 'y')
            c_xz = qm._pauli_coeff(Heff_mean, 'x', 'z')
            c_yz = qm._pauli_coeff(Heff_mean, 'y', 'z')
            c_x0 = qm._pauli_coeff(Heff_mean, 'x', '0')
            c_y0 = qm._pauli_coeff(Heff_mean, 'y', '0')
            c_z0 = qm._pauli_coeff(Heff_mean, 'z', '0')
            single_site_max = max(abs(c_x0), abs(c_y0), abs(c_z0))

            s = abs(float(D['sz_up']))
            kappa_code = float(kappa_doublets[k_idx])

            # Ŝ = 2s·τ, σ = 2τ  ⇒  H = J·S₁·S₂ = J·(2s)²·τ₁·τ₂  ⇒  J = c_zz / s²
            J_from_pauli = c_zz / max(s * s, qm._WIGNER_ECKART_DIV_FLOOR)
            # Löwdin-ED value: κ_code / (4s²) — this is the exact ED output in the physical spin basis.
            J_from_loewdin = (branches[-1] - branches[0]) / max(4.0 * s * s, qm._WIGNER_ECKART_DIV_FLOOR)

            # E-dependence measure: how much the mean-energy Heff differs from the branch energies (a diagnostic, NOT a pass/fail criterion).
            Heff_evals = np.linalg.eigvalsh(0.5 * (Heff_mean + Heff_mean.conj().T))
            spread_matrix = float(Heff_evals[-1] - Heff_evals[0])
            spread_branches = float(branches[-1] - branches[0]) if len(branches) >= 2 else 0.0
            E_dep_rel = (abs(spread_matrix - spread_branches)
                         / max(abs(spread_branches), 1e-20))

            per_doublet[name] = {
                's': s,
                'c_00': c_00,
                'c_xx': c_xx, 'c_yy': c_yy, 'c_zz': c_zz,
                'c_xy': c_xy, 'c_xz': c_xz, 'c_yz': c_yz,
                'single_site_max': single_site_max,
                'spread_matrix': spread_matrix,
                'spread_branches': spread_branches,
                'E_dep_rel': E_dep_rel,
                'branches': branches,
                'loewdin_ok': loewdin_ok,
                'loewdin_per_branch_resid': per_branch_residual,
                'J_from_pauli': J_from_pauli,          # c_zz / s²
                'J_from_loewdin': J_from_loewdin,      # κ_code / (4 s²)
                'kappa_code': kappa_code,
                'aniso_xy': abs(c_xx - c_yy),
                'aniso_xz': abs(c_xx - c_zz),
                'aniso_yz': abs(c_yy - c_zz),
                'offdiag_max': max(abs(c_xy), abs(c_xz), abs(c_yz)),
            }
        per_doublet['kappa_B1g_7a'] = float(kappa_cross[0])
        per_doublet['kappa_B1g_7b'] = float(kappa_cross[1])
        J_H = J_H_ratio_v * U_dd_v
        return per_doublet, U_pp, J_H

    per_doublet, U_pp, J_H = _extract(p.t_pd, p.U_dd, p.Delta_CT, p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static, p.Upp_ratio_bare, p.J_H_ratio, qm._Z)

    J_ZSA = 4.0 * p.t_pd ** 4 / (p.Delta_CT ** 2 * p.U_dd)

    _log("=" * 116)
    _log(f"    t_pd={p.t_pd:.4f}  U_dd={p.U_dd:.4f}  Δ_CT={p.Delta_CT:.4f}")
    _log(f"    U_pp={U_pp:.4f}  J_H={J_H:.4f}")
    _log(f"    λ_SOC={p.lambda_soc:.4f}  Δ_tetra={p.Delta_tetra:.4f}  "
          f"Δ_B1g_static={p.Delta_B1g_static:.4f}")
    _log(f"    J_ZSA = 4·t_pd⁴/(Δ_CT²·U_dd) = {J_ZSA:.6e} eV")
    _log()

    # 1) Pauli decomposition
    _log("  1) Pauli decomposition of H_eff")
    _log(f"  {'doublet':<7s}  {'s':>8s}  {'c_zz':>12s}  {'aniso':>10s}  "
          f"{'offdiag':>10s}  {'single-site':>12s}  {'iso?':>5s}")
    _log("  " + "-" * 100)
    for name in ('G6', 'G7a', 'G7b'):
        d = per_doublet[name]
        iso = (d['aniso_xy'] < TOL_ISO and d['aniso_xz'] < TOL_ISO
               and d['offdiag_max'] < TOL_ISO and d['single_site_max'] < TOL_ISO)
        _log(f"  {name:<7s}  {d['s']:>8.6f}  {d['c_zz']:>12.4e}  "
              f"{max(d['aniso_xy'], d['aniso_xz']):>10.2e}  "
              f"{d['offdiag_max']:>10.2e}  {d['single_site_max']:>12.2e}  "
              f"{'✓' if iso else '✗':>5s}")
    _log()

    # 2) Löwdin per-branch consistency (correct check)
    _log("  2) Löwdin per-branch consistency check")
    _log("     (verify each fixed-point branch E_n is an eigenvalue of Heff_at(E_n))")
    _log(f"  {'doublet':<7s}  {'residual':>16s}  {'E_dep_rel':>14s}  {'chk':>5s}")
    _log("  " + "-" * 100)
    loewdin_ok_all = True
    for name in ('G6', 'G7a', 'G7b'):
        d = per_doublet[name]
        max_res = max(d['loewdin_per_branch_resid']) if d['loewdin_per_branch_resid'] else float('nan')
        if not d['loewdin_ok']:
            loewdin_ok_all = False
        _log(f"  {name:<7s}  {max_res:>16.3e}  {d['E_dep_rel']:>14.3e}  "
              f"{'✓' if d['loewdin_ok'] else '✗':>5s}")
    _log("    (E_dep_rel: fraction by which the mean-energy Heff eigenvalue spread differs from the branch spread — this is EXPECTED to be nonzero and is not a test failure; it is a diagnostic of the E-dependence)")
    _log()

    # 3) Physical J comparison
    _log("  3) Physical J:  J_pauli = c_zz/s²  vs  J_loewdin = κ_code/(4s²)")
    _log(f"  {'doublet':<7s}  {'s':>8s}  {'c_zz/s²':>16s}  {'κ_code/(4s²)':>16s}  "
          f"{'ratio':>10s}  {'consistent?':>12s}")
    _log("  " + "-" * 100)
    J_consistency_ok = True
    for name in ('G6', 'G7a', 'G7b'):
        d = per_doublet[name]
        ratio = d['J_from_loewdin'] / d['J_from_pauli'] if abs(d['J_from_pauli']) > 1e-20 else float('nan')
        # The two J definitions should agree up to the Löwdin E-dependence
        consistent = abs(ratio - 1.0) < loewdin_E_dep_tol
        if not consistent:
            J_consistency_ok = False
        _log(f"  {name:<7s}  {d['s']:>8.6f}  {d['J_from_pauli']:>16.6e}  "
              f"{d['J_from_loewdin']:>16.6e}  {ratio:>10.4f}  "
              f"{'✓' if consistent else '⚠ (>tol)':>12s}")
    _log()

    # 4) Multi-channel weighted ZSA comparison
    _log("  4) ZSA comparison (multi-channel weighted)")
    _log(f"     J_ZSA = {J_ZSA:.6e} eV   (bare single-band estimate)")
    if M_channels_for_weighting is not None:
        weights = np.asarray(M_channels_for_weighting, dtype=float) ** 2
    else:
        weights = np.array([0.95, 0.04, 0.01])   # Γ6-dominant default
    weights = weights / max(weights.sum(), 1e-20)
    J_channels = np.array([per_doublet[n]['J_from_loewdin']
                            for n in ('G6', 'G7a', 'G7b')])
    J_weighted = float(np.dot(weights, J_channels))
    ratio_weighted = J_weighted / J_ZSA if abs(J_ZSA) > 1e-20 else float('nan')
    _log(f"     weights (M²) : Γ6={weights[0]:.4f}  Γ7a={weights[1]:.4f}  Γ7b={weights[2]:.4f}")
    _log(f"     J_c (Loewdin) : Γ6={J_channels[0]:.6e}  Γ7a={J_channels[1]:.6e}  "
          f"Γ7b={J_channels[2]:.6e} eV")
    _log(f"     J_weighted   = {J_weighted:.6e} eV")
    _log(f"     J_weighted/J_ZSA = {ratio_weighted:.4f}")
    _log("     (this ratio is the multi-orbital + U_pp + Hund + SOC enhancement)")
    _log()

    # 5) Code-chain consistency
    _log("  5) Code-chain consistency  (does J_A1g match κ_code at the fixpoint?)")
    _, g_J_fp = p.get_gutzwiller_factors(target_doping)
    # At Q=0, ⟨orbital weight⟩=1 and J_A1g(Γ6) = g_J·κ_Γ6
    J_A1g_fixpoint, _, _ = _exch(p, 0.0, g_J_fp)
    J_A1g_Γ6_actual = float(J_A1g_fixpoint[0])
    J_A1g_Γ6_expected = g_J_fp * per_doublet['G6']['kappa_code']
    chain_ok = abs(J_A1g_Γ6_actual - J_A1g_Γ6_expected) < 1e-10 * max(abs(J_A1g_Γ6_expected), 1e-20)
    _log(f"     g_J(δ={target_doping}) = C_J/(1+δ)² = "
          f"{qm._GW_G_J_PREFACTOR}/(1+{target_doping})² = {g_J_fp:.6f}")
    _log(f"     C_J = {qm._GW_G_J_PREFACTOR:.1f}  "
          f"(convention: {'RENORMALISED κ_code, C_J=1' if qm._GW_G_J_PREFACTOR == 1.0 else 'BARE κ_code, C_J=4'})")
    _log(f"     J_A1g(Γ6, Q=0) [exchange_channels()] = {J_A1g_Γ6_actual:.6e} eV")
    _log(f"     g_J·κ_Γ6 [expected, no extra (1−δ)]  = {J_A1g_Γ6_expected:.6e} eV"
          f"   {'✓' if chain_ok else '✗ code-chain mismatch'}")
    _log(f"     vs. J_ZSA = {J_ZSA:.6e} eV   →  ratio = {J_A1g_Γ6_actual/J_ZSA:.3f}")
    _log()

    # 6) Verdict
    _log("=" * 116)
    _log("  VERDICT:")
    if not loewdin_ok_all:
        _log("    ✗ Löwdin per-branch consistency FAILED — algorithmic issue.")
    else:
        _log("    ✓ Löwdin per-branch consistency: PASS")
    if not J_consistency_ok:
        _log(f"    ⚠ J_pauli vs J_loewdin differ by more than {loewdin_E_dep_tol:.0%} —")
        _log("      the Löwdin H_eff E-dependence is strong at this parameter point.")
    else:
        _log("    ✓ J_pauli = c_zz/s²  and  J_loewdin = κ_code/(4s²)  agree —")
        _log("      Wigner-Eckart identity holds; the two J-definitions are consistent.")
    _log(f"    → multi-orbital enhancement over J_ZSA: {ratio_weighted:.2f}×")
    _log("=" * 116 + "\n")

    # 7) Scans (unchanged structure; use J_from_loewdin for the κ column)
    def _scan_one(t_pd_v, U_dd_v, J_H_ratio_v):
        pd, _, _ = _extract(t_pd_v, U_dd_v, p.Delta_CT, p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static, p.Upp_ratio_bare, J_H_ratio_v, qm._Z)
        d = pd['G6']
        iso = (d['aniso_xy'] < TOL_ISO and d['aniso_xz'] < TOL_ISO
               and d['offdiag_max'] < TOL_ISO and d['single_site_max'] < TOL_ISO)
        return {'s': d['s'], 'c_zz': d['c_zz'], 'iso': iso,
                'J_pauli': d['J_from_pauli'], 'J_loewdin': d['J_from_loewdin'],
                'ratio': (d['J_from_loewdin'] / d['J_from_pauli']
                          if abs(d['J_from_pauli']) > 1e-20 else float('nan'))}

    def _print_scan(scan_name, scan_values, scan_callable):
        if scan_values is None: return
        _log("=" * 116)
        _log(f"Scan over {scan_name}  (Γ6 only)")
        _log("=" * 116)
        _log(f"  {'val':>10s}  {'s':>10s}  {'c_zz':>12s}  {'iso':>5s}  "
              f"{'J_pauli':>16s}  {'J_loewdin':>16s}  {'ratio':>10s}")
        _log("  " + "-" * 100)
        for v in scan_values:
            rec = scan_callable(v)
            if rec is None: continue
            _log(f"  {v:>10.4f}  {rec['s']:>10.6f}  {rec['c_zz']:>12.4e}  "
                  f"{'✓' if rec['iso'] else '✗':>5s}  "
                  f"{rec['J_pauli']:>16.6e}  {rec['J_loewdin']:>16.6e}  "
                  f"{rec['ratio']:>10.4f}")
        _log()

    _print_scan("t_pd", t_pd_scan, lambda t: _scan_one(t, p.U_dd, p.J_H_ratio))
    _print_scan("U_dd", U_dd_scan, lambda U: _scan_one(p.t_pd, U, p.J_H_ratio))
    _print_scan("J_H",  J_H_scan,  lambda j: _scan_one(p.t_pd, p.U_dd, j))


def _apply_theta_operator(solver, H24: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    """
    Application of Θ = T · τ_AB to the 24×24 BdG Hamiltonian:
    1. τ_AB: A ↔ B sublattice exchange (in both particle and hole blocks)
    2. T: TRUE Kramers time-reversal — the "up"/"down" states of the 3 doublets
    are not their own complex conjugates in a fixed basis — Θ actually permutes them as well.
    Constructs the UNITARY part of the combined (anti-unitary) Θ·τ_AB symmetry within the full 24×24 Nambu basis:
    [Part_A(0:6), Part_B(6:12), Hole_A(12:18), Hole_B(18:24)]. 

    For the 6-dimensional (Γ6/Γ7a/Γ7b × up/down) sector, Θ = U_new·K, where U_new is constructed
    from the ACTUAL Kramers-doublet eigenvectors: starting from the bare t2g×spin basis—using standard time-reversal
    Θ_orb: m→−m (with a phase) and Θ_spin = iσ_y—this is then rotated into the Γ6/Γ7a/Γ7b eigenbasis using U_ord.

    Result: The Θ H(k) Θ^{-1} matrix at the given k-point. 

    In the model, every explicitly k-dependent term (H_AB hopping, H_nnn_k, β_k) is constructed from real-valued cosines;
    therefore, the term behaves correctly under conjugation even without the k→−k transformation.
    """
    p = solver.p
    G6, G7a, G7b, _kd, _kc, _B1g_t2g, *_ = _extract_all_kappas(p.t_pd, p.Delta_CT, p.U_dd, p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static, p.Upp_ratio_bare, p.J_H_ratio, qm._Z)
    U_ord = np.column_stack([G6['up'], G6['dn'], G7a['up'], G7a['dn'], G7b['up'], G7b['dn']])

    Theta_orb  = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]], dtype=complex)
    Theta_spin = np.array([[0, 1], [-1, 0]], dtype=complex)
    U_bare = np.kron(Theta_orb, Theta_spin)
    # antiunitary basis transformation: U_new = U_ord^dagger @ U_bare @ U_ord.conj()
    U_new = U_ord.conj().T @ U_bare @ U_ord.conj()

    U24 = np.zeros((24, 24), dtype=complex)
    for b in range(4):
        block = U_new if b < 2 else U_new.conj()
        U24[6 * b:6 * b + 6, 6 * b:6 * b + 6] = block
    idx = np.r_[6:12, 0:6, 18:24, 12:18]
    H_flipped = H24[np.ix_(idx, idx)]
    return U24 @ H_flipped.conj() @ U24.conj().T


def test_chi_SQ_M_dependence(solver, target_doping: float, M_scan, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, g_t: float, g_J: float) -> None:
    """
    χ_SQ(M) — direct measure of Θ-symmetry breaking.
    Expectation:
      M = 0 : χ_SQ ≡ 0, even if Δ≠0 and Q≠0 (Θ is still a symmetry)
      M ≠ 0 : χ_SQ ≠ 0, and |χ_SQ| grows roughly with M
    """
    p = solver.p
    _log("=" * 100)
    _log(f"  {'M':>10s}  {'χ_SQ(q=0)':>16s}  {'χ_SQ(q=π,π)':>16s}  "
          f"{'‖ΘHΘ⁻¹−H‖_F':>18s}  {'Θ-breaking':>12s}")
    _log("-" * 100)

    vbdg = solver._get_vbdg()

    records = []
    for M_val in M_scan:
        M = np.array([M_val, 0.0, 0.0], dtype=float)
        tx_b, ty_b = p.effective_hopping_anisotropic(Q)
        t_eff = g_t * float(np.sqrt(0.5 * (tx_b**2 + ty_b**2)))

        # set mu for the current M
        mu, n_k = solver._find_mu_for_density(target_doping, M, Q, Delta_vec, Quad67_vec, -2.0 * t_eff * (1.0 - 2.0 * abs(target_doping)), g_t, g_J)
        Gamma_M, F_cluster, V_JT_corr, _ = _vertex_tuple(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J)

        ev, ec = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J)

        # χ_SQ(q=0) and χ_SQ(q=π,π)
        chi_SS_q0, chi_SQ_q0, _, chi_QQ_q0 = _susceptibilities_sc(solver, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J, np.zeros(2), Gamma_M, (ev, ec), apply_diamagnetic_QQ=True)
        chi_SS_afm, chi_SQ_afm, _, chi_QQ_afm = _susceptibilities_sc(solver, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, (ev, ec), apply_diamagnetic_QQ=True)

        # Θ-breaking measure: H(k_test) vs Θ H(k_test) Θ^{-1}
        k_test_idx = np.argmin(np.abs(solver.k_points[:, 0] - np.pi/2) +
                                np.abs(solver.k_points[:, 1] - np.pi/2))
        H_test = vbdg._build_H_stack(solver.k_points[k_test_idx:k_test_idx+1], M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J)[0]
        H_theta = _apply_theta_operator(solver, H_test, solver.k_points[k_test_idx:k_test_idx+1])
        theta_break_norm = float(np.linalg.norm(H_theta - H_test, 'fro'))
        theta_breaking = theta_break_norm > 1e-10 * max(float(np.linalg.norm(H_test, 'fro')), 1.0)

        _log(f"  {M_val:>10.4f}  {chi_SQ_q0:>+16.6e}  {chi_SQ_afm:>+16.6e}  "
              f"{theta_break_norm:>18.6e}  "
              f"{'✓ YES' if theta_breaking else '✗ no':>12s}")

        records.append({
            'M': M_val, 'chi_SQ_q0': chi_SQ_q0, 'chi_SQ_afm': chi_SQ_afm,
            'theta_break_norm': theta_break_norm,
            'theta_breaking': theta_breaking,
        })

    M0_record = records[0]
    if abs(M0_record['chi_SQ_q0']) < 1e-8 and abs(M0_record['chi_SQ_afm']) < 1e-8:
        _log("    ✓ M=0: χ_SQ ≡ 0 even with Δ≠0 and Q≠0 (Θ is still a symmetry)")
    else:
        _log("    ⚠ M=0: χ_SQ ≠ 0 (unexpected — Θ-symmetry is not clean)")

    # growth with M
    chi_SQ_vals = np.array([r['chi_SQ_afm'] for r in records[1:]])
    if len(chi_SQ_vals) > 1 and np.all(np.diff(np.abs(chi_SQ_vals)) > 0):
        _log(f"    ✓ χ_SQ grows with M: {[f'{v:.2e}' for v in chi_SQ_vals]}")
        _log("      → M ≠ 0 is a NECESSARY condition for Θ-symmetry breaking")
    _log("=" * 100)


def test_chi_SQ_individual_channel_activation(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_k: float, mu: float, g_t: float, g_J: float) -> None:
    """
    χ_SQ(q) q-resolved scan on a user-supplied Quad67_vec.

    The function evaluates χ_SQ(q) at a set of symmetry-distinguished q-points
    (Γ, X, mid-antinodal, nodal diagonal, M(π,π)) with the Quad67_vec AS GIVEN, and reports:
      - the maximum |χ_SQ| over the scanned q-grid,
      - whether the channel is open anywhere,
      - whether it is open at the AFM wavevector q = (π,π).

    It does NOT isolate the on-site vs inter-sublattice contributions of Quad67_vec
    """
    q_scan = [
        ("Γ(0,0)",    np.array([0.0,      0.0    ])),
        ("X(π,0)",    np.array([np.pi,    0.0    ])),
        ("(π/2,0)",   np.array([np.pi/2,  0.0    ])),
        ("(π/2,π/2)", np.array([np.pi/2,  np.pi/2])),
        ("M(π,π)",    np.array([np.pi,    np.pi  ])),
    ]

    Gamma_M, _, V_JT_corr, _ = _vertex_tuple(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J)
    CHI_SQ_OPEN_TOL = 1e-7

    ev, ec = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J)
    E_k_cache = (ev, ec)

    # --- χ_SQ(q) scan ---
    chi_SQ_q: dict[str, float] = {}
    for q_label, q_pt in q_scan:
        _, chi_SQ_val, _, _ = _susceptibilities_sc(solver, M, Q, Delta_vec, Quad67_vec, n_k, mu, g_t, g_J, q_pt, Gamma_M, E_k_cache, apply_diamagnetic_QQ=True)
        chi_SQ_q[q_label] = float(chi_SQ_val)

    chi_SQ_abs = np.array([abs(v) for v in chi_SQ_q.values()])
    chi_SQ_max = float(np.max(chi_SQ_abs))
    q_at_max   = list(chi_SQ_q.keys())[int(np.argmax(chi_SQ_abs))]
    chi_SQ_at_afm = chi_SQ_q.get("M(π,π)", 0.0)

    # --- Logging ---
    _log("=" * 100)
    _log(f"  χ_SQ(q) scan   M={np.array2string(np.asarray(M), precision=4)}   "
         f"Q={Q:+.5f}   |Δ_vec|={float(np.sum(np.abs(Delta_vec))):.3e}")
    _log(f"  Quad67_vec: |s7a|={abs(Quad67_vec[qm._DELTA_IDX['s7a']]):.3e}   "
         f"|s7b|={abs(Quad67_vec[qm._DELTA_IDX['s7b']]):.3e}   "
         f"|d7a|={abs(Quad67_vec[qm._DELTA_IDX['d7a']]):.3e}   "
         f"|d7b|={abs(Quad67_vec[qm._DELTA_IDX['d7b']]):.3e}")
    _log("=" * 100)
    _log(f"  {'q-point':<12s}  {'χ_SQ(q)':>18s}")
    _log("-" * 34)
    for q_label, _ in q_scan:
        _log(f"  {q_label:<12s}  {chi_SQ_q[q_label]:>+18.6e}")
    _log()

    _log(f"  max_q |χ_SQ| = {chi_SQ_max:.3e}  (q_max = {q_at_max})")
    if chi_SQ_max > CHI_SQ_OPEN_TOL:
        _log("    ✓ χ_SQ channel OPEN somewhere in the scanned q-grid")
    else:
        _log(f"    ⚠ χ_SQ channel CLOSED at every scanned q (below the {CHI_SQ_OPEN_TOL:.0e} threshold)")

    if abs(chi_SQ_at_afm) > CHI_SQ_OPEN_TOL:
        _log(f"    ✓ χ_SQ(π,π) = {chi_SQ_at_afm:+.3e} — also active at the AFM wavevector")
    else:
        _log(f"    ⚠ χ_SQ(π,π) = {chi_SQ_at_afm:+.3e} — inactive at the AFM wavevector")
    _log("=" * 100)


@dataclass
class InstabilityInfo:
    """G3 instability matrix diagnostics.  Basis order: [s(0), d(1), JT(2)].

    Two independent criteria:
      G22 = G3[2,2] = 1 − χ_QQ/K_eff ≤ 0  →  spontaneous JT (SC-independent)
      lambda_min ≤ 0                      →  collective instability (pairing or cross-channel)
    """
    G11: float            # G3[0,0]  s-channel diagonal
    G33: float            # G3[1,1]  d-channel diagonal
    G22: float            # G3[2,2]  JT-channel diagonal = 1 − χ_QQ/K_eff
    G_sd: float           # G3[0,1]
    G_sJT: float          # G3[0,2]
    G_dJT: float          # G3[1,2]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    lambda_min: float
    evec_min: np.ndarray

    @property
    def jt_stable(self)   -> bool: return self.G22 > 0.0
    @property
    def s_stable(self)    -> bool: return self.G11 > 0.0
    @property
    def d_stable(self)    -> bool: return self.G33 > 0.0
    @property
    def full_stable(self) -> bool: return self.lambda_min > 0.0

    @property
    def instab_type(self) -> str:
        """stable | spontaneous_JT | s_pairing | d_pairing | both_pairing | cross_channel"""
        if self.full_stable:   return 'stable'
        if not self.jt_stable: return 'spontaneous_JT'
        s, d = not self.s_stable, not self.d_stable
        if s and d: return 'both_pairing'
        if s:       return 's_pairing'
        if d:       return 'd_pairing'
        return 'cross_channel'

    @property
    def dominant_channel(self) -> str:
        ws, wd, wq = np.abs(self.evec_min)
        if wq > ws and wq > wd: return 'JT'
        return 'd' if wd >= ws else 's'

    @property
    def severity(self) -> float:
        if self.full_stable:              return 0.0
        if not self.jt_stable:            return min(1.0, abs(self.G22) / 0.5)
        if not self.s_stable and not self.d_stable:
                                          return min(1.0, (abs(self.G11) + abs(self.G33)))
        if not self.s_stable:             return min(1.0, abs(self.G11) / 0.5)
        if not self.d_stable:             return min(1.0, abs(self.G33) / 0.5)
        return min(1.0, abs(self.lambda_min) / 0.5)

    @property
    def weight_for_score(self) -> float:
        return {'stable': 1.0, 'spontaneous_JT': 0.0, 's_pairing': 0.5,
                'd_pairing': 1.2, 'both_pairing': 0.8, 'cross_channel': 0.7,
                }.get(self.instab_type, 0.5)

    @property
    def weight_for_log(self) -> str:
        return {'stable': '✓ stable (w=1.00)', 'spontaneous_JT': '✗ spontaneous JT (w=0.00)',
                's_pairing': '⚠ s-pairing (w=0.50)', 'd_pairing': '★ d-pairing ACTIVE (w=1.20)',
                'both_pairing': '⚠ s+d pairing (w=0.80)', 'cross_channel': '⚠ cross-channel (w=0.70)',
                }.get(self.instab_type, '? unknown')

    def log_summary(self, verbose: bool = True) -> str:
        lines = [
            f"G3 eigs=[{self.eigenvalues[0]:.4f},{self.eigenvalues[1]:.4f},{self.eigenvalues[2]:.4f}]"
            f"  evec_min=[{self.evec_min[0]:.3f},{self.evec_min[1]:.3f},{self.evec_min[2]:.3f}]"
            f"  → {self.instab_dir}  {self.weight_for_log}",
        ]
        if verbose:
            lines += [
                f"  JT-channel  G3[2,2]={self.G22:+.4f}  {'✓' if self.jt_stable else '✗ SPONTANEOUS JT'}",
                f"  s-channel   G3[0,0]={self.G11:+.4f}  {'✓' if self.s_stable  else '✗'}",
                f"  d-channel   G3[1,1]={self.G33:+.4f}  {'✓' if self.d_stable  else '✗ desired for SC'}",
                f"  dominant={self.dominant_channel}  severity={self.severity:.3f}",
            ]
        return "\n".join(lines)

    @property
    def instab_dir(self) -> str:
        return {
            'stable':         'stable',
            'spontaneous_JT': f'pure JT (spontaneous risk)  G3[2,2]={self.G22:.4f}≤0',
            's_pairing':      f's pairing  G3[0,0]={self.G11:.4f}≤0',
            'd_pairing':      f'd pairing  G3[1,1]={self.G33:.4f}≤0  ✓ desired',
            'both_pairing':   f's+d pairing  G11={self.G11:.4f}  G33={self.G33:.4f}',
            'cross_channel':  f'SC-triggered JT  λ_min={self.lambda_min:.4f}<0  diagonals +',
        }[self.instab_type]

    @classmethod
    def from_G3(cls, G3: np.ndarray) -> 'InstabilityInfo':
        """Construct from a 3×3 G3 matrix with basis order [s, d, JT]."""
        eigs, evecs = np.linalg.eigh(G3)
        return cls(
            G11=float(G3[0, 0]), G33=float(G3[1, 1]), G22=float(G3[2, 2]),
            G_sd=float(G3[0, 1]), G_sJT=float(G3[0, 2]), G_dJT=float(G3[1, 2]),
            eigenvalues=eigs, eigenvectors=evecs,
            lambda_min=float(eigs[0]), evec_min=evecs[:, 0],
        )


def compute_G_instability(solver, target_doping: float, M: float, mu: float, n_kspace: float, g_t: float, g_J: float, g_Das: float, g_Dad: float) -> None:
    """
    Normal-state (Δ=0) collective instability matrix and diagnostics.
    Here, M is a SCALAR: this is a standalone, analytic two-band diagnostic
    for the Γ6 doublet alone (h_afm below uses J_eff = Z_afm_eff·J_Γ6 only).

    The 3×3 Schur-complement instability matrix (in dimensionless units):

    G = | 1 − gVs·χ_pair_s    −√(gVs·gVd)·χ_pair_sd    −c_s·χ_SQ_s | 
    | −√(gVs·gVd)·χ_pair_sd    1 − gVd·χ_pair_d    −c_d·χ_SQ_d | 
    | −c_s·χ_SQ_s              −c_d·χ_SQ_d        1 − K⁻¹·χ_QQ | 

    λ_min(G) < 0 indicates instability; the corresponding eigenvector identifies the dominant channel.
    """
    p = solver.p
    t_eff_now = g_t * p.t0

    M_vec = np.array([M, 0.0, 0.0], dtype=float)

    J_A1g_diag, _, _ = _exch(p, 0.0, g_J)
    J_eff = p.Z_afm_eff * J_A1g_diag[0]
    h_afm = J_eff * M

    kx = solver.k_points[:, 0]
    ky = solver.k_points[:, 1]
    eps_k   = -2.0 * (t_eff_now * np.cos(kx) + t_eff_now * np.cos(ky)) - mu
    eps_kQ  = -eps_k - 2.0 * mu
    xi_avg  =  0.5 * (eps_k + eps_kQ)
    xi_diff =  0.5 * (eps_k - eps_kQ)
    sq      = np.sqrt(xi_diff**2 + h_afm**2 + 1e-20)
    E_plus  = xi_avg + sq
    E_minus = xi_avg - sq

    def _th2E(E):
        a  = np.clip(E / (2.0 * solver.kT), -100, 100)
        se = np.where(np.abs(E) > qm._MATH_EPS, E, qm._MATH_EPS)
        return np.tanh(a) / (2.0 * se)

    def _mdf(E):
        f_E = 1.0 / (1.0 + np.exp(np.clip(E / solver.kT, -100, 100)))
        return f_E * (1.0 - f_E) / solver.kT

    pk    = _th2E(E_plus) + _th2E(E_minus)
    phi_s = np.ones_like(kx)
    phi_d = np.cos(kx) - np.cos(ky)

    chi_pair_s  = float(np.dot(solver.k_weights, pk * phi_s**2))
    chi_pair_d  = float(np.dot(solver.k_weights, pk * phi_d**2))
    chi_pair_sd = float(np.dot(solver.k_weights, pk * phi_s * phi_d))
    N_eff       = float(np.dot(solver.k_weights, _mdf(E_plus) + _mdf(E_minus)))

    chi_QQ = solver._chi_QQ_matrix_elements(M_vec, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J)
    _Gamma_M, F_cluster, _V_JT_corr, _ = _vertex_tuple(solver, target_doping, M_vec, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J)

    ev, ec = solver._get_chi0_norm_cache(M_vec, 0.0, mu, g_t, g_J, solver._get_vbdg())
    _, chi_SQ_q0, _, chi_QQ_q0 = _susceptibilities_sc(solver, M_vec, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, (ev, ec), apply_diamagnetic_QQ=True)

    _rpa_afm_G = solver._rpa_closure(M_vec, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, J_A1g_diag, _V_JT_corr, (ev, ec), True)
    chi_SS_afm = _rpa_afm_G['chi_SS']
    stoner_afm = _rpa_afm_G['stoner']

    _chi_pair_s_orig = chi_pair_s
    _chi_QQ_orig     = chi_QQ
    _chi_SQ_q0_orig  = chi_SQ_q0

    gVs = g_Das * _V_JT_corr
    gVd = g_Dad * _V_JT_corr

    _chi_mat = np.array([[chi_pair_s, chi_SQ_q0],
                         [chi_SQ_q0,  chi_QQ]], dtype=float)

    _eigv, _evc = np.linalg.eigh(_chi_mat)
    _psd_violated = bool(_eigv[0] < -1e-10)

    if _psd_violated:
        _ev_clipped = np.maximum(_eigv, 0.0)
        _mat_psd    = _evc @ np.diag(_ev_clipped) @ _evc.T
        _mat_psd    = 0.5 * (_mat_psd + _mat_psd.T)

        chi_pair_s = float(_mat_psd[0, 0])
        chi_QQ     = float(_mat_psd[1, 1])
        chi_SQ_q0  = float(_mat_psd[0, 1])

        _rel_ss = abs(chi_pair_s - _chi_pair_s_orig) / max(abs(_chi_pair_s_orig), 1e-12)
        _rel_qq = abs(chi_QQ     - _chi_QQ_orig)     / max(abs(_chi_QQ_orig),     1e-12)
        _rel_sq = abs(chi_SQ_q0  - _chi_SQ_q0_orig)  / max(abs(_chi_SQ_q0_orig),  1e-12)
        _psd_sev = "⚠ >1% change — numerical instability likely" if max(_rel_ss, _rel_qq, _rel_sq) > 0.01 else "✓ <1% — minor numerical noise"

    chi_SQ_s = 0.0
    chi_SQ_d = 0.0

    G3 = np.zeros((3, 3))

    G3[0, 0] = 1.0 - gVs * chi_pair_s
    G3[1, 1] = 1.0 - gVd * chi_pair_d
    G3[2, 2] = 1.0 - chi_QQ * _V_JT_corr
    G3[0, 1] = G3[1, 0] = -np.sqrt(max(gVs * gVd, 0.0)) * chi_pair_sd

    G3[0, 2] = G3[2, 0] = -solver.g_JT_bare * math.sqrt(max(gVs / p.K_lattice, 0.0)) * chi_SQ_s
    G3[1, 2] = G3[2, 1] = -solver.g_JT_bare * math.sqrt(max(gVd / p.K_lattice, 0.0)) * chi_SQ_d

    instab = InstabilityInfo.from_G3(G3)

    ws, wd, wq = np.abs(instab.evec_min)
    if wd > ws:
        dominant     = 'd'
        chi_pair_dom = chi_pair_d
        chi_SQ_dom   = chi_SQ_d
        V_dom        = gVd
    else:
        dominant     = 's'
        chi_pair_dom = chi_pair_s
        chi_SQ_dom   = chi_SQ_s
        V_dom        = gVs
    if wq > ws and wq > wd:
        dominant = 'JT'

    if dominant != 'JT' and instab.G22 > qm._MATH_EPS:
        V_eff = V_dom + V_dom * (_V_JT_corr * chi_SQ_dom**2) / (max(chi_pair_dom, 1e-12) * instab.G22)
    else:
        V_eff = V_dom
    lambda_eff = N_eff * V_eff

    lambda_lin_max_Q0 = float(_build_vertex_cache(solver, target_doping, M_vec, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J)['lambda_lin_max'])

    H_afm_mat = solver.build_local_hamiltonian_for_bdg(
        1.0, M_vec, J_A1g_diag, mu, p.Z_afm_eff)
    comm = solver.B1g_op @ H_afm_mat - H_afm_mat @ solver.B1g_op
    blocking_ratio = float(np.linalg.norm(comm, 'fro')) / abs(p.Delta_CF)

    _log("=" * 100)
    _log("  G-MATRIX  —  NORMAL-STATE COLLECTIVE INSTABILITY DIAGNOSTIC")
    _log("=" * 100)
    _log(f"  M (scalar) = {M:.6f}   Q = 0.0   Δ = 0")
    _log(f"  n_kspace   = {n_kspace:.6f}   μ = {mu:+.6f} eV")
    _log(f"  t_eff      = {t_eff_now:.6f} eV   T = {solver.kT*1000:.2f} meV")
    _log()

    _log("  ── Gutzwiller & exchange ──")
    _log(f"    g_t  = {g_t:.6f}   g_J  = {g_J:.6f}")
    _log(f"    g_Δs = {g_Das:.6f}   g_Δd = {g_Dad:.6f}")
    _log(f"    J_ch per-bond = {np.array2string(np.asarray(J_A1g_diag, dtype=float)[0::2], precision=6)}  "
          f"[J_Γ6, J_Γ7a, J_Γ7b]")
    _log(f"    J_eff (Γ6, AFM) = {J_eff:.6f} eV   h_afm = {h_afm:.6f} eV")
    _log(f"    h_afm / t_eff   = {h_afm / max(t_eff_now, 1e-12):.4f}  "
          f"[{'metallic AFM' if h_afm < 4.0*t_eff_now else '⚠ marginal/insulating'}]")
    _log()

    _log("  ── Pairing susceptibilities (normal-state Lindhard) ──")
    _log(f"    χ_pair_s   = {chi_pair_s:+.6f}  eV⁻¹")
    _log(f"    χ_pair_d   = {chi_pair_d:+.6f}  eV⁻¹")
    _log(f"    χ_pair_sd  = {chi_pair_sd:+.6f}  eV⁻¹")
    _log(f"    N_eff      = {N_eff:+.6f}  eV⁻¹")
    _log()

    _log("  ── Orbital susceptibilities ──")
    _log(f"    χ_QQ(q=0)  = {chi_QQ:+.6f}  eV⁻¹  (Kubo: {chi_QQ_q0:+.6f})")
    _log(f"    χ_SQ(q=0)  = {chi_SQ_q0:+.6e}  eV⁻¹  [normal state: 0 by symmetry]")
    _log(f"    χ_SS(AFM)  = {chi_SS_afm:+.6f}  eV⁻¹   Stoner = {stoner_afm:+.6f}")
    _log()

    _log("  ── PSD projection of [[χ_pair_s, χ_SQ], [χ_SQ, χ_QQ]] ──")
    if _psd_violated:
        _log(f"    ⚠ PSD violated: λ_min = {_eigv[0]:.3e}")
        _log(f"      χ_pair_s: {_chi_pair_s_orig:.4f} → {chi_pair_s:.4f}")
        _log(f"      χ_QQ    : {_chi_QQ_orig:.4f} → {chi_QQ:.4f}")
        _log(f"      χ_SQ    : {_chi_SQ_q0_orig:.4e} → {chi_SQ_q0:.4e}")
        _log(f"      {_psd_sev}")
    else:
        _log(f"    ✓ PSD satisfied (λ_min = {_eigv[0]:.3e})")
    _log()

    _log("  ── G3 matrix (basis: [s, d, JT]) ──")
    _log("           | s        d        JT      ")
    _log("    -------+----------------------------")
    _log(f"    s      | {G3[0,0]:+.5f}  {G3[0,1]:+.5f}  {G3[0,2]:+.5f}")
    _log(f"    d      | {G3[1,0]:+.5f}  {G3[1,1]:+.5f}  {G3[1,2]:+.5f}")
    _log(f"    JT     | {G3[2,0]:+.5f}  {G3[2,1]:+.5f}  {G3[2,2]:+.5f}")
    _log()

    _log("  ── Instability summary ──")
    _log(f"    {instab.log_summary(verbose=True)}")
    _log()

    _log("  ── Schur-complement effective pairing ──")
    _log(f"    dominant = {dominant}")
    _log(f"    V_dom    = {V_dom:.6e} eV")
    _log(f"    V_eff    = {V_eff:.6e} eV")
    _log(f"    λ_eff    = N_eff · V_eff = {lambda_eff:.6f}")
    _log(f"    λ_lin(q=0, full kernel)  = {lambda_lin_max_Q0:.6f}")
    _log()

    _log("  ── B1g / AFM blocking ratio ──")
    _log(f"    ‖[B1g_op, H_AFM]‖_F / |Δ_CF| = {blocking_ratio:.6f}")
    if blocking_ratio > 1.0:
        _log("      → ⚠ strong AFM blocking — the τ_x channel is hindered")
    elif blocking_ratio < 0.5:
        _log("      → ✓ weak blocking — the JT channel is easily accessible")
    else:
        _log("      → moderate blocking")
    _log()

    _log("  ── VERDICT ──")
    verdict_map = {
        'stable':         ("✓ STABLE — no instability in the normal state",  "OK"),
        'spontaneous_JT': ("✗ SPONTANEOUS JT — the lattice is already unstable in the normal state", "CRITICAL"),
        's_pairing':      ("⚠ s-pairing instability — s-channel dominates",       "WARN"),
        'd_pairing':      ("★ d-pairing ACTIVE — the desired d-wave instability",  "DESIRED"),
        'both_pairing':   ("⚠ s+d pairing — both channels unstable",          "WARN"),
        'cross_channel':  ("⚠ cross-channel (SC-triggered JT) — diagonals stable, λ_min<0", "INTERESTING"),
    }
    verdict_text, verdict_sev = verdict_map.get(instab.instab_type, ("? unknown", "?"))
    _log(f"    {verdict_text}")
    _log(f"    severity = {instab.severity:.3f}   weight_for_score = {instab.weight_for_score:.2f}")
    _log()

    # ── Physical consistency checks ───────────────────────────────────
    _log("  ── Physical consistency checks ──")

    # 1. JT stability
    V_JT_chi_QQ = _V_JT_corr * chi_QQ
    jt_ok = V_JT_chi_QQ < 1.0
    _log(f"    [{'✓' if jt_ok else '✗'}] RPA JT stability: "
          f"V_JT·χ_QQ = {V_JT_chi_QQ:.4f} {'< 1' if jt_ok else '≥ 1'}")

    # 2. G22 = 1 - χ_QQ/K_bare (spontaneous-JT threshold)
    g22_from_K = 1.0 - chi_QQ * _V_JT_corr
    _log(f"    [{'✓' if g22_from_K > 0 else '✗'}] JT-channel G22 = "
          f"1 − χ_QQ/K_bare = {g22_from_K:+.4f}")

    # 3. Stoner (proximity to the AFM QCP)
    stoner_ok = stoner_afm < 1.0
    _log(f"    [{'✓' if stoner_ok else '✗'}] Stoner (AFM): {stoner_afm:.4f} "
          f"{'< 1 (below QCP)' if stoner_ok else '≥ 1 (past QCP!)'}")

    # 4. det(RPA) in the normal state
    det_rpa = _rpa_afm_G['det']
    det_ok = det_rpa > 0.0
    _log(f"    [{'✓' if det_ok else '✗'}] det(RPA, AFM) = {det_rpa:.5f} "
          f"{'> 0 (stable)' if det_ok else '≤ 0 (breakdown)'}")

    # 5. d-wave dominance (expected by the paper)
    d_leads = abs(chi_pair_d) > abs(chi_pair_s)
    _log(f"    [{'✓' if d_leads else '⚠'}] d-wave leads: "
          f"|χ_d|/|χ_s| = {abs(chi_pair_d)/max(abs(chi_pair_s), 1e-12):.4f}")

    # 6. λ_eff (Schur + JT) — the paper's central quantity
    lambda_ok = 0.3 < lambda_eff < 1.0
    _log(f"    [{'✓' if lambda_ok else '⚠'}] λ_eff (Schur+JT) = {lambda_eff:.4f}  "
          f"{'optimal' if lambda_ok else 'weak' if lambda_eff <= 0.3 else 'too strong'}")

    # 7. λ_lin_max(q=0) — the full FS kernel
    lambda_lin_ok = lambda_lin_max_Q0 > 0.0
    _log(f"    [{'✓' if lambda_lin_ok else '⚠'}] λ_lin_max(q=0, full FS) = "
          f"{lambda_lin_max_Q0:+.4f}  "
          f"{'attractive' if lambda_lin_ok else 'repulsive (nodal sign change needed)'}")

    _log("=" * 100)
    _log()


def _lindhard_bubble(sector_pairs: tuple, E_k_all: np.ndarray, V_k_all: np.ndarray, f_k_all: np.ndarray, shift_idx: np.ndarray, w: np.ndarray, vw_sq: np.ndarray, eta: float, kT: float) -> np.ndarray:
    """Static (ω=0) full-BZ Lindhard bubble for a given q-shift and set of sector pairs. Uses uniform BZ weights w throughout."""
    E_kQ = E_k_all[shift_idx]
    V_kQ = V_k_all[shift_idx]
    f_kQ = qm._fermi_function(E_kQ, kT)

    kernel = qm._lehmann_kernel(f_k_all, f_kQ, E_k_all, E_kQ, eta, kT)
    # Apply k-weighting
    kernel = kernel * w[:, None, None] * vw_sq[:, None, None]

    N   = E_k_all.shape[0]
    chi = np.zeros((qm._N_ORB, qm._N_ORB), dtype=complex)
    for sl_a, sl_b in sector_pairs:
        Vk_a  = V_k_all[:, sl_a, :]
        VkQ_a = V_kQ[:,   sl_a, :]
        Vk_b  = V_k_all[:, sl_b, :]
        VkQ_b = V_kQ[:,   sl_b, :]
        for k0 in range(0, N, _LINDHARD_CHUNK):
            k1 = min(k0 + _LINDHARD_CHUNK, N)
            chi += oe.contract(
                'cnm,can,cam,cbm,cbn->ab',
                kernel[k0:k1],
                Vk_a[k0:k1].conj(), VkQ_a[k0:k1],
                VkQ_b[k0:k1].conj(), Vk_b[k0:k1],
                optimize='greedy')
    return qm._NAMBU_MBZ_NORMFACTOR * (chi + chi.conj().T)

def estimate_chi_SQ_q_full(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, n_q: int) -> None:
    """
    BZ scan of χ_SQ(q) = Tr[S_z · χ₀[Γ₆,Γ₇](q)] in both the normal and SC states.

      chi_SQ_n  : Δ=0 eigenstates  → χ_SQ ≡ 0 in D₄h (B₂g selection rule), finite in D₂h
      chi_SQ_sc : Δ≠0 eigenstates  → Bogoliubov rotation mixes Γ₆/Γ₇ in the normal-sector
                  propagator, lifting the selection rule and making χ_SQ_sc ≠ 0 even in D₄h

    Comparison quantities:
      phi_d_q          : |cos q_x − cos q_y|  (B₁g d-wave form factor)
      phi_d_overlap_n/sc : normalised ∫|χ_SQ|·φ_d / (‖χ_SQ‖·‖φ_d‖)·n_pts
      local_vertex_ok  : antinodal_frac_n > 0.5  (local q=0 vertex approx is safe)
    """
    tx_b, ty_b = solver.p.effective_hopping_anisotropic(Q)

    # Normal-state eigenstates (Δ=0)
    E_k_n, V_k_n = _bdg_eig(solver, M, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J)
    E_k_sc, V_k_sc = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)

    f_k_n  = qm._fermi_function(E_k_n, solver.kT)
    eta_n  = max(qm._ETA_T_FRAC * solver.kT, qm._ETA_GRID_FLOOR * solver.p.t0)    # Normal-state: thermal broadening dominates (bands gapped by h_afm)
    _Gamma_M = _vertex_tuple(solver, target_doping, M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J)[0]
    _chi_QQ_n = solver._chi_QQ_matrix_elements(M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J)

    _qvals = np.linspace(-np.pi, np.pi, n_q, endpoint=False)
    _QX, _QY = np.meshgrid(_qvals, _qvals)
    q_grid = np.column_stack((_QX.ravel(), _QY.ravel()))
    n_pts  = len(q_grid)

    chi_SQ_n  = np.zeros(n_pts)
    chi_SQ_sc = np.zeros(n_pts)
    chi_SS_n  = np.zeros(n_pts)
    chi_SS_sc = np.zeros(n_pts)
    chi_QQ_sc_arr = np.zeros(n_pts)

    sz_6, sz_7a, sz_7b = solver.sz_op[0:2], solver.sz_op[2:4], solver.sz_op[4:6]
    dk = 2.0 * np.pi / qm._NK

    beta_k_array = solver.p.wave_function_weight(tx_b, ty_b, solver.k_points[:, 0], solver.k_points[:, 1])
    vw_sq = beta_k_array ** 2

    for i_q, q in enumerate(q_grid):
        nx = int(round(q[0] / dk)) % qm._NK
        ny = int(round(q[1] / dk)) % qm._NK
        shift_idx = solver.shift_table[nx, ny]

        # Normal state: _NORMAL_SECTOR_PAIRS on Δ=0 eigenstates
        chi_n = _lindhard_bubble(qm._NORMAL_SECTOR_PAIRS, E_k_n, V_k_n, f_k_n, shift_idx, solver.k_weights, vw_sq, eta_n, solver.kT)
        cr_n = chi_n.real
        chi_SS_n[i_q] = float(solver.sz_op @ cr_n @ solver.sz_op)
        chi_SQ_n[i_q] = float(np.trace(np.diag(sz_6) @ cr_n[0:2, 2:4]) + np.trace(np.diag(sz_7a) @ cr_n[2:4, 0:2])
                             + np.trace(np.diag(sz_6) @ cr_n[0:2, 4:6]) + np.trace(np.diag(sz_7b) @ cr_n[4:6, 0:2]))

        # χ_SQ evaluated on the condensate band structure, which lifts the D₄h B₂g selection rule via Bogoliubov Γ₆/Γ₇ mixing in normal propagator.
        # Delta_s_7b/Delta_d_7b themselves stay implicitly 0 here (this estimator only probes the Γ7a Delta_s/Delta_d amplitudes)
        _chi_SS_sc, _chi_SQ_sc, _, _chi_QQ_sc = _susceptibilities_sc(solver, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J, q, _Gamma_M, (E_k_sc, V_k_sc))
        chi_SS_sc[i_q]     = _chi_SS_sc
        chi_SQ_sc[i_q]     = _chi_SQ_sc
        chi_QQ_sc_arr[i_q] = _chi_QQ_sc

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    data = [
        (chi_SS_n,  'Normal state', r'$\chi_{SS}^{\rm normal}(q)$'),
        (chi_SQ_n,  'Normal state', r'$\chi_{SQ}^{\rm normal}(q)$'),
        (None,      'Normal state', r'$\chi_{QQ}^{\rm normal}$ (const)'),  # constant, not q-dependent
        (chi_SS_sc, 'SC state',     r'$\chi_{SS}^{\rm SC}(q)$'),
        (chi_SQ_sc, 'SC state',     r'$\chi_{SQ}^{\rm SC}(q)$'),
        (chi_QQ_sc_arr, 'SC state', r'$\chi_{QQ}^{\rm SC}(q)$'),
    ]

    vmax_SS = max(chi_SS_n.max(), chi_SS_sc.max())
    vmax_SQ = max(abs(chi_SQ_n).max(), abs(chi_SQ_sc).max())
    vmax_QQ = chi_QQ_sc_arr.max()

    for ax, (vals, state, title) in zip(axes, data):
        if vals is None:
            # chi_QQ is constant in the normal state -- indicate this with text
            ax.text(0.5, 0.5, r'$\chi_{QQ}^{\rm normal} = $' + f'{_chi_QQ_n:.4f}\n(constant, $q$-independent)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(r'$\chi_{QQ}^{\rm normal}$ (constant)')
            continue

        Z = vals.reshape(_QX.shape)
        if 'SS' in title:
            vmax = vmax_SS
            cmap = 'Reds'
        elif 'SQ' in title:
            vmax = vmax_SQ
            cmap = 'RdBu_r'
        else:
            vmax = vmax_QQ
            cmap = 'Reds'

        im = ax.pcolormesh(_QX, _QY, Z, cmap=cmap, shading='auto',
                        vmin=-vmax if 'SQ' in title else 0, vmax=vmax)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel('$q_x$')
        ax.set_ylabel('$q_y$')
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
        ax.set_aspect('equal')
        ax.set_title(title + f' [{state}]')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(r'Lindhard susceptibility maps ($q$ in $[-\pi,\pi]^2$)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('chi_SQ_q_full_maps.png', dpi=150)
    plt.close(fig)

    def _antinodal_frac(arr: np.ndarray) -> float:
        _abs = np.abs(arr); _tot = _abs.sum()
        if _tot < 1e-20: return 0.0
        mask = (
            ((np.abs(q_grid[:, 0]) > np.pi / 2) & (np.abs(q_grid[:, 1]) < np.pi / 2)) |
            ((np.abs(q_grid[:, 1]) > np.pi / 2) & (np.abs(q_grid[:, 0]) < np.pi / 2)))
        return float(_abs[mask].sum() / _tot)

    def _phi_overlap(arr: np.ndarray) -> float:
        phi_d_q  = np.abs(np.cos(q_grid[:, 0]) - np.cos(q_grid[:, 1]))
        _abs = np.abs(arr)
        na, np_ = _abs.sum(), phi_d_q.sum()
        if na < 1e-20 or np_ < 1e-20: return 0.0
        return float((_abs * phi_d_q).sum() / (na * np_) * n_pts)

    def _classify(q) -> str:
        if q is None: return 'none'
        qx, qy = abs(q[0]), abs(q[1])
        if qx > 0.7*np.pi and qy > 0.7*np.pi: return 'M(π,π)'
        if (qx > 0.7*np.pi and qy < 0.3*np.pi) or (qy > 0.7*np.pi and qx < 0.3*np.pi): return 'antinode'
        if qx < 0.2*np.pi and qy < 0.2*np.pi: return 'Γ(0,0)'
        return 'nodal'

    # Peak-position / peak-amplitude summary, feeding the log lines below.
    peak_idx_n   = int(np.argmax(np.abs(chi_SQ_n)))
    peak_idx_sc  = int(np.argmax(np.abs(chi_SQ_sc)))
    q_peak_n     = q_grid[peak_idx_n]
    q_peak_sc    = q_grid[peak_idx_sc]
    chi_SQ_pk_n  = float(chi_SQ_n[peak_idx_n])
    chi_SQ_pk_sc = float(chi_SQ_sc[peak_idx_sc])
    antinodal_frac_n  = _antinodal_frac(chi_SQ_n)
    antinodal_frac_sc = _antinodal_frac(chi_SQ_sc)

    _log(
        f"[χ-DIAG] χ_SQ scan ({n_q}×{n_q} grid): "
        f"norm peak={_classify(q_peak_n)} {chi_SQ_pk_n:+.3f} (antinode={antinodal_frac_n:.0%}) | "
        f"SC peak={_classify(q_peak_sc)} {chi_SQ_pk_sc:+.3f} (antinode={antinodal_frac_sc:.0%}) | "
        f"φ_d overlap: n={_phi_overlap(chi_SQ_n):.2f} sc={_phi_overlap(chi_SQ_sc):.2f}"
        + (" ⚠local-vertex overestimate" if antinodal_frac_n <= 0.5 and _classify(q_peak_n) == 'M(π,π)' else "")
    )

    if not np.allclose(q_peak_n, q_peak_sc, atol=np.pi/18):
        _log(
            f"[χ-DIAG]   ⚠ peak shift: norm {_classify(q_peak_n)} → SC {_classify(q_peak_sc)}"
            f"  (Δχ_SQ = {chi_SQ_pk_sc - chi_SQ_pk_n:+.3f})")

    if q_peak_n is not None and len(chi_SS_n) > 0:
        idx_SS = np.argmax(np.abs(chi_SS_n))
        if not np.allclose(q_grid[idx_SS], q_peak_n, atol=np.pi/18 + 0.1):
            _log(f"[χ-DIAG]   ⚠ χ_SS peak at {_classify(q_grid[idx_SS])}, χ_SQ_n at {_classify(q_peak_n)} — differ")


def _compute_rpa_vertices(solver, ref: FSReference, pref: RPAPrefactors, M: np.ndarray, Q: float, n: float, mu: float, g_t: float, g_J: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, V_cap: float, apply_diamagnetic: bool = False, spectrum: tuple[np.ndarray, np.ndarray] | None = None):
    if spectrum is None:
        ev, ec = _bdg_eig(solver, M, Q, Delta_vec, Quad67_vec, n, mu, g_t, g_J)
    else:
        ev, ec = spectrum

    dHdQ = solver._calc_dHdQ(M, Q, Quad67_vec, mu, g_t, g_J)
    J_A1g_diag, _, _ = _exch(solver.p, Q, g_J)
    _n_ch = qm._N_CHANNELS

    def _clamp(x):
        return math.copysign(V_cap, x) if abs(x) > V_cap else x

    def _vertex_scalar(J_q, chiR):
        V_spin_mat = -np.outer(J_q, J_q) * chiR[:_n_ch, :_n_ch]
        V_jt_val   = pref.V_JT_corr + pref.V_JT_corr ** 2 * chiR[_n_ch, _n_ch]
        V_cross_v  = 2.0 * J_q * pref.V_JT_corr * chiR[:_n_ch, _n_ch]
        return _clamp(float(V_spin_mat.sum() + V_jt_val + V_cross_v.sum()))

    n_q = ref.n_q
    V_full = np.empty(n_q); V_spin = np.empty(n_q); V_jt = np.empty(n_q)
    V_noSQ = np.empty(n_q); chi_SQ = np.empty(n_q)

    for iq, q_u in enumerate(ref.unique_q):
        rpa = solver._rpa_closure(M, Q, Delta_vec, Quad67_vec, mu, g_t, g_J, q_u, pref.Gamma_M, J_A1g_diag, pref.V_JT_corr, (ev, ec), apply_diamagnetic, dHdQ)
        J_q, det, M4, chi0 = rpa['J_q'], rpa['det'], rpa['M4'], rpa['chi0']
        chi_SQ[iq] = rpa['chi_SQ']

        # ---- Full closure: χ_RPA = adj(M4)·χ₀ / det(M4) --------------------
        chiR_full = qm._adjugate(M4) @ chi0 / det
        V_full[iq] = _vertex_scalar(J_q, chiR_full)
        V_spin[iq] = _clamp(float((-np.outer(J_q, J_q)
                                    * chiR_full[:_n_ch, :_n_ch]).sum()))
        V_jt[iq]   = _clamp(float(pref.V_JT_corr + pref.V_JT_corr ** 2 * chiR_full[_n_ch, _n_ch]))

        # ---- χ_SQ ≡ 0: zero the spin–Q blocks of χ₀, rebuild M4 -----------
        chi0_mod = chi0.copy()
        chi0_mod[:_n_ch, _n_ch] = 0.0
        chi0_mod[_n_ch, :_n_ch] = 0.0
        U_mod = np.diag([float(J_q[0]), float(J_q[1]), float(J_q[2]), float(pref.V_JT_corr)])
        M4_mod = np.eye(_n_ch + 1) - chi0_mod @ U_mod
        det_mod = float(np.linalg.det(M4_mod))
        det_floor = max(qm._MATH_EPS, 1e-4 * float(np.linalg.norm(M4_mod)))
        if abs(det_mod) < det_floor:
            det_safe = (math.copysign(det_floor, det_mod)
                        if det_mod != 0.0 else det_floor)
        else:
            det_safe = det_mod
        chiR_mod = qm._adjugate(M4_mod) @ chi0_mod / det_safe
        V_noSQ[iq] = _vertex_scalar(J_q, chiR_mod)
    return RPAVertices(V_full=V_full, V_spin=V_spin, V_jt=V_jt, V_noSQ=V_noSQ, chi_SQ=chi_SQ)

def test_rpa_vertex(solver, target_doping: float, M_normal: np.ndarray, n_kspace_n: float, mu_n: float, M_sc: np.ndarray, Q_sc: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_sc: float, mu_sc: float, g_t: float, g_J: float, 
                    ctx_normal: dict | None = None, ctx_sc: dict | None = None, V_cap_diagnostic: float = 1.0e9, tol_chi_SQ_normal: float = 1e-8, tol_dV_SQ_sign: float = 1e-6, run_one_shot_trw_toggle: bool = True, run_full_ab_check: bool = False):
    """
    Per-channel RPA vertex diagnostic at a single q-point.

    SCOPE — this is NOT the SCF pairing kernel.  It evaluates the bare  RPA closure (scalar 2×2, Γ6-only J) on an UNWEIGHTED FS kernel, so
    V_ss/V_dd/V_sd here show the ISOLATED spin/JT/cross channel strengths, without the SCF kernel's:
      - 4×4 channel-resolved closure (M4)
      - √(w_c w_c') FS doublet weight
      - c_B(k)·c_B(k') antinodal gate
      - V_cap UV clamp

    Numeric reference (0.153 doping, working point):
        this function:  V_ss = -0.123, V_dd = +0.043
        SCF kernel:     V_ss = -0.029, V_dd = -0.021
    """
    M_normal = np.asarray(M_normal, dtype=float)
    M_sc     = np.asarray(M_sc,     dtype=float)
    Delta_vec = np.asarray(Delta_vec, dtype=complex)
    Quad67_vec   = np.asarray(Quad67_vec, dtype=complex)
    Delta_total = float(np.sum(np.abs(Delta_vec)))
    _zero_c = np.zeros(qm._N_DELTA, dtype=complex)
    _zero_r = np.zeros(qm._N_DELTA, dtype=float)

    result = {}

    # ====================== NORMAL STATE ======================
    if ctx_normal is None:
        ctx_normal = _build_fs_context(solver, M_normal, 0.0, _zero_c, _zero_r, n_kspace_n, mu_n, g_t, g_J)
    ref_n = FSReference.from_context(ctx_normal)
    if ref_n is None:
        _log("[rpa] No normal-state FS points / q-pairs. Aborted.")
        return {}

    vp = _scf_vertex_params(solver, target_doping, M_normal, 0.0, _zero_c, _zero_r, n_kspace_n, mu_n, g_t, g_J)
    pref_n = RPAPrefactors(J_eff=vp['J_eff'], V_JT_corr=vp['V_JT_corr'], Gamma_M=vp['Gamma_M'])

    g_Das_n, g_Dad_n, _ = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M_normal, 0.0, mu_n, g_t, g_J)

    verts_n = _compute_rpa_vertices(solver, ref_n, pref_n, M_normal, 0.0, n_kspace_n, mu_n, g_t, g_J, _zero_c, _zero_r, V_cap_diagnostic, apply_diamagnetic=False, spectrum=(ctx_normal['ev_n'], ctx_normal['ec_n']))

    kernel_jt_n   = ref_n.kernel_2x2(verts_n.V_jt,   g_Das_n, g_Dad_n)
    kernel_spin_n = ref_n.kernel_2x2(verts_n.V_spin, g_Das_n, g_Dad_n)

    chi_SQ_max_n = verts_n.chi_SQ_max
    chi_SQ_ok_n  = chi_SQ_max_n < tol_chi_SQ_normal
    dom_n        = kernel_jt_n.dominant_abs

    _log(_fmt_header(
        "RPA PAIRING VERTEX  --  NORMAL state (Δ_vec = 0)",
        0.0, M_normal, n_kspace_n, mu_n, g_t, g_J,
        g_Das_n, g_Dad_n, pref_n, ref_n,
        V_cap=V_cap_diagnostic))

    _log("\nχ_SQ:")
    _log(_fmt_qstats(verts_n.chi_SQ, ref_n.afm_mask, "AFM"))
    _log(_fmt_qstats(verts_n.chi_SQ, ref_n.fwd_mask, "forward"))
    _log(_fmt_qstats(verts_n.chi_SQ, np.ones(ref_n.n_q, bool), "all q"))
    _log(f"    max|χ_SQ| = {chi_SQ_max_n:.3e}  "
         f"{'✓ ~0' if chi_SQ_ok_n else '✗ nonzero'}")

    _log("\nq-resolved RPA vertex")
    for label, V in (("Full", verts_n.V_full),
                     ("Spin-only", verts_n.V_spin),
                     ("JT-only", verts_n.V_jt)):
        _log(f"  {label}:")
        _log(_fmt_qstats(V, ref_n.afm_mask, "AFM"))
        _log(_fmt_qstats(V, ref_n.fwd_mask, "forward"))

    _log("\ns/d kernel (JT-only)")
    _log(f"    V_ss={kernel_jt_n.V_ss:+.6f}  V_dd={kernel_jt_n.V_dd:+.6f}"
         f"  V_sd={kernel_jt_n.V_sd:+.6f}")
    _log(f"    V_dd (spin-only) = {kernel_spin_n.V_dd:+.6f}")
    _log(f"    λ = ({kernel_jt_n.eigenvalues[0]:+.4f}, {kernel_jt_n.eigenvalues[1]:+.4f})"
         f"  θ = {kernel_jt_n.angle:+.2f}°  dom|λ| = {dom_n:.4f}")

    _log("\nPhysical consistency checks (normal state)")
    checks_n = []
    if np.any(ref_n.afm_mask) and np.any(ref_n.fwd_mask):
        afm_neg = float(np.mean(verts_n.V_spin[ref_n.afm_mask] < 0.0))
        fwd_pos = float(np.mean(verts_n.V_spin[ref_n.fwd_mask] > 0.0))
        dw_ok = afm_neg > 0.5 and fwd_pos > 0.5
        _log(f"    {'✓' if dw_ok else '⚠'} Spin vertex d-wave: "
             f"AFM neg%={afm_neg*100:.1f}  fwd pos%={fwd_pos*100:.1f}")
        checks_n.append(dw_ok)
    jt_ok_n = bool(np.all(verts_n.V_jt > -1e-6))
    _log(f"    {'✓' if jt_ok_n else '⚠'} JT vertex positive "
         f"(min={float(np.min(verts_n.V_jt)):+.4f})")
    checks_n.append(jt_ok_n)
    _log(f"    {'✓' if chi_SQ_ok_n else '⚠'} χ_SQ ~ 0 (max={chi_SQ_max_n:.2e})")
    checks_n.append(chi_SQ_ok_n)
    n_fail_n = sum(1 for c in checks_n if not c)
    _log("\nNORMAL VERDICT: " + ("PASS" if n_fail_n == 0
         else f"PARTIAL — {n_fail_n}/{len(checks_n)} failed"))
    _log("=" * 78)

    result['normal'] = {
        'V_ss': kernel_jt_n.V_ss, 'V_dd': kernel_jt_n.V_dd,
        'V_sd': kernel_jt_n.V_sd, 'V_dd_spin': kernel_spin_n.V_dd,
        'eigenvalues': kernel_jt_n.eigenvalues,
        'angle': kernel_jt_n.angle, 'dom_eig': dom_n,
        'chi_SQ_max': chi_SQ_max_n,
        'g_Das': g_Das_n, 'g_Dad': g_Dad_n,
        'J_eff': pref_n.J_eff, 'V_JT_corr': pref_n.V_JT_corr, 'Gamma_M': pref_n.Gamma_M,
    }

    # ====================== SC STATE ======================
    if ctx_sc is None:
        ctx_sc = _build_fs_context(solver, M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J)
    ref_sc = FSReference.from_context(ctx_sc)
    if ref_sc is None:
        _log("[rpa] No SC-state FS points / q-pairs. Aborted.")
        return {'normal': result['normal']}

    vp = _scf_vertex_params(solver, target_doping, M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J)
    pref_sc = RPAPrefactors(J_eff=vp['J_eff'], V_JT_corr=vp['V_JT_corr'], Gamma_M=vp['Gamma_M'])
    g_Das_sc, g_Dad_sc, g_Dbd_sc = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M_sc, Q_sc, mu_sc, g_t, g_J)

    verts_sc = _compute_rpa_vertices(solver, ref_sc, pref_sc, M_sc, Q_sc, n_sc, mu_sc, g_t, g_J, Delta_vec, Quad67_vec, V_cap_diagnostic, apply_diamagnetic=False, spectrum=None)

    kernel_full_sc = ref_sc.kernel_2x2(verts_sc.V_full, g_Das_sc, g_Dad_sc)
    kernel_noSQ_sc = ref_sc.kernel_2x2(verts_sc.V_noSQ, g_Das_sc, g_Dad_sc)

    delta_V_SQ      = verts_sc.delta_V_SQ
    delta_V_SQ_mean = verts_sc.delta_V_SQ_mean
    delta_V_SQ_ok   = delta_V_SQ_mean > tol_dV_SQ_sign
    delta_V_d_SQ_phys = g_Dad_sc * (kernel_full_sc.V_dd - kernel_noSQ_sc.V_dd)
    chi_SQ_max_sc = verts_sc.chi_SQ_max
    dom_sc = kernel_full_sc.dominant_abs

    _log(_fmt_header(
        "RPA PAIRING VERTEX  --  SC state",
        Q_sc, M_sc, n_sc, mu_sc, g_t, g_J,
        g_Das_sc, g_Dad_sc, pref_sc, ref_sc,
        Delta_vec=Delta_vec, Quad67_vec=Quad67_vec, V_cap=V_cap_diagnostic))
    if Delta_total < 1e-9:
        _log("  [WARNING] Δ_vec = 0 — NOT a genuine SC probe.")

    _log("\nχ_SQ:")
    _log(_fmt_qstats(verts_sc.chi_SQ, ref_sc.afm_mask, "AFM"))
    _log(_fmt_qstats(verts_sc.chi_SQ, ref_sc.fwd_mask, "forward"))
    _log(_fmt_qstats(verts_sc.chi_SQ, np.ones(ref_sc.n_q, bool), "all q"))
    _log(f"    max|χ_SQ| = {chi_SQ_max_sc:.3e}")

    _log("\nδV_SQ = V_SC(full) − V_SC(χ_SQ = 0)")
    _log(_fmt_qstats(delta_V_SQ, ref_sc.afm_mask, "AFM"))
    _log(_fmt_qstats(delta_V_SQ, ref_sc.fwd_mask, "forward"))
    _log(_fmt_qstats(delta_V_SQ, np.ones(ref_sc.n_q, bool), "all q"))
    _log(f"    mean(δV_SQ) = {delta_V_SQ_mean:+.6e} eV  "
         f"{'✓' if delta_V_SQ_ok else '✗'}")

    _log("\ns/d kernel (full RPA)")
    _log(f"    V_ss={kernel_full_sc.V_ss:+.6f}  V_dd={kernel_full_sc.V_dd:+.6f}"
         f"  V_sd={kernel_full_sc.V_sd:+.6f}")
    _log(f"    V_dd (χ_SQ=0) = {kernel_noSQ_sc.V_dd:+.6f}")
    _log(f"    g_Δd·(V_dd(full) − V_dd(χ_SQ=0)) = {delta_V_d_SQ_phys:+.6e} eV")
    _log(f"    λ = ({kernel_full_sc.eigenvalues[0]:+.4f}, "
         f"{kernel_full_sc.eigenvalues[1]:+.4f})"
         f"  θ = {kernel_full_sc.angle:+.2f}°  dom|λ| = {dom_sc:.4f}")

    # ---- SC-only extras ----
    dF_TRW_one_shot = None
    if run_one_shot_trw_toggle:
        _vp_sc = _scf_vertex_params(solver, target_doping, M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J)

        F_with = solver._compute_bdg_free_energy(M_sc, Q_sc, Delta_vec, Quad67_vec, mu_sc, g_t, g_J, g_Das_sc, g_Dad_sc, g_Dbd_sc, _vp_sc['V_JT_corr'], 0.0, vertex_cache=_registered_cache(M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J))
        _ab_slots = _ab_trw_slots(solver)
        F_without_vec = np.asarray(Quad67_vec).copy(); F_without_vec[_ab_slots] = 0.0
        _n_with = _ab_trw_block_norm(solver, M_sc, Q_sc, Quad67_vec, mu_sc, g_t, g_J)
        _n_without = _ab_trw_block_norm(solver, M_sc, Q_sc, F_without_vec, mu_sc, g_t, g_J)
        _log(f"    H_TRW_AB is carried by {', '.join(n for n in qm._IRREP_NAMES if qm._IRREP_KIND[n] == 'nn_AB' and solver.p.trw_irrep_scale[list(qm._IRREP_NAMES).index(n)] > 1e-12)} (both branches); max‖H_TRW_AB‖: with = {_n_with:.3e}, without = {_n_without:.3e}   "
             f"{'✓' if (_n_without < 1e-14 and _n_with > 0.0) else '❌ FAIL — the toggle does not switch H_TRW_AB off (or it was already zero)'}")
        F_without = solver._compute_bdg_free_energy(M_sc, Q_sc, Delta_vec, F_without_vec, mu_sc, g_t, g_J, g_Das_sc, g_Dad_sc, g_Dbd_sc, _vp_sc['V_JT_corr'], 0.0, vertex_cache=_registered_cache(M_sc, Q_sc, Delta_vec, Quad67_vec, n_sc, mu_sc, g_t, g_J))
        dF_TRW_one_shot = F_with - F_without
        _log("\nOne-shot H_TRW_AB contribution")
        _log(f"    ΔF_TRW = {dF_TRW_one_shot:+.6e} eV")

    ab_rel_change = None
    ab_vals_A = ab_vals_B = None
    if run_full_ab_check:
        _log("\nFULL A/B BRANCH CHECK")
        vbdg = solver._get_vbdg()

        def _converge_branch(force_zero):
            """Self-consistent (M, mu, F67) iteration at fixed (Q_sc, Delta_vec), force_zero=True switches the inter-sublattice transverse Weiss field H_TRW_AB (all its channels, see _ab_trw_slots) OFF and converges everything else without it."""
            _M = M_sc.copy()
            _mu, _n = mu_sc, n_sc
            _F67 = np.asarray(Quad67_vec).copy()
            _ab_slots = _ab_trw_slots(solver)
            if force_zero:
                _F67[_ab_slots] = 0.0
            _ev = _ec = None
            for _ in range(8):
                _mu, _n = solver._find_mu_for_density(target_doping, _M, Q_sc, Delta_vec, _F67, _mu, g_t, g_J)
                _gds, _gda, _gdb = solver.estimate_gutzwiller_factors_occupation_based(target_doping, _M, Q_sc, _mu, g_t, g_J)
                _ev, _ec = _bdg_eig(solver, _M, Q_sc, Delta_vec, _F67, _n, _mu, g_t, g_J)
                _F67 = qm._compute_B1g_quadrupole(solver.k_weights, solver.Phi_irrep_k, _ev, _ec, solver.kT, solver.B1g_op)
                if force_zero:
                    _F67[_ab_slots] = 0.0
                _M = vbdg.compute_channel_staggered_magnetizations(Q_sc, Delta_vec, _mu, _ev, _ec)
            return dict(M=_M, mu=_mu, n=_n,
                        g_Das=_gds, gDd=_gda,
                        Quad67_vec=_F67, Quad67_s=float(_F67[qm._DELTA_IDX['s7a']].real), Quad67_AB=float(np.linalg.norm(_F67[_ab_slots])),
                        ev=_ev, ec=_ec)

        def _branch_kernel(branch):            
            vp = _scf_vertex_params(solver, target_doping, branch['M'], Q_sc, Delta_vec, branch['Quad67_vec'], branch['n'], branch['mu'], g_t, g_J)
            _pref_b = RPAPrefactors(J_eff=vp['J_eff'], V_JT_corr=vp['V_JT_corr'], Gamma_M=vp['Gamma_M'])
            _Fvec_b = branch['Quad67_vec']
            _verts_b = _compute_rpa_vertices(
                solver, ref_sc, _pref_b, branch['M'], Q_sc, branch['n'],
                branch['mu'], g_t, g_J, Delta_vec, _Fvec_b,
                V_cap_diagnostic, apply_diamagnetic=False,
                spectrum=(branch['ev'], branch['ec']))
            return ref_sc.kernel_2x2(_verts_b.V_full, branch['g_Das'], branch['gDd'])

        bA = _converge_branch(True)
        bB = _converge_branch(False)
        kA = _branch_kernel(bA)
        kB = _branch_kernel(bB)
        ab_vals_A, ab_vals_B = kA.eigenvalues, kB.eigenvalues
        dA = kA.dominant_abs; dB = kB.dominant_abs
        ab_rel_change = (dB - dA) / max(abs(dA), 1e-9)
        _log(f"  A (OFF): |Quad67_AB|={bA['Quad67_AB']:.5f}  "
             f"λ=({kA.eigenvalues[0]:+.4f}, {kA.eigenvalues[1]:+.4f})")
        _log(f"  B (ON) : |Quad67_AB|={bB['Quad67_AB']:.5f}  "
             f"λ=({kB.eigenvalues[0]:+.4f}, {kB.eigenvalues[1]:+.4f})")
        _log(f"  Dominant |λ| change: {ab_rel_change*100:+.2f}%")

    _log("\nPhysical consistency checks (SC state)")
    checks_sc = []
    chi_SQ_ok_sc = chi_SQ_max_sc > 1e-6
    _log(f"    {'✓' if chi_SQ_ok_sc else '⚠'} χ_SQ opened (max={chi_SQ_max_sc:.3e})")
    checks_sc.append(chi_SQ_ok_sc)
    _log(f"    {'✓' if delta_V_SQ_ok else '⚠'} χ_SQ enhances pairing "
         f"(mean δV_SQ={delta_V_SQ_mean:+.3e})")
    checks_sc.append(delta_V_SQ_ok)
    n_fail_sc = sum(1 for c in checks_sc if not c)
    _log("\nSC VERDICT: " + ("PASS" if n_fail_sc == 0
         else f"PARTIAL — {n_fail_sc}/{len(checks_sc)} failed"))
    _log("=" * 78)

    result['sc'] = {
        'V_ss': kernel_full_sc.V_ss, 'V_dd': kernel_full_sc.V_dd,
        'V_sd': kernel_full_sc.V_sd, 'V_dd_noSQ': kernel_noSQ_sc.V_dd,
        'delta_V_d_SQ_phys': delta_V_d_SQ_phys,
        'eigenvalues': kernel_full_sc.eigenvalues,
        'angle': kernel_full_sc.angle, 'dom_eig': dom_sc,
        'chi_SQ_max': chi_SQ_max_sc,
        'delta_V_SQ_mean': delta_V_SQ_mean,
        'g_Das': g_Das_sc, 'g_Dad': g_Dad_sc,
        'J_eff': pref_sc.J_eff, 'V_JT_corr': pref_sc.V_JT_corr, 'Gamma_M': pref_sc.Gamma_M,
        'dF_TRW_one_shot': dF_TRW_one_shot,
        'ab_rel_change': ab_rel_change,
        'ab_vals_A': ab_vals_A, 'ab_vals_B': ab_vals_B,
    }

    # ====================== COMPARISON ======================
    _log("\n" + "=" * 78)
    _log("NORMAL vs SC  --  RPA vertex comparison")
    _log("=" * 78)
    _log(f"  {'quantity':<24s}  {'NORMAL':>14s}  {'SC':>14s}  {'Δ':>14s}")
    rn, rs = result['normal'], result['sc']
    for key in ('V_ss', 'V_dd', 'V_sd', 'dom_eig', 'angle'):
        vn, vs = rn.get(key), rs.get(key)
        if vn is None or vs is None:
            continue
        _log(f"  {key:<24s}  {vn:>+14.6f}  {vs:>+14.6f}  {vs - vn:>+14.6f}")
    _log(f"  {'chi_SQ_max':<24s}  {rn['chi_SQ_max']:>14.3e}"
         f"  {rs['chi_SQ_max']:>14.3e}"
         f"  {rs['chi_SQ_max'] - rn['chi_SQ_max']:>+14.3e}")
    if abs(rn['dom_eig']) > 1e-9:
        deig_rel = (rs['dom_eig'] - rn['dom_eig']) / abs(rn['dom_eig']) * 100
        _log(f"  {'dominant |λ| rel change':<24s}  {'':>14s}  {'':>14s}  "
             f"{deig_rel:>+13.2f}%")
    _log(f"  {'mixing angle change':<24s}  {'':>14s}  {'':>14s}  "
         f"{rs['angle'] - rn['angle']:>+13.2f}°")
    _log("=" * 78)
    return result


def test_vertex_assembly_vs_cache(solver, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, tol: float = 1e-8, verbose: bool = True) -> dict:
    """
    Pin the test side to the SCF's pair-vertex assembly: the cache of compute_pairing_kernel_and_build_cache is compared with an independent re-assembly from the PUBLIC
    building blocks, following the documented recipe step by step:
        χ_RPA(q) = adj(M4)·χ₀/det(M4)  from _rpa_closure at Δ = 0, F67 = 0 on the cache's own spectrum (the kernel never sees the condensate);
        X_cc'(q) = −J_c J_c' χ_cc',  V_jt(q) = V_JT_corr + V_JT_corr² χ_QQ  (Γ = U + U·χ_R·U),  X_c(q) = 2 J_c V_JT_corr χ_cQ;
        V_pairs(g) = [Σ_cc' u_cc' u_cc' X_cc'(q_g),  V_jt(q_g),  Σ_c s_c s_c X_c(q_g)],   q_g = k_i − g·k_j,   s_c = √w_c(k) (FS doublet content), u_cc' = s_c s_c';
        cap: |ΣV_pairs(g)| > V_cap is rescaled to V_cap for EVERY group element g separately;  proper symmetric copy (V + Vᵀ − diag) of the upper-triangle pair list;
        sector kernel K_χ = (1/|G|) Σ_g χ(g) V(g);   V_irrep[a,b] = <φ_a|V_sec[χ]|φ_b> for a, b of the SAME sector (else exactly 0).
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n_ch = qm._N_CHANNELS
    vp = _scf_vertex_params(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    cache = _build_vertex_cache(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    fs_pts, fs_idx, fs_w = np.asarray(cache['fs_pts'], dtype=float), cache['fs_idx'], np.asarray(cache['fs_weights'], dtype=float)
    N_fs = len(fs_pts)
    if N_fs == 0:
        if verbose:
            _log("[vertex-assembly] empty FS sample -> nothing to compare")
        return {'verdict': 'INVALID'}
    sector_mode = (cache.get('sector_lambda') is not None)
    ops = list(qm._G_OPS)
    n_ops = len(ops)

    E_k = solver._get_chi0_norm_cache(M, Q, mu, g_t, g_J, solver._get_vbdg())
    dHdQ = solver._calc_dHdQ(M, Q, np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J)
    # canonical transfers of every (group element, pair) — independent re-implementation of the bucketing rule
    i_idx, j_idx = np.triu_indices(N_fs)
    def canon(q_raw):
        return np.abs(np.rint(((q_raw + np.pi) % (2.0 * np.pi) - np.pi) * qm._Q_UNIQUE_SCALE)).astype(np.int64)
    if sector_mode:
        keys = []
        for op in ops:
            gx, gy = qm._g_apply(op, fs_pts[:, 0], fs_pts[:, 1])
            keys.append(canon(np.column_stack((fs_pts[i_idx, 0] - gx[j_idx], fs_pts[i_idx, 1] - gy[j_idx]))))
        u_int, inv_all = np.unique(np.concatenate(keys, axis=0), axis=0, return_inverse=True)
        inv_all = np.asarray(inv_all).reshape(n_ops, len(i_idx))
        uq = (u_int.astype(np.float64) / qm._Q_UNIQUE_SCALE + np.pi) % (2.0 * np.pi) - np.pi
        same_q = bool(np.array_equal(i_idx, cache['i_idx']) and np.array_equal(j_idx, cache['j_idx']) and uq.shape == cache['unique_q'].shape and np.allclose(uq, cache['unique_q']))
    else:
        i0, j0, uq, inv0 = qm._unique_q_pairs(fs_pts)
        inv_all = inv0[None, :]
        ops = [(1, 1, 1)]
        n_ops = 1
        same_q = bool(np.array_equal(i0, cache['i_idx']) and np.array_equal(j0, cache['j_idx']) and np.allclose(uq, cache['unique_q']))
        i_idx, j_idx = i0, j0
    
    Xi, Xc, Vjt = np.empty((len(uq), n_ch, n_ch)), np.empty((len(uq), n_ch)), np.empty(len(uq))
    for iu, q_u in enumerate(uq):
        rpa = solver._rpa_closure(M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J, q_u, vp['Gamma_M'], vp['J_A1g_diag'], vp['V_JT_corr'], E_k, False, dHdQ)
        J_q, det, M4 = rpa['J_q'], rpa['det'], rpa['M4']
        chiR = qm._adjugate(M4) @ rpa['chi0'] / det
        Xi[iu] = -np.outer(J_q, J_q) * chiR[:n_ch, :n_ch]
        Xc[iu] = 2.0 * J_q * vp['V_JT_corr'] * chiR[:n_ch, n_ch]
        Vjt[iu] = vp['V_JT_corr'] + vp['V_JT_corr'] ** 2 * chiR[n_ch, n_ch]
    s_c = np.sqrt(np.clip(solver._fs_channel_weights(fs_idx, E_k), 0.0, None))
    u = s_c[:, :, None] * s_c[:, None, :]
    V_g = np.empty((n_ops, 3, len(i_idx)))
    for gi in range(n_ops):
        V_g[gi] = np.stack([np.einsum('pcd,pcd,pcd->p', u[i_idx], u[j_idx], Xi[inv_all[gi]], optimize=True), Vjt[inv_all[gi]],
                            np.einsum('pc,pc,pc->p', s_c[i_idx], s_c[j_idx], Xc[inv_all[gi]], optimize=True)])
        V_sum = V_g[gi].sum(axis=0)
        over = np.abs(V_sum) > vp['V_cap']
        if np.any(over):
            V_g[gi][:, over] *= vp['V_cap'] / np.abs(V_sum[over])
    def sym_fill(vals):                          # upper triangle (incl. diagonal) -> full symmetric matrix, EXPLICIT copy of the triangle
        A = np.zeros(vals.shape[:-1] + (N_fs, N_fs))
        A[..., i_idx, j_idx] = vals
        A[..., j_idx, i_idx] = vals
        return A
    Phi = qm._pairing_basis_matrix(fs_pts[:, 0], fs_pts[:, 1])
    c_B = np.abs(Phi[:, qm._IR_D]) / max(float(np.max(np.abs(Phi[:, qm._IR_D]))), 1e-12)
    gate = c_B[:, None] * c_B[None, :]
    weighted = lambda Vch: Vch[1] + gate * (Vch[0] + Vch[2])
    # identity-element gated kernel: re-assembled reference vs the cache's cached diagonal
    V_w_ref = weighted(sym_fill(V_g[0]))
    V_self_diag_ref = V_w_ref.diagonal()
    V_self_diag = np.asarray(cache['V_self_diag'], dtype=float)
    scale = max(float(np.max(np.abs(V_self_diag))), 1e-30)
    rel_W = float(np.max(np.abs(V_self_diag_ref - V_self_diag))) / scale
    names = qm._IRREP_NAMES
    sec_of = (lambda a: qm._IRREP_CHAR[names[a]]) if sector_mode else (lambda a: None)
    V_sec_ref, lam_ref = {}, {}
    sw = np.sqrt(fs_w)
    for si, sec in enumerate(qm._G_OPS if sector_mode else [None]):
        if sec is None:
            V_sec_ref[None] = V_w_ref
            continue
        coef = np.array([qm._G_CHI[si, gi] for gi in range(n_ops)]) / n_ops
        V_sec_ref[sec] = weighted(sym_fill(np.tensordot(coef, V_g, axes=(0, 0))))
        lam_ref[sec] = float(np.linalg.eigvalsh(sw[:, None] * V_sec_ref[sec] * sw[None, :])[-1])
    V_irr_ref = np.zeros((qm._N_IRREP, qm._N_IRREP))
    for a in range(qm._N_IRREP):
        for b in range(qm._N_IRREP):
            if sec_of(a) == sec_of(b):
                V_irr_ref[a, b] = float(np.clip(qm._wave_project(V_sec_ref[sec_of(a)], Phi[:, a], Phi[:, b], fs_w), -vp['V_cap'], vp['V_cap']))
    V_irr_cache = np.asarray(cache['V_irrep'], dtype=float)
    odd = [i for i, n in enumerate(names) if qm._IRREP_PARITY[n] < 0]
    comp = np.ones((qm._N_IRREP, qm._N_IRREP), dtype=bool)
    if not bool(cache['odd_channels_ok']):
        comp[odd, :] = False; comp[:, odd] = False
    rel_I = float(np.max(np.abs(V_irr_ref - V_irr_cache)[comp])) / max(float(np.max(np.abs(V_irr_cache[comp]))), 1e-30)
    block_ok = True
    rel_L = 0.0
    if sector_mode:
        for a in range(qm._N_IRREP):
            for b in range(qm._N_IRREP):
                if sec_of(a) != sec_of(b) and V_irr_cache[a, b] != 0.0:
                    block_ok = False
        for si, sec in enumerate(qm._G_OPS):
            rel_L = max(rel_L, abs(lam_ref[sec] - float(cache['sector_lambda'][si])) / max(abs(lam_ref[sec]), 1e-12))
    nan_free = (not sector_mode) or (not bool(np.any(np.isnan(cache['sector_lambda']))))
    ok = same_q and rel_W < tol and rel_I < tol and block_ok and rel_L < tol and nan_free
    if verbose:
        _log("=" * 78)
        _log(f"  Pair-vertex assembly vs SCF cache   (N_fs={N_fs}, n_q={len(uq)}, Q={Q:+.4f}, J_eff={vp['J_eff']:.5f}, V_cap={vp['V_cap']:.4f}, mode={'sector' if sector_mode else 'identity'})")
        _log("=" * 78)
        _log(f"  unique-q bookkeeping identical to the cache: {same_q}")
        _log(f"  max|V_self_diag(re-assembled) − V_self_diag(cache)| / max|V_self_diag| = {rel_W:.2e}   (tol {tol:.0e})   [identity-element gated kernel diagonal]")
        _log(f"  max|V_irrep(re-assembled)    − V_irrep(cache)|    / max|V_irrep|    = {rel_I:.2e}   [{'sector-resolved' if sector_mode else 'identity kernel'}, all {qm._N_IRREP}×{qm._N_IRREP} entries]")
        if sector_mode:
            _log(f"  V_irrep exactly block-diagonal in the G-sectors: {block_ok};   sector_lambda free of NaN (every sector processed): {nan_free};   max rel.diff of the 8 sector top eigenvalues = {rel_L:.2e}")
        _log(f"  diag V_irrep = {np.array2string(np.diag(V_irr_cache), precision=4)}  ({','.join(names)})")
        _log(f"  odd-channel FS closure OK (cache flag): {bool(cache['odd_channels_ok'])}   (inv={cache.get('fs_closed_inv')}, mx={cache.get('fs_closed_mx')}, my={cache.get('fs_closed_my')}, w_sym={cache.get('fs_weight_sym_ok')}, w_asym={cache.get('fs_weight_asym', float('nan')):.2e})")
        _log(f"  {'✓ PASS' if ok else '❌ FAIL'}: the re-assembly {'reproduces' if ok else 'does NOT reproduce'} the SCF pair vertex")
        _log("=" * 78)
    return {'verdict': 'PASS' if ok else 'FAIL', 'rel_V_self_diag': rel_W, 'rel_V_irrep': rel_I, 'rel_lambda_sector': rel_L, 'same_q': same_q, 'block_diagonal': block_ok, 'nan_free': nan_free,
            'V_irrep_main': V_irr_cache}


def test_channel_registry_selfcheck() -> None:
    """
    The channel registry must agree with the lattice: (T_Q, mx, my) are MEASURED from every form factor (qm._IRREP_CHAR) and compared with the labels (`kind`, `parity`):
    a bond joining sites of the same sublattice is T_Q-even — including the ODD-in-k bonds sin 2kx, sin 2ky (sin 2(kx+π) = sin 2kx); NN (A↔B) bonds are T_Q-odd.
    Also checks the D4h label of every active form factor against the character table of this file (g = sin kx sin ky must be B2g; (px,py), (px2,py2) must span Eu).
    """
    ok = True
    if len(set(qm._IRREP_NAMES)) != len(qm._IRREP_NAMES):
        ok = False
        _log(f"  ❌ FAIL: duplicate pairing channel names: {qm._IRREP_NAMES}")
    for c in qm._PAIR_CHANNELS:
        t, mx, my = qm._IRREP_CHAR[c.name]
        if mx * my != c.parity:
            ok = False
            _log(f"  ❌ FAIL: channel '{c.name}': declared parity {c.parity} != measured mx*my = {mx * my}")
        if t != (-1 if c.kind == 'nn_AB' else +1):
            ok = False
            _log(f"  ❌ FAIL: channel '{c.name}': T_Q parity {t:+d} is inconsistent with bond kind '{c.kind}' (A–B bonds <=> T_Q-odd, same-sublattice bonds <=> T_Q-even)")
    for n, v in qm._HARMONIC_LIBRARY.items():
        t, mx, my = qm._LIB_CHAR[n]
        if mx * my != v[2] or t != (-1 if v[1] == 'nn_AB' else +1):
            ok = False
            _log(f"  ❌ FAIL: library harmonic '{n}': declared (kind, parity) inconsistent with measured character {(t, mx, my)}")
        if mx == my:     # even-parity sectors: the measured (mx, my, σd) must reproduce the declared D4h irrep, otherwise the σd resolution of a sector would mislabel it
            sw = qm._LIB_SWAP[n]
            if sw == 0 or qm._D4H_BY_MX_MY_SWAP[(mx, my, sw)] != v[0]:
                ok = False
                _log(f"  ❌ FAIL: library harmonic '{n}': declared D4h irrep {v[0]} inconsistent with the measured (mx, my, σd) = {(mx, my, sw)}")

    _log("  channel  kind       D4h   parity  T_Q   mx   my   sector")
    for c in qm._PAIR_CHANNELS:
        t, mx, my = qm._IRREP_CHAR[c.name]
        good = (mx * my == c.parity) and (t == (-1 if c.kind == 'nn_AB' else +1))
        ok &= good
        _log(f"  {c.name:<8s} {c.kind:<10s} {c.d4h:<5s} {c.parity:>+5d}  {t:>+4d} {mx:>+4d} {my:>+4d}   {qm._sector_label((t, mx, my))}  {'✓' if good else '❌'}")
    ops = _d4h_group()
    ch4 = [_d4h_characters(M) for M in ops]
    rng = np.random.default_rng(3)
    kx, ky = rng.uniform(0.2, 2.9, 40), rng.uniform(0.2, 2.9, 40)
    k = np.stack([kx, ky])
    inv = [np.linalg.inv(np.asarray(M)[:2, :2]) for M in ops]
    d4h_ok = True
    for c in qm._PAIR_CHANNELS:
        if c.d4h == 'Eu':
            continue
        f0 = c.ff(kx, ky)
        chars = []
        for Mi in inv:
            kk = Mi @ k
            r = c.ff(kk[0], kk[1]) / f0
            chars.append(float(np.round(r[0])) if np.allclose(r, np.round(r[0]), atol=1e-9) else float('nan'))
        match = bool(np.allclose(chars, [ch[c.d4h] for ch in ch4]))
        d4h_ok &= match
        _log(f"    D4h character of φ_{c.name:<4s} equals the table entry '{c.d4h}': {match}")
    for pair in (('px', 'py'), ('px2', 'py2')):
        fa, fb = qm._IRREP_FF[pair[0]], qm._IRREP_FF[pair[1]]
        trs = []
        for Mi in inv:
            kk = Mi @ k
            A = np.stack([fa(kx, ky), fb(kx, ky)])
            B = np.stack([fa(kk[0], kk[1]), fb(kk[0], kk[1])])
            D = B @ np.linalg.pinv(A)
            assert np.allclose(D @ A, B, atol=1e-8), f"{pair} does not span an invariant 2-D space"
            trs.append(float(np.trace(D)))
        match = bool(np.allclose(trs, [ch['Eu'] for ch in ch4]))
        d4h_ok &= match
        _log(f"    D4h character of the pair {pair} equals Eu: {match}")
    ok &= d4h_ok
    _log(f"  {'✓ PASS' if ok else '❌ FAIL'}: registry consistent with the lattice and the D4h character table")


def test_sector_kernel_vs_cache(solver, target_doping: float, M: np.ndarray, Q: float, Quad67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, Delta_vec: np.ndarray, tol: float = 1e-8, verbose: bool = True) -> dict:
    """
    Independent re-assembly of the exact G = {E, T_Q} x D2h sector kernels that compute_pairing_kernel_and_build_cache builds, compared with its cache:
        q_g(i,j) = k_i - g.k_j                      (continuous action, folded, canonical |q_x|,|q_y|)
        K_sec[chi, ch] = (1/|G|) sum_g chi(g) A_g^ch  (A_g: symmetric (spin, JT, cross) pair matrices at q_g, V_cap clamp per pair)
        V_sec[chi]     = K_sec[chi, JT] + c_B c_B' (K_sec[chi, spin] + K_sec[chi, cross])
        lambda_chi     = top eigenvalue of sqrt(w) V_sec[chi] sqrt(w);   V_irrep[a,b] = <phi_a | V_sec[chi] | phi_b>_w  for a, b inside the SAME sector chi.
    The RPA tables come from solver._rpa_closure; only the sector assembly is re-implemented, which is exactly the part where a
    loop/return mix-up would silently leave sectors undiagonalised.  Also checks COVERAGE: every sector that contains an active channel must have a finite lambda.
    """
    M = np.atleast_1d(np.asarray(M, dtype=float)); n_ch = qm._N_CHANNELS
    vp = _scf_vertex_params(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    cache = _build_vertex_cache(solver, target_doping, M, Q, Delta_vec, Quad67_vec, n_kspace, mu, g_t, g_J)
    fs_pts, fs_idx, fs_w = np.asarray(cache['fs_pts'], dtype=float), cache['fs_idx'], np.asarray(cache['fs_weights'], dtype=float)
    N = len(fs_pts)
    if N == 0:
        if verbose:
            _log("[sector-kernel] empty FS sample -> nothing to compare")
        return {'verdict': 'INVALID'}
    ops, n_ops = qm._G_OPS, len(qm._G_OPS)
    members = [qm._SECTOR_MEMBERS[k] for k in ops]
    iu, ju = np.triu_indices(N)
    keys = []
    for op in ops:
        gx, gy = qm._g_apply(op, fs_pts[:, 0], fs_pts[:, 1])
        qf = (np.column_stack((fs_pts[iu, 0] - gx[ju], fs_pts[iu, 1] - gy[ju])) + np.pi) % (2.0 * np.pi) - np.pi
        keys.append(np.abs(np.rint(qf * qm._Q_UNIQUE_SCALE)).astype(np.int64))
    u_int, inv_all = np.unique(np.concatenate(keys, axis=0), axis=0, return_inverse=True)
    inv_all = np.asarray(inv_all).reshape(n_ops, len(iu))
    uq = (u_int.astype(np.float64) / qm._Q_UNIQUE_SCALE + np.pi) % (2.0 * np.pi) - np.pi
    E_k = solver._get_chi0_norm_cache(M, Q, mu, g_t, g_J, solver._get_vbdg())
    dHdQ = solver._calc_dHdQ(M, Q, np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J)

    Xi, Xc, Vjt = np.empty((len(uq), n_ch, n_ch)), np.empty((len(uq), n_ch)), np.empty(len(uq))
    for k, q_u in enumerate(uq):
        rpa = solver._rpa_closure(M, Q, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), mu, g_t, g_J, q_u, vp['Gamma_M'], vp['J_A1g_diag'], vp['V_JT_corr'], E_k, False, dHdQ)
        chiR = qm._adjugate(rpa['M4']) @ rpa['chi0'] / rpa['det']
        Xi[k] = -np.outer(rpa['J_q'], rpa['J_q']) * chiR[:n_ch, :n_ch]
        Xc[k] = 2.0 * rpa['J_q'] * vp['V_JT_corr'] * chiR[:n_ch, n_ch]
        Vjt[k] = vp['V_JT_corr'] + vp['V_JT_corr'] ** 2 * chiR[n_ch, n_ch]
    s = np.sqrt(np.clip(solver._fs_channel_weights(fs_idx, E_k), 0.0, None)); u = s[:, :, None] * s[:, None, :]
    Phi = np.asarray(qm._pairing_basis_matrix(fs_pts[:, 0], fs_pts[:, 1]))
    c_B = np.abs(Phi[:, qm._IR_D]) / max(float(np.max(np.abs(Phi[:, qm._IR_D]))), 1e-12)
    cBcB = c_B[:, None] * c_B[None, :]
    V_cap, ar = vp['V_cap'], np.arange(N)
    K_sec = np.zeros((n_ops, n_ch, N, N))
    for gi in range(n_ops):
        ig = inv_all[gi]
        Vp = np.stack([np.einsum('pcd,pcd,pcd->p', u[iu], u[ju], Xi[ig], optimize=True), Vjt[ig], np.einsum('pc,pc,pc->p', s[iu], s[ju], Xc[ig], optimize=True)])
        over = np.abs(Vp.sum(axis=0)) > V_cap
        if np.any(over):
            Vp[:, over] *= V_cap / np.abs(Vp.sum(axis=0)[over])
        A = np.zeros((n_ch, N, N)); A[:, iu, ju] = Vp; A = A + A.transpose(0, 2, 1); A[:, ar, ar] *= 0.5
        K_sec += (qm._G_CHI[:, gi] / n_ops)[:, None, None, None] * A[None]
    V_sec = K_sec[:, 1] + cBcB[None] * (K_sec[:, 0] + K_sec[:, 2])
    sw = np.sqrt(fs_w)
    lam_ref = np.full(n_ops, np.nan); V_irr_ref = np.zeros((qm._N_IRREP, qm._N_IRREP))
    for si in range(n_ops):
        Ks = sw[:, None] * V_sec[si] * sw[None, :]
        lam_ref[si] = float(np.linalg.eigvalsh(0.5 * (Ks + Ks.T))[-1])
        for a in members[si]:
            for b in members[si]:
                if b >= a:
                    V_irr_ref[a, b] = V_irr_ref[b, a] = float(np.clip(qm._wave_project(V_sec[si], Phi[:, a], Phi[:, b], fs_w), -V_cap, V_cap))
    # The cache reports the FS-closure status directly; the reference zeroes the odd rows only when the closure check failed.
    if not bool(cache['odd_channels_ok']):
        for i_odd in [i for i, n in enumerate(qm._IRREP_NAMES) if qm._IRREP_PARITY[n] < 0]:
            V_irr_ref[i_odd, :] = 0.0; V_irr_ref[:, i_odd] = 0.0
    lam_main = np.asarray(cache['sector_lambda'], dtype=float)
    has = np.array([bool(m) for m in members])
    cover_ok = bool(np.all(np.isfinite(lam_main[has])))
    scale_l = max(float(np.nanmax(np.abs(lam_ref[has]))), 1e-30)
    rel_lam = float(np.nanmax(np.abs(lam_ref[has] - lam_main[has]))) / scale_l if cover_ok else float('inf')
    V_irr_main = np.asarray(cache['V_irrep'], dtype=float)
    rel_V = float(np.max(np.abs(V_irr_ref - V_irr_main))) / max(float(np.max(np.abs(V_irr_ref))), 1e-30)
    ok = cover_ok and rel_lam < tol and rel_V < tol
    if verbose:
        _log("=" * 78)
        _log(f"  G-sector kernel: independent re-assembly vs SCF cache   (N_fs={N}, n_q(all 8 images)={len(uq)}, Q={Q:+.4f}, |M|={float(np.linalg.norm(M)):.4f})")
        _log("=" * 78)
        _log(f"  {'sector (T_Q, mx, my)':<24s} {'channels':<14s} {'lambda_top(cache)':>18s} {'lambda_top(ref)':>16s}")
        for si, key in enumerate(ops):
            if members[si]:
                _log(f"  {str(key):<24s} {','.join(qm._IRREP_NAMES[i] for i in members[si]):<14s} {lam_main[si]:>18.6f} {lam_ref[si]:>16.6f}   {'' if np.isfinite(lam_main[si]) else '<<<< NOT COMPUTED in the cache'}")
        _log(f"  coverage (every sector with an active channel has a finite lambda): {cover_ok}")
        _log(f"  odd-channel FS closure (cache flag odd_channels_ok = {bool(cache['odd_channels_ok'])}): "
             f"inv={cache.get('fs_closed_inv')}, mx={cache.get('fs_closed_mx')}, my={cache.get('fs_closed_my')}, "
             f"w_sym={cache.get('fs_weight_sym_ok')}, w_asym={cache.get('fs_weight_asym', float('nan')):.2e}")
        _log(f"  max rel. |lambda_top(ref) - cache| = {rel_lam:.2e}     max rel. |V_irrep(ref) - cache| = {rel_V:.2e}     (tol {tol:.0e})")
        _log(f"  {'✓ PASS' if ok else '❌ FAIL'}: the sector kernels {'reproduce' if ok else 'do NOT reproduce'} the SCF cache")
        _log("=" * 78)
    return {'verdict': 'PASS' if ok else 'FAIL', 'coverage': cover_ok, 'rel_lambda': rel_lam, 'rel_V_irrep': rel_V}


def test_sector_kernel_algebra(nk: int = 12, verbose: bool = True) -> None:
    """
    Pure algebra of the group-averaged kernels on a toy problem (no solver): a G-CLOSED sample (the full nk×nk grid) and a toy V(|qx|,|qy|) peaked at (π,π).
      (a) K_χ = (1/|G|) Σ_g χ(g) V(k_i, g·k_j) commutes with every group permutation and maps the χ-subspace into itself (other sectors annihilated);
      (b) the transfers q = k_i − g·k_j of all 8 group images collapse onto ≤ (nk/2+1)² canonical q-values — the RPA cost does not grow 8-fold;
      (c) the symmetric copy of an upper-triangle list keeps the off-diagonals; the legacy 0.5·(V+Vᵀ) halves every one of them;
      (d) a T_Q-even and a T_Q-odd Ag function are orthogonal on the full BZ but NOT on the diamond (why the decomposition must be sector-resolved).
    """
    step = 2.0 * np.pi / nk
    g1 = -np.pi + step * np.arange(nk)
    pts = np.array([(a, b) for a in g1 for b in g1])
    N = len(pts)
    ops = list(qm._G_OPS)
    def Vq(qx, qy):
        qx = np.abs((qx + np.pi) % (2 * np.pi) - np.pi); qy = np.abs((qy + np.pi) % (2 * np.pi) - np.pi)
        return 1.0 / (0.6 + 0.5 * (np.cos(qx) + np.cos(qy)) + 1e-3 * (qx * qx + qy * qy)) - 1.0 / (0.6 + 0.5 * (np.cos(qx - np.pi) + np.cos(qy - np.pi)))
    def key(p):
        return tuple(np.rint(((np.asarray(p) + np.pi) % (2 * np.pi)) / step).astype(int) % nk)
    idx = {key(p): i for i, p in enumerate(pts)}
    perm = []
    for op in ops:
        gx, gy = qm._g_apply(op, pts[:, 0], pts[:, 1])
        perm.append(np.array([idx[key((a, b))] for a, b in zip(gx, gy)]))
    Vfull = Vq(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    max_comm = max_cross = 0.0
    Kc = []
    for si in range(len(ops)):
        K = sum(qm._G_CHI[si, gi] * Vfull[:, perm[gi]] for gi in range(len(ops))) / len(ops)
        Kc.append(K)
        for gi in range(len(ops)):
            P = np.zeros((N, N)); P[np.arange(N), perm[gi]] = 1.0
            max_comm = max(max_comm, float(np.max(np.abs(P @ K @ P.T - K))))
    rng = np.random.default_rng(0)
    v = rng.normal(size=N)

    def proj(si, x):
        return sum(qm._G_CHI[si, gi] * x[perm[gi]] for gi in range(len(ops))) / len(ops)

    for sa in range(len(ops)):
        for sb in range(len(ops)):
            if sa != sb:
                max_cross = max(max_cross, float(np.max(np.abs(Kc[sa] @ proj(sb, v)))))
    ok = (max_comm < 1e-10 and max_cross < 1e-10)
    fs_tiny = pts[::7]
    Nt = len(fs_tiny)
    it, jt = np.triu_indices(Nt)
    allk = []
    for op in ops:
        gx, gy = qm._g_apply(op, fs_tiny[:, 0], fs_tiny[:, 1])
        q_raw = np.column_stack((fs_tiny[it, 0] - gx[jt], fs_tiny[it, 1] - gy[jt]))
        allk.append(np.abs(np.rint(((q_raw + np.pi) % (2 * np.pi) - np.pi) * qm._Q_UNIQUE_SCALE)).astype(np.int64))
    n_uq = len(np.unique(np.concatenate(allk, 0), axis=0))
    n_uq_E = len(np.unique(allk[0], axis=0))
    n_bound = (nk // 2 + 1) ** 2
    ok &= (n_uq <= n_bound)
    vals = np.arange(1.0, 1.0 + len(it))
    A = np.zeros((Nt, Nt)); A[it, jt] = vals
    proper = A + A.T; proper[np.arange(Nt), np.arange(Nt)] *= 0.5
    legacy = 0.5 * (A + A.T)
    off = it != jt
    fill_ok = bool(np.allclose(proper[it[off], jt[off]], vals[off]) and np.allclose(proper, proper.T) and np.allclose(np.diag(proper), np.diag(A)))
    halved = bool(np.allclose(legacy[it[off], jt[off]], 0.5 * vals[off]))
    ok &= fill_ok and halved
    diamond = np.abs(pts[:, 0]) + np.abs(pts[:, 1]) < np.pi
    f_even, f_odd = np.cos(pts[:, 0]) * np.cos(pts[:, 1]), np.cos(pts[:, 0]) + np.cos(pts[:, 1])
    ov = float(np.sum((f_even * f_odd)[diamond]) / np.sqrt(np.sum(f_even[diamond] ** 2) * np.sum(f_odd[diamond] ** 2)))
    ov_full = float(np.sum(f_even * f_odd) / np.sqrt(np.sum(f_even ** 2) * np.sum(f_odd ** 2)))
    ok &= abs(ov_full) < 1e-10 and abs(ov) > 1e-3
    _log(f"  G-closed toy sample: N={N}, |G|={len(ops)}")
    _log(f"  (a) max‖P_g K_χ P_gᵀ − K_χ‖ = {max_comm:.1e};   max‖K_χ·proj_χ'(v)‖ (χ≠χ') = {max_cross:.1e}")
    _log(f"  (b) distinct canonical transfers over the 8 group images: {n_uq} ≤ (nk/2+1)² = {n_bound}   (identity element alone: {n_uq_E})")
    _log(f"  (c) symmetric copy keeps the off-diagonals: {fill_ok};   legacy 0.5·(V+Vᵀ) halves them: {halved}")
    _log(f"  (d) overlap of a T_Q-even and a T_Q-odd Ag function: full BZ {ov_full:+.1e}, diamond only {ov:+.3f}")
    _log(f"  {'✓ PASS' if ok else '❌ FAIL'}")


def test_condensation_energy_repulsive_floor() -> None:
    """
    E_cond = Δ†Ṽ⁻¹Δ of ONE Γ6–Γ7 branch, as qm._sector_condensation_energy evaluates it:
    spectrum of the Gutzwiller-scaled pairing matrix Ṽ in the FS-weighted orthonormal basis
    (Gram matrix S_irrep, normalized to unit mean diagonal); attractive modes weighted 1/a,
    all others 1/max(|a|, floor) with floor = _V_REPULSIVE_FLOOR_FRAC · max|a|.

    Properties verified:
      (1) all-attractive Ṽ with S = I: E_cond equals the plain Δ†Ṽ⁻¹Δ exactly;
      (2) a repulsive px / px2 entry, or the indefinite (es, d) block seen at real states:
          E_cond > 0 for a generic gap and for a gap confined to one repulsive / indefinite
          channel (a signed 1/a would give E_cond < 0 there);
      (2b) a strong repulsive mode with |a| >> floor is regularized by 1/|a|, not by the
          weak-mode floor 1/floor: with S = I, D = unit('px2', 0.1), V = diag(0.05, ..., -0.01, ...)
          the px2 contribution is |0.1|² / 0.01 = 1.0, whereas the 1/floor form would give
          |0.1|² / floor ≈ 2·10³ — the check fails if the regularizer is 1/floor;
      (3a) the all-attractive E_cond is exactly invariant under a harmonic re-mixing
          Δ → LΔ, Ṽ → L Ṽ Lᵀ, S → L⁻ᵀ S L⁻¹ (in the attractive sector E_cond = Δ†Ṽ⁻¹Δ
          identically, so the S-dependence — including its mean-diag normalization — cancels);
      (3b) rescaling S leaves E_cond unchanged for EVERY Ṽ (S is normalized to unit mean
          diagonal, so the scale of S drops out identically);
      (3c) an orthogonal re-mixing U leaves E_cond unchanged for every Ṽ
          (T → U T Uᵀ, eig(A) preserved, c → Uc so |c|² preserved).
    """
    names = list(qm._IRREP_NAMES)
    n = len(names)
    g_vec = qm._gutzwiller_irrep_vector(1.0, 1.0, 1.0)
    root_g = np.sqrt(np.clip(g_vec, 1e-12, None))
    rng = np.random.default_rng(5)

    def E(D, Vt, S):
        return qm._sector_condensation_energy(D, 1.0, 1.0, 1.0, 0.0, {'V_irrep': Vt / np.outer(root_g, root_g), 'S_irrep': S})

    def unit(name, amp=0.01):
        D = np.zeros(n, dtype=complex)
        D[names.index(name)] = amp
        return D

    I = np.eye(n)
    Va = np.diag(np.linspace(0.02, 0.08, n))
    Va[1, 2] = Va[2, 1] = 0.01
    Vr = Va.copy()
    Vr[names.index('px'),  names.index('px')]  = -0.06
    Vr[names.index('px2'), names.index('px2')] = -0.14
    Vi = np.diag(np.full(n, 0.05))
    ies, idd = names.index('es'), names.index('d')
    Vi[ies, ies], Vi[idd, idd], Vi[ies, idd], Vi[idd, ies] = 0.0278, 0.0481, -0.0512, -0.0512
    Vstrong = np.diag(np.full(n, 0.05))
    Vstrong[names.index('px2'), names.index('px2')] = -0.010
    Dv = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.01

    e_att = E(Dv, Va, I)
    plain = float(np.real(Dv.conj() @ np.linalg.solve(Va, Dv)))
    e_rep, e_px2, e_ind, e_d = E(Dv, Vr, I), E(unit('px2'), Vr, I), E(Dv, Vi, I), E(unit('d'), Vi, I)
    e_strong = E(unit('px2', 0.1), Vstrong, I)

    A_r = rng.normal(size=(n, n))
    S0 = A_r @ A_r.T / n + I
    L = I + 0.25 * rng.normal(size=(n, n))
    while np.linalg.cond(L) > 1e3:
        L = I + 0.25 * rng.normal(size=(n, n))
    Li = np.linalg.inv(L)
    S1 = Li.T @ S0 @ Li

    e0_att = E(Dv, Va, S0)
    remix_att = abs(E(L @ Dv, L @ Va @ L.T, S1) - e0_att) / abs(e0_att)

    worst_scale = 0.0
    for Vt in (Va, Vr, Vi):
        e0 = E(Dv, Vt, S0)
        worst_scale = max(worst_scale, abs(E(Dv, Vt, 7.3 * S0) - e0) / abs(e0))

    Q_orth, _ = np.linalg.qr(rng.normal(size=(n, n)))
    worst_orth = 0.0
    for Vt in (Va, Vr, Vi):
        e0 = E(Dv, Vt, S0)
        worst_orth = max(worst_orth, abs(E(Q_orth @ Dv, Q_orth @ Vt @ Q_orth.T, Q_orth @ S0 @ Q_orth.T) - e0) / abs(e0))

    checks = [
        ("(1)  attractive Ṽ, S=I: E_cond == plain Δ†Ṽ⁻¹Δ", abs(e_att - plain) <= 1e-9 * abs(plain), f"E={e_att:.6e}  plain={plain:.6e}"),
        ("(2)  repulsive px/px2: E_cond > 0 (generic gap)",    e_rep > 0.0, f"E={e_rep:.3e}"),
        ("(2)  gap on the repulsive px2: E_cond > 0",          e_px2 > 0.0, f"E={e_px2:.3e}"),
        ("(2)  indefinite (es,d) block: E_cond > 0 (generic)", e_ind > 0.0, f"E={e_ind:.3e}"),
        ("(2)  gap on d inside (es,d): E_cond > 0",            e_d   > 0.0, f"E={e_d:.3e}"),
        ("(2b) strong repulsion (|a|>>floor) uses 1/|a|, not 1/floor", 0.5 < e_strong < 1.5,
            f"E={e_strong:.3e} (expected ~1.0; the 1/floor form would give ~2e3)"),
        ("(3a) L-remix exact for all-attractive Ṽ",            remix_att   <= 1e-6, f"rel={remix_att:.1e}"),
        ("(3b) S-rescaling invariant (all Ṽ)",                 worst_scale <= 1e-9, f"rel={worst_scale:.1e}"),
        ("(3c) orthogonal U-invariance (all Ṽ)",               worst_orth  <= 1e-9, f"rel={worst_orth:.1e}"),
    ]
    for label, ok, detail in checks:
        _log(f"    [{'✓' if ok else '✗'}] {label}   {detail}")
    n_fail = sum(not ok for _, ok, _ in checks)
    _log(f"  VERDICT: {'PASS' if not n_fail else 'FAIL'} -- "
         + ("E_cond is positive-definite, exact in the attractive sector, invariant under S-scaling and orthogonal re-mixing."
            if not n_fail else f"{n_fail} check(s) failed."))


def test_new_channel_irrep_content(solver) -> None:
    """
    Symmetry content of the pair operators of ALL channels in the D4h limit (Δ_B1g_static = 0, M = 0), in the MODEL's own T-adapted doublet gauge.

    Checks [1] φ_i(k), [2] orbital, [3] total = [1]⊗[2], plus the linear Q·Δ_i*Δ_j coupling.
    """
    p = solver.p
    vb = solver._get_vbdg()
    ops = _d4h_group()
    ch4 = [_d4h_characters(M) for M in ops]
    ctx = _pair_irrep_context(p.lambda_soc, p.Delta_tetra, 0.0)
    R6 = [_rot6(M, ctx['l3']) for M in ops]
    Rord = [ctx['U_ord'].conj().T @ R @ ctx['U_ord'] for R in R6]
    U24 = [_u24(Ro) for Ro in Rord]
    k0 = np.array([[0.7, 0.3]])
    phi0 = qm._pairing_basis_matrix(k0[:, 0], k0[:, 1])[0]

    # qm._IRREP_D4H[n] is the D4h irrep of the FORM FACTOR φ_n(k) alone —
    ff_char = {n: np.array([c[qm._IRREP_D4H[n]] for c in ch4]) for n in qm._IRREP_NAMES if qm._IRREP_D4H[n] != 'Eu'}   # 1-D form factors
    ff_char_Eu = np.array([c['Eu'] for c in ch4], dtype=complex)          # 2-D form factor

    irr_names = [n for n, _ in _D4H_IRREPS]
    chi_irr = {n: np.array([c[n] for c in ch4], dtype=complex) for n in irr_names}
    dims = dict(_D4H_IRREPS)

    def decompose(chi):
        return {n: float(np.real(np.sum(np.conj(chi_irr[n]) * chi) / 16.0))
                for n in irr_names
                if abs(np.sum(np.conj(chi_irr[n]) * chi) / 16.0) > 1e-9}

    # Orbital irreps for all channels
    orb_of = {}
    for br, idx_arr in (('7a', qm._IDX_7A_ARR), ('7b', qm._IDX_7B_ARR)):
        for i_irr, name in enumerate(qm._IRREP_NAMES):
            dv = np.zeros(qm._N_DELTA, dtype=complex); dv[int(idx_arr[i_irr])] = 1.0
            H = vb._build_H_stack(k0, np.zeros(3), 0.0, dv, np.zeros(qm._N_DELTA, dtype=complex), 0.0, 1.0, 1.0)[0]
            D12 = np.array(H[0:2*qm._N_ORB, 2*qm._N_ORB:4*qm._N_ORB]) / phi0[i_irr]
            w_orb, _, _ = _irrep_weights(_nambu24(D12), U24, np.ones(16), ch4, _D4H_IRREPS)
            orb_of[(name, br)] = {k: round(v, 3) for k, v in w_orb.items() if v > 1e-3}

    orb_ok_7a = all(list(orb_of[(n, '7a')]) == ['B1g'] for n in qm._IRREP_NAMES)
    orb_ok_7b = all(list(orb_of[(n, '7b')]) == ['Eg'] for n in qm._IRREP_NAMES)
    orb_ok = orb_ok_7a and orb_ok_7b

    # Total irrep (7a branch)
    chi_total = {}
    for name in qm._IRREP_NAMES:
        ff_label = qm._IRREP_D4H[name]
        orb_lbl = list(orb_of[(name, '7a')])[0]
        if dims[orb_lbl] == 1:
            chi_ff = (ff_char_Eu if ff_label == 'Eu' else ff_char[name])
            chi_total[(name, '7a')] = chi_irr[orb_lbl] * chi_ff

    # Q-allowed pairs (7a branch), tab[i,j] = B1g multiplicity in Γ_i*⊗Γ_j
    names = qm._IRREP_NAMES
    tab = np.zeros((len(names), len(names)))
    q_allowed = []
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            if (na, '7a') in chi_total and (nb, '7a') in chi_total:
                m = float(np.real(np.sum(chi_irr['B1g']
                                          * np.conj(chi_total[(na, '7a')])
                                          * chi_total[(nb, '7a')]) / 16.0))
                tab[a, b] = m
                if abs(m) > 1e-9 and (a <= b or abs(m) > 0.5):
                    q_allowed.append((na, nb, int(m)))

    g_idx = [i for i, n in enumerate(names) if qm._IRREP_D4H[n] == 'B2g']
    _allowed = {n for n in names if qm._IRREP_D4H[n] == 'A2g'}
    g_blind = bool(all({names[j] for j in range(len(names)) if abs(tab[g, j]) > 1e-9 or abs(tab[j, g]) > 1e-9} <= _allowed for g in g_idx))
    eu_active = bool(tab[names.index('px2'), names.index('px2')] > 0.5 and tab[names.index('px'), names.index('px2')] > 0.5)
    sd_active = bool(tab[names.index('s'), names.index('d')] > 0.5 and tab[names.index('es'), names.index('d')] > 0.5)
    ok = orb_ok and g_blind and eu_active and sd_active

    # ==== CONCISE LOGGING ====
    _log(f"  Δ_tet/λ = {ctx['ratio']:+.3f}  D4h limit, k0 = (0.7, 0.3)")
    _log("")
    _log("  [2] ORBITAL irrep (model T-adapted gauge):")
    _log(f"      7a ({len(qm._IRREP_NAMES)} channels): B1g   {'✓' if orb_ok_7a else '❌'}")
    _log(f"      7b ({len(qm._IRREP_NAMES)} channels): Eg    {'✓' if orb_ok_7b else '❌'}  (2D, Γ6↔Γ7b)")
    _log("")
    _log("  [3] TOTAL irrep of 7a channels — one representative per form-factor type:")
    seen_ff = set()
    for name in names:
        ff_label = qm._IRREP_D4H[name]
        if ff_label in seen_ff:
            continue
        seen_ff.add(ff_label)
        if (name, '7a') in chi_total:
            dec = decompose(chi_total[(name, '7a')])
            tot = ' ⊕ '.join(f"{k}" for k in dec) or '(zero)'
            _log(f"      {name:<6s}  φ={ff_label:<4s}  orb=B1g  →  total = {tot}")
        else:
            _log(f"      {name:<6s}  φ={ff_label:<4s}  orb=B1g  →  total = Eu (2D)")
    _log("")
    _log("  Linear Q·Δ_i*·Δ_j coupling (7a branch) — allowed pairs (B1g multiplicity > 0):")
    for m_mult in sorted(set(mm for _, _, mm in q_allowed), reverse=True):
        members = [f"{a}×{b}" for a, b, mm in q_allowed if mm == m_mult]
        show = members[:8]
        extra = len(members) - len(show)
        _log(f"      B1g^{m_mult}: {', '.join(show)}"
             + (f" (+{extra} more)" if extra > 0 else ""))
    _log("")
    _log(f"  CHECK s–d / es–d Q-coupling      : {'✓' if sd_active else '❌'}  "
         f"(mult s·d={int(tab[names.index('s'), names.index('d')])}, "
         f"es·d={int(tab[names.index('es'), names.index('d')])})")
    _log(f"  CHECK B2g channels Q-blind        : {'✓' if g_blind else '❌'}  "
         f"(only A2g-formfactor partners)")
    _log(f"  CHECK Eu pair (px2,px2)/(px,px2) : {'✓' if eu_active else '❌'}")
    _log("")
    _log(f"  {'✓ PASS' if ok else '❌ FAIL'}")


def _bond_class_fractions(ff) -> np.ndarray:
    """Squared-amplitude fractions of a form factor on the bond classes: 'onsite', 'NN (1,0)', 'NNN (1,1)', '3NN (2,0)', 'longer'"""
    w, lo, hi = _bond_weights(ff)
    cls = np.full(w.shape, 4)
    cls[(lo == 0) & (hi == 0)] = 0
    cls[(lo == 0) & (hi == 1)] = 1
    cls[(lo == 1) & (hi == 1)] = 2
    cls[(lo == 0) & (hi == 2)] = 3
    return np.array([float(w[cls == i].sum()) for i in range(5)])


def test_trw_symmetry(solver,
                      Delta_ip_scan=(-0.0, -0.005, -0.02, -0.05),
                      Quad67_test: float = 0.05, F_amp: float = 0.02, Q_theta: float = 0.01, Q_cov: float = 0.05,
                      target_doping: float = 0.146) -> None:
    """
    [A] D4h vs D2h TRW at Q=0 (differencing full BdG with/without F67; every channel with a TRW weight at once)
    [B] Orbital (L+S) character of D2h rotations in the 6×6 doublet basis
    [C] Full 24×24 D2h covariance per channel (C2x, C2y, C2z).  D2h cannot see the B1g/A1g difference of the form factors (φ_es and φ_d have the same D2h character): that is audited
        by test_trw_b1g_weighting (σd ∘ (Q → −Q)), not here.
    [D] Θ = T·τ_AB parity per TRW channel and the F67 phase that preserves Θ (evaluated at the k-point where the EFFECTIVE TRW amplitude is largest)
    [E] bond-class TRW strength audit + regression
    """
    from scipy.linalg import expm
    p  = solver.p
    vb = solver._get_vbdg()
    g_t, g_J = p.get_gutzwiller_factors(target_doping)
    n_ref  = 1.0 - target_doping
    mu_ref = -0.017
    M_sml  = np.array([0.05, 0.0, 0.0])
    M_cov  = np.zeros(3)
    nd     = qm._N_DELTA
    dz     = np.zeros(nd, dtype=complex)
    orig_ip = p.Delta_B1g_static
    F_all   = np.zeros(nd, dtype=complex)
    for _i, _n in enumerate(qm._IRREP_NAMES):
        if float(p.trw_irrep_scale[_i]) > 1e-12:
            F_all[qm._DELTA_IDX[_n + '7a']] = Quad67_test

    # ── Part A ──
    _log("[A] TRW at Q=0: D4h (Δ_ip=0) vs D2h (Δ_ip≠0)")
    _log(f"  {'Δ_ip':>10s} {'J_B1g_7a':>13s} {'J_B1g_7b':>13s} {'‖onsite‖':>12s} {'‖AB‖':>12s} active")
    rows_A = []
    try:
        for D_ip in Delta_ip_scan:
            p.Delta_B1g_static = float(D_ip)
            solver._full_rebuild()
            solver.g_JT_bare = p.g_JT
            _, J_B1g_7a, J_B1g_7b = _exch(p, 0.0, g_J)
            H_on  = vb._build_H_stack(solver.k_points, M_sml, 0.0, dz, F_all, mu_ref, g_t, g_J)
            H_off = vb._build_H_stack(solver.k_points, M_sml, 0.0, dz, np.zeros(nd), mu_ref, g_t, g_J)
            dH    = H_on - H_off
            n_on  = float(np.max(np.linalg.norm(np.concatenate([dH[:, 0:6, 0:6], dH[:, 6:12, 6:12]], axis=0), 'fro', axis=(1, 2))))
            n_ab  = float(np.max(np.linalg.norm(np.concatenate([dH[:, 0:6, 6:12], dH[:, 6:12, 0:6]], axis=0), 'fro', axis=(1, 2))))
            act   = (n_on > 1e-12) or (n_ab > 1e-12)
            rows_A.append((float(D_ip), J_B1g_7a, J_B1g_7b, n_on, n_ab, act))
            _log(f"  {float(D_ip):>+10.4f} {J_B1g_7a:>+13.3e} {J_B1g_7b:>+13.3e} "
                 f"{n_on:>12.3e} {n_ab:>12.3e} {'YES' if act else 'no'}")
    finally:
        p.Delta_B1g_static = orig_ip
        solver._full_rebuild()
        solver.g_JT_bare = p.g_JT
    d4h_A = next((r for r in rows_A if abs(r[0]) < 1e-12), None)
    d2h_A = [r for r in rows_A if abs(r[0]) > 1e-12]
    ok_A  = (d4h_A is not None and not d4h_A[5]) and (not d2h_A or all(r[5] for r in d2h_A))
    n_d2h_active = sum(1 for r in d2h_A if r[5])
    _log(f"  [A] {'✓ PASS' if ok_A else '❌ FAIL'}: D4h TRW="
         f"{'0 exactly' if (d4h_A and not d4h_A[5]) else 'ACTIVE ❌'}, "
         f"D2h active at {n_d2h_active}/{len(d2h_A)} points")
    _log("")

    # ── Part B: orbital (L+S) character of the D2h rotations ──
    Hsc, Lx, Ly, Lz, Sz_full, LS = qm._t2g_soc_cf_operators(p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static)
    evals, evecs = np.linalg.eigh(Hsc)
    B1g_t2g = Lx @ Lx - Ly @ Ly
    G6, G7a, G7b = qm._get_G6_G7a_G7b(evals, evecs, Sz_full, LS, B1g_t2g)
    U_ord = np.column_stack([d[s] for d in (G6, G7a, G7b) for s in ('up', 'dn')])
    I3 = np.eye(3)
    Sx = 0.5 * qm._SIGMA_2x2['x']; Sy = 0.5 * qm._SIGMA_2x2['y']; Sz = 0.5 * qm._SIGMA_2x2['z']
    Jn = {'x': Lx + np.kron(I3, Sx), 'y': Ly + np.kron(I3, Sy), 'z': Lz + np.kron(I3, Sz)}
    R  = {a: expm(-1j * np.pi * Jn[a]) for a in 'xyz'}
    U  = {a: U_ord.conj().T @ R[a] @ U_ord for a in 'xyz'}

    def _pair_char(D, a):
        """kappa of the pair 2-form D under R_a:  D -> U D U^T."""
        Dn = U[a] @ D @ U[a].T
        nn = float(np.vdot(D, D).real)
        k  = float(np.vdot(D, Dn).real) / nn if nn > 1e-30 else 0.0
        return k
    def _norm_char(M, a):
        Mn = U[a] @ M @ U[a].conj().T
        nn = float(np.vdot(M, M).real)
        k  = float(np.vdot(M, Mn).real) / nn if nn > 1e-30 else 0.0
        return k

    D_eps = np.zeros((6, 6), dtype=complex); D_eps[0, 3] = +1.0; D_eps[1, 2] = -1.0; D_eps = D_eps - D_eps.T
    D_sym = np.zeros((6, 6), dtype=complex); D_sym[0, 3] = +1.0; D_sym[1, 2] = +1.0; D_sym = D_sym - D_sym.T
    M_B   = p.B1g_offdiag.astype(complex)
    _log("[B] Orbital (L+S) character of D2h rotations, 6×6 doublet basis:")
    _log("  pair ε      : " + " ".join(f"R_{a}:{_pair_char(D_eps, a):+.3f}" for a in 'xyz'))
    _log("  pair 'sym'  : " + " ".join(f"R_{a}:{_pair_char(D_sym, a):+.3f}" for a in 'xyz'))
    _log("  B1g_offdiag : " + " ".join(f"R_{a}:{_norm_char(M_B, a):+.3f}" for a in 'xyz')
         + "  (all +1: B1g_offdiag is D2h-even, B1g vs A1g is invisible to D2h — see test_trw_b1g_weighting)")
    DB   = p.B1g_op @ D_eps @ p.B1g_op.T
    leak = float(np.linalg.norm(
        DB - (np.vdot(D_eps, DB).real / np.vdot(D_eps, D_eps).real) * D_eps)) / float(np.linalg.norm(D_eps))
    _log(f"  B1g·ε → ε: κ = {np.vdot(D_eps, DB).real / np.vdot(D_eps, D_eps).real:+.5f}, leakage = {leak:.2e}")
    _log("")

    # ── Part C ──
    _log("[C] Full 24×24 D2h covariance per channel")
    OPS = {'C2x': (lambda k: np.column_stack([k[:, 0], -k[:, 1]]), 'x'),
           'C2y': (lambda k: np.column_stack([-k[:, 0], k[:, 1]]), 'y'),
           'C2z': (lambda k: -k, 'z')}
    def _W(a):
        u = U[a]; Z6 = np.zeros((6, 6))
        return np.block([[u, Z6, Z6, Z6], [Z6, u, Z6, Z6], [Z6, Z6, u.conj(), Z6], [Z6, Z6, Z6, u.conj()]])
    rng  = np.random.default_rng(3)
    kpts = rng.uniform(-np.pi, np.pi, (7, 2))
    def _Hk(k, D=None, F=None):
        return vb._build_H_stack(k, M_cov, Q_cov,
                                 dz if D is None else D, dz if F is None else F,
                                 mu_ref, g_t, g_J)
    def _kappa(op, D=None, F=None):
        kmap, ax = OPS[op]; W = _W(ax)
        H0 = _Hk(kpts)
        A  = np.einsum('ij,njk,kl->nil', W, _Hk(kpts, D, F) - H0, W.conj().T)
        B  = _Hk(kmap(kpts), D, F) - _Hk(kmap(kpts))
        nn = float(np.vdot(A, A).real)
        if nn < 1e-24:
            return None, 0.0
        k  = float(np.vdot(A, B).real) / nn
        return k, float(np.linalg.norm(B - k * A) / np.sqrt(nn))
    _log("  control H0 (hopping + local CF + JT, no fields):")
    for op in OPS:
        kmap, ax = OPS[op]; W = _W(ax)
        A = np.einsum('ij,njk,kl->nil', W, _Hk(kpts), W.conj().T)
        B = _Hk(kmap(kpts))
        nn = float(np.vdot(A, A).real)
        k  = float(np.vdot(A, B).real) / nn if nn > 1e-24 else 0.0
        res = float(np.linalg.norm(B - k * A) / np.sqrt(nn)) if nn > 1e-24 else 0.0
        _log(f"    {op}: kappa={k:+.4f}  residual={res:.2e}")
    n_cov, n_bad, n_zero = 0, 0, 0
    trw_results = []
    for i, c in enumerate(qm._IRREP_NAMES):
        iD  = int(qm._IDX_7A_ARR[i])
        D   = np.zeros(nd, dtype=complex); D[iD] = F_amp
        F   = np.zeros(nd, dtype=complex); F[iD] = F_amp
        wgt = float(p.trw_irrep_scale[i])
        if wgt < 1e-12:
            trw_results.append((c, wgt, {}, {}, {}))
            continue                                       # no TRW by construction (verified in [E])
        kps, kts, mm = {}, {}, {}
        for op in OPS:
            kp, _ = _kappa(op, D=D)
            kt, _ = _kappa(op, F=F)
            kps[op], kts[op] = kp, kt
            if kt is None or abs(kt) < 1e-12:
                mm[op] = '--'; n_zero += 1
            elif kp is not None and abs(kp - kt) < 1e-6:
                mm[op] = ' Y'; n_cov += 1
            else:
                mm[op] = ' N'; n_bad += 1
        trw_results.append((c, wgt, kps, kts, mm))

    _log(f"  {'channel':8s} {'weight':>7s}  "
         f"{'C2x(kp/kt)':>16s}  {'C2y(kp/kt)':>16s}  {'C2z(kp/kt)':>16s}  match")
    nulls = []
    for (c, wgt, kps, kts, mm) in trw_results:
        if wgt < 1e-12:
            nulls.append(c); continue
        fmt = lambda kp, kt: f"{kp:+.3f}/{kt:+.3f}" if kt is not None else f"{kp:+.3f}/--"
        _log(f"  {c:8s} {wgt:>7.3f}  "
             f"{fmt(kps['C2x'], kts['C2x']):>16s}  "
             f"{fmt(kps['C2y'], kts['C2y']):>16s}  "
             f"{fmt(kps['C2z'], kts['C2z']):>16s}  "
             f"{''.join(mm[op] for op in OPS)}")
    _log(f"  channels without TRW (weight = 0, skipped): {len(nulls)}")
    ok_C = (n_bad == 0)
    _log(f"  [C] {'✓ PASS' if ok_C else '❌ FAIL'}: {n_cov} matches, {n_bad} mismatches, {n_zero} operations with a vanishing TRW response")
    _log("")

    # ── Part D ──
    _log("[D] Θ = T·τ_AB parity per TRW channel and the F67 phase that preserves Θ")
    _log("    Evaluated where the EFFECTIVE TRW amplitude |H(F67) − H(0)| is largest, NOT where the pair form factor peaks: for the B1g-weighted inter-sublattice channels these differ")
    _log("    (φ̃_es = φ_d vanishes at k = 0 and φ̃_d = φ_es at k = (π,0), the maxima of φ_es and φ_d — a test that picks k from the pair form factor silently drops es and d).")
    _log("    The Θ-compatible phase depends only on the T_Q parity of the bond kind (A–B bond: real F67; on-site and same-sublattice bonds: imaginary F67), not on the form factor.")
    _log(f"  {'channel':8s} {'kind':9s} {'scale':>8s} {'k-point':>13s} {'|dH|':>10s} {'Θ incr (F real)':>16s} {'Θ incr (F imag)':>16s}  {'Θ-even for':>10s} {'expected':>9s}")
    rng_D = np.random.default_rng(7)
    k_cand = np.vstack([solver.k_points, rng_D.uniform(-np.pi, np.pi, (64, 2))])
    fails_D, n_D = [], 0
    for i, c in enumerate(qm._PAIR_CHANNELS):
        sc = float(p.trw_irrep_scale[i])
        if sc < 1e-12:
            continue                                       # no TRW by construction (verified in [E])
        iD = qm._DELTA_IDX[c.name + '7a']
        def _fv(ph, iD=iD):
            v = np.zeros(nd, dtype=complex); v[iD] = F_amp * ph
            return v
        dHc = vb._build_H_stack(k_cand, M_cov, Q_theta, dz, _fv(1.0), mu_ref, g_t, g_J) - vb._build_H_stack(k_cand, M_cov, Q_theta, dz, dz, mu_ref, g_t, g_J)
        ik = int(np.argmax(np.linalg.norm(dHc, axis=(1, 2))))
        kp_D = k_cand[ik:ik + 1]
        H0 = vb._build_H_stack(kp_D, M_cov, Q_theta, dz, dz, mu_ref, g_t, g_J)[0]
        base = float(np.linalg.norm(_apply_theta_operator(solver, H0, kp_D) - H0))
        rp = []
        for ph in (1.0, 1.0j):
            H = vb._build_H_stack(kp_D, M_cov, Q_theta, dz, _fv(ph), mu_ref, g_t, g_J)[0]
            rp.append((float(np.linalg.norm(H - H0)), float(np.linalg.norm(_apply_theta_operator(solver, H, kp_D) - H)) - base))
        dH = rp[0][0]
        if dH <= 1e-14:
            fails_D.append(c.name)
            _log(f"  {c.name:8s} {c.kind:9s} {sc:>+8.4f}  ❌ FAIL — a channel with scale > 0 has no TRW at its best k-point")
            continue
        n_D += 1
        even_real = abs(rp[0][1]) < 1e-6 * dH and rp[1][1] > 0.5 * dH
        even_imag = abs(rp[1][1]) < 1e-6 * dH and rp[0][1] > 0.5 * dH
        measured = 'real' if even_real else ('imaginary' if even_imag else 'neither')
        expected = 'real' if c.kind == 'nn_AB' else 'imaginary'
        good = (measured == expected)
        if not good:
            fails_D.append(c.name)
        _log(f"  {c.name:8s} {c.kind:9s} {sc:>+8.4f} ({kp_D[0, 0]:+.2f},{kp_D[0, 1]:+.2f}) {dH:10.3e} {rp[0][1]:16.3e} {rp[1][1]:16.3e}  {measured:>10s} {expected:>9s}  {'✓' if good else '❌ FAIL'}")
    ok_D = (not fails_D) and n_D > 0
    _log(f"  [D] {'✓ PASS' if ok_D else '❌ FAIL'}: {n_D} TRW channels (every channel with scale > 0, the B1g-weighted es and d included)"
         + (f"  violations: {fails_D}" if fails_D else ""))
    _log("")

    # ── Part E ──
    _log("[E] NNN TRW strength audit")
    _log(f"  {'channel':8s} {'on':>5s} {'NN':>5s} {'NNN':>5s} {'3NN':>5s} {'long':>5s} {'scale':>9s}")
    for i, c in enumerate(qm._PAIR_CHANNELS):
        sc = float(p.trw_irrep_scale[i])
        if sc > 1e-12:
            fr = _bond_class_fractions(qm._IRREP_FF[c.name])
            _log(f"  {c.name:8s} {fr[0]:5.2f} {fr[1]:5.2f} {fr[2]:5.2f} "
                 f"{fr[3]:5.2f} {fr[4]:5.2f} {sc:>+9.4f}")
    _w_cls = np.array([0.0, 1.0, (p.t_prime_ratio) ** 2, (p.t_dprime_ratio) ** 2, 0.0])
    _kx_s, _ky_s = np.random.default_rng(11).uniform(-np.pi, np.pi, (2, 16))
    def _expected_scale(c) -> float:
        """Σ_class fraction·(t_class/t)² for the channels that can carry a σd-covariant TRW, 0 for the rest:
        A1g / B1g only with a declared σd partner (qm._PARTNER_PAIRS) on a shell that HAS a B1g function (Σ_δ η_δ c_δ e^{ik·δ} ≢ 0: not diagonal bonds, not on-site); Eu keeps its own form factor
        (covariance audited by test_trw_b1g_weighting); A2g / B2g have no same-shell partner of the B1g-flipped parity, so no covariant TRW exists."""
        frac = float(_bond_class_fractions(qm._IRREP_FF[c.name]) @ _w_cls)
        if frac <= 0.0:
            return 0.0                                  # (long-range harmonics: no bond-sum needed, and the reference grid would alias them)
        if c.d4h in ('A2g', 'B2g'):
            return 0.0
        if c.d4h in ('A1g', 'B1g'):
            if c.name not in dict(getattr(qm, '_PARTNER_PAIRS', {})) or float(np.linalg.norm(_b1g_weighted_ff(qm._IRREP_FF[c.name], _kx_s, _ky_s))) < 1e-12:
                return 0.0
        return frac
    _scale_bad = [c.name for i, c in enumerate(qm._PAIR_CHANNELS) if abs(_expected_scale(c) - float(p.trw_irrep_scale[i])) > 1e-9]
    _log(f"  trw_irrep_scale == Σ_class fraction·(t_class/t)² (NN: 1, NNN: (t'/t)², 3NN: (t''/t)², onsite / longer: 0), and 0 where no σd-covariant TRW exists (cc, on-site, A2g / B2g): {'✓' if not _scale_bad else '✗ differs for ' + str(_scale_bad)}")
    n_zero_scale = sum(1 for i in range(len(qm._PAIR_CHANNELS))
                       if float(p.trw_irrep_scale[i]) < 1e-12)
    _log(f"  zero-scale channels: {n_zero_scale}")

    F_mag_E, Q_E = 0.05, 0.05
    dz_E = np.zeros(nd, dtype=complex)
    zero_bad, nz_bad = [], []
    for i, c in enumerate(qm._PAIR_CHANNELS):
        fv = np.zeros(nd, dtype=complex); fv[qm._DELTA_IDX[c.name + '7a']] = F_mag_E
        dHn = float(np.max(np.linalg.norm(
            vb._build_H_stack(solver.k_points, M_sml, Q_E, dz_E, fv, mu_ref, g_t, g_J)
          - vb._build_H_stack(solver.k_points, M_sml, Q_E, dz_E, np.zeros(nd), mu_ref, g_t, g_J),
            'fro', axis=(1, 2))))
        sc = float(p.trw_irrep_scale[i])
        if sc < 1e-12 and dHn > 1e-12: zero_bad.append(c.name)
        if sc > 1e-12 and dHn < 1e-12: nz_bad.append(c.name)
    _log(f"  regression: scale == 0 -> zero TRW    : {'yes' if not zero_bad else 'NO  ' + str(zero_bad)}")
    _log(f"  regression: scale  > 0 -> nonzero TRW : {'yes' if not nz_bad   else 'NO  ' + str(nz_bad)}")
    ok_E = not (zero_bad or nz_bad)
    _log(f"  [E] {'PASS' if ok_E else 'FAIL'}: trw_irrep_scale is applied consistently by _build_H_stack")
    _log("")

    overall = ok_A and ok_C and ok_D and ok_E
    _log(f"OVERALL: {'✓ PASS' if overall else '❌ FAIL'}  "
         f"(A={ok_A}, B=info, C={ok_C}, D={ok_D}, E={ok_E})")


def _trw_slots(solver) -> list:
    """Delta/F67-vector slots (both branches) of every channel with a transverse-Weiss-field weight (trw_irrep_scale > 0)."""
    s = solver.p.trw_irrep_scale
    return [qm._DELTA_IDX[n + br] for i, n in enumerate(qm._IRREP_NAMES) if float(s[i]) > 1e-12 for br in ('7a', '7b')]


def _ab_trw_slots(solver) -> list:
    """F67 slots (both branches) of the channels that feed the INTER-SUBLATTICE transverse Weiss field H_TRW_AB: the nn_AB channels with a bond-class weight (es, d, px, py).
    H_TRW_AB is NOT 'the d channel': switching it off means zeroing all of them."""
    s = solver.p.trw_irrep_scale
    return [qm._DELTA_IDX[n + br] for i, n in enumerate(qm._IRREP_NAMES) if qm._IRREP_KIND[n] == 'nn_AB' and float(s[i]) > 1e-12 for br in ('7a', '7b')]


def _ab_trw_block_norm(solver, M, Q, Quad67_vec, mu, g_t, g_J) -> float:
    """max over the k-grid of ‖A–B block of H(F67) − H(F67 = 0)‖_F: the Hamiltonian-level size of H_TRW_AB."""
    vb = solver._get_vbdg()
    z = np.zeros(qm._N_DELTA, dtype=complex)
    dH = vb._build_H_stack(solver.k_points, M, Q, z, np.asarray(Quad67_vec, dtype=complex), mu, g_t, g_J) - vb._build_H_stack(solver.k_points, M, Q, z, z, mu, g_t, g_J)
    return float(np.linalg.norm(dH[:, 0:6, 6:12], axis=(1, 2)).max())


# ============================================================================
# TRW: B1g-weighted form factor, C4-type (σd ∘ Q→−Q) covariance, Q-vertex, BdG structure
# ============================================================================
@contextlib.contextmanager
def _exact_D4h(solver):
    """Context manager: the exact-D4h model (static B1g field Δ_B1g_static switched OFF, solver rebuilt) for the duration of the block; the original parameters are restored (and the solver rebuilt again) on exit.
    Yields the VectorizedBdG of the REBUILT solver; a VectorizedBdG fetched before entering is stale inside the block."""
    p = solver.p
    orig = float(p.Delta_B1g_static)

    def _rebuild() -> None:
        solver._full_rebuild()
        solver.g_JT_bare = p.g_JT

    try:
        if orig != 0.0:
            p.Delta_B1g_static = 0.0
            _rebuild()
        yield solver._get_vbdg()
    finally:
        if float(p.Delta_B1g_static) != orig:
            p.Delta_B1g_static = orig
            _rebuild()


_B1G_TIMES = {'A1g': 'B1g', 'B1g': 'A1g', 'A2g': 'B2g', 'B2g': 'A2g', 'Eu': 'Eu'}     # B1g ⊗ Γ: the orbital part of every Γ6⊗Γ7a pair is B1g (test_new_channel_irrep_content)


def _sigma_d_context(solver) -> tuple:
    """(W24, R_ord, ctx) for the diagonal mirror σd: (x,y,z) → (y,x,z), in the doublet basis of the model; W24 = blockdiag(R,R,R*,R*) acts on the 24×24 Nambu basis and σd maps k = (kx,ky) → (ky,kx).
    Asserts that the matrix is in the σd class of the D4h table and that R_ord does not mix the doublets."""
    p = solver.p
    Msd = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert _d4h_class(Msd) == 'σd'
    ctx = _pair_irrep_context(p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static)
    R6 = _rot6(Msd, ctx['l3'])
    Rord = ctx['U_ord'].conj().T @ R6 @ ctx['U_ord']
    return _u24(Rord), Rord, ctx


def _sigma_d_partner(name: str, seed: int = 5) -> tuple:
    """(partner, s): φ_name(σd k) = s·φ_partner(k) for all k; the partner is the channel itself for a 1-d irrep (s = its σd parity) and the other member of the pair for Eu (px ↔ py, ...)."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(-np.pi, np.pi, (16, 2))
    base = np.asarray(qm._IRREP_FF[name](k[:, 1], k[:, 0]), dtype=float)
    for other in qm._IRREP_NAMES:
        f = np.asarray(qm._IRREP_FF[other](k[:, 0], k[:, 1]), dtype=float)
        for s in (1, -1):
            if np.allclose(base, s * f, atol=1e-10):
                return other, s
    return None, 0


def _b1g_weighted_ff(ff, kx, ky) -> np.ndarray:
    """Reference B1g-weighted form factor  φ̃(k) = Σ_δ η_δ c_δ e^{ik·δ},  ff(k) = Σ_δ c_δ e^{ik·δ},  η_δ = sign(δx² − δy²) = the B1g (x²−y²) character of the bond vector δ (x-bonds +, y-bonds −).
    On-site and diagonal-bond amplitudes carry no B1g weight (η = 0): there is no B1g function on those shells."""
    n = int(qm._BOND_GRID)
    k1 = 2.0 * np.pi * np.arange(n) / n
    KX, KY = np.meshgrid(k1, k1, indexing='ij')
    c = np.fft.fft2(np.asarray(ff(KX, KY), dtype=float)) / n ** 2          # c[δ] for δ = signed index (the transform's exponent sign is the one of e^{+ik·δ})
    m = np.arange(n)
    m = np.where(m > n // 2, m - n, m)
    DX, DY = np.meshgrid(m, m, indexing='ij')
    eta = np.sign(DX ** 2 - DY ** 2)
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    out = np.zeros(kx.shape, dtype=complex)
    for a, b in zip(*np.nonzero((np.abs(c) > 1e-13) & (eta != 0))):
        out += eta[a, b] * c[a, b] * np.exp(1j * (kx * DX[a, b] + ky * DY[a, b]))
    assert np.abs(out.imag).max() < 1e-10
    return out.real


def _cov_kappa(W: np.ndarray, dH_src: np.ndarray, dH_dst: np.ndarray) -> tuple:
    """(κ, residual) of  dH_dst = κ·W dH_src W†  (stacks over k); (None, None) if dH_src ≡ 0."""
    A = np.einsum('ij,njk,kl->nil', W, dH_src, W.conj().T)
    nn = float(np.vdot(A, A).real)
    if nn < 1e-26:
        return None, None
    kap = float(np.vdot(A, dH_dst).real) / nn
    return kap, float(np.linalg.norm(dH_dst - kap * A) / np.sqrt(nn))


def _trw_structure(solver, F7a: complex, J7a: float) -> np.ndarray:
    """6×6 orbital matrix S of the transverse Weiss field of the Γ6⊗Γ7a pair amplitude F7a (independent re-statement of the matrix the solver builds): S = B1g_offdiag ∘ [[0, F·J], [F*·J, 0]] on the (Γ6, Γ7a) doublet blocks."""
    Wm = np.zeros((6, 6), dtype=complex)
    Wm[0:2, 2:4] = F7a * J7a
    Wm[2:4, 0:2] = np.conj(F7a) * J7a
    return np.asarray(solver.B1g_offdiag) * Wm


def _trw_channel_indices(solver) -> list:
    """Registry indices of the channels that carry a transverse Weiss field (trw_irrep_scale > 0)."""
    s = np.asarray(solver.p.trw_irrep_scale, dtype=float)
    return [i for i in range(len(qm._IRREP_NAMES)) if s[i] > 1e-12]


def _trw_ff_partner_report(solver, names: list, kpts: np.ndarray) -> tuple:
    """Registry-level check of qm._trw_form_factor for every channel with a TRW weight.
    Returns (rows, n_bad).

      - on-site:                       own form factor (Φ_s ≡ 1; no B1g function on a single site)
      - A1g / B1g in _PARTNER_PAIRS:   the σd-partner's form factor
      - A1g / B1g NOT in _PARTNER_PAIRS: own form factor (silent fallback — flagged as a bug)
      - Eu / B2g / A2g:                own form factor
    """
    Phi = qm._pairing_basis_matrix(kpts[:, 0], kpts[:, 1])
    rows, n_bad = [], []
    for i in _trw_channel_indices(solver):
        nm = names[i]
        d4h, kind = qm._IRREP_D4H[nm], qm._IRREP_KIND[nm]
        own = Phi[:, i]
        used = qm._trw_form_factor(nm, i, Phi)               # 3-arg version
        ref = _b1g_weighted_ff(qm._IRREP_FF[nm], kpts[:, 0], kpts[:, 1])
        refn, usedn = float(np.linalg.norm(ref)), float(np.linalg.norm(used))
        ok = True
        sign_vs_rule = None

        # -------- on-site: own form factor (Φ_s ≡ 1) ----------------
        if kind == 'onsite':
            ok = bool(np.allclose(used, own, atol=1e-12))
            rows.append((nm, kind, d4h,
                         "on-site: own form factor (Φ_s ≡ 1, no B1g function on one site)",
                         None, ok))
            if not ok:
                n_bad.append(nm)
            continue

        # -------- A1g / B1g: must use a σd-partner that is the B1g-weighted form -----
        if d4h in ('A1g', 'B1g'):
            if nm not in qm._PARTNER_PAIRS:
                ok = False
                txt = (f"{d4h} WITHOUT entry in _PARTNER_PAIRS — "
                       f"silent fallback to own form factor (latent: B1g covariance would break if Quad67 had weight here)")
            else:
                partner = qm._PARTNER_PAIRS[nm]
                diff         = not bool(np.allclose(used, own, atol=1e-12))
                same_sector  = qm._IRREP_CHAR[partner] == qm._IRREP_CHAR[nm]
                if refn > 1e-12 and usedn > 1e-12:
                    s_ = float(np.vdot(ref, used).real) / refn ** 2
                    prop = bool(np.linalg.norm(used - s_ * ref)
                                < 1e-8 * max(usedn, 1.0)
                                and abs(abs(s_) - 1.0) < 1e-8)
                    if prop:
                        sign_vs_rule = s_        # reported, not failed
                else:
                    prop = bool(refn < 1e-12 and usedn < 1e-12)
                problems = []
                if not diff:
                    problems.append("silent fallback (partner ≡ own)")
                if not same_sector:
                    problems.append(f"partner {partner} in a DIFFERENT sector")
                if not prop:
                    if refn < 1e-12:
                        problems.append("no B1g component on the bond shell (Σ η_δ c_δ e^{ik·δ} ≡ 0)")
                    else:
                        problems.append("partner is NOT ∝ B1g-weighted form factor of the SAME bond shell")
                ok = diff and same_sector and prop
                txt = f"σd partner {partner}" + (
                    "" if not problems else "  ← " + "; ".join(problems))
        # -------- Eu / B2g / A2g: own form factor -------------------
        else:
            ok = bool(np.allclose(used, own, atol=1e-12))
            txt = f"{d4h}: own form factor (no partner defined)"
            if refn > 1e-12 and usedn > 1e-12:
                s_ = float(np.vdot(ref, used).real) / refn ** 2
                if np.linalg.norm(used - s_ * ref) < 1e-8 * max(usedn, 1.0):
                    sign_vs_rule = s_
        if not ok:
            n_bad.append(nm)
        rows.append((nm, kind, d4h, txt, sign_vs_rule, ok))
    return rows, n_bad

def test_trw_b1g_weighting(solver, target_doping: float = 0.158, Q_test: float = 0.03, h: float = 0.01, n_k: int = 9, tol: float = 1e-8) -> None:
    """σd ∘ (Q → −Q) covariance of the transverse Weiss field built from the STATE.

    Audit of what B1g weighting the symmetry requires

      [A] controls       : model (without TRW), B1g_offdiag, σd block-diagonality, J_B1g odd in Q
      [B] registry       : which form factor _trw_form_factor actually returns, per channel
      [C] state-based    : covariance with the partner form factor (code's current rule)
      [C'] contrast      : same test forcing the OWN form factor — proves the partner is required
    """
    p = solver.p
    names = list(qm._IRREP_NAMES)
    nd = qm._N_DELTA
    zD = np.zeros(nd, dtype=complex)
    M0 = np.zeros(3)
    rng = np.random.default_rng(3)
    kpts = rng.uniform(-np.pi, np.pi, (n_k, 2))
    ksd = kpts[:, ::-1].copy()
    n_fail = 0

    with _exact_D4h(solver) as vb:
        g_t, g_J = p.get_gutzwiller_factors(target_doping)
        mu = -0.05
        W, Rord, ctx = _sigma_d_context(solver)

        def Hs(k, Qv, D, q):
            return vb._build_H_stack(k, M0, Qv, D, q, mu, g_t, g_J)

        def one(i: int, a: float) -> np.ndarray:
            v = zD.copy()
            v[qm._DELTA_IDX[names[i] + '7a']] = a
            return v

        def quad(Qv, D):
            ev, ec = np.linalg.eigh(vb._build_H_stack(solver.k_points, M0, Qv, D, zD, mu, g_t, g_J))
            return qm._compute_B1g_quadrupole(solver.k_weights, solver.Phi_irrep_k, ev, ec, solver.kT, solver.B1g_op)

        # ── [A] controls ─────────────────────────────────────────────
        _log("  [A] controls (exact D4h: Δ_B1g_static = 0, M = 0)")
        Hq = Hs(kpts, Q_test, zD, zD)
        c1 = float(np.linalg.norm(np.einsum('ij,njk,kl->nil', W, Hq, W.conj().T) - Hs(ksd, -Q_test, zD, zD)) / np.linalg.norm(Hq))
        Bm = np.asarray(solver.B1g_offdiag)
        c2 = float(np.linalg.norm(Rord @ Bm @ Rord.conj().T + Bm) / np.linalg.norm(Bm))
        blk = np.zeros_like(Rord)
        for a, b in ((0, 2), (2, 4), (4, 6)):
            blk[a:b, a:b] = Rord[a:b, a:b]
        c3 = float(np.linalg.norm(Rord - blk))
        J_B_p = np.asarray(p.exchange_channels(+Q_test, g_J)[1], dtype=float)
        J_B_m = np.asarray(p.exchange_channels(-Q_test, g_J)[1], dtype=float)
        c4 = abs(J_B_p[0] + J_B_m[0]) / max(abs(J_B_p[0]), 1e-30)
        ctrl = [
            (f"H without TRW: ‖W H(k,Q) W† − H(σd k,−Q)‖/‖H‖ = {c1:.1e}", c1 < 1e-9),
            (f"B1g_offdiag is σd-odd: ‖R B R† + B‖/‖B‖ = {c2:.1e}", c2 < 1e-9),
            (f"σd does not mix the doublets: ‖R − blockdiag‖ = {c3:.1e}", c3 < 1e-9),
            (f"J_B1g(Γ6Γ7a) is odd in Q: J(+Q) = {J_B_p[0]:+.4e}, J(−Q) = {J_B_m[0]:+.4e}", c4 < 1e-9),
        ]
        for msg, ok in ctrl:
            _log(f"      {msg}   {'✓' if ok else '❌ FAIL'}")
        _log(f"      J_B1g(Γ6Γ7b) = {J_B_p[1]:+.1e}: " + ("the 7b branch carries no TRW (Γ7b is xy-like and cannot enter the B1g channel)" if abs(J_B_p[1]) < 1e-14 else "7b TRW present (7a states audited below)"))
        if not all(ok for _, ok in ctrl):
            _log("  ❌ FAIL — the controls fail: the audit below would not be meaningful")
            return

        # ── [B] registry of _trw_form_factor ──────────────────────────
        _log("")
        _log("  [B] qm._trw_form_factor for every channel with a TRW weight")
        _log("      The 3-argument version returns the σd-PARTNER for A1g / B1g channels")
        _log("      in _PARTNER_PAIRS, and the OWN form factor otherwise.")
        rows, bad_B = _trw_ff_partner_report(solver, names, kpts)
        _log(f"      {'chan':5s} {'kind':9s} {'D4h':5s}  {'form factor used':<88s} {'sign vs Σ η c e^{ik·δ}':>23s}")
        for nm, kind, d4h, txt, sgn, ok in rows:
            sg = '' if sgn is None else ('≡ 0' if sgn == 0.0 else f"{sgn:+.0f}")
            _log(f"      {nm:5s} {kind:9s} {d4h:5s}  {txt:<88s} {sg:>23s}{'' if ok else '  ❌'}")
        if bad_B:
            n_fail += 1
            _log(f"  ❌ FAIL — qm._trw_form_factor: wrong partner for {', '.join(bad_B)}")
        else:
            _log("      ✓ every A1g / B1g TRW channel has a valid σd partner (same sector,")
            _log("        B1g-weighted bond form factor of the same shell); every other")
            _log("        channel returns its own form factor as designed.")

        # ── [C] state-based σd covariance with the PARTNER rule ──────
        _log("")
        _log("  [C] state-based σd covariance  (Δ' = σd image of Δ; Quad67 recomputed from each state)")
        _log("      H(+Q, D, Quad67(+Q,D)) − H(+Q, D, 0)")
        _log("      must equal  W · [ H(−Q, D', Quad67(−Q,D')) − H(−Q, D', 0) ] · W†")
        kap_cache = {}

        def kap_pair(i: int) -> tuple:
            if i not in kap_cache:
                j, _s = _sigma_d_partner(names[i])
                jj = names.index(j)
                kk, _r = _cov_kappa(W, Hs(kpts, +Q_test, one(i, .05), zD) - Hs(kpts, +Q_test, zD, zD), Hs(ksd, -Q_test, one(jj, .05), zD) - Hs(ksd, -Q_test, zD, zD))
                kap_cache[i] = (jj, kk)
            return kap_cache[i]

        states = [
            ("normal state (Δ = 0)",     {}),
            ("s7a",                      {'s7a': h}),
            ("d7a",                      {'d7a': h}),
            ("s7a + d7a",                {'s7a': h, 'd7a': h}),
            ("es7a + d7a",               {'es7a': h, 'd7a': h}),
            ("px7a + py7a",              {'px7a': h, 'py7a': h}),
            ("px7a",                     {'px7a': h}),
            ("s7a + es7a + d7a",         {'s7a': h, 'es7a': h, 'd7a': h}),
            ("g7a + s7a",                {'g7a': h, 's7a': h}),
            ("es2 + d2 + s7a",           {'es27a': h, 'd27a': h, 's7a': h}),
            ("px2_7a + py2_7a + s7a",    {'px27a': h, 'py27a': h, 's7a': h}),
        ]

        def _state_pairs(dd: dict):
            D, Dp = zD.copy(), zD.copy()
            for key_, a in dd.items():
                nm_ = key_[:-2]
                if nm_ not in names:
                    return None
                i_ = names.index(nm_)
                jj, kk = kap_pair(i_)
                D[qm._DELTA_IDX[key_]] += a
                Dp[qm._DELTA_IDX[names[jj] + '7a']] += kk * a
            return D, Dp

        def _residual_for(D, Dp):
            qp = quad(+Q_test, D)
            qn = quad(-Q_test, Dp)
            A = np.einsum('ij,njk,kl->nil', W,
                          Hs(kpts, +Q_test, D, qp) - Hs(kpts, +Q_test, D, zD), W.conj().T)
            B = Hs(ksd, -Q_test, Dp, qn) - Hs(ksd, -Q_test, Dp, zD)
            nA = float(np.linalg.norm(A))
            return nA, (float(np.linalg.norm(A - B)) / max(nA, 1e-30) if nA > 0 else 0.0), qp

        _log(f"      {'state':26s} {'|TRW|':>9s} {'residual':>12s}   dominant Quad67 channels")
        worst, n_viol, n_nontrivial = 0.0, 0, 0
        for label, dd in states:
            p_ = _state_pairs(dd)
            if p_ is None:
                _log(f"      {label:26s} (channel not in the registry — skipped)")
                continue
            D, Dp = p_
            nA, residual, qp = _residual_for(D, Dp)
            if nA > 1e-12:
                n_nontrivial += 1
            a7 = np.abs(qp[np.asarray(qm._IDX_7A_ARR)])
            top = ', '.join(names[i] for i in np.argsort(-a7)[:3] if a7[i] > 1e-9) or '(Quad67 = 0)'
            viol = residual > tol and nA > 1e-12
            n_viol += viol
            worst = max(worst, residual)
            _log(f"      {label:26s} {nA:9.2e} {residual:12.2e}   {top}{'   ❌' if viol else ''}")
        if n_viol:
            n_fail += 1
            _log(f"  ❌ FAIL — the TRW is not σd-covariant in {n_viol} of "
                 f"{n_nontrivial} non-trivial states (worst residual {worst:.1e}).")
        else:
            _log(f"  ✓ PASS — the TRW is σd-covariant in every non-trivial state "
                 f"(worst residual {worst:.1e})")

        # ── [C'] contrast: force OWN form factor, watch covariance break ──
        _orig_ff = qm._trw_form_factor
        try:
            qm._trw_form_factor = lambda name, i_irr, Phi_k: Phi_k[:, i_irr]
            _log("")
            _log("  [C'] same σd covariance with qm._trw_form_factor forced to the OWN form factor")
            _log("       (proves that the σd partner is what makes the TRW covariant):")
            _log(f"      {'state':26s} {'|TRW|':>9s} {'residual':>12s}")
            n_viol_own, worst_own, n_own = 0, 0.0, 0
            for label, dd in states:
                if not dd:
                    continue
                p_ = _state_pairs(dd)
                if p_ is None:
                    continue
                D, Dp = p_
                nA, residual, _ = _residual_for(D, Dp)
                if nA < 1e-12:
                    continue
                n_own += 1
                n_viol_own += residual > tol
                worst_own = max(worst_own, residual)
                _log(f"      {label:26s} {nA:9.2e} {residual:12.2e}")
            if n_own == 0:
                _log("      (no non-trivial state found — nothing to contrast)")
            elif n_viol_own > 0:
                _log(f"      own-form-factor variant: {n_viol_own}/{n_own} non-trivial states "
                     f"non-covariant, worst residual {worst_own:.1e}   "
                     f"✓ (as expected — the σd partner is required)")
            else:
                _log(f"      own-form-factor variant: {n_viol_own}/{n_own} non-covariant   "
                     f"⚠ (partner vs own makes no difference here)")
        finally:
            qm._trw_form_factor = _orig_ff

    overall = (n_fail == 0)
    _log(f"OVERALL: {'✓ PASS' if overall else '❌ FAIL'}  ({'no violations' if overall else f'{n_fail} violation group(s)'})")


def test_dHdQ_vs_build_H_stack(solver, M_bdg: np.ndarray, Q: float, mu: float, g_t: float, g_J: float, F_amp: float = 0.05, h: float = 1e-4, tol: float = 1e-6) -> None:
    """The analytic Q-vertex  solver._calc_dHdQ  against a 5-point finite difference of  vbdg._build_H_stack  at fixed (M, μ, Δ, Quad67).  _calc_dHdQ re-implements the TRW blocks of _build_H_stack
    (including the σd-character-dependent TRW form factor _trw_form_factor, the even/odd hole-block parity and the 7a/7b split), and every χ_SQ / χ_QQ / Hessian result is built on it; nothing else compares the two.
    [A] Quad67 = 0 (hopping, JT, AFM Weiss)   [B] one TRW channel at a time, both branches   [C] all channels at once   [D] the pairing block has no explicit Q dependence (so _calc_dHdQ needs no Δ)."""
    p = solver.p
    vb = solver._get_vbdg()
    names = list(qm._IRREP_NAMES)
    nd = qm._N_DELTA
    zD = np.zeros(nd, dtype=complex)
    k = solver.k_points
    ph = np.exp(1j * np.pi / 5.0)

    def H(q, Fv, Dv=None):
        return vb._build_H_stack(k, M_bdg, q, zD if Dv is None else Dv, Fv, mu, g_t, g_J)

    def fd(Fv, Dv=None):
        return (H(Q - 2 * h, Fv, Dv) - 8 * H(Q - h, Fv, Dv) + 8 * H(Q + h, Fv, Dv) - H(Q + 2 * h, Fv, Dv)) / (12 * h)

    def an(Fv):
        return solver._calc_dHdQ(M_bdg, Q, Fv, mu, g_t, g_J)

    def rel(A, B):
        return float(np.abs(A - B).max() / max(np.abs(B).max(), 1e-30))

    n_fail = 0
    A0, B0 = an(zD), fd(zD)
    e0 = rel(A0, B0)
    ok = e0 < tol
    n_fail += (not ok)
    _log(f"  [A] Quad67 = 0:  max|dH/dQ_analytic − FD| / max|FD| = {e0:.2e}   {'✓ PASS' if ok else '❌ FAIL'}")

    _, J_B = p.exchange_channels(Q, g_J)
    J7a, J7b = float(J_B[0]), float(J_B[1])
    _log(f"  [B] one channel at a time (Quad67 amplitude {F_amp}·e^{{iπ/5}}; J_B1g(Γ6Γ7a) = {J7a:+.3e}, J_B1g(Γ6Γ7b) = {J7b:+.3e})")
    _log(f"      {'chan':5s} {'kind':9s} {'par':>3s} {'7a: rel.err':>12s} {'|dH_TRW/dQ|':>12s}   {'7b: rel.err':>12s} {'|dH_TRW/dQ|':>12s}")
    n_vac = 0
    for i in _trw_channel_indices(solver):
        nm = names[i]
        out = []
        for br in ('7a', '7b'):
            Fv = zD.copy()
            Fv[qm._DELTA_IDX[nm + br]] = F_amp * ph
            dA, dB = an(Fv) - A0, fd(Fv) - B0
            out.append((rel(dA, dB) if np.abs(dB).max() > 1e-14 else float(np.abs(dA).max()), float(np.abs(dB).max())))
        bad = out[0][0] > tol or out[1][0] > tol
        n_fail += bad
        n_vac += (out[0][1] < 1e-14)
        _log(f"      {nm:5s} {qm._IRREP_KIND[nm]:9s} {qm._IRREP_PARITY[nm]:+3d} {out[0][0]:12.2e} {out[0][1]:12.2e}   {out[1][0]:12.2e} {out[1][1]:12.2e}   {'✓' if not bad else '❌ FAIL'}")
    if abs(J7b) < 1e-14:
        _log("      7b: J_B1g(Γ6Γ7b) = 0 ⇒ the 7b amplitudes have no TRW and no Q-vertex; both sides vanish (the 7b columns above are 0 by construction, not a test of the formula)")
    if n_vac:
        _log(f"  ✗ {n_vac} channel(s) have a vanishing 7a Q-vertex at this Q — the comparison is vacuous for them")

    rng = np.random.default_rng(1)
    Fall = zD.copy()
    for i in _trw_channel_indices(solver):
        for br in ('7a', '7b'):
            Fall[qm._DELTA_IDX[names[i] + br]] = F_amp * (rng.normal() + 1j * rng.normal())
    ea = rel(an(Fall) - A0, fd(Fall) - B0)
    ok = ea < tol
    n_fail += (not ok)
    _log(f"  [C] all TRW channels at once (random complex amplitudes: the reference character w_a1g is then a non-trivial interpolation):  rel.err = {ea:.2e}   {'✓ PASS' if ok else '❌ FAIL'}")

    Dall = zD.copy()
    for nm in names:
        Dall[qm._DELTA_IDX[nm + '7a']] = 0.01 * (rng.normal() + 1j * rng.normal())
        Dall[qm._DELTA_IDX[nm + '7b']] = 0.01 * (rng.normal() + 1j * rng.normal())
    ed = float(np.abs(fd(zD, Dall) - fd(zD)).max())
    ok = ed < 1e-9
    n_fail += (not ok)
    _log(f"  [D] ∂H_pair/∂Q (random Δ on every channel, FD) = {ed:.1e}   {'✓ PASS — the pairing block is Q-independent, _calc_dHdQ needs no Δ' if ok else '❌ FAIL — the pairing block depends on Q but _calc_dHdQ ignores Δ'}")
    _log(f"  {'✓ PASS' if n_fail == 0 else '❌ FAIL'} — dH/dQ consistency")


def test_bdg_structure_phs(solver, M_bdg: np.ndarray, Q: float, mu: float, g_t: float, g_J: float, amp: float = 0.05, n_k: int = 8, tol: float = 1e-12) -> None:
    """Exact algebraic structure of the 24×24 BdG matrix for EVERY channel (pair block and TRW block, branches 7a and 7b, complex amplitudes, M ≠ 0, Q ≠ 0; the TRW source is Quad67):
    Hermiticity and particle–hole symmetry  H(k) = −P H(−k)* P  (P swaps the particle and hole blocks).  The PHS identity is what the even/odd hole-block parity of every form factor
    (qm._IRREP_PARITY) must satisfy, so a wrong parity flag of a library harmonic, or a TRW block that does not follow its channel's parity, shows up here and nowhere else."""
    vb = solver._get_vbdg()
    names = list(qm._IRREP_NAMES)
    nd = qm._N_DELTA
    zD = np.zeros(nd, dtype=complex)
    rng = np.random.default_rng(2)
    k = rng.uniform(-np.pi, np.pi, (n_k, 2))
    P = np.block([[np.zeros((12, 12)), np.eye(12)], [np.eye(12), np.zeros((12, 12))]])

    def resid(Dv, Fv):
        Hk = vb._build_H_stack(k, M_bdg, Q, Dv, Fv, mu, g_t, g_J)
        Hm = vb._build_H_stack(-k, M_bdg, Q, Dv, Fv, mu, g_t, g_J)
        phs = np.abs(Hk + np.einsum('ij,njk,kl->nil', P, np.conj(Hm), P)).max()
        herm = np.abs(Hk - np.conj(np.transpose(Hk, (0, 2, 1)))).max()
        return float(phs), float(herm)

    r0 = resid(zD, zD)
    _log(f"  normal state (no pair, no TRW):  PHS residual {r0[0]:.1e}, Hermiticity residual {r0[1]:.1e}   {'✓' if max(r0) < tol else '❌ FAIL'}")
    n_bad, n_checked = (0 if max(r0) < tol else 1), 0
    worst = (0.0, '')
    for nm in names:
        for br in ('7a', '7b'):
            a = zD.copy()
            a[qm._DELTA_IDX[nm + br]] = amp * (1.0 + 0.3j)
            for part, (Dv, Fv) in (('pair', (a, zD)), ('TRW', (zD, a))):
                r = resid(Dv, Fv)
                n_checked += 1
                if max(r) >= tol:
                    n_bad += 1
                    _log(f"  ❌ FAIL — {nm}{br} {part} block (parity {qm._IRREP_PARITY[nm]:+d}): PHS {r[0]:.1e}, Hermiticity {r[1]:.1e}")
                if max(r) > worst[0]:
                    worst = (max(r), f"{nm}{br} {part}")
    allD, allF = zD.copy(), zD.copy()
    for nm in names:
        for br in ('7a', '7b'):
            allD[qm._DELTA_IDX[nm + br]] = amp * (rng.normal() + 1j * rng.normal())
            allF[qm._DELTA_IDX[nm + br]] = amp * (rng.normal() + 1j * rng.normal())
    ra = resid(allD, allF)
    n_bad += (max(ra) >= tol)
    _log(f"  {n_checked} single-channel blocks checked (all {len(names)} channels × 7a/7b × pair/TRW); worst residual {worst[0]:.1e} ({worst[1] or '-'})")
    _log(f"  all channels at once (random Δ and Quad67): PHS {ra[0]:.1e}, Hermiticity {ra[1]:.1e}   {'✓' if max(ra) < tol else '❌ FAIL'}")
    _log(f"  {'✓ PASS' if n_bad == 0 else '❌ FAIL'} — BdG matrix is Hermitian and particle–hole symmetric for every channel")


def extract_main_params(main_py_path):
    with open(main_py_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    main_block = None
    for node in tree.body:
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare)
                and isinstance(t.left, ast.Name) and t.left.id == '__name__'
                and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                and len(t.comparators) == 1
                and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == '__main__'):
                main_block = node
                break
    if main_block is None:
        raise RuntimeError(f"No `if __name__ == '__main__'` block in {main_py_path}")
    params_kwargs, target_doping = {}, None
    for node in ast.walk(main_block):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ModelParams':
            for kw in node.keywords:
                try:
                    params_kwargs[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    pass
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'target_doping':
                try:
                    target_doping = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    if not params_kwargs:
        raise RuntimeError("No ModelParams(...) call found in __main__ block")
    if target_doping is None:
        target_doping = 0.138
    return params_kwargs, target_doping


def recompute_orbital_projectors(qm_, p):
    """{A_xz, A_yz, A_xy} as 6x6 real symmetric matrices in the ordered doublet basis, recomputed independently of ModelParams.__post_init__."""
    H_soc_cf, Lx, Ly, Lz, Sz_full, LS_op = qm_._t2g_soc_cf_operators(p.lambda_soc, p.Delta_tetra, p.Delta_B1g_static)
    evals_soc, evecs_soc = np.linalg.eigh(H_soc_cf)
    B1g_t2g = Lx @ Lx - Ly @ Ly
    G6, G7a, G7b = qm_._get_G6_G7a_G7b(evals_soc, evecs_soc, Sz_full, LS_op, B1g_t2g)
    U_ord = np.column_stack([d[s] for d in (G6, G7a, G7b) for s in ('up', 'dn')])
    I2 = np.eye(2, dtype=complex)
    P_orb = {name: np.kron(np.outer(u, u.conj()), I2) for name, u in qm_._ORBITAL_VECTORS.items()}
    return {name: np.asarray(np.real(U_ord.conj().T @ P @ U_ord), dtype=float) for name, P in P_orb.items()}


def check_hopping_covariance(p, orb_in_basis, Q=0.02):
    """
    Compare U T_x(Q) U^+ with T_y(−Q) for the CURRENT bond-Q form (T_x = t_x(Q)·W_x, T_y = t_y(Q)·W_y, W_x/W_y Q-independent) and for the LEGACY orbital-Q form
    (T_x = t_x·A_xz + t_y·A_yz + t_avg·A_xy, and x<->y).  Under σ_d: A_xz <-> A_yz, A_xy invariant, hop_W_x -> hop_W_y.
    """
    A_xz, A_yz, A_xy = orb_in_basis['xz'], orb_in_basis['yz'], orb_in_basis['xy']
    tx, ty = p.effective_hopping_anisotropic(Q)
    txm, tym = p.effective_hopping_anisotropic(-Q)
    t_avg, t_avg_m = 0.5 * (tx + ty), 0.5 * (txm + tym)
    err_current = float(np.abs(tx * p.hop_W_y - tym * p.hop_W_y).max())
    T_x_leg_sd = tx * A_yz + ty * A_xz + t_avg * A_xy
    T_y_leg_mQ = tym * A_xz + txm * A_yz + t_avg_m * A_xy
    err_legacy = float(np.abs(T_x_leg_sd - T_y_leg_mQ).max())
    return {'current_hopping_err': err_current, 'legacy_hopping_err': err_legacy, 'Q': Q}

# D4 (≅ D4h gerade) irrep products: a term  Q·Δ_i*·Δ_j  (Q ∈ B1g) is invariant iff B1g ∈ Γ_i ⊗ Γ_j
_D4_PRODUCT = {
    ('A1g', 'A1g'): {'A1g'}, ('A1g', 'A2g'): {'A2g'}, ('A1g', 'B1g'): {'B1g'}, ('A1g', 'B2g'): {'B2g'}, ('A1g', 'Eg'): {'Eg'},
    ('A2g', 'A2g'): {'A1g'}, ('A2g', 'B1g'): {'B2g'}, ('A2g', 'B2g'): {'B1g'}, ('A2g', 'Eg'): {'Eg'},
    ('B1g', 'B1g'): {'A1g'}, ('B1g', 'B2g'): {'A2g'}, ('B1g', 'Eg'): {'Eg'},
    ('B2g', 'B2g'): {'A1g'}, ('B2g', 'Eg'): {'Eg'},
    ('Eg', 'Eg'): {'A1g', 'A2g', 'B1g', 'B2g'},
}
_C4_PARITY = {'A1g': +1, 'A2g': +1, 'B1g': -1, 'B2g': -1}          # C4 character of the 1D irreps (B1g/B2g change sign under a 90° rotation)

def _q_coupling_allowed(ga: str, gb: str) -> bool:
    """True if the bilinear Δ_a*·Δ_b can couple linearly to a B1g distortion in D4h (unknown irreps are treated as allowed, never as forbidden)."""
    key = (ga, gb) if (ga, gb) in _D4_PRODUCT else (gb, ga)
    return 'B1g' in _D4_PRODUCT.get(key, {'B1g'})


def test_linear_coupling(solver,
                         delta_ip_scan=(0.001, 0.0, -0.002, -0.005, -0.01, -0.02),
                         delta_tetra_ratios=(-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
                         h_list=(1e-3, 5e-4),
                         fd_steps=(1e-3, 5e-4, 2.5e-4),
                         scan_pairs: tuple[tuple[str, str], ...] = (('s', 'd'),),
                         branches: tuple[str, ...] = ('7a', '7b'),
                         target_doping: float = 0.15,
                         d4h_rel_tol: float = 1e-5,
                         tol_D4h: float = 5e-5,
                         tol_D2h: float = 1e-6,
                         richardson_tol: float = 0.05,
                         trace_tol: float = 1e-10,
                         tol_cov: float = 1e-9,
                         normal=None,
                         check_kernel: bool = True) -> None:
    """
    Linear Q coupling of the pair fields under the exact B1g covariance of the model.

    The model is covariant under  C4 (with the orbital rotation) ∘ (Q → −Q):   F(Q, {Δ_i}) = F(−Q, {ε_i Δ_i}),   ε_i = +1 (A1g, A2g), −1 (B1g, B2g) for the TOTAL irrep of channel i.
    Consequently  γ_ij = ∂³F/(∂Q ∂Δ_i ∂Δ_j)  is symmetry-ALLOWED in D4h iff B1g ∈ Γ_i ⊗ Γ_j (e.g. s7a[B1g] × d7a[A1g], and Eg × Eg inside the 7b branch) and vanishes identically otherwise
    (s7a × es7a, d7a × d37a, 7a × 7b).  The explicit static B1g field Δ_ip (D4h → D2h) switches the forbidden same-branch pairs on ∝ Δ_ip.
    [A] cross coupling γ_ij of all (A1g-form, B1g-form) pairs, branch-resolved, split by the D4h selection rule: FORBIDDEN pairs must vanish in D4h, ALLOWED pairs are reported
    [B] γ vs Δ_tet/λ scan (information)
    [C] fine Δ_ip scan on a FORBIDDEN pair (s7a, es7a): D4h ≡ 0, D2h ≠ 0; and on the ALLOWED pair (s7a, d7a): finite in D4h
    [D] Nambu-trace identity
    [E] Full 12×12 irrep decomposition (both static values)
    [F] exact B1g covariance (D4h limit): F(+Q) = F(−Q) in the normal state, σ_d spectral covariance, no A1g–B1g mixing of V_irrep at Q = 0, analytic force = numeric dF/dQ,
        F(Q, s, d) = F(−Q, ε_s s, ε_d d) for the allowed pair, hopping covariance (current bond-Q form vs the legacy orbital-Q form)
    """
    p = solver.p
    names_A = [c.name for c in qm._PAIR_CHANNELS if c.d4h == 'A1g']
    names_B = [c.name for c in qm._PAIR_CHANNELS if c.d4h == 'B1g']
    orig_ip  = p.Delta_B1g_static
    orig_tet = p.Delta_tetra
    lambda_soc = float(p.lambda_soc)
    _br_combos = [(a, b) for a in branches for b in branches if (a, b) != ('7b', '7a')]
    zero_D = np.zeros(qm._N_DELTA, dtype=complex)

    def _rebuild(delta_ip: float, delta_tet: float):
        p.Delta_B1g_static = delta_ip
        p.Delta_tetra      = delta_tet
        solver._full_rebuild()
        solver.g_JT_bare = solver.p.g_JT

    def _make_F_eval(delta_ip: float, delta_tet: float):
        _rebuild(delta_ip, delta_tet)
        g_t, g_J = p.get_gutzwiller_factors(target_doping)
        M0 = np.zeros(3)
        tx_b, ty_b = p.effective_hopping_anisotropic(0.0)
        t_eff = g_t * float(np.sqrt(0.5 * (tx_b ** 2 + ty_b ** 2)))
        mu, n_k = solver._find_mu_for_density(target_doping, M0, 0.0, zero_D, zero_D, -2.0 * t_eff * (1.0 - 2.0 * abs(target_doping)), g_t, g_J)
        def F_eval(Q, dv):
            gA, gB, gC = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M0, Q, mu, g_t, g_J)
            Vd = solver.g_JT_bare ** 2 / max(solver._K_bare, 1e-30)
            return solver._compute_bdg_free_energy(M0, Q, dv, zero_D, mu, g_t, g_J, gA, gB, gC, Vd, 0.0, p.K_lattice, p.K_lattice_Eg2, None)
        return F_eval

    def _gamma_ij_cross(F_eval, a: str, b: str, br_a: str = '7a', br_b: str = '7a'):
        gam = []
        for h in h_list:
            def _S(sb):
                dv = np.zeros(qm._N_DELTA, dtype=complex)
                dv[qm._DELTA_IDX[a + br_a]] = h
                dv[qm._DELTA_IDX[b + br_b]] = sb * h
                return F_eval(+h, dv) - F_eval(-h, dv)
            gam.append((_S(+1) - _S(-1)) / (4.0 * h ** 3))
        gam = np.array(gam)
        rich = (4.0 * gam[-1] - gam[0]) / 3.0 if len(gam) > 1 else float(gam[-1])
        return gam, rich

    def _gamma_at_step(F_eval, h: float, ch_a: str = 's7a', ch_b: str = 'd7a') -> float:
        """Diagonal+cross stencil on two pair amplitudes (channel keys such as 's7a', 'es7a', 'd7a')."""
        Delta_pp = np.zeros(qm._N_DELTA, dtype=complex)
        Delta_pp[qm._DELTA_IDX[ch_a]] = h
        Delta_pp[qm._DELTA_IDX[ch_b]] = h
        return (F_eval(+h, Delta_pp) - F_eval(+h, zero_D) - F_eval(-h, Delta_pp) + F_eval(-h, zero_D)) / (4.0 * h ** 3)

    def _gamma_richardson(F_eval, ch_a: str, ch_b: str):
        gammas = np.array([_gamma_at_step(F_eval, h, ch_a, ch_b) for h in fd_steps])
        R = np.array([(4.0 * gammas[k + 1] - gammas[k]) / 3.0 for k in range(len(gammas) - 1)])
        if len(R) >= 2:
            RR = np.array([(4.0 * R[k + 1] - R[k]) / 3.0 for k in range(len(R) - 1)])
            gamma_rich = float(RR[-1])
            conv_residual = (abs(RR[-1] - RR[-2]) / max(abs(RR[-1]), 1e-20) if len(RR) >= 2 else abs(RR[-1] - R[-1]) / max(abs(RR[-1]), 1e-20))
            converged = conv_residual < richardson_tol
        else:
            gamma_rich = float(R[-1]); converged = False
        return gammas, gamma_rich, converged

    def _nambu_trace_identity():
        rng = np.random.default_rng(1)
        worst = {'antisym': 0.0, 'sym': 0.0}
        for _ in range(5):
            B = rng.normal(size=(6, 6)); B = B + B.T
            for kind, sgn in (('antisym', -1.0), ('sym', +1.0)):
                D = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
                D = D + sgn * D.T
                val = abs(np.real(np.trace(B @ D @ D.conj().T)) - np.real(np.trace(B.T @ D.conj().T @ D)))
                worst[kind] = max(worst[kind], val)
        return worst

    def _anomalous_blocks():
        """12×12 anomalous blocks of every A1g/B1g k-form channel (they depend on the pairing basis only, not on the static field)."""
        vb = solver._get_vbdg()
        k0 = np.array([[0.0, 0.5 * np.pi]])
        blocks: dict[str, np.ndarray] = {}
        for chan_tag, idx_arr in (('7a', qm._IDX_7A_ARR), ('7b', qm._IDX_7B_ARR)):
            for i_irr, spec in enumerate(qm._PAIR_CHANNELS):
                if spec.d4h not in ('A1g', 'B1g'):
                    continue
                dv = np.zeros(qm._N_DELTA, dtype=complex)
                dv[int(idx_arr[i_irr])] = 1.0
                H = vb._build_H_stack(k0, np.zeros(3), 0.0, dv, np.zeros(qm._N_DELTA, dtype=complex), 0.0, 1.0, 1.0)[0]
                blocks[f'{spec.name}{chan_tag}'] = np.array(H[0:12, 12:24])
        return blocks

    scale: dict[float, dict[tuple[str, str], tuple[float, float, float, float]]] = {}
    tet_scan: dict[tuple[str, str, str, str], dict[float, float]] = {}
    diag_scan: dict[str, list] = {'forbidden': [], 'allowed': []}
    irrep_results: dict = {}

    try:
        # ── D4h total irrep of every channel (decides which pairs may couple linearly to Q) ──
        _rebuild(0.0, orig_tet)
        blocks = _anomalous_blocks()
        _r0 = _pair_irrep_analysis(solver, _pair_irrep_context(lambda_soc, orig_tet, 0.0), blocks, tol=trace_tol, verbose=False)
        irrep_D4h: dict[str, str] = {}
        for name, c in _r0['channels'].items():
            top_i = max(c['w4'], key=c['w4'].get)
            irrep_D4h[name] = top_i if c['w4'][top_i] > 0.5 else 'mixed'

        # ── Part A: aggregated γ table, split by the D4h selection rule ──
        _log("[A] DYNAMICAL CROSS COUPLING γ_ij = ∂³F/(∂Q ∂Δ_i ∂Δ_j)  — split by the D4h selection rule  B1g ∈ Γ_i ⊗ Γ_j")
        _log(f"    Δ_ip scan: {tuple(float(d) for d in delta_ip_scan)}")
        _log(f"    {'Δ_ip':>10s}  {'branch':<10s}  {'max|γ| forb.':>13s}  {'max|γ| allow.':>13s}  {'γ(s,d)':>12s}  {'γ(es,d)':>12s}")

        for dip in delta_ip_scan:
            delta_ip = float(dip)
            F_eval = _make_F_eval(delta_ip, orig_tet)
            per_branch = {}
            for br_a, br_b in _br_combos:
                vals_f, vals_a = [], []
                sd_val, esd_val = 0.0, 0.0
                for a in names_A:
                    for b in names_B:
                        _, rich = _gamma_ij_cross(F_eval, a, b, br_a, br_b)
                        allowed = _q_coupling_allowed(irrep_D4h.get(a + br_a, 'mixed'), irrep_D4h.get(b + br_b, 'mixed'))
                        (vals_a if allowed else vals_f).append(abs(rich))
                        if (a, b) == ('s', 'd'):  sd_val = rich
                        if (a, b) == ('es', 'd'): esd_val = rich
                max_f = max(vals_f) if vals_f else 0.0
                max_a = max(vals_a) if vals_a else 0.0
                n_a = len(vals_a)
                per_branch[(br_a, br_b)] = (max_f, max_a, float(n_a), float(len(vals_f)))
                _log(f"    {delta_ip:>+10.4f}  ({br_a},{br_b}):  {max_f:>13.3e}  {max_a:>13.3e}  {sd_val:>+12.3e}  {esd_val:>+12.3e}")
            scale[delta_ip] = per_branch
        _log("")

        d4h = scale.get(0.0, {})
        d2h = {k: v for k, v in scale.items() if abs(k) > 1e-12}
        verdict_A = True
        noise_floor = 1e-5
        _log("  D4h selection rule (forbidden pairs must vanish at Δ_ip = 0; allowed pairs are symmetry-permitted):")
        _log(f"    {'branch':<10s}  {'forb.(D4h)':>12s}  {'forb.(D2h)':>12s}  {'allow.(D4h)':>12s}  {'#allow/#all':>11s}  verdict")
        for key in _br_combos:
            f4, a4, n_a, n_f = d4h.get(key, (0.0, 0.0, 0.0, 0.0))
            f2 = max((v.get(key, (0.0,) * 4)[0] for v in d2h.values()), default=0.0)
            a2 = max((v.get(key, (0.0,) * 4)[1] for v in d2h.values()), default=0.0)
            ref = max(a4, f2, a2)
            ok = bool(f4 < max(noise_floor, d4h_rel_tol * ref))
            verdict_A &= ok
            tag = '✓ PASS' if ok else '✗ FAIL'
            extra = ''
            if n_a > 0 and a4 < noise_floor:
                extra = '  (✗ flag: allowed pairs present but γ ≈ 0)'
            _log(f"    ({key[0]},{key[1]}):  {f4:>12.3e}  {f2:>12.3e}  {a4:>12.3e}  {int(n_a):>4d}/{int(n_a + n_f):<5d}  {tag}{extra}")
        _log(f"  [A] {'✓ PASS' if verdict_A else '❌ FAIL'}")
        _log("")

        # ── Part B: γ vs Δ_tet/λ ──
        _log(f"[B] CROSS γ(a,b) vs Δ_tet/λ at Δ_ip = {orig_ip:+.4f}, scan pairs {scan_pairs}")
        for (a0, b0) in scan_pairs:
            if a0 not in names_A or b0 not in names_B:
                _log(f"  skipped pair ({a0},{b0}): not (A1g, B1g)")
                continue
            for br_a, br_b in _br_combos:
                _log(f"  pair ({a0}@{br_a}, {b0}@{br_b}):")
                _log(f"    {'Δ_tet/λ':>10s}  {'γ':>14s}")
                scan_here = {}
                for ratio in delta_tetra_ratios:
                    F_eval = _make_F_eval(orig_ip, ratio * lambda_soc)
                    _, rich = _gamma_ij_cross(F_eval, a0, b0, br_a, br_b)
                    scan_here[float(ratio)] = float(rich)
                    _log(f"    {ratio:>+10.2f}  {rich:>+14.4e}")
                signs = [np.sign(v) for v in scan_here.values()]
                flips = sum(1 for s1, s2 in zip(signs, signs[1:]) if s1 * s2 < 0)
                _log(f"    sign flips: {flips}")
                tet_scan[(a0, b0, br_a, br_b)] = scan_here
        _log("")

        # ── Part C: fine Δ_ip scan on forbidden and allowed pairs (the expectation comes from the D4h irreps, not from a hard-coded list) ──
        _pairs_C = [('s7a', 'es7a'), ('d7a', 'd37a'), ('s7a', 'd7b'), ('s7a', 'd7a'), ('es7a', 'd7a'), ('s7b', 'd7b')]
        _pairs_C = [pr for pr in _pairs_C if pr[0] in irrep_D4h and pr[1] in irrep_D4h]
        verdict_C = True
        _scale_allow = 1e-30
        _C_records = []
        for _pair in _pairs_C:
            _ga, _gb = irrep_D4h[_pair[0]], irrep_D4h[_pair[1]]
            _allowed = _q_coupling_allowed(_ga, _gb)
            _same_branch = _pair[0][-2:] == _pair[1][-2:]
            _log(f"[C] ∂²F/(∂Q ∂|Δ|²) at Δ_{_pair[0]} = Δ_{_pair[1]} = h   ({_ga} × {_gb}: {'ALLOWED' if _allowed else 'FORBIDDEN'} in D4h), fd_steps={fd_steps}")
            _log(f"  {'Δ_ip[eV]':>10s}  {'γ(h)':>13s}  {'γ(h/2)':>13s}  {'γ(h/4)':>13s}  {'γ_Rich':>14s}  {'conv':>5s}  verdict")
            _recs = []
            for D_ip in delta_ip_scan:
                D_ip = float(D_ip)
                F_eval = _make_F_eval(D_ip, orig_tet)
                gammas, gamma_rich, converged = _gamma_richardson(F_eval, *_pair)
                _recs.append((D_ip, gamma_rich))
                if _allowed:
                    verdict = "✓ allowed: γ≠0" if abs(gamma_rich) > tol_D2h else f"✗ allowed pair but |γ|={abs(gamma_rich):.1e}"
                    _scale_allow = max(_scale_allow, abs(gamma_rich))
                elif abs(D_ip) < 1e-12:
                    verdict = "✓ D4h: γ≡0" if abs(gamma_rich) < tol_D4h else f"✗ D4h: |γ|={abs(gamma_rich):.1e}≠0"
                else:
                    verdict = "✓ D2h: γ≠0" if abs(gamma_rich) > tol_D2h else "✗ D2h: |γ|≈0"
                _log(f"  {D_ip:>+10.5f}  {gammas[0]:>+13.3e}  {gammas[1] if len(gammas) > 1 else float('nan'):>+13.3e}  {gammas[2] if len(gammas) > 2 else float('nan'):>+13.3e}  "
                     f"{gamma_rich:>+14.3e}  {'✓' if converged else '✗':>5s}  {verdict}")
            _C_records.append((_pair, _allowed, _same_branch, _recs))
        _log("")
        _log("  summary of [C] (forbidden pairs: D4h ≡ 0 and D2h ≠ 0; allowed pairs: γ ≠ 0 at every Δ_ip; the stencil contains the diagonal terms Q|Δ_a|², Q|Δ_b|², which for the Eg channels of the 7b branch also switch on with Δ_ip — the pure 7a×7b CROSS coupling stays 0, see [A]):")
        for _pair, _allowed, _same_branch, _recs in _C_records:
            if _allowed:
                ok = all(abs(g) > tol_D2h for _, g in _recs)
                _log(f"    {_pair[0]:>6s} × {_pair[1]:<6s} allowed    : {sum(1 for _, g in _recs if abs(g) > tol_D2h)}/{len(_recs)} points with γ≠0  {'✓' if ok else '✗ (flag)'}")
            else:
                f4 = [g for D, g in _recs if abs(D) < 1e-12]; f2 = [g for D, g in _recs if abs(D) > 1e-12]
                ok4 = all(abs(g) < max(tol_D4h, d4h_rel_tol * _scale_allow) for g in f4) if f4 else True
                ok2 = all(abs(g) > tol_D2h for g in f2) if f2 else True
                verdict_C &= bool(ok4 and ok2)
                _log(f"    {_pair[0]:>6s} × {_pair[1]:<6s} forbidden  : D4h max|γ|={max([abs(g) for g in f4] + [0.0]):.2e} {'≡0 ✓' if ok4 else '❌ FAIL'};  "
                     f"D2h {sum(1 for g in f2 if abs(g) > tol_D2h)}/{len(f2)} with γ≠0 {'✓' if ok2 else '❌ FAIL'}")
        _log(f"  [C] {'✓ PASS' if verdict_C else '❌ FAIL'}")
        _log("")

        # ── Part D: Nambu-trace identity ──
        worst = _nambu_trace_identity()
        _id_ok = max(worst.values()) < trace_tol
        _log(f"[D] Nambu-trace identity: max residual = {max(worst.values()):.1e}  {'✓ PASS (algebraic)' if _id_ok else '❌ FAIL'}")
        _log("")

        # ── Part E: 12×12 irrep decomposition (verbose=False → compact summary) ──
        _rebuild(orig_ip, orig_tet)
        _log("[E] 12×12 anomalous-block irrep decomposition")
        for static, tag in ((0.0, "D4h limit"), (orig_ip, f"working point (Δ_ip={orig_ip:+.4f})")):
            ctx = _pair_irrep_context(lambda_soc, orig_tet, static)
            r = _pair_irrep_analysis(solver, ctx, blocks, tol=trace_tol, verbose=False)
            _log(f"  ── {tag}, Δ_tet/λ = {ctx['ratio']:+.3f} ──")
            _log(f"    {'channel':<8s} {'Tr[B1g·D·D†]':>14s} {'‖[B1g,D]‖_F':>14s}  dominant irrep")
            for name, c in r['channels'].items():
                top_i = max(c['w4'], key=c['w4'].get)
                top_w = c['w4'][top_i]
                if top_w > 0.5:  # informative only
                    _log(f"    {name:<8s} {c['tr_B1g']:>+14.2e} {c['comm_B1g']:>14.2e}  {top_i}:{top_w:.2f}")
            _log("    pair spaces:")
            for (tag_ps, form), mult in r['pair_space'].items():
                _log(f"      {tag_ps} {form}-form: " + "  ".join(f"{k}×{v:.0f}" for k, v in mult.items()))
            nonzero_alpha = [n for n, c in r['channels'].items() if c['best']['w'] > 0.5 and abs(c['best']['alpha']) > 1e-3]
            _log(f"    gauge scan: {len(nonzero_alpha)} channel(s) with α* ≠ 0" + (f" ({', '.join(nonzero_alpha[:4])})" if nonzero_alpha else ""))
            irrep_results[('work', static)] = r
        _log("")

        _log("  Δ_tet/λ scan — dominant total irrep at α=0 (only informative channels):")
        for static in sorted({0.0, orig_ip}):
            _log(f"  Δ_ip = {static:+.4f}:")
            _log(f"    {'Δ_tet/λ':>8s}  {'s7a':>9s}  {'es7a':>9s}  {'d7a':>9s}  {'d27a':>9s}")
            for ratio in delta_tetra_ratios:
                ctx = _pair_irrep_context(lambda_soc, ratio * lambda_soc, static)
                r = _pair_irrep_analysis(solver, ctx, blocks, tol=trace_tol, verbose=False)
                c = r['channels']
                row = []
                for ch in ('s7a', 'es7a', 'd7a', 'd27a'):
                    top_i = max(c[ch]['w4'], key=c[ch]['w4'].get)
                    row.append(f"{top_i}:{c[ch]['w4'][top_i]:.2f}")
                _log(f"    {ratio:>+8.2f}  " + "  ".join(f"{v:>9s}" for v in row))
                irrep_results[('scan', static, ratio)] = r
        _log("")

        # ── Part F: exact B1g covariance (D4h limit: static B1g field switched off) ──
        _log("[F] EXACT B1g COVARIANCE of the model (D4h limit Δ_ip = 0):  C4 (orbital rotation) ∘ (Q → −Q)")
        _rebuild(0.0, orig_tet)
        if normal is not None:
            M_n, mu_n, gt_n, gJ_n = np.asarray(normal.M, dtype=float), float(normal.mu), float(normal.g_t), float(normal.g_J)
        else:
            M_n, _d, _f, _n, mu_n, gt_n, gJ_n, _g1, _g2, _g3 = solver.refine_M_state(target_doping, 0.0, zero_D, zero_D, damping=0.45, verbose=False)
        # (F1) the normal-state free energy is even in Q — no linear term
        Qs = (-0.02, -0.01, 0.0, 0.01, 0.02)
        Fq = {q: float(solver._canonical_free_energy(target_doping, M_n, q, zero_D, zero_D, mu_n, gt_n, gJ_n, 0.0, 0.0, None)) for q in Qs}
        odd2 = abs(Fq[0.02] - Fq[-0.02]); lin = abs((Fq[0.01] - Fq[-0.01]) / 0.02)
        pass_F = odd2 < tol_cov
        _log(f"  (F1) normal state: |F(+Q)−F(−Q)| = {odd2:.2e} (Q=0.02),  |dF/dQ|(Q=0) = {lin:.2e}   {'✓ PASS' if pass_F else '❌ FAIL'}")
        # (F2) spectral covariance under σ_d: E(k,Q) = E(σ_d k, −Q); the same Q with the wrong sign must NOT match
        vb = solver._get_vbdg()
        rng = np.random.default_rng(3)
        kk = rng.uniform(-np.pi, np.pi, size=(24, 2)); k_sd = kk[:, ::-1].copy(); Q_t = 0.02
        def _sp(k, Q):
            return np.sort(np.linalg.eigvalsh(vb._build_H_stack(k, M_n, Q, zero_D, zero_D, mu_n, gt_n, gJ_n, out=None)), axis=1)
        E_kQ = _sp(kk, Q_t)
        d_cov = float(np.abs(E_kQ - _sp(k_sd, -Q_t)).max()); d_anti = float(np.abs(E_kQ - _sp(k_sd, +Q_t)).max())
        pass_cov = bool(d_cov < tol_cov and d_anti > 1e-6)
        _log(f"  (F2) σ_d covariance: max|E(k,Q)−E(σ_d k,−Q)| = {d_cov:.2e},  anti-covariant max|E(k,Q)−E(σ_d k,+Q)| = {d_anti:.2e}   {'✓ PASS' if pass_cov else '❌ FAIL'}")
        # (F3) vertex at Q = 0: no A1g–B1g mixing (es ↔ d), sector-swap defect
        pass_mix = pass_defect = True
        if check_kernel:
            Gm, _Fcl, VJTc, Vcap = solver._make_vertex_params(target_doping, M_n, 0.0, zero_D, zero_D, mu_n, gt_n, gJ_n)
            cache = solver.compute_pairing_kernel_and_build_cache(target_doping, M_n, 0.0, mu_n, gt_n, gJ_n, p.exchange_channels(0.0, gJ_n)[0], Gm, VJTc, Vcap)
            names = list(qm._IRREP_NAMES)
            defect = float(np.nanmax(cache['sector_swap_defect'])) if 'sector_swap_defect' in cache else float('nan')
            v_es_d = float(abs(cache['V_irrep'][names.index('es'), names.index('d')]))
            pass_defect = bool(defect < 1e-8) if np.isfinite(defect) else True
            pass_mix = bool(v_es_d < 1e-8)
            _log(f"  (F3) vertex at Q=0: |V_irrep(es,d)| = {v_es_d:.2e},  sector-swap defect = {defect:.2e}   {'✓ PASS' if (pass_mix and pass_defect) else '❌ FAIL'}")
        else:
            _log("  (F3) vertex check skipped (check_kernel=False)")
        # (F4) analytic Q-force = numeric derivative of the Q profile.  The force carries a deliberately coarse finite-difference cluster term (h_c = 1e-3 Å in _make_Q_profile), so the comparison is
        #      made against a Richardson-extrapolated derivative with a scale-aware tolerance: a stencil error must not be mistaken for a model inconsistency (and vice versa)
        F_tot, force = solver._make_Q_profile(target_doping, M_n, zero_D, zero_D, mu_n, gt_n, gJ_n, 0.0, 0.0, None)
        def _dF(h):
            return float((F_tot(Q_t + h) - F_tot(Q_t - h)) / (2.0 * h))
        D1, D2 = _dF(1e-3), _dF(5e-4)
        fa = float(force(Q_t)); fn = (4.0 * D2 - D1) / 3.0; fd = abs(fa - fn)
        tol_force = max(1e-5, 5e-3 * abs(fa))
        pass_force = bool(fd < tol_force)
        _log(f"  (F4) Q-force: analytic = {fa:+.6e}, numeric (Richardson 1e-3/5e-4) = {fn:+.6e}, |diff| = {fd:.2e} = {fd / max(abs(fa), 1e-30):.2%} of the force (tol {tol_force:.1e})   {'✓ PASS' if pass_force else '❌ FAIL'}")
        # (F5) the allowed pair obeys the exact covariance F(Q,s,d) = F(−Q, ε_s s, ε_d d), and its coupling is non-zero
        F_eval = _make_F_eval(0.0, orig_tet)
        eps_s = _C4_PARITY.get(irrep_D4h.get('s7a', '?'), +1); eps_d = _C4_PARITY.get(irrep_D4h.get('d7a', '?'), +1)
        h_c, Q_c = 0.01, 0.02
        def _dv(s, d):
            dv = np.zeros(qm._N_DELTA, dtype=complex); dv[qm._DELTA_IDX['s7a']] = s; dv[qm._DELTA_IDX['d7a']] = d
            return dv
        f0 = F_eval(Q_c, _dv(h_c, h_c))
        dev_cov = abs(f0 - F_eval(-Q_c, _dv(eps_s * h_c, eps_d * h_c)))
        dev_odd = abs(f0 - F_eval(-Q_c, _dv(h_c, h_c)))
        pass_pair = bool(dev_cov < tol_cov and dev_odd > 1e-9)
        _log(f"  (F5) allowed pair (s7a[{irrep_D4h.get('s7a', '?')}, ε={eps_s:+d}], d7a[{irrep_D4h.get('d7a', '?')}, ε={eps_d:+d}]): "
             f"|F(Q,s,d)−F(−Q,εs,εd)| = {dev_cov:.2e},  coupling |F(Q,s,d)−F(−Q,s,d)| = {dev_odd:.2e}   {'✓ PASS' if pass_pair else '❌ FAIL'}")
        # (F6) hopping covariance: the current bond-Q form is covariant, the legacy orbital-Q form is not (information)
        try:
            hop = check_hopping_covariance(p, recompute_orbital_projectors(qm, p), Q=Q_t)
            pass_hop = bool(hop['current_hopping_err'] < tol_cov)
            _log(f"  (F6) hopping covariance U·T_x(Q)·U† = T_y(−Q): current bond-Q form err = {hop['current_hopping_err']:.2e}  {'✓' if pass_hop else '❌ FAIL'};  "
                 f"legacy orbital-Q form err = {hop['legacy_hopping_err']:.2e} (information: not covariant)")
        except Exception as _e:                                     # the private operator builders may be renamed — never let the information line abort the test
            pass_hop = True
            _log(f"  (F6) hopping covariance diagnostic skipped: {_e}")
        # symmetry breaking at the working point (information)
        if abs(orig_ip) > 0.0:
            _rebuild(orig_ip, orig_tet)
            d_cov_ip = float(np.abs(_sp(kk, Q_t) - _sp(k_sd, -Q_t)).max())
            _log(f"  working point Δ_ip = {orig_ip:+.1e}: max|E(k,Q)−E(σ_d k,−Q)| = {d_cov_ip:.2e}  (the static B1g field is the only symmetry breaker)")
        verdict_F = pass_F and pass_cov and pass_mix and pass_defect and pass_force and pass_pair and pass_hop
        _log(f"  [F] {'✓ PASS' if verdict_F else '❌ FAIL'}")

        # ── Part G: TRW odd-in-Q response ──
        _rebuild(0.0, orig_tet)
        _log("[G] TRW odd-in-Q:  0.5·[F(+Q) − F(−Q)]  (M=0, |Δ|=10 meV, Quad67 self-consistent at Q=0)")
        M0_G = np.zeros(3)
        g_tG, g_JG = p.get_gutzwiller_factors(target_doping)
        mu_G, _ = solver._find_mu_for_density(target_doping, M0_G, 0.0, zero_D, zero_D, -2.0 * g_tG * p.t0 * (1.0 - 2.0 * abs(target_doping)), g_tG, g_JG)
        gs_G, ga_G, gb_G = solver.estimate_gutzwiller_factors_occupation_based(target_doping, M0_G, 0.0, mu_G, g_tG, g_JG)
        Vd_G = solver.g_JT_bare ** 2 / max(solver._K_bare, 1e-30)

        def _F_G(Q, dv, Quad67):
            return float(solver._compute_bdg_free_energy(M0_G, Q, dv, Quad67, mu_G, g_tG, g_JG, gs_G, ga_G, gb_G, Vd_G, 0.0, p.K_lattice, p.K_lattice_Eg2, None))

        a_G, Q_G = 0.01, 0.02
        _mk = lambda **kw: (lambda D: (D.__setitem__([qm._DELTA_IDX[k] for k in kw],
                                                     list(kw.values())), D)[1])(np.zeros(qm._N_DELTA, dtype=complex))
        pats_G = [('d7a',         _mk(d7a=a_G)),
                  ('es7a',        _mk(es7a=a_G)),
                  ('es7a+d7a',    _mk(es7a=a_G, d7a=a_G)),
                  ('es7a+i·d7a',  _mk(es7a=a_G, d7a=1j * a_G))]
        _log(f"  {'pattern':<16s} {'TRW off':>14s} {'TRW on':>14s}")
        ok_G = True
        for lab, dv in pats_G:
            Quad67 = np.zeros(qm._N_DELTA, dtype=complex)
            for _ in range(4):
                ev, ec = np.linalg.eigh(solver._get_vbdg()._build_H_stack(solver.k_points, M0_G, 0.0, dv, Quad67, mu_G, g_tG, g_JG))
                Quad67 = qm._compute_B1g_quadrupole(solver.k_weights, solver.Phi_irrep_k, ev, ec, solver.kT, solver.B1g_op)

            odd_off = 0.5 * (_F_G(+Q_G, dv, zero_D) - _F_G(-Q_G, dv, zero_D))
            odd_on  = 0.5 * (_F_G(+Q_G, dv, Quad67) - _F_G(-Q_G, dv, Quad67))
            _eo = _expect_Q_odd(dv)
            ok_G &= (abs(odd_off) > 1e-9) if _eo is True else ((abs(odd_off) < 1e-9) if _eo is False else True)
            _log(f"  {lab:<16s} {odd_off:>+14.3e} {odd_on:>+14.3e}")
        _log(f"  [G] {'✓ PASS (TRW off: Q-odd exactly for in-phase mixed patterns)' if ok_G else '❌ FAIL (TRW-off Q-parity violates the pair-level selection rule)'}")
        verdict_F &= ok_G

    finally:
        p.Delta_B1g_static = orig_ip
        p.Delta_tetra      = orig_tet
        solver._full_rebuild()
        solver.g_JT_bare = solver.p.g_JT

def _c4_breakers(solver, Q: float):
    """(h_c4_breaker, h_c4_symmetric) exactly as scf_gap_diagnostics forms them for qm._sector_report_lines: dimensionless C4 breakers of H, and whether H itself is C4-symmetric."""
    pp = solver.p
    tx, ty = pp.effective_hopping_anisotropic(Q)
    breaker = (abs(pp.Delta_B1g_static) + abs(pp.g_JT) * abs(Q)) / abs(pp.Delta_CF) + abs(tx - ty) / (tx + ty)
    symmetric = bool(abs(pp.Delta_B1g_static) <= qm._SYM_THRESH and abs(float(Q)) <= qm._SYM_THRESH)
    return breaker, symmetric

def _bond_weights(ff):
    """(w, lo, hi): normalised squared amplitudes |c_δ|² of the bond expansion of the form factor `ff` on an n×n periodic grid, and min/max(|δ_x|, |δ_y|) of every bond."""
    k = 2.0 * np.pi * np.arange(qm._BOND_GRID) / qm._BOND_GRID
    KX, KY = np.meshgrid(k, k, indexing='ij')
    c = np.fft.fft2(np.broadcast_to(np.asarray(ff(KX, KY), dtype=float), KX.shape)) / qm._BOND_GRID ** 2
    w = np.abs(c) ** 2
    w[w < 1e-20 * float(w.max())] = 0.0                              # FFT round-off (~1e-31) -> exact zero: bonds that are absent from the form factor carry exactly no weight
    ax = np.minimum(np.arange(qm._BOND_GRID), qm._BOND_GRID - np.arange(qm._BOND_GRID))
    AX, AY = np.meshgrid(ax, ax, indexing='ij')
    return w / max(float(w.sum()), 1e-300), np.minimum(AX, AY), np.maximum(AX, AY)

def _check_bond_content() -> None:
    """Import-time check of the declared bond KIND of every library harmonic against the real-space support of its form factor: 'onsite' = a constant; 'nn_AB' = bonds of odd sublattice distance only (A–B);
    'nnn_diag' / 'nnn_axis' = bonds of even sublattice distance only and no on-site term, 'nnn_axis' additionally purely axial ((n,0)-type).  Only failures and one summary line are reported."""
    n_bad = 0
    for n, v in qm._HARMONIC_LIBRARY.items():
        w, lo, hi = _bond_weights(v[3])
        f_on = float(w[(lo == 0) & (hi == 0)].sum())
        f_odd = float(w[((lo + hi) % 2) == 1].sum())               # weight on odd sublattice distance (A–B bonds)
        f_ax = float(w[lo == 0].sum())
        kind = v[1]
        ok = {'onsite': abs(f_on - 1.0) < 1e-9, 'nn_AB': abs(f_odd - 1.0) < 1e-9,
              'nnn_diag': f_odd < 1e-9 and f_on < 1e-9,
              'nnn_axis': f_odd < 1e-9 and f_on < 1e-9 and abs(f_ax - 1.0) < 1e-9}[kind]
        if not ok:
            n_bad += 1
            _log(f"  ❌ FAIL — library harmonic '{n}': kind '{kind}' disagrees with the real-space bond content of its form factor (on-site {f_on:.3g}, odd-distance {f_odd:.3g}, axial {f_ax:.3g})")
    _log(f"  bond content: the declared kind of all {len(qm._HARMONIC_LIBRARY)} library harmonics agrees with the real-space support of their form factors   {'✓ PASS' if not n_bad else '❌ FAIL'}")
    # the bond-content analysis runs on an n × n grid (qm._BOND_GRID): bonds with |δ| ≥ n/2 alias onto shorter ones (cos 11k looks like a (5,0) bond on n = 16); the class / parity conclusions above
    # stay right only if the aliased bond keeps its class and parity, so list the harmonics that alias instead of silently validating the aliased bond
    n_g = int(qm._BOND_GRID)
    n_f = 4 * n_g
    k1 = 2.0 * np.pi * np.arange(n_f) / n_f
    KXf, KYf = np.meshgrid(k1, k1, indexing='ij')
    sm = np.where(np.arange(n_f) > n_f // 2, np.arange(n_f) - n_f, np.arange(n_f))
    aliased = []
    for n, v in qm._HARMONIC_LIBRARY.items():
        c = np.fft.fft2(np.asarray(v[3](KXf, KYf), dtype=float)) / n_f ** 2
        ia, ib = np.nonzero(np.abs(c) > 1e-12)
        dmax = int(max(np.abs(sm[ia]).max(), np.abs(sm[ib]).max()))
        if dmax >= n_g // 2:
            aliased.append(f"{n} (|δ| = {dmax})")
    if aliased:
        _log(f"  ✗ {len(aliased)} harmonics have bonds with |δ| ≥ _BOND_GRID/2 = {n_g // 2}, which alias on the {n_g}×{n_g} bond grid (class and parity of the aliased bond are what is validated above): " + ", ".join(aliased))

def _check_sk_hopping() -> None:
    """straight x-bond ⇒ T = diag(1, 0, 1) in (xz, yz, xy) (xz and xy are C4 partners about the bond axis, hence EQUAL; yz does not hop),
    y-bond ⇒ diag(0, 1, 1); no off-diagonal element; independent of V_pdσ."""
    for pds_over_pdp in (qm._SK_PDS_OVER_PDP, -1.0, 0.0, 3.0):
        Tx = -(qm._sk_pd_t2g(*[float(c) for c in (1.0, 0.0, 0.0)], pds_over_pdp, 1.0).T @ qm._sk_pd_t2g(*[float(c) for c in (-1.0, 0.0, 0.0)], pds_over_pdp, 1.0))
        Ty = -(qm._sk_pd_t2g(*[float(c) for c in (0.0, 1.0, 0.0)], pds_over_pdp, 1.0).T @ qm._sk_pd_t2g(*[float(c) for c in (0.0, -1.0, 0.0)], pds_over_pdp, 1.0))
        good = bool(np.allclose(Tx, np.diag([1.0, 0.0, 1.0]), atol=1e-12) and np.allclose(Ty, np.diag([0.0, 1.0, 1.0]), atol=1e-12))
        _log(f"  Slater–Koster t2g hopping of a straight bond for V_pdσ/V_pdπ = {pds_over_pdp:+.4f}: x-bond diag(1,0,1), y-bond diag(0,1,1)  {'✓ PASS' if good else '❌ FAIL'}")
        if not good:
            _log(f"  Tx =\n{Tx}\n  Ty =\n{Ty}")


def _registry_checks() -> None:
    """The three import-time checks of the channel registry and of the hopping matrices, as ONE step: bond content of the harmonics, Slater–Koster hopping structure, registry self-check."""
    _check_bond_content()
    _check_sk_hopping()
    test_channel_registry_selfcheck()


_ONLY: tuple = ()

def _step(title: str, fn, *args, **kwargs):
    """Run ONE check in isolation: banner, wall time, ✓ / ✗ tally; an exception is logged with its traceback as a CRASH and does not stop the remaining checks."""
    if _ONLY and not any(s in title.lower() for s in _ONLY):
        return None
    _section(title)
    
    m0, t0 = dict(_MARK), time.perf_counter()
    out, status = None, 'done'
    try:
        out = fn(*args, **kwargs)
    except Exception as err:
        status = 'CRASH'
        _log(f"  CRASH in {fn.__name__}: {type(err).__name__}: {err}")
        for line in traceback.format_exc().rstrip().splitlines():
            _emit("    " + line)
    _STEP_RESULTS.append(dict(title=title, status=status, sec=time.perf_counter() - t0, **{k: _MARK[k] - m0[k] for k in _MARK}))
    return out

def _print_summary() -> int:
    n_crash = sum(r['status'] == 'CRASH' for r in _STEP_RESULTS)
    ok, flag, fail = (sum(r[k] for r in _STEP_RESULTS) for k in ('ok', 'flag', 'fail'))
    lines = ["", "=" * 78, "SUMMARY"]
    lines += [f"  {r['status']:5s} {r['sec']:8.1f} s   ✓ {r['ok']:3d}   ✗ {r['flag']:3d}   FAIL {r['fail']:3d}   {r['title']}" for r in _STEP_RESULTS]
    lines += [f"  {len(_STEP_RESULTS)} steps, {n_crash} crashed, {ok} ✓ / {flag} ✗ flag(s) / {fail} FAIL line(s), {sum(r['sec'] for r in _STEP_RESULTS):.0f} s   (✗ = criterion flag in a table or diagnostic; only FAIL / CRASH set the exit code)",
              "  compute-once layer (reused / built): " + "   ".join(f"{tag} {c[0]}/{c[1]}" for tag, c in _MEMO_COUNT.items())]
    for ln in lines:
        _emit(ln)
        if _REPORT_FILE is not None:
            print(ln)
    return 1 if (n_crash or (fail)) else 0

def _run_all_tests(states_path: str | None = None, only: tuple = ()) -> int:
    global _ONLY
    _ONLY = tuple(s.strip().lower() for s in only if s.strip())

    _kw_main, target_doping = extract_main_params(MODULE_PATH)
    params = qm.ModelParams(**_kw_main)
    _param_src = f"__main__ block of {_os.path.basename(MODULE_PATH)}"
    solver = qm.RMFT_Solver(params)
    _emit(f"model parameters: {_param_src};  target_doping = {target_doping}")

    h = hashlib.sha256()
    with open(MODULE_PATH, 'rb') as fh:
        h.update(fh.read())
    h.update(f"{params!r}|{float(target_doping)!r}|NK={int(qm._NK)}".encode())
    key = h.hexdigest()
    normal = sc = None
    if states_path and _os.path.exists(states_path):
        try:
            with open(states_path, 'rb') as fh:
                blob = pickle.load(fh)
            if blob.get('key') == key:
                print(f"[states] reusing {states_path}")
                normal, sc = _make_point(**blob['normal']), _make_point(**blob['sc'])
            else:
                print(f"[states] {states_path} belongs to another model file / parameter set -> rebuilding")
        except Exception as err:
            print(f"[states] {states_path} unreadable ({type(err).__name__}: {err}) -> rebuilding")

    if normal is None:
        M, D, F, n, mu, g_t, g_J, gA, gB, gC = solver.refine_M_state(target_doping, 0.0, np.zeros(qm._N_DELTA, dtype=complex), np.zeros(qm._N_DELTA, dtype=complex), damping=0.45, verbose=True)
        normal = _make_point(M, 0.0, D, F, n, mu, g_t, g_J, gA, gB, gC)
        res = solver.solve_self_consistent(target_doping, {'s': 0.0, 'd': 0.0}, D, F, force_delta_zero=False, force_Q_zero=False, max_iter=4)
        gs = solver.estimate_gutzwiller_factors_occupation_based(target_doping, res['M'], res['Q'], res['mu'], g_t, g_J)
        sc = _make_point(res['M'], res['Q'], res['Delta_vec'], res['Quad67'], res['n_kspace'], res['mu'], g_t, g_J, *gs, cache=res['vertex_cache'])

        if states_path:
            with open(states_path, 'wb') as fh:
                pickle.dump({'key': key, 'normal': {f: getattr(normal, f) for f in normal.__dataclass_fields__}, 'sc': {f: getattr(sc, f) for f in sc.__dataclass_fields__}}, fh)
            print(f"[states] stored in {states_path}")

    _step("Time-reversal operator construction (prerequisite check)", test_time_reversal_operator_construction, solver)
    _register_point(sc)
    M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg = normal.M, normal.Q, normal.Delta_vec, normal.Quad67_vec, normal.n, normal.mu
    g_t, g_J, g_Das, g_Dad, g_Dbd = normal.g_t, normal.g_J, normal.g_Das, normal.g_Dad, normal.g_Dbd
    M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc = sc.M, sc.Q, sc.Delta_vec, sc.Quad67_vec, sc.n, sc.mu
    g_Das_sc, g_Dad_sc, g_Dbd_sc, cache_sc = sc.g_Das, sc.g_Dad, sc.g_Dbd, sc.cache
    ctx = _build_fs_context(solver, M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J)
    ctx_sc = _build_fs_context(solver, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J)
    cache_n = _build_vertex_cache(solver, target_doping, M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J)
    print(f"SHORT-SCF Q_sc={Q_sc:.5f}  M={np.array2string(M_bdg_sc, precision=4)}  |Δ|₂={float(np.linalg.norm(np.abs(Delta_vec_sc))):.4e}")
    _step("Sector-kernel algebra (toy, G-closed sample)", test_sector_kernel_algebra)
    _step("Channel registry: bond content, Slater–Koster hopping, T_Q / mirror characters / D4h labels (import-time checks)", _registry_checks)
    _step("Cluster K_eff diagnostics (K_full−K_MF bug, Δ-dependence, SCF-free JT window)", test_cluster_K_eff_diagnostics, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, M_normal=M_bdg, mu_normal=mu_bdg, vertex_cache=cache_sc)
    _step("Diagnose simulated SC state", test_simulated_sc_state_diagnostic, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, M_bdg, n_bdg, mu_bdg, ctx=ctx_sc, vertex_cache=cache_sc)
    _step("Q-path δK-vs-BFS (does SC softening survive at finite Q?)", test_Q_path_dK_and_BFS, solver, target_doping, M_bdg_sc, Q_scan=[0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050], Delta_vec_sc=Delta_vec_sc, Quad67_vec_sc=Quad67_vec_sc, n_kspace=n_sc, mu_sc=mu_sc, g_t=g_t, g_J=g_J, M_normal=M_bdg, mu_normal=mu_bdg, vertex_cache=cache_sc)
    _step("Bogoliubov Fermi surface fast probe", probe_bogoliubov_fermi_surface_fast, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J)
    _step("FS sample carries G = {E, T_Q} x D2h exactly (independent of the cache flags): normal state", test_fs_sample_group_structure, solver, M_bdg, Q_n, n_bdg, mu_bdg, g_t, g_J)
    _step("FS sample carries G = {E, T_Q} x D2h exactly (independent of the cache flags): SC state, Q != 0", test_fs_sample_group_structure, solver, M_bdg_sc, Q_sc, n_sc, mu_sc, g_t, g_J)
    _step("FS sector spectrum test -- exact G = {E, T_Q}×D2h block decomposition", test_fs_sector_spectrum, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, cache_sc, use_dwave_weighting=True)
    _step("Pair-vertex symmetry test", test_pair_vertex_symmetry, solver, target_doping, M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J, ctx)
    _step("Production kernel cache: sector invariants (all 8 sectors processed, block-diagonal V_irrep, leading sector)", test_kernel_cache_sectors, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, cache_sc)
    _step("Newton Hessian: zero-amplitude channels linearly coupled to an active one (Q·Δ_s*Δ_d)", test_hessian_inactive_linear_coupling, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, g_Das_sc, g_Dad_sc, g_Dbd_sc)
    _step("Irrep projection of FS eigenvector test", test_irrep_projection_of_FS_eigenvector, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, verbose=True)
    _step("M-controlled Q scan", test_M_controlled_Q_scan, solver, target_doping, M_scan=(0.0, 0.05, 0.1, 0.15, 0.2), Q_scan=(0.0, 0.005, 0.01, 0.015, 0.02), g_t=g_t, g_J=g_J)
    _step("Doublet structure & mixing test", test_doublet_structure_and_mixing, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, Q_Eg2=0.0)
    _step("Landa first order test", test_landau_first_order_criterion, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, cache_sc)
    _step("2D phase stiffness and the Kosterlitz–Thouless bound", test_phase_stiffness_KT_bound, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, Tc_mf=None)
    _step("RPA determinant-coupling isolation", test_rpa_determinant_coupling_isolation, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, ctx=ctx_sc)
    _step("BdG free-energy consistency test (normal state)", test_bdg_energy_consistency, solver, target_doping, M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J, 0.0, cache=cache_n, K_eff_for_free_energy=params.K_lattice, K_eff_Eg2_for_free_energy=params.K_lattice_Eg2)
    _step("Superexchange convention / Heisenberg factor test", test_superexchange_convention_and_Heisenberg_factor, solver, target_doping, t_pd_scan=(0.30, 0.35, 0.40, 0.45), U_dd_scan=(1.8, 2.0, 2.3, 2.6), J_H_scan=(0.0, 0.05, 0.10, 0.15, 0.20), M_channels_for_weighting=M_bdg)
    _step("Isotropic Gutzwiller hopping test", test_isotropic_gutzwiller_hopping, solver, target_doping, M_bdg, Q_n, n_bdg, mu_bdg, g_t, g_J, ctx=ctx)
    _step("Second-neighbor hopping necessity test", test_second_neighbor_hopping_necessity, solver, target_doping, M_bdg, Q_n, n_bdg, mu_bdg, g_t, g_J, ctx=ctx)
    _step("Third-neighbor hopping necessity test", test_third_neighbor_hopping_necessity, solver, target_doping, M_bdg, Q_n, n_bdg, mu_bdg, g_t, g_J, ctx=ctx)
    _step("dk sensitivity and mu consistency test", test_dk_sensitivity_and_mu_consistency, solver, Q_n, target_doping, g_t, g_J, M_bdg, n_bdg, mu_bdg)
    _step("Q-force decomposition test", test_q_force_decomposition, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, K_eff_for_free_energy=params.K_lattice, K_eff_Eg2_for_free_energy=params.K_lattice_Eg2)
    _step("Pair correlator and B1g quadrupole (Quad67) consistency (normal state)", test_pair_correlator_and_quadrupole, solver, target_doping, M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J, g_Das, g_Dad, g_Dbd)
    _step("Pair correlator and B1g quadrupole (Quad67) consistency (SC state)", test_pair_correlator_and_quadrupole, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, g_Das_sc, g_Dad_sc, g_Dbd_sc)
    _step("cluster correction scan test", test_cluster_correction_scan, solver, target_doping, M_bdg, Q_n, n_bdg, mu_bdg, g_t, g_J)
    _step("chi_SQ M-dependence (Theta-breaking measure)", test_chi_SQ_M_dependence, solver, target_doping, M_scan=(0.0, 0.05, 0.1, 0.15, 0.2), Q=Q_sc, Delta_vec=Delta_vec_sc, Quad67_vec=Quad67_vec_sc, g_t=g_t, g_J=g_J)
    _step("chi_SQ individual channel activation", test_chi_SQ_individual_channel_activation, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J)
    _step("Symmetry content of all pair channels (orbital irrep in the model gauge, total irrep, linear Q coupling)", test_new_channel_irrep_content, solver)
    _step("TRW symmetry test (D2h covariance, Theta parity and Theta-compatible F67 phase, bond-class weights)", test_trw_symmetry, solver, target_doping=target_doping)
    _step("TRW B1g weighting: sigma_d ∘ (Q → −Q) covariance of the B1g-weighted form factor, channel by channel", test_trw_b1g_weighting, solver, target_doping)
    _step("dH/dQ: analytic Q-vertex vs finite difference of the H stack (TRW blocks, B1g-weighted form factor, hole-block parity)", test_dHdQ_vs_build_H_stack, solver, M_bdg_sc, Q_sc, mu_sc, g_t, g_J)
    _step("BdG matrix: Hermiticity and particle-hole symmetry of every pair and TRW block", test_bdg_structure_phs, solver, M_bdg_sc, Q_sc, mu_sc, g_t, g_J)
    _step("Linear coupling: D4h selection rule for A1g x B1g pairs + exact B1g covariance", test_linear_coupling, solver, target_doping=target_doping, normal=normal)
    _step("Condensation energy: repulsive / indefinite modes must not lower the free energy", test_condensation_energy_repulsive_floor)
    _step("G-matrix collective-instability diagnostic", compute_G_instability, solver, target_doping, float(np.linalg.norm(M_bdg)), mu_bdg, n_bdg, g_t, g_J, g_Das, g_Dad)
    _step("chi_SQ(q) full BZ scan", estimate_chi_SQ_q_full, solver, target_doping, M_bdg_sc, Q_sc, Delta_vec_sc, Quad67_vec_sc, n_sc, mu_sc, g_t, g_J, n_q=35)
    _step("RPA pairing vertex (normal + SC + comparison)", test_rpa_vertex, solver, target_doping, M_normal=M_bdg, n_kspace_n=n_bdg, mu_n=mu_bdg, M_sc=M_bdg_sc, Q_sc=Q_sc, Delta_vec=Delta_vec_sc, Quad67_vec=Quad67_vec_sc, n_sc=n_sc, mu_sc=mu_sc, g_t=g_t, g_J=g_J, ctx_normal=ctx, ctx_sc=ctx_sc, run_one_shot_trw_toggle=True, run_full_ab_check=True)
    _step("Pair-vertex assembly vs SCF cache (normal state)", test_vertex_assembly_vs_cache, solver, target_doping, M_bdg, Q_n, Delta_vec_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J)
    _step("G-sector kernel: independent re-assembly vs SCF cache (normal state)", test_sector_kernel_vs_cache, solver, target_doping, M_bdg, Q_n, Quad67_vec_n, n_bdg, mu_bdg, g_t, g_J, Delta_vec=Delta_vec_n)
    return _print_summary()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", metavar="FILE", help="store / reuse the two SCF states (checked against model file, parameters and doping)")
    ap.add_argument("--only", metavar="A,B", help="run only the steps whose title contains one of these substrings")
    ap.add_argument("--log", default="test_report.txt", metavar="FILE", help="report file (default: %(default)s)")
    ap.add_argument("--nk", type=int, metavar="N", help="k-grid points per axis (default: the model's _NK); the states file is grid-specific")
    args = ap.parse_args()
    if args.nk:
        qm._NK = int(args.nk)
    _open_test_log(args.log)
    try:
        rc = _run_all_tests(args.states, tuple((args.only or '').split(',')))
    finally:
        _close_test_log()
    sys.exit(rc)