import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import opt_einsum as oe
from scipy.linalg import eigh as _scipy_eigh
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from dataclasses import dataclass, replace
from typing import Tuple, Dict, List, Optional, Callable, Union
import warnings
import copy
import gc
import time as _time
import math
import concurrent.futures
import threading as _threading
import matplotlib.pyplot as plt

_log_lock = _threading.Lock()

# ── Pairing-irrep basis ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Irrep:
    name:   str
    d4h:    str                # D4h irrep label ('A1g', 'B1g', 'Eu', ...)
    kind:   str                # 'onsite' | 'nn_AB' | 'nnn_diag' | 'nnn_axis'
    parity: int                # φ(-k) = parity · φ(k)   (+1 even, −1 odd)
    ff:     object             # callable(kx, ky) -> ndarray
    scale:  float              #

def _irrep_character(ff, n_probe: int = 9, seed: int = 7) -> tuple:
    """(t, mx, my) character of a form factor under G, MEASURED numerically from φ(g·k)/φ(k) at generic points (no hand-assigned labels that could be wrong)."""
    rng = np.random.default_rng(seed)
    kx = rng.uniform(-np.pi, np.pi, n_probe)
    ky = rng.uniform(-np.pi, np.pi, n_probe)
    base = np.asarray(ff(kx, ky), dtype=float)
    ok = np.abs(base) > 1e-6
    if not np.any(ok):
        raise ValueError("form factor vanishes at every probe point")
    out = []
    for gen in ((-1, 1, 1), (1, -1, 1), (1, 1, -1)):
        gx, gy = _g_apply(gen, kx, ky)
        ratio = np.asarray(ff(gx, gy), dtype=float)[ok] / base[ok]
        if not (np.all(np.abs(ratio - 1.0) < 1e-9) or np.all(np.abs(ratio + 1.0) < 1e-9)):
            raise ValueError("form factor is not a 1-D character of {E, T_Q} x D2h")
        out.append(int(np.sign(ratio[0])))
    return tuple(out)

def _g_apply(op: tuple, kx, ky):
    """Exact continuous-momentum action g·k = (mx·kx + s, my·ky + s), s = π for the T_Q-containing elements (no lattice rounding)."""
    t, mx, my = op
    sh = np.pi if t == -1 else 0.0
    return mx * kx + sh, my * ky + sh

def _swap_parity(ff, n_probe: int = 9, seed: int = 7) -> int:
    """Parity (+1 / −1) of a form factor under the diagonal mirror σd: (kx, ky) → (ky, kx), MEASURED numerically from φ(ky, kx)/φ(kx, ky) at generic points.
    0 if φ is not a σd eigenfunction (the px / py partners are exchanged by σd). σd is not in G (the kernel need not be C4-covariant), it only resolves the D4h irreps that share a G-sector."""
    rng = np.random.default_rng(seed)
    kx = rng.uniform(-np.pi, np.pi, n_probe)
    ky = rng.uniform(-np.pi, np.pi, n_probe)
    base = np.asarray(ff(kx, ky), dtype=float)
    ok = np.abs(base) > 1e-6
    if not np.any(ok):
        raise ValueError("form factor vanishes at every probe point")
    ratio = np.asarray(ff(ky, kx), dtype=float)[ok] / base[ok]
    if np.all(np.abs(ratio - 1.0) < 1e-9):
        return 1
    if np.all(np.abs(ratio + 1.0) < 1e-9):
        return -1
    return 0

# ── Harmonic library ──────────────────────────────────────────────────────────
# The library is the definition of the basis. the first field is the D4h irrep of the form factor φ_i(k) ALONE
#   kind  : 'onsite'   -> same-sublattice (A–A / B–B), k-independent, T_Q-even
#           'nn_AB'    -> inter-sublattice (A–B / B–A): bonds with ODD |δx|+|δy|   <=> T_Q-ODD
#           'nnn_diag' -> same-sublattice, bonds (±n,±n), even |δx|+|δy|           <=> T_Q-EVEN
#           'nnn_axis' -> same-sublattice, bonds (±n,0), (0,±n), even |δx|+|δy|   <=> T_Q-EVEN
_HARMONIC_LIBRARY = {
    # ── D4h A1g :  even-k, even under kx↔ky, T_Q-even ──────────────────────────
    's':   ('A1g', 'onsite',   +1, lambda kx, ky: np.ones_like(kx),                                              1.0),
    'es':  ('A1g', 'nn_AB',    +1, lambda kx, ky: np.cos(kx) + np.cos(ky),                                       2.0),
    'es2': ('A1g', 'nnn_axis', +1, lambda kx, ky: np.cos(2.0 * kx) + np.cos(2.0 * ky),                           2.0),
    'es3': ('A1g', 'nn_AB',    +1, lambda kx, ky: np.cos(kx) * np.cos(2.0 * ky) + np.cos(2.0 * kx) * np.cos(ky), 2.0),
    'cc':  ('A1g', 'nnn_diag', +1, lambda kx, ky: np.cos(kx) * np.cos(ky),                                       1.0),
    'cc2': ('A1g', 'nnn_diag', +1, lambda kx, ky: np.cos(2.0 * kx) * np.cos(2.0 * ky),                           1.0),
    'es4': ('A1g', 'nnn_axis', +1, lambda kx, ky: np.cos(4.0 * kx) + np.cos(4.0 * ky),                           2.0),
    'cc3': ('A1g', 'nnn_diag', +1, lambda kx, ky: np.cos(3.0 * kx) * np.cos(3.0 * ky),                           1.0),

    # ── D4h A2g :  even-k, odd under kx↔ky, T_Q-odd ────────────────────────────
    'gA2': ('A2g', 'nn_AB',    +1, lambda kx, ky: np.sin(kx) * np.sin(ky) * (np.cos(kx) - np.cos(ky)),           1.0),
    'gA3': ('A2g', 'nn_AB',    +1, lambda kx, ky: np.sin(kx) * np.sin(ky) * (np.cos(3.0 * kx) - np.cos(3.0 * ky)), 1.0),

    # ── D4h B1g :  even-k, x²−y², T_Q-odd ─────────────────────────────────────
    'd':   ('B1g', 'nn_AB',    +1, lambda kx, ky: np.cos(kx) - np.cos(ky),                                       2.0),
    'd2':  ('B1g', 'nnn_axis', +1, lambda kx, ky: np.cos(2.0 * kx) - np.cos(2.0 * ky),                           2.0),
    'd3':  ('B1g', 'nn_AB',    +1, lambda kx, ky: np.cos(kx) * np.cos(2.0 * ky) - np.cos(2.0 * kx) * np.cos(ky), 2.0),

    # ── D4h B2g :  even-k, xy, T_Q-even ───────────────────────────────────────
    'g':   ('B2g', 'nnn_diag', +1, lambda kx, ky: np.sin(kx) * np.sin(ky),                                       1.0),
    'g2':  ('B2g', 'nnn_diag', +1, lambda kx, ky: np.sin(2.0 * kx) * np.sin(2.0 * ky),                           1.0),
    'g3':  ('B2g', 'nnn_diag', +1, lambda kx, ky: np.sin(3.0 * kx) * np.sin(3.0 * ky),                           1.0),
    'g4':  ('B2g', 'nnn_diag', +1, lambda kx, ky: np.sin(4.0 * kx) * np.sin(4.0 * ky),                           1.0),
    'g5':  ('B2g', 'nnn_diag', +1, lambda kx, ky: np.sin(5.0 * kx) * np.sin(5.0 * ky),                           1.0),
    'gB2': ('B2g', 'nn_AB',    +1, lambda kx, ky: np.sin(kx) * np.sin(ky) * (np.cos(kx) + np.cos(ky)),           1.0),
    'gB3': ('B2g', 'nn_AB',    +1, lambda kx, ky: np.sin(kx) * np.sin(ky) * (np.cos(3.0 * kx) + np.cos(3.0 * ky)), 1.0),

    # ── D4h Eu (x-partner), T_Q-odd, mx=-1, my=+1  (sector (-,-,+)) ───────────
    'px':  ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(kx),                                                    1.0),
    'px4': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(kx) * np.cos(2.0 * ky),                                 1.0),
    'px5': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(3.0 * kx),                                              1.0),
    'px6': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(5.0 * kx),                                              1.0),
    'px7': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(kx) * np.cos(4.0 * ky),                                 1.0),
    'px8': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(3.0 * kx) * np.cos(2.0 * ky),                           1.0),

    # ── D4h Eu (y-partner), T_Q-odd, mx=+1, my=-1  (sector (-,+,-)) ───────────
    'py':  ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(ky),                                                    1.0),
    'py4': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.cos(2.0 * kx) * np.sin(ky),                                 1.0),
    'py5': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(3.0 * ky),                                              1.0),
    'py6': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.sin(5.0 * ky),                                              1.0),
    'py7': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.cos(4.0 * kx) * np.sin(ky),                                 1.0),
    'py8': ('Eu',  'nn_AB',    -1, lambda kx, ky: np.cos(2.0 * kx) * np.sin(3.0 * ky),                           1.0),

    # ── D4h Eu (x-partner), T_Q-even, mx=-1, my=+1  (sector (+,-,+)) ──────────
    'px2': ('Eu',  'nnn_axis', -1, lambda kx, ky: np.sin(2.0 * kx),                                              1.0),
    'px3': ('Eu',  'nnn_diag', -1, lambda kx, ky: np.sin(kx) * np.cos(ky),                                       1.0),
    'px9': ('Eu',  'nnn_axis', -1, lambda kx, ky: np.sin(4.0 * kx),                                              1.0),
    'pxA': ('Eu',  'nnn_diag', -1, lambda kx, ky: np.sin(2.0 * kx) * np.cos(2.0 * ky),                           1.0),
    'pxB': ('Eu',  'nnn_diag', -1, lambda kx, ky: np.sin(kx) * np.cos(3.0 * ky),                                 1.0),

    # ── D4h Eu (y-partner), T_Q-even, mx=+1, my=-1  (sector (+,+,-)) ──────────
    'py2': ('Eu',  'nnn_axis', -1, lambda kx, ky: np.sin(2.0 * ky),                                              1.0),
    'py3': ('Eu',  'nnn_diag', -1, lambda kx, ky: np.cos(kx) * np.sin(ky),                                       1.0),
    'py9': ('Eu',  'nnn_axis', -1, lambda kx, ky: np.sin(4.0 * ky),                                              1.0),
    'pyA': ('Eu',  'nnn_diag', -1, lambda kx, ky: np.cos(2.0 * kx) * np.sin(2.0 * ky),                           1.0),
    'pyB': ('Eu',  'nnn_diag', -1, lambda kx, ky: np.cos(3.0 * kx) * np.sin(ky),                                 1.0),
}
# Every library entry is an active pairing channel — uniform treatment, no filter.
_PAIR_CHANNELS = tuple(_Irrep(_name, *_HARMONIC_LIBRARY[_name]) for _name in _HARMONIC_LIBRARY)

_IRREP_NAMES  = tuple(c.name      for c in _PAIR_CHANNELS)
_IRREP_D4H    = {c.name: c.d4h    for c in _PAIR_CHANNELS}
_IRREP_KIND   = {c.name: c.kind   for c in _PAIR_CHANNELS}
_IRREP_PARITY = {c.name: c.parity for c in _PAIR_CHANNELS}
_IRREP_FF     = {c.name: c.ff     for c in _PAIR_CHANNELS}

_N_IRREP              : int = len(_IRREP_NAMES)
# canonical positions WITHIN the irrep basis (0.._N_IRREP-1): the index space of V_irrep, S_irrep, v_raw, frac. Distinct from the Delta_vec/F67_vec position constants below. Derived by name, so re-ordering _PAIR_CHANNELS cannot desynchronise them.
_IR_S, _IR_ES, _IR_D, _IR_PX, _IR_PY = (_IRREP_NAMES.index(_n) for _n in ('s', 'es', 'd', 'px', 'py'))
_IR_G, _IR_PX2, _IR_PY2 = (_IRREP_NAMES.index(_n) if _n in _IRREP_NAMES else -1 for _n in ('g', 'px2', 'py2'))

# ── Symmetry group of the collinear-AFM normal state, as it acts on the pair amplitude Δ(k):  G = {E, T_Q} × {E, kx→−kx, ky→−ky, k→−k}  (abelian, order 8;  "D2h" in the comments below always means this k-space Z2×Z2) ────────────────
# Every gap function with a definite symmetry is a 1-D character χ = (t, mx, my) ∈ {±1}³ of G: t = parity under the AFM translation T_Q: k→k+(π,π), mx / my = parity under kx→−kx / ky→−ky.
# What G is, and what it is not:
#  • The magnetic space group of the collinear Néel state is of Shubnikov type IV. Its magnetic point group is 4/mm'm' (order 16: 8 unitary + 8 antiunitary elements) for M ∥ z, Q = 0 and no static B1g field, and m'm'm (order 8) as soon as
#    Q ≠ 0 or Δ_B1g_static ≠ 0 (an in-plane moment would break C4 through the SOC as well).  M ∥ z by itself does NOT break C4.
#  • T_Q is not a point-group element: it is the translation by one lattice vector combined with the sublattice exchange A↔B (the magnetic anti-translation Θ = T·τ_AB carries the time reversal).  Θ is not an element of G; what it implies for the
#    normal state is implemented explicitly where it acts: the T_Q parity t of every pair channel (this table) and χ_SQ = 0 / the vanishing spin–JT cross vertex of the normal state (_rpa_closure, _scan_incommensurate_nesting).
#  • G is the group that acts on the pair amplitude inside the diamond MBZ (the fundamental domain of T_Q on which the FS sample lives). mx, my are the k-space actions of the (primed) mirrors σx, σy, their product that of C2z (≙ inversion for even Δ);
#    the 8 antiunitary elements act on the real symmetric kernel only through |q_x|, |q_y| (_unique_q_pairs, the folding of q in the sector kernel).  C4 and σd (kx↔ky) are symmetries of the Q = 0, Δ_B1g = 0 normal state but are NOT in G: a character of G
#    labels a SECTOR, not a D4h irrep.  Sectors with several D4h irreps:  (+,+,+): A1g {s, es2, cc, ...} ⊕ B1g {d2};  (−,+,+): A1g {es, es3} ⊕ B1g {d, d3};  (−,−,−): A2g {gA2, gA3} ⊕ B2g {gB2, gB3};  'g' (B2g) is alone in (+,−,−);
#    px / py each have their own sector.  The irreps inside a sector are separated by the σd parity of the sample (cache keys sector_swap_lambda / sector_swap_defect); that split is exact only if the kernel is C4-covariant.
_G_OPS : tuple = tuple((t, mx, my) for t in (1, -1) for mx in (1, -1) for my in (1, -1))   # identity first; op = (T_Q applied?, kx→−kx?, ky→−ky?)
_IRREP_CHAR = {c.name: _irrep_character(c.ff) for c in _PAIR_CHANNELS}                   # name -> (t, mx, my)
_LIB_CHAR   = {n: _irrep_character(v[3]) for n, v in _HARMONIC_LIBRARY.items()}          # same for every library harmonic
_IRREP_SWAP = {c.name: _swap_parity(c.ff) for c in _PAIR_CHANNELS}                       # name -> σd (kx↔ky) parity, +1 / −1; 0 for the px/py partners (not swap eigenfunctions)
_LIB_SWAP   = {n: _swap_parity(v[3]) for n, v in _HARMONIC_LIBRARY.items()}              # same for every library harmonic
_D4H_BY_MX_MY_SWAP = {(1, 1, 1): 'A1g', (1, 1, -1): 'B1g', (-1, -1, 1): 'B2g', (-1, -1, -1): 'A2g'}   # D4h irrep selected by (mx, my, σd) in the even-parity (mx = my) sectors; checked against the library in _check_registry
_G_CHI = np.array([[(t if a == -1 else 1.0) * (mx if b == -1 else 1.0) * (my if c == -1 else 1.0) for (a, b, c) in _G_OPS] for (t, mx, my) in _G_OPS], dtype=float)   # (sector, op) character table, sector key = (t, mx, my) in _G_OPS order
_SECTOR_MEMBERS = {key: [i for i, n in enumerate(_IRREP_NAMES) if _IRREP_CHAR[n] == key] for key in _G_OPS}   # sector -> basis positions it contains (empty list: sector NOT covered by the basis)

# ── Delta_vec / F67_vec layout ───────────────────────────────────────────────
_N_CH67               : int = 2                                                        # Gamma6-Gamma7a and Gamma6-Gamma7b sectors
_N_DELTA              : int = _N_CH67 * _N_IRREP                                       # 2 · _N_IRREP (currently 2 · 42 = 84)
_DELTA_NAMES = (tuple(f'{n}7a' for n in _IRREP_NAMES) + tuple(f'{n}7b' for n in _IRREP_NAMES))
_IDX_7A_ARR = np.array([_DELTA_NAMES.index(f'{n}7a') for n in _IRREP_NAMES])
_IDX_7B_ARR = np.array([_DELTA_NAMES.index(f'{n}7b') for n in _IRREP_NAMES])
_DELTA_IDX = {name: i for i, name in enumerate(_DELTA_NAMES)}

# ── Orbital / BdG basis size ──────────────────────────────────────────────────
_N_ORB                : int   = 6                                                      # orbital flavors [Γ6↑,Γ6↓,Γ7a↑,Γ7a↓,Γ7b↑,Γ7b↓]: full 3-Kramers-doublet CF+SOC manifold, no downfolding.
_N_BDG                : int   = 4 * _N_ORB                                             # 24: BdG dimension = 2 sublattices (A,B) × particle/hole × N_ORB.
_CHANNEL_ORB_IDX      : tuple = ((0, 1), (2, 3), (4, 5))                               # orbital indices per channel in the 6-dim basis

# ── Cluster (local irreducible vertex) ────────────────────────────────────────
_N_CHANNELS           : int   = 3                                                      # Channel-wise (Γ6, Γ7a, Γ7b) order parameter resolution
_CLUSTER_SIZE         : int   = 4                                                      # 2x2 plaquette
_N_CLUSTER            : int   = _N_ORB ** _CLUSTER_SIZE

# ── Real-space bond content of a pairing form factor φ(k) = Σ_δ c_δ e^{i k·δ} (registry-driven; weights the transverse Weiss field TRW of every channel and validates the declared bond kind) ──────────
_Z                    : int   = 4                                                      # metal's coordination number in 2D square lattice
_BOND_GRID            : int   = 16                                                     # exact for bonds with |δ_x|, |δ_y| < 8 (the library reaches 5)
_SIGN_M = (+1.0, -1.0, +1.0, -1.0)
_BONDS  = ((0, 1, +1.0), (1, 2, -1.0), (2, 3, +1.0), (3, 0, -1.0))
_Z_EFF   = _Z - 2.0 * len(_BONDS) / _CLUSTER_SIZE                                      # intra-cluster NN bonds per site (2 for the 2x2 plaquette) must be subtracted to get external NN neighbours per site

_NORMAL_SECTOR_PAIRS  : tuple = (
    (slice(0,         _N_ORB),     slice(0,         _N_ORB)),                          # A-A particle
    (slice(  _N_ORB,  2*_N_ORB),   slice(  _N_ORB,  2*_N_ORB)),                        # B-B particle
    (slice(2*_N_ORB,  3*_N_ORB),   slice(2*_N_ORB,  3*_N_ORB)),                        # A-A hole
    (slice(3*_N_ORB,  _N_BDG),     slice(3*_N_ORB,  _N_BDG)),                          # B-B hole
    (slice(0,         _N_ORB),     slice(  _N_ORB,  2*_N_ORB)),                        # A-B particle
    (slice(  _N_ORB,  2*_N_ORB),   slice(0,         _N_ORB)),                          # B-A particle
    (slice(2*_N_ORB,  3*_N_ORB),   slice(3*_N_ORB,  _N_BDG)),                          # A-B hole
    (slice(3*_N_ORB,  _N_BDG),     slice(2*_N_ORB,  3*_N_ORB)),                        # B-A hole
)                                                                                      # Nambu sector pairs for normal-state (Δ=0) Lindhard sum; anomalous Part↔Hole pairs vanish at Δ=0

# ── Pairing-kernel construction mode ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_SECTOR_NEAR_FRAC      : float = 0.25               # covered sectors with λ_top ≥ this fraction of the leading covered λ_top are "near-degenerate": their eigenvector shapes also seed the kick (frac_seed)
_V_REPULSIVE_FLOOR_FRAC: float = 1e-4               # eigenvalues of the (FS-weighted) pairing matrix below this fraction of its largest one count as 'not attractive'
_GRAM_RCOND            : float = 1e-12              # relative floor of the FS-weighted Gram-matrix eigenvalues in the attractive/non-attractive split
_BDY_TOL               : float = 1e-9               # |(|kx|+|ky|) − π| below this: the point lies on the T_Q-paired diamond boundary
_HESS_DELTA_EPS_MIN    : float = 1e-5               # floor of the |Δ| finite-difference step (F is even in Δ, the stencil may straddle Δ = 0)
_HESS_INACTIVE_EPS     : float = 4e-4               # FD step for the amplitudes that are exactly 0 in compute_dF_dDelta_and_d2F (two-point fit of gradient AND curvature)
_HESS_LINEAR_TOL       : float = 2e-5               # |∂F/∂|Δ_i|| above which a zero-amplitude channel counts as linearly coupled to an active one (mixed derivatives are then evaluated)
_SECTOR_SCAN_MIN_GAIN  : float = 0.02               # library scan: a candidate harmonic is suggested only if it raises the resolved weight of the sector eigenvector by at least this much

# ── NOT free parameters ───────────────────────────────────────────────────────
_EV_TO_K              : float = 11604.518121        # 1 eV in Kelvin
_GW_G_J_PREFACTOR     : float = 4.0                 # g_J = 4/(1+δ)²  Gutzwiller exchange renormalization prefactor (slave-boson / Kotliar-Ruckenstein derivation, half-filling limit)
_GW_G_T_NUMERATOR     : float = 2.0                 # g_t = 2δ/(1+δ)  Gutzwiller hopping prefactor numerator coefficient

# ── Fermi surface sampling and k-grid ─────────────────────────────────────────
_NK                   : int   = 60                  # k-grid points per direction (even required for commensurate q_AFM=(π,π))
_FS_SAMPLING          : float = 4.4                 # integration window around the Fermi level
_FS_THERMAL_THRESHOLD : float = 0.0025              # 1% of peak value of f(E)*(1-f(E)) = 1/4, used as baseline for thermal FS weight
_FS_ORBIT_TIE_RTOL    : float = 1e-9                # relative score difference below which two FS orbits are TIED (degenerate by a symmetry of the spectrum) and enter / leave the sample together
_VERTEX_DIAG_MIN_FS   : int   = 10                  # minimum FS points required for V_mat structure diagnostics; below this std/mean statistics are unreliable.
_NODAL_REGION_PCTL    : int   = 25                  # lower and upper 25% percentile for node gap estimation
_Q_UNIQUE_SCALE       : int   = 100000              # integer scaling factor for unique q pairs
_BZ_NORM              : float = (2.0 * np.pi) ** 2  # BZ area in reduced coordinates (a=1, ħ=1): area = (2π)², FS arc-length integration measure is dl/((2π)²·vF)
_FS_CACHE_TOL         : float = 1e-3                # parameter-change threshold for FS-points cache invalidation

# ── Tc diagnostics ────────────────────────────────────────────────────────────
_BCS_RATIO_STRONG     : float = 3.8                 # 2Δ/kTc > this → strong coupling
_BCS_RATIO_EXOTIC     : float = 6.0                 # 2Δ/kTc > this → exotic / non-phononic
_BCS_RATIO_S_WAVE     : float = 3.528               # weak-coupling BCS, isotropic gap: 2Δ/kTc = 2π/e^γ. The three thresholds above are calibrated to this baseline
_GAP_NODAL_ANISO      : float = 0.25                # min/max spectral FS gap below this → gap has (near-)nodes (diagnostic label only)
_GAP_REF_TOP_FRAC     : float = 0.10                # Δ_ref = FS-weighted mean of the largest this-fraction of the spectral FS gap (robust FS maximum)
_GAP_N_SUB            : int   = 60                  # FS points (farthest-point sampled: uniform coverage of the FS contour) on which the spectral gap is searched; one line search per point
_GAP_WIN              : float = 0.030               # eV: minimum line-search half-window along the FS normal (converted to k through |v_F|, capped at _GAP_WIN_KMAX); widened to 3·max|Δ_vec|
_GAP_SCENTER_MAX      : float = 2.0                 # rad: cap of the first-order shift s0 = −ξ/|v_F| from a sample point to the normal-state FS crossing
_GAP_WIN_KMAX         : float = 2.0                 # rad: cap of that line-search half-window
_GAP_VF_MIN           : float = 1.0e-5              # eV·(rad⁻¹): FS points with |v_F| below this have no usable normal direction and are EXCLUDED from the gap statistics
_GAP_NLINE            : int   = 1201                # points of the line search per FS point
_GAP_BFS_TOL          : float = 1.0e-4              # eV: FS points with min|E| below this count as gapless (Bogoliubov Fermi surface)
_ODD_VTX_TOL          : float = 1e-9                # relative tolerance of the p-channel (odd-parity) vertex validation
_MAD_DENOM            : float = 1.13                # Allen–Dynes strong-coupling corrections give denom≈1.45 for λ~1 spin-fluctuation spectra; Millis–Monien–Pines quote denom≈1.13 for the SF model.
_MAD_NUM              : float = 1.04                # exponent prefactor (weak-coupling limit, spectrum-independent)
_GL_MIN_PTS           : int   = 2                   # minimum stable SC points required for GL fit
_GL_MAX_PTS           : int   = 4                   # upper window of recent stable points used in GL regression
_GL_DELTA_MIN         : float = 2e-3                # |Δ| floor for GL fit points (below this: numerically unreliable)
_GL_TC_MARGIN         : float = 0.05                # max relative deviation |Tc_GL−T_spinodal|/T_max to accept GL result
_GL_SPINODAL_JUMP     : float = 0.15                # Δ_spinodal/Δ_0 below this → GL extrapolation reliable
_THRESHOLD_RATIO      : float = 0.02                # 2% difference threshold for first-order classification
_TC_LIN_DELTA_EPS     : float = 2.0e-5              # FD step for ∂F_raw_g/∂Δ at Δ=0 (F_raw_g is ODD in Δ (anomalous amplitude ∝ Δ), so the truncation error is (eps/ξ_eff)² with ξ_eff = min(kT, t_eff)
_TC_LIN_EPS_TMIN_FRAC : float = 0.1                 # warn if eps > this × min(T_grid): FD linearisation unreliable
_TC_LIN_COMPLEX_TOL   : float = 0.1                 # max(|Λ| − |Re Λ|) / max|Λ| above this → complex (TRS-broken) mode
_TC_TARGET_K          : float = 230

# ── Lindhard / χ₀ kernel ─────────────────────────────────────────────────────
_NAMBU_MBZ_NORMFACTOR : float = 0.125               # Normalization: 0.5 × (two sublattices A/B) 0.5 (Nambu particle-hole doubling) × 0.5 (hermitisation of chi[a,b]↔chi[b,a])
_MATH_EPS             : float = 1e-9                # general protection against division by zero
_FD_MASK_DF           : float = 1e-12               # |Δf| mask threshold in χ₀ Lehmann sums
_FD_MASK_DE           : float = 1e-6                # |ΔE| mask threshold in χ₀ Lehmann sums
_ETA_T_FRAC           : float = 0.10                # normal-state η = _ETA_T_FRAC · kT  (thermal broadening)
_ETA_DELTA_FRAC       : float = 0.02                # anomalous η += _ETA_DELTA_FRAC · |Δ|  (gap-scale broadening)
_ETA_GRID_FLOOR       : float = 0.001               # η ≥ _ETA_GRID_FLOOR · t0  (k-grid aliasing floor; ~bandwidth/N_k²)

# ── RPA / Moriya spin-fluctuation damping ─────────────────────────────────────
_MORIYA_LANDAU_M_STEP : float = 0.06                # M-probe step for the model-derived (Landau a,b) Gamma_M estimate
_RPA_BW_FACTOR        : float = 8.0                 # 2D t-t'-t'' BW = 8t exactly when |t'/t|<0.5 and t''/t<(1−t'/t)/2 (t' and t'' cancel at (0,0) and (π,π) corners).
_RPA_V_CAP_ALPHA      : float = 2.2                 # Perturbative RPA breaks down when V_pair ~ O(bandwidth); V_cap = α·max(8·max(|tx|,|ty|), J_eff). 2.2× headroom above the BEC-BCS crossover energy while preventing runaway at the AFM QCP
_SCHUR_HDD_FLOOR      : float = 1e-6                #       
_RPA_DET_WARN         : float = 0.11                # QCP proximity warning threshold for diagnostics and SCF adaptive mixing.
_RPA_QCP_PENALTY      : float = 0.40                # α reduction per unit |det_afm|<0 past QCP (used in SCF near-critical detection, BO near_qcp flag).
_DET_AFM_FLOOR        : float = 0.5                 # default det_afm when vertex cache is absent (normal state, no QCP)
_DET_SIGN_FLIP_SCALE  : float = 0.05                # |det_afm| scale for V_d sign-flip EMA suppression (determines the sigmoid midpoint)
_EMA_SIGN_FLIP_W_MIN  : float = 0.20                # minimum w_factor on V_d sign flip; preserves adaptation even at det≈0
_EMA_SIGN_FLIP_SLOPE  : float = 6.0                 # sigmoid steepness in sign-flip EMA: w=w_min+(1-w_min)/[1+exp(-k·(|det|/floor-0.5))]
_V_PREV_SIGN_FLOOR    : float = 1e-6                # |V_d_prev| below this → treat as zero, skip sign-flip check
_DET_DEPTH_CAP        : float = 5.0                 # max det_depth in jump-cap exponential suppression
_DET_JUMP_HALF_SCALE  : float = 0.5                 # exp(−this·det_depth) decay rate for gap jump cap past QCP
_JUMP_CAP_FLOOR       : float = 1.05                # minimum effective_jump_cap (prevents total freeze past QCP)
_QQ_DELTA_THRESH      : float = 1e-8                # |Δ| threshold below which χ_SQ enforced zero

# ── JT / SC-triggered Jahn–Teller ─────────────────────────────────────────────
_JT_PHONON_MEV_CONST  : float = 0.00417998454784    # force-constant-to-phonon-energy conversion (derived from hbar, eV, Ang, amu)
_LAMBDA_JT_VIABLE     : float = 0.05                # Minimum λ_JT_sc = g²·|χ_τ_sc|/K_eff for SC-triggered JT viability. Sets upper K_SC = g²·|χ_τ_sc|/0.05 bound on K_lattice
_JT_ACT_THR           : float = 0.04                # The threshold of Γ₆–Γ₇ mixing induced by SC condensate
_DQ_FS_VERTEX         : float = 0.03                # Å — MINIMUM finite-difference step for ∂λ/∂Q on FS; adaptiv mode: h = max(_DQ_FS_VERTEX, _DQ_FS_VERTEX_FRAC*|Q|).
_DQ_FS_VERTEX_FRAC    : float = 0.05                # Å/Å — adaptive step fraction: h = max(_DQ_FS_VERTEX, this*|Q|); near QCP where |Q|->0, falls back to minimum; large Q -> 5% gives noise protection.
_JT_FD_H2_BASE        : float = 3e-8                # Q-derivative finite-difference step, h(Q) = sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF·Q²)
_JT_FD_H2_QCOEF       : float = 6e-7                # quadratic coefficient of the Q-dependent FD step (see above)

# ── Saddle‑escape / Kick ──────────────────────────────────────────────────────
_MODE_FRAC_DOMINANT   : float = 0.48                # fX > this → mode dominated by X (pure-SC, pure-JT, pure-AFM)
_KICK_MIN_DELTA_FRAC  : float = 0.25                # saddle-escape kick only if the Δ fraction of the soft mode exceeds this (pure-SC AND mixed Δ–Q 
_MODE_PULL_FRAC       : float = 0.30                # fraction of (M − M_phys_est) used as kick pull in pure-SC/SC-JT mode
_KICK_BASE_FRACTION   : float = 0.05                # "Trust Region" box specifies the step size as a percentage
_KICK_M_EXCESS_CTR    : float = 0.70                # M-excess onset for kick reduction
_KICK_JCHI_EXCESS_CTR : float = 0.70                # J·χ_SS-excess onset for kick reduction
_KICK_REDUCTION_AMP   : float = 3.88                # amplitude of M-kick reduction: M_kick × (1 − this × excess)
_Q_SEED_THR           : float = 2e-4
_KICK_M_CLIP_LO       : float = _MATH_EPS           # hard lower clip on M_kick (normal SCF path)
_KICK_M_CLIP_HI       : float = 0.9                 # hard upper clip on M_kick
_KICK_DELTA_MAX_FRAC  : float = 0.4                 # maximum allowed seed gap as a fraction of the effective hopping scale t_eff.
_KICK_MIXING_FLOOR    : float = 0.004               # minimum mixing weight in the kick; prevents α from collapsing to zero when λ_plus is huge.
_KICK_MIXING_SCALE    : float = 4.0                 # damping scale for λ_plus in α = _MIXING / (1 + scale·log1p(λ_plus)).
_EARLY_KICK_BASE      : float = 0.01                # base step fraction in the coupled space
_MAX_ITER_FOR_REFINE_M: float = 130

# ── SCF iteration / mixing adaptive control ────────────────────────────────────
_MAX_ITER             : int   = 500
_MIN_ITER             : int   = 4
_MIXING               : float = 0.19                # base weight of the newly computed residual in the solution update; lower values improve stability at the cost of slower convergence.
_ALPHA_HF             : float = 0.85                # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton)
_ALPHA_Q              : float = 0.5                 # relax it with THIS weight (and keep it out of the Anderson least-squares) instead of the Δ/M weight alpha~0.03, which would need >100 iterations to reach Q_out.
_ALPHA_Q_MIN          : float = 0.10                # floor of the Q relaxation weight while Q oscillates (halved at every detected oscillation, restored when Q has been calm for a window)
_Q_TRUST_ABS          : float = 0.006               # Å: half-width of the window around the CURRENT Q in which solve_Q_slave looks for the new Q (the V_irrep cache is frozen at Q_in, so the profile is trustworthy only near Q_in)
_Q_TRUST_REL          : float = 0.50                # … plus this fraction of |Q_in| (the vertex itself varies on the scale of |Q|: λ_FS falls ≈25 % between Q=0.016 and 0.026 Å)
_Q_TRUST_MIN_SCALE    : float = 0.05                # lower bound of the oscillation-adaptive scale of the window half-width
_Q_WINDOW_NPTS        : int   = 7                   # profile evaluations inside the window (spacing = half-width/2); the force root is refined inside the bracket of the grid minimum
_Q_FLAT_EV            : float = 1e-9                # eV: a window profile flatter than this is numerical noise → Q stays where it is
_Q_CYCLE_WINDOW       : int   = 7                   # signed-Q iterates inspected by the oscillation detector (catches period-2 and period-3 cycles)
_Q_CYCLE_ZIGZAG       : float = 0.5                 # fraction of consecutive Q increments that reverse direction above which Q is declared oscillating
_Q_CYCLE_AMP_MIN      : float = 5e-4                # Å: Q increments below this are noise, not an oscillation
_Q_THR_REL            : float = 2e-5                # fraction of lambda_hop; Q change below this skips vertex rebuild
_M_THR_REL            : float = 0.01                # absolute M change threshold
_EMA_NEW_WEIGHT       : float = 0.12                # EMA weight for V_d, Λ_inst
_SCF_DIVERGE_RATIO    : float = 1.05                # max_diff > prev × this → SCF classified as diverging
_SCF_STAGNATE_RATIO   : float = 0.95                # max_diff > prev × this (and not diverging) → SCF stagnating
_SCF_ALPHA_DECAY      : float = 0.95                # α *= this when SC+JT active and converging (mild damping)
_SCF_ALPHA_QMOVE_CAP  : float = 0.30                # while Q is still moving (|Q_mixed−Q| > _Q_THR_REL·λ_hop) α is capped at _MIXING × this (0.30 → α ≤ 0.045); the cap is released once Q has settled.  NOTE: the threshold is tiny (2.4e-5 Å), so the cap is active during practically the whole Q approach → this, not _SCF_ALPHA_DECAY (a one-off ×0.95 in one branch), is what limits α
_SCF_ALPHA_RECOVER    : float = 1.60                # α *= this on freeze-recovery (restores mobility after stagnation)
_SCF_FREEZE_THR       : int   = 10                  # α_freeze_count ≥ this triggers freeze-recovery boost
_SCF_ALPHA_FREEZE_LO  : float = 0.15                # α < _MIXING × this → too frozen, trigger recovery
_SCF_ALPHA_FREEZE_HI  : float = 0.60                # α recovery ceiling: min(α×_RECOVER, _MIXING×this)
_SCF_ALPHA_CONVG_BOOST: float = 1.09                # α boosted (× this) when SC+JT active and converging (mild)
_SCF_ALPHA_CONVG_CAP  : float = 0.75                # α ceiling (× _MIXING) during SC+JT active converging branch

# ── M, Q, Δ Newton solver step control ──────────────────────────────────────────
_M_STEP_FLOOR_REL     : float = 0.005               # _step_floor = max(_M_STEP_FLOOR_REL × |M|, _M_STEP_FLOOR_ABS)
_M_STEP_FLOOR_ABS     : float = 0.002               # absolute minimum M step (eV·site) regardless of |M|
_M_J_EFF_FLOOR_FRAC   : float = 0.20                # j_eff_floor = max(|J_eff|, this × t_eff, ε) — QCP guard preventing ΔM∝1/J_eff→∞
_MU_LM                : float = 3.0                 # Levenberg–Marquardt floor for M Newton step, larger → smaller γ_M → more conservative M update.
_TR_M_STEP_MAX        : float = 0.1                 # Upper bound on |ΔM| (eV). Reduced when J_eff/t_eff is large (stiff landscape) or near QCP (flat curvature).
_TR_M_STEP_MIN_FLOOR  : float = 1e-3                # absolute minimum step — prevents total freeze near M→0
_Q_LM_FRAC            : float = 0.08                # Q-channel LM floor as a *fraction of self._K_bare*, not an absolute eV/Å² constant
_MU_LM_DELTA          : float = 3.0                 # Levenberg–Marquardt floor for the Δ Newton step (2×2 analogue of _MU_LM)
_ALPHA_HF_DELTA       : float = 0.75                # Newton vs BdG-fixpoint blend for Δ update (0=fixpoint, 1=Newton); more conservative than _ALPHA_HF since Δ has two coupled channels and is already jump-capped downstream
_TR_DELTA_STEP_MAX    : float = 0.1                 # Upper bound on |Δ_newton_step| per channel per iteration (eV, same scale as _TR_M_STEP_MAX)

# ── Anderson mixing ──────────────────────────────────────────────────────────────
_ANDERSON_TIKHONOV    : float = 1e-8                # Tikhonov β / diag_max in Anderson normal equations
_ANDERSON_TRUST       : float = 2.4                 # trust-region step-size limit (multiples of simple step)
_ANDERSON_W_LO        : float = 0.3                 # lower blend weight between Anderson and simple mixing
_ANDERSON_W_HI        : float = 0.8                 # upper blend weight
_IDX_ANDERSON_EXT     : np.ndarray = np.array([_i for _i in range(_N_DELTA) if _i not in (_DELTA_IDX['s7a'], _DELTA_IDX['d7a'])])   # everything except the primary (s7a, d7a) pair

# ── Limit-cycle detector ─────────────────────────────────────────────────────────
_CYCLE_WINDOW         : int   = 20                  # iteration window to detect oscillation
_CYCLE_THRESHOLD      : float = 0.25                # std/mean of |Δ| above this → oscillatory regime
_CYCLE_DAMP_FAC       : float = 0.45                # alpha reduction factor when oscillation detected

# ── Orbital-character / coherence-length thresholds ──────────────────────────────
_XI_NODAL_MIN         : float = 2.0                 # ξ/a > this required for coherent nodal quasiparticles (BCS side)
_ORBITAL_SEL_FRAC     : float = 0.15                # |ξ_Γ₆ − ξ_Γ₇|/ξ > this → system classified as orbitally selective

# ── IC (inter-channel) correction ratio bounds ───────────────────────────────────
_IC_RATIO_FLOOR       : float = 1.05                # r < this → negligible IC, no suppression applied
_IC_RATIO_CAP         : float = 3.00                # r > this → very strong IC, cap reduction to avoid too small M

# ── Thermodynamics / μ-solver ────────────────────────────────────────────────────
_FERMI_ARG_CLIP       : float = 100.0               # clip argument of exp() in Fermi function
_DEN_DERIV_FLOOR      : float = 1e-12               # ∂n/∂μ floor in Newton μ-finder
_BRENTQ_TOL           : float = 1e-6                # brentq μ-bracketing tolerance
_MU_NEWTON_MAXIT      : int   = 20                  # Newton/backtrack iteration budget before the guaranteed Brent finish
_MU_DENSITY_TOL       : float = 1e-8                # |n(μ)−target_n| convergence tolerance
_MU_SC_DERIV_THRESH   : float = 1e-4                # eV; |Δ_s|+|Δ_d| above which the analytic dn/dμ (exact only for pure-particle/hole BdG branches, i.e. Δ=0) is replaced by a centred numeric derivative
_MU_BACKTRACK_MAX     : int   = 6                   # max step-halvings per Newton iteration before declaring the direction unreliable
_MU_BACKTRACK_FLOOR   : float = 0.05                # minimum backtracking damping factor η before giving up on the current Newton direction

# ── Gap (Δ) update ───────────────────────────────────────────────────────────────
_KERNEL_BLEND_DECAY   : float = 3.0                 # kernel-direction steering is a SEEDING device: its weight is multiplied by exp(−|D_fp|/(DECAY·seed_val)); 0 = constant weight (fixed point ≠ gap-equation solution)
_ALPHA_MIX_2X2        : float = 0.56                # blend weight of 2×2 eigenvector direction vs fixed-point gap update, 0 = pure fixed-point; 1 = pure 2×2 eigenvector
_BCS_SEED_FRACTION    : float = 0.1                 # BCS seed magnitude as fraction of t_eff
_DELTA_JUMP_CAP       : float = 5.0                 # max |Δ_new| / |Δ_current| ratio per iteration
_DELTA_ABS_FLOOR      : float = 1e-4                # eV — absolute |Δ| below which the jump limiter is bypassed (seed-gap free-growth phase). floor is ~0.7–2% of t0
_PHI_D_FLOOR          : float = 1e-3                # minimum φ_d max value to enable nodal/antinodal decomposition
_KERNEL_DIR_MIN_FRAC  : float = 0.5                 # when the fixed-point gap amplitude is below this fraction of the 2×2 kernel eigenvector amplitude, the 2×2 eigenvector direction is allowed to dominate the mixing

# ── Physical thresholds (SC viability / Mott) ────────────────────────────────────
_G_T_COHERENCE_MIN    : float = 0.10                # g_t floor for coherent ZRS band (δ<0.053 → incoherent FS). Used in: _scf_kick, Mott filter, _score, __main__ doping floor.
_JCHI_HARD_REJECT     : float = 2.0                 # J·χ_SS > this → score = 0 (deeply AFM, SC impossible)
_V_CUT                : float = 20.0                # pairing vertex near-divergence detector threshold
_MBZ_DEGEN_FRAC       : float = 2e-3                # energy-based tie-break scale to what extent the JT member disrupts the degeneration
_WIGNER_ECKART_DIV_FLOOR: float = 1e-6              # denominator floor in the Wigner–Eckart factor κ_c/(4 s_eff²)
_S_EFF_FLOOR          : float = 0.15                # Numerical floor on the effective spin projection used in κ_c/(4 s_eff²). A Kramers doublet with s_eff ≡ 0 is unphysical
_ANISO_WARN_TRESH     : float = 5e-4                # non-Heisenberg coefficient warning threshold in the intra-doublet κ extraction
_B1G_BLOCK_MIN_FRAC   : float = 0.02                # min share of ‖B1g_op‖_F² (=4) a Γ6↔Γ7 block must carry to resolve its B1g⊗B1g exchange channel: J_B1g ∝ 1/‖M‖² and its Weiss field J_B1g·B_offdiag ∝ 1/‖M‖ (≈2× amplified at 5 %) diverge for selection-rule-suppressed blocks (Γ6↔Γ7b)
_SYM_THRESH           : float = 1e-8                # symmetry-breaking floor for Δ_B1g_static / |Q| classification
_C4_SPECTRAL_TOL      : float = 1e-8                # eV: C4 is an exact symmetry of the normal-state BdG spectrum if max_{k,n}|E_n(k) − E_n(R·k)| is below this

_I6 = np.eye(_N_ORB, dtype=complex)
_SIGMA_2x2 = {
    '0': np.eye(2, dtype=complex),
    'x': np.array([[0, 1], [1, 0]], dtype=complex),
    'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'z': np.array([[1, 0], [0, -1]], dtype=complex),
}
_ORBITAL_VECTORS = {
    'xz': np.array([-1.0, 0.0,  1.0], dtype=complex) / np.sqrt(2.0),
    'yz': np.array([-1j,  0.0, -1j ], dtype=complex) / np.sqrt(2.0),
    'xy': np.array([ 0.0, 1.0,  0.0], dtype=complex),
}
# ── Time reversal on the t2g ⊗ spin space ───────────────────────────────────────────
_U_ORB = np.column_stack(list(_ORBITAL_VECTORS.values()))
_U_THETA_T2G = np.kron(_U_ORB @ _U_ORB.T, 1j * _SIGMA_2x2['y'])

def _scf_log(tag: str, msg: str, verbose: bool = True) -> None:
    """Thread-safe logger.  tag is left-padded to 18 chars so columns stay aligned."""
    if not verbose:
        return
    with _log_lock:
        print(f"[{tag:<18s}] {msg}", flush=True)

def _check_registry() -> None:
    """Import-time consistency checks of the basis registry: unique names, inversion parity = mx·my, T_Q parity consistent with the bond kind (A–B bonds are T_Q-odd, everything else T_Q-even)."""
    if len(set(_IRREP_NAMES)) != len(_IRREP_NAMES):
        raise ValueError(f"duplicate pairing channel names: {_IRREP_NAMES}")
    for c in _PAIR_CHANNELS:
        t, mx, my = _IRREP_CHAR[c.name]
        if mx * my != c.parity:
            raise ValueError(f"channel '{c.name}': declared parity {c.parity} != measured mx*my = {mx * my}")
        if t != (-1 if c.kind == 'nn_AB' else +1):
            raise ValueError(f"channel '{c.name}': T_Q parity {t:+d} is inconsistent with bond kind '{c.kind}' (A–B bonds <=> T_Q-odd, same-sublattice bonds <=> T_Q-even)")
    for n, v in _HARMONIC_LIBRARY.items():
        t, mx, my = _LIB_CHAR[n]
        if mx * my != v[2] or t != (-1 if v[1] == 'nn_AB' else +1):
            raise ValueError(f"library harmonic '{n}': declared (kind, parity) inconsistent with measured character {(t, mx, my)}")
        if mx == my:     # even-parity sectors: the measured (mx, my, σd) must reproduce the declared D4h irrep, otherwise the σd resolution of a sector would mislabel it
            sw = _LIB_SWAP[n]
            if sw == 0 or _D4H_BY_MX_MY_SWAP[(mx, my, sw)] != v[0]:
                raise ValueError(f"library harmonic '{n}': declared D4h irrep {v[0]} inconsistent with the measured (mx, my, σd) = {(mx, my, sw)}")
_check_registry()

def _bond_weights(ff, n: int = _BOND_GRID):
    """(w, lo, hi): normalised squared amplitudes |c_δ|² of the bond expansion of the form factor `ff` on an n×n periodic grid, and min/max(|δ_x|, |δ_y|) of every bond."""
    k = 2.0 * np.pi * np.arange(n) / n
    KX, KY = np.meshgrid(k, k, indexing='ij')
    c = np.fft.fft2(np.broadcast_to(np.asarray(ff(KX, KY), dtype=float), KX.shape)) / n ** 2
    w = np.abs(c) ** 2
    w[w < 1e-20 * float(w.max())] = 0.0                              # FFT round-off (~1e-31) -> exact zero: bonds that are absent from the form factor carry exactly no weight
    ax = np.minimum(np.arange(n), n - np.arange(n))
    AX, AY = np.meshgrid(ax, ax, indexing='ij')
    return w / max(float(w.sum()), 1e-300), np.minimum(AX, AY), np.maximum(AX, AY)

_BOND_CLASSES = ('onsite', 'NN (1,0)', 'NNN (1,1)', '3NN (2,0)', 'longer')

def _bond_class_fractions(ff) -> np.ndarray:
    """Squared-amplitude fractions of the form factor on the bond classes _BOND_CLASSES. A bond carries a B1g⊗B1g exchange (hence a Weiss field) only if the model has a hopping on it: t, t', t'' for the NN, NNN, 3NN class."""
    w, lo, hi = _bond_weights(ff)
    cls = np.full(w.shape, 4)
    cls[(lo == 0) & (hi == 0)] = 0
    cls[(lo == 0) & (hi == 1)] = 1
    cls[(lo == 1) & (hi == 1)] = 2
    cls[(lo == 0) & (hi == 2)] = 3
    return np.array([float(w[cls == i].sum()) for i in range(5)])

def _check_bond_content() -> None:
    """Import-time check of the declared bond KIND of every library harmonic against the real-space support of its form factor: 'onsite' = a constant; 'nn_AB' = bonds of odd sublattice distance only (A–B);
    'nnn_diag' / 'nnn_axis' = bonds of even sublattice distance only and no on-site term, 'nnn_axis' additionally purely axial ((n,0)-type). (_check_registry already ties kind <-> T_Q parity.)"""
    for n, v in _HARMONIC_LIBRARY.items():
        w, lo, hi = _bond_weights(v[3])
        f_on = float(w[(lo == 0) & (hi == 0)].sum())
        f_odd = float(w[((lo + hi) % 2) == 1].sum())               # weight on odd sublattice distance (A–B bonds)
        f_ax = float(w[lo == 0].sum())
        kind = v[1]
        ok = {'onsite': abs(f_on - 1.0) < 1e-9, 'nn_AB': abs(f_odd - 1.0) < 1e-9,
              'nnn_diag': f_odd < 1e-9 and f_on < 1e-9,
              'nnn_axis': f_odd < 1e-9 and f_on < 1e-9 and abs(f_ax - 1.0) < 1e-9}[kind]
        if not ok:
            raise ValueError(f"library harmonic '{n}': kind '{kind}' is inconsistent with the real-space bond content of its form factor (on-site {f_on:.3g}, odd-distance {f_odd:.3g}, axial {f_ax:.3g})")
_check_bond_content()
_TRW_BOND_FRAC = {_n: _bond_class_fractions(_IRREP_FF[_n]) for _n in _IRREP_NAMES}    # name -> fractions on _BOND_CLASSES

def _sector_label(key: tuple) -> str:
    """Readable sector label: the character plus the library harmonics living in that sector ('*' marks the ones in the active BdG basis)."""
    t, mx, my = key
    names = [n + ('*' if n in _IRREP_NAMES else '') for n in _HARMONIC_LIBRARY if _LIB_CHAR[n] == key]
    return f"(TQ{'+' if t > 0 else '-'},mx{'+' if mx > 0 else '-'},my{'+' if my > 0 else '-'}) [{','.join(names[:6])}]"

def _t2g_soc_cf_operators(lambda_soc: float, Delta_tetra: float, Delta_inplane: float = 0.0):
    """
    Build the 6-dim (3 t2g orbitals × 2 spin) SOC + crystal-field Hamiltonian and the operators needed to analyse it.

    Returns
    -------
    H       : (6,6) complex   H_SOC + H_CF
    Lx_t2g, Ly_t2g, Lz_t2g : (6,6) complex   orbital operators embedded in the 6-dim spin space (⊗I2) — e.g. for the B1g operator Lx²−Ly².
    Sz      : bare 3-dim identity and 2-dim S_z, for building S_z in the full 6-dim space (I3 ⊗ Sz) where the caller needs it.
    LS_op   : (6,6) complex   L·S operator (Γ6 vs Γ7 identification via ⟨L·S⟩)
    """
    Lz = np.diag([1.0, 0.0, -1.0])
    Lp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    Lm = Lp.T.conj()
    Lx = (Lp + Lm) / 2.0
    Ly = (Lp - Lm) / 2.0j
    I2 = np.eye(2, dtype=complex)
    Sx = 0.5 * _SIGMA_2x2['x']
    Sy = 0.5 * _SIGMA_2x2['y']
    Sz = 0.5 * _SIGMA_2x2['z']
    I3 = np.eye(3, dtype=complex)

    H_SOC = lambda_soc * (np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz))
    Lx_t2g = np.kron(Lx, I2)
    Ly_t2g = np.kron(Ly, I2)
    Lz_t2g = np.kron(Lz, I2)
    H_CF = (Delta_tetra * (Lz_t2g @ Lz_t2g) + Delta_inplane * (Lx_t2g @ Lx_t2g - Ly_t2g @ Ly_t2g))
    LS_op = np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)
    return H_SOC + H_CF, Lx_t2g, Ly_t2g, Lz_t2g, np.kron(I3, Sz), LS_op

def _fermi_function(ev: np.ndarray, kT: float) -> np.ndarray:
    arg = np.clip(ev / kT, -_FERMI_ARG_CLIP, _FERMI_ARG_CLIP)
    return 1.0 / (1.0 + np.exp(arg))

def _get_nambu_spinors(ec: np.ndarray):
    """
    Slice BdG eigenvector array into Nambu spinors per sublattice.
    Layout (matches _build_H_stack), full 6-level manifold — no downfolding:
        rows  0– 5 : particle A (Γ₆↑, Γ₆↓, Γ₇ₐ↑, Γ₇ₐ↓, Γ₇ᵦ↑, Γ₇ᵦ↓)
        rows  6–11 : particle B
        rows 12–17 : hole A
        rows 18–23 : hole B
    Returns uA, uB, vA, vB — each (N_k, _N_ORB, _N_BDG).
    """
    return ec[:, 0:_N_ORB, :], ec[:, _N_ORB:2*_N_ORB, :], ec[:, 2*_N_ORB:3*_N_ORB, :], ec[:, 3*_N_ORB:_N_BDG, :]

def _compute_F67_pair_amplitudes(k_weights: np.ndarray, Phi_irrep_k: np.ndarray, ev: np.ndarray, ec: np.ndarray, kT: float, Delta_vec: np.ndarray, g_Delta_s: float, g_Delta_ad: float, g_Delta_bd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the Γ₆–Γ₇ₐ and Γ₆–Γ₇b inter-orbital singlet anomalous amplitudes, one per (channel x irrep) entry of Delta_vec (_N_DELTA = 2·_N_IRREP components in total): an on-site (A–A / B–B) amplitude
    for the 's' irrep, related to Δ_s; inter-sublattice (A–B / B–A) amplitudes with the appropriate NN-type bond form factor φ_name(k) for every kind='nn_AB' irrep ('es', 'd', 'px', 'py');
    and SAME-sublattice (A–A / B–B) amplitudes with the form factor φ_name(k) for the kind='nnn_*' irreps ('g', 'g2', 'px2', 'py2'): the pair correlator is the one of the on-site channel, weighted by φ_name(k).

    They are defined as
        F = Σ_k w_k (1-2f_n) [u_{6↑} v*_{7↓} − u_{6↓} v*_{7↑}] / 4,
    extended with φ_name(k) for every bond channel (φ_s ≡ 1 for the on-site channel); F_raw ∝ Δ (it has the phase of its own gap function).

    phase: F67[i] = g_i · F_raw,i · exp(−iφ_ref)    complex, gauge-invariant; φ_ref = phase of the largest |Δ| of Delta_vec (0 if Δ=0); exactly 0 for a channel with Δ_i = 0.  Returns a complex array.

    Phi_irrep_k : (N_k, _N_IRREP) — _pairing_basis_matrix evaluated on the same k-grid as ev/ec, in _IRREP_NAMES order. Column _IR_D reproduces the original φ_k = cos kx − cos ky exactly.
    """
    f12 = 1.0 - 2.0 * _fermi_function(ev, kT)
    uA, uB, vA, vB = _get_nambu_spinors(ec)

    # --- Γ₆ ↔ Γ₇ₐ (7a) ---
    anom_A_7a  = uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
    anom_B_7a  = uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])
    anom_AB_7a = uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
    anom_BA_7a = uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])

    # --- Γ₆ ↔ Γ₇b (7b) ---
    anom_A_7b  = uA[:, 0, :] * np.conj(vA[:, 5, :]) - uA[:, 1, :] * np.conj(vA[:, 4, :])
    anom_B_7b  = uB[:, 0, :] * np.conj(vB[:, 5, :]) - uB[:, 1, :] * np.conj(vB[:, 4, :])
    anom_AB_7b = uA[:, 0, :] * np.conj(vB[:, 5, :]) - uA[:, 1, :] * np.conj(vB[:, 4, :])
    anom_BA_7b = uB[:, 0, :] * np.conj(vA[:, 5, :]) - uB[:, 1, :] * np.conj(vA[:, 4, :])

    # BZ sums, Gutzwiller-weighted.
    onsite_7a   = complex(np.einsum('k,kn,kn->', k_weights, f12, (anom_A_7a + anom_B_7a)) / 4.0)
    onsite_7b   = complex(np.einsum('k,kn,kn->', k_weights, f12, (anom_A_7b + anom_B_7b)) / 4.0)
    bond_sum_7a = anom_AB_7a + anom_BA_7a     # inter-sublattice pair correlator: bonds with odd sublattice distance (NN-type, T_Q-odd form factors)
    bond_sum_7b = anom_AB_7b + anom_BA_7b
    same_sum_7a = anom_A_7a + anom_B_7a       # same-sublattice pair correlator: bonds with even sublattice distance (NNN / 3NN-type, T_Q-even form factors)
    same_sum_7b = anom_A_7b + anom_B_7b

    ref_phase = float(np.angle(Delta_vec[int(np.argmax(np.abs(np.asarray(Delta_vec))))])) if float(np.abs(np.asarray(Delta_vec)).max()) > _MATH_EPS else 0.0

    # Gutzwiller factor of EVERY channel: the very same per-irrep vectors that the pairing vertex, the condensation energy and the gap equation use (on-site AND same-sublattice NNN / 3NN / longer bonds:
    # g_Δs = g_t; NN-type A–B bonds: g_Δad (7a) / g_Δbd (7b)). With g_Δad/g_Δbd for the same-sublattice channels F67 was inconsistent with V·√(g_i g_j) in the gap equation and with the Hellmann–Feynman identity ∂Ω/∂|Δ_i| = −(2/g_i)·Re(F_g,i e^{−iφ_i}).
    g_vec_7a = _gutzwiller_irrep_vector(g_Delta_s, g_Delta_ad, g_Delta_s)
    g_vec_7b = _gutzwiller_irrep_vector(g_Delta_s, g_Delta_bd, g_Delta_s)

    F67_vec = np.zeros(_N_DELTA, dtype=complex)
    F_raw_g = np.zeros(_N_DELTA, dtype=complex)
    for i_irr, _name in enumerate(_IRREP_NAMES):
        pos_7a, pos_7b = int(_IDX_7A_ARR[i_irr]), int(_IDX_7B_ARR[i_irr])
        g_a, g_b = float(g_vec_7a[i_irr]), float(g_vec_7b[i_irr])
        if _is_onsite(_name):
            raw_7a, raw_7b = onsite_7a, onsite_7b
        else:
            phi_name_k = Phi_irrep_k[:, i_irr]
            sum_7a, sum_7b = (bond_sum_7a, bond_sum_7b) if _is_nn_bond(_name) else (same_sum_7a, same_sum_7b)
            raw_7a = complex(np.einsum('k,k,kn,kn->', k_weights, phi_name_k, f12, sum_7a) / 4.0)
            raw_7b = complex(np.einsum('k,k,kn,kn->', k_weights, phi_name_k, f12, sum_7b) / 4.0)
        for pos, g_c, F67_raw_c in ((pos_7a, g_a, raw_7a), (pos_7b, g_b, raw_7b)):
            F67_vec[pos] = g_c * complex(F67_raw_c * np.exp(-1j * ref_phase))
            F_raw_g[pos] = g_c * F67_raw_c
    return F67_vec, F_raw_g

def _uniform_bz_weights_2d(nx: int, ny: int) -> np.ndarray:
    """
    BZ integration weights for a 2D periodic endpoint=False k-grid →  dk = 2π/n.

    Convention: Σ_k w_k = 1  (normalised BZ average, not physical (2π)² measure).
    All callers use  Σ_k w_k f(k)  to compute a BZ average <f>_BZ directly.

    The correct rule for a PERIODIC function on an endpoint=False uniform grid is
    the UNIFORM (rectangular) rule, which converges exponentially for smooth
    periodic functions (Poisson summation / aliasing argument).

    Why NOT composite Simpson?
        Composite Simpson is designed for CLOSED intervals [a, b].
        On a periodic BZ the pattern [1, 4, 2, …, 4, 1] assigns different weights
        to the boundary points, breaking translational invariance and introducing
        a boundary-bias O((dk)²) per cell edge.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Grid sizes must be positive: got {nx}, {ny}")
    n_total = nx * ny
    return np.full(n_total, 1.0 / n_total)

def _lehmann_kernel(f_k: np.ndarray, f_kQ: np.ndarray, E_k: np.ndarray, E_kQ: np.ndarray, eta: float, kT: float) -> np.ndarray:
    """
    Continuous (ω→0) Lehmann-weight particle-hole kernel, shared by every
    bubble in this file:
        |ΔE| ≫ η  →  Δf·ΔE / ΔE²   (standard particle-hole bubble)
        |ΔE| ≪ η  →  -f'           (Fermi-surface term, Taylor limit)
    f_k, E_k are occupations/energies at k, shape (N_k, n_bands); f_kQ, E_kQ
    are the same at k+q. Returns the (N_k, n_bands, n_bands) kernel *before* any k-weighting
    """
    df = f_k[:, :, None] - f_kQ[:, None, :]
    dE = E_kQ[:, None, :] - E_k[:, :, None]

    df_dE_k  = -f_k  * (1.0 - f_k)  / kT
    df_dE_kQ = -f_kQ * (1.0 - f_kQ) / kT
    df_dE_avg = 0.5 * (df_dE_k[:, :, None] + df_dE_kQ[:, None, :])  # (N_k, n, n)

    df_safe = np.where(np.abs(df) > _FD_MASK_DF, df, 0.0)
    de_safe = np.where(np.abs(dE.real) > _FD_MASK_DE, dE.real, 0.0)
    return (df_safe * de_safe + (-df_dE_avg) * eta**2) / (de_safe**2 + eta**2)

def _apply_hop(bits, i, j):
    """Return (new_bits, sign) for h_i^† h_j, or None if forbidden."""
    if not ((bits >> j) & 1):       # source must be occupied
        return None
    if (bits >> i) & 1:             # target must be empty (Pauli)
        return None

    # Fermion sign from removing j ...
    sign = 1
    sign *= (-1) ** bin(bits & ((1 << j) - 1)).count("1")
    bits_after_ann = bits ^ (1 << j)
    sign *= (-1) ** bin(bits_after_ann & ((1 << i) - 1)).count("1")
    return bits_after_ann | (1 << i), sign

def _multiorbital_site_of(sp):
    """Map a spin-orbital index (0..17) of Single-particle Hamiltonian for the SOC/CF-aware d1-p-d2 superexchange cluster to (site, channel m, spin s)."""
    if sp < 6:
        return 'd1', sp // 2, sp % 2
    if sp < 12:
        return 'p', (sp - 6) // 2, (sp - 6) % 2
    return 'd2', (sp - 12) // 2, (sp - 12) % 2

def _multiorbital_2hole_hamiltonian(t_pd, Delta_CT, H_soc_cf, U_dd, U_pp, J_H):
    """
    Full 2-hole Hamiltonian on the C(18,2)=153 Fock states of the 3-orbital t2g manifold for a d-p-d chain.

    Layout (18 spin-orbitals):
        d1 = 0..5, p_(m,s) = 6..11, d2 = 12..17,
        local index = 2*m + s, with m=0,1,2 ↔ Lz=+1,0,-1 and s=0,1 ↔ up,dn.

    The ligand consists of THREE independent modes, one for each m, rather than a shared
    ligand orbital. This preserves Jz=Lz+Sz after SOC mixing and avoids dependence on
    arbitrary inter-channel phase choices. Equal bare t_pd for all three modes directly
    matches the orbital-blind reference hopping Tx_op(Q=0)=t0*I6 in hopping_matrices().

    Kanamori interaction (d-ions, density-density only)
    ---------------------------------------------------
        same orbital, opposite spin          → U_dd
        different orbitals, same spin        → U_dd − 3 J_H
        different orbitals, opposite spin    → U_dd − 2 J_H
    Spin-flip and pair-hopping are excluded, consistent with the density-density-only approximation stated in the paper.

    Ligand interaction
    ------------------
    Only same-mode p_(m,s) density-density repulsion (U_pp) is retained.
    Different-m correlations on the ligand are excluded by construction (three independent modes).

    Returns
    -------
    (H, basis) : (153×153 complex Hermitian, list of 153 sorted index pairs)
    """
    # --- Single-particle 18×18 block ---
    H1 = np.zeros((3*_N_ORB, 3*_N_ORB), dtype=complex)
    H1[0:_N_ORB,   0:_N_ORB] = H_soc_cf
    H1[2*_N_ORB:3*_N_ORB, 2*_N_ORB:3*_N_ORB] = H_soc_cf
    for m in range(3):
        for s in range(2):
            p_i = _N_ORB + 2 * m + s
            H1[p_i, p_i] += Delta_CT
            d1_i, d2_i = 2 * m + s, 2*_N_ORB + 2 * m + s
            H1[d1_i, p_i] = H1[p_i, d1_i] = t_pd
            H1[d2_i, p_i] = H1[p_i, d2_i] = t_pd
    
    H1 = 0.5 * (H1 + H1.conj().T)

    basis = list(itertools.combinations(range(3*_N_ORB), 2))
    idx_of = {st: i for i, st in enumerate(basis)}
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)

    for idx, (p_, q_) in enumerate(basis):
        bits = (1 << p_) | (1 << q_)

        # --- Diagonal single-particle energy ---
        H[idx, idx] += H1[p_, p_].real + H1[q_, q_].real

        # --- On-site Coulomb / Kanamori (density-density) ---
        site_p, m_p, s_p = _multiorbital_site_of(p_)
        site_q, m_q, s_q = _multiorbital_site_of(q_)
        if site_p == site_q:
            if site_p == 'p':
                if m_p == m_q:
                    H[idx, idx] += U_pp
            else:
                if m_p == m_q:
                    # Same orbital → Pauli forces opposite spin.
                    H[idx, idx] += U_dd
                else:
                    H[idx, idx] += (U_dd - 3.0 * J_H) if s_p == s_q else (U_dd - 2.0 * J_H)

        # --- Off-diagonal single-particle hopping (t_pd, SOC within d-shell) ---
        for source in (p_, q_):
            for target in range(3*_N_ORB):
                if target == source:
                    continue
                hij = H1[target, source]
                if abs(hij) < 1e-13:
                    continue
                res = _apply_hop(bits, target, source)
                if res is None:
                    continue
                new_bits, sign = res
                new_sp = tuple(sorted(i for i in range(3*_N_ORB) if (new_bits >> i) & 1))
                H[idx, idx_of[new_sp]] += hij * sign

    H = 0.5 * (H + H.conj().T)
    return H, basis

def _multiorbital_reference_vector(basis, u_local, u_offset, v_local, v_offset, n_sp=None):
    """Antisymmetrized 2-hole Slater-determinant vector |u>_site1 x |v>_site2 for single-ion
    Kramers vectors u_local, v_local, embedded at disjoint site offsets (0 for d1, 12 for d2 in the 18-state layout).
    n_sp : number of spin-orbitals of the cluster; inferred from `basis` (all 2-subsets of range(n_sp)) when omitted -> 18 for d-p-d, 24 for d-p-p'-d'."""
    n_sp = (max(max(st) for st in basis) + 1) if n_sp is None else int(n_sp)
    u = np.zeros(n_sp, dtype=complex); u[u_offset:u_offset + 6] = u_local
    v = np.zeros(n_sp, dtype=complex); v[v_offset:v_offset + 6] = v_local
    vec = np.array([u[i] * v[j] - u[j] * v[i] for (i, j) in basis], dtype=complex)
    return vec / np.linalg.norm(vec)

def _lowdin_branch_energies(evals, evecs, V, E_guess, spread, n_seeds: int = 15, n_iter: int = 200, tol: float = 1e-10, dedup_tol: float = 1e-6):
    """
    Exact quasi-degenerate (Löwdin/Feshbach) projection of H onto the reference subspace span(V):

        H_eff(E) = P H P + P H Q (E − QHQ)⁻¹ Q H P ,   Q = 1 − P

    via the Schur-complement identity  P (E − H)⁻¹ P = (E − H_eff(E))⁻¹, evaluated from the full spectrum {evals, evecs} of H —
    exact, no powerseries truncation in t_pd. For every true eigenvalue E_n of H with nonzero P-weight, H_eff(E_n) has E_n as an
    exact eigenvalue;each branch is found by its own fixed-point iteration E ← λ_min(H_eff(E)) from n_seeds starting energies across `spread` (one shared trial
    energy does not converge to the true spectrum), then deduplicated.
    Returns (uniq_branches: sorted distinct branch energies, Heff_at_mean: H_eff at the mean branch energy — for the Pauli-tensor diagnostic only, H_eff being energy dependent).
    """
    Vc = evecs.conj().T @ V
    dim = V.shape[1]

    def Heff_at(E):
        denom = E - evals
        denom = np.where(np.abs(denom) < 1e-15, np.copysign(1e-15, denom), denom)
        G = (Vc.conj().T * (1.0 / denom)) @ Vc
        Heff = E * np.eye(dim) - np.linalg.inv(G)
        return 0.5 * (Heff + Heff.conj().T)

    found = []
    for s in np.linspace(E_guess - spread, E_guess + spread, n_seeds):
        E = float(s)
        for _ in range(n_iter):
            ev = np.linalg.eigvalsh(Heff_at(E))
            E_new = float(ev[np.argmin(np.abs(ev - E))])
            if abs(E_new - E) < tol:
                E = E_new
                break
            E = E_new
        found.append(E)

    uniq = []
    for e in sorted(found):
        if not uniq or abs(e - uniq[-1]) > dedup_tol:
            uniq.append(e)
    return uniq, Heff_at(float(np.mean(uniq)))

def _intra_doublet_kappa(basis, evals_2hole, evecs_2hole, pairs):
    """
    Intra-doublet exchange κ = (E_triplet − E_singlet) / 4s² in the NORMALISED pseudospin τ basis (τ_z = ±1/2).

    Wigner–Eckart identity: on a Kramers doublet the physical spin operator is Ŝ = s_eff · σ
    (eigenvalues ±s_eff), so the physical Heisenberg coupling is J = κ / (4 s_eff²) .

    Without this factor, J_A1g_diag would over-strengthen the AFM Weiss field by a doublet-dependent
    amount (up to ~2 orders of magnitude for Γ7b in the strong-SOC regime, where s_eff ≪ 1/2).

    Caveats:
    - Only the density-density Kanamori terms are included; spin-flip and pair-hopping are omitted.
    - For this symmetric, inversion-centred 2-site cluster, the result is isotropic (Heisenberg-like).
      Lower-symmetry clusters or the full rotationally invariant Kanamori interaction may generate Ising/
      DM-like terms that are not captured here.
    - The ligand consists of 3 independent modes with equal t_pd and no O-2p multiplet/Hund term.
    - The trace-based J_B1g extraction is exact for a scalar off-diagonal block; deviation is flagged but
      the scalar value is still used.
    """
    kappas = np.zeros(len(pairs), dtype=float)
    for i, (name, D) in enumerate(pairs):
        V = np.column_stack([
            _multiorbital_reference_vector(basis, D[a], 0, D[b], 12)
            for a in ('up', 'dn') for b in ('up', 'dn')
        ])
        branches, Heff = _lowdin_branch_energies(
            evals_2hole, evecs_2hole, V,
            E_guess=2.0 * D['energy'], spread=0.3)

        off_diag_max = max(
            (abs(np.real(np.trace(np.kron(_SIGMA_2x2[a], _SIGMA_2x2[b]).conj().T @ Heff))) / 4.0
             for a in _SIGMA_2x2 for b in _SIGMA_2x2
             if a != '0' and b != '0' and a != b),
            default=0.0,
        )
        if off_diag_max > _ANISO_WARN_TRESH:
            _scf_log("RMFT-WARN", f"kappa_superexchange_doublets[{name}]: non-Heisenberg component detected (|coeff|max={off_diag_max:.2e})")

        kappa_raw = 0.0 if len(branches) == 1 else float(branches[-1] - branches[0])
        
        # In the spin-switch regime Δ_tetra/λ_SOC ≳ 0.5 the Γ6 doublet becomes mostly xy-like and s_eff drops from ~0.39 to ~0.09. The physical exchange is instead set by the orbital-mediated contribution scale
        s_eff  = abs(D['sz_up'])
        s_safe = max(s_eff, _S_EFF_FLOOR)
        denom  = max(4.0 * s_safe ** 2, _WIGNER_ECKART_DIV_FLOOR)
        kappas[i] = kappa_raw / denom

        if s_eff < _S_EFF_FLOOR:
            _scf_log("RMFT-WARN", f"kappa_superexchange_doublets[{name}]: s_eff={s_eff:.4f} below floor {_S_EFF_FLOOR:.2f}; J_eff capped to avoid Wigner–Eckart divergence.")
    return kappas

def _cross_doublet_kappas(basis, evals_2hole, evecs_2hole, G6, pairs, B1g_t2g, verbose: bool = False, d2_offset: int = 12):
    """Cross-doublet coefficient J_B1g for the Γ6↔Γ7a / Γ6↔Γ7b channels.

    The model multiplies J_B1g by the physical operator product B1g_op(i)·B1g_op(j).
    On the 8 reference states {|Γ6 a⟩₁|Γ7 b⟩₂} ∪ {|Γ7 a⟩₁|Γ6 b⟩₂}, the product has matrix elements
    M[a,a']·conj(M[b',b]) = (M ⊗ M†)[(a,b),(a',b')] with M = U6† B1g_t2g U7 (2×2, no fermionic sign).
    The operator coefficient is

        J_B1g = Re Tr[(M ⊗ M†)† · H_block] / ‖M‖_F⁴ ,

    the exact analogue of the intra-doublet 4s² factor.  H_block is the energy-independent Bloch–des Cloizeaux
    effective Hamiltonian on span(V): the d = 8 exact eigenstates with the largest weight in span(V) are
    projected and Löwdin-orthonormalised, Φ = P Ψ (Ψ†P Ψ)^(-1/2),  H_eff = Φ diag(E) Φ†, so 4·c_zz reproduces
    E_triplet − E_singlet exactly for an isotropic block. 

    Hard-exclusion guards:
      * reference subspace (near-)singular: σ_min/σ_max < 1e-10
      * projected reference-space overlap singular: s[0] < 1e-8
      * Γ6↔Γ7 block too small: ‖M‖_F² < _B1G_BLOCK_MIN_FRAC · ‖B1g_op‖_F²
      * ‖M‖_F⁴ ≤ 0 (degenerate / null block)

    A soft warning is issued when the reference space is not isolated
    (min P-weight of exact eigenstates < 0.5); the projection is still computed but flagged as unreliable.
    """
    U6      = np.column_stack((G6['up'], G6['dn']))
    B_norm2 = float(np.linalg.norm(B1g_t2g, 'fro') ** 2)

    def _kappa_for_pair(D_ref, label) -> float:
        # ---- Build the 8-dim reference subspace ----
        refs = [_multiorbital_reference_vector(basis, G6[a],    0, D_ref[b], d2_offset)
                for a in ('up', 'dn') for b in ('up', 'dn')] \
            + [_multiorbital_reference_vector(basis, D_ref[a],  0, G6[b],    d2_offset)
                for a in ('up', 'dn') for b in ('up', 'dn')]
        V = np.column_stack(refs)                           # (153, 8)

        # ---- Guard 1: reference subspace (near-)singular ----
        sv = np.linalg.svd(V, compute_uv=False)
        if sv[-1] / sv[0] < 1e-10:
            if verbose:
                _scf_log("RMFT-WARN", f"cross-doublet {label} reference subspace is (near-)singular (σ_min/σ_max = {sv[-1] / sv[0]:.2e}); κ_B1g_{label} = 0.0")
            return 0.0

        # ---- 2×2 transition block of the B1g operator ----
        U7 = np.column_stack((D_ref['up'], D_ref['dn']))
        M  = U6.conj().T @ B1g_t2g @ U7
        R2 = float(np.linalg.norm(M, 'fro') ** 2)

        # ---- Guard 2: Γ6↔Γ7 B1g block too small ----
        if R2 < max(_B1G_BLOCK_MIN_FRAC * B_norm2, _WIGNER_ECKART_DIV_FLOOR):
            if verbose:
                _scf_log("RMFT-WARN",
                    f"cross-doublet {label}: ‖B1g_op^67‖_F² = {R2:.2e} is {R2 / B_norm2:.1e} of ‖B1g_op‖_F² "
                    f"(< {_B1G_BLOCK_MIN_FRAC:.0e}): B1g⊗B1g channel not resolvable (selection rule); κ_B1g_{label} = 0.0")
            return 0.0

        # ---- Bloch–des Cloizeaux H_eff on span(V) ----
        d   = V.shape[1]
        A   = evecs_2hole.conj().T @ V
        w   = np.sum(np.abs(A) ** 2, axis=1)
        sel = np.argsort(w)[::-1][:d]
        C   = A[sel, :].conj().T
        s, U = np.linalg.eigh(C.conj().T @ C)

        # ---- Guard 3: projected overlap singular ----
        if s[0] < 1e-8:
            if verbose:
                _scf_log("RMFT-WARN", f"cross-doublet {label}: projected reference-space overlap is singular; κ_B1g_{label} = 0.0")
            return 0.0

        # ---- Soft warning: reference space not isolated ----
        if float(np.min(w[sel])) < 0.5 and verbose:
            _scf_log("RMFT-WARN", f"cross-doublet {label}: reference space not isolated (min P-weight of exact eigenstates = {float(np.min(w[sel])):.2f}); quasi-degenerate projection unreliable.")

        Phi  = C @ ((U * s ** -0.5) @ U.conj().T)         # Löwdin-orthonormalised projected eigenstates
        Heff = (Phi * evals_2hole[sel]) @ Phi.conj().T
        Heff = 0.5 * (Heff + Heff.conj().T)

        T  = np.kron(M, M.conj().T)
        t2 = float(np.real(np.vdot(T, T)))
        if t2 <= 0.0:
            return 0.0
        return float(np.real(np.vdot(T, Heff[0:4, 4:8])) / t2)
    return np.array([_kappa_for_pair(D_ref, label) for (D_ref, label) in pairs], dtype=float)

def _pauli_coeff(H, a, b):
    op = np.kron(_SIGMA_2x2[a], _SIGMA_2x2[b])
    return float(np.real(np.trace(op.conj().T @ H)) / 4.0)

def _unique_q_pairs(fs_pts: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Using the full D₂h magnetic Shubnikov symmetry of the collinear AFM state.
    
    Returns
    -------
    i_idx, j_idx : upper-triangle pair indices  (n_pairs,)
    unique_q     : canonical q-vectors in (−π, π]²  (n_unique, 2)
    inv_idx      : pair → unique_q row mapping  (n_pairs,)
    """
    i_idx, j_idx = np.triu_indices(len(fs_pts))
    q_raw = fs_pts[i_idx] - fs_pts[j_idx]

    # Fold into (−π, π] interval
    q_arr = (q_raw + np.pi) % (2.0 * np.pi) - np.pi

    # Convert to integer representation (robust against floating-point noise)
    q_int_raw = np.rint(q_arr * _Q_UNIQUE_SCALE).astype(np.int64)

    # Map to the first quadrant using component-wise absolute values. This step combines:
    #   - inversion symmetry (q → −q)
    #   - two-fold rotations / mirrors combined with time reversal (σ_x·T : q_x → −q_x,  σ_y·T : q_y → −q_y) ⇒ χ(q) = χ(|q_x|, |q_y|)
    qx_canon = np.abs(q_int_raw[:, 0])
    qy_canon = np.abs(q_int_raw[:, 1])
    q_int_canon = np.stack([qx_canon, qy_canon], axis=1)

    # Unique canonical vectors and inverse index for each original pair
    u_int, inv_idx = np.unique(q_int_canon, axis=0, return_inverse=True)

    # Convert back to floating-point (units of 2π/_Q_UNIQUE_SCALE)
    unique_q = u_int.astype(np.float64) / _Q_UNIQUE_SCALE
    # Ensure the points stay in the correct BZ (should already be in [0, π])
    unique_q = (unique_q + np.pi) % (2.0 * np.pi) - np.pi
    if verbose:
        plt.figure(figsize=(8,6))
        plt.hexbin(unique_q[:,0], unique_q[:,1], gridsize=50, cmap='viridis')
        plt.colorbar(label='Count')
        plt.xlabel('$q_x$')
        plt.ylabel('$q_y$')
        plt.title(f"Unique q distribution (N_q={len(unique_q)})")
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.savefig('unique_q_hist.png', dpi=150)
        plt.close()
    return i_idx, j_idx, unique_q, inv_idx

def _build_H_AB_block(kx: np.ndarray, ky: np.ndarray, Tx_op: np.ndarray, Ty_op: np.ndarray, g_t: float) -> np.ndarray:
    """
    Vectorized, orbital-selective inter-sublattice hopping block:
        H_AB(k) = -2·g_t·[cos(kx)·Tx_op + cos(ky)·Ty_op]      shape (N_k, 6, 6)
    """
    cos_kx = np.cos(kx)[:, None, None]
    cos_ky = np.cos(ky)[:, None, None]
    return (-2.0 * g_t * (cos_kx * Tx_op[None, :, :] + cos_ky * Ty_op[None, :, :])).astype(complex)

def _expand_M_channels(M_channels: np.ndarray) -> np.ndarray:
    """
    (3,) [M_Γ6, M_Γ7a, M_Γ7b] -> (6,) channel-resolved, Kramers-doubled M, matching the [Γ6,Γ6,Γ7a,Γ7a,Γ7b,Γ7b] orbital order of sz_op/J_A1g_diag.
    Scalar (or size-1) M is broadcast to all 3 channels, preserving callers with shared order parameter (e.g. 4-site cluster-ED embedding, which uses only J_A1g_diag[0]).
    """
    arr = np.atleast_1d(np.asarray(M_channels, dtype=float))
    if arr.size == 1:
        arr = np.full(_N_CHANNELS, float(arr[0]))
    return np.repeat(arr, 2)

def _adjugate(A: np.ndarray) -> np.ndarray:
    """adj(A) = det(A)·A⁻¹ of a real square matrix via the SVD, A = UΣVᵀ  ⇒  adj(A) = det(U)det(V)·V diag(Π_{j≠i} σ_j) Uᵀ  — finite also when A is (near-)singular,
    which is exactly where the floor-regularised RPA determinant is used:  χ_RPA = adj(M)·χ₀ / det_safe."""
    U, s, Vt = np.linalg.svd(A)
    others = np.array([np.prod(np.delete(s, i)) for i in range(len(s))])
    return float(np.linalg.det(U) * np.linalg.det(Vt)) * ((Vt.T * others) @ U.T)

def _gamma_splitting(lambda_soc: float, Delta_tetra: float, Delta_inplane: float = 0.0) -> float:
    """Fast standalone Δ_CF = E(Γ7a) − E(Γ6) (eV) for the crystal-field pre-scan only."""
    H_soc_cf, Lx_t2g, Ly_t2g, _, Sz_full, LS_op = _t2g_soc_cf_operators(lambda_soc, Delta_tetra, Delta_inplane)
    evals, evecs_soc = np.linalg.eigh(H_soc_cf)
    B1g_t2g = Lx_t2g @ Lx_t2g - Ly_t2g @ Ly_t2g
    doublets = []
    for i in (0, 2, 4):
        v = evecs_soc[:, i]
        vp = evecs_soc[:, i + 1]
        U = np.column_stack((v, vp))
        sz_vals, sz_vecs = np.linalg.eigh(U.conj().T @ Sz_full @ U)
        if sz_vals[1] < 0:
            sz_vals = sz_vals[::-1]
            sz_vecs = sz_vecs[:, ::-1]
        dn = U @ sz_vecs[:, 0]
        up = U @ sz_vecs[:, 1]
        doublets.append({
            'idx': i, 'energy': float(evals[i]),
            'ls_val': float(np.real(v.conj() @ LS_op @ v)),
            'v': v, 'v_p': vp, 'up': up, 'dn': dn,
            'sz_up': float(np.real(sz_vals[1])), 'sz_dn': float(np.real(sz_vals[0])),
        })
    doublets.sort(key=lambda d: d['ls_val'])
    G6 = doublets[0]                # Γ6: most negative <L·S> → j_eff = 1/2
    G7_candidates = doublets[1:]    # two Γ7 candidates

    # Γ7a ≡ the doublet whose B1g (Lx²-Ly²) matrix element TO Γ6 is larger in magnitude. |μz| tracks this closely for Δ_tetra>0 but can decouple
    # from it once Δ_tetra<0 or Δ_B1g_static admixes the states differently — silently moving the real coupling onto the doublet everything downstream still calls "Γ7b".
    U6 = np.column_stack((G6['up'], G6['dn']))
    for d in G7_candidates:
        U7 = np.column_stack((d['up'], d['dn']))
        d['b1g_to_G6'] = float(np.linalg.norm(U6.conj().T @ B1g_t2g @ U7))
    G7_candidates.sort(key=lambda x: -x['b1g_to_G6'])
    G7a, G7b = G7_candidates[0], G7_candidates[1]
    return float(G7a['energy'] - G6['energy'])

def compute_and_log_hessian_diagnostics(target_doping: float, g_t: float, g_J: float, results: Dict[str, Optional[dict]], solvers: Dict[str, 'RMFT_Solver']):
    """
    Unified post-SCF Hessian diagnostics for ALL scenarios (ref / normal / SC_Q0).
    Per scenario (parallel):
      1. F_can — bulk-only mean-field F at the probe point
      2. Hessian (always, expensive) — beyond-MF F_bdg (authoritative), eigenvalues, and bulk-only F_bulk (for the cross-check).
      3. Inline reliability: converged + not past_qcp + no saddle.
    """
    scenario_present = [n for n, r in results.items() if r is not None]
    if not scenario_present:
        return results

    def _reliability_inline(res: dict, hess: dict) -> Tuple[bool, str]:
        if not res.get("converged", False):
            if res.get("past_qcp", False) and res.get("scf_iteration_converged", False):
                return False, f"past QCP (det={res.get('det_afm_final', float('nan')):+.4f})"
            return False, "not converged"
        if res.get("past_qcp", False):
            return False, f"past QCP (det={res.get('det_afm_final', float('nan')):+.4f})"
        if hess.get('error'):
            return False, "Hessian failed"
        if not all(e > -1e-6 for e in hess['eigenvalues']):
            return False, "saddle"
        return True, "ok"

    def _compute_one(name: str) -> Tuple[str, dict]:
        solver = solvers[name]
        res    = results[name]
        M         = np.asarray(res['M'], dtype=float)
        Q         = float(res['Q'])
        Delta_vec = np.asarray(res['Delta_vec'], dtype=complex)
        F67_vec   = np.asarray(res['F67_vec'], dtype=complex)
        n_kspace  = float(res['n_kspace'])
        mu        = float(res['mu'])
        V_JT_corr = float(res['V_JT_corr'])
        vertex_cache = res['vertex_cache']

        _, F_cluster, *_ = solver._make_vertex_params(target_doping, M, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, vertex_cache=vertex_cache)
        F_can = float(F_cluster['F_can'])

        hess = {'F_bdg': float('nan'), 'F_bulk': float('nan'), 'eigenvalues': None, 'lambda_min': float('nan')}
        try:
            _hk = solver.compute_hessian(target_doping, M, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, V_JT_corr, vertex_cache=vertex_cache)
            hess = {
                'F_bdg':       float(_hk['F_bdg']),      # beyond-MF
                'F_bulk':      float(_hk['F_bulk']),     # bulk-only
                'eigenvalues': _hk['eigenvalues'],
                'lambda_min':  float(_hk.get('lambda_min', float('nan'))),
            }
        except Exception as e:
            hess['error'] = str(e)
        return name, {'F_can': F_cluster['F_can'], 'hessian_result': hess}

    with ThreadPoolExecutor(max_workers=min(3, len(scenario_present))) as pool:
        futures = {pool.submit(_compute_one, n): n for n in scenario_present}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, out = fut.result()
                results[name]['F_bulk']         = out['hessian_result']['F_bulk']   # bulk (for cross-check)
                results[name]['hessian_result'] = out['hessian_result']
                results[name]['F_can']          = out['F_can']
            except Exception as e:
                results[name]['F_bulk']         = float('nan')
                results[name]['hessian_result'] = {'eigenvalues': None, 'error': str(e)}
                results[name]['F_can']          = float('nan')

    for name in scenario_present:
        ok, why = _reliability_inline(results[name], results[name]['hessian_result'])
        results[name]['hess_ok']  = ok
        results[name]['hess_why'] = why

    _scf_log("HESS-DIAG", f"  {'scenario':<10s} {'F_bdg [eV]':>13s} {'λ_min':>10s} {'status':>25s}")
    _scf_log("HESS-DIAG", "  " + "-" * 62)
    for name in scenario_present:
        r = results[name]
        eigs = r['hessian_result'].get('eigenvalues')
        lmin = float(eigs[0]) if eigs is not None else float('nan')
        status = "✓ ok" if r['hess_ok'] else f"✗ {r['hess_why']}"
        _scf_log("HESS-DIAG", f"  {name:<10s} {r['F_bdg']:>+13.6f} {lmin:>+10.4f} {status:>25s}")

    # F_bulk ↔ F_can cross-check: both bulk-only via different code paths,
    _pairs = [(results[n].get('F_bulk'), results[n].get('F_can')) for n in scenario_present]
    _pairs = [(a, b) for a, b in _pairs if np.isfinite(a) and np.isfinite(b)]
    if _pairs:
        _dmax = max(abs(a - b) for a, b in _pairs)
        if _dmax > 1e-4:
            _scf_log("HESS-DIAG", f"  ⚠ |F_bulk(Hess) − F_can(cluster)| = {_dmax:.2e} eV  (should be ~0)")

    # --- Free-energy comparison (beyond-MF) + Q ratio ---
    _ALL = ('ref', 'normal', 'SC_Q0')
    if all(n in scenario_present for n in _ALL):
        en  = {n: results[n]['F_bdg']   for n in _ALL}
        rel = {n: results[n]['hess_ok'] for n in _ALL}

        _scf_log("FREE-ENERGY",
            f"  F_ref    = {en['ref']:+.6f} eV  {'✓' if rel['ref'] else '⚠ ' + results['ref']['hess_why']}")
        _scf_log("FREE-ENERGY",
            f"  F_normal = {en['normal']:+.6f} eV  "
            f"(ΔE = {(en['normal']-en['ref'])*1e3:+8.3f} meV)  {'✓' if rel['normal'] else '⚠ ' + results['normal']['hess_why']}")
        _scf_log("FREE-ENERGY",
            f"  F_SC_Q0  = {en['SC_Q0']:+.6f} eV  "
            f"(ΔE = {(en['SC_Q0']-en['ref'])*1e3:+8.3f} meV)  {'✓' if rel['SC_Q0'] else '⚠ ' + results['SC_Q0']['hess_why']}")
        if not all(rel.values()):
            _scf_log("FREE-ENERGY", "  → comparison NOT trustworthy")
        elif en['ref'] < en['normal'] and en['ref'] < en['SC_Q0']:
            _scf_log("FREE-ENERGY", "  → SC+JT is the ground state ✓")
        else:
            _scf_log("FREE-ENERGY", "  → SC+JT is NOT the lowest energy state ✗")

        Q_sc = float(results['ref'].get('Q', 0.0))
        Q_n  = float(results['normal'].get('Q', 0.0))
        Q_ratio = abs(Q_sc) / max(abs(Q_n), 1e-12)
        hess_ok_ref = results['ref']['hess_ok']

        # K_eff at Q = 0 for the NORMAL state.
        Kn = results['normal']['K_eff']
        if np.isfinite(Kn):
            verdict = (
                "✓ SC-enhanced distortion" if (hess_ok_ref and Kn > 0 and Q_ratio > 5.0)
                else "— Q is static-field response" if Q_ratio <= 5.0
                else "⚠ large ratio but H_SC not verified min" if Kn <= 0
                else "? unclear")
        else:
            verdict = "? K_eff(normal, Q=0) unavailable"

        _scf_log("FREE-ENERGY",
            f"  Q(SC)={Q_sc:+.5f} Å  Q(N)={Q_n:+.5f} Å  "
            f"|Q_SC/Q_N|={Q_ratio:.1f}  K_eff(N,Q=0)={Kn:+.4f} eV/Å²  → {verdict}")
    return results

def _grid_sym_permutation(fs_pts: np.ndarray, op: str) -> Tuple[np.ndarray, bool]:
    """Index permutation of a symmetry image of the FS sample on the _NK×_NK k-grid: op ∈ {'inv': k→−k, 'mx': kx→−kx, 'my': ky→−ky}.
    Returns (perm, closed): perm[i] = index of the image of point i, closed = every image is in the sample (then perm is a true involution)."""
    step = 2.0 * np.pi / _NK
    ix = np.round((fs_pts[:, 0] + np.pi) / step).astype(np.int64) % _NK
    iy = np.round((fs_pts[:, 1] + np.pi) / step).astype(np.int64) % _NK
    if op == 'swap':
        jx, jy = iy, ix
    else:
        jx = (-ix) % _NK if op in ('inv', 'mx') else ix
        jy = (-iy) % _NK if op in ('inv', 'my') else iy
    keys = ix * _NK + iy
    order = np.argsort(keys)
    sk = keys[order]
    tk = jx * _NK + jy
    pos = np.clip(np.searchsorted(sk, tk), 0, len(sk) - 1)
    found = bool(np.all(sk[pos] == tk))
    perm = order[pos]
    return perm, bool(found and np.array_equal(perm[perm], np.arange(len(perm))))

def _on_tq_boundary(fs_pts: np.ndarray) -> np.ndarray:
    """Mask of the points on the diamond boundary |kx|+|ky| = π: T_Q maps this set onto itself in pairs (no fixed point); each partner carries weight ½."""
    return np.abs(np.abs(fs_pts[:, 0]) + np.abs(fs_pts[:, 1]) - np.pi) < _BDY_TOL

def _tq_partner(fs_pts: np.ndarray) -> np.ndarray:
    """Sample index of the T_Q image k+(π,π) of every point (−1: the image is not in the sample), on the _NK×_NK grid."""
    step = 2.0 * np.pi / _NK
    ix = np.round((fs_pts[:, 0] + np.pi) / step).astype(np.int64) % _NK
    iy = np.round((fs_pts[:, 1] + np.pi) / step).astype(np.int64) % _NK
    keys = ix * _NK + iy
    order = np.argsort(keys)
    sk = keys[order]
    tk = ((ix + _NK // 2) % _NK) * _NK + (iy + _NK // 2) % _NK
    pos = np.clip(np.searchsorted(sk, tk), 0, len(sk) - 1)
    return np.where(sk[pos] == tk, order[pos], -1)

def _check_fs_closure(fs_pts: np.ndarray, fs_weights: np.ndarray, tol: float = _ODD_VTX_TOL) -> Tuple[bool, bool, bool, bool, float]:
    """Closure of the FS sample (assumptions behind the odd-parity projections and the T_Q image terms of the sector kernel):
        closed_inv / closed_mx / closed_my : the sample is closed under k→−k / kx→−kx / ky→−ky;
        weight_sym  : weights are symmetric under k→−k AND the sample is a T_Q fundamental domain: no interior point has its T_Q image in the sample, every boundary point
                      (|kx|+|ky| = π) has it, with the same weight;
        weight_asym : largest relative violation of that weight symmetry (inf if the T_Q pairing itself fails).
    The interior must NOT be T_Q-closed: the sector kernel adds the images V(k_i − T_Q k_j) explicitly, a T_Q-closed interior would count every state twice."""
    w = np.asarray(fs_weights, dtype=float)
    perm_inv, closed_inv = _grid_sym_permutation(fs_pts, 'inv')
    _,         closed_mx = _grid_sym_permutation(fs_pts, 'mx')
    _,         closed_my = _grid_sym_permutation(fs_pts, 'my')
    w_max = max(float(w.max()), 1e-30)
    asym_inv = float(np.max(np.abs(w - w[perm_inv]))) / w_max if closed_inv else float('inf')
    tq, bdy = _tq_partner(fs_pts), _on_tq_boundary(fs_pts)
    if np.any((tq >= 0) != bdy):
        asym_tq = float('inf')
    else:
        asym_tq = float(np.max(np.abs(w[bdy] - w[tq[bdy]]))) / w_max if np.any(bdy) else 0.0
    weight_asym = max(asym_inv, asym_tq)
    return bool(closed_inv), bool(closed_mx), bool(closed_my), bool(weight_asym < tol), float(weight_asym)

def _swap_resolved_top(K: np.ndarray, perm: np.ndarray) -> Tuple[float, float, float]:
    """Top eigenvalues of the symmetric FS kernel K restricted to the σd-even and σd-odd subspaces of the FS sample, plus the covariance defect.

    perm : involution of the sample for the diagonal mirror (kx, ky) → (ky, kx), closed sample with σd-invariant weights).
    Returns (λ_even, λ_odd, defect), defect = max|K − P K Pᵀ| / max|K|. If K commutes with the swap (a C4-covariant kernel) λ_even / λ_odd are the exact top eigenvalues of the two D4h irreps that share the sector
    (A1g / B1g for mx = my = +1, B2g / A2g for mx = my = −1).  If it does not (defect > 0), they are still the best swap-even / swap-odd pairing states (variational), but no longer exact irreps."""
    n = len(perm)
    ar = np.arange(n)
    idx_fix = np.flatnonzero(perm == ar)
    idx_a = np.flatnonzero(perm > ar)
    idx_b = perm[idx_a]
    nf, npair = len(idx_fix), len(idx_a)
    B_e = np.zeros((n, nf + npair))
    B_o = np.zeros((n, npair))
    B_e[idx_fix, np.arange(nf)] = 1.0
    r = np.arange(npair)
    B_e[idx_a, nf + r] = B_e[idx_b, nf + r] = 1.0 / math.sqrt(2.0)
    B_o[idx_a, r] = 1.0 / math.sqrt(2.0)
    B_o[idx_b, r] = -1.0 / math.sqrt(2.0)
    K = 0.5 * (K + K.T)
    lam_e = float(np.linalg.eigvalsh(B_e.T @ K @ B_e)[-1]) if B_e.shape[1] else float('nan')
    lam_o = float(np.linalg.eigvalsh(B_o.T @ K @ B_o)[-1]) if B_o.shape[1] else float('nan')
    defect = float(np.max(np.abs(K - K[np.ix_(perm, perm)])) / max(float(np.max(np.abs(K))), 1e-300))
    return lam_e, lam_o, defect

def _sc_gap_amplitude(res: dict) -> np.ndarray:
    if isinstance(res, dict):
        Delta_vec = res.get('Delta_vec')
        if Delta_vec is not None:
            return np.asarray(Delta_vec, dtype=complex)
    return np.zeros(_N_DELTA, dtype=complex)

def _fd_grad_hessian(F, x0: np.ndarray, eps: np.ndarray, F0: float = None, need_grad: bool = True) -> Tuple[float, Optional[np.ndarray], np.ndarray]:
    """
    Generic central finite-difference gradient and Hessian of a scalar function F: R^n -> R at x0,
    with independent per-component step sizes `eps`.

    Diagonal:      H[i,i] = (F(x0+e_i) - 2F0 + F(x0-e_i)) / eps_i^2                  (3-point stencil)
    Off-diagonal:  H[i,j] = (F(++) - F(+-) - F(-+) + F(--)) / (4 eps_i eps_j)        (4-point cross stencil)
    Gradient:      grad[i] = (F(x0+e_i) - F(x0-e_i)) / (2 eps_i)                     (shares the diagonal's Fp/Fm, no extra F-evals)

    This is the single numerical core shared by compute_dF_dM_channels_and_hessian (n=3, M-channels), compute_hessian (n=5, M+Q+|Δ|),
    and compute_dF_dDelta_and_d2F (n=_N_DELTA, the full extended Δ vector). Each caller only differs in how it builds the closure F(x)
    and in what it does with (grad, Hess) afterwards — the stencil arithmetic itself was previously copy-pasted three times.

    need_grad=False skips writing the gradient but costs no extra F-evaluations either way.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.shape[0]
    if F0 is None:
        F0 = F(x0)
    grad = np.zeros(n, dtype=float) if need_grad else None
    Hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps[i]; xm[i] -= eps[i]
        Fp, Fm = F(xp), F(xm)
        if need_grad:
            grad[i] = (Fp - Fm) / (2.0 * eps[i])
        Hess[i, i] = (Fp - 2.0 * F0 + Fm) / eps[i]**2

    for i in range(n):
        for j in range(i + 1, n):
            xpp, xmm, xpm, xmp = x0.copy(), x0.copy(), x0.copy(), x0.copy()
            xpp[i] += eps[i]; xpp[j] += eps[j]
            xmm[i] -= eps[i]; xmm[j] -= eps[j]
            xpm[i] += eps[i]; xpm[j] -= eps[j]
            xmp[i] -= eps[i]; xmp[j] += eps[j]
            val = (F(xpp) - F(xpm) - F(xmp) + F(xmm)) / (4.0 * eps[i] * eps[j])
            Hess[i, j] = Hess[j, i] = val
    return F0, grad, Hess

def _lm_damping(lam_min: float, mu_base: float, scale: float, near_critical_mu: float = None) -> float:
    """
    Levenberg-Marquardt damping parameter μ_LM from the softest curvature eigenvalue λ_min of a
    (possibly indefinite) Hessian, shared by the Q / M-channels / Δ_s,Δ_d Newton steps in the SCF loop.

        λ_min ≫ 0  (deep stable minimum)      -> μ small          -> fast, lightly-damped Newton step
        λ_min ≈ 0  (flat / near a QCP)         -> μ = near_critical_mu or mu_base -> cautious step
        λ_min ≪ 0  (saddle / unstable)         -> μ = |λ_min| + mu_base -> guarantees a PD damped Hessian

    `scale` is the physical curvature scale (typically t_eff or K_bare) that makes λ_min/scale
    dimensionless in the stable branch. `near_critical_mu` lets a caller override the flat-λ_min
    branch (the M-channel step reduces μ as |Δ| grows, to unfreeze once SC-AFM coupling turns on;
    Q and Δ use a plain mu_base there)
    """
    if lam_min > _MATH_EPS:
        return max(mu_base / (1.0 + lam_min / scale), mu_base * 0.1)
    elif lam_min < -_MATH_EPS:
        return abs(lam_min) + mu_base
    else:
        return mu_base if near_critical_mu is None else near_critical_mu

def _q_fd_step(Q: float) -> Tuple[float, float]:
    """
    Canonical central finite-difference step size for every derivative taken with respect to the B1g distortion Q.
    h(Q) = sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q^2):
      - at Q=0:      h = sqrt(_JT_FD_H2_BASE) ≈ 1.7e-4, in the standard double-precision-optimal
                     range (~eps_machine^(1/4)) for a 3-point second-derivative stencil of an
                     O(1)-scale free energy.
      - for large Q: h grows ~linearly in |Q| (relative-step behaviour), smoothly (no kink),
    """
    h2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
    return math.sqrt(h2), h2

def _mbz_deduplication(k_all: np.ndarray, Emin: np.ndarray, kT: float) -> np.ndarray:
    # For collinear AFM with Q_AFM=(π,π), k and k+Q_AFM are the same physical state. Keeping both corrupts K_dd because φ_d(k+Q) = -φ_d(k) → negative cross-term.
    half_NK = _NK // 2
    
    ix_all = np.round((k_all[:, 0] + np.pi) * _NK / (2.0 * np.pi)).astype(int) % _NK
    iy_all = np.round((k_all[:, 1] + np.pi) * _NK / (2.0 * np.pi)).astype(int) % _NK
    code_all = (ix_all * _NK + iy_all).astype(np.int64)
    partner_code = (((ix_all + half_NK) % _NK) * _NK + (iy_all + half_NK) % _NK).astype(np.int64)

    order = np.argsort(code_all)
    code_sorted = code_all[order]
    pos = np.clip(np.searchsorted(code_sorted, partner_code), 0, len(code_sorted) - 1)
    partner_idx = order[pos]
    partner_idx = np.where(code_all[partner_idx] == partner_code, partner_idx, np.arange(len(code_all)))
    # At Q=0 (and near it) Emin[k] and Emin[partner_idx[k]] are degenerate to float precision
    dE_partner = Emin - Emin[partner_idx]
    combinatorial_mask = code_all < partner_code
    return np.where(np.abs(dE_partner) < _MBZ_DEGEN_FRAC * kT, combinatorial_mask, dE_partner < 0.0)

def _wave_project(V_channel: np.ndarray, phi_bra: np.ndarray, phi_ket: np.ndarray, fs_weights: np.ndarray) -> float:
    x_bra = phi_bra * fs_weights
    x_ket = phi_ket * fs_weights
    ns_bra = max(float(phi_bra @ (phi_bra * fs_weights)), 1e-30)
    ns_ket = max(float(phi_ket @ (phi_ket * fs_weights)), 1e-30)
    return float((x_bra @ V_channel @ x_ket) / (ns_bra * ns_ket))

def _pairing_basis_matrix(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """(len(kx), _N_IRREP) matrix stacking every basis form factor at each k-point, in _IRREP_NAMES order."""
    return np.stack([_IRREP_FF[name](kx, ky) for name in _IRREP_NAMES], axis=1)

def _is_onsite(name: str) -> bool:
    return _IRREP_KIND[name] == 'onsite'

def _is_nn_bond(name: str) -> bool:
    """Inter-sublattice (A–B / B–A) bond channel <=> T_Q-ODD form factor. These carry the B1g transverse Weiss field (TRW) in the inter-sublattice (A–B) block."""
    return _IRREP_KIND[name] == 'nn_AB'

def _is_nnn_bond(name: str) -> bool:
    """Diagonal (1,1) / axial (2,0) bond: even lattice parity, so it connects sites of the SAME sublattice (A–A / B–B block).
    The same-sublattice NNN / 3NN bonds also carry a TRW, but with a smaller amplitude (κ_nnn ≈ (t'/t)²·κ_NN) and in the A–A / B–B diagonal blocks"""
    return _IRREP_KIND[name] in ('nnn_diag', 'nnn_axis')

def _afm_shift_character(name: str) -> int:
    """σ = ±1 with φ(k + (π,π)) = σ φ(k) for the channel's form factor. Channels with σ = −1 (es, d, px, py) change sign under the AFM shift and therefore see the AFM-peaked
    V_spin(q ≈ (π,π)) as ATTRACTION (and may flip sign with det_afm_sc near the QCP -> EMA protection in compute_gap_eq_vectorized); σ = +1 (s, g, px2, py2) do not: like the
    on-site channel they see it repulsively and need no smoothing."""
    kx0, ky0 = 0.731, 0.419
    f  = float(_IRREP_FF[name](np.array([kx0]), np.array([ky0]))[0])
    fs = float(_IRREP_FF[name](np.array([kx0 + np.pi]), np.array([ky0 + np.pi]))[0])
    return int(np.sign(fs / f))

def _mirror_characters(name: str) -> Tuple[int, int]:
    """(χ_mx, χ_my): sign of the form factor under kx→−kx and ky→−ky.  A kernel V(k,k') that is mirror-symmetric (V depends on |q_x|, |q_y| only) can only connect
    channels with identical (χ_mx, χ_my): this is the D2h selection rule that separates g (−,−) from s/es/d (+,+) and px/px2 (−,+) from py/py2 (+,−)."""
    kx0, ky0 = 0.731, 0.419            # generic point: no accidental zero of any form factor
    f  = float(_IRREP_FF[name](np.array([kx0]),  np.array([ky0]))[0])
    fx = float(_IRREP_FF[name](np.array([-kx0]), np.array([ky0]))[0])
    fy = float(_IRREP_FF[name](np.array([kx0]),  np.array([-ky0]))[0])
    return int(np.sign(fx / f)), int(np.sign(fy / f))

def _gutzwiller_irrep_vector(g_Delta_s: float, g_Delta_bond: float, g_Delta_nnn: float) -> np.ndarray:
    """
    Per-irrep Gutzwiller renormalisation, canonical _IRREP_NAMES order, for ONE Gamma6-Gamma7 channel group. The on-site 's' channel gets its own
    g_Delta_s; every NN-bond irrep (es, d, px, py) shares g_Delta_bond, since Gutzwiller / no-double-occupancy renormalisation depends on the pair's
    real-space range (on-site vs. nearest-neighbour bond), not on which symmetry channel a given bond amplitude belongs to -- exactly how the
    original 2-channel code already reused a single g_Delta_ad/g_Delta_bd for its one bond channel (d-wave).
    """
    g = np.empty(_N_IRREP, dtype=float)
    for i, name in enumerate(_IRREP_NAMES):
        if _is_onsite(name):
            g[i] = g_Delta_s
        elif _is_nnn_bond(name):
            g[i] = g_Delta_nnn
        else:
            g[i] = g_Delta_bond
    return g

def _m2_weighted_J(J_q: np.ndarray, M: np.ndarray) -> float:
    M = np.asarray(M, dtype=float)
    w = M ** 2
    if (not np.all(np.isfinite(w))) or float(np.max(np.abs(M))) < 1e-12:
        w = np.zeros_like(w)
        w[0] = 1.0
    return float(np.average(np.asarray(J_q, dtype=float), weights=w))

def _gram_sqrt(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(S^{1/2}, S^{-1/2}) of the FS-weighted Gram matrix, normalised to unit mean diagonal (the scale drops out of E_cond and the projector); eigenvalues below _GRAM_RCOND·max are clipped."""
    S = 0.5 * (S + S.T)
    s, Us = np.linalg.eigh(S / max(float(np.mean(np.diag(S))), 1e-300))
    r = np.sqrt(np.maximum(s, _GRAM_RCOND * max(float(s[-1]), 1e-300)))
    return (Us * r) @ Us.T, (Us / r) @ Us.T

def _pairing_spectral_decomposition(g_vec: np.ndarray, V_JT_corr: float, vertex_cache: Optional[dict]) -> tuple:
    """Spectrum of the Gutzwiller-scaled pairing matrix Ṽ_ij = V_irrep_ij·√(g_i g_j) of ONE Γ6–Γ7 branch, in the FS-weighted orthonormal basis Δ̃ = S^{1/2}Δ
    (S = Gram matrix of the basis harmonics on the FS sample):  A = S^{1/2} Ṽ S^{1/2} = U diag(a) Uᵀ.
    E_cond = Δ†Ṽ⁻¹Δ is an indefinite quadratic form, so its attractive/non-attractive split exists only with respect to a metric. The FS-weighted norm is the physical one and makes the split
    invariant under any rescaling or re-mixing of the harmonics; a plain eigh(Ṽ) is not (it cut the physical es⊕d mixtures of a shared sector along an arbitrary Euclidean direction).
    Returns (a, U, T = S^{1/2}, T⁻¹, att = a > floor, floor = _V_REPULSIVE_FLOOR_FRAC·a_max). Without a cache: Ṽ = V_JT_corr·diag(g), S = 1."""
    n = len(g_vec)
    S = None if vertex_cache is None else vertex_cache.get('S_irrep')
    if vertex_cache is None:
        V = V_JT_corr * np.diag(g_vec)
    else:
        sq = np.sqrt(np.clip(g_vec, 0.0, None))
        V = vertex_cache['V_irrep'] * np.outer(sq, sq)
    T, Tinv = _gram_sqrt(np.eye(n) if S is None else np.asarray(S, dtype=float))
    A = T @ (0.5 * (np.asarray(V, dtype=float) + np.asarray(V, dtype=float).T)) @ T
    a, U = np.linalg.eigh(0.5 * (A + A.T))
    floor = max(_V_REPULSIVE_FLOOR_FRAC * float(a[-1]), _MATH_EPS)
    return a, U, T, Tinv, a > floor, floor

def _sector_condensation_energy(D: np.ndarray, g_t: float, g_Delta_s: float, g_Delta_d: float, V_JT_corr: float, vertex_cache: Optional[dict]) -> float:
    """
    Condensation-energy density Re[D† V⁻¹ D] in the attractive eigenspace of V, in FS-weighted orthonormal basis (a = eigenvalues of A, A = S^{1/2} Ṽ S^{1/2}). 

    Attractive modes (att = a > floor): 1/a — the physical V⁻¹. 
    Repulsive modes (a ≤ floor): 1/floor — constant, finite penalty.
    """
    g_vec = _gutzwiller_irrep_vector(g_Delta_s, g_Delta_d, g_t)
    a, U, T, _, att, floor = _pairing_spectral_decomposition(g_vec, V_JT_corr, vertex_cache)
    c2 = np.abs(U.T @ (T @ np.asarray(D, dtype=complex))) ** 2
    a_safe = np.where(att, np.maximum(a, floor), floor)
    stiff  = 1.0 / a_safe
    return float(np.sum(c2 * stiff))

def _irrep_decomposition(eigvec_euclid: np.ndarray, sqrt_w: np.ndarray, fs_weights: np.ndarray, Phi: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Decompose a FS-weighted dominant eigenvector (Euclidean-orthonormal
    `eigvec_euclid`, i.e. sum(eigvec_euclid**2)==1, from eigh of
    K_fs = sqrt(w) V sqrt(w)) onto the (generally non-orthogonal) basis `Phi`
    (columns = _IRREP_NAMES form factors evaluated at each FS point).

    using the proper (Loewdin) formula for the weight captured by a non-orthogonal subspace:
        resolved_weight = c^T S^+ c,   S_ij = <phi_i|phi_j>_w,  c_i = <phi_i|psi>_w,
    where psi = eigvec_euclid/sqrt(w) is the FS-measure-normalised eigenvector (sum_k w_k psi_k^2 == 1) and S^+ is the (pseudo-)inverse Gram matrix.
    This equals 1 exactly iff psi lies entirely in span(Phi); 1-resolved_weight is precisely the "unresolved weight" of the irrep-projection test.
    """
    psi_w = fs_weights * (eigvec_euclid / np.maximum(sqrt_w, 1e-300))   # = psi(k) * w(k)
    S = Phi.T @ (fs_weights[:, None] * Phi)
    c = Phi.T @ psi_w
    diag_norms = np.clip(np.diag(S), 1e-30, None)
    v_raw = c / np.sqrt(diag_norms)
    w_frac = np.abs(v_raw)
    frac = w_frac / max(float(w_frac.sum()), 1e-12)
    S_pinv = np.linalg.pinv(S, rcond=1e-10)
    resolved_weight = float(np.clip(c @ S_pinv @ c, 0.0, 1.0))
    return {
        'S':                 S,
        'c':                 c,
        'v_raw':             v_raw,
        'frac':              frac,
        'resolved_weight':   resolved_weight,
        'unresolved_weight': 1.0 - resolved_weight,
    }

def _sector_report_lines(vc: dict, kmax: int = 3) -> list:
    """Human-readable table of the G-sector spectrum + completeness scan."""
    fs_pts, w = vc['fs_pts'], vc['fs_weights']
    sqw = np.sqrt(w)
    cache = {}

    def phi(n):
        if n not in cache:
            cache[n] = np.asarray(_HARMONIC_LIBRARY[n][3](fs_pts[:, 0], fs_pts[:, 1]), dtype=float)
        return cache[n]

    def resolved(psi, names):
        if not names:
            return 0.0
        return float(_irrep_decomposition(psi, sqw, w, np.stack([phi(n) for n in names], axis=1))['resolved_weight'])

    def sector(si, key):
        psi = vc['sector_psi'][si]
        active = [n for n in _IRREP_NAMES if _IRREP_CHAR[n] == key]
        pool = [n for n in _HARMONIC_LIBRARY if _LIB_CHAR[n] == key and n not in active]
        r_act = resolved(psi, active)
        chosen, cur = [], r_act

        for _ in range(kmax):
            best = max(
                ((n, resolved(psi, active + chosen + [n]))
                 for n in pool if n not in chosen),
                key=lambda item: item[1],
                default=None,
            )
            if best is None or best[1] - cur < _SECTOR_SCAN_MIN_GAIN:
                break
            chosen.append(best[0])
            cur = best[1]
        d4h = {}                                        # D4h content of the sector's top eigenvector: Σ of the Loewdin |v_raw| fractions of the active basis functions, grouped by the D4h irrep of each function
        for _i in _SECTOR_MEMBERS[key]:
            _lab = _IRREP_D4H[_IRREP_NAMES[_i]]
            d4h[_lab] = d4h.get(_lab, 0.0) + float(vc['sector_frac'][si][_i])
        return {
            'key': key,
            'label': _sector_label(key),
            'lam': float(vc['sector_lambda'][si]),
            'lam2': float(vc['sector_lambda2'][si]),
            'covered': bool(active),
            'drives': (bool(vc['sector_drives'][si]) if vc.get('sector_drives') is not None else bool(active)),
            'resolved_active': r_act,
            'suggest': chosen,
            'resolved_best': cur,
            'd4h': d4h,
        }

    lines = ["  G = {E,T_Q}×{kx→−kx, ky→−ky} sector spectrum of the FS pairing kernel  (a sector is a character of G, NOT a D4h irrep; * = harmonic in the active BdG basis; resolved = Loewdin weight of the sector's leading eigenvector; D4h = its irrep content):"]
    for si, key in sorted(enumerate(_G_OPS), key=lambda item: -float(vc['sector_lambda'][item[0]])):
        r = sector(si, key)
        par = 'even-k' if key[1] * key[2] > 0 else 'odd-k '
        sug = (
            '   → add ' + ', '.join(r['suggest'])
            + f" (resolved → {r['resolved_best'] * 100:.0f}%)"
            if r['suggest'] else ''
        )
        cov = (
            f"resolved(active)={r['resolved_active'] * 100:5.1f}%"
            if r['covered'] else "NOT COVERED by the active basis"
        )
        irr = (
            '  D4h:' + ''.join(f" {_k} {100.0 * _v:.0f}%" for _k, _v in sorted(r['d4h'].items(), key=lambda kv: -kv[1]) if _v >= 0.005)
            if r['covered'] else ''
        )
        lines.append(
            f"   {r['label']:<58s} {par}  "
            f"λ_top={r['lam']:+8.4f}  λ2={r['lam2']:+8.4f}  "
            f"{cov}{irr}{sug}"
        )

    # σd (kx↔ky) resolution of the even-parity sectors: the D4h irreps that share a G-sector, one eigenvalue each
    sw_lam = vc.get('sector_swap_lambda')
    sw_def = vc.get('sector_swap_defect')
    sw_ok = sw_lam is not None and bool(vc.get('fs_closed_swap', False))
    if sw_ok:
        lines.append("  D4h irreps inside the even-parity sectors (σd = kx↔ky parity of the FS sample; exact irreps only if the kernel is C4-covariant, defect = max|K − P K Pᵀ|/max|K| = 0):")
        for si, key in sorted(((s, k) for s, k in enumerate(_G_OPS) if k[1] == k[2]), key=lambda it: -float(np.nanmax(sw_lam[it[0]]))):
            parts = []
            for ip, par_sw in enumerate((1, -1)):
                members = [_IRREP_NAMES[i] for i in _SECTOR_MEMBERS[key] if _IRREP_SWAP[_IRREP_NAMES[i]] == par_sw]
                tag = (','.join(members[:4]) + ('…' if len(members) > 4 else '')) if members else 'no basis function'
                parts.append(f"{_D4H_BY_MX_MY_SWAP[(key[1], key[2], par_sw)]} λ={sw_lam[si, ip]:+.4f} [{tag}]")
            lines.append(f"   (TQ{'+' if key[0] > 0 else '-'},mx{'+' if key[1] > 0 else '-'},my{'+' if key[2] > 0 else '-'})  " + "   |   ".join(parts) + f"   defect={sw_def[si]:.1e}")
        _defects = [float(d) for d in np.asarray(sw_def, dtype=float) if np.isfinite(d)]
        _defect_max = max(_defects) if _defects else 0.0
        _pair_gap = 0.0                                   # px/py partner sectors are C4 images of each other: their λ coincide in a C4-covariant kernel
        for _t in (1, -1):
            _a, _b = _G_OPS.index((_t, -1, 1)), _G_OPS.index((_t, 1, -1))
            _pair_gap = max(_pair_gap, abs(float(vc['sector_lambda'][_a]) - float(vc['sector_lambda'][_b])))
        _brk = float(vc.get('h_c4_breaker', 0.0))
        if max(_defect_max, _pair_gap) > max(1e-6, 3.0 * _brk):
            lines.append(
                f"  ⚠ the pairing kernel is not C4-covariant: σd defect up to {_defect_max:.1e}, px/py partner sectors differ by {_pair_gap:.1e} in λ, V_irrep couples es↔d — "
                + ("although H itself is exactly C4-symmetric (Δ_B1g_static = 0, Q = 0)" if bool(vc.get('h_c4_symmetric', False))
                   else f"while the explicit C4 breakers of H (Δ_B1g_static/Δ_CF + |Q|/λ_hop) are only {_brk:.1e}, i.e. not the cause")
                + ". Source in this model: the Q-vertex ∂H/∂Q = local g_JT·β_k·B1g_op (σd-odd) + hopping part ∂T_x/∂Q = −∂T_y/∂Q = (t0/λ_hop)(A_xz − A_yz) at Q = 0 (orbital-B1g × k-B1g = A1g, σd-even); "
                "their cross term in χ_QQ(q) is odd under q_x↔q_y. The σd split above is then variational, not an exact irrep decomposition.")
        _lead_cov_lam = max((float(vc['sector_lambda'][si]) for si, key in enumerate(_G_OPS) if _SECTOR_MEMBERS[key] and (vc.get('sector_drives') is None or bool(vc['sector_drives'][si]))), default=0.0)
        _unseeded = []
        for si, key in enumerate(_G_OPS):
            if key[1] != key[2] or not _SECTOR_MEMBERS[key] or (vc.get('sector_drives') is not None and not bool(vc['sector_drives'][si])):
                continue
            _ip_top = int(np.nanargmax(sw_lam[si]))
            _ip_oth = 1 - _ip_top
            _par_oth = (1, -1)[_ip_oth]
            _mem_oth = [_IRREP_NAMES[i] for i in _SECTOR_MEMBERS[key] if _IRREP_SWAP[_IRREP_NAMES[i]] == _par_oth]
            _lam_oth = float(sw_lam[si, _ip_oth])
            if _mem_oth and _lam_oth > 0.0 and _lam_oth >= _SECTOR_NEAR_FRAC * _lead_cov_lam:
                _unseeded.append(f"{_D4H_BY_MX_MY_SWAP[(key[1], key[2], _par_oth)]} {_mem_oth[:3]} λ={_lam_oth:+.4f} (sector top: {_D4H_BY_MX_MY_SWAP[(key[1], key[2], (1, -1)[_ip_top])]} λ={float(sw_lam[si, _ip_top]):+.4f})")
        if _unseeded:
            lines.append(
                f"  ⚠ irreps that share a sector with a stronger one and are NOT followed by the sector seed (λ ≥ {_SECTOR_NEAR_FRAC:.2f}·λ_lead): " + "; ".join(_unseeded)
                + " — to add one of them to the seed pass seed_complex_phase={'<channel>': 0.0}: a channel that the seed leaves (almost) empty is then populated at 50 % of the largest seed amplitude.")

    if 'fs_boundary_frac' in vc:
        lines.append(f"  FS sample: N={len(w)}  T_Q boundary (paired, weight ½): {100.0 * vc['fs_boundary_frac']:.1f}% of the weight  T_Q-consistent: {'✓' if vc.get('fs_tq_ok', True) else '⚠'}  "
                     f"closed under k→−k / kx→−kx / ky→−ky: {'✓' if vc.get('fs_closed_inv') else '✗'} / {'✓' if vc.get('fs_closed_mx') else '✗'} / {'✓' if vc.get('fs_closed_my') else '✗'}  "
                     f"odd-k sectors: {'validated' if vc.get('odd_channels_ok', True) else 'EXCLUDED'}")
    if 'S_irrep' in vc:
        _se = np.linalg.eigvalsh(0.5 * (vc['S_irrep'] + vc['S_irrep'].T))
        _rk = int(np.sum(_se > 1e-10 * max(float(_se[-1]), 1e-300)))
        lines.append(f"  basis on the sample: rank(S) = {_rk}/{len(_se)}" + ("  ⚠ some harmonics are linearly dependent on the FS sample (too few points: raise thermal_fs_nat_size)" if _rk < len(_se) else ""))
    lead_all = max(
        (sector(si, key) for si, key in enumerate(_G_OPS)),
        key=lambda r: r['lam'],
    )
    cov_sectors = [
        sector(si, key)
        for si, key in enumerate(_G_OPS)
        if any(_IRREP_CHAR[n] == key for n in _IRREP_NAMES) and (vc.get('sector_drives') is None or bool(vc['sector_drives'][si]))
    ]
    lead_cov = max(cov_sectors, key=lambda r: r['lam'], default=None)

    lines.append(f"  leading sector (all)    : {lead_all['label']}  λ={lead_all['lam']:+.4f}")

    if lead_cov is not None:
        lines.append(
            f"  leading sector (covered): {lead_cov['label']}  "
            f"λ={lead_cov['lam']:+.4f}   [drives lambda_lin_max / v_raw / frac]"
        )
        near = [
            r for r in cov_sectors
            if r['key'] != lead_cov['key']
            and r['lam'] > 0
            and r['lam'] >= _SECTOR_NEAR_FRAC * lead_cov['lam']
        ]
        if near:
            lines.append(
                "  near-degenerate covered sectors (seeded together with the leading one; the nonlinear BdG gap equation decides): "
                + "; ".join(f"{r['label']} λ={r['lam']:+.4f}" for r in near)
            )

        if lead_all['covered'] and not lead_all['drives'] and lead_all['lam'] > lead_cov['lam'] * (1.0 + _MATH_EPS):
            lines.append(
                f"  ⚠ the leading sector {lead_all['label']} (λ={lead_all['lam']:+.4f} > {lead_cov['lam']:+.4f}) is an ODD-k sector that is EXCLUDED from the selection / seeds / Tc ")
        if (lead_all['lam'] > lead_cov['lam'] * (1.0 + _MATH_EPS) and not lead_all['covered']):
            lines.append(
                f"  ⚠ BASIS GAP: the leading sector {lead_all['label']} contains NO active basis function "
                f"(λ={lead_all['lam']:+.4f} > covered {lead_cov['lam']:+.4f}); library harmonics of that sector: "
                f"{[n for n in _HARMONIC_LIBRARY if _LIB_CHAR[n] == lead_all['key']]}"
            )

        if lead_cov['resolved_active'] < 0.5 and lead_cov['suggest']:
            lines.append(f"  ⚠ only {lead_cov['resolved_active'] * 100:.0f}% of the covered leading eigenvector is spanned by the active basis; candidates: {lead_cov['suggest']}")
    return lines

def _basis(d):
    """Columns [up, dn] of a Kramers doublet as a 6×2 matrix."""
    return np.column_stack((d['up'], d['dn']))

def _get_G6_G7a_G7b(evals_soc, evecs_soc, Sz_full, LS_op, B1g_t2g):
    doublets = []

    for idx in (0, 2, 4):
        U = evecs_soc[:, idx:idx + 2]
        sz_vals, sz_vecs = np.linalg.eigh(U.conj().T @ Sz_full @ U)
        if sz_vals[1] < 0:
            sz_vals, sz_vecs = sz_vals[::-1], sz_vecs[:, ::-1]
        states = U @ sz_vecs          # columns [dn, up] in physical Sz basis
        # T-adapted gauge: eigh returns every eigenvector with an arbitrary phase. Define |dn⟩ := Θ|up⟩ exactly, so that the pair operator
        #   c†_{a↑} c†_{b↓} − c†_{a↓} c†_{b↑} is a true pseudospin singlet with the same time-reversal convention for all orbitals.
        _partner = _U_THETA_T2G @ states[:, 1].conj()

        if abs(np.vdot(states[:, 0], _partner)) < 1.0 - 1e-8:
            raise RuntimeError(
                "Θ|up⟩ is not the Sz-down state of the doublet: the time-reversal matrix does not match the t2g operators"
            )

        states = np.column_stack((_partner, states[:, 1]))  # [dn := Θ|up⟩, up]
        v0 = evecs_soc[:, idx]
        doublets.append({
            'idx':    idx,
            'energy': float(evals_soc[idx]),
            'ls_val': float(np.real(v0.conj() @ LS_op @ v0)),
            'up':     states[:, 1],
            'dn':     states[:, 0],
            'sz_up':  float(np.real(sz_vals[1])),
            'sz_dn':  float(np.real(sz_vals[0])),
        })

    # Sort Γ6 / Γ7 according to the spin-orbit expectation value.
    doublets.sort(key=lambda d: d['ls_val'])
    G6, G7_candidates = doublets[0], doublets[1:]

    # Γ7a / Γ7b are defined by the direct Γ6–Γ7 B1g coupling.
    # Γ7b is retained explicitly because its finite matrix elements contribute to the virtual high-energy sums.
    U6 = _basis(G6)
    for d in G7_candidates:
        d['b1g_to_G6'] = float(np.linalg.norm(U6.conj().T @ B1g_t2g @ _basis(d)))
    G7_candidates.sort(key=lambda d: -d['b1g_to_G6'])
    G7a, G7b = G7_candidates
    return G6, G7a, G7b

def _fmt_amplitude_line(vec: np.ndarray, names: tuple, top_n: int = 4, active_thr: float = 1e-6, zero_thr: float = 1e-9, signed: bool = False) -> str:
    v = np.asarray(vec)
    if v.size == 0:
        return "<empty>"
    a = np.abs(v)
    idx = np.argsort(a)[::-1][:top_n]
    if signed:
        top = ", ".join(f"{names[i]}={np.real(v[i]):+.4f}" for i in idx if a[i] > zero_thr)
    else:
        top = ", ".join(f"{names[i]}={a[i]:.4e}" for i in idx if a[i] > zero_thr)
    nact = int(np.sum(a > active_thr))
    parts = []
    parts.append(f"top{top_n}=[{top}]")
    parts.append(f"nact={nact}/{a.size}")
    return "  ".join(parts)

@dataclass
class ModelParams:
    t_pd:             float      # eV    pd hybridisation integral
    t_prime_ratio:    float      # —     2nd-neighbor (diagonal, (1,1)-type) hopping ratio
    t_dprime_ratio:   float      # —     3rd-neighbor (axial, (2,0)-type) hopping ratio
    U_dd:             float      # —     Coulomb repulsion in d orbit, typ charge-transfer 10·t0
    lambda_soc:       float      # eV    atomic SOC λ t2g shell; determines Γ₆–Γ₇ splitting
    Delta_tetra:      float      # eV    tetragonal axial CF Δ_tet·Lz²; negative = z-compression;  Partial cancellation with SOC tunes Γ₆–Γ₇ gap independently of λ
    g_JT:             float      # eV/Å  Jahn–Teller electron–phonon coupling (B1g channel: Q·(Lx²-Ly²))  increasing g_JT beyond the SC-triggered window is risky, because spontaneous JT (G3[2,2] < 0) always precedes the RPA cross-channel divergence.
    K_lattice:        float      # eV/Å² bare lattice spring constant, B1g channel (phonon stiffness, no exchange);  K_lattice must satisfy: K_spont = g²/Δ_CF < K < g²/(0.05·π·t0) for SC-JT window (λ_JT > 0.05).
    lambda_hop:       float      # Å     hopping decay length for B₁g anisotropy: t(Q) = t0·exp(±Q/λ_hop)
    g_Eg2:            float      # eV/Å  Jahn-Teller electron-phonon coupling, Eg,2 channel (Q_Eg2·(LyLz+LzLy));  Eg,2 is local (no hopping-renormalisation term, unlike B1g's t(Q) anisotropy).         
    K_lattice_Eg2:    float      # eV/Å² bare lattice spring constant, Eg,2 channel (independent stiffness from K_lattice)
    Delta_B1g_static: float      # eV    static B1g (x²-y²) crystal-field splitting, Δ_ip·(Lx²-Ly²); splits Γ₇→Γ₇a+Γ₇b.
    hybrid_scale:     float      #       above 4.0 assumes stronger covalency (the charge carriers would typically reside less than ~80% on the metal ion)
    Upp_ratio_bare:   float      #       the bare (without hybridization) U_pp / U_dd ratio varies between 0.2 and 2.0 for 3d-5d transition metals due to the spatial extent of the ligand orbital
    J_H_ratio:        float      #       Kanamori Hund's coupling for the virtual d^2 superexchange intermediate state
    Delta_CT:         float      # eV    charge-transfer gap (ZSA scale); sets scale for CT-insulator crossover
    M_eff:            float      # amu   it is a normal-mode reduced mass, weighted by how much each atom actually moves in that specific eigenmode), but for the B1g-type octahedral distortions this model targets, the metal ion is close to a symmetry-imposed node, so the mode is usually dominated by ligand motion
    kT:               float      # eV    emperature — keep kT < Tc to allow gap to open
    M_axis:           str        #       direction of the staggered (Néel) moment: 'x' | 'y' | 'z' (x, y ∥ the two Cu–O bond directions). The AFM Weiss field and the M observable use the doublet-projected spin S_axis.
    tol:              float
    trw_far_scale:    float = 1.0   # —  multiplies the TRW weight of every NON-NN bond class (NNN, 3NN): 1 = the (t_δ/t)² estimate, 0 = far-bond TRW switched off, >1 = stress test

    def __post_init__(self):
        if self.M_axis not in ('x', 'y', 'z'):
            raise ValueError(f"M_axis must be 'x', 'y' or 'z', got {self.M_axis!r}")

        # ── 1. SOC + CF Hamiltonian and t2g operators ──────────────────────────
        H_soc_cf, Lx_t2g, Ly_t2g, Lz_t2g, Sz_full, LS_op = _t2g_soc_cf_operators(self.lambda_soc, self.Delta_tetra, self.Delta_B1g_static)
        evals_soc, evecs_soc = np.linalg.eigh(H_soc_cf)

        B1g_t2g  = Lx_t2g @ Lx_t2g - Ly_t2g @ Ly_t2g
        Eg2_t2g  = Ly_t2g @ Lz_t2g + Lz_t2g @ Ly_t2g
        mu_z_t2g = Lz_t2g + 2.0 * Sz_full

        # ── 2. Orbital vectors and their projectors in the spin⊗orbital basis ──
        I2 = np.eye(2, dtype=complex)
        P_orb = {name: np.kron(np.outer(u, u.conj()), I2) for name, u in _ORBITAL_VECTORS.items()}

        # ── 3. Kramers doublets: identify, order, warn ─────────────────────────
        G6, G7a, G7b = _get_G6_G7a_G7b(evals_soc, evecs_soc, Sz_full, LS_op, B1g_t2g)

        # Diagnostic cross-check vs. the |μz| criterion; disagreement flags the mixed CF/SOC regime.
        mu7 = [np.linalg.norm(_basis(d).conj().T @ mu_z_t2g @ _basis(d)) for d in (G7a, G7b)]
        if mu7[0] < mu7[1]:
            _scf_log("RMFT-WARN",
                f"⚠ Γ7a (B1g-coupling-sorted) has smaller |μz| than Γ7b: the two criteria disagree "
                f"(Δ_tetra/λ_SOC={self.Delta_tetra / max(self.lambda_soc, 1e-12):.2f}). "
                f"|Sz|: {abs(G7a['sz_up']):.4f} vs {abs(G7b['sz_up']):.4f}  "
                f"μz: {mu7[0]:.4f} vs {mu7[1]:.4f}  "
                f"B1g-to-Γ6: {G7a['b1g_to_G6']:.4f} vs {G7b['b1g_to_G6']:.4f}"
            )
        
        # ── 4. Ordered 6×6 basis: rotation + rotated operators ────────────────────
        doublets = (G6, G7a, G7b)
        # z-polarised ordered doublet basis Γ6⊕Γ7a⊕Γ7b in the t2g⊗spin space (T-adapted gauge: |↓⟩ = Θ|↑⟩)
        U_ord = np.column_stack([d[s] for d in doublets for s in ('up', 'dn')])  
        evals_ord = evals_soc[[d['idx'] + k for d in doublets for k in (0, 1)]]

        self.sz_op = np.array([d[s] for d in doublets for s in ('sz_up', 'sz_dn')], dtype=float)

        # ── 5. Doublet-projected spin operators S_x, S_y, S_z in the ordered Γ6⊕Γ7a⊕Γ7b basis ───
        # The exchange couples per doublet (J_A1g_diag[c] uniform inside a Kramers pair), so only the INTRA-doublet 2×2 blocks of the spin operator enter the Weiss field / observables; the Γ6↔Γ7 inter-doublet
        # elements are kept separately (S_inter, diagnostics only). For S_z the block-diagonal projection IS diag(sz_op) exactly (the doublet basis is the S_z eigenbasis inside each doublet).
        _S_full = {'x': np.kron(np.eye(3, dtype=complex), 0.5 * _SIGMA_2x2['x']),
                   'y': np.kron(np.eye(3, dtype=complex), 0.5 * _SIGMA_2x2['y']),
                   'z': Sz_full}
        self.S_ops   = {}
        self.S_inter = {}
        for _ax, _Sf in _S_full.items():
            _S6  = U_ord.conj().T @ _Sf @ U_ord
            _blk = np.zeros((_N_ORB, _N_ORB), dtype=complex)
            for _idx in _CHANNEL_ORB_IDX:
                _blk[np.ix_(_idx, _idx)] = _S6[np.ix_(_idx, _idx)]
            self.S_ops[_ax]   = 0.5 * (_blk + _blk.conj().T)
            self.S_inter[_ax] = _S6 - _blk
        if not np.allclose(self.S_ops['z'], np.diag(self.sz_op), atol=1e-10):
            raise RuntimeError("block-diagonal projected S_z must equal diag(sz_op)")
        self.S_ops['z'] = np.diag(self.sz_op).astype(complex)   # use the EXACT diagonal (the numerical projection carries ~1e-16 round-off): M_axis='z' then reproduces the previous numerics bit for bit
        self.S_axis_op = self.S_ops[self.M_axis]            # (6,6) Hermitian, block-diagonal per doublet: the Néel-axis spin used by the Weiss field, the M observable and the vertices

        if any(d['sz_up'] * d['sz_dn'] >= 0 for d in doublets):
            _scf_log("RMFT-WARN",
                "⚠ Sz diagonalisation returned same-sign partners: "
                + "  ".join(f"{name}: {d['sz_up']:+.4f},{d['sz_dn']:+.4f}"
                            for name, d in zip(('Γ6', 'Γ7a', 'Γ7b'), doublets))
                + ". Possible near-degenerate subspace."
            )

        self.Delta_CF = float(evals_ord[2] - evals_ord[0])   # Γ7a − Γ6 gap (JT-active)
        self.g7split  = float(evals_ord[4] - evals_ord[2])   # Γ7b − Γ7a splitting

        # B1g = Lx² − Ly² (JT-active); off-diagonal part separated out for convenience.
        self.B1g_op      = np.asarray(np.real(U_ord.conj().T @ B1g_t2g @ U_ord), dtype=float)
        self.B1g_offdiag = self.B1g_op - np.diag(np.diag(self.B1g_op))

        # Eg,2 = Ly Lz + Lz Ly (symmetrised); enforce exact Hermiticity against roundoff.
        eg2 = U_ord.conj().T @ Eg2_t2g @ U_ord
        self.Eg2_op = 0.5 * (eg2 + eg2.conj().T)

        # Orbital-selective hopping building blocks in the ordered subspace.
        orb_in_basis = {name: np.asarray(np.real(U_ord.conj().T @ P @ U_ord), dtype=float) for name, P in P_orb.items()}
        self.Tx_A_xz = orb_in_basis['xz']
        self.Tx_A_yz = orb_in_basis['yz']
        self.Tx_A_xy = orb_in_basis['xy']

        # Completeness: P_xz + P_yz + P_xy = 1 on the orbital triplet ⇒ sum = I6.
        _sum_A = self.Tx_A_xz + self.Tx_A_yz + self.Tx_A_xy
        if not np.allclose(_sum_A, np.eye(6), atol=1e-8):
            _scf_log("RMFT-WARN", f"⚠ A_xz+A_yz+A_xy deviates from I6 (max|Δ|={np.max(np.abs(_sum_A - np.eye(6))):.2e})")

        # ── 6. Model parameters and the 2-hole Hamiltonian ─────────────────────
        self.t0       = self.t_pd ** 2 / self.Delta_CT
        self.t_prime  = self.t_prime_ratio  * self.t0
        self.t_dprime = self.t_dprime_ratio * self.t0
        # FS k-points used in the vertex q-loop; resolves the d-wave node and B1g anti-nodal hot spots.
        self.thermal_fs_nat_size = max(int(self.kT / self.t0 * _NK ** 2), 32)

        self.J2_over_J1 = self.t_prime_ratio  ** 2
        self.J3_over_J1 = self.t_dprime_ratio ** 2
        # ── Weight of the B1g transverse Weiss field (TRW) of every pairing channel relative to the NN-bond value J_B1g  The B1g⊗B1g exchange on a bond δ is ∝ (t_δ/t)² (the same bridge counting as J2/J1, J3/J1 above): NN 1, diagonal NNN (1,1) → (t'/t)², axial 3NN (2,0) → (t''/t)², bonds that carry no hopping in this model. The weight of a channel is the squared-amplitude average over the bond classes of its form factor.
        _w_cls = np.array([1.0, 1.0, self.J2_over_J1 * self.trw_far_scale, self.J3_over_J1 * self.trw_far_scale, 0.0])      # on-site | NN | NNN | 3NN | longer
        self.trw_irrep_scale = np.array([float(_TRW_BOND_FRAC[_n] @ _w_cls) for _n in _IRREP_NAMES])
        self.Z_afm_eff  = _Z * (1.0 - self.J2_over_J1 - self.J3_over_J1)  # A1g multiplier from J2/J3 frustration for the longitudinal AFM Weiss field

        # Ligand coordination + weak-hybridisation limit for U_pp.
        z_O  = _Z / 2.0
        self.U_pp = (self.Upp_ratio_bare * self.U_dd) / (
            1.0 + z_O * self.t_pd ** 2 / (self.Delta_CT * (self.Delta_CT + self.U_dd)))
        self.J_H  = self.J_H_ratio * self.U_dd

        H_2hole, basis = _multiorbital_2hole_hamiltonian(self.t_pd, self.Delta_CT, H_soc_cf, self.U_dd, self.U_pp, self.J_H)
        evals_2hole, evecs_2hole = np.linalg.eigh(H_2hole)

        # ── 7. κ from Löwdin downfolding (intra + cross) ───────────────────────
        self.kappa_doublets = _intra_doublet_kappa(basis, evals_2hole, evecs_2hole, [('G6', G6), ('G7a', G7a), ('G7b', G7b)])
        self.kappa_cross = _cross_doublet_kappas(basis, evals_2hole, evecs_2hole, G6, [(G7a, "7a"), (G7b, "7b")], B1g_t2g)

        # ── 8. Orbital weights per doublet (3×3) ───────────────────────────────
        #   _w_orb[c, orb] = (1/2) Tr(P_orb · ρ_doublet_c)  ∈ [0, 1]
        #   row index c ∈ {0: Γ6, 1: Γ7a, 2: Γ7b};  col index orb ∈ {0: xz, 1: yz, 2: xy}
        self._w_orb = np.stack([0.5 * np.diagonal(op).reshape(3, 2).sum(axis=1) for op in [self.Tx_A_xz, self.Tx_A_yz, self.Tx_A_xy]], axis=-1)

        # ── 9. k-grid and cyclic-shift table for k+q on the 2π/_NK grid ────────
        assert _NK % 2 == 0, f"_NK={_NK} must be even for commensurate q_AFM=(π,π)"

        k_scf = np.linspace(-np.pi, np.pi, _NK, endpoint=False)
        KX_scf, KY_scf = np.meshgrid(k_scf, k_scf)
        self.k_points  = np.column_stack((KX_scf.ravel(), KY_scf.ravel()))
        self.N_k       = len(self.k_points)
        self.k_weights = _uniform_bz_weights_2d(_NK, _NK)

        # E(k+q)[i] = E(k)[shift_table[nx, ny, i]] for q = (nx, ny) · 2π/_NK.
        flat   = np.arange(self.N_k)
        kx_idx = flat %  _NK
        ky_idx = flat // _NK
        nx     = np.arange(_NK)[:, None, None]   # (_NK, 1, 1)
        ny     = np.arange(_NK)[None, :, None]   # (1, _NK, 1)
        self.shift_table = (
            ((ky_idx[None, None, :] + ny) % _NK) * _NK
        + ((kx_idx[None, None, :] + nx) % _NK)
        ).astype(np.int32)
    
    def get_gutzwiller_factors(self, target_doping: float) -> Tuple[float, float]:
        """
        g_t       = 2δ/(1+δ)         kinetic energy; → 0 at half-filling (Mott insulator)
        g_J       = 4/(1+δ)²         exchange enhancement; → 4 at half-filling (J = 4t²/U)

        For the d-channel: the B₁g pairing channel arises from virtual Γ₆↔Γ₇ transitions via  the superexchange tensor J_B1g = (J_CT/2)·sinh(2Q/λ)·η·τ_x.
        The renormalisation interpolates between Γ₇ decoupled (Δ_CF→∞ → g_Delta_ad → g_t) and Γ₆–Γ₇ degenerate (Δ_CF→0 → g_Delta_ad → g_J) 
        """
        abs_delta = max(abs(target_doping), 1e-6)
        g_t       = (_GW_G_T_NUMERATOR * abs_delta) / (1.0 + abs_delta)
        g_J       = _GW_G_J_PREFACTOR / ((1.0 + abs_delta) ** 2)
        return g_t, g_J
    
    def exchange_channels(self, Q: float, g_J: float) -> Tuple[np.ndarray, float, float]:
        """
        Q-dependent multipolar superexchange in the full Γ6⊕Γ7a⊕Γ7b basis.

            J_A1g_diag[c]  = g_J · κ_c · ⟨orbital weight⟩_c(Q)          (longitudinal, even)
            J_B1g_scalar_c = g_J · [κ_c^{B1g} √(f6,orb f7c,orb)] · (j_xz(Q) − j_yz(Q))   (transverse, odd)

        Convention (ZGRS / Lee–Nagaosa–Wen)
        ------------------------------------
        κ_c is the PHYSICAL, absolute-normalised "bare" J extracted via Löwdin/ Feshbach projection of the 2-hole cluster ED (4·c_zz exactly reproduces
        E_triplet − E_singlet). It already carries the ZSA factor of 4 — do NOT multiply by 4 again.

        g_J(δ) = 4/(1+δ)² is the Gutzwiller/RVB renormalisation (ZGRS 1988), applied on top of this bare J. The spin-dilution from δ is carried by
        the self-consistent order parameters (M_c in the AFM Weiss field (Z_afm_eff/2)·Σ_c J_c·M_c²), NOT by an additional factor on J_c itself.

        Caveat: g_J is an RVB variational ansatz, not the exact ground state (2-site Gutzwiller projection washes out S·S).
        """
        _jxz = np.exp(+Q / self.lambda_hop) ** 2
        _jyz = np.exp(-Q / self.lambda_hop) ** 2
        _jxy = 0.5 * (_jxz + _jyz)

        # Physical exchange per doublet with orbital-weighted hopping factor per doublet (3-vector)
        J_A1g_diag = np.repeat(g_J * (self.kappa_doublets[:, None] * self._w_orb) @ np.array([_jxz, _jyz, _jxy]), 2)  # single-bond ZSA superexchange: even, longitudinal scale

        # A Γ6↔Γ7 virtual d-p-d path factorises into the geometric mean of the two doublets' orbital weights;
        # xy is A1g-like and cannot enter the B1g channel. J_B1g_scalar(Q=0) vanishes exactly only for equal xz/yz weights; with Δ_B1g_static ≠ 0 a residual static B1g exchange survives at Q = 0.
        J_B1g_2ch = g_J * self.kappa_cross * (np.array([_jxz, -_jyz]) @ np.sqrt(self._w_orb[0, :2, None] * self._w_orb[1:, :2].T))  # odd, transverse scale
        return J_A1g_diag,  J_B1g_2ch[0], J_B1g_2ch[1]

    def _exchange_J_q(self, J_A1g_diag: np.ndarray, qx: float, qy: float) -> np.ndarray:
        """
        Channel extraction AND momentum evaluation of the A1g exchange coupling

        J_A1g_diag :  Per-orbital (Kramers-doubled) A1g exchange, or already-extracted per-channel exchange.

        Both forms are evaluated by the same NN + 2nd-NN + 3rd-NN momentum sum, tied to the single-bond NN
        value J1 = J_A1g_diag[c] via the same t²-superexchange scaling that already governs J1's own Q-dependence:

            J_c(q) = J_A1g_diag[c] · [ −2·(cos qx + cos qy)
                                    − 4·(J2/J1)·cos qx·cos qy
                                    − 2·(J3/J1)·(cos 2qx + cos 2qy) ]

        with J2/J1 = (t'/t)², J3/J1 = (t''/t)². Special points:

            q = (π,π):  J_c(q) = Z_afm_eff · J_A1g_diag[c]
                        (AFM peak; Z_afm_eff = Z·(1 − t'²/t² − t''²/t²))
            q = (0,0):  J_c(q) = −(4 + 4·J2/J1 + 4·J3/J1) · J_A1g_diag[c] < 0
                        (ferro/uniform channel is exchange-SUPPRESSED)
        """
        J = np.asarray(J_A1g_diag, dtype=float)[0::2]
        cqx  = np.cos(qx)
        cqy  = np.cos(qy)
        c2qx = np.cos(2.0 * qx)
        c2qy = np.cos(2.0 * qy)
        shell_nn  = -2.0 * (cqx + cqy)
        shell_2nn = -4.0 * cqx * cqy
        shell_3nn = -2.0 * (c2qx + c2qy)
        scale = shell_nn + self.J2_over_J1 * shell_2nn + self.J3_over_J1 * shell_3nn
        return J * scale   # [J_Γ6(q), J_Γ7a(q), J_Γ7b(q)]

    def effective_hopping_anisotropic(self, Q: float) -> Tuple[float, float]:
        """
        B₁g JT distortion breaks x-y symmetry: tx ≠ ty
        
        Exponential hopping law (Harrison + bond-length argument):
        tx(Q) = t₀ * exp(+Q / lambda_hop)   [elongation along x → shorter bond → larger t]
        ty(Q) = t₀ * exp(-Q / lambda_hop)   [compression along y → longer bond → smaller t]
        """
        tx = self.t0 * np.exp(+Q / self.lambda_hop)
        ty = self.t0 * np.exp(-Q / self.lambda_hop)
        return tx, ty

    def hopping_matrices(self, Q: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Orbital-selective 6×6 inter-sublattice hopping matrices

            T_x(Q) = t_x(Q)·A_xz + t_y(Q)·A_yz + ½(t_x(Q)+t_y(Q))·A_xy
            T_y(Q) = t_y(Q)·A_xz + t_x(Q)·A_yz + ½(t_x(Q)+t_y(Q))·A_xy

        The x↔y swap in T_y (not T_y = T_x) is what makes the B1g distortion C4-covariant: under the 90° rotation combined with Q → −Q,

            T_x(−Q) = T_y(Q),     T_y(−Q) = T_x(Q),

        so the full inter-sublattice block

            H_AB(k, Q) = −2·g_t·[cos(k_x)·T_x(Q) + cos(k_y)·T_y(Q)]

        satisfies H_AB(k, Q) = H_AB(R·k, −Q) with R: (k_x, k_y) → (−k_y, k_x), the B1g-covariant identity. (With T_y = T_x this identity fails for Q ≠ 0.)

        Returns (Tx_op, Ty_op), each a real symmetric (6,6) ndarray. Γ7b hops through the SAME orbital projection as Γ6/Γ7a, so it disperses as a genuine band rather than sitting flat.
        """
        tx_b, ty_b = self.effective_hopping_anisotropic(Q)
        t_avg = 0.5 * (tx_b + ty_b)

        Tx_op = tx_b * self.Tx_A_xz + ty_b * self.Tx_A_yz + t_avg * self.Tx_A_xy
        Ty_op = ty_b * self.Tx_A_xz + tx_b * self.Tx_A_yz + t_avg * self.Tx_A_xy
        return Tx_op, Ty_op


    def hopping_matrices_dQ(self, Q: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact analytic ∂T_x/∂Q and ∂T_y/∂Q, the companions of hopping_matrices().

        With  t_x(Q) = t0·exp(+Q/λ_hop),  t_y(Q) = t0·exp(−Q/λ_hop),

            dt_x/dQ = +t_x/λ_hop,     dt_y/dQ = −t_y/λ_hop.

        The dT_y/dQ = the x↔y swap of dT_x/dQ is inherited from hopping_matrices(), so the analytic derivative is exactly consistent with the finite-difference
        Q-derivative of hopping_matrices() at every Q (including Q = 0, where the σd-odd ∂T_x/∂Q = −∂T_y/∂Q = (t0/λ_hop)(A_xz − A_yz) and the hopping part of ∂H/∂Q is σd-even).
        """
        tx_b, ty_b = self.effective_hopping_anisotropic(Q)
        dtx = tx_b / self.lambda_hop
        dty = -ty_b / self.lambda_hop
        dt_avg = 0.5 * (dtx + dty)

        dTx_op = dtx * self.Tx_A_xz + dty * self.Tx_A_yz + dt_avg * self.Tx_A_xy
        dTy_op = dty * self.Tx_A_xz + dtx * self.Tx_A_yz + dt_avg * self.Tx_A_xy
        return dTx_op, dTy_op

    def wave_function_weight(self, tx_b: float, ty_b: float, kx: float, ky: float) -> np.ndarray:
        """Downfolding factor from the multiband (metal dd + ligand pp) system to the Γ₆–Γ₇ subspace, measuring quasiparticle localization on the central ion versus the ligands; scaling from 2nd order perturbation theory"""
        t_avg = 0.5 * (tx_b + ty_b)
        dt    = 0.5 * (tx_b - ty_b)
        k_sigmoid = 10.0
        x0 = 0.5
        
        def beta_sq_at(kx_v, ky_v):
            Ag_part  = np.cos(kx_v) + np.cos(ky_v)
            B1g_part = np.cos(kx_v) - np.cos(ky_v)
            x = 1.0 - self.hybrid_scale * (t_avg * Ag_part + dt * B1g_part) / self.Delta_CT
            return 1.0 / (1.0 + np.exp(-k_sigmoid * (x - x0)))
        
        # T_Q-symmetrization
        return 0.5 * (beta_sq_at(kx, ky) + beta_sq_at(kx + np.pi, ky + np.pi))


@dataclass
class _SolveState:
    """
    Mutable SCF-run-local state passed explicitly into compute_gap_eq_vectorized.
    """
    V_irrep_ema: Optional[np.ndarray] = None   # sign-flip EMA state for diag(V_irrep) (generalises the old scalar V_d_ema to all _N_IRREP channels)
    _ema_kick_pending: bool  = False           # True for one iter after a kick: doubles blend weight so EMA adapts faster

def lambda_channel_decomposition(solver, cache: dict, verbose: bool = True) -> dict:
    """Exact channel decomposition of lambda_FS (top eigenvalue of the leading covered sector) by Rayleigh quotient:  lambda_FS = C_JT + C_spin + C_cross  identically,
    C_ch = psi^T (sqrt_w V^ch sqrt_w) psi with psi the unit top eigenvector of that sector's FS kernel."""
    si_lead = list(cache['sector_keys']).index(cache['leading_sector'])
    lam_FS = float(cache['sector_lambda'][si_lead])
    if cache.get('sector_C') is not None:
        C_spin, C_JT, C_cross = (float(c) for c in cache['sector_C'][si_lead])
    else:
        K_sec = cache.get('K_sec')              # (n_ops, 3, N_fs, N_fs): spin, JT, cross
        if K_sec is None:
            raise ValueError("neither sector_C nor K_sec in the cache: rebuild it with compute_pairing_kernel_and_build_cache")
        psi = np.asarray(cache['sector_psi'][si_lead], dtype=float)
        psi = psi / max(float(np.linalg.norm(psi)), 1e-30)
        x = np.sqrt(np.asarray(cache['fs_weights'], dtype=float)) * psi
        y = np.asarray(cache['c_B'], dtype=float) * x
        C_spin, C_JT, C_cross = float(y @ K_sec[si_lead, 0] @ y), float(x @ K_sec[si_lead, 1] @ x), float(y @ K_sec[si_lead, 2] @ y)
    total = C_JT + C_spin + C_cross
    if verbose:
        den = lam_FS if abs(lam_FS) > 1e-30 else 1e-30
        _scf_log("PAIR-DIAG", f"  Channel decomposition of lambda_FS = {lam_FS:+.6f}:")
        _scf_log("PAIR-DIAG", f"    C_JT    = {C_JT:+.6f}   ({100 * C_JT / den:+.1f}%)")
        _scf_log("PAIR-DIAG", f"    C_spin  = {C_spin:+.6f}   ({100 * C_spin / den:+.1f}%)")
        _scf_log("PAIR-DIAG", f"    C_cross = {C_cross:+.6f}   ({100 * C_cross / den:+.1f}%)")
        _scf_log("PAIR-DIAG", f"    Sum     = {total:+.6f}   {'✓' if abs(total - lam_FS) < 1e-8 else '❌'}   (sum - lambda_FS = {total - lam_FS:+.1e})")
    return {'C_JT': C_JT, 'C_spin': C_spin, 'C_cross': C_cross, 'lambda_FS': lam_FS}


class RMFT_Solver:

    def __init__(self, params: ModelParams):
        self.p = params

        self.kT          = max(params.kT, _MATH_EPS)
        self.sz_op       = params.sz_op
        self.k_points    = params.k_points
        self.N_k         = params.N_k
        self.k_weights   = params.k_weights
        self.shift_table = params.shift_table           # (nk, nk, N_k) int32 — cyclic shift index
        self.g_JT_bare   = params.g_JT                  # keeping it strictly bare avoids double-counting local-correlation

        # Orbital operators derived from the SOC+CF eigenbasis.
        self._rebuild_orbital_operators()
        self.phi_k = (np.cos(self.k_points[:, 0])
                      - np.cos(self.k_points[:, 1]))
        # Full extended-basis form factors (every channel of _IRREP_NAMES) on the same grid;
        self.Phi_irrep_k = _pairing_basis_matrix(self.k_points[:, 0], self.k_points[:, 1])
        
        self._reset_transient_state()
        self._K_bare: float = params.K_lattice  # immutable bare lattice spring constant (eV/Å²)
        self.thermal_fs_nat_size = params.thermal_fs_nat_size

    def _rebuild_orbital_operators(self) -> None:
        """
        Rebuild all SOC+CF-basis-dependent operators.
        Call after mutating lambda_soc, Delta_tetra, or Delta_B1g_static (after __post_init__).
        """
        z6 = np.zeros((_N_ORB, _N_ORB), dtype=complex)
        # B₁g phonon operator in the full Γ₆⊕Γ₇a⊕Γ₇b subspace (no downfolding)
        self.B1g_op = self.p.B1g_op
        self.B1g_offdiag = self.p.B1g_offdiag

        # 24×24 Nambu extension of B1g_op for per-site ⟨B1g⟩ evaluation.
        # Layout: [Part_A, Part_B, Hole_A, Hole_B]; hole block carries −B1g_op^T.
        _B1g_c = self.B1g_op.astype(complex)
        _B1g_h = (-self.B1g_op.T).astype(complex)
        self.B1g_24 = np.block([
            [_B1g_c, z6,      z6,      z6     ],
            [z6,     _B1g_c,  z6,      z6     ],
            [z6,     z6,      _B1g_h,  z6     ],
            [z6,     z6,      z6,      _B1g_h ],
        ])

        # Eg,2 phonon operator in the full Γ₆⊕Γ₇a⊕Γ₇b subspace (M=T∘C2x-even, symmetry-unprotected channel)
        self.Eg2_op = self.p.Eg2_op

        # 24×24 Nambu extension of Eg2_op, mirroring B1g_24 exactly.
        _Eg2_c = self.Eg2_op.astype(complex)
        _Eg2_h = (-self.Eg2_op.T).astype(complex)
        self.Eg2_24 = np.block([
            [_Eg2_c, z6,      z6,      z6     ],
            [z6,     _Eg2_c,  z6,      z6     ],
            [z6,     z6,      _Eg2_h,  z6     ],
            [z6,     z6,      z6,      _Eg2_h ],
        ])

        # Nambu vertex matrices for the unified χ_SQ susceptibility: lift physical operators into the full 24×24 Nambu space as  M_O = block_diag(O_6, O_6, -O_6^T, -O_6^T)
        _sz6 = np.diag(self.sz_op).astype(complex)          # Sz vertex: O_6 = diag(sz_op)
        _sz6_h = (-_sz6.T).astype(complex)                  # hole block = −diag(sz_op)^T = −diag(sz_op)
        self.Sz_nambu = np.block([
            [_sz6,   z6,      z6,       z6    ],   # particle A
            [z6,     _sz6,    z6,       z6    ],   # particle B
            [z6,     z6,      _sz6_h,   z6    ],   # hole A
            [z6,     z6,      z6,       _sz6_h],   # hole B
        ])  # (24,24) complex

        _nambu_sign = np.diag([1, -1, -1, 1])
        self.Sz_stag_nambu_channels = [
            np.kron(_nambu_sign, np.diag(self.sz_op * np.isin(np.arange(_N_ORB), idx)))
            for idx in _CHANNEL_ORB_IDX
        ]

        # Uniform (sublattice-even) per-doublet spin vertices for the channel-resolved RPA. Diagonal in the 24-dim Nambu basis (index = 6·block + orbital;
        # blocks = [particle A, particle B, hole A, hole B]) → stored as (3, 24) diagonals; Σ_c O_c = Sz_nambu exactly, so the channel bubbles add up to the total-S_z χ_SS of the scalar routine.
        _nambu_uniform_sign = np.array([1.0, 1.0, -1.0, -1.0])
        self._sz_nambu_diag_channels = np.stack([
            np.kron(_nambu_uniform_sign, self.sz_op * np.isin(np.arange(_N_ORB), idx)) for idx in _CHANNEL_ORB_IDX
        ]).astype(complex)                                           # (3, 24)
        self.Sz_nambu_channels = [np.diag(_d) for _d in self._sz_nambu_diag_channels]
        if not np.allclose(sum(self.Sz_nambu_channels), self.Sz_nambu):
            raise RuntimeError("Sz_nambu_channels must sum to Sz_nambu")
        self._chi_MA_buf = None                                      # reusable (4, N_k, 24, 24) band-basis vertex buffer

        # ── Néel axis.  The doublet-projected spin along the ordering axis (block-diagonal per doublet; = diag(sz_op) for 'z')
        self.Sx_op, self.Sy_op, self.Sz_op = (np.asarray(self.p.S_ops[_a], dtype=complex) for _a in ('x', 'y', 'z'))
        self.S_axis_op   = np.asarray(self.p.S_axis_op, dtype=complex)
        self._axis_is_z  = (self.p.M_axis == 'z')
        if not self._axis_is_z:
            # x / y: the spin vertices are not diagonal → dense 24×24 Nambu matrices instead of the diagonal-vector paths.
            #   staggered (A:+, B:−) :  blockdiag( S_c, −S_c, −S_c^T, +S_c^T )      uniform :  blockdiag( S_c, S_c, −S_c^T, −S_c^T )     (hole block of a Hermitian O is −O^T)
            def _nambu_blocks(a, b, c, d):
                out = np.zeros((_N_BDG, _N_BDG), dtype=complex)
                out[0:_N_ORB, 0:_N_ORB], out[_N_ORB:2*_N_ORB, _N_ORB:2*_N_ORB], out[2*_N_ORB:3*_N_ORB, 2*_N_ORB:3*_N_ORB], out[3*_N_ORB:_N_BDG, 3*_N_ORB:_N_BDG] = a, b, c, d
                return out
            _S_ch = []
            for _idx in _CHANNEL_ORB_IDX:
                _mask = np.zeros((_N_ORB, _N_ORB)); _mask[np.ix_(_idx, _idx)] = 1.0
                _S_ch.append(self.S_axis_op * _mask)
            self.Sz_stag_nambu_channels = [_nambu_blocks(S, -S, -S.T, S.T) for S in _S_ch]
            self.Sz_nambu_channels      = [_nambu_blocks(S, S, -S.T, -S.T) for S in _S_ch]
            self._sz_nambu_diag_channels = None                      # sentinel: dense path in _chi0_channels_raw
            self.Sz_nambu               = sum(self.Sz_nambu_channels)
            self._S_stag_nambu_total    = sum(self.Sz_stag_nambu_channels)

    def _get_vbdg(self) -> 'VectorizedBdG':
        if self._vbdg is None:
            self._vbdg = VectorizedBdG(self)
        return self._vbdg

    def _get_chi0_norm_cache(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, vbdg: 'VectorizedBdG') -> Tuple[np.ndarray, np.ndarray]:
        """
        Cached on (M, Q, n_kspace, mu, g_t, g_J)
        Within a single SCF iteration these quantities are constant across the entire q-loop, avoiding O(N_q) redundant eigh calls on the N_k × 24 matrix.
        Return (E_k_all, V_k_all) for the Δ=0 BdG on k_points.
        """
        if self._chi0_norm_cache is not None:
            _E, _V, _M_old, _Q_old, _n_kspace_old, _mu_old, _gt_old, _gJ_old = self._chi0_norm_cache
            if (np.linalg.norm(M - _M_old) < _M_THR_REL * max(np.linalg.norm(M), 1.0) and
                abs(Q - _Q_old) < _Q_THR_REL * self.p.lambda_hop and
                abs(mu - _mu_old) < 1e-4 * max(abs(mu), 1.0) and
                abs(g_t - _gt_old) < 1e-4 and
                abs(g_J - _gJ_old) < 1e-4 and
                abs(n_kspace - _n_kspace_old) < 1e-6):
                return _E, _V
        
        E_k, V_k = np.linalg.eigh(
            vbdg._build_H_stack(self.k_points, M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, out=vbdg._H_stack)
            )
        self._chi0_norm_cache = (E_k, V_k, M, Q, n_kspace, mu, g_t, g_J)
        return E_k, V_k

    def _compute_orbital_densities(self, ev, ec):
        """Calculates orbital (Γ6↑, Γ6↓, Γ7a↑, Γ7a↓, Γ7b↑, Γ7b↓) densities from the BdG spectrum."""
        f = _fermi_function(ev, self.kT)
        fbar = 1.0 - f
        uA, uB, vA, vB = _get_nambu_spinors(ec)
        occ_A = np.abs(uA)**2 * f[:, None, :] + np.abs(vA)**2 * fbar[:, None, :]
        occ_B = np.abs(uB)**2 * f[:, None, :] + np.abs(vB)**2 * fbar[:, None, :]
        occ_total = occ_A + occ_B
        dens = np.einsum('k,kin->i', self.k_weights, occ_total, optimize=True) / 4.0
        return dens

    def estimate_gutzwiller_factors_occupation_based(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float) -> Tuple[float, float, float]:
        """
        Physical per-orbital occupations <c^dag_alpha c_alpha> for alpha = (Γ₆↑, Γ₆↓, Γ₇ₐ↑, Γ₇ₐ↓, Γ₇ᵦ↑, Γ₇ᵦ↓),
        evaluated on sublattice A of the normal-state (Δ=0) BdG Hamiltonian at (M, Q, mu, g_t, g_J).

        - g_t and g_J are fixed by the macroscopic hole doping x, following the exact large-U single-band Gutzwiller limit.
        - The Luttinger/hole-counting constraint x = 1 − Σ_α n_α now sums over ALL SIX states (Γ6, Γ7a, AND Γ7b)
        - The orbital character weight is updated dynamically from the occupations and used to interpolate the d-wave pairing vertex (g_Δd).
        """
        ev, ec = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self._get_vbdg())
        dens = self._compute_orbital_densities(ev, ec)
        n6  = dens[0] + dens[1]
        n7a = dens[2] + dens[3]
        n7b = dens[4] + dens[5]

        # dynamic p_7 weight from orbital coherence
        n_total = max(n6 + n7a + n7b, 1e-6)
        # pairing vertices (robust interpolation)
        g_Delta_s = g_t
        g_Delta_ad = g_t + (g_J - g_t) * np.clip(2.0 * n7a / n_total, 0.0, 1.0)
        g_Delta_bd = g_t + (g_J - g_t) * np.clip(2.0 * n7b / n_total, 0.0, 1.0)
        return g_Delta_s, g_Delta_ad, g_Delta_bd

    def _reset_transient_state(self) -> None:
        """
        Reset all mutable per-solve caches on a solver clone.

        Must be called after copy.copy(solver) to guarantee that:
          - _vbdg          : gets a fresh VectorizedBdG with this clone's k-grids
          - _K_bare        : NOT cleared (immutable per __init__ contract)
        """
        self._vbdg            = None   # re-created on first _get_vbdg()
        self._chi0_norm_cache = None   # normal-state χ₀ eigenvector cache
        self._fs_cache_dict   = None   # unified FS sample cache (fs_pts, vF) keyed by (params, n_fs)
        self._moriya_landau_cache = {} # Landau-derived Gamma_M is kT-dependent (via the BdG spectrum); keyed only by doping, both of which cloning can change; must not be inherited by reference from a clone.
        self._dFc_ref_cache = {} 

    def _shallow_clone(self) -> 'RMFT_Solver':
        """Shallow-copy self and self.p (params), so the clone's ModelParams can be mutated independently of the original solver, without touching anything else."""
        s = copy.copy(self)
        s.p = copy.copy(self.p)
        return s

    def _clone_solver_at_T(self, T: float) -> 'RMFT_Solver':
        """
        Performs a shallow copy of self and self.p, then resets all mutable per-solve caches.
        The immutable bare stiffness _K_bare is carried over unchanged.

        Return a fully independent solver clone with kT = T.
        """
        s = self._shallow_clone()
        s.kT = T
        s._K_bare = self._K_bare
        s.thermal_fs_nat_size = max(int(T / self.p.t0 * _NK**2), 32)
        s._reset_transient_state()
        return s

    def _full_rebuild(self) -> None:
        """Always call this single method after mutating any ModelParams field on a solver clone."""
        self.p.__post_init__()
        self._K_bare = self.p.K_lattice
        self._rebuild_orbital_operators()
        self._reset_transient_state()

    def build_local_hamiltonian_for_bdg(self, sign_M: float, M_channels: np.ndarray, J_A1g_per_channel: np.ndarray, mu: float, Z: float) -> np.ndarray:
        """
        Local 6×6 BdG Hamiltonian for one sublattice, basis [6↑, 6↓, 7ₐ↑, 7ₐ↓, 7ᵦ↑, 7ᵦ↓] — full 3 Kramers-doublet manifold, no downfolding. sign_M = ±1 for sublattices A/B (staggered AFM).

        Terms:
          1. Chemical potential −μ (all six states — Γ7b is a genuine dynamical band, not a frozen core level,
             so its occupation responds to μ exactly like Γ6/Γ7a; this is what lets Γ6/Γ7a bands rise to, or past, Γ7b's level without any special-casing).
          2. Crystal field splitting: Δ_CF on Γ₇ₐ, Δ_CF+g7split on Γ₇ᵦ.
          3. Longitudinal (diagonal) AFM Weiss field from J_A1g: this is purely diagonal (spin-preserving). It shifts the orbital energies of all three doublets,
             weighted by each one's own <Sz> (sz_op) — so the large-|μz| Γ7b state always participates in the AFM physics, regardless of how weak its JT (B1g) matrix elements happen to be.
             J_A1g_diag is DIFFERENT for each channel (Γ6/Γ7a/Γ7b), a single common M cannot be simultaneously (i) the parameter of the actual Weiss space AND (ii) the exact self-consistent
             fixed point of the unweighted Sz_stag-observable on all three channels. However, with a separate M_c per channel, this is EXACTLY true based on the Hellmann-Feynman identity,
             because J_A1g_diag is uniform within each channel (on the two members of the Kramers pair) by construction.
        """
        H = np.zeros((_N_ORB, _N_ORB), dtype=complex)

        np.fill_diagonal(H, -mu)
        H[2, 2] += self.p.Delta_CF
        H[3, 3] += self.p.Delta_CF
        H[4, 4] += self.p.Delta_CF + self.p.g7split
        H[5, 5] += self.p.Delta_CF + self.p.g7split

        M6 = _expand_M_channels(M_channels)            # (6,) # all chanels carrying its own Weiss space for the 2 orbitals of the corresponding Kramers doublet.
        if self._axis_is_z:
            h_J_unit = Z * J_A1g_per_channel * self.sz_op         # (6,)
            H -= np.diag(sign_M * h_J_unit * M6)
        else:
            # Weiss field along x / y.  The coupling J_c·M_c is uniform inside each Kramers pair, so  diag(J·M) commutes with the block-diagonal S_axis and the term stays Hermitian.
            _s_op = {'x': self.Sx_op, 'y': self.Sy_op, 'z': self.Sz_op}[self.p.M_axis]
            H = H - sign_M * Z * (np.diag(np.asarray(J_A1g_per_channel, dtype=float) * M6) @ _s_op)
        return H

    def _calc_dHdQ(self, M: np.ndarray, Q: float, F67_vec: np.ndarray, mu: float, g_t: float, g_J: float) -> np.ndarray:
        """
        Bare explicit Q-vertex
            dH/dQ = (∂H/∂Q)_{M, μ, F67 fixed}
        for the 24×24 BdG Hamiltonian, with block-symmetric TRW weights.

        Explicit Q-dependent channels differentiated here:
            1. Local JT:                  ∂/∂Q [β_k(Q) · g_JT · Q · B1g_op]
            2. Inter-sublattice hopping:  ∂H_AB/∂Q  (analytic via hopping_matrices_dQ)
            3. AFM Weiss:                 ∂/∂Q [Z_afm · J_A1g(Q)] · M · sz_op
            4. On-site TRW (s):           ∂/∂Q [β_k(Q) · Z · J_B1g^{(s)}(Q)] · F67_s · B1g_offdiag
            5. NN-bond TRW (es,d,px,py):  ∂/∂Q [J_B1g^{(i)}(Q)] · φ_i(k) · F67_i · B1g_offdiag
            6. NNN / 3NN TRW (g, ...):    ∂/∂Q [J_B1g^{(i)}(Q)] · φ_i(k) · F67_i · B1g_offdiag

        Hole-block parity handling:
            Hole blocks are −h(−k)^*.  The EVEN-in-k parts (JT, AFM, s, es, d, g-type even NNN) keep their sign under k → −k, the ODD parts (px, py and any odd-parity NNN such as px2, py2, px3, py3) flip sign.
            The TRW contributions are therefore SPLIT into even/odd and added to the hole blocks separately, instead of being routed through the simple −conj() of the particle block.
        """
        F67a_s, F67b_s = F67_vec[_DELTA_IDX['s7a']], F67_vec[_DELTA_IDX['s7b']]
        
        kx, ky = self.k_points[:, 0], self.k_points[:, 1]

        # ---- β_k(Q) and its 5-point Q-derivative ----
        def _beta_at(Qv: float) -> np.ndarray:
            tx_v, ty_v = self.p.effective_hopping_anisotropic(Qv)
            return np.asarray(self.p.wave_function_weight(tx_v, ty_v, kx, ky), dtype=float)

        def _exchange_at(Qv: float):
            return self.p.exchange_channels(Qv, g_J)
        
        # Internal finite-difference step
        eps_beta, _ = _q_fd_step(Q)
        beta_k = _beta_at(Q)
        beta_p1, beta_m1 = _beta_at(Q + eps_beta), _beta_at(Q - eps_beta)
        beta_p2, beta_m2 = _beta_at(Q + 2.0 * eps_beta), _beta_at(Q - 2.0 * eps_beta)
        dbeta_dQ = (beta_m2 - 8.0 * beta_m1 + 8.0 * beta_p1 - beta_p2) / (12.0 * eps_beta)

        J_A1g,  J_7a,  J_7b  = _exchange_at(Q)
        J_A1g_p1, J_7a_p1, J_7b_p1 = _exchange_at(Q + eps_beta)
        J_A1g_m1, J_7a_m1, J_7b_m1 = _exchange_at(Q - eps_beta)
        J_A1g_p2, J_7a_p2, J_7b_p2 = _exchange_at(Q + 2.0 * eps_beta)
        J_A1g_m2, J_7a_m2, J_7b_m2 = _exchange_at(Q - 2.0 * eps_beta)

        dJ_7a_dQ = (J_7a_m2 - 8.0 * J_7a_m1 + 8.0 * J_7a_p1 - J_7a_p2) / (12.0 * eps_beta)
        dJ_7b_dQ = (J_7b_m2 - 8.0 * J_7b_m1 + 8.0 * J_7b_p1 - J_7b_p2) / (12.0 * eps_beta)

        # --- 1. Local JT: d/dQ [β_k(Q) · g_JT · Q · B1g_op] ---
        gJT_B1g = (self.g_JT_bare * self.B1g_op).astype(complex)
        dH_JT_k = (beta_k[:, None, None] * gJT_B1g[None, :, :]
                + dbeta_dQ[:, None, None] * (Q * gJT_B1g)[None, :, :])

        # --- 2. Hopping: analytic ∂T/∂Q ---
        dTx_op, dTy_op = self.p.hopping_matrices_dQ(Q)
        dH_AB_hop_dQ = _build_H_AB_block(kx, ky, dTx_op, dTy_op, g_t)

        # --- 3. AFM Weiss: d/dQ [Z_afm · J_A1g(Q)] · M · sz_op (via 5-point FD on J_A1g) ---
        def _Hloc(sign: float, JA1g) -> np.ndarray:
            return self.build_local_hamiltonian_for_bdg(sign, M, JA1g, mu, self.p.Z_afm_eff)

        def _dHloc_dQ(sign: float) -> np.ndarray:
            return (_Hloc(sign, J_A1g_m2)
                    - 8.0 * _Hloc(sign, J_A1g_m1)
                    + 8.0 * _Hloc(sign, J_A1g_p1)
                    - _Hloc(sign, J_A1g_p2)) / (12.0 * eps_beta)

        dH_A_dQ = _dHloc_dQ(+1.0)
        dH_B_dQ = _dHloc_dQ(-1.0)

        # --- 4. TRW: block-symmetric (0↔1) and (0↔2) weight matrix on 6×6 ---
        # The Γ6↔Γ7a entry pairs with J_B1g_7a, the Γ6↔Γ7b entry with J_B1g_7b. np.kron(w_blocks, ones(2,2))
        # replicates each scalar weight across the full 2×2 Kramers block, so the final B1g_offdiag product inherits the 2×2 inner structure from B1g_offdiag *and* the block weight from w_blocks
        def _trw_weight(w_7a: float, w_7b: float) -> np.ndarray:
            w_blocks = np.zeros((3, 3), dtype=complex)
            w_blocks[0, 1], w_blocks[1, 0] = w_7a, np.conj(w_7a)      # Hermitian partner blocks
            w_blocks[0, 2], w_blocks[2, 0] = w_7b, np.conj(w_7b)
            return np.kron(w_blocks, np.ones((2, 2), dtype=complex))

        B1g_off = self.B1g_offdiag  # real, symmetric (6, 6)

        # On-site s-channel: ∂/∂Q [β_k(Q) · J_B1g(Q)] · F67 · B1g_offdiag
        trw_static_s = B1g_off * _trw_weight(F67a_s * J_7a, F67b_s * J_7b)
        trw_dQ_s     = B1g_off * _trw_weight(F67a_s * dJ_7a_dQ, F67b_s * dJ_7b_dQ)
        dH_TRW_k = _Z * (
            dbeta_dQ[:, None, None] * trw_static_s[None, :, :]
            + beta_k[:, None, None] * trw_dQ_s[None, :, :]
        )

        # --- 5. Inter-sublattice (NN) TRW: ∂/∂Q [J_B1g^{(i)}(Q)] · φ_i(k) · F67_i · B1g_offdiag ---
        # Split even (es, d) / odd (px, py): even part survives k → −k, odd part flips sign. The hole block below must treat them separately
        dH_TRW_AB_even = np.zeros((self.N_k, _N_ORB, _N_ORB), dtype=complex)
        dH_TRW_AB_odd  = np.zeros((self.N_k, _N_ORB, _N_ORB), dtype=complex)
        for i_irr, _name in enumerate(_IRREP_NAMES):
            if not _is_nn_bond(_name):          # on-site: local term above; g / px2 / py2: same-sublattice blocks below
                continue
            _s_trw = float(self.p.trw_irrep_scale[i_irr])            # same bond-class weight as in _build_H_stack
            F67a_bond = _s_trw * F67_vec[_IDX_7A_ARR[i_irr]]
            F67b_bond = _s_trw * F67_vec[_IDX_7B_ARR[i_irr]]
            trw_dQ_bond = B1g_off * _trw_weight(F67a_bond * dJ_7a_dQ, F67b_bond * dJ_7b_dQ)
            _c = self.Phi_irrep_k[:, i_irr][:, None, None] * trw_dQ_bond[None, :, :]
            if _IRREP_PARITY[_name] > 0:
                dH_TRW_AB_even += _c
            else:
                dH_TRW_AB_odd  += _c

        # --- 6. Same-sublattice (NNN / 3NN) TRW ---
        # Split even / odd: some NNN channels are k-odd (px2, py2, px3, py3 have parity −1), and their hole-block sign must follow the k → −k rule.
        dH_TRW_nnn_even = np.zeros((self.N_k, _N_ORB, _N_ORB), dtype=complex)
        dH_TRW_nnn_odd  = np.zeros((self.N_k, _N_ORB, _N_ORB), dtype=complex)
        for i_irr, _name in enumerate(_IRREP_NAMES):
            if not _is_nnn_bond(_name):
                continue
            _s_trw = float(self.p.trw_irrep_scale[i_irr])
            F67a_nnn = _s_trw * F67_vec[_IDX_7A_ARR[i_irr]]
            F67b_nnn = _s_trw * F67_vec[_IDX_7B_ARR[i_irr]]
            if F67a_nnn == 0 and F67b_nnn == 0:
                continue
            trw_dQ_nnn = B1g_off * _trw_weight(F67a_nnn * dJ_7a_dQ, F67b_nnn * dJ_7b_dQ)
            if _IRREP_PARITY[_name] > 0:
                dH_TRW_nnn_even += self.Phi_irrep_k[:, i_irr][:, None, None] * trw_dQ_nnn[None, :, :]
            else:
                dH_TRW_nnn_odd  += self.Phi_irrep_k[:, i_irr][:, None, None] * trw_dQ_nnn[None, :, :]

        # --- Assemble 24×24 BdG derivative ---
        dH_AB_even = dH_AB_hop_dQ + dH_TRW_AB_even
        dH_AB_odd  = dH_TRW_AB_odd
        dH_AB_dQ   = dH_AB_even + dH_AB_odd
        dH_AB_dQ_T = dH_AB_dQ.transpose(0, 2, 1)

        dHdQ = np.zeros((self.N_k, _N_BDG, _N_BDG), dtype=complex)

        # Particle inter-sublattice hopping
        dHdQ[:, 0:_N_ORB, _N_ORB:2*_N_ORB]  += dH_AB_dQ
        dHdQ[:, _N_ORB:2*_N_ORB, 0:_N_ORB]  += np.conj(dH_AB_dQ_T)

        # Particle-diagonal blocks: split even/odd (only dH_TRW_nnn_odd is k-odd)
        dH_diag_A_even = dH_JT_k + dH_A_dQ[None, :, :] - dH_TRW_k - dH_TRW_nnn_even
        dH_diag_A_odd  = -dH_TRW_nnn_odd
        dH_diag_B_even = dH_JT_k + dH_B_dQ[None, :, :] + dH_TRW_k + dH_TRW_nnn_even
        dH_diag_B_odd  = +dH_TRW_nnn_odd

        dHdQ[:, 0:_N_ORB, 0:_N_ORB] = dH_diag_A_even + dH_diag_A_odd
        dHdQ[:, _N_ORB:2*_N_ORB, _N_ORB:2*_N_ORB] = dH_diag_B_even + dH_diag_B_odd

        # Hole-diagonal blocks: −conj(even) + conj(odd)
        dHdQ[:, 2*_N_ORB:3*_N_ORB, 2*_N_ORB:3*_N_ORB] = -np.conj(dH_diag_A_even) + np.conj(dH_diag_A_odd)
        dHdQ[:, 3*_N_ORB:_N_BDG, 3*_N_ORB:_N_BDG] = -np.conj(dH_diag_B_even) + np.conj(dH_diag_B_odd)

        # Hole inter-sublattice blocks: even part via −conj, odd part via +conj
        dHdQ[:, 2*_N_ORB:3*_N_ORB, 3*_N_ORB:_N_BDG] += -np.conj(dH_AB_even) + np.conj(dH_AB_odd)
        dHdQ[:, 3*_N_ORB:_N_BDG, 2*_N_ORB:3*_N_ORB] += -dH_AB_even.transpose(0, 2, 1) + dH_AB_odd.transpose(0, 2, 1)

        # Hermitian symmetrization (numerical cleanup)
        dHdQ = 0.5 * (dHdQ + dHdQ.conj().transpose(0, 2, 1))
        return dHdQ

    def _eval_B1g_and_kweights_at_Q(self, target_doping: float, M: np.ndarray, Q_val: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, mu_guess: float, g_t: float, g_J: float) -> Tuple[float, np.ndarray]:
        """Scalar <B1g> and its k-resolved (beta^2(k)-weighted, /4) array at fixed (M, Q, Δ, F67); mu is re-solved at fixed density."""
        vbdg = self._get_vbdg()
        mu_v, n_v = self._find_mu_for_density(M, Q_val, Delta_vec, F67_vec, target_doping, mu_guess, g_t, g_J)
        txb, tyb = self.p.effective_hopping_anisotropic(Q_val)
        ev, ec = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M, Q_val, Delta_vec, F67_vec, n_v, mu_v, g_t, g_J, out=vbdg._H_stack))
        f_n = _fermi_function(ev, self.kT)
        diag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), self.B1g_24, ec).real
        beta = self.p.wave_function_weight(txb, tyb, self.k_points[:, 0], self.k_points[:, 1])
        B1g_k = np.einsum('kn,kn->k', diag_qp, f_n) * beta / 4.0
        return float(np.dot(self.k_weights, B1g_k)), B1g_k

    def Eg2_expectation(self, E_k_cache: tuple) -> float:
        """Per-site ⟨Eg2_op⟩ in the BdG ground state."""
        ev, ec = E_k_cache
        f_n = _fermi_function(ev, self.kT)
        diag_qp = np.einsum('kan, ab, kbn -> kn', ec.conj(), self.Eg2_24, ec).real
        exp_k = np.einsum('kn,kn->k', diag_qp, f_n)
        return float(np.dot(self.k_weights, exp_k)) / 4.0

    def _chi_QQ_matrix_elements(self, M: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, Q_Eg2: float = 0.0, return_matrix: bool = False):
        """
        Bare JT orbital susceptibility: χ_QQ = −∂²Ω/∂Q² at the state (M, Q, Δ_vec, F67_vec, μ).

        State-dependent.  Pass (Δ=0, F67=0, Q=0) with the normal-state self-consistent (M, μ) for the normal-state quantity used
        in the spontaneous-JT criterion; pass the SC-state values for χ_QQ(SC).  The two differ and must not be interchanged.

        return_matrix=True: full 2×2 matrix {χ_QQ[B1g,B1g], χ_QQ[B1g,Eg2]; χ_QQ[Eg2,B1g], χ_QQ[Eg2,Eg2]} via mixed finite differences of the same grand potential Ω(Q, Q_Eg2).
        """
        eps2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
        dQ   = np.sqrt(eps2)
        vbdg = self._get_vbdg()

        def omega(Qval, Qeg2val):
            ev = np.linalg.eigvalsh(
                vbdg._build_H_stack(self.k_points, M, Qval, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, vbdg._H_stack, Qeg2val)
                )
            arg = np.clip(np.abs(ev) / self.kT, 0.0, _FERMI_ARG_CLIP)
            Omega_kn = np.minimum(0.0, ev) - self.kT * np.log1p(np.exp(-arg))
            return np.sum(self.k_weights[:, None] * Omega_kn)
        
        Ωp = omega(Q + dQ, Q_Eg2)
        Ω0 = omega(Q, Q_Eg2)
        Ωm = omega(Q - dQ, Q_Eg2)
        
        # −∂²Ω/∂Q²: positive for a stable metal (χ_QQ > 0 convention used in G3[2,2]); division by 4.0 due to 2 (sublattice) * 2 (particle-hole) Nambu doubling
        chi_QQ = -(Ωp - 2.0 * Ω0 + Ωm) / (4.0 * eps2)
        chi_QQ_n = chi_QQ / self.g_JT_bare**2

        if not return_matrix:
            return chi_QQ_n
        
        eps2_e = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q_Eg2**2
        dQe = np.sqrt(eps2_e)
        Ωep = omega(Q, Q_Eg2 + dQe)
        Ωem = omega(Q, Q_Eg2 - dQe)
        chi_QQ_eg2 = -(Ωep - 2.0 * Ω0 + Ωem) / (4.0 * eps2_e)

        # Mixed partial via 4-point cross stencil: d²Ω/dQdQ_Eg2 ≈ [Ω(+,+)-Ω(+,-)-Ω(-,+)+Ω(-,-)] / (4 dQ dQe)
        Ω_pp = omega(Q + dQ, Q_Eg2 + dQe)
        Ω_pm = omega(Q + dQ, Q_Eg2 - dQe)
        Ω_mp = omega(Q - dQ, Q_Eg2 + dQe)
        Ω_mm = omega(Q - dQ, Q_Eg2 - dQe)
        d2Omega_cross = (Ω_pp - Ω_pm - Ω_mp + Ω_mm) / (4.0 * dQ * dQe)
        chi_QQ_cross = -d2Omega_cross / 4.0
        
        g_Eg2_2  = max(self.p.g_Eg2**2, _MATH_EPS)
        g_cross  = max(self.g_JT_bare * self.p.g_Eg2, _MATH_EPS)
        chi = np.array([[chi_QQ / self.g_JT_bare**2,      chi_QQ_cross / g_cross],
                        [chi_QQ_cross / g_cross, chi_QQ_eg2 / g_Eg2_2]], dtype=float)
        return chi

    def _diamagnetic_QQ_term(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, F67_vec: np.ndarray, E_k_all: np.ndarray, V_k_all: np.ndarray) -> float:
        """
        Full diamagnetic/contact contribution to χ_QQ. The static QQ susceptibility is decomposed as

            χ_QQ = χ_QQ^para - <∂²H/∂Q²>,

        where χ_QQ^para is the Nambu/Lehmann bubble generated by the first-derivative vertex ∂H/∂Q.
        The contact term must contain the second derivative of ALL FOUR channels:
            ∂²H/∂Q² = ∂²H_JT/∂Q² + ∂²H_hop/∂Q² + ∂²H_AFM/∂Q² + ∂²H_TRW/∂Q².

        In particular, although the bare factor Q in the JT coupling is linear, the actual Hamiltonian contains β_k(Q) Q, so
            ∂²/∂Q² [β_k(Q) Q] = 2 β'_k(Q) + Q β''_k(Q),
        which is generally nonzero.
        Likewise the TRW term contains
            ∂²/∂Q² [β_k J_B1g] = β''_k J_B1g + 2 β'_k J'_B1g + β_k J''_B1g.

        The derivative is evaluated at fixed M, μ, g_t, g_J and F67_s.
        Thus this is the explicit-Hamiltonian second derivative (∂²H/∂Q²)_{M,μ,g_t,g_J,F67_s},
        not the total second derivative of the fully self-consistent free energy along an SCF solution branch.
        """
        # The outer stencil stays accurate down to a roundoff floor roughly 2-3 orders of magnitude below this step
        h_QQ = np.sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2)

        dHdQ_m2 = self._calc_dHdQ(M, Q - 2.0 * h_QQ, F67_vec, mu, g_t, g_J)
        dHdQ_m1 = self._calc_dHdQ(M, Q - h_QQ, F67_vec, mu, g_t, g_J)
        dHdQ_p1 = self._calc_dHdQ(M, Q + h_QQ, F67_vec, mu, g_t, g_J)
        dHdQ_p2 = self._calc_dHdQ(M, Q + 2.0 * h_QQ, F67_vec, mu, g_t, g_J)
        
        # 5-point derivative of the COMPLETE first-derivative vertex.
        d2HdQ2 = (dHdQ_m2 - 8.0 * dHdQ_m1 + 8.0 * dHdQ_p1 - dHdQ_p2) / (12.0 * h_QQ)

        # Remove tiny numerical anti-Hermitian noise introduced by the nested finite differences. The exact ∂²H/∂Q² is Hermitian.
        d2HdQ2 = 0.5 * (d2HdQ2 + np.conj(np.swapaxes(d2HdQ2, 1, 2)))

        # Thermal expectation value

        f_k = _fermi_function(E_k_all, self.kT)
        H_QQ_diag = np.einsum('kin,kij,kjn->kn', np.conj(V_k_all), d2HdQ2, V_k_all, optimize=True)

        # The diagonal expectation values must be real for a Hermitian operator.
        H_QQ_expect = np.sum(self.k_weights[:, None] * f_k * np.real(H_QQ_diag))
        return H_QQ_expect / 4.0  # 2 (sublattice) × 2 (particle-hole) Nambu doubling

    def _chi0_channels_raw(self, M: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, q: np.ndarray, E_k_cache: tuple, apply_diamagnetic_QQ: bool = False, dHdQ_precomputed: np.ndarray = None, mechanical: bool = True) -> np.ndarray:
        """
        Raw (no Moriya damping, no PSD projection) 4×4 Lehmann/Kubo susceptibility matrix in the channel basis [S_Γ6, S_Γ7a, S_Γ7b, Q]:  χ_ab(q) = ⟨⟨O_a ; O_b⟩⟩,
        O_c = per-doublet S_z (uniform Nambu vertices, Σ_c O_c = total S_z),  O_Q = ∂H/∂Q, normalised to a common 1/eV scale (spin–Q entries / g_JT, QQ / g_JT²).

        All vertices are Hermitian, so ⟨k+q,m|O_b|k,n⟩ = conj(⟨k,n|O_b|k+q,m⟩) and the whole matrix is ONE Gram product of the four band-basis vertex tensors: the same number of
        band-basis transforms as the scalar routine's two vertex pairs (S_z, ∂H/∂Q).  Σ_cc' χ_cc' reproduces the scalar total-S_z χ_SS, Σ_c χ_cQ its χ_SQ.
        """
        E_k_all, V_k_all = E_k_cache
        _Delta_amp = float(np.sum(np.abs(Delta_vec)))
        eta = max(_ETA_DELTA_FRAC * _Delta_amp, _ETA_GRID_FLOOR * self.p.t0)

        dk = 2.0 * np.pi / _NK
        nx = int(round(q[0] / dk)) % _NK
        ny = int(round(q[1] / dk)) % _NK
        shift_idx = self.shift_table[nx, ny]
        V_kQ = V_k_all[shift_idx]
        E_kQ = E_k_all[shift_idx]                     # (N_k, 24)
        f_k  = _fermi_function(E_k_all, self.kT)      # (N_k, 24)
        f_kQ = _fermi_function(E_kQ, self.kT)         # (N_k, 24)

        kernel = _lehmann_kernel(f_k, f_kQ, E_k_all, E_kQ, eta, self.kT)
        kernel = kernel * self.k_weights[:, None, None]  # (N_k, 24, 24), real, k-weighted

        if mechanical:
            dHdQ = dHdQ_precomputed if dHdQ_precomputed is not None else self._calc_dHdQ(M, Q, F67_vec, mu, g_t, g_J)
        else:
            dH_Q_k = (self.g_JT_bare * self.B1g_24).astype(complex)
            dHdQ = np.broadcast_to(dH_Q_k[None, :, :], (len(E_k_all), _N_BDG, _N_BDG))

        n_k = len(E_k_all)
        _buf = self._chi_MA_buf
        if _buf is None or _buf.shape[1] != n_k:
            _buf = self._chi_MA_buf = np.empty((_N_CHANNELS + 1, n_k, _N_BDG, _N_BDG), dtype=complex)
        for c in range(_N_CHANNELS):
            if self._sz_nambu_diag_channels is not None:
                _buf[c] = np.einsum('kan,a,kam->knm', V_k_all.conj(), self._sz_nambu_diag_channels[c], V_kQ, optimize=True)
            else:   # Néel axis x / y: non-diagonal (dense) spin vertex
                _buf[c] = np.einsum('kan,ab,kbm->knm', V_k_all.conj(), self.Sz_nambu_channels[c], V_kQ, optimize=True)
        _buf[_N_CHANNELS] = np.einsum('kan,kab,kbm->knm', V_k_all.conj(), dHdQ, V_kQ, optimize=True)

        X = _buf.reshape(_N_CHANNELS + 1, -1)
        G = ((X * kernel.reshape(1, -1)) @ X.conj().T) / 4.0             # 4 = Nambu particle–hole doubling
        chi = 0.5 * (G.real + G.real.T)                                  # static limit: Re part; χ_ab = χ_ba for Hermitian vertices

        if apply_diamagnetic_QQ:
            chi[_N_CHANNELS, _N_CHANNELS] -= self._diamagnetic_QQ_term(M, Q, n_kspace, mu, g_t, g_J, F67_vec, E_k_all, V_k_all)   # eV/Å²
        chi[:_N_CHANNELS, _N_CHANNELS] /= self.g_JT_bare                                   # (1/Å) / (eV/Å) = 1/eV
        chi[_N_CHANNELS, :_N_CHANNELS] /= self.g_JT_bare
        chi[_N_CHANNELS, _N_CHANNELS]  /= self.g_JT_bare ** 2                              # (eV/Å²) / (eV²/Å²) = 1/eV
        return chi

    def _rpa_closure(self, M: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, q: np.ndarray, Gamma_M: float, J_A1g_diag: np.ndarray, V_JT_corr: float, E_k_cache: tuple, apply_diamagnetic_QQ: bool = False, dHdQ_precomputed: np.ndarray = None) -> dict:
        """
        Single entry point for every RPA quantity the SCF driver needs at one q, for the ACTIVE closure
            'chi_SS', 'chi_SQ', 'chi_QS', 'chi_QQ' : scalar (total-S_z) susceptibilities — legacy-compatible (channel mode: Σ over the 3 spin channels of the 4×4 block)
            'det'    : floor-regularised RPA determinant  (2×2 scalar closure, or 4×4 det(1 − χ₀U))
            'stoner' : Stoner factor of the spin sector   (J_eff_scalar·χ_SS, or largest eigenvalue of diag(J_c(q))·χ_cc')
            'chi0', 'J_q' : the 4×4 matrix and the per-channel J_c(q) (channel mode only; None otherwise)
        """
        q = np.asarray(q, dtype=float)
        # Off-diagonal spin–spin entries χ_cc' (c ≠ c') vanish for orbitally-isotropic hopping at Q = 0, Δ = 0, but not for Q ≠ 0 (JT-induced Γ6–Γ7 mixing) or in the SC state.
        chi0 = self._chi0_channels_raw(M, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, q, E_k_cache, apply_diamagnetic_QQ=apply_diamagnetic_QQ, dHdQ_precomputed=dHdQ_precomputed, mechanical=True)
        w_s, U_s = np.linalg.eigh(chi0[:_N_CHANNELS, :_N_CHANNELS])
        w_d = w_s / np.maximum(1.0 + Gamma_M * w_s, _MATH_EPS)
        _S = (U_s * w_d) @ U_s.T
        chi0[:_N_CHANNELS, :_N_CHANNELS] = 0.5 * (_S + _S.T)
        ev, evc = np.linalg.eigh(chi0)
        if ev[0] < 0.0:
            _P = (evc * np.maximum(ev, 0.0)) @ evc.T
            chi0 = 0.5 * (_P + _P.T)
        
        J_q = self.p._exchange_J_q(J_A1g_diag, q[0], q[1])

        # 4×4 RPA denominator M(q) = 1 − χ₀·U, U = diag(J_Γ6(q), J_Γ7a(q), J_Γ7b(q), V_JT_corr), and its floor-regularised determinant; χ_RPA = M⁻¹χ₀. With only the Γ6 channel populated it reduces to the 2×2 scalar M(q)
        U = np.diag([float(J_q[0]), float(J_q[1]), float(J_q[2]), float(V_JT_corr)])
        M4 = np.eye(_N_CHANNELS + 1) - chi0 @ U
        det = float(np.linalg.det(M4))
        det_floor = max(_MATH_EPS, 1e-4 * float(np.linalg.norm(M4)))      # Frobenius norm sets the natural scale
        if abs(det) < det_floor:
            det_safe = math.copysign(det_floor, det) if det != 0.0 else det_floor
        else:
            det_safe = det

        _cSQ = float(chi0[:_N_CHANNELS, _N_CHANNELS].sum())
        S = np.diag(np.asarray(J_q, dtype=float)) @ chi0[:_N_CHANNELS, :_N_CHANNELS]
        return {'chi_SS': float(chi0[:_N_CHANNELS, :_N_CHANNELS].sum()), 'chi_SQ': _cSQ, 'chi_QS': _cSQ, 'chi_QQ': float(chi0[_N_CHANNELS, _N_CHANNELS]),
                'det': det_safe, 'M4': M4, 'stoner': float(np.max(np.linalg.eigvals(S).real)), 'chi0': chi0, 'J_q': J_q}

    def _fs_channel_weights(self, fs_idx: np.ndarray, E_k_cache: tuple) -> np.ndarray:
        """Doublet content w_c(k) (Σ_c w_c = 1) of the normal-state Fermi-surface quasiparticle at every FS sample point; same band selection as _get_fs_points (lowest positive-energy BdG band)."""
        ev, ec = E_k_cache
        ev_f = ev[fs_idx]
        band = np.argmin(np.where(ev_f > 0.0, ev_f, np.inf), axis=1)
        psi = np.take_along_axis(ec[fs_idx], band[:, None, None], axis=2)[:, :, 0]      # (N_fs, 24)
        prob = np.abs(psi) ** 2
        w = np.stack([prob[:, [_N_ORB * b + o for b in range(4) for o in idx]].sum(axis=1) for idx in _CHANNEL_ORB_IDX], axis=1)
        return w / np.maximum(w.sum(axis=1, keepdims=True), _MATH_EPS)

    def _moriya_gamma_landau(self, target_doping: float, n_kspace: float, mu: float, g_t: float, g_J: float, J_eff: float, V_JT_corr: float) -> float:
        """
        Self-consistent Moriya damping from the model's own Landau coefficients.
        Solves  Γ_M = b·<δM²>  with  <δM²> = kT / (a + b·<δM²>), yielding
            Γ_M = (-a + sqrt(a² + 4·b·kT)) / 2.
        For a > 0 and a² ≫ 4b·kT this recovers the classical b·kT/a limit, while staying finite as a → 0 instead of diverging.
        """
        cache = self.__dict__.setdefault('_moriya_landau_cache', {})
        key = (round(float(target_doping), 6), round(float(g_t), 8), round(float(g_J), 8))
        if key in cache:
            return cache[key]
        
        Q0 = 0.0
        t_eff0 = g_t * self.p.t0

        def F_of_M(Mval: float, n_kspace: float, mu: float, g_t: float, g_J: float) -> float:
            Mc = np.zeros(_N_CHANNELS, dtype=float)
            Mc[0] = Mval    # probe Γ6 order only   
            mu, n_kspace = self._find_mu_for_density(Mc, Q0, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), target_doping, mu, g_t, g_J)
            g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(Mc, Q0, n_kspace, mu, g_t, g_J)
            Omega = self._compute_bdg_free_energy(Mc, Q0, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd, V_JT_corr)
            return Omega + mu * n_kspace  # Legendre correction

        # Five-point stencil on -2h, -h, 0, h, 2h
        h = _MORIYA_LANDAU_M_STEP
        Fm2 = F_of_M(-2.0 * h, n_kspace, mu, g_t, g_J)
        Fm1 = F_of_M(-h, n_kspace, mu, g_t, g_J)
        F0  = F_of_M(0.0, n_kspace, mu, g_t, g_J)
        Fp1 = F_of_M(h, n_kspace, mu, g_t, g_J)
        Fp2 = F_of_M(2.0 * h, n_kspace, mu, g_t, g_J)

        a_landau = (-Fm2 + 16.0 * Fm1 - 30.0 * F0 + 16.0 * Fp1 - Fp2) / (12.0 * h**2)
        b_landau = (Fm2 - 4.0 * Fm1 + 6.0 * F0 - 4.0 * Fp1 + Fp2) / (6.0 * h**4)

        a_val = float(a_landau)
        b_safe = max(float(b_landau), 0.0)   # quartic must be non-negative
        kT = float(self.kT)
        if b_safe < _MATH_EPS:
            # Nearly Gaussian limit: use classical b·kT/a
            a_safe = max(a_val, _MATH_EPS)
            Gamma_M_landau = b_safe * kT / a_safe
        else:
            disc = a_val**2 + 4.0 * b_safe * kT
            Gamma_M_landau = 0.5 * (-a_val + math.sqrt(disc))
        
        # UV safety cap: Gamma_M cannot exceed ~(correlated bandwidth)^2/J_eff. Built from Gutzwiller-renormalised RMS hopping at Q=0
        gamma_max = 4.0 * t_eff0**2 / (np.pi * J_eff)
        Gamma_M_final = min(max(Gamma_M_landau, 0.0), gamma_max)

        cache[key] = float(Gamma_M_final)
        return cache[key]

    def preflight_report(self, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, tx_bare: float, ty_bare: float, g_t: float, g_J: float, omega0_bare: float, F_cluster: dict, E_k_cache: tuple, vertex_cache: dict):
        params = self.p

        # ── Derived quantities ─────────────────────────────────────────────
        g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M, Q, n_kspace, mu, g_t, g_J)
        t_eff  = g_t * np.sqrt(0.5 * (tx_bare**2 + ty_bare**2))
        J_A1g0 = params.exchange_channels(0.0, g_J)[0]
        J_eff  = params.Z_afm_eff * J_A1g0[0]
        kTc    = _TC_TARGET_K / _EV_TO_K

        dCF_over_teff   = params.Delta_CF / max(t_eff, _MATH_EPS)
        J_over_teff     = J_eff / max(t_eff, _MATH_EPS)
        U_over_t0       = params.U_dd / max(params.t0, _MATH_EPS)
        omega_over_teff = omega0_bare / max(t_eff, _MATH_EPS)
        omega_over_dCF  = omega0_bare / max(params.Delta_CF, _MATH_EPS)
        dB1g_over_dCF   = abs(params.Delta_B1g_static) / max(params.Delta_CF, _MATH_EPS)
        kTc_over_teff   = kTc / max(t_eff, _MATH_EPS)
        g7split_over_kT = abs(params.g7split) / max(params.kT, _MATH_EPS)

        def _flag(ok, good, bad):
            return ("✓ " + good) if ok else ("⚠ " + bad)

        # ── 1. Energy scales (one line) ───────────────────────────────────
        _scf_log("PREFLIGHT",
            f"scales: t_eff={t_eff*1e3:.1f}  ω₀={omega0_bare*1e3:.1f}  Δ_CF={params.Delta_CF*1e3:.1f}  Γ7split={params.g7split*1e3:.2f} meV  "
            f"T_c^target={_TC_TARGET_K:.0f} K (kTc/t_eff={kTc_over_teff:.2f})")

        # ── 2. Regime flags (one per line) ────────────────────────────────
        _scf_log("PREFLIGHT", "  " + _flag(
            omega_over_dCF < 1.0,
            f"ω₀/Δ_CF={omega_over_dCF:.2f} < 1 — phonon below doublet gap (no Γ6↔Γ7 scattering by the phonon)",
            f"ω₀/Δ_CF={omega_over_dCF:.2f} ≥ 1 — phonon can scatter between doublets"))
        _scf_log("PREFLIGHT", "  " + _flag(
            5.0 <= U_over_t0 <= 20.0,
            f"U_dd/t0={U_over_t0:.1f} ∈ [5, 20] — intermediate charge-transfer",
            f"U_dd/t0={U_over_t0:.1f} outside [5, 20] — {'weak coupling, no Mott' if U_over_t0 < 5.0 else 'frozen Mott, coherence suppressed'}"))
        _scf_log("PREFLIGHT", "  " + _flag(
            J_over_teff < 1.0,
            f"J_eff/t_eff={J_over_teff:.2f} < 1 — itinerant quasiparticles (BdG valid)",
            f"J_eff/t_eff={J_over_teff:.2f} ≥ 1 — AFM-localized, itinerant BdG invalid"))
        _scf_log("PREFLIGHT", "  " + _flag(
            omega_over_teff < 0.5,
            f"ω₀/t_eff={omega_over_teff:.2f} < 0.5 — adiabatic (static Q valid)",
            f"ω₀/t_eff={omega_over_teff:.2f} ≥ 0.5 — quantum phonon, static Q invalid (decrease K_lattice or increase M_eff)"))
        _scf_log("PREFLIGHT", "  " + _flag(
            dCF_over_teff < 2.0,
            f"Δ_CF/t_eff={dCF_over_teff:.2f} < 2 — Γ7a inside the coherent band",
            f"Δ_CF/t_eff={dCF_over_teff:.2f} ≥ 2 — Γ7a near band edge, JT channel weakened"))
        _scf_log("PREFLIGHT", "  " + _flag(
            g7split_over_kT > 2.0,
            f"Γ7split/kT={g7split_over_kT:.1f} > 2 — Γ7a/Γ7b resolved",
            f"Γ7split/kT={g7split_over_kT:.1f} ≤ 2 — Γ7a/Γ7b near-degenerate, JT partner ambiguous"))
        _scf_log("PREFLIGHT", "  " + _flag(
            dB1g_over_dCF < 0.1,
            f"Δ_B1g/Δ_CF={dB1g_over_dCF:.2e} < 0.1 — D4h good approximation",
            f"Δ_B1g/Δ_CF={dB1g_over_dCF:.2e} ≥ 0.1 — static B1g breaks D4h, mechanism modified"))
        _scf_log("PREFLIGHT", "  " + _flag(
            kTc_over_teff < 0.5,
            f"kTc/t_eff={kTc_over_teff:.2f} < 0.5 — Tc below coherence scale",
            f"kTc/t_eff={kTc_over_teff:.2f} ≥ 0.5 — target Tc above bandwidth, mean-field BdG cannot reach it"))

        # ── 3. State content (density, doublet, b1g_weight) ───────────────
        total_n   = np.sum(self._compute_orbital_densities(*E_k_cache))
        beta_all  = self.p.wave_function_weight(tx_bare, ty_bare, self.k_points[:, 0], self.k_points[:, 1])
        b1g_ratio = float(np.linalg.norm(self.B1g_offdiag)) / max(float(np.linalg.norm(np.diag(self.B1g_op))), _MATH_EPS)
        b1g_weight = float(b1g_ratio / (1.0 + b1g_ratio))
        _w_fs = np.einsum('k,kc->c', vertex_cache['fs_weights'], self._fs_channel_weights(vertex_cache['fs_idx'], E_k_cache))
        _w_fs = _w_fs / max(_w_fs.sum(), _MATH_EPS)
        _delta_B1g = self._eval_B1g_and_kweights_at_Q(target_doping, M, Q, Delta_vec, F67_vec, mu, g_t, g_J)[0] \
                    - self._eval_B1g_and_kweights_at_Q(target_doping, M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), mu, g_t, g_J)[0]
        _doublet_tag = ('mixed (mechanism-friendly)' if 0.5 < _w_fs[0] <= 0.85
                        else 'pure Γ6 (weak Γ7a partner — mechanism suppressed)' if _w_fs[0] > 0.85
                        else 'strongly mixed (FS dominated by Γ7)')
        _scf_log("PREFLIGHT",
            f"state: δ={target_doping:.4f}  n={total_n:.6f} (target {1.0 - target_doping:.6f})  "
            f"b1g_weight={b1g_weight:.4f} [{'SC-triggered only' if b1g_weight > 0.90 else 'partial D2h mixing'}]  "
            f"frac_low={float(np.mean(beta_all < 0.25)):.4f}  δB1g(Δ)={_delta_B1g:+.4f}")
        _scf_log("PREFLIGHT",
            f"state: E_F doublet Γ6={_w_fs[0]:.2f} Γ7a={_w_fs[1]:.2f} Γ7b={_w_fs[2]:.2f}  [{_doublet_tag}]")

        # ── 4. ⟨B1g⟩ cluster-vs-BdG benchmark (single line) ───────────────
        _B1g_exp = self._eval_B1g_and_kweights_at_Q(target_doping, M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), mu, g_t, g_J)[0]
        b_mean = F_cluster['b_mean']
        bA_bdg, bB_bdg = self._bdg_b1g_per_sublattice(*E_k_cache)
        bA_cl = 0.5 * (b_mean[0] + b_mean[2])
        bB_cl = 0.5 * (b_mean[1] + b_mean[3])
        _relA = abs(bA_cl - bA_bdg) / max(abs(bA_cl), abs(bA_bdg), 1e-6)
        _relB = abs(bB_cl - bB_bdg) / max(abs(bB_cl), abs(bB_bdg), 1e-6)
        _scf_log("PREFLIGHT",
            f"⟨B1g⟩: cluster/BdG={np.mean(b_mean)/max(_B1g_exp, _MATH_EPS):.4f}  "
            f"A={bA_cl:+.4f}/{bA_bdg:+.4f} (Δ={_relA:.1%})  B={bB_cl:+.4f}/{bB_bdg:+.4f} (Δ={_relB:.1%})")

        # ── 5. h_afm chain + κ_B1g perturbativity ─────────────────────────
        _J_q_afm  = self.p._exchange_J_q(self.p.exchange_channels(Q, g_J)[0], np.pi, np.pi)
        _s_ax_amp = np.array([np.max(np.abs(np.linalg.eigvalsh(self.S_axis_op[2*_c:2*_c+2, 2*_c:2*_c+2]))) for _c in range(_N_CHANNELS)])
        h_afm_M0  = float((_J_q_afm * np.asarray(M, dtype=float) * _s_ax_amp)[0])
        _h_over_halfBW = h_afm_M0 / (_Z * t_eff)
        _h_over_dCF    = h_afm_M0 / abs(self.p.Delta_CF)
        _kappa_over_dCF = max(abs(self.p.kappa_cross[0]), abs(self.p.kappa_cross[1])) / abs(self.p.Delta_CF)
        _scf_log("PREFLIGHT",
            f"h_afm={h_afm_M0*1e3:.1f} meV  h_afm/(Z·t_eff)={_h_over_halfBW:.2f} [{'metallic AFM' if _h_over_halfBW < 1.0 else 'marginal/insulating'}]  "
            f"h_afm/Δ_CF={_h_over_dCF:.2f} [{'not marginal' if _h_over_dCF < 0.6 else 'level-crossing proximity'}]  "
            f"κ_B1g/Δ_CF={_kappa_over_dCF:.3f} [{'perturbative' if _kappa_over_dCF < 0.4 else 'Löwdin estimate biased'}]")

        # ── 6. SCF state + coupling summary ───────────────────────────────
        _regime = 'subcritical' if vertex_cache['lambda_lin_max'] < 0.7 else ('critical' if vertex_cache['lambda_lin_max'] <= 1.4 else 'supercritical')
        _scf_log("PREFLIGHT",
            f"state: M₀={np.array2string(M, precision=4)}  Q₀={Q:+.5f} Å  |Δ|₂={float(np.linalg.norm(np.abs(Delta_vec))):.4e}  "
            f"K_eff={F_cluster['K_eff']:+.4f} eV/Å²  λ_lin_max={vertex_cache['lambda_lin_max']:.3f} [{_regime}]")
        _scf_log("PREFLIGHT",
            f"coupling: g_t={g_t:.4f}  g_J={g_J:.4f}  g_Δs={g_Delta_s:.4f}  g_Δad={g_Delta_ad:.4f}  g_Δbd={g_Delta_bd:.4f}")

    def _make_vertex_params(self, target_doping: float, M: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, chi_QQ: float = 0.0, Q_Eg2: float = 0.0, E_k_cache: tuple = None, vertex_cache: Optional[dict] = None, verbose: bool = False) -> Tuple[float, dict, float, float, float]:
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)

        fs_pts, _, _, _ = self._get_fs_points(M, Q, n_kspace, mu, g_t, g_J, store_cache=False, verbose=verbose)
        Phi_fs = _pairing_basis_matrix(fs_pts[:, 0], fs_pts[:, 1])
        
        gap_a = Phi_fs @ Delta_vec[_IDX_7A_ARR]
        gap_b = Phi_fs @ Delta_vec[_IDX_7B_ARR]
        Delta_retard = float(np.max(np.abs(gap_a)**2 + np.abs(gap_b)**2))

        _omega_sq_JT = max(_JT_PHONON_MEV_CONST * (self._K_bare - self.g_JT_bare**2 * chi_QQ) / self.p.M_eff, _MATH_EPS)
        V_JT_eff  = self.g_JT_bare **2 / (self._K_bare * (1.0 + Delta_retard / _omega_sq_JT))                     # bare RPA/mean-field-level JT pairing vertex with retardation reduction

        F_cluster = self.compute_cluster_free_energy(target_doping, M, Q, Delta_vec, F67_vec, n_kspace, mu, tx_bare, ty_bare, g_t, g_J, V_JT_eff, Q_Eg2, vertex_cache)  # JT stability: Cluster ED embedding field uses only the leading (Γ6) channel -- it has no separate per-channel Weiss field

        V_JT_corr = F_cluster['V_JT_corr']                                                                        # local cluster-ED irreducible B1g vertex correction V_irr_QQ, which is the purely electronic irreducible vertex
        J_eff   = _m2_weighted_J(self.p._exchange_J_q(self.p.exchange_channels(Q, g_J)[0], np.pi, np.pi), M)
        V_cap   = _RPA_V_CAP_ALPHA * max(_RPA_BW_FACTOR * g_t * np.sqrt(0.5 * (tx_bare**2 + ty_bare**2)), J_eff)  # BEC boundary (8t), UV cap on the RPA pairing vertex
        Gamma_M = self._moriya_gamma_landau(target_doping, n_kspace, mu, g_t, g_J, J_eff, V_JT_corr)              # model-derived Moriya damping Γ_M

        if verbose:
            self.preflight_report(target_doping, M, Q, Delta_vec, F67_vec, n_kspace, mu, tx_bare, ty_bare, g_t, g_J, math.sqrt(_omega_sq_JT), F_cluster, E_k_cache, vertex_cache)
        return Gamma_M, F_cluster, V_JT_eff, V_JT_corr, V_cap

    def _rpa_vertex_tables(self, unique_q: np.ndarray, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, Gamma_M: float, J_A1g_diag: np.ndarray, V_JT_eff: float, V_JT_corr: float, E_k_cache: tuple, dHdQ_precomputed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RPA-dressed vertex tables (spin Xi, cross Xc, JT Vjt) at every q of `unique_q` (normal state, Δ=0)."""
        n_q = len(unique_q)
        Xi   = np.empty((n_q, _N_CHANNELS, _N_CHANNELS))
        Xc   = np.empty((n_q, _N_CHANNELS))
        Vjt  = np.empty(n_q)
        for u_idx, q_u in enumerate(unique_q):
            _rpa = self._rpa_closure(M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, q_u, Gamma_M, J_A1g_diag, V_JT_corr, E_k_cache, apply_diamagnetic_QQ=False, dHdQ_precomputed=dHdQ_precomputed)
            J_q, det, M4 = _rpa['J_q'], _rpa['det'], _rpa['M4']
            chiR = _adjugate(M4) @ _rpa['chi0'] / det
            Xi[u_idx]  = -np.outer(J_q, J_q) * chiR[:_N_CHANNELS, :_N_CHANNELS]
            Xc[u_idx]  = 2.0 * J_q * V_JT_eff * chiR[:_N_CHANNELS, _N_CHANNELS]
            Vjt[u_idx] = V_JT_eff ** 2 * chiR[_N_CHANNELS, _N_CHANNELS]
        return Xi, Xc, Vjt

    def compute_pairing_kernel_and_build_cache(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, J_A1g_diag: np.ndarray, Gamma_M: float, V_JT_eff: float, V_JT_corr: float, V_cap: float) -> Dict:
        """
        Full-Fermi-surface pairing kernel with symmetry-weighted vertex, used consistently for BOTH the eigenvalue diagonalization AND the (s, d)
        channel projections feeding the gap equation and the free-energy condensation term.

        Symmetry-weighted vertex
        ------------------------
        Local B1g character at each FS point:
            c_B(k) = |φ_d(k)| / max_k' |φ_d(k')|       ∈ [0, 1]

        The three RPA channels (spin, JT/orbital, spin-JT cross) are combined as:

            pair_char(k, k') = c_B(k) · c_B(k')
            V_w(k, k')       = V_JT_eff(k, k')
                            + pair_char(k, k') · [ V_spin(k, k') + V_cross(k, k') ]

        Exchange of RPA-dressed fluctuations between the pair (k,−k) → (k',−k') with the full response χ_RPA = M⁻¹χ₀ (4×4), U = diag(J_c(q), V_JT_corr), q = k − k':
            V_spin (k,k') = − Σ_cc' √(w_c w_c')(k) · √(w_c w_c')(k') · J_c J_c' [χ_RPA]_cc'(q)
            V_JT   (k,k') =   V_JT_eff² · [χ_RPA]_QQ(q)
            V_cross(k,k') =   2 Σ_c √(w_c(k) w_c(k')) · J_c V_JT_eff · [χ_RPA]_cQ(q)
        where w_c(k) is the doublet-c content of the FS Bloch state (the S_z^(c) vertex of a k → k' scattering ∝ √(w_c(k) w_c(k'))); the JT vertex keeps unit weight as in the scalar kernel.

        Boundary conditions (this is the decisive consistency test):
            Both electrons antinodal  (c_B = c_B' = 1): pair_char = 1
                → V_w = V_JT + V_spin + V_cross = V_full    ✓ (full RPA, deepest trust)
            Both electrons nodal      (c_B = c_B' = 0): pair_char = 0
                → V_w = V_JT only                            ✓ (conservative, on-site-like)
            Mixed (one nodal, one antinodal): pair_char = 0
                → V_w = V_JT only                            ✓ (d-wave symmetry broken at one leg)

        The spin and cross channels share the SAME weighting because they share the SAME q-structure: both are mediated by the AFM-peaked J(q), projected
        by the d-wave form factor, and both contribute to the d-wave pairing gap which is maximal at the antinodes. There is no physical reason to gate them differently.

        The FS-weighted symmetric kernel
            K_fs(k, k') = √(w_k) · V_w(k, k') · √(w_k')
        is diagonalised; its largest eigenvalue is `lambda_lin_max` and its eigenvector is decomposed onto irrep basis, giving the symmetry-channel weights
        that feed the gap equation plus the 'resolved_weight' / 'unresolved_weight' irrep-projection diagnostic.

        The c_B gate above stays tied specifically to the B1g/antinodal structure and weights the SAME V_weighted kernel that all five channels are projected from
        
        Exact G-covariant sector kernel
        ---------------------------------------------------------
        G = {E, T_Q} × {E, kx→−kx, ky→−ky, k→−k} (abelian, order 8; T_Q: k → k+(π,π); the k-space Z2×Z2 is what the code's comments call "D2h") is the group by which the normal state of the collinear AFM acts on the pair
        amplitude Δ(k) inside the diamond MBZ; it is a SUBGROUP of the true symmetry (magnetic point group 4/mm'm' at Q = 0, Δ_B1g_static = 0; m'm'm otherwise): C4 / σd and the antiunitary elements are not listed in G (see the
        symmetry note at _G_OPS for what carries them). `_get_fs_points` returns the FUNDAMENTAL DOMAIN of T_Q (diamond MBZ, closed under the k-space Z2×Z2 but not under T_Q), so the plain kernel V(k_i − k_j) on it is not
        G-covariant: the full-BZ gap equation also contains the image terms V(k_i − T_Q·k_j), which enter with the T_Q parity of the gap function (+ for s-like, − for d-like). Every gap function with a definite symmetry is a
        1-D character χ = (t, mx, my) of G, and the exact projected kernel is

            K_χ(k_i, k_j) = (1/|G|) Σ_{g∈G} χ(g) · V_w(k_i, g·k_j),      V_w(k_i, g·k_j) evaluated at q_g = k_i − g·k_j, with the V_cap clamp applied to every image pair.

        * The 8 sector kernels are diagonalised separately; lambda_lin_max is the largest λ_top among the sectors that contain at least one basis function ("covered"), `lambda_lin_max_all` the largest
          of all 8 (a larger uncovered λ is a BASIS GAP, reported by _sector_report_lines together with the library harmonics that would close it).
        * V_irrep, W_irrep, S_irrep are block-diagonal in χ (functions of different T_Q parity cannot mix at the linear level; the nonlinear BdG equations still couple them);
          v_raw / resolved_weight come from the Loewdin decomposition of the leading covered sector's eigenvector onto the basis functions OF THAT SECTOR.
        * 'frac_seed' additionally mixes in the near-degenerate covered sectors (λ_χ ≥ _SECTOR_NEAR_FRAC·λ_lead)
        * A sector is a character of G, not a D4h irrep: (+,+,+) holds A1g ⊕ B1g {d2}, (−,+,+) holds A1g {es, es3} ⊕ B1g {d, d3}, (−,−,−) holds A2g ⊕ B2g. `sector_lambda` is the top eigenvalue over everything in the sector; the
          per-irrep eigenvalues come from the σd (kx↔ky) split of the FS sample, cache keys 'sector_swap_lambda' (σd-even, σd-odd) and 'sector_swap_defect' (≠ 0: the kernel is not C4-covariant, the split is then only variational).
          Which irrep a sector's top eigenvector belongs to is read from 'sector_frac' / 'sector_vraw' (Loewdin weights).
        """
        # --- Fermi-surface points ---
        fs_pts, vF_arr, fs_idx, fs_weights = self._get_fs_points(M, Q, n_kspace, mu, g_t, g_J, store_cache=True)

        Phi_fs  = _pairing_basis_matrix(fs_pts[:, 0], fs_pts[:, 1])   # (N_fs, _N_IRREP), _IRREP_NAMES order
        phi_d   = Phi_fs[:, _IR_D]
        phi_s_d = Phi_fs[:, _IR_S]     # == np.ones(N_fs)

        E_k_cache = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self._get_vbdg())
        dHdQ_precomputed = self._calc_dHdQ(M, Q, np.zeros(_N_DELTA, dtype=complex), mu, g_t, g_J)

        N_fs = len(fs_pts)
        _ar  = np.arange(N_fs)

        # √w_c(k) — amplitude (square root of the doublet fraction) per FS point, c ∈ {Γ6, Γ7a, Γ7b}.
        sqrt_doublet_frac = np.sqrt(np.clip(self._fs_channel_weights(fs_idx, E_k_cache), 0.0, None))   # (N_fs, 3)
        c_B    = np.abs(phi_d) / max(float(np.max(np.abs(phi_d))), 1e-12)
        sqrt_w = np.sqrt(fs_weights)
        Xw     = Phi_fs * fs_weights[:, None]   # (N_fs, N_IRREP) FS-weighted basis functions

        i_idx, j_idx = np.triu_indices(N_fs)
        n_pairs = len(i_idx)

        sqrt_doublet_frac_pair = sqrt_doublet_frac[i_idx] * sqrt_doublet_frac[j_idx]   # (P, 3)

        def _pair_vertex(ig, Xi, Xc, Vjt):
            """(spin, JT, cross) vertex of the FS pairs at the q-table rows `ig`, V_cap-clamped on the 3-channel sum."""
            V_pairs = np.stack([
                np.einsum('pc,pcd,pd->p', sqrt_doublet_frac_pair, Xi[ig], sqrt_doublet_frac_pair, optimize=True),
                Vjt[ig],
                np.einsum('pc,pc->p',     sqrt_doublet_frac_pair, Xc[ig],                      optimize=True),
            ])
            _V_sum = V_pairs.sum(axis=0)
            _over  = np.abs(_V_sum) > V_cap
            if np.any(_over):
                V_pairs[:, _over] *= V_cap / np.abs(_V_sum[_over])
            return V_pairs

        def _sym_full(V_pairs, i_idx, j_idx):
            """Correct symmetric (3, N, N) matrices from the upper-triangle (incl. diagonal) pair list."""
            A = np.zeros((3, N_fs, N_fs), dtype=float)
            A[:, i_idx, j_idx] = V_pairs
            A = A + A.transpose(0, 2, 1)
            A[:, _ar, _ar] *= 0.5
            return A

        odd_idx = [i for i, n in enumerate(_IRREP_NAMES) if _IRREP_PARITY[n] < 0]

        # exact G = {E, T_Q} × {kx→−kx, ky→−ky} sector kernels
        n_ops = len(_G_OPS)

        # canonical q of every (group element, pair): q_g = k_i − g·k_j (exact continuous action, folded, |q_x|,|q_y| canonical -- the same D2h-Shubnikov assumption as _unique_q_pairs).
        # The T_Q image maps the canonical q-set (π − |q|) onto itself, so the number of RPA evaluations stays bounded by the (NK/2+1)² canonical grid however many images are used.
        key_list = []
        for _op in _G_OPS:
            gx, gy = _g_apply(_op, fs_pts[:, 0], fs_pts[:, 1])
            q_raw  = np.column_stack((fs_pts[i_idx, 0] - gx[j_idx], fs_pts[i_idx, 1] - gy[j_idx]))
            q_fold = (q_raw + np.pi) % (2.0 * np.pi) - np.pi
            key_list.append(np.abs(np.rint(q_fold * _Q_UNIQUE_SCALE)).astype(np.int64))
        u_int, inv_all = np.unique(np.concatenate(key_list, axis=0), axis=0, return_inverse=True)
        inv_all = np.asarray(inv_all).reshape(n_ops, n_pairs)
        unique_q = (u_int.astype(np.float64) / _Q_UNIQUE_SCALE + np.pi) % (2.0 * np.pi) - np.pi

        Xi, Xc, Vjt = self._rpa_vertex_tables(unique_q, M, Q, n_kspace, mu, g_t, g_J, Gamma_M, J_A1g_diag, V_JT_eff, V_JT_corr, E_k_cache, dHdQ_precomputed)

        # K_sec[χ, ch] = (1/|G|) Σ_g χ(g)·V^ch(k_i, g·k_j)   (spin / JT / cross, ungated, V_cap-clamped per image pair)
        K_sec = np.zeros((n_ops, _N_CHANNELS, N_fs, N_fs), dtype=float)
        V_parts_diag = np.zeros((3, _N_IRREP))
        for gi in range(n_ops):
            V_pairs_g = _pair_vertex(inv_all[gi], Xi, Xc, Vjt)
            A_g = _sym_full(V_pairs_g, i_idx, j_idx)
            K_sec += (_G_CHI[:, gi] / n_ops)[:, None, None, None] * A_g[None, :, :, :]

        cBcB = c_B[:, None] * c_B[None, :]
        V_sec = K_sec[:, 1] + cBcB[None] * (K_sec[:, 0] + K_sec[:, 2])   # (n_sectors, N, N): gated kernel per sector

        V_pairs_0 = _pair_vertex(inv_all[0], Xi, Xc, Vjt)
        A_0 = _sym_full(V_pairs_0, i_idx, j_idx)
        V_identity_gated = A_0[1] + cBcB * (A_0[0] + A_0[2])

        V_irrep = np.zeros((_N_IRREP, _N_IRREP), dtype=float)
        W_irrep = np.zeros((_N_IRREP, _N_IRREP), dtype=float)
        S_irrep = np.zeros((_N_IRREP, _N_IRREP), dtype=float)
        sector_lam      = np.full(n_ops, np.nan)
        sector_lam2     = np.full(n_ops, np.nan)
        sector_psi      = np.zeros((n_ops, N_fs))
        sector_vraw     = np.zeros((n_ops, _N_IRREP))
        sector_frac     = np.zeros((n_ops, _N_IRREP))
        sector_resolved = np.zeros(n_ops)
        sector_c        = np.zeros((n_ops, _N_IRREP))

        for si, key in enumerate(_G_OPS):
            K_s = sqrt_w[:, None] * V_sec[si] * sqrt_w[None, :]
            ev_s, U_s = np.linalg.eigh(0.5 * (K_s + K_s.T))
            sector_lam[si]  = float(ev_s[-1])
            sector_lam2[si] = float(ev_s[-2]) if N_fs > 1 else float('nan')
            sector_psi[si]  = U_s[:, -1]
            cols = _SECTOR_MEMBERS[key]
            if not cols:
                continue
            for a in cols:
                V_parts_diag[0, a] = _wave_project(cBcB * K_sec[si, 0], Phi_fs[:, a], Phi_fs[:, a], fs_weights)
                V_parts_diag[1, a] = _wave_project(       K_sec[si, 1], Phi_fs[:, a], Phi_fs[:, a], fs_weights)
                V_parts_diag[2, a] = _wave_project(cBcB * K_sec[si, 2], Phi_fs[:, a], Phi_fs[:, a], fs_weights)
                for b in cols:
                    if b >= a:
                        V_irrep[a, b] = V_irrep[b, a] = float(np.clip(_wave_project(V_sec[si], Phi_fs[:, a], Phi_fs[:, b], fs_weights), -V_cap, V_cap))
            W_irrep[np.ix_(cols, cols)] = Xw[:, cols].T @ V_sec[si] @ Xw[:, cols]
            dec = _irrep_decomposition(U_s[:, -1], sqrt_w, fs_weights, Phi_fs[:, cols])     # Loewdin projection onto the SAME-sector basis functions only
            S_irrep[np.ix_(cols, cols)] = dec['S']
            sector_c[si, cols]    = dec['c']
            sector_vraw[si, cols] = dec['v_raw']
            sector_frac[si, cols] = dec['frac']
            sector_resolved[si]   = dec['resolved_weight']

        # --- σd (kx↔ky) resolution of the even-parity (mx = my) sectors ---
        # G contains neither C4 nor σd, so a sector can hold two D4h irreps (A1g ⊕ B1g for mx = my = +1, B2g ⊕ A2g for −1) and its top eigenvalue alone does not say which leads (e.g. 'es' vs 'd' in (−,+,+)). Splitting the FS sample
        # into σd-even / σd-odd functions gives each irrep's top eigenvalue: exact for a C4-covariant kernel; `sector_swap_defect` = max|K − P K Pᵀ| / max|K| measures the deviation.
        perm_sw, closed_sw = _grid_sym_permutation(fs_pts, 'swap')
        swap_ok = bool(closed_sw and float(np.max(np.abs(fs_weights - fs_weights[perm_sw]))) <= _ODD_VTX_TOL * max(float(np.max(fs_weights)), 1e-300))
        sector_swap_lam    = np.full((n_ops, 2), np.nan)          # [:, 0] σd-even, [:, 1] σd-odd;  NaN for mx ≠ my (σd maps such a sector onto its px ↔ py partner, whose λ must coincide in a C4-covariant kernel)
        sector_swap_defect = np.full(n_ops, np.nan)
        if swap_ok:
            for si, key in enumerate(_G_OPS):
                if key[1] != key[2]:
                    continue
                K_s = sqrt_w[:, None] * V_sec[si] * sqrt_w[None, :]
                sector_swap_lam[si, 0], sector_swap_lam[si, 1], sector_swap_defect[si] = _swap_resolved_top(K_s, perm_sw)
        h_c4_symmetric = bool(abs(float(getattr(self.p, "Delta_B1g_static", 0.0))) <= _SYM_THRESH and abs(float(Q)) <= _SYM_THRESH)   # H itself is C4-symmetric (no static B1g field, Q = 0)
        h_c4_breaker = float(abs(float(getattr(self.p, "Delta_B1g_static", 0.0))) / max(abs(float(self.p.Delta_CF)), _MATH_EPS) + abs(float(Q)) / max(float(self.p.lambda_hop), _MATH_EPS))   # dimensionless size of the explicit C4 breakers of H: static B1g field / crystal-field gap + distortion / hopping decay length

        # --- FS-closure check for the odd-parity channels ---
        # Kernel-level D2h selection rules follow from G-character orthogonality. The only additional requirement is an inversion-closed FS sample with inversion-symmetric weights, ensuring well-defined projections of the odd-parity (px, py, px2, py2) form factors onto V_sec[χ].
        closed_inv, closed_mx, closed_my, weight_sym_ok, weight_asym = _check_fs_closure(fs_pts, fs_weights)
        odd_ok = bool(closed_inv and weight_sym_ok)
        _bdy = _on_tq_boundary(fs_pts)
        tq_ok = bool(not np.any((_tq_partner(fs_pts) >= 0) != _bdy))                     # interior without T_Q image in the sample, every boundary point paired
        fs_boundary_frac = float(np.sum(fs_weights[_bdy]) / max(float(np.sum(fs_weights)), 1e-300))
        if not tq_ok:
            _scf_log("PAIR-WARN", "  ⚠ the FS sample is not a T_Q fundamental domain (unpaired boundary points or T_Q images inside): the image terms of EVERY sector kernel are biased, not only the odd ones")
        if not odd_ok:
            for _i_odd in odd_idx:
                V_irrep[_i_odd, :] = 0.0
                V_irrep[:, _i_odd] = 0.0
                W_irrep[_i_odd, :] = 0.0
                W_irrep[:, _i_odd] = 0.0
                V_parts_diag[:, _i_odd] = 0.0
            _scf_log("PAIR-WARN", "  ⚠ odd-channel FS-closure check FAILED — odd rows/columns of V_irrep, W_irrep, V_parts_diag set to 0: "
                    f"closed_inv={closed_inv}  weight_asym={weight_asym:.2e}  closed_mx={closed_mx}  closed_my={closed_my}")

        # --- Sector selection with explicit numerical-validity handling ---
        all_active = [si for si, key in enumerate(_G_OPS) if _SECTOR_MEMBERS[key]]
        if not all_active:
            raise RuntimeError("No pairing sector is covered by the active basis.")
        valid_active = [si for si in all_active if odd_ok or (_G_OPS[si][1] * _G_OPS[si][2] > 0)]
        if not valid_active:
            raise RuntimeError("No numerically validated pairing sector is available.")

        sector_drives = np.zeros(n_ops, dtype=bool)
        sector_drives[valid_active] = True
        si_lead = max(valid_active, key=lambda si: sector_lam[si])
        si_all  = int(np.nanargmax(sector_lam))
        lambda_lin_max = float(sector_lam[si_lead])
        sym_lin = _IRREP_NAMES[int(np.argmax(np.abs(sector_vraw[si_lead])))]

        # Exact Rayleigh-quotient decomposition of every sector's top eigenvalue into the three kernel channels:
        sector_C = np.full((n_ops, _N_CHANNELS), np.nan)          # columns follow K_sec: 0 = spin, 1 = JT, 2 = cross
        for _si in range(n_ops):
            _x = sqrt_w * sector_psi[_si]
            for _ch in range(_N_CHANNELS):
                _y = _x if _ch == 1 else c_B * _x
                sector_C[_si, _ch] = float(_y @ K_sec[_si, _ch] @ _y)
        frac = sector_frac[si_lead].copy()

        seed_w = np.zeros(n_ops)
        for si in valid_active:                              # near-degenerate validated sectors also seed the kick (weight λ_χ/λ_lead)
            if lambda_lin_max > 0.0 and sector_lam[si] > 0.0 and sector_lam[si] >= _SECTOR_NEAR_FRAC * lambda_lin_max:
                seed_w[si] = sector_lam[si] / lambda_lin_max * sector_resolved[si]
        frac_seed = (seed_w[:, None] * sector_frac).sum(axis=0)
        frac_seed = frac_seed / frac_seed.sum() if frac_seed.sum() > 0.0 else frac.copy()
        v_irr_signed = np.zeros(_N_IRREP, dtype=float)
        for _si in np.flatnonzero(seed_w):
            _vs = np.asarray(sector_vraw[_si], dtype=float)
            # sign-alignment per sector: aligns to the sign of the largest |v| component
            _i_max = int(np.argmax(np.abs(_vs)))
            _sign = 1.0 if _vs[_i_max] >= 0.0 else -1.0
            v_irr_signed += seed_w[_si] * _sign * _vs
        v_irr_signed /= max(float(np.abs(v_irr_signed).sum()), 1e-12)

        _d_sec = _G_OPS.index(_IRREP_CHAR['d'])
        vertex_cache = {
            'M':                  M,
            'Q':                  Q,
            'fs_pts':             fs_pts,
            'vF_arr':             vF_arr,
            'fs_idx':             fs_idx,
            'phi_s_d':            phi_s_d,
            'phi_d':              phi_d,
            'Phi_fs':             Phi_fs,
            'fs_weights':         fs_weights,
            'V_irrep':            V_irrep,
            'W_irrep':            W_irrep,
            'S_irrep':            S_irrep,
            'c_irrep':            sector_c[si_lead].copy(),
            'sym_lin':            sym_lin,
            'frac_seed':          frac_seed,
            'frac_seed_signed':   v_irr_signed,
            'seed_w':             seed_w,
            'resolved_weight':    float(sector_resolved[si_lead]),
            'unresolved_weight':  float(1.0 - sector_resolved[si_lead]),
            'unique_q':           unique_q,
            'i_idx':              i_idx,
            'j_idx':              j_idx,
            'lambda_lin_max':     lambda_lin_max,
            'lambda_lin_max_all': float(sector_lam[si_all]),
            'V_JT_eff':           V_JT_eff,
            'V_JT_corr':          V_JT_corr,
            'c_B':                c_B,
            'W22_JT':             float(np.dot(phi_d * fs_weights, np.dot(K_sec[_d_sec, 1], phi_d * fs_weights))),
            'S_cond':             max((float(np.linalg.cond(S_irrep[np.ix_(_c, _c)])) for _c in _SECTOR_MEMBERS.values() if _c), default=1.0),
            'sector_lambda':      sector_lam,
            'sector_lambda2':     sector_lam2,
            'sector_psi':         sector_psi,
            'sector_C':           sector_C,                # (n_ops, 3): Rayleigh contributions to sector_lambda, columns spin / JT / cross
            'sector_frac':        sector_frac,
            'sector_keys':        list(_G_OPS),
            'V_parts_diag':       V_parts_diag,
            'sector_vraw':        sector_vraw,
            'sector_resolved':    sector_resolved,
            'leading_sector':     _G_OPS[si_lead],
            'leading_sector_all': _G_OPS[si_all],
            'sector_drives':      sector_drives,
            'K_d_spin':           K_sec[_d_sec, 0],
            'V_self_diag':        V_identity_gated.diagonal().copy(),
            'fs_closed_inv':      closed_inv,
            'fs_closed_mx':       closed_mx,
            'fs_closed_my':       closed_my,
            'fs_weight_sym_ok':   weight_sym_ok,
            'fs_weight_asym':     weight_asym,
            'fs_tq_ok':           tq_ok,                   # sample is a T_Q fundamental domain (interior free of T_Q images, boundary paired)
            'fs_boundary_frac':   fs_boundary_frac,        # weight fraction on the T_Q-paired boundary |kx|+|ky| = π (carried at ½ per point)
            'odd_channels_ok':    odd_ok,
            'fs_closed_swap':     swap_ok,                 # FS sample closed under kx↔ky with σd-invariant weights (the σd resolution below is defined)
            'sector_swap_lambda': sector_swap_lam,         # (n_ops, 2): top eigenvalue in the σd-even / σd-odd subspace (mx = my sectors only)
            'sector_swap_defect': sector_swap_defect,      # (n_ops,): max|K − P K Pᵀ| / max|K|  (0 for a C4-covariant kernel)
            'h_c4_symmetric':     h_c4_symmetric,          # no explicit C4 breaker in H (Δ_B1g_static = 0, Q = 0): a nonzero defect is then a defect of the kernel, not a physical C4 breaking
            'h_c4_breaker':       h_c4_breaker,            # Δ_B1g_static/Δ_CF + |Q|/λ_hop: scale of the explicit C4 breakers of H, to be compared with sector_swap_defect
        }
        return vertex_cache

    def scf_gap_diagnostics(self, Delta_vec: np.ndarray, g_Delta_s: float, g_Delta_ad: float, E_k_cache: tuple, vertex_cache: dict = None) -> Dict:
        """Diagnostic post-processing of the linearised gap equation results. Returns a new dictionary that merges the vertex_cache with additional diagnostics"""
        fs_pts = vertex_cache['fs_pts']
        vF_arr = vertex_cache['vF_arr']
        fs_idx = vertex_cache['fs_idx']
        frac   = vertex_cache['frac_seed']
        Phi_fs = vertex_cache.get('Phi_fs')
        ev, ec = E_k_cache
        # The Γ7a and Γ7b branches pair Γ6 with two different Γ7 doublets (orthogonal pair channels): combine their magnitudes in quadrature, not as a complex sum.
        Delta_s = float(math.hypot(abs(Delta_vec[_DELTA_IDX['s7a']]), abs(Delta_vec[_DELTA_IDX['s7b']])))
        Delta_d = float(math.hypot(abs(Delta_vec[_DELTA_IDX['d7a']]), abs(Delta_vec[_DELTA_IDX['d7b']])))

        lambda_lin_max = float(vertex_cache['lambda_lin_max'])
        vF_mag = np.linalg.norm(vF_arr, axis=1)

        # Per-irrep linear eigenvalue decomposition (generalises lambda_s/lambda_d to all _N_IRREP channels); lambda_s/lambda_d are kept as their original scalar aliases.
        _g_irrep = _gutzwiller_irrep_vector(g_Delta_s, g_Delta_ad, g_Delta_s)    # the same per-irrep g as the vertex / gap equation (same-sublattice bonds: g_Δs)
        lambda_irrep = _g_irrep * lambda_lin_max * frac
        if vertex_cache.get('sector_lambda') is not None:
            # 'sector' mode: channel i reports the instability of ITS OWN G-sector, λ_irrep[i] = g_i · λ_top(sector of i) · (weight of i inside that sector's eigenvector);
            # for the leading covered sector this is identical to the line above, for the other sectors it is their own (non-zero) λ.
            lambda_irrep = np.zeros(_N_IRREP)
            for _si, _key in enumerate(_G_OPS):
                for _i in _SECTOR_MEMBERS[_key]:
                    lambda_irrep[_i] = _g_irrep[_i] * float(vertex_cache['sector_lambda'][_si]) * float(vertex_cache['sector_frac'][_si][_i])
        lambda_s = float(lambda_irrep[_IR_S])
        lambda_d = float(lambda_irrep[_IR_D])

        # Coherence length — nodal/antinodal decomposition for d-wave validity; for d-wave Δ(k)=Δ_d·(cos kx−cos ky): ξ diverges at nodes, shortest at antinodes.
        # BdG validity (phase coherence) is governed by the NODAL sector (superfluid density ∝ v_F at nodes).
        if Phi_fs is not None and len(fs_pts) > 0:
            # Exact order-parameter magnitude |Δ(k)| on the actual FS over all _N_IRREP irreps: irreps add COHERENTLY within a Γ7 branch (their relative phases matter: s+id is nodeless), the two branches in QUADRATURE (orthogonal pair channels)
            gap_a = Phi_fs @ Delta_vec[_IDX_7A_ARR]
            gap_b = Phi_fs @ Delta_vec[_IDX_7B_ARR]
            gap_k = np.sqrt(np.abs(gap_a)**2 + np.abs(gap_b)**2)
            Delta_0 = float(np.max(gap_k))
        else:
            gap_k = None
            Delta_0 = max(abs(Delta_s), 2.0 * abs(Delta_d))
        if Delta_0 > 1e-8:
            vF_avg = float(np.mean(vF_mag))

            # Nodal / antinodal regions are defined on the GAP SHAPE |Δ(k)| itself (for a pure d-wave state |Δ(k)| ∝ |φ_d(k)| (for a pure-s state Δ_d = 0 gave Δ_nodal = ε and a divergent ξ_nodal).
            if gap_k is not None:
                shape_abs = gap_k
                shape_ok  = float(shape_abs.max()) > _MATH_EPS
            else:
                shape_abs = np.abs(Phi_fs[:, _IR_D]) if Phi_fs is not None else np.abs(np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1]))
                shape_ok  = float(shape_abs.max()) > _PHI_D_FLOOR
            # An (almost) flat gap has no nodal/antinodal structure -> isotropic ξ, as for the former 'pure s-wave' branch.
            _aniso_ok = shape_ok and (float(shape_abs.min()) < 0.9 * float(shape_abs.max()))

            if _aniso_ok:
                nodal_mask = shape_abs < np.percentile(shape_abs, _NODAL_REGION_PCTL)
                antinodal_mask = shape_abs > np.percentile(shape_abs, 100 - _NODAL_REGION_PCTL)

                # Nodal ξ: 25th percentile of the gap inside the nodal region as gap scale (conservative).
                if nodal_mask.sum() >= _VERTEX_DIAG_MIN_FS :
                    vF_nodal     = float(np.mean(vF_mag[nodal_mask]))
                    shape_nod_vals = shape_abs[nodal_mask]
                    Delta_nodal  = max(float(np.percentile(shape_nod_vals, _NODAL_REGION_PCTL)) if gap_k is not None
                                       else abs(Delta_d) * float(np.percentile(shape_nod_vals, _NODAL_REGION_PCTL)), _MATH_EPS)
                    xi_nodal     = vF_nodal / (np.pi * Delta_nodal)
                else:
                    vF_nodal = vF_avg
                    xi_nodal = vF_avg / (np.pi * max(abs(Delta_d) * 0.2, _MATH_EPS))

                # Antinodal ξ: gap ≈ Δ_0 at antinodes is diagnostic for BEC–BCS crossover;
                if antinodal_mask.sum() >= _VERTEX_DIAG_MIN_FS:
                    vF_antinodal = float(np.mean(vF_mag[antinodal_mask]))
                    xi_antinodal = vF_antinodal / (np.pi * Delta_0)
                else:
                    xi_antinodal = vF_avg / (np.pi * Delta_0)
            else:
                # Pure s-wave or very small d-wave form factor: use isotropic ξ
                xi_nodal     = vF_avg / (np.pi * Delta_0)
                xi_antinodal = xi_nodal

            if fs_idx is not None and ev is not None and ec is not None and len(fs_idx) >= _VERTEX_DIAG_MIN_FS:
                # Full orbital-resolved ξ computation (Γ₆ vs Γ₇ character)
                window = _FS_SAMPLING * self.kT
                w6_arr, w7_arr = [], []
                for i in range(len(fs_idx)):
                    ev_k = ev[fs_idx[i]]
                    ec_k = ec[fs_idx[i]]
                    weights = np.exp(-ev_k**2 / (2 * window**2))
                    wsum = np.sum(weights)
                    if wsum < 1e-30:
                        # fall back to uniform weight so the k-point still contributes
                        weights = np.ones_like(weights)
                        wsum = float(len(weights))
                    weights /= wsum
                    w6 = float(np.sum(weights * np.sum(np.abs(ec_k[0:2, :])**2, axis=0)))
                    w7 = float(np.sum(weights * np.sum(np.abs(ec_k[2:6, :])**2, axis=0)))   # Γ7a+Γ7b combined
                    norm = max(w6 + w7, 1e-12)
                    w6_arr.append(w6 / norm)
                    w7_arr.append(w7 / norm)
                vF_G6 = float(np.average(vF_mag, weights=np.array(w6_arr) + 1e-12))
                vF_G7 = float(np.average(vF_mag, weights=np.array(w7_arr) + 1e-12))
                xi_Gamma6 = vF_G6 / (np.pi * Delta_0)
                xi_Gamma7 = vF_G7 / (np.pi * Delta_0)
                orbital_selective = abs(xi_Gamma6 - xi_Gamma7) / max(xi_nodal, 1e-12) > _ORBITAL_SEL_FRAC
            else:
                xi_Gamma6 = xi_nodal
                xi_Gamma7 = xi_nodal
                orbital_selective = False

            # BdG validity: nodal coherence length must exceed 2 lattice constants.
            valid_BdG = xi_nodal > _XI_NODAL_MIN
        else:
            xi_Gamma6 = xi_Gamma7 = xi_nodal = xi_antinodal = vF_avg = 0.0
            valid_BdG = False
            orbital_selective = False

        # ---- Assemble result (vertex cache + new diagnostics) ----
        result = dict(vertex_cache)
        result.update({
            'lambda_s': lambda_s,
            'lambda_d': lambda_d,
            'lambda_irrep': lambda_irrep,
            'xi_Gamma6': xi_Gamma6,
            'xi_Gamma7': xi_Gamma7,
            'xi_nodal': xi_nodal,
            'xi_antinodal': xi_antinodal if Delta_0 > 1e-8 else 0.0,
            'valid_BdG': valid_BdG,
            'orbital_selective': orbital_selective,
        })
        return result

    def _find_mu_for_density(self, M: float, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, target_doping: float, mu_guess: float, g_t: float, g_J: float) -> Tuple[float, float]:
        """
        Find the chemical potential μ such that the BdG electron density satisfies n(μ) = 1 − target_doping.
        The Gutzwiller factors and the exchange prefactor are functions of the total density, so the fixpoint values are determined analytically.

        The cheap estimator: ∂n/∂μ ≈ Σₖ,ₙ wₖ f(Eₙ)[1−f(Eₙ)]/kT  is the standard BCS approximation. It effectively replaces the exact Hellmann–Feynman slope,
          ∂Eₙ/∂μ = ⟨n|∂H/∂μ|n⟩ = −(Pₙ−Hₙ),  Pₙ+Hₙ=1,
        by −1, assuming purely particle-like BdG states (Pₙ=1). This is exact only for Δ=0, where eigenstates are pure particle or hole branches.
        For finite Δₛ or Δ_d, Bogoliubov mixing gives Pₙ≠1, so ∂Eₙ/∂μ = 1−2Pₙ ≠ −1, and dn/dμ acquires additional eigenvector (Sternheimer-type) contributions omitted by the shortcut.
        Thus, once superconductivity is present, the analytic slope is only approximate: although μ enters H only through the diagonal −μτ_z term, the density's μ-dependence also reflects the evolving u/v coherence factors.
        A plain Newton step can overshoot when the DOS—and hence ∂n/∂μ—changes rapidly (e.g. near a Van Hove singularity). Instead, each step is backtracked (halved) until |n(μ)−target_n| decreases.
        If backtracking fails, the routine switches to a numerical derivative; if convergence is still not achieved within the iteration budget, it falls back to a guaranteed bracketed Brent solve rather than returning an unconverged μ.
        """
        target_n = 1.0 - target_doping
        vbdg = self._get_vbdg()
        _use_numeric_deriv = np.sum(np.abs(Delta_vec)) > _MU_SC_DERIV_THRESH
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)
        t_eff = g_t * np.sqrt(0.5 * (tx_bare**2 + ty_bare**2))

        def _diag_and_density(mu_val: float) -> Tuple[np.ndarray, np.ndarray, float]:
            ev, ec = np.linalg.eigh(
                vbdg._build_H_stack(self.k_points, M, Q, Delta_vec, F67_vec, target_n, mu_val, g_t, g_J, out=vbdg._H_stack)
            )
            dens = self._compute_orbital_densities(ev, ec)
            n_total = np.sum(dens)
            return ev, ec, n_total

        def _analytic_dn_dmu(ev: np.ndarray, ec: np.ndarray) -> float:
            # ∂n/∂μ ≈ Σ_k (f(1-f)/kT) * (|u_A|²+|u_B|²+|v_A|²+|v_B|²) / 4 — exact only when every BdG branch is pure particle/hole (Δ=0); see docstring.
            f = _fermi_function(ev, self.kT)
            fb = 1.0 - f
            uA, uB, vA, vB = _get_nambu_spinors(ec)
            df_dE = f * fb / self.kT
            w_A = np.sum(np.abs(uA)**2 + np.abs(vA)**2, axis=1)
            w_B = np.sum(np.abs(uB)**2 + np.abs(vB)**2, axis=1)
            return float(np.einsum('k,kn,kn->', self.k_weights, df_dE, w_A + w_B)) / 4.0

        def _numeric_dn_dmu(mu_val: float) -> float:
            # Centred finite difference of the FULL n(μ) (coherence factors included): captures both the (1−2P_n) Hellmann–Feynman correction and the eigenvector response that the analytic estimator misses. Costs two extra diagonalisations.
            h = max(1e-5, 1e-4 * self.p.t0)
            _, _, n_p = _diag_and_density(mu_val + h)
            _, _, n_m = _diag_and_density(mu_val - h)
            return (n_p - n_m) / (2.0 * h)

        def density_error(mu_val: float) -> float:
            _, _, n = _diag_and_density(mu_val)
            return n - target_n

        def _bracket_and_brent(mu_center: float) -> Optional[float]:
            """Guaranteed-convergence fallback: expand a bracket around mu_center until the sign
            of density_error flips (n(μ) is monotonic in μ at fixed M,Q,Δ), then Brent."""
            w = _RPA_BW_FACTOR * t_eff
            mu_min, mu_max = mu_center - w, mu_center + w
            err_min, err_max = density_error(mu_min), density_error(mu_max)
            for _ in range(10):
                if err_min * err_max <= 0.0:
                    return brentq(density_error, mu_min, mu_max, xtol=_BRENTQ_TOL)
                if err_min > 0.0:
                    mu_min -= w
                    err_min = density_error(mu_min)
                else:
                    mu_max += w
                    err_max = density_error(mu_max)
            return None

        mu = mu_guess
        n_at_mu = target_n
        err = 0.0

        for _ in range(_MU_NEWTON_MAXIT):
            ev, ec, n = _diag_and_density(mu)
            err = n - target_n
            n_at_mu = n

            if abs(err) < _MU_DENSITY_TOL:
                return mu, n_at_mu
            deriv = _numeric_dn_dmu(mu) if _use_numeric_deriv else _analytic_dn_dmu(ev, ec)

            if abs(deriv) < _DEN_DERIV_FLOOR:
                mu_root = _bracket_and_brent(mu)
                if mu_root is not None:
                    return mu_root, density_error(mu_root) + target_n
                continue   # genuinely degenerate derivative; let the next pass re-evaluate
            raw_step = float(np.clip(err / deriv, -self.p.t0, self.p.t0))

            # Backtracking/damped Newton: protects against Van Hove-induced overshoot
            eta = 1.0
            mu_trial = mu - eta * raw_step
            err_trial = density_error(mu_trial)
            _bt = 0
            while abs(err_trial) >= abs(err) and eta > _MU_BACKTRACK_FLOOR and _bt < _MU_BACKTRACK_MAX:
                eta *= 0.5
                mu_trial = mu - eta * raw_step
                err_trial = density_error(mu_trial)
                _bt += 1

            if abs(err_trial) >= abs(err):
                # Backtracking exhausted with no improvement: the local derivative estimate (analytic or numeric) is unreliable here
                # — escalate to the numeric derivative and retry from the same μ rather than accepting a non-improving step.
                _use_numeric_deriv = True
                continue

            mu = mu_trial
            n_at_mu = err_trial + target_n

        if abs(n_at_mu - target_n) >= _MU_DENSITY_TOL:
            mu_root = _bracket_and_brent(mu)
            if mu_root is not None:
                mu = mu_root
                n_at_mu = density_error(mu) + target_n
            else:
                _M_repr = np.array2string(np.atleast_1d(M), precision=4) if np.ndim(M) > 0 else f"{M:.4f}"
                warnings.warn(
                    f"_find_mu_for_density: failed to bracket the density root after {_MU_NEWTON_MAXIT} Newton iterations (M={_M_repr}, Q={Q:.4f}, "
                    f"|Δ|={np.sum(np.abs(Delta_vec)):.4f}); returning best estimate μ={mu:.6f} with |n−target|={abs(n_at_mu-target_n):.2e}.",
                    RuntimeWarning,
                )
        return mu, n_at_mu

    def _compute_bdg_free_energy(self, M_channels: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_ad: float, g_Delta_bd: float, V_JT_corr: float, Q_Eg2: float = 0.0, K_eff_for_free_energy: float = 0.0, K_eff_Eg2_for_free_energy: float = 0.0, vertex_cache: dict = None, return_cond: bool = False):
        """
        Generalized free energy with CHANNEL-PER-SIZE (Γ6, Γ7a, Γ7b) order parameters, including both Γ6↔Γ7a and Γ6↔Γ7b pairing channels.
        The grand potential per site is computed from the k-space BdG spectrum:

            Ω = (1/4) Σ_{k,n} w_k [E_n f_n − T S(f_n)] + (1/4)·Tr[H_particle]
                + E_cond
                + (K_spring_for_free_energy/2) Q²
                + (Z_afm_eff/2) Σ_c J_c M_c²

        Condensation energy: Γ7a and Γ7b share the on-site s-wave vertex V_s, but their NN-bond vertices (es, d, px, py) differ due to their distinct
        orbital character and Gutzwiller renormalisation factors. The condensation energy is the sum of two independent N_IRREP-channel contributions:

            E_cond = Re[Δ_7a† · V_a^{-1} · Δ_7a] + Re[Δ_7b† · V_b^{-1} · Δ_7b]

        V_a and V_b are from the shared V_irrep, rescaled by the Gutzwiller vector of each channel group. Δ_7a and Δ_7b  are extracted from Delta_vec
        using _IDX_7A_ARR and _IDX_7B_ARR. A small Tikhonov regularisation (lam = 1e-6·I) is added before inversion to handle near-singular V matrices.
        The Γ7a↔Γ7b cross term, which would arise from two-body Γ6-mediated processes not included in the current mean-field decoupling, is omitted
        consistently with the block-diagonal pairing structure of the BdG Hamiltonian.

        Sign convention
        ---------------
        - V > 0: attractive → positive term; finite Δ incurs an energy cost.
        - V ≤ 0: repulsive/absent → Δ = 0; contribution is negligible.
        """
        # BdG diagonalization with 4-channel Hamiltonian
        vbdg = self._get_vbdg()
        M_channels = np.asarray(M_channels, dtype=float)
        _J_A1g_diag, _, _ = self.p.exchange_channels(Q, g_J)
        _J_q_afm = self.p._exchange_J_q(_J_A1g_diag, np.pi, np.pi)

        ev_all, ec_all = np.linalg.eigh(
            vbdg._build_H_stack(self.k_points, M_channels, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, out=vbdg._H_stack, Q_Eg2=Q_Eg2)
        )
        
        E_mf_correction = 0.5 * float(np.dot(_J_q_afm, M_channels**2))

        _arg = np.clip(np.abs(ev_all) / self.kT, 0.0, _FERMI_ARG_CLIP)
        Omega_kn = np.minimum(0.0, ev_all) - self.kT * np.log1p(np.exp(-_arg))
        Omega_cell = np.einsum('k,kn->', self.k_weights, Omega_kn)
        Omega_trace = np.einsum('k,k->', self.k_weights, np.real(np.trace(vbdg._H_stack[:, 0:12, 0:12], axis1=1, axis2=2)))

        elastic_energy = 0.5 * K_eff_for_free_energy * Q**2
        if Q_Eg2 != 0.0 or K_eff_Eg2_for_free_energy != 0.0:
            elastic_energy += 0.5 * K_eff_Eg2_for_free_energy * Q_Eg2**2
        
        condensation = _sector_condensation_energy(Delta_vec[_IDX_7A_ARR], g_t, g_Delta_s, g_Delta_ad, V_JT_corr, vertex_cache) + _sector_condensation_energy(Delta_vec[_IDX_7B_ARR], g_t, g_Delta_s, g_Delta_bd, V_JT_corr, vertex_cache)
        F_total = 0.25 * (Omega_cell+Omega_trace) + elastic_energy + condensation + E_mf_correction
        return (F_total, float(condensation)) if return_cond else F_total

    def _canonical_free_energy(self, target_doping: float, M_channels: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu_guess: float, g_t: float, g_J: float, V_JT_corr: float, Q_Eg2: float, vertex_cache: dict = None, return_cond: bool = False):
        """
        Canonical-ensemble (fixed particle number) free energy F = Ω + μN at a probed point (M_channels, Q, Delta_vec):
        refits μ at the Q-consistent Fermi-velocity scale, evaluates the BdG grand potential at the refit point, and Legendre-transforms it.
        return_cond=True returns (F, E_cond) — the condensation-energy part of the SAME evaluation (used to split K_eff into its parts).
        """
        mu_opt, n_opt = self._find_mu_for_density(M_channels, Q, Delta_vec, F67_vec, target_doping, mu_guess, g_t, g_J)
        g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M_channels, Q, n_opt, mu_opt, g_t, g_J)
        Omega = self._compute_bdg_free_energy(M_channels, Q, Delta_vec, F67_vec, n_opt, mu_opt, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd, V_JT_corr, Q_Eg2, self.p.K_lattice, self.p.K_lattice_Eg2, vertex_cache, return_cond)
        if return_cond:
            return Omega[0] + mu_opt * n_opt, Omega[1]
        return Omega + mu_opt * n_opt

    def compute_dF_dM_channels_and_hessian(self, target_doping: float, M_channels: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, V_JT_corr: float, Q_Eg2: float, vertex_cache: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        3-component gradient (∂F/∂M_Γ6, ∂F/∂M_Γ7a, ∂F/∂M_Γ7b) and 3×3 Hessian of the
        canonical free energy  F(M, Q) = F_bulk(M, Q) + ΔF_cluster(M, Q) − ΔF_cluster(0, Q)
        w.r.t. the channel-resolved staggered magnetizations M_c.

        """
        M0 = np.asarray(M_channels, dtype=float).copy()
        eps = np.maximum(1e-4, np.abs(M0) * 1e-3)

        _dFc_ref = self._dFc_reference(target_doping, mu, g_t, g_J)

        def F_eval(Mc: np.ndarray) -> float:
            """Total canonical free energy at fixed density (μ is profiled out internally)."""
            F_bulk = self._canonical_free_energy(target_doping, Mc, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, V_JT_corr, Q_Eg2, vertex_cache)
            dFc = self._dF_cluster(Mc, Q, target_doping, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J)
            return F_bulk + (dFc - _dFc_ref)

        _, grad, Hess = _fd_grad_hessian(F_eval, M0, eps)
        return grad, Hess

    def _build_H_cluster(self, M_vec, Q_val, g_J, Z_val, g_eff, beta_cluster, mu_v, F67_vec, include_exchange, include_TRW):
        """
        Cluster BdG Hamiltonian on the 2x2 plaquette. All inputs explicit — no closures.

        Parameters
        ----------
        M_vec         : (3,) per-channel staggered magnetization (only M_vec[0]=Γ6 enters)
        Q_val         : float  B1g distortion (Å)
        g_J           : float  Gutzwiller exchange factor
        Z_val         : float  coordination used for the Weiss embedding
        g_eff         : float  = beta_cluster · g_JT_bare (effective JT coupling)
        beta_cluster  : float  thermal-FS-averaged β_k
        mu_v          : float  chemical potential, pre-solved for (M_vec, Q_val) at fixed density
        F67_vec       : (_N_DELTA,) anomalous Γ6–Γ7 amplitudes

        TRW scope (documented limitation)
        ---------------------------------
        Only the on-site (s) channel contributes, since the plaquette embedding treats each site's Weiss field independently:
        Hs -= sign · β_cluster·Z·J_B1g·F67_s·B1g_offdiag. NN (es, d, px, py) and NNN (g, px2, py2) TRW would be **bond** operators
        and require `_embed2` with the corresponding Fourier coefficient c_δ of the form factor φ_i(k) = Σ_δ c_δ e^{ik·δ}.
        """
        F67a_s = F67_vec[_DELTA_IDX['s7a']]
        F67b_s = F67_vec[_DELTA_IDX['s7b']]

        _S_ch = []
        for _c in range(3):
            _P = np.zeros((6, 6))
            _P[2*_c:2*_c+2, 2*_c:2*_c+2] = np.eye(2)
            _S_ch.append(_P @ np.asarray(self.S_axis_op) @ _P)

        H = np.zeros((_N_CLUSTER, _N_CLUSTER), dtype=complex)
        H_JT = (g_eff * Q_val * self.p.B1g_op).astype(complex)
        J_A1g_diag, J_B1g_bare, J_B1g_bare_7b = self.p.exchange_channels(Q_val, g_J)

        if include_TRW:
            _w_trw = np.zeros((3, 3), dtype=complex)
            _w_trw[0, 1], _w_trw[1, 0] = J_B1g_bare    * F67a_s, J_B1g_bare    * np.conj(F67a_s)
            _w_trw[0, 2], _w_trw[2, 0] = J_B1g_bare_7b * F67b_s, J_B1g_bare_7b * np.conj(F67b_s)
            H_TRW = (beta_cluster * Z_val * self.p.B1g_offdiag
                    * np.kron(_w_trw, np.ones((2, 2), dtype=complex))).astype(complex)
        else:
            H_TRW = None

        for site, sign in enumerate(_SIGN_M):
            Hs = self.build_local_hamiltonian_for_bdg(sign, M_vec, J_A1g_diag, mu_v, Z_val)
            Hs = Hs + H_JT
            if include_TRW:
                Hs = Hs - sign * H_TRW
            H += self._embed1(0.5 * (Hs + Hs.conj().T), site)

        if include_exchange:
            for i, j, eta in _BONDS:
                _J_ex = sum(J_A1g_diag[_c] * self._embed2(_S_ch[_c], i, _S_ch[_c], j)
                            for _c in range(3))
                H += (_J_ex + eta * J_B1g_bare * self._embed2(self.p.B1g_op, i, self.p.B1g_op, j))
        return 0.5 * (H + H.conj().T)

    def compute_cluster_free_energy(self, target_doping: float, M_channels: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, tx_bare: float, ty_bare: float, g_t: float, g_J: float, V_JT_eff: float, Q_Eg2: float = 0.0, vertex_cache: Optional[dict] = None) -> Dict:
        """
        Cluster exact diagonalization on a 2×2 plaquette.

        Returns SIX distinct quantities:
        V_irr_QQ       : particle-hole irreducible B1g vertex (→ V_JT_corr)
        Delta_K_cluster: (K_full − K_full_MF)/4 — beyond-MF stiffness correction, where per site
                         K_full   = _Z_EFF = 2 external bonds at MF + n_intra = 2 intra bonds EXACT
                         K_full_MF= the same cluster with ALL _Z = 4 bonds at MF (the lattice coordination)
                         The bulk BdG already treats all Z = 4 bonds at MF; subtracting K_full_MF removes exactly that MF content.
                         Classical-limit check (M_ext = local moment, kT → 0, channel-diagonal exchange): K_full − K_full_MF → 0
        Q_fluct       : RMS ⟨B1g⟩ fluctuation (grows when kT ≳ cluster B1g level spacing)
        b_mean        : per-site ⟨B1g⟩

        K_schur is computed separately in solve_Q_slave via a 2D (Q, |Δ|) Hessian and is capped at the same |K_bare|. It is the curvature along the CURRENT Δ-direction only,
        not the full Δ-space.

            Γ_ED = χ0_ED^{-1} - χ_ED^{-1},

        projected onto the staggered (spin) / uniform (B1g) subspace AFTER the inversion (projection and inversion do not commute).
        Geometry (checkerboard, open plaquette):
            0 --x-- 1
            |       |
            y       y
            |       |
            3 --x-- 2

        Sites 0,2 = sublattice A (sign_M=+1); sites 1,3 = sublattice B (sign_M=-1).
        Bonds: (0,1) x, (1,2) y, (2,3) x, (3,0) y — each carries η=+1 (x) or η=−1 (y) in the B1g channel; the A1g (magnetic) channel is direction-independent.
        Every site has exactly 2 intra-cluster NN bonds and 2 external neighbours (_Z_EFF = Z − n_intra = 4 − 2 = 2).
        """
        M_ext = float(M_channels[0])
        zero_Delta = np.zeros(_N_DELTA, dtype=complex)
        zero_F67 = np.zeros(_N_DELTA, dtype=complex)
        _F67a_s, _F67b_s = F67_vec[_DELTA_IDX['s7a']], F67_vec[_DELTA_IDX['s7b']]

        def _apply_at_site(evecs_tensor: np.ndarray, op: np.ndarray, site: int) -> np.ndarray:
            """Apply a 6×6 operator to one site-leg of the rank-5 eigenvector tensor."""
            t = np.tensordot(op, evecs_tensor, axes=([1], [site]))
            return np.moveaxis(t, 0, site)

        def _expect_from_applied(evecs_tensor: np.ndarray, applied: np.ndarray) -> np.ndarray:
            """Expectation value per eigenstate from a site-applied operator."""
            return np.einsum('ijkln,ijkln->n', evecs_tensor.conj(), applied).real

        # Q-dependent μ (bulk, fixed density)
        mu_state_cache: Dict[float, float] = {}
        def _mu_state_at(q_val):
            key = round(float(q_val), 10)
            if key not in mu_state_cache:
                mu_q, _ = self._find_mu_for_density(M_channels, q_val, Delta_vec, F67_vec, target_doping, mu, g_t, g_J)
                mu_state_cache[key] = float(mu_q)
            return mu_state_cache[key]

        def _mu_state_at_normal(q_val):
            key = ("norm", round(float(q_val), 10))
            if key not in mu_state_cache:
                mu_q, _ = self._find_mu_for_density(M_channels, q_val, zero_Delta, zero_F67, target_doping, mu, g_t, g_J)
                mu_state_cache[key] = float(mu_q)
            return mu_state_cache[key]

        def _cluster_stiffness(include_exchange: bool, Z_val: float, need_evecs: bool = False):
            """Cluster mechanical stiffness using a 5-point second derivative."""
            h_Q = max(np.sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2), 1e-4)
            M_vec = np.array([M_ext, 0.0, 0.0], dtype=float)
            spectra = []
            for i in (-2, -1, 0, 1, 2):
                Qv = Q + i * h_Q
                mu_v = _mu_state_at(Qv)
                H = self._build_H_cluster(M_vec, Qv, g_J, Z_val, g_eff, beta_cluster, mu_v, F67_vec, include_exchange, include_TRW=True)
                if i == 0 and need_evecs:
                    evals, evecs = np.linalg.eigh(H)
                else:
                    evals, evecs = np.linalg.eigvalsh(H), None
                spectra.append((evals, evecs))

            F = []
            for k_idx, (e, _) in enumerate(spectra):
                e = np.asarray(e, dtype=float)
                E0 = float(e[0])
                w = np.exp(np.clip(-(e - E0) / self.kT, -700.0, 0.0))
                q_k = Q + (k_idx - 2) * h_Q
                J_val = float(self.p.exchange_channels(q_k, g_J)[0][0])
                F_dc = _CLUSTER_SIZE * 0.5 * Z_val * J_val * (M_ext ** 2)
                F.append(E0 - self.kT * np.log(max(float(w.sum()), _MATH_EPS)) + F_dc)
            
            F = np.asarray(F, dtype=float)
            K = np.dot([-1, 16, -30, 16, -1], F) / (12.0 * h_Q**2)
            return float(K), spectra[2][0], spectra[2][1], float(F[2])

        def _site_channel_susceptibility_tensor(include_exchange: bool, Z_val: float, Q_ref: float = 0.0) -> np.ndarray:
            M_vec_zero = np.array([0.0, 0.0, 0.0], dtype=float)
            mu_v = _mu_state_at_normal(Q_ref)
            H = self._build_H_cluster(M_vec_zero, Q_ref, g_J, Z_val, g_eff, beta_cluster, mu_v, zero_F67, include_exchange, include_TRW=False)
            try:
                evals, evecs = np.linalg.eigh(H)
            except np.linalg.LinAlgError:
                try:
                    evals, evecs = _scipy_eigh(H, driver='evr')
                except np.linalg.LinAlgError:
                    evals, evecs = _scipy_eigh(H, driver='ev')
            
            eta = max(_ETA_T_FRAC * self.kT, _ETA_GRID_FLOOR * self.p.t0)
            beta = 1.0 / self.kT
            E = np.asarray(evals, dtype=float) - float(evals[0])
            p = np.exp(np.clip(-beta * E, -700.0, 0.0))
            p /= max(float(p.sum()), _MATH_EPS)
            
            # Site-averaged local operators in eigenbasis, with disconnected part removed.
            O_eig = []
            for channel in range(2):
                op_local = self.S_axis_op if channel == 0 else self.p.B1g_op      # S_axis_op == diag(sz_op) for M_axis='z'
                for site in range(4):
                    op_e = evecs.conj().T @ (self._embed1(op_local, site) @ evecs)
                    op_avg = float(np.sum(p * np.diag(op_e).real))
                    O_eig.append(op_e - op_avg * np.eye(len(E)))

            dE = E[None, :] - E[:, None]
            dp = p[:, None] - p[None, :]
            dE2 = dE.real**2

            # Static Lehmann kernel: (p_n-p_m)/(E_m-E_n), with beta*p_n at degeneracy.
            kernel = np.empty_like(dE2)
            mask = dE2 > eta**2
            kernel[mask] = dp[mask] / dE[mask]
            kernel[~mask] = beta * 0.5 * (p[:, None] + p[None, :])[~mask]

            chi_tensor = np.zeros((8, 8), dtype=float)
            for k in range(8):
                for l in range(k, 8):
                    val = np.sum(kernel * O_eig[k] * O_eig[l].conj()).real
                    chi_tensor[k, l] = val
                    if k != l:
                        chi_tensor[l, k] = val
            return 0.5 * (chi_tensor + chi_tensor.T)

        def _stable_inverse(A: np.ndarray) -> np.ndarray:
            """Stable symmetric inverse in the full site x channel space. Only numerically null eigenvalues are regularised; the physical susceptibility eigenvalues are not projected away."""
            A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
            w, U = np.linalg.eigh(A)
            floor = 1e-12 * max(float(np.max(np.abs(w))), 1.0)
            w_inv = np.where(np.abs(w) >= floor, 1.0 / w, 0.0)
            return 0.5 * ((U * w_inv) @ U.T + ((U * w_inv) @ U.T).T)

        def _project_vertex_to_staggered(Gamma: np.ndarray) -> np.ndarray:
            """
            Project the full 8x8 irreducible vertex onto:
            - staggered spin channel:  0.5 * (1, -1, 1, -1)
            - uniform B1g channel:     0.5 * (1,  1, 1,  1)
            """
            P = np.zeros((8, 2), dtype=float)
            P[0:4, 0] = 0.5 * np.array([1.0, -1.0, 1.0, -1.0])
            P[4:8, 1] = 0.5
            V = P.T @ Gamma @ P
            return 0.5 * (V + V.T)

        # 1. Thermodynamics & Cluster Stiffness (Physical State)
        beta_k = np.asarray(self.p.wave_function_weight(tx_bare, ty_bare, self.k_points[:, 0], self.k_points[:, 1]), dtype=float)
        vbdg = self._get_vbdg()
        mu_now = _mu_state_at(Q)
        ev_now, _ = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M_channels, Q, zero_Delta, zero_F67, n_kspace, mu_now, g_t, g_J, out=vbdg._H_stack))
        f_now = _fermi_function(ev_now, self.kT)
        w_fermi = np.sum(f_now * (1.0 - f_now) / self.kT, axis=1)
        denom = float(np.sum(self.k_weights * w_fermi))
        beta_cluster = float(np.sum(self.k_weights * w_fermi * beta_k) / denom) if denom >= _MATH_EPS else float(np.mean(beta_k))
        g_eff = beta_cluster * self.p.g_JT

        # Three references, all with the SAME kT-based thermal averaging and 5-point stencil:
        K_full,    evals_full, evecs_full, F_full = _cluster_stiffness(True,   _Z_EFF, need_evecs=True)   # _Z_EFF external MF + exact intra
        K_full_MF, _,          _,          _      = _cluster_stiffness(False, _Z,     need_evecs=False)  # _Z_EFF external MF + intra M
        K_bare,    _,          _,          _      = _cluster_stiffness(False, _Z_EFF,  need_evecs=False)  # _Z_EFF external MF only; no intra at all

        # TRUE beyond-MF: exact intra minus the intra MF content that the bulk BdG already includes.
        Delta_K_cluster = (K_full - K_full_MF) / _CLUSTER_SIZE

        # 2. Irreducible Vertex Extraction (Normal State Reference)
        chi0_tensor     = _site_channel_susceptibility_tensor(include_exchange=False, Z_val=_Z_EFF, Q_ref=0.0)
        chi_full_tensor = _site_channel_susceptibility_tensor(include_exchange=True,  Z_val=_Z_EFF, Q_ref=0.0)
        inv_chi0        = _stable_inverse(chi0_tensor)
        inv_chi_full    = _stable_inverse(chi_full_tensor)
        Gamma_ED        = 0.5 * ((inv_chi0 - inv_chi_full) + (inv_chi0 - inv_chi_full).T)

        # Only now project onto the uniform q=0 [spin, B1g] channel subspace.
        V_irr = _project_vertex_to_staggered(Gamma_ED)
        V_irr_QQ = float(V_irr[1, 1])
        V_JT_corr = V_JT_eff + V_irr_QQ
        
        # The 1D Q-stencil (K_bulk, K_econd) and the 2D (Q,|Δ|) Schur-Hessian evaluate the same points: F(Q, Δ), F(Q±ε_Q, Δ).
        Delta_total   = float(np.sum(np.abs(Delta_vec)))
        _delta_active = Delta_total > _QQ_DELTA_THRESH
        if _delta_active:
            _abs = np.abs(Delta_vec)
            Delta_frac = _abs / Delta_total
            phase = np.where(_abs > 1e-12, Delta_vec / np.where(_abs > 1e-12, _abs, 1.0), 1.0 + 0j)

        _dF_can_cache: Dict[Tuple[float, float], Tuple[float, float]] = {}

        def _F_can(qv: float, dv: float) -> Tuple[float, float]:
            """(F, E_cond) at (Q=qv, |Δ|=dv), proportional rescaling of all _N_DELTA
            channels when Δ is non-negligible, otherwise the exact Delta_vec."""
            key = (round(float(qv), 12), round(float(dv), 12))
            if key not in _dF_can_cache:
                d_vec = phase * abs(dv) * Delta_frac if _delta_active else Delta_vec
                _dF_can_cache[key] = self._canonical_free_energy(target_doping, M_channels, qv, d_vec, F67_vec, n_kspace, mu, g_t, g_J, V_JT_corr, Q_Eg2, vertex_cache, return_cond=True)
            return _dF_can_cache[key]

        eps_Q, eps_Q2 = _q_fd_step(Q)

        (F_p2, E_p2) = _F_can(Q + 2.0 * eps_Q, Delta_total)
        (F_p1, E_p1) = _F_can(Q +       eps_Q, Delta_total)
        (F_0,  E_0 ) = _F_can(Q,               Delta_total)
        (F_m1, E_m1) = _F_can(Q -       eps_Q, Delta_total)
        (F_m2, E_m2) = _F_can(Q - 2.0 * eps_Q, Delta_total)

        # K_econd comes from the frozen-vertex E_cond(Q) is visible and not mistaken for a response of the BdG ground state; K_bulk is the curvature of the code's own F_bdg (what the Q-slave minimises)
        K_bulk  = (-F_p2 + 16.0 * F_p1 - 30.0 * F_0 + 16.0 * F_m1 - F_m2) / (12.0 * eps_Q2)
        K_econd = (-E_p2 + 16.0 * E_p1 - 30.0 * E_0 + 16.0 * E_m1 - E_m2) / (12.0 * eps_Q2)

        if _delta_active:
            eps_D = max(1e-6, min(Delta_total * 5e-4, 0.05))
            _, _, H_2d = _fd_grad_hessian(lambda x2: _F_can(x2[0], x2[1])[0], np.array([Q, Delta_total]), np.array([eps_Q, eps_D]), F0=F_0, need_grad=False)
            H_QD, H_DD = H_2d[0, 1], H_2d[1, 1]
            if H_DD > _SCHUR_HDD_FLOOR:
                K_schur = min(H_QD**2 / H_DD, abs(self._K_bare))
            elif H_DD < -_SCHUR_HDD_FLOOR:
                K_schur = 0.0
                _scf_log("CLUSTER-WARN", f"  ⚠ K_schur=0: H_DD={H_DD:+.3e} < 0 (indefinite Δ curvature — F_can is NON-MONOTONE in Δ; the SC state is not a local min along Δ)")
            else:
                K_schur = 0.0
        else:
            K_schur = 0.0

        # 3. Per-site B1g expectation values and fluctuation amplitude
        evecs_tensor = evecs_full.reshape(_N_ORB, _N_ORB, _N_ORB, _N_ORB, evecs_full.shape[1])
        B_applied = [_apply_at_site(evecs_tensor, self.p.B1g_op, s) for s in range(4)]
        b_s = [_expect_from_applied(evecs_tensor, B_applied[s]) for s in range(4)]
        
        _bweights = np.exp(np.clip(-(evals_full - evals_full[0]) / self.kT, -700.0, 0.0))
        p_full = _bweights / max(float(_bweights.sum()), _MATH_EPS)
        b_mean = np.array([np.sum(p_full * b_s[s]) for s in range(4)])
        b2_diag = [_expect_from_applied(evecs_tensor, _apply_at_site(evecs_tensor, self.p.B1g_op @ self.p.B1g_op, s)) for s in range(4)]
        b2_mean = np.array([np.sum(p_full * b2_diag[s]) for s in range(4)])
        Q_fluct = float(np.sqrt(max(0.0, np.mean(b2_mean - b_mean**2))))

        # 4. Projected susceptibilities are retained only for diagnostics.
        P = np.zeros((8, 2), dtype=float)
        P[0:4, 0] = 0.5 * np.array([1.0, -1.0, 1.0, -1.0])
        P[4:8, 1] = 0.5
        chi0_loc = 0.5 * ((P.T @ chi0_tensor @ P) + (P.T @ chi0_tensor @ P).T)
        chi_loc_full = 0.5 * ((P.T @ chi_full_tensor @ P) + (P.T @ chi_full_tensor @ P).T)

        return {
            'b_mean':          b_mean,
            'Q_fluct':         Q_fluct,
            'V_irr_QQ':        V_irr_QQ,
            'V_JT_corr':       V_JT_corr,
            'chi0_loc':        chi0_loc,
            'chi_loc_full':    chi_loc_full,
            'Delta_K_cluster': Delta_K_cluster,
            'K_eff':           K_bulk + Delta_K_cluster - K_schur,
            'K_bulk':          K_bulk,
            'K_econd':         K_econd,
            'K_schur':         K_schur,
            'K_full':          K_full,
            'K_bare':          K_bare,
            'K_full_MF':       K_full_MF,
            'F_can':           float(F_0),
            'F_per_site':      float(F_full / _CLUSTER_SIZE),
        }

    def _bdg_b1g_per_sublattice(self, ev: np.ndarray, ec: np.ndarray) -> Tuple[float, float]:
        """Mean-field BdG <B1g> per sublattice at the given (already-diagonalized) spectrum."""
        fn = _fermi_function(ev, self.kT)
        fbar = 1.0 - fn
        uA, uB, vA, vB = _get_nambu_spinors(ec)

        def _sandwich(u: np.ndarray) -> np.ndarray:
            return np.real(np.einsum('kan,ab,kbn->kn', u.conj(), self.B1g_op, u))

        bA_per_k = np.sum(_sandwich(uA) * fn + _sandwich(vA) * fbar, axis=1)
        bB_per_k = np.sum(_sandwich(uB) * fn + _sandwich(vB) * fbar, axis=1)
        bA_bdg = float(np.dot(self.k_weights, bA_per_k)) / 2.0
        bB_bdg = float(np.dot(self.k_weights, bB_per_k)) / 2.0
        return bA_bdg, bB_bdg

    def refine_M_state(self, target_doping: float, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, damping: float = 0.45, anderson_m: int = 5, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float, float]:
        """
        Converge the channel-resolved staggered magnetization M and the chemical potential mu self-consistently at fixed (Q, Delta_s, Delta_d).
        anderson_m : Anderson history depth.  5 is the standard choice.
        """
        # --- Initial guesses ---
        g_t, g_J = self.p.get_gutzwiller_factors(target_doping)
        n_kspace = 1.0 - target_doping
        mu = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        vbdg = self._get_vbdg()
        M = np.full(_N_CHANNELS, 0.01, dtype=float)
        M_out = M.copy()

        x_hist: list = []
        f_hist: list = []
        it = 0
        
        for it in range(_MAX_ITER_FOR_REFINE_M):
            M_in = M.copy()
            mu, n_kspace = self._find_mu_for_density(M_in, Q, Delta_vec, F67_vec, target_doping, mu, g_t, g_J)
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M_in, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, out=vbdg._H_stack))

            g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M_in, Q, n_kspace, mu, g_t, g_J)
            F67_vec, _ = _compute_F67_pair_amplitudes(self.k_weights, self.Phi_irrep_k, ev, ec, self.kT, Delta_vec, g_Delta_s, g_Delta_ad, g_Delta_bd)

            M_out = vbdg.compute_channel_staggered_magnetizations(Q, Delta_vec, mu, ev, ec)
            x_hist.append(M_in.copy())
            f_hist.append(M_out.copy())
            
            if len(x_hist) > anderson_m + 1:
                x_hist.pop(0)
                f_hist.pop(0)
                M_new = self._anderson_mix(x_hist, f_hist, m=anderson_m, alpha=damping)
            else:
                M_new = (1.0 - damping) * M_in + damping * M_out

            if not np.all(np.isfinite(M_new)):
                M_new = (1.0 - damping) * M_in + damping * M_out
            M = np.clip(M_new, 0.0, 1.0)
        
        M_out = M_out.copy()
        M_out[np.abs(M_out) < 1e-12] = 0.0
        mu, n_kspace = self._find_mu_for_density(M_out, Q, Delta_vec, F67_vec, target_doping, mu, g_t, g_J)
        ev_p, ec_p = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M_out, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, out=vbdg._H_stack))

        g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M_out, Q, n_kspace, mu, g_t, g_J)
        F67_vec, _ = _compute_F67_pair_amplitudes(self.k_weights, self.Phi_irrep_k, ev_p, ec_p, self.kT, Delta_vec, g_Delta_s, g_Delta_ad, g_Delta_bd)

        if verbose:
            M_bdg_check = vbdg.compute_channel_staggered_magnetizations(Q, Delta_vec, mu, ev_p, ec_p)
            final_resid = float(np.linalg.norm(M_out - M_bdg_check))
            dens_err = abs(n_kspace - (1.0 - target_doping))
            _scf_log("REFINE-M",
                    f"|M-M_BdG|={final_resid:.3e}  "
                    f"|Δn|={dens_err:.3e}  "
                    f"M_out={np.array2string(M_out, precision=6)}")
        return (M_out, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd)

    def _scf_kick(self, target_doping: float, seed_complex_phase: Union[float, Dict[str, float]], Delta_vec: np.ndarray = np.zeros(_N_DELTA, dtype=complex), F67_vec: np.ndarray = np.zeros(_N_DELTA, dtype=complex), Q_Eg2: float = 0.0, verbose: bool = False) -> Dict:
        vbdg = self._get_vbdg()
        initial_Q = 0.0
        M_kick, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd = self.refine_M_state(target_doping, initial_Q, Delta_vec, F67_vec, damping = 0.45, verbose=verbose)

        # --- Anisotropic hopping ---
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(initial_Q)
        t_eff = g_t * np.sqrt(0.5 * (tx_bare**2 + ty_bare**2))
        
        Gamma_M, F_cluster, V_JT_eff, V_JT_corr, V_cap = self._make_vertex_params(target_doping, M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J)

        # --- Linearised BdG+RPA eigenproblem ---
        J_A1g_diag, J_B1g_bare, J_B1g_bare_7b = self.p.exchange_channels(initial_Q, g_J)
        vertex_cache = self.compute_pairing_kernel_and_build_cache(M_kick, initial_Q, n_kspace, mu, g_t, g_J, J_A1g_diag, Gamma_M, V_JT_eff, V_JT_corr, V_cap)

        lambda_lin_max = float(vertex_cache['lambda_lin_max'])
        lambda_excess = max(0.0, lambda_lin_max - 1.0) / max(lambda_lin_max, 1e-12)

        # --- Early Hessian in the seed neighborhood ---
        frac = np.tile(vertex_cache['frac_seed_signed'], 2)
        hk_early = self.compute_hessian(target_doping, M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, V_JT_corr, Q_Eg2, vertex_cache)
        lambda_min = hk_early['lambda_min']

        ev_n, ec_n = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, out=vbdg._H_stack))
        # --- Delta seeding ---
        if lambda_min < 0.0:
            # physical_dir basis: [M_Γ6(0), M_Γ7a(1), M_Γ7b(2), Q(3), Δ(4)]
            step = self._project_kick_from_hessian(hk_early, (_EARLY_KICK_BASE + lambda_excess) * min(1.0, abs(lambda_min)), sign_ref=initial_Q)
            M_kick = np.clip(M_kick + step[0:3], _KICK_M_CLIP_LO, _KICK_M_CLIP_HI)
            new_Delta_total = complex(np.clip(t_eff * 0.5 / np.sinh(1.0 / max(lambda_lin_max, 0.1)), _DELTA_ABS_FLOOR, _KICK_DELTA_MAX_FRAC * t_eff))
            
            _abs_D = np.abs(Delta_vec_kick)
            _phase = np.divide(Delta_vec_kick, _abs_D, out=np.ones(_N_DELTA, dtype=complex), where=_abs_D > _MATH_EPS)
            Delta_vec_kick = new_Delta_total * frac * _phase
        else:
            _rpa_afm_seed = self._rpa_closure(M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, J_A1g_diag, V_JT_corr, (ev_n, ec_n), apply_diamagnetic_QQ=True)
            reduction = _KICK_REDUCTION_AMP * np.maximum(0.0, (M_kick - _KICK_M_EXCESS_CTR)) * max(0.0, (_rpa_afm_seed['stoner'] - _KICK_JCHI_EXCESS_CTR)) * lambda_excess
            M_kick = M_kick * (1.0 - reduction)
            Delta_vec_kick = complex(np.clip(t_eff * 0.5 / np.sinh(1.0 / max(lambda_lin_max, 0.1)), _DELTA_ABS_FLOOR, _KICK_DELTA_MAX_FRAC * t_eff)) * frac
        
        spec = {_DELTA_IDX[f'{n}7{b}']: float(p) for n, p in seed_complex_phase.items() for b in 'ab'}
        idx = list(spec)
        Delta_vec_kick[idx] = np.abs(Delta_vec_kick[idx]) * np.exp(1j * np.fromiter(spec.values(), dtype=float))
        F67_vec_kick, _ = _compute_F67_pair_amplitudes(self.k_weights, self.Phi_irrep_k, ev_n, ec_n, self.kT, Delta_vec_kick, g_Delta_s, g_Delta_ad, g_Delta_bd)
        ev_n, ec_n = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, out=vbdg._H_stack))
        F67_vec_kick, _ = _compute_F67_pair_amplitudes(self.k_weights, self.Phi_irrep_k, ev_n, ec_n, self.kT, Delta_vec_kick, g_Delta_s, g_Delta_ad, g_Delta_bd)
        mu, n_kspace = self._find_mu_for_density(M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, target_doping, mu, g_t, g_J)
        g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M_kick, initial_Q, n_kspace, mu, g_t, g_J)
        ev, ec = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, out=vbdg._H_stack))
        chi_QQ = self._rpa_closure(M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M, J_A1g_diag, V_JT_corr, (ev, ec), apply_diamagnetic_QQ=True)['chi_QQ']
        Gamma_M, _, V_JT_eff, V_JT_corr, V_cap = self._make_vertex_params(target_doping, M_kick, initial_Q, Delta_vec_kick, F67_vec_kick, n_kspace, mu, g_t, g_J, chi_QQ, Q_Eg2, (ev, ec), vertex_cache, verbose)
        Delta_vec_kick, _, vertex_cache = vbdg.compute_gap_eq_vectorized(M_kick, initial_Q, Delta_vec_kick, n_kspace, mu, t_eff, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd, J_A1g_diag, Gamma_M, V_JT_eff, V_JT_corr, V_cap, 0.0, None, ev, ec, vertex_cache, verbose=verbose)

        # --- Adaptive mixing ---
        _alpha = max(_KICK_MIXING_FLOOR, _MIXING / (1 + _KICK_MIXING_SCALE * np.log1p(lambda_lin_max)))
        return {
            'n_kspace':       n_kspace,
            'mu':             mu,
            'M_kick':         M_kick,
            'Delta_vec':      Delta_vec_kick,
            'F67_vec':        F67_vec_kick,
            'alpha':          _alpha,
            'lambda_lin_max': lambda_lin_max,
            'V_cap':          V_cap,
            'Gamma_M':        Gamma_M,
            'chi_QQ':         chi_QQ,
        }

    def _vertex_matrix_at_Q(self, M: np.ndarray, Qv: float, n_kspace: float, mu: float, g_t: float, g_J: float, Gamma_M: float, V_JT_eff: float, V_JT_corr: float, V_cap: float) -> Tuple[float, np.array]:
        """
        Evaluate the linearised pairing eigenvalue and diagonal RPA vertex at a trial JT distortion Qv.
        Used by the ∂λ/∂Q finite-difference scan (SC-triggered JT consistency check).
        Builds the full N_FS × N_FS RPA vertex matrix at Qv from the normal-state (Δ=0) susceptibilities

        Returns
        -------
        lambda_lin_max : float  — largest eigenvalue of the linearised gap kernel at Qv.
        V_diag         : ndarray — diagonal of the N_FS × N_FS vertex matrix (for diagnostics).
        """
        J_A1g_diag, _, _ = self.p.exchange_channels(Qv, g_J)
        lin = self.compute_pairing_kernel_and_build_cache(M, Qv, n_kspace, mu, g_t, g_J, J_A1g_diag, Gamma_M, V_JT_eff, V_JT_corr, V_cap)
        return float(lin['lambda_lin_max']), np.asarray(lin['V_self_diag'], dtype=float).copy()

    def _classify_scf_dynamics(self, delta_history: list) -> dict:
        """
        Classify the SCF |Δ| trajectory.

        Regimes: 'converging' | 'limit_cycle' | 'first_order_jump' | 'hysteretic'
          limit_cycle      : rel_std > threshold AND max/min < 2
          first_order_jump : max/min > 2 AND last value near max (jumped up, held)
          hysteretic       : max/min > 2 AND last value flips between high/low
          converging       : none of the above.
        """
        if len(delta_history) < _CYCLE_WINDOW:
            return {'in_cycle': False, 'regime': 'converging', 'rel_std': 0.0, 'jump_ratio': 1.0}
        arr  = np.array(delta_history[-_CYCLE_WINDOW:], dtype=float)
        mean = float(np.mean(arr))
        if mean < 1e-10:
            return {'in_cycle': False, 'regime': 'converging', 'rel_std': 0.0, 'jump_ratio': 1.0}
        std      = float(np.std(arr))
        rel_std  = std / mean
        arr_max  = float(np.max(arr))
        arr_min  = float(np.min(arr))
        jump_ratio = arr_max / max(arr_min, 1e-12)

        if rel_std <= _CYCLE_THRESHOLD:
            return {'in_cycle': False, 'regime': 'converging', 'rel_std': rel_std, 'jump_ratio': jump_ratio}

        # Oscillation detected — classify sub-type.
        if jump_ratio > 2.0:
            # Large amplitude: first-order-like behaviour; check if last value is near the high end (jumped and held) or alternating.
            _half = _CYCLE_WINDOW // 2
            _early_mean = float(np.mean(arr[:_half]))
            _late_mean  = float(np.mean(arr[_half:]))
            alternating = abs(_late_mean - _early_mean) < 0.3 * std * _half**0.5
            regime = 'hysteretic' if alternating else 'first_order_jump'
        else:
            regime = 'limit_cycle'

        return {'in_cycle': True, 'regime': regime, 'rel_std': rel_std, 'jump_ratio': jump_ratio}

    def _classify_q_dynamics(self, q_hist: list) -> dict:
        if len(q_hist) < _Q_CYCLE_WINDOW:
            return {'in_cycle': False, 'zigzag': 0.0, 'amp': 0.0}
        arr = np.asarray(q_hist[-_Q_CYCLE_WINDOW:], dtype=float)
        d = np.diff(arr)
        amp = float(np.max(np.abs(d)))
        big = np.abs(d) > _Q_CYCLE_AMP_MIN
        rev = (d[1:] * d[:-1] < 0.0) & big[1:] & big[:-1]
        zigzag = float(np.sum(rev)) / max(len(d) - 1, 1)
        return {'in_cycle': bool(zigzag >= _Q_CYCLE_ZIGZAG and amp > _Q_CYCLE_AMP_MIN), 'zigzag': zigzag, 'amp': amp}

    def solve_self_consistent(self, target_doping: float, seed_complex_phase: Union[float, Dict[str, float]], initial_Delta: np.ndarray, initial_F67: np.ndarray, Q_Eg2: float = 0.0, force_delta_zero=False, force_Q_zero=False, max_iter: int = _MAX_ITER, verbose: bool = False) -> Dict:
        """
        Coupled (M, Q, Δ_s, Δ_d, μ) SCF via Anderson(5)-accelerated fixed-point + LM Newton.

        Order parameters
        ----------------
        M       : staggered AFM magnetisation (Gutzwiller-renormalised BdG Weiss field)
        Q       : B₁g JT lattice distortion (Å); zero in the AFM normal state by symmetry
        Δ_vec   : pairing amplitudes by symmetry channel
        μ       : chemical potential enforcing ⟨n⟩ = 1 − δ (Newton + Brentq fallback)

        Per-iteration algorithm:
          1. BdG diagonalisation H(k; M,Q,Δ,μ) → observables (M_bdg, ⟨τ_x⟩, ⟨B₁g⟩, pairs).
          2. SC+JT active (Δ>0, Q>0): inject F67s (Gorkov Γ₆–Γ₇ singlet amplitude) into J_B1g off-diagonal Weiss field, rebuild BdG cache.
          3. Gap equation Δ_out = g_Δ·V(q)·F_AA/AB via RPA vertex (always from Δ=0 χ₀). After fixed-point gap update, blend in the N_IRREP×N_IRREP
             pairing kernel eigenvector direction to ensure the SCF can always find a slope toward the dominant instability channel even when |Δ|≈0.
             The N_IRREP×N_IRREP kernel K_pair is built from V_irrep[i,j] projections over the extended basis; its eigenvector defines the optimal Δ*
             ratio that is mixed in with weight _ALPHA_MIX_2X2, preventing artificial channel locking.
          4. Cluster-ED (2×2 plaquette) mechanical stiffness correction and update K_eff: bare-electronic + AFM-MF Hessian in Q.
          5. Newton step for M (LM-damped, trust-region); blended with the *linearly*-mixed BdG fixpoint M_bdg.
          6. Q update (LOCAL window of half-width Q_trust around the current Q + bracketed root-find + curvature check in solve_Q_slave; the window keeps the choice between the two
             nearly degenerate ±Q wells of a B1g double well branch-continuous and never extrapolates the V_irrep cache — frozen at Q_in — far away from Q_in):
             The scan picks the correct basin; inside it brentq roots the Hellmann–Feynman force
                F_Q = ⟨∂H/∂Q⟩ + ½·Z_afm·Σ_c M_c²·∂J_c/∂Q + K_lattice·Q + ∂E_cond/∂Q,
             with the cached-V_irrep perturbation for the condensation term. Q_out is the raw profiler output.
          7. Anderson(5) mix on (Q, |Δ_s|, |Δ_d|) μ enforcing density is then re-solved at the freshly mixed (M,Q,Δ) μ is exactly profiled out as a slave
             variable at every accepted (M,Q,Δ) point rather than being an extra, imperfectly converged "fast" degree of freedom whose residual error could
             otherwise leak into the (Q,Δ_s,Δ_d) Anderson history from one iteration to the next.
          8. Adaptive α via Λ_inst = EMA[max(λ_pair, λ_JT, J·χ_SS)]: α_eff = α₀/(1+Λ); halved + history reset on divergence; past-QCP (det<0): exponential
             penalty on α; near-QCP: hard cap. FS-resolved ∂λ/∂Q computed at post-convergence to confirm SC-triggered JT consistency on hot-spot level.
             Limit-cycle detector: if |Δ| oscillates with relative std > _CYCLE_THRESHOLD, α is reduced by _CYCLE_DAMP_FAC and the Anderson history is reset to escape the cycle.
             Q has its own detector (signed Q increments that keep reversing): α_Q and the Q-window are halved, α is damped.

        Converged when max(|ΔM|,|ΔQ|,|ΔΔ_s|,|ΔΔ_d|)<tol and |n−(1−δ)|<10·tol.
        Post-convergence: 3×3 Hessian, linearised λ_max/gap symmetry, χ₀(q_AFM), χ_τ/λ_JT, Λ_inst.
        """
        converged = False
        kick = self._scf_kick(target_doping, seed_complex_phase, initial_Delta, initial_F67, Q_Eg2, verbose)

        M = kick['M_kick']
        _Delta_vec = kick['Delta_vec']
        _F67_vec = kick['F67_vec']
        n_kspace = kick['n_kspace']
        mu = kick['mu']
        _alpha = kick['alpha']
        _lambda_lin_max = kick['lambda_lin_max']
        _Gamma_M = kick['Gamma_M']
        _V_cap = kick['V_cap']
        _chi_QQ_sc_q0 = kick['chi_QQ']
        _chi_SS_sc_pipi = 0.0
        _chi_SQ_sc_pipi = 0.0
        _chi_SQ_sc_q0 = 0.0

        history = {
            'M': [], 'Q': [], 'Delta_total': [], 'n_kspace': [], 'F_cluster': [], 'mu': [], 'mixing': [],
        }
        scf_x_hist: list = []
        scf_f_hist: list = []
        _vertex_cache: Optional[dict] = None
        _max_diff_prev = float('inf')
        _max_diff = float('inf')

        _stagnation_count   = 0                    # consecutive near-stagnation iterations
        _alpha_freeze_count = 0
        _selection_ratio    = 0.0
        _det_afm_sc = _DET_AFM_FLOOR
        _delta_run_hist: list = []
        _q_run_hist: list = []                     # signed Q iterates, for the Q-oscillation detector
        _alpha_Q: float = _ALPHA_Q
        _q_trust_scale: float = 1.0                # adaptive scale of the Q-window half-width
        _q_calm: int = 0                           # iterations since the last detected Q oscillation
        _scf_dynamics_regime: str = 'converging'
        _ansatz_unstable_ever: bool = False
        _solve_state = _SolveState()
        _vbdg = self._get_vbdg()

        if force_Q_zero:
            Q = 0.0
        else:
            Q = _Q_SEED_THR
        
        g_t, g_J = self.p.get_gutzwiller_factors(target_doping)
        mu, n_kspace = self._find_mu_for_density(M, Q, _Delta_vec, _F67_vec, target_doping, mu, g_t, g_J)
        _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(_vbdg._build_H_stack(self.k_points, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, out=_vbdg._H_stack))

        for iteration in range(max_iter):
            if force_delta_zero:
                _Delta_vec = np.zeros(_N_DELTA, dtype=complex)
            
            _iter_t0 = _time.time()
            _had_cache = _vertex_cache is not None

            g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M, Q, n_kspace, mu, g_t, g_J)

            # Bare exchange couplings (single bond, no Z)
            J_A1g_diag, J_B1g_bare, J_B1g_bare_7b = self.p.exchange_channels(Q, g_J)
            # J_eff comes exclusively from the analytic Gutzwiller/Kotliar–Ruckenstein exchange renormalisation.
            # This is J(q_AFM); Z_afm_eff folds in the 2nd/3rd-neighbour (t', t'') exchange J2, J3 so it stays the exact AFM-point value of the same J(q) used q-resolved below.
            _J_eff = _m2_weighted_J(self.p._exchange_J_q(J_A1g_diag, np.pi, np.pi), M)

            tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)
            _t_eff = g_t * np.sqrt(0.5 * (tx_bare**2 + ty_bare**2))
            _Gamma_M, F_cluster, _V_JT_eff, _V_JT_corr, _V_cap = self._make_vertex_params(target_doping, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, _chi_QQ_sc_q0, Q_Eg2, (_bdg_ev_sc, _bdg_ec_sc), _vertex_cache)

            # ── SC-state (Δ≠0) RPA determinant ──────────────
            _rpa_pipi = self._rpa_closure(M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, J_A1g_diag, _V_JT_corr, (_bdg_ev_sc, _bdg_ec_sc), apply_diamagnetic_QQ=True)
            _chi_SS_sc_pipi, _chi_SQ_sc_pipi = _rpa_pipi['chi_SS'], _rpa_pipi['chi_SQ']
            _det_afm_sc = _rpa_pipi['det']
            _rpa_q0_sc = self._rpa_closure(M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, J_A1g_diag, _V_JT_corr, (_bdg_ev_sc, _bdg_ec_sc), apply_diamagnetic_QQ=True)
            _chi_SQ_sc_q0, _chi_QQ_sc_q0 = _rpa_q0_sc['chi_SQ'], _rpa_q0_sc['chi_QQ']
            
            # Update Λ_inst: Rayleigh λ_pair, λ_JT=(g²/K)·χ_QQ, J·χ_SS
            if _vertex_cache is not None:
                _Lambda_raw = float(np.clip(_vertex_cache['lambda_lin_max'], 0.0, 10.0))
                _lambda_lin_max = (1-_EMA_NEW_WEIGHT) * _lambda_lin_max + _EMA_NEW_WEIGHT * _Lambda_raw

            # Update Δ before Anderson mix so M update sees current SC state; Δ Newton step (2×2 analogue of the M Newton step above); its stationary point coincides with the gap equation's fixed point 4D Newton step in the Δ-amplitude subspace.
            if force_delta_zero:
                _delta_step = np.zeros(_N_DELTA, dtype=complex)
            else:
                _grad_delta, _H_delta = self.compute_dF_dDelta_and_d2F(M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd, _V_JT_corr, Q_Eg2, _vertex_cache)

                # Adaptive LM floor, generalized to the 4×4 channel Hessian via its MINIMUM eigenvalue:
                #   λ_min >> 0  (deep SC minimum)         -> μ_LM small  -> near-bare Newton step
                #   λ_min ~  0  (near Tc / RPA QCP)       -> μ_LM = _MU_LM_DELTA -> cautious step
                #   λ_min <  0  (saddle, Δ still growing) -> μ_LM = |λ_min| + _MU_LM_DELTA -> guarantees PD damped H
                _lam_min_D = float(np.linalg.eigvalsh(_H_delta)[0])
                _mu_LM_D   = _lm_damping(_lam_min_D, _MU_LM_DELTA, max(_t_eff, _MATH_EPS))
                _H_damped  = _H_delta + _mu_LM_D * np.eye(_N_DELTA)
                try:
                    _delta_step = -np.linalg.solve(_H_damped, _grad_delta)
                except np.linalg.LinAlgError:
                    _delta_step = -_grad_delta / max(_mu_LM_D, 1e-6)
                _delta_step = np.clip(_delta_step, -_TR_DELTA_STEP_MAX, _TR_DELTA_STEP_MAX)
            
            _abs_D = np.abs(_Delta_vec)
            _phase = np.divide(_Delta_vec, _abs_D, out=np.ones(_N_DELTA, dtype=complex), where=_abs_D > _MATH_EPS)
            # _N_DELTA-D Newton target: every amplitude updated, phases preserved.
            Delta_vec_newton = np.clip(_abs_D + _delta_step, 0.0, 1.0) * _phase

            # Gap equation: V(q) always from Δ=0 χ₀; BdG amplitudes (u,v) from SC state.
            Delta_out, F67_vec_new, _vertex_cache = _vbdg.compute_gap_eq_vectorized(M, Q, _Delta_vec, n_kspace, mu, _t_eff, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd, J_A1g_diag, _Gamma_M, _V_JT_eff, _V_JT_corr, _V_cap, _det_afm_sc, _solve_state, _bdg_ev_sc, _bdg_ec_sc, _vertex_cache, verbose and iteration == 0)
            Delta_vec_mixed = self._blend_update_candidates(_Delta_vec, Delta_vec_newton, Delta_out, alpha=_alpha, beta=_ALPHA_HF_DELTA) if not force_delta_zero else np.zeros(_N_DELTA, dtype=complex)

            # Local Q window: the vertex cache is frozen at the CURRENT Q, so the profile is only trusted around it.  Radius ≥ 5·tol so that a window-limited step can never pass the convergence test.
            if not force_Q_zero:
                Q_out, _qslave_info = self.solve_Q_slave(target_doping, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, _V_JT_corr, _q_trust_scale, vertex_cache=_vertex_cache)
                # Primary 2D Anderson vector: (|Δ_s7a|/t₀, |Δ_d7a|/t₀) — scaled to O(1) for a balanced least-squares solve. The other _N_DELTA - 3 amplitudes (s7b, d7b, es/px/py × 7a/7b) previously converged only through the fixed-point+Newton blend above;
                # They now also get an Anderson step of their own, so that a state dominated by them is not left to the slowest, unaccelerated path and they cannot spoil the conditioning of the primary block.
                Q_mixed = float(np.clip(Q + _alpha_Q * (Q_out - Q), -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))
                if verbose:
                    _mir_txt  = ('n/a' if not np.isfinite(_qslave_info['F_mirror']) else f"{_qslave_info['F_mirror']*1e3:+.4f} meV")
                    _noise_ratio = (abs(_qslave_info['F_mirror']) / max(_qslave_info['dF_window_ueV'] * 1e-6, 1e-12)
                                    if np.isfinite(_qslave_info['F_mirror']) and _qslave_info['mode'] == 'local' and _qslave_info['dF_window_ueV'] > 1e-6 else float('nan'))
                    _noise_flag = ('⚠ ±Q branches degenerate (<10% of window depth): sign of Q* is set by noise/cache drift' if np.isfinite(_noise_ratio) and _noise_ratio < 0.1
                                else '✓ ±Q branches resolved (splitting > window depth)' if np.isfinite(_noise_ratio) and _noise_ratio > 1.0
                                else '~ mixed')
                    _dF_txt = (f"{_qslave_info['dF_window_ueV']:.1f} μeV" if _qslave_info['mode'] == 'local' else 'n/a')
                    _scf_log("Q-SLAVE",
                        f"  Q_in={Q:+.5f} → Q_out={Q_out:+.5f} → Q_next={Q_mixed:+.5f} Å  "
                        f"win=±{0.5 * (_qslave_info['hi_w'] - _qslave_info['lo_w']):.4f}  ΔF_win={_dF_txt}  ({'edge' if _qslave_info['edge'] else 'interior min'})  "
                        f"F(−Q)−F(+Q)={_mir_txt}  |mirror|/window-depth={_noise_ratio:.2f}  {_noise_flag}")
            else:
                Q_out   = 0.0
                Q_mixed = 0.0

            x_in_2d  = np.array([_abs_D[_DELTA_IDX['s7a']] / self.p.t0,
                                _abs_D[_DELTA_IDX['d7a']] / self.p.t0])
            x_out_2d = np.array([abs(Delta_vec_mixed[_DELTA_IDX['s7a']]) / self.p.t0,
                                abs(Delta_vec_mixed[_DELTA_IDX['d7a']]) / self.p.t0])
            scf_x_hist.append(np.concatenate([x_in_2d,  _abs_D[_IDX_ANDERSON_EXT] / self.p.t0]))
            scf_f_hist.append(np.concatenate([x_out_2d, np.abs(Delta_vec_mixed[_IDX_ANDERSON_EXT]) / self.p.t0]))
            x_new_2d = self._anderson_mix([_h[:2] for _h in scf_x_hist], [_h[:2] for _h in scf_f_hist], m=5, alpha=_alpha)

            # Anderson updates; phases kept from the mixed vector.
            _dv_s = Delta_vec_mixed[_DELTA_IDX['s7a']]; _dv_d = Delta_vec_mixed[_DELTA_IDX['d7a']]
            Delta_vec_mixed[_DELTA_IDX['s7a']] = np.clip(x_new_2d[0] * self.p.t0, 0.0, 1.0) * (_dv_s / abs(_dv_s) if abs(_dv_s) > _MATH_EPS else 1.0+0j)
            Delta_vec_mixed[_DELTA_IDX['d7a']] = np.clip(x_new_2d[1] * self.p.t0, 0.0, 1.0) * (_dv_d / abs(_dv_d) if abs(_dv_d) > _MATH_EPS else 1.0+0j)

            # Anderson step for the extended amplitudes
            if float(np.max(scf_f_hist[-1][2:])) > 1e-12 or float(np.max(scf_x_hist[-1][2:])) > 1e-12:
                x_new_ext  = self._anderson_mix([_h[2:] for _h in scf_x_hist], [_h[2:] for _h in scf_f_hist], m=5, alpha=_alpha)
                _mag_ext   = np.clip(x_new_ext * self.p.t0, 0.0, 1.0)
                _dv_ext    = Delta_vec_mixed[_IDX_ANDERSON_EXT]
                _phase_ext = np.divide(_dv_ext, np.abs(_dv_ext), out=np.ones(len(_IDX_ANDERSON_EXT), dtype=complex), where=np.abs(_dv_ext) > _MATH_EPS)
                Delta_vec_mixed[_IDX_ANDERSON_EXT] = _mag_ext * _phase_ext

            # Run history: total |Δ| across ALL _N_DELTA channels (monitors the limit cycle).
            _delta_run_hist.append(float(np.sum(np.abs(Delta_vec_mixed))))
            _q_run_hist.append(Q_mixed)

            # V_s/V_d/K_eff are irrelevant here: the elastic and condensation terms have zero M-derivatives.
            grad_M, Hess_M = self.compute_dF_dM_channels_and_hessian(target_doping, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, _V_JT_corr, Q_Eg2, _vertex_cache)
            
            # Adaptive LM floor, generalized to the 3×3 channel Hessian (Hess_M) via its MINIMUM eigenvalue
            #   λ_min(Hess_M) >> 0  (deep stable minimum) → μ_LM small  → fast, lightly-damped Newton step
            #   λ_min(Hess_M) ≈ 0   (flat / near AFM QCP)  → μ_LM = _MU_LM → cautious step, dominated by 1/μ_LM
            #   λ_min(Hess_M) <  0  (saddle / unstable)    → μ_LM = |λ_min| + _MU_LM → guarantees a PD damped Hessian
            _lam_min_M = float(np.linalg.eigvalsh(Hess_M)[0])
            _near_crit_mu_M = _MU_LM / (1.0 + float(np.sum(np.abs(Delta_vec_mixed))) / (2 * self.p.t0))
            _mu_LM_eff = _lm_damping(_lam_min_M, _MU_LM, _t_eff, near_critical_mu=_near_crit_mu_M)
            
            # 1. Regularization of the LM-damped 3×3 Newton system and J_eff thresholds (against Anderson overshoot)
            _H_damped_M = Hess_M + _mu_LM_eff * np.eye(_N_CHANNELS)
            _j_eff_floor = max(abs(_J_eff), _M_J_EFF_FLOOR_FRAC * _t_eff, 1e-4)

            # 2. Trust-region upper bound: J/t stiffness cut + curvature-based penalty near QCP
            _cap_stiff = _TR_M_STEP_MAX / max(1.0, abs(_J_eff) / (2.0 * max(_t_eff, 1e-6)))
            _cap_curv = 0.5 + 0.5 * (max(_lam_min_M, 0.0) / max(_lam_min_M + _mu_LM_eff, 1e-6))
            _step_upper = float(np.clip(_cap_stiff * _cap_curv, _TR_M_STEP_MIN_FLOOR, _TR_M_STEP_MAX))

            # 3. Enforcing a dynamic step limit between the lower and upper trust-region boundaries (per channel)
            _step_floor = np.maximum(_M_STEP_FLOOR_REL * np.abs(M), _M_STEP_FLOOR_ABS)
            _step_limit = np.clip(max(self.kT, 0.05 * _t_eff) / _j_eff_floor, _step_floor, _step_upper)

            # 4. M update and hybrid mixing (linear BdG fixed point + Newton trajectory), per channel
            try:
                _raw_step_M = np.linalg.solve(_H_damped_M, -grad_M)
            except np.linalg.LinAlgError:
                _raw_step_M = -grad_M / max(_mu_LM_eff, 1e-6)
            M_newton = np.clip(M + np.clip(_raw_step_M, -_step_limit, _step_limit), 0.0, 1.0)
            M_bdg = _vbdg.compute_channel_staggered_magnetizations(Q, _Delta_vec, mu, _bdg_ev_sc, _bdg_ec_sc)
            M_mixed = self._blend_update_candidates(M, M_newton, M_bdg, alpha=_alpha, beta=_ALPHA_HF)

            if len(scf_x_hist) > 1 and (Q * Q_mixed < 0):
                # A genuine Q sign change (rare now: the Q window is local).  The Anderson history and the V_irrep EMA refer to the other branch → reset them.
                scf_x_hist.clear()
                scf_f_hist.clear()
                _solve_state.V_irrep_ema = None   # EMA from old topology is invalid after Q sign flip
                self._chi0_norm_cache = None # χ₀ eigenvectors keyed on Q → must rebuild
            
            # max_diff now monitors all four Δ components + F67 amplitudes + M + Q
            _max_diff = max(
                float(np.max(np.abs(M_bdg - M))),
                float(abs(Q_out - Q)),                                                  # RESIDUAL of the Q profile (a step |Q_mixed-Q| = alpha*residual would declare convergence too early)
                float(np.max(np.abs(np.abs(Delta_out) - np.abs(_Delta_vec)))),
                float(self._relative_phase_change(_Delta_vec, Delta_out)),        # gauge-invariant: sees phase drifts (s+id relative phase) that |Δ| alone misses
                float(np.max(np.abs(np.abs(F67_vec_new) - np.abs(_F67_vec)))),
            )

            _selection_ratio = float(np.clip(
                float(np.sum(np.abs(Delta_vec_mixed))) / max(self.p.Delta_CF, 1e-9), 0.0, 1.0
            )) * abs(_F67_vec[0])

            if iteration >= 5 and iteration % 5 == 0:
                _diverging  = _max_diff > _max_diff_prev * _SCF_DIVERGE_RATIO
                _stagnating = _max_diff > _max_diff_prev * _SCF_STAGNATE_RATIO and not _diverging

                # Unified α_eff = α₀ / (1 + 0.5·Λ): Λ_inst measures pairing instability, the 0.5 prefactor suppresses overdamping
                _alpha_base = float(np.clip(
                    _MIXING / (1.0 + 0.5 * _lambda_lin_max),
                    _MIXING / 6.0,
                    _MIXING,
                ))

                # --- Limit cycle / first-order dynamics detection ---
                _dyn = self._classify_scf_dynamics(_delta_run_hist)
                _dyn_Q = self._classify_scf_dynamics(list(np.abs(_q_run_hist)))
                if _dyn['in_cycle']:
                    _scf_dynamics_regime = _dyn['regime']
                    _alpha = max(_alpha_base * _CYCLE_DAMP_FAC, _MIXING / 16.0)
                    if len(scf_x_hist) > 4:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                    _stagnation_count   = 0
                    _alpha_freeze_count = 0
                    if verbose:
                        _scf_log("LIMIT-CYCLE",
                            f"⚠ SCF dynamics [{_dyn['regime']}] iter={iteration} rel_std={_dyn['rel_std']:.3f}  jump_ratio={_dyn['jump_ratio']:.2f}"
                            f"  → α damped to {_alpha:.5f} {'  [first-order basin: SCF may not converge to single point]' if _dyn['regime'] in ('first_order_jump', 'hysteretic') else ''}")
            
                elif _dyn_Q['in_cycle']:
                    _scf_dynamics_regime = f"Q-{_dyn_Q['regime']}"
                    _alpha = max(_alpha_base * _CYCLE_DAMP_FAC, _MIXING / 16.0)
                    if verbose:
                        _scf_log("LIMIT-CYCLE",
                            f"⚠ Q dynamics [{_dyn_Q['regime']}] iter={iteration} "
                            f"rel_std={_dyn_Q['rel_std']:.3f} jump_ratio={_dyn_Q['jump_ratio']:.2f} → α damped to {_alpha:.5f}")

                # --- Diverging ---
                elif _diverging:
                    _alpha = max(_alpha_base * 0.5, _MIXING / 8.0)
                    scf_x_hist.clear()
                    scf_f_hist.clear()
                    _stagnation_count   = 0
                    _alpha_freeze_count = 0

                # --- Stagnating ---
                elif _stagnating:
                    _stagnation_count   += 1
                    _alpha_freeze_count += 1
                    if _stagnation_count >= 2:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _stagnation_count = 0
                    # Subcritical plateau: boost α slightly if pairing is weak.
                    if _lambda_lin_max < 0.3:
                        _alpha = min(_alpha_base * 1.2, _MIXING)
                    else:
                        _alpha = _alpha_base

                # --- Converging ---
                else:
                    _stagnation_count   = 0
                    _alpha_freeze_count = max(_alpha_freeze_count - 1, 0)
                    if _selection_ratio > 0.05 and abs(Q) > 1e-4:
                        # SC+JT active: mild boost, Λ-capped
                        _alpha = min(_alpha_base * _SCF_ALPHA_CONVG_BOOST, _MIXING * _SCF_ALPHA_CONVG_CAP)
                    elif _selection_ratio > 0.05:
                        _alpha = _alpha_base * _SCF_ALPHA_DECAY
                    else:
                        _alpha = _alpha_base
                    # Q injected this iteration (genuine displacement) → be more conservative
                    if abs(Q_mixed - Q) > _Q_THR_REL * self.p.lambda_hop:
                        _alpha = min(_alpha, _MIXING * _SCF_ALPHA_QMOVE_CAP)

                # --- Alpha freeze recovery ---
                if _alpha_freeze_count >= _SCF_FREEZE_THR and _alpha < _MIXING * _SCF_ALPHA_FREEZE_LO:
                    _alpha = min(_alpha * _SCF_ALPHA_RECOVER, _MIXING * _SCF_ALPHA_FREEZE_HI)
                    _alpha_freeze_count = 0
                    if verbose:
                        _scf_log("SCF", f"↺ α-recovery after freeze → {_alpha:.5f}")
            _max_diff_prev = _max_diff
            
            if _det_afm_sc < 0.0:
                _ansatz_unstable_ever = True   # physical instability: collinear AFM+SC ansatz broke down
                # Past QCP: exponential α penalty ∝ |det_afm|/det_warn; Λ_inst boosted to keep Anderson conservative.
                _det_penalty = float(np.clip(abs(_det_afm_sc) / max(_RPA_DET_WARN, 1e-6), 0.0, 5.0))
                _alpha = float(np.clip(_alpha * math.exp(-_RPA_QCP_PENALTY * _det_penalty), _MIXING / 16.0, _alpha))
                _lambda_lin_max = float(np.clip(_lambda_lin_max + 1.5 * _det_penalty, 0.0, 10.0))
                # Count iterations where det-penalty pins alpha at floor → enables recovery
                if _alpha <= _MIXING / 16.0 * 1.1:
                    _alpha_freeze_count += 1
            # Near QCP: belt-and-suspenders α cap
            elif (_det_afm_sc < _RPA_DET_WARN) or (_vertex_cache is not None and abs(_vertex_cache['V_irrep'][_IR_D, _IR_D]) > _V_CUT):
                _alpha = min(_alpha, _MIXING / (1.0 + _lambda_lin_max))
            else:
                # Safe zone (det > 0): fast exponential forgetting of past instability
                if _lambda_lin_max > 1.0:
                    _lambda_lin_max = float(np.clip((1-_EMA_NEW_WEIGHT) * _lambda_lin_max, 0.0, 10.0))

            # Store current det_afm so next iteration's sign-flip check can compare the cached value with the live value.
            if _vertex_cache is not None:
                _vertex_cache['det_afm_current'] = _det_afm_sc

            if verbose:
                _scf_log("SCF-A",
                    f"{iteration+1:3d}/{max_iter}"
                    f"  conv={_max_diff:.1e}  M={np.array2string(M, precision=3)}"
                    f"  Q={Q:+.4f} Å  mu={mu:.5f}  J_eff={_J_eff:.4f} eV"
                    f"  g_Delta_ad={g_Delta_ad:.4f}  g_Delta_bd={g_Delta_bd:.4f}"
                    f"  F_can={F_cluster['F_can']:.4f} eV  F_cluster={F_cluster['F_per_site']:.4f} eV"
                    f"  α={_alpha:.4f}  {(_time.time() - _iter_t0):3.0f}s/it")
                _scf_log("SCF-B",
                    f"  Q_fluct={F_cluster['Q_fluct']:.3f}  Γ_M={_Gamma_M*1000:.2f} meV"
                    f"  K_eff={F_cluster['K_eff']:+.4f} eV/Å²  [K_bulk={F_cluster['K_bulk']:+.4f} (E_cond''={F_cluster['K_econd']:+.4f})"
                    f"   + K_cluster={F_cluster['Delta_K_cluster']:+.4f} K_schur={F_cluster['K_schur']:.4f}]")
                _scf_log("SCF-C",
                    f"  |Δ|₂={float(np.linalg.norm(np.abs(_Delta_vec))):.4e}"
                    f"  Δ: {_fmt_amplitude_line(_Delta_vec, _DELTA_NAMES, top_n=4)}"
                    f"  F67: {_fmt_amplitude_line(_F67_vec, _DELTA_NAMES, top_n=4)}")
                if _vertex_cache is not None:
                    _scf_log("SCF-D",
                        f"  Stoner: FM={_rpa_q0_sc['stoner']:+.3f}  SDW={_rpa_pipi['stoner']:+.3f}  V_JT_corr*χQQ(q): (0,0)={_V_JT_corr*_vertex_cache['chi_QQ_q0']:.4f}  (π,π)={_V_JT_corr*_vertex_cache['chi_QQ_afm']:.4f}"
                        f"  χSQ_sc(q=π,π)={_chi_SQ_sc_pipi:.4f}  χSQ_sc(q=0)={_chi_SQ_sc_q0:.4f}"
                        f"  V_diag: {_fmt_amplitude_line(_vertex_cache['V_irrep'].diagonal(), _IRREP_NAMES, top_n=4, signed=True)}  V_sd={_vertex_cache['V_irrep'][_IR_S, _IR_D]:.3f}  lead={_vertex_cache.get('leading_sector')}")

            mu_new, n_kspace_new = self._find_mu_for_density(M_mixed, Q_mixed, Delta_vec_mixed, F67_vec_new, target_doping, mu, g_t, g_J)

            history['M'].append(M_mixed.copy())
            history['Q'].append(abs(Q_mixed))
            history['Delta_total'].append(float(np.sum(np.abs(Delta_vec_mixed))))
            history['n_kspace'].append(n_kspace_new)
            history['F_cluster'].append(F_cluster['F_per_site'])
            history['mu'].append(mu_new)
            history['mixing'].append(_alpha)

            M, Q, mu, n_kspace = M_mixed, Q_mixed, mu_new, n_kspace_new

            # ── Q-oscillation guard (signed Q iterates) ──
            _qdyn = self._classify_q_dynamics(_q_run_hist)
            if _qdyn['in_cycle']:
                _alpha_Q       = max(_alpha_Q * 0.5, _ALPHA_Q_MIN)
                _q_trust_scale = max(_q_trust_scale * 0.5, _Q_TRUST_MIN_SCALE)
                _alpha         = max(min(_alpha, _MIXING / (1.0 + 0.5 * _lambda_lin_max)) * _CYCLE_DAMP_FAC, _MIXING / 16.0)
                _q_calm = 0
                _q_run_hist[:] = _q_run_hist[-1:]          # a fresh window is needed before the next reaction
                if verbose:
                    _scf_log("Q-CYCLE", f"⚠ Q oscillates (zigzag={_qdyn['zigzag']:.2f}, max|ΔQ|={_qdyn['amp']:.4f} Å)  → α_Q={_alpha_Q:.3f}  Q-window scale={_q_trust_scale:.2f}  α={_alpha:.5f}")
            else:
                _q_calm += 1
                if _q_calm >= _Q_CYCLE_WINDOW and (_alpha_Q < _ALPHA_Q or _q_trust_scale < 1.0):
                    _alpha_Q       = min(_ALPHA_Q, _alpha_Q * 1.25)
                    _q_trust_scale = min(1.0, _q_trust_scale * 1.25)
            _Delta_vec = Delta_vec_mixed.copy()
            _F67_vec = F67_vec_new.copy()
            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                _vbdg._build_H_stack(self.k_points, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, out=_vbdg._H_stack)
                )
            
            # Saddle-escape: |Δ|≈0 + negative Hessian eigenvalue → kick along λ_min eigenvector.
            Delta_total = float(np.sum(np.abs(_Delta_vec)))
            if Delta_total < 5.0 * self.p.tol and iteration > 3 and iteration % 8 == 0:
                _hk = self.compute_hessian(target_doping, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, _V_JT_corr, Q_Eg2, _vertex_cache)
                
                lambda_min = _hk['lambda_min']
                if np.isfinite(lambda_min) and lambda_min < 0:
                    _mode = _hk['mode']

                    _kick_damp = 1.0 / (1.0 + _lambda_lin_max)   # Kick magnitude Λ-damped: 1/(1+Λ)
                    _curvature = min(abs(lambda_min), 1.0)
                    _kick_mag  = _KICK_BASE_FRACTION * _kick_damp * _curvature
                    step = self._project_kick_from_hessian(_hk, _kick_mag)

                    # Gentle pull toward the Γ6-only Stoner estimate: detect overshoot via M[0], then shrink all 3 channels uniformly to preserve their relative weights.
                    _M_phys_est = self.refine_M_state(target_doping, 0.0, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), damping = 0.45)[0][0]
                    _m_was_pulled = False
                    if _mode in ('pure-SC', 'SC-triggered-JT') and M[0] > 2.0 * _M_phys_est:
                        _pull_frac = _MODE_PULL_FRAC * _kick_damp
                        _shrink = 1.0 - _pull_frac * (1.0 - _M_phys_est / max(M[0], 1e-12))
                        _M_kick_component = np.clip(M * _shrink, 0.02, M)
                        _m_was_pulled = True
                    else:
                        _M_kick_component = np.clip(M + step[0:3], 0.0, 1.0)
                    
                    # Apply the kick only if the Δ component of the soft mode is significant
                    if _hk['frac_D'] > _KICK_MIN_DELTA_FRAC:
                        # Δ: signed kick along eigenvector component, preserving the full per-channel ratio (Delta_frac, now _N_DELTA components).
                        _D_kick_signed = max(0.0, Delta_total + step[4])
                        _phase_old = np.divide(_Delta_vec, np.abs(_Delta_vec), out=np.ones(_N_DELTA, dtype=complex), where=np.abs(_Delta_vec) > _MATH_EPS)                        
                        M = _M_kick_component
                        Q = float(np.clip(Q + step[3], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))
                        _Delta_vec = np.clip(_D_kick_signed * _hk['Delta_frac'], 0.0, 0.3) * _phase_old
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _vertex_cache = None
                        # Mark the EMA as stale so the vertex loop uses a higher blend weight for the first post-kick iteration
                        _solve_state._ema_kick_pending = True
                        if verbose:
                            _scf_log("SADDLE-ESC",
                                f"⚡kick iter={iteration}  mode={_mode}  λ_min={lambda_min:+.4f}  "
                                f"  damp={_kick_damp:.3f}  → M={np.array2string(M, precision=3)} Q={Q:+.4f} |Δ|={_D_kick_signed:.4f}  {' [M-pulled]' if _m_was_pulled else ''}")

            # Post-kick convergence re-evaluation
            _M_post  = abs(M) - abs(M_mixed)
            _Q_post  = abs(Q) - abs(Q_mixed)
            _Deltas_post = np.abs(_Delta_vec) - np.abs(Delta_vec_mixed)
            _kick_fired = (np.sum(np.abs(_M_post)) + abs(_Q_post) + np.sum(np.abs(_Deltas_post))) > 1e-10
            gc.collect()

            if (not _kick_fired
                    and not iteration < _MIN_ITER
                    and not _det_afm_sc < 0.0
                    and _max_diff < self.p.tol
                    and abs(n_kspace - (1 - target_doping)) < self.p.tol * 10):
                converged = True
                break
        
        if not converged and verbose:
            _scf_log("SCF-RES",
                f"⚠ no conv after {_MAX_ITER} iters,  max_diff={_max_diff:.2e}"
                f"  dens_err={abs(n_kspace-(1-target_doping)):.2e}"
                f"  M={np.array2string(M, precision=4)}  Q={Q:+.4f}  |Δ|(7a s+d)={abs(_Delta_vec[_DELTA_IDX['s7a']])+abs(_Delta_vec[_DELTA_IDX['d7a']]):.4f}  |Δ|(total)={float(np.sum(np.abs(_Delta_vec))):.4f}"
                f"  dyn={_scf_dynamics_regime}"
                f"{'  [ansatz unstable]' if _ansatz_unstable_ever else ''}")

        # Post-loop diagnostic: λ_max and Rayleigh JT projection and store converged gap and distortion
        J_A1g_diag, J_B1g_bare, J_B1g_bare_7b = self.p.exchange_channels(Q, g_J)
        if _vertex_cache is None:
            _vertex_cache = self.compute_pairing_kernel_and_build_cache(M, Q, n_kspace, mu, g_t, g_J, J_A1g_diag, _Gamma_M, _V_JT_eff, _V_JT_corr, _V_cap)
        _vertex_cache = self.scf_gap_diagnostics(_Delta_vec, g_Delta_s, g_Delta_ad, self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, _vbdg), _vertex_cache)

        _Delta_s_mag = abs(_Delta_vec[_DELTA_IDX['s7a']])
        _Delta_d_mag = abs(_Delta_vec[_DELTA_IDX['d7a']])
        _Delta_irrep_scf_mag = np.abs(_Delta_vec[_IDX_7A_ARR])

        _dlam_dQ_fs = float('nan')     # only meaningful once converged with a non-negligible gap
        _hot_spot_frac = 0.0

        # ── FS-resolved ∂λ/∂Q and gap-channel decomposition (SC state only) ──────
        if converged and (_Delta_s_mag + _Delta_d_mag) > _QQ_DELTA_THRESH:
            # Adaptive finite-difference step: 5 % of |Q| protects against noise at larger distortions.
            _dQ_step = max(_DQ_FS_VERTEX, _DQ_FS_VERTEX_FRAC * abs(Q))
            lmax_p, V_diag_p = self._vertex_matrix_at_Q(M, Q + _dQ_step, n_kspace, mu, g_t, g_J, _Gamma_M, _V_JT_eff, _V_JT_corr, _V_cap)
            lmax_m, V_diag_m = self._vertex_matrix_at_Q(M, Q - _dQ_step, n_kspace, mu, g_t, g_J, _Gamma_M, _V_JT_eff, _V_JT_corr, _V_cap)
            _dlam_dQ_fs = (lmax_p - lmax_m) / (2.0 * _dQ_step)
            if len(V_diag_p) == len(V_diag_m):
                _dV_dQ_diag = (V_diag_p - V_diag_m) / (2.0 * _dQ_step)
                _hot_spot_frac = float(np.mean(_dV_dQ_diag > 0)) if len(_dV_dQ_diag) > 0 else 0.0
            else:
                # FS point count differs at Q±dQ_step (likely a pocket appearing/merging near pinned Q≈0). Keep _hot_spot_frac = 0.0 rather than aborting over this post-hoc diagnostic.
                _scf_log("SCF-RES", f"⚠ hot-spot diagnostic skipped: FS point count differs at Q±dQ ({len(V_diag_p)} vs {len(V_diag_m)})")

            # Channel-resolved pairing strength from the full N_IRREP x N_IRREP kernel projection — handles arbitrary irrep mixing exactly via the off-diagonal entries of V_irrep, unlike a single-eigenvector scalar split.
            _lam_max_s = _vertex_cache['lambda_s']
            _lam_max_d = _vertex_cache['lambda_d']
            _lambda_irrep_vc = _vertex_cache['lambda_irrep']    # (_N_IRREP,), canonical order

            # Dominant symmetry channel:
            _sym_scf = _IRREP_NAMES[int(np.argmax(_Delta_irrep_scf_mag))]
            _sym_lin = _vertex_cache['sym_lin']
            _sym_mismatch = _sym_scf != _sym_lin

            if _vertex_cache['frac_seed'] is not None:
                gap_symmetry = f'{_IRREP_D4H[_sym_lin]} ({_sym_lin}-wave)'

            if _sym_mismatch:
                gap_symmetry = (gap_symmetry.split(' [')[0] + f' [lin={_sym_lin}, SCF={_sym_scf}: nonlinear quench]')
            else:
                gap_symmetry += f' [lin={_sym_lin}, SCF={_sym_scf}: consistent]'

            if verbose:
                _neg_note = ('  [⚠ λ<0: FS-avg repulsive — instability requires nodal sign change]'
                            if _vertex_cache['lambda_lin_max'] < 0 else '')
                _scf_log("SCF-RES",
                    f"  ∂λ/∂Q(FS)={_dlam_dQ_fs:+.4f} eV⁻¹"
                    f"  {'✓ SC condensate enhances pairing at hot spots' if np.isfinite(_dlam_dQ_fs) and _dlam_dQ_fs > 0 else '⚠ SC-JT FS coupling absent'}"
                    f"  hot_spot_frac={_hot_spot_frac:.2f}  {'[concentrated]' if _hot_spot_frac > 0.6 else '[diffuse]' if _hot_spot_frac > 0.3 else '[anti-nodal suppressed]'}")
                _scf_log("SCF-RES",
                    f"  λ_lin_max={_vertex_cache['lambda_lin_max']:.3f}"
                    f"  λ_irrep({','.join(_IRREP_NAMES)})={np.array2string(_lambda_irrep_vc, precision=4)}   [{gap_symmetry}]{_neg_note}"
                    f"  |Δ_7a|({','.join(_IRREP_NAMES)})={np.array2string(_Delta_irrep_scf_mag * 1000, precision=3)}meV"
                    f"  unresolved_weight={float(_vertex_cache['unresolved_weight']) * 100:.1f}%"
                    f"  {'[NL SYMMETRY MIXING]' if (_sym_mismatch and _Delta_s_mag > 1e-5 and _Delta_d_mag > 1e-5) else ''}")

        # ── Incommensurate nesting scan ──
        # Normal-state (Δ = 0) Lindhard kernel over the BdG eigenstates at the converged (M, Q, μ); only χ_SS is evaluated since χ_SQ = 0 in the normal state (Shubnikov argument: the odd-in-k integrand cancels over the MBZ).
        # The BdG H is commensurate (q_AFM = (π,π)); incommensurate nesting (δq > 0) is detected via the Lindhard sum at shifted momenta q = (π, π−δq), keeping Δ = 0 in the BdG.
        _ic_dq_max  = 0.0
        _ic_chi_max = 0.0
        _ic_chi_0   = 0.0
        if converged:
            _ic_dq_max, _ic_chi_max, _ic_chi_0 = self._scan_incommensurate_nesting(M, Q, mu, g_t, g_J, n_kspace)
            _ic_flag = 0.05 * np.pi < _ic_dq_max < 0.10 * np.pi
            _scf_log("SCF-RES",
                f"Incommensurate scan: dq*={_ic_dq_max/np.pi:.3f}π"
                f"  χ(0)={_ic_chi_0:.4f}  χ(max)={_ic_chi_max:.4f}"
                f"  {'⚠ incommensurate' if _ic_flag else '✓ commensurate'}")

        # ── SCF convergence summary ──────────────────────────────────────────────
        if verbose and converged:
            _scf_log("SCF-RES",
                f"  dyn={_scf_dynamics_regime}{'  ⚠ ANSATZ UNSTABLE' if _ansatz_unstable_ever else ''}"
                f"  χ_SQ^SC(π,π)={_chi_SQ_sc_pipi:+.5f} eV⁻¹  χ_SQ/χ_SS={_chi_SQ_sc_pipi / _chi_SS_sc_pipi if abs(_chi_SS_sc_pipi) > 1e-12 else float('nan'):+.4f}"
                f"  [{'strong' if abs(_chi_SQ_sc_pipi) > 0.10 else 'moderate' if abs(_chi_SQ_sc_pipi) > 0.02 else 'weak'}]")
        
        # ── Post-SCF Mott filter ──────────────────────────────────────────────
        # g_t<0.10 (δ<0.053): incoherent ZRS band; ξ/a<1.0: BEC limit, Cooper pairs not lattice-coherent.
        _mott_suspect = (g_t < _G_T_COHERENCE_MIN) or (_vertex_cache['xi_nodal'] < 1.0)
        if _mott_suspect:
            _Delta_vec = np.zeros(_N_DELTA, dtype=complex)
            converged = False
            if verbose:
                _reason = 'g_t<min (Mott)' if g_t < _G_T_COHERENCE_MIN else f"ξ/a={_vertex_cache['xi_nodal']:.2f}<1 (BEC)"
                _scf_log("SCF-RES", f"⚠ MOTT-SUSPECT [{_reason}]  g_t={g_t:.3f}  ξ/a={_vertex_cache['xi_nodal']:.2f}  — gap suppressed")
        
        _, _Fcl_sc_fin, _V_JT_eff, _V_JT_corr, _ = self._make_vertex_params(target_doping, M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, _chi_QQ_sc_q0, Q_Eg2, (_bdg_ev_sc, _bdg_ec_sc), _vertex_cache)
        _dFc_ref = self._dFc_reference(target_doping, mu, g_t, g_J)
        _dFc_sc  = self._dF_cluster(M, Q, target_doping, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J)
        
        # ── Post-convergence Eg,2 PHONON STABILITY diagnostic
        _chi_Eg2Eg2_final = float(self._chi_QQ_matrix_elements(M, Q, _Delta_vec, _F67_vec, n_kspace, mu, g_t, g_J, Q_Eg2, return_matrix=True)[1, 1])
        _G44_final        = 1.0 - _chi_Eg2Eg2_final * self.p.g_Eg2**2 / self.p.K_lattice_Eg2
        _eg2_exp_final    = self.Eg2_expectation((_bdg_ev_sc, _bdg_ec_sc))
        if verbose:
            _scf_log("SCF-EG2",
                f"δ={target_doping:.4f}  Q_Eg2(fixed)={Q_Eg2:+.4f}  <Eg2>={_eg2_exp_final:+.3e}  "
                f"χ_Eg2,Eg2={_chi_Eg2Eg2_final:+.4f}  G44={_G44_final:+.4f} [{'stable' if _G44_final > 0 else 'SOFT Eg2 PHONON'}]")
        
        # ── Post-SCF QCP check ────────────────────────────────────────────────
        _past_qcp = bool(_det_afm_sc < 0.0)
        _converged_reported = bool(converged) and not _past_qcp
        if _past_qcp:
            _scf_log("SCF-RES", f"⚠ PAST-QCP: det(1−χ₀U)(π,π) = {_det_afm_sc:+.5f} < 0 on the returned state — the collinear AFM(+SC) ansatz is RPA-unstable, converged → {_converged_reported}")

        result = dict(_vertex_cache)
        result.update({
            'past_qcp': _past_qcp,                        # True: det(RPA)(π,π) < 0 on the returned state (post-SCF check); downstream users must exclude this point from free-energy comparisons
            'det_afm_final': _det_afm_sc,
            'scf_iteration_converged': bool(converged),   # raw tolerance-level convergence, before the QCP policy is applied to 'converged'
            'M': M,
            'Q': Q,
            'Eg2_expectation': _eg2_exp_final,            # sanity check: should be ~0 for real Delta_s,Delta_d
            'chi_Eg2Eg2': _chi_Eg2Eg2_final,
            'G44_Eg2_stable': bool(_G44_final > 0) if math.isfinite(_G44_final) else None,
            'Delta_s': _Delta_s_mag,
            'Delta_d': _Delta_d_mag,
            'Delta_irrep': {_n: float(abs(_Delta_vec[_DELTA_IDX[_n + '7a']])) for _n in _IRREP_NAMES},   # |Δ_7a| of EVERY channel of the active basis (incl. g, px2, py2)
            'Delta_vec': _Delta_vec,
            'F67_vec': _F67_vec,
            'V_JT_corr': _V_JT_corr,
            'F_cluster': _Fcl_sc_fin,
            'mu': mu,
            'n_kspace': n_kspace,
            'F_bdg': _Fcl_sc_fin['F_can'] + (_dFc_sc - _dFc_ref),
            'J_eff': _J_eff,
            'target_doping': target_doping,
            'afm_unstable': _det_afm_sc <= 0.0,
            'selection_ratio': _selection_ratio,
            'history': history,
            'converged': _converged_reported,
            'mott_suspect': _mott_suspect,
            'scf_dynamics_regime': _scf_dynamics_regime,   # 'converging'|'limit_cycle'|'first_order_jump'|'hysteretic'
            'ansatz_unstable': _ansatz_unstable_ever,      # True if det(RPA)<0 at any SCF iteration
            'dlam_dQ_fs': _dlam_dQ_fs,
            'incommensurate_dq': _ic_dq_max,
            'incommensurate_chi_ratio': _ic_chi_max / max(_ic_chi_0, 1e-12) if _ic_chi_0 else float('nan'),
            'vertex_cache': _vertex_cache,
        })
        return result

    def _dF_cluster(self, M_vec: np.ndarray, Q: float, target_doping: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu_guess: float, g_t: float, g_J: float, mu_v: Optional[float] = None) -> float:
        """
        Beyond-MF cluster correction at (M_vec, Q):
            ΔF_cluster = (F_cluster^exact − F_cluster^MF) / _CLUSTER_SIZE
        with both cluster-ED free energies evaluated at the Q-consistent μ fixed by
        the bulk density constraint.
        """
        M_vec = np.atleast_1d(np.asarray(M_vec, dtype=float))

        # μ at fixed density (bulk constraint)
        if mu_v is None:
            mu_v, _ = self._find_mu_for_density(
                M_vec, Q, Delta_vec, F67_vec, target_doping, mu_guess, g_t, g_J,
            )

        # Thermal-FS average of β_k at Q
        tx_b, ty_b = self.p.effective_hopping_anisotropic(Q)
        beta_k = np.asarray(
            self.p.wave_function_weight(tx_b, ty_b, self.k_points[:, 0], self.k_points[:, 1]),
            dtype=float,
        )
        vbdg = self._get_vbdg()
        ev_n, _ = np.linalg.eigh(vbdg._build_H_stack(
            self.k_points, M_vec, Q,
            np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex),
            n_kspace, mu_v, g_t, g_J,
        ))
        f_n = _fermi_function(ev_n, self.kT)
        w_fermi = np.sum(f_n * (1.0 - f_n) / self.kT, axis=1)
        denom = float(np.sum(self.k_weights * w_fermi))
        beta_cluster = (
            float(np.sum(self.k_weights * w_fermi * beta_k) / denom)
            if denom >= _MATH_EPS else float(np.mean(beta_k))
        )
        g_eff = beta_cluster * self.g_JT_bare

        J_val = float(self.p.exchange_channels(Q, g_J)[0][0])

        _e_full = np.linalg.eigvalsh(self._build_H_cluster(M_vec, Q, g_J, _Z_EFF, g_eff, beta_cluster, mu_v, F67_vec, include_exchange=True, include_TRW=True))
        _E0_full = float(_e_full[0])
        _w_full = np.exp(np.clip(-(_e_full - _E0_full) / self.kT, -700.0, 0.0))
        _F_dc_full = _CLUSTER_SIZE * 0.5 * _Z_EFF * J_val * (M_vec[0] ** 2)
        _F_full = _E0_full - self.kT * np.log(max(float(_w_full.sum()), _MATH_EPS)) + _F_dc_full
        
        _e_MF = np.linalg.eigvalsh(self._build_H_cluster(M_vec, Q, g_J, _Z, g_eff, beta_cluster, mu_v, F67_vec, include_exchange=False, include_TRW=True))
        _E0_MF = float(_e_MF[0])
        _w_MF = np.exp(np.clip(-(_e_MF - _E0_MF) / self.kT, -700.0, 0.0))
        _F_dc_MF = _CLUSTER_SIZE * 0.5 * _Z * J_val * (M_vec[0] ** 2)
        _F_MF = _E0_MF - self.kT * np.log(max(float(_w_MF.sum()), _MATH_EPS)) + _F_dc_MF
        return float((_F_full - _F_MF) / _CLUSTER_SIZE)

    def _make_Q_profile(self, target_doping: float, M: np.ndarray, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu_guess: float, g_t: float, g_J: float, V_JT_corr: float, Q_Eg2: float = 0.0, vertex_cache: Optional[dict] = None) -> Tuple[Callable[[float], float], Callable[[float], float]]:
        """F_total(Q) = F_can(Q; μ(Q), V_JT_corr) + ΔF_cluster(Q) − ΔF_cluster(0); returns (F_total, force)."""
        vbdg = self._get_vbdg()
        mu_cache: Dict[float, Tuple[float, float]] = {}
        dFc_cache: Dict[float, float] = {}
        econd_cache: Dict[float, float] = {}

        def mu_at(Qv: float) -> Tuple[float, float]:
            k = round(float(Qv), 10)
            if k not in mu_cache:
                mu_cache[k] = self._find_mu_for_density(M, Qv, Delta_vec, F67_vec, target_doping, mu_guess, g_t, g_J)
            return mu_cache[k]

        def dFc_at(Qv: float) -> float:
            k = round(float(Qv), 10)
            if k not in dFc_cache:
                dFc_cache[k] = self._dF_cluster(M, Qv, target_doping, Delta_vec, F67_vec, n_kspace, mu_guess, g_t, g_J, mu_v=mu_at(Qv)[0])
            return dFc_cache[k]

        # ---- E_cond(Q) at frozen V_irrep (for the force derivative) ----
        def E_cond_at(Qv: float) -> float:
            k = round(float(Qv), 10)
            if k not in econd_cache:
                mu_v, n_v = mu_at(Qv)
                g_Delta_s, g_Delta_ad, g_Delta_bd = self.estimate_gutzwiller_factors_occupation_based(M, Qv, n_v, mu_v, g_t, g_J)
                econd_cache[k] = _sector_condensation_energy(Delta_vec[_IDX_7A_ARR], g_t, g_Delta_s, g_Delta_ad, V_JT_corr, vertex_cache) + _sector_condensation_energy(Delta_vec[_IDX_7B_ARR], g_t, g_Delta_s, g_Delta_bd, V_JT_corr, vertex_cache)
            return econd_cache[k]

        def F_total(Qv: float) -> float:
            F_bulk = self._canonical_free_energy(target_doping, M, Qv, Delta_vec, F67_vec, n_kspace, mu_guess, g_t, g_J, V_JT_corr, Q_Eg2, vertex_cache)
            return F_bulk + dFc_at(Qv) - dFc_at(0.0)

        def force(Qv: float) -> float:
            mu_q, n_q = mu_at(Qv)
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M, Qv, Delta_vec, F67_vec, n_q, mu_q, g_t, g_J, out=vbdg._H_stack))
            dHdQ = self._calc_dHdQ(M, Qv, F67_vec, mu_q, g_t, g_J)
            # Hellmann–Feynman:
            hf = float(np.sum(self.k_weights[:, None] * _fermi_function(ev, self.kT) * np.einsum('kin,kij,kjn->kn', ec.conj(), dHdQ, ec).real) / 4.0)
            
            # AFM Weiss-field force
            eps, _ = _q_fd_step(Qv)
            J_p, _, _ = self.p.exchange_channels(Qv + eps, g_J)
            J_m, _, _ = self.p.exchange_channels(Qv - eps, g_J)
            ex = 0.5 * float(np.dot(M ** 2, (self.p._exchange_J_q(J_p, np.pi, np.pi) - self.p._exchange_J_q(J_m, np.pi, np.pi)) / (2.0 * eps)))
            
            # Condensation-energy force (V_irrep frozen → only μ(Q) and g_i(Q) vary)
            dE_cond_dQ = (E_cond_at(Qv + eps) - E_cond_at(Qv - eps)) / (2.0 * eps)

            # Cluster-correction force (h_c = 1e-3 Å: coarse on purpose — dFc is smooth on that scale)
            h_c = 1e-3
            dFc_dQ = (dFc_at(Qv + h_c) - dFc_at(Qv - h_c)) / (2.0 * h_c)
            return hf + ex + self.p.K_lattice * Qv + dE_cond_dQ + dFc_dQ
        return F_total, force

    def solve_Q_slave(self, target_doping: float, M: np.ndarray, Q_prev: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu_guess: float, g_t: float, g_J: float, V_JT_corr: float, q_trust_scale: float = 1.0, vertex_cache: Optional[dict] = None) -> Tuple[float, dict]:
        """
        Q-profile minimization: F_total(Q) = F_can(Q) + ΔF_cluster(Q) − ΔF_cluster(0).

        LOCAL mode (Q_prev AND Q_trust given — what the SCF uses):
            Trust-region search in [Q_prev − Q_trust, Q_prev + Q_trust] with _Q_WINDOW_NPTS points. V_irrep is frozen at Q_prev and strongly Q-dependent, so the profile is only trustworthy
            near Q_prev; the B1g profile is a shallow, almost symmetric double well (Z2 partners differ by a few μeV — less than the cache drift from one iteration to the next), 
            A trust region around the current Q is branch-continuous BY CONSTRUCTION without modifying F. If the profile is flat to numerical noise (< _Q_FLAT_EV), Q_prev is returned unchanged.

        Envelope theorem: the Schur term (Δ-relaxation) is NOT part of the Q equation; it only changes the curvature, i.e. K_eff diagnostics; the SCF fixpoint is stationary on F_total by construction.
        """
        _F_total, _total_force = self._make_Q_profile(target_doping, M, Delta_vec, F67_vec, n_kspace, mu_guess, g_t, g_J, V_JT_corr, 0.0, vertex_cache)

        def _refine(lo: float, hi: float, Q_grid: float) -> float:
            if lo > hi:
                lo, hi = hi, lo
            f_lo, f_hi = _total_force(lo), _total_force(hi)
            Q_ref = float(Q_grid)
            if f_lo * f_hi < 0.0:
                try:
                    Q_root = float(brentq(_total_force, lo, hi, xtol=1e-6, rtol=1e-10))
                except ValueError:
                    Q_root = None
                if Q_root is not None:
                    eps_c, _ = _q_fd_step(Q_root)
                    curvature = (_total_force(Q_root + eps_c) - _total_force(Q_root - eps_c)) / (2.0 * eps_c)
                    if curvature > 0.0:
                        Q_ref = Q_root
            return Q_ref

        r_w = q_trust_scale * (_Q_TRUST_ABS + _Q_TRUST_REL * abs(Q_prev))
        Q_cap = 0.5 * self.p.lambda_hop
        lo_w, hi_w = max(-Q_cap, float(Q_prev) - r_w), min(Q_cap, float(Q_prev) + r_w)

        Qs = np.linspace(lo_w, hi_w, _Q_WINDOW_NPTS)
        Fs = np.array([_F_total(q) for q in Qs], dtype=float)
        i_min = int(np.argmin(Fs))

        if float(Fs.max() - Fs.min()) < _Q_FLAT_EV:
            Q_loc, edge = float(Q_prev), False
        elif i_min == 0 or i_min == len(Qs) - 1:
            Q_loc, edge = float(Qs[i_min]), True
        else:
            Q_loc, edge = _refine(
                float(Qs[i_min - 1]),
                float(Qs[i_min + 1]),
                float(Qs[i_min]),
            ), False

        F_mirror = float(_F_total(-Q_loc) - _F_total(Q_loc)) if abs(Q_loc) > 1e-4 else float('nan')

        qslave_info = {
            'mode': 'local',
            'edge': edge,
            'lo_w': float(lo_w),
            'hi_w': float(hi_w),
            'dF_window_ueV': float((Fs.max() - Fs.min()) * 1e6),
            'F_mirror': F_mirror,
            'Qs': np.asarray(Qs) if Qs is not None else np.array([]),
            'Fs_phys': (
                np.asarray(Fs - np.min(Fs))
                if Fs is not None and len(Fs) > 0
                else np.array([])
            ),
        }
        return Q_loc, qslave_info

    def _scan_incommensurate_nesting(self, M: np.ndarray, Q: float, mu: float, g_t: float, g_J: float, n_kspace: float) -> Tuple[float, float, float]:
        """
        Incommensurate nesting scan: q* = (π, π−δq) that maximizes χ_SS. The BdG Hamiltonian is fixed to a commensurate q_AFM = (π, π),
        so any incommensurate nesting is only detected here. Uses the normal-state (Δ=0) Lindhard kernel evaluated at the converged (M, Q, μ).
        Consistent with the Shubnikov-group analysis: in the collinear AFM normal state χ_SQ vanishes by k → −k symmetry, so only χ_SS is evaluated.
        
        Known limitation: the vertex is the STAGGERED Sz operator, which is exact only at q = (π, π). For δq ≠ 0 the proper vertex carries
        e^{iq·r_α} phase factors on each sublattice. Because the code's convention uses the unfolded chemical BZ with explicit A/B sublattices,
        this correction is not included here and the δq ≠ 0 values should be read as a "staggered-approximation" scan, not the exact χ_SS^{phys}(q).
        """
        sz_bdg24 = np.concatenate([
            self.sz_op,   # particle A   (+)
            -self.sz_op,   # particle B  (stagger)
            -self.sz_op,   # hole A      (p-h conjugate)
            self.sz_op,   # hole B       (double flip)
        ])
        _vbdg = self._get_vbdg()
        _Ek_ic, _Vk_ic = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, _vbdg)

        ev_pos = np.where(_Ek_ic > 0.0, _Ek_ic, np.inf)
        Emin = ev_pos[np.arange(len(ev_pos)), np.argmin(ev_pos, axis=1)]
        
        mbz_mask_k = _mbz_deduplication(np.asarray(self.k_points), Emin, self.kT)
        
        # MBZ weight: zero the redundant half, then restore normalization.
        # The MBZ is exactly half the full BZ, so doubling the surviving half preserves the integral.
        w_mbz = 2.0 * self.k_weights * mbz_mask_k.astype(float)
        
        eta_norm = max(_ETA_T_FRAC * self.kT, _ETA_GRID_FLOOR * self.p.t0)
        _dq_scan = np.linspace(0.0, 0.15 * np.pi, 7)
        _chi_SS_q_scan = []

        for _dq_v in _dq_scan:
            _dk_ic = 2.0 * np.pi / _NK
            _q_ic = np.array([np.pi, np.pi - _dq_v])
            if _dq_v == 0.0:
                # Commensurate point q=(π,π): k and k+q are pairwise identified under the magnetic-zone folding
                _nxq = int(round(_q_ic[0] / _dk_ic)) % _NK
                _nyq = int(round(_q_ic[1] / _dk_ic)) % _NK
                _shift_q = self.shift_table[_nxq, _nyq]
                _Ekq_ic = _Ek_ic[_shift_q]
                _Vkq_ic = _Vk_ic[_shift_q]
            else:
                _kpts_q = (self.k_points + _q_ic[None, :] + np.pi) % (2 * np.pi) - np.pi
                _Ekq_ic, _Vkq_ic = np.linalg.eigh(
                    _vbdg._build_H_stack(_kpts_q, M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J)
                )

            _fk  = _fermi_function(_Ek_ic, self.kT)
            _fkq = _fermi_function(_Ekq_ic, self.kT)

            # <n, k| Sz_stag |m, k+q>: apply Sz to the q-shifted states
            if self._axis_is_z:
                _SzV = sz_bdg24[None, :, None] * _Vkq_ic
            else:
                _SzV = np.einsum('ab,kbm->kam', self._S_stag_nambu_total, _Vkq_ic)
            _Mm  = np.einsum('kin,kim->knm', _Vk_ic.conj(), _SzV)
            _kernel = _lehmann_kernel(_fk, _fkq, _Ek_ic, _Ekq_ic, eta_norm, self.kT)
            _r = w_mbz[:, None, None] * np.abs(_Mm) ** 2 * _kernel
            _chi_SS_q_scan.append(_NAMBU_MBZ_NORMFACTOR * float(_r.sum()))

        _idx_max    = int(np.argmax(_chi_SS_q_scan))
        _ic_dq_max  = float(_dq_scan[_idx_max])
        _ic_chi_max = _chi_SS_q_scan[_idx_max]
        _ic_chi_0   = _chi_SS_q_scan[0]
        return _ic_dq_max, _ic_chi_max, _ic_chi_0
    
    def _anderson_mix(self, x_history: list, f_history: list, m: int = 5, alpha: float = None) -> np.ndarray:
        """
        Robust Anderson(m) acceleration with scale normalisation, adaptive Tikhonov regularisation,
        condition-number guard, residual-norm acceptance test, trust-region safeguard,
        and automatic fallback to linear mixing. Safe for stiff SCF problems (BdG + JT + feedback).
        """
        alpha = _MIXING if alpha is None else alpha
        x_last = np.asarray(x_history[-1], dtype=float)
        f_last = np.asarray(f_history[-1], dtype=float)

        # Simple linear mixing fallback
        x_simple = (1.0 - alpha) * x_last + alpha * f_last

        n = min(len(x_history), m)
        if n < 2:
            return x_simple

        # Build history arrays
        X = np.asarray(x_history[-n:], dtype=float)
        F = np.asarray(f_history[-n:], dtype=float)
        R = F - X   # residuals
        dR = np.diff(R, axis=0)
        dX = np.diff(X, axis=0)
        r_last = R[-1]

        # Component scaling
        scale = np.ones_like(x_last)

        dX_scaled = dX * scale           # shape (n-1, d)
        dR_scaled = dR * scale           # shape (n-1, d)
        r_scaled  = r_last * scale       # shape (d,)
        x_last_scaled = x_last * scale   # shape (d,)

        # Regularised normal equations in scaled space: min ||r_scaled − dR_scaled·θ||² + β||θ||²
        A = dR_scaled @ dR_scaled.T
        b = dR_scaled @ r_scaled
        diag_max = max(float(np.max(np.abs(np.diag(A)))), 1e-30)
        A.flat[::A.shape[0] + 1] += _ANDERSON_TIKHONOV * diag_max

        # Condition number guard
        try:
            condA = np.linalg.cond(A)
            if not np.isfinite(condA) or condA > 1e12:
                return x_simple
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return x_simple

        # Canonical Anderson update fully in scaled space, then unscale.
        correction_scaled = (dX_scaled + dR_scaled).T @ theta
        x_opt_scaled = x_last_scaled + r_scaled - correction_scaled
        x_opt = x_opt_scaled / scale

        if not np.all(np.isfinite(x_opt)):
            return x_simple

        # Only accept Anderson if it actually reduces residual relative to the simple-mixing fallback
        r_simple = f_last - x_simple
        r_opt    = f_last - x_opt

        norm_simple = np.linalg.norm(r_simple)
        norm_opt    = np.linalg.norm(r_opt)

        # If Anderson worsens residual vs. simple mixing -> fallback
        if norm_opt > norm_simple:
            return x_simple

        # Trust region safeguard (limit step size)
        step_simple = x_simple - x_last
        step_opt    = x_opt - x_last

        norm_step_simple = np.linalg.norm(step_simple)
        norm_step_opt    = np.linalg.norm(step_opt)

        if norm_step_opt > _ANDERSON_TRUST * max(norm_step_simple, 1e-12):
            # Too aggressive → shrink
            shrink = (_ANDERSON_TRUST * norm_step_simple) / (norm_step_opt + 1e-12)
            x_opt = x_last + shrink * step_opt

        # Blended final step: conservative near small alpha, more Anderson near full mixing.
        w = float(np.clip(alpha / max(_MIXING, _MATH_EPS), _ANDERSON_W_LO, _ANDERSON_W_HI))
        x_new = w * x_opt + (1.0 - w) * x_simple

        if not np.all(np.isfinite(x_new)):
            return x_simple
        return x_new

    def _dFc_reference(self, target_doping, mu_guess, g_t, g_J):
        """
        Beyond-MF cluster-correction at the FULLY NORMAL reference state (M=0, Q=0, Δ=0, F67=0), per site.  A fixed constant across all SCF scenarios
        sharing (target_doping, g_t, g_J): n_kspace is a pass-through into _build_H_stack, and mu_guess only seeds the μ-solver, which converges to the unique normal-state μ.
        """
        key = (round(float(target_doping), 6), round(float(g_t), 8), round(float(g_J), 8))
        cache = self.__dict__.setdefault('_dFc_ref_cache', {})
        if key not in cache:
            try:
                val = self._dF_cluster(np.zeros(_N_CHANNELS, dtype=float), 0.0, target_doping, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), 1.0 - target_doping, mu_guess, g_t, g_J)
                cache[key] = float(val) if np.isfinite(val) else 0.0
            except Exception:
                cache[key] = 0.0
        return cache[key]

    def compute_hessian(self, target_doping: float, M_channels: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, V_JT_corr: float, Q_Eg2: float, vertex_cache: dict = None) -> Dict:
        """(3+2)×(3+2) finite-difference Hessian in (M_c, Q, |Δ|) of the beyond-MF free energy F_phys = F_bulk + ΔF_cluster(M,Q,Δ) − ΔF_cluster(0,0,0) at fixed Q_Eg2; the reference is the fully normal state
        (M = Q = Δ = 0, F67 = 0), shared with _dFc_reference. |Δ| is the total amplitude over all _N_DELTA channels; the probed direction rescales all of them together and keeps their converged relative phases
        (collapsing them onto a real, phase-stripped split would evaluate F off the stationary point whenever the state carries a nontrivial relative phase)."""
        vbdg = self._get_vbdg()
        M_channels = np.asarray(M_channels, dtype=float)
        Delta_vec  = np.asarray(Delta_vec, dtype=complex)

        Delta = float(np.sum(np.abs(Delta_vec)))
        _abs = np.abs(Delta_vec)
        Delta_frac = (_abs / Delta) if Delta > _QQ_DELTA_THRESH else np.full(_N_DELTA, 1.0 / _N_DELTA)
        # Converged relative phase, held fixed for every probe point below. Falls back to 1.0+0j (real, positive) when a channel's amplitude is too small to define a phase
        phase = np.where(_abs > 1e-12, Delta_vec / np.where(_abs > 1e-12, _abs, 1.0), 1.0 + 0j)

        eps_M = np.maximum(1e-4, np.abs(M_channels) * 1e-3)
        eps_Q, _ = _q_fd_step(Q)
        eps_D = max(_HESS_DELTA_EPS_MIN, abs(Delta) * 1e-3)       # not ≤ |Δ|/2: at Δ = 0 that gave 1e-10 and a curvature that was pure rounding noise (F is even in Δ, so the stencil may straddle Δ = 0)
        eps5 = np.array([eps_M[0], eps_M[1], eps_M[2], eps_Q, eps_D])

        _dFc_ref = self._dFc_reference(target_doping, mu, g_t, g_J)

        def F(x5: np.ndarray) -> float:
            m_vals     = x5[0:3]
            q_val      = x5[3]
            delta_val  = x5[4]
            d_vec      = phase * abs(delta_val) * Delta_frac
            F_bulk = self._canonical_free_energy(target_doping, m_vals, q_val, d_vec, F67_vec, n_kspace, mu, g_t, g_J, V_JT_corr, Q_Eg2, vertex_cache)
            dFc = self._dF_cluster(m_vals, q_val, target_doping, d_vec, F67_vec, n_kspace, mu, g_t, g_J)
            return F_bulk + (dFc - _dFc_ref)

        x0 = np.array([M_channels[0], M_channels[1], M_channels[2], Q, Delta])
        F0, _, H = _fd_grad_hessian(F, x0, eps5, need_grad=False)

        # beyond-MF correction at x0
        F_bulk_x0 = self._canonical_free_energy(target_doping, M_channels, Q, Delta_vec, F67_vec, n_kspace, mu, g_t, g_J, V_JT_corr, Q_Eg2, vertex_cache)

        _evals, _evecs = np.linalg.eigh(H)
        
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)
        t_eff = g_t * float(np.sqrt(0.5 * ((tx_bare)**2 + (ty_bare)**2)))
        _scales_h = np.array([_KICK_M_CLIP_HI, _KICK_M_CLIP_HI, _KICK_M_CLIP_HI, self.p.lambda_hop, t_eff])
        _S_h = np.diag(_scales_h)
        _evals_scaled, _evecs_scaled = np.linalg.eigh(_S_h @ H @ _S_h)
        _idx_min_scaled = int(np.argmin(_evals_scaled))
        _physical_dir = _scales_h * _evecs_scaled[:, _idx_min_scaled]

        _wsum = max(float(np.linalg.norm(_physical_dir[0:3])) + abs(_physical_dir[3]) + abs(_physical_dir[4]), 1e-12)
        if abs(_physical_dir[4]) / _wsum > _MODE_FRAC_DOMINANT:                       _mode = 'pure-SC'
        elif abs(_physical_dir[3]) / _wsum > _MODE_FRAC_DOMINANT:                     _mode = 'pure-JT'
        elif float(np.linalg.norm(_physical_dir[0:3])) / _wsum > _MODE_FRAC_DOMINANT: _mode = 'AFM-fluctuation'
        else:                                                                         _mode = 'mixed'

        return {
            'Delta_frac':      Delta_frac,
            'F_bdg':           F0,
            'F_bulk':          F_bulk_x0,
            'F_cluster_corr':  F0 - F_bulk_x0,
            'eigenvectors':    _evecs,
            'eigenvalues':     _evals,
            'physical_dir':    _physical_dir,
            'lambda_min':      float(_evals_scaled[_idx_min_scaled]),
            'mode':            _mode,
            'frac_D':          float(abs(_physical_dir[4]) / _wsum),
    }

    def compute_dF_dDelta_and_d2F(self, M_channels: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_ad: float, g_Delta_bd: float, V_JT_corr: float, Q_Eg2: float, vertex_cache: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gradient and Hessian of the BdG free energy w.r.t. the _N_DELTA gap amplitudes, at fixed relative phases.
        
        Active amplitudes (|Δ_i| > _MATH_EPS): central finite differences.
        Inactive amplitudes (|Δ_i| ≈ 0): F is not even in |Δ_i| here, because U(1) invariance allows bilinear couplings such as Q·Re(Δ_s*Δ_d) that
        make F linear in |Δ_i| near the origin. A one-sided evaluation therefore returns the slope as a spurious curvature; a two-point fit
        at steps (ε, 2ε) separates slope and curvature. The step is chosen ≥ _HESS_INACTIVE_EPS to stay above eigensolver noise.

        Mixed inactive↔active derivatives are computed only for channels whose slope exceeds _HESS_LINEAR_TOL (linearly coupled to an active one);
        inactive↔inactive cross terms are omitted.
        """
        amp = np.abs(Delta_vec)
        phase = np.divide(Delta_vec, amp, out=np.ones(_N_DELTA, dtype=complex), where=amp > _MATH_EPS)
        eps = np.where(amp > _MATH_EPS, np.minimum(np.maximum(1e-5, amp * 1e-3), np.maximum(amp / 2.0, _MATH_EPS)), 1e-5)

        def F_vec(x: np.ndarray) -> float:
            return self._compute_bdg_free_energy(M_channels, Q, phase * np.abs(x), F67_vec, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_ad, g_Delta_bd, V_JT_corr, Q_Eg2, self.p.K_lattice, self.p.K_lattice_Eg2, vertex_cache)
        
        act  = np.flatnonzero(amp > _MATH_EPS)            # same threshold as the eps / phase selection above
        inac = np.flatnonzero(~(amp > _MATH_EPS))

        def F_sub(xa: np.ndarray) -> float:
            x = amp.copy()
            x[act] = xa
            return F_vec(x)

        F0 = F_vec(amp)
        grad = np.zeros(_N_DELTA, dtype=float)
        Hess = np.zeros((_N_DELTA, _N_DELTA), dtype=float)
        if act.size:
            _, g_a, H_a = _fd_grad_hessian(F_sub, amp[act], eps[act], F0=F0)
            grad[act] = g_a
            Hess[np.ix_(act, act)] = H_a

        lin = []
        for i in inac:
            e = max(float(eps[i]), _HESS_INACTIVE_EPS)
            x1 = amp.copy(); x1[i] = e
            x2 = amp.copy(); x2[i] = 2.0 * e
            F1, F2 = F_vec(x1), F_vec(x2)
            grad[i] = (4.0 * F1 - F2 - 3.0 * F0) / (2.0 * e)
            Hess[i, i] = (F2 - 2.0 * F1 + F0) / e ** 2
            if act.size and abs(grad[i]) > _HESS_LINEAR_TOL:
                lin.append((i, e))
        if lin:
            F_pm = {}
            for j in act:
                for sg in (+1.0, -1.0):
                    x = amp.copy(); x[j] = max(x[j] + sg * eps[j], 0.0)
                    F_pm[(j, sg)] = (x, F_vec(x))
            for i, e in lin:
                for j in act:
                    g_pm = []
                    for sg in (+1.0, -1.0):
                        x, Fj0 = F_pm[(j, sg)]
                        xi = x.copy(); xi[i] = e
                        g_pm.append((F_vec(xi) - Fj0) / e)
                    Hess[i, j] = Hess[j, i] = (g_pm[0] - g_pm[1]) / (2.0 * eps[j])
        return grad, Hess

    @staticmethod
    def _relative_phase_change(D_old: np.ndarray, D_new: np.ndarray) -> float:
        """
        Gauge-invariant change of the Δ vector between two iterates (eV): both vectors are referenced to the phase of the OLD vector's largest component (removing the global U(1) gauge), then
        max_i |Δ_new,i·e^{−iφ_new,ref} − Δ_old,i·e^{−iφ_old,ref}|.  Unlike max|(|Δ_new| − |Δ_old|)| it sees phase drifts such as the relaxation of an s+id relative phase.
        """
        D_old = np.asarray(D_old, dtype=complex); D_new = np.asarray(D_new, dtype=complex)
        a_old = np.abs(D_old)
        if float(a_old.max()) <= _MATH_EPS or float(np.abs(D_new).max()) <= _MATH_EPS:
            return 0.0
        i_ref = int(np.argmax(a_old))
        ref_old = D_old[i_ref] / a_old[i_ref]
        ref_new = D_new[i_ref] / abs(D_new[i_ref]) if abs(D_new[i_ref]) > _MATH_EPS else ref_old
        return float(np.max(np.abs(D_new * np.conj(ref_new) - D_old * np.conj(ref_old))))

    @staticmethod
    def _embed1(op: np.ndarray, site: int) -> np.ndarray:
        """Embed a 6x6 operator into the 4-site 1296-dim tensor-product space."""
        mats = [_I6, _I6, _I6, _I6]
        mats[site] = np.asarray(op, dtype=complex)
        return np.kron(np.kron(mats[0], mats[1]), np.kron(mats[2], mats[3]))

    @staticmethod
    def _embed2(opA: np.ndarray, siteA: int, opB: np.ndarray, siteB: int) -> np.ndarray:
        """Embed a two-site operator opA⊗opB at sites siteA, siteB."""
        mats = [_I6, _I6, _I6, _I6]
        mats[siteA] = np.asarray(opA, dtype=complex)
        mats[siteB] = np.asarray(opB, dtype=complex)
        return np.kron(np.kron(mats[0], mats[1]), np.kron(mats[2], mats[3]))

    def _blend_update_candidates(self, old, newton, fixed_point, alpha=None, beta=None):
        alpha = _MIXING if alpha is None else alpha
        beta = _ALPHA_HF if beta is None else beta

        d_newton = np.asarray(newton) - np.asarray(old)
        d_fixed = np.asarray(fixed_point) - np.asarray(old)
        direction = (1.0 - beta) * d_fixed + beta * d_newton
        cand = np.asarray(old) + alpha * direction
        if np.iscomplexobj(cand):
            # complex gap vector: clip the modulus, keep the phase (np.clip on complex numbers orders lexicographically and would zero every component with Re < 0)
            amp = np.abs(cand)
            phase = np.divide(cand, amp, out=np.ones(cand.shape, dtype=complex), where=amp > _MATH_EPS)
            return np.clip(amp, 0.0, 1.0) * phase
        return np.clip(cand, 0.0, 1.0)

    def _project_kick_from_hessian(self, hessian_result: Dict, kick_scale: float, sign_ref: Optional[float] = None) -> np.ndarray:
        """
        Convert a compute_hessian() result into a signed step vector [ΔM_Γ6, ΔM_Γ7a, ΔM_Γ7b, ΔQ, ΔΔ] along the most-unstable direction, scaled by kick_scale.
        
        sign_ref: if given, flip physical_dir so its Q-component (index 3) matches sign_ref, preserving the external initial_Q/Q sign convention.
        
        step[4] is the linear eigenvector projection of Δ. It is used directly for saddle-escape kicks, but NOT for cold-start seed sizing:
        near weak-coupling SC instabilities, Δ ~ exp(-1/λ), so the linear projection can severely underestimate the physical seed magnitude.
        """
        edir = np.array(hessian_result['physical_dir'], dtype=float)
        if sign_ref is not None and sign_ref * edir[3] < 0:
            edir = -edir
        return kick_scale * edir

    def _bcs_ratio_2D(self, sc_result: dict, Tc: float, doping: Optional[float] = None, Q_Eg2: float = 0.0, refine: bool = True) -> Dict:
        """
        Spectral FS gap of a converged SC state: 2Δ_ref/kTc is the measured coupling ratio. Empirical cuprate range: 4–8

        Δ_gap(k_F) = min_n |E_n(k_F)|, refined by a 1D line search along the local FS normal starting at the ξ=0 crossing.
        The FS sample from _get_fs_points is a thermal shell (|ξ| ≲ 6 kT), not the exact ξ=0 surface, so the line search is
        needed to extract the physical gap. Thresholds _BCS_RATIO_STRONG/_BCS_RATIO_EXOTIC are calibrated to the weak-coupling
        s-wave value (2π/e^γ); no shape prediction is applied — a pure d-wave state will land in the 'very-strong' band despite
        being weak-coupling, because the analytical d-wave weak-coupling value is 4.28 (not 3.53).
        """
        empty = {'Delta_max': 0.0, 'Delta_min': 0.0, 'Delta_ref': 0.0, 'nodal': False, 'dominant': None,
                'coupling_regime': "unknown", 'T_state': float(self.kT), 'gap_saturated': False}
        if sc_result is None:
            return empty
        Delta_vec = np.asarray(sc_result.get('Delta_vec', np.zeros(_N_DELTA, dtype=complex)), dtype=complex)
        if Delta_vec.size != _N_DELTA or float(np.max(np.abs(Delta_vec))) < _MATH_EPS:
            return empty
        if doping is None:
            doping = float(sc_result['target_doping'])

        M    = np.asarray(sc_result['M'])
        Q    = float(sc_result['Q'])
        mu   = float(sc_result['mu'])
        n_ks = float(sc_result['n_kspace'])
        F67  = np.asarray(sc_result['F67_vec'])
        g_t, g_J = self.p.get_gutzwiller_factors(doping)

        fs_pts, vF, _, fs_w = self._get_fs_points(M, Q, n_ks, mu, g_t, g_J, store_cache=False)
        if len(fs_pts) == 0:
            return empty
        vbdg = self._get_vbdg()

        def _gaps_at(kpts: np.ndarray) -> np.ndarray:
            H = vbdg._build_H_stack(kpts, M, Q, Delta_vec, F67, n_ks, mu, g_t, g_J, Q_Eg2=Q_Eg2)
            return np.min(np.abs(np.linalg.eigvalsh(H)), axis=1)

        # Farthest-point subsampling for uniform FS coverage
        # Sorting by polar angle about Γ and cutting at equal cumulative chord length is NOT arc-length sampling here: neighbouring points in polar order jump between branches of the FS
        n_sub = min(len(fs_pts), _GAP_N_SUB)
        _dk   = fs_pts[:, None, :] - fs_pts[None, :, :]
        _dk   = (_dk + np.pi) % (2.0 * np.pi) - np.pi                                   # periodic k-space distance
        _D    = np.sqrt(np.sum(_dk ** 2, axis=2))
        _sel  = [int(np.argmin(np.arctan2(fs_pts[:, 1], fs_pts[:, 0])))]                # deterministic start
        _dmin = _D[_sel[0]].copy()
        while len(_sel) < n_sub:
            _nxt = int(np.argmax(_dmin))
            _sel.append(_nxt)
            _dmin = np.minimum(_dmin, _D[_nxt])
        sub = np.array(_sel, dtype=int)

        pts_s = fs_pts[sub]
        v_s = np.linalg.norm(vF[sub], axis=1) if vF is not None else np.zeros(len(sub))
        n_s = (vF[sub] / np.maximum(v_s, 1e-12)[:, None]) if vF is not None else np.zeros_like(pts_s)
        gaps = _gaps_at(pts_s)

        # Line search at the ξ=0 crossing along the local FS normal
        valid = np.ones(len(sub), dtype=bool)
        if refine and vF is not None:
            e_n = np.linalg.eigvalsh(vbdg._build_H_stack(pts_s, M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_ks, mu, g_t, g_J, Q_Eg2=Q_Eg2)[:, :_N_BDG // 2, :_N_BDG // 2])
            xi_n = e_n[np.arange(len(sub)), np.argmin(np.abs(e_n), axis=1)]            # normal-state band energy closest to E_F

            # Physical line-search half-window (eV → k): |Δ|/v_F scale, floored at _GAP_WIN.
            win_eV = max(_GAP_WIN, 3.0 * float(np.max(np.abs(Delta_vec))))

            for i in range(len(sub)):
                if v_s[i] < _GAP_VF_MIN:
                    valid[i] = False   # flat band: no FS normal → skip
                    continue
                # Extrapolate to the true ξ=0 crossing along the local normal:  ξ(k_F) ≈ ξ(k_sample) + v_F · s  = 0  →  s = -ξ/v_F
                s0 = float(np.clip(-xi_n[i] / v_s[i], -_GAP_SCENTER_MAX, _GAP_SCENTER_MAX))
                W = float(np.clip(win_eV / v_s[i], 0.02, _GAP_WIN_KMAX))
                line = pts_s[i][None, :] + (s0 + np.linspace(-W, W, _GAP_NLINE))[:, None] * n_s[i][None, :]
                gaps[i] = min(gaps[i], float(np.min(_gaps_at(line))))

        if not valid.any():
            return empty
        fs_pts = pts_s[valid]
        fs_w = np.asarray(fs_w)[sub][valid]
        gaps = gaps[valid]

        # Gap statistics
        weights = np.asarray(fs_w, dtype=float)
        if (not np.all(np.isfinite(weights))) or weights.sum() <= 0.0:
            weights = np.ones_like(gaps)
        D_max = float(np.max(gaps))
        D_min = float(np.min(gaps))
        top = gaps >= np.quantile(gaps, 1.0 - _GAP_REF_TOP_FRAC)
        D_ref = float(np.average(gaps[top], weights=weights[top])) if (top.any() and weights[top].sum() > 0.0) else D_max
        amp_irrep = np.sqrt(np.abs(Delta_vec[_IDX_7A_ARR])**2 + np.abs(Delta_vec[_IDX_7B_ARR])**2)
        dominant = _IRREP_NAMES[int(np.argmax(amp_irrep))]
        nodal = bool(D_min < _GAP_NODAL_ANISO * D_max)
        bfs_frac = float(np.average(gaps < _GAP_BFS_TOL, weights=weights))

        # Measured coupling ratio and regime
        gap_saturated = bool(Tc > 1e-8 and self.kT <= 0.25 * Tc)
        ratio_2D = (2.0 * D_ref / Tc) if (Tc > 1e-8 and D_ref > 0.0) else 0.0
        if bfs_frac > 0.5:
            coupling_regime = f"gapless ({100*bfs_frac:.0f}% BFS)"
        elif not gap_saturated:
            coupling_regime = "undetermined (T not << Tc)"
        elif ratio_2D < _BCS_RATIO_STRONG:
            coupling_regime = "BCS-like"
        elif ratio_2D < _BCS_RATIO_EXOTIC:
            coupling_regime = "very-strong"
        else:
            coupling_regime = "exotic"

        return {'Delta_max': D_max, 'Delta_min': D_min, 'Delta_ref': D_ref, 'nodal': nodal, 'dominant': dominant,
                'coupling_regime': coupling_regime, 'T_state': float(self.kT), 'gap_saturated': gap_saturated}

    def compute_Tc_linearized_normal(self, target_doping: float, T_grid: Optional[np.ndarray] = None, eps: float = _TC_LIN_DELTA_EPS) -> Dict:
        """Second-order instability temperature Tc of the model's own gap equation linearised around Δ = 0 (pure BdG map: no kernel steering, caps or EMA):
            Δ_new[b][i] = Σ_j V_irrep[i,j]·√(g_b,i/g_b,j)·F_raw_g[b][j](Δ) = (B_b F_raw_g[b])[i],  b ∈ {7a, 7b};   B: 84×84 block-diagonal (the gap equation is branch-diagonal);
            R(T)[j,k] = ∂F_raw_g[k]/∂Δ[j] at Δ = 0 by one-sided FD with step eps ≪ k_B T;   Λ(T) = max|eig(B·R)|;   Tc = root of Λ(T) = 1.
        Caveats: M and μ are FROZEN at their call values (exact for Tc ≪ T_Néel; the model's Tc ~ 230 K is not small against the AFM scale); use a fine T_grid near the crossing (linear interpolation underestimates Tc where Λ ~ ln(1/T)).
        B is the UN-projected map: it does not contain the attractive-subspace constraint of the SCF (_attractive_projector). The two agree only if the leading eigenvector of B·R lies in the attractive subspace
        (cache key 'proj_cut' ≈ 0); otherwise Tc(lin) belongs to the unconstrained gap equation and the SCF onset can differ (A/B test: _ATTRACTIVE_PROJECTION)."""
        # ── 1. Normal-state SCF at Q = 0 ────────────────────────────────────
        zero = np.zeros(_N_DELTA, dtype=complex)
        (M, _, _, n, mu, g_t, g_J, _, _, _) = self.refine_M_state(target_doping, 0.0, zero, zero, damping=0.45, verbose=False)

        # ── 2. RPA vertex cache → V_irrep ───────────────────────────────────
        Gm, _, V_JT_eff, V_JT_corr, V_cap = self._make_vertex_params(target_doping, M, 0.0, zero, zero, n, mu, g_t, g_J)
        J_A1g = self.p.exchange_channels(0.0, g_J)[0]
        cache = self.compute_pairing_kernel_and_build_cache(M, 0.0, n, mu, g_t, g_J, J_A1g, Gm, V_JT_eff, V_JT_corr, V_cap)
        V_irrep = cache['V_irrep']
        lambda_lin_max_FS = float(cache['lambda_lin_max'])

        # ── 3. Linearized gap equation on T_grid ────────────────────────────
        vbdg  = self._get_vbdg()
        zeroF = np.zeros(_N_DELTA, dtype=complex)

        if T_grid is None:
            T_grid = np.geomspace(0.5 * self.kT, 0.15, 14)
        T_grid = np.asarray(T_grid, dtype=float)
        if eps > _TC_LIN_EPS_TMIN_FRAC * float(np.min(T_grid)):
            warnings.warn(
                f"compute_Tc_linearized_normal: eps={eps:.2e} is not << T_min={T_grid.min():.2e}; the FD linearization may be inaccurate.",
                RuntimeWarning,
            )

        g_Ds, g_Da, g_Db = self.estimate_gutzwiller_factors_occupation_based(M, 0.0, n, mu, g_t, g_J)
        g_a = _gutzwiller_irrep_vector(g_Ds, g_Da, g_t)
        g_b = _gutzwiller_irrep_vector(g_Ds, g_Db, g_t)

        B = np.zeros((_N_DELTA, _N_DELTA))
        B[np.ix_(_IDX_7A_ARR, _IDX_7A_ARR)] = V_irrep * np.sqrt(np.outer(g_a, 1.0 / np.maximum(g_a, 1e-12)))
        B[np.ix_(_IDX_7B_ARR, _IDX_7B_ARR)] = V_irrep * np.sqrt(np.outer(g_b, 1.0 / np.maximum(g_b, 1e-12)))

        # R[it, k, j] = ∂F_raw_g[k] / ∂Δ[j]  at  T = T_grid[it]
        R = np.zeros((len(T_grid), _N_DELTA, _N_DELTA))
        for j in range(_N_DELTA):
            D = np.zeros(_N_DELTA, dtype=complex)
            D[j] = eps
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M, 0.0, D, zeroF, n, mu, g_t, g_J, out=vbdg._H_stack))
            for it, T in enumerate(T_grid):
                _, F_raw_g = _compute_F67_pair_amplitudes(self.k_weights, self.Phi_irrep_k, ev, ec, float(T), D, g_Ds, g_Da, g_Db)
                R[it, :, j] = np.real(F_raw_g) / eps

        # Eigenvalues of B·R(T)
        lam_abs = np.zeros(len(T_grid))
        lam_re  = np.zeros(len(T_grid))
        vec0    = None
        for it in range(len(T_grid)):
            w, U = np.linalg.eig(B @ R[it])
            idx = int(np.argmax(np.abs(w)))
            lam_abs[it] = float(abs(w[idx]))
            lam_re[it]  = float(w[idx].real)
            if it == 0:
                vec0 = np.real(U[:, idx])

        # ── 4. Tc = root of Λ(T) = 1 ────────────────────────────────────────
        if lam_abs[0] < 1.0:
            status, Tc = 'below', float('nan')
        elif lam_abs[-1] >= 1.0:
            status, Tc = 'above', float('nan')
        else:
            status = 'inside'
            i = int(np.flatnonzero(
                (lam_abs[:-1] >= 1.0) & (lam_abs[1:] < 1.0))[-1])
            Tc = float(T_grid[i]
                    + (lam_abs[i] - 1.0)
                        / max(lam_abs[i] - lam_abs[i + 1], 1e-30)
                        * (T_grid[i + 1] - T_grid[i]))

        complex_mode = bool(np.max(np.abs(lam_abs - np.abs(lam_re))) > _TC_LIN_COMPLEX_TOL * max(float(np.max(lam_abs)), 1e-30))
        names = ([_DELTA_NAMES[i] for i in np.argsort(-np.abs(vec0))[:4]]
                if vec0 is not None else [])

        return {
            'Tc':                Tc,
            'T':                 T_grid,
            'Lambda':            lam_abs,
            'Lambda_real':       lam_re,
            'status':            status,
            'vec':               vec0,
            'leading_channels':  names,
            'complex_mode':      complex_mode,
            'lambda_lin_max_FS': lambda_lin_max_FS,
            'M_normal':          M,
            'mu_normal':         mu,
            'n_normal':          n,
            'frozen_M_mu':       True,
        }

    def compute_Tc_thermodynamic(self, doping: float, sc_result: dict, T_min: float = 1e-4, T_max: float = 0.20, n_scan: int = 15, n_bisect: int = 10) -> dict:
        """
        Thermodynamic Tc via warm-start upward temperature scan. For the SC-triggered JT mechanism the effective Landau potential
         F_eff(Δ) = a(T)Δ² + [b−γ²/(2K_eff)]Δ⁴+… can have a negative quartic term; the transition is then first-order and Tc* occurs
         where a(T*)>0.  Cooling from Δ≈0 misses this and returns Tc ≪ Tc*.

        Algorithm: (1) warm-start heating from the T≈0 SC+JT basin; (2) at each T compare F_SC vs F_name;
        (3) Tc = max(spinodal collapse, thermodynamic crossing); (4) bisection between last SC-wins and first NM-wins T.
        """
        Delta_vec0 = _sc_gap_amplitude(sc_result)
        Delta0 = float(np.linalg.norm(np.asarray(Delta_vec0)))
        M0 = float(np.asarray(sc_result['M'])[0])   # leading (Γ6) channel

        if (not sc_result.get('converged', False) or Delta0 < _DELTA_ABS_FLOOR):
            return {
                'Tc': 0.0, 'Tc_spinodal': 0.0, 'T_cross': 0.0, 'T_spinodal_cool': float('nan'),
                'transition_order': 'unknown', 'Q_at_Tc': 0.0, 'Delta_jump': 0.0, 'hysteresis': 0.0, 'history': []
            }

        def _eval_sc_basin(solver: 'RMFT_Solver', seed_D: np.ndarray, seed_F67: np.ndarray) -> tuple:
            try:
                res = solver.solve_self_consistent(doping, initial_Delta=seed_D, initial_F67=seed_F67)
                D_vec = np.asarray(_sc_gap_amplitude(res), dtype=complex)
                Q_eff = float(res['Q'])
                M_eff = float(np.asarray(res['M'])[0])
                F_sc = float(res['F_bdg'])
                return D_vec, Q_eff, M_eff, F_sc
            except Exception as _e:
                _scf_log("TC-THERMO", f"⚠ SC-basin SCF raised {type(_e).__name__}: {_e} -- the point is scored as 'SC loses' (F=1e30)")
                return np.asarray(seed_D, dtype=complex), 0.0, M0, 1e30

        def _eval_normal_basin(solver: 'RMFT_Solver') -> tuple:
            """Returns (F_name, converged) for the normal-state basin."""
            try:
                res = solver.solve_self_consistent(doping, initial_Delta = np.zeros(_N_DELTA, dtype=complex), initial_F67=np.zeros(_N_DELTA, dtype=complex))
                return float(res['F_bdg'])
            except Exception as _e:
                _scf_log("TC-THERMO", f"⚠ normal-basin SCF raised {type(_e).__name__}: {_e} -- F=1e30")
                return 1e30

        # ── Helper: single upward heating pass — crossing + spinodal in one scan ─
        def _find_crossing_and_spinodal(T_vals: list, seed: dict) -> tuple:
            """
            Single pass over ascending T_vals.  Per T-point: one SC-basin SCF and one
            NM-basin SCF (same cost as before, but only once per point instead of twice).

            Crossing (F_SC = F_name): interpolated between last sc_wins and first sc_loses.
            After the crossing the seed keeps advancing so hysteresis remains trackable.

            Spinodal: recorded at first collapse; pass terminates immediately after.
            When collapse precedes crossing (2nd-order like), T_cross = 0.
            """
            history      = []
            _seed        = seed.copy()
            T_cross      = 0.0;  D_cross      = 0.0;  Q_cross      = 0.0
            T_spinodal   = 0.0;  D_spinodal   = 0.0;  Q_spinodal   = 0.0
            _cross_found = False

            for i, T in enumerate(T_vals):
                s = self._clone_solver_at_T(T)
                D_vec, Q_eff, M_eff, F_sc = _eval_sc_basin(s, _seed['Delta_vec'], _seed['F67_vec'])
                F_nm = _eval_normal_basin(s)

                D_eff = float(np.linalg.norm(D_vec))  
                collapsed = D_eff < _DELTA_ABS_FLOOR
                sc_wins = (not collapsed) and (F_sc < F_nm)
                history.append((T, D_eff, Q_eff, F_sc, F_nm, sc_wins))

                # ── spinodal: SC basin collapsed ─────────────────────────────────────
                if collapsed and T_spinodal == 0.0:
                    T_spinodal = T
                    if i > 0:
                        D_spinodal = history[-2][1]
                        Q_spinodal = history[-2][2]
                    break   # no SC state to track further

                # ── crossing: F_SC first exceeds F_NM ───────────────────────────
                if not _cross_found:
                    if sc_wins:
                        D_cross = D_eff
                        Q_cross = Q_eff
                    elif i > 0:
                        prev    = history[-2]
                        dF_prev = prev[3] - prev[4]   # F_sc - F_nm at T_{i-1}
                        dF_curr = F_sc - F_nm
                        denom_interp = dF_curr - dF_prev
                        T_cross = (T_vals[i-1] - dF_prev * (T - T_vals[i-1]) / denom_interp
                                   if abs(denom_interp) > 1e-30 else T_vals[i-1])
                        D_cross = prev[1]
                        Q_cross = prev[2]
                        _cross_found = True

                # advance warm seed while SC basin is alive
                _seed['M'] = M_eff
                _seed['Q'] = Q_eff
                _seed['Delta_vec'] = D_vec 
            return T_cross, D_cross, Q_cross, T_spinodal, D_spinodal, Q_spinodal, history

        def _find_spinodal_cooling(T_vals: list) -> tuple:
            """
            Cool from normal state to find where SC first appears.
            Returns (T_spinodal_cool, Delta_at_appearance, Q_at_appearance, history).
            """
            history = []
            T_spinodal = 0.0
            D_appear = 0.0
            Q_appear = 0.0

            # Start from normal state at highest T
            prev_D = 0.0

            for i, T in enumerate(reversed(T_vals)):  # cool down
                s = self._clone_solver_at_T(T)
                # Cold start from normal state (Δ=0)
                try:
                    res = s.solve_self_consistent(doping, initial_Delta = np.zeros(_N_DELTA, dtype=complex), initial_F67=np.zeros(_N_DELTA, dtype=complex))
                    D_eff = float(np.linalg.norm(np.asarray(_sc_gap_amplitude(res))))
                    Q_eff = float(res['Q'])
                    converged = res['converged']
                except Exception as _e:
                    _scf_log("TC-COOL", f"⚠ SCF at T={T*1000:.2f} meV raised {type(_e).__name__}: {_e} -- counted as Δ = 0")
                    D_eff = 0.0
                    Q_eff = 0.0
                    converged = False

                history.append((T, D_eff, Q_eff, converged))

                if D_eff > _DELTA_ABS_FLOOR and prev_D < _DELTA_ABS_FLOOR and i > 0:
                    # SC just appeared
                    T_spinodal = T
                    D_appear = D_eff
                    Q_appear = Q_eff
                    break

                prev_D = D_eff
            return T_spinodal, D_appear, Q_appear, history

        # ── 1. Build temperature grid ────────────────────────────────────────────
        T_vals = list(np.geomspace(T_min, T_max, n_scan + 1))

        # Initial seed from low-T SC result
        seed_init = {'M': M0, 'Delta_vec': Delta_vec0, 'F67_vec': sc_result['F67_vec']}

        # ── 2. Single upward heating pass: crossing + spinodal together ──────────
        (T_cross, D_cross, Q_cross, T_spinodal, D_spinodal, Q_spinodal, heating_history) = _find_crossing_and_spinodal(T_vals, seed_init)

        # ── 3. Ginzburg-Landau extrapolation for precise Tc near second-order transition ──
        def _gl_extrapolate_Tc(history: list) -> float:
            # (non-zero Delta) to include all alive-SC points regardless of whether F_SC < F_NM, since GL fit needs the Δ(T) collapse shape.
            valid = [h for h in history if h[1] > _GL_DELTA_MIN]
            if len(valid) < _GL_MIN_PTS:
                return float('nan')
            pts = valid[-_GL_MAX_PTS:]
            T_v  = np.array([h[0] for h in pts])
            D2_v = np.array([h[1]**2 for h in pts])
            a, b = np.polyfit(T_v, D2_v, 1)
            if a >= 0.0:   # Δ² increasing with T: unphysical, discard
                return float('nan')
            return float(-b / a)

        _Tc_GL = _gl_extrapolate_Tc(heating_history)
        # Apply GL refinement only when transition is near-second-order: for first-order transitions Δ² vanishes discontinuously, making linear extrapolation unreliable.
        # Guard: last-stable Δ must be < 15% of Δ₀ to qualify as continuous collapse.
        _gl_delta_jump_ok = (D_spinodal / max(Delta0, 1e-9)) < _GL_SPINODAL_JUMP
        if (np.isfinite(_Tc_GL) and T_spinodal > 0.0
                and abs(_Tc_GL - T_spinodal) < _GL_TC_MARGIN * T_max
                and _gl_delta_jump_ok):
            T_spinodal = _Tc_GL   # refine coarse grid point with GL fit

        # ── 4. Optional: find spinodal from cooling (hysteresis check) ──────────
        T_spinodal_cool = float('nan')
        hysteresis = 0.0
        if T_cross > 0.0:
            # Use a finer grid around expected transition
            T_cool_vals = list(np.geomspace(T_min, min(T_cross * 1.2, T_max), n_scan + 1))
            T_spinodal_cool, D_cool, Q_cool, cool_history = _find_spinodal_cooling(T_cool_vals)
            if np.isfinite(T_spinodal_cool):
                hysteresis = T_spinodal_cool - T_spinodal

        # ── 5. Determine transition order and Tc ─────────────────────────────────
        def _snap_to_crossing():
            return T_cross, D_cross, Q_cross, D_cross

        def _snap_to_spinodal():
            return T_spinodal, D_spinodal, Q_spinodal, 0.0

        if T_cross > 0.0 and T_spinodal > 0.0:
            if T_cross > T_spinodal * (1.0 + _THRESHOLD_RATIO):
                transition_order = 'first-order'
                Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_crossing()
            elif T_spinodal > T_cross * (1.0 + _THRESHOLD_RATIO):
                # This should not happen physically, but handle it
                transition_order = 'second-order'
                Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_spinodal()
            else:
                # T_cross and T_spinodal are close → weakly first-order or second-order
                # Use the higher temperature as Tc (thermodynamic stability)
                if T_cross > T_spinodal:
                    Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_crossing()
                    transition_order = 'weakly-first-order' if D_cross > 0.1 * Delta0 else 'second-order'
                else:
                    Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_spinodal()
                    transition_order = 'second-order'
        elif T_cross > 0.0:
            # Only crossing found, spinodal not reached within scan range
            Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_crossing()
            transition_order = 'first-order' if D_cross > 0.1 * Delta0 else 'unknown'
        elif T_spinodal > 0.0:
            # Only spinodal found
            Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_spinodal()
            transition_order = 'second-order'
        else:
            # No transition found in scan range
            Tc = T_max
            Delta_at_Tc = 0.0
            Q_at_Tc = 0.0
            Delta_jump = 0.0
            transition_order = 'unknown'

        # ── 6. Compute ratio and combine history ─────────────────────────────────
        gap_info = self._bcs_ratio_2D(sc_result, Tc, doping)

        # heating_history is the single shared scan; label each row for clarity.
        combined_history = [('heating',) + h for h in heating_history]

        return {
            'Tc':               Tc,
            'T_cross':          T_cross,
            'Tc_spinodal':      T_spinodal,
            'T_spinodal_cool':  T_spinodal_cool,
            'transition_order': transition_order,
            'Q_at_Tc':          Q_at_Tc,
            'gap_info':         gap_info,
            'Delta_jump':       Delta_jump,
            'hysteresis':       hysteresis,
            'history':          combined_history,
        }

    def _get_fs_points(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, store_cache: bool = True, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Normal-state (Δ=0) Fermi-surface sample: (points, Hellmann–Feynman velocities [diagnostic only], full-grid indices, integration weights).

        Weights: w_k = f(1−f)/kT, averaged over the orbit group (no 1/v_F singularity), × dA/(2π)², renormalised so that the truncated sample keeps the thermal weight of the whole shell.
        Orbit group: D4h if the normal-state spectrum is C4-symmetric (max|E(k)−E(Rk)| < _C4_SPECTRAL_TOL), else D2h; decided from E(k) alone (h_breaks_C4, the parameter-based prediction, is a diagnostic).
        M ∥ z does not break C4; a static B1g field and Q ≠ 0 do. Whole orbits are taken in order of their thermal score and tied orbits enter/leave together, so the sample is closed under
        k→−k, kx→−kx, ky→−ky (and kx↔ky in the D4h regime) and every non-A1g moment cancels exactly. The kernel itself uses the D2h×T_Q character group and does not assume C4.

        T_Q fundamental domain: the sample lives in the diamond |kx|+|ky| ≤ π, a fundamental domain of T_Q: k→k+(π,π). The sector kernel adds the T_Q images V(k_i − T_Q k_j) explicitly (with the
        T_Q parity of the gap function), so the interior must NOT be T_Q-closed. T_Q maps the boundary |kx|+|ky| = π onto itself in pairs (no fixed point): it is kept with weight ½ per point and both
        partners are always selected together (the D2h boundary orbits (a,b)/(b,a) form one atomic orbit), so every pair counts exactly once. The boundary holds the nodal points (π/2,π/2) of the
        AFM hole pockets (φ_d = 0, φ_px = 1 there): the uploaded version dropped it altogether, which removed that part of the Fermi surface from the channels unevenly.
        """
        _M_key = tuple(np.asarray(M, dtype=float).ravel())
        _cache_key_vals = (_M_key, float(Q), float(n_kspace), float(mu), float(g_t), float(g_J), int(self.thermal_fs_nat_size))
        vbdg = self._get_vbdg()

        # ---- Cache lookup ----
        if self._fs_cache_dict is not None:
            for key, val in self._fs_cache_dict.items():
                if (np.allclose(key[0], _M_key, atol=_FS_CACHE_TOL, rtol=0.0)
                        and all(abs(float(key[i]) - float(_cache_key_vals[i])) < _FS_CACHE_TOL for i in range(1, 6))
                        and int(key[6]) == int(self.thermal_fs_nat_size)):
                    return val

        # ---- 1. Normal-state BdG spectrum ----
        ev_all, ec_all = np.linalg.eigh(vbdg._build_H_stack(self.k_points, M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, out=vbdg._H_stack))
        ev_pos = np.where(ev_all > 0.0, ev_all, np.inf)
        min_band_idx = np.argmin(ev_pos, axis=1)
        Emin = ev_pos[np.arange(len(ev_pos)), min_band_idx]

        k_all = np.asarray(self.k_points)
        N_k_total = len(k_all)

        # ---- 2. Diamond |kx|+|ky| ≤ π = T_Q fundamental domain: interior weight 1, boundary (T_Q-paired) weight ½ ----
        _r1 = np.abs(k_all[:, 0]) + np.abs(k_all[:, 1])
        on_bdy_k = np.abs(_r1 - np.pi) < _BDY_TOL
        mbz_mask_k = _r1 < np.pi + _BDY_TOL
        bdy_factor_k = np.where(on_bdy_k, 0.5, 1.0)

        # ---- 3. Symmetry-regime determination (D4h vs D2h) ----
        d_B1g_static = abs(float(getattr(self.p, "Delta_B1g_static", 0.0)))
        M_norm = float(np.linalg.norm(np.atleast_1d(np.asarray(M, dtype=float))))
        Q_norm = abs(float(Q))

        # ---- 4. Thermal shell with orbit-symmetrised occupation ----
        # NO D2h canonical filter: the D2h subgroup is smaller than the true Q=0 point group D4h, and truncating to it drops the C4/diagonal-mirror images that are needed for exact cancellation of non-A1g functions.
        # The D2h subgroup is only the residual symmetry of a nonzero B1g order parameter, not of the underlying Hamiltonian at Q=0.
        f_all = _fermi_function(Emin, self.kT)
        therm_raw = f_all * (1.0 - f_all) / self.kT

        # Grid-index → array-index map so we can look up symmetry images on the full BZ grid.
        grid_step = 2.0 * np.pi / _NK
        ix_all = np.round((k_all[:, 0] + np.pi) / grid_step).astype(int) % _NK
        iy_all = np.round((k_all[:, 1] + np.pi) / grid_step).astype(int) % _NK
        idx_map = np.empty((_NK, _NK), dtype=np.int64)
        idx_map[ix_all, iy_all] = np.arange(N_k_total)

        ix_n = (-ix_all) % _NK
        iy_n = (-iy_all) % _NK

        # ---- 5. Symmetry regime: is C4 an exact symmetry of the normal-state spectrum? ----
        # Decided from E(k) alone: with D2h-only orbits the C4-partner orbits (a,b)/(b,a) of a C4-symmetric spectrum tie, and the n_target truncation would keep one and drop the other (sample not C4-closed).
        # The explicit C4-breakers (static B1g field, Q ≠ 0) are reported as `h_breaks_C4` (diagnostic): they act on the orbital part of the pair vertex, not on the point set; c4_spec_dev catches a really C4-asymmetric spectrum.
        ev_rot = ev_all[idx_map[iy_n, ix_all]]                          # spectrum at R·k,  R: (kx, ky) → (−ky, kx)
        c4_spec_dev = float(np.max(np.abs(ev_all - ev_rot)))
        tq_spec_dev = float(np.max(np.abs(ev_all - ev_all[idx_map[(ix_all + _NK // 2) % _NK, (iy_all + _NK // 2) % _NK]])))   # T_Q-invariance of the normal-state spectrum (needed for T_Q-consistent weights)
        h_breaks_C4 = bool(d_B1g_static > _SYM_THRESH or Q_norm > _SYM_THRESH)
        breaks_C4 = c4_spec_dev > _C4_SPECTRAL_TOL
        symmetry_group = "D2h" if breaks_C4 else "D4h"

        if symmetry_group == "D4h":
            orbit_idx = np.stack([
                idx_map[ix_all, iy_all], idx_map[ix_n,   iy_all],
                idx_map[ix_all, iy_n  ], idx_map[ix_n,   iy_n  ],
                idx_map[iy_all, ix_all], idx_map[iy_n,   ix_all],
                idx_map[iy_all, ix_n  ], idx_map[iy_n,   ix_n  ],
            ], axis=1)
        else:
            orbit_idx = np.stack([
                idx_map[ix_all, iy_all], idx_map[ix_n,   iy_all],
                idx_map[ix_all, iy_n  ], idx_map[ix_n,   iy_n  ],
            ], axis=1)

        therm_sym = therm_raw[orbit_idx].mean(axis=1)
        near_fs = therm_sym > (_FS_THERMAL_THRESHOLD / self.kT)
        shell_mask = near_fs & mbz_mask_k
        fs_idx_shell = np.flatnonzero(shell_mask)

        if len(fs_idx_shell) == 0:
            empty_pts = np.empty((0, 2), dtype=float)
            return empty_pts, empty_pts.copy(), np.empty(0, dtype=int), np.empty(0, dtype=float)

        # ---- 6. Orbit grouping on the shell (canonical wedge, grid-index hashing) ----
        k_shell = k_all[fs_idx_shell]

        if symmetry_group == "D4h":
            kx_c = np.maximum(np.abs(k_shell[:, 0]), np.abs(k_shell[:, 1]))
            ky_c = np.minimum(np.abs(k_shell[:, 0]), np.abs(k_shell[:, 1]))
        else:
            kx_c = np.abs(k_shell[:, 0])
            ky_c = np.abs(k_shell[:, 1])

        kx_q = np.round(kx_c / grid_step).astype(np.int64)
        ky_q = np.round(ky_c / grid_step).astype(np.int64)

        orbit_of_shell = {}
        for j, i_shell in enumerate(fs_idx_shell):
            oid = (int(kx_q[j]), int(ky_q[j]))
            orbit_of_shell.setdefault(oid, []).append(int(i_shell))

        orbit_scores = [(float(np.mean(therm_sym[idx_list])), oid, idx_list)
                        for oid, idx_list in orbit_of_shell.items()]
        orbit_scores.sort(key=lambda x: -x[0])

        # ---- 7. Whole-orbit selection (symmetry-closed subset of the shell) ----
        # Orbits with equal scores are degenerate by a symmetry of the spectrum (or accidentally) and their order is arbitrary, so a tie block is ATOMIC (kept or dropped as a whole); without ties every block is one orbit:
        # orbits are taken in score order until the next does not fit.
        n_target = min(int(self.thermal_fs_nat_size), len(fs_idx_shell))
        tie_blocks = []
        for _item in orbit_scores:
            if tie_blocks and abs(_item[0] - tie_blocks[-1][0][0]) <= _FS_ORBIT_TIE_RTOL * max(abs(tie_blocks[-1][0][0]), 1e-300):
                tie_blocks[-1].append(_item)
            else:
                tie_blocks.append([_item])

        if verbose:
            _block_sizes = [sum(len(_il) for (_s, _o, _il) in _b) for _b in tie_blocks]
            _scf_log("FS-DEBUG",
                    f"  N_shell={len(fs_idx_shell)}  n_target={n_target}  "
                    f"n_tie_blocks={len(tie_blocks)}  first5_block_sizes={_block_sizes[:5]}")

        selected_full = []
        for _block in tie_blocks:
            idx_list = [_i for (_s, _o, _il) in _block for _i in _il]
            if len(selected_full) + len(idx_list) > n_target and selected_full:
                if len(selected_full) < 0.5 * n_target:
                    selected_full.extend(idx_list)
                break
            selected_full.extend(idx_list)
        if not selected_full:
            # Fallback: keep at least the highest-scoring full orbit.
            selected_full = list(orbit_scores[0][2])

        fs_idx = np.asarray(sorted(selected_full), dtype=int)
        fs_pts = k_all[fs_idx]
        bands_sel = min_band_idx[fs_idx]
        evecs_sel = ec_all[fs_idx]
        n_sel = len(fs_pts)

        # ---- 8. Hellmann-Feynman Fermi velocities (diagnostic only) ----
        dk = min(1e-3, max(1e-5, grid_step / 10.0))
        kx, ky = fs_pts[:, 0], fs_pts[:, 1]
        H_buf_p = np.empty((n_sel, _N_BDG, _N_BDG), dtype=complex)
        H_buf_m = np.empty((n_sel, _N_BDG, _N_BDG), dtype=complex)

        Hp_x = vbdg._build_H_stack(np.column_stack((kx + dk, ky)), M, Q,
                                np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex),
                                n_kspace, mu, g_t, g_J, out=H_buf_p).copy()
        Hm_x = vbdg._build_H_stack(np.column_stack((kx - dk, ky)), M, Q,
                                np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex),
                                n_kspace, mu, g_t, g_J, out=H_buf_m).copy()
        dH_dkx = (Hp_x - Hm_x) / (2.0 * dk)

        Hp_y = vbdg._build_H_stack(np.column_stack((kx, ky + dk)), M, Q,
                                np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex),
                                n_kspace, mu, g_t, g_J, out=H_buf_p).copy()
        Hm_y = vbdg._build_H_stack(np.column_stack((kx, ky - dk)), M, Q,
                                np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex),
                                n_kspace, mu, g_t, g_J, out=H_buf_m).copy()
        dH_dky = (Hp_y - Hm_y) / (2.0 * dk)

        psi_sel = np.take_along_axis(evecs_sel, bands_sel[:, None, None], axis=2)[:, :, 0]
        vF_x = np.real(np.einsum("ni,nij,nj->n", psi_sel.conj(), dH_dkx, psi_sel, optimize=True))
        vF_y = np.real(np.einsum("ni,nij,nj->n", psi_sel.conj(), dH_dky, psi_sel, optimize=True))
        vF_vec = np.column_stack((vF_x, vF_y))

        # ---- 9. Integration weights from the 2D thermal element ----
        dA = grid_step ** 2
        therm_sel = bdy_factor_k[fs_idx] * therm_sym[fs_idx]        # boundary points count ½ (T_Q-paired)

        # Renormalise so that the truncated sample keeps the thermal weight of the whole shell.
        w_full_shell = float(np.sum(bdy_factor_k[fs_idx_shell] * therm_sym[fs_idx_shell])) * dA
        w_sel        = float(np.sum(therm_sel)) * dA
        scale = (w_full_shell / w_sel) if w_sel > 1e-30 else 1.0

        weights = therm_sel * dA / _BZ_NORM * scale

        # ---- 10. Diagnostics ----
        if verbose and n_sel > 0:
            print(f"[FS-SYM] orbit group (spectrum)={symmetry_group}  explicit C4-breaker in H={'yes' if h_breaks_C4 else 'no'}  "
                f"Δ_B1g={d_B1g_static:+.3e}  |M|={M_norm:.3e}  |Q|={Q_norm:.3e}  max|E(k)-E(Rk)|={c4_spec_dev:.2e} eV")
            print(f"[FS-SYM] N={n_sel}  orbits={len(orbit_scores)}  "
                f"shell={len(fs_idx_shell)}  n_target={self.thermal_fs_nat_size}")
            _bdy_sel = on_bdy_k[fs_idx]
            print(f"[FS-SYM] T_Q: {int(_bdy_sel.sum())}/{n_sel} boundary points (paired, weight ½) carry {100.0 * float(np.sum(weights[_bdy_sel])) / max(float(np.sum(weights)), 1e-300):.1f}% of the weight  "
                f"max|E(k)-E(k+Q)|={tq_spec_dev:.2e} eV")
            if tq_spec_dev > 1e-8:
                print("[FS-WARN] the normal-state spectrum is not T_Q-invariant — the T_Q fundamental-domain weights are not valid.")

            phi_d_fs = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
            phi_s_fs = np.ones_like(phi_d_fs)
            s_norm = float(np.sum(weights))
            if s_norm > 0.0:
                avg_d  = float(np.dot(weights, phi_d_fs) / s_norm)
                avg_px = float(np.dot(weights, np.sin(fs_pts[:, 0])) / s_norm)
                avg_py = float(np.dot(weights, np.sin(fs_pts[:, 1])) / s_norm)
                denom = np.sqrt(np.sum(weights * phi_s_fs ** 2)
                                * np.sum(weights * phi_d_fs ** 2))
                S_sd = (float(np.sum(weights * phi_s_fs * phi_d_fs) / denom)
                        if denom > 0.0 else 0.0)

                print(f"[FS-SYM] <phi_d>={avg_d:+.3e}  "
                    f"<sin kx>={avg_px:+.3e}  <sin ky>={avg_py:+.3e}  S_sd={S_sd:+.3e}")

                # Odd moments: D2h-odd in both regimes → must cancel exactly.
                if max(abs(avg_px), abs(avg_py)) > 1e-10:
                    print("[FS-WARN] D2h-odd moment did not cancel "
                        "— check orbit closure / grid indexing.")
                # S_sd: only required to vanish in the D4h regime.
                if symmetry_group == "D4h" and abs(S_sd) > 1e-10:
                    print("[FS-WARN] D4h regime but S_sd != 0 "
                        "— check D4h orbit closure.")

            w_max = float(np.max(weights))
            w_med = float(np.median(weights))
            if w_med > 0.0 and w_max > 5.0 * w_med:
                top_k = min(5, n_sel)
                top_idx = np.argsort(weights)[::-1][:top_k]
                pts_str = "  ".join(
                    f"k=({fs_pts[i, 0]:+.3f},{fs_pts[i, 1]:+.3f}) w={weights[i]:.3e}"
                    for i in top_idx)
                print(f"[FS-WARN] dominant FS weight: w_max/w_median={w_max / w_med:.1f}x  top: {pts_str}")
            _scf_log("FS-DEBUG", f"N_shell={len(fs_idx_shell)}, N_target={n_target}, n_tie_blocks={len(tie_blocks)}, first_3_block_sizes={[sum(len(il) for (_,_,il) in b) for b in tie_blocks[:3]]}, selected={len(selected_full)}")

        # ---- 9. Cache ----
        result = (fs_pts, vF_vec, fs_idx, weights)
        if store_cache:
            if self._fs_cache_dict is None:
                self._fs_cache_dict = {}
            if len(self._fs_cache_dict) >= 32:
                self._fs_cache_dict.pop(next(iter(self._fs_cache_dict)))
            self._fs_cache_dict[_cache_key_vals] = result
        return result

class VectorizedBdG:
    def __init__(self, solver: 'RMFT_Solver'):
        self.solver    = solver
        self._H_stack  = np.zeros((solver.N_k, _N_BDG, _N_BDG), dtype=complex)  # SCF grid buffer
        self.Z_afm_eff = solver.p.Z_afm_eff  # q_AFM-consistent A1g coupling multiplier (includes J2, J3)
        self.g_Eg2     = solver.p.g_Eg2
        self.t_prime   = solver.p.t_prime
        self.t_dprime  = solver.p.t_dprime
        self.Sz_stag_nambu_channels = solver.Sz_stag_nambu_channels
    
    def _build_H_stack(self, kpts: np.ndarray, M: np.ndarray, Q: float, Delta_vec: np.ndarray, F67_vec: np.ndarray, n_kspace: float, mu: float, g_t: float, g_J: float, out: Optional[np.ndarray] = None, Q_Eg2: float = 0.0) -> np.ndarray:
        """
        Build the (N, 24, 24) BdG Hamiltonian stack for an arbitrary kpts array.

        24×24 Nambu basis: [Part_A(0:6), Part_B(6:12), Hole_A(12:18), Hole_B(18:24)], each sub-block in the FULL [6↑, 6↓, 7ₐ↑, 7ₐ↓, 7ᵦ↑, 7ᵦ↓] orbital basis
        (no downfolding; all three Kramers doublets fully dynamical).

        ┌──────────────┬──────────────────────────────┐
        │  H_A   T_AB  │  D_s        D_bond            │  Part_A, Part_B
        │  T_AB† H_B   │  D_bond     D_s               │
        ├──────────────┼──────────────────────────────┤
        │  D_s†  D_bond†│ −H_A*      −T_AB†            │  Hole_A, Hole_B
        │  D_bond† D_s† │ −T_AB*     −H_B*             │
        └──────────────┴──────────────────────────────┘

        D_bond packs the inter-sublattice (nn_AB) symmetry channels (es, d, px, py); the same-sublattice bond channels (g, px2, py2: kind 'nnn_*') are written
        ONLY into the A–A / B–B PAIRING blocks together with the on-site D_s.

        Pairing amplitudes (`Delta_vec`, `F67_vec`, both _N_DELTA = 2·_N_IRREP components):
        • 's'                          → on-site A1g (A–A / B–B)
        • 'es', 'd', 'px', 'py'        → NN-bond (A–B / B–A), form factor φ(k)
        • 'g'                          → NNN diagonal (A–A / B–B), sin kx sin ky, T_Q-even
        • 'px2', 'py2'                 → NNN axial (A–A / B–B), sin 2kx / sin 2ky, T_Q-even
        The 7a / 7b labels denote the two Γ7-derived Kramers doublets (orthogonal pair channels).

        Transverse (anomalous) Weiss fields (H_TRW):
        • On-site (s)             → β_k(Q) · Z · J_B1g^{(s)}(Q) · F67_s · B1g_offdiag
        • NN-bond (es, d, px, py) → φ_i(k) · J_B1g^{(i)}(Q) · F67_i · B1g_offdiag
        • NNN / 3NN (g, es2, d2, px2, py2, ...) → φ_i(k) · J_B1g^{(i)}(Q) · F67_i · B1g_offdiag
        The NNN amplitude is small (κ_nnn ≈ (t'/t)²·κ_NN) but not symmetry-forbidden: the relative sign on the two sublattices is fixed by the same Θ·τ_AB rule as the
        on-site term (A gets −, B gets + for a uniform sign convention). φ_i(k)·F_i·B1g_offdiag IS D2h-mirror covariant once F_i is transformed with the channel's mirror
        character together with Δ_i.

        Fermion antisymmetry, Δ_ij(k) = -Δ_ji(-k)
        ------------------------------------------
        An EVEN-in-k pairing form factor (s, es, d, g; φ(-k)=+φ(k)) is combined with an ANTIsymmetric (site+orbital+spin) index structure.  An ODD-in-k form factor
        (px, py, px2, py2; φ(-k)=-φ(k)) instead uses a SYMMETRIC index structure (identical value at [i,j] and [j,i]), added as a separate contribution (Delta_full_odd)
        — never antisymmetrised.

        Hole-block convention (k → −k):
        The hole blocks are −H_part(−k)^*.  Since even parts satisfy f(−k)=f(k) and odd parts satisfy f(−k)=−f(k), the ODD contributions enter the hole block with the
        OPPOSITE sign of the particle block.  This applies to H_AB (hopping + NN TRW).
        """
        N = len(kpts)
        solver = self.solver

        Delta_s_7a, Delta_d_7a, Delta_s_7b, Delta_d_7b = Delta_vec[_DELTA_IDX['s7a']], Delta_vec[_DELTA_IDX['d7a']], Delta_vec[_DELTA_IDX['s7b']], Delta_vec[_DELTA_IDX['d7b']]
        Delta_es_7a, Delta_es_7b = Delta_vec[_DELTA_IDX['es7a']], Delta_vec[_DELTA_IDX['es7b']]
        F67a_s, F67a_d, F67b_s, F67b_d = F67_vec[_DELTA_IDX['s7a']], F67_vec[_DELTA_IDX['d7a']], F67_vec[_DELTA_IDX['s7b']], F67_vec[_DELTA_IDX['d7b']]

        if out is None:
            H = np.zeros((N, _N_BDG, _N_BDG), dtype=complex)
        else:
            H = out
            H[:] = 0.0 + 0.0j

        # --- Hopping and exchange ---
        tx_b, ty_b = solver.p.effective_hopping_anisotropic(Q)
        Tx_op, Ty_op = solver.p.hopping_matrices(Q)
        J_A1g_diag, J_B1g_scalar_7a, J_B1g_scalar_7b = solver.p.exchange_channels(Q, g_J)

        # --- Local Hamiltonians (A/B sublattice), full 6×6 (Γ6⊕Γ7a⊕Γ7b) ---
        H_A = solver.build_local_hamiltonian_for_bdg(+1.0, M, J_A1g_diag, mu, self.Z_afm_eff)
        H_B = solver.build_local_hamiltonian_for_bdg(-1.0, M, J_A1g_diag, mu, self.Z_afm_eff)

        # --- k-dependent quantities ---
        _kx = kpts[:, 0]
        _ky = kpts[:, 1]
        beta_k = solver.p.wave_function_weight(tx_b, ty_b, _kx, _ky)

        # --- Local JT coupling (β_k-weighted) ---
        H_JT_loc = (solver.g_JT_bare * Q * solver.B1g_op).astype(complex)
        H_JT_k = beta_k[:, None, None] * H_JT_loc[None, :, :]

        # --- On-site (s-channel) TRW ---
        # Active only when Q ≠ 0 AND F67 ≠ 0; the coordination factor is the BARE Z (single-site operator)
        # The Γ6↔Γ7a block is weighted by F67a_s · J_B1g^{(s)}_7a, the Γ6↔Γ7b block by F67b_s · J_B1g^{(s)}_7b.
        TRW_weight_s = np.zeros((_N_ORB, _N_ORB), dtype=complex)
        TRW_weight_s[0:2, 2:4] = F67a_s * J_B1g_scalar_7a
        TRW_weight_s[2:4, 0:2] = np.conj(F67a_s) * J_B1g_scalar_7a
        TRW_weight_s[0:2, 4:6] = F67b_s * J_B1g_scalar_7b
        TRW_weight_s[4:6, 0:2] = np.conj(F67b_s) * J_B1g_scalar_7b

        f67s_loc_matrix = _Z * (solver.B1g_offdiag * TRW_weight_s)
        H_TRW = beta_k[:, None, None] * f67s_loc_matrix[None, :, :]

        # --- Form-factor matrix on the current k-points (needed for both the NN-bond and the NNN-bond TRW) ---
        if N == solver.N_k and (kpts is solver.k_points or np.array_equal(kpts, solver.k_points)):
            Phi_irrep_k_here = solver.Phi_irrep_k
        else:
            Phi_irrep_k_here = _pairing_basis_matrix(_kx, _ky)

        # --- Eg,2 distortion: UNIFORM across sublattices ---
        H_Eg2 = (self.g_Eg2 * Q_Eg2) * solver.Eg2_op

        # --- 2nd/3rd-neighbor (t', t'') dispersion ---
        _disp_nnn = (-4.0 * g_t * self.t_prime * np.cos(_kx) * np.cos(_ky) - 2.0 * g_t * self.t_dprime * (np.cos(2.0 * _kx) + np.cos(2.0 * _ky)))
        H_nnn_k = _disp_nnn[:, None, None] * np.eye(_N_ORB, dtype=complex)[None, :, :]

        # --- Same-sublattice (NNN / 3NN) transverse Weiss field ---
        # Split even (parity +1) / odd (parity -1): the odd part flips sign under k → −k,
        # so it enters the HOLE block with the opposite sign relative to the particle block.
        H_TRW_nnn_even = np.zeros((N, _N_ORB, _N_ORB), dtype=complex)
        H_TRW_nnn_odd  = np.zeros((N, _N_ORB, _N_ORB), dtype=complex)
        for i_irr, _name in enumerate(_IRREP_NAMES):
            if not _is_nnn_bond(_name):
                continue
            _s_trw = float(solver.p.trw_irrep_scale[i_irr])          # bond-class weight (t_δ/t)² of this channel
            F67a_nnn = _s_trw * F67_vec[_IDX_7A_ARR[i_irr]]
            F67b_nnn = _s_trw * F67_vec[_IDX_7B_ARR[i_irr]]
            if F67a_nnn == 0 and F67b_nnn == 0:
                continue
            TRW_weight_nnn = np.zeros((_N_ORB, _N_ORB), dtype=complex)
            TRW_weight_nnn[0:2, 2:4] = F67a_nnn * J_B1g_scalar_7a
            TRW_weight_nnn[2:4, 0:2] = np.conj(F67a_nnn) * J_B1g_scalar_7a
            TRW_weight_nnn[0:2, 4:6] = F67b_nnn * J_B1g_scalar_7b
            TRW_weight_nnn[4:6, 0:2] = np.conj(F67b_nnn) * J_B1g_scalar_7b
            f67_nnn_matrix = solver.B1g_offdiag * TRW_weight_nnn
            contribution = Phi_irrep_k_here[:, i_irr][:, None, None] * f67_nnn_matrix[None, :, :]
            if _IRREP_PARITY[_name] > 0:
                H_TRW_nnn_even += contribution
            else:
                H_TRW_nnn_odd  += contribution

        # --- Particle and hole diagonal blocks ---
        H_part_0_base = H_A + H_Eg2 + H_JT_k - H_TRW - H_TRW_nnn_even - H_TRW_nnn_odd + H_nnn_k
        H_part_1_base = H_B + H_Eg2 + H_JT_k + H_TRW + H_TRW_nnn_even + H_TRW_nnn_odd + H_nnn_k

        H[:, 0:_N_ORB, 0:_N_ORB] = H_part_0_base
        H[:, _N_ORB:2*_N_ORB, _N_ORB:2*_N_ORB] = H_part_1_base

        # Hole-diagonal blocks: −H_part(−k)^*.  EVEN pieces keep their sign, ODD pieces flip.
        H_part_0_even = H_A + H_Eg2 + H_JT_k - H_TRW - H_TRW_nnn_even + H_nnn_k
        H_part_1_even = H_B + H_Eg2 + H_JT_k + H_TRW + H_TRW_nnn_even + H_nnn_k
        H_part_0_odd  = -H_TRW_nnn_odd
        H_part_1_odd  = +H_TRW_nnn_odd

        H[:, 2*_N_ORB:3*_N_ORB, 2*_N_ORB:3*_N_ORB] = -np.conj(H_part_0_even) + np.conj(H_part_0_odd)
        H[:, 3*_N_ORB:_N_BDG, 3*_N_ORB:_N_BDG] = -np.conj(H_part_1_even) + np.conj(H_part_1_odd)

        # --- Inter-sublattice transverse Weiss field, every NN-bond irrep (es, d, px, py) ---
        # Kept split into an EVEN part (es, d) and an ODD part (px, py): both are fine in the PARTICLE block (Hermiticity alone governs it), but the hole block below needs them with opposite signs.
        H_TRW_AB_even = np.zeros((N, _N_ORB, _N_ORB), dtype=complex)
        H_TRW_AB_odd  = np.zeros((N, _N_ORB, _N_ORB), dtype=complex)
        for i_irr, _name in enumerate(_IRREP_NAMES):
            if not _is_nn_bond(_name):       # TRW lives on inter-sublattice (NN-type) bonds only
                continue
            _s_trw = float(solver.p.trw_irrep_scale[i_irr])
            F67a_bond = _s_trw * F67_vec[_IDX_7A_ARR[i_irr]]
            F67b_bond = _s_trw * F67_vec[_IDX_7B_ARR[i_irr]]
            TRW_weight_bond = np.zeros((_N_ORB, _N_ORB), dtype=complex)
            TRW_weight_bond[0:2, 2:4] = F67a_bond * J_B1g_scalar_7a
            TRW_weight_bond[2:4, 0:2] = np.conj(F67a_bond) * J_B1g_scalar_7a
            TRW_weight_bond[0:2, 4:6] = F67b_bond * J_B1g_scalar_7b
            TRW_weight_bond[4:6, 0:2] = np.conj(F67b_bond) * J_B1g_scalar_7b
            f67_bond_inter_matrix = solver.B1g_offdiag * TRW_weight_bond
            contribution = Phi_irrep_k_here[:, i_irr][:, None, None] * f67_bond_inter_matrix[None, :, :]
            if _IRREP_PARITY[_name] > 0:
                H_TRW_AB_even += contribution
            else:
                H_TRW_AB_odd += contribution

        H_AB_even = _build_H_AB_block(_kx, _ky, Tx_op, Ty_op, g_t) + H_TRW_AB_even   # (N,6,6): orbital-selective hopping (even) + even-parity TRW
        H_AB_odd  = H_TRW_AB_odd                                                     # odd-parity (px, py) TRW only
        H_AB = H_AB_even + H_AB_odd

        H_AB_T = H_AB.transpose(0, 2, 1)

        # Particle sector: A→B is H_AB, B→A is its conjugate transpose (Hermiticity at fixed k; parity-independent)
        H[:, 0:_N_ORB, _N_ORB:2*_N_ORB] += H_AB
        H[:, _N_ORB:2*_N_ORB, 0:_N_ORB] += np.conj(H_AB_T)

        # Hole sector needs H_AB evaluated at -k: the EVEN part reuses H_AB_even(k), while the ODD part flips sign under k->-k 
        H_AB_even_T = H_AB_even.transpose(0, 2, 1)
        H_AB_odd_T  = H_AB_odd.transpose(0, 2, 1)
        H[:, 2*_N_ORB:3*_N_ORB, 3*_N_ORB:_N_BDG] += -np.conj(H_AB_even) + np.conj(H_AB_odd)
        H[:, 3*_N_ORB:_N_BDG, 2*_N_ORB:3*_N_ORB] += -H_AB_even_T + H_AB_odd_T
        
        # Pairing block (particle-hole), 12×12 in (sublattice ⊗ orbital) space
        Delta_full = np.zeros((N, 12, 12), dtype=complex)

        # --- On-site singlet ('s' channel only): Γ6↔Γ7a AND Γ6↔Γ7b ---
        D_on = np.zeros((_N_ORB, _N_ORB), dtype=complex)
        D_on[0, 3] =  Delta_s_7a       # Γ6↑ ↔ Γ7a↓
        D_on[1, 2] = -Delta_s_7a       # Γ6↓ ↔ Γ7a↑  (singlet sign)
        D_on[0, 5] =  Delta_s_7b       # Γ6↑ ↔ Γ7b↓
        D_on[1, 4] = -Delta_s_7b       # Γ6↓ ↔ Γ7b↑  (singlet sign)

        Delta_full[:, 0:_N_ORB,  0:_N_ORB] = D_on[None, :, :]   # A-A
        Delta_full[:, _N_ORB:2*_N_ORB, _N_ORB:2*_N_ORB] = D_on[None, :, :]   # B-B

        # ---- Inter-sublattice EVEN-parity NN-bond channels (es, d), Γ6↔Γ7a AND Γ6↔Γ7b ----
        phi_7a = np.zeros(N, dtype=complex)
        phi_7b = np.zeros(N, dtype=complex)
        for i_irr, _name in enumerate(_IRREP_NAMES):
            if _is_nn_bond(_name) and _IRREP_PARITY[_name] > 0:
                phi_7a = phi_7a + Phi_irrep_k_here[:, i_irr] * Delta_vec[_IDX_7A_ARR[i_irr]]
                phi_7b = phi_7b + Phi_irrep_k_here[:, i_irr] * Delta_vec[_IDX_7B_ARR[i_irr]]

        # A→B block (particle A rows 0:6, hole B cols 6:12 of Delta_full)
        Delta_full[:, 0,  9] += phi_7a   # A: Γ6↑, B: Γ7a↓ (j = 3)
        Delta_full[:, 1,  8] -= phi_7a   # A: Γ6↓, B: Γ7a↑ (j = 2)
        Delta_full[:, 0, 11] += phi_7b   # A: Γ6↑, B: Γ7b↓ (j = 5)
        Delta_full[:, 1, 10] -= phi_7b   # A: Γ6↓, B: Γ7b↑ (j = 4)

        # B→A block (particle B rows 6:12, hole A cols 0:6 of Delta_full)
        Delta_full[:, 6, 3] += phi_7a    # B: Γ6↑, A: Γ7a↓ (j = 3)
        Delta_full[:, 7, 2] -= phi_7a    # B: Γ6↓, A: Γ7a↑ (j = 2)
        Delta_full[:, 6, 5] += phi_7b    # B: Γ6↑, A: Γ7b↓ (j = 5)
        Delta_full[:, 7, 4] -= phi_7b    # B: Γ6↓, A: Γ7b↑ (j = 4)

        # ---- Same-sublattice EVEN-parity bond channels beyond on-site (NNN / 3NN: 'g', ...): A–A and B–B blocks carrying the k-dependent form factor φ(k) ----
        # Same Γ6↔Γ7a / Γ6↔Γ7b index and singlet-sign pattern as the on-site D_on above; identical amplitude on both sublattices (translation-invariant pairing => T_Q-even).
        for i_irr, _name in enumerate(_IRREP_NAMES):
            if _IRREP_KIND[_name] not in ('nnn_diag', 'nnn_axis') or _IRREP_PARITY[_name] < 0:
                continue
            d_e_7a = Delta_vec[_IDX_7A_ARR[i_irr]]
            d_e_7b = Delta_vec[_IDX_7B_ARR[i_irr]]
            if d_e_7a == 0 and d_e_7b == 0:
                continue
            phi_e_k = Phi_irrep_k_here[:, i_irr]
            for _o in (0, _N_ORB):                                  # A–A block (rows/cols 0:6), B–B block (6:12)
                Delta_full[:, _o + 0, _o + 3] += phi_e_k * d_e_7a   # Γ6↑ ↔ Γ7a↓
                Delta_full[:, _o + 1, _o + 2] -= phi_e_k * d_e_7a   # Γ6↓ ↔ Γ7a↑ (singlet sign)
                Delta_full[:, _o + 0, _o + 5] += phi_e_k * d_e_7b   # Γ6↑ ↔ Γ7b↓
                Delta_full[:, _o + 1, _o + 4] -= phi_e_k * d_e_7b   # Γ6↓ ↔ Γ7b↑ (singlet sign)

        # --- Antisymmetrization (Pauli principle: Δ_ij(k) = -Δ_ji(k) at fixed k -- valid for the EVEN-in-k channels above; see docstring) ---
        Delta_asym = Delta_full - Delta_full.transpose(0, 2, 1)

        # ---- Inter-sublattice ODD-parity NN-bond channels (px, py), Γ6↔Γ7a AND Γ6↔Γ7b ----
        # Same singlet spin-sign convention as above (+ for ↑↓, - for ↓↑), but filled SYMMETRICALLY ([i,j] and [j,i] get the SAME value) rather than antisymmetrised
        Delta_full_odd = np.zeros((N, 12, 12), dtype=complex)

        def _add_symmetric(i, j, val):
            Delta_full_odd[:, i, j] += val
            Delta_full_odd[:, j, i] += val

        for ir_idx, _name in enumerate(_IRREP_NAMES):
            if _IRREP_PARITY[_name] > 0:
                continue
            phi_odd_k = Phi_irrep_k_here[:, ir_idx]
            d_o_7a = phi_odd_k * Delta_vec[_IDX_7A_ARR[ir_idx]]
            d_o_7b = phi_odd_k * Delta_vec[_IDX_7B_ARR[ir_idx]]
            if _is_nn_bond(_name):           # inter-sublattice (A–B / B–A): px, py
                _add_symmetric(0, 9,   d_o_7a)   # A:Γ6↑ / B:Γ7a↓
                _add_symmetric(1, 8,  -d_o_7a)   # A:Γ6↓ / B:Γ7a↑
                _add_symmetric(6, 3,   d_o_7a)   # B:Γ6↑ / A:Γ7a↓
                _add_symmetric(7, 2,  -d_o_7a)   # B:Γ6↓ / A:Γ7a↑
                _add_symmetric(0, 11,  d_o_7b)   # A:Γ6↑ / B:Γ7b↓
                _add_symmetric(1, 10, -d_o_7b)   # A:Γ6↓ / B:Γ7b↑
                _add_symmetric(6, 5,   d_o_7b)   # B:Γ6↑ / A:Γ7b↓
                _add_symmetric(7, 4,  -d_o_7b)   # B:Γ6↓ / A:Γ7b↑
            else:                            # same-sublattice (A–A / B–B): px2, py2 (3rd-neighbour axial bonds, odd in k, T_Q-even)
                for _o in (0, _N_ORB):
                    _add_symmetric(_o + 0, _o + 3,  d_o_7a)   # Γ6↑ / Γ7a↓
                    _add_symmetric(_o + 1, _o + 2, -d_o_7a)   # Γ6↓ / Γ7a↑
                    _add_symmetric(_o + 0, _o + 5,  d_o_7b)   # Γ6↑ / Γ7b↓
                    _add_symmetric(_o + 1, _o + 4, -d_o_7b)   # Γ6↓ / Γ7b↑

        Delta_asym = Delta_asym + Delta_full_odd

        # Particle-hole block and its Hermitian conjugate (hole-particle)
        H[:, 0:2*_N_ORB, 2*_N_ORB:_N_BDG] = Delta_asym
        H[:, 2*_N_ORB:_N_BDG, 0:2*_N_ORB] = np.conj(Delta_asym).transpose(0, 2, 1)

        # Final Hermitian symmetrization (numerical cleanup against roundoff)
        H[:] = 0.5 * (H + H.conj().transpose(0, 2, 1))
        return H

    def compute_channel_staggered_magnetizations(self, Q: float, Delta_vec: np.ndarray, mu: float, ev: np.ndarray, ec: np.ndarray) -> np.ndarray:
        """
        Channel-resolved (Γ6, Γ7a, Γ7b) BdG staggered magnetization; the full unweighted S_z_stag trace
        is exactly the sum of the three channel traces, since the channels orthogonally partition the 6 orbitals.
        """
        solver = self.solver
        fn = _fermi_function(ev, solver.p.kT)
        M_obs = np.zeros(_N_CHANNELS)
        for c in range(_N_CHANNELS):
            diag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), self.Sz_stag_nambu_channels[c], ec).real
            exp_k = np.einsum('kn,kn->k', diag_qp, fn)
            M_obs[c] = float(np.dot(solver.k_weights, exp_k)) / 4.0
        return M_obs

    def compute_gap_eq_vectorized(self, M: np.ndarray, Q: float, Delta_vec: np.ndarray, n_kspace: float, mu: float, t_eff: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_ad: float, g_Delta_bd: float, J_A1g_diag: np.ndarray, Gamma_M: float, V_JT_eff: float, V_JT_corr: float, V_cap: float, det_afm_sc: float, solve_state: '_SolveState', ev: np.ndarray, ec: np.ndarray, vertex_cache: dict = None, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Gap equation with a q-dependent RPA pairing vertex V(q) built from normal-state (Δ=0) susceptibilities.

        Extended-basis form: for each anomalous branch b ∈ {7a, 7b} the raw gap update is Δ_raw[b, i] = Σ_j V_irrep[i, j] · √(g_i / g_j) · F67_raw[b, j],   i, j ∈ _IRREP_NAMES,
        with V_irrep the N_IRREP × N_IRREP FS-kernel projection matrix, and g the branch's per-irrep Gutzwiller vector. The kernel's eigenvector blend below likewise acts on all
        N_IRREP channels, so the  iteration is pulled toward whichever symmetry the FS kernel's dominant eigenvector actually carries.

        The V_irrep is the G-covariant sector-kernel projection: only the sector-character-matched blocks are filled, everything else is exactly zero.
        """
        solver = self.solver
        # --- Vertex cache invalidation (Δ-independent for normal-state part!) ---
        staleness = (
            not isinstance(vertex_cache, dict)
            or float(np.max(np.abs(M - vertex_cache['M']))) > _M_THR_REL * float(np.sqrt(abs(det_afm_sc)))
            or abs(Q - vertex_cache['Q']) > max(_Q_THR_REL * solver.p.lambda_hop, 1e-4)
            or (det_afm_sc * vertex_cache['det_afm_current']) < 0.0
            or abs(V_JT_eff  - vertex_cache['V_JT_eff']) / V_JT_eff  > 1e-3
            or abs(V_JT_corr - vertex_cache['V_JT_corr']) / V_JT_corr > 1e-3
        )
        if staleness:
            # Compute kernel and obtain base cache
            vertex_cache = solver.compute_pairing_kernel_and_build_cache(M, Q, n_kspace, mu, g_t, g_J, J_A1g_diag, Gamma_M, V_JT_eff, V_JT_corr, V_cap)
            ev_n0, ec_n0 = solver._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self)

            _rp_q0  = solver._rpa_closure(M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M, J_A1g_diag, V_JT_corr, (ev_n0, ec_n0), apply_diamagnetic_QQ=True)
            _rp_afm = solver._rpa_closure(M, Q, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, J_A1g_diag, V_JT_corr, (ev_n0, ec_n0), apply_diamagnetic_QQ=True)

            vertex_cache.update({
                'chi_SS_q0':  _rp_q0['chi_SS'],
                'chi_SQ_q0':  _rp_q0['chi_SQ'],
                'chi_QQ_q0':  _rp_q0['chi_QQ'],
                'chi_SS_afm': _rp_afm['chi_SS'],
                'chi_SQ_afm': _rp_afm['chi_SQ'],
                'chi_QQ_afm': _rp_afm['chi_QQ'],
                'det_afm_current': det_afm_sc,
            })

        # ---- Anomalous pair amplitudes (always computed with current Δ) ----
        # The Bogoliubov u·v* coherence factors here inherit whatever relative phase every Δ channel currently carries.
        F67_vec, F_raw_g = _compute_F67_pair_amplitudes(solver.k_weights, solver.Phi_irrep_k, ev, ec, solver.kT, Delta_vec, g_Delta_s, g_Delta_ad, g_Delta_bd)
        F67a_s, F67a_d, F67b_s, F67b_d = F67_vec[_DELTA_IDX['s7a']], F67_vec[_DELTA_IDX['d7a']], F67_vec[_DELTA_IDX['s7b']], F67_vec[_DELTA_IDX['d7b']]   # s/d amplitudes of both branches (diagnostics below)

        # ---- Extract kernel-matrix and full-FS kernel info from cache ----
        V_irrep_raw    = vertex_cache['V_irrep']
        seed_w         = vertex_cache['seed_w']
        sector_vraw    = vertex_cache['sector_vraw']
        V_parts_diag   = vertex_cache['V_parts_diag']
        S_irrep        = vertex_cache['S_irrep']
        c_irrep        = vertex_cache['c_irrep']
        phi_s_d        = vertex_cache['phi_s_d']
        phi_d          = vertex_cache['phi_d']
        Phi_fs         = vertex_cache['Phi_fs']
        fs_weights     = vertex_cache['fs_weights']
        lambda_lin_max = vertex_cache['lambda_lin_max']

        # Working copy of the kernel matrix: its diagonal may be replaced by an EMA below, off-diagonals stay raw (as V_sd always did).
        V_irrep = V_irrep_raw.copy()

        # --- Diagonal EMA sign-flip protection: only if solve_state is provided ---
        # EMA of the diagonal vertex of every NN-bond channel (d, es, px, py). All four have form factors that change sign under the AFM shift k → k+(π,π),
        # so all four see the same AFM-peaked V_spin(q) attraction and can flip sign together with det_afm_sc near the QCP.
        if solve_state is not None:
            V_diag_raw = np.diag(V_irrep_raw).copy()
            if solve_state.V_irrep_ema is None:
                solve_state.V_irrep_ema = V_diag_raw.copy()
            else:
                _ema_prev   = solve_state.V_irrep_ema
                _ema_new    = _ema_prev.copy()
                _kick_boost = 2.0 if solve_state._ema_kick_pending else 1.0
                for i_irr, _name in enumerate(_IRREP_NAMES):
                    if _afm_shift_character(_name) > 0:
                        _ema_new[i_irr] = V_diag_raw[i_irr]      # not smoothed; tracked only so the state array stays aligned
                        continue
                    sign_flipped = (V_diag_raw[i_irr] * _ema_prev[i_irr] < 0.0 and abs(_ema_prev[i_irr]) > _V_PREV_SIGN_FLOOR)
                    if sign_flipped:
                        if V_diag_raw[i_irr] > 0.0:
                            _ema_w = _EMA_NEW_WEIGHT
                        else:
                            _ema_w = _EMA_NEW_WEIGHT * (_EMA_SIGN_FLIP_W_MIN + (1.0 - _EMA_SIGN_FLIP_W_MIN) / (1.0 + math.exp(-_EMA_SIGN_FLIP_SLOPE * (abs(det_afm_sc) / _DET_SIGN_FLIP_SCALE - 0.5))))
                    else:
                        _ema_w = min(_EMA_NEW_WEIGHT * _kick_boost, 1.0)
                    _ema_new[i_irr] = (1.0 - _ema_w) * _ema_prev[i_irr] + _ema_w * V_diag_raw[i_irr]
                solve_state._ema_kick_pending = False
                solve_state.V_irrep_ema = _ema_new
            for i_irr, _name in enumerate(_IRREP_NAMES):
                if _afm_shift_character(_name) < 0:
                    V_irrep[i_irr, i_irr] = solve_state.V_irrep_ema[i_irr]

        # ---- N_IRREP × N_IRREP reduction ----
        # None of the quantities in this block feed the gap equation itself (that uses V_irrep from the cache);
        g_vec_a = _gutzwiller_irrep_vector(g_Delta_s, g_Delta_ad, g_t)    # 7a-group Gutzwiller vector (this block has always used the 7a group's g's)
        g_vec_b = _gutzwiller_irrep_vector(g_Delta_s, g_Delta_bd, g_t)    # 7b-group Gutzwiller vector (used by the gap equation below)
        sq_g_a  = np.sqrt(np.clip(g_vec_a, 0.0, None))

        Xw     = Phi_fs * fs_weights[:, None]                        # (N_fs, N_IRREP) FS-weighted basis functions
        W_raw  = vertex_cache['W_irrep']
        W_pair = W_raw * np.outer(sq_g_a, sq_g_a)                    # diag: g_i W_ii ; off-diag: √(g_i g_j) W_ij  (was W11, W22, W12)

        s_eval, U = np.linalg.eigh(S_irrep)
        S_inv_sqrt = (U / np.sqrt(np.maximum(s_eval, 1e-12))) @ U.T

        # Use the properly orthonormalised Rayleigh matrix W_ortho = <ψ_i|V|ψ_j>, ψ = S^{-1/2}φ, which is required for non-orthogonal basis functions on the FS.
        # V_cap bounds individual V_unique[q], not their FS-weighted double sums; clip W_ortho before diagonalisation so lambda_lin_max, eigvecs, and the couplings entering the gap equation share the same bounded interaction.
        W_ortho  = S_inv_sqrt @ W_pair @ S_inv_sqrt
        _cap_mat = V_cap * np.sqrt(np.maximum(np.outer(g_vec_a, g_vec_a), 1e-12))   # off-diagonal cap V_cap·√(g_i g_j) (1e-12 floor, as the old g_sd)
        np.fill_diagonal(_cap_mat, V_cap * g_vec_a)                                 # diagonal cap V_cap·g_i (as the old V_cap·g_Delta_s / V_cap·g_Delta_ad)
        W_ortho_capped = np.clip(W_ortho, -_cap_mat, _cap_mat)
        lambda_lin_max_2d = float(np.linalg.eigvalsh(W_ortho_capped)[-1])           # largest eigenvalue of the N_IRREP-dimensional orthonormalised kernel

        # --- linearised kernel with the JT-only d-d element (diagnostic only: only the d-d entry is replaced) ---
        W_JT = W_pair.copy()
        W_JT[_IR_D, _IR_D] = g_Delta_ad * vertex_cache['W22_JT']

        _eigval_JT, _eigvec_JT = np.linalg.eigh(W_JT)
        lambda_JT_kernel = float(_eigval_JT[-1])

        # --- relative gain from inter-channel mixing (diagnostic) ---
        max_diag_ortho = float(np.max(np.diag(W_ortho_capped)))
        lambda_gain_rel = (lambda_lin_max_2d - max_diag_ortho) / max(abs(max_diag_ortho), 1e-12)

        # Gap equations + jump limiter: F67 already carries BdG saturation via the anomalous Green functions, so a λ_pair-based f_stab would double-count the suppression; the channel ratios are preserved during clamping.
        # Matrix form Δ_raw[i] = Σ_j V_irrep[i,j] √(g_i/g_j) F67[j] per branch.
        if lambda_lin_max > 0:
            _ratio_a = np.sqrt(np.outer(g_vec_a, 1.0 / np.maximum(g_vec_a, 1e-12)))   # √(g_i / g_j), 7a group
            _ratio_b = np.sqrt(np.outer(g_vec_b, 1.0 / np.maximum(g_vec_b, 1e-12)))   # √(g_i / g_j), 7b group
            D_raw = np.array([(V_irrep  * _ratio_a) @ F_raw_g[_IDX_7A_ARR],
                              (V_irrep  * _ratio_b) @ F_raw_g[_IDX_7B_ARR]], dtype=complex)              # (2 branches, _N_IRREP)
        else:
            D_raw = np.zeros((2, _N_IRREP), dtype=complex)
        
        # --- Sign-align the kernel-eigenvector direction with the primary branch ---
        # Leading covered sector + near-degenerate covered sectors (weight λ_χ/λ_lead): superpose their eigenvector directions, each sign-aligned with the current candidate D_raw[0]
        # (the sign of a sector eigenvector is arbitrary), so the kernel pull does not suppress a sector that is almost as unstable as the leading one.
        v_irr = np.zeros(_N_IRREP)
        for _si in np.flatnonzero(seed_w):
            _vs = np.asarray(sector_vraw[_si], dtype=float)
            v_irr += seed_w[_si] * (1.0 if np.real(np.sum(np.conj(D_raw[0]) * _vs)) >= 0 else -1.0) * _vs
        # L1-normalised eigenvector direction from cache (all N_IRREP channels); align sign with the current 7a-branch direction.
        sign_align = 1.0 if np.real(np.sum(np.conj(D_raw[0]) * v_irr)) >= 0 else -1.0
        v_irr *= sign_align / max(float(np.abs(v_irr).sum()), 1e-12)

        # --- Per-branch fixed-point magnitude + seed threshold ---
        D_fp_pre = np.abs(D_raw).sum(axis=1)                                # (2,)
        seeded   = D_fp_pre < _QQ_DELTA_THRESH
        seed_val = float(max(_BCS_SEED_FRACTION * t_eff * np.exp(-1.0 / max(lambda_lin_max, 0.1)), _DELTA_ABS_FLOOR))
        D_fp        = np.where(seeded, seed_val, D_fp_pre)                  # (2,)
        alpha_blend = np.where(seeded, 0.5 * _ALPHA_MIX_2X2, _ALPHA_MIX_2X2)  # (2,)  weights the full irrep-kernel direction
        # The kernel direction only STEERS the iterate while |Δ| is still of the order of the seed. With a constant weight the fixed point of the blended map is NOT a solution of D = V_att·F(D).
        # Fading the weight with |D_fp| keeps the steering at Δ≈0 and makes the converged state an exact solution of the model's own gap equation.
        if _KERNEL_BLEND_DECAY > 0.0:
            _blend_decay = np.exp(-D_fp_pre / (_KERNEL_BLEND_DECAY * max(seed_val, _DELTA_ABS_FLOOR)))   # (2,)  =1 for |Δ|→0
        else:
            _blend_decay = np.ones_like(D_fp_pre)
        alpha_blend = alpha_blend * _blend_decay

        # Kernel target per branch: D_kernel[branch, irrep] = D_fp * v_irr
        D_kernel = D_fp[:, None] * v_irr[None, :]                           # (2, _N_IRREP)

        # Phase-preserving blend: the kernel eigenvector only constrains the REAL relative signs between channels;
        # Blending magnitudes and then multiplying by a real v_irr would silently zero the channels' relative phases every iteration.
        phases = D_raw / np.maximum(np.abs(D_raw), 1e-30)                   # (2, _N_IRREP)

        D_mag = (1.0 - alpha_blend[:, None]) * np.maximum(
            np.abs(D_raw), (_KERNEL_DIR_MIN_FRAC * _blend_decay)[:, None] * np.abs(D_kernel)
        ) + alpha_blend[:, None] * np.abs(D_kernel)                         # (2, _N_IRREP)

        # Below seed threshold → use the (real) kernel sign directly.
        use_kernel_sign = D_fp < _QQ_DELTA_THRESH                           # (2,)
        sign_v_irr = np.where(v_irr != 0.0, np.sign(v_irr), 1.0)            # (_N_IRREP,)

        D_out = np.where(
            use_kernel_sign[:, None],
            D_mag * sign_v_irr[None, :],
            D_mag * phases,
        )

        # --- Det-proportional jump cap ---
        if det_afm_sc < 0.0:
            _det_depth = float(np.clip(abs(det_afm_sc) / max(_RPA_DET_WARN, 1e-6), 0.0, _DET_DEPTH_CAP))
            effective_jump_cap = max(_JUMP_CAP_FLOOR, _DELTA_JUMP_CAP * math.exp(-_DET_JUMP_HALF_SCALE * _det_depth))
        else:
            effective_jump_cap = _DELTA_JUMP_CAP
        
        # ---- Single, final hard cap (per branch) ----
        phys_ceiling = max(effective_jump_cap * np.linalg.norm(Delta_vec), _DELTA_ABS_FLOOR)
        D_totals = np.abs(D_out).sum(axis=1)
        scale = np.minimum(1.0, phys_ceiling / np.maximum(D_totals, 1e-12))
        D_out = D_out * scale[:, None]

        if verbose:
            c_B = vertex_cache['c_B']
            # --- Per-channel (spin / JT / cross) from the sector-kernel projection ---
            # V_parts_diag[ch, i] = sector-kernel projection of the {spin, JT, cross} channel onto form factor φ_i.  Their sum over ch reproduces V_irrep[i,i] (before the V_cap clip).
            V_s_spin, V_s_JT, V_s_cross = (float(V_parts_diag[_k, _IR_S]) for _k in range(3))
            V_d_spin, V_d_JT, V_d_cross = (float(V_parts_diag[_k, _IR_D]) for _k in range(3))

            V_s_total = V_s_spin + V_s_JT + V_s_cross
            V_d_total = V_d_spin + V_d_JT + V_d_cross

            # --- Fractional decomposition of the d-wave coupling ---
            _denom = max(abs(V_d_total), _MATH_EPS)
            dwave_spin_frac  = V_d_spin  / _denom
            dwave_JT_frac    = V_d_JT    / _denom
            dwave_cross_frac = V_d_cross / _denom

            # --- Relative phase between spin and JT channels in the d-wave projection ---
            # spin×JT > 0 : constructive (both push the same way)
            # spin×JT < 0 : destructive (channels compete)
            if abs(V_d_spin) > _MATH_EPS and abs(V_d_JT) > _MATH_EPS:
                dwave_spin_JT_phase = float(np.sign(V_d_spin * V_d_JT))
            else:
                dwave_spin_JT_phase = 0.0
                    
            _scf_log("PAIR-DIAG",
                f"  dwave_spin_frac={dwave_spin_frac:+.2f}  dwave_JT_frac={dwave_JT_frac:+.2f}  phase(↑,JT)={dwave_spin_JT_phase:+.0f}"
                f"  V_s: spin={V_s_spin:+.4f}  JT={V_s_JT:+.4f}  cross={V_s_cross:+.4f} → V_s_tot={V_s_total:+.4f}"
                f"  V_d: spin={V_d_spin:+.4f}  JT={V_d_JT:+.4f}  cross={V_d_cross:+.4f} → V_d_tot={V_d_total:+.4f}")
            _scf_log("PAIR-DIAG",
                f"{'✓ BS+attract' if bool(V_s_spin < 0.0 and V_d_spin > 0.0) and bool(V_s_JT > 0.0 and V_d_JT > 0.0) else '⚠ anomalous'}"
                f"{'✓ d-wave leads' if bool(V_d_total > V_s_total) else '⚠ s-wave leads'}")

            for i_irr, _name in enumerate(_IRREP_NAMES):
                if _name in ('s', 'd'):
                    continue
                _p_spin, _p_JT, _p_cross = (float(V_parts_diag[_k, i_irr]) for _k in range(3))
                _scf_log("PAIR-DIAG",
                    f"  V_{_name} ({_IRREP_D4H[_name]}): spin={_p_spin:+.4f}  JT={_p_JT:+.4f} cross={_p_cross:+.4f}  → V_{_name}_tot={_p_spin + _p_JT + _p_cross:+.4f}")

            # --- V_irrep diagonal vs. sector-kernel projection consistency ---
            _V_diag_irrep = np.diag(V_irrep_raw)
            _V_diag_sum   = V_parts_diag.sum(axis=0)
            _diag_err     = float(np.max(np.abs(_V_diag_sum - _V_diag_irrep)))

            _scf_log("PAIR-DIAG",
                f"  V_irrep diag: {_fmt_amplitude_line(_V_diag_irrep, _IRREP_NAMES, top_n=4, signed=True)}"
                f"  max|offdiag|={float(np.max(np.abs(V_irrep_raw - np.diag(_V_diag_irrep)))):.4f}")
            _scf_log("PAIR-DIAG",
                f"  Σ_ch (spin+JT+cross): {_fmt_amplitude_line(_V_diag_sum, _IRREP_NAMES, top_n=4, signed=True)}"
                f"  |Σ_ch − V_irrep_diag|_∞ = {_diag_err:.2e}  {'✓' if _diag_err < 1e-10 else '⚠ (V_cap clip or odd-channel reset)'}")

            # --- Antinodal/nodal decomposition in the spin channel (from the V_irrep spectrum) ---
            _K_d_spin = vertex_cache['K_d_spin']
            _anti_mask = c_B > 0.7
            _nod_mask = c_B < 0.3
            wx = phi_d * fs_weights

            def _subregion(Kch, mi, mj):
                return float((mi * wx) @ Kch @ (mj * wx)) / (phi_d @ wx) ** 2

            V_d_spin_aa = _subregion(_K_d_spin, _anti_mask, _anti_mask)
            V_d_spin_nn = _subregion(_K_d_spin, _nod_mask,  _nod_mask)
            V_d_spin_an = _subregion(_K_d_spin, _anti_mask, _nod_mask)
            V_d_spin_na = _subregion(_K_d_spin, _nod_mask,  _anti_mask)
            _scf_log("PAIR-DIAG",
                f"  V_d_spin decomposition (d-sector spin channel): aa={V_d_spin_aa:+.4f}  nn={V_d_spin_nn:+.4f}  an+na={V_d_spin_an + V_d_spin_na:+.4f}")

            # --- Spread of the V_irrep diagonal ---
            _diag_spread = float(np.std(_V_diag_irrep)) / max(float(np.abs(np.mean(_V_diag_irrep))), _MATH_EPS)
            _scf_log("PAIR-DIAG",
                f"  lambda_gain_rel={lambda_gain_rel:.3e}"
                f"  lambda_JT_kernel={lambda_JT_kernel:.3e}"
                f"  λ_FS={lambda_lin_max:.4e}  λ_irr={lambda_lin_max_2d:.4e}"
                f"  std(V_irrep diag)/|mean|={_diag_spread:.3e}")

            # --- Capture-fraction: irrep-projection test of the dominant FS eigenvector ---
            # Loewdin (least-squares) weight of the eigenvector inside a possibly non-orthogonal subspace, c_sub^T S_sub^+ c_sub.
            _sd_idx = [_IR_S, _IR_D]
            _G_sd   = S_irrep[np.ix_(_sd_idx, _sd_idx)]
            _c_sd   = c_irrep[_sd_idx]
            capture_sd    = float(np.clip(_c_sd @ np.linalg.pinv(_G_sd, rcond=1e-10) @ _c_sd, 0.0, 1.0))
            residual_sd   = max(0.0, 1.0 - capture_sd)
            capture_full  = float(vertex_cache['resolved_weight'])
            residual_full = float(vertex_cache['unresolved_weight'])

            _scf_log("GAP-DIAG",
                f"  φ_eig irrep-projection: (s,d)-only capture={capture_sd*100:.1f}% [unresolved {residual_sd*100:.1f}%]"
                f"  →  full basis capture={capture_full*100:.1f}% [unresolved {residual_full*100:.1f}%]  {'✓' if residual_full < 0.10 else '⚠'}")
            _scf_log("GAP-DIAG",
                f"  seed direction v_irr (multi-sector): "
                f"{_fmt_amplitude_line(v_irr, _IRREP_NAMES, top_n=6, signed=True)}")
            if residual_full > 0.25:
                _scf_log("GAP-DIAG", f"  ⚠ {residual_full*100:.1f}% of the dominant FS eigenvector still lies outside the {_N_IRREP}-function basis.")
            
            f12 = 1.0 - 2.0 * _fermi_function(ev, solver.kT)
            uA, uB, vA, vB = _get_nambu_spinors(ec)
            pair_intra6_s_k = np.sum(
                (uA[:, 0, :] * np.conj(vA[:, 1, :]) - uA[:, 1, :] * np.conj(vA[:, 0, :])
                + uB[:, 0, :] * np.conj(vB[:, 1, :]) - uB[:, 1, :] * np.conj(vB[:, 0, :])) * f12,
                axis=1)
            pair_intra7_s_k = np.sum(
                (uA[:, 2, :] * np.conj(vA[:, 3, :]) - uA[:, 3, :] * np.conj(vA[:, 2, :])
                + uB[:, 2, :] * np.conj(vB[:, 3, :]) - uB[:, 3, :] * np.conj(vB[:, 2, :])) * f12,
                axis=1)

            _scf_log("GAP-DIAG", f"  F67a: {_fmt_amplitude_line(F67_vec[_IDX_7A_ARR], _IRREP_NAMES, top_n=4)}")
            _scf_log("GAP-DIAG", f"  F67b: {_fmt_amplitude_line(F67_vec[_IDX_7B_ARR], _IRREP_NAMES, top_n=4)}")
            _scf_log("GAP-DIAG", f"  F66a_s={g_Delta_s*abs(np.dot(solver.k_weights, pair_intra6_s_k))/4.0:.3f}  F77a_d={g_Delta_ad*abs(np.dot(solver.k_weights, pair_intra7_s_k))/4.0:.3e}")
            
            lambda_channel_decomposition(self, vertex_cache, verbose=True)
            for _ln in _sector_report_lines(vertex_cache):
                _scf_log("SECTOR", _ln)

        Delta_out = np.zeros(_N_DELTA, dtype=complex)
        Delta_out[_IDX_7A_ARR] = D_out[0]
        Delta_out[_IDX_7B_ARR] = D_out[1]
        return Delta_out, F67_vec, vertex_cache


def _solve_normal_state(solver: 'RMFT_Solver', target_doping: float) -> dict:
    zero = np.zeros(_N_DELTA, dtype=complex)
    M_N, _, F67_N, n_N, mu_N, g_t_N, g_J_N, _, _, _ = solver.refine_M_state(target_doping, 0.0, zero, zero, damping=0.45, verbose=False)

    _, _, _, V_JT_corr_N_0, _ = solver._make_vertex_params(target_doping, M_N, 0.0, zero, zero, n_N, mu_N, g_t_N, g_J_N)
    Q_N = 0.0
    _done = False
    for _it in range(12):
        Q_out, _info = solver.solve_Q_slave(target_doping, M_N, Q_N, zero, zero, n_N, mu_N, g_t_N, g_J_N, V_JT_corr_N_0, vertex_cache=None)
        _done = abs(Q_out - Q_N) < 1e-4
        Q_N = Q_out
        if _done:
            break

    _, Fcl_N, _, V_JT_corr_N, _ = solver._make_vertex_params(target_doping, M_N, Q_N, zero, zero, n_N, mu_N, g_t_N, g_J_N)
    _dFc_ref = solver._dFc_reference(target_doping, mu_N, g_t_N, g_J_N)
    _dFc_N   = solver._dF_cluster(M_N, Q_N, target_doping, zero, F67_N, n_N, mu_N, g_t_N, g_J_N)
    _F_bdg_N = Fcl_N['F_can'] + (_dFc_N - _dFc_ref)

    return {
        'M': M_N, 'Q': Q_N, 'Delta_vec': zero, 'F67_vec': F67_N,
        'n_kspace': n_N, 'mu': mu_N, 'g_t': g_t_N, 'g_J': g_J_N,
        'V_JT_corr': V_JT_corr_N,
        'F_bdg': _F_bdg_N,
        'K_eff': Fcl_N['K_eff'],
        'hessian_result': {'eigenvalues': None},
        'hess_ok': False, 'hess_why': 'not verified here',
        'converged': bool(_done), 'target_doping': target_doping,
        'history': {'F_cluster': [Fcl_N['F_per_site']]},
        'vertex_cache': None, 'scf_iteration_converged': bool(_done), 'past_qcp': False,
    }

def plot_ground_state_comparison(results: Dict[str, dict], labels: Optional[Dict[str, str]] = None):
    default_labels = {'ref': 'ref (SC+JT)', 'normal': 'normal (Δ=0)', 'SC_Q0': 'SC, Q=0'}
    default_colors = {'ref': 'tab:blue', 'normal': 'tab:gray', 'SC_Q0': 'tab:orange'}
    labels = {**default_labels, **(labels or {})}

    preferred_order = ['ref', 'normal', 'SC_Q0']
    ordered_keys = ([k for k in preferred_order if results.get(k)]
                     + [k for k in results if k not in preferred_order and results.get(k)])
    scenarios = [(k, results[k]) for k in ordered_keys]
    if not scenarios:
        raise ValueError("plot_ground_state_comparison: no non-empty results to plot.")

    _gs_fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    _gs_fig.suptitle('Ground-state comparison: ref vs. normal vs. SC-only (Q=0)', fontsize=14, fontweight='bold')

    # (0,0) F_cluster per iteration
    ax = axes[0, 0]
    for key, res in scenarios:
        c, lbl = default_colors.get(key), labels.get(key, key)
        ok  = bool(res.get('hess_ok', True))
        why = res.get('hess_why', '')
        ax.plot(res['history']['F_cluster'], '-' if ok else '--', color=c, linewidth=2,
                 label=lbl + ('' if ok else f'  ⚠ {why}'))
        ax.axhline(res.get('F_bdg', np.nan), color=c, linestyle=':', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Free energy (eV)')
    ax.set_title('F_cluster per iteration  (dotted = converged F_bdg)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) Final F_bdg bar
    ax = axes[0, 1]
    keys   = [k for k, _ in scenarios]
    F_vals = [res.get('F_bdg', np.nan) for _, res in scenarios]
    bars = ax.bar([labels.get(k, k) for k in keys], F_vals,
                   color=[default_colors.get(k, 'gray') for k in keys], alpha=0.85)
    for (k, res), bar in zip(scenarios, bars):
        if not res.get('hess_ok', True):
            bar.set_hatch('//'); bar.set_edgecolor('red')
    finite = [(k, v) for k, v in zip(keys, F_vals) if np.isfinite(v)]
    if finite:
        best_key = min(finite, key=lambda kv: kv[1])[0]
        ax.set_title(f'Final F_bdg  (lowest = ground state → {labels.get(best_key, best_key)})', fontsize=11)
    ax.set_ylabel('F_bdg (eV)'); ax.grid(True, alpha=0.3, axis='y'); ax.tick_params(axis='x', labelrotation=10)

    # (1,0) AFM order parameter (Gamma6 channel) convergence, all scenarios overlaid.
    ax = axes[1, 0]
    for key, res in scenarios:
        m0_hist = [float(m[0]) for m in res['history']['M']]
        ax.plot(m0_hist, color=default_colors.get(key), linewidth=2, label=labels.get(key, key))
    ax.set_xlabel('Iteration'); ax.set_ylabel('Magnetization M[Γ6]')
    ax.set_title('AFM order parameter convergence', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (1,1) |Q| convergence -- the key symmetry-forbiddenness check: "normal" should relax back to ~0 on its own (Q was NOT forced there), while "ref" settles at a finite value.
    ax = axes[1, 1]
    for key, res in scenarios:
        ax.plot(res['history']['Q'], color=default_colors.get(key), linewidth=2, label=labels.get(key, key))
    ax.axhline(0.0, color='k', linewidth=0.7, linestyle=':')
    ax.set_xlabel('Iteration'); ax.set_ylabel('|Q| (Å)')
    ax.set_title('JT distortion: "normal" should self-consistently relax to ≈0', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _gs_path = "ground_state_comparison.png"
    _gs_fig.savefig(_gs_path, dpi=150)
    plt.close(_gs_fig)
    _scf_log("PLOT", f"  Ground-state comparison saved -> {_gs_path}")


if __name__ == "__main__":
    with _log_lock:
        print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  SC-Activated JT Model - Variational Free Energy Minimization     ║
    ║  Implements: SC → Γ₆–Γ₇ mixing → JT via ∂F/∂M = ∂F/∂Q = 0         ║
    ║  Optimizer: Unified 5D pipeline (DE→GP→TuRBO→LocalRefine)         ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """, flush=True)

    params = ModelParams(
        t_pd             = 0.370,
        t_prime_ratio    = -0.17,
        t_dprime_ratio   = 0.08,
        U_dd             = 2.200,
        lambda_soc       = 0.024,
        Delta_tetra      = -0.057,
        g_JT             = 0.315,
        K_lattice        = 2.300,
        lambda_hop       = 1.200,
        g_Eg2            = 0.100,
        K_lattice_Eg2    = 6.500,
        Delta_CT         = 1.750,
        Delta_B1g_static = 5e-5,
        hybrid_scale     = 5.000,
        Upp_ratio_bare   = 0.400,
        J_H_ratio        = 0.09,
        M_eff            = 8.0,
        kT               = 0.005,
        M_axis           = 'z',
        tol              = 1e-4,
        )

    target_doping = 0.160
    doping_margin = 0.20          # scan covers target ± 20 %
    min_doping    = max(target_doping * (1.0 - doping_margin), _G_T_COHERENCE_MIN / (2.0 - _G_T_COHERENCE_MIN))
    max_doping    = target_doping * (1.0 + doping_margin)
    
    Delta_vec_ans = np.zeros(_N_DELTA, dtype=complex)
    F67_vec_ans = np.zeros(_N_DELTA, dtype=complex)
    seed_complex_phase = {'s': 0.0, 'd': 0.0}
    # = {'s': 0.0, 'd': np.pi / 2.0}    s + i·d  or = {'d7a': 0.0, 'd7b': np.pi, 'px': np.pi / 2}    d7a real, d7b anti-phase, px rotated

    _scf_log("INIT",
        f"t_pd={params.t_pd:.3f} eV  t_prime={params.t_prime_ratio:.3f}  t_dprime={params.t_dprime_ratio:.3f}  U_dd={params.U_dd:.4f} eV  λ_SOC={params.lambda_soc:.3f} eV"
        f"  Δ_tetra={params.Delta_tetra:.3f} eV  bare g_JT={params.g_JT:.3f} eV  K_lattice={params.K_lattice:.3f}  lambda_hop={params.lambda_hop:.3f}  Δ_CT={params.Delta_CT:.3f} eV"
        f"  Δ_ip={params.Delta_B1g_static:.1e} eV  hyb_scale={params.hybrid_scale:.2f}  M_eff={params.M_eff}  Z={_Z}  kT={params.kT*1000:.2f} meV  N_k={params.N_k}  δ={target_doping:.4f}")
    _scf_log("DERIVED",
        f"  U_pp={params.U_pp:.4f}  J_H={params.J_H:.4f}  t0={params.t0:.4f} eV  Δ_CF={params.Delta_CF:.4f} eV "
        f"  kappa_G6={params.kappa_doublets[0]:.4f} eV  Γ₇split={params.g7split:.5f} eV [{'⚠ < 2kT' if abs(params.g7split) < 2.0 * params.kT else '✓'}]")

    # J_eff chain: κ from the two-hole ED, Z_afm_eff from J2/J3 frustration, 1/(4 s_eff² from Wigner–Eckart (the Weiss-field sum Σ_j J_ij)
    g_t, g_J = params.get_gutzwiller_factors(target_doping)
    _J_A1g0 = params.exchange_channels(0.0, g_J)[0]
    _scf_log("DERIVED",
        f"  J-chain(Γ6): κ_G6={params.kappa_doublets[0]*1000:.2f} meV [=(E_T−E_S)/(4 s_eff²)]  ×g_J={g_J:.3f}  → single-bond J_A1g={_J_A1g0[0]*1000:.2f} meV"
        f"  ×Z_afm_eff={params.Z_afm_eff:.2f} → J_q(π,π)={params._exchange_J_q(_J_A1g0, np.pi, np.pi)[0]*1000:.1f} meV (=J_eff, Weiss-field scale; nearest-neighbour coupling alone: {_J_A1g0[0]*1000:.1f} meV)")

    bandwidth = _RPA_BW_FACTOR * g_t * params.t0
    E_G7b = params.Delta_CF + params.g7split
    _scf_log("DERIVED",
        f"Γ7b (E={E_G7b:.3f} eV) vs. estimated Γ6/Γ7a bandwidth ({bandwidth:.3f} eV): "
        + ("within bandwidth — bands may cross/overlap; handled exactly by the full 3-band " if E_G7b < bandwidth else "well separated from the Γ6/Γ7a manifold."))

    solver_ref    = RMFT_Solver(copy.deepcopy(params))
    solver_normal = RMFT_Solver(copy.deepcopy(params))
    solver_Q0     = RMFT_Solver(copy.deepcopy(params))
    
    # --- 1. Reference SCF ---
    _scf_log("REF-SCF", "="*60) 
    tasks = {
        "ref":    lambda: solver_ref.solve_self_consistent(target_doping, seed_complex_phase, Delta_vec_ans, F67_vec_ans, force_delta_zero=False, force_Q_zero=False, verbose=True),
        "normal": lambda: _solve_normal_state(solver_normal, target_doping),
        "SC_Q0":  lambda: solver_Q0.solve_self_consistent(target_doping, seed_complex_phase, Delta_vec_ans, F67_vec_ans, force_delta_zero=False, force_Q_zero=True)
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(task): name for name, task in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                _scf_log("PARALLEL", f"Task {name} failed: {e}")

    ref_result = results.get("ref")
    # --- 2. SCF diagnostics ---
    if ref_result is not None:
        _ref_M    = ref_result['M']
        _ref_Q    = ref_result['Q']
        _ref_mu   = ref_result['mu']
        _ref_n_kspace = ref_result['n_kspace']

        _scf_log("REF-SCF",
            f"  converged={ref_result['converged']}"
            f"  (raw SCF flag: {ref_result.get('scf_iteration_converged')})"
            f"  mott_suspect={ref_result.get('mott_suspect', False)}"
            f"  past_qcp={ref_result.get('past_qcp')}"
            f"  det_afm(final)={ref_result.get('det_afm_final', float('nan')):+.5f}")
        _scf_log("REF-SCF",
            f"  M={np.array2string(_ref_M, precision=4)}  Q={_ref_Q:+.5f} Å  μ={ref_result['mu']:.4f} eV"
            f"  Δs={ref_result['Delta_s']:.5f}  Δd={ref_result['Delta_d']:.5f}"
            f"  IrrepR={ref_result['selection_ratio']:.4f}"
            f"  JT={'ALLOWED' if ref_result['selection_ratio'] > _JT_ACT_THR else 'BLOCKED'}")

        # --- 3. K_eff trajectory (same M, Q, μ; only Δ varies) ---
        _, Fcl_N,  *_ = solver_ref._make_vertex_params(target_doping, results['normal']['M'], 0.0, np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), results['normal']['n_kspace'], results['normal']['mu'], g_t, g_J)
        Fcl_SC = ref_result['F_cluster']

        _F_tot_N,  _ = solver_ref._make_Q_profile(target_doping, results['normal']['M'], np.zeros(_N_DELTA, dtype=complex), np.zeros(_N_DELTA, dtype=complex), results['normal']['n_kspace'], results['normal']['mu'], g_t, g_J, results['normal']['V_JT_corr'], 0.0, None)
        _F_tot_SC, _ = solver_ref._make_Q_profile(target_doping, _ref_M, ref_result['Delta_vec'], ref_result['F67_vec'], _ref_n_kspace, _ref_mu, g_t, g_J, ref_result['V_JT_corr'], 0.0, ref_result['vertex_cache'])

        _dN  = _F_tot_N(0.0)  - _F_tot_N(results['normal']['Q'])
        _dSC = _F_tot_SC(0.0) - _F_tot_SC(ref_result['Q'])

        if abs(results['normal']['Q']) < _Q_THR_REL and abs(ref_result['Q']) > _Q_THR_REL and _dSC > _MATH_EPS:
            _jt = '✓ SC-triggered JT'
        elif abs(results['normal']['Q']) > _Q_THR_REL and _dN > _MATH_EPS:
            _jt = '✗ spontaneous JT'
        elif abs(results['normal']['Q']) < _Q_THR_REL and abs(ref_result['Q']) < _Q_THR_REL:
            _jt = '— no JT'
        else:
            _jt = '? unclear'

        _scf_log("SCF-RES",
            f"JT (F_total): N: Q={results['normal']['Q']:+.4f} Å ΔF={_dN*1e3:+.2f} meV | "
            f"SC: Q={ref_result['Q']:+.4f} Å ΔF={_dSC*1e3:+.2f} meV → {_jt}")

        _scf_log("SCF-RES",
            f"K_eff (diag, Q=0): N={Fcl_N['K_eff']:+.3f}  SC={Fcl_SC['K_eff']:+.3f}  "
            f"[SC parts: K_bulk={Fcl_SC['K_bulk']:+.3f}  E_cond''={Fcl_SC['K_econd']:+.3f}  "
            f"K_cluster={Fcl_SC['Delta_K_cluster']:+.3f}  K_schur={Fcl_SC['K_schur']:.3f}]")

        # --- 4. Static-field test (explicit B1g symmetry breaker switched off) ---
        solver_ns = RMFT_Solver(replace(params, Delta_B1g_static=0.0))
        _, Fcl_ns, _, V_JT_corr_ns, _ = solver_ns._make_vertex_params(target_doping, _ref_M, _ref_Q, ref_result['Delta_vec'], ref_result['F67_vec'], ref_result['n_kspace'], ref_result['mu'], g_t, g_J)
        s0 = max(0.5 * abs(_ref_Q), 1e-2)
        _qscale = max(1.0, 2.2 * s0 / (_Q_TRUST_ABS + _Q_TRUST_REL * s0))
        Qpm, _qedge = {}, {}
        for tag, q0 in (('+', +s0), ('-', -s0)):
            Qpm[tag], _qinfo = solver_ns.solve_Q_slave(target_doping, _ref_M, q0, ref_result['Delta_vec'], ref_result['F67_vec'], ref_result['n_kspace'], ref_result['mu'], g_t, g_J, V_JT_corr_ns, q_trust_scale=_qscale)
            _qedge[tag] = bool(_qinfo.get('edge', False))
        _F_ns, _ = solver_ns._make_Q_profile(target_doping, _ref_M, ref_result['Delta_vec'], ref_result['F67_vec'], ref_result['n_kspace'], ref_result['mu'], g_t, g_J, V_JT_corr_ns)
        F0, Fp, Fm = _F_ns(0.0), _F_ns(Qpm['+']), _F_ns(Qpm['-'])

        qa, qb = max(abs(Qpm['+']), abs(Qpm['-'])), min(abs(Qpm['+']), abs(Qpm['-']))
        _wlim      = _qedge['+'] or _qedge['-']
        _same_side = (Qpm['+'] > 0.0) and (Qpm['-'] < 0.0)
        _mag_ok    = (qa > 1e-3) and (qa >= 0.5 * abs(_ref_Q)) and (qb >= 0.8 * qa)
        _e_gain    = (min(Fp, Fm) < F0 - 1e-4)                     # >= 0.1 meV energy gain

        if _mag_ok and _same_side and _e_gain and not _wlim:
            verdict = '✓ spontaneous (SC-driven): a Z2 pair ±Q survives without the static field'
        elif _mag_ok and _same_side and _wlim:
            verdict = (f'✓ likely spontaneous, but |Q*| is a LOWER BOUND '
                    f'(window-limited: edge min (+,-) = {_qedge["+"]}, {_qedge["-"]})')
        elif qa <= max(0.1 * abs(_ref_Q), 1e-3) and not _wlim:
            verdict = '— no distortion without the static field (Q is the amplified Δ_B1g_static response)'
        elif _wlim:
            verdict = '? inconclusive: the minimum sits at the edge of the local window (|Q*| is only a lower bound)'
        else:
            verdict = '≈ partly spontaneous / partly static-field driven'

        _scf_log("SCF-RES", f"STATIC-FIELD: Q*(Δ_B1g_static=0) = ({Qpm['+']:+.5f}, {Qpm['-']:+.5f}) Å  [F−F(0) = {Fp-F0:+.3e}, {Fm-F0:+.3e} eV]  vs  "
            f"Q(Δ_B1g_static={params.Delta_B1g_static:.1e} eV) = {_ref_Q:+.5f} Å  → {verdict}")
        
        # --- 5. Hessian ---
        compute_and_log_hessian_diagnostics(target_doping, g_t, g_J, results, solvers={'ref': solver_ref, 'normal': solver_normal, 'SC_Q0': solver_Q0})

        # --- 6. Coherence lengths (single line; orbital selectivity shown when relevant) ---
        _scf_log("SCF-RES", f"Gap eq: λ_lin_max={ref_result['lambda_lin_max']:.4f}")
        if not ref_result['valid_BdG']:
            _xi_note = f"⚠ ξ_nodal/a={ref_result['xi_nodal']:.2f} < 2 (marginal)"
        elif ref_result['orbital_selective']:
            _xi_note = (f"✓ ξ_nodal/a={ref_result['xi_nodal']:.2f}"
                        f"  ξ_anti/a={ref_result.get('xi_antinodal', float('nan')):.2f}"
                        f"  ORBITAL-SEL: ξ_Γ6/a={ref_result['xi_Gamma6']:.2f}  ξ_Γ7/a={ref_result['xi_Gamma7']:.2f}")
        else:
            _xi_note = (f"✓ ξ_nodal/a={ref_result['xi_nodal']:.2f}"
                        f"  ξ_anti/a={ref_result.get('xi_antinodal', float('nan')):.2f} (uniform)")
        _scf_log("SCF-RES", f"  {_xi_note}")

        # --- 7. Three independent Tc estimates ---
        #   Tc₁: McMillan (analytical, ω_SF = J_A1g[0])
        _pre_Delta_total = float(np.linalg.norm(np.asarray(_sc_gap_amplitude(ref_result))))
        _scf_log("TC-PRELIM", f"Pre-BO Tc estimates at |Δ|={_pre_Delta_total*1000:.3f} meV")
        _sc_viable = (not ref_result.get('mott_suspect', False)) and (float(g_t) >= _G_T_COHERENCE_MIN) and bool(ref_result['converged'])
        if _sc_viable:
            _omega_SF  = float(ref_result['J_eff'])
            _lmax_safe = max(ref_result['lambda_lin_max'], _MATH_EPS)
            _Tc1_eV    = float((_omega_SF / _MAD_DENOM) * np.exp(-_MAD_NUM * (1.0 + _lmax_safe) / _lmax_safe))
            _scf_log("TC-PRELIM",
                    f"  Tc₁(Allen–Dynes-SF): ω_SF(J_eff=J_q(π,π), Weiss-field sum; NN coupling only would be {float(_omega_SF)*1000:.1f} meV)"
                    f"  → {_Tc1_eV*1000:.2f} meV  ({_Tc1_eV*_EV_TO_K:.1f} K) [denom={_MAD_DENOM} (Allen–Dynes SF compromise); for phonon Debye use 1.45]")
        
        #   Tc₂: linearized normal
        _tc_lin = solver_ref.compute_Tc_linearized_normal(target_doping)
        _Lam58 = float(np.interp(solver_ref.kT, _tc_lin['T'], _tc_lin['Lambda']))
        _scf_log("TC-PRELIM",
            f"  Tc₂(linearized gap eq)={_tc_lin['Tc']*1000:.2f} meV"
            f"  ({_tc_lin['Tc']*_EV_TO_K:.1f} K)  status={_tc_lin['status']}"
            f"  λ_FS={_tc_lin['lambda_lin_max_FS']:.4f}"
            f"{'  ⚠ complex mode' if _tc_lin['complex_mode'] else ''}"
            f"  Λ(kT={solver_ref.kT*_EV_TO_K:.0f} K)={_Lam58:.2f};  leading channels {_tc_lin['leading_channels']}")
        if _tc_lin['status'] != 'inside':
            _scf_log("TC-PRELIM",
                f"  ⚠ Λ=1 crossing lies {_tc_lin['status'].upper()} T_grid range "
                f"[{_tc_lin['T'][0]*_EV_TO_K:.0f}, {_tc_lin['T'][-1]*_EV_TO_K:.0f}] K — extend T_grid for a Tc value")

        #   Tc₃: thermodynamic Tc from reference SCF (first-order aware free-energy crossing)
        try:
            _Tc_ref_thermo = solver_ref.compute_Tc_thermodynamic(
                target_doping,
                sc_result = ref_result,
                T_min     = 1e-4,
                T_max     = 0.25,
                n_scan    = 12,
                n_bisect  = 8,
            )
            _Tc_rt = _Tc_ref_thermo['Tc']
            _Tc_spinodal = _Tc_ref_thermo['Tc_spinodal']   # metastability limit, companion to Tc₃
            _scf_log("TC-THERMO",
                        f"  Tc₃(thermo)={_Tc_rt*1000:.2f} meV ({_Tc_rt*_EV_TO_K:.1f} K)  Tc_spinodal(spinodal)={_Tc_spinodal*1000:.2f} meV"
                        f"  order={_Tc_ref_thermo['transition_order']}"
                        f"  Δ_jump={_Tc_ref_thermo['Delta_jump']*1000:.2f} meV")
            _scf_log("TC-THERMO", f"  vs. target {_TC_TARGET_K:.0f} K: Tc₃={_Tc_rt*_EV_TO_K:.1f} K = {_Tc_rt*_EV_TO_K/_TC_TARGET_K:.0%} of the target")
            _gi = _Tc_ref_thermo['gap_info']
            _scf_log("TC-THERMO",
                        f"  spectral FS gap: Δ_ref={_gi['Delta_ref']*1000:.3f}  Δ_max={_gi['Delta_max']*1000:.3f}  Δ_min={_gi['Delta_min']*1000:.3f} meV"
                        f"  [dominant irrep: {_gi['dominant']}{', nodal FS points' if _gi['nodal'] else ''}]   vs amplitude |Δ_vec|={_pre_Delta_total*1000:.3f} meV")
            _scf_log("TC-THERMO", f" weak-coupling BCS for this gap shape: {_gi['coupling_regime']}")
            if not _gi['gap_saturated']:
                _scf_log("TC-THERMO", f"  ⚠ SCF state at kT={_gi['T_state']*1000:.2f} meV is not ≲0.25·Tc: Δ_ref is below its T=0 value, ratio underestimated")
            if _Tc_spinodal > 1e-6:
                _uplift_r = (_Tc_rt - _Tc_spinodal) / _Tc_spinodal * 100.0
                _scf_log("TC-THERMO",
                            f"  Tc uplift (Tc₃ vs Tc_spinodal): {_uplift_r:+.1f}%"
                            f"  [{'first-order dominant' if abs(_uplift_r)>20 else 'weakly first-order' if abs(_uplift_r)>5 else 'effectively second-order'}]")
        except Exception as _Tc_rt_err:
            _scf_log("TC-THERMO", f" Tc₃(thermo) failed: {_Tc_rt_err}")
    else:
        _scf_log("TC-THERMO", " Thermodynamic Tc skipped: reference SCF not converged.")

    # --- 8. Ground-state comparison plot (ref vs. normal vs. SC-only Q=0) ---
    _scf_log("PLOT", "=" * 60)
    plot_ground_state_comparison(results)