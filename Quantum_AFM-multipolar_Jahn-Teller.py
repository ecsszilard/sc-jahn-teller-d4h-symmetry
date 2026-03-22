import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import numpy as np
import opt_einsum as oe
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.optimize import brentq, differential_evolution
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings
import copy
import time as _time
import concurrent.futures
import sys
import threading as _threading
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from threadpoolctl import threadpool_limits as _tpl_ctx

_log_lock = _threading.Lock()

# ── Module-level physical and numerical constants ─────────────────────────────
# RPA / Moriya damping
_ALPHA_MORIYA:      float = 0.15    # Moriya mode-coupling coefficient baseline: Γ_M = α_M · J_eff · t_eff
                                    # The *effective* α_M is computed at runtime by _moriya_alpha();
_MORIYA_C:          float = 0.30    # dimensionless prefactor in α_M = C · δ · (t_eff / J_eff); tuned to give α_M ~ 0.15 at δ~0.15, t_eff/J_eff ~ 3
_CHI_QQ_CLAMP:      float = 0.95    # χ_QQ upper bound fraction of 1/V_JT (prevents self-crossing JT instability)
_RPA_DET_FLOOR:     float = 1e-4    # absolute det fallback floor — NUMERICAL emergency only.
                                    # Moriya damping is the primary QCP regulariser; this floor is only reached if damping is insufficient.
_RPA_DET_WARN:      float = 0.09    # QCP proximity warning threshold for diagnostics and SCF adaptive mixing.
                                    # Used in: SCF near-critical detection, BO near_qcp flag
_CHI_SQ_EPS:        float = 1e-12   # numerical noise threshold for χ_SQ/χ_QS symmetry enforcement

# SCF numerics
_FERMI_ARG_CLIP:    float = 100.0   # clip argument of exp() in Fermi function
_ENTROPY_CLIP:      float = 1e-12   # lower clip for f in entropy -f·ln(f)
_KT_FLOOR:          float = 1e-8    # temperature floor to avoid division by zero
_DEN_DERIV_FLOOR:   float = 1e-12   # ∂n/∂μ floor in Newton μ-finder
_BRENTQ_TOL:        float = 1e-5    # brentq μ-bracketing tolerance
_FD_MASK_DF:        float = 1e-12   # |Δf| mask threshold in χ₀ Lehmann sums
_FD_MASK_DE:        float = 1e-6    # |ΔE| mask threshold in χ₀ Lehmann sums
_FD_MASK_DE8:       float = 1e-8    # tighter |ΔE| mask for d²F/dM² off-diagonal
_VF_FLOOR:          float = 1e-4    # Fermi velocity floor (prevents 1/|v_F|→∞ at hot spots)
_VF_FLOOR_TIGHT:    float = 1e-8    # tighter v_F floor in linearised gap kernel
_QQ_DELTA_THRESH:   float = 1e-8    # |Δ| threshold below which χ_DQ enforced zero
_JCHI_HARD_REJECT:  float = 2.0     # J·χ_SS > this → score = 0 (deeply AFM, SC impossible)
_V_CUT:             float = 20.0    # pairing vertex near-divergence detector threshold
_G_T_COHERENCE_MIN: float = 0.10    # g_t = 2δ/(1+δ) floor for coherent ZRS band; below this δ < 0.053 the Fermi surface is incoherent.
                                    # Used in: _scf_jacobi_kick, post-SCF Mott filter, _eval_constraints H4, _score, and __main__ doping floor.
_FS_SAMPLING:       float = 2.3     # integration window around the Fermi level
_FS_N_SAMPLE:       int   = 42      # FS k-points kept for geometric representation (topology, hot spots)
_FS_N_VERTEX:       int   = 72      # FS k-points used in the vertex q-loop. _FS_N_VERTEX > _FS_N_SAMPLE is allowed — the gap kernel
                                    # samples the full k-grid while _FS_N_SAMPLE controls topology outputs. Angular resolution need to resolve the d-wave node at (π/2,π/2) and the B₁g anti-nodal hot spots.
_Q_THR_REL:         float = 0.02    # fraction of lambda_hop; Q change below this skips vertex rebuild
_DELTA_THR_REL:     float = 0.15    # relative Δ change threshold for vertex cache invalidation
_M_THR_REL:         float = 0.03    # absolute M change threshold (M is O(0.1–0.5))
_DELTA_THR_ABS:     float = 0.008   # absolute Δ floor: guards against spurious rebuilds near Δ≈0

# Anderson mixing
_ANDERSON_TIKHONOV: float = 1e-8    # Tikhonov β / diag_max in Anderson normal equations
_ANDERSON_TRUST:    float = 2.5     # trust-region step-size limit (multiples of simple step)
_ANDERSON_W_LO:     float = 0.4     # lower blend weight between Anderson and simple mixing
_ANDERSON_W_HI:     float = 0.9     # upper blend weight

# Bayesian optimiser scoring
_BO_MAX_WORKERS:    int   = 6       # hard ceiling on ThreadPoolExecutor workers
_BO_OPT_JCHI:       float = 0.875   # optimal J·χ_SS for Gauss gate (near-QCP but still metallic)
_BO_SIG_JCHI:       float = 0.15    # Gaussian width σ for J·χ_SS gate
_BO_JCHI_FLOOR:     float = 0.3     # score floor when J·χ unavailable (jchi≈0)
_BO_W_STONER_BAD:   float = 0.20    # score weight when AFM Stoner criterion violated
_BO_SPONT_JT_PEN:   float = 0.05    # penalty factor when G3[2,2] ≤ 0 (spontaneous-JT risk)
_BO_G_FALLBACK:     float = 5e-3    # overall scale for G-matrix proxy (no-gap region)
_BO_SIGMOID_W:      float = 0.30    # sigmoid width for G22 continuous gate
_BO_SC_HESS_SIG:    float = 0.05    # eV — sc_hessian_f sigmoid width around lambda_min=0
_BO_JCHI_NOISE:     float = 0.05    # J·χ below this is numerical noise, apply floor

def _moriya_alpha(doping: float, t_eff: float, J_eff: float) -> float:
    """
    Moriya spin-fluctuation damping: Γ_M = α_M · J_eff · t_eff,
        α_M = max(C · δ · (t_eff / J_eff),  _ALPHA_MORIYA)

    Vanishes at half-filling (δ→0, long-range AFM), grows with doping as
    metallic screening broadens the QCP. Floor _ALPHA_MORIYA guards against
    numerical runaway at very low doping.
    """
    abs_d  = max(abs(doping), 1e-4)
    J_safe = max(abs(J_eff),  1e-9)
    t_safe = max(abs(t_eff),  1e-9)
    alpha  = _MORIYA_C * abs_d * (t_safe / J_safe)
    return float(max(alpha, _ALPHA_MORIYA))

def _scf_log(tag: str, msg: str, verbose: bool = True) -> None:
    """Thread-safe logger.  tag is left-padded to 18 chars so columns stay aligned."""
    if not verbose:
        return
    with _log_lock:
        print(f"[{tag:<18s}] {msg}", flush=True)

@dataclass
class ModelParams:
    """
    Derived fields (set in __post_init__):
      t0       = t_pd²/Δ_CT            (dd effective hopping, eV)
      Delta_CF = Γ₆–Γ₇ SOC+CF gap     (eV)
      U        = u·t0                  (Hubbard U, eV)
      U_mf     = Z·J_CT/2              (bare Weiss amplitude, eV)
      doping_0 = z_ZRS/(1−z_ZRS)       (ZRS coherence crossover; floor in J_eff only)
    """
    # --- Primary inputs ---
    t_pd:          float      # eV    pd hybridisation integral (independent of Δ_CT; typ. 0.8–1.5 eV)
    u:             float      # —     U/t0 ratio; U = u·t0 = u·t_pd²/Δ_CT (charge-transfer: typ. 6–12)
    lambda_soc:    float      # eV    atomic SOC λ t2g shell; determines Γ₆–Γ₇ splitting
    Delta_tetra:   float      # eV    tetragonal axial CF Δ_tet·Lz²; negative = z-compression
                              #       Partial cancellation with SOC tunes Γ₆–Γ₇ gap independently of λ
    g_JT:          float      # eV/Å  Jahn–Teller electron–phonon coupling
                              #       increasing g_JT beyond the SC-triggered window is risky, because spontaneous JT (G3[2,2] < 0) always precedes the RPA cross-channel divergence.
    K_lattice:     float      # eV/Å² bare lattice spring constant (phonon stiffness, no exchange)
                              #       Physical: ω_JT=60meV, mass~Cu → K~1-2 eV/Å²; K_eff < K_lattice after exchange correction
                              #       K_lattice must satisfy: K_spont = g²/Δ_CF < K < g²/(0.05·π·t0) for SC-JT window (λ_JT > 0.05).
    lambda_hop:    float      # Å     hopping decay length for B₁g anisotropy: t(Q) = t0·exp(±Q/λ_hop)

    # --- Charge-transfer / RPA / gap symmetry ---
    Delta_inplane: float      # eV    B2g in-plane anisotropy Δ_ip·(Lx²−Ly²); splits Γ₇ into Γ₇a+Γ₇b
                              #       (preserves Kramers, prevents spontaneous JT from residual Γ₇ degeneracy)
    Delta_CT:      float      # eV    charge-transfer gap (ZSA scale); sets scale for CT-insulator crossover
    omega_JT:      float      # eV    JT phonon frequency (40–80 meV); enters only D_phonon = 2/ω_JT
                              #       All free-energy magnitudes use adiabatic g²/K

    # --- SCF numerical hyper-parameters (tune once, do NOT Bayesian-optimise) ---
    mu_LM:         float      # Levenberg–Marquardt floor for M Newton step (default 4.0), larger → smaller γ_M → more conservative M update.
    ALPHA_HF:      float      # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton; default 0.2)

    # --- Numerics ---
    Z:             int        # 2D square lattice coordination number
    nk:            int        # k-grid points per direction (even required for commensurate q_AFM=(π,π))
    kT:            float      # eV  temperature — keep kT < Tc to allow gap to open;
    a:             float      # Å   lattice constant (used only for ξ/a; set to physical value for correct units)
    max_iter:      int
    tol:           float
    mixing:        float

    def __post_init__(self):
        evals, _evecs_soc = np.linalg.eigh(_build_soc_cf_hamiltonian(self.lambda_soc, self.Delta_tetra, self.Delta_inplane))
        self.Delta_CF: float     = float(evals[2] - evals[0])
        self.g7split: float      = float(evals[4] - evals[2])    # Γ₇a–Γ₇b internal split
        self.U_gamma: np.ndarray = _evecs_soc                    # Diagonalise H_SOC+CF; U_gamma columns = eigenvectors (ascending energy):
        self._U4:     np.ndarray = _evecs_soc[:, 0:4]            # _U4 = U_gamma[:, 0:4] is the 4-dim BdG projection (exact when Δ_CF ≫ kT).

        _Sz_t2g = np.kron(np.eye(3, dtype=complex), 0.5 * np.array([[1, 0], [0, -1]], dtype=complex))
        _v6 = _evecs_soc[:, 0]   # first Kramers partner of Γ₆
        _v7 = _evecs_soc[:, 2]   # first Kramers partner of Γ₇a
        _me6 = abs(float(np.real(_v6.conj() @ _Sz_t2g @ _v6)))
        _me7 = abs(float(np.real(_v7.conj() @ _Sz_t2g @ _v7)))
        # Clip: lower 0.1 guards the near-degenerate limit where numerical eigenvector mixing can reduce _me6 artificially.
        self.eta   = float(np.clip(_me7 / max(_me6, 1e-9), 0.1, 5.0))  # η_Sz

        # η_J: orbital-character-derived exchange weight ratio J_Γ₇/J_Γ₆.
        # Each irrep has a different mix of d_xz, d_yz, d_xy character; since J ∝ t² and hopping is orbital-selective (t_xz ∝ tx, t_yz ∝ ty, t_xy ∝ (tx+ty)/2), the exchange felt by Γ₆ and Γ₇ differs when Q≠0.
        # mL↔orbital unitaries (t2g standard):
        #   |mL=+1⟩ = −(1/√2)(|d_xz⟩ + i|d_yz⟩)
        #   |mL= 0⟩ = |d_xy⟩
        #   |mL=−1⟩ = +(1/√2)(|d_xz⟩ − i|d_yz⟩)
        _u_xy = np.array([0, 1, 0], dtype=complex)
        _u_xz = (-1.0/np.sqrt(2)) * np.array([1,  0, -1], dtype=complex)
        _u_yz = (-1j /np.sqrt(2)) * np.array([1,  0,  1], dtype=complex)
        _I2c  = np.eye(2, dtype=complex)
        _P_xz = np.kron(np.outer(_u_xz, _u_xz.conj()), _I2c)
        _P_yz = np.kron(np.outer(_u_yz, _u_yz.conj()), _I2c)
        _P_xy = np.kron(np.outer(_u_xy, _u_xy.conj()), _I2c)
        # Orbital weights in Γ₆ and Γ₇a (first Kramers partner of each)
        self._w6_xz = float(np.real(_v6.conj() @ _P_xz @ _v6))
        self._w6_yz = float(np.real(_v6.conj() @ _P_yz @ _v6))
        self._w6_xy = float(np.real(_v6.conj() @ _P_xy @ _v6))
        self._w7_xz = float(np.real(_v7.conj() @ _P_xz @ _v7))
        self._w7_yz = float(np.real(_v7.conj() @ _P_yz @ _v7))
        self._w7_xy = float(np.real(_v7.conj() @ _P_xy @ _v7))

        self.t0: float = self.t_pd**2 / max(self.Delta_CT, 1e-9)
        self.U: float = self.u * self.t0
        _dct  = max(self.Delta_CT, 1e-9)
        _U    = max(self.U, 1e-9)
        _J_ct: float = (2.0 * self.t_pd**4 / _dct**2) * (1.0 / _U + 1.0 / (_dct + _U / 2.0))

        # ZSA superexchange (CT insulator): 1/U (dd, Mott) + 1/(Δ_CT+U/2) (pp, ZR).
        # z_ZRS ≈ t_pd²/(Δ_CT²+t_pd²), doping_0 = z/(1−z): ZRS coherence scale (loss below δ₀) + J_eff floor.
        # f_J = max(δ,δ₀)/(max(δ,δ₀)+δ₀) ⇒ J_eff → 2·J_CT at half-filling (Mott limit).
        # Weiss uses (1−δ); BdG applies g_J = 4/(1+δ)² → 4 (finite).
        _z_ZRS = self.t_pd**2 / (_dct**2 + self.t_pd**2)
        self.doping_0: float = float(_z_ZRS / max(1.0 - _z_ZRS, 1e-9))
        self.U_mf: float = self.Z * _J_ct / 2.0
        self.J_CT: float = _J_ct
        self._t0_ref: float = self.t0  # store reference hopping for Q-scaling
        if self.nk % 2 != 0:
            self.nk = self.nk + 1

    def summary(self, delta: float = 0.15) -> None:
        g_t   = 2.0 * delta / (1.0 + delta)
        g_J   = 4.0 / (1.0 + delta) ** 2

        # Representative anisotropic point (Q = 5% of lambda_hop)
        _Q_rep     = 0.05 * self.lambda_hop
        _tx_r = self.t0 * np.exp(+_Q_rep / max(self.lambda_hop, 1e-9))
        _ty_r = self.t0 * np.exp(-_Q_rep / max(self.lambda_hop, 1e-9))
        t_sq_aniso = 0.5 * (_tx_r**2 + _ty_r**2)

        f_d   = 1.0 - delta   # RMFT spin-site fraction
        # h_afm prefactor — isotropic (Q=0) and anisotropic (Q=0.05·λ) versions
        _hp_aniso = g_J * f_d * (self.U_mf / 2.0 + self.Z * 2.0 * g_t**2 * t_sq_aniso / self.U)

        M_phys       = 0.15
        h_afm_Mphys  = _hp_aniso * M_phys / 2.0
        # Metallic AFM criterion: Weiss field < half-bandwidth
        # Use t_eff = g_t * sqrt(t_sq_aniso) as effective hopping
        t_eff_aniso = g_t * np.sqrt(t_sq_aniso)
        ok_metal     = h_afm_Mphys < 2.0 * t_eff_aniso

        K_spont = self.g_JT**2 / max(self.Delta_CF, 1e-9)
        k_scf          = np.linspace(-np.pi, np.pi, self.nk, endpoint=False)
        KX_scf, KY_scf = np.meshgrid(k_scf, k_scf)
        self.k_points  = np.column_stack((KX_scf.flatten(), KY_scf.flatten()))
        self.N_k       = len(self.k_points)
        self.k_weights = _simpson_weights_2d(self.nk, self.nk)   # uniform 1/N weights (periodic BZ)

        # χ₀ even grid — kept separate for the commensurate q_AFM=(π,π) index trick.
        k_even = np.linspace(-np.pi, np.pi, self.nk, endpoint=False)
        KX_ev, KY_ev = np.meshgrid(k_even, k_even)
        self.k_points_even   = np.column_stack((KX_ev.flatten(), KY_ev.flatten()))
        self.N_k_even        = len(self.k_points_even)
        # Consistent with k_weights: both grids use uniform 1/N weights on the periodic BZ.
        self.k_weights_even  = _simpson_weights_2d(self.nk, self.nk)
        self.chi0_Q_idx = ((np.arange(self.N_k_even) // self.nk + self.nk // 2) % self.nk) * self.nk + (np.arange(self.N_k_even) %  self.nk + self.nk // 2) % self.nk  # Precompute AFM shift index: chi0_Q_idx[i] = index of k_i + Q_AFM in k_points_even

        # General shift-index table for ALL q-vectors on the 2π/nk grid.
        # For q = (nx, ny) * 2π/nk the k+q grid is a cyclic PERMUTATION of k_even:
        #   E(k+q)[i] = E(k)[shift_table[nx, ny, i]]
        # This makes k+q resolution zero-cost (integer index reorder, no eigh) for all
        # on-grid q in solve_linearized_gap_equation and compute_gap_eq_vectorized.
        _flat = np.arange(self.N_k_even)
        _kx_idx = _flat % self.nk          # column index 0..nk-1
        _ky_idx = _flat // self.nk         # row index    0..nk-1
        # shift_table[nx, ny, k_flat] = index of k + (nx,ny)*dk in k_even
        _nx = np.arange(self.nk)[:, None, None]   # (nk, 1, 1)
        _ny = np.arange(self.nk)[None, :, None]   # (1, nk, 1)
        self.shift_table = (
            ((_ky_idx[None, None, :] + _ny) % self.nk) * self.nk
          + ((_kx_idx[None, None, :] + _nx) % self.nk)
        ).astype(np.int32)   # (nk, nk, N_k_even)

        print("\n================ MODEL PARAMS SUMMARY ================\n")
        print("Primary inputs:")
        print(f"  t_pd={self.t_pd:.4f} eV   Δ_CT={self.Delta_CT:.4f} eV   → t0={self.t0:.4f} eV (derived)")
        print(f"  u={self.u:.3f}   U={self.U:.4f} eV")
        print(f"  λ_SOC={self.lambda_soc:.4f} eV   Δ_tet={self.Delta_tetra:.4f} eV"
              f"   Δ_ip={self.Delta_inplane:.4f} eV")
        print(f"  ω_JT={self.omega_JT:.4f} eV")
        print(f"  g_JT={self.g_JT:.4f} eV/Å")
        print(f"  Z={self.Z}   η={self.eta:.4f} δ₀={self.doping_0:.4f}")
        print(f"  {self.N_k} k-pts (SCF/gap nk={self.nk}), {self.N_k_even} k-pts (χ₀ nk={self.nk})  [uniform 1/N weights, Σw=1, periodic BZ]")

        print("\nDerived quantities (from __post_init__):")
        print(f"  Δ_CF   = {self.Delta_CF:.5f} eV   (Γ₆–Γ₇a SOC+CF gap)")
        print(f"  t0     = {self.t0:.5f} eV   (= t_pd²/Δ_CT, ZSA dd hopping)")
        print(f"  J_CT   = {self.J_CT:.5f} eV   (ZSA CT superexchange: 2t_pd⁴/Δ_CT²·(1/U + 1/(Δ_CT+U/2)))")
        print(f"  U_mf   = {self.U_mf:.5f} eV   (= Z·J_CT/2, bare MF Weiss amplitude)")
        print(f"  V_eff_bare = {self.g_JT**2 / max(self.K_lattice, 1e-9):.5f} eV  (= g²/K_lattice, bare adiabatic JT pairing scale)")
        print(f"  Γ₇ split = {self.g7split:.5f} eV"
              f"  [{'⚠ < 2kT — residual Γ₇ degeneracy' if self.g7split < 2.0 * self.kT else '✓ > 2kT'}]")
        print(f"\nMagnetic regime check (δ={delta:.3f}, anisotropic Q={_Q_rep:.3f} Å):")
        print(f"  h_afm prefactor (aniso) = {_hp_aniso:.5f} eV")
        print(f"  M={M_phys:.2f} (typical SC+AFM): h_afm = {h_afm_Mphys:.5f} eV  vs  2t_eff = {2*t_eff_aniso:.5f} eV"
              f"  {'✓ metallic AFM' if ok_metal else '⚠ marginal/insulating'}")
        print("\nJT mechanism (static / pre-SCF estimate):")
        print(f"  K_spont = g²/Δ_CF = {K_spont:.5f} eV/Å²  (local atomic JT threshold, NO exchange)")
        print(f"  K_lattice = {self.K_lattice:.5f} eV/Å²")
        if self.K_lattice > K_spont:
            print(f"  ✓ K_lattice > K_spont — spontaneous (atomic) JT blocked by bare lattice stiffness.")
            print(f"    SC-triggered JT requires K_eff = K_lattice + ∂²F_ex/∂Q² > 0 AND G3[2,2] > 0.")
        else:
            print(f"  ✗ K_lattice < K_spont ⚠ bare lattice already soft — spontaneous JT risk even without SC!")

        print("\nNumerics:")
        print(f" kT={self.kT*1000:.2f} meV  mixing={self.mixing:.4f}")
        print("=======================================================\n")

def _build_soc_cf_hamiltonian(lambda_soc: float, Delta_tetra: float, Delta_inplane: float = 0.0) -> np.ndarray:
    """
    H = λ·L·S + Δ_axial·Lz² + Δ_inplane·(Lx²−Ly²)  in the t2g manifold.

    Basis: {|mL=+1,↑⟩, |0,↑⟩, |−1,↑⟩, |+1,↓⟩, |0,↓⟩, |−1,↓⟩}  (6×6 complex).

    Parameters
    ----------
    lambda_soc   : SOC strength λ > 0 (t2g effective, same sign convention as before)
    Delta_tetra  : axial (tetragonal) crystal-field term Δ_axial·Lz²
                   Negative = compressive along z → partially cancels SOC, reduces Δ_CF.
    Delta_inplane: in-plane B2g anisotropy Δ_ip·(Lx²−Ly²).
                   Splits the Γ₇ quartet into two Kramers doublets (Γ₇a, Γ₇b)
                   Prevents spontaneous JT from the 4-fold degenerate Γ₇ level

    SOC spectrum (t2g, L_eff = 1):
      Γ₆  (j_eff=1/2-like, 2-fold Kramers) — GROUND STATE for λ>0 and this sign
      Γ₇a (j_eff=3/2 m=±3/2 component, 2-fold Kramers) — first excited
      Γ₇b (j_eff=3/2 m=±1/2 component, 2-fold Kramers) — second excited
    """
    # L = 1 operators
    Lz = np.diag([1.0, 0.0, -1.0])
    Lp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    Lm = Lp.T.conj()
    Lx = (Lp + Lm) / 2.0
    Ly = (Lp - Lm) / 2.0j
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    H_SOC = lambda_soc * (
        np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)
    )
    Lx_f = np.kron(Lx, I2)
    Ly_f = np.kron(Ly, I2)
    Lz_f = np.kron(Lz, I2)
    H_CF = (Delta_tetra * (Lz_f @ Lz_f)
            + Delta_inplane * (Lx_f @ Lx_f - Ly_f @ Ly_f))
    return H_SOC + H_CF

def _simpson_weights_2d(nx: int, ny: int) -> np.ndarray:
    """
    BZ integration weights for a 2D periodic endpoint=False k-grid.

    Convention: Σ_k w_k = 1  (normalised BZ average, not physical (2π)² measure).
    All callers use  Σ_k w_k f(k)  to compute a BZ average <f>_BZ directly.

    Grid: k = linspace(-π, π, n, endpoint=False)  →  dk = 2π/n.

    The correct rule for a PERIODIC function on an endpoint=False uniform grid is
    the UNIFORM (rectangular) rule, which converges exponentially for smooth
    periodic functions (Poisson summation / aliasing argument).

    Why NOT composite Simpson?
        Composite Simpson is designed for CLOSED intervals [a, b].
        On a periodic BZ the pattern [1, 4, 2, …, 4, 1] assigns different weights
        to the boundary points, breaking translational invariance and introducing
        a boundary-bias O((dk)²) per cell edge.  For smooth periodic integrands
        the uniform rule is strictly superior.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Grid sizes must be positive: got {nx}, {ny}")
    n_total = nx * ny
    return np.full(n_total, 1.0 / n_total)

class ClusterMF:
    """
    Sites A and B (AFM sublattices); exact diagonalization of O⊗O multipolar exchange with mean-field boundary coupling to external magnetization.
    Captures: quantum multipolar correlations, orbital mixing and spin-orbit coupling, thermal fluctuations.
    Does NOT capture: fermionic antisymmetrization, charge-transfer fluctuations.
    Valid when multipolar degrees of freedom dominate over charge fluctuations and system is not deep in the Mott insulator.
    """
    
    def __init__(self, params: ModelParams):
        self.p = params
        self.CLUSTER_SIZE = 2
        self.Z_BOUNDARY = params.Z - 1  # One link is within cluster, Z-1 are boundary
    
    def build_multipolar_operator(self, eta: float) -> np.ndarray:
        P6_diag = np.array([1.0, 1.0, 0.0, 0.0])    # Projects to 6↑, 6↓
        P7_diag = np.array([0.0, 0.0, 1.0, 1.0])    # Projects to 7↑, 7↓
        sz_diag = np.array([1.0, -1.0, 1.0, -1.0])  # Spin polarization σz: ↑=+1, ↓=-1
        O_diag = (P6_diag + eta * P7_diag) * sz_diag
        return np.diag(O_diag)
    
    def build_cluster_hamiltonian(self, H_sp_A: np.ndarray, H_sp_B: np.ndarray, J_eff: float, M_ext: float, eta: float, U_mf_stoner: float = 0.0) -> np.ndarray:
        """
        H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)               [single-particle terms]
                  + J·O_A⊗O_B                                [intra-cluster multipolar exchange]
                  + H_boundary(A) ⊗ I + I ⊗ H_boundary(B)   [inter-cluster MF]

          - J_eff:          Gutzwiller-renormalised Heisenberg superexchange: ZSA charge-transfer superexchange + kinematic dd-exchange
          - U_mf_stoner/2:  Stoner (on-site Hubbard) contribution to the Weiss field;
                            consistent with BdG h_afm = g_J·f_d·(U_mf/2 + Z·2t²/U)·M/2.
                            If U_mf_stoner=0 (default), reverts to Heisenberg-only boundary.
        """
        I4 = np.eye(4, dtype=complex)
        H_cluster = np.kron(H_sp_A, I4) + np.kron(I4, H_sp_B)
        O = self.build_multipolar_operator(eta)
        H_cluster += J_eff * np.kron(O, O)

        # Boundary coupling: Heisenberg + Stoner, matching BdG Weiss-field definition
        H_bound = self.Z_BOUNDARY * (J_eff + U_mf_stoner / 2.0) * M_ext * O
        H_cluster += np.kron(H_bound, I4)  # site A
        H_cluster += np.kron(I4, H_bound)  # site B
        return H_cluster
    
    def cluster_expectation(self, evals: np.ndarray, evecs: np.ndarray, Operator: np.ndarray, temperature: float, site_index: int = -1) -> float:
        if Operator.shape[0] == 4:
            I4 = np.eye(4, dtype=complex)
            if site_index == 0:
                Operator = np.kron(Operator, I4)
            elif site_index == 1:
                Operator = np.kron(I4, Operator)
            else:
                raise ValueError("site_index must be 0 or 1 when Operator is 4×4")

        if temperature < 1e-6:
            psi = evecs[:, 0]
            return float(np.real(np.vdot(psi, Operator @ psi))) # ⟨ψ₀|O|ψ₀⟩ (ground state)

        E = evals - evals[0]
        weights = np.exp(-E / temperature)
        Z = weights.sum()
        Oevecs = Operator @ evecs
        diag   = np.einsum('ij,ij->j', evecs.conj(), Oevecs)
        return float(np.real((weights * diag).sum() / Z))

class SusceptibilityMixin:
    """
    Three-level susceptibility interface for RMFT_Solver.

    Levels
    ------
    'fast'   — analytical 2-band AFM model; Δ=0, isotropic tx=ty.
               Used by: BO scout, FeasibilityScanner, G3-matrix pre-SCF estimate.
    'normal' — exact Lindhard χ₀(q) tensor; Δ=0, arbitrary tx,ty,q.
               Used by: solve_linearized_gap_equation, compute_gap_eq_vectorized.
    'sc'     — exact χ_QQ at Δ≠0 (numerical ∂²Ω/∂Q²), q=0;
               χ_DD/N_eff from 'fast' (see _sus_sc docstring for justification).
               Used by: SCF rigidity update, post-SCF SC-JT validation.
    """

    def get_susceptibilities_fast(self, target_doping: float, M: float, Q: float, Delta_s: complex = 0.0, Delta_d: complex = 0.0) -> dict:
        """
        Analytical susceptibilities from the 2-band AFM model.

        Return dict keys
        chi_DD_s, chi_DD_d, chi_DD_sd  : pairing susceptibilities [eV⁻¹]
        chi_DQ_s, chi_DQ_d             : SC–JT cross terms [eV⁻¹]  (0 if Δ=0)
        chi_QQ                         : orbital JT susceptibility [eV/Å²]
        """
        result = self._compute_afm2band_susceptibilities(target_doping, M, Q, Delta_s, Delta_d)
        result['chi_QQ'] = self._chi_QQ_matrix_elements(M, Q, target_doping, Delta_s, Delta_d, result['mu_n'])
        return result

    def get_susceptibilities_normal(self, q: np.ndarray, M: float, Q: float,
                    target_doping: float, mu: float,
                    tx: float, ty: float, g_J: float,
                    Delta_s: complex = 0.0+0j, Delta_d: complex = 0.0+0j,
                    _E_k_cache: tuple = None,
                    vertex_params: dict = None,
                    _chi_QQ_cache: float = None,
                    actual_doping: float = None) -> dict:
        """
        Exact q-dependent χ₀(q) tensor in the normal state (Δ=0), with optional in-place RPA vertex calculation.

        Thin wrapper around compute_chi0_tensor.  Any non-zero Δ passed
        via Delta_s / Delta_d is silently forced to zero (Lindhard is not
        Bogoliubov); delta_warning=True is set in that case.

        Susceptibility projections (basis [Γ₆↑,Γ₆↓,Γ₇↑,Γ₇↓]):
            χ_SS = Tr[S_z · χ₀[Γ₆,Γ₆] · S_z]  spin–spin, Γ₆ sector
            χ_SQ = Tr[S_z · χ₀[Γ₆,Γ₇]]        spin–quadrupole cross (SC-opened)
            χ_QS = Tr[χ₀[Γ₇,Γ₆] · S_z]        quadrupole–spin cross

        Parameters
        ----------
        vertex_params : dict or None
            If provided, the full RPA pairing vertex is computed from the χ₀
            Required keys:
                'J_eff_x'       : float  superexchange along x  J_x = g_J·f_J·2t_x²·(1/U+…) [eV]
                'J_eff_y'       : float  superexchange along y  J_y = g_J·f_J·2t_y²·(1/U+…) [eV]
                'V_JT'          : float  g_JT²/K_bare [eV]
                'chi_QQ_normal' : float  −∂²Ω/∂Q²|_{Δ=0} [eV/Å²]
            Optional keys:
                'return_det'    : bool   include 'rpa_det' in result

        _chi_QQ_cache : float or None
            Pre-computed χ_QQ(Δ=0) value.  χ_QQ depends only on (M, Q, Δ, μ),
            NOT on the transfer momentum q, so when called in a q-loop it is
            wasteful to recompute it each iteration.

        Return dict keys (always present)
        ----------------------------------
        chi0_tensor, chi_SS, chi_SQ, chi_QS, chi_QQ, mode, delta_warning

        Additional keys (only when vertex_params is provided)
        ------------------------------------------------------
        V_full, V_spin, V_jt  : float  RPA pairing vertices [eV]
        chi_SS_moriya          : float  Moriya-damped spin susceptibility
        rpa_near_qcp           : bool   det < _RPA_DET_WARN
        rpa_det                : float  (only if return_det=True)
        """
        delta_warning = False
        _Ds = Delta_s; _Dd = Delta_d
        if abs(Delta_s) + abs(Delta_d) > _QQ_DELTA_THRESH:
            delta_warning = True
            _Ds = 0.0+0j; _Dd = 0.0+0j   # Lindhard requires Δ=0

        chi0_tensor = self.compute_chi0_tensor(
            q, M, Q, _Ds, _Dd, target_doping, mu, tx, ty, g_J,
            _E_k_cache=_E_k_cache)

        # Projections
        S_z    = np.array([[1.0, 0.0], [0.0, -1.0]])
        chi_66 = chi0_tensor[0:2, 0:2]
        chi_67 = chi0_tensor[0:2, 2:4]
        chi_76 = chi0_tensor[2:4, 0:2]

        chi_SS_bare = float(np.real(np.trace(S_z @ chi_66 @ S_z)))
        chi_SQ      = float(np.real(np.trace(S_z @ chi_67)))
        chi_QS      = float(np.real(np.trace(chi_76 @ S_z)))

        # χ_QQ is q-independent: reuse cache if caller provides it.
        # Computing it fresh requires 3× eigvalsh (±dQ finite-difference).
        chi_QQ_n = (_chi_QQ_cache if _chi_QQ_cache is not None
                    else self._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu))

        result = {
            'chi0_tensor':   chi0_tensor,
            'chi_SS':        chi_SS_bare,
            'chi_SQ':        chi_SQ,
            'chi_QS':        chi_QS,
            'chi_QQ':        chi_QQ_n,
            'delta_warning': delta_warning,
        }

        if vertex_params is None:
            return result

        # Correct Stoner criterion at q=Q_AFM=(π,π):
        #   J(Q_AFM) = −(J_x+J_y)  →  |J(Q_AFM)| = J_x+J_y
        # J_eff = ½(J_x+J_y): positive scalar Stoner parameter.
        # The J_x≠J_y anisotropy enters correctly through χ_SS(q) via the BdG dispersion.
        _Jx   = float(vertex_params['J_eff_x'])
        _Jy   = float(vertex_params['J_eff_y'])
        J_eff = 0.5 * (_Jx + _Jy)   # positive scalar; anisotropy enters via χ_SS(q), not J
        V_JT          = float(vertex_params['V_JT'])
        chi_QQ_normal = float(vertex_params['chi_QQ_normal'])
        return_det    = bool(vertex_params.get('return_det', False))

        K_bare = max(self._K_bare, 1e-9)

        _moriya_doping = actual_doping if actual_doping is not None else target_doping
        _abs_d_m       = max(abs(_moriya_doping), 1e-6)
        _t_eff_proxy   = float(self.p.t0 * (2.0 * _abs_d_m) / (1.0 + _abs_d_m))   # g_t · t0
        _alpha_M       = _moriya_alpha(_moriya_doping, _t_eff_proxy, J_eff)
        _Gamma_M       = _alpha_M * max(J_eff, 1e-9) * _t_eff_proxy
        chi_SS_moriya  = chi_SS_bare / (1.0 + _Gamma_M * max(chi_SS_bare, 0.0))

        # χ_SQ clamp: dynamic floor avoids spurious zeroing at Δ≠0 (active B₁g channel).
        # Floor = max(1e−8, 1e−4·√(χ_SS·χ_QQ)) ∝ geometric mean ⇒ large if active, small if suppressed.
        # _CHI_SQ_EPS = 1e−12: fallback when χ_QQ → 0 (insulating limit).
        _gap_amp = self._gap_amplitude
        _chi_sq_dyn  = max(_CHI_SQ_EPS,
                           1e-4 * float(np.sqrt(max(chi_SS_bare, 0.0) * max(chi_QQ_n, 0.0))))
        _thr     = max(_chi_sq_dyn, 1e-6 * _gap_amp) if _gap_amp > _QQ_DELTA_THRESH else _chi_sq_dyn
        chi_SQ_v = 0.0 if abs(chi_SQ) < _thr else chi_SQ
        chi_QS_v = 0.0 if abs(chi_QS) < _thr else chi_QS

        # χ_QQ clamp: prevent JT self-crossing before SC acts
        chi_QQ_clamped = min(max(chi_QQ_normal, 0.0), _CHI_QQ_CLAMP / max(V_JT, 1e-9) * K_bare)
        chi_QQ_tilde   = chi_QQ_clamped / K_bare

        def _rpa_vertex(J: float, V: float) -> tuple:
            a = 1.0 - J * chi_SS_moriya
            b =     - J * chi_SQ_v
            c =     - V * chi_QS_v
            d = 1.0 - V * chi_QQ_tilde
            det = a * d - b * c
            if det < _RPA_DET_FLOOR:
                _safe  = max(abs(det), _RPA_DET_FLOOR)
                _taper = float(np.clip(det / _safe, -1.0, 1.0)) if det < 0 else 1.0
                i00 =  d/_safe; i01 = -b/_safe; i10 = -c/_safe; i11 = a/_safe
            else:
                _taper = 1.0
                i00 =  d/det;  i01 = -b/det;  i10 = -c/det;  i11 = a/det
            rss = i00*chi_SS_moriya + i01*chi_QS_v
            rqq = i10*chi_SQ_v     + i11*chi_QQ_tilde
            rsq = i00*chi_SQ_v     + i01*chi_QQ_tilde
            rqs = i10*chi_SS_moriya + i11*chi_QS_v
            Vp  = float((J**2*rss + V**2*rqq + J*V*(rsq+rqs)) * max(_taper, 0.0))
            return Vp, det

        V_full, det_full = _rpa_vertex(J_eff, V_JT)
        V_spin, _        = _rpa_vertex(J_eff, 0.0)
        V_jt,   _        = _rpa_vertex(0.0,   V_JT)

        result['V_full']        = V_full
        result['V_spin']        = V_spin
        result['V_jt']          = V_jt
        result['chi_SS_moriya'] = chi_SS_moriya
        result['rpa_near_qcp']  = det_full < _RPA_DET_WARN
        if return_det:
            result['rpa_det'] = det_full
        return result

    def get_susceptibilities_sc(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> dict:
        """
        Exact susceptibilities in the SC state (Δ≠0), evaluated at q=0.

        χ_QQ is computed exactly via numerical ∂²Ω/∂Q² at the given Δ
        (this is the SC-state JT rigidity used by the Hessian and SCF
        rigidity update).

        χ_DD_s/d, χ_DQ, N_eff are taken from the 'fast' 2-band model
        evaluated at the same (M, Q, Δ_s, Δ_d).  Justification: these
        quantities enter the G3-matrix and the SC-JT diagnostic only as
        pre-factors; their Δ-dependence is weak (the gap modifies the
        quasiparticle weights but not the AFM band structure topology
        that dominates χ_DD).  Computing them exactly would require a
        full BdG Lindhard sum at Δ≠0, which is forbidden in the pairing
        vertex context to avoid double-counting the condensate feedback.
        The fast proxy is therefore the correct approximation here.

        sc_jt_signal = True when χ_QQ(SC) > χ_QQ(normal), the key qualitative indicator of SC-triggered JT.
        """
        chi_QQ_sc = self._chi_QQ_matrix_elements(M, Q, target_doping, Delta_s, Delta_d, mu)
        chi_QQ_n = self._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)

        result = self._compute_afm2band_susceptibilities(target_doping, M, Q, Delta_s, Delta_d)

        result['chi_QQ'] = chi_QQ_sc
        result['chi_QQ_normal'] = chi_QQ_n
        result['chi_QQ_ratio'] = float(chi_QQ_sc / max(abs(chi_QQ_n), 1e-12))
        result['sc_jt_signal'] = bool(chi_QQ_sc > chi_QQ_n)
        return result

class RMFT_Solver(SusceptibilityMixin):

    def __init__(self, params: ModelParams):
        self.p = params
        self.cluster_mf = ClusterMF(params)

        self.k_points       = params.k_points
        self.k_points_even  = params.k_points_even
        self.N_k            = params.N_k
        self.N_k_even       = params.N_k_even
        self.k_weights      = params.k_weights
        self.k_weights_even = params.k_weights_even
        self.chi0_Q_idx     = params.chi0_Q_idx
        self.shift_table    = params.shift_table   # (nk, nk, N_k_even) int32 — cyclic shift index

        # Orbital operators derived from the SOC+CF eigenbasis.
        # sz_op and sz_bdg16 are also set inside _rebuild_orbital_operators.
        self._rebuild_orbital_operators(params)

        self.phi_k = (np.cos(self.k_points[:, 0])
                      - np.cos(self.k_points[:, 1]))
        self.phi_k_even = (np.cos(self.k_points_even[:, 0])
                           - np.cos(self.k_points_even[:, 1]))
        self._D_phonon: float = 2.0 / max(params.omega_JT, 1e-6)
        _scf_log("RMFT-INIT",
                 f"t_pd={params.t_pd:.4f} eV  Δ_CT={params.Delta_CT:.4f} eV"
                 f"  t0={params.t0:.4f} eV  U={params.U:.4f} eV"
                 f"  Δ_CF={params.Delta_CF:.4f} eV"
                 f"  g_JT={params.g_JT:.4f} eV/Å"
                 f"  V_eff_bare={params.g_JT**2 / params.K_lattice:.4f} eV"
                 f"  Δ_tetra={params.Delta_tetra:.4f} eV"
                 f"  lambda_soc={params.lambda_soc:.4f} eV"
                 f"  u = {params.u:.4f}")
        self._vbdg: Optional['VectorizedBdG'] = None
        self._scf_bdg_cache: Optional[tuple] = None
        self._cluster_j_renorm: float = 1.0   # cluster ED vertex correction; 1.0 = bare Gutzwiller
        self._gap_amplitude:    float = 0.0   # current |Δ_s|+|Δ_d|; updated each SCF iteration
        self._K_bare: float = params.K_lattice # immutable bare lattice spring constant (eV/Å²)
        self._chi0_norm_cache: Optional[tuple] = None   # (E, V, M, Q, mu, tx, ty, g_J)

    def _rebuild_orbital_operators(self, params: 'ModelParams') -> None:
        """
        Rebuild all SOC+CF-basis-dependent operators from params._U4.

        Must be called whenever params.lambda_soc, params.Delta_tetra, or params.Delta_inplane
        changes and params.__post_init__() has been called (which regenerates _U4, η, η_J).
        """
        U4 = params._U4  # columns = {Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓}
        tau_x_t2g = (U4[:, 0:2] @ U4[:, 2:4].conj().T
                   + U4[:, 2:4] @ U4[:, 0:2].conj().T)           # (6,6)
        tau_x_op  = (U4.conj().T @ tau_x_t2g @ U4)               # (4,4), complex
        z4 = np.zeros((4, 4), dtype=complex)
        # BdG particle–hole symmetry requires the hole block to carry −τ_x^T, nambu structure: O_Nambu = block_diag(O_AA, -O_AA^T)
        tau_x_op_T = tau_x_op.T
        self.tau16 = np.block([
            [tau_x_op,    z4,              z4,                z4              ],
            [z4,          tau_x_op,        z4,                z4              ],
            [z4,          z4,             -tau_x_op_T,        z4              ],
            [z4,          z4,              z4,               -tau_x_op_T      ],
        ])
        self.tau_x_mat = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ], dtype=float)

        # sz_op  (4,)  : orbital Sz weights in the [Γ₆↑,Γ₆↓,Γ₇↑,Γ₇↓] basis.
        # sz_bdg16 (16,) : the same weights extended to the full 16-component Nambu basis [pA, pB, hA, hB] × sz_op
        self.sz_op = np.array([1.0, -1.0, params.eta, -params.eta], dtype=float)
        self.sz_bdg16 = np.concatenate([
             self.sz_op,   # particle A  (+)
            -self.sz_op,   # particle B  (stagger)
            -self.sz_op,   # hole A      (p-h conjugate of A)
             self.sz_op,   # hole B      (p-h conjugate of B = double flip)
        ])

    def _get_vbdg(self) -> 'VectorizedBdG':
        if self._vbdg is None:
            self._vbdg = VectorizedBdG(self)
        return self._vbdg

    _CHI0_CACHE_TOL: float = 1e-5   # parameter-change threshold for cache invalidation

    def _get_chi0_norm_cache(self, M: float, Q: float, mu: float,
                             tx: float, ty: float, g_J: float,
                             vbdg: 'VectorizedBdG',
                             target_doping: float = 0.0) -> tuple:
        """
        Return (E_k_all, V_k_all) for the Δ=0 BdG on k_points_even.

        Cached on (M, Q, mu, tx, ty, g_J, target_doping); tolerance _CHI0_CACHE_TOL.
        Within a single SCF iteration these quantities are constant across the
        entire q-loop in solve_linearized_gap_equation and compute_gap_eq_vectorized,
        avoiding O(N_q) redundant eigh calls on the N_k_even × 16 matrix.
        """
        key = (M, Q, mu, tx, ty, g_J, target_doping)
        if self._chi0_norm_cache is not None:
            _E, _V, *_key = self._chi0_norm_cache
            if all(abs(a - b) < self._CHI0_CACHE_TOL
                   for a, b in zip(key, _key)):
                return _E, _V
        E_k, V_k = np.linalg.eigh(
            vbdg._build_H_stack(
                vbdg._kpts_ev, M, Q, 0.0 + 0j, 0.0 + 0j,
                target_doping, mu, tx, ty, g_J, out=vbdg._H_stack_ev))
        self._chi0_norm_cache = (E_k, V_k, M, Q, mu, tx, ty, g_J, target_doping)
        return E_k, V_k

    def _reset_transient_state(self) -> None:
        """
        Reset all mutable per-solve caches on a solver clone.

        Must be called after copy.copy(solver) to guarantee that:
          - _vbdg          : gets a fresh VectorizedBdG with this clone's k-grids
          - _scf_bdg_cache : previous BdG (ev, ec) from a different solve is not reused
          - _cluster_j_renorm : exchange vertex correction starts at bare value
          - _K_bare        : NOT cleared (immutable per __init__ contract)
        """
        self._vbdg             = None   # re-created on first _get_vbdg()
        self._scf_bdg_cache    = None   # no stale (ev, ec) from parent solve
        self._cluster_j_renorm = 1.0    # bare Gutzwiller vertex
        self._chi0_norm_cache  = None   # normal-state χ₀ eigenvector cache
        self._gap_amplitude    = 0.0    # reset so cloned solvers start from normal state

    def get_gutzwiller_factors(self, delta: float) -> Tuple[float, float, float, float]:
        """
        g_t       = 2δ/(1+δ)         kinetic energy; → 0 at half-filling (Mott insulator)
        g_J       = 4/(1+δ)²         exchange enhancement; → 4 at half-filling (J = 4t²/U)
        g_Delta_s = g_t              on-site inter-orbital Γ₆⊗Γ₇ singlet (kinetic origin)
        g_Delta_d                    inter-site d-wave B₁g renormalisation — see below.

        B₁g (d-wave) RENORMALISATION

        The B₁g pairing channel arises from virtual Γ₆↔Γ₇ transitions via the
        superexchange tensor J_B1g = (J_CT/2)·sinh(2Q/λ)·η·τ_x. Its many-body
        renormalisation differs from pure kinetic (g_t) or exchange (g_J)
        because the Γ₇ sector has reduced spectral weight due to CF splitting
        (Δ_CF) and SOC.

        DOUBLE-COUNTING AVOIDANCE
        -------------------------
        • η-weight (AFM selection rule) is already included in J_alpha_beta_Q → V_d_scalar.
        • j_renorm (cluster vertex correction) is also applied there.

        GUTZWILLER FACTOR
        -----------------
        The B₁g vertex mixes Γ₆ and Γ₇ sectors, so the pairing renormalisation
        interpolates between:

            g_t  : kinetic limit (Γ₇ decoupled)
            g_J  : full exchange limit (Γ₆–Γ₇ degenerate)

        LIMITS
        ------
        Δ_CF → ∞, λ_SOC → 0   → p_7 → 0   → g_Delta_d → g_t
        Δ_CF → 0,  λ_SOC → ∞  → p_7 → 0.5 → g_Delta_d → g_J
        """
        abs_delta = max(abs(delta), 1e-6)
        g_t       = (2.0 * abs_delta) / (1.0 + abs_delta)
        g_J       = 4.0 / ((1.0 + abs_delta) ** 2)
        g_Delta_s = g_t   # s-channel is kinetic in origin → follows g_t

        # Γ₇ spectral weight from SOC+CF eigenvectors: U_gamma column order: [Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓, Γ₇b↑, Γ₇b↓] (6D t2g basis).
        # p_7 = (Σ |U_gamma[i,2:4]|² over Γ₆ indices) / 2  measures Γ₇ admixture in the Γ₆ doublet due to SOC.
        # p_7 is a single-ion SOC+CF property that determines how strongly the exchange J_B1g renormalises the B₁g pairing vertex.
        # Computed using the truncated SOC matrix _U4 (Γ₆/Γ₇ subspace): _U4[:,0], _U4[:,1] → Γ₆↑, Γ₆↓ eigenvectors, rows 2–3  → Γ₇ components
        U4 = self.p._U4
        # Weight of Γ₇ character (rows 2:4) in the Γ₆ doublet (columns 0:2)
        p_7_up = float(np.sum(np.abs(U4[2:4, 0])**2))   # Γ₇ weight in Γ₆↑ eigenvec
        p_7_dn = float(np.sum(np.abs(U4[2:4, 1])**2))   # Γ₇ weight in Γ₆↓ eigenvec
        p_7    = 0.5 * (p_7_up + p_7_dn)                # average ∈ [0, 0.5]

        # Interpolation: w_norm = p_7 / 0.5 ∈ [0, 1]
        w_norm    = float(np.clip(p_7 / 0.5, 0.0, 1.0))
        g_Delta_d = float(np.clip(g_t + (g_J - g_t) * w_norm, g_t, g_J))
        return g_t, g_J, g_Delta_s, g_Delta_d

    def effective_hopping_anisotropic(self, Q: float) -> Tuple[float, float]:
        """
        B₁g JT distortion breaks x-y symmetry: tx ≠ ty
        
        Exponential hopping law (Harrison + bond-length argument):
        tx(Q) = t₀ * exp(+Q / lambda_hop)   [elongation along x → shorter bond → larger t]
        ty(Q) = t₀ * exp(-Q / lambda_hop)   [compression along y → longer bond → smaller t]
        """
        tx = self.p.t0 * np.exp(+Q / self.p.lambda_hop)
        ty = self.p.t0 * np.exp(-Q / self.p.lambda_hop)
        return tx, ty
    
    def effective_superexchange(self, g_J: float, tx_bare: float, ty_bare: float, doping: float, direction: str = 'z') -> float:
        """
        Gutzwiller-renormalised superexchange J_eff for the cluster Hamiltonian.

        J_eff = g_J · f_J(δ) · J_CT

        g_J = 4/(1+δ)²  — Gutzwiller exchange factor (kinematic blocking of
                           virtual hops by mobile holes).

        f_J(δ) = max(δ,δ₀) / (max(δ,δ₀) + δ₀)  — ZRS coherence floor.
            Saturates at f_J=0.5 as δ→0 (ZRS band incoherent but local J survives),
            Note: g_J·f_J together still give J_eff → g_J(0)·0.5·J_CT = 2·J_CT
            at half-filling, consistent with the Mott limit.
        """
        abs_doping = max(abs(doping), 1e-6)
        d_fl  = max(abs_doping, self.p.doping_0)          # floor at doping_0
        f_J   = d_fl / (d_fl + self.p.doping_0)           # ∈ [0.5, 1)
        _dct = max(self.p.Delta_CT, 1e-9)
        _U = max(self.p.U, 1e-9)

        if direction == 'x':
            t_sq = tx_bare**2
        elif direction == 'y':
            t_sq = ty_bare**2
        else:
            t_sq = 0.5 * (tx_bare**2 + ty_bare**2)
        return g_J * f_J * 2.0 * t_sq * (1.0/_U + 1.0/(_dct + _U/2.0))

    def J_alpha_beta_Q(self, Q: float, lambda_hop: float) -> np.ndarray:
        """
        Q-dependent multipolar exchange matrix in the [Γ₆↑, Γ₆↓, Γ₇↑, Γ₇↓] basis.

        Irrep decomposition (D₄h):
            J(Q) = J_A1g(Q) · P_A1g  +  J_B1g(Q) · P_B1g
            P_A1g = diag(1,1,η_J²,η_J²),   P_B1g = η_J·τ_x   (Γ₆–Γ₇ mixing).

        Microscopic origin (hopping anisotropy):
            t_x = t₀ e^{+Q/λ},  t_y = t₀ e^{-Q/λ},  with J ∝ t²  ⇒
            J_A1g ∝ (J_CT/2) cosh(2Q/λ)   (even in Q),
            J_B1g ∝ (J_CT/2) sinh(2Q/λ)   (odd, vanishes at Q=0).

        MF normalization:
            the 1/2 factor comes from counting each bond once in the two-sublattice Heisenberg mean-field equation.

        Physical consequence:
            Q = 0  →  J_B1g = 0, Γ₆–Γ₇ mixing forbidden (AFM selection rule).
            Q ≠ 0  →  B1g channel opens, enabling multipolar/JT response.
        """
        lam = max(lambda_hop, 1e-9)

        scale_A1g = float(np.cosh(2.0 * Q / lam))
        scale_B1g = float(np.sinh(2.0 * Q / lam))

        # η_J(Q): orbital-character-weighted exchange ratio J_Γ₇/J_Γ₆.
        # Superexchange J ∝ t² is orbital-selective: d_xz hops only along x,
        # d_yz only along y, d_xy along both.  When tx≠ty the Γ₆ (xz-dominant)
        # and Γ₇ (yz-dominant) sectors feel different effective exchanges.
        tx_q = self.p.t0 * np.exp(+Q / lam)
        ty_q = self.p.t0 * np.exp(-Q / lam)
        t0   = max(self.p.t0, 1e-9)
        # Normalised hopping weights (relative to isotropic t0²)
        _jxz = (tx_q / t0) ** 2
        _jyz = (ty_q / t0) ** 2
        _jxy = 0.5 * (_jxz + _jyz)
        _J6  = (self.p._w6_xz * _jxz + self.p._w6_yz * _jyz + self.p._w6_xy * _jxy)
        _J7  = (self.p._w7_xz * _jxz + self.p._w7_yz * _jyz + self.p._w7_xy * _jxy)
        eta_J = float(np.sqrt(max(_J7 / max(_J6, 1e-9), 0.0)))  # sqrt because J_A1g uses eta_J², at Q=0 (tx=ty=t0): eta_J = 1.0 exactly.

        J_A1g = self._cluster_j_renorm * (self.p.J_CT / 2.0) * scale_A1g * np.diag([1.0, 1.0, eta_J**2, eta_J**2])
        J_B1g = self._cluster_j_renorm * (self.p.J_CT / 2.0) * scale_B1g * eta_J * self.tau_x_mat
        return J_A1g + J_B1g
    
    def compute_JT_rigidity_from_exchange(self, M: float, Q: float, mu: float, g_J: float, target_doping: float, g_t_loc: float) -> Dict:
        """
        Exchange contribution to the JT stiffness: ∂²F_ex/∂Q².

            F_ex = Σ_{αβ} J_{αβ}(Q) ⟨O_α(Q)⟩⟨O_β(Q)⟩

        Full second derivative via product rule:
            ∂²F_ex/∂Q² = O·(∂²J/∂Q²)·O + 4·(∂O/∂Q)·J·(∂O/∂Q)
                        + 2·O·J·(∂²O/∂Q²) + 4·O·(∂J/∂Q)·(∂O/∂Q)

        All four terms are included. At Q=0 the B1g selection rule forces ∂O/∂Q = 0 and ∂²J/∂Q² = 0 (sinh→cosh, leading term ∝ Q²),
        so only the term 2·O·J·(∂²O/∂Q²) survives — but at Q≠0 all terms contribute and omitting any would bias the SCF Q-update.

        Effective stiffness:
            K_eff = K_lattice + ∂²F_ex/∂Q²
            ∂²F_ex/∂Q² < 0  →  exchange softens the mode
            ∂²F_ex/∂Q² > 0  →  exchange stiffens the mode

        Stability criterion:
            G3[2,2] = 1 − χ_QQ / K_eff < 0  →  SC-triggered JT

        SC limit: M → 0 ⇒ ⟨O_α⟩ → 0 ⇒ ∂²F_ex/∂Q² → 0.
        """
        eps    = max(1e-4, 0.01 * abs(Q) + 1e-4)
        abs_d  = max(abs(target_doping), 1e-6)
        f_d    = 1.0 - abs_d   # spin-site fraction: RMFT h_afm ∝ g_J*(1−δ)*J*M
        sz_op  = self.sz_op

        vbdg = self._get_vbdg()

        tx_0, ty_0 = self.effective_hopping_anisotropic(Q)
        tx_p, ty_p = self.effective_hopping_anisotropic(Q + eps)
        tx_m, ty_m = self.effective_hopping_anisotropic(Q - eps)

        W = np.zeros((4, 16), dtype=float)
        for a in range(4):
            W[a, a]    =  sz_op[a]   # particle A
            W[a, a+4]  =  sz_op[a]   # particle B
            W[a, a+8]  = -sz_op[a]   # hole A   (p-h conjugate sign)
            W[a, a+12] = -sz_op[a]   # hole B

        H_0 = vbdg._build_H_stack(vbdg._kpts, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, g_t_loc * tx_0, g_t_loc * ty_0, g_J)
        H_p = vbdg._build_H_stack(vbdg._kpts, M, Q + eps, 0.0+0j, 0.0+0j, target_doping, mu, g_t_loc * tx_p, g_t_loc * ty_p, g_J)
        H_m = vbdg._build_H_stack(vbdg._kpts, M, Q - eps, 0.0+0j, 0.0+0j, target_doping, mu, g_t_loc * tx_m, g_t_loc * ty_m, g_J)

        H_batch = np.concatenate([H_0, H_p, H_m], axis=0) 
        ev_batch, ec_batch = np.linalg.eigh(H_batch)

        Nk = vbdg._N_k
        ev_0, ev_p, ev_m = ev_batch[:Nk], ev_batch[Nk:2*Nk], ev_batch[2*Nk:]
        ec_0, ec_p, ec_m = ec_batch[:Nk], ec_batch[Nk:2*Nk], ec_batch[2*Nk:]

        def _calc_O_exp(ev, ec):
            f_kn = self.fermi_function(ev)
            ec2  = np.abs(ec)**2
            return np.einsum('k,kn,ai,kin->a', self.k_weights, f_kn, W, ec2, optimize=True)

        O_exp_Q = _calc_O_exp(ev_0, ec_0)
        O_p     = _calc_O_exp(ev_p, ec_p)
        O_m     = _calc_O_exp(ev_m, ec_m)

        dO_dQ   = (O_p - O_m) / (2.0 * eps)
        d2O_dQ2 = (O_p - 2.0 * O_exp_Q + O_m) / (eps ** 2)

        J_Q   = self.J_alpha_beta_Q(Q, self.p.lambda_hop)
        J_p   = self.J_alpha_beta_Q(Q + eps, self.p.lambda_hop)
        J_m   = self.J_alpha_beta_Q(Q - eps, self.p.lambda_hop)

        dJ_dQ   = (J_p - J_m) / (2.0 * eps)
        d2J_dQ2 = (J_p - 2.0 * J_Q + J_m) / (eps ** 2)

        # Full ∂²F_ex/∂Q²
        term_J2    = float(O_exp_Q @ d2J_dQ2 @ O_exp_Q)           # O·(∂²J/∂Q²)·O
        term_dO2   = 4.0 * float(dO_dQ @ J_Q @ dO_dQ)             # 4·(∂O/∂Q)·J·(∂O/∂Q)
        term_O2    = 2.0 * float(O_exp_Q @ J_Q @ d2O_dQ2)         # 2·O·J·(∂²O/∂Q²)
        term_mix   = 4.0 * float(O_exp_Q @ dJ_dQ @ dO_dQ)         # 4·O·(∂J/∂Q)·(∂O/∂Q)
        d2F_ex_dQ2 = term_J2 + term_dO2 + term_O2 + term_mix

        K_eff = self.p.K_lattice + d2F_ex_dQ2

        # Commutator diagnostic: [τ_x, H_AFM] at Q=0 — B1g blocking strength
        h_afm_diag = self.J_alpha_beta_Q(0.0, self.p.lambda_hop) @ O_exp_Q
        H_afm_mat  = np.diag(h_afm_diag)
        tau_x_4    = self.tau_x_mat
        comm       = tau_x_4 @ H_afm_mat - H_afm_mat @ tau_x_4
        comm_norm  = float(np.linalg.norm(comm, 'fro'))

        return {
            'K_eff':         K_eff,
            'O_exp':         O_exp_Q,
            'dO_dQ':         dO_dQ,
            'comm_tau_H':    comm,
            'comm_norm':     comm_norm,
            'blocking_ratio': comm_norm / max(abs(self.p.Delta_CF), 1e-9),
        }
    
    def fermi_function(self, E: np.ndarray) -> np.ndarray:
        arg = E / self.p.kT
        arg = np.clip(arg, -_FERMI_ARG_CLIP, _FERMI_ARG_CLIP)
        return 1.0 / (1.0 + np.exp(arg))

    def compute_rank2_multipole_expectation(self, Delta: complex, tau_x_bdg: float) -> Dict:
        """
        In the AFM-only state: P_eff = P6, ⟨τ_x⟩_P6 = 0 (τ_x off-diagonal).
        In the SC state: w = |Δ|/Δ_CF mixes in Γ₇, ⟨τ_x⟩_eff = w·|τ_x_bdg|.
        Selection ratio R = ⟨τ_x⟩_eff / 1: R≈0 → barrier intact; R→1 → JT allowed.
        """
        Delta_CF = max(self.p.Delta_CF, 1e-9)
        w = float(np.clip(abs(Delta) / Delta_CF, 0.0, 1.0))
        tau_x_free_max = 1.0
        tau_x_projected = w * abs(tau_x_bdg)
        selection_ratio = tau_x_projected / max(tau_x_free_max, 1e-9)
        return {
            'w':                        w,
            'selection_ratio':          selection_ratio,
            'jt_algebraically_allowed': selection_ratio > 0.05,
            'tau_x_projected':          tau_x_projected,
            'tau_x_free_max':           tau_x_free_max,
        }

    def compute_static_chi0_afm(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> Dict:
        """
        Static transverse spin susceptibility χ₀(q_AFM) at q = (π,π).
        Uses the even k-grid so commensurate q_AFM=(π,π) maps k→k+Q exactly onto another grid point: k_i + π ≡ k_{i + nk/2} (mod 2π, endpoint=False)

        Formula: χ₀ = Σ_{k,n,m} |⟨ψ_n(k)|Ŝ_z|ψ_m(k+Q)⟩|² · (f_n − f_m) / (E_m − E_n)
        Ŝ_z in [6↑,6↓,7↑,7↓] = diag(+1,−1,+η,−η) on sublattice A (staggered in BdG).

        Return dict keys
            'chi0'        : float, static susceptibility (eV⁻¹)
            'chi0_moriya' : float, Moriya-damped susceptibility used in Stoner denominator
            'U_eff_chi'   : float, renormalised magnetic coupling used in Stoner denominator (eV),  NOT the bare Hubbard U. This keeps U_eff_chi · χ₀ ~ O(1) within the ordered AFM phase
            'rpa_factor'  : float, AFM QCP crossed (magnetic instability); returns 1.0 (no enhancement) — the ordered state has broken down and the linear RPA is invalid.
            'afm_unstable': bool, True if stoner_denom ≤ 0 (AFM QCP crossed, magnetically unstable)
        """
        # Spin operator in the 16-dim Nambu basis.
        #
        # BdG Nambu layout: [Part_A(0:4), Part_B(4:8), Hole_A(8:12), Hole_B(12:16)]
        # Sublattice stagger: B carries −S_z relative to A (collinear AFM).
        # Particle-hole conjugation: holes carry −S_z relative to particles.
        sz_diag  = self.sz_bdg16   # (16,) Nambu Sz, consistent with _rebuild_orbital_operators

        # chi0_Q_idx is a permutation of [0..N_even) so E(k+Q) and V(k+Q) are simply row-permutations of E(k) and V(k) 
        vbdg = self._get_vbdg()
        E_k_all, V_k_all = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts_ev, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack_ev))  # (N,16), (N,16,16)

        E_kQ_all = E_k_all[self.chi0_Q_idx]   # (N,16)  — free permutation, no LAPACK
        V_kQ_all = V_k_all[self.chi0_Q_idx]   # (N,16,16)

        f_k_all  = self.fermi_function(E_k_all)    # (N, 16)
        f_kQ_all = self.fermi_function(E_kQ_all)   # (N, 16)

        SzV_kQ  = sz_diag[None, :, None] * V_kQ_all                  # (N,16,16): [k,i,m]
        M_mat   = np.einsum('kin,kim->knm', V_k_all.conj(), SzV_kQ)  # (N,16,16)
        M2      = np.abs(M_mat)**2  # (N,16,16)

        df = f_k_all[:, :, None] - f_kQ_all[:, None, :]   # (N,16,16)
        dE = E_kQ_all[:, None, :] - E_k_all[:, :, None]   # (N,16,16)

        mask    = (np.abs(df) > _FD_MASK_DF) & (np.abs(dE) > _FD_MASK_DE)
        safe_dE = np.where(mask, dE, 1.0)
        ratio   = np.where(mask, self.k_weights_even[:, None, None] * M2 * df / safe_dE, 0.0)
        chi0 = float(ratio.sum())
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        J_eff_now = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping)
        # Apply Moriya damping to chi0 before computing the Stoner denominator. Uses g_t·t0 (Gutzwiller-renormalised hopping) as the spin-fermion energy scale.
        _t_eff_proxy_s  = float(self.p.t0 * (2.0 * abs(target_doping)) / (1.0 + abs(target_doping) + 1e-9))
        _alpha_M_s      = _moriya_alpha(target_doping, _t_eff_proxy_s, J_eff_now)
        _Gamma_M_s      = _alpha_M_s * max(J_eff_now, 1e-9) * _t_eff_proxy_s
        chi0_moriya     = chi0 / (1.0 + _Gamma_M_s * max(chi0, 0.0))
        stoner_denom = 1.0 - J_eff_now * chi0_moriya
        return {
            'chi0':         chi0,
            'chi0_moriya':  chi0_moriya,
            'U_eff_chi':    J_eff_now,
            'rpa_factor':   1.0 / max(stoner_denom, _RPA_DET_FLOOR) if stoner_denom > 0.0 else 1.0,
            'afm_unstable': stoner_denom <= 0.0,
        }

    def _compute_chi_tau(self, M: float, Q: float, target_doping: float, Delta_s: complex = 0.0, Delta_d: complex = 0.0,  mu: float = 0.0) -> Dict:
        """
        Multipolar susceptibility χ_τx = |∂⟨τ_x⟩/∂(g_JT·δQ)| via finite-difference BdG.

        At each perturbed Q value the full BdG is rediagonalised with the corresponding
        self-consistent parameters: hopping t(Q±δQ) AND the AFM Weiss field h_afm(Q±δQ)
        recomputed from the updated t_eff.
        """
        g_t, g_J, _, _ = self.get_gutzwiller_factors(target_doping)
        tx_bare_0, ty_bare_0 = self.effective_hopping_anisotropic(Q)
        tx0 = g_t * tx_bare_0
        ty0 = g_t * ty_bare_0

        t_eff_avg = np.sqrt(0.5 * (tx0**2 + ty0**2))
        N0        = 1.0 / (np.pi * max(t_eff_avg, 1e-6))
        Ut_ratio  = self.p.U / max(t_eff_avg, 1e-6)
        
        dQ    = max(1e-4, 1e-3 * abs(Q) + 1e-5)   # Adaptive step: avoids catastrophic cancellation when Q ~ 0 or Q ~ dQ.
        vbdg  = self._get_vbdg()

        def _tau_x_expectation(Q_val: float) -> float:
            tx_b, ty_b = self.effective_hopping_anisotropic(Q_val)
            tx_v = g_t * tx_b
            ty_v = g_t * ty_b
            
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q_val, Delta_s, Delta_d, target_doping, mu, tx_v, ty_v, g_J, out=vbdg._H_stack))
            f_n  = self.fermi_function(ev)
            fbar = 1.0 - f_n
            uA, uB, vA, vB = VectorizedBdG._get_nambu_spinors(ec)
            tau_A = (2.0 * np.real(uA[:, 0, :]*np.conj(uA[:, 2, :]) + uA[:, 1, :]*np.conj(uA[:, 3, :])) * f_n
                   + 2.0 * np.real(vA[:, 0, :]*np.conj(vA[:, 2, :]) + vA[:, 1, :]*np.conj(vA[:, 3, :])) * fbar)
            tau_B = (2.0 * np.real(uB[:, 0, :]*np.conj(uB[:, 2, :]) + uB[:, 1, :]*np.conj(uB[:, 3, :])) * f_n
                   + 2.0 * np.real(vB[:, 0, :]*np.conj(vB[:, 2, :]) + vB[:, 1, :]*np.conj(vB[:, 3, :])) * fbar)
            return float(np.einsum('k,kn->', self.k_weights, (tau_A + tau_B))) / 4.0

        tau_p = _tau_x_expectation(Q + dQ)
        tau_m = _tau_x_expectation(Q - dQ)
        tau_diff = tau_p - tau_m
        denom_fd = max(self.p.g_JT * 2.0 * dQ, 1e-12)
        # In the pure AFM normal state <tau_x>=0 (symmetry), but the anomalous pair amplitude creates a non-zero d<tau_x>/dQ|_{Q=0} channel — this is chi_tau: linear orbital susceptibility to the JT distortion.
        # Physical role: lambda_JT = (g_JT^2 / K) * chi_tau measures the dimensionless SC-triggered JT coupling strength. Viable SC+JT coexistence requires 0.05 < lambda_JT < 1.
        chi_tau  = abs(tau_diff / denom_fd) if abs(tau_diff) > 1e-10 else 0.0
        return {'chi_tau': chi_tau, 'N0': N0, 'Ut_ratio': Ut_ratio}

    def _chi_QQ_matrix_elements(self, M: float, Q: float, target_doping: float, Delta_s: complex, Delta_d: complex, mu: float) -> float:
        """
        g²-weighted JT orbital susceptibility: χ_QQ = g_JT² · χ_orbital = −g_JT² · ∂²Ω/∂(g_JT·Q)².
        """
        dQ = max(1e-4, 1e-3 * abs(Q) + 1e-5)   # adaptive step; consistent with _compute_chi_tau
        g_t, g_J, _, _ = self.get_gutzwiller_factors(target_doping)
        vbdg = self._get_vbdg()

        def omega(Qval):
            tx_b, ty_b = self.effective_hopping_anisotropic(Qval)
            tx, ty = g_t * tx_b, g_t * ty_b
            ev = np.linalg.eigvalsh(
                vbdg._build_H_stack(
                    vbdg._kpts, M, Qval, Delta_s, Delta_d,
                    target_doping, mu, tx, ty, g_J,
                    out=vbdg._H_stack
                )
            )
            arg = np.clip(np.abs(ev) / self.p.kT, 0.0, 100.0)
            Omega_kn = np.minimum(0.0, ev) - self.p.kT * np.log1p(np.exp(-arg))
            return np.sum(self.k_weights[:, None] * Omega_kn)
        
        Ωp = omega(Q + dQ)
        Ω0 = omega(Q)
        Ωm = omega(Q - dQ)

        # −∂²Ω/∂Q²: positive for a stable metal (χ_QQ > 0 convention used in G3[2,2]).
        chi_QQ = -(Ωp - 2.0 * Ω0 + Ωm) / (dQ ** 2)
        return chi_QQ

    def compute_chi0_tensor(self, q: np.ndarray, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, _E_k_cache: tuple = None) -> np.ndarray:
        """
        Orbital bare susceptibility tensor chi0^{ab}(q) in [6↑,6↓,7↑,7↓] basis.

        chi0^{ab}(q) = -Σ_{k,n,m} <n,k|P_a|m,k+q><m,k+q|P_b|n,k>
                        * (f_n(k) - f_m(k+q)) / (E_m(k+q) - E_n(k))

        Sign convention: χ₀ > 0 for stable metal → RPA: χ_RPA = (I − U·χ₀)⁻¹·χ₀.

        The 16x16 BdG Nambu basis is [Part_A(0:4), Part_B(4:8), Hole_A(8:12), Hole_B(12:16)].
        eigh diagonalises the full Nambu matrix; the resulting eigenvectors already
        encode the Bogoliubov rotation including particle-hole signs.
        
        Anomalous sector pairs (Part<->Hole) are EXCLUDED:
            - At Delta=0 they vanish exactly.
            - At Delta!=0 they would double-count F_AA/F_AB already in the gap equation.
        Only normal (same-type) intra- and inter-sublattice pairs are included.

        _E_k_cache : (E_k_all, V_k_all) pre-computed at Δ=0 on k_points_even.
            Callers that loop over many q-vectors (solve_linearized_gap_equation,
            compute_gap_eq_vectorized) build this once and reuse it.  When None,
            the cache is built internally (single-q callers).

        k+q resolution uses shift_table (a cyclic integer-permutation index built
        in ModelParams.__post_init__) for all q on the 2π/nk grid — zero eigh cost.
        """
        vbdg = self._get_vbdg()

        # ── k-cache (Δ=0) — build once per q-loop, reuse across q ───────────────
        if _E_k_cache is not None:
            E_k_all, V_k_all = _E_k_cache
        else:  # Fallback: build at Delta=0 as safety net.
            E_k_all, V_k_all = np.linalg.eigh(
                vbdg._build_H_stack(vbdg._kpts_ev, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack_ev))

        # ── k+q via shift_table (free permutation) or off-grid fallback ──────────
        nk = self.p.nk
        dk = 2.0 * np.pi / nk
        nx_f = q[0] / dk
        ny_f = q[1] / dk
        nx = int(round(nx_f)) % nk
        ny = int(round(ny_f)) % nk
        on_grid = (abs(nx_f - round(nx_f)) < 1e-6) and (abs(ny_f - round(ny_f)) < 1e-6)

        # Free permutation
        idx = self.shift_table[nx, ny]   # (N_k_even,) int32
        E_kQ_all = E_k_all[idx]
        V_kQ_all = V_k_all[idx]

        f_k_all  = self.fermi_function(E_k_all)
        f_kQ_all = self.fermi_function(E_kQ_all)

        # Lindhard weights (N_k, 16_k, 16_kq)
        df      = f_k_all[:, :, None] - f_kQ_all[:, None, :]
        dE      = E_kQ_all[:, None, :] - E_k_all[:, :, None]
        mask    = (np.abs(df) > _FD_MASK_DF) & (np.abs(dE) > _FD_MASK_DE)
        safe_dE = np.where(mask, dE, 1.0)
        factor  = -np.where(mask, self.k_weights_even[:, None, None] * df / safe_dE, 0.0)

        # 8-sector Lindhard sum — only normal (same-type) Nambu pairs are included.
        SECTOR_PAIRS = [
            (slice(0,  4), slice(0,  4)),
            (slice(4,  8), slice(4,  8)),
            (slice(8,  12), slice(8,  12)),
            (slice(12, 16), slice(12, 16)),
            (slice(0,  4), slice(4,  8)),
            (slice(4,  8), slice(0,  4)),
            (slice(8,  12), slice(12, 16)),
            (slice(12, 16), slice(8,  12)),
        ]
        chi0 = np.zeros((4, 4), dtype=complex)
        N = len(self.k_points_even)
        CHUNK = 128
        for sl_k, sl_kQ in SECTOR_PAIRS:
            Vk_s  = V_k_all[:,  sl_k,  :]   # (N, 4, 16) — rows = orbital index a, cols = band n
            VkQ_s = V_kQ_all[:, sl_kQ, :]   # (N, 4, 16) — rows = orbital index b, cols = band m
            for k0 in range(0, N, CHUNK):
                k1    = min(k0 + CHUNK, N)
                fac_c = factor[k0:k1]       # (C, 16_k, 16_kq)
                Vk_c  = Vk_s[k0:k1]         # (C,  4,   16_k)
                VkQ_c = VkQ_s[k0:k1]        # (C,  4,   16_kq)
                chi0 += oe.contract('cnm,can,cbn,cam,cbm->ab',
                                    fac_c, Vk_c.conj(), Vk_c, VkQ_c, VkQ_c.conj(),
                                    optimize='optimal')
        return chi0

    def _get_fermi_surface_sample(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, n_fs: int = _FS_N_SAMPLE) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample k-points near the Fermi surface and estimate Fermi velocities.

        A k-point is 'near the FS' if at least one quasiparticle band satisfies
        |E_n(k)| < 3kT.  The Fermi velocity proxy is the minimum positive
        quasiparticle energy (a monotone proxy that is zero at a node).

        Return
            fs_pts : (N, 2)  k-points on or near the Fermi surface
            vF     : (N,)    |v_F| proxy (eV); proportional to DOS weight
        """
        # Vectorised: diagonalise all k at once, then filter near-FS points
        vbdg   = self._get_vbdg()
        ev_all, _ = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))

        near_fs = np.any(np.abs(ev_all) < _FS_SAMPLING * self.p.kT, axis=1)

        ev_pos = np.where(ev_all > 0, ev_all, np.inf)
        vF_all = ev_pos.min(axis=1)
        vF_all = np.where(np.isinf(vF_all), self.p.kT, vF_all)
        vF_all = np.maximum(vF_all, _VF_FLOOR)
        vF_all = np.where(near_fs, vF_all, np.nan)

        fs_idx = np.where(near_fs)[0]
        if len(fs_idx) == 0:
            fs_idx = np.arange(min(n_fs, self.N_k))
            vF_arr = np.ones(len(fs_idx))
        else:
            fs_idx = fs_idx[:n_fs]
            vF_arr = vF_all[fs_idx].astype(float)

        fs_pts = self.k_points[fs_idx]
        vF     = vF_arr
        return fs_pts, vF

    def solve_linearized_gap_equation(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, actual_doping: float = None) -> Dict:
        """
        Linearised gap equation solved as an eigenvalue problem on the Fermi surface.

        λ Δ(k_i) = Σ_j Γ_ij Δ(k_j)

        Γ_ij = g_Δ · V(k_i−k_j) / √(|v_F(i) v_F(j)|)

        with V(q) the full RPA vertex
        """
        fermi_pts, vF = self._get_fermi_surface_sample(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        N = fermi_pts.shape[0]

        # k_i − k_j vectors
        i_idx, j_idx = np.triu_indices(N)

        q_raw = fermi_pts[i_idx] - fermi_pts[j_idx]
        q_arr = (q_raw + np.pi) % (2*np.pi) - np.pi

        scale = 1e5
        q_int = np.rint(q_arr * scale).astype(np.int64)
        unique_int, inv_idx = np.unique(q_int, axis=0, return_inverse=True)
        unique_q = unique_int.astype(np.float64) / scale

        # interaction scales — anisotropic J(q) = J_x·cos qx + J_y·cos qy
        # This correctly captures the d-wave channel asymmetry from B₁g distortion:
        # J_x ∝ tx², J_y ∝ ty² → J_x ≠ J_y when Q≠0, which directly renormalises
        # the d-wave pairing vertex V_d ∝ ⟨J(q)·(cos qx − cos qy)⟩_FS.
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        J_eff_x = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping, direction='x')
        J_eff_y = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping, direction='y')
        # Scalar fallback for diagnostics / G-matrix (uses isotropic average)
        J_eff   = 0.5 * (J_eff_x + J_eff_y)

        # Build (or reuse) the Δ=0 BdG eigenvector cache for the q-loop.
        # The cache key is (M, Q, mu, tx, ty, g_J, target_doping); tolerance _CHI0_CACHE_TOL.
        vbdg = self._get_vbdg()
        E_k_cache, U_k_cache = self._get_chi0_norm_cache(
            M, Q, mu, tx, ty, g_J, vbdg, target_doping=target_doping)

        # χ_QQ is q-independent: cache for q-loop.
        V_JT = self.p.g_JT**2 / max(self._K_bare, 1e-9)
        chi_QQ_normal = self._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)

        # k+q is resolved by shift_table inside compute_chi0_tensor (zero eigh cost).
        n_q = len(unique_q)
        V_unique = np.empty(n_q)
        V_spin_u = np.empty(n_q)
        V_JT_u   = np.empty(n_q)

        for u_idx, q_u in enumerate(unique_q):
            _sus_qu = self.get_susceptibilities_normal(
                q=q_u, M=M, Q=Q,
                target_doping=target_doping, mu=mu, tx=tx, ty=ty, g_J=g_J,
                _E_k_cache=(E_k_cache, U_k_cache),
                vertex_params={'J_eff_x': J_eff_x, 'J_eff_y': J_eff_y, 'V_JT': V_JT, 'chi_QQ_normal': chi_QQ_normal},
                _chi_QQ_cache=chi_QQ_normal,
                actual_doping=actual_doping)

            V_unique[u_idx] = _sus_qu['V_full']
            V_spin_u[u_idx] = _sus_qu['V_spin']
            V_JT_u[u_idx]   = _sus_qu['V_jt']

        # DOS weights
        vF_safe = np.maximum(np.abs(vF), _VF_FLOOR_TIGHT)
        inv_svF = 1.0 / np.sqrt(vF_safe)

        weights = inv_svF[i_idx] * inv_svF[j_idx]
        vals = weights * V_unique[inv_idx]

        # symmetric kernel
        Gamma = np.zeros((N, N), dtype=float)
        Gamma[i_idx, j_idx] = vals
        Gamma += Gamma.T

        # symmetry detection
        phi_s = np.ones(N)
        phi_d = np.cos(fermi_pts[:,0]) - np.cos(fermi_pts[:,1])

        phi_s /= np.linalg.norm(phi_s)
        phi_d /= np.linalg.norm(phi_d)

        eigvals_tmp, eigvecs_tmp = np.linalg.eigh(Gamma)
        idx_tmp = np.argmax(eigvals_tmp)
        gap_tmp = eigvecs_tmp[:, idx_tmp]

        w_s = abs(gap_tmp @ phi_s)
        w_d = abs(gap_tmp @ phi_d)

        gap_symmetry = 'B1g (d-wave)' if w_d > w_s else 'A1g (s-wave)'
        lambda_raw = float(eigvals_tmp[idx_tmp])
        gap_vector = gap_tmp

        # Gutzwiller factors
        _, _, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)
        g_Delta_dom = g_Delta_d if w_d > w_s else g_Delta_s
        lambda_max = lambda_raw * g_Delta_dom

        # Rayleigh projection on JT-only pairing kernel
        vals_JT = weights * V_JT_u[inv_idx]
        Gamma_JT = np.zeros((N, N), dtype=float)
        Gamma_JT[i_idx, j_idx] = vals_JT
        Gamma_JT += Gamma_JT.T
        gv_norm = gap_vector / max(float(np.linalg.norm(gap_vector)), 1e-12)
        lambda_JT_kernel = float(gv_norm @ (Gamma_JT @ gv_norm))

        # vertex diagnostics
        V_spin_mean = float(np.mean(V_spin_u))
        V_JT_mean   = float(np.mean(V_JT_u))
        V_rpa_mean  = float(np.mean(V_unique))
        V_cross_mean = V_rpa_mean - V_spin_mean - V_JT_mean

        return {
            'lambda_max': lambda_max,
            'lambda_max_raw': lambda_raw,
            'lambda_JT_kernel': lambda_JT_kernel,
            'g_delta_dom': g_Delta_dom,
            'gap_vector': gap_vector,
            'fs_pts': fermi_pts,
            'gap_symmetry': gap_symmetry,
            'V_spin_mean': V_spin_mean,
            'V_JT_mean': V_JT_mean,
            'V_cross_mean': V_cross_mean,
            'V_rpa_mean': V_rpa_mean
        }
    
    def _compute_orbital_coherence_from_pairs(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> float:
        """
        Anomalous orbital coherence ⟨τ_x⟩_anom from off-diagonal BdG amplitudes (u·v).

        Definition:
            ⟨τ_x⟩_anom = Σ_k (1−2f_n) Re[u*_6 v_7 + h.c.]

        Properties:
            Δ = 0  →  ⟨τ_x⟩_anom = 0  (Γ₆/Γ₇ do not mix; selection rule intact)
            Δ ≠ 0  →  ⟨τ_x⟩_anom ≠ 0  (SC condensate unlocks B1g JT channel)

        The lattice couples to the total ⟨τ_x⟩; this quantity isolates the
        pure SC-induced contribution and verifies that Q is condensate-driven.
        """
        vbdg = self._get_vbdg()
        ev_all, ec_all = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))
        f_n_all  = self.fermi_function(ev_all)
        omf_all  = 1.0 - 2.0 * f_n_all
        
        # Sublattice amplitudes: (N,4,16)
        uA = ec_all[:, 0:4,   :]
        vA = ec_all[:, 8:12,  :]
        uB = ec_all[:, 4:8,   :]
        vB = ec_all[:, 12:16, :]

        # Anomalous orbital coherence per sublattice and state:
        anom_A = (np.real(uA[:, 0, :] * np.conj(vA[:, 2, :]))
                + np.real(uA[:, 2, :] * np.conj(vA[:, 0, :]))
                + np.real(uA[:, 1, :] * np.conj(vA[:, 3, :]))
                + np.real(uA[:, 3, :] * np.conj(vA[:, 1, :])))
        anom_B = (np.real(uB[:, 0, :] * np.conj(vB[:, 2, :]))
                + np.real(uB[:, 2, :] * np.conj(vB[:, 0, :]))
                + np.real(uB[:, 1, :] * np.conj(vB[:, 3, :]))
                + np.real(uB[:, 3, :] * np.conj(vB[:, 1, :])))
        tau_x_anom = float(
            np.einsum('k,kn,kn->', self.k_weights, omf_all, (anom_A + anom_B)) / 4.0
        )
        return float(tau_x_anom)

    def build_local_hamiltonian_for_bdg(self, sign_M: float, M: float, Q: float, mu: float, g_J: float, target_doping: float, tx: float, ty: float, O_expectation: np.ndarray = None) -> np.ndarray:
        """
        Local 4×4 BdG Hamiltonian for one sublattice, basis [6↑, 6↓, 7↑, 7↓].
        sign_M = ±1 for sublattices A/B (staggered AFM).

        Terms:
          1. Chemical potential −μ
          2. Crystal field splitting Δ_CF on Γ₇
          3. AFM Weiss field:
               h_α = g_J·(1−δ)·M · [h_kin_scalar · sz_α  +  Σ_β J_{αβ} · sz_β]
        where:
          (1−δ)        : RMFT spin-site fraction — fraction of sites with an S=1/2 spin.
                         Maximal at half-filling (Mott: all sites magnetic), → 0 at δ→1.
          h_kin_scalar = Z · 2·t_eff²/U    (kinematic dd-exchange, Q-dependent)
          J_{αβ}(Q)    = J_A1g + J_B1g     (superexchange tensor from J_alpha_beta_Q;
                                             J_eff uses f_J floor, not (1-δ))
          4. JT distortion:  H_JT = g_JT · Q · τ_x

        tx, ty : Gutzwiller-renormalised hoppings g_t·t(Q) (eV).

        O_expectation: optional 4-element array ⟨O_β⟩ for each orbital.
            If None: uses the MF approximation ⟨O_β⟩ = g_J·(1−δ)·M · sz_β.

        Weiss vs J_eff scaling:
            h_afm ∝ g_J·(1−δ)·J·M  — RMFT: (1-δ) spin-site fraction; maximal at half-filling.
            J_eff uses f_J = max(δ,δ₀)/(max(δ,δ₀)+δ₀) ≥ 0.5 (finite floor at δ→0).

        SC–JT chain:
            Δ≠0 → F(k)≠0 → ⟨τ_x⟩≠0 → Q≠0 → H_JT≠0

        No explicit anomalous Σ ∝ Δ term (would double-count BdG feedback).
        """
        H = np.zeros((4, 4), dtype=complex)

        # 1. Chemical potential
        np.fill_diagonal(H, -mu)

        # 2. Crystal field splitting Δ_CF on Γ₇
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓

        # 3. Mean-field AFM Weiss field
        # f_d = (1-δ): fraction of sites with an S=1/2 spin (standard RMFT site-dilution).
        # g_J*(1-δ) is maximal at half-filling (Mott limit) and decreases with doping.
        abs_delta = max(abs(target_doping), 1e-6)
        f_delta   = 1.0 - abs_delta

        # 4. Kinematic exchange scalar
        t_sq_avg = 0.5 * (tx**2 + ty**2)
        h_kin_scalar = self.p.Z * 2.0 * t_sq_avg / max(self.p.U, 1e-9) / 2.0

        if O_expectation is None:
            O_exp = g_J * f_delta * M * self.sz_op
        else:
            O_exp = np.asarray(O_expectation, dtype=float)

        h_J_vec = self.J_alpha_beta_Q(Q, self.p.lambda_hop) @ O_exp
        h_kin_vec = h_kin_scalar * O_exp
        # 5. Total Weiss field per orbital, with staggered sign
        h_vec = sign_M * (h_J_vec + h_kin_vec)
        
        # 6. B₁g multipolar selection rule
        H[0, 0] -= h_vec[0]   # 6↑
        H[1, 1] -= h_vec[1]   # 6↓
        H[2, 2] -= h_vec[2]   # 7↑
        H[3, 3] -= h_vec[3]   # 7↓

        # 7. JT distortion: H_JT = g_JT · Q · τ_x  (orbital mixing, spin-conserving)
        #    Q=0 in normal AFM state  →  no orbital mixing  →  τ_x forbidden (correct).
        #    Q≠0 only when SC condensate has generated ⟨τ_x⟩_anom ≠ 0 (SC-triggered).
        h_jt = self.p.g_JT * Q
        H[0, 2] = h_jt   # 6↑ ↔ 7↑
        H[2, 0] = h_jt
        H[1, 3] = h_jt   # 6↓ ↔ 7↓
        H[3, 1] = h_jt
        return H

    def build_single_particle_hamiltonian(self, Q: float, mu: float) -> np.ndarray:
        H = np.zeros((4, 4), dtype=complex)
        np.fill_diagonal(H, -mu)
        H[2, 2] += self.p.Delta_CF
        H[3, 3] += self.p.Delta_CF
        h_jt = self.p.g_JT * Q
        H[0, 2] = H[2, 0] = h_jt
        H[1, 3] = H[3, 1] = h_jt
        return H
    
    def _compute_site_magnetization_and_quadrupole(self, vec: np.ndarray, u_slice: slice, v_slice: slice, f: float, f_bar: float) -> Tuple[float, float]:
        u = vec[u_slice]
        v = vec[v_slice]
        # Magnetization: both terms are positive: |u|²·f gives the particle contribution, |v|²·(1-f) gives the filled-band (hole) electron contribution.
        # sz_op carries the ±1 spin weighting; the minus for spin-down comes from sz_op itself.
        m = (np.abs(u)**2 @ self.sz_op) * f + (np.abs(v)**2 @ self.sz_op) * f_bar
        
        # Quadrupole τ_x mixing between orbitals 6 and 7
        tau_u = 2.0 * np.real(np.vdot(u[[0, 1]], u[[2, 3]]))
        tau_v = 2.0 * np.real(np.vdot(v[[0, 1]], v[[2, 3]]))
        tau = tau_u * f + tau_v * f_bar
        return m, tau
    
    def _compute_density_at_mu(self, mu: float, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, tx: float, ty: float, g_J: float) -> float:
        vbdg = self._get_vbdg()
        ev_all, ec_all = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))

        fn     = self.fermi_function(ev_all)
        fn_bar = 1.0 - fn

        uA, uB, vA, vB = VectorizedBdG._get_nambu_spinors(ec_all)
        dens_A, dens_B = VectorizedBdG._compute_densities(uA, uB, vA, vB, fn, fn_bar)

        n_avg = (dens_A + dens_B) / 4.0  # BdG doubling correction
        return float(np.dot(self.k_weights, n_avg))
    
    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, tx: float, ty: float, mu_guess: float, g_J: float) -> Tuple[float, Optional[tuple]]:
        """
        Returns (mu, last_bdg_cache) where last_bdg_cache = (ev, ec) at the converged mu, so the caller can skip the redundant eigh in _compute_density_at_mu for the same (M,Q,Δ,μ) point.
        """
        target_n = 1.0 - target_doping
        _last_cache: Optional[tuple] = None

        def density_and_deriv(mu_val: float):
            nonlocal _last_cache
            vbdg = self._get_vbdg()
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu_val, tx, ty, g_J, out=vbdg._H_stack))
            _last_cache = (ev, ec)
            f   = self.fermi_function(ev)
            fb  = 1.0 - f

            uA, uB, vA, vB = VectorizedBdG._get_nambu_spinors(ec)
            dens_A, dens_B = VectorizedBdG._compute_densities(uA, uB, vA, vB, f, fb)
            n = float(np.dot(self.k_weights, dens_A + dens_B)) / 4.0

            # ∂n/∂μ: −∂f/∂E = f(1−f)/kT ≥ 0; total weight per (k,n) is |u_A|²+|u_B|²+|v_A|²+|v_B|² = 1 (BdG normalization within each sublattice pair)
            df_dE = f * fb / max(self.p.kT, 1e-10)   # (N_k,16), ≥ 0
            w_A = np.sum(np.abs(uA)**2 + np.abs(vA)**2, axis=1)   # (N_k,16)
            w_B = np.sum(np.abs(uB)**2 + np.abs(vB)**2, axis=1)
            dn_dmu = float(np.einsum('k,kn,kn->', self.k_weights, df_dE, w_A + w_B)) / 4.0
            return n - target_n, dn_dmu

        mu = mu_guess
        for _ in range(20):
            err, deriv = density_and_deriv(mu)
            if abs(err) < 1e-6:
                return mu, _last_cache
            if abs(deriv) < _DEN_DERIV_FLOOR:
                break   # flat → fall through to brentq
            step = err / max(abs(deriv), 1e-10)
            step = float(np.clip(step, -self.p.t0, self.p.t0))
            mu -= step

        # Fallback: brentq (rare — only if Newton diverges or lands on flat region)
        def density_error(mu_val):
            nonlocal _last_cache
            vbdg = self._get_vbdg()
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu_val, tx, ty, g_J, out=vbdg._H_stack))
            _last_cache = (ev, ec)
            f  = self.fermi_function(ev)
            fb = 1.0 - f
            uA, uB, vA, vB = VectorizedBdG._get_nambu_spinors(ec)
            dA, dB = VectorizedBdG._compute_densities(uA, uB, vA, vB, f, fb)
            return float(np.dot(self.k_weights, dA + dB)) / 4.0 - target_n

        w = 6.0 * self.p.t0
        mu_min, mu_max = mu - w, mu + w
        try:
            err_min = density_error(mu_min)
            err_max = density_error(mu_max)
            for _ in range(10):
                if err_min * err_max <= 0:
                    break
                if err_min > 0:
                    mu_min -= w;  err_min = density_error(mu_min)
                else:
                    mu_max += w;  err_max = density_error(mu_max)
            if err_min * err_max <= 0:
                mu = brentq(density_error, mu_min, mu_max, xtol=_BRENTQ_TOL)
                return mu, _last_cache
        except Exception:
            pass
        return mu, _last_cache
    
    def compute_bdg_free_energy(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, K_eff_for_free_energy: float = None, _ev_cache: np.ndarray = None, V_s: float = None, V_d: float = None) -> float:
        """
        Grand potential per site computed from the k-space BdG spectrum.

        Ω = (1/2) Σ_{k,n} w_k [E_n f_n − T S(f_n)]
            + |Δ_s|² / (g_s · V_s)    ← condensation correction, s-channel
            + |Δ_d|² / (g_d · V_d)    ← condensation correction, d-channel
            + (K_eff/2) Q²             ← elastic cost

        Notes on condensation correction:
        - Restores variational stationarity: ∂Ω/∂Δ_ℓ = 0 ↔ Δ_ℓ = g_ℓ · V_ℓ · F_ℓ_BZ (gap equation).
        - V_ℓ > 0 : attractive channel → term positive, energy cost to maintain Δ_ℓ; quasiparticle gain included in Ω_BdG.
        - V_ℓ ≤ 0 : repulsive / absent → Δ_ℓ = 0; term omitted.
        - V_ℓ = None (pre-cache) : fall back to bare JT vertex to allow SCF startup.

        Elastic energy uses full effective stiffness K_eff = K_lattice + ∂²F_ex/∂Q².
        """
        if _ev_cache is not None:
            ev_all = _ev_cache
        else:
            vbdg = self._get_vbdg()
            ev_all, _ = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))
        f_n = self.fermi_function(ev_all)   # (N_k, 16)

        # quasiparticle energy term
        Ef = np.einsum('k,kn,kn->', self.k_weights, ev_all, f_n)

        # entropy contribution
        if self.p.kT > _KT_FLOOR:
            f_c = np.clip(f_n, _ENTROPY_CLIP, 1 - _ENTROPY_CLIP)
            S_kn = -(f_c*np.log(f_c) + (1-f_c)*np.log(1-f_c))
            S_term = self.p.kT * np.einsum('k,kn->', self.k_weights, S_kn)
        else:
            S_term = 0.0

        # Elastic energy uses the full effective spring constant (K_eff = K_lattice + ∂²F_ex/∂Q²).
        _K_eff = max(K_eff_for_free_energy if K_eff_for_free_energy is not None else self._K_bare, 1e-9)
        elastic_energy = 0.5 * _K_eff * Q**2

        # Condensation correction: |Δ_ℓ|² / (g_ℓ · V_ℓ)
        _V_JT = self.p.g_JT**2 / max(self._K_bare, 1e-9)
        _, _, g_s, g_d = self.get_gutzwiller_factors(target_doping)

        condensation = 0.0
        _V_s = V_s if V_s is not None else _V_JT
        if _V_s > 0.0:
            condensation += abs(Delta_s)**2 / (g_s * _V_s)
        _V_d = V_d if V_d is not None else _V_JT
        if _V_d > 0.0:
            condensation += abs(Delta_d)**2 / (g_d * _V_d)

        Omega_cell = Ef - S_term
        return 0.5 * Omega_cell + elastic_energy + condensation
    
    def compute_cluster_free_energy(self, M: float, Q: float, mu: float, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> Dict:
        H_sp_A = self.build_single_particle_hamiltonian(Q, mu)
        H_sp_B = self.build_single_particle_hamiltonian(Q, mu)

        J_eff = self.effective_superexchange(g_J, tx_bare, ty_bare, doping)
        # U_mf_stoner: Stoner boundary Weiss field. Scales as g_J*(1-δ) consistent with
        # the BdG Weiss field (RMFT site-dilution: (1-δ) spins per site feel the field).
        abs_d_cl    = max(abs(doping), 1e-6)
        f_d_cl      = 1.0 - abs_d_cl
        U_mf_stoner = g_J * f_d_cl * self.p.U_mf
        H_cluster = self.cluster_mf.build_cluster_hamiltonian(H_sp_A, H_sp_B, J_eff, M, self.p.eta, U_mf_stoner=U_mf_stoner)

        evals, evecs = eigh(H_cluster)

        if self.p.kT < _KT_FLOOR:
            F_total = evals[0]
        else:
            E_shifted = evals - evals[0]
            weights   = np.exp(-E_shifted / self.p.kT)
            Z         = weights.sum()
            F_total   = evals[0] - self.p.kT * np.log(Z)

        O_mag = self.cluster_mf.build_multipolar_operator(self.p.eta)
        M_A = self.cluster_mf.cluster_expectation(evals, evecs, O_mag, self.p.kT, site_index=0)
        M_B = self.cluster_mf.cluster_expectation(evals, evecs, O_mag, self.p.kT, site_index=1)
        M_cluster = abs(M_A - M_B) / 2.0

        tau_x = np.zeros((4, 4), dtype=complex)
        tau_x[0, 2] = tau_x[2, 0] = 1.0  # 6↑ ↔ 7↑
        tau_x[1, 3] = tau_x[3, 1] = 1.0  # 6↓ ↔ 7↓

        Q_A_exp = self.cluster_mf.cluster_expectation(evals, evecs, tau_x, self.p.kT, site_index=0)
        Q_B_exp = self.cluster_mf.cluster_expectation(evals, evecs, tau_x, self.p.kT, site_index=1)
        Q_exp   = (Q_A_exp + Q_B_exp) / 2.0   # signed: respects Z2 symmetry

        tau_x_sq = tau_x @ tau_x
        Q2_A = self.cluster_mf.cluster_expectation(evals, evecs, tau_x_sq, self.p.kT, site_index=0)
        Q2_B = self.cluster_mf.cluster_expectation(evals, evecs, tau_x_sq, self.p.kT, site_index=1)
        Q_rms       = np.sqrt(abs(Q2_A + Q2_B) / 2.0)
        fluctuation = np.sqrt((abs(Q2_A - Q_A_exp**2) + abs(Q2_B - Q_B_exp**2)) / 2.0)
        return {
            'F_per_site':    F_total / self.cluster_mf.CLUSTER_SIZE,
            'M':             M_cluster,
            'Q_exp':         Q_exp,
            'Q_rms':         Q_rms,
            'Q_fluctuation': fluctuation,
            'J_eff':         J_eff
        }

    def _scf_jacobi_kick(self, target_doping: float, initial_M: float, initial_Q: float, initial_Delta: float) -> Dict:
        """
        Estimate the dominant Jacobi eigenvalue λ₊ of the two-channel (Δ, Q) SCF map
        and generate physics-informed seed values  for (M, Q, Δ_s, Δ_d).

        Linearised Jacobian:
            J = [ ∂Δ_out/∂Δ   ∂Δ_out/∂Q ]
                [ ∂Q_out/∂Δ          0  ]

        Define:
            A = ∂Δ_out/∂Δ ≈ g_Δ · V_pair · N0
            C = ∂Q_out/∂Δ ≈ (g_JT / K) · χ_τ
            B = ∂Δ_out/∂Q · ∂Q_out/∂Δ

        ∂Δ_out/∂Q is dominated by Γ₆–Γ₇ mixing:
            ∂Δ_out/∂Q ~ g_Δ · V_pair · N0 · (g_JT / Δ_CF)

        Hence:
            B_raw ~ A · (g_JT² / (Δ_CF · K)) · (χ_τ / N0)

        A Padé saturation B = B_raw / (1 + B_raw/A) regularises the map near criticality (λ₊ → 1), mimicking quartic Landau corrections.

        λ₊ = ½ [ A + sqrt(A² + 4 B C) ]

        Regimes:
            λ₊ << 1   : subcritical  → small Δ seed, standard mixing
            λ₊ ~  1   : critical     → thermal-scale Δ, reduced mixing
            λ₊ >> 1   : supercritical→ large Δ, strong mixing reduction
        """
        p = self.p
        abs_d = max(abs(target_doping), 1e-6)
        g_t   = (2.0 * abs_d) / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d) ** 2
        t_eff = g_t * p.t0
        N0 = 1.0 / (np.pi * max(t_eff, 1e-6))

        # Mott guard (g_t<0.10): incoherent FS, seed Δ=0
        if g_t < _G_T_COHERENCE_MIN:
            return {
                'M_kick':      float(np.clip(initial_M, 0.05, 0.45)),
                'Q_kick':      0.0,
                'Delta_kick':  0.0,    # crucial: no SC seed in Mott regime
                'mixing_kick': p.mixing * 0.5,
                'lambda_plus': 0.0,
                'regime':      'mott',
                'chi_tau':     0.0,
            }

        chi0_est = N0 / (1.0 + (p.U_mf / max(np.pi * t_eff, 1e-9))**2)
        U_pair = g_J * p.U

        # V_spin_est: Moriya-damped warm-start estimate U²·χ / (1 + Γ_M·χ).
        # This is a cheap approximation sufficient for seeding (M₀, Q₀, Δ₀);
        # the full SCF vertex uses the proper RPA J²·χ / (1 − J·χ)² via get_susceptibilities_normal.
        _Gamma_M_jk   = _moriya_alpha(target_doping, t_eff, g_J * p.J_CT) * max(g_J * p.J_CT, 1e-9) * t_eff
        V_spin_est = U_pair**2 * chi0_est / (1.0 + _Gamma_M_jk * max(chi0_est, 0.0))

        V_eff_bare = p.g_JT**2 / max(self._K_bare, 1e-9)
        V_pair = max(V_eff_bare + V_spin_est, V_eff_bare)

        g_Delta = g_t
        chi_tau_val = self._compute_chi_tau(initial_M, initial_Q, target_doping)['chi_tau']

        A = g_Delta * V_pair * N0
        B_raw = A * (p.g_JT**2 / (max(p.Delta_CF, 1e-9) * max(self._K_bare, 1e-9))) \
                    * (chi_tau_val / max(N0, 1e-12))
        B = B_raw / (1.0 + B_raw / max(A, 1e-9))
        C = (p.g_JT / max(self._K_bare, 1e-9)) * chi_tau_val

        discriminant = A**2 + 4.0 * B * C

        if discriminant >= 0.0:
            # Real eigenvalues: standard overdamped regime.
            lambda_plus = 0.5 * (A + np.sqrt(discriminant))
        else:
            # Complex eigenvalues: B·C < 0 → spiral (oscillatory) convergence.
            # The amplitude envelope grows as exp(Re(λ)·t) = exp(A/2 · t), while
            # the angular frequency is Im(λ) = sqrt(|discriminant|)/2.
            # We set lambda_plus to the envelope rate so that the mixing / seed
            # logic is calibrated to the *amplitude* growth, not the real part alone.
            # sign(B-C) determines whether the spiral is clockwise or anti-clockwise,
            # but for seeding purposes only the amplitude matters.
            lambda_plus = 0.5 * (A + np.sqrt(abs(discriminant)))

        # ── Smart Seed: treat initial_Delta as a lower bound.
        # If initial_Delta > 0 (warm start), the Jacobi estimate may be too small.
        # Using floor to prevents SCF from drifting to the trivial Δ = 0 when a good seed exists.
        _Delta_floor = float(initial_Delta)   # caller-supplied lower bound

        if lambda_plus < 0.7:
            regime       = 'subcritical'
            Delta_kick   = max(initial_Q * p.g_JT * 0.5, p.kT, _Delta_floor)
            M_kick       = initial_M
            Q_kick       = initial_Q
            mixing_kick  = p.mixing

        elif lambda_plus <= 1.4:
            regime       = 'critical'
            # Near λ₊≈1 the map is nearly neutral: use a thermal-scale Δ seed to avoid the trivial Δ=0 fixpoint while remaining in the physical basin.
            Delta_kick   = max(3.0 * p.kT, 0.5 * p.g_JT * abs(initial_Q), _Delta_floor)
            M_kick       = initial_M
            # Seed Q from the self-consistent JT equilibrium at this Δ_kick:
            Q_kick_est   = (p.g_JT / max(self._K_bare, 1e-9)) * (Delta_kick / max(p.Delta_CF, 1e-9)) * N0
            Q_kick       = float(np.clip(Q_kick_est, initial_Q, 0.1 * p.lambda_hop))
            mixing_kick  = min(p.mixing * 0.5, 0.02) # Reduce mixing to slow down the neutral mode
        else:
            regime       = 'supercritical'
            Delta_kick   = float(np.clip( max(2.0 * p.kT * np.exp(min(1.0 / max(lambda_plus - 1.0, 0.05), 10.0)), _Delta_floor), 0.01, 0.3))
            M_kick       = initial_M * 0.8   # slight reduction: SC competes with AFM
            Q_kick_est   = (p.g_JT / max(self._K_bare, 1e-9)) * (Delta_kick / max(p.Delta_CF, 1e-9)) * N0
            Q_kick       = float(np.clip(Q_kick_est, initial_Q, 0.2 * p.lambda_hop))
            mixing_kick  = min(p.mixing * 0.25, 0.01)
        return {
            'M_kick':      M_kick,
            'Q_kick':      Q_kick,
            'Delta_kick':  Delta_kick,
            'mixing_kick': mixing_kick,
            'lambda_plus': lambda_plus,
            'regime':      regime,
            'chi_tau':     chi_tau_val,
        }

    def solve_self_consistent(self, target_doping: float, initial_M: float, initial_Q: float, initial_Delta: float, verbose: bool = True) -> Dict:
        """
        Coupled (M, Q, Δ_s, Δ_d, μ) SCF via Anderson-accelerated fixed-point + LM Newton.

        Order parameters
        ----------------
        M       : staggered AFM magnetisation (Gutzwiller-renormalised BdG Weiss field)
        Q       : B₁g JT lattice distortion (Å); zero in the AFM normal state by symmetry
        Δ_s     : on-site inter-orbital (Γ₆⊗Γ₇) singlet pairing amplitude (eV)
        Δ_d     : inter-site d-wave B₁g singlet pairing amplitude (eV)
        μ       : chemical potential enforcing ⟨n⟩ = 1 − δ (Newton + Brentq fallback)

        Algorithm per iteration
        -----------------------
        1. Build 16×16 BdG Hamiltonian H(k; M, Q, Δ_s, Δ_d, μ) on the uniform k-grid.
        2. Diagonalise → (E_k, ψ_k); compute observables: M_BdG, ⟨τ_x⟩, Pair_s, Pair_d.
        3. If SC+JT active (Δ>0, Q>0): inject anomalous orbital coherence ⟨τ_x⟩_anom
           into the Weiss field O_expectation, rebuild BdG cache.
        4. On iter 0 and when ≥5 iters passed AND |ΔM| > 0.02:
           update K_eff = K_lattice + ∂²F_ex/∂Q² (exchange rigidity correction).
        5. Solve gap equations for (Δ_s_out, Δ_d_out) via RPA vertex fixed-point.
        6. Update cluster free energy (DMFT-like vertex renormalisation of J_eff).
        7. Newton step for M via ∂F/∂M and ∂²F/∂M² (LM-damped); blend with BdG fixpoint.
        8. Update Q via the adiabatic JT equilibrium: Q_out = −(g_JT/K_eff)·⟨τ_x⟩.
        9. Apply Anderson(5) acceleration to (M, Q, |Δ_s|, |Δ_d|) jointly.
        10. Find μ to enforce density; compute F_BdG and F_cluster diagnostics.
        11. Adaptive mixing every 5 iters: halve α on divergence (max_diff > 1.05×prev),
            recover α on good convergence (×1.35); cap α near QCP (×0.6).
            Reset Anderson history on divergence, stagnation, or Q sign flip.

        Converged when max(|ΔM|,|ΔQ|,|ΔΔ_s|,|ΔΔ_d|) < tol and |n−(1−δ)| < tol×10.
        Near SC critical point (0.8<λ_max<1.8): tolerance relaxed to 5×tol.

        Post-convergence diagnostics
        ----------------------------
        - 3×3 Hessian of F(M, Q, Δ) (finite-difference); confirms free-energy minimum.
        - Linearised gap equation: largest eigenvalue λ_max and gap symmetry (B₁g / A₁g).
        - Static χ₀(q_AFM) and Stoner denominator (AFM stability check).
        - χ_τ multipolar susceptibility and λ_JT = (g²/K)·χ_τ (SC-triggered JT strength).
        """
        ALPHA_HF = self.p.ALPHA_HF
        converged = False
        # K_eff is tracked as a LOCAL variable throughout the SCF loop.
        _K_eff_scf: float = self._K_bare
        self._cluster_j_renorm = 1.0
        _mu0_est: float
        if abs(target_doping) < 0.01:
            _mu0_est = 0.0
        elif target_doping > 0:
            _mu0_est = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        else:
            _mu0_est = 2.0 * self.p.t0 * np.tanh(abs(target_doping) / 0.1)
        _mu0_est += 0.5 * self.p.Delta_CF

        kick = self._scf_jacobi_kick(target_doping, initial_M, initial_Q, float(initial_Delta))

        M = kick['M_kick']
        Q = kick['Q_kick']

        # Equal initial split between s and d channel
        _Delta_seed = max(kick['Delta_kick'], float(initial_Delta))
        Delta_s = _Delta_seed * 0.5 + 0.0j   # on-site orbital B₁g
        Delta_d = _Delta_seed * 0.5 + 0.0j   # inter-site d-wave B₁g

        mu = _mu0_est
        _alpha = kick['mixing_kick']

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [],
            'F_bdg': [], 'F_cluster': [],
            'g_t': [], 'g_J': [], 'mu': [],
            'mixing': [],       # adaptive mixing rate per iteration
        }

        if verbose:
            with _log_lock:
                print(f"[SCF δ={target_doping:.4f}] start  "
                      f"λ₊={kick['lambda_plus']:.3f}[{kick['regime']}]  "
                      f"M₀={M:.4f}  Q₀={Q:.5f}  Δ₀={_Delta_seed:.5f}  "
                      f"α={_alpha:.4f}", flush=True)

        scf_x_hist: list = []
        scf_f_hist: list = []

        chi0         = 0.0
        rpa_factor   = 1.0
        afm_unstable = False
        chi_tau      = 0.0   # computed once after convergence
        Ut_ratio     = 0.0   # computed once after convergence
        chi0_result  = {'Ut_ratio': 0.0}
        _vertex_cache: Optional[dict] = None
        irrep_info = {'w': 0.0, 'selection_ratio': 0.0,
                      'jt_algebraically_allowed': False,
                      'tau_x_projected': 0.0, 'tau_x_free_max': 1.0}
        _K_eff_last_M  = initial_M + 999.0
        _K_eff_last_iter = -5

        _scf_t0 = _time.time()
        _max_diff_prev = float('inf')    # previous iteration's max_diff
        _stagnation_count = 0            # consecutive near-stagnation iterations
        _lambda_max   = 0.0
        # QCP detection thresholds: pairing vertex amplitude — V_s catches single-channel (pure spin or pure JT) near-divergence.
        _gap_symmetry = 'unknown'
        g_t, g_J, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)

        for iteration in range(self.p.max_iter):
            _iter_t0 = _time.time()

            tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
            tx, ty = g_t * tx_bare, g_t * ty_bare

            _vbdg_scf = self._get_vbdg()
            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(_vbdg_scf._build_H_stack(_vbdg_scf._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=_vbdg_scf._H_stack))
            self._scf_bdg_cache = (_bdg_ev_sc, _bdg_ec_sc)

            obs = self._get_vbdg().compute_observables_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                _bdg_cache=(_bdg_ev_sc, _bdg_ec_sc))
            tau_x     = obs['Q']
            M_bdg     = obs['M']        # BdG response: lattice magnetization
            Delta_eff = abs(Delta_s) + abs(Delta_d)   # combined for irrep mixing weight

            # Feed back the SC-induced orbital coherence ⟨τ_x⟩ into O_expectation.
            # When Δ≠0 and Q≠0, the BdG eigenvectors carry anomalous Γ₆↔Γ₇ mixing absent in the diagonal approximation O_exp = g_J·f_d·M·sz_op.
            # tau_x_anom = Σ_k Re(u_k·v_k*) anomalous orbital weight (Eq. 6↔⟦7⟩ off-diag pairing).
            # It enters build_local_hamiltonian via J_B1g @ O_exp, i.e. it corrects the Weiss field to reflect that the SC condensate has made the B1g channel accessible.
            #     O_expectation  -> single-particle Weiss field in H_A / H_B
            #     gap_eq vertex  -> two-particle RPA pairing kernel V(q) from chi0_normal
            Delta_eff_now = abs(Delta_s) + abs(Delta_d)
            if Delta_eff_now > 1e-4 and abs(Q) > 1e-5:
                tau_x_anom   = self._compute_orbital_coherence_from_pairs(
                    M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
                abs_d_oe     = max(abs(target_doping), 1e-6)
                f_d_oe       = 1.0 - abs_d_oe           # RMFT: (1-δ) spin-site fraction
                O_exp_diag   = g_J * f_d_oe * M * self.sz_op   # (4,) diagonal part

                # off-diagonal τ_x couples only the Γ₆↑↔Γ₇↑ (index 0↔2) and Γ₆↓↔Γ₇↓ (index 1↔3)
                tau_x_component = tau_x_anom * np.array([1.0, 1.0, -1.0, -1.0])
                O_expectation_scf = O_exp_diag + tau_x_component
            else:
                O_expectation_scf = None   # use fast diagonal approximation

            irrep_info = self.compute_rank2_multipole_expectation(Delta_eff, tau_x)

            # Rebuild SC-state BdG cache with updated O_expectation if SC+JT active. This ensures the K-space Hamiltonian used for gap-eq and K_eff update
            if O_expectation_scf is not None:
                _vbdg_scf2 = self._get_vbdg()
                _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                    _vbdg_scf2._build_H_stack(_vbdg_scf2._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, O_expectation_scf, out = _vbdg_scf2._H_stack)
                )
                self._scf_bdg_cache = (_bdg_ev_sc, _bdg_ec_sc)

            _K_eff_update_needed = (
                iteration == 0
                or (iteration - _K_eff_last_iter >= 5 and abs(M - _K_eff_last_M) > 0.02)
            )
            if _K_eff_update_needed:
                _rigidity = self.compute_JT_rigidity_from_exchange(M, Q, mu, g_J, target_doping, g_t)
                _K_eff_scf       = max(_rigidity['K_eff'], 1e-9)
                _K_eff_last_M    = M
                _K_eff_last_iter = iteration

            # Gap equation fixed-point: Δ_out = g_Δ · V(q) · F_AA/AB(k; Δ, M, Q)
            # _bdg_cache provides the SC-state BdG amplitudes (u, v) for F_AA / F_AB.
            # The RPA vertex V(q) is rebuilt from Δ=0 susceptibilities inside compute_gap_eq_vectorized.
            Delta_s_out, Delta_d_out, _vertex_cache = self._get_vbdg().compute_gap_eq_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, g_Delta_s, g_Delta_d,
                _bdg_cache=(_bdg_ev_sc, _bdg_ec_sc),
                _vertex_cache=_vertex_cache)

            # Cluster ED: DMFT-like vertex renormalisation of J_eff beyond bare Gutzwiller.
            # _cluster_j_renorm = J_cluster / J_bare ∈ [0.5, 2.0] is read by J_alpha_beta_Q.
            cluster_result_pre = self.compute_cluster_free_energy(M, Q, mu, g_J, tx_bare, ty_bare, target_doping)
            J_eff_cluster  = cluster_result_pre['J_eff']
            J_eff_bare     = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping)
            if abs(J_eff_bare) > 1e-10:
                _j_renorm = float(np.clip(J_eff_cluster / J_eff_bare, 0.5, 2.0))
            else:
                _j_renorm = 1.0
            self._cluster_j_renorm = _j_renorm

            dF_dM_0, d2F_dM2 = self.compute_dF_dM_and_d2F(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            self._scf_bdg_cache = None   # cache consumed; clear to prevent stale reuse

            # Adaptive LM floor: large μ_LM can overdamp M even when Δ grows, freezing SC–AFM coupling.
            # Use Delta_s_out + Delta_d_out so the M Newton step already knows about the SC gap that just opened this iteration.
            _Delta_out_now = abs(Delta_s_out) + abs(Delta_d_out)
            _mu_LM_eff = self.p.mu_LM / (1.0 + 10.0 * _Delta_out_now / max(self.p.t0, 1e-9))

            # LM denominator: d2F_dM2 + mu_LM_eff (positive shift guarantees a positive denominator while preserving sign)
            # When d2F < 0 (saddle/instability), abs() would flip the Newton direction and push M away from the minimum, blocking convergence.
            M_newton = M - dF_dM_0 / (d2F_dM2 + _mu_LM_eff)
            # Safety: clamp Newton proposal to physical range before blending
            M_newton = float(np.clip(M_newton, 0.0, 1.0))
            # Self-consistent fixpoint: ⟨S_z⟩ = M from BdG Green function = ∂Ω_BdG/∂h|_{h→0}.
            M_out = float(np.clip(
                (1.0 - ALPHA_HF) * M_bdg + ALPHA_HF * M_newton,
                0.0, 1.0
            ))
            
            # The cluster ⟨τ_x⟩ is NOT used here — it carries only local fluctuations and has no k-resolved Fermi surface weighting.
            Q_out = -(self.p.g_JT / max(_K_eff_scf, 1e-9)) * tau_x
            Q_out = float(np.clip(Q_out, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # Δ update BEFORE Anderson mix of (M, Q) so the M update sees the current SC state, avoiding delayed convergence from asymmetric M→Δ feedback.
            Delta_s_mixed = self._mix(Delta_s, Delta_s_out, alpha=_alpha)
            Delta_d_mixed = self._mix(Delta_d, Delta_d_out, alpha=_alpha)

            # 4D Anderson vector (M, Q, |Δ_s|, |Δ_d|) to capture ∂M/∂Δ and ∂Δ/∂M coupling in the Jacobian.
            # Scale variables to similar magnitudes in the least-squares solve: M ×1,  Q ×1/λ_hop,  Δ_s ×t₀,  Δ_d ×t₀.
            _t0_sc = max(self.p.t0, 1e-6)
            _lhop  = max(self.p.lambda_hop, _KT_FLOOR)
            x_in_4d  = np.array([M,             Q / _lhop,
                                  abs(Delta_s) * _t0_sc, abs(Delta_d) * _t0_sc])
            x_out_4d = np.array([M_out,             Q_out / _lhop,
                                  abs(Delta_s_mixed) * _t0_sc, abs(Delta_d_mixed) * _t0_sc])
            scf_x_hist.append(x_in_4d)
            scf_f_hist.append(x_out_4d)

            x_new_4d = self._anderson_mix(scf_x_hist, scf_f_hist, m=5, alpha=_alpha)
            M_mixed    = float(np.clip(x_new_4d[0], 0.0, 1.0))
            Q_mixed    = float(np.clip(x_new_4d[1] * _lhop, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # Anderson updates |Δ| only; apply magnitude corrections but keep the phase from the linear-mix value (important for BdG).
            _Ds_abs_new = float(np.clip(x_new_4d[2] / _t0_sc, 0.0, 1.0))
            _Dd_abs_new = float(np.clip(x_new_4d[3] / _t0_sc, 0.0, 1.0))
            
            _phase_s = Delta_s_mixed / (abs(Delta_s_mixed) + 1e-30)
            _phase_d = Delta_d_mixed / (abs(Delta_d_mixed) + 1e-30)
            Delta_s_mixed = complex(_Ds_abs_new) * _phase_s
            Delta_d_mixed = complex(_Dd_abs_new) * _phase_d

            if len(scf_x_hist) > 1 and (Q * Q_mixed < 0):
                scf_x_hist.clear()
                scf_f_hist.clear()
                _vertex_cache = None         # Q sign flip → FS topology may change
                self._scf_bdg_cache = None   # topology change → stale SC cache unsafe
                self._chi0_norm_cache = None  # χ₀ eigenvectors keyed on Q → must rebuild

            tx_mixed_bare, ty_mixed_bare = self.effective_hopping_anisotropic(Q_mixed)
            tx_mixed, ty_mixed = g_t * tx_mixed_bare, g_t * ty_mixed_bare
            
            mu_new, _mu_bdg_cache = self._find_mu_for_density(M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, tx_mixed, ty_mixed, mu, g_J)

            # Reuse it directly instead of calling eigh a second time for n(μ_new).
            if _mu_bdg_cache is not None:
                _ev_mu, _ec_mu = _mu_bdg_cache
                _fn_mu  = self.fermi_function(_ev_mu)
                _fnb_mu = 1.0 - _fn_mu
                _uA_mu  = _ec_mu[:, 0:4,   :]
                _uB_mu  = _ec_mu[:, 4:8,   :]
                _vA_mu  = _ec_mu[:, 8:12,  :]
                _vB_mu  = _ec_mu[:, 12:16, :]
                _dA = np.sum(np.abs(_uA_mu)**2 * _fn_mu[:, None, :]
                           + np.abs(_vA_mu)**2 * _fnb_mu[:, None, :], axis=(1, 2))
                _dB = np.sum(np.abs(_uB_mu)**2 * _fn_mu[:, None, :]
                           + np.abs(_vB_mu)**2 * _fnb_mu[:, None, :], axis=(1, 2))
                n_kspace_new = float(np.dot(self.k_weights, _dA + _dB)) / 4.0
            else:
                n_kspace_new = self._compute_density_at_mu(mu_new, M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, tx_mixed, ty_mixed, g_J)
            
            # Pass V_s, V_d from vertex cache → consistent with gap equation
            _V_s_fbdg = _vertex_cache.get('V_s_scalar', None) if _vertex_cache else None
            _V_d_fbdg = _vertex_cache.get('V_d_scalar', None) if _vertex_cache else None
            F_bdg = self.compute_bdg_free_energy(
                M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, mu_new,
                tx_mixed, ty_mixed, g_J, K_eff_for_free_energy=_K_eff_scf,
                _ev_cache=_mu_bdg_cache[0] if _mu_bdg_cache is not None else None,
                V_s=_V_s_fbdg, V_d=_V_d_fbdg
            )
            F_cluster = self.compute_cluster_free_energy(M_mixed, Q_mixed, mu_new, g_J, tx_mixed_bare, ty_mixed_bare, target_doping)
            Delta_s_abs = abs(Delta_s_mixed)
            Delta_d_abs = abs(Delta_d_mixed)
            # max_diff tracks order-parameter convergence (M, Q, Δ_s, Δ_d).
            max_diff = max(
                abs(M_mixed - M),
                abs(Q_mixed - Q),
                abs(Delta_s_abs - abs(Delta_s)),
                abs(Delta_d_abs - abs(Delta_d)),
            )

            if iteration >= 5 and iteration % 5 == 0:
                if max_diff > _max_diff_prev * 1.05:
                    # Diverging: halve alpha and reset history
                    _alpha = max(_alpha * 0.5, self.p.mixing / 8.0)
                    scf_x_hist.clear()
                    scf_f_hist.clear()
                    _stagnation_count = 0
                elif max_diff > _max_diff_prev * 0.85:
                    # Stagnating: count and reset history if persistent
                    _stagnation_count += 1
                    if _stagnation_count >= 2:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _stagnation_count = 0
                else:
                    # Converging well: recover alpha toward nominal faster
                    _stagnation_count = 0
                    if _alpha < self.p.mixing * 0.95:
                        _alpha = min(_alpha * 1.35, self.p.mixing)
            _max_diff_prev = max_diff

            # QCP detection: RPA det + V_s proxy
            _det_now = _vertex_cache.get('det_q0', 1.0) if _vertex_cache else 1.0
            _Vs_now  = abs(_vertex_cache.get('V_s_scalar', 0.0)) if _vertex_cache else 0.0
            _near_critical = (_det_now < _RPA_DET_WARN) or (_Vs_now > _V_CUT)
            _tol_use = self.p.tol * (5.0 if _near_critical else 1.0)
            if _near_critical:
                # Cap alpha near QCP: large steps overshoot the near-singular gap vertex
                _alpha = min(_alpha, self.p.mixing * 0.6)
            if _vertex_cache is not None:
                _vertex_cache['near_critical'] = _near_critical
                # Update actual_doping from the μ-finder density — used by Moriya damping in the *next* vertex rebuild so it reflects the current ⟨n⟩.
                _vertex_cache['actual_doping'] = float(1.0 - n_kspace_new)

            _iter_ms = (_time.time() - _iter_t0) * 1000.0

            history['M'].append(abs(M_mixed))
            history['Q'].append(abs(Q_mixed))
            history['Delta'].append(Delta_s_abs + Delta_d_abs)
            history['density'].append(n_kspace_new)
            history['F_bdg'].append(F_bdg)
            history['F_cluster'].append(F_cluster['F_per_site'])
            history['g_t'].append(g_t)
            history['g_J'].append(g_J)
            history['mu'].append(mu_new)
            history['mixing'].append(_alpha)

            if verbose:
                _elapsed = _time.time() - _scf_t0
                _frac    = (iteration + 1) / self.p.max_iter
                _w       = 38
                _filled  = int(_w * _frac)
                _bar     = "█" * _filled + "░" * (_w - _filled)
                _eta_s   = (_elapsed / max(iteration + 1, 1)) * (self.p.max_iter - iteration - 1)
                _h, _r   = divmod(int(_eta_s), 3600)
                _m, _s   = divmod(_r, 60)
                _qcp_tag = " QCP!" if _near_critical else "     "
                with _log_lock:
                    sys.stdout.write(
                        f"\r  SCF δ={target_doping:.3f} [{_bar}]"
                        f" {iteration+1:3d}/{self.p.max_iter}"
                        f"  conv={max_diff:.1e}  M={M:.3f}  Q={Q:+.4f}"
                        f"  |Δ|={(abs(Delta_s)+abs(Delta_d)):.4f}"
                        f"  det={_det_now:.3f}  Vs={_Vs_now:.1f}"
                        f"  {_iter_ms:4.0f}ms/it{_qcp_tag}"
                        f"  ETA {_h}:{_m:02d}:{_s:02d}  "
                    )
                    sys.stdout.flush()

            M, Q, Delta_s, Delta_d, mu = M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, mu_new

            # Saddle-escape kick: if |Δ| ≈ 0 and the Hessian has a negative eigenvalue, SCF sits at a saddle of F.
            # Kick along the λ_min eigenvector (steepest descent) to reveal the preferred (M, Q, Δ) direction.
            #   |e[2]| ≫ |e[0]|,|e[1]|  → pure SC instability  (clean Δ kick)
            #   |e[1]| ≫ |e[0]|,|e[2]|  → pure JT instability  (kick Q too)
            #   |e[1]| and |e[2]| both large → SC-triggered JT  (the target!)
            #   |e[0]| dominant           → AFM fluctuation     (kick M)
            _Delta_abs_now = abs(Delta_s) + abs(Delta_d)
            _kick_eligible = (
                _Delta_abs_now < 5.0 * self.p.tol
                and iteration > 3
                and iteration % 8 == 0
            )
            if _kick_eligible:
                try:
                    _hk = self.compute_hessian(
                        M, Q, max(_Delta_abs_now, 1e-6), target_doping, mu, g_t, g_J,
                        Delta_s_frac=0.5)
                    _lmin_k = _hk.get('min_curvature', 0.0)
                    if (_lmin_k is not None and np.isfinite(_lmin_k)
                            and _lmin_k < -0.3):
                        _evals_k, _evecs_k = np.linalg.eigh(_hk['H'])
                        _edir = _evecs_k[:, 0]   # eigenvector of λ_min: (M, Q, Δ)

                        # Physical scale normalisation before interpreting components:
                        #   M  ~ O(0.1–0.5)      → scale 1
                        #   Q  ~ O(lambda_hop)   → scale lambda_hop  [Å]
                        #   Δ  ~ O(kT–0.1 eV)   → scale 1
                        _lhop = max(self.p.lambda_hop, 1e-4)
                        _scale = np.array([1.0, _lhop, 1.0])
                        _edir_phys = _edir * _scale
                        _edir_phys /= max(np.linalg.norm(_edir_phys), 1e-12)

                        # Mode identification (use unscaled components for classification)
                        _wM  = abs(_edir[0])
                        _wQ  = abs(_edir[1])
                        _wD  = abs(_edir[2])
                        _wsum = max(_wM + _wQ + _wD, 1e-12)
                        _fM  = _wM / _wsum
                        _fQ  = _wQ / _wsum
                        _fD  = _wD / _wsum

                        if _fD > 0.6:
                            _mode = 'pure-SC'
                        elif _fQ > 0.6:
                            _mode = 'pure-JT'
                        elif _fD > 0.3 and _fQ > 0.3:
                            _mode = 'SC-triggered-JT'   # the target!
                        elif _fM > 0.6:
                            _mode = 'AFM-fluctuation'
                        else:
                            _mode = 'mixed'

                        # The kick magnitude is scaled by 2·kT so it is a thermal fluctuation amplitude
                        _kick_mag = 2.0 * self.p.kT

                        M_kick  = float(np.clip(
                            M + _kick_mag * _edir_phys[0], 0.0, 1.0))
                        Q_kick  = float(np.clip(
                            Q + _kick_mag * _edir_phys[1],
                            -0.5 * _lhop, 0.5 * _lhop))
                        # Δ: preserve s/d ratio, boost magnitude along eigenvector
                        _D_base  = max(_Delta_abs_now, 1e-6)
                        _D_kick  = float(np.clip(
                            _D_base + _kick_mag * abs(_edir_phys[2]), 0.0, 0.4))
                        _ratio_s = (abs(Delta_s) / (_D_base + 1e-30))
                        Delta_s_kick = complex(np.clip(_D_kick * _ratio_s, 0.0, 0.3))
                        Delta_d_kick = complex(np.clip(_D_kick * (1.0 - _ratio_s), 0.0, 0.3))

                        # Only apply the kick if the Δ component is meaningful
                        # (pure AFM fluctuations are better left to the M Newton step)
                        if _fD > 0.15:
                            M       = M_kick
                            Q       = Q_kick
                            Delta_s = Delta_s_kick
                            Delta_d = Delta_d_kick
                            scf_x_hist.clear()
                            scf_f_hist.clear()
                            _vertex_cache = None
                            self._scf_bdg_cache = None
                            if verbose:
                                _scf_log(f"SCF δ={target_doping:.3f}",
                                         f"⚡ kick iter={iteration}  mode={_mode}"
                                         f"  λ_min={_lmin_k:+.3f}"
                                         f"  fM={_fM:.2f} fQ={_fQ:.2f} fΔ={_fD:.2f}"
                                         f"  → M={M:.3f} Q={Q:+.4f} |Δ|={_D_kick:.4f}")
                except Exception:
                    pass   # saddle kick is best-effort; never abort SCF

            if max_diff < _tol_use and abs(n_kspace_new - (1 - target_doping)) < _tol_use * 10:
                converged = True
                break

        if not converged and verbose:
            tag = f"SCF δ={target_doping:.4f}"
            with _log_lock:
                print(f"", flush=True)
            _scf_log(tag, f"⚠ no conv after {self.p.max_iter} iters  "
                          f"max_diff={max_diff:.2e}  "
                          f"dens_err={abs(n_kspace_new-(1-target_doping)):.2e}")

        # Post-loop diagnostic: λ_max and Rayleigh JT projection.
        _lin: Dict = self.solve_linearized_gap_equation(M, Q, Delta_s, Delta_d, target_doping, mu, tx_mixed, ty_mixed, g_J, actual_doping=float(1.0 - n_kspace_new))
        _lambda_max      = _lin['lambda_max']
        _gap_symmetry    = _lin['gap_symmetry']
        _lambda_JT_kernel = _lin['lambda_JT_kernel']

        if converged:
            _Delta_total = abs(Delta_s) + abs(Delta_d)
            _Delta_s_frac = (abs(Delta_s) / _Delta_total) if _Delta_total > 1e-10 else 0.5
            # Single eigh at the converged point — reused for F0 in compute_hessian, skipping 1 of its 13 diagonalisations.
            _vbdg_conv = self._get_vbdg()
            _ev_conv, _ = np.linalg.eigh(
                _vbdg_conv._build_H_stack(
                    _vbdg_conv._kpts, M, Q, Delta_s, Delta_d,
                    target_doping, mu, tx_mixed, ty_mixed, g_J,
                    out=_vbdg_conv._H_stack))
            hessian_result = self.compute_hessian(
                M, Q, _Delta_total, target_doping, mu, g_t, g_J,
                Delta_s_frac=_Delta_s_frac,
                K_eff_for_free_energy=_K_eff_scf,   # Post-convergence Hessian / curvature test
                _ev0_cache=_ev_conv,
                V_s=_vertex_cache.get('V_s_scalar', None) if _vertex_cache else None,
                V_d=_vertex_cache.get('V_d_scalar', None) if _vertex_cache else None)
        else:
            hessian_result = {'H': None, 'eigenvalues': None, 'is_minimum': None, 'min_curvature': None}

        chi0_result = self.compute_static_chi0_afm(M, Q, Delta_s, Delta_d, target_doping, mu, tx_mixed, ty_mixed, g_J)
        chi0         = chi0_result['chi0']
        rpa_factor   = chi0_result['rpa_factor']
        afm_unstable = chi0_result['afm_unstable']
        _chi_tau_result = self._compute_chi_tau(M, Q, target_doping, Delta_s, Delta_d, mu)
        chi_tau  = _chi_tau_result['chi_tau']
        Ut_ratio = _chi_tau_result['Ut_ratio']

        # ── SC-triggered JT channel diagnostic: χ_SQ(Δ=0) vs χ_SQ(Δ≠0)
        if converged and verbose:
            _tag_chi = f"SCF δ={target_doping:.4f}"
            try:
                # Normal-state χ_SQ (Δ=0), τ_x projection built in.
                _n_sus = self.get_susceptibilities_normal(
                    q=np.zeros(2), M=M, Q=Q,
                    target_doping=target_doping, mu=mu,
                    tx=tx_mixed, ty=ty_mixed, g_J=g_J,
                    actual_doping=float(1.0 - n_kspace_new))
                _chi_SQ_normal = _n_sus['chi_SQ']

                # SC-state χ_SQ: Δ≠0 → condensate opens the Γ₆↔Γ₇ channel.
                # Direct τ_x projection from chi0_tensor at Δ≠0 is unavailable without a BdG Lindhard sum
                # — the QQ-ratio proxy is the correct approximation here
                _sc_sus = self.get_susceptibilities_sc(
                    M=M, Q=Q,
                    Delta_s=complex(Delta_s), Delta_d=complex(Delta_d),
                    target_doping=target_doping, mu=mu,
                    tx=tx_mixed, ty=ty_mixed, g_J=g_J)
                # χ_SQ proxy in SC state: Δχ_QQ = χ_QQ(SC) − χ_QQ(N) ∝ χ_SQ
                _chi_SQ_sc_proxy = _sc_sus['chi_QQ'] - _sc_sus['chi_QQ_normal']
                _chi_QQ_ratio    = _sc_sus['chi_QQ_ratio']
                _channel_opened  = _sc_sus['sc_jt_signal']

                with _log_lock:
                    print(
                        f"[{_tag_chi}] SC-triggered χ_SQ channel:"
                        f"  χ_SQ(Δ=0,τ_x-proj)={_chi_SQ_normal:.6e}"
                        f"  χ_QQ(SC)/χ_QQ(N)={_chi_QQ_ratio:.3f}×"
                        f"  Δχ_QQ={_chi_SQ_sc_proxy:.4e}"
                        f"  {'✓ CHANNEL OPEN — SC-triggered JT active' if _channel_opened else '✗ channel still closed'}",
                        flush=True
                    )
            except Exception as _chi_sq_err:
                _scf_log(_tag_chi, f"χ_SQ diagnostic failed: {_chi_sq_err}")
            
            # Lambda_s / Lambda_d channel decomposition: if lambda_d > lambda_s but Delta_s != 0, the SCF has mixed channels beyond what the linear kernel predicts.
            try:
                _gap_vec   = _lin['gap_vector']
                _fs_pts    = _lin['fs_pts']
                _phi_s     = np.ones(len(_fs_pts));    _phi_s /= np.linalg.norm(_phi_s)
                _phi_d     = np.cos(_fs_pts[:,0]) - np.cos(_fs_pts[:,1])
                _nd        = np.linalg.norm(_phi_d)
                if _nd > 1e-10:
                    _phi_d /= _nd
                _w_s_post   = float(abs(_gap_vec @ _phi_s))
                _w_d_post   = float(abs(_gap_vec @ _phi_d))
                _lam_max_s  = _lin['lambda_max'] * _w_s_post / max(_w_s_post + _w_d_post, 1e-10)
                _lam_max_d  = _lin['lambda_max'] * _w_d_post / max(_w_s_post + _w_d_post, 1e-10)
                _Delta_s_mag = abs(Delta_s)
                _Delta_d_mag = abs(Delta_d)
                _nl_mixing = (_lam_max_d > _lam_max_s) and (_Delta_s_mag > 1e-5) and (_Delta_d_mag > 1e-5)
                _scf_log(_tag_chi,
                         f"Gap channels (post-SCF): lambda_s={_lam_max_s:.4f}  lambda_d={_lam_max_d:.4f}"
                         f"  |Delta_s|={_Delta_s_mag*1000:.3f}meV  |Delta_d|={_Delta_d_mag*1000:.3f}meV"
                         f"  {'[NONLINEAR s-d MIXING: lambda_d>lambda_s but Delta_s!=0]' if _nl_mixing else '[dominant channel consistent]'}")
            except Exception as _lam_ch_err:
                _scf_log(_tag_chi, f"lambda_s/d channel decomposition failed: {_lam_ch_err}")

            # Incommensurate Q diagnostic: check whether the AFM ordering wavevector has shifted from (pi,pi).
            try:
                _dq_scan   = np.linspace(0.0, 0.15 * np.pi, 7)
                _chi_SS_scan = []
                _vbdg_ic   = self._get_vbdg()
                _Ek_ic, _Vk_ic = np.linalg.eigh(
                    _vbdg_ic._build_H_stack(_vbdg_ic._kpts_ev, M, Q, 0.0+0j, 0.0+0j,
                                            target_doping, mu, tx_mixed, ty_mixed, g_J,
                                            out=_vbdg_ic._H_stack_ev))
                _Sz_diag_ic = self.sz_bdg16   # [pA,pB,hA,hB] × [Γ₆↑,Γ₆↓,Γ₇↑,Γ₇↓]
                for _dq_v in _dq_scan:
                    _q_ic   = np.array([np.pi, np.pi - _dq_v])
                    _kpts_q = (self.k_points_even + _q_ic[None,:] + np.pi) % (2*np.pi) - np.pi
                    _Ekq_ic, _Vkq_ic = np.linalg.eigh(
                        _vbdg_ic._build_H_stack(_kpts_q, M, Q, 0.0+0j, 0.0+0j,
                                                target_doping, mu, tx_mixed, ty_mixed, g_J))
                    _fk  = self.fermi_function(_Ek_ic)
                    _fkq = self.fermi_function(_Ekq_ic)
                    _SzV = _Sz_diag_ic[None,:,None] * _Vkq_ic
                    _Mm  = np.einsum('kin,kim->knm', _Vk_ic.conj(), _SzV)
                    _M2  = np.abs(_Mm)**2
                    _df  = _fk[:,:,None] - _fkq[:,None,:]
                    _dE  = _Ekq_ic[:,None,:] - _Ek_ic[:,:,None]
                    _msk = (np.abs(_df) > _FD_MASK_DF) & (np.abs(_dE) > _FD_MASK_DE)
                    _sdE = np.where(_msk, _dE, 1.0)
                    _r   = np.where(_msk, self.k_weights_even[:,None,None]*_M2*_df/_sdE, 0.0)
                    _chi_SS_scan.append(float(_r.sum()))
                _idx_max = int(np.argmax(_chi_SS_scan))
                _dq_max  = float(_dq_scan[_idx_max])
                _scf_log(_tag_chi,
                         f"Incommensurate Q scan: chi_SS max at dq={_dq_max/np.pi:.3f}*pi"
                         f"  chi_SS(0)={_chi_SS_scan[0]:.4f}  chi_SS(max)={_chi_SS_scan[_idx_max]:.4f}"
                         f"  {'[WARNING: incommensurate tendency dq>0.05*pi -- Q=(pi,pi-dq) may be preferred]' if _dq_max > 0.05 * np.pi else '[commensurate AFM confirmed]'}")
            except Exception as _ic_err:
                _scf_log(_tag_chi, f"Incommensurate scan failed: {_ic_err}")

        if verbose:
            _tag = f"SCF δ={target_doping:.4f}"
            if hessian_result.get('eigenvalues') is not None:
                _eigs  = hessian_result['eigenvalues']
                _hstat = "✓MIN" if hessian_result['is_minimum'] else "⚠SADDLE"
                _hstr  = f"H=[{_eigs[0]:.3f},{_eigs[1]:.3f},{_eigs[2]:.3f}]{_hstat}"
            else:
                _hstr = "H=n/a"
            with _log_lock:
                print(f"", flush=True)   # newline after \r progress bar
                print(
                    f"[{_tag}] ✓ conv {iteration+1} iters"
                    f"  M={M:.4f}  Q={Q:+.4f}  |Δs|={abs(Delta_s):.4f}  |Δd|={abs(Delta_d):.4f}"
                    f"  n={n_kspace_new:.4f}  μ={mu:.4f}  F={F_bdg:.5f}"
                    f"  λ_max={_lambda_max:.3f}({_gap_symmetry[:3]})"
                    f"  JT={'✓' if irrep_info['jt_algebraically_allowed'] else '✗'}"
                    f"  {_hstr}",
                    flush=True)

        # ── Post-SCF validations ──────────────────────────────────────────────
        # 1. λ_min(Hessian_SC): the authoritative SC-triggered JT criterion.
        #    λ_min < 0  →  SC has genuinely triggered JT (condensate softened the mode).
        #    λ_min > 0  →  the converged SC+AFM state is a true local minimum without
        #                   orbital distortion — JT was NOT triggered.
        #    This is categorically different from the pre-SCF G3[2,2] > 0 check, because here M, Q, Δ are all fully self-consistent fixpoint.
        _hess_lmin_sc = float('nan')
        _hess_sc_triggered = False
        try:
            if hessian_result.get('eigenvalues') is not None:
                _hess_lmin_sc = float(hessian_result['eigenvalues'][0])
                _hess_sc_triggered = _hess_lmin_sc < 0.0
        except Exception:
            pass

        # 2. Coherence length — post-SCF, uses converged (M, Q, Δ, μ, tx, ty).
        _sc_result_for_xi = {
            'M': M, 'Q': Q,
            'Delta_s': abs(Delta_s), 'Delta_d': abs(Delta_d),
            'mu': mu, 'tx': tx_mixed, 'ty': ty_mixed,
            'g_J': g_J, 'converged': converged,
        }
        try:
            _xi_res = self.compute_coherence_length(target_doping, _sc_result_for_xi)
        except Exception as _xi_err:
            _scf_log(f"SCF δ={target_doping:.4f}", f"coherence length failed: {_xi_err}")
            _xi_res = {'xi_over_a': float('nan'), 'valid_BdG': False,
                       'orbital_selective': False, 'note': f'failed: {_xi_err}'}

        if verbose:
            _xi_note = _xi_res['note']
            _hmin_str = (f"λ_min(H_SC)={_hess_lmin_sc:+.4f}"
                         f"  {'✓ SC-triggered JT' if _hess_sc_triggered else '— JT not triggered'}")
            _scf_log(f"SCF δ={target_doping:.4f}",
                     f"Post-SCF:  ξ/a={_xi_res['xi_over_a']:.2f}"
                     f"  {'✓ BdG valid' if _xi_res['valid_BdG'] else '⚠ BdG marginal'}"
                     f"  orbital_sel={'✓' if _xi_res['orbital_selective'] else '—'}"
                     f"  {_hmin_str}")
            _scf_log(f"SCF δ={target_doping:.4f}", f"  ξ note: {_xi_note}")

        # ── Post-SCF Mott filter ──────────────────────────────────────────────
        # (a) g_t < 0.10 (δ < 0.053): primary Mott guard prevents incoherent ZRS band
        #     The Gutzwiller factor encodes the full doping-dependent Mott suppression;
        #     no SC gap can be physical without coherent hopping.
        # (b) ξ/a < 1.0: BEC/artefact limit — Cooper pairs not coherent across a lattice site;
        #     Δ is suppressed post-hoc (BdG mean-field breaks down in this regime).
        _mott_g_t       = g_t
        _mott_xi_over_a = _xi_res['xi_over_a']
        _mott_suspect   = (_mott_g_t < _G_T_COHERENCE_MIN) or (_mott_xi_over_a < 1.0)
        if _mott_suspect:
            Delta_s   = 0.0 + 0.0j
            Delta_d   = 0.0 + 0.0j
            converged = False
            if verbose:
                _reason = ('g_t<_G_T_COHERENCE_MIN (Mott)' if _mott_g_t < _G_T_COHERENCE_MIN
                           else f'ξ/a={_mott_xi_over_a:.2f}<1 (BEC/artefact)')
                _scf_log(f"SCF δ={target_doping:.4f}",
                         f"⚠ MOTT-SUSPECT [{_reason}]:"
                         f" g_t={_mott_g_t:.3f} ξ/a={_mott_xi_over_a:.2f}"
                         f" — gap suppressed")

        return {
            'M': M,
            'Q': Q,
            'Delta_s': abs(Delta_s),
            'Delta_d': abs(Delta_d),
            'chi_tau': chi_tau,
            'Ut_ratio': Ut_ratio,
            'density': n_kspace_new,
            'mu': mu,
            'g_t': g_t,
            'g_J': g_J,
            'F_bdg': F_bdg,
            'F_cluster': F_cluster['F_per_site'],
            'tx': tx_mixed,
            'ty': ty_mixed,
            'J_eff': F_cluster['J_eff'],
            'target_doping': target_doping,
            'chi0': chi0,
            'chi0_moriya': chi0_result['chi0_moriya'],
            'rpa_factor': rpa_factor,
            'afm_unstable': afm_unstable,
            'irrep_info': irrep_info,
            'history': history,
            'hessian': hessian_result,
            'hessian_lmin_sc': _hess_lmin_sc,
            'sc_jt_confirmed': _hess_sc_triggered,
            'coherence': _xi_res,
            'xi_over_a': _xi_res['xi_over_a'],
            'lambda_max': _lambda_max,
            'lambda_max_raw': _lin.get('lambda_max_raw', float('nan')),
            'g_delta_dom': _lin.get('g_delta_dom', float('nan')),
            'gap_vector': _lin.get('gap_vector', None),
            'fs_pts': _lin.get('fs_pts', None),
            'V_spin_mean': _lin.get('V_spin_mean', float('nan')),
            'V_JT_mean': _lin.get('V_JT_mean', float('nan')),
            'V_cross_mean': _lin.get('V_cross_mean', float('nan')),
            'V_rpa_mean': _lin.get('V_rpa_mean', float('nan')),
            'gap_symmetry': _gap_symmetry,
            'lambda_JT_kernel': _lambda_JT_kernel,
            'lambda_plus': kick['lambda_plus'],
            'regime': kick['regime'],
            'K_eff_scf': _K_eff_scf,
            'converged': converged,
            'mott_suspect': _mott_suspect,
        }

    def _anderson_mix(self, x_history: list, f_history: list, m: int = 5, alpha: float = None) -> np.ndarray:
        """
        Robust Anderson(m) acceleration with:
            • scale normalisation (component balancing)
            • adaptive Tikhonov regularisation
            • condition-number guard
            • residual-norm acceptance test
            • trust-region safeguard
            • automatic fallback to linear mixing

        Safe for stiff SCF problems (BdG + JT + feedback).
        """
        alpha = self.p.mixing if alpha is None else alpha

        x_last = np.asarray(x_history[-1], dtype=float)
        f_last = np.asarray(f_history[-1], dtype=float)

        # Simple linear mixing fallback
        x_simple = (1.0 - alpha) * x_last + alpha * f_last

        n = min(len(x_history), m)
        if n < 2:
            return x_simple

        # Build history arrays
        X = np.asarray(x_history[-n:], dtype=float)   # (n, d)
        F = np.asarray(f_history[-n:], dtype=float)
        R = F - X                                    # residuals

        dR = np.diff(R, axis=0)                      # (n-1, d)
        dX = np.diff(X, axis=0)
        r_last = R[-1]

        # Component scaling: prevent M and Q from dominating each other numerically.
        scale = np.ones_like(x_last)

        # If second component is Q, scale by lambda_hop
        if len(scale) >= 2 and hasattr(self.p, "lambda_hop"):
            scale[1] = 1.0 / max(self.p.lambda_hop, _KT_FLOOR)

        R_scaled  = R  * scale
        dR_scaled = dR * scale
        r_scaled  = r_last * scale

        # Solve regularised normal equations
        # min || r_last - dR theta ||² + beta ||theta||²
        A = dR_scaled @ dR_scaled.T
        b = dR_scaled @ r_scaled

        # Adaptive Tikhonov regularisation
        diag_max = max(float(np.max(np.abs(np.diag(A)))), 1e-30)
        beta = _ANDERSON_TIKHONOV * diag_max
        A.flat[::A.shape[0] + 1] += beta

        # Condition number guard
        try:
            condA = np.linalg.cond(A)
            if not np.isfinite(condA) or condA > 1e12:
                return x_simple
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return x_simple

        # Canonical Anderson update: x_new = x_last + r_last - (dX + dR) @ theta
        correction = (dX + dR).T @ theta
        x_opt = x_last + r_last - correction

        if not np.all(np.isfinite(x_opt)):
            return x_simple

        # Residual-norm acceptance test
        # Only accept Anderson if it actually reduces residual.
        r_simple = f_last - x_simple
        r_opt    = f_last - x_opt

        norm_last   = np.linalg.norm(r_last)
        norm_simple = np.linalg.norm(r_simple)
        norm_opt    = np.linalg.norm(r_opt)

        # If Anderson worsens residual → fallback
        if norm_opt > norm_last:
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
        w = float(np.clip(alpha / max(self.p.mixing, _KT_FLOOR), _ANDERSON_W_LO, _ANDERSON_W_HI))
        x_new = w * x_opt + (1.0 - w) * x_simple

        if not np.all(np.isfinite(x_new)):
            return x_simple
        return x_new

    def compute_dF_dM_and_d2F(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> Tuple[float, float]:
        t_sq_avg    = 0.5 * (tx**2 + ty**2)   # renormalized: consistent with BdG spectrum
        abs_d       = max(abs(target_doping), 1e-6)
        f_d         = 1.0 - abs_d              # RMFT: (1-δ) spin-site fraction
        h_prefactor = g_J * f_d * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) / 2.0

        sz_orb = self.sz_op * h_prefactor
        dH_diag = np.array([
            -sz_orb[0], -sz_orb[1], -sz_orb[2], -sz_orb[3],   # particle A
            +sz_orb[0], +sz_orb[1], +sz_orb[2], +sz_orb[3],   # particle B
            +sz_orb[0], +sz_orb[1], +sz_orb[2], +sz_orb[3],   # hole A
            -sz_orb[0], -sz_orb[1], -sz_orb[2], -sz_orb[3],   # hole B
        ])

        vbdg = self._get_vbdg()
        if hasattr(self, '_scf_bdg_cache') and self._scf_bdg_cache is not None:
            ev, ec = self._scf_bdg_cache
        else:
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))
        f_all = self.fermi_function(ev)
        exp_nn = np.einsum('i,kin->kn', dH_diag, np.abs(ec)**2)

        grad = float(np.einsum('k,kn,kn->', self.k_weights, f_all, exp_nn)) / 2.0

        kT = self.p.kT
        df_dE = -f_all * (1.0 - f_all) / max(kT, 1e-10)   # (N,16)  ≤ 0
        term1 = float(np.einsum('k,kn,kn->', self.k_weights, df_dE, exp_nn**2))

        off = np.einsum('i,kin,kim->knm', dH_diag, ec.conj(), ec)
        off2 = np.abs(off)**2   # |matrix element|²

        dE_nm = ev[:, None, :] - ev[:, :, None]   # E_m − E_n,  (N,16,16)
        df_nm = f_all[:, :, None] - f_all[:, None, :]
        safe  = np.abs(dE_nm) > _FD_MASK_DE8
        ratio = np.where(safe, df_nm / np.where(safe, dE_nm, 1.0), df_dE[:, :, None])
        np.einsum('knn->kn', ratio)[:] = 0.0
        term2 = float(np.einsum('k,knm,knm->', self.k_weights, ratio, off2))
        d2F = (term1 + term2) / 2.0
        return grad, d2F

    def compute_hessian(self, M: float, Q: float, Delta: float, target_doping: float, mu: float, g_t: float, g_J: float, Delta_s_frac: float = 0.5, eps_M: float = 1e-3, eps_Q: float = 1e-4, eps_D: float = 1e-4, K_eff_for_free_energy: float = None, _ev0_cache: np.ndarray = None, V_s: float = None, V_d: float = None) -> Dict:
        """ 3×3 finite-difference Hessian of F(M,Q,Δ).

        V_s / V_d: full RPA pairing vertex for the condensation correction in
        compute_bdg_free_energy.  Must match the vertex used in the SCF gap
        equation so that ∂F/∂Δ = 0 at the converged point.
        """
        Delta_s_frac = float(np.clip(Delta_s_frac, 0.0, 1.0))
        Delta_d_frac = 1.0 - Delta_s_frac

        eps_M = max(1e-4, abs(M)     * 1e-3)
        eps_Q = max(5e-3 * self.p.lambda_hop, abs(Q) * 1e-3 * self.p.lambda_hop)  # floor ~6.4e-3 Å prevents noise-dominated H[1,1] at Q≈0
        eps_D = max(1e-5, abs(Delta) * 1e-3)

        def F(m, q, d, _ev_c=None):
            tb_x, tb_y = self.effective_hopping_anisotropic(q)
            ds = complex(d * Delta_s_frac)
            dd = complex(d * Delta_d_frac)
            return self.compute_bdg_free_energy(
                m, q, ds, dd, target_doping, mu, g_t * tb_x, g_t * tb_y, g_J,
                K_eff_for_free_energy=K_eff_for_free_energy,
                _ev_cache=_ev_c, V_s=V_s, V_d=V_d,
            )

        F0 = F(M, Q, Delta, _ev_c=_ev0_cache)   # skips eigh if cache provided
        H = np.zeros((3, 3))

        F_Mpp = F(M + eps_M, Q, Delta);  F_Mmm = F(M - eps_M, Q, Delta)
        F_Qpp = F(M, Q + eps_Q, Delta);  F_Qmm = F(M, Q - eps_Q, Delta)
        F_Dpp = F(M, Q, Delta + eps_D);  F_Dmm = F(M, Q, Delta - eps_D)

        H[0, 0] = (F_Mpp - 2*F0 + F_Mmm) / eps_M**2
        H[1, 1] = (F_Qpp - 2*F0 + F_Qmm) / eps_Q**2
        H[2, 2] = (F_Dpp - 2*F0 + F_Dmm) / eps_D**2

        F_MQ_pp = F(M+eps_M, Q+eps_Q, Delta); F_MQ_mm = F(M-eps_M, Q-eps_Q, Delta)
        F_MQ_pm = F(M+eps_M, Q-eps_Q, Delta); F_MQ_mp = F(M-eps_M, Q+eps_Q, Delta)
        H[0, 1] = H[1, 0] = (F_MQ_pp - F_MQ_pm - F_MQ_mp + F_MQ_mm) / (4*eps_M*eps_Q)

        F_MD_pp = F(M+eps_M, Q, Delta+eps_D); F_MD_mm = F(M-eps_M, Q, Delta-eps_D)
        F_MD_pm = F(M+eps_M, Q, Delta-eps_D); F_MD_mp = F(M-eps_M, Q, Delta+eps_D)
        H[0, 2] = H[2, 0] = (F_MD_pp - F_MD_pm - F_MD_mp + F_MD_mm) / (4*eps_M*eps_D)

        F_QD_pp = F(M, Q+eps_Q, Delta+eps_D); F_QD_mm = F(M, Q-eps_Q, Delta-eps_D)
        F_QD_pm = F(M, Q+eps_Q, Delta-eps_D); F_QD_mp = F(M, Q-eps_Q, Delta+eps_D)
        H[1, 2] = H[2, 1] = (F_QD_pp - F_QD_pm - F_QD_mp + F_QD_mm) / (4*eps_Q*eps_D)

        eigvals = np.linalg.eigvalsh(H)
        return {
            'H': H,
            'eigenvalues': eigvals,
            'is_minimum': bool(np.all(eigvals > -1e-6)),
            'min_curvature': float(eigvals[0])
        }

    def _mix(self, old, new, alpha=None):
        a = alpha if alpha is not None else self.p.mixing
        return (1 - a) * old + a * new

    def _compute_afm2band_susceptibilities(self, target_doping: float, M: float, Q: float, Delta_s: complex, Delta_d: complex) -> dict:
        """
        Compute 2-band AFM susceptibility tensor on the full k-grid.

        - Normal-state limit (Δ=0, Q=0) enforces χ_DQ = 0:
            • Orbital-mixing vertex τ_x (Γ6↔Γ7) is the only contribution.
            • In the 2-band AFM-folded model, ξ_avg = -μ ⇒ proj(k)·Δ_k·φ_c(k) is
            odd under k→k+Q, so BZ integral vanishes analytically.

        - SC-state (Δ≠0) induces finite χ_DQ via the orbital-mixing kernel,
        capturing SC response to JT distortion Q. The reduced 2-band model
        cannot compute the full χ_DQ, but the selection rule Δ=0 → χ_DQ=0
        is enforced.

        - χ_DD_s, χ_DD_d, χ_DD_sd computed from weighted k-grid sums with
        tanh(E/2kT) kernels.
        - χ_QQ = −∂²Ω/∂Q² evaluated at given Q, Δ_s, Δ_d, M, μ.
        The negative sign ensures χ_QQ > 0 for a stable metal.
        The g_JT factor is already included in the Hamiltonian.

        - N_eff gives effective density of states for SC instability estimates.
        """
        p     = self.p
        abs_d = max(abs(target_doping), 1e-6)
        g_t   = (2.0 * abs_d) / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d) ** 2
        kT    = max(p.kT, _KT_FLOOR)

        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        tx_eff = g_t * tx_bare
        ty_eff = g_t * ty_bare
        t_eff  = np.sqrt(0.5 * (tx_eff**2 + ty_eff**2))   # kept for BCS Tc estimate only

        f_d   = 1.0 - abs_d   # RMFT: (1-δ) spin-site fraction
        h_afm = g_J * f_d * (p.U_mf / 2.0 + p.Z * 2.0 * t_eff**2 / max(p.U, 1e-6)) * M / 2.0
        mu_n  = -2.0 * t_eff * (1.0 - 2.0 * abs_d)

        kx = self.k_points[:, 0]
        ky = self.k_points[:, 1]
        eps_k  = -2.0 * (tx_eff * np.cos(kx) + ty_eff * np.cos(ky)) - mu_n
        eps_kQ = -eps_k - 2.0 * mu_n
        xi_avg  = 0.5 * (eps_k + eps_kQ)
        xi_diff = 0.5 * (eps_k - eps_kQ)
        sq      = np.sqrt(xi_diff**2 + h_afm**2 + 1e-20)
        Ep = xi_avg + sq
        Em = xi_avg - sq

        def _th2E(E):
            a  = np.clip(E / (2.0 * kT), -100, 100)
            se = np.where(np.abs(E) > _KT_FLOOR, E, _KT_FLOOR)
            return np.tanh(a) / (2.0 * se)

        def _mdf(E):
            f_E = 1.0 / (1.0 + np.exp(np.clip(E / kT, -100, 100)))
            return f_E * (1.0 - f_E) / kT

        w_k  = self.k_weights
        pk   = _th2E(Ep) + _th2E(Em)
        proj = xi_diff / np.where(sq > 1e-9, sq, 1e-9)
        mix  = _th2E(Ep) * proj - _th2E(Em) * proj   # orbital-mixing kernel (proxy for chi_DQ at Δ≠0)

        phi_s = np.ones_like(kx)
        phi_d = np.cos(kx) - np.cos(ky)

        chi_DD_s  = float(np.dot(w_k, pk * phi_s**2))
        chi_DD_d  = float(np.dot(w_k, pk * phi_d**2))
        chi_DD_sd  = float(np.dot(w_k, pk * phi_s * phi_d))

        Delta_mag = np.sqrt(abs(Delta_s)**2 + abs(Delta_d)**2)
        if Delta_mag > _KT_FLOOR:
            chi_DQ_s = p.g_JT * float(np.dot(w_k, mix * phi_s))
            chi_dQ_d = p.g_JT * float(np.dot(w_k, mix * phi_d))
        else:
            # Normal state: chi_DQ = 0 enforced (selection rule + the analytic 2-band formula vanishes identically — two independent reasons).
            chi_DQ_s = 0.0
            chi_dQ_d = 0.0
        N_eff   = float(np.dot(w_k, _mdf(Ep) + _mdf(Em)))

        return {
            'chi_DD_s': chi_DD_s, 'chi_DD_d': chi_DD_d, 'chi_DD_sd': chi_DD_sd,
            'chi_DQ_s': chi_DQ_s, 'chi_DQ_d': chi_dQ_d, 'N_eff': N_eff,
            'E_plus': Ep, 'E_minus': Em, 'h_afm': float(h_afm),
            'mu_n': float(mu_n), 't_eff': float(t_eff),
        }

    def compute_Tc_by_gap_suppression(self, doping: float, sc_result: dict, T_min: float = 1e-4, T_max: float = 0.20, n_bracket: int = 12, n_bisect: int = 16, Delta_tol: float = 1e-5, use_free_energy: bool = False, verbose: bool = False) -> dict:
        """
        Find Tc by scanning temperature from the normal-state seed at each T.

        Algorithm:
        1. Always start each temperature from a *normal-state* seed (Δ ~ 0).
           → eliminates metastable SC continuation.
        2. Automatically bracket Tc by linear scan.
        3. Bisection refinement.
        4. Optional free-energy comparison (use_free_energy=True).
        """
        if sc_result is None or not sc_result.get("converged", False):
            return {'Tc': 0.0, 'Delta_at_Tc': 0.0, 'ratio_2D': 0.0, 'history': []}

        Delta_s0 = sc_result.get('Delta_s', 0.0)
        Delta_d0 = sc_result.get('Delta_d', 0.0)
        Delta0   = (Delta_s0**2 + Delta_d0**2) ** 0.5

        if Delta0 < Delta_tol:
            return {'Tc': 0.0, 'Delta_at_Tc': 0.0, 'ratio_2D': 0.0, 'history': []}
        history = []

        def _make_solver_at_T(T: float) -> 'RMFT_Solver':
            """Return a fully independent solver clone with kT=T.  Never modifies self."""
            s = copy.copy(self)
            s.p = copy.copy(self.p)
            s.p.kT = T
            s._K_bare = self._K_bare     # carry over immutable bare stiffness
            s._reset_transient_state()
            return s

        def gap_at_T(T: float) -> float:
            s = _make_solver_at_T(T)
            try:
                res = s.solve_self_consistent(
                    target_doping = doping,
                    initial_M     = self._estimate_M0(doping, sc_result),
                    initial_Q     = 1e-6,
                    initial_Delta = 1e-8,   # normal-state seed (below nucleation floor)
                    verbose       = False,
                )
                Ds = res['Delta_s']
                Dd = res['Delta_d']
                D  = (Ds**2 + Dd**2) ** 0.5

                if use_free_energy and D > Delta_tol:
                    res_normal = s.solve_self_consistent(
                        target_doping = doping,
                        initial_M     = self._estimate_M0(doping, sc_result),
                        initial_Q     = 1e-6,
                        initial_Delta = 1e-8,
                        verbose       = False,
                    )
                    if res_normal['F_bdg'] < res['F_bdg']:
                        return 0.0
                return D
            except Exception:
                return 0.0

        # 1. Bracketing stage
        T_vals   = list(np.geomspace(T_min, T_max, n_bracket + 1))
        SC_flags = []
        for T in T_vals:
            D = gap_at_T(T)
            history.append((T, D))
            SC_flags.append(D > Delta_tol)
            if verbose:
                print(f"[Tc-scan] T={T*1000:6.1f} meV  Δ={D:.6f}")

        bracket_found = False
        for i in range(len(T_vals) - 1):
            if SC_flags[i] and not SC_flags[i+1]:
                T_lo = T_vals[i]
                T_hi = T_vals[i+1]
                bracket_found = True
                break

        if not bracket_found:
            return {'Tc': 0.0, 'Delta_at_Tc': 0.0, 'ratio_2D': 0.0, 'history': history}

        # 2. Bisection refinement
        for _ in range(n_bisect):
            T_mid = 0.5 * (T_lo + T_hi)
            D_mid = gap_at_T(T_mid)
            history.append((T_mid, D_mid))
            if D_mid > Delta_tol:
                T_lo = T_mid
            else:
                T_hi = T_mid

        Tc    = 0.5 * (T_lo + T_hi)
        D_Tc  = gap_at_T(Tc)
        ratio = (2.0 * Delta0 / Tc) if Tc > 1e-8 else 0.0

        if verbose:
            print(f"[Tc] Tc={Tc*1000:.2f} meV  2Δ/kTc={ratio:.2f}")
        return {
            'Tc': Tc,
            'Delta_at_Tc': D_Tc,
            'ratio_2D': ratio,
            'history': history,
        }

    def _estimate_M0(self, target_doping: float, sc_result: dict = None) -> float:
        """Warm-start M estimate: use converged M if available, otherwise, analytical AFM suppression estimation from Gutzwiller-RVB scaling."""
        if sc_result is not None and sc_result.get('converged', False):
            return float(np.clip(sc_result['M'], 0.02, 0.45))

        abs_d     = max(abs(target_doping),1e-6)
        g_J       = 4.0/(1.0+abs_d)**2
        t_eff     = self.p.t0 * (2.0*abs_d)/(1.0+abs_d)
        bandwidth = 8.0 * max(t_eff,1e-6)
        N0        = 2.0 / bandwidth
        J_eff     = g_J * self.p.J_CT
        S         = np.clip(J_eff * N0,0.0,5.0)
        M_stoner  = np.clip(0.18*(S/max(S,1.0))*g_J/4.0,0.05,0.20)
        delta_c   = 0.23
        M_stoner  *= max(1.0 - abs_d/delta_c,0.0)
        M_prior   = np.clip(0.18 - 0.40*(target_doping-0.06),0.08,0.22)
        w         = np.clip(abs_d/0.20,0.0,1.0)
        M0        = (1-w)*M_stoner + w*M_prior
        return float(np.clip(M0, 0.02, 0.45))

    def compute_lambda_vs_T(self, doping: float, sc_result: dict) -> Dict:
        """
        Compute the linearised gap eigenvalue λ_max(T) across a temperature range.

        λ_max(T) is the largest eigenvalue of the linearised gap equation kernel at each T.
        It measures the strength of the pairing instability: λ_max(Tc) = 1 by definition.

        Diagnostics available from the curve:
          • Slope |dλ/dT|_Tc  — steeper → stronger coupling, less fluctuation-dominated
          • Strong-coupling signal: λ_max(T) deviates strongly from BCS tanh(T/Tc) form
          • Non-monotonic λ_max(T): indicates competing orders or fluctuation enhancement
          • Asymptotic λ_max(T→0): should saturate; if still rising → not fully converged k-grid
        """
        kT0 = max(self.p.kT, 1e-4)
        T_points = np.geomspace(kT0 * 0.25, kT0 * 4.0, 20)

        lam_arr   = np.zeros(len(T_points))
        sym_list  = []

        for i, T in enumerate(T_points):
            s_T = copy.copy(self)
            s_T.p = copy.copy(self.p)
            s_T.p.kT = T
            s_T._K_bare = self._K_bare
            s_T._reset_transient_state()
            try:
                res = s_T.solve_self_consistent(
                    target_doping  = doping,
                    initial_M      = self._estimate_M0(doping, sc_result),
                    initial_Q      = 1e-5,
                    initial_Delta  = 0.0,   # The SC gap is intentionally set to zero so λ_max(T) measures the linearised pairing instability, not the already-condensed state.
                    verbose        = False,
                )
                M       = res['M']
                Q       = res['Q']
                mu      = res['mu']
                tx      = res['tx']
                ty      = res['ty']
                g_J     = res['g_J']
                Delta_s = complex(res['Delta_s'])
                Delta_d = complex(res['Delta_d'])
                actual_doping = float(1.0 - res['density'])

                lin = s_T.solve_linearized_gap_equation(M, Q, Delta_s, Delta_d, doping, mu, tx, ty, g_J, actual_doping=actual_doping)
                lam_arr[i] = float(lin['lambda_max'])
                sym_list.append(lin['gap_symmetry'])
            except Exception:
                lam_arr[i] = 0.0
                sym_list.append('error')

        Tc_lambda  = 0.0
        slope_at_Tc = 0.0
        for i in range(len(T_points) - 1):
            l0, l1 = lam_arr[i], lam_arr[i + 1]
            t0, t1 = T_points[i], T_points[i + 1]
            if l0 >= 1.0 >= l1 and abs(l1 - l0) > 1e-10:
                frac = (1.0 - l0) / (l1 - l0)
                Tc_lambda   = float(t0 + frac * (t1 - t0))  # last T where λ_max crosses 1 from above (linear interpolation)
                slope_at_Tc = float((l1 - l0) / (t1 - t0))  # dλ/dT at the crossing (eV⁻¹); large → strong coupling
                break  # take the first (lowest-T) crossing 

        return {
            'T':            T_points,
            'lambda_max':   lam_arr,
            'gap_symmetry': sym_list,
            'Tc_lambda':    Tc_lambda,
            'slope_at_Tc':  slope_at_Tc,
        }

    def compute_gap_ratio(self, doping: float, sc_result: dict) -> Dict:
        """
        2Δ₀/kTc ratio — primary strong-coupling diagnostic.

        BCS weak-coupling value: 2Δ₀/kTc = 3.52.
        Values above 3.52 indicate enhanced pairing beyond BCS:
          3.52 – 4.5  : moderate strong coupling (phonon mechanism, λ ~ 1)
          4.5  – 6.0  : strong coupling (λ ~ 2, e.g. Pb, Hg)
          > 6.0        : very strong coupling or non-phononic (exotic) pairing mechanism
        In this SC-triggered JT model the expected ratio is > 3.52 because:
          (a) the JT feedback loop cooperatively enhances Δ beyond the linearised BCS value
          (b) the proximity to the AFM quantum critical region suppresses Tc relative to Δ₀
              via pair-breaking spin fluctuations → same Δ₀, lower Tc → higher ratio

        Returns
        -------
        dict with keys:
          'ratio_2D'     : float, 2Δ₀/kTc (0.0 if Tc or Δ₀ unavailable)
          'Delta_0'      : float, |Δ_s| + |Δ_d| from sc_result (eV)
          'Tc'           : float, critical temperature (eV)
          'Tc_K'         : float, Tc in Kelvin (eV / k_B)
          'coupling_regime': str, 'BCS-like' | 'strong' | 'very-strong' | 'exotic'
        """
        Delta_0 = (sc_result.get('Delta_s', 0.0) + sc_result.get('Delta_d', 0.0)
                   if sc_result else 0.0)
        if Delta_0 < 1e-6:
            return {'ratio_2D': 0.0, 'Delta_0': 0.0, 'Tc': 0.0, 'Tc_K': 0.0, 'coupling_regime': 'no SC'}

        tc_res = self.compute_Tc_by_gap_suppression(doping, sc_result)
        Tc     = float(tc_res['Tc'])
        if Tc < _KT_FLOOR:
            return {'ratio_2D': 0.0, 'Delta_0': float(Delta_0), 'Tc': 0.0, 'Tc_K': 0.0, 'coupling_regime': 'Tc not found'}

        ratio = 2.0 * Delta_0 / Tc
        Tc_K  = Tc / 8.617333e-5

        if ratio < 3.8:
            regime = 'BCS-like'
        elif ratio < 5.0:
            regime = 'strong'
        elif ratio < 7.0:
            regime = 'very-strong'
        else:
            regime = 'exotic / non-phononic'

        return {
            'ratio_2D':        float(ratio),
            'Delta_0':         float(Delta_0),
            'Tc':              float(Tc),
            'Tc_K':            float(Tc_K),
            'coupling_regime': regime,
        }

    def compute_coherence_length(self, doping: float, sc_result: dict) -> Dict:
        """
        BCS coherence length ξ = ℏv_F / (π Δ₀) estimated from the converged SCF state.

        Physical significance
        ---------------------
        ξ controls whether Cooper pairs are spatially coherent across multiple
        lattice sites.  If ξ < 2a the pairing is essentially on-site (pair-density
        wave / localised limit) and the BdG mean-field description breaks down.

        In the SC-triggered JT model, a too-small ξ also undermines the orbital
        selectivity argument: Γ₆ and Γ₇ bands need a coherence length large enough
        for the band-selective renormalisation of V_pair to develop.

        Algorithm
        ---------
        1. Compute the BdG spectrum at the converged (M, Q, Δ_s, Δ_d, μ).
        2. Identify quasi-particle states within ±_FS_SAMPLING·kT of the Fermi level
           (the "Fermi surface window").
        3. Estimate |v_F(k)| = |∇_k E_k| via finite differences on the k-grid.
        4. ξ = v_F_avg / (π · Δ₀)  [in units of the lattice constant a if a=1].

        Orbital-selective variant
        -------------------------
        Γ₆ and Γ₇ bands generally carry different effective masses (different
        bandwidth due to SOC+CF splitting and JT-induced tx≠ty).  The function
        reports ξ_Γ6 and ξ_Γ7 separately via the Nambu spinor weight of each
        quasi-particle state on the FS.

        Returns
        -------
        dict with:
          'xi_Gamma6'  : float  — Γ₆-weighted coherence length
          'xi_Gamma7'  : float  — Γ₇-weighted coherence length
          'xi_over_a'  : float  — ξ/a  (> 2 needed for BdG validity)
          'vF_avg'     : float  — average Fermi velocity (eV·Å)
          'Delta_0'    : float  — |Δ_s| + |Δ_d| (eV)
          'valid_BdG'  : bool   — True if ξ/a > 2
          'orbital_selective': bool — True if |ξ_Γ6 − ξ_Γ7| / ξ > 0.15
          'note'       : str
        """
        if sc_result is None or not sc_result.get('converged', False):
            return {'xi_Gamma6': 0.0, 'xi_Gamma7': 0.0, 'xi_over_a': 0.0, 'vF_avg': 0.0,
                    'Delta_0': 0.0, 'valid_BdG': False, 'orbital_selective': False, 'note': 'SCF not converged'}

        M       = float(sc_result.get('M',  0.1))
        Q       = float(sc_result.get('Q',  0.0))
        Delta_s = complex(sc_result.get('Delta_s', 0.0))
        Delta_d = complex(sc_result.get('Delta_d', 0.0))
        mu      = float(sc_result.get('mu',  0.0))
        tx      = float(sc_result.get('tx',  self.p.t0))
        ty      = float(sc_result.get('ty',  self.p.t0))
        g_J     = float(sc_result.get('g_J', 1.0))

        Delta_0 = abs(Delta_s) + abs(Delta_d)
        if Delta_0 < 1e-8:
            return {'xi_Gamma6': 0.0, 'xi_Gamma7': 0.0, 'xi_over_a': 0.0, 'vF_avg': 0.0,
                    'Delta_0': 0.0, 'valid_BdG': False, 'orbital_selective': False, 'note': 'no SC gap'}

        vbdg = self._get_vbdg()
        ev, ec = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d,
                                doping, mu, tx, ty, g_J, out=vbdg._H_stack))

        # Fermi-surface window
        near_fs = np.any(np.abs(ev) < _FS_SAMPLING * self.p.kT, axis=1)
        fs_idx  = np.where(near_fs)[0]
        if len(fs_idx) == 0:
            fs_idx = np.arange(min(64, self.N_k))

        kpts_fs = self.k_points[fs_idx]   # (N_fs, 2) 
        dk = 2.0 * np.pi / self.p.nk      # grid spacing

        # Finite-difference Fermi velocity for the two lowest positive-energy BdG bands
        # (these are the particle-like quasi-particles closest to the FS).
        vF_list, w6_list, w7_list = [], [], []

        for ki, k in enumerate(kpts_fs):
            # Evaluate spectrum at k±dk in x and y
            k_px = np.array([[k[0] + dk, k[1]]])
            k_mx = np.array([[k[0] - dk, k[1]]])
            k_py = np.array([[k[0], k[1] + dk]])
            k_my = np.array([[k[0], k[1] - dk]])

            def _min_pos_ev(kpt):
                ev_loc = np.linalg.eigvalsh(
                    vbdg._build_H_stack(kpt, M, Q, Delta_s, Delta_d,
                                        doping, mu, tx, ty, g_J))
                pos = ev_loc[ev_loc > 0]
                return float(pos[0]) if len(pos) > 0 else 0.0

            dEx = (_min_pos_ev(k_px) - _min_pos_ev(k_mx)) / (2.0 * dk)
            dEy = (_min_pos_ev(k_py) - _min_pos_ev(k_my)) / (2.0 * dk)
            vF  = max(np.sqrt(dEx**2 + dEy**2), _VF_FLOOR)
            vF_list.append(vF)

            # Orbital weight: Γ₆ = rows 0–1, Γ₇ = rows 2–3 in particle-A spinor
            # (averaged over quasi-particle states near FS)
            ec_k = ec[fs_idx[ki]]   # (16, 16)
            w6 = float(np.sum(np.abs(ec_k[0:2, :])**2))
            w7 = float(np.sum(np.abs(ec_k[2:4, :])**2))
            norm67 = max(w6 + w7, 1e-12)
            w6_list.append(w6 / norm67)
            w7_list.append(w7 / norm67)

        vF_arr  = np.array(vF_list)
        w6_arr  = np.array(w6_list)
        w7_arr  = np.array(w7_list)

        vF_avg  = float(np.mean(vF_arr))
        vF_G6   = float(np.average(vF_arr, weights=w6_arr + 1e-12))
        vF_G7   = float(np.average(vF_arr, weights=w7_arr + 1e-12))

        # ξ [Å]: k-grid uses dimensionless ka units so vF=dE/dk is in eV/ka, not eV·Å
        xi_over_a = vF_avg / (np.pi * Delta_0)            # dimensionless (lattice units)
        xi        = xi_over_a * self.p.a                         # Å
        xi_G6     = vF_G6 / (np.pi * Delta_0) * self.p.a         # Å
        xi_G7     = vF_G7 / (np.pi * Delta_0) * self.p.a         # Å

        valid_BdG = xi_over_a > 2.0   # ξ > 2a: standard BdG validity (correct after unit fix)
        orbital_selective = abs(xi_G6 - xi_G7) / max(xi, 1e-12) > 0.15

        if not valid_BdG:
            note = f"⚠ ξ/a={xi_over_a:.2f} < 2 — BdG validity marginal; Cooper pairs not coherent across lattice"
        elif orbital_selective:
            note = f"✓ ξ/a={xi_over_a:.2f}  ORBITAL-SELECTIVE: ξ_Γ6={xi_G6/self.p.a:.2f}a  ξ_Γ7={xi_G7/self.p.a:.2f}a — JT-driven band splitting enhances selectivity"
        else:
            note = f"✓ ξ/a={xi_over_a:.2f}  orbitally uniform (|ξ_Γ6−ξ_Γ7|/ξ < 15%)"

        return {
            'xi_Gamma6':        float(xi_G6),
            'xi_Gamma7':        float(xi_G7),
            'xi_over_a':        float(xi_over_a),
            'vF_avg':           float(vF_avg),
            'vF_Gamma6':        float(vF_G6),
            'vF_Gamma7':        float(vF_G7),
            'Delta_0':          float(Delta_0),
            'valid_BdG':        valid_BdG,
            'orbital_selective': orbital_selective,
            'note':             note,
        }

    def _build_G3_matrix(self, chi: dict, gVs: float, gVd: float, K_eff: float) -> tuple:
        """
        Assemble the 3×3 SC–JT instability matrix from a susceptibility dict.

        Required keys : chi_DD_s, chi_DD_d, chi_QQ
        Optional keys : chi_DD_sd (default 0), chi_DQ_s (0), chi_DQ_d (0)

        Derivation: F = F_s + F_d + F_Q + F_sQ + F_dQ where
            F_s  = Δ_s²/(gVs) − χ_ss·Δ_s²
            F_d  = Δ_d²/(gVd) − χ_dd·Δ_d²
            F_Q  = K·Q²/2 − χ_QQ·Q²/2          (χ_QQ already g²-weighted)
            F_sQ = −χ_sQ·Δ_s·Q                  (χ_sQ already g_JT-weighted)
            F_dQ = −χ_dQ·Δ_d·Q

        G = | 1 − gVs·χ_ss       −√(gVs·gVd)·χ_sd   −c_s·χ_sQ |
            | −√(gVs·gVd)·χ_sd   1 − gVd·χ_dd        −c_d·χ_dQ |
            | −c_s·χ_sQ           −c_d·χ_dQ            1 − K⁻¹·χ_QQ |

        This is the standard Schur-complement form in dimensionless units where each axis has been rescaled by the geometric mean of its diagonal stiffnesses.
        """
        G3 = np.zeros((3, 3))
        Kinv = 1.0 / max(K_eff, 1e-9)

        G3[0, 0] = 1.0 - gVs * chi['chi_DD_s']
        G3[1, 1] = 1.0 - gVd * chi['chi_DD_d']
        G3[2, 2] = 1.0 - chi['chi_QQ'] * Kinv
        G3[0, 1] = G3[1, 0] = -np.sqrt(max(gVs * gVd, 0.0)) * chi.get('chi_DD_sd', 0.0)

        c_s = np.sqrt(max(gVs * Kinv, 0.0))
        c_d = np.sqrt(max(gVd * Kinv, 0.0))
        G3[0, 2] = G3[2, 0] = -c_s * chi.get('chi_DQ_s', 0.0)
        G3[1, 2] = G3[2, 1] = -c_d * chi.get('chi_DQ_d', 0.0)

        eigs3, evecs3 = np.linalg.eigh(G3)
        lam_min = float(eigs3[0])
        evec_min = evecs3[:, 0]

        if lam_min < 0.5:
            nm = np.abs(evec_min)
            ws, wd, wq = nm
            sc_weight = ws + wd

            if lam_min < 0:
                if wq > 0.6 and sc_weight < 0.3:
                    instab = 'pure JT (spontaneous risk)'
                elif wq > 0.4 and sc_weight > 0.3:
                    instab = 'SC-triggered JT'
                elif ws > 0.6:
                    instab = 's pairing'
                elif wd > 0.6:
                    instab = 'd pairing'
                else:
                    instab = 'mixed SC+JT'
            else:
                mc = int(np.argmax(nm))
                instab = f"near-critical ({'Δ_s' if mc==0 else 'Δ_d' if mc==1 else 'Q'}-dominant)"
        else:
            instab = 'stable'
        return G3, eigs3, lam_min, instab, evec_min

    def compute_G_instability(self, target_doping: float, M: float, compute_dlambda: bool = True) -> dict:
        """
        Compute normal-state (Δ=0) collective instability matrix and diagnostics.

        λ_eff = N_eff · V_eff measures proximity to the QCP of the dominant
        channel (SC or JT) from the normal state. This is NOT a pairing eigenvalue.

        Correct usage:
        - BO Phase 1 (DE scout): enforces H1/H2/H3 hard constraints from the G3 outputs;
          λ_eff used as soft-constraint S2 signal, not as a hard SCF gatekeeper.
        - BO prior: bias sampling toward QCP without hard rejection.
        - G22 > 0: normal-state JT stable (prerequisite for SC-triggered mechanism).

        Notes:
        - K_eff incorporates exchange-corrected normal-state stiffness, used
            consistently in SC vertices (gVs, gVd) and the JT diagonal
            (G3[2,2] = 1 − χ_QQ/K_eff), ensuring Schur complement and off-diagonal
            SC–JT couplings are built from a single reference stiffness.

        - Dominant SC channel is determined from the eigenvector of G3, not
            from the diagonal alone, to account for off-diagonal SC–JT mixing.

        - Tc_est via λ_eff is only an estimate; authoritative Tc comes from
            the SCF-converged gap and Hessian.

        - SC-triggered JT coupling:
            • χ_QQ^normal (Δ = 0) gives the normal-state Q stiffness.
            • χ_QQ^SC (Δ ≠ 0, adiabatic/linear approximation) accounts for some
                SC-induced Q-dependence, but underestimates full coupling.
            • The full Hessian ∂²F/∂Q²|_{Δ≠0} includes Δ(Q) and M(Q) feedback,
                capturing implicit Q-dependence and SC-triggered JT effects.
                This can make ∂²F/∂Q²|_{Δ≠0} larger than K − χ_QQ^SC, or even
                positive while χ_QQ^SC > χ_QQ^normal.

        - Using the full Hessian ensures a faithful test of SC–JT coupling while correctly reflecting SC–JT interplay.
        """
        abs_d = max(abs(target_doping), 1e-6)
        g_t, g_J, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)

        # chi must be computed first: provides mu_n for the rigidity BdG (mu=0 is wrong for δ≠0)
        chi = self.get_susceptibilities_fast(target_doping, M, Q=0.0, Delta_s=0.0, Delta_d=0.0)
        t_eff = chi['t_eff']
        N_eff = chi['N_eff']

        rigidity = self.compute_JT_rigidity_from_exchange(M, 0.0, chi['mu_n'], g_J, target_doping, g_t)
        K_eff_here  = max(rigidity['K_eff'], 1e-9)

        V_base = self.p.g_JT**2 / K_eff_here
        gVs = g_Delta_s * V_base
        gVd = g_Delta_d * V_base

        G3, eigs3, lam_min, instab_dir, evec_min = self._build_G3_matrix(chi, gVs, gVd, K_eff_here)
        det_G = float(np.linalg.det(G3))    

        # Dominant channel from the *eigenvector*, not from diagonal alone.
        weights = np.abs(evec_min)
        ws, wd, wq = weights
        if wd > ws:
            dominant   = 'd'
            G11, G12   = G3[1, 1], G3[1, 2]
            chi_DD_dom = chi['chi_DD_d']
            chi_DQ_dom = chi['chi_DQ_d']
            V_dom      = gVd
        else:
            dominant   = 's'
            G11, G12   = G3[0, 0], G3[0, 2]
            chi_DD_dom = chi['chi_DD_s']
            chi_DQ_dom = chi['chi_DQ_s']
            V_dom      = gVs

        if wq > ws and wq > wd:
            dominant   = 'JT'
        
        # V_eff Schur complement:
        G22 = G3[2, 2]
        if dominant != 'JT' and G22 > _KT_FLOOR:
            V_eff = V_dom + (V_dom / K_eff_here * chi_DQ_dom**2) / G22
        else:
            V_eff = V_dom   # spontaneous-JT regime: no SC-triggered boost
        lambda_eff = N_eff * V_eff
        Tc_est  = float(1.13 * t_eff * np.exp(-1.0 / lambda_eff)) if lambda_eff > 1e-3 else 0.0
        d2F_Q_normal = K_eff_here - chi['chi_QQ']   # normal-state Q-curvature; exact at Δ=0

        J_eff = self.effective_superexchange(g_J, self.p.t0, self.p.t0, target_doping)

        # Moriya-damped chi_DD_s for H2 / jchi_gate — consistent with get_susceptibilities_normal.
        _t_eff_gi       = float(self.p.t0 * (2.0 * abs(target_doping)) / (1.0 + abs(target_doping) + 1e-9))
        _alpha_M_gi     = _moriya_alpha(target_doping, _t_eff_gi, J_eff)
        _Gamma_M_gi     = _alpha_M_gi * max(J_eff, 1e-9) * _t_eff_gi
        _chi_DD_s_raw   = chi['chi_DD_s']
        chi_DD_s_moriya = _chi_DD_s_raw / (1.0 + _Gamma_M_gi * max(_chi_DD_s_raw, 0.0))

        # ── ∂λ_pair/∂Q diagnostic: 5-point quadratic polynomial fit λ(Q) around Q=0 ──
        # Measures whether JT distortion renormalises the pairing vertex upward.
        # Evaluated at Δ=0 (normal-state linearised gap equation) so it is honest:
        # solve_linearized_gap_equation uses normal-state chi0 internally.
        dlambda_dQ = float('nan')
        if compute_dlambda:
            try:
                _dQ = max(1e-3, 0.01 * self.p.lambda_hop)

                def _lambda_at_Q(Qv: float) -> float:
                    tx_b, ty_b = self.effective_hopping_anisotropic(Qv)
                    lin = self.solve_linearized_gap_equation(
                        M, Qv, 0.0+0j, 0.0+0j,
                        target_doping, chi['mu_n'],
                        g_t * tx_b, g_t * ty_b, g_J,
                        actual_doping=target_doping)
                    return lin['lambda_max']

                _Q_offsets = np.array([-2*_dQ, -_dQ, 0.0, _dQ, 2*_dQ])
                _lam_vals  = np.array([_lambda_at_Q(q) for q in _Q_offsets])

                # np.polyfit returns [c, b, a]; b = coeffs[1] is ∂λ/∂Q|_{Q=0}
                _coeffs    = np.polyfit(_Q_offsets, _lam_vals, 2)
                dlambda_dQ = float(_coeffs[1])
            except Exception as _dl_err:
                _scf_log("G-INST", f"∂λ_pair/∂Q diagnostic failed: {_dl_err}")

        return {
            'chi_DD_dom':      chi_DD_dom,
            'chi_DD_s':        chi['chi_DD_s'],
            'chi_DD_d':        chi['chi_DD_d'],
            'chi_DD_sd':       chi['chi_DD_sd'],
            'chi_QQ':          chi['chi_QQ'],
            'chi_DQ_dom':      chi_DQ_dom,
            'chi_DQ_s':        chi['chi_DQ_s'],
            'chi_DQ_d':        chi['chi_DQ_d'],
            'dominant':        dominant,
            'instab_dir':      instab_dir,
            'evec_min':        evec_min,
            'h_afm':           chi['h_afm'],
            'E_plus_mean':     float(np.mean(chi['E_plus'])),
            'N_eff':           float(N_eff),
            'G3':              G3,
            'eigs3':           eigs3,
            'lambda_min':      lam_min,
            'det_G':           det_G,
            'V_eff':           float(V_eff),
            'lambda_eff':      float(lambda_eff),
            'Tc_estimate':     Tc_est,
            'comm_norm':       float(rigidity['comm_norm']),
            'blocking_ratio':  float(rigidity['blocking_ratio']),
            'K_eff':           float(rigidity['K_eff']),
            'd2F_Q_normal':    float(d2F_Q_normal),   # ∂²F/∂Q²|_{Δ=0} — hard exclusion if < 0
            'sc_triggered_jt': False,                  # set True by SCF when Q≠0 and Δ≠0 converge
            'dlambda_pair_dQ': dlambda_dQ,             # ∂λ_pair/∂Q — positive: JT renormalises V_pair upward
            'G11': G11, 'G22': G22, 'G12': G12, 'K_spont_blocked': G22 > 0.0,
            'g_t':             float(g_t),
            'g_J':             float(g_J),
            'J_eff':           float(J_eff),
            'chi_DD_s_moriya': float(chi_DD_s_moriya),
            'mu_n':            float(chi['mu_n']),
        }

class VectorizedBdG:
    def __init__(self, solver: 'RMFT_Solver'):
        self.solver    = solver
        self._kpts     = solver.k_points        # (N_k, 2)    — SCF / gap grid (endpoint=False)
        self._kpts_ev  = solver.k_points_even   # (N_k_even, 2) — chi0 / commensurate grid
        self._N_k      = solver.N_k
        self._N_k_ev   = solver.N_k_even
        self._H_stack    = np.zeros((self._N_k,    16, 16), dtype=complex)  # SCF grid buffer
        self._H_stack_ev = np.zeros((self._N_k_ev, 16, 16), dtype=complex)  # chi0 grid buffer

    def _build_H_stack(self, kpts: np.ndarray, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, O_expectation: Optional[np.ndarray] = None, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build the (N, 16, 16) BdG Hamiltonian stack for an arbitrary kpts array.

        16×16 Nambu basis: [Part_A(0:4), Part_B(4:8), Hole_A(8:12), Hole_B(12:16)],
        each sub-block in [6↑, 6↓, 7↑, 7↓] orbital basis:

        ┌──────────────┬──────────────────────────────┐
        │  H_A   T_AB  │  D_s        D_d              │  Part_A, Part_B
        │  T_AB† H_B   │  D_d        D_s              │
        ├──────────────┼──────────────────────────────┤
        │  D_s†  D_d†  │  −H_A*     −T_AB†            │  Hole_A, Hole_B
        │  D_d†  D_s†  │  −T_AB*    −H_B*             │
        └──────────────┴──────────────────────────────┘

        D_s (on-site, channel s):  Δ_s · [6↑↔7↓ singlet, φ=1]
        D_d (inter-site, channel d): Δ_d · φ(k) · [6↑↔7↓ singlet, φ(k)=cos kx−cos ky]
        F_AA = u_A·v_A* → feeds Δ_s gap eq.   F_AB = u_A·v_B* → feeds Δ_d gap eq.

        Parameters
        ----------
        kpts : (N, 2) array of k-points.  May be self._kpts (full grid) or any
               arbitrary sub-grid (e.g. k+Q shifted for chi0 routines).
        out  : optional pre-allocated (N, 16, 16) complex buffer.  If provided
               the result is written in-place (no heap allocation) and the same
               array is returned.  Pass self._H_stack here for the hot SCF path
               to avoid per-iteration GC pressure.
        """
        N = len(kpts)
        # Allocate / reuse output
        if out is None:
            H = np.zeros((N, 16, 16), dtype=complex)
        else:
            H = out
            H[:] = 0.0 + 0.0j


        # ---- Local Hamiltonians (A/B sublattice) ----
        local_kwargs = dict(tx=tx, ty=ty)
        if O_expectation is not None:
            local_kwargs["O_expectation"] = O_expectation

        H_A = self.solver.build_local_hamiltonian_for_bdg(
            +1.0, M, Q, mu, g_J, target_doping, **local_kwargs
        )
        H_B = self.solver.build_local_hamiltonian_for_bdg(
            -1.0, M, Q, mu, g_J, target_doping, **local_kwargs
        )

        # ---- On-site singlet pairing (Delta_s) ----
        D_on = np.zeros((4, 4), dtype=complex)
        D_on[0, 3] = Delta_s
        D_on[1, 2] = -Delta_s
        D_dag = np.conj(D_on).T

        # Particle blocks
        H[:, 0:4,   0:4  ] = H_A
        H[:, 4:8,   4:8  ] = H_B

        # Hole blocks (−H*)
        H[:, 8:12,  8:12 ] = -np.conj(H_A)
        H[:, 12:16, 12:16] = -np.conj(H_B)

        # Particle–hole off-diagonal (Delta_s)
        H[:, 0:4,   8:12 ] = D_on
        H[:, 4:8,  12:16 ] = D_on
        H[:, 8:12,  0:4  ] = D_dag
        H[:, 12:16, 4:8  ] = D_dag

        # ---- Inter-sublattice hopping gamma_k ----
        kx = kpts[:, 0]
        ky = kpts[:, 1]
        gamma_k = -2.0 * (tx * np.cos(kx) + ty * np.cos(ky))

        di = np.arange(4)

        # Particle sector
        H[:, di,      di + 4 ] = gamma_k[:, None]
        H[:, di + 4,  di     ] = gamma_k[:, None]

        # Hole sector (−γ*)
        H[:, di + 8,  di + 12] = -gamma_k[:, None]
        H[:, di + 12, di + 8 ] = -gamma_k[:, None]

        # ---- d-wave pairing Delta_d ----
        if N == self._N_k:
            phi_d_k = self.solver.phi_k
        elif N == self._N_k_ev:
            phi_d_k = self.solver.phi_k_even
        else:
            phi_d_k = np.cos(kx) - np.cos(ky)

        phi = phi_d_k * Delta_d

        # Particle ↔ Hole couplings (singlet structure)
        H[:, 0,  15] +=  phi
        H[:, 1,  14] -=  phi
        H[:, 4,  11] +=  phi
        H[:, 5,  10] -=  phi

        phi_c = np.conj(phi)
        H[:, 15,  0] +=  phi_c
        H[:, 14,  1] -=  phi_c
        H[:, 11,  4] +=  phi_c
        H[:, 10,  5] -=  phi_c

        # Enforce exact Hermiticity: prevents eigh convergence failures from floating-point asymmetry accumulated across the block assembly above.
        H[:] = 0.5 * (H + H.conj().transpose(0, 2, 1))
        return H

    @staticmethod
    def _get_nambu_spinors(ec: np.ndarray):
        """
        Slice BdG eigenvector array into Nambu spinors per sublattice.

        Layout (matches _build_H_stack):
          rows  0– 3 : particle A (Γ₆↑, Γ₆↓, Γ₇↑, Γ₇↓)
          rows  4– 7 : particle B
          rows  8–11 : hole A
          rows 12–15 : hole B

        Returns uA, uB, vA, vB — each (N_k, 4, 16).
        """
        return ec[:, 0:4, :], ec[:, 4:8, :], ec[:, 8:12, :], ec[:, 12:16, :]

    @staticmethod
    def _compute_densities(uA, uB, vA, vB, f, fbar):
        """
        Per-k sublattice densities: n_X(k) = Σ_{orb,n}[|u_X|²f + |v_X|²f̄].

        Returns dens_A, dens_B — each (N_k,).
        """
        dens_A = np.sum(np.abs(uA)**2 * f[:, None, :] + np.abs(vA)**2 * fbar[:, None, :], axis=(1, 2))
        dens_B = np.sum(np.abs(uB)**2 * f[:, None, :] + np.abs(vB)**2 * fbar[:, None, :], axis=(1, 2))
        return dens_A, dens_B

    def compute_observables_vectorized(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, _bdg_cache: tuple = None) -> Dict:
        """ Vectorised observables: M, Q (τ_x), density, Pair_s, Pair_d. """
        solver = self.solver
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = np.linalg.eigh(self._build_H_stack(self._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=self._H_stack))

        # Fermi-Dirac factors: (N_k, 16)
        arg  = np.clip(ev / solver.p.kT, -100, 100)
        f    = 1.0 / (1.0 + np.exp(arg))
        fbar = 1.0 - f
        f12  = 1.0 - 2.0 * f

        uA, uB, vA, vB = self._get_nambu_spinors(ec)
        dens_A, dens_B = self._compute_densities(uA, uB, vA, vB, f, fbar)

        mag_A = np.sum((np.abs(uA)**2 * solver.sz_op[None, :, None]) * f[:, None, :]
                     + (np.abs(vA)**2 * solver.sz_op[None, :, None]) * fbar[:, None, :], axis=(1, 2))
        mag_B = np.sum((np.abs(uB)**2 * solver.sz_op[None, :, None]) * f[:, None, :]
                     + (np.abs(vB)**2 * solver.sz_op[None, :, None]) * fbar[:, None, :], axis=(1, 2))

        # Quadrupole τ_x = 2 Re(u₀*u₂ + u₁*u₃)  — orbital mixing indicator
        tau_u_A = 2.0 * np.real(uA[:, 0, :] * np.conj(uA[:, 2, :])
                               + uA[:, 1, :] * np.conj(uA[:, 3, :]))   # (N_k, 16)
        tau_v_A = 2.0 * np.real(vA[:, 0, :] * np.conj(vA[:, 2, :])
                               + vA[:, 1, :] * np.conj(vA[:, 3, :]))
        tau_u_B = 2.0 * np.real(uB[:, 0, :] * np.conj(uB[:, 2, :])
                               + uB[:, 1, :] * np.conj(uB[:, 3, :]))
        tau_v_B = 2.0 * np.real(vB[:, 0, :] * np.conj(vB[:, 2, :])
                               + vB[:, 1, :] * np.conj(vB[:, 3, :]))
        quad_A  = np.sum(tau_u_A * f + tau_v_A * fbar, axis=1)
        quad_B  = np.sum(tau_u_B * f + tau_v_B * fbar, axis=1)

        # On-site pairing amplitude (channel s): u_A[6↑]·v_A[7↓]* − u_A[6↓]·v_A[7↑]*
        pair_s_A = uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
        pair_s_B = uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])
        pair_s   = np.sum((pair_s_A + pair_s_B) * f12, axis=1)   # (N_k,)

        # Inter-site pairing amplitude (channel d)
        pair_AB = uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
        pair_BA = uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])
        pair_d  = np.sum(0.5 * (pair_AB + pair_BA) * f12, axis=1)   # (N_k,)

        # k-weighted averages
        w = self.solver.k_weights   # (N_k,)
        n_avg  = float(np.dot(w, dens_A + dens_B))  / 4.0
        M_stag = float(np.dot(w, mag_A  - mag_B))   / 4.0
        Q_unif = float(np.dot(w, quad_A + quad_B))  / 4.0
        Pair_s = complex(np.dot(w, pair_s))         / 4.0
        Pair_d = complex(np.dot(w, pair_d))         / 4.0
        return {'n': n_avg, 'M': M_stag, 'Q': Q_unif, 'Pair_s': Pair_s, 'Pair_d': Pair_d, 'Pair': Pair_s + Pair_d}

    def compute_gap_eq_vectorized(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, g_Delta_s: float, g_Delta_d: float, _bdg_cache: tuple = None, _vertex_cache: dict = None) -> Tuple[float, float, dict]:
        """
        Gap equation with q-dependent RPA pairing vertex (spin + JT channels).

        The BdG amplitudes (pair_s_k, pair_d_k) are evaluated at the current (Δ_s, Δ_d)
        to give the anomalous Green function F_AA / F_AB at the converged SC state.

        The RPA pairing vertex V(q) is always built from normal-state (Δ=0) susceptibilities:
          chi0 / chi_QQ_normal_v at Δ=0 — feeding back Δ≠0 susceptibilities into the
          interaction that caused Δ would be self-referential double-counting.

        Returns (Δ_s_new, Δ_d_new, _vertex_cache).
        """
        solver = self.solver

        # _gap_amplitude propagated to get_susceptibilities_normal via self.solver._gap_amplitude:
        # χ_SQ clamp threshold is relaxed when Δ≠0 so genuine cross-channel signal is not zeroed.
        solver._gap_amplitude = abs(Delta_s) + abs(Delta_d)

        # --- BdG amplitudes on the full k-grid ---
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = np.linalg.eigh(self._build_H_stack(self._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=self._H_stack))

        arg = np.clip(ev / solver.p.kT, -100, 100)
        f12 = 1.0 - 2.0 / (1.0 + np.exp(arg))   # tanh(E/2kT); (N_k, 16)

        uA, uB, vA, vB = self._get_nambu_spinors(ec)

        # Full-BZ pair amplitudes (consistent with compute_observables_vectorized Pair_s/d)
        pair_s_k = np.sum(
            (uA[:, 0, :]*np.conj(vA[:, 3, :]) - uA[:, 1, :]*np.conj(vA[:, 2, :])
           + uB[:, 0, :]*np.conj(vB[:, 3, :]) - uB[:, 1, :]*np.conj(vB[:, 2, :])) * f12,
            axis=1)
        pair_d_k = np.sum(
            0.5*(uA[:, 0, :]*np.conj(vB[:, 3, :]) - uA[:, 1, :]*np.conj(vB[:, 2, :])
               + uB[:, 0, :]*np.conj(vA[:, 3, :]) - uB[:, 1, :]*np.conj(vA[:, 2, :])) * f12,
            axis=1)

        # Full-BZ integrals for the gap equation RHS.
        # k_weights are uniform 1/N (Σw_k = 1). /4.0 corrects for the 16-dim BdG space: A/B sublattice × particle-hole doubling.
        F_AA_BZ = float(np.real(np.dot(solver.k_weights, pair_s_k))) / 4.0  # on-site s-channel anomalous amplitude (phi_s = 1, no k-factor needed).
        F_AB_BZ = float(np.real(np.dot(solver.k_weights, pair_d_k))) / 4.0  # inter-site d-channel anomalous amplitude. The d-wave symmetry projection is handled entirely on the VERTEX side (V_d_scalar = phi_d @ V_mat @ phi_d / phi2).

        # --- Fermi-surface sampling — used ONLY for the q-dependent vertex V(k-k') ---
        near_fs = np.any(np.abs(ev) < _FS_SAMPLING * solver.p.kT, axis=1)
        fs_idx  = np.where(near_fs)[0]
        if len(fs_idx) == 0:
            fs_idx = np.arange(min(_FS_N_VERTEX, solver.N_k))
        fs_idx  = fs_idx[:_FS_N_VERTEX]
        fs_pts  = solver.k_points[fs_idx]
        N_fs    = len(fs_pts)
        phi_d   = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])

        # --- Vertex cache invalidation ---
        _cache_M          = _vertex_cache.get('M',        float('nan')) if _vertex_cache else float('nan')
        _cache_Q          = _vertex_cache.get('Q',        float('nan')) if _vertex_cache else float('nan')
        _cache_Delta      = _vertex_cache.get('Delta',    float('nan')) if _vertex_cache else float('nan')
        _cache_j_renorm   = _vertex_cache.get('j_renorm', float('nan')) if _vertex_cache else float('nan')
        _cache_fs         = _vertex_cache.get('fs_idx',   None)         if _vertex_cache else None
        _cache_chi_normal = _vertex_cache.get('chi_QQ_from_normal', False) if _vertex_cache else False
        Delta_eff  = abs(Delta_s) + abs(Delta_d)
        _delta_rel = abs(Delta_eff - _cache_Delta) / max(abs(_cache_Delta), 1e-6) if _vertex_cache else float('inf')

        # Q enters the vertex through tx(Q), ty(Q) → J_eff(Q) → chi_SS(Q) → V_spin(Q).
        _Q_thr = max(_Q_THR_REL * solver.p.lambda_hop, 1e-4)

        _j_renorm_now   = solver._cluster_j_renorm
        _j_renorm_stale = abs(_j_renorm_now - _cache_j_renorm) > 0.05

        _vertex_stale = (
            _vertex_cache is None
            or not _cache_chi_normal       # cache built with Δ≠0 chi_QQ → must rebuild
            or abs(M - _cache_M) > _M_THR_REL
            or abs(Q - _cache_Q) > _Q_thr
            or abs(Delta_eff - _cache_Delta) > _DELTA_THR_ABS
            or _delta_rel > _DELTA_THR_REL
            or _j_renorm_stale
            or _cache_fs is None
            or len(_cache_fs) != len(fs_idx)
            or not np.array_equal(_cache_fs, fs_idx)
        )

        if _vertex_stale:
            V_JT = solver.p.g_JT**2 / max(solver._K_bare, 1e-9)   # [eV]

            tx_bare_v, ty_bare_v = solver.effective_hopping_anisotropic(Q)
            J_eff_v_x = solver.effective_superexchange(g_J, tx_bare_v, ty_bare_v, target_doping, direction='x')
            J_eff_v_y = solver.effective_superexchange(g_J, tx_bare_v, ty_bare_v, target_doping, direction='y')

            # Reuse solver-level Δ=0 BdG eigenvector cache across the q-loop.
            E_k_cache_normal = solver._get_chi0_norm_cache(
                M, Q, mu, tx, ty, g_J, self, target_doping=target_doping)

            chi_QQ_normal_v = solver._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)

            _vp_v = {'J_eff_x': J_eff_v_x, 'J_eff_y': J_eff_v_y, 'V_JT': V_JT, 'chi_QQ_normal': chi_QQ_normal_v, 'return_det': True}

            # s-channel: single q=0 call — chi0 + vertex in one Lindhard sum.
            _actual_dop_v = _vertex_cache.get('actual_doping', target_doping) if _vertex_cache else target_doping
            _n_sus_q0_v = solver.get_susceptibilities_normal(
                q=np.zeros(2), M=M, Q=Q,
                target_doping=target_doping, mu=mu, tx=tx, ty=ty, g_J=g_J,
                _E_k_cache=E_k_cache_normal,
                vertex_params=_vp_v,
                _chi_QQ_cache=chi_QQ_normal_v,
                actual_doping=_actual_dop_v)
            V_s_scalar = _n_sus_q0_v['V_full']
            _det_q0    = _n_sus_q0_v['rpa_det']

            # d-channel: q-dependent vertex (only if Delta_d has nucleated)
            if abs(Delta_d) > 1e-4:
                iu, ju   = np.triu_indices(N_fs)
                q_raw    = fs_pts[iu] - fs_pts[ju]
                q_arr    = (q_raw + np.pi) % (2.0 * np.pi) - np.pi

                q_int              = np.round(q_arr * 1e5).astype(np.int64)
                u_q_int, inv_idx   = np.unique(q_int, axis=0, return_inverse=True)
                u_q_vecs           = u_q_int.astype(np.float64) / 1e5

                # k+q via shift_table (zero eigh cost).
                V_rpa = np.empty(len(u_q_vecs), dtype=float)
                for ui, q_u in enumerate(u_q_vecs):
                    _n_sus_qu = solver.get_susceptibilities_normal(
                        q=q_u, M=M, Q=Q,
                        target_doping=target_doping, mu=mu, tx=tx, ty=ty, g_J=g_J,
                        _E_k_cache=E_k_cache_normal,
                        vertex_params=_vp_v,
                        _chi_QQ_cache=chi_QQ_normal_v,
                        actual_doping=_actual_dop_v)
                    V_rpa[ui] = _n_sus_qu['V_full']

                # Vectorised symmetric fill: avoids Python loop over ij pairs
                V_mat          = np.zeros((N_fs, N_fs))
                vvals          = V_rpa[inv_idx]
                V_mat[iu, ju]  = vvals
                V_mat[ju, iu]  = vvals

                V_d_proj = phi_d @ V_mat
            else:
                V_d_proj = np.full(N_fs, V_s_scalar)

            phi2_cache   = float(np.dot(phi_d, phi_d))
            V_d_scalar_c = float(np.dot(phi_d, V_d_proj)) / max(phi2_cache, 1e-12)
            _vertex_cache = {
                'M':                  M,
                'Q':                  Q,
                'Delta':              Delta_eff,
                'j_renorm':           _j_renorm_now,
                'fs_idx':             fs_idx.copy(),
                'V_s_scalar':         V_s_scalar,
                'V_d_scalar':         V_d_scalar_c,
                'V_d_proj':           V_d_proj.copy(),
                'phi_d':              phi_d.copy(),
                'det_q0':             _det_q0,  # RPA det at q=0: near_critical proxy
                'near_critical':      False,    # updated by SCF loop each iteration
                'chi_QQ_from_normal': True,     # chi_QQ_normal_v was computed at Δ=0
                'actual_doping':      target_doping,
            }
        else:
            V_s_scalar = _vertex_cache['V_s_scalar']
            V_d_proj   = _vertex_cache['V_d_proj']
            phi_d      = _vertex_cache['phi_d']
            # V_d_scalar may be missing in old caches (before this fix)
            if 'V_d_scalar' not in _vertex_cache:
                phi2_r = float(np.dot(phi_d, phi_d))
                _vertex_cache['V_d_scalar'] = float(np.dot(phi_d, V_d_proj)) / max(phi2_r, 1e-12)

        # --- Gap equations: V [eV] × F_BZ [dimensionless] → Δ [eV] ---
        Delta_s_new = abs(g_Delta_s * V_s_scalar * F_AA_BZ)
        phi2       = float(np.dot(phi_d, phi_d))
        V_d_scalar = float(np.dot(phi_d, V_d_proj)) / max(phi2, 1e-12)  # V_d_scalar: φ(k)-weighted projection of V(k−k') onto d-wave symmetry.
        Delta_d_new = abs(g_Delta_d * V_d_scalar * F_AB_BZ)
        return Delta_s_new, Delta_d_new, _vertex_cache

class OptimPoint:
    __slots__ = ('doping', 'Delta_tetra', 'u', 'g_JT', 't_pd',
                 'Delta_total', 'converged', 'result',
                 'lambda_JT', 'lambda_max', 'stoner_ok', 'score', 'Tc',
                 'lambda_soc', '_exclude_from_gp')

    def __init__(self, doping, Delta_tetra, u, g_JT, t_pd, Delta_total, converged, result=None,
                 lambda_JT=0.0, lambda_max=0.0, stoner_ok=True, score=0.0, Tc=0.0,
                 lambda_soc=None):
        self.doping      = doping
        self.Delta_tetra = Delta_tetra
        self.u           = u
        self.g_JT        = g_JT
        self.t_pd        = t_pd
        self.Delta_total = Delta_total
        self.converged   = converged
        self.result      = result
        self.lambda_JT   = lambda_JT
        self.lambda_max  = lambda_max
        self.stoner_ok   = stoner_ok
        self.score       = score
        self.Tc          = Tc
        self.lambda_soc       = lambda_soc
        self._exclude_from_gp = False   # set True for G22>0 / spont-JT failures to keep them out of GP training

    def __repr__(self):
        regime = ('SC-trig' if 0.05 < self.lambda_JT < 1.0
                  else ('spont?' if self.lambda_JT >= 1.0 else 'closed'))
        lsoc_str = f", λ_soc={self.lambda_soc:.4f}" if self.lambda_soc is not None else ""
        return (f"OptimPoint(δ={self.doping:.3f}, Δ_tet={self.Delta_tetra:.3f}, "
                f"u={self.u:.2f}, g={self.g_JT:.3f}, t_pd={self.t_pd:.4f}{lsoc_str}, "
                f"Δ={self.Delta_total:.5f}, Tc={self.Tc*1000:.2f}meV, score={self.score:.5f}, "
                f"λ_JT={self.lambda_JT:.3f}[{regime}])")

def check_sc_jt_window(g_JT: float, Delta_CF: float, chi_tau: float, chi0: float, K_lattice: float, K_eff: float, lambda_min: float) -> Dict:
    """
    SC-triggered JT viability check and window diagnostics.

    Window: K_spont < K_lattice < K_SC
        K_spont = g²/Δ_CF         lower bound — spontaneous atomic JT onset
        K_SC    = g²·χ_τ/λ_min    upper bound — minimum K for λ_JT > λ_min

    chi_tau   : multipolar susceptibility χ_τx (eV⁻¹); MUST be the SC-state value
                (from _compute_chi_tau(Δ≠0) or a post-SCF proxy).
                Normal-state χ_τ is typically too small to open the window.
    chi0      : orbital susceptibility scale (eV⁻¹); used for structural_ok gate

    Returns
    -------
    dict with keys
        viable        : bool — window open AND K_lattice inside AND structural_ok
        window_open   : bool — K_SC > K_spont (g_JT-independent window condition: χ_τ·Δ_CF > λ_min)
        K_in_window   : bool — K_spont < K_lattice < K_SC
        structural_ok : bool — g²·χ₀ < K_eff  (G3[2,2] > 0 in normal state)
        K_spont       : lower window bound (eV/Å²)
        K_SC          : upper window bound (eV/Å²)
        K_opt         : geometric mean midpoint — diagnostic only (eV/Å²)
        K_distance    : log-distance of K_lattice from K_opt: log(K_lattice/K_opt)
                        negative = below midpoint (closer to K_spont)
        lambda_JT     : g²·χ_τ / K_lattice (actual coupling)
        lambda_JT_opt : g²·χ_τ / K_opt     (coupling at geometric midpoint)
        lambda_min    : echoed back for traceability
        window_width  : K_SC − K_spont (eV/Å²) — linear width for diagnostics
        note          : human-readable diagnosis string
    """
    g2 = g_JT ** 2
    K_spont = g2 / max(Delta_CF, 1e-12)
    K_SC    = g2 * chi_tau / max(lambda_min, 1e-12)

    window_open   = K_SC > K_spont          # χ_τ·Δ_CF > λ_min
    K_in_window   = K_spont < K_lattice < K_SC
    structural_ok = (g2 * chi0) < K_eff

    if window_open:
        K_opt           = float(np.sqrt(K_spont * K_SC))
        lambda_JT_spont = g2 * chi_tau / max(K_spont, 1e-12)
        lambda_JT_opt   = float(np.sqrt(lambda_JT_spont * lambda_min))
        window_width    = K_SC - K_spont
        K_distance      = float(np.log(max(K_lattice, 1e-12) / max(K_opt, 1e-12)))
    else:
        K_opt         = K_SC  = K_spont   # degenerate
        lambda_JT_opt = 0.0
        window_width  = 0.0
        K_distance    = float('nan')

    lambda_JT = g2 * chi_tau / max(K_lattice, 1e-12)
    viable    = window_open and K_in_window and structural_ok

    if not window_open:
        note = (f"Window closed: χ_τ·Δ_CF={chi_tau*Delta_CF:.4f} < λ_min={lambda_min:.4f}. "
                f"Need SC χ_τ > {lambda_min/max(Delta_CF,1e-12):.3f} eV⁻¹.")
    elif not K_in_window:
        if K_lattice <= K_spont:
            note = f"K_lattice={K_lattice:.4f} ≤ K_spont={K_spont:.4f}: spontaneous JT risk."
        else:
            note = (f"K_lattice={K_lattice:.4f} ≥ K_SC={K_SC:.4f}: "
                    f"λ_JT={lambda_JT:.4f} < λ_min={lambda_min:.4f}.")
    elif not structural_ok:
        note = f"G3[2,2] unstable: g²·χ₀={g2*chi0:.4f} ≥ K_eff={K_eff:.4f}."
    else:
        frac = (K_lattice - K_spont) / max(window_width, 1e-12)
        note = (f"Viable. K_lattice at {frac*100:.0f}% of window ")

    return {
        'viable':        viable,
        'window_open':   window_open,
        'K_in_window':   K_in_window,
        'structural_ok': structural_ok,
        'K_spont':       K_spont,
        'K_SC':          K_SC,
        'K_opt':         K_opt,
        'K_distance':    K_distance,
        'lambda_JT':     lambda_JT,
        'lambda_JT_opt': lambda_JT_opt,
        'lambda_min':    lambda_min,
        'window_width':  window_width,
        'note':          note,
    }

class UnifiedBayesianOptimizer:
    """
    Unified 5D Bayesian optimiser over the full (Delta_tetra, lambda_soc, u, g_JT, t_pd) space.

    All cross-couplings are handled correctly because every candidate evaluation calls
    __post_init__ + _rebuild_orbital_operators unconditionally:
      Delta_CF = f(Delta_tetra, lambda_soc)  — derived, never a free parameter
      t0       = t_pd^2 / Delta_CT           — derived
      J_CT     = f(u, t_pd)                  — derived
      chi_QQ, chi_tau depend on the full band structure, which depends on all five.

    Four-phase pipeline
    -------------------
    Phase 1 — DE scout  (analytic G-matrix, no SCF):
        5D Differential Evolution using compute_G_instability() only.
        Enforces the same H1/H2/H3 hard constraints and S1–S4 soft-penalty weights
        as the original FeasibilityScanner.  Returns a ranked feasible archive.
        No BdG / SCF → ~100x faster per point than full evaluation.

    Phase 2 — GP seed  (parallel SCF on DE top-k):
        Full SCF on the top_k DE candidates, run in parallel via ThreadPoolExecutor.
        Results seed the ARD Matern-2.5 GP.  Hilbert-space rebuild (_rebuild_orbital_operators)
        is performed for every solver clone, so lambda_soc changes are safe.

    Phase 3 — TuRBO  (trust-region GP-EI, batch parallel):
        Sequential iterations; each proposes n_batch candidates via greedy EI inside
        an adaptive hypersphere trust-region.  Batch SCF runs in parallel.
        Trust-region shrinks on failure (x0.65), expands on consecutive improvement (x1.35).
        Trust-region state is guarded by _tr_lock; _gp_obs / observations by _gp_lock.

    Phase 4 — Local refinement (optional):
        Dense random sampling in a ±margin hypercube around the global best.

    Hard constraints (H1–H3): score = 0, excluded from GP training set.
      H1: d2F/dQ^2 |_{Delta=0} > 0   — normal-state Q-stability (no spontaneous JT)
      H2: J_eff * chi_SS < 1          — below Stoner QCP
      H3: G22 > 0                     — JT channel not self-crossing in normal state
      H4: g_t >= _G_T_COHERENCE_MIN  — coherent Fermi surface (Mott guard; g_t encodes full doping-dependent Mott suppression)

    Soft constraints / DE penalty (S1–S4, weights sum to 1.0):
      S1 (w=0.25): 0 < lambda_min(G3) < 0.15  — near-critical, not past QCP
      S2 (w=0.25): reward larger lambda_max monotonically; only penalise near-divergence (λ_max > 0.95) and unsolvable cases.
                   first-order transitions with small λ_max in the normal state are not penalised.
      S3 (w=0.20): lambda_JT > 0.05           — SC-JT coupling above threshold
      S4 (w=0.30): d_lambda_pair/dQ > 0       — JT renormalises V_pair upward

    Post-SCF scoring gates (multiplicative):
      Tier 1 hard guards: mott_suspect / g_t<_G_T_COHERENCE_MIN / ξ/a<1 (new), jchi, G22/λ_min.
      Tier 2 smooth weights (no hard clips):
        w_lJT        : parabolic arch on [0,1], peak at λ_JT=0.45
        w_lJT_kernel : sigmoid(10·(lJTk−0.05))
        w_hessian    : sigmoid(−λ_min_SC/0.05), floor 0.30
      Tier 3 objective: Tc_proxy × conv_f × stoner_f × g22_f × xi_f
                        × lmax_boost × jchi_gate
        lmax_boost = 0.6·softplus(λ_max) + 0.4·(∂λ/∂Q)·σ(10·(λ_max−0.70))/0.5

    Thread safety
    -------------
    _gp_lock : guards _gp, _gp_obs, observations (register + fit_gp snapshot pattern).
    _tr_lock : guards _tr_radius, _tr_center, _improve, _no_improve.
    Trust-region state is mutated only from the main thread (after batch join)
    """
    _NDIMS   = 5
    _SEED_DE = 42
    _SEED_LHS= 43

    # Soft-constraint weights (must sum to 1.0)
    _W_LMIN = 0.25   # lambda_min(G3) near-critical window
    _W_LEFF = 0.25   # lambda_max pairing-vertex window
    _W_LJT  = 0.20   # lambda_JT SC-JT threshold
    _W_DLAM = 0.30   # d_lambda_pair/dQ > 0

    # Trust-region parameters (normalised [0,1]^5 space)
    _TR_INIT   = 0.80
    _TR_MIN    = 0.10
    _TR_MAX    = 1.00
    _TR_SHRINK = 0.65
    _TR_EXPAND = 1.35

    def __init__(self, solver: 'RMFT_Solver', n_doping_scan: int = 7):
        self.solver        = solver
        self.n_doping_scan = n_doping_scan
        self.observations: List[OptimPoint] = []
        self._gp_obs:      List[OptimPoint] = []
        self._gp_lock      = _threading.Lock()
        self._tr_lock      = _threading.Lock()
        self._gp           = None
        self._bounds: Optional[Dict] = None
        self._tr_center:   Optional[np.ndarray] = None
        self._tr_radius:   float = self._TR_INIT
        self._no_improve:  int   = 0
        self._improve:     int   = 0

    # ── Normalisation ────────────────────────────────────────────────────────
    def _normalize(self, dt, ls, u, gJT, tpd) -> np.ndarray:
        b = self._bounds
        return np.array([
            (dt  - b['dt'][0])  / max(b['dt'][1]  - b['dt'][0],  1e-12),
            (ls  - b['ls'][0])  / max(b['ls'][1]  - b['ls'][0],  1e-12),
            (u   - b['u'][0])   / max(b['u'][1]   - b['u'][0],   1e-12),
            (gJT - b['g'][0])   / max(b['g'][1]   - b['g'][0],   1e-12),
            (tpd - b['tpd'][0]) / max(b['tpd'][1] - b['tpd'][0], 1e-12),
        ])

    def _denormalize(self, x) -> tuple:
        b = self._bounds
        return (
            float(b['dt'][0]  + x[0] * (b['dt'][1]  - b['dt'][0])),
            float(b['ls'][0]  + x[1] * (b['ls'][1]  - b['ls'][0])),
            float(b['u'][0]   + x[2] * (b['u'][1]   - b['u'][0])),
            float(b['g'][0]   + x[3] * (b['g'][1]   - b['g'][0])),
            float(b['tpd'][0] + x[4] * (b['tpd'][1] - b['tpd'][0])),
        )

    # ── Solver clone ─────────────────────────────────────────────────────────
    def _make_solver(self, dt: float, ls: float, u: float,
                     gJT: float, tpd: float) -> 'RMFT_Solver':
        """Clone solver with all five parameters set; always rebuilds orbital operators."""
        s = copy.copy(self.solver)
        s.p = copy.copy(self.solver.p)
        s.p.Delta_tetra = float(dt)
        s.p.lambda_soc = float(ls)
        s.p.u = float(u)
        s.p.g_JT = float(gJT)
        s.p.t_pd = float(tpd)
        s.p.__post_init__()
        s._K_bare = s.p.K_lattice
        s._rebuild_orbital_operators(s.p)
        s._reset_transient_state()
        return s

    # ── GP infrastructure ────────────────────────────────────────────────────
    def _build_gp(self) -> None:
        n = self._NDIMS
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(n),
                           length_scale_bounds=[(1e-2, 10.0)] * n, nu=2.5)
                  + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 0.1)))
        self._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=10, normalize_y=True)

    def _obs_to_X(self, obs: 'OptimPoint') -> np.ndarray:
        return self._normalize(obs.Delta_tetra, obs.lambda_soc or 0.0,
                               obs.u, obs.g_JT, obs.t_pd)

    def _fit_gp(self) -> None:
        """Snapshot-pattern GP fit: copy data under lock, fit outside, swap under lock."""
        with self._gp_lock:
            if self._gp is None or len(self._gp_obs) < self._NDIMS + 1:
                return
            X       = np.array([self._obs_to_X(o) for o in self._gp_obs])
            y       = np.array([o.score           for o in self._gp_obs])
            gp_snap = copy.deepcopy(self._gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_snap.fit(X, y)
        with self._gp_lock:
            self._gp = gp_snap

    def _register(self, pt: 'OptimPoint') -> None:
        """Thread-safe: append to observations; add to GP set if constraint-valid."""
        with self._gp_lock:
            self.observations.append(pt)
            if not pt._exclude_from_gp and pt.score > 0.0:
                self._gp_obs.append(pt)

    def _lhs_sample(self, n: int, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed=seed if seed is not None else self._SEED_LHS)
        s = np.zeros((n, self._NDIMS))
        for j in range(self._NDIMS):
            perm = rng.permutation(n)
            s[:, j] = (perm + rng.uniform(size=n)) / n
        return s

    # ── Constraint evaluation (H1–H3 hard, S1–S4 soft) ──────────────────────
    def _eval_constraints(self, s: 'RMFT_Solver', doping: float) -> Dict:
        """
        Evaluate H1/H2/H3 hard and S1–S4 soft constraints on a solver clone.

        Phase 1 — cheap  (~few ms):
            compute_G_instability(compute_dlambda=False)
            → H1: d²F/dQ² > 0   (normal-state Q-stability)
            → H2: J·χ_SS < 1    (below Stoner QCP)
            → H3: G22 > 0        (JT channel stable)
            → S1: λ_min near-critical window
            → S2: λ_eff in pairing window
            → S3: λ_JT above threshold
        Early exit: if any H fails or partial_penalty(S1+S2+S3) ≥ _S4_SKIP_THRESHOLD,
        return S4 = nan (treated as 0.5 by the DE objective).

        Phase 2 (expensive, ~186 s at nk=74; only for promising points):
            compute_G_instability(compute_dlambda=True)
            → S4: ∂λ_pair/∂Q > 0  (JT increases V_pair)

        Skip rule: partial_penalty ≥ feasibility_threshold = 0.25, the point is infeasible regardless
        of S4 (S4 ≥ 0 and _W_DLAM > 0), so skipping S4 preserves the DE ordering
        of infeasible candidates. Only points with partial_penalty < 0.25 require the full evaluation.
        """
        _FEASIBILITY_THRESHOLD = 0.25   # penalty >= this → infeasible regardless of S4

        # Pre-SCF Mott hard-reject
        _abs_d_pre = max(abs(doping), 1e-6)
        _g_t_pre   = (2.0 * _abs_d_pre) / (1.0 + _abs_d_pre)
        if _g_t_pre < _G_T_COHERENCE_MIN:
            return {
                'hard_fail': True,
                'penalty':   100.0,
                'H1': 0.0, 'H2': 0.0, 'H3': 0.0,
                'jchi': 0.0,
                'mott_reject': True,
                'G_res': {},
            }

        M0 = s._estimate_M0(doping)

        # ── Phase 1: cheap G-matrix without dlambda ────────────────────────────
        G_cheap = s.compute_G_instability(doping, M0, compute_dlambda=False)

        H1   = float(G_cheap['d2F_Q_normal'])
        H3   = float(G_cheap['G22'])
        jchi = G_cheap['J_eff'] * G_cheap['chi_DD_s_moriya']
        H2   = 1.0 - jchi

        if H1 <= 0.0 or H2 <= 0.0 or H3 <= 0.0:
            return {'hard_fail': True,
                    'penalty': (max(0.0, -H1) + max(0.0, -H2) + max(0.0, -H3)) * 10.0,
                    'H1': H1, 'H2': H2, 'H3': H3, 'jchi': jchi, 'G_res': G_cheap}

        lmin    = float(G_cheap['lambda_min'])
        V_JT    = s.p.g_JT**2 / max(s._K_bare, 1e-9)
        K_eff   = float(G_cheap['K_eff'])
        chi_orb = float(G_cheap['chi_QQ']) / max(s.p.g_JT**2, 1e-12)
        lam_JT  = V_JT * chi_orb   # = chi_QQ/K_bare; dimensionless

        S1 = (0.0 if 0.0 < lmin < 0.15
              else min(abs(lmin) if lmin <= 0 else max(0.0, lmin - 0.15), 1.0))

        # S2: λ_max from the linearised gap equation at Δ=0.
        try:
            _g_t_s2   = float(G_cheap['g_t'])
            _g_J_s2   = float(G_cheap['g_J'])
            _mu_n_s2  = float(G_cheap['mu_n'])
            _t_eff_s2 = _g_t_s2 * s.p.t0
            _lin_s2   = s.solve_linearized_gap_equation(
                M0, 0.0, 0.0+0j, 0.0+0j, doping, _mu_n_s2,
                _t_eff_s2, _t_eff_s2, _g_J_s2, actual_doping=doping)
            lmax_s2 = float(_lin_s2['lambda_max'])
        except Exception:
            lmax_s2 = float('nan')

        if not np.isfinite(lmax_s2):
            S2 = 1.0   # unknown → maximal penalty
        elif lmax_s2 > 0.95:
            # RPA vertex near-divergent: numerically unreliable
            S2 = float(1.0 / (1.0 + np.exp(20.0 * (lmax_s2 - 0.95))))
        else:
            # Monotonic reward: sigmoid turn-on above noise floor ~0.05.
            # λ_max ≤ 0 → S2 ≈ 1 (maximal penalty); λ_max = 0.5 → S2 ≈ 0.01; λ_max ≥ 0.7 → S2 ≈ 0.
            S2 = float(1.0 - 1.0 / (1.0 + np.exp(-15.0 * (lmax_s2 - 0.15))))

        S3 = max(0.0, 0.05 - lam_JT) / 0.05

        partial_penalty = self._W_LMIN * S1 + self._W_LEFF * S2 + self._W_LJT * S3

        # ── Phase 2: expensive dlambda only for potentially feasible candidates ─
        S4 = 0.5
        G_res = G_cheap

        if partial_penalty < _FEASIBILITY_THRESHOLD:
            G_res = s.compute_G_instability(doping, M0, compute_dlambda=True)
            dlam = float(G_res['dlambda_pair_dQ'])

            if not np.isnan(dlam):
                S4 = float(np.clip(max(0.0, -dlam) / max(abs(dlam), 1e-6), 0.0, 1.0))

        penalty = partial_penalty + self._W_DLAM * S4
        return {
            'hard_fail': False, 'penalty': float(penalty),
            'feasible':  penalty < _FEASIBILITY_THRESHOLD,
            'H1': H1, 'H2': H2, 'H3': H3, 'jchi': jchi,
            'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4,
            'lam_JT': lam_JT, 'lmin': lmin, 'lmax_gap': lmax_s2,
            'G_res': G_res,
        }

    # ── Phase 1: DE scout ────────────────────────────────────────────────────
    def run_de_phase(self, doping: float, param_bounds_5d: Dict[str, tuple], popsize: int = 10, maxiter: int = 50, verbose: bool = True) -> Dict:
        """
        Phase 1: 5D Differential Evolution using analytic G-matrix only (no SCF).

        Objective: penalty = H-violations * 10  +  sum(W_i * S_i) when hard-feasible.
        DE minimises penalty in normalised [0,1]^5 space (workers=1, updating='immediate').
        Returns ranked feasible archive for GP seeding.
        """
        b = param_bounds_5d
        self._bounds = {'dt': b['Delta_tetra'], 'ls': b['lambda_soc'],
                        'u':  b['u'],            'g':  b['g_JT'],
                        'tpd': b['t_pd']}
        self._build_gp()

        _archive: List[Dict] = []

        def _obj(x: np.ndarray) -> float:
            dt, ls, u, gJT, tpd = self._denormalize(x)
            try:
                s   = self._make_solver(dt, ls, u, gJT, tpd)
                res = self._eval_constraints(s, doping)
            except Exception:
                return 999.0
            _archive.append({'x': x.copy(), 'dt': dt, 'ls': ls, 'u': u,
                              'gJT': gJT, 'tpd': tpd,
                              'penalty': res['penalty'],
                              'feasible': res.get('feasible', False),
                              'hard_fail': res['hard_fail'],
                              'score': max(0.0, 1.0 - res['penalty'])})
            return res['penalty']

        t0 = _time.time()
        if verbose:
            _scf_log("DE-SCOUT", "="*60)
            _scf_log("DE-SCOUT", f"5D DE scout (analytic G-matrix, no SCF)"
                                  f"  pop={popsize*self._NDIMS}  maxiter={maxiter}"
                                  f"  doping={doping:.3f}")
            _scf_log("DE-SCOUT", "="*60)

        de_res = differential_evolution(
            _obj, bounds=[(0.0, 1.0)] * self._NDIMS,
            popsize=popsize, maxiter=maxiter, seed=self._SEED_DE,
            tol=1e-4, mutation=(0.5, 1.2), recombination=0.85,
            polish=False, workers=1, updating='immediate')

        feasible = sorted([r for r in _archive if r.get('feasible', False)],
                          key=lambda r: r['penalty'])
        if verbose:
            _scf_log("DE-SCOUT", f"Done ({_time.time()-t0:.1f}s)"
                                  f"  n_eval={len(_archive)}  n_feasible={len(feasible)}")
            for i, r in enumerate(feasible[:5]):
                _scf_log("DE-SCOUT",
                         f"  top-{i+1}: Dt={r['dt']:.3f} ls={r['ls']:.4f}"
                         f" u={r['u']:.2f} g={r['gJT']:.4f} tpd={r['tpd']:.4f}"
                         f"  penalty={r['penalty']:.4f}")
        return {'archive': _archive, 'feasible': feasible,
                'de_result': de_res, 'elapsed_s': _time.time() - t0}

    def _make_phase_grid(self, doping_bounds: tuple) -> tuple:
        """
        Return (dg, fallback_point) shared by all three optimisation phases.

        dg             : np.ndarray — linspace doping grid for _scan_doping
        fallback_point : OptimPoint — safe zero-score sentinel used when an SCF
                         call raises an exception inside a worker thread.
                         Parameters are deliberately conservative (high u=8,
                         small g_JT=0.2) so the GP treats the fallback as a
                         genuinely bad point, not as a spurious attractor.
        """
        d_mid = 0.5 * (doping_bounds[0] + doping_bounds[1])
        dg    = np.linspace(doping_bounds[0], doping_bounds[1], self.n_doping_scan)
        fb    = OptimPoint(d_mid, 0.0, 8.0, 0.2, 0.5, 0.0, False, score=0.0,
                           lambda_soc=self.solver.p.lambda_soc)
        return dg, fb

    # ── Phase 2: GP seed ─────────────────────────────────────────────────────
    def run_gp_seed_phase(self, doping_bounds: tuple, de_feasible: list, top_k: int = 12, verbose: bool = True) -> None:
        """
        Phase 2: full SCF on top_k DE candidates in parallel; results seed the GP.
        Falls back to LHS if de_feasible is empty.
        Each worker thread gets _tpl_ctx(limits=1) to prevent BLAS oversubscription.
        """
        if not de_feasible:
            _scf_log("GP-SEED", "No feasible DE points — falling back to LHS seeding.")
            lhs_pts   = [self._denormalize(x) for x in self._lhs_sample(top_k)]
            candidates = [{'dt': p[0], 'ls': p[1], 'u': p[2],
                           'gJT': p[3], 'tpd': p[4]} for p in lhs_pts]
        else:
            candidates = de_feasible[:top_k]

        dg, fb = self._make_phase_grid(doping_bounds)
        t0    = _time.time()

        if verbose:
            _scf_log("GP-SEED", "="*60)
            _scf_log("GP-SEED", f"Parallel SCF seed: {len(candidates)} candidates"
                                 f"  doping-scan: {self.n_doping_scan} pts/material")

        def _eval(cand):
            try:
                s = self._make_solver(cand['dt'], cand['ls'], cand['u'],
                                      cand['gJT'], cand['tpd'])
                return self._scan_doping(s, dg, cand['dt'], cand['u'],
                                         cand['gJT'], cand['tpd'], lsoc=cand['ls'])
            except Exception as e:
                _scf_log("GP-SEED", f"SCF error: {e}")
                return fb

        n_w = min(_os.cpu_count() or 1, len(candidates), _BO_MAX_WORKERS)
        if n_w > 1:
            def _worker(c):
                with _tpl_ctx(limits=1):
                    return _eval(c)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_w) as ex:
                results = list(ex.map(_worker, candidates))
        else:
            results = [_eval(c) for c in candidates]

        for pt in results:
            self._register(pt)
        self._fit_gp()
        self._update_tr_center()

        if verbose:
            _scf_log("GP-SEED", f"Done ({(_time.time()-t0)/60:.1f} min)"
                                  f"  GP obs={len(self._gp_obs)}/{len(self.observations)}")
            if self._gp_obs:
                _scf_log("GP-SEED", f"Best seed: "
                         f"{max(self._gp_obs, key=lambda o: o.score)}")

    # ── Trust-region helpers ─────────────────────────────────────────────────
    def _update_tr_center(self) -> None:
        """Set trust-region centre to the current best GP obs.  Thread-safe."""
        with self._gp_lock:
            if not self._gp_obs:
                return
            best_x = self._obs_to_X(max(self._gp_obs, key=lambda o: o.score))
        with self._tr_lock:
            self._tr_center = best_x.copy()

    def _update_trust_region(self, improved: bool) -> None:
        """Adapt TR radius.  Must be called from the main thread only (after batch join)."""
        with self._tr_lock:
            if improved:
                self._improve    += 1
                self._no_improve  = 0
                if self._improve >= 2:
                    self._tr_radius = min(self._tr_radius * self._TR_EXPAND, self._TR_MAX)
                    self._improve   = 0
            else:
                self._no_improve += 1
                self._improve     = 0
                if self._no_improve >= 2:
                    self._tr_radius = max(self._tr_radius * self._TR_SHRINK, self._TR_MIN)
                    self._no_improve = 0

    def _expected_improvement_tr(self, n_batch: int = 3, xi: float = 0.01, n_cand: int = 3000) -> List[np.ndarray]:
        """
        TR-constrained greedy batch EI (Kriging Believer diversity).
        All GP reads are done under _gp_lock; TR state under _tr_lock.
        Candidate sampling and EI computation happen outside the lock.
        """
        with self._gp_lock:
            if self._gp is None or len(self._gp_obs) < self._NDIMS + 1:
                return [np.random.default_rng().uniform(size=self._NDIMS)
                        for _ in range(n_batch)]
            y_best = max(o.score for o in self._gp_obs)
            gp     = self._gp   # read-only reference to immutable snapshot
        with self._tr_lock:
            center = self._tr_center.copy() if self._tr_center is not None \
                     else self._obs_to_X(max(self._gp_obs, key=lambda o: o.score))
            radius = self._tr_radius

        rng   = np.random.default_rng()
        raw   = rng.uniform(-1.0, 1.0, size=(n_cand, self._NDIMS))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        r     = rng.uniform(0.0, radius, size=(n_cand, 1)) ** (1.0 / self._NDIMS)
        cand  = np.clip(center + raw / np.maximum(norms, 1e-12) * r, 0.0, 1.0)

        mu, sigma = gp.predict(cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        z     = (mu - y_best - xi) / sigma
        ei    = np.maximum((mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z), 0.0)

        selected = []
        ei_work  = ei.copy()
        for _ in range(n_batch):
            bi = int(np.argmax(ei_work))
            selected.append(cand[bi].copy())
            ei_work[np.linalg.norm(cand - cand[bi], axis=1) < 0.05] = 0.0
        return selected

    # ── Phase 3: TuRBO ───────────────────────────────────────────────────────
    def run_turbo_phase(self, doping_bounds: tuple, n_iterations: int = 30, n_batch: int = 3, verbose: bool = True) -> None:
        """
        Phase 3: trust-region GP-EI with parallel batch SCF per iteration.

        TR state is mutated only from the main thread after each batch completes,
        avoiding any concurrent writes.  _register() is thread-safe via _gp_lock.
        """
        dg, fb = self._make_phase_grid(doping_bounds)
        t0    = _time.time()

        if verbose:
            with self._tr_lock:
                r0 = self._tr_radius
            _scf_log("TURBO", "="*60)
            _scf_log("TURBO", f"TuRBO: {n_iterations} iters x {n_batch} pts/iter"
                               f"  TR_init={r0:.2f}")

        with self._gp_lock:
            prev_best = max((o.score for o in self._gp_obs), default=0.0)

        for i in range(n_iterations):
            self._fit_gp()
            batch_x  = self._expected_improvement_tr(n_batch=n_batch)
            batch_bp = [self._denormalize(x) for x in batch_x]

            # Evaluate batch in parallel; use default-arg binding to capture bp correctly
            def _eval_bp(bp, _dg=dg, _fb=fb):
                dt, ls, u, gJT, tpd = bp
                try:
                    s = self._make_solver(dt, ls, u, gJT, tpd)
                    return self._scan_doping(s, _dg, dt, u, gJT, tpd, lsoc=ls)
                except Exception as e:
                    _scf_log("TURBO", f"SCF error: {e}")
                    return _fb

            n_w = min(_os.cpu_count() or 1, len(batch_bp), _BO_MAX_WORKERS)
            if n_w > 1:
                def _worker(bp, _eval=_eval_bp):
                    with _tpl_ctx(limits=1):
                        return _eval(bp)
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_w) as ex:
                    pts = list(ex.map(_worker, batch_bp))
            else:
                pts = [_eval_bp(bp) for bp in batch_bp]

            for pt in pts:
                self._register(pt)

            # TR update — main thread only, no race
            with self._gp_lock:
                cur_best = max((o.score for o in self._gp_obs), default=0.0)
            improved = cur_best > prev_best + 1e-6
            self._update_trust_region(improved)
            self._update_tr_center()

            if verbose:
                with self._gp_lock:
                    bp_pt = max(self._gp_obs, key=lambda o: o.score) if self._gp_obs else fb
                with self._tr_lock:
                    tr_r = self._tr_radius
                _scf_log("TURBO",
                         f"[{i+1:3d}/{n_iterations}]  TR={tr_r:.3f}"
                         f"  best Tc={bp_pt.Tc*1000:.2f}meV  score={bp_pt.score:.5f}"
                         f"  {'↑' if improved else '—'}  ({_time.time()-t0:.0f}s)")

            prev_best = cur_best
            with self._tr_lock:
                if self._tr_radius <= self._TR_MIN:
                    _scf_log("TURBO", f"TR min reached — early stop at iter {i+1}.")
                    break

        if verbose:
            _scf_log("TURBO", f"Done ({(_time.time()-t0)/60:.1f} min)"
                               f"  GP obs={len(self._gp_obs)}/{len(self.observations)}")

    # ── Phase 4: Local refinement ────────────────────────────────────────────
    def run_local_refinement(self, doping_bounds: tuple, n_grid: int = 12, margin: float = 0.10, verbose: bool = True) -> None:
        """Phase 4 (optional): dense random sampling around the global best ±margin."""
        with self._gp_lock:
            if not self._gp_obs:
                return
            best_x = self._obs_to_X(max(self._gp_obs, key=lambda o: o.score))

        rng   = np.random.default_rng(seed=99)
        pts_x = np.clip(rng.uniform(best_x - margin, best_x + margin,
                                    size=(n_grid, self._NDIMS)), 0.0, 1.0)
        dg, fb = self._make_phase_grid(doping_bounds)
        if verbose:
            _scf_log("LOCAL-REF", f"Local refinement: {n_grid} pts, margin={margin:.2f}")

        for x in pts_x:
            dt, ls, u, gJT, tpd = self._denormalize(x)
            try:
                pt = self._scan_doping(self._make_solver(dt, ls, u, gJT, tpd),
                                       dg, dt, u, gJT, tpd, lsoc=ls)
                self._register(pt)
            except Exception as e:
                _scf_log("LOCAL-REF", f"SCF error: {e}")
                self._register(fb)
        self._fit_gp()

    # ── Shared SCF helpers ───────────────────────────────────────────────────
    def _scan_doping(self, solver, doping_grid, Delta_tetra, u, gJT, t_pd,
                     lsoc=None) -> 'OptimPoint':
        """Warm-started doping scan; returns best-scoring OptimPoint across the grid."""
        best: Optional[OptimPoint] = None
        prev: Optional[Dict]       = None
        iM0 = solver._estimate_M0(doping_grid[0])
        iQ0, iD0 = 1e-4, 0.02
        for doping in doping_grid:
            iM = prev['M']                               if prev else iM0
            iQ = prev['Q']                               if prev else iQ0
            iD = max(prev['Delta_s']+prev['Delta_d'], iD0) if prev else iD0
            pt = self._eval_one_doping(solver, doping, Delta_tetra, u, gJT, t_pd,
                                       iM, iQ, iD, lambda_soc=lsoc)
            if pt.result:
                prev = pt.result
            if best is None or pt.score > best.score:
                best = pt
        return best

    def _eval_one_doping(self, solver, doping, Delta_tetra, u, gJT, t_pd, initial_M, initial_Q, initial_Delta, lambda_soc=None) -> 'OptimPoint':
        """Single-doping SCF evaluation with dual-basin JT probe."""
        tag = f"FULL d={doping:.3f}"
        t0  = _time.time()
        result = None; Delta = 0.0; converged = False

        try:
            result    = solver.solve_self_consistent(
                doping, initial_M, initial_Q, initial_Delta, verbose=False)
            Delta     = result['Delta_s'] + result['Delta_d']
            converged = result['converged']
        except Exception as e:
            _scf_log(tag, f"SCF error: {e}")

        Tc = 0.0
        if converged and Delta > 1e-6:
            try:
                Tc = solver.compute_Tc_by_gap_suppression(doping, sc_result=result)['Tc']
            except Exception:
                pass

        # Dual-basin probe: if SC gap exists but Q≈0, nudge toward JT basin
        if (result is not None and converged
                and Delta > solver.p.tol * 5
                and abs(result['Q']) < 1e-3):
            try:
                r2 = solver.solve_self_consistent(
                    doping, result['M'], 0.3 * solver.p.lambda_hop,
                    max(result['Delta_s'] + result['Delta_d'], initial_Delta),
                    verbose=False)
                if (r2['converged'] and abs(r2['Q']) > 1e-3
                        and r2['Delta_s'] + r2['Delta_d'] > solver.p.tol
                        and r2['F_bdg'] < result['F_bdg'] + 1e-3):
                    result = r2
                    Delta  = r2['Delta_s'] + r2['Delta_d']
                    converged = True
                    Tc = 0.0
                    try:
                        Tc = solver.compute_Tc_by_gap_suppression(
                            doping, sc_result=result)['Tc']
                    except Exception:
                        pass
            except Exception:
                pass

        M_conv = result['M'] if result is not None else initial_M
        G_post = solver.compute_G_instability(doping, M=M_conv) if result is not None else {}

        if Delta < 1e-8 and Tc < 1e-6:
            return self._g_fallback_score(initial_M, doping, Delta_tetra, u, gJT, t_pd)

        stoner_ok = not result['afm_unstable']

        _chi_tau_val = result.get('chi_tau')
        if _chi_tau_val is None:  # Recompute chi_tau from the available (M, Q, μ) if SCF did not converge fully.
            _chi_tau_val = solver._compute_chi_tau(
                result.get('M', initial_M), result.get('Q', 0.0), doping,
                complex(result.get('Delta_s', 0.0)), complex(result.get('Delta_d', 0.0)),
                result.get('mu', 0.0))['chi_tau']
        
        # lambda_JT = (g²/K)·χ_tau: SC-triggered JT coupling (requires Δ≠0).
        lambda_JT = (solver.p.g_JT**2 / max(solver._K_bare, 1e-9)) * _chi_tau_val

        score = self._score(Delta, converged, result, Tc, G_post, lambda_JT=lambda_JT)
        lambda_max = result['lambda_max']
        lambda_JT_kernel = result.get('lambda_JT_kernel', float('nan'))
        G_chi_K   = G_post['chi_QQ'] / max(G_post['K_eff'], 1e-9)
        regime    = ('SC-triggered' if 0.05 < lambda_JT < 1.0
                     else ('strong-coupling' if lambda_JT >= 1.0 else 'JT-closed'))
        _hmin = (result['hessian'].get('min_curvature') or float('nan'))
        _scf_log(tag,
                 f"D={Delta:.5f} Tc={Tc*1000:.2f}meV score={score:.5f}"
                 f" lJT={lambda_JT:.3f}[{regime}]"
                 f" lJT_ker={lambda_JT_kernel:.3f}"
                 f" lmax={lambda_max:.3f}({result['gap_symmetry']})"
                 f" cQQ/K={G_chi_K:.3f} lmin(H)={_hmin:+.4f}[{G_post['instab_dir']}]"
                 f" {'ok' if converged else 'nc'} ({_time.time()-t0:.1f}s)")
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, Delta, converged,
                          result, lambda_JT, lambda_max, stoner_ok, score, Tc, lambda_soc)

    def _g_fallback_score(self, M0, doping, Delta_tetra, u, gJT, t_pd) -> 'OptimPoint':
        """Cheap G-matrix proxy score when SCF finds no SC gap."""
        try:
            s2 = copy.copy(self.solver); s2.p = copy.copy(self.solver.p)
            s2.p.Delta_tetra = float(Delta_tetra); s2.p.u = float(u)
            s2.p.g_JT = float(gJT); s2.p.t_pd = float(t_pd)
            s2.p.__post_init__(); s2._K_bare = s2.p.K_lattice
            s2._reset_transient_state()
            G = s2.compute_G_instability(doping, M0)
        except Exception:
            return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)
        G22 = G['G22']; lm = G['lambda_min']; Te = G['Tc_estimate']
        if G22 <= 0.0 or lm <= 0.0:
            return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)
        g22_f = _BO_SPONT_JT_PEN + (1.0 - _BO_SPONT_JT_PEN) / (1.0 + np.exp(-G22 / _BO_SIGMOID_W))
        sc = _BO_G_FALLBACK * (1.0 - min(lm, 1.0)) * g22_f * (1.0 + min(Te / 0.004, 8.0))
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=sc)

    def _score(self, Delta: float, converged: bool, result: dict, Tc: float,
               G_post: dict, lambda_JT: float = float('nan')) -> float:
        """
        Post-SCF scoring: three-tier multiplicative architecture.

        Tier 1 — Hard physical constraints (return 0 immediately):
            mott    : g_t < _G_T_COHERENCE_MIN or ξ/a < 1.0  — incoherent / artefact SC
            jchi    : J·χ_SS > _JCHI_HARD_REJECT — deep AFM, SC impossible
            g22     : G22 ≤ 0 or λ_min(G3) ≤ 0  — spontaneous JT

        Tier 2 — Smooth mechanism weights (no hard clips; continuous in [0,1]):
            w_lJT        : λ_JT ∈ (0,1) parabola peak at 0.45; zero at 0 and 1.
                           Replaces the old rise×fall step with a single smooth arch
                           that is easy to optimise through.
            w_lJT_kernel : sigmoid(10·(lJTk − 0.05)).  Soft turn-on above the noise
                           floor; saturates smoothly near 1 instead of hard-clipping.
            w_hessian    : sigmoid(−lmin_sc / 0.05).  Already a sigmoid — kept, but
                           floor raised to 0.30 (was 0.10) so absent data is less
                           catastrophic for unconverged points.

        Tier 3 — Optimisation objective:
            Tc_proxy   : Tc if converged, else Δ·0.3
            xi_f       : coherence length sigmoid gate (centre ξ/a=2, width k=2)
            conv_f     : 1.0 if converged, else 0.10
            stoner_f   : 1.0 if AFM stable, else _BO_W_STONER_BAD
            lmax_boost : sigmoid-gated ∂λ/∂Q bonus.
                         softplus(λ_max) captures pairing strength continuously.
                         σ(10·(λ_max − 0.70)) gates the ∂λ/∂Q term so a large
                         derivative only matters when λ_max > 0.70 (already in
                         strong-pairing territory).  Replaces ratio_bonus (2Δ/kTc),
                         which was redundant with Tc_proxy.
            g22_f      : continuous spontaneous-JT softening (unchanged)
            jchi_gate  : Gaussian near-QCP sweet-spot reward (unchanged)
        """
        # ── Tier 1: hard guards ───────────────────────────────────────────────
        # Mott / incoherence guard (mirrors post-SCF Mott filter in solve_self_consistent)
        if result.get('mott_suspect', False):
            return 0.0
        _g_t_sc = float(result.get('g_t', 1.0))
        _xia_sc = float(result.get('xi_over_a', float('nan')))
        if _g_t_sc < _G_T_COHERENCE_MIN or (np.isfinite(_xia_sc) and _xia_sc < 1.0):
            return 0.0

        _jchi = float(np.clip(result['J_eff'] * result['chi0_moriya'], 0.0, 10.0))
        if _jchi > _JCHI_HARD_REJECT:
            return 0.0
        G22    = G_post['G22']
        lmin_n = G_post['lambda_min']
        if G22 <= 0.0 or lmin_n <= 0.0:
            return 0.0

        # ── Tier 2: smooth mechanism weights ─────────────────────────────────
        # λ_JT arch: parabola on [0,1], peak=1 at lJT=0.45, zero at endpoints.
        # Unknown (nan) → 0.5 (neutral). Over-coupled (≥1) → 0.10 (soft penalty).
        lJT = float(lambda_JT)
        if not np.isfinite(lJT):
            w_lJT = 0.5
        elif lJT >= 1.0:
            w_lJT = 0.10
        else:
            lJT_c = float(np.clip(lJT, 0.0, 1.0))
            # parabola through (0,0),(0.45,1),(1,0): f(x)=−x(x−1)/0.2025 normalised to 1
            w_lJT = float(np.clip(-lJT_c * (lJT_c - 1.0) / 0.2025, 0.0, 1.0))

        # λ_JT_kernel: sigmoid(10·(lJTk − 0.05))
        lJTk = float(result.get('lambda_JT_kernel', float('nan')))
        if not np.isfinite(lJTk):
            w_lJT_kernel = 0.5
        elif lJTk >= 1.0:
            w_lJT_kernel = 0.30   # the Rayleigh quotient exceeding 1 signals numerical over-saturation (kernel eigenvalue > pairing eigenvalue
        else:
            w_lJT_kernel = float(1.0 / (1.0 + np.exp(-10.0 * (lJTk - 0.05))))

        # Hessian: sigmoid(−lmin_sc / 0.05). Floor 0.30 for missing/unconverged data.
        lmin_sc = result.get('hessian', {}).get('min_curvature', None)
        if lmin_sc is not None and np.isfinite(lmin_sc):
            w_hessian = float(1.0 / (1.0 + np.exp(lmin_sc / _BO_SC_HESS_SIG)))
        else:
            w_hessian = 0.30

        # ── Tier 3: optimisation objective ───────────────────────────────────
        g22_f = _BO_SPONT_JT_PEN + (1.0 - _BO_SPONT_JT_PEN) / (
            1.0 + np.exp(-G22 / _BO_SIGMOID_W))

        # Coherence gate: hard zero below ξ/a=1 (already checked above for finite values),
        # sigmoid ramp centred at ξ/a=2, saturates at ξ/a=4.
        if   not np.isfinite(_xia_sc): xi_f = 0.5
        elif _xia_sc < 4.0:            xi_f = float(np.clip(1.0 / (1.0 + np.exp(-2.0 * (_xia_sc - 2.0))), 0.0, 1.0))
        else:                          xi_f = 1.0

        Tc_proxy = Tc if Tc > 1e-6 else Delta * 0.3
        conv_f   = 1.0 if converged else 0.10
        stoner_f = 1.0 if not result['afm_unstable'] else _BO_W_STONER_BAD

        # lmax_boost: sigmoid(10·(λ_max − 0.70)) gates ∂λ/∂Q so the derivative
        # reward only activates when the system is already in strong-pairing territory.
        # softplus(λ_max) = ln(1 + e^λ) grows continuously with pairing strength.
        # Weights: w1=0.6 (strength), w2=0.4 (JT upward renorm), scale 0.5 eV/Å → O(1).
        _lmax = float(result.get('lambda_max', 0.0))
        _dlam = float(G_post.get('dlambda_pair_dQ', float('nan')))
        _lmax_gate = 1.0 / (1.0 + np.exp(-10.0 * (_lmax - 0.70)))
        _softplus  = float(np.log1p(np.exp(np.clip(_lmax, -10.0, 10.0))))
        _dlam_pos  = max(_dlam, 0.0) if np.isfinite(_dlam) else 0.0
        lmax_boost = float(np.clip(0.6 * _softplus + 0.4 * _dlam_pos * _lmax_gate / 0.5,
                                   0.0, 2.0))

        # jchi_gate shapes the score *within* the feasible region to prefer the near-QCP sweet spot.
        jchi_gate = float(np.exp(-0.5 * ((_jchi - _BO_OPT_JCHI) / _BO_SIG_JCHI) ** 2))
        jchi_gate = float(np.clip(
            jchi_gate + (_BO_JCHI_FLOOR if _jchi < _BO_JCHI_NOISE else 0.0), 0.0, 1.0))

        return (Tc_proxy * conv_f * stoner_f * g22_f
                * w_lJT * w_lJT_kernel * w_hessian
                * xi_f * lmax_boost * jchi_gate)

    def _jt_causality_test(self, solver, result) -> dict:
        """SC-triggered JT causality test using G-matrix + SC Hessian eigenvalue."""
        try:
            G = solver.compute_G_instability(
                result.get('target_doping', 0.15), result.get('M', 0.11))
        except Exception as e:
            return {'sc_triggered': False, 'error': str(e), 'note': 'G-matrix failed',
                    'Q_sc': 0.0, 'Q_normal': 0.0, 'level1_ok': False, 'level2_ok': False}

        G22       = G['G22']
        Q_sc      = abs(result['Q'])
        Delta_sc  = abs(result['Delta_s']) + abs(result['Delta_d'])
        lmin_sc = result.get('hessian', {}).get('min_curvature')
        if lmin_sc is None:
            lmin_sc = float('nan')

        if Delta_sc < 1e-4 or Q_sc < 1e-5:
            return {'sc_triggered': False, 'G22_normal': G22,
                    'd2F_normal': G['d2F_Q_normal'], 'd2F_sc': float('nan'),
                    'lmin_sc_hessian': lmin_sc, 'note': 'No SC/JT order in converged state',
                    'Q_sc': Q_sc, 'Q_normal': 0.0, 'level1_ok': False, 'level2_ok': False}

        stability      = float(np.clip(G22 / 0.5, 0.0, 1.0)) if np.isfinite(G22) else 0.0
        hess_confirmed = np.isfinite(lmin_sc) and lmin_sc < 0.0
        hess_metric    = float(np.clip(-lmin_sc / 0.1, 0.0, 1.0)) if np.isfinite(lmin_sc) else 0.0
        score          = 0.0 if stability < 0.1 else (stability + hess_metric) / 2.0

        if   score > 0.7 and stability > 0.6 and hess_confirmed: regime = 'CONFIRMED SC-triggered JT'
        elif score > 0.4 and stability > 0.3:                    regime = 'PARTIAL: SC-JT coupling present'
        elif stability < 0.3:                                    regime = 'WARNING: Normal state JT-unstable'
        elif not hess_confirmed:                                 regime = 'WARNING: SC does not soften JT mode'
        else:                                                    regime = 'INCONCLUSIVE: Mixed signals'

        return {
            'sc_triggered': score > 0.5 or hess_confirmed, 'score': score,
            'stability': stability, 'hess_confirmed': hess_confirmed,
            'lmin_sc_hessian': lmin_sc, 'G22_normal': G22,
            'd2F_normal': G['d2F_Q_normal'], 'regime': regime,
            'note': f"{regime} (score={score:.3f}, lmin_H={lmin_sc:.4f})",
            'Q_sc': Q_sc, 'Q_normal': 0.0,
            'level1_ok': stability > 0.3, 'level2_ok': hess_confirmed,
        }

    @staticmethod
    def _progress_bar(done, total, elapsed_s, width=40, prefix="") -> str:
        frac   = done / max(total, 1)
        filled = int(width * frac)
        bar    = "█" * filled + "░" * (width - filled)
        if done > 0 and elapsed_s > 0:
            eta_s = elapsed_s / done * (total - done)
            h, r  = divmod(int(eta_s), 3600); m, s = divmod(r, 60)
            eta   = f"ETA {h}:{m:02d}:{s:02d}"
        else:
            eta = "ETA --:--:--"
        return (f"\r{prefix}[{bar}] {done}/{total} {int(100*frac)}%"
                f" {int(elapsed_s//60)}m{int(elapsed_s%60):02d}s {eta}  ")

    @staticmethod
    def _pick_best(observations: list) -> tuple:
        best     = max(observations, key=lambda o: o.score)
        best_raw = max(observations, key=lambda o: o.Delta_total)
        valid    = [o for o in observations if o.converged and o.lambda_JT > 0.0]
        return best, (max(valid, key=lambda o: o.score) if valid else best), best_raw

    # ── Main entry point ─────────────────────────────────────────────────────
    def optimize(self, doping_bounds: tuple, param_bounds_5d: Dict[str, tuple],
                 de_popsize: int = 10, de_maxiter: int = 50,
                 gp_seed_top_k: int = 12, turbo_iterations: int = 30,
                 turbo_batch: int = 3, local_refine: bool = True,
                 local_n_grid: int = 10, verbose: bool = True) -> Dict:
        """
        Full four-phase optimisation.

        Parameters
        ----------
        doping_bounds    : (min, max) carrier doping range
        param_bounds_5d  : {'Delta_tetra':…, 'lambda_soc':…, 'u':…, 'g_JT':…, 't_pd':…}
        de_popsize       : DE population size multiplier (total = de_popsize * 5)
        de_maxiter       : DE maximum generations
        gp_seed_top_k    : DE candidates evaluated with full SCF for GP seeding
        turbo_iterations : TuRBO iteration count
        turbo_batch      : parallel proposals per TuRBO iteration
        local_refine     : enable Phase 4 local refinement
        local_n_grid     : Phase 4 grid density
        verbose          : detailed logging
        """
        t_global = _time.time()

        de_phase = self.run_de_phase(
            doping=0.5*(doping_bounds[0]+doping_bounds[1]),
            param_bounds_5d=param_bounds_5d,
            popsize=de_popsize, maxiter=de_maxiter, verbose=verbose)

        self.run_gp_seed_phase(
            doping_bounds=doping_bounds, de_feasible=de_phase['feasible'],
            top_k=gp_seed_top_k, verbose=verbose)

        self.run_turbo_phase(
            doping_bounds=doping_bounds, n_iterations=turbo_iterations,
            n_batch=turbo_batch, verbose=verbose)

        if local_refine and len(self._gp_obs) >= self._NDIMS + 1:
            self.run_local_refinement(
                doping_bounds=doping_bounds, n_grid=local_n_grid, verbose=verbose)

        best, best_valid, best_raw = self._pick_best(self.observations)
        elapsed = _time.time() - t_global

        if verbose:
            n_excl = sum(1 for o in self.observations if o._exclude_from_gp)
            _scf_log("UNIFIED-BO", "="*60)
            _scf_log("UNIFIED-BO", f"Done ({elapsed/60:.1f} min total)"
                                    f"  GP={len(self._gp_obs)}/{len(self.observations)}"
                                    f"  ({n_excl} hard-constrained excluded)")
            _scf_log("UNIFIED-BO", f"Best: {best}")

        return {'best_point': best, 'best_valid': best_valid, 'best_raw': best_raw,
                'observations': self.observations, 'de_archive': de_phase['archive'],
                'elapsed_s': elapsed}


def plot_phase_diagrams(solver: RMFT_Solver, initial_M: float, initial_Q: float, initial_Delta: float, doping_range: np.ndarray, cf_min: float = 0.05, cf_max: float = 0.20, N_cf: int = 10, opt_result: Optional[Dict] = None):
    phase_data = {
        'target_doping': [], 'M': [], 'Q': [],
        'Delta_s': [], 'Delta_d': [],
        'mu': [], 'density': [], 'F_bdg': [],
        'chi_tau': [], 'Ut_ratio': []
    }

    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", f"PHASE DIAGRAM SCAN  n={len(doping_range)} δ: {doping_range[0]:.3f}→{doping_range[-1]:.3f}")
    _scf_log("MAIN", f"V_s,V_d: g²/K·χ_RPA  U/t0={solver.p.u:.2f} η={solver.p.eta:.3f} δ₀={solver.p.doping_0:.3f}")
    _scf_log("MAIN", "="*60)

    all_results = []   # store every result for convergence history plots
    prev_result = None

    def _gamma_splitting(lambda_soc: float, Delta_tetra: float, Delta_inplane: float = 0.0) -> float:
        evals = np.linalg.eigvalsh(_build_soc_cf_hamiltonian(lambda_soc, Delta_tetra, Delta_inplane))
        return float(evals[2] - evals[0])

    for i, target_doping in enumerate(doping_range):
        if prev_result is not None:
            init_M     = prev_result['M']
            init_Q     = prev_result['Q']
            init_Delta = prev_result['Delta_s'] + prev_result['Delta_d']
        else:
            init_M, init_Q, init_Delta = initial_M, initial_Q, initial_Delta

        result = solver.solve_self_consistent(
            target_doping=target_doping,
            initial_M=init_M,
            initial_Q=init_Q,
            initial_Delta=init_Delta,
            verbose=True
        )

        has_afm = result['M'] > 0.15
        has_sc  = result['Delta_d'] > 2 * solver.p.kT
        has_jt  = abs(result['Q']) > 1e-4
        if   has_afm and not has_sc:  phase = 'AFM'
        elif has_sc  and has_jt:      phase = 'SC+JT'
        elif has_afm and has_sc:      phase = 'MIX'
        else:                         phase = 'NM'

        density_error = abs(result['density'] - (1 - target_doping))
        dens_warn = '⚠' if density_error > 0.01 else ' '
        chi_tau  = result['chi_tau']
        Ut_ratio = result['Ut_ratio']

        _scf_log("SCAN", f"[{i+1:2d}/{len(doping_range)}] δ={target_doping:.3f}  "
              f"[{phase:6s}]  M={result['M']:.3f}  Q={result['Q']:+.4f}  "
              f"Δs={result['Delta_s']:.4f}  Δd={result['Delta_d']:.4f}  "
              f"χτ={chi_tau:.3f}  U/t={Ut_ratio:.2f}  "
              f"n={result['density']:.4f}{dens_warn}")

        phase_data['target_doping'].append(target_doping)
        phase_data['M'].append(result['M'])
        phase_data['Q'].append(result['Q'])
        phase_data['Delta_s'].append(result['Delta_s'])
        phase_data['Delta_d'].append(result['Delta_d'])
        phase_data['mu'].append(result['mu'])
        phase_data['density'].append(result['density'])
        phase_data['F_bdg'].append(result['F_bdg'])
        phase_data['chi_tau'].append(chi_tau)
        phase_data['Ut_ratio'].append(Ut_ratio)

        all_results.append(result)
        prev_result = result

    # ── Per-doping Tc and instability summary table ──────────────────────────
    _scf_log("SCAN", "── Tc vs doping summary ──")
    _scf_log("SCAN", f"  {'δ':>6}  {'Tc_lin(meV)':>12}  {'Tc_BCS(meV)':>12}  {'λ_min':>8}  {'G22':>8}  {'2Δ/kTc':>8}  regime")
    _tc_list, _g22_list, _lmin_list, _ratio_list = [], [], [], []
    for i_d, (_d, _res) in enumerate(zip(doping_range, all_results)):
        try:
            _M_d = float(_res['M'])
            _Gd  = solver.compute_G_instability(_d, M=_M_d)
            _Tcl_res = solver.compute_Tc_by_gap_suppression(_d, sc_result=_res)
            _Tcl = float(_Tcl_res['Tc'])
            _gr_d = solver.compute_gap_ratio(_d, _res)
            _tc_list.append(_Tcl)
            _g22_list.append(_Gd['G22'])
            _lmin_list.append(_Gd['lambda_min'])
            _ratio_list.append(_gr_d['ratio_2D'])
            _scf_log("SCAN", f"  {_d:6.3f}  {_Tcl*1000:12.2f}  "
                  f"{_Gd['Tc_estimate']*1000:12.2f}  "
                  f"{_Gd['lambda_min']:8.4f}  "
                  f"{_Gd['G22']:8.4f}  "
                  f"{_gr_d['ratio_2D']:8.2f}  {_gr_d['coupling_regime']}")
        except Exception:
            _tc_list.append(0.0); _g22_list.append(float('nan'))
            _lmin_list.append(float('nan')); _ratio_list.append(0.0)
            _scf_log("SCAN", f"  {_d:6.3f}  diagnostics failed")
    _scf_log("MAIN", f"Target ΔCF window: [{cf_min:.3f}, {cf_max:.3f}] eV  (scanned via Δ_tet)")

    ref_doping_idx = len(doping_range) // 2
    ref_doping = doping_range[ref_doping_idx]
    _scf_log("MAIN", f"Reference doping: δ={ref_doping:.3f}")
    _dt_prescan  = np.linspace(-0.60, 0.10, 200)
    _cf_prescan  = np.array([
        _gamma_splitting(solver.p.lambda_soc, dt, solver.p.Delta_inplane)
        for dt in _dt_prescan
    ])
    _mask        = (_cf_prescan >= cf_min) & (_cf_prescan <= cf_max)

    if _mask.sum() < 2:
        _mask = np.ones(len(_dt_prescan), dtype=bool)
    _dt_lo, _dt_hi = _dt_prescan[_mask][[0, -1]]
    dt_scan_grid = np.linspace(_dt_lo, _dt_hi, N_cf)
    cf_scan_actual = np.array([
        _gamma_splitting(solver.p.lambda_soc, dt, solver.p.Delta_inplane)
        for dt in dt_scan_grid
    ])

    cf_gaps, cf_Q_values, cf_M_values, cf_actual_CF = [], [], [], []
    cf_previous = None

    for dt, cf_actual in zip(dt_scan_grid, cf_scan_actual):
        p_cf = copy.copy(solver.p)
        p_cf.Delta_tetra = float(dt)
        p_cf.__post_init__()

        cf_solver = copy.copy(solver)
        cf_solver.p = p_cf
        cf_solver._vbdg = None   # force fresh VectorizedBdG for the new params

        if cf_previous is not None:
            init_M     = cf_previous['M']
            init_Q     = cf_previous['Q']
            init_Delta = cf_previous['Delta_s'] + cf_previous['Delta_d']
        else:
            init_M     = phase_data['M'][ref_doping_idx]
            init_Q     = phase_data['Q'][ref_doping_idx]
            init_Delta = (phase_data['Delta_s'][ref_doping_idx]
                          + phase_data['Delta_d'][ref_doping_idx])

        cf_result = cf_solver.solve_self_consistent(target_doping=ref_doping, initial_M=init_M, initial_Q=init_Q, initial_Delta=init_Delta, verbose=False)
        cf_gaps.append(cf_result['Delta_d'])
        cf_Q_values.append(cf_result['Q'])
        cf_M_values.append(cf_result['M'])
        cf_actual_CF.append(cf_actual)
        cf_previous = {
            'M': cf_result['M'], 'Q': cf_result['Q'],
            'Delta_s': cf_result['Delta_s'], 'Delta_d': cf_result['Delta_d']
        }
        _scf_log("CF-SCAN", f"Δ_tet={dt:+.4f}  ΔCF={cf_actual:.4f} eV → "
              f"Δs={cf_result['Delta_s']:.5f}  Δd={cf_result['Delta_d']:.5f} "
              f"Q={cf_result['Q']:+.5f}  M={cf_result['M']:.4f}")
        _scf_log("CF-SCAN", f"χ₀={cf_result['chi0']:.4f}"
              f"  |  RPA factor = {cf_result['rpa_factor']:.3f}×")
        _irr = cf_result['irrep_info']
        _scf_log("CF-SCAN", f"Irrep R={_irr['selection_ratio']:.4f} "
              f"JT {'ALLOWED ✓' if _irr['jt_algebraically_allowed'] else 'BLOCKED ✗'}")

    cf_range = np.array(cf_actual_CF)

    max_gap_idx = np.argmax(cf_gaps)
    sweet_spot_cf = cf_range[max_gap_idx]
    max_gap = cf_gaps[max_gap_idx]
    _scf_log("MAIN", f"✓ Sweet spot: ΔCF = {sweet_spot_cf:.3f} eV"
          f"  (Δ_tet = {dt_scan_grid[max_gap_idx]:+.4f} eV)"
          f"  Δmax = {max_gap:.4f} eV")

    n_rows = 4 if opt_result is not None else 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    fig.suptitle('SC-Activated JT Model – Full Results', fontsize=15, fontweight='bold')

    cmap = plt.cm.plasma
    n_dop = len(doping_range)
    colors = [cmap(i / max(n_dop - 1, 1)) for i in range(n_dop)]

    _plot_phase_data(axes[0, 0], phase_data)
    # Overlay Tc(δ) on the phase diagram (right y-axis)
    _ax0_r = axes[0, 0].twinx()
    _tc_arr = np.array(_tc_list) * 1000  # meV
    _ax0_r.plot(doping_range[:len(_tc_arr)], _tc_arr, 'k--o', linewidth=1.5,
                markersize=4, label='Tc (meV)')
    _ax0_r.set_ylabel('Tc (meV)', fontsize=10, color='k')
    _ax0_r.tick_params(axis='y', labelcolor='k')
    _ax0_r.legend(fontsize=8, loc='upper left')

    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(cf_range, cf_gaps, 'b-o', linewidth=2, markersize=5, label='Δ_d (SC gap)')
    ax.plot(cf_range, cf_Q_values, 'g-s', linewidth=1.5, markersize=4, label='Q (JT)')
    ax2.plot(cf_range, cf_M_values, 'r-^', linewidth=1.5, markersize=4, label='M (AFM)')
    ax.axvline(sweet_spot_cf, color='gray', linestyle='--', linewidth=1,
               label=f'Sweet spot {sweet_spot_cf:.3f} eV')
    ax.set_xlabel('Crystal Field Δ_CF (eV)', fontsize=11)
    ax.set_ylabel('Gap / Distortion (eV or Å)', fontsize=10)
    ax2.set_ylabel('Magnetization M', fontsize=10, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax.set_title(f'CF Sweet-Spot Search (δ={ref_doping:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    _plot_dos(axes[0, 2], solver, all_results[-1])

    ax = axes[1, 0]
    for idx, res in enumerate(all_results):
        ax.plot(res['history']['M'], color=colors[idx],
                linewidth=1.5, label=f'δ={doping_range[idx]:.3f}')
    ax.set_ylabel('Magnetization M', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title('SCF Convergence: M (per target doping)', fontsize=11)
    ax.legend(fontsize=7, ncol=max(1, n_dop // 4), loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for idx, res in enumerate(all_results):
        ax.plot(res['history']['Q'], color=colors[idx],
                linewidth=1.5, label=f'δ={doping_range[idx]:.3f}')
    ax.set_ylabel('JT Distortion Q (Å)', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title('SCF Convergence: Q (per target doping)', fontsize=11)
    ax.legend(fontsize=7, ncol=max(1, n_dop // 4), loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for idx, res in enumerate(all_results):
        ax.plot(res['history']['Delta'], color=colors[idx],
                linewidth=1.5, label=f'δ={doping_range[idx]:.3f}')
    ax.set_ylabel('SC Gap |Δ| (eV)', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title('SCF Convergence: |Δ| (per target doping)', fontsize=11)
    ax.legend(fontsize=7, ncol=max(1, n_dop // 4), loc='upper right')
    ax.grid(True, alpha=0.3)

    last_hist = all_results[-1]['history']
    last_label = f'δ={doping_range[-1]:.3f}'

    ax = axes[2, 0]
    ax.plot(last_hist['F_bdg'],     'k-',  linewidth=2,   label='F_bdg')
    ax.plot(last_hist['F_cluster'], 'r--', linewidth=1.5, alpha=0.8, label='F_cluster')
    ax.set_ylabel('Free Energy (eV)', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title(f'Free Energy (may be non-monotonic) [{last_label}]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(last_hist['g_t'], 'c-', linewidth=2, label='g_t (kinetic)')
    ax.plot(last_hist['g_J'], 'm-', linewidth=2, label='g_J (exchange)')
    ax.set_ylabel('Renormalization Factor', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title(f'Gutzwiller Factors [{last_label}]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    _tc_arr_plot = np.array(_tc_list) * 1000   # meV
    _g22_arr     = np.array(_g22_list, dtype=float)
    _lmin_arr    = np.array(_lmin_list, dtype=float)
    _dops_plot   = np.array(doping_range[:len(_tc_arr_plot)])

    ax.plot(_dops_plot, _tc_arr_plot, 'b-o', linewidth=2, markersize=5, label='Tc (meV)')
    ax.set_ylabel('Tc (meV)', fontsize=11, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_xlabel('Doping δ', fontsize=11)
    ax.set_title(f'Tc & JT Stability vs Doping [{last_label}]', fontsize=11)
    ax.grid(True, alpha=0.3)

    _ax22r = ax.twinx()
    _ax22r.plot(_dops_plot, _g22_arr, 'r--^', linewidth=1.5, markersize=4, label='G22 (JT stab.)')
    _ax22r.axhline(0.0, color='r', linewidth=0.7, linestyle=':')
    _ax22r.set_ylabel('G3[2,2]  (>0 = SC-trig JT)', fontsize=9, color='r')
    _ax22r.tick_params(axis='y', labelcolor='r')

    lines1b, labs1b = ax.get_legend_handles_labels()
    lines2b, labs2b = _ax22r.get_legend_handles_labels()
    ax.legend(lines1b + lines2b, labs1b + labs2b, fontsize=8, loc='upper right')

    plt.tight_layout()

    if opt_result is not None:
        all_obs = opt_result.get('observations', opt_result.get('all_obs', []))
        if all_obs:
            deltas  = [o.Delta_total for o in all_obs]
            scores   = [o.score for o in all_obs]
            dopings  = [o.doping for o in all_obs]
            dt_vals  = [o.Delta_tetra for o in all_obs]
            lJT_vals = [o.lambda_JT for o in all_obs]
            colours  = ['green' if 0.05 < lj < 1.0 else ('orangered' if lj >= 1.0 else 'orange') for lj in lJT_vals]

            conv_mask = [o.converged for o in all_obs]
            running = np.full(len(scores), np.nan)
            best_so_far = -np.inf
            for _i, (sc, cv) in enumerate(zip(scores, conv_mask)):
                if cv:
                    best_so_far = max(best_so_far, sc)
                if np.isfinite(best_so_far):
                    running[_i] = best_so_far

            conv_scores = [scores[i] for i, cv in enumerate(conv_mask) if cv]
            best_idx = int(np.argmax(scores))

            ax_p = axes[3, 0]
            idx_conv  = [i for i, cv in enumerate(conv_mask) if cv]
            idx_nconv = [i for i, cv in enumerate(conv_mask) if not cv]
            if idx_conv:
                ax_p.plot(idx_conv, [deltas[i] for i in idx_conv],
                          'o', alpha=0.5, color='steelblue', markersize=4, label='Δ (conv)')
                ax_p.plot(idx_conv, [scores[i] for i in idx_conv],
                          's', alpha=0.5, color='darkgreen', markersize=4, label='score (conv)')
            if idx_nconv:
                ax_p.plot(idx_nconv, [scores[i] for i in idx_nconv],
                          's', alpha=0.25, color='gray', markersize=4,
                          markerfacecolor='none', label='score (non-conv)')
            ax_p.plot(running, 'g-', linewidth=2, label='best (conv only)')
            ax_p.set_xlabel('Evaluation'); ax_p.set_ylabel('eV')
            ax_p.set_title('BO progress (green=SC-trig JT, red=strong-coupling, orange=closed)', fontsize=11)
            ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)

            ax_d = axes[3, 1]
            ax_d.scatter(dopings, scores, c=colours, s=40, alpha=0.7)
            ax_d.axvline(dopings[best_idx], color='gold', linewidth=1.5, linestyle='--')
            ax_d.set_xlabel('Doping δ'); ax_d.set_ylabel('score')
            ax_d.set_title('BO: doping vs score', fontsize=11); ax_d.grid(True, alpha=0.3)

            ax_a = axes[3, 2]
            ax_a.scatter(dt_vals, scores, c=colours, s=40, alpha=0.7)
            ax_a.axvline(dt_vals[best_idx], color='gold', linewidth=1.5, linestyle='--')
            ax_a.set_xlabel('Δ_tetra (eV)'); ax_a.set_ylabel('score')
            ax_a.set_title('BO: Δ_tetra vs score', fontsize=11); ax_a.grid(True, alpha=0.3)
        plt.tight_layout()
    return fig

def _plot_phase_data(ax, phase_data: Dict):
    doping = np.array(phase_data['target_doping'])
    M = np.array(phase_data['M'])
    Q = np.array(phase_data['Q']) 
    
    ax.plot(doping, M, 'r-o', linewidth=2, markersize=6, label='AFM (M)')
    ax.plot(doping, Q, 'g-s', linewidth=2, markersize=6, label='JT Distortion (Q)')
    Delta_s_arr = np.array(phase_data['Delta_s'])
    Delta_d_arr = np.array(phase_data['Delta_d'])
    ax.plot(doping, Delta_s_arr,               'b--^', linewidth=1.5, markersize=5, label='Δ_s (on-site B₁g)')
    ax.plot(doping, Delta_d_arr,               'c--v', linewidth=1.5, markersize=5, label='Δ_d (d-wave B₁g)')
    ax.plot(doping, Delta_s_arr + Delta_d_arr, 'b-^',  linewidth=2,   markersize=6, label='|Δ| total')
    
    ax.set_xlabel('Doping δ', fontsize=14)
    ax.set_ylabel('Order Parameters', fontsize=14)
    ax.set_title('Phase Diagram: SC-Activated JT Mechanism', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([doping[0], doping[-1]])
    
    if doping[-1] >= 0.03:
        ax.axvspan(0, 0.03, alpha=0.1, color='red', label='AFM dominant')
    if doping[-1] >= 0.15:
        ax.axvspan(0.05, 0.15, alpha=0.1, color='blue', label='SC+JT coexistence')
    
def _plot_dos(ax, solver: 'RMFT_Solver', result: Dict):
    M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J = (result['M'], result['Q'], result['Delta_s'], result['Delta_d'], result['target_doping'], result['mu'], result['tx'], result['ty'], result['g_J'])

    vbdg = solver._get_vbdg()
    H_stack = vbdg._build_H_stack(solver.k_points, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
    all_energies = np.linalg.eigvalsh(H_stack).ravel()

    ax.hist(all_energies, bins=200, density=True, color='blue', alpha=0.7, label='DOS')
    ax.axvline(x=0.0, color='red', linestyle='--', label='Fermi Level ($E_F$)')
    ax.set_title(
        f"Density of States (DOS)\n"
        f"$\\Delta_{{CF}}={solver.p.Delta_CF:.4f}$ eV, "
        f"Doping δ={target_doping}"
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    hist, bin_edges = np.histogram(all_energies, bins=200, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peaks, _ = find_peaks(hist, prominence=0.1 * np.max(hist))
    vhs_energies = bin_centers[peaks]
    _scf_log("MAIN", f"VHS energies: {vhs_energies}")
    fermi_distance = np.min(np.abs(vhs_energies)) if len(vhs_energies) > 0 else np.inf
    _scf_log("MAIN", f"Closest VHS to Fermi: {fermi_distance:.4f} eV")
    
if __name__ == "__main__":
    with _log_lock:
        print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  SC-Activated JT Model - Variational Free Energy Minimization     ║
    ║  Implements: SC → Γ₆–Γ₇ mixing → JT via ∂F/∂M = ∂F/∂Q = 0         ║
    ║  Optimizer: Unified 5D pipeline (DE→GP→TuRBO→LocalRefine)         ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """, flush=True)

    _blas_threads = _os.environ.get("OMP_NUM_THREADS", "?")
    _scf_log("INIT", f"BLAS/OMP threads={_blas_threads} cpu_count={_os.cpu_count()}  max_parallel_workers≤4")

    params = ModelParams(
        t_pd         = 0.495,
        u            = 15.500,
        lambda_soc   = 0.215,
        Delta_tetra  = -0.140,
        g_JT         = 0.235,
        K_lattice    = 1.400,
        lambda_hop   = 1.280,
        Delta_CT     = 2.000,
        omega_JT     = 0.057,
        Delta_inplane= 0.050,
        mu_LM        = 4.5,
        ALPHA_HF     = 0.2,
        Z            = 4,
        nk           = 80,
        kT           = 0.01,
        a            = 3.8,
        max_iter     = 500,
        tol          = 1e-4,
        mixing       = 0.05,
    )
    params.summary()
    solver        = RMFT_Solver(params)
    target_doping  = 0.21
    doping_margin  = 0.20          # scan covers target ± 20 %
    min_doping     = max(target_doping * (1.0 - doping_margin), _G_T_COHERENCE_MIN / (2.0 - _G_T_COHERENCE_MIN))
    max_doping     = target_doping * (1.0 + doping_margin)
    supposed_M     = solver._estimate_M0(target_doping)
    initial_Q      = 1e-5
    initial_Delta  = 1e-5

    # ── Section 1: Reference SCF ─────────────────────────────────────────────
    # Run SCF first.  All subsequent diagnostics use the self-consistent (M, μ)
    _scf_log("REF-SCF", "="*60)
    _scf_log("REF-SCF", f"Reference SCF at δ={target_doping:.3f}")
    _ref_result = None
    try:
        _ref_result = solver.solve_self_consistent(
            target_doping,
            initial_M     = supposed_M,
            initial_Q     = initial_Q,
            initial_Delta = initial_Delta,
            verbose       = False,
        )
        _ref_M     = _ref_result['M']
        _ref_Q     = _ref_result['Q']
        _ref_mu    = _ref_result['mu']
        _ref_g_t   = _ref_result['g_t']
        _ref_Delta = _ref_result['Delta_s'] + _ref_result['Delta_d']
        _ref_conv  = _ref_result['converged']
        _ref_mott  = _ref_result.get('mott_suspect', False)
        _scf_log("REF-SCF", f"  converged={_ref_conv}  mott_suspect={_ref_mott}")
        _scf_log("REF-SCF", f"  M={_ref_M:.4f}  Q={_ref_Q:+.5f}  |Δ|={_ref_Delta:.5f} eV"
                 f"  μ={_ref_mu:.4f} eV  g_t={_ref_g_t:.3f}")
    except Exception as _ref_err:
        _scf_log("REF-SCF", f"  Reference SCF failed: {_ref_err}")
        _ref_M   = supposed_M
        _ref_mu  = 0.0
        _ref_Q   = initial_Q; _ref_Delta = 0.0
        _ref_conv = False; _ref_mott = False

    # ── Section 2: G-matrix at self-consistent M ────────────────────────────
    _scf_log("G-MATRIX", "="*60)
    G_base = solver.compute_G_instability(target_doping=target_doping, M=_ref_M)
    _scf_log("G-MATRIX", f"h_afm={G_base['h_afm']:.4f} eV")
    _scf_log("G-MATRIX", f"χ_ΔΔ (dom)={G_base['chi_DD_dom']:.4f}  χ_DD_s={G_base['chi_DD_s']:.4f}"
             f"  χ_DD_d={G_base['chi_DD_d']:.4f}  χ_DD_sd={G_base['chi_DD_sd']:.4f}  [eV⁻¹]")
    _scf_log("G-MATRIX", f"χ_ΔQ (dom)={G_base['chi_DQ_dom']:.4f}  χ_ΔQ_s={G_base['chi_DQ_s']:.4f}"
             f"  χ_ΔQ_d={G_base['chi_DQ_d']:.4f}  [eV⁻¹]")
    _scf_log("G-MATRIX", f"χ_QQ(normal)={G_base['chi_QQ']:.4f} eV⁻¹")
    _scf_log("G-MATRIX", f"N_eff={G_base['N_eff']:.4f} eV⁻¹  K_eff={G_base['K_eff']:.4f} eV/Å²")
    _scf_log("G-MATRIX", f"3×3 eigs: [{G_base['eigs3'][0]:.4f},{G_base['eigs3'][1]:.4f},{G_base['eigs3'][2]:.4f}]")
    _scf_log("G-MATRIX", f"evec_min=[{G_base['evec_min'][0]:.3f},{G_base['evec_min'][1]:.3f},"
             f"{G_base['evec_min'][2]:.3f}]  → instab_dir: {G_base['instab_dir']}")
    _scf_log("G-MATRIX", f"G11={G_base['G11']:.4f}  G3[2,2]={G_base['G22']:.4f}  G12={G_base['G12']:.4f}"
             f"  dom={G_base['dominant']}")
    _lmin_val  = G_base['lambda_min']
    _g22_val   = G_base['G22']
    _lmin_note = ("✗ SPONTANEOUS instability" if _lmin_val <= 0
                  else ("⚠ near-critical (0 < λ_min < 0.1)" if _lmin_val < 0.1
                        else "✓ normal-state stable"))
    _g22_note  = "✓ G22>0: spontaneous JT blocked" if _g22_val > 0 else "✗ G22≤0: spontaneous JT risk"
    _scf_log("G-MATRIX", f"λ_min={_lmin_val:.4f}  [{_lmin_note}]")
    _scf_log("G-MATRIX", f"G3[2,2]={_g22_val:.4f}  [{_g22_note}]")
    _lambda_eff = G_base['lambda_eff']
    _leff_status = ("✓ optimal" if 0.3 < _lambda_eff < 1.0
                    else ("⚠ weak — increase J_eff (↓u or ↑t_pd/Δ_CT)" if _lambda_eff <= 0.3
                          else "⚠ too strong — risk of spontaneous JT / AFM QCP"))
    _scf_log("G-MATRIX", f"λ_eff=N_eff·V_eff={_lambda_eff:.4f}  [{_leff_status}]")
    _scf_log("G-MATRIX", f"∂²F/∂Q²|Δ=0={G_base['d2F_Q_normal']:+.5f} eV/Å²  "
             f"{'✓ normal-state Q-stable' if G_base['d2F_Q_normal'] > 0 else '✗ spontaneous JT!'}")
    _dlam_dQ = G_base['dlambda_pair_dQ']
    _dlam_note = ("✓ JT renormalises V_pair upward" if (not np.isnan(_dlam_dQ) and _dlam_dQ > 0)
                  else ("✗ JT suppresses V_pair" if (not np.isnan(_dlam_dQ) and _dlam_dQ < 0)
                        else "n/a"))
    _scf_log("G-MATRIX", f"∂λ_pair/∂Q={_dlam_dQ:+.5f}  [{_dlam_note}]")
    _scf_log("G-MATRIX", f"||[τ_x,H]||={G_base['comm_norm']:.4f} eV  blocking={G_base['blocking_ratio']:.4f}")

    # ── Section 3: Linearised gap equation result — from SCF result dict ────
    _lmax_ref  = float(_ref_result.get('lambda_max', float('nan')))  if _ref_result else float('nan')
    _gsym_ref  = _ref_result.get('gap_symmetry', '?')                if _ref_result else '?'
    _lraw_ref  = float(_ref_result.get('lambda_max_raw', float('nan'))) if _ref_result else float('nan')
    _gdel_ref  = float(_ref_result.get('g_delta_dom', float('nan')))    if _ref_result else float('nan')
    _V_spin    = float(_ref_result.get('V_spin_mean',  float('nan')))   if _ref_result else float('nan')
    _V_JT      = float(_ref_result.get('V_JT_mean',   float('nan')))   if _ref_result else float('nan')
    _V_cr      = float(_ref_result.get('V_cross_mean', float('nan')))   if _ref_result else float('nan')
    _V_tot     = float(_ref_result.get('V_rpa_mean',  float('nan')))   if _ref_result else float('nan')
    _gap_vec   = _ref_result.get('gap_vector', None) if _ref_result else None
    _fs_pts    = _ref_result.get('fs_pts', None)     if _ref_result else None

    _scf_log("G-MATRIX", "Linearised gap equation (from SCF; full RPA vertex, self-consistent M and μ):")
    _scf_log("G-MATRIX", f"  λ_max={_lmax_ref:.4f} (raw={_lraw_ref:.4f}"
             f"  × g_Δ={_gdel_ref:.3f})  sym={_gsym_ref}")
    if _gap_vec is not None and _fs_pts is not None:
        _phi_s = np.ones(len(_fs_pts)); _phi_s /= np.linalg.norm(_phi_s)
        _phi_d = np.cos(_fs_pts[:,0]) - np.cos(_fs_pts[:,1])
        _nd = np.linalg.norm(_phi_d)
        if _nd > 1e-10: _phi_d /= _nd
        _w_s = float(abs(_gap_vec @ _phi_s))
        _w_d = float(abs(_gap_vec @ _phi_d))
        _norm = max(_w_s + _w_d, 1e-10)
        _ls = _lmax_ref * _w_s / _norm
        _ld = _lmax_ref * _w_d / _norm
        # d-wave: negative FS-average is EXPECTED because forward scattering q≈0 is repulsive and dominates the mean;
        # the instability comes from backscattering at q≈(π,π) which is captured by the dominant eigenvector
        _ch_note = ('d-wave dominant' if _w_d > _w_s else 's-wave dominant')
        _neg_note = ('  [⚠ λ<0: FS-avg vertex repulsive — instability requires nodal sign change]'
                     if _lmax_ref < 0 else '')
        _scf_log("G-MATRIX", f"  Channel decomp: λ_s={_ls:.4f}  λ_d={_ld:.4f}"
                 f"  [{_ch_note}]{_neg_note}")

    # Moriya/Stoner diagnostic — from G_base (self-consistent M)
    _t_eff_proxy = float(params.t0 * (2.0 * abs(target_doping)) / (1.0 + abs(target_doping) + 1e-9))
    _J_eff_log   = G_base['J_eff']
    _alpha_M_log = _moriya_alpha(target_doping, _t_eff_proxy, float(_J_eff_log))
    _chi_SS_log  = G_base['chi_DD_s']
    _chi_SS_M    = _chi_SS_log / (1.0 + _alpha_M_log * max(float(_J_eff_log), 1e-9) * _t_eff_proxy * max(_chi_SS_log, 0.0))
    _stoner      = _J_eff_log * _chi_SS_M
    _ston_status = ('✓ near QCP' if 1.0 > _stoner > 0.7
                    else ('⚠ near/past AFM QCP' if 2.0 > _stoner >= 1.0
                          else ('safe' if _stoner <= 0.7 else '✗ deeply past QCP')))
    _scf_log("G-MATRIX", f"  Moriya: α_M={_alpha_M_log:.3f}  χ_SS(bare)={_chi_SS_log:.4f}"
             f"  χ_SS(Moriya)={_chi_SS_M:.4f}")
    _scf_log("G-MATRIX", f"  J_eff·χ_SS(Moriya)={_stoner:.4f}  [{_ston_status}]")
    if np.isfinite(_V_tot) and abs(_V_tot) > 1e-4:
        # V_RPA(FS-avg) < 0 is expected for d-wave: forward scattering (q≈0) is repulsive and dominates the Fermi-surface average.
        _v_note = ('⚠ V_avg<0: d-wave backscattering dominant (normal)' if _V_tot < 0
                   else '✓ positive avg')
        _scf_log("G-MATRIX", f"  V_RPA(FS-avg)={_V_tot:.4f} eV  [{_v_note}]"
                 f"  spin={_V_spin/_V_tot*100:.0f}%  JT={_V_JT/_V_tot*100:.0f}%"
                 f"  cross={_V_cr/_V_tot*100:.0f}%")
    elif np.isfinite(_V_tot):
        _scf_log("G-MATRIX", f"  V_RPA(FS-avg)={_V_tot:.2e} eV"
                 f"  [spin={_V_spin:.3f}  JT={_V_JT:.3f}  cross={_V_cr:.3f} eV]")

    # ── Section 4: SC-JT window — chi_tau from SCF if available ─────────────
    if _ref_result is not None and _ref_result.get('chi_tau') is not None:
        _chi_tau_ref = float(_ref_result['chi_tau'])
        _chi_tau_src = "SC-enhanced (from converged SCF)"
    else:
        _chi_tau_ref = solver._compute_chi_tau(_ref_M, _ref_Q, target_doping)['chi_tau']
        _chi_tau_src = "normal-state estimate (SCF not converged)"
    _jt_win = check_sc_jt_window(
        g_JT       = solver.p.g_JT,
        Delta_CF   = solver.p.Delta_CF,
        chi_tau    = _chi_tau_ref,
        chi0       = G_base['chi_QQ'] / max(solver.p.g_JT**2, 1e-12),
        K_lattice  = solver._K_bare,
        K_eff      = G_base['K_eff'],
        lambda_min = max(G_base['lambda_min'], 1e-4),
    )
    _scf_log("G-MATRIX", f"SC-JT window  [χ_τ source: {_chi_tau_src}]:")
    _scf_log("G-MATRIX", f"  K_spont={_jt_win['K_spont']:.4f}  K_SC={_jt_win['K_SC']:.4f}"
             f"  K_opt={_jt_win['K_opt']:.4f}  K_lattice={solver._K_bare:.4f}")
    _scf_log("G-MATRIX", f"  λ_JT={_jt_win['lambda_JT']:.4f}  λ_JT_opt={_jt_win['lambda_JT_opt']:.4f}"
             f"  K_dist={_jt_win['K_distance']:+.3f}"
             f"  in_window={_jt_win['K_in_window']}  open={_jt_win['window_open']}")
    _scf_log("G-MATRIX", f"  → {_jt_win['note']}")

    # ── Section 5: Gap=0 diagnosis — from G-matrix and SCF result ───────────
    _gap_zero = _ref_Delta < params.tol * 10
    _dlam_g   = G_base.get('dlambda_pair_dQ', float('nan'))
    if _gap_zero and np.isfinite(_lmax_ref) and _lmax_ref > 0.5:
        if np.isfinite(_V_tot) and abs(_V_tot) < 0.05:
            _cause = f"(A) V_RPA={_V_tot:.4f} eV too small — consider ↓u or ↑t_pd"
        elif _ref_mott:
            _cause = f"(B) Mott-suspect: g_t={_ref_g_t:.3f}"
        elif abs(_ref_mu - G_base['mu_n']) > 0.1:
            _cause = f"(C) μ shift: pre-SCF mu_n={G_base['mu_n']:.4f} → SC μ={_ref_mu:.4f}"
        elif np.isfinite(_dlam_g) and abs(_dlam_g) < 1e-4:
            _cause = f"(D) ∂λ/∂Q≈0: JT has no upward effect on V_pair at this point"
        else:
            _cause = f"(D) ∂λ/∂Q={_dlam_g:+.4f} eV/Å — sensitivity present but λ below SCF threshold"
        _scf_log("G-MATRIX", f"⚠ GAP=0 despite λ_max={_lmax_ref:.3f} — likely cause: {_cause}")
    elif _ref_Delta > params.tol * 10:
        _scf_log("G-MATRIX", f"✓ SC gap found at reference params  |Δ|={_ref_Delta:.5f} eV")
    else:
        _scf_log("G-MATRIX", f"Gap=0 (λ_max={_lmax_ref:.3f} < threshold) — expected, DE will search")
    _scf_log("G-MATRIX", "="*60)

    # ── Section 6: Unified 5D optimisation ──────────────────────────────────
    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", "UNIFIED 5D OPTIMISATION  (DE scout → GP seed → TuRBO → local refine)")
    _scf_log("MAIN", "Search space: (Delta_tetra, lambda_soc, u, g_JT, t_pd)  — no parameter splitting")
    _scf_log("MAIN", "="*60)

    _5d_bounds = {
        'Delta_tetra': (-0.22, -0.07),
        'lambda_soc':  ( 0.13,  0.26),
        'u':           ( 11.0,  20.0),
        'g_JT':        ( 0.19,  0.26),
        't_pd':        ( 0.40,  0.60),
    }

    unified_bo = UnifiedBayesianOptimizer(solver, n_doping_scan=7)
    res_unified = unified_bo.optimize(
        doping_bounds    = (min_doping, max_doping),
        param_bounds_5d  = _5d_bounds,
        de_popsize       = 10,    # total population = 10 * 5 = 50 candidates/generation
        de_maxiter       = 50,    # DE maximum generations
        gp_seed_top_k    = 12,    # DE candidates evaluated with full SCF for GP seeding
        turbo_iterations = 30,    # TuRBO iteration count
        turbo_batch      = 3,     # parallel proposals per TuRBO iteration
        local_refine     = True,  # enable Phase 4 local refinement
        local_n_grid     = 10,    # Phase 4 grid density
        verbose          = True,
    )

    best_final = res_unified['best_valid'] or res_unified['best_point']
    all_obs    = res_unified['observations']

    # ── Section 4: SC-triggered JT causality test on top-5 converged points ─────
    _scf_log("MAIN", "SC-triggered JT causality test (top-5 converged points)...")
    _top5 = sorted([o for o in all_obs if o.converged],
                   key=lambda o: o.score, reverse=True)[:5]
    for top_pt in _top5:
        if top_pt.result:
            _ls_val = top_pt.lambda_soc or params.lambda_soc
            s_test  = copy.copy(solver); s_test.p = copy.copy(solver.p)
            s_test.p.Delta_tetra = top_pt.Delta_tetra
            s_test.p.lambda_soc  = _ls_val
            s_test.p.u           = top_pt.u
            s_test.p.g_JT        = top_pt.g_JT
            s_test.p.t_pd        = top_pt.t_pd
            s_test.p.__post_init__()
            s_test._K_bare = s_test.p.K_lattice
            s_test._rebuild_orbital_operators(s_test.p)
            s_test._reset_transient_state()
            ct = unified_bo._jt_causality_test(s_test, top_pt.result)
            _scf_log("CAUSAL",
                     f"Δ_tet={top_pt.Delta_tetra:.3f}  λ_soc={_ls_val:.4f}"
                     f"  u={top_pt.u:.2f}  g_JT={top_pt.g_JT:.3f}"
                     f"  t_pd={top_pt.t_pd:.4f}"
                     f"  Tc={top_pt.Tc*1000:.2f}meV → {ct['note']}")
            _scf_log("CAUSAL",
                     f"  G22_N={ct['G22_normal']:.4f}"
                     f"  λ_min(H)={ct['lmin_sc_hessian']:+.4f}"
                     f"  hess={'✓' if ct['hess_confirmed'] else '✗'}"
                     f"  L1={'✓' if ct.get('level1_ok') else '✗'}"
                     f"  L2={'✓' if ct.get('level2_ok') else '✗'}")

    # ── Global optimum summary ───────────────────────────────────────────────
    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", f"OPTIMISATION COMPLETE  ({res_unified['elapsed_s']/60:.1f} min total)")
    _scf_log("MAIN", "Global optimal parameters:")
    _best_lsoc = best_final.lambda_soc or params.lambda_soc
    _scf_log("MAIN", f"  Δ_tet={best_final.Delta_tetra:.4f}"
                     f"  λ_soc={_best_lsoc:.4f}"
                     f"  u={best_final.u:.4f}"
                     f"  g_JT={best_final.g_JT:.4f}"
                     f"  t_pd={best_final.t_pd:.4f} eV"
                     f"  K_latt={params.K_lattice:.4f} eV/Å²")
    _scf_log("MAIN", f"  |Δ|={best_final.Delta_total:.6f} eV"
                     f"  Tc={best_final.Tc*1000:.2f} meV"
                     f"  score={best_final.score:.6f}")

    # ── Section 5: Rebuild solver at global optimum and run final diagnostics ────
    params.Delta_tetra = best_final.Delta_tetra
    params.lambda_soc  = _best_lsoc
    params.u           = best_final.u
    params.g_JT        = best_final.g_JT
    params.t_pd        = best_final.t_pd
    params.__post_init__()
    solver_opt = RMFT_Solver(params)

    try:
        _sc_opt = solver_opt.solve_self_consistent(
            best_final.doping,
            initial_M     = supposed_M,
            initial_Q     = initial_Q,
            initial_Delta = initial_Delta,
            verbose       = False)
        _M_opt  = float(_sc_opt['M'])
        _G_opt  = solver_opt.compute_G_instability(best_final.doping, M=_M_opt)
        _Tc_res = solver_opt.compute_Tc_by_gap_suppression(
            best_final.doping, sc_result=_sc_opt)
        _Tc_lin = float(_Tc_res['Tc'])
        _hess   = _sc_opt['hessian']
        _lmin   = _hess['min_curvature'] if _hess['min_curvature'] is not None else float('nan')

        _scf_log("MAIN", "── Diagnostics at global optimum ──")
        _scf_log("MAIN", f"Tc(BdG bisect)={_Tc_lin*1000:.2f} meV ({_Tc_lin*11604:.1f} K)")
        _scf_log("MAIN",
                 f"Tc(G-BCS)={_G_opt['Tc_estimate']*1000:.2f} meV"
                 f"  λ_min(G3,Δ=0)={_G_opt['lambda_min']:.4f}"
                 f"  G22={_G_opt['G22']:.4f}")
        _scf_log("MAIN",
                 f"SC Hessian λ_min(H_3x3,Δ≠0)={_lmin:+.4f}"
                 f"  {'✓ SC-triggered JT confirmed' if _lmin < 0 else '— JT not triggered'}")
        _sc_jt_confirmed = _sc_opt['sc_jt_confirmed']
        _hess_lmin_sc    = _sc_opt['hessian_lmin_sc']
        _scf_log("MAIN",
                 f"Post-SCF λ_min(Hessian_SC)={_hess_lmin_sc:+.4f}"
                 f"  {'✓ SC-triggered JT CONFIRMED (Δ≠0, Q≠0, converged)' if _sc_jt_confirmed else '— JT not triggered in converged state'}")
        _xi_res = _sc_opt['coherence']
        _scf_log("MAIN",
                 f"Coherence: ξ/a={_xi_res['xi_over_a']:.2f}"
                 f"  {'✓ BdG valid' if _xi_res['valid_BdG'] else '⚠ BdG marginal (ξ/a < 2)'}"
                 f"  orbital_selective={'✓' if _xi_res['orbital_selective'] else '—'}")
        if _xi_res['orbital_selective']:
            _scf_log("MAIN",
                     f"  ξ_Γ6={_xi_res['xi_Gamma6']*1e10:.1f} a  "
                     f"ξ_Γ7={_xi_res['xi_Gamma7']*1e10:.1f} a  "
                     f"[orbital-selective SC: JT-driven tx≠ty enhances Γ₆/Γ₇ selectivity]")
        _dlam_opt = _G_opt['dlambda_pair_dQ']
        _scf_log("MAIN",
                 f"∂λ_pair/∂Q={_dlam_opt:+.5f}"
                 f"  [{'✓ JT renormalises V_pair upward → indirect Tc boost' if (not np.isnan(_dlam_opt) and _dlam_opt > 0) else '⚠ JT does not boost V_pair at this point'}]")
        _gr = solver_opt.compute_gap_ratio(best_final.doping, _sc_opt)
        _scf_log("MAIN", f"2Δ₀/kTc = {_gr['ratio_2D']:.3f}  [{_gr['coupling_regime']}]"
                         f"  Δ₀={_gr['Delta_0']*1000:.2f} meV  Tc={_gr['Tc_K']:.1f} K")
        try:
            _lT = solver_opt.compute_lambda_vs_T(best_final.doping, _sc_opt)
            _scf_log("MAIN",
                     f"Tc(λ=1)={_lT['Tc_lambda']*1000:.2f} meV"
                     f"  slope={_lT['slope_at_Tc']*1000:.3f} meV⁻¹")
        except Exception as _le:
            _scf_log("MAIN", f"λ_max(T) curve failed: {_le}")
    except Exception as _e:
        _sc_opt = None
        _M_opt  = supposed_M
        _scf_log("MAIN", f"Optimal-point SCF failed: {_e}")

    _scf_log("MAIN", "="*60)

    fig = plot_phase_diagrams(
        solver_opt,
        initial_M     = _M_opt,
        initial_Q     = initial_Q,
        initial_Delta = initial_Delta if _sc_opt is None else (
            _sc_opt['Delta_s'] + _sc_opt['Delta_d']),
        doping_range  = np.linspace(min_doping, max_doping, 10),
        opt_result    = {'observations': all_obs},
    )
    plt.show()