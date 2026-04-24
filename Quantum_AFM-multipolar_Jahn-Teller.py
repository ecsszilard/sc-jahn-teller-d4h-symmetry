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
import math
import concurrent.futures
import sys
import threading as _threading
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from threadpoolctl import threadpool_limits as _tpl_ctx

_log_lock = _threading.Lock()

# ── Module-level physical and numerical constants ─────────────────────────────
# RPA / Moriya damping
_ALPHA_MORIYA:      float = 0.05    # Moriya mode-coupling coefficient baseline: Γ_M = α_M · J_eff · t_eff
                                    # The *effective* α_M is computed at runtime by moriya_gamma();
_MORIYA_C:          float = 0.45    # dimensionless prefactor in α_M = C · δ · (t_eff / J_eff); tuned to give α_M ~ 0.175 at δ~0.15, t_eff/J_eff ~ 3
                                    # stronger Moriya damping ensures det never reaches _RPA_DET_FLOOR in the physically relevant near-QCP regime,
                                    # so the hard floor truly becomes a numerical-only emergency and does not cut the critical scaling.
_CHI_DQ_S_PADE_W:   float = 0.10    # width parameter w for the Padé χ_SQ regularisation: chi_SQ_v = chi_SQ / (1 + |chi_SQ| / w)
                                    # linear at |chi_SQ| ≪ w, saturates asymptotically to ±w, smooth gradient near QCP
_LAMBDA_JT_VIABLE:  float = 0.05    # minimum λ_JT = g²·χ_τ/K for SC-triggered JT viability.
                                    # Sets K_SC = g²·χ_τ_sc / _LAMBDA_JT_VIABLE (the K_lattice upper bound above which λ_JT drops below the threshold).
_RPA_DET_REG:       float = 1e-9    # Floor for 2×2 RPA determinant when det>0 but tiny (exact QCP numerical zero). Only applied when det>0 — negative det means past QCP and is left intact.
_RPA_DET_WARN:      float = 0.09    # QCP proximity warning threshold for diagnostics and SCF adaptive mixing.
_RPA_V_SOFT_CAP:    float = 50.0    # eV — Padé soft cap on |Vp| when det < 0; preserves divergence character but prevents overflow
_RPA_QCP_PENALTY:   float = 0.30    # fraction by which α is reduced per unit of |det_afm| < 0 past QCP
                                    # Used in: SCF near-critical detection, BO near_qcp flag
_CHI_DQ_S_EPS:      float = 1e-12   # numerical noise threshold for χ_SQ/χ_QS symmetry enforcement
_CHI0_CACHE_TOL:    float = 1e-5    # parameter-change threshold for cache invalidation
_VAR_MIN:           float = 1e-10   # variance minimum to avoid overshoot instability

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
_FS_N_VERTEX:       int   = 100     # FS k-points used in the vertex q-loop.
                                    # samples the full k-grid, angular resolution need to resolve the d-wave node at (π/2,π/2) and the B₁g anti-nodal hot spots.
_Q_THR_REL:         float = 0.04    # fraction of lambda_hop; Q change below this skips vertex rebuild
_M_THR_REL:         float = 0.05    # absolute M change threshold (M is O(0.1–0.5))
_MU_LM:             float = 3.5     # Levenberg–Marquardt floor for M Newton step (default 4.0), larger → smaller γ_M → more conservative M update.
_ALPHA_RENORM:      float = 0.4     # smooths out noise from quantum fluctuations
_ALPHA_HF:          float = 0.25    # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton; default 0.2)
_NK:                int   = 70      # k-grid points per direction (even required for commensurate q_AFM=(π,π))
_CLUSTER_SIZE:      int   = 2
_MAX_ITER:          int   = 800
_MIN_ITER:           int   = 4
_MIXING:            float = 0.05

_Q_UPDATE_PERIOD:   int   = 4      # update Q every N inner iterations
_Q_INNER_TOL_FAC:   float = 5.0    # inner (Δ,M,μ) convergence tolerance = tol * this factor
_ALPHA_MIX_2X2:     float = 0.35   # blend weight of 2x2 eigenvector direction vs fixed-point gap update, 0 = pure fixed-point; 1 = pure 2x2 eigenvector
_DQ_FS_VERTEX:      float = 0.02   # SC-triggered JT FS sensitivity, finite-difference step for ∂V(k,k')/∂Q on FS

# Limit-cycle detector
_CYCLE_WINDOW:      int   = 20     # iteration window to detect oscillation
_CYCLE_THRESHOLD:   float = 0.30   # std/mean of |Δ| above this → oscillatory regime
_CYCLE_DAMP_FAC:    float = 0.50   # alpha reduction factor when oscillation detected

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
_BO_SPONT_JT_PEN:   float = 0.05    # penalty floor in g22_f (used only in _g_fallback_score)
_BO_G_FALLBACK:     float = 5e-3    # overall scale for G-matrix proxy (no-gap region)
_BO_SIGMOID_W:      float = 0.30    # sigmoid width for g22_f gate (fallback-only)
_BO_SC_HESS_SIG:    float = 0.05    # eV — sc_hessian_f sigmoid width around lambda_min=0
_BO_JCHI_NOISE:     float = 0.05    # J·χ below this is numerical noise, apply floor
_BO_G22_MARGIN_CTR: float = 0.25    # G22 sweet-spot centre for g22_margin_f sigmoid
_BO_G22_MARGIN_W:   float = 0.15    # sigmoid width for g22_margin_f
_DE_G22M_SAFE:      float = 0.25    # G22 value considered safely above spontaneous-JT boundary

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
    Delta_inplane: float      # eV    B2g in-plane anisotropy Δ_ip·(Lx²-Ly²); splits Γ₇ into Γ₇a+Γ₇b.
                              #       At Δ_ip=0 (D₄h) B1g_op is a pure singlet (spin-flip), so JT is
                              #       strictly SC-triggered. Finite Δ_ip (D₂h) adds spin-conserving and
                              #       diagonal elements to B1g_op, partially activating JT without SC.
    Delta_CT:      float      # eV    charge-transfer gap (ZSA scale); sets scale for CT-insulator crossover
    omega_JT:      float      # eV    JT phonon frequency (40–80 meV); enters only D_phonon = 2/ω_JT
                              #       All free-energy magnitudes use adiabatic g²/K

    # --- Numerics ---
    Z:             int        # 2D square lattice coordination number
    kT:            float      # eV  temperature — keep kT < Tc to allow gap to open;
    tol:           float

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
        self.eta        = float(np.clip(_me7 / max(_me6, 1e-9), 0.1, 5.0))  # η_Sz
        self.multi_op = self.build_multipolar_operator()

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
        self.U_mf: float = self.Z * _J_ct / 2.0  # bare Weiss amplitude
        self.J_CT: float = _J_ct
        assert _NK % 2 == 0, f"_NK={_NK} must be even for commensurate q_AFM=(π,π)"

        k_scf          = np.linspace(-np.pi, np.pi, _NK, endpoint=False)
        KX_scf, KY_scf = np.meshgrid(k_scf, k_scf)
        self.k_points  = np.column_stack((KX_scf.flatten(), KY_scf.flatten()))
        self.N_k       = len(self.k_points)
        self.k_weights = _uniform_bz_weights_2d(_NK, _NK)   # uniform 1/N weights (periodic BZ)

        # χ₀ even grid — kept separate for the commensurate q_AFM=(π,π) index trick.
        k_even = np.linspace(-np.pi, np.pi, _NK, endpoint=False)
        KX_ev, KY_ev = np.meshgrid(k_even, k_even)
        self.k_points_even  = np.column_stack((KX_ev.flatten(), KY_ev.flatten()))
        self.N_k_even       = len(self.k_points_even)
        self.k_weights_even = _uniform_bz_weights_2d(_NK, _NK)

        # General shift-index table for ALL q-vectors on the 2π/_NK grid.
        # For q = (nx, ny) * 2π/_NK the k+q grid is a cyclic PERMUTATION of k_even:
        #   E(k+q)[i] = E(k)[shift_table[nx, ny, i]]
        _flat   = np.arange(self.N_k_even)
        _kx_idx = _flat % _NK
        _ky_idx = _flat // _NK
        _nx = np.arange(_NK)[:, None, None]   # (_NK, 1, 1)
        _ny = np.arange(_NK)[None, :, None]   # (1, _NK, 1)
        self.shift_table = (
            ((_ky_idx[None, None, :] + _ny) % _NK) * _NK
          + ((_kx_idx[None, None, :] + _nx) % _NK)
        ).astype(np.int32)   # (_NK, _NK, N_k_even)

    def estimate_M0(self, target_doping: float, lambda_lin_max: float) -> float:
        """
        Warm-start AFM order-parameter estimate.
        """
        abs_d     = max(abs(target_doping), 1e-6)
        g_J       = 4.0 / (1.0 + abs_d) ** 2
        t_eff     = self.t0 * (2.0 * abs_d) / (1.0 + abs_d)
        bandwidth = 8.0 * max(t_eff, 1e-6)
        N0        = 2.0 / bandwidth
        J_eff     = g_J * self.Z * self.J_CT
        S         = float(np.clip(J_eff * N0, 0.0, 5.0))
        M_stoner  = float(np.clip(0.18 * (S / max(S, 1.0)) * g_J / 4.0, 0.05, 0.20))
        delta_c   = 0.23
        M_stoner  *= max(1.0 - abs_d / delta_c, 0.0)
        M_prior   = float(np.clip(0.18 - 0.40 * (target_doping - 0.06), 0.08, 0.22))
        w         = float(np.clip(abs_d / 0.20, 0.0, 1.0))
        M0        = (1.0 - w) * M_stoner + w * M_prior

        # Supercritical correction: when λ_lin > 1 the system is past the AFM QCP.
        # The linearised gap eigenvalue λ_lin overestimates the Stoner parameter because it includes vertex corrections and JT coupling.
        if lambda_lin_max > 1.0:
            JN = float(np.clip(lambda_lin_max, 1.001, 20.0))
            # Solve M* = tanh(JN · M*) by fixed-point iteration
            M_sc = 0.5
            for _ in range(50):
                M_sc_new = float(np.tanh(JN * M_sc))
                if abs(M_sc_new - M_sc) < 1e-10:
                    break
                M_sc = M_sc_new
            
            M_stoner_sc = float(np.clip(M_sc, 0.0, 0.45))
            
            # Blend weight: grows with λ_lin, capped at 0.75
            _w_sc = float(np.clip((lambda_lin_max - 1.0) / 15.0, 0.0, 0.75))
            M0 = (1.0 - _w_sc) * M0 + _w_sc * M_stoner_sc
        return float(np.clip(M0, 0.02, 0.45))
    
    def get_gutzwiller_factors(self, target_doping: float) -> Tuple[float, float, float, float]:
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
        • η-weight (AFM selection rule) is already included in _exchange_channels → V_d_scalar.
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
        abs_delta = max(abs(target_doping), 1e-6)
        g_t       = (2.0 * abs_delta) / (1.0 + abs_delta)
        g_J       = 4.0 / ((1.0 + abs_delta) ** 2)
        g_Delta_s = g_t   # s-channel is kinetic in origin → follows g_t

        # Γ₇ spectral weight from SOC+CF eigenvectors: U_gamma column order: [Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓, Γ₇b↑, Γ₇b↓] (6D t2g basis).
        # p_7 = (Σ |U_gamma[i,2:4]|² over Γ₆ indices) / 2  measures Γ₇ admixture in the Γ₆ doublet due to SOC.
        # p_7 is a single-ion SOC+CF property that determines how strongly the exchange J_B1g renormalises the B₁g pairing vertex.
        # Computed using the truncated SOC matrix _U4 (Γ₆/Γ₇ subspace): _U4[:,0], _U4[:,1] → Γ₆↑, Γ₆↓ eigenvectors, rows 2–3  → Γ₇ components
        # Weight of Γ₇ character (rows 2:4) in the Γ₆ doublet (columns 0:2)
        p_7_up = float(np.sum(np.abs(self._U4[2:4, 0])**2))   # Γ₇ weight in Γ₆↑ eigenvec
        p_7_dn = float(np.sum(np.abs(self._U4[2:4, 1])**2))   # Γ₇ weight in Γ₆↓ eigenvec
        p_7    = 0.5 * (p_7_up + p_7_dn)                # average ∈ [0, 0.5]

        # Interpolation: w_norm = p_7 / 0.5 ∈ [0, 1]
        w_norm    = float(np.clip(p_7 / 0.5, 0.0, 1.0))
        g_Delta_d = float(np.clip(g_t + (g_J - g_t) * w_norm, g_t, g_J))
        return g_t, g_J, g_Delta_s, g_Delta_d
    
    def moriya_gamma(self, doping: float, t_eff: float, J_eff: float) -> float:
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
        Gamma_M = float(max(alpha, _ALPHA_MORIYA)) * J_safe * t_eff
        return  min(Gamma_M, 4.0 * t_eff / np.pi)

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
    
    def effective_superexchange(self, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> float:
        """
        Gutzwiller-renormalised single-bond ZSA superexchange (without cluster vertex correction).

        J_bare = g_J · f_J(δ) · J_CT(tx, ty)

        f_J(δ) = max(δ,δ₀) / (max(δ,δ₀) + δ₀)  — ZRS coherence floor.
            Saturates at f_J=0.5 as δ→0 (ZRS band incoherent but local J survives).
            Z·g_J·f_J together give J_eff → Z·g_J(0)·0.5·J_CT = Z·2·J_CT
            at half-filling, consistent with the Mott limit.
        """
        abs_doping = max(abs(doping), 1e-6)
        d_fl = max(abs_doping, self.doping_0)          # floor at doping_0
        f_J  = d_fl / (d_fl + self.doping_0)           # ∈ [0.5, 1)
        _dct = max(self.Delta_CT, 1e-9)
        _U   = max(self.U, 1e-9)
        return g_J * f_J * (tx_bare**2 + ty_bare**2) * (1.0/_U + 1.0/(_dct + _U/2.0))
    
    def build_multipolar_operator(self) -> np.ndarray:
        P6_diag = np.array([1.0, 1.0, 0.0, 0.0])    # Projects to 6↑, 6↓
        P7_diag = np.array([0.0, 0.0, 1.0, 1.0])    # Projects to 7↑, 7↓
        sz_diag = np.array([1.0, -1.0, 1.0, -1.0])  # Spin polarization σz: ↑=+1, ↓=-1
        O_diag = (P6_diag + self.eta * P7_diag) * sz_diag
        return np.diag(O_diag)

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
                   Splits the Γ₇ quartet into two Kramers doublets (Γ₇a, Γ₇b),
                   preventing spontaneous JT from the 4-fold degenerate Γ₇ level.

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

def _uniform_bz_weights_2d(nx: int, ny: int) -> np.ndarray:
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

class RMFT_Solver:

    def __init__(self, params: ModelParams):
        self.p = params

        self.k_points       = params.k_points
        self.k_points_even  = params.k_points_even
        self.N_k            = params.N_k
        self.N_k_even       = params.N_k_even
        self.k_weights      = params.k_weights
        self.k_weights_even = params.k_weights_even
        self.shift_table    = params.shift_table   # (nk, nk, N_k_even) int32 — cyclic shift index

        # Orbital operators + sz_op and sz_bdg16 derived from the SOC+CF eigenbasis.
        self._rebuild_orbital_operators()

        self.phi_k = (np.cos(self.k_points[:, 0])
                      - np.cos(self.k_points[:, 1]))
        self.phi_k_even = (np.cos(self.k_points_even[:, 0])
                           - np.cos(self.k_points_even[:, 1]))

        K_spont   = params.g_JT**2 / max(params.Delta_CF, 1e-9)
        _scf_log("RMFT-INIT",
                 f"t_pd={params.t_pd:.3f} eV  u={params.u:.3f}  λ_SOC={params.lambda_soc:.3f} eV"
                 f"   Δ_tetra={params.Delta_tetra:.3f} eV  g_JT={params.g_JT:.3f} eV"
                 f"   K_lattice={params.K_lattice:.3f} eV/Å²  K_spont={K_spont:.3f} eV/Å²"
                 f"  {'✓ K_lattice > K_spont, SC-JT window possible' if params.K_lattice > K_spont else '⚠ spontaneous JT risk'}"
                 f"  lambda_hop={params.lambda_hop:.3f}  Δ_CT={params.Delta_CT:.3f} eV   ω_JT={params.omega_JT:.3f} eV   Δ_ip={params.Delta_inplane:.3f} eV"
                 f"  kT={params.kT*1000:.2f} meV  Z={params.Z}  N_k={self.N_k}")
        _scf_log("RMFT-DERIVED",
                 f"t0={params.t0:.4f} eV  U={params.U:.4f} eV  "
                 f"  Δ_CF={params.Delta_CF:.5f} eV   J_CT={params.J_CT:.4f} eV  U_mf={params.U_mf:.4f} eV"
                 f"  Γ₇split={params.g7split:.5f} eV [{'⚠ < 2kT' if params.g7split < 2.0 * params.kT else '✓'}]"
                 f"η={params.eta:.4f}  δ₀={params.doping_0:.4f}  ")
        self._vbdg: Optional['VectorizedBdG'] = None
        self._scf_bdg_cache: Optional[tuple] = None
        self._gap_amplitude: float = 0.0   # current |Δ_s|+|Δ_d|; updated each SCF iteration
        self._K_bare: float = params.K_lattice # immutable bare lattice spring constant (eV/Å²)
        self._chi0_norm_cache: Optional[tuple] = None   # (E, V, M, Q, mu, tx, ty, g_J)
        self._fs_cache_dict:   Optional[dict]  = None   # key → (fs_pts, vF); single unified FS cache

    def _rebuild_orbital_operators(self) -> None:
        """
        Rebuild all SOC+CF-basis-dependent operators from params._U4.

        Must be called whenever params.lambda_soc, params.Delta_tetra, or params.Delta_inplane
        changes and params.__post_init__() has been called (which regenerates _U4, η, η_J).

        B₁g JT phonon operator — symmetry-correct treatment
        ────────────────────────────────────────────────────
        The physical electron–phonon coupling for the B₁g mode is

            H_JT = g_JT · Q · O_B1g

        where O_B1g = (Lx²−Ly²) in the t₂g manifold, projected to the active Γ₆⊕Γ₇a subspace:

            self.B1g_op = U4† · (Lx²−Ly²)_t2g · U4   (4×4, real, hermitian)

          D₄h (Δ_inplane = 0):
            B1g_op is purely anti-diagonal: couples (6↑)↔(7a↓) and (6↓)↔(7a↑).
            This is a SINGLET pairing operator — it can only be activated by Cooper
            pairs (SC-triggered JT selection rule: ⟨B1g⟩_normal = 0 exactly).

          D₂h (Δ_inplane ≠ 0, site symmetry reduced):
            B1g_op gains additional elements:
              • Spin-preserving off-diagonal (6↑↔7a↑, 6↓↔7a↓): the D₂h-induced
                τ_x channel — active in the normal state, no Cooper pairs required.
              • Diagonal (τ_z = diag(+1,+1,−1,−1)): Q-dependent renormalisation
                of Δ_CF; shifts Γ₆ and Γ₇a levels in opposite directions.
            Both extra terms are included automatically via the full projection.
        """
        self.multi_op = self.p.build_multipolar_operator()
        U4 = self.p._U4  # columns = {Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓}
        z4 = np.zeros((4, 4), dtype=complex)
        # BdG particle–hole symmetry requires the hole block to carry −τ_x^T, nambu structure: O_Nambu = block_diag(O_AA, -O_AA^T)
        self.tau_x_mat = np.array([
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ], dtype=float)

        # B₁g phonon operator in Γ₆⊕Γ₇a subspace
        _Lp = np.array([[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        _Lm = _Lp.T.conj()
        _Lx_f = np.kron((_Lp + _Lm) / 2.0,      np.eye(2, dtype=complex))
        _Ly_f = np.kron((_Lp - _Lm) / 2.0j,     np.eye(2, dtype=complex))
        _B1g_t2g = _Lx_f @ _Lx_f - _Ly_f @ _Ly_f          # (6,6) hermitian
        self.B1g_op = np.asarray(
            np.real(U4.conj().T @ _B1g_t2g @ U4),         # (4,4) real, hermitian
            dtype=float)

        # 16×16 Nambu extension of B1g_op for per-site ⟨B1g⟩ evaluation.
        # Layout: [Part_A, Part_B, Hole_A, Hole_B]; hole block carries −B1g_op^T.
        _B1g_c = self.B1g_op.astype(complex)
        _B1g_h = (-self.B1g_op.T).astype(complex)
        self.B1g_16 = np.block([
            [_B1g_c, z4,      z4,      z4     ],
            [z4,     _B1g_c,  z4,      z4     ],
            [z4,     z4,      _B1g_h,  z4     ],
            [z4,     z4,      z4,      _B1g_h ],
        ])

        # sz_op  (4,)  : orbital Sz weights in the [Γ₆↑,Γ₆↓,Γ₇↑,Γ₇↓] basis.
        # sz_bdg16 (16,) : the same weights extended to the full 16-component Nambu basis [pA, pB, hA, hB] × sz_op
        self.sz_op = np.array([1.0, -1.0, self.p.eta, -self.p.eta], dtype=float)
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

    def _get_chi0_norm_cache(self, M: float, Q: float, mu: float, tx: float, ty: float, g_J: float, vbdg: 'VectorizedBdG', target_doping: float) -> tuple:
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
            if all(abs(a - b) < _CHI0_CACHE_TOL
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
          - _K_bare        : NOT cleared (immutable per __init__ contract)
        """
        self._vbdg             = None   # re-created on first _get_vbdg()
        self._scf_bdg_cache    = None   # no stale (ev, ec) from parent solve
        self._chi0_norm_cache  = None   # normal-state χ₀ eigenvector cache
        self._gap_amplitude    = 0.0    # reset so cloned solvers start from normal state
        self._fs_cache_dict    = None   # unified FS sample cache (fs_pts, vF) keyed by (params, n_fs)

    def _full_rebuild(self) -> None:
        """Always call this single method after mutating any ModelParams field on a solver clone."""
        self.p.__post_init__()
        self._K_bare = self.p.K_lattice
        self._rebuild_orbital_operators()
        self._reset_transient_state()

    def _exchange_channels(self, Q: float) -> tuple:
        """
        Q-dependent multipolar exchange in the [Γ₆↑, Γ₆↓, Γ₇↑, Γ₇↓] basis.

        D₄h decomposition:
            J(Q) = J_A1g(Q)·P_A1g + J_B1g(Q)·P_B1g,
            P_A1g = diag(1,1,η_J²,η_J²),  P_B1g = η_J·τ_x  (Γ₆–Γ₇ mixing).

        Microscopic origin (anisotropic hopping t_x≠t_y, J ∝ t²):
            t_x = t₀ e^{+Q/λ},  t_y = t₀ e^{-Q/λ}  ⇒
            J_A1g ∝ J_CT·cosh(2Q/λ)  (even, longitudinal),
            J_B1g ∝ J_CT·sinh(2Q/λ)  (odd, transverse; vanishes at Q=0).

        At Q=0:
            J_B1g = 0 (Γ₆–Γ₇ mixing forbidden), 
            J_A1g = J_CT·diag(1,1,η_J²,η_J²), consistent with effective_superexchange.

        Physical consequence:
            Q=0 → pure A1g (AFM selection rule);
            Q≠0 → B1g channel opens (multipolar/JT response).

        Returns:
            J_A1g_diag   — Z·J_CT·scale_A1g·[1,1,η_J²,η_J²]
            J_B1g_scalar — Z·J_CT·scale_B1g·η_J

        η_J(Q): orbital-selective superexchange (J ∝ t²) yields Γ₆ (xz) vs Γ₇ (yz)
        imbalance when t_x≠t_y (d_xy contributes to both).
        """
        lam   = max(self.p.lambda_hop, 1e-9)
        e_p   = np.exp(+Q / lam)
        e_m = np.exp(-Q / lam)
        _jxz  = e_p ** 2           # (tx_q/t0)²
        _jyz  = e_m ** 2           # (ty_q/t0)²
        _jxy  = 0.5 * (_jxz + _jyz)
        _J6   = self.p._w6_xz * _jxz + self.p._w6_yz * _jyz + self.p._w6_xy * _jxy
        _J7   = self.p._w7_xz * _jxz + self.p._w7_yz * _jyz + self.p._w7_xy * _jxy
        eta_J = float(np.sqrt(max(_J7 / max(_J6, 1e-9), 0.0)))  # orbital-weight ratio
        s_A   = float(np.cosh(2.0 * Q / lam))  # even, longitudinal scale
        s_B   = float(np.sinh(2.0 * Q / lam))  # odd,  transverse scale

        J_A1g_diag = self.p.Z * self.p.J_CT * s_A * np.array([1.0, 1.0, eta_J**2, eta_J**2])
        J_B1g_scalar  = self.p.Z * self.p.J_CT * s_B * eta_J
        return J_A1g_diag, J_B1g_scalar
    
    def cluster_expectation(self, evals: np.ndarray, evecs: np.ndarray, Operator: np.ndarray, temperature: float, site_index: int = -1) -> float:
        """
        Sites A and B (AFM sublattices); exact diagonalization of O⊗O multipolar exchange with mean-field boundary coupling to external magnetization.
        Captures: quantum multipolar correlations, orbital mixing and spin-orbit coupling, thermal fluctuations.
        Does NOT capture: fermionic antisymmetrization, charge-transfer fluctuations.
        Valid when multipolar degrees of freedom dominate over charge fluctuations and system is not deep in the Mott insulator.
        """
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
        Oevecs = Operator @ evecs
        diag   = np.einsum('ij,ij->j', evecs.conj(), Oevecs)
        return float(np.real((weights * diag).sum() / weights.sum()))

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
        sz_op  = self.sz_op

        vbdg = self._get_vbdg()

        tx_0, ty_0 = self.p.effective_hopping_anisotropic(Q)
        tx_p, ty_p = self.p.effective_hopping_anisotropic(Q + eps)
        tx_m, ty_m = self.p.effective_hopping_anisotropic(Q - eps)

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
        
        def _J_alpha_beta_Q(Qv):
            # Full 4×4 exchange matrix for the rigidity ∂²F_ex/∂Q² derivative.
            # J_B1g·τ_x contributes through the O_exp_Q @ J @ O_exp_Q contraction only when ⟨τ_x⟩≠0.
            J_A1g_diag_v, J_B1g_scalar_v = self._exchange_channels(Qv)
            return np.diag(J_A1g_diag_v) + J_B1g_scalar_v * self.tau_x_mat

        O_exp_Q = _calc_O_exp(ev_0, ec_0)
        O_p     = _calc_O_exp(ev_p, ec_p)
        O_m     = _calc_O_exp(ev_m, ec_m)

        dO_dQ   = (O_p - O_m) / (2.0 * eps)
        d2O_dQ2 = (O_p - 2.0 * O_exp_Q + O_m) / (eps ** 2)

        J_Q   = _J_alpha_beta_Q(Q)
        J_p   = _J_alpha_beta_Q(Q + eps)
        J_m   = _J_alpha_beta_Q(Q - eps)

        dJ_dQ   = (J_p - J_m) / (2.0 * eps)
        d2J_dQ2 = (J_p - 2.0 * J_Q + J_m) / (eps ** 2)

        # Full ∂²F_ex/∂Q²
        term_J2    = float(O_exp_Q @ d2J_dQ2 @ O_exp_Q)           # O·(∂²J/∂Q²)·O
        term_dO2   = 2.0 * float(dO_dQ @ J_Q @ dO_dQ)             # 2·(∂O/∂Q)·J·(∂O/∂Q)
        term_O2    = 2.0 * float(O_exp_Q @ J_Q @ d2O_dQ2)         # 2·O·J·(∂²O/∂Q²)
        term_mix   = 4.0 * float(O_exp_Q @ dJ_dQ @ dO_dQ)         # 4·O·(∂J/∂Q)·(∂O/∂Q)
        d2F_ex_dQ2 = term_J2 + term_dO2 + term_O2 + term_mix

        K_eff = self.p.K_lattice + d2F_ex_dQ2

        # Commutator diagnostic: [τ_x, H_AFM] at Q=0 — B1g blocking strength
        h_afm_diag = _J_alpha_beta_Q(0.0) @ O_exp_Q
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

    def compute_chi_ss_with_infinitesimal_gap(self, M: float, G_res: dict, target_doping: float, delta_test: float = 1e-4) -> float:
        """
        Physically: a tiny Δ opens a gap at the Fermi surface and suppresses the
        low-energy spin-fluctuation spectral weight responsible for the Stoner pole.
        This mimics what happens in the SC state without solving the full gap equation.

        The gap is applied as a pure d-wave seed (Delta_d = delta_test, Delta_s = 0)
        so that the nodal structure is respected.
        """
        vbdg = self._get_vbdg()

        g_t, g_J, _, _  = self.p.get_gutzwiller_factors(target_doping)
        t_eff = g_t * self.p.t0   # isotropic seed; Q=0 pre-SCF

        # Clone the output buffer to avoid mutating in-flight SCF cache
        _H_buf = vbdg._H_stack_ev.copy()

        def _diag(Delta_d_val, Delta_s_val):
            return np.linalg.eigh(
                vbdg._build_H_stack(
                    vbdg._kpts_ev, M, 0.0,
                    complex(Delta_s_val), complex(Delta_d_val),
                    target_doping, G_res['mu_n'], t_eff, t_eff, g_J,
                    out=_H_buf,
                )
            )

        # Try gapped first; fall back to Δ=0 on numerical failure
        try:
            ev, ec = _diag(delta_test, 0.0)
        except Exception:
            try:
                ev, ec = _diag(0.0, 0.0)
            except Exception:
                return float('nan')

        f_k = self.fermi_function(ev)   # (N, 16)

        # χ_SS Lehmann sum
        sz_diag = self.sz_bdg16   # (16,) staggered Sz in 2-sublattice Nambu basis
        SzV  = sz_diag[None, :, None] * ec                          # (N,16,16)
        Mmat = np.einsum('kin,kim->knm', ec.conj(), SzV)            # (N,16,16)
        M2   = np.abs(Mmat) ** 2

        df = f_k[:, :, None] - f_k[:, None, :]   # (N,16,16)
        dE = ev[:, None, :] - ev[:, :, None]      # (N,16,16)

        _eta_sq = max(0.01 * self.p.t0, _FD_MASK_DE) ** 2
        denom   = dE.real ** 2 + _eta_sq
        df_mask = np.where(np.abs(df) > _FD_MASK_DF, df, 0.0)
        
        ratio   = self.k_weights_even[:, None, None] * M2 * df_mask * dE.real / denom
        chi_ss  = float(ratio.sum())

        # J_eff for Moriya damping: use J_eff_renorm
        J_eff = self.p.Z * self.p.effective_superexchange(g_J, t_eff, t_eff, target_doping)

        _Gamma_M      = self.p.moriya_gamma(target_doping, t_eff, J_eff)
        _chi_ss_pos   = max(chi_ss, 0.0)
        chi_ss_moriya = _chi_ss_pos / (1.0 + _Gamma_M * _chi_ss_pos)
        return chi_ss_moriya

    def _B1g_expectation(self, M: float, Q_val: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, g_t: float, g_J: float) -> float:
        """
        Per-site ⟨B1g_op⟩ in the BdG ground state at distortion Q_val.

        Uses the full 16-component Nambu eigenstates so that the anomalous
        u·v amplitudes — which carry the SC-triggered orbital coherence — are
        fully included.  The /4 factor corrects for the Nambu doubling
        (2 sublattices × particle-hole redundancy).
        """
        tx_b, ty_b = self.p.effective_hopping_anisotropic(Q_val)
        tx_v = g_t * tx_b
        ty_v = g_t * ty_b
        vbdg = self._get_vbdg()
        ev, ec = np.linalg.eigh(
            vbdg._build_H_stack(
                vbdg._kpts, M, Q_val, Delta_s, Delta_d,
                target_doping, mu, tx_v, ty_v, g_J,
            )
        )
        f_n = self.fermi_function(ev)  # (Nk, 16)
        # ⟨B1g⟩_k = Tr[B1g_16 · ρ_k]  where ρ_k = Σ_n (u_n u_n† f_n + v_n v_n† (1-f_n))
        # Diagonal of ec† B1g_16 ec in the quasiparticle basis: a, b: component indices (rows and columns of the orbitals / Nambu operator), n: band index (columns of the eigenvector in the ec matrix
        diag_qp = np.einsum('kan, ab, kbn -> kn', ec.conj(), self.B1g_16, ec).real
        
        # Thermal avg in Nambu basis. Summing over all 16 bands automatically covers the hole components via f(-E) = 1 - f(E).
        exp_k = np.einsum('kn,kn->k', diag_qp, f_n)
        return float(np.dot(self.k_weights, exp_k)) / 4.0

    def _compute_chi_tau(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_t: float, g_J: float) -> Dict:
        """
        The JT coupling is H_JT = g_JT · Q · B1g_op.  B1g_op = U4†(Lx²−Ly²)U4 is the JT phonon coupling (not only spin-preserving)

        JT orbital susceptibility χ_τ = ∂⟨B1g_op⟩ / ∂(g_JT · Q)

        via Richardson-extrapolated central finite difference of the per-site ⟨B1g_op⟩ expectation value.     

        δχ_τ = χ_τ(Δ≠0) − χ_τ(Δ=0) isolates the condensate-induced contribution.
        In D₄h ⟨B1g_op⟩=0 exactly in the normal state (symmetry) so χ_τ(Δ=0)=0 and
        δχ_τ = χ_τ(Δ≠0).  In D₂h (Δ_inplane≠0) a small normal-state baseline can
        exist; subtracting it prevents a D₂h normal-state signal from masquerading
        as SC-triggered.

        Richardson extrapolation
        ------------------------
        Three step sizes h, h/2, h/4 give three central-difference estimates.
        First-order Richardson: R1 = (4·CD(h/2) − CD(h)) / 3  (O(h²)→O(h⁴)).
        Second-order Richardson: R2 = (4·CD(h/4) − CD(h/2)) / 3.
        Final estimate: mean(R1, R2).
        Quality flags:
        richardson_ok : |R1−R2|/max(|est|,ε) < 3%  — extrapolation converged (distinguishes "used reliable value" from "fell back")
        nonlinear     : |CD(h)−CD(h/2)|/max(|CD(h/2)|,ε) > 20% — step too large relative to the curvature of ⟨B1g⟩(Q)),
        response is nonlinear; result unreliable. Falling back to 0.0 is conservative:
          • For chi_tau_sc = 0.0: λ_JT = 0 → no spurious SC-triggered JT signal.
          • For chi_tau_n  = 0.0: no baseline subtraction, mild overcount risk in D₂h, but safer than using a wrong (potentially sign-flipped) derivative.
        """
        t_eff_avg = max(np.sqrt(0.5 * (tx**2 + ty**2)), 1e-6)
        N0        = 1.0 / (np.pi * t_eff_avg)
        Ut_ratio  = self.p.U / t_eff_avg

        g_JT = max(self.p.g_JT, 1e-12)
        scale = self.p.Delta_CF / g_JT
        h_phys = 0.05 * scale
        h_floor = 1e-4
        h = float(np.clip(1e-3 * max(abs(Q), scale), h_floor, h_phys))

        def _cd(dq: float, ds: complex, dd: complex) -> float:
            """Central difference d⟨B1g⟩/dQ at step dq."""
            vp = self._B1g_expectation(M, Q + dq, ds, dd, target_doping, mu, g_t, g_J)
            vm = self._B1g_expectation(M, Q - dq, ds, dd, target_doping, mu, g_t, g_J)
            return (vp - vm) / (2.0 * dq)

        def _richardson(ds: complex, dd: complex) -> tuple:
            cd1 = _cd(h,       ds, dd)
            cd2 = _cd(h / 2.0, ds, dd)
            cd3 = _cd(h / 4.0, ds, dd)
            
            R1  = (4.0 * cd2 - cd1) / 3.0
            R2  = (4.0 * cd3 - cd2) / 3.0
            est = 0.5 * (R1 + R2)
            
            err       = abs(R1 - R2) / max(abs(est), 1e-12)
            converged = err < 0.05
            nonlinear = abs(cd1 - cd2) / max(abs(cd2), 1e-12) > 0.2
            
            if converged:
                return est, True, nonlinear
            elif not nonlinear:
                return cd1, False, nonlinear
            else:
                # Highly non-linear response: fall back to the smallest step that isn't R2 to localize the response.
                return cd2, False, True

        # SC-state χ_τ
        dB1g_sc, ok_sc, nonlin_sc = _richardson(Delta_s, Delta_d)

        # Normal-state baseline: MUST evaluate even when Δ is small, because in D₂h symmetry the normal state has a non-zero B₁g response that must be subtracted.
        dB1g_n, ok_n, nonlin_n = _richardson(0.0+0j, 0.0+0j)

        # ── Reliability guard ──────────────────────────────────────────────────────
        if nonlin_sc:
            chi_tau_sc = 0.0   # unreliable derivative — treat as zero (conservative)
        else:
            chi_tau_sc = dB1g_sc / g_JT   # sign physical: negative=stiffening

        if nonlin_n:
            chi_tau_n = 0.0    # unreliable baseline — skip subtraction
        else:
            chi_tau_n = dB1g_n / g_JT

        richardson_ok = ok_sc and ok_n and not nonlin_sc and not nonlin_n

        return {
            'chi_tau_sc':    chi_tau_sc,
            'chi_tau_n':     chi_tau_n,
            'richardson_ok': richardson_ok,
            'nonlin_sc':     nonlin_sc,
            'nonlin_n':      nonlin_n,
            'N0':            N0,
            'Ut_ratio':      Ut_ratio,
        }

    def _chi_QQ_matrix_elements(self, M: float, Q: float, target_doping: float, Delta_s: complex, Delta_d: complex, mu: float) -> float:
        """
        g²-weighted JT orbital susceptibility: χ_QQ = g_JT² · χ_orbital = −g_JT² · ∂²Ω/∂(g_JT·Q)².
        """
        dQ = max(1e-4, 1e-3 * abs(Q) + 1e-5)   # adaptive step
        g_t, g_J, _, _ = self.p.get_gutzwiller_factors(target_doping)
        vbdg = self._get_vbdg()

        def omega(Qval):
            tx_b, ty_b = self.p.effective_hopping_anisotropic(Qval)
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
    
    def get_susceptibilities_normal(self, q: np.ndarray, M: float, Q: float, target_doping: float, actual_doping: float, mu: float,
                                    tx: float, ty: float, g_J: float, chi_QQ_n: float, K_eff: float, J_eff: float, _E_k_cache: tuple = None, _enforce_c4: bool = False, z_qp: float = 1.0) -> dict:
        """
        Static bare susceptibility tensor χ⁰^{ab}(q) in [Γ₆↑,Γ₆↓,Γ₇↑,Γ₇↓] basis,
        with RPA resummation and pairing vertex.  Δ=0 strictly enforced.

        Lindhard formula (static, ω=0):

            χ₀^{ab}(q) = Σ_{k,n,m} (f_n(k) − f_m(k+q)) / (E_m(k+q) − E_n(k) + iη)
                          × V*_k[a,n] · V_{k+q}[a,m] · V*_{k+q}[b,m] · V_k[b,n]

        The imaginary part (∝ η/(dE²+η²)) vanishes in the static limit and is
        dropped from the kernel for efficiency, while being retained in the formula
        comment for physical transparency.

        The 16x16 BdG Nambu basis is [Part_A(0:4), Part_B(4:8), Hole_A(8:12), Hole_B(12:16)].
        eigh diagonalises the full Nambu matrix; the resulting eigenvectors already
        encode the Bogoliubov rotation including particle-hole signs.
        
        a and b are ORBITAL indices (not band), n and m are quasiparticle bands.
        The 0.5 prefactor corrects for double-counting: SECTOR_PAIRS includes both
        (sl_α, sl_β) and (sl_β, sl_α) for every inter-sublattice pair.

        Anomalous sector pairs (Part↔Hole) are excluded: they vanish at Δ=0 and
        would double-count the anomalous Green function F at Δ≠0.

        Susceptibility projections (B1g operator structure in D₄h/D₂h):
            χ_DD_s: Tr[S_z · χ⁰[Γ₆,Γ₆] · S_z] — longitudinal spin-spin, Γ₆ sector.
                    Stoner parameter for AFM QCP.

            χ_DQ_s: Tr[S_z · χ⁰[Γ₆,Γ₇]] — spin-Z / orbital-diagonal cross.
                    τ_x (= tau_x_mat) is spin-PRESERVING: 6↑↔7↑, 6↓↔7↓.
                    The relevant S_z–B1g cross-channel is therefore the DIAGONAL
                    spin projection of the Γ₆→Γ₇ susceptibility block:
                        χ_DQ = χ⁰[0,2] − χ⁰[1,3] = Tr[S_z · χ⁰[Γ₆,Γ₇]]

        _E_k_cache : (E_k_all, V_k_all) at Δ=0 on k_points_even — pass to skip eigh.
        _enforce_c4 : average χ⁰ over C₄-related q-points (removes grid anisotropy).
        """
        vbdg   = self._get_vbdg()
        eta    = max(0.01 * self.p.t0, _FD_MASK_DE)   # η ~ 1% of bandwidth; FS-projection reduces sensitivity but too small η causes spurious peaks

        SECTOR_PAIRS = [
            (slice(0,  4), slice(0,  4)),   # A-A particle
            (slice(4,  8), slice(4,  8)),   # B-B particle
            (slice(8, 12), slice(8, 12)),   # A-A hole
            (slice(12,16), slice(12,16)),   # B-B hole
            (slice(0,  4), slice(4,  8)),   # A-B particle
            (slice(4,  8), slice(0,  4)),   # B-A particle
            (slice(8, 12), slice(12,16)),   # A-B hole
            (slice(12,16), slice(8, 12)),   # B-A hole
        ]
        CHUNK = 128

        if _E_k_cache is not None:
            E_k_all, V_k_all = _E_k_cache
        else:
            E_k_all, V_k_all = np.linalg.eigh(
                vbdg._build_H_stack(vbdg._kpts_ev, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack_ev))

        f_k_all = self.fermi_function(E_k_all)   # (N_k, 16)
        N = len(self.k_points_even)
        w = self.k_weights_even                  # (N_k,)

        def _chi0_at_shift(shift_idx: np.ndarray) -> np.ndarray:
            """χ⁰(q') via shift-table permutation; reuses cached E_k, V_k, f_k."""
            E_kQ = E_k_all[shift_idx]
            V_kQ = V_k_all[shift_idx]
            f_kQ = self.fermi_function(E_kQ)

            df      = f_k_all[:, :, None] - f_kQ[:, None, :]   # (N_k,16,16)
            dE      = E_kQ[:, None, :] - E_k_all[:, :, None]
            df_safe = np.where(np.abs(df) > _FD_MASK_DF, df, 0.0)
            kernel  = w[:, None, None] * df_safe * dE.real / (dE.real**2 + eta * eta)

            chi = np.zeros((4, 4), dtype=complex)
            for sl_a, sl_b in SECTOR_PAIRS:
                # Standard Lindhard: V_k[a,n]* · V_{k+q}[a,m] · V_{k+q}[b,m]* · V_k[b,n]
                # a indexes sl_a sector at k; b indexes sl_b sector at k.
                Vk_a  = V_k_all[:, sl_a, :]   # (N_k, 4, 16) — sl_a sector at k
                VkQ_a = V_kQ[:,   sl_a, :]    # (N_k, 4, 16) — sl_a sector at k+q
                Vk_b  = V_k_all[:, sl_b, :]   # (N_k, 4, 16) — sl_b sector at k
                VkQ_b = V_kQ[:,   sl_b, :]    # (N_k, 4, 16) — sl_b sector at k+q
                for k0 in range(0, N, CHUNK):
                    k1 = min(k0 + CHUNK, N)
                    chi += oe.contract(
                        'cnm,can,cam,cbm,cbn->ab',
                        kernel[k0:k1], Vk_a[k0:k1].conj(),
                        VkQ_a[k0:k1], VkQ_b[k0:k1].conj(),
                        Vk_b[k0:k1], optimize='greedy')
            chi *= 0.5
            return chi

        dk = 2.0 * np.pi / _NK
        nx = int(round(q[0] / dk)) % _NK
        ny = int(round(q[1] / dk)) % _NK
        chi0 = _chi0_at_shift(self.shift_table[nx, ny])

        if _enforce_c4:
            chi0 += _chi0_at_shift(self.shift_table[(-ny) % _NK,  nx        % _NK])
            chi0 += _chi0_at_shift(self.shift_table[(-nx) % _NK, (-ny)      % _NK])
            chi0 += _chi0_at_shift(self.shift_table[  ny  % _NK, (-nx)      % _NK])
            chi0 *= 0.25

        chi0 = 0.5 * (chi0 + chi0.conj().T)   # enforce Hermiticity
        chi0_tensor = chi0.real   # Lindhard requires Δ=0, but anyway, this is the only way to avoid self-referential double-counting  

        # Susceptibility projections
        chi_66 = chi0_tensor[0:2, 0:2]
        chi_67 = chi0_tensor[0:2, 2:4]
        chi_76 = chi0_tensor[2:4, 0:2]

        # χ_DD_s: longitudinal spin-spin in Γ₆ sector.
        sz_vec   = np.array([1.0, -1.0])
        chi_DD_s = float(sz_vec @ chi_66 @ sz_vec)

        # χ_DQ_s / χ_QD_s: spin-Z / orbital-diagonal Γ₆→Γ₇ cross-channel.
        # The relevant coupling is the spin-diagonal projection: Tr[S_z · χ[Γ₆,Γ₇]].
        chi_DQ_s = float(np.trace(np.diag(sz_vec) @ chi_67))
        chi_QD_s = float(np.trace(chi_76 @ np.diag(sz_vec)))

        # ── Moriya damping and RPA ────────────────────────────────────────────────
        V_JT   = self.p.g_JT**2 / K_eff

        _moriya_doping  = actual_doping if actual_doping is not None else target_doping
        g_t, _, _, _    = self.p.get_gutzwiller_factors(target_doping)
        _Gamma_M_bare   = self.p.moriya_gamma(_moriya_doping, g_t * self.p.t0, J_eff)

        # Moriya damping with correlation correction: only penalize excess J (r_J > 1) beyond Gutzwiller to avoid RPA runaway;
        # leave r_J ≤ 1 unchanged. Apply correction to alpha (not Γ_M) so Γ_M stays within its physical limit.
        # With r_J = 1/z_qp, excess = max(0, r_J - 1). Γ_M remains capped (≤ 4·t_eff/π) and is at most doubled vs. bare.
        _r_J_eff  = 1.0 / max(z_qp, 0.1)      # r_J from quasiparticle weight
        _rJ_excess = max(0.0, _r_J_eff - 1.0)  # only penalise the overcorrelated part
        _rJ_boost  = min(_rJ_excess, 1.0)       # cap boost at ×2 to preserve SC-JT window
        _Gamma_M   = _Gamma_M_bare * (1.0 + _rJ_boost)

        _chi_DD_s_pos   = max(chi_DD_s, 0.0)
        chi_DD_s_moriya = _chi_DD_s_pos / max((1.0 + _Gamma_M * _chi_DD_s_pos), 1e-10) 

        _chi_QQ_pos   = max(chi_QQ_n, 0.0)
        _chi_DQ_s_dyn = max(_CHI_DQ_S_EPS, 1e-4 * float(np.sqrt(_chi_DD_s_pos * _chi_QQ_pos)))
        # χ_SQ vanishes by symmetry in the normal state, scales ∝ Δ in the superconducting state, and is Padé-regularized as χ_SQ_v = χ_SQ / (1 + |χ_SQ| / w), mimicking a resummed perturbative vertex.
        # Padé width w: Δ-adaptive when Δ > threshold, else geometric-mean floor.
        _w_sq   = max(_chi_DQ_s_dyn, 1e-6 * self._gap_amplitude) if self._gap_amplitude > _QQ_DELTA_THRESH else _chi_DQ_s_dyn
        _w_sq   = max(_w_sq, _CHI_DQ_S_PADE_W * max(abs(chi_DQ_s), abs(chi_QD_s), _CHI_DQ_S_EPS))
        chi_DQ_s_v = float(chi_DQ_s) / (1.0 + abs(chi_DQ_s) / max(_w_sq, 1e-30))
        chi_QD_s_v = float(chi_QD_s) / (1.0 + abs(chi_QD_s) / max(_w_sq, 1e-30))

        # χ_QQ soft Dyson regularisation (re-summation of bubble diagrams), as chi_QQ → ∞: ≈ K_bare / V_JT
        chi_QQ_eff  = _chi_QQ_pos / (1.0 + _chi_QQ_pos / K_eff)
        chi_QQ_tilde = chi_QQ_eff / max(self.p.g_JT**2, 1e-12)

        def _rpa_vertex(J: float, V: float) -> tuple[float, float]:
            """
            RPA vertex with sign-preserving determinant regularization.

            det > 0 (paramagnetic side of QCP): standard RPA inversion; floor only
                     prevents exact-zero numerical accident.
            det < 0 (AFM ordered, past QCP): matrix formally inverted past convergence
                     radius. Physical divergence preserved via universal Padé soft-cap;
                     sign enforced attractive when J*chi > 1.
            det ~ 0 (exact QCP): copysign(floor, det) keeps sign but |Vp| can reach
                     ~1/det_floor ~ 1e9 eV even when math.isfinite(Vp) is True.
                     (Root cause of V_s=3.6e11 observed in logs.)
                     Universal cap _RPA_V_SOFT_CAP applied for ALL det values.
            """
            a = 1.0 - J * chi_DD_s_moriya
            b = -J * chi_DQ_s_v
            c = -V * chi_QD_s_v
            d = 1.0 - V * chi_QQ_tilde
            det = a * d - b * c

            M_frob    = math.sqrt(a*a + b*b + c*c + d*d)  # Frobenius norm sets natural scale of matrix
            det_floor = max(_RPA_DET_REG, 1e-4 * M_frob)
            if abs(det) < det_floor:
                det_safe = math.copysign(det_floor, det) if det != 0.0 else det_floor
            else:
                det_safe = det

            i00 =  d / det_safe
            i01 = -b / det_safe
            i10 = -c / det_safe
            i11 =  a / det_safe

            rss = i00 * chi_DD_s_moriya + i01 * chi_QD_s_v
            rqq = i10 * chi_DQ_s_v      + i11 * chi_QQ_tilde
            rsq = i00 * chi_DQ_s_v      + i01 * chi_QQ_tilde
            rqs = i10 * chi_DD_s_moriya + i11 * chi_QD_s_v
            Vp  = J**2 * rss + V**2 * rqq + J * V * (rsq + rqs)

            if abs(Vp) > _RPA_V_SOFT_CAP:
                Vp = math.copysign(_RPA_V_SOFT_CAP, Vp)

            if det < 0 and J * chi_DD_s_moriya > 1.0 and Vp < 0:
                Vp = abs(Vp)

            if not math.isfinite(Vp):
                Vp = _RPA_V_SOFT_CAP
            return float(Vp), float(det)

        V_full, det_full = _rpa_vertex(J_eff, V_JT)
        V_spin, _        = _rpa_vertex(J_eff, 0.0)
        V_jt,   _        = _rpa_vertex(0.0,   V_JT)

        return {
            'chi0_tensor':     chi0_tensor,
            'chi_DD_s':        chi_DD_s,
            'chi_DQ_s':        chi_DQ_s,
            'chi_QD_s':        chi_QD_s,
            'chi_QQ':          chi_QQ_n,
            'V_full':          V_full,
            'V_spin':          V_spin,
            'V_jt':            V_jt,
            'chi_DD_s_moriya': chi_DD_s_moriya,
            'rpa_det':         det_full,
            'Gamma_M_eff':     _Gamma_M,
            'r_J_excess':      _rJ_excess,
        }
    
    def solve_linearized_gap_equation(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, K_eff: float, J_eff: float, actual_doping: float = None, z_qp: float = 1.0, _fs_precomputed: tuple = None) -> Dict:
        """
        Builds the full N_FS × N_FS kernel matrix Γ_ij = √(dl_i·dl_j/(vFi·vFj)) · V(k_i−k_j).
        Finds the LARGEST EIGENVALUE λ_max and its eigenvector (gap symmetry).
        
        Improvements:
        1. Arc-length (dl) weighting for proper FS integration measure
        2. C₄ symmetry averaging only when Q ≈ 0 (D₄h valid)
        3. Exact Hermiticity enforcement: 0.5*(Γ+Γᵀ)
        4. Proper Gutzwiller renormalization on dominant channel
        5. Consistent handling of bare vs. renormalized eigenvalues
        
        Linearised gap equation solved as an eigenvalue problem on the Fermi surface:
            λ Δ(k_i) = Σ_j Γ_ij Δ(k_j)
        """
        # FS points
        if _fs_precomputed is not None:
            fermi_pts, vF_arr = _fs_precomputed
            fs_idx, ev, ec = None, None, None
        else:
            fermi_pts, vF_arr, fs_idx, ev, ec = self._get_fs_points(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                store_cache=True, compute_vF=True)
        N = fermi_pts.shape[0]
        
        # Sort points by polar angle to estimate spacing along FS
        angles = np.arctan2(fermi_pts[:, 1], fermi_pts[:, 0])
        sort_idx = np.argsort(angles)
        sorted_pts = fermi_pts[sort_idx]
        
        dl_sorted = np.zeros(N)
        for i in range(N):
            # Distance to neighboring points (periodic along FS)
            p_prev = sorted_pts[(i - 1) % N]
            p_next = sorted_pts[(i + 1) % N]
            dl_sorted[i] = 0.5 * (np.linalg.norm(sorted_pts[i] - p_prev) + np.linalg.norm(sorted_pts[i] - p_next))
        
        # Map back to original point ordering
        dl_arr = np.zeros(N)
        dl_arr[sort_idx] = dl_sorted
        
        # build q = k_i - k_j
        i_idx, j_idx = np.triu_indices(N)
        q_raw = fermi_pts[i_idx] - fermi_pts[j_idx]
        q_arr = (q_raw + np.pi) % (2 * np.pi) - np.pi

        # Unique q cache
        scale = 1e5
        q_int = np.rint(q_arr * scale).astype(np.int64)
        unique_int, inv_idx = np.unique(q_int, axis=0, return_inverse=True)
        unique_q = unique_int.astype(np.float64) / scale
        
        # Build the χ₀ cache once for the normal state
        chi_QQ_normal = self._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)
        vbdg = self._get_vbdg()
        E_k_cache_normal = self._get_chi0_norm_cache(M, Q, mu, tx, ty, g_J, vbdg, target_doping=target_doping)

        # For each unique q, evaluate V(q).
        n_q = len(unique_q)
        V_unique = np.empty(n_q)
        V_spin_u = np.empty(n_q)
        V_JT_u   = np.empty(n_q)

        for u_idx, q_u in enumerate(unique_q):
            _sus_qu = self.get_susceptibilities_normal(
                q_u, M, Q, target_doping, actual_doping, mu, tx, ty, g_J,
                chi_QQ_normal, K_eff, J_eff, E_k_cache_normal, z_qp=z_qp,
                _enforce_c4=(abs(Q) < 1e-3 * self.p.lambda_hop))  # _enforce_c4=True when Q≈0 (unbroken D₄h)
            V_unique[u_idx] = _sus_qu['V_full']
            V_spin_u[u_idx] = _sus_qu['V_spin']
            V_JT_u[u_idx]   = _sus_qu['V_jt']
        
        # Build bare kernel matrices with arc-length weights
        vF_safe = np.maximum(np.abs(vF_arr), _VF_FLOOR_TIGHT)
        w_array = np.sqrt(dl_arr / vF_safe)  # sqrt(dl / vF)
        W_mat = np.outer(w_array, w_array)   # W_ij = √(dl_i/vF_i) · √(dl_j/vF_j)
        
        # Build bare vertex matrices (only upper triangle)
        V_mat_bare = np.zeros((N, N), dtype=float)
        V_mat_bare[i_idx, j_idx] = V_unique[inv_idx]
        V_mat_bare = 0.5 * (V_mat_bare + V_mat_bare.T)  # enforce exact Hermiticity
        
        V_mat_JT_bare = np.zeros((N, N), dtype=float)
        V_mat_JT_bare[i_idx, j_idx] = V_JT_u[inv_idx]
        V_mat_JT_bare = 0.5 * (V_mat_JT_bare + V_mat_JT_bare.T)
        
        # Full kernel matrices and solve bare eigenvalue problem
        Gamma_bare = W_mat * V_mat_bare
        Gamma_JT_bare = W_mat * V_mat_JT_bare
        eigvals_bare, eigvecs_bare = np.linalg.eigh(Gamma_bare)
        lambda_bare = float(eigvals_bare[-1])
        gap_vector = eigvecs_bare[:, -1]
        
        # Basis functions for s-wave and d-wave
        phi_s = np.ones(N) / np.sqrt(N)
        phi_d_raw = np.cos(fermi_pts[:, 0]) - np.cos(fermi_pts[:, 1])
        norm_d = np.linalg.norm(phi_d_raw)
        phi_d = phi_d_raw / norm_d if norm_d > 1e-12 else phi_d_raw
        
        # Project gap vector onto s- and d-wave channels
        w_s = abs(np.dot(gap_vector, phi_s))
        w_d = abs(np.dot(gap_vector, phi_d))
        
        # Get Gutzwiller factors
        _, _, g_Delta_s, g_Delta_d = self.p.get_gutzwiller_factors(target_doping)
        
        # Apply channel-specific Gutzwiller renormalization and determine dominant simmetry
        if w_d > w_s:
            lambda_lin_max = g_Delta_d * lambda_bare
            gap_symmetry = 'B1g (d-wave)'
            g_delta_dom = g_Delta_d
        else:
            lambda_lin_max = g_Delta_s * lambda_bare
            gap_symmetry = 'A1g (s-wave)'
            g_delta_dom = g_Delta_s
        
        # JT-projection (Rayleight quotient)
        gv_norm = gap_vector / max(float(np.linalg.norm(gap_vector)), 1e-12)
        lambda_JT_kernel = g_delta_dom * float(np.dot(gv_norm, np.dot(Gamma_JT_bare, gv_norm)))
        
        # Check for nonlinear quenching (SCF vs linear gap symmetry mismatch)
        _Ds_mag = abs(Delta_s)
        _Dd_mag = abs(Delta_d)
        if _Ds_mag + _Dd_mag > 1e-4:
            _scf_dom = 'd' if _Dd_mag > _Ds_mag else 's'
            _lin_dom = 'd' if w_d > w_s else 's'
            if _scf_dom != _lin_dom:
                gap_symmetry += f' [lin={_lin_dom}, SCF={_scf_dom}: nonlinear quench]'
        
        # Vertex diagnostics (mean values over unique q)
        V_spin_mean = float(np.mean(V_spin_u))
        V_JT_mean   = float(np.mean(V_JT_u))
        V_rpa_mean  = float(np.mean(V_unique))
        V_cross_mean = V_rpa_mean - V_spin_mean - V_JT_mean
        
        # Cohernece length estimation
        Delta_0 = max(abs(Delta_s), 2.0 * abs(Delta_d))
        if Delta_0 > 1e-8:
            vF_avg = float(np.mean(vF_arr))
            xi_over_a = vF_avg / (np.pi * Delta_0)
            
            if fs_idx is not None and ev is not None and ec is not None:
                # Full orbital-resolved ξ computation
                kT_val = self.p.kT
                window = _FS_SAMPLING * kT_val
                w6_arr, w7_arr = [], []
                for i in range(len(fs_idx)):
                    ev_k = ev[fs_idx[i]]
                    ec_k = ec[fs_idx[i]]
                    weights = np.exp(-ev_k**2 / (2 * window**2))
                    weights[ev_k < 0] *= 0.5
                    weights /= np.sum(weights)
                    w6 = float(np.sum(weights * np.sum(np.abs(ec_k[0:2, :])**2, axis=0)))
                    w7 = float(np.sum(weights * np.sum(np.abs(ec_k[2:4, :])**2, axis=0)))
                    norm = max(w6 + w7, 1e-12)
                    w6_arr.append(w6 / norm)
                    w7_arr.append(w7 / norm)
                vF_G6 = float(np.average(vF_arr, weights=np.array(w6_arr) + 1e-12))
                vF_G7 = float(np.average(vF_arr, weights=np.array(w7_arr) + 1e-12))
                xi_Gamma6 = vF_G6 / (np.pi * Delta_0)
                xi_Gamma7 = vF_G7 / (np.pi * Delta_0)
                orbital_selective = abs(xi_Gamma6 - xi_Gamma7) / max(xi_over_a, 1e-12) > 0.15
            else:
                xi_Gamma6 = xi_over_a
                xi_Gamma7 = xi_over_a
                orbital_selective = False
            
            valid_BdG = xi_over_a > 2.0
        else:
            xi_Gamma6 = 0.0
            xi_Gamma7 = 0.0
            xi_over_a = 0.0
            vF_avg = 0.0
            valid_BdG = False
            orbital_selective = False
        
        return {
            'xi_Gamma6': xi_Gamma6,
            'xi_Gamma7': xi_Gamma7,
            'xi_over_a': xi_over_a,
            'vF_avg': vF_avg,
            'valid_BdG': valid_BdG,
            'orbital_selective': orbital_selective,
            'lambda_lin_max': lambda_lin_max,
            'lambda_JT_kernel': lambda_JT_kernel,
            'g_delta_dom': g_delta_dom,
            'gap_vector': gap_vector,
            'fs_pts': fermi_pts,
            'gap_symmetry': gap_symmetry,
            'V_spin_mean': V_spin_mean,
            'V_JT_mean': V_JT_mean,
            'V_cross_mean': V_cross_mean,
            'V_rpa_mean': V_rpa_mean
        }
        
    def _compute_orbital_coherence_from_pairs(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, _bdg_cache: tuple = None) -> float:
        """
        Anomalous SC-orbital coherence ⟨τ_x⟩_anom from off-diagonal BdG amplitudes (u·v). This probes the same Γ₆↔Γ₇ off-diagonal coherence driven by B1g_op.

        Definition:
            ⟨τ_x⟩_anom = Σ_k (1−2f_n) Re[u*_6 v_7 + h.c.]

        Properties:
            Δ = 0  →  ⟨τ_x⟩_anom = 0  (selection rule exact in D₄h)
            Δ ≠ 0  →  ⟨τ_x⟩_anom ≠ 0, SC condensate unlocks B1g JT channel (only Q≠0)

        _bdg_cache : (ev, ec) from the current SCF iteration — reuse to avoid a
            redundant eigh call AND to prevent the out= buffer from corrupting the
            live _scf_bdg_cache tuple.
        """
        vbdg = self._get_vbdg()
        if _bdg_cache is not None:
            ev_all, ec_all = _bdg_cache
        else:
            ev_all, ec_all = np.linalg.eigh(
                vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J))
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

    def build_local_hamiltonian_for_bdg(self, sign_M: float, M: float, Q: float, mu: float, g_J: float, target_doping: float, tx: float, ty: float, tau_x_anom_mf: float = 0.0) -> np.ndarray:
        """
        Local 4×4 BdG Hamiltonian for one sublattice, basis [6↑, 6↓, 7↑, 7↓].
        sign_M = ±1 for sublattices A/B (staggered AFM).

        Terms:
          1. Chemical potential −μ
          2. Crystal field splitting Δ_CF on Γ₇
          3. Longitudinal (diagonal) AFM Weiss field from J_A1g:
               h_z[α] = sign_M · J_A1g[α,α] · g_J·(1−δ)·M·sz[α]
             J_A1g = J_CT · cosh(2Q/λ) · diag(1, 1, η_J², η_J²)
             This is purely diagonal (spin-preserving). It shifts the orbital energies.
          4. Transverse (off-diagonal) anomalous Weiss field from J_B1g:
               H_B1g = −sign_M · J_B1g_scalar · ⟨τ_x⟩_anom · τ_x_mat
             J_B1g_scalar = J_CT · sinh(2Q/λ) · η_J
             tau_x_mat is off-diagonal in the 4×4 basis: couples 6↑↔7↑ and 6↓↔7↓.
             This term is ZERO in the normal state (Q=0 → sinh=0; or ⟨τ_x⟩_anom=0).
             It is activated only when BOTH Q≠0 AND the SC condensate generates
             anomalous orbital coherence ⟨τ_x⟩_anom ≠ 0.
             Placing it off-diagonal is physically correct: J_B1g mixes Γ₆ and Γ₇,
             it does NOT shift their energies diagonally.
          5. JT distortion: H_JT = g_JT · Q · B1g_op

        tau_x_anom_mf: Gutzwiller-renormalised anomalous orbital coherence
            ⟨τ_x⟩_renorm = g_Delta_s · ⟨τ_x⟩_BdG, passed from the SCF loop.
            Default 0.0 (normal state, first iterations, chi0 calls).

        SC–JT chain: Δ≠0 → ⟨τ_x⟩_anom≠0 → H_B1g≠0 → mixes Γ₆/Γ₇ → modifies
        spectrum → feeds back into gap equation self-consistently.
        """
        H = np.zeros((4, 4), dtype=complex)

        # 1. Chemical potential
        np.fill_diagonal(H, -mu)

        # 2. Crystal field splitting Δ_CF on Γ₇
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓

        # 3. AFM Weiss field: longitudinal (J_A1g) + transverse (J_B1g)
        J_A1g_diag, J_B1g_scalar = self._exchange_channels(Q)
        abs_delta = max(abs(target_doping), 1e-6)
        O_exp_z   = g_J * (1.0 - abs_delta) * M * self.sz_op / 2.0  # /2 factor due to the Mean-Field Weiss field related to one longitudinal spin
        h_z = sign_M * (J_A1g_diag * O_exp_z)

        H[0, 0] -= h_z[0]   # 6↑
        H[1, 1] -= h_z[1]   # 6↓
        H[2, 2] -= h_z[2]   # 7↑
        H[3, 3] -= h_z[3]   # 7↓

        # Transverse term: active only when Q≠0 AND condensate has ⟨τ_x⟩_anom≠0.
        # H_Weiss = −J·⟨O⟩; J_B1g is off-diagonal (6↑↔7↑, 6↓↔7↓), never diagonal.
        if abs(tau_x_anom_mf) > 1e-12:
            H -= sign_M * (J_B1g_scalar * tau_x_anom_mf) * self.tau_x_mat

        # 4. JT distortion: H_JT = g_JT · Q · B1g_op
        H += (self.p.g_JT * Q) * self.B1g_op
        return H

    def build_single_particle_hamiltonian(self, Q: float, mu: float) -> np.ndarray:
        """Non-magnetic single-particle Hamiltonian (cluster ED input, no Weiss field)."""
        H = np.zeros((4, 4), dtype=complex)
        np.fill_diagonal(H, -mu)
        H[2, 2] += self.p.Delta_CF
        H[3, 3] += self.p.Delta_CF
        H += (self.p.g_JT * Q) * self.B1g_op
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
    
    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, tx: float, ty: float, mu_guess: float, g_J: float) -> Tuple[float, Optional[tuple], float]:
        """
        Returns (mu, last_bdg_cache, n_at_mu) where:
          - last_bdg_cache = (ev, ec) at the converged mu — caller can skip a redundant eigh.
          - n_at_mu        = electron density at the converged mu
        """
        target_n = 1.0 - target_doping
        _last_cache: Optional[tuple] = None
        _last_n: float = target_n   # updated each Newton/brentq step; returned to caller

        def density_and_deriv(mu_val: float):
            nonlocal _last_cache, _last_n
            vbdg = self._get_vbdg()
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu_val, tx, ty, g_J, out=vbdg._H_stack))
            _last_cache = (ev, ec)
            f   = self.fermi_function(ev)
            fb  = 1.0 - f

            uA, uB, vA, vB = VectorizedBdG._get_nambu_spinors(ec)
            dens_A, dens_B = VectorizedBdG._compute_densities(uA, uB, vA, vB, f, fb)
            n = float(np.dot(self.k_weights, dens_A + dens_B)) / 4.0
            _last_n = n

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
                return mu, _last_cache, _last_n
            if abs(deriv) < _DEN_DERIV_FLOOR:
                break   # flat → fall through to brentq
            step = err / max(abs(deriv), 1e-10)
            step = float(np.clip(step, -self.p.t0, self.p.t0))
            mu -= step

        # Fallback: brentq (rare — only if Newton diverges or lands on flat region)
        def density_error(mu_val):
            nonlocal _last_cache, _last_n
            vbdg = self._get_vbdg()
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu_val, tx, ty, g_J, out=vbdg._H_stack))
            _last_cache = (ev, ec)
            f  = self.fermi_function(ev)
            fb = 1.0 - f
            uA, uB, vA, vB = VectorizedBdG._get_nambu_spinors(ec)
            dA, dB = VectorizedBdG._compute_densities(uA, uB, vA, vB, f, fb)
            n_val = float(np.dot(self.k_weights, dA + dB)) / 4.0
            _last_n = n_val
            return n_val - target_n

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
                return mu, _last_cache, _last_n
        except Exception:
            pass
        return mu, _last_cache, _last_n
    
    def compute_bdg_free_energy(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, V_s: float, V_d: float, K_eff_for_free_energy: float, _ev_cache: np.ndarray = None) -> float:
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
        elastic_energy = 0.5 * K_eff_for_free_energy * Q**2

        # Condensation correction: |Δ_ℓ|² / (g_ℓ · V_ℓ)
        _, _, g_s, g_d = self.p.get_gutzwiller_factors(target_doping)

        condensation = 0.0
        if V_s > 0.0:
            condensation += abs(Delta_s)**2 / (g_s * V_s)
        if V_d > 0.0:
            condensation += abs(Delta_d)**2 / (g_d * V_d)

        Omega_cell = Ef - S_term
        return 0.5 * Omega_cell + elastic_energy + condensation
    
    def compute_cluster_free_energy(self, M_ext: float, Q: float, mu: float, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> Dict:
        """
        Cluster exact diagonalization with DMFT-like vertex correction.

            H = H_0 + J_eff · O_AB, where O_AB = O_A ⊗ O_B
        
        Hellmann-Feynman: dE_n/dJ_eff = ⟨ψ_n| O_AB |ψ_n⟩, extracts renormalized J_eff from the full cluster spectrum via
        linear regression: E_n = const + J_eff · ⟨O_A ⊗ O_B⟩_n, weighted by Boltzmann factors to emphasize low-energy states.

        H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)               [single-particle terms]
                  + J·O_A⊗O_B                                [intra-cluster multipolar exchange]
                  + H_boundary(A) ⊗ I + I ⊗ H_boundary(B)   [inter-cluster MF]

        J_bond_bare: Gutzwiller-renormalised Heisenberg superexchange (ZSA CT + kinematic dd).
                     Boundary coupling: h_bound = Z_BOUNDARY · J_eff · (M_ext/2) · O, consistent
                     with BdG Weiss field h_afm = g_J·f_d·Z·J_CT·M/2. It is not allowed to explicitly
                     include the Stoner term, since J_CT already derives from 2nd-order in t/U.
        """
        # Single-particle Hamiltonians
        H_sp_A = self.build_single_particle_hamiltonian(Q, mu)
        H_sp_B = self.build_single_particle_hamiltonian(Q, mu)

        # Bare exchange and Stoner scaling
        J_bond_bare = self.p.effective_superexchange(g_J, tx_bare, ty_bare, doping)
        I4 = np.eye(4, dtype=complex)

        # ── Build cluster Hamiltonian ────────────────────────────────────────────
        H_int = np.kron(H_sp_A, I4) + np.kron(I4, H_sp_B)
        H_int += J_bond_bare * np.kron(self.multi_op, self.multi_op)

        H_bound_op = (self.p.Z - 1) * J_bond_bare * (M_ext / 2.0) * self.multi_op
        H_bound_full = np.kron(-H_bound_op, I4) + np.kron(I4, H_bound_op)

        H_cluster = H_int + H_bound_full
        evals, evecs = np.linalg.eigh(H_cluster)

        # ── Free energy: uses FULL evals (boundary coupling is real thermodynamics) ─
        if self.p.kT < _KT_FLOOR:
            F_total = evals[0]
        else:
            weights = np.exp(-(evals - evals[0]) / self.p.kT)
            F_total = evals[0] - self.p.kT * np.log(weights.sum())

        # ── Observables ─────────────────────────────────────────────────────────
        M_A = self.cluster_expectation(evals, evecs, self.multi_op, self.p.kT, site_index=0)
        M_B = self.cluster_expectation(evals, evecs, self.multi_op, self.p.kT, site_index=1)
        M_cluster = abs(M_A - M_B) / 2.0

        B1g_op = self.B1g_op
        Q_A  = self.cluster_expectation(evals, evecs, B1g_op, self.p.kT, site_index=0)
        Q_B  = self.cluster_expectation(evals, evecs, B1g_op, self.p.kT, site_index=1)
        B1g_sq = B1g_op @ B1g_op
        Q2_A = self.cluster_expectation(evals, evecs, B1g_sq, self.p.kT, site_index=0)
        Q2_B = self.cluster_expectation(evals, evecs, B1g_sq, self.p.kT, site_index=1)
        Q_fluct = np.sqrt(max(0.0, 0.5 * ((Q2_A - Q_A**2) + (Q2_B - Q_B**2))))

        # ── Connected correlator (state-by-state, removes classical MF bias) ────
        O_AB = np.kron(self.multi_op, self.multi_op)
        O_A  = np.kron(self.multi_op, I4)
        O_B  = np.kron(I4, self.multi_op)

        corr_raw  = np.einsum('ij,ij->j', evecs.conj(), O_AB @ evecs).real
        mA_states = np.einsum('ij,ij->j', evecs.conj(), O_A  @ evecs).real
        mB_states = np.einsum('ij,ij->j', evecs.conj(), O_B  @ evecs).real
        corr_vals = corr_raw - mA_states * mB_states

        # ── Weiss-field decontamination of eigenvalues for regression ───────────
        # ΔE_n^(bound) = ⟨ψ_n| H_bound_full |ψ_n⟩: the Weiss-field energy per eigenstate.
        # Subtracting this from evals leaves evals_int whose variation with <O_AB>_n
        # is driven purely by the internal J·O_A⊗O_B term — the quantity we want to fit.
        bound_exp = np.einsum('ij,ij->j', evecs.conj(), H_bound_full @ evecs).real
        evals_int = evals - bound_exp   # internal energy: E_n - <H_bound>_n

        # ── Fit temperature ──────────────────────────────────────────────────────
        _T_floor  = max(self.p.kT, 0.1 * np.sqrt(0.5 * (tx_bare**2 + ty_bare**2)), 1e-4)
        T_fit_raw = min(max(_T_floor, abs(J_bond_bare) * 0.5), abs(J_bond_bare) * 2.0)
        T_fit     = T_fit_raw if T_fit_raw >= _T_floor else _T_floor

        # Boltzmann weights from FULL evals (thermal population is physical)
        fit_weights = np.exp(-(evals - evals[0]) / T_fit)
        fit_weights /= fit_weights.sum()

        # ── Weighted regression on decontaminated internal energies ─────────────
        corr_mean  = np.sum(fit_weights * corr_vals)
        E_int_mean = np.sum(fit_weights * evals_int)   # mean of internal energy

        dc = corr_vals - corr_mean
        dE = evals_int - E_int_mean                    # variation of INTERNAL energy only

        w_cov = np.sum(fit_weights * dE * dc)
        w_var = np.sum(fit_weights * dc * dc)

        if w_var > _VAR_MIN:
            J_raw = w_cov / w_var
            r_J   = J_raw / max(J_bond_bare, 1e-12)
            # Variance-adaptive upper bound: large variance → regression reliable → allow wider range.
            # Physical ceiling at 3.0 (U/J scale); variance floor prevents noise-driven overestimate.
            _r_J_max = min(3.0, 1.0 + 2.0 * (w_var / (abs(corr_mean) + 1e-6)))
            r_J = float(np.clip(r_J, 0.3, _r_J_max))
        else:
            r_J = 1.0   # variance collapsed: extreme AFM freezing, use bare exchange

        J_bond_cluster = r_J * J_bond_bare
        return {
            'F_per_site':    F_total / _CLUSTER_SIZE,
            'M':             M_cluster,
            'Q_fluctuation': Q_fluct,
            'J_eff':         self.p.Z * J_bond_cluster,
            'j_renorm':      r_J,
        }

    def _scf_jacobi_kick(self, target_doping: float, initial_M: float, initial_Q: float, initial_Delta: float, g_t: float, g_J: float, g_Delta_eff: float, mu: float) -> Dict:
        """
        Estimate the dominant Jacobi eigenvalue λ₊ of the two-channel (Δ, Q) SCF map
        and generate physics-informed seed values for (M, Q, Δ_s, Δ_d).

        Linearised Jacobian of the (Δ, Q) fixed-point map:
            J = [ A   B ]    A = g_Δ·V_pair·N₀
                [ C   0 ]    C = (g_JT/K)·χ_τ
        B_raw = A·(g_JT²/(Δ_CF·K))·(χ_τ/N₀), Padé saturated: B = B_raw/(1+B_raw/|A|)
        λ₊ = ½[A + √(A²+4BC)]; complex → spectral radius = ½√(A²+|disc|)
        Regimes: λ₊<0.7 subcritical, λ₊∈[0.7,1.4] critical, λ₊>1.4 supercritical.
        """
        t_eff = g_t * self.p.t0
        N0 = 1.0 / (np.pi * max(t_eff, 1e-6))

        # ---- Mott guard ----
        if g_t < _G_T_COHERENCE_MIN:
            return {
                'M_kick':        float(np.clip(initial_M, 0.05, 0.45)),
                'Q_kick':        0.0,
                'Delta_s_kick':  0.0j,
                'Delta_d_kick':  0.0j,
                'mixing_kick':   _MIXING * 0.5,
                'lambda_plus':   0.0,
                'lambda_lin_max': 0.0,
                'regime':        'mott',
                'J_eff_scalar':  0.0,
                'chi0_moriya':   0.0,
                'chi_tau_val':   0.0,
                'M_phys':        float(np.clip(initial_M, 0.05, 0.45)),
            }

        # ---- χ₀ estimate + Moriya damping ----
        chi0_est = N0 / (1.0 + (self.p.U_mf / max(np.pi * t_eff, 1e-9))**2)

        J_eff_scalar = self.p.Z * self.p.effective_superexchange(g_J, self.p.t0, self.p.t0, target_doping)
        _Gamma_M = self.p.moriya_gamma(target_doping, t_eff, J_eff_scalar)
        chi0_moriya = chi0_est / (1.0 + _Gamma_M * max(chi0_est, 0.0))

        # ---- Spin vertex (RPA-like proxy) ----
        denom = max(1.0 - J_eff_scalar * chi0_moriya, 0.1)
        V_spin_est = (J_eff_scalar**2 * chi0_moriya) / denom

        # ---- Total pairing vertex (Proxy for Jacobian A) ----
        V_JT_bare = self.p.g_JT**2 / max(self._K_bare, 1e-9)
        V_pair = max(V_spin_est + V_JT_bare, V_JT_bare)

        # ---- Anisotropic hopping ----
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(initial_Q)
        tx = g_t * tx_bare
        ty = g_t * ty_bare

        # ---- Linearised BdG+RPA eigenproblem ----
        _lin_seed = self.solve_linearized_gap_equation(
            initial_M, initial_Q, 0.0j, 0.0j, target_doping, mu,
            tx, ty, g_J, self._K_bare, J_eff_scalar, actual_doping=target_doping)

        lambda_lin_max = float(_lin_seed['lambda_lin_max'])

        # ---- Get physically motivated M estimate for χ_τ calculation ----
        _M_phys = self.p.estimate_M0(target_doping, lambda_lin_max)

        # ---- χ_τ (Magnitude of softening) ----
        # Linear response that seeds SC→JT. chi_tau < 0 → lattice softening.
        chi_tau_raw = self._compute_chi_tau(_M_phys, initial_Q, 0.0+0j, 0.0+0j, target_doping, mu, tx, ty, g_t, g_J)['chi_tau_n']
        
        # We need the magnitude of softening for the Jacobian C element
        chi_tau_val = max(-chi_tau_raw, 0.0) if chi_tau_raw < -1e-6 else 0.0

        # ---- Jacobian elements ----
        A = g_Delta_eff * V_pair * N0
        B_raw = A * (self.p.g_JT**2 / (max(self.p.Delta_CF, 1e-9) * max(self._K_bare, 1e-9))) \
                * (chi_tau_val / max(N0, 1e-12))
        B = B_raw / (1.0 + B_raw / max(abs(A), 1e-8))  # Stable Padé
        C = (self.p.g_JT**2 / max(self._K_bare, 1e-9)) * chi_tau_val

        # ---- Eigenvalue (λ₊) ----
        disc = A**2 + 4.0 * B * C
        if disc >= 0.0:
            lambda_plus = 0.5 * (A + np.sqrt(disc))
        else:
            # Complex eigenvalues (B·C < 0): use spectral radius
            lambda_plus = 0.5 * np.sqrt(A**2 + abs(disc))

        # ---- Regime classification ----
        if lambda_plus < 0.7:
            regime = 'subcritical'
        elif lambda_plus <= 1.4:
            regime = 'critical'
        else:
            regime = 'supercritical'

        # ---- Q kick ----
        Q_kick = initial_Q if abs(initial_Q) > 1e-6 else 1e-6 * np.random.uniform(-1, 1)
        Q_kick = float(np.clip(Q_kick, -0.05 * self.p.lambda_hop, 0.05 * self.p.lambda_hop))

        # ---- M kick: blend initial_M toward physics estimate ----
        if lambda_lin_max > 1.0:
            w_M = float(np.clip((lambda_lin_max - 1.0) / 10.0, 0.0, 0.8))
            M_kick = (1.0 - w_M) * initial_M + w_M * _M_phys
        else:
            M_kick = initial_M

        # Overshoot protection: if initial_M is suspiciously large relative to the AFM stiffness and J·χ is near-critical, reduce it.
        jchi = J_eff_scalar * chi0_moriya
        m_excess = max(0.0, (initial_M - 0.7) / 0.3)
        stiffness_deficit = max(0.0, (jchi - 0.7) / 0.3)
        lambda_excess     = max(0.0, lambda_plus - 1.0)
        reduction = 0.35 * m_excess * stiffness_deficit * (lambda_excess / (1.0 + lambda_excess))
        M_kick = M_kick * (1.0 - reduction)

        # Supercritical extra pull: if _M_phys < M_kick, gently pull down.
        if lambda_lin_max > 5.0:
            pull = float(np.clip((lambda_lin_max - 5.0) / 15.0, 0.0, 0.6))
            if _M_phys < M_kick:
                M_kick = M_kick - pull * (M_kick - _M_phys)

        M_kick = float(np.clip(M_kick, 0.02, 0.45))
        fs_pts = _lin_seed['fs_pts']
        gap_vec = _lin_seed['gap_vector']

        phi_s = np.ones(len(fs_pts))
        phi_s /= np.linalg.norm(phi_s)

        phi_d = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
        phi_d /= max(np.linalg.norm(phi_d), 1e-10)

        w = np.abs([gap_vec @ phi_s, gap_vec @ phi_d])
        frac = w / max(w.sum(), 1e-10)

        # ---- Δ kick ----
        base_scale = float(np.clip(
            t_eff * np.exp(-1.0 / max(lambda_lin_max, 0.1)) * 0.1,
            1e-6, 0.3 * t_eff
        ))
        boost = 1.0 + 3.0 * lambda_excess / (1.0 + lambda_excess)
        Delta_kick = float(np.clip(base_scale * boost, 1e-6, 0.3 * t_eff))

        Delta_s_kick = complex(Delta_kick * frac[0])
        Delta_d_kick = complex(Delta_kick * frac[1])

        # ---- Adaptive mixing ----
        lin_factor = 1.0 + np.log1p(max(0.0, lambda_lin_max))
        jac_factor = 1.0 + 0.3 * np.log1p(max(0.0, lambda_plus))
        mixing_kick = max(_MIXING / (lin_factor * jac_factor), _MIXING / 16.0)

        # Supercritical mixing reduction
        if lambda_lin_max > 5.0:
            sc_factor = 1.0 + 2.0 * np.log10(max(lambda_lin_max / 5.0, 1.0))
            mixing_kick = max(mixing_kick / sc_factor, _MIXING / 16.0)

        return {
            'M_kick':        M_kick,
            'Q_kick':        Q_kick,
            'Delta_s_kick':  Delta_s_kick,
            'Delta_d_kick':  Delta_d_kick,
            'mixing_kick':   mixing_kick,
            'lambda_plus':   lambda_plus,
            'lambda_lin_max': lambda_lin_max,
            'regime':        regime,
            'J_eff_scalar':  J_eff_scalar,
            'chi0_moriya':   float(chi0_moriya),
            'chi_tau_val':   float(chi_tau_val),
            'M_phys':        _M_phys,
        }

    def _compute_2x2_pairing_kernel(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, K_eff: float, g_Delta_s: float, g_Delta_d: float, j_renorm: float, z_qp: float = 1.0) -> dict:
        """
        Build the 2×2 pairing kernel matrix K_pair in the (s, d) channel basis
        and return its dominant eigenvector as the optimal Δ direction.

        Instead of treating Δ_s and Δ_d as independent fixed-point scalars, we build
        the symmetric kernel:
            K_pair = [[K_ss,  K_sd],
                      [K_sd,  K_dd]]
        where:
            K_ss  = g_s  · Σ_{k,k'} φ_s(k) · V(k−k') · φ_s(k') / √(vF_k · vF_k') / N_s²
            K_dd  = g_d  · Σ_{k,k'} φ_d(k) · V(k−k') · φ_d(k') / √(vF_k · vF_k') / N_d²
            K_sd  = √(g_s·g_d) · Σ_{k,k'} φ_s(k) · V(k−k') · φ_d(k') / √(vF_k · vF_k') / √(N_s·N_d)

        V(k−k') is the full RPA vertex evaluated at q=k−k' using the Δ=0 χ₀
        and the same susceptibility infrastructure as solve_linearized_gap_equation.

        V_sd is computed as the proper FS-averaged cross-vertex integral:
            V_sd = [Σ_{k,k'} φ_s(k)·V(k−k')·φ_d(k')·w(k)·w(k')] / [Σ_{k,k'} φ_s(k)·φ_d(k')·w(k)·w(k')]
        where w(k) = 1/√(vF_k) is the FS DOS weight.

        The dominant eigenvector (Δ_s*, Δ_d*)/|(Δ_s*, Δ_d*)| gives the optimal
        hybridisation between s and d pairing channels, automatically:
          • mixing channels when Γ₆–Γ₇ off-diagonal elements are non-negligible,
          • finding the optimum B₁g hybrid when both channels have similar λ,
          • avoiding artificial channel locking from independent fixed-point iteration.

        Returns
        -------
        dict with keys:
          'vec_s'      : s component of dominant eigenvector (real, signed)
          'vec_d'      : d component of dominant eigenvector
          'lambda_2x2' : dominant eigenvalue of K_pair
          'K_pair'     : 2×2 array
          'V_ss'       : FS-averaged s-channel vertex  (diagnostic)
          'V_dd'       : FS-averaged d-channel vertex  (diagnostic)
          'V_sd'       : FS-averaged cross vertex      (diagnostic)
        """
        try:
            vbdg = self._get_vbdg()
            fs_pts, vF_arr, _, _, _ = self._get_fs_points(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                store_cache=False, compute_vF=True)
            N_fs = len(fs_pts)
            if N_fs < 4:
                return {'vec_s': 1.0, 'vec_d': 0.0, 'lambda_2x2': 0.0,
                        'K_pair': np.zeros((2, 2)), 'V_ss': 0.0, 'V_dd': 0.0, 'V_sd': 0.0}

            phi_s = np.ones(N_fs, dtype=float)
            phi_d = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
            inv_vF = 1.0 / np.maximum(np.abs(vF_arr), _VF_FLOOR_TIGHT)
            inv_svF = np.sqrt(inv_vF)   # 1/√|vF| DOS weight

            # DOS normalisers: N_eff_a = Σ_k φ_a(k)² / |vF_k|
            ns  = max(float(np.dot(phi_s**2, inv_vF)), 1e-12)
            nd  = max(float(np.dot(phi_d**2, inv_vF)), 1e-12)

            # ── Proper FS-averaged vertex integrals (TODO resolved) ────────────────
            # V_ab = Σ_{k,k'} φ_a(k)·V(k−k')·φ_b(k')·w_k·w_k' / (N_a · N_b)
            # where w_k = 1/√|vF_k|.
            # We need unique q=k−k' vectors and V(q) for each
            tx_bare_v, ty_bare_v = self.p.effective_hopping_anisotropic(Q)
            J_eff_loc = j_renorm * self.p.Z * self.p.effective_superexchange(
                g_J, tx_bare_v, ty_bare_v, target_doping)
            chi_QQ_n  = self._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)
            E_k_cache = self._get_chi0_norm_cache(M, Q, mu, tx, ty, g_J, vbdg, target_doping)

            # Build q=k_i−k_j for all FS pairs; deduplicate exactly as in solve_linearized_gap_equation.
            i_idx, j_idx = np.triu_indices(N_fs)
            q_raw  = fs_pts[i_idx] - fs_pts[j_idx]
            q_arr  = (q_raw + np.pi) % (2 * np.pi) - np.pi
            scale  = 1e5
            q_int  = np.rint(q_arr * scale).astype(np.int64)
            unique_int, inv_idx = np.unique(q_int, axis=0, return_inverse=True)
            unique_q = unique_int.astype(np.float64) / scale

            # Compute V(q) for each unique q — reusing the same susceptibility machinery.
            n_q = len(unique_q)
            V_unique = np.empty(n_q, dtype=float)
            for u_idx, q_u in enumerate(unique_q):
                _sus = self.get_susceptibilities_normal(
                    q_u, M, Q, target_doping, target_doping, mu, tx, ty,
                    g_J, chi_QQ_n, K_eff, J_eff_loc, E_k_cache, z_qp=z_qp)
                V_unique[u_idx] = _sus['V_full']

            # FS-weight product w_k · w_k' for each (i,j) pair.
            w_ij = inv_svF[i_idx] * inv_svF[j_idx]   # shape (n_pairs,)
            V_ij  = V_unique[inv_idx]                  # V(k_i − k_j)

            # Symmetrise: each upper-triangle pair (i≠j) contributes twice.
            def _fs_avg(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
                """Σ_{k,k'} φ_a(k)·V(k−k')·φ_b(k')·w_k·w_k' with proper symmetrisation."""
                # Upper triangle (i,j), i<j: each pair counted once → multiply by 2.
                # Diagonal (i=i): counted once.
                contrib = phi_a[i_idx] * V_ij * phi_b[j_idx] * w_ij
                total = float(np.sum(contrib))
                # Add lower-triangle contribution (transpose of upper, excluding diag).
                diag_mask = (i_idx == j_idx)
                total_offdiag_upper = float(np.sum(contrib[~diag_mask]))
                total = float(np.sum(contrib[diag_mask])) + 2.0 * total_offdiag_upper
                return total

            V_ss_raw = _fs_avg(phi_s, phi_s)
            V_dd_raw = _fs_avg(phi_d, phi_d)
            V_sd_raw = _fs_avg(phi_s, phi_d)

            # Normalise by N_eff_a · N_eff_b so K_ab is dimensionally consistent
            # with the BCS coupling constant λ_ab = g_a · V_ab / N_eff_ab.
            V_ss = V_ss_raw / max(ns, 1e-12)
            V_dd = V_dd_raw / max(nd, 1e-12)
            V_sd = V_sd_raw / max(float(np.sqrt(ns * nd)), 1e-12)

            # ── Build 2×2 kernel ──────────────────────────────────────────────────
            K11 = g_Delta_s * V_ss
            K22 = g_Delta_d * V_dd
            K12 = float(np.sqrt(max(g_Delta_s * g_Delta_d, 0.0))) * V_sd

            K_pair = np.array([[K11, K12], [K12, K22]], dtype=float)
            evals, evecs = np.linalg.eigh(K_pair)
            # Dominant (most unstable) eigenvector: last column (ascending order from eigh).
            lam_2x2 = float(evals[-1])
            v = evecs[:, -1]
            return {
                'vec_s':      float(v[0]),
                'vec_d':      float(v[1]),
                'lambda_2x2': lam_2x2,
                'K_pair':     K_pair,
                'V_ss':       V_ss,
                'V_dd':       V_dd,
                'V_sd':       V_sd,
            }
        except Exception:
            return {'vec_s': 1.0, 'vec_d': 0.0, 'lambda_2x2': 0.0,
                    'K_pair': np.zeros((2, 2)), 'V_ss': 0.0, 'V_dd': 0.0, 'V_sd': 0.0}

    def _dlambda_dQ_fs(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, K_eff: float, j_renorm: float, z_qp: float = 1.0, dQ: float = _DQ_FS_VERTEX) -> float:
        """
        FS-resolved ∂λ_pair/∂Q via finite difference of the full pairing eigenvalue.

        Computes the linearised gap eigenvalue λ_max at Q±dQ and returns the central
        difference.  This gives a locally resolved measure of how much a small JT
        distortion enhances the dominant pairing channel at every k on the Fermi surface.

        A positive return value confirms SC-triggered JT consistency: the condensate
        (existing Δ) strengthens the pairing kernel when Q deforms the lattice.

        The Fermi surface points and velocities change only weakly with Q over the tiny
        step dQ ~ 0.02 Å.  We therefore compute fs_pts and vF_arr ONCE at the middle Q
        and reuse them for both Q+dQ and Q-dQ evaluations of λ_max. The residual error
        from the Q-frozen FS is O(dQ²) in λ_max
        """
        try:
            g_t_loc, g_J_loc, g_ds, g_dd = self.p.get_gutzwiller_factors(target_doping)

            # Compute FS geometry once at the middle Q: the FS geometry changes negligibly over ±dQ
            # so we freeze it and only recompute the pairing vertex V(q) (which is computed from Δ=0 χ₀ anyway).
            tx_mid, ty_mid = g_t_loc * self.p.effective_hopping_anisotropic(Q)[0], \
                             g_t_loc * self.p.effective_hopping_anisotropic(Q)[1]
            try:
                fs_pts_mid, vF_mid, _, _, _ = self._get_fs_points(
                    M, Q, Delta_s, Delta_d, target_doping, mu, tx_mid, ty_mid, g_J_loc,
                    store_cache=False, compute_vF=True)
                _fs_precomputed = (fs_pts_mid, vF_mid) if len(fs_pts_mid) >= 4 else None
            except Exception:
                _fs_precomputed = None

            def _lmax_at_Q(Qv: float) -> float:
                tx_b, ty_b = self.p.effective_hopping_anisotropic(Qv)
                tx_v = g_t_loc * tx_b
                ty_v = g_t_loc * ty_b
                J_eff_v = j_renorm * self.p.Z * self.p.effective_superexchange(
                    g_J_loc, tx_b, ty_b, target_doping)
                K_eff_v = max(K_eff, 1e-9)
                lin = self.solve_linearized_gap_equation(
                    M, Qv, Delta_s, Delta_d, target_doping, mu, tx_v, ty_v, g_J_loc,
                    K_eff_v, J_eff_v, actual_doping=target_doping, z_qp=z_qp,
                    _fs_precomputed=_fs_precomputed)
                return float(lin['lambda_lin_max'])

            lp = _lmax_at_Q(Q + dQ)
            lm = _lmax_at_Q(Q - dQ)
            return (lp - lm) / (2.0 * dQ)
        except Exception:
            return 0.0

    def _detect_limit_cycle(self, delta_history: list) -> bool:
        """
        detect oscillatory / limit-cycle behaviour in |Δ|(iteration).

        Returns True when the relative standard deviation of |Δ| over the last
        _CYCLE_WINDOW iterations exceeds _CYCLE_THRESHOLD, which indicates the SCF
        is trapped in an oscillation rather than converging to a fixed point.
        In such cases the caller should damp α further (_CYCLE_DAMP_FAC).
        """
        w = _CYCLE_WINDOW
        if len(delta_history) < w:
            return False
        arr = np.array(delta_history[-w:], dtype=float)
        mean = float(np.mean(arr))
        if mean < 1e-10:
            return False
        std = float(np.std(arr))
        return (std / mean) > _CYCLE_THRESHOLD

    def solve_self_consistent(self, target_doping: float, initial_M: float, initial_Q: float, initial_Delta: float, verbose: bool = True, _ic_retry: bool = False) -> Dict:
        """
        Coupled (M, Q, Δ_s, Δ_d, μ) SCF via Anderson(5)-accelerated fixed-point + LM Newton.

        Order parameters
        ----------------
        M       : staggered AFM magnetisation (Gutzwiller-renormalised BdG Weiss field)
        Q       : B₁g JT lattice distortion (Å); zero in the AFM normal state by symmetry
        Δ_s     : on-site inter-orbital (Γ₆⊗Γ₇) singlet pairing amplitude (eV)
        Δ_d     : inter-site d-wave B₁g singlet pairing amplitude (eV)
        μ       : chemical potential enforcing ⟨n⟩ = 1 − δ (Newton + Brentq fallback)

        Per-iteration algorithm:
          1. BdG diagonalisation H(k; M,Q,Δ,μ) → observables (M_BdG, ⟨τ_x⟩, ⟨B₁g⟩, pairs).
          2. SC+JT active (Δ>0, Q>0): inject ⟨τ_x⟩_anom into J_B1g off-diagonal Weiss field,
             rebuild BdG cache.
          3. Update K_eff=K_lattice+∂²F_ex/∂Q² when |ΔM|>0.02 or j_renorm/Q changed.
          4. Gap equation Δ_out = g_Δ·V(q)·F_AA/AB via RPA vertex (always from Δ=0 χ₀).
             After fixed-point gap update, blend in the 2×2 pairing kernel eigenvector direction
             to ensure the SCF can always find a slope toward the dominant instability channel even when |Δ| ≈ 0.
             The 2×2 kernel K_pair is built from (V_s, V_d, V_sd) projections;
             its eigenvector defines the optimal (Δ_s*, Δ_d*) ratio that
             is mixed in with weight _ALPHA_MIX_2X2, preventing artificial channel locking.
          5. Cluster ED → j_renorm (EMA-smoothed), z_qp=1/r_J for Moriya boost.
          6. Newton step for M (LM-damped, trust-region); blend with BdG fixpoint.
          7. Hierarchical SCF: Q updates only every
             _Q_UPDATE_PERIOD iterations via Hellmann-Feynman; intervening iterations
             converge (Δ,M,μ) at frozen Q, consistent with adiabatic lattice timescale.
          8. Anderson(5) mix (M, Q, |Δ_s|, |Δ_d|); find μ enforcing density.
          9. Adaptive α via Λ_inst = EMA[max(λ_pair, λ_JT, J·χ_SS)]:
             α_eff = α₀/(1+Λ); halved + history reset on divergence;
             past-QCP (det<0): exponential penalty on α; near-QCP: hard cap.
             FS-resolved ∂λ/∂Q computed at post-convergence to confirm SC-triggered JT consistency on hot-spot level.
             Limit-cycle detector: if |Δ| oscillates with relative std > _CYCLE_THRESHOLD,
             α is reduced by _CYCLE_DAMP_FAC and the Anderson history is reset to escape the cycle.

        Converged when max(|ΔM|,|ΔQ|,|ΔΔ_s|,|ΔΔ_d|)<tol and |n−(1−δ)|<10·tol.
        Near onset (0.8<λ_max<1.8): tol relaxed ×5.

        Post-convergence: 3×3 Hessian, linearised λ_max/gap symmetry, χ₀(q_AFM),
        χ_τ/λ_JT, Λ_inst — all exported in result dict.
        """
        converged = False
        # K_eff is tracked as a LOCAL variable throughout the SCF loop.
        _K_eff_scf: float = self._K_bare
        if abs(target_doping) < 0.01:
            mu = 0.0
        elif target_doping > 0:
            mu = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        else:
            mu = 2.0 * self.p.t0 * np.tanh(abs(target_doping) / 0.1)
        mu += 0.5 * self.p.Delta_CF

        g_t, g_J, g_Delta_s, g_Delta_d = self.p.get_gutzwiller_factors(target_doping) 
        kick = self._scf_jacobi_kick(target_doping, initial_M, initial_Q, initial_Delta, g_t, g_J, max(g_Delta_s, g_Delta_d), mu)

        M = kick['M_kick']
        Q = kick['Q_kick']
        Delta_s = kick['Delta_s_kick']   # on-site B₁g singlet
        Delta_d = kick['Delta_d_kick']   # inter-site d-wave B₁g
        _alpha = kick['mixing_kick']
        _Lambda_inst = kick['lambda_plus']
        _j_eff_kick  = kick['J_eff_scalar']
        _chi0m_kick  = kick['chi0_moriya']

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [],
            'F_bdg': [], 'F_cluster': [], 'g_t': [],
            'g_J': [], 'mu': [], 'mixing': [],       # adaptive mixing rate per iteration
        }

        if verbose:
            _Q_rep      = 0.05 * self.p.lambda_hop
            _tx_r       = self.p.t0 * np.exp(+_Q_rep / max(self.p.lambda_hop, 1e-9))
            _ty_r       = self.p.t0 * np.exp(-_Q_rep / max(self.p.lambda_hop, 1e-9))
            scale_A1g   = float(np.cosh(2.0 * _Q_rep / max(self.p.lambda_hop, 1e-9)))
            h_afm_M0    = self.p.Z * g_J * (1.0 - target_doping) * self.p.J_CT * scale_A1g * initial_M
            t_eff_aniso = g_t * np.sqrt(0.5 * (_tx_r**2 + _ty_r**2))
            
            _scf_log("SCF-INIT", f"δ={target_doping:.4f}  M₀={initial_M:.3f}  M_kick={M:.3f}  M_phys={kick['M_phys']:.3f}  g_t={g_t:.4f}  g_J={g_J:.4f}  g_Delta_s={g_Delta_s:.4f}  g_Delta_d={g_Delta_d:.4f}")
            _scf_log("SCF-INIT", f"h_afm(M₀)={h_afm_M0:.4f} eV  t_eff={t_eff_aniso:.4f} eV  {'✓ metallic AFM' if h_afm_M0 < 2.0 * t_eff_aniso else '⚠ marginal/insulating'}")
            _scf_log("SCF-INIT", f"λ₊={kick['lambda_plus']:.3f}  λ_lin_max={kick['lambda_lin_max']:.3f}[{kick['regime']}]"
                     f"  Λ₀={_Lambda_inst:.3f}  M₀={M:.4f}  Q₀={Q:.5f}  |Δ|₀={abs(Delta_s)+abs(Delta_d):.5f}  α={_alpha:.4f}")

        scf_x_hist: list = []
        scf_f_hist: list = []

        _vertex_cache: Optional[dict] = None
        _K_eff_last_M    = initial_M + 999.0
        _K_eff_last_iter = -5
        _K_eff_last_Q    = initial_Q + 999.0
        _z_qp_scf        = 1.0   # quasiparticle weight; updated after first cluster-ED call
        _K_eff_last_j_renorm = 1.0

        _scf_t0 = _time.time()
        _max_diff_prev = float('inf')    # previous iteration's max_diff
        _stagnation_count = 0            # consecutive near-stagnation iterations
        _pairing_strength_proxy = 0.0    # must be initialised before loop; only updated when _vertex_cache is not None
        _k2x2_cache: Optional[dict] = None   # cache for _compute_2x2_pairing_kernel (invalidated on M/Q/j_renorm change)
        _k2x2_cache_key: tuple = ()

        # Initialise Λ_inst from kick proxies; take the most pessimistic channel.
        _lambda_JT_proxy = (self.p.g_JT**2 / max(self._K_bare, 1e-9)) * kick.get('chi_tau_val', 0.0)
        _jchi_proxy      = float(np.clip(_j_eff_kick * _chi0m_kick, 0.0, _JCHI_HARD_REJECT))
        _Lambda_inst     = float(np.clip(max(_Lambda_inst, kick['lambda_lin_max'], _lambda_JT_proxy, _jchi_proxy), 0.0, 10.0))

        tx_mixed, ty_mixed = g_t * self.p.t0, g_t * self.p.t0
        mu_new        = mu
        n_kspace_new  = 1.0 - target_doping
        max_diff      = float('inf')
        F_bdg         = 0.0
        F_cluster     = {'F_per_site': 0.0, 'J_eff': kick['J_eff_scalar'], 'j_renorm': 1.0}
        _j_renorm_cur = 1.0   # cluster vertex correction for current iteration; starts bare
        _det_q0             = 1.0
        _det_afm            = 1.0
        _det_q0_last_valid  = 1.0   # last finite det value; survives cache invalidation
        _det_afm_last_valid = 1.0
        _alpha_freeze_count = 0
        _near_fm_qcp  = False
        _near_afm_qcp = False
        _tol_use      = self.p.tol
        _Q_iter_counter  = 0      # counts iterations since last Q update
        _Q_frozen        = False  # True during inner sub-iterations (Q held fixed)
        selection_ratio = 0.0
        _delta_run_hist: list = []

        for iteration in range(_MAX_ITER):
            _iter_t0 = _time.time()

            tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)
            tx, ty = g_t * tx_bare, g_t * ty_bare

            F_cluster = self.compute_cluster_free_energy(M, Q, mu, g_J, tx_bare, ty_bare, target_doping)
            _j_renorm_cur = _ALPHA_RENORM * F_cluster['j_renorm'] + (1.0 - _ALPHA_RENORM) * _j_renorm_cur

            _z_qp_scf = float(np.clip(1.0 / max(_j_renorm_cur, 0.3), 0.3, 1.0))

            _vbdg_scf = self._get_vbdg()
            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(_vbdg_scf._build_H_stack(_vbdg_scf._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=_vbdg_scf._H_stack))
            self._scf_bdg_cache = (_bdg_ev_sc, _bdg_ec_sc)

            obs = self._get_vbdg().compute_observables_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                _bdg_cache=(_bdg_ev_sc, _bdg_ec_sc))
            tau_x   = obs['tau_x']    # off-diagonal Γ₆↔Γ₇ mixing
            B1g_exp = obs['B1g_exp']  # Hellmann-Feynman force ⟨B̂₁g⟩
            M_bdg   = obs['M_stag']   # BdG staggered magnetisation

            # SC+JT active: ⟨τ_x⟩_anom from anomalous amplitudes (u·v), Gutzwiller-renormalised, fed back into J_B1g off-diagonal Weiss field. Zero by symmetry when Δ=0 or Q=0.
            Delta_eff_now = abs(Delta_s) + abs(Delta_d)
            if Delta_eff_now > 1e-5 and abs(Q) > 1e-8:
                tau_x_anom = self._compute_orbital_coherence_from_pairs(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
                tau_x_renorm = g_Delta_s * tau_x_anom
                tau_x_anom_for_sel = tau_x_anom
            else:
                tau_x_renorm = 0.0
                tau_x_anom_for_sel = 0.0

            # Rebuild SC-state BdG cache with the off-diagonal anomalous Weiss field.
            if abs(tau_x_renorm) > 1e-12:
                _vbdg_scf2 = self._get_vbdg()
                _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                    _vbdg_scf2._build_H_stack(_vbdg_scf2._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, tau_x_anom_mf=tau_x_renorm, out=_vbdg_scf2._H_stack)
                )
                self._scf_bdg_cache = (_bdg_ev_sc, _bdg_ec_sc)

            _K_eff_update_needed = (
                iteration == 0
                or (iteration - _K_eff_last_iter >= 5 and abs(M - _K_eff_last_M) > 0.02)
                or abs(F_cluster['j_renorm'] - _K_eff_last_j_renorm) > 0.05
                or abs(Q - _K_eff_last_Q) > _Q_THR_REL * self.p.lambda_hop
            )
            if _K_eff_update_needed:
                _rigidity = self.compute_JT_rigidity_from_exchange(M, Q, mu, g_J, target_doping, g_t)
                _K_eff_scf          = max(_rigidity['K_eff'], 1e-9)
                _K_eff_last_M       = M
                _K_eff_last_Q       = Q
                _K_eff_last_iter    = iteration
                _K_eff_last_j_renorm = _j_renorm_cur

            # Gap equation: V(q) always from Δ=0 χ₀; BdG amplitudes (u,v) from SC state.
            Delta_s_out, Delta_d_out, _vertex_cache = self._get_vbdg().compute_gap_eq_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, _K_eff_scf, g_Delta_s, g_Delta_d, _j_renorm_cur,
                _bdg_cache=(_bdg_ev_sc, _bdg_ec_sc), _vertex_cache=_vertex_cache, z_qp=_z_qp_scf)

            # Blend fixed-point gap with 2×2 pairing kernel eigenvector for optimal s/d mixing.
            # Near QCP (|det|<0.05): direction-only update (10% weight), fixed amplitude.
            # Far from QCP: full blend (weight _ALPHA_MIX_2X2), with BCS seed when Δ≈0.
            # Guard: |Δ_new| ≤ max(5×|Δ_current|, 0.02 eV)
            _det_afm = _vertex_cache.get('det_afm', 1.0) if _vertex_cache else 1.0
            _det_q0  = _vertex_cache.get('det_q0',  1.0) if _vertex_cache else 1.0
            _near_det_zero = abs(_det_afm) < 0.05 or abs(_det_q0) < 0.05
            _D_curr = abs(Delta_s) + abs(Delta_d)

            try:
                # --- Cache 2×2 kernel ---
                _key = (
                    round(M / 0.05) * 0.05,
                    round(Q / max(0.04 * self.p.lambda_hop, 1e-6)) * 0.04 * self.p.lambda_hop,
                    round(_j_renorm_cur / 0.05) * 0.05,
                )
                if _k2x2_cache is None or _key != _k2x2_cache_key:
                    _k2x2_cache = self._compute_2x2_pairing_kernel(
                        M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                        _K_eff_scf, g_Delta_s, g_Delta_d, _j_renorm_cur, z_qp=_z_qp_scf)
                    _k2x2_cache_key = _key

                # --- Eigenvector direction ---
                _v_s = abs(_k2x2_cache['vec_s'])
                _v_d = abs(_k2x2_cache['vec_d'])
                _v_norm = max(_v_s + _v_d, 1e-10)
                _lam_2x2 = _k2x2_cache['lambda_2x2']

                _Ds_fp = abs(Delta_s_out)
                _Dd_fp = abs(Delta_d_out)
                _D_fp = _Ds_fp + _Dd_fp

                if _near_det_zero:
                    if _D_fp > 1e-10:
                        α = _ALPHA_MIX_2X2 * 0.10   # strongly damped, direction only
                        _Ds_dir = _D_fp * _v_s / _v_norm
                        _Dd_dir = _D_fp * _v_d / _v_norm
                        _Ds = (1-α)*_Ds_fp + α*_Ds_dir
                        _Dd = (1-α)*_Dd_fp + α*_Dd_dir
                    else:
                        _Ds, _Dd = _Ds_fp, _Dd_fp
                else:
                    if _D_fp < 1e-10:
                        # Δ=0: BCS seed from 2×2 eigenvalue
                        t_eff = g_t * self.p.t0
                        _D_lin = float(np.clip(
                            t_eff * np.exp(-1.0 / max(_lam_2x2, 0.05)) * 0.05,
                            1e-7, 0.05 * t_eff))
                        α = _ALPHA_MIX_2X2 * 0.5   # gentle injection from zero
                    else:
                        _D_lin = _D_fp
                        α = _ALPHA_MIX_2X2

                    _Ds_2x2 = _D_lin * _v_s / _v_norm
                    _Dd_2x2 = _D_lin * _v_d / _v_norm

                    # Fallback for dead channels
                    _Ds_base = _Ds_fp if _Ds_fp > 0 else 0.5 * _Ds_2x2
                    _Dd_base = _Dd_fp if _Dd_fp > 0 else 0.5 * _Dd_2x2

                    _Ds = (1-α)*_Ds_base + α*_Ds_2x2
                    _Dd = (1-α)*_Dd_base + α*_Dd_2x2

                # --- Jump limiter ---
                _D_new = _Ds + _Dd
                _D_max = max(5.0 * max(_D_curr, 1e-6), 0.02)
                if _D_new > _D_max:
                    scale = _D_max / _D_new
                    _Ds *= scale
                    _Dd *= scale
                Delta_s_out = complex(_Ds)
                Delta_d_out = complex(_Dd)
            except Exception:
                pass  # fall back to plain fixed-point gap update

            # hierarchical Q update: Q is the adiabatic (slow) variable; electrons converge much faster.
            # Q updates only every _Q_UPDATE_PERIOD inner iterations.
            # This prevents over-correlation between Q and Δ and avoids the numerical oscillation that arises from updating both simultaneously at each step.
            _Q_iter_counter += 1
            _q_update_now = (_Q_iter_counter >= _Q_UPDATE_PERIOD) or (iteration == 0)
            if _q_update_now:
                Q_out = -(self.p.g_JT / max(_K_eff_scf, 1e-9)) * B1g_exp
                Q_out = float(np.clip(Q_out, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))
                _Q_iter_counter = 0
            else:
                # Q frozen: keep current Q as output so Anderson does not try to change it.
                Q_out = Q

            # Update Λ_inst: Rayleigh λ_pair, λ_JT=(g²/K)·χ_QQ, J·χ_SS; EMA(0.3 new, 0.7 old).
            if _vertex_cache is not None:
                _t_eff_now = g_t * self.p.t0
                _N0_now    = 1.0 / (np.pi * max(_t_eff_now, 1e-6))
                _pairing_strength_proxy = float(np.clip(max(
                    g_Delta_s * max(_vertex_cache['V_s_scalar'], 0.0) * _N0_now,
                    g_Delta_d * max(_vertex_cache['V_d_scalar'], 0.0) * _N0_now
                ), 0.0, 10.0))
                _chi_QQ_vc  = float(_vertex_cache.get('chi_QQ', 0.0))
                _lam_JT_vc  = float(np.clip(self.p.g_JT**2 / max(_K_eff_scf, 1e-9) * abs(_chi_QQ_vc), 0.0, 10.0))
                _chi_DD_vc  = float(_vertex_cache.get('chi_DD_s_moriya_full',
                                     _vertex_cache.get('chi_DD_s_full', 0.0)))
                _jchi_vc    = float(np.clip(F_cluster['J_eff'] * max(_chi_DD_vc, 0.0), 0.0, _JCHI_HARD_REJECT))
                # Unified instability measure: smooth exponential moving average to avoid single-iteration spikes (weight 0.3 new, 0.7 old).
                _Lambda_raw = float(np.clip(max(_pairing_strength_proxy, _lam_JT_vc, _jchi_vc), 0.0, 10.0))
                _Lambda_inst = 0.7 * _Lambda_inst + 0.3 * _Lambda_raw

            # Δ update before Anderson mix so M update sees current SC state.
            Delta_s_mixed = self._mix(Delta_s, Delta_s_out, alpha=_alpha)
            Delta_d_mixed = self._mix(Delta_d, Delta_d_out, alpha=_alpha)

            # 4D Anderson vector (M, Q/λ_hop, |Δ_s|·t₀, |Δ_d|·t₀): scaled for balanced Jacobian solve.
            _t0_sc = max(self.p.t0, 1e-6)
            _lhop  = max(self.p.lambda_hop, _KT_FLOOR)
            x_in_4d  = np.array([M,     Q / _lhop,     abs(Delta_s) * _t0_sc,       abs(Delta_d) * _t0_sc])
            x_out_4d = np.array([M_bdg, Q_out / _lhop, abs(Delta_s_mixed) * _t0_sc, abs(Delta_d_mixed) * _t0_sc])
            scf_x_hist.append(x_in_4d)
            scf_f_hist.append(x_out_4d)

            x_new_4d = self._anderson_mix(scf_x_hist, scf_f_hist, m=5, alpha=_alpha)

            dF_dM_0, d2F_dM2 = self.compute_dF_dM_and_d2F(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, tau_x_anom_mf=tau_x_renorm)
            self._scf_bdg_cache = None   # cache consumed; clear to prevent stale reuse

            # Adaptive LM floor: reduce overdamping when Δ grows (SC–AFM coupling unfreezes).
            _mu_LM_eff = _MU_LM / (1.0 + 10.0 * (abs(Delta_s_out) + abs(Delta_d_out)) / max(self.p.t0, 1e-9))

            # LM Newton for M: positive Hessian → standard LM; saddle (d²F<0) → abs floor preserves gradient direction.
            if d2F_dM2 > 0:
                _lm_denom = d2F_dM2 + _mu_LM_eff
            else:
                _lm_denom = max(abs(d2F_dM2) * 0.1, _mu_LM_eff)
            M_newton = M - dF_dM_0 / _lm_denom
            M_newton = float(np.clip(M_newton, 0.0, 1.0))

            # Trust-region: |ΔM| ≤ clip(max(kT, 0.1·t_eff)/J_eff, 0.02, 0.30).
            _t_eff_now  = g_t * self.p.t0
            _J_eff_now  = float(F_cluster['J_eff'])
            _step_limit = float(np.clip(max(self.p.kT, 0.1 * _t_eff_now) / max(abs(_J_eff_now), 1e-4), 0.02, 0.30))
            M_newton    = float(np.clip(M_newton, M - _step_limit, M + _step_limit))
            M_newton    = float(np.clip(M_newton, 0.0, 1.0))
            # Blend: Anderson fixpoint (BdG ⟨S_z⟩) with Newton correction.
            M_mixed = float(np.clip(
                (1.0 - _ALPHA_HF) * x_new_4d[0] + _ALPHA_HF * M_newton,
                0.0, 1.0
            ))
            if _J_eff_now * _vertex_cache.get('chi_DD_s_moriya_full', float('nan')) > 1.0:
                # A rendszer instabil a QCP alatt, húzzuk az M-et az analitikus becslés felé
                M_mixed = 0.8 * M_mixed + 0.2 * kick['M_phys']

            Q_mixed    = float(np.clip(x_new_4d[1] * _lhop, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # Anderson updates |Δ| magnitude; phase kept from linear mix (BdG phase convention).
            _Ds_abs_new = float(np.clip(x_new_4d[2] / _t0_sc, 0.0, 1.0))
            _Dd_abs_new = float(np.clip(x_new_4d[3] / _t0_sc, 0.0, 1.0))
            
            _phase_s = Delta_s_mixed / (abs(Delta_s_mixed) + 1e-30)
            _phase_d = Delta_d_mixed / (abs(Delta_d_mixed) + 1e-30)
            Delta_s_mixed = complex(_Ds_abs_new) * _phase_s
            Delta_d_mixed = complex(_Dd_abs_new) * _phase_d

            # _vertex_cache may be None after a Q sign-flip reset.
            _V_s_now = _vertex_cache['V_s_scalar'] if _vertex_cache is not None else 0.0
            _V_d_now = _vertex_cache['V_d_scalar'] if _vertex_cache is not None else 0.0

            if len(scf_x_hist) > 1 and (Q * Q_mixed < 0):
                scf_x_hist.clear()
                scf_f_hist.clear()
                _vertex_cache = None         # Q sign flip → FS topology may change
                self._scf_bdg_cache = None   # topology change → stale SC cache unsafe
                self._chi0_norm_cache = None # χ₀ eigenvectors keyed on Q → must rebuild
                _k2x2_cache = None           # 2×2 kernel FS geometry also keyed on Q
                _k2x2_cache_key = ()
                _Lambda_inst = max(_Lambda_inst, 2.0)

            tx_mixed_bare, ty_mixed_bare = self.p.effective_hopping_anisotropic(Q_mixed)
            tx_mixed, ty_mixed = g_t * tx_mixed_bare, g_t * ty_mixed_bare
            
            mu_new, _mu_bdg_cache, n_kspace_new = self._find_mu_for_density(M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, tx_mixed, ty_mixed, mu, g_J)

            # _mu_bdg_cache provides (ev, ec) for free-energy reuse, pass V_s, V_d from vertex cache
            F_bdg = self.compute_bdg_free_energy(
                M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, mu_new,
                tx_mixed, ty_mixed, g_J, _V_s_now, _V_d_now, _K_eff_scf,
                _ev_cache=_mu_bdg_cache[0] if _mu_bdg_cache is not None else None)

            Delta_s_abs = abs(Delta_s_mixed)
            Delta_d_abs = abs(Delta_d_mixed)
            # max_diff tracks order-parameter convergence (M, Q, Δ_s, Δ_d).
            max_diff = max(
                abs(M_mixed - M),
                abs(Q_mixed - Q),
                abs(Delta_s_abs - abs(Delta_s)),
                abs(Delta_d_abs - abs(Delta_d)),
            )

            # track |Δ| run history for limit-cycle detection.
            _delta_run_hist.append(Delta_s_abs + Delta_d_abs)
            
            # SC-triggered JT selection rule proxy:
            #   |Δ|/Δ_CF         — condensate mixing of Γ₆↔Γ₇ (0=normal state, 1=full mixing)
            #   |τ_x_anom|       — ANOMALOUS SC-induced orbital coherence (not normal-state quad!)
            # this is exactly zero at Δ=0 (D₄h selection rule) and nonzero only when the condensate generates anomalous u·v off-diagonal coherence.
            selection_ratio = float(np.clip(
                (Delta_s_abs + Delta_d_abs) / max(self.p.Delta_CF, 1e-9), 0.0, 1.0
            )) * abs(tau_x_anom_for_sel)

            if iteration >= 5 and iteration % 5 == 0:
                _diverging  = max_diff > _max_diff_prev * 1.05
                _stagnating = max_diff > _max_diff_prev * 0.95 and not _diverging

                # Unified α_eff = α₀ / (1 + 0.5·Λ): Λ_inst measures pairing instability, the 0.5 prefactor suppresses overdamping
                _alpha_base = float(np.clip(
                    _MIXING / (1.0 + 0.5 * _Lambda_inst),
                    _MIXING / 6.0,
                    _MIXING,
                ))

                # --- Limit cycle detection ---
                _in_cycle = self._detect_limit_cycle(_delta_run_hist)
                if _in_cycle:
                    _alpha = max(_alpha_base * _CYCLE_DAMP_FAC, _MIXING / 16.0)
                    if len(scf_x_hist) > 4:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                    _stagnation_count   = 0
                    _alpha_freeze_count = 0
                    if verbose:
                        _scf_log("SCF", f"⚠ LIMIT CYCLE detected iter={iteration} → α damped to {_alpha:.5f}")

                # --- Diverging ---
                elif _diverging:
                    _alpha = max(_alpha_base * 0.5, _MIXING / 8.0)
                    scf_x_hist.clear()
                    scf_f_hist.clear()
                    _stagnation_count   = 0
                    _alpha_freeze_count = 0

                # --- Stagnating (not diverging) ---
                elif _stagnating:
                    _stagnation_count   += 1
                    _alpha_freeze_count += 1
                    if _stagnation_count >= 2:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _stagnation_count = 0
                    # Subcritical plateau: boost α slightly if pairing is weak.
                    if _pairing_strength_proxy < 0.3:
                        _alpha = min(_alpha_base * 1.2, _MIXING)
                    else:
                        _alpha = _alpha_base

                # --- Converging ---
                else:
                    _stagnation_count   = 0
                    _alpha_freeze_count = max(_alpha_freeze_count - 1, 0)
                    if selection_ratio > 0.05 and abs(Q) > 1e-4:
                        # SC+JT active: mild boost, Λ-capped
                        _alpha = min(_alpha_base * 1.15, _MIXING * 0.75)
                    elif selection_ratio > 0.05:
                        _alpha = _alpha_base * 0.95
                    else:
                        _alpha = _alpha_base
                    # Q update this iteration → be more conservative
                    if _q_update_now:
                        _alpha = min(_alpha, _MIXING * 0.3)

                # --- Alpha freeze recovery ---
                if _alpha_freeze_count >= 10 and _alpha < _MIXING * 0.15:
                    _alpha = min(_alpha * 1.6, _MIXING * 0.6)
                    _alpha_freeze_count = 0
                    if verbose:
                        _scf_log("SCF", f"↺ α-recovery after freeze → {_alpha:.5f}")
            _max_diff_prev = max_diff

            # QCP detection (None after Q sign-flip → treat as non-critical).
            if _vertex_cache is not None:
                _det_q0_raw  = _vertex_cache['det_q0']
                _det_afm_raw = _vertex_cache['det_afm']
                # Only update last_valid if finite — prevents nan propagation
                if math.isfinite(_det_q0_raw):
                    _det_q0            = _det_q0_raw
                    _det_q0_last_valid = _det_q0_raw
                else:
                    _det_q0 = _det_q0_last_valid   # hold last known good value
                if math.isfinite(_det_afm_raw):
                    _det_afm            = _det_afm_raw
                    _det_afm_last_valid = _det_afm_raw
                else:
                    _det_afm = _det_afm_last_valid
                _near_fm_qcp  = (_det_q0  < _RPA_DET_WARN) or (abs(_vertex_cache['V_s_scalar']) > _V_CUT)
                _near_afm_qcp = (_det_afm < _RPA_DET_WARN) or (abs(_vertex_cache['V_d_scalar']) > _V_CUT)
            else:
                # Cache invalidated (Q sign-flip etc.): hold last valid — not a physical event
                _det_q0       = _det_q0_last_valid
                _det_afm      = _det_afm_last_valid
                _near_fm_qcp  = False
                _near_afm_qcp = False
            _tol_use = self.p.tol * (5.0 if _near_afm_qcp else 1.0)

            # Near QCP: belt-and-suspenders α cap; past QCP (det<0): handled above by exponential penalty.
            if _near_afm_qcp:
                _alpha = min(_alpha, _MIXING / (1.0 + _Lambda_inst))
            if _det_afm < 0.0:
                # Past QCP: exponential α penalty ∝ |det_afm|/det_warn; Λ_inst boosted to keep Anderson conservative.
                _det_penalty = float(np.clip(abs(_det_afm) / max(_RPA_DET_WARN, 1e-6), 0.0, 5.0))
                _alpha = float(np.clip(_alpha * math.exp(-_RPA_QCP_PENALTY * _det_penalty),
                                       _MIXING / 16.0, _alpha))
                _Lambda_inst = float(np.clip(_Lambda_inst + 1.5 * _det_penalty, 0.0, 10.0))
                # Count iterations where det-penalty pins alpha at floor → enables recovery
                if _alpha <= _MIXING / 16.0 * 1.1:
                    _alpha_freeze_count += 1

            _iter_s = (_time.time() - _iter_t0)

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
                _qcp_tag = (" AFM!" if _near_afm_qcp and _det_afm >= 0 else
                            (" PAST!" if _det_afm < 0.0 else
                            (" FM!  " if _near_fm_qcp else "      ")))

                _scf_log("SCF", f"δ={target_doping:.4f} {iteration+1:3d}/{_MAX_ITER}"
                        f"  conv={max_diff:.1e}  M={M:.3f}  Q={Q:+.4f}"
                        f"  |Δ|={(abs(Delta_s)+abs(Delta_d)):.4f}"
                        f"  J_eff={_J_eff_now:.4f} eV  j_renorm={F_cluster['j_renorm']:.4f}  mu={mu_new:.3f} "
                        f"  dFM={_det_q0:.3f}  dAFM={_det_afm:.3f}"
                        f"  _V_s_now={_V_s_now:.3f} _V_d_now={_V_d_now:.3f}"
                        f"  {_iter_s:3.0f}s/it{_qcp_tag}"
                        f"  Λ_inst={_Lambda_inst:.3f}  max_diff={max_diff:.1e}  α→{_alpha:.4f}", verbose=verbose)

            # Save the Anderson-mixed values for post-kick convergence check.
            _M_pre_kick      = M_mixed
            _Q_pre_kick      = Q_mixed
            _Ds_pre_kick     = Delta_s_mixed
            _Dd_pre_kick     = Delta_d_mixed

            M, Q, Delta_s, Delta_d, mu = M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, mu_new

            # Saddle-escape: |Δ|≈0 + negative Hessian eigenvalue → kick along λ_min eigenvector.
            # Modes from component fractions (fM,fQ,fΔ): pure-SC(fΔ>0.6), pure-JT(fQ>0.6),
            # SC-triggered-JT(fΔ>0.3 AND fQ>0.3, target), AFM-fluctuation(fM>0.6).
            # Kick magnitude Λ-damped: 1/(1+Λ); applied only when fΔ>0.25.
            if abs(Delta_s) + abs(Delta_d) < 5.0 * self.p.tol and iteration > 3 and iteration % 8 == 0:
                try:
                    if abs(Delta_s) + abs(Delta_d) < 1e-10:
                        _D_base = 1e-6
                        Delta_s_frac = 0.5
                    else:
                        _D_base = abs(Delta_s) + abs(Delta_d)
                        Delta_s_frac = float(np.clip(abs(Delta_s) / _D_base, 0.0, 1.0))

                    if _vertex_cache is not None:
                        V_s = _vertex_cache['V_s_scalar']
                        V_d = _vertex_cache['V_d_scalar']
                    else:
                        V_s = self.p.g_JT**2 / max(self._K_bare, 1e-9)
                        V_d = V_s

                    _hk = self.compute_hessian(M, Q, _D_base, target_doping, mu, g_t, g_J, Delta_s_frac, V_s, V_d, max(self._K_bare, 1e-9))
                    
                    _lmin_k = _hk['min_curvature']
                    if np.isfinite(_lmin_k) and _lmin_k < -self.p.kT:
                        _evals_k, _evecs_k = np.linalg.eigh(_hk['H'])
                        _edir = _evecs_k[:, 0]   # eigenvector of λ_min in Hessian units (M, Q, Δ)

                        # Scale to physical units (Q in Å), normalise to unit vector.
                        _lhop = max(self.p.lambda_hop, 1e-4)
                        _scale = np.array([1.0, _lhop, 1.0])
                        _edir_raw  = _edir * _scale
                        _edir_raw /= max(np.linalg.norm(_edir_raw), 1e-12)

                        _wM, _wQ, _wD = abs(_edir_raw[0]), abs(_edir_raw[1]), abs(_edir_raw[2])
                        _wsum = max(_wM + _wQ + _wD, 1e-12)
                        _fM, _fQ, _fD = _wM / _wsum, _wQ / _wsum, _wD / _wsum

                        if _fD > 0.6:     _mode = 'pure-SC'
                        elif _fQ > 0.6:   _mode = 'pure-JT'
                        elif _fD > 0.3 and _fQ > 0.3: _mode = 'SC-triggered-JT'
                        elif _fM > 0.6:   _mode = 'AFM-fluctuation'
                        else:             _mode = 'mixed'

                        _kick_damp = 1.0 / (1.0 + _Lambda_inst)
                        _curvature = min(abs(_lmin_k), 1.0)
                        _kick_mag  = min(2.0 * self.p.kT, 0.1 * _D_base) * _kick_damp * _curvature

                        # M: gentle pull toward Stoner estimate when SC mode and M overshoots.
                        _M_phys_est = self.p.estimate_M0(target_doping, kick['lambda_lin_max'])
                        _m_was_pulled = False
                        if _mode in ('pure-SC', 'SC-triggered-JT') and M > 3.0 * _M_phys_est:
                            _pull_frac = 0.30 * _kick_damp
                            _M_kick_component = float(np.clip(M - _pull_frac * (M - _M_phys_est), 0.02, M))
                            _m_was_pulled = True
                        else:
                            _M_kick_component = float(np.clip(M + _kick_mag * _edir_raw[0], 0.0, 1.0))
                        Q_kick = float(np.clip(
                            Q + _kick_mag * _edir_raw[1],
                            -0.5 * _lhop, 0.5 * _lhop))

                        # Δ: signed kick along eigenvector component, preserving s/d ratio.
                        _delta_sign = np.sign(_edir_raw[2]) if abs(_edir_raw[2]) > 1e-6 else 1.0
                        _D_kick_signed = max(0.0, _D_base + _kick_mag * _delta_sign * abs(_edir_raw[2]))

                        Delta_s_kick = complex(np.clip(_D_kick_signed * Delta_s_frac, 0.0, 0.3))
                        Delta_d_kick = complex(np.clip(_D_kick_signed * (1.0 - Delta_s_frac), 0.0, 0.3))
                        
                        # Apply kick only if Δ component is significant (>25%)
                        if _fD > 0.25:
                            M = _M_kick_component
                            Q = Q_kick
                            Delta_s = Delta_s_kick
                            Delta_d = Delta_d_kick
                            scf_x_hist.clear()
                            scf_f_hist.clear()
                            _vertex_cache = None
                            self._scf_bdg_cache = None
                            _scf_log("SCF", f"δ={target_doping:.3f} ⚡kick iter={iteration}  mode={_mode}"
                                 f"  λ_min={_lmin_k:+.4f}  Λ={_Lambda_inst:.3f}  damp={_kick_damp:.3f}"
                                 f"  fM={_fM:.2f} fQ={_fQ:.2f} fΔ={_fD:.2f}"
                                 f"  → M={M:.3f} Q={Q:+.4f} |Δ|={_D_kick_signed:.4f}"
                                 f"{' [M-pulled]' if _m_was_pulled else ''}", verbose=verbose)
                except Exception as e:
                    pass

            # Re-evaluate convergence against the current (possibly kicked) state.
            _M_post   = abs(M)          - abs(_M_pre_kick)
            _Q_post   = abs(Q)          - abs(_Q_pre_kick)
            _Ds_post  = abs(Delta_s)    - abs(_Ds_pre_kick)
            _Dd_post  = abs(Delta_d)    - abs(_Dd_pre_kick)
            _kick_fired = (abs(_M_post) + abs(_Q_post) + abs(_Ds_post) + abs(_Dd_post)) > 1e-10

            _past_qcp = (_det_afm < 0.0)
            _too_early = (iteration < _MIN_ITER)
            if (not _kick_fired
                    and not _too_early
                    and not _past_qcp
                    and max_diff < _tol_use
                    and abs(n_kspace_new - (1 - target_doping)) < _tol_use * 10):
                converged = True
                break

        if not converged and verbose:
            _scf_log("SCF-RES", f"δ={target_doping:.4f} ⚠ no conv after {_MAX_ITER} iters  max_diff={max_diff:.2e}"
                     f"  dens_err={abs(n_kspace_new-(1-target_doping)):.2e}")

        # Post-loop diagnostic: λ_max and Rayleigh JT projection.
        _tx_mixed_bare, _ty_mixed_bare = self.p.effective_hopping_anisotropic(Q)
        _J_eff_lin = self.p.Z * self.p.effective_superexchange(g_J, _tx_mixed_bare, _ty_mixed_bare, target_doping) * F_cluster['j_renorm']
        _lin: Dict = self.solve_linearized_gap_equation(M, Q, Delta_s, Delta_d, target_doping, mu, tx_mixed, ty_mixed, g_J, _K_eff_scf, _J_eff_lin, actual_doping=float(1.0 - n_kspace_new), z_qp=_z_qp_scf)

        if converged:
            _Delta_total = abs(Delta_s) + abs(Delta_d)
            _Delta_s_frac = (abs(Delta_s) / _Delta_total) if _Delta_total > 1e-10 else 0.5
            # Single eigh at the converged point — reused for F0 in compute_hessian, skipping 1 of its 13 diagonalisations.
            _vbdg_conv = self._get_vbdg()
            _ev_conv_cache, _ = np.linalg.eigh(
                _vbdg_conv._build_H_stack(
                    _vbdg_conv._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx_mixed, ty_mixed, g_J,
                    out=_vbdg_conv._H_stack))
            hessian_result = self.compute_hessian(
                M, Q, _Delta_total, target_doping, mu, g_t, g_J, _Delta_s_frac,
                _vertex_cache['V_s_scalar'] if _vertex_cache is not None else 0.0,
                _vertex_cache['V_d_scalar'] if _vertex_cache is not None else 0.0,
                _K_eff_scf, _ev_conv_cache)

        else:
            hessian_result = {'H': None, 'eigenvalues': None, 'is_minimum': None, 'min_curvature': None}

        _chi_tau_result  = self._compute_chi_tau(M, Q, Delta_s, Delta_d, target_doping, mu, tx_mixed, ty_mixed, g_t, g_J)

        # ── FS-resolved ∂λ_pair/∂Q ────────────────────
        # Compute the finite-difference sensitivity of the dominant pairing eigenvalue to the JT distortion Q using the full BdG+RPA linearised gap equation at Q±dQ.
        # A positive value confirms the SC-triggered JT mechanism: the condensate (Δ≠0) genuinely increases the pairing strength when Q deforms the lattice.
        _dlam_dQ_fs = 0.0
        if converged and (abs(Delta_s) + abs(Delta_d)) > 1e-5:
            try:
                _tx_b_dlam, _ty_b_dlam = self.p.effective_hopping_anisotropic(Q)
                _J_eff_dlam = self.p.Z * self.p.effective_superexchange(
                    g_J, _tx_b_dlam, _ty_b_dlam, target_doping) * _j_renorm_cur
                _dlam_dQ_fs = self._dlambda_dQ_fs(
                    M, Q, Delta_s, Delta_d, target_doping, mu, tx_mixed, ty_mixed,
                    g_J, _K_eff_scf, _j_renorm_cur, z_qp=_z_qp_scf)
            except Exception:
                _dlam_dQ_fs = float('nan')

        if converged and verbose:
            _tag_chi = f"SCF-RES δ={target_doping:.4f}"
            # Lambda_s / Lambda_d channel decomposition: if lambda_d > lambda_s but Delta_s != 0, the SCF has mixed channels beyond what the linear kernel predicts.
            try:
                _gap_vec = _lin['gap_vector']
                _fs_pts  = _lin['fs_pts']
                _phi_s   = np.ones(len(_fs_pts))
                _phi_s /= np.linalg.norm(_phi_s)
                _phi_d   = np.cos(_fs_pts[:,0]) - np.cos(_fs_pts[:,1])
                _nd      = np.linalg.norm(_phi_d)
                if _nd > 1e-10:
                    _phi_d /= _nd
                _w_s_post   = float(abs(_gap_vec @ _phi_s))
                _w_d_post   = float(abs(_gap_vec @ _phi_d))
                _lam_max_s  = _lin['lambda_lin_max'] * _w_s_post / max(_w_s_post + _w_d_post, 1e-10)
                _lam_max_d  = _lin['lambda_lin_max'] * _w_d_post / max(_w_s_post + _w_d_post, 1e-10)
                _Delta_s_mag = abs(Delta_s)
                _Delta_d_mag = abs(Delta_d)
                _scf_d_frac = _Delta_d_mag / max(_Delta_s_mag + _Delta_d_mag, 1e-12)
                _nl_mixing = (_lam_max_d > _lam_max_s) and (_Delta_s_mag > 1e-5) and (_Delta_d_mag > 1e-5)
                # Eigenvector vs SCF amplitude mismatch: linearised gap eq says d-wave dominates but SCF converged to s-wave (Δ_s >> Δ_d). This indicates that the non-linear
                # fixpoint (large Δ) has quenched the d-wave channel via spectrum depletion.
                _sym_scf = 'd' if _scf_d_frac > 0.5 else 's'
                _sym_lin = 'd' if _lam_max_d > _lam_max_s else 's'
                _sym_mismatch = _sym_scf != _sym_lin
                _scf_log(_tag_chi,
                         f"Gap channels (post-SCF): lambda_s={_lam_max_s:.4f}  lambda_d={_lam_max_d:.4f}"
                         f"  |Delta_s|={_Delta_s_mag*1000:.3f}meV  |Delta_d|={_Delta_d_mag*1000:.3f}meV"
                         f"  SCF-sym={_sym_scf}  lin-sym={_sym_lin}"
                         f"  {'[⚠ SYMMETRY MISMATCH: lin-gap predicts ' + _sym_lin + ' but SCF converged to ' + _sym_scf + ']' if _sym_mismatch else '[consistent]'}"
                         f"  {'[NONLINEAR s-d MIXING: lambda_d>lambda_s but Delta_s!=0]' if _nl_mixing else ''}")
            except Exception as _lam_ch_err:
                _scf_log(_tag_chi, f"lambda_s/d channel decomposition failed: {_lam_ch_err}")

            # Incommensurate nesting scan: q* = (π, π−δq) that maximizes χ_SS. The BdG Hamiltonian is fixed to a commensurate q_AFM = (π, π),
            # so any incommensurate nesting is only detected here. The auto-retry (softened M) still uses the same BdG and can help only near the threshold (δq ≳ 0.05π);
            # for δq > 0.10π it is purely diagnostic. (_ic_dq_max/_ic_chi_max/_ic_chi_0 = 0.0 above)
            try:
                _dq_scan = np.linspace(0.0, 0.15 * np.pi, 7)
                _chi_SS_scan = []
                _vbdg_ic = self._get_vbdg()
                _Ek_ic, _Vk_ic = np.linalg.eigh(
                    _vbdg_ic._build_H_stack(_vbdg_ic._kpts_ev, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx_mixed, ty_mixed, g_J,
                                            out=_vbdg_ic._H_stack_ev))
                _Sz_diag_ic = self.sz_bdg16
                for _dq_v in _dq_scan:
                    _q_ic   = np.array([np.pi, np.pi - _dq_v])
                    _kpts_q = (self.k_points_even + _q_ic[None,:] + np.pi) % (2*np.pi) - np.pi
                    _Ekq_ic, _Vkq_ic = np.linalg.eigh(
                        _vbdg_ic._build_H_stack(_kpts_q, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx_mixed, ty_mixed, g_J))
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
                _idx_max    = int(np.argmax(_chi_SS_scan))
                _ic_dq_max  = float(_dq_scan[_idx_max])
                _ic_chi_max = _chi_SS_scan[_idx_max]
                _ic_chi_0   = _chi_SS_scan[0]
                # Retry only for borderline incommensurate (0.05π < δq < 0.10π).
                _ic_flag = 0.05 * np.pi < _ic_dq_max < 0.10 * np.pi
                _scf_log(_tag_chi,
                         f"Incommensurate scan: χ_SS max at dq={_ic_dq_max/np.pi:.3f}π"
                         f"  χ(0)={_ic_chi_0:.4f}  χ(max)={_ic_chi_max:.4f}"
                         f"  {'⚠ incommensurate — auto-retry below' if _ic_flag else '✓ commensurate'}")
                # Auto-retry with softened AFM seed (single recursion, _ic_retry flag prevents further recursion).
                if _ic_flag and not _ic_retry:
                    try:
                        _M_ic = float(np.clip(M * 0.85, 0.02, 0.45))   # soften AFM seed
                        _ic_result = self.solve_self_consistent(
                            target_doping,
                            initial_M=_M_ic, initial_Q=Q,
                            initial_Delta=float(abs(Delta_s) + abs(Delta_d)),
                            verbose=verbose, _ic_retry=True)
                        if _ic_result.get('converged', False):
                            _scf_log(_tag_chi, f"IC retry converged: M={_ic_result['M']:.4f}  Q={_ic_result['Q']:+.4f}  |Δ|={_ic_result['Delta_s']+_ic_result['Delta_d']:.4f}")
                            _ic_result['incommensurate_dq'] = _ic_dq_max
                            _ic_result['incommensurate_chi_ratio'] = _ic_chi_max / max(_ic_chi_0, 1e-12)
                            return _ic_result
                        else:
                            _scf_log(_tag_chi, "IC retry did not converge — keeping commensurate result")
                    except Exception as _ic_retry_err:
                        _scf_log(_tag_chi, f"IC retry failed: {_ic_retry_err}")
            except Exception as _ic_err:
                _scf_log(_tag_chi, f"Incommensurate scan failed: {_ic_err}")

        # ── Post-SCF validations ──────────────────────────────────────────────
        # λ_min(Hessian_SC): the authoritative SC-triggered JT criterion.
        # λ_min < 0  →  SC has genuinely triggered JT (condensate softened the mode).
        # λ_min > 0  →  the converged SC+AFM state is a true local minimum without orbital distortion — JT was NOT triggered.
        #    This is categorically different from the pre-SCF G3[2,2] > 0 check, because here M, Q, Δ are all fully self-consistent fixpoint.
        _ic_dq_max  = 0.0
        _ic_chi_max = 0.0
        _ic_chi_0   = 0.0

        # χ_AFM: use the Δ=0 values from the vertex cache (computed by get_susceptibilities_normal at q=(π,π), Δ=0, SECTOR_PAIRS-masked — the correct Stoner input).
        _vc_ok        = isinstance(_vertex_cache, dict) 
        _chi_DD_s     = _vertex_cache.get('chi_DD_s_full',        float('nan')) if _vc_ok else float('nan')
        _chi_DD_s_mor = _vertex_cache.get('chi_DD_s_moriya_full', float('nan')) if _vc_ok else float('nan')
        _stoner_denom  = (1.0 - _J_eff_lin * _chi_DD_s_mor) if np.isfinite(_chi_DD_s_mor) else 1.0
        _rpa_factor    = (1.0 / max(_stoner_denom, _RPA_DET_REG) if _stoner_denom > 0.0 else 1.0)
        _afm_unstable  = _stoner_denom <= 0.0

        if verbose:
            _hstr = "H=n/a"
            if hessian_result['eigenvalues'] is not None:
                _eigs  = hessian_result['eigenvalues']
                _hstr  = f"H=[{_eigs[0]:.3f},{_eigs[1]:.3f},{_eigs[2]:.3f}]{'✓MIN' if hessian_result['is_minimum'] else '⚠SADDLE'}"
            _det_afm_log = _vertex_cache.get('det_afm', float('nan')) if _vc_ok else float('nan')
            with _log_lock:
                sys.stdout.write("\n"); sys.stdout.flush()
            _scf_log("SCF-RES", f"δ={target_doping:.4f} ✓ conv {iteration+1} iters"
                     f"  M={M:.4f}  Q={Q:+.4f}  |Δs|={abs(Delta_s):.4f}  |Δd|={abs(Delta_d):.4f}"
                     f"  n={n_kspace_new:.4f}  μ={mu:.4f}  F={F_bdg:.5f}"
                     f"  j_renorm={_j_renorm_cur:.3f}  det_AFM={_det_afm_log:.4f}"
                     f"  JT={'✓' if selection_ratio > 0.05 else '✗'}  {_hstr}")
            if (abs(Delta_s) + abs(Delta_d)) > 1e-5:
                _scf_log("SCF-RES", f"δ={target_doping:.4f}  ∂λ/∂Q(FS)={_dlam_dQ_fs:+.4f} eV⁻¹"
                         f"  {'✓ SC-triggered JT consistent: ∂λ/∂Q > 0' if _dlam_dQ_fs > 0 else '⚠ ∂λ/∂Q ≤ 0: JT does not enhance pairing on FS'}")

        # ── Post-SCF Mott filter ──────────────────────────────────────────────
        # (a) g_t < 0.10 (δ < 0.053): primary Mott guard prevents incoherent ZRS band
        #     The Gutzwiller factor encodes the full doping-dependent Mott suppression;
        #     no SC gap can be physical without coherent hopping.
        # (b) ξ/a < 1.0: BEC/artefact limit — Cooper pairs not coherent across a lattice site;
        #     Δ is suppressed post-hoc (BdG mean-field breaks down in this regime).
        _mott_suspect   = (g_t < _G_T_COHERENCE_MIN) or (_lin['xi_over_a'] < 1.0)
        if _mott_suspect:
            Delta_s   = 0.0 + 0.0j
            Delta_d   = 0.0 + 0.0j
            converged = False
            if verbose:
                _reason = 'g_t<min (Mott)' if g_t < _G_T_COHERENCE_MIN else f'ξ/a={_lin['xi_over_a']:.2f}<1 (BEC)'
                _scf_log("SCF-RES", f"δ={target_doping:.4f} ⚠ MOTT-SUSPECT [{_reason}]  g_t={g_t:.3f}  ξ/a={_lin['xi_over_a']:.2f}  — gap suppressed")

        result = dict(_lin)
        result.update({
            'M': M,
            'Q': Q,
            'Delta_s': abs(Delta_s),
            'Delta_d': abs(Delta_d),
            'chi_tau_sc': _chi_tau_result['chi_tau_sc'],
            'chi_tau_n': _chi_tau_result['chi_tau_n'],
            'richardson_ok': _chi_tau_result['richardson_ok'],
            'Ut_ratio': _chi_tau_result['Ut_ratio'],
            'density': n_kspace_new,
            'mu': mu,
            'g_t': g_t,
            'g_J': g_J,
            'F_bdg': F_bdg,
            'F_per_site': F_cluster['F_per_site'],
            'tx': tx_mixed,
            'ty': ty_mixed,
            'J_eff': _J_eff_lin,
            'j_renorm': F_cluster['j_renorm'],
            'z_qp_scf': _z_qp_scf,
            'target_doping': target_doping,
            'chi_DD_s': _chi_DD_s,
            'chi_DD_s_moriya': _chi_DD_s_mor,
            'rpa_factor': _rpa_factor,
            'afm_unstable': _afm_unstable,
            'selection_ratio': selection_ratio,
            'history': history,
            'hessian_result': hessian_result,
            'lambda_plus': kick['lambda_plus'],
            'regime': kick['regime'],
            'K_eff_scf': _K_eff_scf,
            'converged': converged,
            'mott_suspect': _mott_suspect,
            'dlam_dQ_fs': _dlam_dQ_fs,
            'incommensurate_dq': _ic_dq_max,
            'incommensurate_chi_ratio': _ic_chi_max / max(_ic_chi_0, 1e-12) if _ic_chi_0 else float('nan'),
        })
        return result

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
        alpha = _MIXING if alpha is None else alpha
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
        w = float(np.clip(alpha / max(_MIXING, _KT_FLOOR), _ANDERSON_W_LO, _ANDERSON_W_HI))
        x_new = w * x_opt + (1.0 - w) * x_simple

        if not np.all(np.isfinite(x_new)):
            return x_simple
        return x_new

    def compute_dF_dM_and_d2F(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, tau_x_anom_mf: float = 0.0) -> Tuple[float, float]:
        abs_d = max(abs(target_doping), 1e-6)
        f_d   = 1.0 - abs_d              # RMFT: (1-δ) spin-site fraction
        
        # ∂H/∂M per orbital in the 16-component Nambu basis.
        # Only J_A1g (diagonal) contributes: J_B1g (off-diagonal, τ_x_mat) has zero
        # diagonal elements so (J_B1g_scalar * tau_x_mat) @ O_exp_unit is off-diagonal
        # and does not enter the diagonal |ec|²-weighted gradient / Kubo sum here.
        # However, the eigenvectors ec must come from the FULL Hamiltonian including
        # the J_B1g off-diagonal Weiss field (tau_x_anom_mf), so that the Kubo-formula
        # term2 correctly captures inter-band matrix elements through the modified bands.
        O_exp_unit = g_J * f_d * self.sz_op / 2.0
        J_A1g_diag, _ = self._exchange_channels(Q)
        h_J_unit = J_A1g_diag * O_exp_unit   # element-wise: diagonal ∂H/∂M
        
        dH_diag = np.array([
            -h_J_unit[0], -h_J_unit[1], -h_J_unit[2], -h_J_unit[3],   # particle A  (sign_M=+1)
            +h_J_unit[0], +h_J_unit[1], +h_J_unit[2], +h_J_unit[3],   # particle B  (sign_M=−1)
            +h_J_unit[0], +h_J_unit[1], +h_J_unit[2], +h_J_unit[3],   # hole A      (PH of A)
            -h_J_unit[0], -h_J_unit[1], -h_J_unit[2], -h_J_unit[3],   # hole B      (PH of B)
        ])

        vbdg = self._get_vbdg()
        if hasattr(self, '_scf_bdg_cache') and self._scf_bdg_cache is not None:
            ev, ec = self._scf_bdg_cache
        else:
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(
                vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, tau_x_anom_mf=tau_x_anom_mf, out=vbdg._H_stack))
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

    def compute_hessian(self, M: float, Q: float, Delta: float, target_doping: float, mu: float, g_t: float, g_J: float, Delta_s_frac: float, V_s: float, V_d: float, K_eff_for_free_energy: float, _ev0_cache: np.ndarray = None) -> Dict:
        """ 3×3 finite-difference Hessian of F(M,Q,Δ).

        V_s / V_d: full RPA pairing vertex for the condensation correction in compute_bdg_free_energy.
        Must match the vertex used in the SCF gap equation so that ∂F/∂Δ = 0 at the converged point.
        """
        Delta_s_frac = float(np.clip(Delta_s_frac, 0.0, 1.0))
        Delta_d_frac = 1.0 - Delta_s_frac

        eps_M = max(1e-4, abs(M) * 1e-3)
        eps_Q = max(5e-3 * self.p.lambda_hop, abs(Q) * 1e-3 * self.p.lambda_hop)  # floor ~6.4e-3 Å prevents noise-dominated H[1,1] at Q≈0
        # eps_D must satisfy Delta - eps_D >= 0; negative amplitude is unphysical and
        # causes abs(Delta_s/d) asymmetry in the BdG, biasing the Hessian off-diagonals.
        # Clip to at most Delta/2 so both Delta±eps_D stay non-negative.
        eps_D = min(max(1e-5, abs(Delta) * 1e-3), max(abs(Delta) / 2.0, 1e-10))

        def F(m, q, d, _ev_c=None):
            tb_x, tb_y = self.p.effective_hopping_anisotropic(q)
            # abs() ensures the amplitude is non-negative even at d = Delta - eps_D ≈ 0;
            # the free energy is even in Delta so F(|d|) = F(-d) is the correct branch.
            ds = complex(abs(d) * Delta_s_frac)
            dd = complex(abs(d) * Delta_d_frac)
            return self.compute_bdg_free_energy(
                m, q, ds, dd, target_doping, mu, g_t * tb_x, g_t * tb_y, g_J, V_s, V_d, K_eff_for_free_energy,
                _ev_cache=_ev_c)

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
        alpha = _MIXING if alpha is None else alpha
        return (1 - alpha) * old + alpha * new

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
                    initial_M     = float(np.clip(sc_result['M'], 0.02, 0.45)) if sc_result['converged'] else self.p.estimate_M0(doping, sc_result['lambda_lin_max']),
                    initial_Q     = 1e-6,
                    initial_Delta = 1e-8,   # normal-state seed (below nucleation floor)
                    verbose       = False,
                )
                Ds = res['Delta_s']
                Dd = res['Delta_d']
                D  = (Ds**2 + Dd**2) ** 0.5

                if use_free_energy and D > Delta_tol:
                    s_n = copy.copy(s)
                    s_n.p = copy.copy(s.p)
                    s_n._K_bare = s._K_bare
                    s_n._reset_transient_state()
                    res_normal = s_n.solve_self_consistent(
                        target_doping = doping,
                        initial_M     = float(np.clip(sc_result['M'], 0.02, 0.45)) if sc_result['converged'] else self.p.estimate_M0(doping, sc_result['lambda_lin_max']),
                        initial_Q     = 0.0,
                        initial_Delta = 0.0,
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

    def compute_Tc_thermodynamic(self, doping: float, sc_result: dict, T_min: float = 1e-4, T_max: float = 0.20, n_scan: int = 15, n_bisect: int = 10, Delta_tol: float = 1e-4) -> dict:
        """
        Thermodynamic Tc via warm-start upward temperature scan.

        compute_Tc_by_gap_suppression starts every temperature from a near-zero
        Δ seed ("cooling from normal state").  This finds only the spinodal
        (second-order instability boundary) — i.e. the temperature where the
        *local* SC minimum first appears.  For the SC-triggered JT mechanism the
        effective Landau potential is

            F_eff(Δ) = a(T)Δ² + [b − γ²/(2K_eff)]Δ⁴ + …

        If ∂λ/∂Q > 0 is large enough the quartic coefficient becomes negative and
        the transition is first-order: the system jumps to finite (Δ, Q) at a Tc*
        where a(T*)>0 (the normal state is still locally stable).  Cooling from
        Δ≈0 misses this entirely and returns Tc ≪ Tc*.

        Algorithm
        ---------
        1. "Warm-start heating": start from the T≈0 SC+JT basin and track the
        state upward in T, using the previous step's (M, Q, Δ) as seed.
        2. At each T: run both the warm-started SC+JT solve and a cold normal-
        state solve; compare free energies F_BdG.
        3. Tc is where the SC+JT basin either collapses (Δ→0) OR is no longer
        thermodynamically preferred (F_SC > F_NM) — whichever comes first.
        The first condition catches spinodal collapse; the second catches the
        first-order crossing.  The thermodynamic Tc is max of the two.
        4. Bisection between the last SC-wins temperature and the first NM-wins
        temperature.

        Returns
        -------
        dict with keys:
        'Tc'              : float  — thermodynamic critical temperature (eV)
        'Tc_spinodal'     : float  — spinodal (second-order boundary), 0 if pure 1st-order
        'transition_order': str    — 'first-order', 'second-order', or 'unknown'
        'Delta_at_Tc'     : float  — gap just below Tc
        'Q_at_Tc'         : float  — JT distortion just below Tc
        'ratio_2D'        : float  — 2Δ₀/kTc ratio
        'Delta_jump'      : float  — |Δ| discontinuity at Tc (0 for 2nd-order)
        'hysteresis'      : float  — T_spinodal_cool − T_spinodal (>0 = hysteresis)
        'history'         : list   — scan history tuples
        """

        Delta_s0 = sc_result['Delta_s']
        Delta_d0 = sc_result['Delta_d']
        Delta0   = float(np.sqrt(Delta_s0**2 + Delta_d0**2))
        Q0       = float(sc_result['Q'])
        M0       = float(sc_result['M'])

        if (not sc_result.get('converged', False)
            or Delta0 < Delta_tol):
            
            return {
                'Tc': 0.0, 'Tc_spinodal': 0.0, 'T_cross': 0.0, 'T_spinodal_cool': float('nan'),
                'transition_order': 'unknown', 'Delta_at_Tc': 0.0, 'Q_at_Tc': 0.0,
                'ratio_2D': 0.0, 'Delta_jump': 0.0, 'hysteresis': 0.0, 'history': []
            }

        def _make_solver_at_T(T: float) -> 'RMFT_Solver':
            s = copy.copy(self)
            s.p = copy.copy(self.p)
            s.p.kT = T
            s._K_bare = self._K_bare
            s._reset_transient_state()
            return s

        def _eval_sc_basin(solver: 'RMFT_Solver', seed_M: float, seed_Q: float, seed_D: float) -> tuple:
            """
            Evaluate SC+JT basin at given solver.
            Returns (Δ_eff, Q_eff, M_eff, F_sc, converged, collapsed).
            collapsed = True if SC minimum vanished (Δ < Delta_tol).
            """
            try:
                res = solver.solve_self_consistent(
                    target_doping=doping,
                    initial_M=seed_M,
                    initial_Q=seed_Q,
                    initial_Delta=seed_D,
                    verbose=False,
                )
                D_eff = float(np.sqrt(res['Delta_s']**2 + res['Delta_d']**2))
                Q_eff = float(res['Q'])
                M_eff = float(res['M'])
                F_sc = float(res['F_bdg']) + float(res['F_per_site'])
                converged = res['converged']
                collapsed = D_eff < Delta_tol
                return D_eff, Q_eff, M_eff, F_sc, converged, collapsed
            except Exception:
                return 0.0, 0.0, seed_M, 1e30, False, True

        def _eval_normal_basin(solver: 'RMFT_Solver') -> tuple:
            """
            Evaluate normal-state basin at given solver.
            Returns (F_nm, converged).
            """
            try:
                res = solver.solve_self_consistent(
                    target_doping=doping,
                    initial_M=solver.p.estimate_M0(doping, 0.0),
                    initial_Q=1e-6,
                    initial_Delta=0.0,
                    verbose=False,
                )
                F_nm = float(res['F_bdg']) + float(res['F_per_site'])
                return F_nm, res['converged']
            except Exception:
                return 1e30, False

        # ── Helper: find crossing temperature (F_SC = F_NM) ──────────────────────
        def _find_crossing(T_vals: list, seed: dict) -> tuple:
            """Returns (T_cross, last_D, last_Q, history) or (0.0, 0.0, 0.0, []) if not found."""
            history = []
            _seed = seed.copy()
            T_cross = 0.0
            last_D = 0.0
            last_Q = 0.0

            for i, T in enumerate(T_vals):
                s = _make_solver_at_T(T)
                D_eff, Q_eff, M_eff, F_sc, _, collapsed = _eval_sc_basin(s, _seed['M'], _seed['Q'], _seed['Delta'])
                F_nm, _ = _eval_normal_basin(s)

                sc_wins = (not collapsed) and (F_sc < F_nm)

                history.append((T, D_eff, Q_eff, F_sc, F_nm, sc_wins))

                if sc_wins and not collapsed:
                    _seed['M'] = M_eff
                    _seed['Q'] = Q_eff
                    _seed['Delta'] = max(D_eff, Delta_tol * 10)
                    last_D = D_eff
                    last_Q = Q_eff
                elif not sc_wins and not collapsed and i > 0:
                    # Linear interpolation of the crossing: F_SC(T_{i-1}) < F_NM, F_SC(T_i) >= F_NM
                    prev = history[-2]
                    dF_prev = prev[3] - prev[4]   # F_sc - F_nm at T_{i-1}
                    dF_curr = F_sc - F_nm           # F_sc - F_nm at T_i
                    denom_interp = dF_curr - dF_prev
                    if abs(denom_interp) > 1e-30:
                        T_cross = T_vals[i-1] - dF_prev * (T - T_vals[i-1]) / denom_interp
                    else:
                        T_cross = T_vals[i-1]
                    last_D = prev[1]
                    last_Q = prev[2]
                    break
            return T_cross, last_D, last_Q, history

        # ── Helper: find spinodal temperature (SC minimum vanishes) ─────────────
        def _find_spinodal_heating(T_vals: list, seed: dict) -> tuple:
            """Returns (T_spinodal, last_D, last_Q, history)."""
            history = []
            _seed = seed.copy()
            T_spinodal = 0.0
            last_D = 0.0
            last_Q = 0.0

            for i, T in enumerate(T_vals):
                s = _make_solver_at_T(T)
                D_eff, Q_eff, M_eff, F_sc, conv_sc, collapsed = _eval_sc_basin(s, _seed['M'], _seed['Q'], _seed['Delta'])

                history.append((T, D_eff, Q_eff, F_sc, float('nan'), not collapsed))

                if collapsed:
                    T_spinodal = T
                    if i > 0:
                        last_D = history[-2][1]
                        last_Q = history[-2][2]
                    break

                _seed['M'] = M_eff
                _seed['Q'] = Q_eff
                _seed['Delta'] = max(D_eff, Delta_tol * 10)
                last_D = D_eff
                last_Q = Q_eff
            return T_spinodal, last_D, last_Q, history

        # ── Helper: find spinodal from cooling (normal → SC) ─────────────────────
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
                s = _make_solver_at_T(T)
                # Cold start from normal state (Δ=0)
                try:
                    res = s.solve_self_consistent(
                        target_doping=doping,
                        initial_M=s.p.estimate_M0(doping, 0.0),
                        initial_Q=1e-6,
                        initial_Delta=0.0,
                        verbose=False,
                    )
                    D_eff = float(np.sqrt(res['Delta_s']**2 + res['Delta_d']**2))
                    Q_eff = float(res['Q'])
                    converged = res['converged']
                except Exception:
                    D_eff = 0.0
                    Q_eff = 0.0
                    converged = False

                history.append((T, D_eff, Q_eff, converged))

                if D_eff > Delta_tol and prev_D < Delta_tol and i > 0:
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
        seed_init = {'M': M0, 'Q': Q0, 'Delta': max(Delta0, 1e-3)}

        # ── 2. Find crossing temperature ─────────────────────────────────────────
        T_cross, D_cross, Q_cross, cross_history = _find_crossing(T_vals, seed_init)

        # ── 3. Find spinodal from heating ────────────────────────────────────────
        T_spinodal, D_spinodal, Q_spinodal, spinod_history = _find_spinodal_heating(T_vals, seed_init)

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
        THRESHOLD_RATIO = 0.02  # 2% difference threshold for first-order classification
        if T_cross > 0.0 and T_spinodal > 0.0:
            if T_cross > T_spinodal * (1.0 + THRESHOLD_RATIO):
                transition_order = 'first-order'
                Tc = T_cross
                Delta_at_Tc = D_cross
                Q_at_Tc = Q_cross
                Delta_jump = D_cross
            elif T_spinodal > T_cross * (1.0 + THRESHOLD_RATIO):
                # This should not happen physically, but handle it
                transition_order = 'second-order'
                Tc = T_spinodal
                Delta_at_Tc = D_spinodal
                Q_at_Tc = Q_spinodal
                Delta_jump = 0.0
            else:
                # T_cross and T_spinodal are close → weakly first-order or second-order
                # Use the higher temperature as Tc (thermodynamic stability)
                if T_cross > T_spinodal:
                    Tc = T_cross
                    Delta_at_Tc = D_cross
                    Q_at_Tc = Q_cross
                    Delta_jump = D_cross
                    transition_order = 'weakly-first-order' if D_cross > 0.1 * Delta0 else 'second-order'
                else:
                    Tc = T_spinodal
                    Delta_at_Tc = D_spinodal
                    Q_at_Tc = Q_spinodal
                    Delta_jump = 0.0
                    transition_order = 'second-order'
        elif T_cross > 0.0:
            # Only crossing found, spinodal not reached within scan range
            Tc = T_cross
            Delta_at_Tc = D_cross
            Q_at_Tc = Q_cross
            Delta_jump = D_cross
            transition_order = 'first-order' if D_cross > 0.1 * Delta0 else 'unknown'
        elif T_spinodal > 0.0:
            # Only spinodal found
            Tc = T_spinodal
            Delta_at_Tc = D_spinodal
            Q_at_Tc = Q_spinodal
            Delta_jump = 0.0
            transition_order = 'second-order'
        else:
            # No transition found in scan range
            Tc = T_max
            Delta_at_Tc = 0.0
            Q_at_Tc = 0.0
            Delta_jump = 0.0
            transition_order = 'unknown'

        # ── 6. Compute ratio and combine history ─────────────────────────────────
        ratio = (2.0 * Delta0 / Tc) if Tc > 1e-8 else 0.0

        # Combine histories for return
        combined_history = []
        for h in cross_history:
            combined_history.append(('cross',) + h)
        for h in spinod_history:
            combined_history.append(('spinod',) + h)

        return {
            'Tc':               Tc,
            'T_cross':          T_cross,
            'Tc_spinodal':      T_spinodal,
            'T_spinodal_cool':  T_spinodal_cool,
            'transition_order': transition_order,
            'Delta_at_Tc':      Delta_at_Tc,
            'Q_at_Tc':          Q_at_Tc,
            'ratio_2D':         ratio,
            'Delta_jump':       Delta_jump,
            'hysteresis':       hysteresis,
            'history':          combined_history,
        }
    
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
                    initial_M      = float(np.clip(sc_result['M'], 0.02, 0.45)) if sc_result['converged'] else self.p.estimate_M0(doping, sc_result['lambda_lin_max']),
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
                actual_doping = float(1.0 - res['density'])

                _tx_b_lT, _ty_b_lT = s_T.p.effective_hopping_anisotropic(Q)
                _J_eff_lT = self.p.Z * self.p.effective_superexchange(g_J, _tx_b_lT, _ty_b_lT, target_doping) * res['j_renorm']
                _z_qp_lT  = float(res.get('z_qp_scf', 1.0))
                lin = s_T.solve_linearized_gap_equation(M, Q, complex(res['Delta_s']), complex(res['Delta_d']), doping, mu, tx, ty, g_J, res['K_eff_scf'], _J_eff_lT, actual_doping=actual_doping, z_qp=_z_qp_lT)
                lam_arr[i] = float(lin['lambda_lin_max'])
                sym_list.append(lin['gap_symmetry'])
            except Exception:
                lam_arr[i] = 0.0
                sym_list.append('error')

        Tc_lambda   = 0.0
        slope_at_Tc = 0.0
        _crossings  = []
        for i in range(len(T_points) - 1):
            l0, l1 = lam_arr[i], lam_arr[i + 1]
            t0, t1 = T_points[i], T_points[i + 1]
            if l0 >= 1.0 >= l1 and abs(l1 - l0) > 1e-10:
                frac = (1.0 - l0) / (l1 - l0)
                _crossings.append((float(t0 + frac * (t1 - t0)),
                                   float((l1 - l0) / (t1 - t0))))

        if _crossings:
            Tc_lambda, slope_at_Tc = _crossings[0]   # lowest-T (physical) crossing
            if len(_crossings) > 1:
                # Non-monotone λ(T): can occur near first-order SC-JT regime.
                # Report but do not abort — caller can inspect the full T/λ arrays.
                _scf_log("TC-LAMBDA", f"⚠ non-monotone λ(T): {len(_crossings)} crossings detected "
                         f" (T={[f'{c[0]*1000:.1f}meV' for c in _crossings]}); using lowest-T crossing as Tc.")

        return {
            'T':              T_points,
            'lambda_lin_max': lam_arr,
            'gap_symmetry':   sym_list,
            'Tc_lambda':      Tc_lambda,
            'slope_at_Tc':    slope_at_Tc,
            'n_crossings':    len(_crossings),
        }

    def compute_G_instability(self, target_doping: float, M: float, compute_dlambda: bool) -> dict:
        """
        Compute normal-state (Δ=0) collective instability matrix and diagnostics.

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

        Notes:
        - Only the orbital-mixing vertex τ_x (Γ6↔Γ7) contributes to χ_DQ.
        - The reduced 2-band AFM-folded model does not capture the full χ_DQ, but preserves
        the selection rule Δ=0 → χ_DQ=0, because ξ_avg = -μ ⇒ proj(k)·Δ_k·φ_c(k) is odd under k→k+Q,
        so the BZ integral vanishes, enforcing χ_DQ = 0 in the normal state
        - For Δ≠0, χ_DQ becomes finite via the same orbital-mixing kernel,
        describing the SC response to JT distortion Q.

        - χ_DD_s, χ_DD_d, χ_DD_sd computed from weighted k-grid sums with
        tanh(E/2kT) kernels.
        - χ_QQ = −∂²Ω/∂Q² evaluated at given Q, Δ_s, Δ_d, M, μ.
        The negative sign ensures χ_QQ > 0 for a stable metal.
        The g_JT factor is already included in the Hamiltonian.

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
        g_t, g_J, g_Delta_s, g_Delta_d = self.p.get_gutzwiller_factors(target_doping)

        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q=0.0)
        tx_eff = g_t * tx_bare
        ty_eff = g_t * ty_bare
        t_eff  = np.sqrt(0.5 * (tx_eff**2 + ty_eff**2))   # kept for BCS Tc estimate only

        abs_d = max(abs(target_doping), 1e-6)
        f_d   = 1.0 - abs_d   # RMFT: (1-δ) spin-site fraction
        h_afm = g_J * f_d * self.p.Z * self.p.J_CT * M / 2.0
        mu_n  = -2.0 * t_eff * (1.0 - 2.0 * abs_d)

        kx = self.k_points[:, 0]
        ky = self.k_points[:, 1]
        eps_k  = -2.0 * (tx_eff * np.cos(kx) + ty_eff * np.cos(ky)) - mu_n
        eps_kQ = -eps_k - 2.0 * mu_n
        xi_avg  = 0.5 * (eps_k + eps_kQ)
        xi_diff = 0.5 * (eps_k - eps_kQ)
        sq      = np.sqrt(xi_diff**2 + h_afm**2 + 1e-20)
        E_plus = xi_avg + sq
        E_minus = xi_avg - sq

        kT = max(self.p.kT, _KT_FLOOR)
        def _th2E(E):
            a  = np.clip(E / (2.0 * kT), -100, 100)
            se = np.where(np.abs(E) > _KT_FLOOR, E, _KT_FLOOR)
            return np.tanh(a) / (2.0 * se)

        def _mdf(E):
            f_E = 1.0 / (1.0 + np.exp(np.clip(E / kT, -100, 100)))
            return f_E * (1.0 - f_E) / kT

        w_k  = self.k_weights
        pk   = _th2E(E_plus) + _th2E(E_minus)
        phi_s = np.ones_like(kx)
        phi_d = np.cos(kx) - np.cos(ky)

        chi_DD_s  = float(np.dot(w_k, pk * phi_s**2))
        chi_DD_d  = float(np.dot(w_k, pk * phi_d**2))
        chi_DD_sd  = float(np.dot(w_k, pk * phi_s * phi_d))
        # Normal state: chi_DQ = 0 enforced (selection rule + the analytic 2-band formula vanishes identically — two independent reasons).
        chi_DQ_s = 0.0
        chi_DQ_d = 0.0
        N_eff   = float(np.dot(w_k, _mdf(E_plus) + _mdf(E_minus)))
        
        chi_QQ = self._chi_QQ_matrix_elements(M, 0.0, target_doping, 0.0, 0.0, mu_n)
        rigidity = self.compute_JT_rigidity_from_exchange(M, 0.0, mu_n, g_J, target_doping, g_t)

        K_eff_here  = max(rigidity['K_eff'], 1e-9)
        V_base = self.p.g_JT**2 / K_eff_here
        gVs = g_Delta_s * V_base
        gVd = g_Delta_d * V_base

        G3 = np.zeros((3, 3))
        Kinv = 1.0 / K_eff_here

        G3[0, 0] = 1.0 - gVs * chi_DD_s
        G3[1, 1] = 1.0 - gVd * chi_DD_d
        G3[2, 2] = 1.0 - chi_QQ * Kinv
        G3[0, 1] = G3[1, 0] = -np.sqrt(max(gVs * gVd, 0.0)) * chi_DD_sd

        c_s = np.sqrt(max(gVs * Kinv, 0.0))
        c_d = np.sqrt(max(gVd * Kinv, 0.0))
        G3[0, 2] = G3[2, 0] = -self.p.g_JT * c_s * chi_DQ_s
        G3[1, 2] = G3[2, 1] = -self.p.g_JT * c_d * chi_DQ_d

        eigs3, evecs3 = np.linalg.eigh(G3)
        lam_min = float(eigs3[0])
        evec_min = evecs3[:, 0]

        # Dominant channel from the *eigenvector*, not from diagonal alone.
        weights = np.abs(evec_min)
        ws, wd, wq = weights

        if lam_min < 0.5:
            if lam_min < 0:
                sc_weight = ws + wd
                if wq > 0.6 and sc_weight < 0.3:
                    instab_dir = 'pure JT (spontaneous risk)'
                elif wq > 0.4 and sc_weight > 0.3:
                    instab_dir = 'SC-triggered JT'
                elif ws > 0.6:
                    instab_dir = 's pairing'
                elif wd > 0.6:
                    instab_dir = 'd pairing'
                else:
                    instab_dir = 'mixed SC+JT'
            else:
                mc = int(np.argmax(weights))
                instab_dir = f"near-critical ({'Δ_s' if mc==0 else 'Δ_d' if mc==1 else 'Q'}-dominant)"
        else:
            instab_dir = 'stable'

        if wd > ws:
            dominant   = 'd'
            G11, G12   = G3[1, 1], G3[1, 2]
            chi_DD_dom = chi_DD_d
            chi_DQ_dom = chi_DQ_d
            V_dom      = gVd
        else:
            dominant   = 's'
            G11, G12   = G3[0, 0], G3[0, 2]
            chi_DD_dom = chi_DD_s
            chi_DQ_dom = chi_DQ_s
            V_dom      = gVs
        if wq > ws and wq > wd:
            dominant   = 'JT'
        
        # V_eff Schur complement:
        G22 = G3[2, 2]
        if dominant != 'JT' and G22 > _KT_FLOOR:
            V_eff = V_dom + V_dom * (self.p.g_JT**2 * chi_DQ_dom**2) / (max(chi_DD_dom, 1e-12) * K_eff_here * G22)
        else:
            V_eff = V_dom   # spontaneous-JT regime: no SC-triggered boost
        lambda_eff = N_eff * V_eff        #  λ_eff used as soft-constraint S2 signal, not as a hard SCF gatekeeper.
        Tc_est  = float(1.13 * t_eff * np.exp(-1.0 / lambda_eff)) if lambda_eff > 1e-3 else 0.0


        J_eff           = self.p.Z * self.p.effective_superexchange(g_J, self.p.t0, self.p.t0, target_doping)
        _Gamma_M        = self.p.moriya_gamma(target_doping, g_t * self.p.t0, J_eff)
        _chi_DD_s_pos   = max(chi_DD_s, 0.0)
        chi_DD_s_moriya = _chi_DD_s_pos / max((1.0 + _Gamma_M * _chi_DD_s_pos), 1e-10)  # Moriya-damped chi_DD_s for H2 / jchi_gate

        # ── ∂λ_pair/∂Q diagnostic: 5-point quadratic polynomial fit λ(Q) around Q=0 ──
        # Measures whether JT distortion renormalises the pairing vertex upward.
        # Evaluated at Δ=0 (normal-state linearised gap equation) so it is honest:
        # solve_linearized_gap_equation uses normal-state chi0 internally, but mu_n must be recomputed at each Qv
        dlambda_dQ = float('nan')
        if compute_dlambda:
            try:
                _dQ = max(1e-3, 0.01 * self.p.lambda_hop)

                def _lambda_at_Q(Qv: float) -> float:
                    tx_b, ty_b = self.p.effective_hopping_anisotropic(Qv)
                    tx_eff_v   = g_t * tx_b
                    ty_eff_v   = g_t * ty_b
                    t_eff_v    = float(np.sqrt(0.5 * (tx_eff_v**2 + ty_eff_v**2)))
                    J_eff_v    = self.p.Z * self.p.effective_superexchange(g_J, tx_b, ty_b, target_doping)
                    mu_n_v     = -2.0 * t_eff_v * (1.0 - 2.0 * abs_d)   # charge-neutral μ at Qv
                    lin = self.solve_linearized_gap_equation(M, Qv, 0.0+0j, 0.0+0j, target_doping, mu_n_v, tx_eff_v, ty_eff_v, g_J, K_eff_here, J_eff_v, target_doping)
                    return lin['lambda_lin_max']

                _Q_offsets = np.array([-2*_dQ, -_dQ, 0.0, _dQ, 2*_dQ])
                _lam_vals  = np.array([_lambda_at_Q(q) for q in _Q_offsets])

                # np.polyfit returns [c, b, a]; b = coeffs[1] is ∂λ/∂Q|_{Q=0}
                _coeffs    = np.polyfit(_Q_offsets, _lam_vals, 2)
                dlambda_dQ = float(_coeffs[1])
            except Exception as _dl_err:
                _scf_log("G-INST", f"∂λ_pair/∂Q diagnostic failed: {_dl_err}")

        result = rigidity

        result['chi_DD_s']        = chi_DD_s
        result['chi_DD_d']        = chi_DD_d
        result['chi_DD_sd']       = chi_DD_sd
        result['chi_DQ_s']        = chi_DQ_s
        result['chi_DQ_d']        = chi_DQ_d
        result['N_eff']           = N_eff
        result['h_afm']           = h_afm
        result['mu_n']            = mu_n
        result['chi_QQ']          = chi_QQ
        result['chi_DD_dom']      = chi_DD_dom
        result['chi_DQ_dom']      = chi_DQ_dom
        result['dominant']        = dominant
        result['instab_dir']      = instab_dir
        result['evec_min']        = evec_min
        result['E_plus_mean']     = np.mean(E_plus)
        result['G3']              = G3
        result['eigs3']           = eigs3
        result['lambda_min']      = lam_min
        result['det_G']           = float(np.linalg.det(G3))
        result['V_eff']           = float(V_eff)
        result['lambda_eff']      = float(lambda_eff)
        result['Tc_estimate']     = Tc_est
        result['d2F_Q_normal']    = float(K_eff_here - chi_QQ)   # normal-state Q-curvature: ∂²F/∂Q²|_{Δ=0} — hard exclusion if < 0
        result['sc_triggered_jt'] = False   # set True by SCF when Q≠0 and Δ≠0 converge
        result['dlambda_pair_dQ'] = dlambda_dQ  # ∂λ_pair/∂Q — positive: JT renormalises V_pair upward
        result['G11']             = G11
        result['G22']             = G22
        result['G12']             = G12
        result['K_spont_blocked'] = G22 > 0.0
        result['g_t']             = float(g_t)
        result['g_J']             = float(g_J)
        result['J_eff']           = float(J_eff)
        result['chi_DD_s_moriya'] = float(chi_DD_s_moriya)
        return result
    
    def _get_fs_points(self, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, store_cache=True, compute_vF=False):
        """
        Return Fermi-surface k-points with even angular coverage.

        FS cache: returns the identical (fs_pts, vF) pair. The cache key uses |Δ_s|, |Δ_d|
        (not the complex values) because the BdG spectrum depends only on the gap magnitudes for the purpose of FS geometry

        Parameters
        ----------
        store_cache : write result to self._fs_cache_dict (default True)
        compute_vF  : if True, also compute |∇_k E_min| via central finite
                      differences on the oversampled FS (4 extra eigh calls on
                      3×n_fs points).  If False, vF is returned as an array of
                      ones (uniform weighting, no velocity bias).

        Returns
        -------
        fs_pts : (n_fs, 2) k-points near the Fermi surface
        vF     : (n_fs,)   |v_F| estimates (eV in ka units); ones if compute_vF=False
        fs_idx : (n_fs,)   global integer indices into self.k_points (direct array access, no search)
        """
        n_fs = _FS_N_VERTEX
        _cache_key = (
            float(M), float(Q),
            float(abs(Delta_s)), float(abs(Delta_d)),
            float(target_doping),
            float(mu), float(tx), float(ty), float(g_J),
            int(n_fs), bool(compute_vF),
        )

        if self._fs_cache_dict is not None and _cache_key in self._fs_cache_dict:
            return self._fs_cache_dict[_cache_key]  # (fs_pts, vF, fs_idx)

        vbdg = self._get_vbdg()

        # ── Step 1: oversample FS at 3×n_fs for better vF estimation ──────────────
        # For the initial energy scan we always run on the full k-grid to get a reliable near-FS mask; the result is NOT cached separately.
        ev_all, ec_all = np.linalg.eigh(
            vbdg._build_H_stack(
                vbdg._kpts, M, Q, Delta_s, Delta_d,
                target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))

        ev_pos = np.where(ev_all > 0, ev_all, np.inf)
        Emin   = ev_pos.min(axis=1)

        near_fs    = Emin < (_FS_SAMPLING * self.p.kT)
        fs_idx_all = np.where(near_fs)[0]

        if len(fs_idx_all) == 0:
            fs_idx_all = np.arange(min(3 * n_fs, self.N_k))

        kxy_all   = self.k_points[fs_idx_all]
        Emin_over = Emin[fs_idx_all]

        # ── Step 2: optionally compute |v_F| via ∇_k E_min ───────────────────────
        if compute_vF:
            dk = min(1e-2, max(1e-4, 2.0 * np.pi / _NK / 6.0))
            kx, ky = kxy_all[:, 0], kxy_all[:, 1]

            def _Emin_batch(kpts):
                ev_b, _ = np.linalg.eigh(
                    vbdg._build_H_stack(kpts, M, Q, Delta_s, Delta_d,
                                        target_doping, mu, tx, ty, g_J))
                return np.where(ev_b > 0, ev_b, np.inf).min(axis=1)

            dE_dx = (_Emin_batch(np.c_[kx + dk, ky]) -
                     _Emin_batch(np.c_[kx - dk, ky])) / (2.0 * dk)
            dE_dy = (_Emin_batch(np.c_[kx, ky + dk]) -
                     _Emin_batch(np.c_[kx, ky - dk])) / (2.0 * dk)
            vF_over = np.maximum(np.hypot(dE_dx, dE_dy), _VF_FLOOR)
        else:
            vF_over = np.ones(len(fs_idx_all), dtype=float)

        # ── Step 3: angular-stratified subsampling ───────────────────────────────
        # Primary criterion: FS proximity (small Emin).
        # Secondary: mild vF regularisation (weight 0.1) avoids systematically
        # picking hot-spot nodes where vF→0 and 1/vF→∞ destabilises the kernel.
        angles  = np.arctan2(kxy_all[:, 1], kxy_all[:, 0])
        angles  = (angles + np.pi) % (2.0 * np.pi) - np.pi
        bins    = np.linspace(-np.pi, np.pi, n_fs + 1, endpoint=True)
        bin_ids = np.clip(np.digitize(angles, bins) - 1, 0, n_fs - 1)

        selected = []
        for b in range(n_fs):
            mask = (bin_ids == b)
            if mask.any():
                score = Emin_over[mask] * (1.0 + 0.1 * vF_over[mask])
                selected.append(int(np.flatnonzero(mask)[np.argmin(score)]))

        if len(selected) < n_fs:
            already = set(selected)
            for i in np.argsort(Emin_over):
                if i not in already:
                    selected.append(int(i))
                    if len(selected) >= n_fs:
                        break

        sel    = np.array(selected[:n_fs], dtype=int)
        fs_pts = kxy_all[sel]
        vF     = vF_over[sel]
        fs_idx = fs_idx_all[sel]   # global indices into self.k_points

        if store_cache:
            if self._fs_cache_dict is None:
                self._fs_cache_dict = {}
            # Evict oldest entry if dict grows too large (simple LRU-like bound).
            if len(self._fs_cache_dict) >= 64:
                self._fs_cache_dict.pop(next(iter(self._fs_cache_dict)))
            self._fs_cache_dict[_cache_key] = (fs_pts, vF, fs_idx)
        return fs_pts, vF, fs_idx, ev_all, ec_all

class VectorizedBdG:
    def __init__(self, solver: 'RMFT_Solver'):
        self.solver    = solver
        self._kpts     = solver.k_points        # (N_k, 2)    — SCF / gap grid (endpoint=False)
        self._kpts_ev  = solver.k_points_even   # (N_k_even, 2) — chi0 / commensurate grid
        self._N_k      = solver.N_k
        self._N_k_ev   = solver.N_k_even
        self._H_stack    = np.zeros((self._N_k,    16, 16), dtype=complex)  # SCF grid buffer
        self._H_stack_ev = np.zeros((self._N_k_ev, 16, 16), dtype=complex)  # chi0 grid buffer

    def _build_H_stack(self, kpts: np.ndarray, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, tau_x_anom_mf: float = 0.0, out: Optional[np.ndarray] = None) -> np.ndarray:
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
        kpts          : (N, 2) array of k-points.
        tau_x_anom_mf : Gutzwiller-renormalised anomalous orbital coherence
                        ⟨τ_x⟩_renorm passed from the SCF loop. Default 0.0
                        (normal state, chi0 calls, first iterations).
        out           : optional pre-allocated (N, 16, 16) complex buffer.
        """
        N = len(kpts)
        if out is None:
            H = np.zeros((N, 16, 16), dtype=complex)
        else:
            H = out
            H[:] = 0.0 + 0.0j

        # ---- Local Hamiltonians (A/B sublattice) ----
        H_A = self.solver.build_local_hamiltonian_for_bdg(
            +1.0, M, Q, mu, g_J, target_doping, tx, ty, tau_x_anom_mf=tau_x_anom_mf,
        )
        H_B = self.solver.build_local_hamiltonian_for_bdg(
            -1.0, M, Q, mu, g_J, target_doping, tx, ty, tau_x_anom_mf=tau_x_anom_mf,
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
        """
        Vectorised observables: M, tau_x (off-diagonal orbital mixing), B1g_exp
        (full Hellmann-Feynman force), density, Pair_s, Pair_d.

        tau_x   — off-diagonal Γ₆↔Γ₇ mixing, τ_x = 2 Re(u₀*u₂ + u₁*u₃) per sublattice;
        anomalous orbital coherence used for Weiss-field feedback.

         B1g_exp — full ⟨B̂₁g⟩ = Tr[B1g_16 · ρ̂] per site;
         Hellmann–Feynman force, Q_eq = −(g_JT/K_eff)⟨B̂₁g⟩.
         Uses full B1g_op (all D₄h + D₂h terms), unlike tau_x.
        """
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

        # ── Quadrupole τ_x = 2 Re(u₀*u₂ + u₁*u₃): off-diagonal Γ₆↔Γ₇ mixing (D₄h component of B1g_op) ──────────
        # Normal-state diagnostic; also used as seed for _compute_orbital_coherence_from_pairs
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

        # ── B1g_exp: full ⟨B̂₁g⟩ via B1g_16 — Hellmann-Feynman consistent ──────
        # ⟨B1g⟩_k = Tr[B1g_16 · ρ_k] = Σ_n (ec† B1g_16 ec)_nn · f_n
        # ec layout from np.linalg.eigh: ec[k, orbital, band], i.e. (k, a, n).
        # The hole-block sign (−B1g_op^T) is already encoded in B1g_16, so weighting
        # by f alone (not fbar) gives the full particle + hole contribution correctly.
        Bdiag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), solver.B1g_16, ec).real
        exp_k    = np.einsum('kn,kn->k', Bdiag_qp, f)

        # On-site pairing amplitude (channel s): u_A[6↑]·v_A[7↓]* − u_A[6↓]·v_A[7↑]*
        pair_s_A = uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
        pair_s_B = uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])
        pair_s   = np.sum((pair_s_A + pair_s_B) * f12, axis=1)   # (N_k,)

        # Inter-site pairing amplitude (channel d)
        pair_AB = uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
        pair_BA = uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])
        pair_d  = np.sum(0.5 * (pair_AB + pair_BA) * f12, axis=1)   # (N_k,)

        # k-weighted averages; /4 corrects for 2-sublattice × particle-hole Nambu doubling
        w = self.solver.k_weights   # (N_k,)
        n_avg   = float(np.dot(w, dens_A + dens_B))  / 4.0
        M_stag  = float(np.dot(w, mag_A  - mag_B))   / 4.0
        Q_unif  = float(np.dot(w, quad_A + quad_B))  / 4.0
        B1g_exp = float(np.dot(w, exp_k))            / 4.0
        Pair_s  = complex(np.dot(w, pair_s))         / 4.0
        Pair_d  = complex(np.dot(w, pair_d))         / 4.0
        return {
            'n_avg':   n_avg,
            'M_stag':  M_stag,
            'tau_x':   Q_unif,
            'B1g_exp': B1g_exp,
            'Pair_s':  Pair_s,
            'Pair_d':  Pair_d,
        }

    def compute_gap_eq_vectorized(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, K_eff: float, g_Delta_s: float, g_Delta_d: float, _j_renorm: float, _bdg_cache: tuple = None, _vertex_cache: dict = None, lambda_0: float = 1.5, z_qp: float = 1.0) -> Tuple[float, float, dict]:
        """
        Gap equation with q-dependent RPA pairing vertex V(q) built from normal-state
        (Δ=0) susceptibilities. V(q) is ALWAYS from the normal state; Δ enters only
        via F_AA/F_AB. Sign convention: V>0 attractive, V<0 repulsive (Δ forced to 0).
        abs() is NOT applied to the full product — vertex sign is physically respected.

        V_d_scalar is the d-wave Rayleigh projection onto φ_d = cos(kx)−cos(ky) and
        CAN be negative (spin-fluctuations repulsive in d-channel → d-wave suppressed).

        Stabilization: λ_pair estimated via O(N_fs) Rayleigh quotient, no eigh.
        Only the super-critical excess is penalised (Hessian sign change criterion):
            lambda_excess = max(λ_pair − 1, 0)
            f_stab = 1 / (1 + lambda_excess / λ_0),  floor 0.05
        λ_0 (default 1.5) is the damping scale above the critical point.

          - s-channel: V_s_scalar = <phi_s|V_mat|phi_s>/<phi_s|phi_s>, FS projection with phi_s=1.
            Physically correct: the s-channel singlet Γ₆⊗Γ₇ is inter-orbital
            on the same site, so q=0 is the appropriate momentum.
          - d-channel: V_d_scalar = FS-projected Rayleigh quotient of V(q) over
            all q = k_i − k_j pairs on the Fermi surface. This is a mean-field
            approximation to the d-wave gap equation kernel.
          - F_AA/F_AB: full-BZ BdG anomalous amplitudes at the CURRENT (Δ,M,Q).
        """
        solver = self.solver
        solver._gap_amplitude = 0.0

        # --- BdG amplitudes on the full k-grid (SC state, includes Δ) ---
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = np.linalg.eigh(self._build_H_stack(self._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=self._H_stack))

        arg = np.clip(ev / solver.p.kT, -100, 100)
        f12 = 1.0 - 2.0 / (1.0 + np.exp(arg))   # tanh(E/2kT); (N_k, 16)

        uA, uB, vA, vB = self._get_nambu_spinors(ec)

        # Full-BZ pair amplitudes (consistent with compute_observables_vectorized Pair_s/d)
        pair_s_k = np.sum(
            (uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
                + uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])) * f12,
            axis=1)
        pair_d_k = np.sum(
            0.5 * (uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
                    + uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])) * f12,
            axis=1)

        # Full-BZ integrals
        F_AA_BZ = float(np.real(np.dot(solver.k_weights, pair_s_k))) / 4.0
        F_AB_BZ = float(np.real(np.dot(solver.k_weights, pair_d_k))) / 4.0

        # --- Fermi-surface points for vertex q-loop (SC state geometry) ---
        fs_pts, vF_arr, _, _, _ = solver._get_fs_points(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, store_cache=True, compute_vF=True)
        N_fs = len(fs_pts)
        phi_d = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
        phi_s = np.ones(N_fs)

        # --- Vertex cache invalidation (Δ-independent!) ---
        _vc_is_dict = isinstance(_vertex_cache, dict)
        _cache_doping = _vertex_cache.get('target_doping', float('inf')) if _vc_is_dict else float('inf')
        _doping_changed = abs(target_doping - _cache_doping) > 1e-4

        _vertex_stale = (
            not _vc_is_dict
            or not _vertex_cache.get('chi_QQ_from_normal', False)   # must be normal-state χ
            or abs(M - _vertex_cache.get('M', 0.0)) > _M_THR_REL
            or abs(Q - _vertex_cache.get('Q', 0.0)) > max(_Q_THR_REL * solver.p.lambda_hop, 1e-4)
            or abs(_j_renorm - _vertex_cache.get('j_renorm', 0.0)) > 0.05
            or abs(z_qp   - _vertex_cache.get('z_qp',    1.0)) > 0.05
            or _vertex_cache.get('fs_pts') is None
            or len(_vertex_cache['fs_pts']) != N_fs
            or _doping_changed
        )

        if _vertex_stale:
            tx_bare_v, ty_bare_v = solver.p.effective_hopping_anisotropic(Q)
            J_eff = _j_renorm * solver.p.Z * solver.p.effective_superexchange(g_J, tx_bare_v, ty_bare_v, target_doping)

            q_zero = np.zeros(2)
            _q_afm = np.array([np.pi, np.pi])
            chi_QQ_normal = solver._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)
            E_k_cache_normal = solver._get_chi0_norm_cache(M, Q, mu, tx, ty, g_J, self, target_doping=target_doping)

            # FM/Pomeranchuk QCP check at q=0
            _n_sus_q0_v = solver.get_susceptibilities_normal(q_zero, M, Q, target_doping, target_doping, mu, tx, ty, g_J, chi_QQ_normal, K_eff, J_eff,
                                                            E_k_cache_normal, z_qp=z_qp)
            _det_q0 = _n_sus_q0_v['rpa_det']

            # AFM QCP check at q=(π,π)
            _n_sus_afm_v = solver.get_susceptibilities_normal(_q_afm, M, Q, target_doping, target_doping, mu, tx, ty, g_J, chi_QQ_normal, K_eff, J_eff,
                                                            E_k_cache_normal, z_qp=z_qp)
            _det_afm = _n_sus_afm_v['rpa_det']
            _chi_DD_s_full = _n_sus_afm_v['chi_DD_s']
            _chi_DD_s_moriya_full = _n_sus_afm_v['chi_DD_s_moriya']

            # V(q) is evaluated at all q = k_i - k_j on the FS (including q=0,
            # which carries the JT/Pomeranchuk physics) and assembled into V_mat.
            # Channel projections via BCS/Eliashberg basis functions:
            #   V_d = <phi_d | V_mat | phi_d> / <phi_d | phi_d>  (d-wave, B1g)
            #   V_s = <phi_s | V_mat | phi_s> / <phi_s | phi_s>  (s-wave, A1g)
            # This integrates the full width of the V(q) peaks, correctly weights
            # the formfactor (phi_d changes sign at the d-wave nodes, cancelling
            # forward scattering), and tracks FS anisotropy induced by Q != 0
            if N_fs > 6:
                iu, ju  = np.triu_indices(N_fs)
                q_raw   = fs_pts[iu] - fs_pts[ju]
                q_arr   = (q_raw + np.pi) % (2.0 * np.pi) - np.pi

                # q ↔ -q symmetry reduction: canonical representation
                # V(q) = V(-q) because χ₀(q) = χ₀(-q) (time-reversal of the Lindhard sum) and the RPA denominator is even in q.
                _scale_int = int(1e5)

                def _canonical_q_int(q_norm: np.ndarray) -> np.ndarray:
                    qi = np.rint(q_norm * _scale_int).astype(np.int64)
                    for c in qi:
                        if c < 0: return -qi
                        if c > 0: return  qi
                    return qi

                q_int_canonical  = np.array([_canonical_q_int(q) for q in q_arr])
                u_q_int, inv_idx = np.unique(q_int_canonical, axis=0, return_inverse=True)
                u_q_vecs         = u_q_int.astype(np.float64) / _scale_int

                V_rpa = np.empty(len(u_q_vecs), dtype=float)
                for ui, q_u in enumerate(u_q_vecs):
                    _n_sus_qu = solver.get_susceptibilities_normal(q_u, M, Q, target_doping, target_doping, mu, tx, ty, g_J, chi_QQ_normal, K_eff, J_eff,
                                                                E_k_cache_normal, z_qp=z_qp)
                    V_rpa[ui] = _n_sus_qu['V_full']

                V_mat = np.zeros((N_fs, N_fs))
                vvals = V_rpa[inv_idx]
                V_mat[iu, ju] = vvals
                V_mat[ju, iu] = vvals

                phi_s_norm = max(float(np.dot(phi_s, phi_s)), 1e-12)
                phi_d_norm = max(float(np.dot(phi_d, phi_d)), 1e-12)

                V_s_scalar = float(phi_s @ V_mat @ phi_s) / phi_s_norm
                V_d_proj   = phi_d @ V_mat
                V_d_scalar = float(np.dot(phi_d, V_d_proj)) / phi_d_norm
            else:
                V_d_proj     = np.zeros(N_fs)
                V_d_scalar = 0.0
                V_s_scalar   = solver.p.g_JT**2 / max(solver._K_bare, 1e-9)

            _vertex_cache = {
                'M': M,
                'Q': Q,
                'j_renorm': _j_renorm,
                'z_qp': z_qp,
                'fs_pts': fs_pts.copy(),
                'V_s_scalar': V_s_scalar,
                'V_d_scalar': V_d_scalar,
                'V_d_proj': V_d_proj.copy(),
                'phi_d': phi_d.copy(),
                'det_q0': _det_q0,
                'det_afm': _det_afm,
                'chi_QQ': chi_QQ_normal,
                'chi_QQ_from_normal': True,
                'target_doping': target_doping,
                'chi_DD_s_full': _chi_DD_s_full,
                'chi_DD_s_moriya_full': _chi_DD_s_moriya_full,
                'vF_arr': vF_arr,
            }

        # Rayleigh-quotient λ_pair proxy: _{ij} = V(k_i−k_j) / √(|vF_i|·|vF_j|) is the linearised gap kernel. Rayleigh quotient with trial vector φ gives: λ ≈ <φ|Γ|φ>/<φ|φ> ≈ V_scalar · <φ²·(1/|vF|)>/<φ²> = V_scalar · N_eff
        _inv_vF  = 1.0 / np.maximum(np.abs(_vertex_cache['vF_arr'] ), _VF_FLOOR_TIGHT)

        # d-wave: DOS-weighted Rayleigh  <φ_d²·(1/|vF|)> / <φ_d²>
        _phi_d   = _vertex_cache['phi_d']
        _phi2    = max(float(np.dot(_phi_d, _phi_d)), 1e-12)
        _N_eff_d = float(np.dot(_phi_d**2, _inv_vF)) / _phi2

        # s-wave: uniform trial vector → mean(1/|vF|)
        _N_eff_s = float(np.mean(_inv_vF))

        # max(V, 0): only attractive channels drive the pairing instability
        lambda_d_ray = max(g_Delta_d * _vertex_cache['V_d_scalar'] * _N_eff_d, 0.0)
        lambda_s_ray = max(g_Delta_s * _vertex_cache['V_s_scalar'] * _N_eff_s, 0.0)
        lambda_pair  = max(lambda_d_ray, lambda_s_ray)

        # ── Stabilization factor ──────────────────────────────────────────────
        # Only penalise the super-critical excess: λ_excess = max(λ_pair − 1, 0).
        # When λ_pair < 1 the Hessian ∂²F/∂Δ² is still positive → f_stab = 1 exactly.
        # When λ_pair > 1 the Hessian has gone negative → damp proportionally.
        #
        #   f_stab = 1 / (1 + lambda_excess / λ_0)
        #
        # λ_0 (parameter, default 1.5): damping scale ABOVE the critical point.
        # Larger λ_0 allows stronger super-critical coupling before significant damping.
        # Floor 0.05: keeps SCF contractive even in deep instability regions.
        lambda_excess = max(lambda_pair - 1.0, 0.0)
        f_stab = 1.0 / (1.0 + lambda_excess / max(lambda_0, 1e-12))
        f_stab = max(f_stab, 0.05)

        _vertex_cache['lambda_pair'] = lambda_pair
        _vertex_cache['f_stab']      = f_stab

        # ── Gap equations — vertex sign respected, no fake attraction ─────────
        # V > 0: attractive → Cooper pairs form.
        # V ≤ 0: repulsive  → channel suppressed (Δ = 0), not abs()-rescued.
        # abs() applied only to the final amplitude (phase convention).
        V_s = _vertex_cache['V_s_scalar']
        V_d = _vertex_cache['V_d_scalar']

        Delta_s_new = abs(g_Delta_s * V_s * F_AA_BZ * f_stab) if V_s > 0.0 else 0.0
        Delta_d_new = abs(g_Delta_d * V_d * F_AB_BZ * f_stab) if V_d > 0.0 else 0.0
        return Delta_s_new, Delta_d_new, _vertex_cache


class OptimPoint:
    __slots__ = ('doping', 'Delta_tetra', 'u', 'g_JT', 't_pd',
                 'Delta_total', 'converged', 'result',
                 'lambda_JT', 'lambda_lin_max', 'stoner_ok', 'score', 'Tc',
                 'lambda_soc', '_exclude_from_gp')

    def __init__(self, doping, Delta_tetra, u, g_JT, t_pd, Delta_total, converged, result=None,
                 lambda_JT=0.0, lambda_max=0.0, stoner_ok=True, score=0.0, Tc=0.0, lambda_soc=None):
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

def check_sc_jt_window(params: ModelParams, result: Dict, G_base: Dict) -> Dict:
    """
    SC-triggered JT viability check and window diagnostics.

    Window: K_spont < K_lattice < K_SC
        K_spont = g²/Δ_CF                         lower bound — spontaneous atomic JT onset
        K_SC    = g²·|χ_τ_sc| / _LAMBDA_JT_VIABLE upper bound — K at which λ_JT = 0.05

    Returns
    ───────
    viable        : bool — window open AND K_lattice inside AND structural_ok
    window_open   : bool — K_SC > K_spont
    K_in_window   : bool — K_spont < K_lattice < K_SC
    structural_ok : bool — K_eff > chi_QQ AND lambda_min > 0
    """
    g_JT       = params.g_JT
    Delta_CF   = params.Delta_CF
    K_lattice  = params.K_lattice

    chi_tau_sc    = float(result['chi_tau_sc'])
    K_eff         = max(float(result['K_eff_scf']), 1e-12)

    # total electronic softening curvature (eV/Å²).
    chi_QQ     = G_base['chi_QQ'] 
    lambda_min = G_base['lambda_min']

    K_spont = g_JT ** 2 / max(Delta_CF, 1e-12)
    chi_tau_sc_mag = max(-chi_tau_sc, 0.0)
    K_SC = g_JT ** 2 * chi_tau_sc_mag / max(_LAMBDA_JT_VIABLE, 1e-12)
    window_open = K_SC > K_spont

    if window_open:
        K_opt           = float(np.sqrt(K_spont * K_SC))
        lambda_JT_opt   = float(np.sqrt(_LAMBDA_JT_VIABLE * g_JT ** 2 * chi_tau_sc_mag / max(K_spont, 1e-12)))
        window_width    = K_SC - K_spont
        K_distance      = float(np.log(K_lattice / max(K_opt, 1e-12)))
    else:
        K_opt         = K_SC  = K_spont
        lambda_JT_opt = 0.0
        window_width  = 0.0
        K_distance    = float('nan')

    # the physical coupling is determined by the actual stiffness
    lambda_JT_sc = g_JT ** 2 * chi_tau_sc_mag / K_eff

    K_in_window     = K_spont < K_lattice < K_SC  
    normal_unstable = (lambda_min <= 0.0)
    viable       = window_open and K_in_window and not normal_unstable
    
    if normal_unstable:
        _base = (f"Normal state spontaneously unstable (λ_min={lambda_min:.4f} ≤ 0). "
                 f"SC-triggered JT viable={viable}; ")
    else:
        _base = ""
        
    if not window_open:
        note = (_base + f"Window closed: |χ_τ_sc|·Δ_CF={chi_tau_sc_mag*Delta_CF:.4f} < "
                f"λ_JT_viable={_LAMBDA_JT_VIABLE:.2f}. "
                f"Need |χ_τ_sc| > {_LAMBDA_JT_VIABLE/max(Delta_CF,1e-12):.4f} eV⁻¹"
                f" (current={chi_tau_sc:.4e}).")
    elif not K_in_window:
        if K_lattice <= K_spont:
            note = _base + f"K_lattice={K_lattice:.4f} ≤ K_spont={K_spont:.4f}: spontaneous JT risk."
        else:
            note = (_base + f"K_lattice={K_lattice:.4f} ≥ K_SC={K_SC:.4f}: lattice too stiff for SC-JT")
    else:
        frac = (K_lattice - K_spont) / max(window_width, 1e-12)
        note = _base + f"Viable. K_lattice at {frac*100:.0f}% of window"

    return {
        'viable':           viable,
        'window_open':      window_open,
        'K_in_window':      K_in_window,
        'normal_unstable':  normal_unstable,
        'K_eff':            K_eff,
        'K_spont':          K_spont,
        'K_SC':             K_SC,
        'K_opt':            K_opt,
        'K_distance':       K_distance,
        'lambda_JT_sc':     lambda_JT_sc,
        'lambda_JT_opt':    lambda_JT_opt,
        'window_width':     window_width,
        'note':             note,
    }

class UnifiedBayesianOptimizer:
    """
    Unified 5D Bayesian optimiser over the full (Delta_tetra, lambda_soc, u, g_JT, t_pd) space.

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
      H3: G22 > 0                     — JT channel not self-crossing in normal state.
              G3[2,2] = 1 − χ_QQ/K_eff; captures spontaneous-JT risk from any source,
              including large Delta_inplane (D₂h spin-conserving elements in B1g_op).
      H4: g_t >= _G_T_COHERENCE_MIN  — coherent Fermi surface (Mott guard; g_t encodes full doping-dependent Mott suppression)

    Soft constraints / DE penalty (S1–S5, weights sum to 1.0):
      S1 (w=0.225): 0 < lambda_min(G3) < 0.15  — near-critical, not past QCP
      S2 (w=0.225): reward larger lambda_max monotonically; only penalise near-divergence (λ_max > 0.95) and unsolvable cases.
                    first-order transitions with small λ_max in the normal state are not penalised.
      S3 (w=0.180): lambda_JT > 0.05            — SC-JT coupling above threshold
      S4 (w=0.270): λ_JT window position parabolic arch [0.05, 1.0], peak at 0.45.
      S5 (w=0.100): G22-margin > _DE_G22M_SAFE  — distance from spontaneous-JT boundary

    Post-SCF scoring gates (multiplicative):
      Tier 1 hard guards: mott_suspect / g_t<_G_T_COHERENCE_MIN / ξ/a<1, jchi, G22/λ_min.
      Tier 2 smooth weights:
        w_lJT        : parabolic arch on [0,1], peak at λ_JT=0.45
        w_lJT_kernel : sigmoid(10·(lJTk−0.05))
        w_hessian    : sigmoid(−λ_min_SC/0.05), floor 0.30
      Tier 3 objective: Tc_proxy × conv_f × stoner_f × g22_margin_f × xi_f
                        × lmax_boost × jchi_gate
        g22_margin_f : sigmoid((G22 − _BO_G22_MARGIN_CTR) / _BO_G22_MARGIN_W)
                       rewards distance from spontaneous-JT boundary continuously.

    Thread safety
    -------------
    _gp_lock : guards _gp, _gp_obs, observations (register + fit_gp snapshot pattern).
    _tr_lock : guards _tr_radius, _tr_center, _improve, _no_improve.
    Trust-region state is mutated only from the main thread (after batch join)
    """
    _NDIMS   = 5
    _SEED_DE = 42
    _SEED_LHS= 43

    # Soft-constraint weights
    _W_LMIN = 0.225  # S1: lambda_min(G3) near-critical window
    _W_LEFF = 0.225  # S2: lambda_max pairing-vertex window
    _W_LJT  = 0.180  # S3: lambda_JT SC-JT threshold
    _W_DLAM = 0.270  # S4: λ_JT parabolic arch penalty
    _W_G22M = 0.10   # S5: optimal delta_inplane (weight in DE penalty) 

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
    def _make_solver(self, dt: float, ls: float, u: float, gJT: float, tpd: float) -> 'RMFT_Solver':
        """Clone solver with all five parameters set; always rebuilds orbital operators."""
        s = copy.copy(self.solver)
        s.p = copy.copy(self.solver.p)
        s.p.Delta_tetra = float(dt)
        s.p.lambda_soc  = float(ls)
        s.p.u           = float(u)
        s.p.g_JT        = float(gJT)
        s.p.t_pd        = float(tpd)
        s._full_rebuild()
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
    def _eval_constraints(self, solver: 'RMFT_Solver', doping: float) -> Dict:
        """
        Evaluate H1/H2/H3 hard and S1–S5 soft constraints on a solver clone.

        Phase 1 — cheap  (~few ms):
            → H1: d²F/dQ² > 0   (normal-state Q-stability)
            → H2: J·χ_SS < 1    (below Stoner QCP)
            → H3: G22 > 0        (JT channel stable)
            → S1: λ_min near-critical window
            → S2: λ_eff in pairing window
            → S3: λ_JT above threshold
            → S5: G22-margin — rewards distance from the spontaneous-JT boundary
        Early exit: if any H fails or partial_penalty(S1+S2+S3+S5) ≥ _S4_SKIP_THRESHOLD,
        return S4 = nan (treated as 0.5 by the DE objective).

        Phase 2 (cheap; only for promising points):
            Uses lam_JT already computed in Phase 1 (no extra G-matrix call).
            → S4: λ_JT in SC-triggered window [0.05, 1.0]  (parabolic arch penalty)

        Skip rule: partial_penalty ≥ 0.25 → infeasible regardless of S4.
        """
        _FEASIBILITY_THRESHOLD = 0.25   # penalty >= this → infeasible regardless of S4

        # Pre-SCF Mott hard-reject
        _abs_d_pre = max(abs(doping), 1e-6)
        if (2.0 * _abs_d_pre) / (1.0 + _abs_d_pre) < _G_T_COHERENCE_MIN:
            return {
                'hard_fail': True,
                'penalty':   100.0,
                'H1': 0.0, 'H2': 0.0, 'H3': 0.0,
                'jchi': 0.0,
                'mott_reject': True,
                'G_res': {},
            }

        M0 = solver.p.estimate_M0(doping, 0.0)

        # ── Phase 1: cheap G-matrix without dlambda ────────────────────────────
        G_res = solver.compute_G_instability(doping, M0, compute_dlambda=False)

        H1 = float(G_res['d2F_Q_normal'])
        H3 = float(G_res['G22'])
        jchi = G_res['J_eff'] * G_res['chi_DD_s_moriya']
        chi_ss_gapped = float('nan')
        if jchi >= 1.0:
            try:
                chi_ss_gapped = solver.compute_chi_ss_with_infinitesimal_gap(M0, G_res, doping, delta_test=1e-5)
                if (np.isfinite(chi_ss_gapped)
                        and chi_ss_gapped < G_res['chi_DD_s_moriya']   # gap must suppress χ
                        and G_res['J_eff'] * chi_ss_gapped < 0.98):    # cap: not unconditionally safe
                    jchi = G_res['J_eff'] * chi_ss_gapped
                # else: keep jchi = jchi → H2 ≤ 0 → hard-fail as version A
            except Exception:
                pass   # keep jchi = jchi

        H2 = 1.0 - jchi

        if H1 <= 0.0 or H2 <= 0.0 or H3 <= 0.0:
            return {'hard_fail': True,
                    'penalty': (max(0.0, -H1) + max(0.0, -H2) + max(0.0, -H3)) * 10.0,
                    'H1': H1, 'H2': H2, 'H3': H3, 'jchi': jchi,
                    'jchi': jchi, 'chi_ss_gapped': chi_ss_gapped,
                    'G_res': G_res}

        lmin    = float(G_res['lambda_min'])
        V_JT    = solver.p.g_JT**2 / max(solver._K_bare, 1e-9)
        chi_orb = float(G_res['chi_QQ']) / max(solver.p.g_JT**2, 1e-12)
        lam_JT  = V_JT * chi_orb   # = chi_QQ/K_bare; dimensionless

        S1 = (0.0 if 0.0 < lmin < 0.15
              else min(abs(lmin) if lmin <= 0 else max(0.0, lmin - 0.15), 1.0))

        # S2: λ_max from the linearised gap equation at Δ=0.
        try:
            _t_eff_s2 = G_res['g_t'] * solver.p.t0
            lmax_s2 = solver.solve_linearized_gap_equation(M0, 0.0, 0.0+0j, 0.0+0j, doping, float(G_res['mu_n']), _t_eff_s2, _t_eff_s2, float(G_res['g_J']), float(G_res['K_eff']), float(G_res['J_eff']), actual_doping=doping)['lambda_lin_max']
        except Exception:
            lmax_s2 = float('nan')

        if not np.isfinite(lmax_s2):
            S2 = 1.0   # unknown → maximal penalty
        elif lmax_s2 > 0.95:
            # RPA vertex near-divergent: numerically unreliable
            S2 = float(1.0 / (1.0 + np.exp(20.0 * (lmax_s2 - 0.95))))
        else:
            S2 = float(1.0 - 1.0 / (1.0 + np.exp(-15.0 * (lmax_s2 - 0.15))))

        S3 = max(0.0, 0.05 - lam_JT) / 0.05
        
        # ── Phase 2: expensive dlambda only for potentially feasible candidates ─
        S4 = 0.5
        S5 = float(1.0 - np.tanh(H3 / max(_DE_G22M_SAFE, 1e-9)))
        partial_penalty = (self._W_LMIN * S1 + self._W_LEFF * S2 + self._W_LJT * S3 + self._W_G22M * S5)

        if partial_penalty < _FEASIBILITY_THRESHOLD:
            # S4: reward λ_JT in the SC-triggered window (0.05, 1.0); penalise outside.
            # lam_JT = g²·χ_QQ/K_bare (normal-state, cheap proxy for SC-triggered strength).
            # Optimal target: λ_JT ∈ [0.10, 0.80] — parabolic arch peaking at 0.45.
            if lam_JT <= 0.0 or lam_JT >= 1.0:
                S4 = 1.0   # outside window → maximum penalty
            else:
                lJT_c = float(np.clip(lam_JT, 0.0, 1.0))
                # Invert the parabolic arch used in _score so the penalty is 0 at peak (0.45) and 1 at boundaries.  S4 = 0 → no penalty; S4 = 1 → max penalty.
                arch  = float(np.clip(-lJT_c * (lJT_c - 1.0) / 0.2025, 0.0, 1.0))
                S4    = 1.0 - arch   # penalty = 1 - reward

        penalty = partial_penalty + self._W_DLAM * S4
        return {
            'hard_fail': False, 'penalty': float(penalty),
            'feasible':  penalty < _FEASIBILITY_THRESHOLD,
            'H1': H1, 'H2': H2, 'H3': H3, 'jchi': jchi,
            'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5,
            'lam_JT': lam_JT, 'lmin': lmin, 'G_res': G_res,
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
                              'S5': res.get('S5', float('nan')),
                              'score': max(0.0, 1.0 - res['penalty'])})
            return res['penalty']

        t0 = _time.time()
        if verbose:
            _scf_log("DE-SCOUT", "="*60)
            _scf_log("DE-SCOUT", f"5D DE scout (analytic G-matrix, no SCF)"
                                 f"  pop={popsize*self._NDIMS}  maxiter={maxiter}  doping={doping:.3f}")
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
                         f"  penalty={r['penalty']:.4f}"
                         f"  S5={r.get('S5', float('nan')):.3f}")
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
            _scf_log("GP-SEED", f"Parallel SCF seed: {len(candidates)} candidates,  doping-scan: {self.n_doping_scan} pts/material")

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
            _scf_log("GP-SEED", f"Done ({(_time.time()-t0)/60:.1f} min)  GP obs={len(self._gp_obs)}/{len(self.observations)}")
            if self._gp_obs:
                _scf_log("GP-SEED", f"Best seed: {max(self._gp_obs, key=lambda o: o.score)}")

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
            _best_x_snap = self._obs_to_X(max(self._gp_obs, key=lambda o: o.score))
        with self._tr_lock:
            center = self._tr_center.copy() if self._tr_center is not None \
                     else _best_x_snap
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
            _scf_log("TURBO", f"TuRBO: {n_iterations} iters x {n_batch} pts/iter  TR_init={r0:.2f}")

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
            _scf_log("TURBO", f"Done ({(_time.time()-t0)/60:.1f} min)  GP obs={len(self._gp_obs)}/{len(self.observations)}")

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
    def _scan_doping(self, solver, doping_grid, Delta_tetra, u, gJT, t_pd, lsoc=None) -> 'OptimPoint':
        """Warm-started doping scan; returns best-scoring OptimPoint across the grid."""
        best: Optional[OptimPoint] = None
        prev: Optional[Dict]       = None
        iM0 = solver.p.estimate_M0(doping_grid[0], 0.0)
        iQ0, iD0 = 1e-4, 0.02
        for doping in doping_grid:
            iM = prev['M']                                 if prev else iM0
            iQ = prev['Q']                                 if prev else iQ0
            iD = max(prev['Delta_s']+prev['Delta_d'], iD0) if prev else iD0
            pt = self._eval_one_doping(solver, doping, Delta_tetra, u, gJT, t_pd, iM, iQ, iD, lambda_soc=lsoc)
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
                # Compare total free energy (BdG + cluster) to avoid basin bias from local K_eff differences between the Q=0 and Q≠0 solutions.
                _F1_tot = result['F_bdg'] + result.get('F_per_site', 0.0)
                _F2_tot = r2['F_bdg']    + r2.get('F_per_site', 0.0)
                if (r2['converged'] and abs(r2['Q']) > 1e-3
                        and r2['Delta_s'] + r2['Delta_d'] > solver.p.tol
                        and _F2_tot < _F1_tot + 1e-4):
                    result = r2
                    Delta  = r2['Delta_s'] + r2['Delta_d']
                    converged = True
                    Tc = 0.0
                    try:
                        Tc = solver.compute_Tc_by_gap_suppression(doping, sc_result=result)['Tc']
                    except Exception:
                        pass
            except Exception:
                pass

        M_conv = result['M'] if result is not None else initial_M
        G_n = (solver.compute_G_instability(doping, M_conv, compute_dlambda=False) if result is not None else {})
        if Delta < 1e-8 and Tc < 1e-6:
            return self._g_fallback_score(initial_M, doping, Delta_tetra, u, gJT, t_pd)

        stoner_ok = not result['afm_unstable']

        # lambda_JT = (g²/K)·χ_tau: SC-triggered JT coupling (requires Δ≠0).
        K_eff = max(result['K_eff_scf'], 1e-9)
        lambda_JT = (solver.p.g_JT**2 / K_eff) * max(-float(result['chi_tau_sc']), 0.0)

        # G_sc: SC-state instability proxy needs two quantities in _score:
        #   • d2F_Q_normal  : ∂²F/∂Q²|_{Δ≠0}  — SC-induced Q-mode softening
        #   • chi_SQ        : spin–quadrupole cross-channel strength at Δ≠0
        if result is not None and converged and Delta > solver.p.tol:
            chi_QQ_sc = solver._chi_QQ_matrix_elements(result['M'], result['Q'], doping, complex(result['Delta_s']), complex(result['Delta_d']), result['mu'])
            chi_QQ_n = solver._chi_QQ_matrix_elements(result['M'], result['Q'], doping, 0j, 0j, result['mu'])
            _chi_SQ_sc_proxy = chi_QQ_sc - chi_QQ_n
            G_sc = {
                'd2F_Q_normal': float(result['hessian_result']['H'][1][1]) if (result.get('hessian_result') and result['hessian_result'].get('H') is not None) else float('nan'),   # ∂²F/∂Q²|_{Δ≠0} from Hessian H[1,1]
                'chi_SQ':       _chi_SQ_sc_proxy,   # Δχ_QQ proxy for spin-quadrupole cross-channel
            }
        else:
            # No SC gap or not converged — _score handles nan gracefully via fallback 0.5
            G_sc: dict = {}

        score = self._score(Delta, converged, result, Tc, G_n, G_sc, lambda_JT)
        lambda_lin_max = result['lambda_lin_max']
        lambda_JT_kernel = result.get('lambda_JT_kernel', float('nan'))
        G_chi_K = G_n['chi_QQ'] / max(G_n['K_eff'], 1e-9)
        regime  = ('SC-triggered' if 0.05 < lambda_JT < 1.0
                     else ('strong-coupling' if lambda_JT >= 1.0 else 'JT-closed'))
        _hmin = (result['hessian_result'].get('min_curvature') or float('nan'))
        _scf_log(tag,
                 f"D={Delta:.5f} Tc={Tc*1000:.2f}meV score={score:.5f}"
                 f" lJT={lambda_JT:.3f}[{regime}]"
                 f" lJT_ker={lambda_JT_kernel:.3f}"
                 f" λ_lin_max={lambda_lin_max:.3f}({result['gap_symmetry']})"
                 f" cQQ/K={G_chi_K:.3f} lmin(H)={_hmin:+.4f}[{G_n['instab_dir']}]"
                 f" {'ok' if converged else 'nc'} ({_time.time()-t0:.1f}s)")
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, Delta, converged, result, lambda_JT, lambda_lin_max, stoner_ok, score, Tc, lambda_soc)

    def _g_fallback_score(self, M0, doping, Delta_tetra, u, gJT, t_pd) -> 'OptimPoint':
        """Cheap G-matrix proxy score when SCF finds no SC gap."""
        try:
            s2 = copy.copy(self.solver)
            s2.p = copy.copy(self.solver.p)
            s2.p.Delta_tetra = float(Delta_tetra)
            s2.p.u   = float(u)
            s2.p.g_JT = float(gJT)
            s2.p.t_pd = float(t_pd)
            s2._full_rebuild()
            G = s2.compute_G_instability(doping, M0, compute_dlambda=False)
        except Exception:
            return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)
        G22 = G['G22']; lm = G['lambda_min']; Te = G['Tc_estimate']
        if G22 <= 0.0 or lm <= 0.0:
            return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)
        g22_f = _BO_SPONT_JT_PEN + (1.0 - _BO_SPONT_JT_PEN) / (1.0 + np.exp(-G22 / _BO_SIGMOID_W))
        sc = _BO_G_FALLBACK * (1.0 - min(lm, 1.0)) * g22_f * (1.0 + min(Te / 0.004, 8.0))
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=sc)

    def _score(self, Delta: float, converged: bool, result: dict, Tc: float, G_n: dict, G_sc: dict, lambda_JT: float = float('nan')) -> float:
        """
        Post-SCF scoring: three-tier multiplicative architecture.

        Tier 1 — Hard physical constraints (return 0 immediately):
            mott    : g_t < _G_T_COHERENCE_MIN or ξ/a < 1.0  — incoherent / artefact SC
            jchi    : J·χ_SS > _JCHI_HARD_REJECT — deep AFM, SC impossible
            g22     : G22 ≤ 0 or λ_min(G3) ≤ 0  — spontaneous JT / unstable normal state
            scwo    : By analogy with the Schrieffer-Wolff correction, a simple penalty term when J_eff/Δ_CF ~ 0.2–0.35 → non-negligible

        Tier 2 — Smooth mechanism weights (no hard clips; continuous in [0,1]):
            w_lJT        : λ_JT ∈ (0,1) parabola peak at 0.45; zero at 0 and 1.
            w_lJT_kernel : sigmoid(10·(lJTk − 0.05))
            w_hessian    : sigmoid(−lmin_sc / 0.05)
            w_softening  : sigmoid(−(d²F_Q_sc − d²F_Q_n) / 0.05) — SC-induced Q-mode softening
                           d²F_Q_sc from converged Hessian H3[2,2]; d²F_Q_n from G_n['d2F_Q_normal']
            w_chisq      : sigmoid(|χ_SQ| / 0.1) — spin-orbital cross-channel strength
                           χ_SQ from G_sc proxy: Δχ_QQ = χ_QQ(Δ≠0) − χ_QQ(Δ=0)

        Tier 3 — Optimisation objective:
            Tc_proxy     : Tc if converged, else Δ·0.3
            xi_f         : coherence length sigmoid gate (centre ξ/a=2, width k=2)
            conv_f       : 1.0 if converged, else 0.10
            stoner_f     : 1.0 if AFM stable, else _BO_W_STONER_BAD
            lmax_boost   : softplus(λ_max)
            g22_margin_f : smooth reward for G22 above spontaneous-JT boundary (centre 0.25)
            jchi_gate    : Gaussian near-QCP sweet-spot reward
        """
        # ── Tier 1: hard guards ───────────────────────────────────────────────
        # Mott / incoherence guard (mirrors post-SCF Mott filter in solve_self_consistent)
        if result.get('mott_suspect', False):
            return 0.0
        _g_t_sc = float(result.get('g_t', 1.0))
        _xia_sc = float(result.get('xi_over_a', float('nan')))
        if _g_t_sc < _G_T_COHERENCE_MIN or (np.isfinite(_xia_sc) and _xia_sc < 1.0):
            return 0.0

        _jchi = float(np.clip(result['J_eff'] * result['chi_DD_s_moriya'], 0.0, 10.0))
        if _jchi > _JCHI_HARD_REJECT:
            return 0.0
        G22 = G_n['G22']
        lmin_n = G_n['lambda_min']
        if G22 <= 0.0 or lmin_n <= 0.0:
            return 0.0

        # 4×4 projection (Γ₆⊕Γ₇a) is valid when Δ_CF ≫ J_eff. Virtual Γ₇b contributions scale as (J_eff/Δ_CF)².
        Delta_CF = self.solver.p.Delta_CF
        J_eff = result.get('J_eff', 0.0)
        if Delta_CF > 1e-9 and J_eff > 0.0:
            ratio = J_eff / Delta_CF
            # Penalty grows quadratically; 0 at ratio=0, 0.25 at ratio=0.5, 1 at ratio=1
            proj_penalty = float(np.clip(ratio**2, 0.0, 1.0))
        else:
            proj_penalty = 0.0
        proj_factor = 1.0 - 0.5 * proj_penalty
        
        # ── Tier 2: smooth mechanism weights ─────────────────────────────────
        # λ_JT arch
        lJT = float(lambda_JT)
        if not np.isfinite(lJT):
            w_lJT = 0.5
        elif lJT >= 1.0:
            w_lJT = 0.10
        else:
            lJT_c = float(np.clip(lJT, 0.0, 1.0))
            w_lJT = float(np.clip(-lJT_c * (lJT_c - 1.0) / 0.2025, 0.0, 1.0))

        # λ_JT_kernel
        lJTk = float(result.get('lambda_JT_kernel', float('nan')))
        if not np.isfinite(lJTk):
            w_lJT_kernel = 0.5
        elif lJTk >= 1.0:
            w_lJT_kernel = 0.30   # the Rayleigh quotient exceeding 1 signals numerical over-saturation (kernel eigenvalue > pairing eigenvalue
        else:
            w_lJT_kernel = float(1.0 / (1.0 + np.exp(-10.0 * (lJTk - 0.05))))

        # Hessian: sigmoid(−lmin_sc / 0.05). Floor 0.30 for missing/unconverged data.
        lmin_sc = result.get('hessian_result', {}).get('min_curvature', None)
        if lmin_sc is not None and np.isfinite(lmin_sc):
            w_hessian = float(1.0 / (1.0 + np.exp(lmin_sc / _BO_SC_HESS_SIG)))
        else:
            w_hessian = 0.30

        # SC-induced Q-mode softening (negative = softening)
        d2F_Q_sc = G_sc.get('d2F_Q_normal', float('nan'))
        d2F_Q_n = G_n.get('d2F_Q_normal', float('nan'))
        if np.isfinite(d2F_Q_sc) and np.isfinite(d2F_Q_n):
            jt_softening = d2F_Q_sc - d2F_Q_n      # negative = SC softens Q-mode
            w_softening = float(1.0 / (1.0 + np.exp(jt_softening / 0.05)))  # 1 if strongly negative
        else:
            w_softening = 0.5

        # χ_SQ spin-orbital cross-channel strength
        chi_sq = G_sc.get('chi_SQ', float('nan'))
        if np.isfinite(chi_sq):
            w_chisq = float(1.0 / (1.0 + np.exp(-abs(chi_sq) / 0.1)))  # 1 when χ_SQ > 0.1
        else:
            w_chisq = 0.5

        # A sigmoid centered at _BO_G22_MARGIN_CTR maps G22 just above 0 to ~0 and large values to ~1.
        g22_margin_f = float(1.0 / (1.0 + np.exp(-(G22 - _BO_G22_MARGIN_CTR) / _BO_G22_MARGIN_W)))

        # ── Tier 3: optimisation objective ───────────────────────────────────
        # Coherence gate: hard zero below ξ/a=1 (already checked above for finite values), sigmoid ramp centred at ξ/a=2, saturates at ξ/a=4.
        if not np.isfinite(_xia_sc):
            xi_f = 0.5
        elif _xia_sc < 4.0:
            xi_f = float(np.clip(1.0 / (1.0 + np.exp(-2.0 * (_xia_sc - 2.0))), 0.0, 1.0))
        else:
            xi_f = 1.0

        Tc_proxy = Tc if Tc > 1e-6 else Delta * 0.3
        conv_f = 1.0 if converged else 0.10
        # afm_unstable is evaluated at Δ=0 (vertex cache); it flags that J·χ(Δ=0)≥1 and the
        # RPA vertex is singular.  If True the λ_max / Tc estimates are unreliable, so we apply a strong penalty rather than a hard zero
        stoner_f = 1.0 if not result['afm_unstable'] else _BO_W_STONER_BAD

        # softplus(λ_max) grows continuously with pairing strength.
        _softplus = float(np.log1p(np.exp(np.clip(result['lambda_lin_max'], -10.0, 10.0))))
        lmax_boost = float(np.clip(_softplus, 0.0, 2.0))

        # jchi_gate shapes the score *within* the feasible region to prefer the near-QCP sweet spot.
        jchi_gate = float(np.exp(-0.5 * ((_jchi - _BO_OPT_JCHI) / _BO_SIG_JCHI) ** 2))
        jchi_gate = float(np.clip(
            jchi_gate + (_BO_JCHI_FLOOR if _jchi < _BO_JCHI_NOISE else 0.0), 0.0, 1.0))
        return (Tc_proxy * conv_f * stoner_f * g22_margin_f * w_lJT * w_lJT_kernel * w_hessian * w_softening * w_chisq * xi_f * lmax_boost * jchi_gate * proj_factor)

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
                                   f"  GP={len(self._gp_obs)}/{len(self.observations)}  ({n_excl} hard-constrained excluded)")
            _scf_log("UNIFIED-BO", f"Best: {best}")

        return {'best_point': best, 'best_valid': best_valid, 'best_raw': best_raw,
                'observations': self.observations, 'de_archive': de_phase['archive'],
                'elapsed_s': elapsed}

    
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
        t_pd         = 0.600,
        u            = 17.500,
        lambda_soc   = 0.180,
        Delta_tetra  = -0.180,
        g_JT         = 0.190,
        K_lattice    = 1.800,
        lambda_hop   = 1.200,
        Delta_CT     = 3.000,
        omega_JT     = 0.057,
        Delta_inplane= 0.03,
        Z            = 4,
        kT           = 0.01,
        tol          = 1e-4,
        )

    target_doping  = 0.20
    doping_margin  = 0.20          # scan covers target ± 20 %
    min_doping     = max(target_doping * (1.0 - doping_margin), _G_T_COHERENCE_MIN / (2.0 - _G_T_COHERENCE_MIN))
    max_doping     = target_doping * (1.0 + doping_margin)
    _ref_M         = params.estimate_M0(target_doping, 0.0)
    _ref_Q         = 1e-8
    initial_Delta  = 1e-4
    solver         = RMFT_Solver(params)

    need_optimalization = False
    # ── Section 0: Unified 5D optimisation ───────────────────────────────────────────────────
    if need_optimalization:
        _scf_log("MAIN", "="*60)
        _scf_log("MAIN", "UNIFIED 5D OPTIMISATION  (DE scout → GP seed → TuRBO → local refine)")
        _scf_log("MAIN", "Search space: (Delta_tetra, lambda_soc, u, g_JT, t_pd)  — no parameter splitting")
        _scf_log("MAIN", "="*60)

        _5d_bounds = {
            'Delta_tetra': (-0.09, -0.03),
            'lambda_soc':  ( 0.18,  0.34),
            'u':           ( 10.0,  20.0),
            'g_JT':        ( 0.11,  0.24),
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

        _scf_log("MAIN", "="*60)
        _scf_log("MAIN", f"OPTIMISATION COMPLETE  ({res_unified['elapsed_s']/60:.1f} min total)")
        _scf_log("MAIN", "Global optimal parameters:")
        _scf_log("MAIN", f"  Δ_tet={best_final.Delta_tetra:.4f}"
                        f"  λ_soc={best_final.lambda_soc:.4f}"
                        f"  u={best_final.u:.4f}"
                        f"  g_JT={best_final.g_JT:.4f}"
                        f"  t_pd={best_final.t_pd:.4f} eV"
                        f"  K_latt={params.K_lattice:.4f} eV/Å²")
        _scf_log("MAIN", f"  |Δ|={best_final.Delta_total:.6f} eV"
                        f"  Tc={best_final.Tc*1000:.2f} meV"
                        f"  score={best_final.score:.6f}")

        params.Delta_tetra = best_final.Delta_tetra
        params.lambda_soc  = best_final.lambda_soc
        params.u           = best_final.u
        params.g_JT        = best_final.g_JT
        params.t_pd        = best_final.t_pd
        params.__post_init__()
        solver = RMFT_Solver(params)
    
    # ── Section 1: Reference SCF ─────────────────────────────────────────────
    _scf_log("REF-SCF", "="*60)   # Run SCF first.  All subsequent diagnostics use the self-consistent (M, μ)
    _scf_log("REF-SCF", f"Reference SCF at δ={target_doping:.3f}")
    _ref_result = None
    try:
        _ref_result = solver.solve_self_consistent(
            target_doping,
            initial_M     = _ref_M,
            initial_Q     = _ref_Q,
            initial_Delta = initial_Delta,
            verbose       = True,
        )
        _ref_M = _ref_result['M']
        _ref_Q = _ref_result['Q']
        _scf_log("REF-SCF", f"  converged={_ref_result['converged']}  mott_suspect={_ref_result.get('mott_suspect', False)}")
        _scf_log("REF-SCF", f"  M={_ref_M:.4f}  Q={_ref_Q:+.5f}  Δs={_ref_result['Delta_s']:.5f} eV  Δd={_ref_result['Delta_d']:.5f} eV  μ={_ref_result['mu']:.4f} eV  g_t={_ref_result['g_t'] :.3f}")
        _scf_log("REF-SCF", f"  Irrep R={_ref_result['selection_ratio']:.4f}  JT {'ALLOWED ✓' if _ref_result['selection_ratio'] > 0.05 else 'BLOCKED ✗'}")
    except Exception as _ref_err:
        import traceback as _tb
        _scf_log("REF-SCF", f"  Reference SCF failed: {_ref_err}")
        _scf_log("REF-SCF", _tb.format_exc())

    # ── Section 2: G-matrix at self-consistent M ────────────────────────────
    _scf_log("G-MATRIX", "="*60)
    G_base = solver.compute_G_instability(target_doping, _ref_M, compute_dlambda=False)
    _scf_log("G-MATRIX", f"h_afm={G_base['h_afm']:.4f} eV, N_eff={G_base['N_eff']:.4f} eV⁻¹")
    _scf_log("G-MATRIX", f"χ_ΔΔ (dom)={G_base['chi_DD_dom']:.4f}  χ_DD_s={G_base['chi_DD_s']:.4f}"
             f"  χ_DD_d={G_base['chi_DD_d']:.4f}  χ_DD_sd={G_base['chi_DD_sd']:.4f}  [eV⁻¹]")
    _scf_log("G-MATRIX", f"χ_ΔQ (dom)={G_base['chi_DQ_dom']:.4f}  χ_ΔQ_s={G_base['chi_DQ_s']:.4f}"
             f"  χ_ΔQ_d={G_base['chi_DQ_d']:.4f}  [eV⁻¹]")
    _scf_log("G-MATRIX", f"K_eff_norm = {G_base['K_eff']:.4f} eV/Å²  χ_QQ_norm = {G_base['chi_QQ']:.4f} eV⁻¹")
    _scf_log("G-MATRIX", f"λ_JT_norm=χ_QQ_norm/K_eff_norm = {G_base['chi_QQ']/G_base['K_eff']:.4f}  ({'unstable, λ_JT_norm<1' if G_base['chi_QQ']/G_base['K_eff'] > 1 else 'stable, λ_JT_norm>1 '})")
    _scf_log("G-MATRIX", f"3×3 eigs: [{G_base['eigs3'][0]:.4f},{G_base['eigs3'][1]:.4f},{G_base['eigs3'][2]:.4f}]")
    _scf_log("G-MATRIX", f"evec_min=[{G_base['evec_min'][0]:.3f},{G_base['evec_min'][1]:.3f},"
             f"{G_base['evec_min'][2]:.3f}]  → instab_dir: {G_base['instab_dir']}")
    _scf_log("G-MATRIX", f"G3[2,2]={G_base['G22']:.4f}  [{"✓ G22>0: spontaneous JT blocked" if G_base['G22'] > 0 else "✗ G22≤0: spontaneous JT risk"}]")
    _scf_log("G-MATRIX", f"G11={G_base['G11']:.4f}  G12={G_base['G12']:.4f}  dom={G_base['dominant']}")

    _lmin_val  = G_base['lambda_min']
    _lmin_note = ("✗ SPONTANEOUS instability" if _lmin_val <= 0
                  else ("⚠ near-critical (0 < λ_min < 0.1)" if _lmin_val < 0.1
                        else "✓ normal-state stable"))
    _scf_log("G-MATRIX", f"λ_min={_lmin_val:.4f}  [{_lmin_note}]")
    _lambda_eff = G_base['lambda_eff']
    _leff_status = ("✓ optimal" if 0.3 < _lambda_eff < 1.0
                    else ("⚠ weak — increase J_eff (↓u or ↑t_pd/Δ_CT)" if _lambda_eff <= 0.3
                          else "⚠ too strong — risk of spontaneous JT / AFM QCP"))
    _scf_log("G-MATRIX", f"λ_eff=N_eff·V_eff={_lambda_eff:.4f}  [{_leff_status}]")
    _scf_log("G-MATRIX", f"∂²F/∂Q²|Δ=0={G_base['d2F_Q_normal']:+.5f} eV/Å²  "
             f"{'✓ normal-state Q-stable' if G_base['d2F_Q_normal'] > 0 else '✗ spontaneous JT!'}")
    _scf_log("G-MATRIX", f"||[τ_x,H]||={G_base['comm_norm']:.4f} eV  blocking={G_base['blocking_ratio']:.4f}")
    _scf_log("G-MATRIX", f" J_eff={G_base['J_eff']:.4f} χ_moriya={G_base['chi_DD_s_moriya']:.4f}")

    # ── Sections 3: SCF diagnostics, SC-JT window, Tc pre-estimates ────────
    if _ref_result is not None:
        _lmax_ref = float(_ref_result['lambda_lin_max'])
        _V_spin   = float(_ref_result['V_spin_mean'])
        _V_JT     = float(_ref_result['V_JT_mean'])
        _V_cr     = float(_ref_result['V_cross_mean'])
        _V_tot    = float(_ref_result['V_rpa_mean'])
        _hessian  = _ref_result['hessian_result']
        _gap_vec  = _ref_result.get('gap_vector', None)
        _fs_pts   = _ref_result.get('fs_pts',     None)

        _scf_log("SCF-RES", "Linearised gap equation (from SCF; full RPA vertex, self-consistent M and μ):")
        _scf_log("SCF-RES", f"  λ_max={_lmax_ref:.4f}  × g_Δ={_ref_result['g_delta_dom']:.3f})  sym={_ref_result['gap_symmetry']}")
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
            _ch_note  = 'd-wave dominant' if _w_d > _w_s else 's-wave dominant'
            _neg_note = ('  [⚠ λ<0: FS-avg vertex repulsive — instability requires nodal sign change]'
                         if _lmax_ref < 0 else '')
            _scf_log("SCF-RES", f"  Channel decomp: λ_s={_ls:.4f}  λ_d={_ld:.4f}"
                     f"  [{_ch_note}]{_neg_note}")
            
            if not _ref_result['valid_BdG']:
                note = f"⚠ ξ/a={_ref_result['xi_over_a']:.2f} < 2 — BdG marginal"
            elif _ref_result['orbital_selective']:
                note = (f"✓ ξ/a={_ref_result['xi_over_a']:.2f}  ORBITAL-SELECTIVE: "
                        f"ξ_Γ6/a={_ref_result['xi_Gamma6']:.2f} ξ_Γ7/a={_ref_result['xi_Gamma7']:.2f}")
            else:
                note = f"✓ ξ/a={_ref_result['xi_over_a']:.2f}  orbitally uniform"
            _scf_log("SCF-RES", f"[{note}]")

            _hess_lmin_sc = float(_hessian['eigenvalues'][0]) if _hessian.get('eigenvalues') is not None else float('nan')
            _scf_log("SCF-RES", f"  λ_min(H_SC)={_hess_lmin_sc:+.4f}  {'✓ SC-triggered JT CONFIRMED' if np.isfinite(_hess_lmin_sc) and _hess_lmin_sc < 0.0 else '— JT not triggered in converged state'}")
            # Improvement 4 (Critique 4): FS-resolved ∂λ/∂Q diagnostic
            _dlam_fs = _ref_result.get('dlam_dQ_fs', float('nan'))
            _scf_log("SCF-RES",
                     f"  ∂λ/∂Q(FS-resolved)={_dlam_fs:+.4f} eV⁻¹"
                     f"  {'✓ positive: SC condensate enhances pairing at hot spots under JT distortion'  if np.isfinite(_dlam_fs) and _dlam_fs > 0 else '⚠ non-positive or undefined: FS-level SC-JT coupling absent'}")
        
        # d-wave: negative FS-average is EXPECTED because forward scattering q≈0 is repulsive and dominates the mean;
        # the instability comes from backscattering at q≈(π,π) which is captured by the dominant eigenvector
        if np.isfinite(_V_tot) and abs(_V_tot) > 1e-4:
            _v_note = ('⚠ V_avg<0: d-wave backscattering dominant (normal)' if _V_tot < 0
                       else '✓ positive avg')
            _scf_log("G-MATRIX", f"  V_RPA(FS-avg)={_V_tot:.4f} eV  [{_v_note}]"
                     f"  spin={_V_spin/_V_tot*100:.0f}%  JT={_V_JT/_V_tot*100:.0f}%"
                     f"  cross={_V_cr/_V_tot*100:.0f}%")
        elif np.isfinite(_V_tot):
            _scf_log("G-MATRIX", f"  V_RPA(FS-avg)={_V_tot:.2e} eV"
                     f"  [spin={_V_spin:.3f}  V_JT={_V_JT:.3f}  cross={_V_cr:.3f} eV]")

        # — Stoner/Moriya diagnostic: via exact BdG χ_AFM —
        _stoner_r    = _ref_result['J_eff'] * _ref_result['chi_DD_s_moriya']
        _ston_status = ('✓ near QCP' if 1.0 > _stoner_r > 0.7
            else ('⚠ near/past AFM QCP' if 2.0 > _stoner_r >= 1.0
                else ('safe' if _stoner_r <= 0.7 else '✗ deeply past QCP')))
        _scf_log("SCF-RES", f"χ_AFM(Δ=0): χ_DD_s={_ref_result['chi_DD_s']:.4f}"
                 f"  χ_moriya={_ref_result['chi_DD_s_moriya']:.4f}  J_eff={_ref_result['J_eff']:.4f}"
                 f"  J_eff·χ_moriya={_stoner_r:.4f} [{_ston_status}]  RPA×={_ref_result['rpa_factor']:.3f}")
        _z_scf      = _ref_result.get('z_qp_scf', 1.0)
        _rJ_val     = 1.0 / max(_z_scf, 0.1)
        _rJ_excess  = max(0.0, _rJ_val - 1.0)
        _scf_log("SCF-RES", f"  r_J={_rJ_val:.3f}  (z_qp=1/r_J={_z_scf:.3f})"
                 f"  j_renorm={_ref_result.get('j_renorm', 1.0):.3f}"
                 f"  r_J_excess={_rJ_excess:.3f}"
                 f"  [correlation correction: χ₀ untouched (Gutzwiller bands); Γ_M × (1 + {_rJ_excess:.2f})]")
        
        # — SC-JT window (chi_tau keys always present in result dict) —
        _jt_win = check_sc_jt_window(params, _ref_result, G_base)
        diff_chi_tau = abs(_ref_result['chi_tau_sc']) - abs(_ref_result['chi_tau_n'])
        _scf_log("SCF-RES", f"  χ_τ_sc={_ref_result['chi_tau_sc']:.4f} eV⁻¹ (signed)  χ_τ_n={_ref_result['chi_tau_n']:.4f} eV⁻¹"
                f"  δχ_τ=|χ_τ_sc|-|χ_τ_n|={diff_chi_tau:.4f} eV⁻¹")
        _scf_log("SCF-RES", f"  {'✓ condensate enhances B1g softening: δχ_τ > 1e-6' if diff_chi_tau > 1e-6 else '⚠ no SC-induced B1g enhancement: δχ_τ < 1e-6'}")
        _scf_log("SCF-RES", f"  {'✓ electron system softens the lattice: χ_τ_sc < 0' if _ref_result['chi_tau_sc'] < 0 else '⚠ electron system stiffens the lattice: χ_τ_sc > 0'}")
        _scf_log("SCF-RES", f" K_eff={_jt_win['K_eff']:.4f}  K_spont={_jt_win['K_spont']:.4f}  K_SC={_jt_win['K_SC']:.4f}"
                f"  K_dist={_jt_win['K_distance']:+.3f}"
                f"  in_window={_jt_win['K_in_window']}  open={_jt_win['window_open']}"
                f"  K_opt={_jt_win['K_opt']:.4f}  K_lattice={solver._K_bare:.4f}")
        _scf_log("SCF-RES", f"  λ_JT_sc={_jt_win['lambda_JT_sc']:.4f}"
                f"  λ_JT_opt={_jt_win['lambda_JT_opt']:.4f}"
                f" { '✓ Richardson ok' if _ref_result['richardson_ok'] else '  ⚠ Richardson inconsistent'}"
                f" {'  ⚠ normal-state unstable' if _jt_win['normal_unstable'] else ''}")
        _scf_log("SCF-RES", f"{'⚠ SC could distort lattice (λ_JT > 1, spontaneous JT regime)' if _jt_win['lambda_JT_sc'] > 1 else '✓ SC-JT coupling active' if _jt_win['lambda_JT_sc'] > _LAMBDA_JT_VIABLE else '✗ SC-JT coupling too weak'}")
        _scf_log("SCF-RES", f"  → {_jt_win['note']}")

        # — Variables used in TC-PRELIM below —
        _pre_Delta_total = float(_ref_result['Delta_s']) + float(_ref_result['Delta_d'])
        _t_eff2    = float(np.sqrt(0.5 * (max(float(_ref_result['tx']), 1e-9)**2
                                         + max(float(_ref_result['ty']), 1e-9)**2)))
        _sc_viable = (not _ref_result.get('mott_suspect', False)) and (float(_ref_result['g_t']) >= _G_T_COHERENCE_MIN) and bool(_ref_result['converged'])

        _scf_log("TC-PRELIM", "="*60)
        _scf_log("TC-PRELIM", f"Pre-BO Tc estimates at |Δ|={_pre_Delta_total*1000:.3f} meV")
        # Tc₁: G-BCS analytic: uses λ_eff = N_eff · V_eff from the G-matrix (Schur-complement corrected).
        # Tc₂: λ_max-BCS with phonon cutoff: λ_max is the Fermi-surface-resolved pairing eigenvalue from the full RPA vertex.
        # The BCS prefactor uses max(t_eff, ω_JT) as the effective cutoff energy: in the SC-triggered JT picture the JT phonon sets the relevant boson scale when ω_JT > t_eff.    
        if _sc_viable:
            _scf_log("TC-PRELIM",
                    f"  Tc₁(G-BCS):    λ_eff={G_base['lambda_eff']:.4f}"
                    f"  → {G_base['Tc_estimate']*1000:.2f} meV  ({G_base['Tc_estimate'] / 8.617333e-5:.1f} K)")
            
            _cutoff = max(_t_eff2, params.omega_JT)     # relevant boson cutoff
            _Tc2_eV = float(1.13 * _cutoff * np.exp(-1.0 / max(_lmax_ref, 1e-9)))
            _scf_log("TC-PRELIM",
                    f"  Tc₂(λ_max-BCS):λ_max={_lmax_ref:.4f}"
                    f"  ω_JT={params.omega_JT*1000:.1f} meV"
                    f"  → {_Tc2_eV*1000:.2f} meV  ({_Tc2_eV / 8.617333e-5:.1f} K)")
        
            _lT = solver.compute_lambda_vs_T(target_doping, _ref_result)
            _scf_log("TC-SIMPLE", f"Tc(λ=1)={_lT['Tc_lambda']*1000:.2f} meV  slope={_lT['slope_at_Tc']*1000:.3f} meV⁻¹")

        # ── TC-PRELIM: Thermodynamic Tc from reference SCF (ad-hoc params) ──────
        if _pre_Delta_total > 1e-5:
            # In this SC-triggered JT model the expected ratio is > 3.52 because:
            #  the JT feedback loop cooperatively enhances Δ beyond the linearised BCS value
            #  the proximity to the AFM quantum critical region suppresses Tc relative to Δ₀ via pair-breaking spin fluctuations → same Δ₀, lower Tc → higher ratio
            Tc = solver.compute_Tc_by_gap_suppression(target_doping, _ref_result)['Tc']
            ratio_2D = 2.0 * _pre_Delta_total / Tc

            if ratio_2D < 3.8:
                coupling_regime = 'BCS-like'
            elif ratio_2D < 5.0:
                coupling_regime = 'strong'
            elif ratio_2D < 7.0:
                coupling_regime = 'very-strong'
            else:
                coupling_regime = 'exotic / non-phononic'
            _scf_log("TC-SIMPLE", f"2Δ₀/kTc = {ratio_2D:.3f}  [{coupling_regime}]  Tc={Tc / 8.617333e-5:.1f} K")

            _scf_log("TC-PRELIM", "  Thermodynamic Tc from reference SCF (warm-start; first-order aware):")
            try:
                _Tc_ref_thermo = solver.compute_Tc_thermodynamic(
                    target_doping,
                    sc_result  = _ref_result,
                    T_min      = 1e-4,
                    T_max      = 0.25,
                    n_scan     = 12,
                    n_bisect   = 8,
                    Delta_tol  = 1e-4,
                )
                _Tc_rt   = _Tc_ref_thermo['Tc']
                _Tc_rs   = _Tc_ref_thermo['Tc_spinodal']
                _ord_r   = _Tc_ref_thermo['transition_order']
                _Dj_r    = _Tc_ref_thermo['Delta_jump']
                _scf_log("TC-THERMO",
                         f"  Tc(thermo)={_Tc_rt*1000:.2f} meV ({_Tc_rt*11604:.1f} K)"
                         f"  Tc(spinodal)={_Tc_rs*1000:.2f} meV"
                         f"  order={_ord_r}"
                         f"  Δ_jump={_Dj_r*1000:.2f} meV")
                _scf_log("Tc-THERMO", f"  2Δ₀/kTc={_Tc_ref_thermo['ratio_2D']:.2f}")
                if _Tc_rs > 1e-6:
                    _uplift_r = (_Tc_rt - _Tc_rs) / _Tc_rs * 100.0
                    _scf_log("TC-THERMO",
                             f"  Tc uplift from SC-JT: {_uplift_r:+.1f}%"
                             f"  [{'first-order dominant' if abs(_uplift_r)>20 else 'weakly first-order' if abs(_uplift_r)>5 else 'effectively second-order'}]")
            except Exception as _Tc_rt_err:
                _scf_log("TC-THERMO", f"  Thermodynamic Tc (ref) failed: {_Tc_rt_err}")
        else:
            _scf_log("TC-THERMO",
                     f"  Thermodynamic Tc skipped: |Δ|={_pre_Delta_total*1000:.3f} meV < threshold")
    else:
        _scf_log("TC-THERMO", "  Thermodynamic Tc skipped: reference SCF not converged.")