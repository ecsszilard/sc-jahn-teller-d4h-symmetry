import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import numpy as np
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import opt_einsum as oe
from scipy.optimize import brentq, differential_evolution
from scipy.stats import norm, t as tdist
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings
import copy
import gc
import time as _time
import math
import concurrent.futures
import threading as _threading
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt

_log_lock = _threading.Lock()

# ── NOT free parameters ───────────────────────────────────────────────────────
_EV_TO_K              : float = 11604.518121        # 1 eV in Kelvin
_GW_G_J_PREFACTOR     : float = 4.0                 # g_J = 4/(1+δ)²  Gutzwiller exchange renormalization prefactor (slave-boson / Kotliar-Ruckenstein derivation, half-filling limit)
_GW_G_T_NUMERATOR     : float = 2.0                 # g_t = 2δ/(1+δ)  Gutzwiller hopping prefactor numerator coefficient
_PI_INT               : int   = 314159              # π in scaled integer units

# ── Fermi surface sampling and k-grid ──────────────────────────────────────────
_NK                   : int   = 72                  # k-grid points per direction (even required for commensurate q_AFM=(π,π))
_N_FS                 : int   = 130                 # FS k-points used in the vertex q-loop; samples the full k-grid, angular resolution need to resolve the d-wave node at (π/2,π/2) and the B₁g anti-nodal hot spots.
_FS_SAMPLING          : float = 4.4                 # integration window around the Fermi level
_FS_THERMAL_THRESHOLD : float = 0.0025              # 1% of peak value of f(E)*(1-f(E)) = 1/4, used as baseline for thermal FS weight
_VF_FLOOR             : float = 1e-4                # Fermi velocity floor (prevents 1/|v_F|→∞ at hot spots), Physical scale: ~0.01·t0·a/ħ in units; Used in geometric FS sampling weight (hypot(dE_dx, dE_dy)).
_VF_FLOOR_TIGHT       : float = _VF_FLOOR * 1e-1    # = 1e-5 : tighter v_F floor in the 1/vF integration kernel (dl/vF arc-length weight); must be  _VF_FLOOR so it never dominates the physical weight;
_VF_FLOOR_REL_FRAC    : float = 0.05                # Relative v_F floor: max(_VF_FLOOR_TIGHT, this x median(|vF| over the FS)).
_VERTEX_DIAG_MIN_FS   : int   = 10                  # minimum FS points required for V_mat structure diagnostics; below this std/mean statistics are unreliable.
_NODAL_REGION_PCTL    : int   = 25                  # lower and upper 25% percentile for node gap estimation
_Q_UNIQUE_SCALE       : int   = 100000              # integer scaling factor for unique q pairs
_BZ_NORM              : float = (2.0 * np.pi) ** 2  # BZ area in reduced coordinates (a=1, ħ=1): area = (2π)², FS arc-length integration measure is dl/((2π)²·vF)

# ── Tc diagnostics ────────────────────────────────────────────────────────────
_BCS_RATIO_STRONG     : float = 3.8                 # 2Δ/kTc > this → strong coupling
_BCS_RATIO_VSTRONG    : float = 5.0                 # 2Δ/kTc > this → very-strong coupling
_BCS_RATIO_EXOTIC     : float = 7.0                 # 2Δ/kTc > this → exotic / non-phononic
_MAD_DENOM            : float = 1.13                # Allen–Dynes strong-coupling corrections give denom≈1.45 for λ~1 spin-fluctuation spectra; Millis–Monien–Pines quote denom≈1.13 for the SF model.
_MAD_NUM              : float = 1.04                # exponent prefactor (weak-coupling limit, spectrum-independent)
_GL_MIN_PTS           : int   = 2                   # minimum stable SC points required for GL fit
_GL_MAX_PTS           : int   = 4                   # upper window of recent stable points used in GL regression
_GL_DELTA_MIN         : float = 2e-3                # |Δ| floor for GL fit points (below this: numerically unreliable)
_GL_TC_MARGIN         : float = 0.05                # max relative deviation |Tc_GL−T_spinodal|/T_max to accept GL result

# ── Orbital / BdG basis size ──────────────────────────────────────────────────
_N_ORB                : int   = 6                   # orbital flavors [Γ6↑,Γ6↓,Γ7a↑,Γ7a↓,Γ7b↑,Γ7b↓]: full 3-Kramers-doublet CF+SOC manifold, no downfolding.
_N_BDG                : int   = 4 * _N_ORB          # 24: BdG dimension = 2 sublattices (A,B) × particle/hole × N_ORB.

# ── Cluster (local irreducible vertex) ────────────────────────────────────────
_N_CHANNELS           : int   = 3                   # Channel-wise (Γ6, Γ7a, Γ7b) order parameter resolution
_CLUSTER_SIZE         : int   = 4                   # 2x2 plaquette
_N_CLUSTER            : int   = _N_ORB ** _CLUSTER_SIZE

# ── Lindhard / χ₀ kernel ─────────────────────────────────────────────────────
_MATH_EPS             : float = 1e-9                # general protection against division by zero
_LINDHARD_CHUNK       : int   = 128                 # k-point batch size in oe.contract loops (memory vs speed)
_FD_MASK_DF           : float = 1e-12               # |Δf| mask threshold in χ₀ Lehmann sums
_FD_MASK_DE           : float = 1e-6                # |ΔE| mask threshold in χ₀ Lehmann sums
_FD_MASK_DE8          : float = 1e-8                # tighter |ΔE| mask for d²F/dM² off-diagonal
_FS_CACHE_TOL         : float = 1e-3                # parameter-change threshold for FS-points cache invalidation
_ETA_T_FRAC           : float = 0.10                # normal-state η = _ETA_T_FRAC · kT  (thermal broadening)
_ETA_DELTA_FRAC       : float = 0.02                # anomalous η += _ETA_DELTA_FRAC · |Δ|  (gap-scale broadening)
_ETA_GRID_FLOOR       : float = 0.001               # η ≥ _ETA_GRID_FLOOR · t0  (k-grid aliasing floor; ~bandwidth/N_k²)

# ── RPA / Moriya spin-fluctuation damping ─────────────────────────────────────
_MORIYA_LANDAU_M_STEP : float = 0.06                # M-probe step for the model-derived (Landau a,b) Gamma_M estimate
_RPA_BW_FACTOR        : float = 8.0                 # Bandwidth = 8·t in 2D tight-binding (square lattice, nearest-neighbour only).
_RPA_V_CAP_ALPHA      : float = 2.2                 # Perturbative RPA breaks down when V_pair ~ O(bandwidth); V_cap = α·max(8·max(|tx|,|ty|), J_eff). 2.2× headroom above the BEC-BCS crossover energy while preventing runaway at the AFM QCP
_DK_CORR_CAP_MULT     : float = 1.0                 # direct cap on |dK_corr| relative to K_bare -- capping rqq alone does not bound dK_corr/K_bare (verified: reached -6.7x with rqq still under its own cap); a "correction" several times larger than the bare stiffness is not perturbative
_RPA_DET_WARN         : float = 0.11                # QCP proximity warning threshold for diagnostics and SCF adaptive mixing.
_RPA_QCP_PENALTY      : float = 0.40                # α reduction per unit |det_afm|<0 past QCP (used in SCF near-critical detection, BO near_qcp flag).
_DET_AFM_FLOOR        : float = 0.5                 # default det_afm when vertex cache is absent (normal state, no QCP)
_DET_SIGN_FLIP_SCALE  : float = 0.05                # |det_afm| scale for V_d sign-flip EMA suppression (determines the sigmoid midpoint)
_EMA_SIGN_FLIP_W_MIN  : float = 0.20                # minimum w_factor on V_d sign flip; preserves adaptation even at det≈0
_EMA_SIGN_FLIP_SLOPE  : float = 6.0                 # sigmoid steepness in sign-flip EMA: w=w_min+(1-w_min)/[1+exp(-k·(|det|/floor-0.5))]
_VMAT_LOW_VAR_FRAC    : float = 0.10                # std(V)/|mean(V)| < this → vertex low-variance flag
_V_PREV_SIGN_FLOOR    : float = 1e-6                # |V_d_prev| below this → treat as zero, skip sign-flip check
_DET_DEPTH_CAP        : float = 5.0                 # max det_depth in jump-cap exponential suppression
_DET_JUMP_HALF_SCALE  : float = 0.5                 # exp(−this·det_depth) decay rate for gap jump cap past QCP
_JUMP_CAP_FLOOR       : float = 1.05                # minimum effective_jump_cap (prevents total freeze past QCP)
_QQ_DELTA_THRESH      : float = 1e-8                # |Δ| threshold below which χ_SQ enforced zero

# ── JT / SC-triggered Jahn–Teller ──────────────────────────────────────────────
_LAMBDA_JT_VIABLE     : float = 0.05                # Minimum λ_JT_sc = g²·|χ_τ_sc|/K_eff for SC-triggered JT viability. Sets upper K_SC = g²·|χ_τ_sc|/0.05 bound on K_lattice
_JT_ACT_THR           : float = 0.04                # The threshold of Γ₆–Γ₇ mixing induced by SC condensate
_DQ_FS_VERTEX         : float = 0.03                # Å — MINIMUM finite-difference step for ∂λ/∂Q on FS; adaptiv mode: h = max(_DQ_FS_VERTEX, _DQ_FS_VERTEX_FRAC*|Q|).
_DQ_FS_VERTEX_FRAC    : float = 0.05                # Å/Å — adaptive step fraction: h = max(_DQ_FS_VERTEX, this*|Q|); near QCP where |Q|->0, falls back to minimum; large Q -> 5% gives noise protection.
_JT_FD_H2_BASE        : float = 3e-8                # Q-derivative finite-difference step, h(Q) = sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF·Q²)
_JT_FD_H2_QCOEF       : float = 6e-7

# ── Saddle‑escape / Jacobi‑kick ─────────────────────────────────────────────────
_MODE_FRAC_DOMINANT   : float = 0.60                # fX > this → mode dominated by X (pure-SC, pure-JT, pure-AFM)
_MODE_FRAC_MIXED      : float = 0.30                # fX > this (both Δ and Q) → SC-triggered-JT mode
_MODE_PULL_FRAC       : float = 0.30                # fraction of (M − M_phys_est) used as kick pull in pure-SC/SC-JT mode
_KICK_BASE_FRACTION   : float = 0.05                # "Trust Region" box specifies the step size as a percentage
_KICK_M_EXCESS_CTR    : float = 0.70                
_KICK_JCHI_EXCESS_CTR : float = 0.70
_KICK_REDUCTION_AMP   : float = 3.88                # amplitude of M-kick reduction: M_kick × (1 − this × excess)
_KICK_BOOST_Q         : float = 0.01                # Q-kick boost
_KICK_M_CLIP_LO       : float = 0.02                # hard lower clip on M_kick (normal SCF path)
_KICK_M_CLIP_HI       : float = 0.9                 # hard upper clip on M_kick
_KICK_DELTA_MAX_FRAC  : float = 0.4                 # maximum allowed seed gap as a fraction of the effective hopping scale t_eff.
_KICK_MIXING_FLOOR    : float = 0.004               # minimum mixing weight in the kick; prevents α from collapsing to zero when λ_plus is huge.
_KICK_MIXING_SCALE    : float = 4.0                 # damping scale for λ_plus in α = _MIXING / (1 + scale·log1p(λ_plus)).
_M0_WARMSTART_MIN     : float = 0.1                 # |M| below this is treated as "no real information" (crude/near-zero seed), not a genuine converged warm start
_EARLY_KICK_BASE      : float = 0.01                # base step fraction in the coupled space

# ── SCF iteration / mixing adaptive control ─────────────────────────────────────
_MAX_ITER             : int   = 700
_MIN_ITER             : int   = 4
_MIXING               : float = 0.06                # base weight of the newly computed residual in the solution update; lower values improve stability at the cost of slower convergence.
_ALPHA_HF             : float = 0.31                # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton)
_Q_UPDATE_PERIOD      : int   = 3                   # update Q every N inner iterations
_Q_THR_REL            : float = 0.016               # fraction of lambda_hop; Q change below this skips vertex rebuild
_Q_SEED_THR           : float = 1e-4                # if initial_Q is already nonzero, trust it as the best current estimate.
_M_THR_REL            : float = 0.01                # absolute M change threshold
_EMA_NEW_WEIGHT       : float = 0.14                # EMA weight for V_d, Λ_inst
_SCF_DIVERGE_RATIO    : float = 1.05                # max_diff > prev × this → SCF classified as diverging
_SCF_STAGNATE_RATIO   : float = 0.95                # max_diff > prev × this (and not diverging) → SCF stagnating
_SCF_ALPHA_DECAY      : float = 0.95                # α *= this when SC+JT active and converging (mild damping)
_SCF_ALPHA_RECOVER    : float = 1.60                # α *= this on freeze-recovery (restores mobility after stagnation)
_SCF_FREEZE_THR       : int   = 10                  # α_freeze_count ≥ this triggers freeze-recovery boost
_SCF_ALPHA_FREEZE_LO  : float = 0.15                # α < _MIXING × this → too frozen, trigger recovery
_SCF_ALPHA_FREEZE_HI  : float = 0.60                # α recovery ceiling: min(α×_RECOVER, _MIXING×this)
_SCF_ALPHA_CONVG_BOOST: float = 1.15                # α boosted (× this) when SC+JT active and converging (mild)
_SCF_ALPHA_CONVG_CAP  : float = 0.75                # α ceiling (× _MIXING) during SC+JT active converging branch

# ── Limit-cycle detector ────────────────────────────────────────────────────────
_CYCLE_WINDOW         : int   = 20                  # iteration window to detect oscillation
_CYCLE_THRESHOLD      : float = 0.25                # std/mean of |Δ| above this → oscillatory regime
_CYCLE_DAMP_FAC       : float = 0.45                # alpha reduction factor when oscillation detected

# ── M Newton solver step control ────────────────────────────────────────────────
_M_STEP_FLOOR_REL     : float = 0.005               # _step_floor = max(_M_STEP_FLOOR_REL × |M|, _M_STEP_FLOOR_ABS)
_M_STEP_FLOOR_ABS     : float = 0.002               # absolute minimum M step (eV·site) regardless of |M|
_M_STEP_FLOOR_M_MIN   : float = 0.010               # reference M scale in step floor: max(|M|, this)
_M_J_EFF_FLOOR_FRAC   : float = 0.20                # j_eff_floor = max(|J_eff|, this × t_eff, ε) — QCP guard preventing ΔM∝1/J_eff→∞
_MU_LM                : float = 3.0                 # Levenberg–Marquardt floor for M Newton step, larger → smaller γ_M → more conservative M update.
_TR_M_STEP_MAX        : float = 0.1                 # Upper bound on |ΔM| (eV). Reduced when J_eff/t_eff is large (stiff landscape) or near QCP (flat curvature).
_TR_M_STEP_MIN_FLOOR  : float = 1e-3                # absolute minimum step — prevents total freeze near M→0

# ── Q Newton solver step control (mirrors the M block above) ──────────────────
_Q_LM_FRAC            : float = 0.08                # Q-channel LM floor as a *fraction of self._K_bare*, not an absolute eV/Å² constant
_TR_Q_STEP_FRAC       : float = 0.10                # Upper bound on |Q_out_raw - Q| per iteration, as a fraction of lambda_hop.
_TR_Q_STEP_MIN_FLOOR  : float = 1e-4                # absolute minimum Q step (Å) — prevents total freeze near the JT QCP.

# ── Anderson mixing ───────────────────────────────────────────────────────────
_ANDERSON_TIKHONOV    : float = 1e-8                # Tikhonov β / diag_max in Anderson normal equations
_ANDERSON_TRUST       : float = 2.4                 # trust-region step-size limit (multiples of simple step)
_ANDERSON_W_LO        : float = 0.3                 # lower blend weight between Anderson and simple mixing
_ANDERSON_W_HI        : float = 0.8                 # upper blend weight

# ── Orbital-character / coherence-length thresholds ──────────────────────────
_XI_NODAL_MIN         : float = 2.0                 # ξ/a > this required for coherent nodal quasiparticles (BCS side)
_ORBITAL_SEL_FRAC     : float = 0.15                # |ξ_Γ₆ − ξ_Γ₇|/ξ > this → system classified as orbitally selective
_GL_SPINODAL_JUMP     : float = 0.15                # D_spinodal/Δ₀ < this → GL extrapolation considered reliable (small first-order jump)

# ── IC (inter-channel) correction ratio bounds ────────────────────────────────
_IC_RATIO_FLOOR       : float = 1.05                # r < this → negligible IC, no suppression applied
_IC_RATIO_CAP         : float = 3.00                # r > this → very strong IC, cap reduction to avoid too small M

# ── Thermodynamics / μ-solver ─────────────────────────────────────────────────
_FERMI_ARG_CLIP       : float = 100.0               # clip argument of exp() in Fermi function
_DEN_DERIV_FLOOR      : float = 1e-12               # ∂n/∂μ floor in Newton μ-finder
_BRENTQ_TOL           : float = 1e-5                # brentq μ-bracketing tolerance
_MU_NEWTON_MAXIT      : int   = 20                  # Newton/backtrack iteration budget before the guaranteed Brent finish
_MU_DENSITY_TOL       : float = 1e-6                # |n(μ)−target_n| convergence tolerance
_MU_SC_DERIV_THRESH   : float = 1e-4                # eV; |Δ_s|+|Δ_d| above which the analytic dn/dμ (exact only for pure-particle/hole BdG branches, i.e. Δ=0) is replaced by a centred numeric derivative
_MU_BACKTRACK_MAX     : int   = 6                   # max step-halvings per Newton iteration before declaring the direction unreliable
_MU_BACKTRACK_FLOOR   : float = 0.05                # minimum backtracking damping factor η before giving up on the current Newton direction

# ── Gap (Δ) update ────────────────────────────────────────────────────────────
_ALPHA_MIX_2X2        : float = 0.56                # blend weight of 2×2 eigenvector direction vs fixed-point gap update, 0 = pure fixed-point; 1 = pure 2×2 eigenvector
_BCS_SEED_FRACTION    : float = 0.1                 # BCS seed magnitude as fraction of t_eff
_DELTA_JUMP_CAP       : float = 5.0                 # max |Δ_new| / |Δ_current| ratio per iteration
_DELTA_ABS_FLOOR      : float = 1e-4                # eV — absolute |Δ| below which the jump limiter is bypassed (seed-gap free-growth phase). floor is ~0.7–2% of t0
_PHI_D_FLOOR          : float = 1e-3                # minimum φ_d max value to enable nodal/antinodal decomposition
_KERNEL_DIR_MIN_FRAC  : float = 0.5                 # when the fixed-point gap amplitude is below this fraction of the 2×2 kernel eigenvector amplitude, the 2×2 eigenvector direction is allowed to dominate the mixing

# ── Δ Newton solver step control (mirrors the M/Q Newton blocks above) ────────
_MU_LM_DELTA          : float = 3.0                 # Levenberg–Marquardt floor for the Δ Newton step (2×2 analogue of _MU_LM)
_ALPHA_HF_DELTA       : float = 0.20                # Newton vs BdG-fixpoint blend for Δ update (0=fixpoint, 1=Newton); more conservative than _ALPHA_HF (0.31) since Δ has two coupled channels and is already jump-capped downstream
_TR_DELTA_STEP_MAX    : float = 0.1                 # Upper bound on |Δ_newton_step| per channel per iteration (eV, same scale as _TR_M_STEP_MAX)

# ── Physical thresholds (SC viability / Mott) ─────────────────────────────────
_G_T_COHERENCE_MIN    : float = 0.10                # g_t floor for coherent ZRS band (δ<0.053 → incoherent FS). Used in: _scf_jacobi_kick, Mott filter, _score, __main__ doping floor.
_JCHI_HARD_REJECT     : float = 2.0                 # J·χ_SS > this → score = 0 (deeply AFM, SC impossible)
_V_CUT                : float = 20.0                # pairing vertex near-divergence detector threshold
_MBZ_DEGEN_TOL        : float = 1e-8                # eV; Energy-based tie-break, but only when the split is resolved above numerical noise.

_CHANNEL_ORB_IDX      : tuple = (
    (0, 1), (2, 3), (4, 5)
    )                                               # orbital indices per channel in the 6-dim basis
_NORMAL_SECTOR_PAIRS  : tuple = (
    (slice(0,  6),  slice(0,  6)),  # A-A particle
    (slice(6,  12), slice(6,  12)), # B-B particle
    (slice(12, 18), slice(12, 18)), # A-A hole
    (slice(18, 24), slice(18, 24)), # B-B hole
    (slice(0,  6),  slice(6,  12)), # A-B particle
    (slice(6,  12), slice(0,  6)),  # B-A particle
    (slice(12, 18), slice(18, 24)), # A-B hole
    (slice(18, 24), slice(12, 18)), # B-A hole
)                                                  # Nambu sector pairs for normal-state (Δ=0) Lindhard sum. Excludes anomalous Part↔Hole pairs, which vanish at Δ=0.

def _scf_log(tag: str, msg: str, verbose: bool = True) -> None:
    """Thread-safe logger.  tag is left-padded to 18 chars so columns stay aligned."""
    if not verbose:
        return
    with _log_lock:
        print(f"[{tag:<18s}] {msg}", flush=True)

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
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I3 = np.eye(3, dtype=complex)

    H_SOC = lambda_soc * (np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz))
    Lx_t2g = np.kron(Lx, I2)
    Ly_t2g = np.kron(Ly, I2)
    Lz_t2g = np.kron(Lz, I2)
    H_CF = (Delta_tetra * (Lz_t2g @ Lz_t2g) + Delta_inplane * (Lx_t2g @ Lx_t2g - Ly_t2g @ Ly_t2g))
    LS_op = np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)
    return H_SOC + H_CF, Lx_t2g, Ly_t2g, Lz_t2g, np.kron(I3, Sz), LS_op

def _find_kramers_doublets(evals: np.ndarray, evecs_soc: np.ndarray, Sz_full: np.ndarray, LS_op: np.ndarray):
    """
    Identify the three Kramers doublets and their z-polarised Kramers partners by diagonalising Sz within each 2D subspace.
    The resulting doublets are sorted by <L·S>.
    """
    doublets = []
    for i in (0, 2, 4):
        v = evecs_soc[:, i]
        vp = evecs_soc[:, i + 1]
        U = np.column_stack((v, vp))
        sz_vals, sz_vecs = np.linalg.eigh(U.conj().T @ Sz_full @ U)
        dn = U @ sz_vecs[:, 0]
        up = U @ sz_vecs[:, 1]
        doublets.append({
            'idx': i,
            'energy': float(evals[i]),
            'ls_val': float(np.real(v.conj() @ LS_op @ v)),
            'v': v,
            'v_p': vp,
            'up': up,
            'dn': dn,
            'sz_up': float(np.real(sz_vals[1])),
            'sz_dn': float(np.real(sz_vals[0])),
        })
    doublets.sort(key=lambda d: d['ls_val'])
    G6 = doublets[0]   # Γ6: most negative <L·S> → j_eff = 1/2
    G7_candidates = doublets[1:]   # two Γ7 candidates
    G7_candidates.sort(key=lambda x: (x['energy'], -abs(x['sz_up'])))
    G7a, G7b = G7_candidates[0], G7_candidates[1]
    return G6, G7a, G7b

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
    return ec[:, 0:6, :], ec[:, 6:12, :], ec[:, 12:18, :], ec[:, 18:24, :]

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

def _lindhard_bubble(sector_pairs: tuple, E_k_all: np.ndarray, V_k_all: np.ndarray, f_k_all: np.ndarray, shift_idx: np.ndarray, w: np.ndarray, vw_sq: np.ndarray, eta: float, kT: float) -> np.ndarray:
    """Static (ω=0) full-BZ Lindhard bubble for a given q-shift and set of sector pairs. Uses uniform BZ weights w throughout."""
    E_kQ = E_k_all[shift_idx]
    V_kQ = V_k_all[shift_idx]
    f_kQ = _fermi_function(E_kQ, kT)

    kernel = _lehmann_kernel(f_k_all, f_kQ, E_k_all, E_kQ, eta, kT)
    # Apply k-weighting
    kernel = kernel * w[:, None, None] * vw_sq[:, None, None]

    N   = E_k_all.shape[0]
    chi = np.zeros((_N_ORB, _N_ORB), dtype=complex)
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
    return 0.125 * (chi + chi.conj().T)   # 0.5 × (two sublattices A/B) 0.5 (Nambu particle-hole doubling) × 0.5 (hermitisation of chi[a,b]↔chi[b,a])

# Helper: apply h_i^dag h_j to a two-hole occupation-number state; bits is an integer mask of the occupied spin-orbitals.
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
    # ... and adding i.
    sign *= (-1) ** bin(bits_after_ann & ((1 << i) - 1)).count("1")
    return bits_after_ann | (1 << i), sign

def _build_block(basis, t_pd, Delta_CT, U_dd, U_pp):
    """
    Build the Hamiltonian block for a given two-hole basis.
    basis : list of sorted tuples (spinorb1, spinorb2)
    """
    dim = len(basis)
    H = np.zeros((dim, dim))

    for idx, (p, q) in enumerate(basis):
        bits = (1 << p) | (1 << q)

        # Local energy: ligand p (orbital index 1) is shifted by Delta_CT.
        for sp in (p, q):
            if sp % 3 == 1:
                H[idx, idx] += Delta_CT

        # On-site Coulomb: both holes in the same orbital (necessarily opposite spins).
        if p % 3 == q % 3:
            orb = p % 3
            H[idx, idx] += U_pp if orb == 1 else U_dd

        # Hopping along the 0-1-2 chain (spin conserved).
        for source in (p, q):
            source_orb = source % 3
            spin = source // 3

            targets = []
            if source_orb == 0:
                targets.append(1)
            elif source_orb == 1:
                targets.extend([0, 2])
            elif source_orb == 2:
                targets.append(1)

            for target_orb in targets:
                target = target_orb + 3 * spin
                res = _apply_hop(bits, target, source)
                if res is not None:
                    new_bits, sign = res
                    new_sp = sorted([i for i in range(6) if (new_bits >> i) & 1])
                    new_idx = basis.index(tuple(new_sp))
                    H[idx, new_idx] += t_pd * sign
    return H

def _kappa_superexchange(t_pd, Delta_CT, U_dd, U_pp):
    """Correction factor for the bare ZSA superexchange prefactor J_pdct."""
    # Spin-up orbital indices: 0,1,2 ; spin-down: 3,4,5.
    up_orbs = [0, 1, 2]
    down_orbs = [3, 4, 5]

    # S_z = +1 sector (both holes spin up): choose 2 from 3 up spin-orbitals.
    basis_up = list(itertools.combinations(up_orbs, 2))
    H_T = _build_block(basis_up, t_pd, Delta_CT, U_dd, U_pp)
    E_T = float(np.linalg.eigvalsh(H_T)[0])

    # S_z = 0 sector (one up, one down): 3 x 3 = 9 states.
    basis_sz0 = [(u, d) for u in up_orbs for d in down_orbs]
    H_S0 = _build_block(basis_sz0, t_pd, Delta_CT, U_dd, U_pp)
    E_S0 = float(np.linalg.eigvalsh(H_S0)[0])
    # Exact superexchange for a half-filled 3-orbital d-p-d cluster with two holes.
    return (E_T - E_S0) / 2.0

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

    # Map to the first quadrant using component-wise absolute values.
    # This step combines:
    #   - inversion symmetry (q → −q)
    #   - two-fold rotations / mirrors combined with time reversal
    #     (σ_x·T : q_x → −q_x,  σ_y·T : q_y → −q_y) ⇒ χ(q) = χ(|q_x|, |q_y|)
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

def _channel_J3(J_A1g_diag: np.ndarray) -> np.ndarray:
    """It extracts the 3 independent channels from the 6-component (Kramers-doubled) J_A1g_diag."""
    return np.asarray(J_A1g_diag)[0::2]

def _gamma_splitting(lambda_soc: float, Delta_tetra: float, Delta_inplane: float = 0.0) -> float:
    """Fast standalone Δ_CF = E(Γ7a) − E(Γ6) (eV) for the crystal-field pre-scan only."""
    H_soc_cf, _, _, _, Sz_full, LS_op = _t2g_soc_cf_operators(lambda_soc, Delta_tetra, Delta_inplane)
    evals, evecs_soc = np.linalg.eigh(H_soc_cf)
    G6, G7a, _ = _find_kramers_doublets(evals, evecs_soc, Sz_full, LS_op)
    return float(G7a['energy'] - G6['energy'])

def _scf_result_reliability(res: Optional[dict]) -> Tuple[bool, str]:
    """
    (ok, reason): whether a solve_self_consistent() result is trustworthy
    enough for a free-energy comparison across scenarios -- converged, has a
    Hessian, and that Hessian has no negative eigenvalue
    """
    if not res or not res.get("converged", False):
        return False, "not converged"
    if res.get("hessian_result", {}).get("eigenvalues") is None:
        return False, "no Hessian"
    if not all(e > -1e-6 for e in res["hessian_result"]["eigenvalues"]):
        return False, "saddle (Hessian has negative eigenvalue)"
    return True, "ok"

def _scf_result_hessian_min(res: Optional[dict]) -> float:
    """Smallest Hessian eigenvalue for a solve_self_consistent() result, or NaN if unavailable."""
    if not res:
        return float('nan')
    hess = res.get("hessian_result")
    if hess is None or hess.get("eigenvalues") is None:
        return float('nan')
    return float(np.min(hess["eigenvalues"]))

def _bcs_coupling_regime(ratio_2D: float) -> str:
    """Label 2Δ/kTc with the same thresholds as the Tc₃ diagnostic in __main__."""
    if ratio_2D < _BCS_RATIO_STRONG:
        return 'BCS-like'
    elif ratio_2D < _BCS_RATIO_VSTRONG:
        return 'strong'
    elif ratio_2D < _BCS_RATIO_EXOTIC:
        return 'very-strong'
    return 'exotic / non-phononic'

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
    def instab_dir(self) -> str:
        return {
            'stable':         'stable',
            'spontaneous_JT': f'pure JT (spontaneous risk)  G3[2,2]={self.G22:.4f}≤0',
            's_pairing':      f's pairing  G3[0,0]={self.G11:.4f}≤0',
            'd_pairing':      f'd pairing  G3[1,1]={self.G33:.4f}≤0  ✓ desired',
            'both_pairing':   f's+d pairing  G11={self.G11:.4f}  G33={self.G33:.4f}',
            'cross_channel':  f'SC-triggered JT  λ_min={self.lambda_min:.4f}<0  diagonals +',
        }[self.instab_type]

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

@dataclass
class ModelParams:
    # --- Primary inputs ---
    t_pd:             float      # eV    pd hybridisation integral
    U_dd:             float      # —     Coulomb repulsion in d orbit, typ charge-transfer 10·t0
    lambda_soc:       float      # eV    atomic SOC λ t2g shell; determines Γ₆–Γ₇ splitting
    Delta_tetra:      float      # eV    tetragonal axial CF Δ_tet·Lz²; negative = z-compression
                                 #       Partial cancellation with SOC tunes Γ₆–Γ₇ gap independently of λ
    g_JT:             float      # eV/Å  Jahn–Teller electron–phonon coupling (B1g channel: Q·(Lx²-Ly²))
                                 #       increasing g_JT beyond the SC-triggered window is risky, because spontaneous JT (G3[2,2] < 0) always precedes the RPA cross-channel divergence.
    K_lattice:        float      # eV/Å² bare lattice spring constant, B1g channel (phonon stiffness, no exchange)
                                 #       K_lattice must satisfy: K_spont = g²/Δ_CF < K < g²/(0.05·π·t0) for SC-JT window (λ_JT > 0.05).
    lambda_hop:       float      # Å     hopping decay length for B₁g anisotropy: t(Q) = t0·exp(±Q/λ_hop)
    g_Eg2:            float      # eV/Å  Jahn-Teller electron-phonon coupling, Eg,2 channel (Q_Eg2·(LyLz+LzLy))
                                 #       Eg,2 is local (no hopping-renormalisation term, unlike B1g's t(Q) anisotropy).         
    K_lattice_Eg2:    float      # eV/Å² bare lattice spring constant, Eg,2 channel (independent stiffness from K_lattice)

    # --- Charge-transfer / RPA / gap symmetry ---
    Delta_B1g_static: float      # eV    static B1g (x²-y²) crystal-field splitting, Δ_ip·(Lx²-Ly²); splits Γ₇→Γ₇a+Γ₇b.
    hybrid_scale:     float      #       above 4.0 assumes stronger covalency (the charge carriers would typically reside less than ~80% on the metal ion)
    Upp_ratio_bare:   float      #       the bare (without hybridization) U_pp / U_dd ratio varies between 0.2 and 2.0 for 3d-5d transition metals due to the spatial extent of the ligand orbital
    Delta_CT:         float      # eV    charge-transfer gap (ZSA scale); sets scale for CT-insulator crossover

    # --- Numerics ---
    Z:                int        #       metal's coordination number in 2D square lattice
    kT:               float      #       eV  temperature — keep kT < Tc to allow gap to open;
    tol:              float

    def __post_init__(self):
        # SOC + crystal-field Hamiltonian
        H_soc_cf, _Lx_t2g, _Ly_t2g, _Lz_t2g, _Sz_full, _LS_op = _t2g_soc_cf_operators(self.lambda_soc, self.Delta_tetra, self.Delta_B1g_static)
        evals, evecs_soc = np.linalg.eigh(H_soc_cf)        
        
        G6, G7a, G7b = _find_kramers_doublets(evals, evecs_soc, _Sz_full, _LS_op)

        # Diagnostic cross-check with μz = Lz + 2Sz norm; in the pure-j limit both agree; disagreement flags the mixed CF/SOC regime.
        _mu_z = _Lz_t2g + 2.0 * _Sz_full
        def _kramers_moment(mu_op, up, dn):
            U = np.column_stack((up, dn))
            return float(np.linalg.norm(U.conj().T @ mu_op @ U))
        
        mu7 = [
            _kramers_moment(_mu_z, G7a['up'], G7a['dn']),
            _kramers_moment(_mu_z, G7b['up'], G7b['dn'])
        ]
        if mu7[0] < mu7[1]:
            _scf_log("RMFT-WARN",
                f"⚠ Γ7a: spin_pol and μz DISAGREE "
                f"(Δ_tetra/λ_SOC={self.Delta_tetra/max(self.lambda_soc,1e-12):.2f}). "
                f"|Sz|: {abs(G7a['sz_up']):.4f} vs {abs(G7b['sz_up']):.4f}  "
                f"μz: {mu7[0]:.4f} vs {mu7[1]:.4f}")

        # ── Build the z-polarised basis [Γ6↑, Γ6↓, Γ7a↑, Γ7a↓, Γ7b↑, Γ7b↓] ────
        _v6_up,  _v6_dn  = G6['up'],  G6['dn']
        _v7_up,  _v7_dn  = G7a['up'], G7a['dn']
        _v7b_up, _v7b_dn = G7b['up'], G7b['dn']
        evecs_soc = np.column_stack([_v6_up, _v6_dn, _v7_up, _v7_dn, _v7b_up, _v7b_dn])
        new_order = [G6['idx'], G6['idx']+1, G7a['idx'], G7a['idx']+1, G7b['idx'], G7b['idx']+1]
        evals     = evals[new_order]

        sz6_up, sz6_dn   = G6['sz_up'],  G6['sz_dn']
        sz7_up, sz7_dn   = G7a['sz_up'], G7a['sz_dn']
        sz7b_up, sz7b_dn = G7b['sz_up'], G7b['sz_dn']

        # Sanity-check: eigh on a 2x2 Hermitian op guarantees opposite-sign partners
        if sz6_up * sz6_dn >= 0 or sz7_up * sz7_dn >= 0 or sz7b_up * sz7b_dn >= 0:
            _scf_log("RMFT-WARN",
                f"⚠ Sz diagonalisation returned same-sign partners: "
                f"Γ6: {sz6_up:+.4f},{sz6_dn:+.4f}  Γ7a: {sz7_up:+.4f},{sz7_dn:+.4f}  "
                f"Γ7b: {sz7b_up:+.4f},{sz7b_dn:+.4f}. "
                f"Possible near-degenerate subspace.")

        # Exact <Sz> matrix elements from the subspace diagonalisation
        self.sz_op = np.array([sz6_up, sz6_dn, sz7_up, sz7_dn, sz7b_up, sz7b_dn], dtype=float)

        # Effective multipolar spin operator for the superexchange H_exch = J · kron(multi_op, multi_op).
        P6_diag  = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        P7_diag  = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        P7b_diag = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        sz_diag  = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.multi_op = np.diag(
            (abs(sz6_up) * P6_diag + abs(sz7_up) * P7_diag + abs(sz7b_up) * P7b_diag) * sz_diag
        )

        # ── PHYSICAL GAPS ──────────────────────────────────────────────────────
        self.Delta_CF = float(evals[2] - evals[0])   # Γ7a - Γ6 gap (JT-active)
        self.g7split  = float(evals[4] - evals[2])   # Γ7b - Γ7a splitting

        # B1g operator in the t2g manifold (Lx^2 - Ly^2)
        _B1g_t2g_pi = _Lx_t2g @ _Lx_t2g - _Ly_t2g @ _Ly_t2g
        # B₁g JT phonon operator: U6†·(Lx²−Ly²)_t2g·U6  (6×6 Γ6⊕Γ7a⊕Γ7b subspace, real, hermitian;
        # Γ7b is kept explicitly, so its own B1g matrix elements, however weak, are carried through exactly rather than discarded; at high energy virtual processes it can no longer be neglected).
        #   D₄h (Δ_B1g_static=0): anti-diagonal real (Γ₆↔Γ₇), diagonal = 0.
        #   D₂h (Δ_B1g_static≠0): both real diagonal AND real off-diagonal elements. The diagonal A₁g component partially lifts the normal-state selection rule for χ_SQ even before superconductivity.
        _U6 = evecs_soc[:, 0:6]
        self.B1g_op = np.asarray(np.real(_U6.conj().T @ _B1g_t2g_pi @ _U6), dtype=float)
        self.B1g_offdiag = self.B1g_op - np.diag(np.diag(self.B1g_op))

        # ── Eg,2 operator in the 6×6 Γ6⊕Γ7a⊕Γ7b subspace ──────────────────────
        _Eg2_t2g = _Ly_t2g @ _Lz_t2g + _Lz_t2g @ _Ly_t2g
        self.Eg2_op = np.asarray(_U6.conj().T @ _Eg2_t2g @ _U6, dtype=complex)
        self.Eg2_op = 0.5 * (self.Eg2_op + self.Eg2_op.conj().T)   # enforce exact Hermiticity against roundoff

        # ── orbital character of Γ6 and Γ7a (d-orbital weights) ────────────────
        _u_xy = np.array([0, 1, 0], dtype=complex)
        _u_xz = (-1.0/np.sqrt(2)) * np.array([1, 0, -1], dtype=complex)
        _u_yz = (-1j/np.sqrt(2)) * np.array([1, 0, 1], dtype=complex)
        _I2c = np.eye(2, dtype=complex)
        _P_xz = np.kron(np.outer(_u_xz, _u_xz.conj()), _I2c)
        _P_yz = np.kron(np.outer(_u_yz, _u_yz.conj()), _I2c)
        _P_xy = np.kron(np.outer(_u_xy, _u_xy.conj()), _I2c)

        # Average over both Kramers partners for each doublet
        self._w6_xz = 0.5 * (float(np.real(_v6_up.conj() @ _P_xz @ _v6_up)) + 
                            float(np.real(_v6_dn.conj() @ _P_xz @ _v6_dn)))
        self._w6_yz = 0.5 * (float(np.real(_v6_up.conj() @ _P_yz @ _v6_up)) + 
                            float(np.real(_v6_dn.conj() @ _P_yz @ _v6_dn)))
        self._w6_xy = 0.5 * (float(np.real(_v6_up.conj() @ _P_xy @ _v6_up)) + 
                            float(np.real(_v6_dn.conj() @ _P_xy @ _v6_dn)))
        self._w7_xz = 0.5 * (float(np.real(_v7_up.conj() @ _P_xz @ _v7_up)) + 
                            float(np.real(_v7_dn.conj() @ _P_xz @ _v7_dn)))
        self._w7_yz = 0.5 * (float(np.real(_v7_up.conj() @ _P_yz @ _v7_up)) + 
                            float(np.real(_v7_dn.conj() @ _P_yz @ _v7_dn)))
        self._w7_xy = 0.5 * (float(np.real(_v7_up.conj() @ _P_xy @ _v7_up)) + 
                            float(np.real(_v7_dn.conj() @ _P_xy @ _v7_dn)))
        self._w7b_xz = 0.5 * (float(np.real(_v7b_up.conj() @ _P_xz @ _v7b_up)) +
                             float(np.real(_v7b_dn.conj() @ _P_xz @ _v7b_dn)))
        self._w7b_yz = 0.5 * (float(np.real(_v7b_up.conj() @ _P_yz @ _v7b_up)) +
                             float(np.real(_v7b_dn.conj() @ _P_yz @ _v7b_dn)))
        self._w7b_xy = 0.5 * (float(np.real(_v7b_up.conj() @ _P_xy @ _v7b_up)) +
                             float(np.real(_v7b_dn.conj() @ _P_xy @ _v7b_dn)))

        # Orbital-selective hopping building blocks in the 6×6 Γ6⊕Γ7a⊕Γ7b SUBSPACE
        self.Tx_A_xz = np.asarray(np.real(_U6.conj().T @ _P_xz @ _U6), dtype=float)
        self.Tx_A_yz = np.asarray(np.real(_U6.conj().T @ _P_yz @ _U6), dtype=float)
        self.Tx_A_xy = np.asarray(np.real(_U6.conj().T @ _P_xy @ _U6), dtype=float)
        # Completeness check (P_xz+P_yz+P_xy = 1 on the orbital triplet ⇒ sum = I6)
        _sum_A = self.Tx_A_xz + self.Tx_A_yz + self.Tx_A_xy
        if not np.allclose(_sum_A, np.eye(6), atol=1e-8):
            _scf_log("RMFT-WARN", f"⚠ A_xz+A_yz+A_xy deviates from I6 (max|Δ|={np.max(np.abs(_sum_A-np.eye(6))):.2e})")
        
        # Γ7 (|Lz|=1) orbital character weight in the Γ6 doublet; v6_up/v6_dn are normalized eigenvectors of a 6x6 Hermitian problem, need Γ₆ weight = 1-p_7 < 0.5
        self.p_7 = 0.5 * (np.sum(np.abs(_v6_up[[0,1,4,5]])**2) + np.sum(np.abs(_v6_dn[[0,1,4,5]])**2))
        self.t0  = self.t_pd**2 / self.Delta_CT

        # Ligand coordination number: typically hybridizes with two neighboring metal atoms
        z_O = self.Z / 2.0
        # Weak-hybridization limit: recovers the Wannier-derived form U_pp ≈ r0 * U_dd - alpha * t_pd², where r0 encodes the bare p-d orbital extent mismatch and alpha captures delocalization-driven screening.
        self.U_pp = (self.Upp_ratio_bare * self.U_dd) / (1.0 + z_O * self.t_pd**2 / (self.Delta_CT * (self.Delta_CT + self.U_dd)))

        self.J_pdct = _kappa_superexchange(self.t_pd, self.Delta_CT, self.U_dd, self.U_pp) / self.t0**2

        assert _NK % 2 == 0, f"_NK={_NK} must be even for commensurate q_AFM=(π,π)"
        k_scf          = np.linspace(-np.pi, np.pi, _NK, endpoint=False)
        KX_scf, KY_scf = np.meshgrid(k_scf, k_scf)
        self.k_points  = np.column_stack((KX_scf.flatten(), KY_scf.flatten()))
        self.N_k       = len(self.k_points)
        self.k_weights = _uniform_bz_weights_2d(_NK, _NK)   # uniform 1/N weights (periodic BZ)

        # General shift-index table for ALL q-vectors on the 2π/_NK grid.
        # For q = (nx, ny) * 2π/_NK the k+q grid is a cyclic per,utation of k_even:  E(k+q)[i] = E(k)[shift_table[nx, ny, i]]
        _flat   = np.arange(self.N_k)
        _kx_idx = _flat % _NK
        _ky_idx = _flat // _NK
        _nx = np.arange(_NK)[:, None, None]   # (_NK, 1, 1)
        _ny = np.arange(_NK)[None, :, None]   # (1, _NK, 1)
        self.shift_table = (
            ((_ky_idx[None, None, :] + _ny) % _NK) * _NK
          + ((_kx_idx[None, None, :] + _nx) % _NK)
        ).astype(np.int32)   # (_NK, _NK, N_k)

    def estimate_M0(self, target_doping: float, stoner: float, M_seed: float = None) -> float:
        """Warm-start AFM order-parameter estimate."""        
        M_sc = M_seed if (M_seed is not None and abs(M_seed) > _M0_WARMSTART_MIN) else 0.5
        for _ in range(50):
            M_sc_new = float(np.tanh(stoner * M_sc))
            if abs(M_sc_new - M_sc) < 1e-9:
                break
            M_sc = M_sc_new
        M0 = float(np.clip(M_sc, 0.0, _KICK_M_CLIP_HI))
        return M0
    
    def get_gutzwiller_factors(self, target_doping: float) -> Tuple[float, float, float, float]:
        """
        g_t       = 2δ/(1+δ)         kinetic energy; → 0 at half-filling (Mott insulator)
        g_J       = 4/(1+δ)²         exchange enhancement; → 4 at half-filling (J = 4t²/U)
        g_Delta_s = g_t              on-site inter-orbital Γ₆⊗Γ₇ singlet (kinetic origin)
        g_Delta_d                    inter-site d-wave B₁g renormalisation.

        For the d-channel: the B₁g pairing channel arises from virtual Γ₆↔Γ₇ transitions via
        the superexchange tensor J_B1g = (J_CT/2)·sinh(2Q/λ)·η·τ_x. The renormalisation interpolates
        between Γ₇ decoupled (Δ_CF→∞ → g_Delta_d → g_t) and Γ₆–Γ₇ degenerate (Δ_CF→0 → g_Delta_d → g_J) 
        """
        abs_delta = max(abs(target_doping), 1e-6)
        g_t       = (_GW_G_T_NUMERATOR * abs_delta) / (1.0 + abs_delta)
        g_J       = _GW_G_J_PREFACTOR / ((1.0 + abs_delta) ** 2)
        g_Delta_s = g_t   # s-channel is kinetic in origin → follows g_t

        # Interpolation: the d-wave Gutzwiller factor depends solely on how much the lowest energy Γ6 state itself loses its pure 1/2 character
        g_Delta_d = g_t + (g_J - g_t) * self.p_7
        return g_t, g_J, g_Delta_s, g_Delta_d
    
    def exchange_channels(self, Q: float, n_kspace: float, tx_bare: float, ty_bare: float, g_J: float) -> Tuple[np.ndarray, float]:
        """
        Q-dependent multipolar exchange in the full [Γ₆↑, Γ₆↓, Γ₇ₐ↑, Γ₇ₐ↓, Γ₇ᵦ↑, Γ₇ᵦ↓] basis (no downfolding).
            D₄h decomposition: J(Q) = J_A1g(Q)·diag(1,1,η_J7a²,η_J7a²,η_J7b²,η_J7b²) + J_B1g(Q)·B1g_op
        
        Due to hole doping:
            n_kspace → 1 → J_eff → 4J_bare (Mott insulator)
            n_kspace → 0 → J_eff → 0 (empty band)

        No additional ZRS coherence factor: which is already contained in the effective hopping renormalization J is now of order t0² (4th order superexchange),
        which already includes the suppression of ligand hybridization
        """
        _jxz  = np.exp(+Q / self.lambda_hop) ** 2
        _jyz  = np.exp(-Q / self.lambda_hop) ** 2
        _jxy  = 0.5 * (_jxz + _jyz)
        _J6   = self._w6_xz  * _jxz + self._w6_yz  * _jyz + self._w6_xy  * _jxy
        _J7a  = self._w7_xz  * _jxz + self._w7_yz  * _jyz + self._w7_xy  * _jxy
        _J7b  = self._w7b_xz * _jxz + self._w7b_yz * _jyz + self._w7b_xy * _jxy
        eta_J7a = float(np.sqrt(max(_J7a / max(_J6, 1e-9), 0.0)))  # orbital-weight ratio, Γ7a
        eta_J7b = float(np.sqrt(max(_J7b / max(_J6, 1e-9), 0.0)))  # orbital-weight ratio, Γ7b

        # free d-site probability (1−δ): fraction of singly occupied sites; suppresses J_eff towards the overdoped limit.
        J_A1g_diag = n_kspace * g_J * self.J_pdct * (tx_bare**2 + ty_bare**2) * np.array([1.0, 1.0, eta_J7a**2, eta_J7a**2, eta_J7b**2, eta_J7b**2]) # single-bond ZSA superexchange: even, longitudinal scale
        J_B1g_scalar = n_kspace * g_J * self.J_pdct * (tx_bare**2 - ty_bare**2) * np.sqrt(eta_J7a**2 + eta_J7b**2)  # odd, transverse scale
        return J_A1g_diag, J_B1g_scalar

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
        Rigorous orbital-selective 6×6 inter-sublattice hopping matrices in the [Γ6↑,Γ6↓,Γ7a↑,Γ7a↓,Γ7b↑,Γ7b↓] basis

            T_x(Q) = t_x(Q)·A_xz + t_y(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy
            T_y(Q) = t_y(Q)·A_xz + t_x(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy   (x↔y swap)
        Returns (Tx_op, Ty_op), each a real symmetric (6,6) ndarray. Γ7b hops through the SAME
        rigorous orbital projection as Γ6/Γ7a, so it disperses as a genuine band rather than sitting flat.
        """
        tx_b, ty_b = self.effective_hopping_anisotropic(Q)
        t_avg = 0.5 * (tx_b + ty_b)
        Tx_op = tx_b * self.Tx_A_xz + ty_b * self.Tx_A_yz + t_avg * self.Tx_A_xy
        Ty_op = ty_b * self.Tx_A_xz + tx_b * self.Tx_A_yz + t_avg * self.Tx_A_xy
        return Tx_op, Ty_op

    def hopping_matrices_dQ(self, Q: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact analytic ∂T_x/∂Q, ∂T_y/∂Q (companion to hopping_matrices()), using
        d t_x/dQ = +t_x/λ_hop, d t_y/dQ = -t_y/λ_hop.
        """
        tx_b, ty_b = self.effective_hopping_anisotropic(Q)
        dtx = tx_b / self.lambda_hop
        dty = -ty_b / self.lambda_hop
        dt_avg = 0.5 * (dtx + dty)
        dTx_op = dtx * self.Tx_A_xz + dty * self.Tx_A_yz + dt_avg * self.Tx_A_xy
        dTy_op = dty * self.Tx_A_xz + dtx * self.Tx_A_yz + dt_avg * self.Tx_A_xy
        return dTx_op, dTy_op

    def wave_function_weight(self, tx_b: float, ty_b: float, kx: float, ky: float) -> np.ndarray:
        """
        Downfolding factor from the multiband (metal dd + ligand pp) system to the Γ₆–Γ₇ subspace, measuring quasiparticle localization on the central ion versus the ligands;
        scaling from 2nd order perturbation theory
        """
        t_avg = 0.5 * (tx_b + ty_b)
        dt    = 0.5 * (tx_b - ty_b)
        Ag_part  = np.cos(kx) + np.cos(ky)
        B1g_part = np.cos(kx) - np.cos(ky)
        x = 1.0 - self.hybrid_scale * (t_avg * Ag_part + dt * B1g_part) / self.Delta_CT
        k_sigmoid = 10.0   # steepness
        x0 = 0.5           # centre of the transition
        return 1.0 / (1.0 + np.exp(-k_sigmoid * (x - x0)))

@dataclass
class _SolveState:
    """
    Mutable SCF-run-local state passed explicitly into compute_gap_eq_vectorized.
    """
    V_d_ema: Optional[float] = None   # sign-flip EMA state for V_d_scalar
    _ema_kick_pending: bool  = False  # True for one iter after a kick: doubles blend weight so EMA adapts faster

class RMFT_Solver:

    def __init__(self, params: ModelParams):
        self.p = params

        self.kT             = max(params.kT, _MATH_EPS)
        self.multi_op       = params.multi_op
        self.sz_op          = params.sz_op
        self.k_points       = params.k_points
        self.N_k            = params.N_k
        self.k_weights      = params.k_weights
        self.shift_table    = params.shift_table           # (nk, nk, N_k) int32 — cyclic shift index
        self.g_JT_bare      = params.g_JT                  # keeping it strictly bare avoids double-counting local-correlation

        # Orbital operators derived from the SOC+CF eigenbasis.
        self._rebuild_orbital_operators()
        self.phi_k = (np.cos(self.k_points[:, 0])
                      - np.cos(self.k_points[:, 1]))
        
        self._reset_transient_state()
        self._K_bare: float = params.K_lattice # immutable bare lattice spring constant (eV/Å²)

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

        # ── Nambu vertex matrices for the unified χ_SQ susceptibility ────────────
        # lift physical operators into the full 24×24 Nambu space as  M_O = block_diag(O_6, O_6, -O_6^T, -O_6^T)
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
            vbdg._build_H_stack(vbdg._kpts, M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=vbdg._H_stack)
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

    def estimate_gutzwiller_factors_occupation_based(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float) -> Tuple[float, float]:
        """
        Physical per-orbital occupations <c^dag_alpha c_alpha> for alpha = (Γ₆↑, Γ₆↓, Γ₇ₐ↑, Γ₇ₐ↓, Γ₇ᵦ↑, Γ₇ᵦ↓),
        evaluated on sublattice A of the normal-state (Δ=0) BdG Hamiltonian at (M, Q, mu, g_t, g_J).

        - g_t and g_J are fixed by the macroscopic hole doping x, following the exact large-U single-band Gutzwiller limit.
        - The Luttinger/hole-counting constraint x = 1 − Σ_α n_α now sums over ALL SIX states (Γ6, Γ7a, AND Γ7b)
        - The orbital character weight p_7 is updated dynamically from the Γ₆/Γ₇ₐ occupations (the pairing-active
          channel) and used to interpolate the d-wave pairing vertex (g_Δd), while the s-wave vertex stays at g_t.
        """
        ev, ec = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self._get_vbdg())
        dens = self._compute_orbital_densities(ev, ec)
        n6  = dens[0] + dens[1]
        n7  = dens[2] + dens[3]
        n7b = dens[4] + dens[5]
        # macroscopic hole concentration (Luttinger constraint)
        x = float(np.clip(1.0 - float(np.sum(dens)), 1e-4, 1.0))

        # macroscopic Mott limit (strictly kept!)
        g_t = float(np.clip((_GW_G_T_NUMERATOR * x) / (1.0 + x), 0.01, 1.0))
        # superexchange is a virtual process that is not weakened by forbidden double loadings, but rather strengthened.
        g_J = float(np.clip(_GW_G_J_PREFACTOR / ((1.0 + x) ** 2), 0.01, 4.0))

        # dynamic p_7 weight from orbital coherence
        #    If the system is purely single-band (n7=0), p_7 = 0.
        #    If both manifolds are active, p_7 grows and activates the inter-orbital vertex enhancement.
        n_total_active = max(n6 + n7, 1e-6)
        p_7_dynamic = np.clip(2.0 * n7 / n_total_active, 0.0, 1.0)

        # pairing vertices using the original, robust interpolation
        g_Delta_s = g_t
        g_Delta_d = g_t + (g_J - g_t) * p_7_dynamic
        return g_t, g_J, g_Delta_s, g_Delta_d

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

    def _shallow_clone(self) -> 'RMFT_Solver':
        """
        Shallow-copy self and self.p (params), so the clone's ModelParams can be mutated independently of the original solver, without touching
        anything else. The caller is responsible for the appropriate follow-up: _reset_transient_state() alone if nothing Hamiltonian-
        affecting changed if a ModelParams field that feeds the Hamiltonian was changed
        """
        s = copy.copy(self)
        s.p = copy.copy(self.p)
        return s

    def _clone_solver_at_T(self, T: float) -> 'RMFT_Solver':
        """
        Return a fully independent solver clone with kT = T.

        Performs a shallow copy of self and self.p, then resets all mutable
        per-solve caches via _reset_transient_state().  The immutable bare
        stiffness _K_bare is carried over unchanged.
        """
        s = self._shallow_clone()
        s.kT = T
        s._K_bare = self._K_bare
        s._reset_transient_state()
        return s

    def _full_rebuild(self) -> None:
        """Always call this single method after mutating any ModelParams field on a solver clone."""
        self.p.__post_init__()
        self._K_bare = self.p.K_lattice
        self._rebuild_orbital_operators()
        self._reset_transient_state()

    def _pairing_strengths(self, vertex_cache: Optional[dict], g_Delta_s: float, g_Delta_d: float, V_JT: float) -> Tuple[float, float]:
        if vertex_cache is not None:
            return vertex_cache['V_s_scalar'] * g_Delta_s, vertex_cache['V_d_scalar'] * g_Delta_d
        return V_JT * g_Delta_s, V_JT * g_Delta_d

    def build_local_hamiltonian_for_bdg(self, sign_M: float, M_channels: np.ndarray, J_A1g_diag: np.ndarray, mu: float, Z: float) -> np.ndarray:
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

        h_J_unit = Z * J_A1g_diag * self.sz_op         # (6,)
        M6 = _expand_M_channels(M_channels)            # (6,) # all chanels carrying its own Weiss space for the 2 orbitals of the corresponding Kramers doublet.

        H -= np.diag(sign_M * h_J_unit * M6)
        return H

    def _calc_dHdQ(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float = 0.0) -> np.ndarray:
        """
        Bare explicit Q-vertex
            dH/dQ = (∂H/∂Q)_{M, μ, F67s_mf, ... fixed}
        for the 24×24 BdG Hamiltonian. The explicit Q dependence retained here has FOUR channels:
            1. Local JT:
                ∂H_JT/∂Q = g_JT [β_k + Q β'_k] B1g_op.
            2. Dispersive inter-sublattice hopping:
                H_AB(Q,k)
            3. AFM Weiss field:
                ∂H_AFM/∂Q ∝ M Z J'_A1g(Q).
            4. Transverse anomalous transverse Weiss field at fixed F67s_mf:
                ∂H_TRW/∂Q = Z F67s_mf [β'_k J_B1g + β_k J'_B1g] B1g_offdiag.
        """
        vbdg = self._get_vbdg()

        kpts = np.asarray(vbdg._kpts)
        N = len(kpts)
        kx = kpts[:, 0]
        ky = kpts[:, 1]

        # Helper functions
        def _beta_at(Qv: float) -> np.ndarray:
            tx_v, ty_v = self.p.effective_hopping_anisotropic(Qv)
            beta = np.asarray(self.p.wave_function_weight(tx_v, ty_v, kx, ky), dtype=float)
            return beta

        def _exchange_at(Qv: float):
            tx_v, ty_v = self.p.effective_hopping_anisotropic(Qv)
            return self.p.exchange_channels(Qv, n_kspace, tx_v, ty_v, g_J)
        
        # Internal finite-difference step
        eps_beta = np.sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2)
        beta_k = _beta_at(Q)

        beta_p1 = _beta_at(Q + eps_beta)
        beta_m1 = _beta_at(Q - eps_beta)

        beta_p2 = _beta_at(Q + 2.0 * eps_beta)
        beta_m2 = _beta_at(Q - 2.0 * eps_beta)

        # 5-point O(h^4) derivative:
        dbeta_dQ = (
            beta_m2 - 8.0 * beta_m1 + 8.0 * beta_p1 - beta_p2
        ) / (12.0 * eps_beta)

        # 1. Local JT derivative
        dH_loc = (self.g_JT_bare * self.B1g_op).astype(complex)
        dH_loc_Q = (self.g_JT_bare * Q * self.B1g_op).astype(complex)

        dH_JT_k = (
            beta_k[:, None, None] * dH_loc[None, :, :]
            + dbeta_dQ[:, None, None] * dH_loc_Q[None, :, :]
        )

        # 2. Dispersive hopping analytical derivative
        dTx_op, dTy_op = self.p.hopping_matrices_dQ(Q)
        dH_AB_dQ = _build_H_AB_block(kx, ky, dTx_op, dTy_op, g_t)
        dH_AB_dQ_T = dH_AB_dQ.transpose(0, 2, 1)
        
        # 3. AFM Weiss-field derivative
        J_A1g, J_B1g_bare = _exchange_at(Q)

        J_A1g_p1, J_B1g_p1 = _exchange_at(Q + eps_beta)
        J_A1g_m1, J_B1g_m1 = _exchange_at(Q - eps_beta)

        J_A1g_p2, J_B1g_p2 = _exchange_at(Q + 2.0 * eps_beta)
        J_A1g_m2, J_B1g_m2 = _exchange_at(Q - 2.0 * eps_beta)

        # Sublattice A 5-point FD
        H_A_p1 = self.build_local_hamiltonian_for_bdg(+1.0, M, J_A1g_p1, mu, self.p.Z)
        H_A_m1 = self.build_local_hamiltonian_for_bdg(+1.0, M, J_A1g_m1, mu, self.p.Z)
        H_A_p2 = self.build_local_hamiltonian_for_bdg(+1.0, M, J_A1g_p2, mu, self.p.Z)
        H_A_m2 = self.build_local_hamiltonian_for_bdg(+1.0, M, J_A1g_m2, mu, self.p.Z)

        dH_A_dQ = (H_A_m2 - 8.0 * H_A_m1 + 8.0 * H_A_p1 - H_A_p2) / (12.0 * eps_beta)

        # Sublattice B 5-point FD
        H_B_p1 = self.build_local_hamiltonian_for_bdg(-1.0, M, J_A1g_p1, mu, self.p.Z)
        H_B_m1 = self.build_local_hamiltonian_for_bdg(-1.0, M, J_A1g_m1, mu, self.p.Z)
        H_B_p2 = self.build_local_hamiltonian_for_bdg(-1.0, M, J_A1g_p2, mu, self.p.Z)
        H_B_m2 = self.build_local_hamiltonian_for_bdg(-1.0, M, J_A1g_m2, mu, self.p.Z)

        dH_B_dQ = (H_B_m2 - 8.0 * H_B_m1 + 8.0 * H_B_p1 - H_B_p2) / (12.0 * eps_beta)
        # 4. TRW derivative
        dJ_B1g_dQ = (
            J_B1g_m2 - 8.0 * J_B1g_m1 + 8.0 * J_B1g_p1 - J_B1g_p2
        ) / (12.0 * eps_beta)

        trw_prefactor = (
            dbeta_dQ * J_B1g_bare
            + beta_k * dJ_B1g_dQ
        ) * (
            self.p.Z * F67s_mf
        )

        dH_TRW_k = (
            trw_prefactor[:, None, None]
            * self.B1g_offdiag[None, :, :]
        )

        # Assemble complete 24×24 BdG derivative
        dHdQ = np.zeros((N, _N_BDG, _N_BDG), dtype=complex)

        # Particle A: H_A = H_AFM + H_JT - H_TRW
        dHdQ[:, 0:6, 0:6] = dH_JT_k + dH_A_dQ[None, :, :] - dH_TRW_k

        # Particle B: H_B = H_AFM(B) + H_JT + H_TRW
        dHdQ[:, 6:12, 6:12] = dH_JT_k + dH_B_dQ[None, :, :] + dH_TRW_k

        # Particle hopping
        dHdQ[:, 0:6, 6:12] += dH_AB_dQ
        dHdQ[:, 6:12, 0:6] += np.conj(dH_AB_dQ_T)

        # Hole A/B: dH_hole/dQ = -conj(dH_particle/dQ)
        dHdQ[:, 12:18, 12:18] = -np.conj(dHdQ[:, 0:6, 0:6])
        dHdQ[:, 18:24, 18:24] = -np.conj(dHdQ[:, 6:12, 6:12])

        # Hole hopping: A->B: -H_AB*, B->A: -H_AB^T
        dHdQ[:, 12:18, 18:24] += -np.conj(dH_AB_dQ)
        dHdQ[:, 18:24, 12:18] += -dH_AB_dQ_T

        dHdQ = 0.5 * (dHdQ + dHdQ.conj().transpose(0, 2, 1))
        return dHdQ

    def compute_K_eff_full(self, target_doping: float, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, Gamma_M: float, V_irr_QQ: float, F67s_mf: float = 0.0, Q_Eg2: float = 0.0, vertex_cache: dict = None) -> Tuple[float, float]:
        eps2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
        eps = np.sqrt(eps2)

        def F_can_at_Q(Q_val: float) -> float:
            tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q_val)
            tx, ty = g_t * tx_bare, g_t * ty_bare
            t_eff = np.sqrt(0.5 * (tx**2 + ty**2))
            mu_opt, n_kspace_opt = self._find_mu_for_density(M, Q_val, Delta_s, Delta_d, target_doping, mu, t_eff, g_t, g_J, F67s_mf=F67s_mf)
            
            omega = self._compute_bdg_free_energy(M, Q_val, Delta_s, Delta_d, n_kspace_opt, mu_opt, g_t, g_J, F67s_mf, Q_Eg2)
            return omega + mu_opt *n_kspace_opt   # Legendre transform

        # Canonical free energy around Q
        F0 = F_can_at_Q(Q)
        Fp = F_can_at_Q(Q + eps)
        Fm = F_can_at_Q(Q - eps)

        if vertex_cache is not None:
            chi_SS_q0 = vertex_cache['chi_SS_q0']
            chi_SQ_q0 = vertex_cache['chi_SQ_q0']
            chi_QS_q0 = vertex_cache['chi_QS_q0']
            chi_QQ_q0 = vertex_cache['chi_QQ_q0']
        else:
            ev, ec = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self._get_vbdg())
            chi_SS_q0, chi_SQ_q0, chi_QS_q0, chi_QQ_q0 = self.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M, F67s_mf, (ev, ec), apply_diamagnetic_QQ=False, mechanical=False)

        rqq = self._rpa_vertex(0.0, 0.0, V_irr_QQ, chi_SS_q0, chi_SQ_q0, chi_QS_q0, chi_QQ_q0, 0.0)[1]
        dK_corr = (chi_QQ_q0 - rqq) * self.g_JT_bare**2
        cap = _DK_CORR_CAP_MULT * max(abs(self._K_bare), _MATH_EPS)
        dK_corr = math.copysign(min(abs(dK_corr), cap), dK_corr)
        K_can = self._K_bare + (Fp - 2.0 * F0 + Fm) / eps2 + dK_corr
        return K_can, F0

    def B1g_expectation(self, tx_b: float, ty_b: float, E_k_cache: tuple) -> float:
        """
        Per-site ⟨B1g_op⟩ in the BdG ground state
        Uses the full 24-component Nambu eigenstates so that the anomalous u·v amplitudes — which carry the SC-triggered orbital coherence — are fully included.
        The /4 factor corrects for the Nambu doubling (2 sublattices × particle-hole redundancy).
        """
        ev, ec = E_k_cache
        vbdg = self._get_vbdg()
        _kx = vbdg._kpts[:, 0]
        _ky = vbdg._kpts[:, 1]
        f_n = _fermi_function(ev, self.kT)  # (Nk, 24)
        # ⟨B1g⟩_k = Tr[B1g_24 · ρ_k]  where ρ_k = Σ_n (u_n u_n† f_n + v_n v_n† (1-f_n))
        # Diagonal of ec† B1g_24 ec in the quasiparticle basis: a, b: component indices (rows and columns of the orbitals / Nambu operator), n: band index (columns of the eigenvector in the ec matrix
        diag_qp = np.einsum('kan, ab, kbn -> kn', ec.conj(), self.B1g_24, ec).real
        
        # Thermal avg in Nambu basis. Summing over all 24 bands automatically covers the hole components via f(-E) = 1 - f(E).
        exp_k = np.einsum('kn,kn->k', diag_qp, f_n) * self.p.wave_function_weight(tx_b, ty_b, _kx, _ky)
        return max(float(np.dot(self.k_weights, exp_k)) / 4.0, _MATH_EPS)

    def Eg2_expectation(self, E_k_cache: tuple) -> float:
        """Per-site ⟨Eg2_op⟩ in the BdG ground state."""
        ev, ec = E_k_cache
        f_n = _fermi_function(ev, self.kT)
        diag_qp = np.einsum('kan, ab, kbn -> kn', ec.conj(), self.Eg2_24, ec).real
        exp_k = np.einsum('kn,kn->k', diag_qp, f_n)
        return float(np.dot(self.k_weights, exp_k)) / 4.0

    def _compute_chi_tau(self, M: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float = 0.0) -> Dict:
        """
        JT orbital susceptibility χ_τ = ∂⟨B1g_op⟩/∂(g_JT·Q) via Richardson-extrapolated central finite difference of the per-site ⟨B1g_op⟩ expectation value.

        δχ_τ = χ_τ(Δ=0) - χ_τ(Δ≠0) isolates the condensate contribution.
        In D₄h: ⟨B1g_op⟩=0 exactly in normal state → δχ_τ = χ_τ(Δ≠0).
        In D₂h: a small normal-state baseline can exist; subtraction prevents D₂h signal from masquerading as SC-triggered.

        Richardson extrapolation (3 primary step sizes h, h/2, h/4):
          R1 = (4·CD(h/2)−CD(h))/3, R2 = (4·CD(h/4)−CD(h/2))/3, est = mean(R1,R2).
          Converged: |R1−R2|/max(|est|,ε) < 3%  → return est (O(h⁴) accurate).
          Nonlinear: |CD(h)−CD(h/2)|/max(|CD(h/2)|,ε) > 20%
            → try h/8 fallback; if still nonlinear → return 0.0 (conservative).
        """
        vbdg = self._get_vbdg()
        scale = self.p.Delta_CF / self.g_JT_bare
        h_floor = 1e-4
        h = float(np.clip(1e-3 * max(abs(Q), scale), h_floor, 0.05 * scale))

        def _cd(dq: float, ds: complex, dd: complex) -> float:
            """Central difference d⟨B1g⟩/dQ at step dq."""
            vp = self.B1g_expectation(*self.p.effective_hopping_anisotropic(Q + dq), np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q + dq, ds, dd, n_kspace, mu, g_t, g_J, F67s_mf, out=vbdg._H_stack)))
            vm = self.B1g_expectation(*self.p.effective_hopping_anisotropic(Q - dq), np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q - dq, ds, dd, n_kspace, mu, g_t, g_J, F67s_mf, out=vbdg._H_stack)))
            return (vp - vm) / (2.0 * dq)

        def _richardson(ds: complex, dd: complex) -> tuple:
            cd1 = _cd(h,       ds, dd)
            cd2 = _cd(h / 2.0, ds, dd)
            cd3 = _cd(h / 4.0, ds, dd)
            
            R1  = (4.0 * cd2 - cd1) / 3.0
            R2  = (4.0 * cd3 - cd2) / 3.0
            est = 0.5 * (R1 + R2)
            err       = abs(R1 - R2) / max(abs(est), 1e-12)
            converged = err < 0.03
            nonlinear = abs(cd1 - cd2) / max(abs(cd2), 1e-12) > 0.2
            
            if converged:
                return est, True, False, 1.0
            
            if not nonlinear:
                return cd1, False, False, 1.0
            
            # Nonlinear branch: try one more refinement level
            cd4 = _cd(h / 8.0, ds, dd)
            nonlinear2 = abs(cd3 - cd4) / max(abs(cd4), 1e-12) > 0.2

            if not nonlinear2:
                return cd4, False, True, 0.5
            
            # Both scales nonlinear → derivative unresolvable at this Q
            return 0.0, False, True, 0.0

        # SC-state χ_τ
        dB1g_sc, ok_sc, nonlin_sc, w_sc = _richardson(Delta_s, Delta_d)

        # Normal-state baseline: MUST evaluate even when Δ is small, because in D₂h symmetry the normal state has a non-zero B₁g response that must be subtracted.
        dB1g_n,  ok_n,  nonlin_n,  w_n  = _richardson(0.0j, 0.0j)

        # ── Reliability guard ──────────────────────────────────────────────────────
        if w_sc == 0.0:
            chi_tau_sc = 0.0  # fully unresolvable (both scales nonlinear): use zero
        else:
            chi_tau_sc = w_sc * dB1g_sc / self.g_JT_bare   # sign physical: negative = stiffening

        if w_n == 0.0:
            chi_tau_n = 0.0   # unreliable baseline — skip subtraction
        else:
            chi_tau_n = w_n * dB1g_n / self.g_JT_bare

        if nonlin_sc and w_sc > 0.0:
            _scf_log("CHI-TAU",
                     f"  ⚠ χ_τ nonlinear at Q={Q:+.5f} Å — finer-scale estimate used"
                     f"  (weight={w_sc:.1f}, value={chi_tau_sc:+.5f} eV⁻¹)."
                     f"  Near first-order SC-JT boundary; feedback halved.")
        elif nonlin_sc:
            _scf_log("CHI-TAU",
                     f"  ⚠ χ_τ unresolvable at Q={Q:+.5f} Å — zero used."
                     f"  Both Richardson scales nonlinear; SCF feedback suppressed.")
        
        richardson_ok = ok_sc and ok_n and not nonlin_sc and not nonlin_n
        chi_tau_net = max(chi_tau_n - chi_tau_sc , 0.0) if richardson_ok else 0.0
        return {
            'chi_tau_sc':     chi_tau_sc,
            'chi_tau_n':      chi_tau_n,
            'chi_tau_net':    chi_tau_net,
            'richardson_ok':  richardson_ok,
            'chi_tau_weight': w_sc,   # 1.0=full, 0.5=halved, 0.0=suppressed
        }

    def _chi_QQ_matrix_elements(self, M: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float = 0.0, Q_Eg2: float = 0.0, return_matrix: bool = False):
        """
        Bare JT orbital susceptibility: χ_QQ = −∂²Ω/∂Q² evaluated at Δ=0. χ_QQ is a normal-state quantity return_matrix=True:
        returns the full 2×2 matrix {χ_QQ[B1g,B1g], χ_QQ[B1g,Eg2]; χ_QQ[Eg2,B1g], χ_QQ[Eg2,Eg2]} via mixed finite differences of the SAME grand potential Ω(Q, Q_Eg2),
        using a 9-point stencil (the B1g-only diagonal term reuses the original 3-point formula exactly).
        """
        eps2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
        dQ   = np.sqrt(eps2)
        vbdg = self._get_vbdg()

        def omega(Qval, Qeg2val):
            ev = np.linalg.eigvalsh(
                vbdg._build_H_stack(vbdg._kpts, M, Qval, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, F67s_mf, vbdg._H_stack, Qeg2val)
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

    def estimate_chi_SQ_q_full(self, target_doping: float, M: np.ndarray, Q: float, Delta_s: float, Delta_d: float, n_kspace: float, mu: float, J_eff: float, F67s_mf: float, n_q: int):
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
        vbdg = self._get_vbdg()
        g_t, g_J, _, _ = self.p.get_gutzwiller_factors(target_doping)
        tx_b, ty_b = self.p.effective_hopping_anisotropic(Q)
        tx, ty = g_t * tx_b, g_t * ty_b

        # Normal-state eigenstates (Δ=0)
        E_k_n, V_k_n = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=vbdg._H_stack)
            )
        E_k_sc, V_k_sc = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, F67s_mf, out=vbdg._H_stack)
            )
        
        f_k_n  = _fermi_function(E_k_n, self.kT)
        eta_n  = max(_ETA_T_FRAC * self.kT, _ETA_GRID_FLOOR * self.p.t0)    # Normal-state: thermal broadening dominates (bands gapped by h_afm).
        _Gamma_M = self._moriya_gamma_landau(target_doping)
        _chi_QQ_n = self._chi_QQ_matrix_elements(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J)
        
        _qvals = np.linspace(-np.pi, np.pi, n_q, endpoint=False)
        _QX, _QY = np.meshgrid(_qvals, _qvals)
        q_grid = np.column_stack((_QX.ravel(), _QY.ravel()))
        n_pts  = len(q_grid)

        chi_SQ_n  = np.zeros(n_pts)
        chi_SQ_sc = np.zeros(n_pts)
        chi_SS_n  = np.zeros(n_pts)
        chi_SS_sc = np.zeros(n_pts)
        chi_QQ_sc_arr = np.zeros(n_pts)

        sz_6, sz_7a, sz_7b = self.sz_op[0:2], self.sz_op[2:4], self.sz_op[4:6]
        dk = 2.0 * np.pi / _NK

        beta_k_array = self.p.wave_function_weight(tx_b, ty_b, self.k_points[:, 0], self.k_points[:, 1])
        vw_sq = beta_k_array ** 2

        for i_q, q in enumerate(q_grid):
            nx = int(round(q[0] / dk)) % _NK
            ny = int(round(q[1] / dk)) % _NK
            shift_idx = self.shift_table[nx, ny]

            # Normal state: _NORMAL_SECTOR_PAIRS on Δ=0 eigenstates
            chi_n = _lindhard_bubble(_NORMAL_SECTOR_PAIRS, E_k_n, V_k_n, f_k_n, shift_idx, self.k_weights, vw_sq, eta_n, self.kT)
            cr_n = chi_n.real
            chi_SS_n[i_q] = float(self.sz_op @ cr_n @ self.sz_op)
            chi_SQ_n[i_q] = float(np.trace(np.diag(sz_6) @ cr_n[0:2, 2:4]) + np.trace(np.diag(sz_7a) @ cr_n[2:4, 0:2])
                                 + np.trace(np.diag(sz_6) @ cr_n[0:2, 4:6]) + np.trace(np.diag(sz_7b) @ cr_n[4:6, 0:2]))

            # χ_SQ evaluated on the condensate band structure, which lifts the D₄h B₂g selection rule via Bogoliubov Γ₆/Γ₇ mixing in normal propagator.
            # SC-state eigenstates (Δ≠0) — same sector pairs, different bands
            _chi_SS_sc, _chi_SQ_sc, _, _chi_QQ_sc = self.get_susceptibilities_sc(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, q, _Gamma_M, F67s_mf, (E_k_sc, V_k_sc))
            chi_SS_sc[i_q]     = _chi_SS_sc
            chi_SQ_sc[i_q]     = _chi_SQ_sc
            chi_QQ_sc_arr[i_q] = _chi_QQ_sc
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        data = [
            (chi_SS_n,  'Normal state', r'$\chi_{SS}^{\rm normal}(q)$'),
            (chi_SQ_n,  'Normal state', r'$\chi_{SQ}^{\rm normal}(q)$'),
            (None,      'Normal state', r'$\chi_{QQ}^{\rm normal}$ (const)'),  # konstans, nem q-függő
            (chi_SS_sc, 'SC state',     r'$\chi_{SS}^{\rm SC}(q)$'),
            (chi_SQ_sc, 'SC state',     r'$\chi_{SQ}^{\rm SC}(q)$'),
            (chi_QQ_sc_arr, 'SC state', r'$\chi_{QQ}^{\rm SC}(q)$'),
        ]

        vmax_SS = max(chi_SS_n.max(), chi_SS_sc.max())
        vmax_SQ = max(abs(chi_SQ_n).max(), abs(chi_SQ_sc).max())
        vmax_QQ = chi_QQ_sc_arr.max()

        for ax, (vals, state, title) in zip(axes, data):
            if vals is None:
                # χ_QQ normál állapotban konstans – szövegesen jelezzük
                ax.text(0.5, 0.5, r'$\chi_{QQ}^{\rm normal} = $' + f'{_chi_QQ_n:.4f}\n(konstans, $q$-független)',
                        transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(r'$\chi_{QQ}^{\rm normal}$ (konstans)')
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

        _scf_log("χ-DIAG", 
            f"χ_SQ scan ({n_q}×{n_q} grid): "
            f"norm peak={_classify(q_peak_n)} {chi_SQ_pk_n:+.3f} (antinode={antinodal_frac_n:.0%}) | "
            f"SC peak={_classify(q_peak_sc)} {chi_SQ_pk_sc:+.3f} (antinode={antinodal_frac_sc:.0%}) | "
            f"φ_d overlap: n={_phi_overlap(chi_SQ_n):.2f} sc={_phi_overlap(chi_SQ_sc):.2f}"
            + (" ⚠local-vertex overestimate" if antinodal_frac_n <= 0.5 and _classify(q_peak_n) == 'M(π,π)' else "")
        )

        if not np.allclose(q_peak_n, q_peak_sc, atol=np.pi/18):
            _scf_log("χ-DIAG", 
                f"  ⚠ peak shift: norm {_classify(q_peak_n)} → SC {_classify(q_peak_sc)}"
                f"  (Δχ_SQ = {chi_SQ_pk_sc - chi_SQ_pk_n:+.3f})")
        
        if q_peak_n is not None and len(chi_SS_n) > 0:
            idx_SS = np.argmax(np.abs(chi_SS_n))
            if not np.allclose(q_grid[idx_SS], q_peak_n, atol=np.pi/18 + 0.1):
                _scf_log("χ-DIAG",
                    f"  ⚠ χ_SS peak at {_classify(q_grid[idx_SS])}, χ_SQ_n at {_classify(q_peak_n)} — differ")

    def _compute_nambu_kernel(self, E_k_all: np.ndarray, shift_idx: np.ndarray, eta: float) -> np.ndarray:
        """
        The (N_k,24,24) Lehmann-weight kernel shared by every χ_AB(q) contraction at a given (E_k_all, shift_idx, eta)
        -- i.e. it does NOT depend on which vertex matrices M_A/M_B are being sandwiched. Factored out of _compute_nambu_susceptibility
        so callers that need χ_SS, χ_SQ, χ_QS, χ_QQ at the SAME q can build this once and reuse it four times, instead of rebuilding the identical (N_k,24,24) arrays four times over.
        """
        E_kQ = E_k_all[shift_idx]                     # (N_k, 24)
        f_k  = _fermi_function(E_k_all, self.kT)      # (N_k, 24)
        f_kQ = _fermi_function(E_kQ, self.kT)         # (N_k, 24)

        kernel = _lehmann_kernel(f_k, f_kQ, E_k_all, E_kQ, eta, self.kT)

        # Apply k-weighting
        kernel = kernel * self.k_weights[:, None, None]
        return kernel

    def _compute_nambu_susceptibility(self, E_k_all: np.ndarray, M_A_bands: np.ndarray, M_B_bands: np.ndarray, shift_idx: np.ndarray, eta: float, kernel_precomputed: np.ndarray = None) -> complex:
        """
        Unified Nambu Lehmann sum for the retarded susceptibility χ_{AB}(q, ω→0).

        Standard Kubo–Lehmann convention:
        ──────────────────────────────────────────────────────────────────────
        χ_{AB}(q) = (1/N_k) Σ_{k,n,m}
                        (f_n(k) − f_m(k+q)) · (E_m(k+q) − E_n(k))
                        ─────────────────────────────────────────────
                        (E_m(k+q) − E_n(k))² + η²
                        × ⟨k,n|M_A|k+q,m⟩ · ⟨k+q,m|M_B|k,n⟩

        where n labels bands at k, m labels bands at k+q. M_A, M_B : 24×24 Nambu vertex matrices.
        
        This single sum automatically captures ALL Gorkov contributions (GG and FF Nambu sectors) without any manual sector splitting or double-counting.
        """
        kernel = kernel_precomputed if kernel_precomputed is not None else self._compute_nambu_kernel(E_k_all, shift_idx, eta)

        # Contraction with transformed Nambu vertices; the 4.0 division is strictly required to compensate for Nambu particle-hole doubling.
        chi_val = np.einsum('knm,knm,kmn->', kernel, M_A_bands, M_B_bands, optimize=True)
        return complex(chi_val) / 4.0

    def _diamagnetic_QQ_term(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float, E_k_all: np.ndarray, V_k_all: np.ndarray) -> float:
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

        The derivative is evaluated at fixed M, μ, g_t, g_J and F67s_mf.
        Thus this is the explicit-Hamiltonian second derivative (∂²H/∂Q²)_{M,μ,g_t,g_J,F67s_mf},
        not the total second derivative of the fully self-consistent free energy along an SCF solution branch.

        The supplied E_k_all and V_k_all are the eigenvalues/eigenvectors of the ORIGINAL Hamiltonian at Q.
        They are used only to evaluate the expectation value of ∂²H/∂Q² in that state.
        """
        # The outer stencil stays accurate down to a roundoff floor roughly 2-3 orders of magnitude below this step
        h_QQ = np.sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2)

        dHdQ_m2 = self._calc_dHdQ(M, Q - 2.0 * h_QQ, n_kspace, mu, g_t, g_J, F67s_mf)
        dHdQ_m1 = self._calc_dHdQ(M, Q - h_QQ, n_kspace, mu, g_t, g_J, F67s_mf)
        dHdQ_p1 = self._calc_dHdQ(M, Q + h_QQ, n_kspace, mu, g_t, g_J, F67s_mf)
        dHdQ_p2 = self._calc_dHdQ(M, Q + 2.0 * h_QQ, n_kspace, mu, g_t, g_J, F67s_mf)
        
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

    def get_susceptibilities_sc(self, M: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, q: np.ndarray, Gamma_M: float, F67s_mf: float, E_k_cache: tuple, apply_diamagnetic_QQ: bool = False, dHdQ_precomputed: np.ndarray = None, mechanical: bool = True) -> Tuple[float, float, float, float]:
        """
        SC-state spin–quadrupole cross-susceptibility χ_{Sz, ∂H/∂Q}^SC(q).

        Kubo formula:  χ_SQ(q) = ⟨⟨S_z ; ∂H/∂Q⟩⟩_{ω=0}

        The Hamiltonian H(Q) depends on Q through THREE channels:
          ∂H/∂Q = g_JT · B1g_op  [local JT coupling, k-independent]
                + ∂H_AB/∂Q  [dispersive hopping renorm, k-dependent, orbital-selective 6×6 matrix]
                + ∂H_A(B)/∂Q  [AFM Weiss field via J_A1g_diag(Q), k-independent, ∝ M]
        
        Set `apply_diamagnetic_QQ=True` to add that missing term back in — do this ONLY at q that correspond to the magnetic-cell zone center (q=(0,0) or, after AFM Umklapp folding, q=(π,π)
        """
        E_k_all, V_k_all = E_k_cache
        _Delta_amp = abs(Delta_s) + abs(Delta_d)

        # Gap-proportional broadening resolves Bogoliubov coherence peaks at Δ scale; k-grid floor prevents aliasing when a Bogoliubov band crossing falls between grid points.
        eta = max(_ETA_DELTA_FRAC * _Delta_amp, _ETA_GRID_FLOOR * self.p.t0)
        
        dk = 2.0 * np.pi / _NK
        nx = int(round(q[0] / dk)) % _NK
        ny = int(round(q[1] / dk)) % _NK
        shift_idx = self.shift_table[nx, ny]
        V_kQ = V_k_all[shift_idx]

        # Lehmann kernel only depends on (E_k_all, shift_idx, eta)
        kernel = self._compute_nambu_kernel(E_k_all, shift_idx, eta)

        # Spin vertex
        M_A_bands = np.einsum('kan,ab,kbm->knm', V_k_all.conj(), self.Sz_nambu, V_kQ, optimize=True)
        M_B_bands = np.einsum('kam,ab,kbn->kmn', V_kQ.conj(), self.Sz_nambu, V_k_all, optimize=True)
        # χ_SS: spin-spin, both vertices are Sz_nambu
        chi_SS_cplx = self._compute_nambu_susceptibility(E_k_all, M_A_bands, M_B_bands, shift_idx, eta, kernel_precomputed=kernel)

        # Construct the Q vertex according to the requested mode
        if mechanical:
            dHdQ = dHdQ_precomputed if dHdQ_precomputed is not None else self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J, F67s_mf)
        else:
            dH_Q_k = (self.g_JT_bare * self.B1g_24).astype(complex)   # shape (24,24)
            dHdQ = np.broadcast_to(dH_Q_k[None, :, :], (len(E_k_all), _N_BDG, _N_BDG))
            
        # Transform to band basis
        M_A_bands_SQ = np.einsum('kan,kab,kbm->knm', V_k_all.conj(), dHdQ, V_kQ, optimize=True)
        M_B_bands_SQ = np.einsum('kam,kab,kbn->kmn', V_kQ.conj(),    dHdQ, V_k_all, optimize=True)

        # χ_SQ: build the full k-dependent ∂H/∂Q in the 24×24 Nambu basis
        chi_SQ_cplx = self._compute_nambu_susceptibility(E_k_all, M_A_bands,    M_B_bands_SQ, shift_idx, eta, kernel_precomputed=kernel)
        chi_QS_cplx = self._compute_nambu_susceptibility(E_k_all, M_A_bands_SQ, M_B_bands,    shift_idx, eta, kernel_precomputed=kernel)
        chi_QQ_cplx = self._compute_nambu_susceptibility(E_k_all, M_A_bands_SQ, M_B_bands_SQ, shift_idx, eta, kernel_precomputed=kernel)

        # Static (ω=0) limit: imaginary part ∝ η → 0
        chi_SS_val = float(chi_SS_cplx.real)   # 1/eV
        chi_SQ_val = float(chi_SQ_cplx.real)   # 1/Å
        chi_QS_val = float(chi_QS_cplx.real)   # 1/Å
        chi_QQ_val = float(chi_QQ_cplx.real)   # eV/Å²

        if apply_diamagnetic_QQ:
            chi_QQ_val -= self._diamagnetic_QQ_term(M, Q, n_kspace, mu, g_t, g_J, F67s_mf, E_k_all, V_k_all)   # eV/Å²
        
        # ---- Normalise to common 1/eV units ----
        chi_SS = chi_SS_val                   # 1/eV
        chi_SQ = chi_SQ_val / self.g_JT_bare            # (1/Å) / (eV/Å) = 1/eV
        chi_QS = chi_QS_val / self.g_JT_bare             # (1/Å) / (eV/Å) = 1/eV
        chi_QQ = chi_QQ_val / self.g_JT_bare**2   # (eV/Å²) / (eV²/Å²) = 1/eV

        # Symmetric average of χ_SQ/χ_QS
        chi_SQ_sym = 0.5 * (chi_SQ + chi_QS)

        # ---- Moriya damping on spin channel ----
        chi_SS = chi_SS / max(1.0 + Gamma_M * chi_SS, _MATH_EPS)

        # ---- PSD projection of [[χ_SS, χ_SQ], [χ_SQ, χ_QQ]] ----
        _psd_mat = np.array([[chi_SS,     chi_SQ_sym],
                             [chi_SQ_sym, chi_QQ]], dtype=float)
        _psd_eigv, _psd_evc = np.linalg.eigh(_psd_mat)
        if _psd_eigv[0] < 0.0:
            _ev_clipped = np.maximum(_psd_eigv, 0.0)
            _mat_proj = _psd_evc @ np.diag(_ev_clipped) @ _psd_evc.T
            _mat_proj = 0.5 * (_mat_proj + _mat_proj.T)   # enforce symmetry
            chi_SS     = float(_mat_proj[0, 0])
            chi_SQ_sym = float(_mat_proj[0, 1])
            chi_QQ     = float(_mat_proj[1, 1])

        return chi_SS, chi_SQ_sym, chi_SQ_sym, chi_QQ

    def _rpa_det(self, J_eff: float, V_JT_corr: float, chi_SS_moriya: float, chi_SQ_v: float, chi_QS_v: float, chi_QQ_v: float) -> Tuple[float, float, float, float, float]:
        a = 1.0 - (J_eff * chi_SS_moriya)
        b = -V_JT_corr * chi_QS_v
        c = -J_eff * chi_SQ_v
        d = 1.0 - (V_JT_corr * chi_QQ_v)

        det = a * d - b * c
        M_frob    = math.sqrt(a*a + b*b + c*c + d*d)  # Frobenius norm sets natural scale of matrix
        det_floor = max(_MATH_EPS, 1e-4 * M_frob)
        if abs(det) < det_floor:
            det_safe = math.copysign(det_floor, det) if det != 0.0 else det_floor
        else:
            det_safe = det
        return det_safe, a, b, c, d
    
    def _rpa_vertex(self, J_eff: float, V_JT: float, V_JT_corr: float, chi_SS_moriya: float, chi_SQ_v: float, chi_QS_v: float, chi_QQ_v: float, V_cap: float) ->  Tuple[float, float]:
        """
        RPA pairing vertex from the local irreducible vertex and the bare bubble chi0.
        The full two-particle response is obtained from the Bethe-Salpeter/Dyson equation
        The pairing vertex in the spin/JT channel is then
            Vp = J_eff² χ_SS^full + V_JT² χ_QQ^full
                + J_eff V_JT (χ_SQ^full + χ_QS^full).
        """
        det, a, b, c, d = self._rpa_det(J_eff, V_JT_corr, chi_SS_moriya, chi_SQ_v, chi_QS_v, chi_QQ_v)

        # chi_full = M^{-1} chi0, evaluated analytically for the 2×2 case.
        rss = (d * chi_SS_moriya - b * chi_QS_v) / det
        rsq = (d * chi_SQ_v      - b * chi_QQ_v) / det
        rqs = (-c * chi_SS_moriya + a * chi_QS_v) / det
        rqq = (-c * chi_SQ_v      + a * chi_QQ_v) / det

        # Spin/JT pairing vertex.
        Vp = J_eff**2 * rss + V_JT**2 * rqq + J_eff * V_JT * (rsq + rqs)

        # UV cap: perturbative RPA breaks down near the QCP.
        if abs(Vp) > V_cap:
            Vp = math.copysign(V_cap, Vp)

        if not math.isfinite(Vp):
            Vp = V_cap
        return float(Vp), rqq

    def _moriya_gamma_landau(self, target_doping: float) -> float:
        """
        Moriya SCR self-consistency in the static/Gaussian approximation: Γ_M ≈ b · ⟨δM²⟩ = b · kT/a,
        where a and b are the model's own Landau coefficients, F(M) ≈ F₀ + (a/2)M² + (b/4)M⁴, obtained from
        free energy at M=0 (paramagnetic reference, Q=0).

        This is a static proxy for the full dynamical SCR condition, where ⟨δM²⟩ is frequency- and momentum-summed
        rather than given by the classical kT/a estimate. Its magnitude is therefore expected to be accurate only
        within a factor of a few, but it is derived from the model's own free energy and empirically removes
        the ~10⁴–10⁵× magnitude mismatch of the t²/J-based proxy.
        """
        cache = self.__dict__.setdefault('_moriya_landau_cache', {})
        key = round(float(target_doping), 8)
        if key in cache:
            return cache[key]

        g_t, g_J, _, _ = self.p.get_gutzwiller_factors(target_doping)
        Q0 = 0.0
        mu_guess = -2.0 * self.p.t0 * math.tanh(target_doping / 0.1)
        tx_bare0, ty_bare0 = self.p.effective_hopping_anisotropic(Q0)
        tx0, ty0 = g_t * tx_bare0, g_t * ty_bare0
        t_eff0 = float(np.sqrt(0.5 * (tx0**2 + ty0**2)))

        def F_of_M(Mval: float) -> float:
            Mc = np.full(_N_CHANNELS, Mval, dtype=float)
            mu_opt, nk = self._find_mu_for_density(Mc, Q0, 0.0j, 0.0j, target_doping, mu_guess, t_eff0, g_t, g_J, F67s_mf=0.0)
            Omega = self._compute_bdg_free_energy(Mc, Q0, 0.0j, 0.0j, nk, mu_opt, g_t, g_J, 0.0, 0.0)
            return Omega + mu_opt * nk  # Legendre correction

        h = _MORIYA_LANDAU_M_STEP
        F0  = F_of_M(0.0)
        Fp1 = F_of_M(h);      Fm1 = F_of_M(-h)
        Fp2 = F_of_M(2.0*h);  Fm2 = F_of_M(-2.0*h)
        a_landau = (Fp1 - 2.0*F0 + Fm1) / h**2
        d4F_dM4  = (Fp2 - 4.0*Fp1 + 6.0*F0 - 4.0*Fm1 + Fm2) / h**4
        b_landau = d4F_dM4 / 6.0   # d^4/dM^4[(1/4)*b*M^4] = 6b

        a_safe = max(a_landau, _MATH_EPS)
        b_safe = max(b_landau, 0.0)   # a negative quartic coefficient signals the simple Landau/SCR picture itself is breaking down
        Gamma_M_landau = b_safe * self.kT / a_safe

        cache[key] = float(Gamma_M_landau)
        return cache[key]

    def _make_vertex_params(self, target_doping: float, tx: float, ty: float, g_t: float, J_eff: float) -> Tuple[float, float, float]:
        Gamma_M = self._moriya_gamma_landau(target_doping)  # model-derived Moriya damping Γ_M
        V_JT    = self.g_JT_bare **2 / max(self._K_bare, _MATH_EPS)  # bare JT pairing vertex
        V_cap   = _RPA_V_CAP_ALPHA * max(_RPA_BW_FACTOR * max(abs(tx), abs(ty), 1e-6), J_eff)  # BEC boundary (8t), UV cap on the RPA pairing vertex
        return Gamma_M, V_JT, V_cap

    def compute_pairing_kernel_and_build_cache(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, J_eff: float, Gamma_M: float, V_JT: float, V_JT_corr: float, V_cap: float, det_afm_sc: float = 1.0, solve_state: '_SolveState' = None) -> Dict:
        """
        Compute basis functions, channel scalars (V_s, V_d), the 2x2 pairing kernel K_pair, and related diagnostics from the pre-built RPA vertex matrices on the d- and s-FS.
        The kernel is reduced to the 2D subspace spanned by phi_s=1 and phi_d=cos(kx)-cos(ky), yielding a 2×2 matrix
          K = [[g_s V_ss, sqrt(g_s g_d) V_sd],
               [sqrt(g_s g_d) V_sd, g_d V_dd]]
        whose larger eigenvalue is lambda_lin_max.
        If solve_state is None, the V_d EMA is skipped; used in linearized solve as a 2×2 eigenvalue problem in the (s,d) channel subspace.
          s-channel: uses JT-phonon vertex only (V_jt), with Gutzwiller factor g_Delta_s = g_t.
          d-channel: uses full RPA vertex (spin+JT+cross), with g_Delta_d = g_t + (g_J-g_t)*p_7.
        """
        # --- Fermi-surface points ---
        fs_pts, vF_arr, fs_idx, inv_vF = self._get_fs_points(M, Q, n_kspace, mu, g_t, g_J, store_cache=True)
        N_fs = len(fs_pts)
        i_idx, j_idx, unique_q, inv_idx = _unique_q_pairs(fs_pts)

        # Build the χ₀ cache once for the normal state
        ev, ec = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self._get_vbdg())
        # Build vertex matrices on the two FS sets.
        dHdQ_precomputed = self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J)

        n_q = len(unique_q)
        V_unique = np.empty(n_q, dtype=float)
        V_spin_u = np.empty(n_q, dtype=float)
        V_jt_u   = np.empty(n_q, dtype=float)

        # Full and JT-only RPA vertex matrices on a given set of Fermi-surface points.
        for u_idx, q_u in enumerate(unique_q):
            chi_SS, chi_SQ, chi_QS, chi_QQ = self.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, q_u, Gamma_M, 0.0, (ev, ec), apply_diamagnetic_QQ=False, dHdQ_precomputed=dHdQ_precomputed)
            V_unique[u_idx] = self._rpa_vertex(J_eff, V_JT, V_JT_corr, chi_SS, chi_SQ, chi_QS, chi_QQ, V_cap)[0]
            V_spin_u[u_idx] = self._rpa_vertex(J_eff, 0.0,  V_JT_corr, chi_SS, chi_SQ, chi_QS, chi_QQ, V_cap)[0]
            V_jt_u[u_idx]   = self._rpa_vertex(0.0,   V_JT, V_JT_corr, chi_SS, chi_SQ, chi_QS, chi_QQ, V_cap)[0]

        # Build symmetric matrices
        V_ij_full = np.zeros((N_fs, N_fs), dtype=float)
        V_ij_full[i_idx, j_idx] = V_unique[inv_idx]
        V_ij_full = 0.5 * (V_ij_full + V_ij_full.T)

        V_ij_jt = np.zeros((N_fs, N_fs), dtype=float)
        V_ij_jt[i_idx, j_idx] = V_jt_u[inv_idx]
        V_ij_jt = 0.5 * (V_ij_jt + V_ij_jt.T)

        # --- basis functions ---
        phi_d   = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
        phi_s_d = np.ones(len(fs_pts), dtype=float)   # s‑wave on d‑FS

        # --- normalisation factors (Overlap matrix S) ---
        ns   = max(float(phi_s_d @ (phi_s_d * inv_vF)), 1e-12)
        nd   = max(float(phi_d   @ (phi_d   * inv_vF)), 1e-12)
        n_sd = float((phi_s_d * phi_d) @ inv_vF)
        
        S_pair = np.array([[ns, n_sd], [n_sd, nd]], dtype=float)

        # --- Unnormalized Interaction Matrix (W = <phi_i | V | phi_j>) ---
        W11 = g_Delta_s * np.dot(phi_s_d * inv_vF, np.dot(V_ij_jt, phi_s_d * inv_vF))
        W22 = g_Delta_d * np.dot(phi_d * inv_vF, np.dot(V_ij_full, phi_d * inv_vF))
        W12 = math.sqrt(max(g_Delta_s * g_Delta_d, 0.0)) * np.dot(phi_s_d * inv_vF, np.dot(V_ij_jt, phi_d * inv_vF))
        
        W_pair = np.array([[W11, W12], [W12, W22]], dtype=float)

        # --- convert generalized eigenproblem to ordinary symmetric eigenproblem ---
        s_eval, U = np.linalg.eigh(S_pair)
        S_inv_sqrt = (U / np.sqrt(np.maximum(s_eval, 1e-12))) @ U.T

        # Use the properly orthonormalised Rayleigh matrix W_ortho = <ψ_i|V|ψ_j>, ψ = S_pair^{-1/2}φ, which is required for non-orthogonal φ_s,φ_d. Keep scalar couplings consistent with this matrix.
        # V_cap bounds individual V_unique[q], not their FS-weighted double sums; clip W_ortho before diagonalisation so lambda_lin_max, eigvecs, and the couplings entering the gap equation share the same bounded interaction.
        W_ortho = S_inv_sqrt @ W_pair @ S_inv_sqrt
        g_sd = math.sqrt(max(g_Delta_s * g_Delta_d, 1e-12))
        W_ortho_capped = np.array([
            [np.clip(W_ortho[0, 0], -V_cap * g_Delta_s, V_cap * g_Delta_s),
             np.clip(W_ortho[0, 1], -V_cap * g_sd,      V_cap * g_sd)],
            [np.clip(W_ortho[1, 0], -V_cap * g_sd,      V_cap * g_sd),
             np.clip(W_ortho[1, 1], -V_cap * g_Delta_d, V_cap * g_Delta_d)],
        ])
        eigvals, eigvecs_ortho = np.linalg.eigh(W_ortho_capped)
        eigvecs = S_inv_sqrt @ eigvecs_ortho

        lambda_lin_max = float(eigvals[-1])
        v_s_raw = float(eigvecs[0, -1])
        v_d_raw = float(eigvecs[1, -1])

        # --- channel scalars (physically weighted averages in eV dimension) ---
        V_s_scalar  = float(np.clip((phi_s_d * inv_vF) @ V_ij_jt @ (phi_s_d * inv_vF) / (ns * ns), -V_cap, V_cap))
        V_d_scalar  = float(np.clip((phi_d   * inv_vF) @ V_ij_full @ (phi_d * inv_vF) / (nd * nd), -V_cap, V_cap))
        V_sd_scalar = float(np.clip((phi_s_d * inv_vF) @ V_ij_jt @ (phi_d * inv_vF) / (ns * nd), -V_cap, V_cap))
        # 1D vector projection on the Fermi surface; to keep it in eV dimension, it should be divided by nd (since sum(V*phi*w) is dimensionless)
        V_d_proj = np.clip((phi_d * inv_vF) @ V_ij_full / nd, -V_cap, V_cap)

        # --- V_d EMA: only if solve_state is provided (only in non-linear SCF) ---
        if solve_state is not None:
            if solve_state.V_d_ema is None:
                solve_state.V_d_ema = V_d_scalar
            else:
                sign_flipped = (V_d_scalar * solve_state.V_d_ema < 0.0 and abs(solve_state.V_d_ema) > _V_PREV_SIGN_FLOOR)
                if sign_flipped:
                    if V_d_scalar > 0.0:
                        _ema_w = _EMA_NEW_WEIGHT
                    else:
                        _ema_w = _EMA_NEW_WEIGHT * (_EMA_SIGN_FLIP_W_MIN + (1.0 - _EMA_SIGN_FLIP_W_MIN) / (1.0 + math.exp(-_EMA_SIGN_FLIP_SLOPE * (abs(det_afm_sc) / _DET_SIGN_FLIP_SCALE - 0.5))))
                else:
                    kick_boost = 2.0 if solve_state._ema_kick_pending else 1.0
                    _ema_w = min(_EMA_NEW_WEIGHT * kick_boost, 1.0)
                solve_state._ema_kick_pending = False
                solve_state.V_d_ema = (1.0 - _ema_w) * solve_state.V_d_ema + _ema_w * V_d_scalar
                V_d_scalar = solve_state.V_d_ema

        # --- vertex structure flags for SCF log ---
        V_flat = V_ij_full[i_idx, j_idx]
        vmat_low_var   = float(np.std(V_flat)) < _VMAT_LOW_VAR_FRAC * abs(float(np.mean(V_flat))) + 1e-12
        vmat_same_sign = (float(np.min(V_flat)) > 0.0) or (float(np.max(V_flat)) < 0.0)

        # --- q-resolved vertex diagnostics ---
        dq_tol = np.pi / _NK
        afm_mask = (np.abs(unique_q[:, 0] - np.pi) < dq_tol) | (np.abs(unique_q[:, 1] - np.pi) < dq_tol)
        fwd_mask = (np.abs(unique_q[:, 0]) < dq_tol) & (np.abs(unique_q[:, 1]) < dq_tol)
        V_afm_mean = float(np.mean(V_unique[afm_mask])) if afm_mask.any() else float('nan')
        V_fwd_mean = float(np.mean(V_unique[fwd_mask])) if fwd_mask.any() else float('nan')
        V_neg_frac = float(np.mean(V_unique < 0.0))
        
        # --- linearised 2×2 kernel with full weight (for diagnostic only) ---
        W22_JT = g_Delta_d * np.dot(phi_d * inv_vF, np.dot(V_ij_jt, phi_d * inv_vF))
        W_JT = np.array([[W11, W12], [W12, W22_JT]], dtype=float)
        evec_max = np.array([v_s_raw, v_d_raw])
        lambda_JT_kernel = float(evec_max @ W_JT @ evec_max)

        # --- gap vector and channel fractions ---
        gap_vector = v_s_raw * phi_s_d + v_d_raw * phi_d
        w = np.abs([v_s_raw, v_d_raw])
        frac = w / max(w.sum(), 1e-12)

        # --- relative gain from inter‑channel mixing ---
        lambda_s_bare = W11 / ns
        lambda_d_bare = W22 / nd
        max_diag = max(lambda_s_bare, lambda_d_bare)
        lambda_gain_rel = (lambda_lin_max - max_diag) / max(abs(max_diag), 1e-12) if max_diag > 0 else 0.0

        V_spin_mean = float(np.mean(V_spin_u))
        V_JT_mean   = float(np.mean(V_jt_u))
        V_rpa_mean  = float(np.mean(V_unique))

        # ---- assemble vertex cache ----
        vertex_cache = {
            'M':               M,
            'Q':               Q,
            'fs_pts':          fs_pts,
            'vF_arr':          vF_arr,
            'fs_idx':          fs_idx,
            'ev':              ev,
            'ec':              ec,
            'V_s_scalar':      V_s_scalar,
            'V_d_scalar':      V_d_scalar,
            'V_d_proj':        V_d_proj.copy(),
            'V_sd':            V_sd_scalar,
            'vmat_low_var':    vmat_low_var,
            'vmat_same_sign':  vmat_same_sign,
            'V_afm_mean':      V_afm_mean,
            'V_fwd_mean':      V_fwd_mean,
            'V_spin_mean':     V_spin_mean,
            'V_JT_mean':       V_JT_mean,
            'V_rpa_mean':      V_rpa_mean,
            'V_neg_frac':      V_neg_frac,
            'v_s_raw':         v_s_raw,
            'v_d_raw':         v_d_raw,
            'frac':            frac,
            'gap_vector':      gap_vector,
            'K12':             W12 / max(math.sqrt(ns * nd), 1e-12),   # NAIVE (n_sd-blind) cross term, diagnostic/log only -- see V_sd for the corrected, gap-equation-consistent value
            'V_ij_full':       V_ij_full,
            'unique_q':        unique_q,
            'i_idx':           i_idx,
            'j_idx':           j_idx,
            'lambda_lin_max':  lambda_lin_max,
            'lambda_gain_rel': lambda_gain_rel,
            'lambda_JT_kernel':lambda_JT_kernel,
        }
        return vertex_cache
    
    def scf_gap_diagnostics(self, Delta_s: complex, Delta_d: complex, g_Delta_s: float, g_Delta_d: float, vertex_cache: dict = None) -> Dict:
        """Diagnostic post-processing of the linearised gap equation results. Returns a new dictionary that merges the vertex_cache with additional diagnostics"""
        # ---- Extract data from cache ----
        fs_pts = vertex_cache['fs_pts']
        vF_arr = vertex_cache['vF_arr']
        fs_idx = vertex_cache['fs_idx']
        ev     = vertex_cache['ev']
        ec     = vertex_cache['ec']
        K12    = vertex_cache['K12']
        frac   = vertex_cache['frac']

        lambda_lin_max = float(vertex_cache['lambda_lin_max'])
        v_s_raw = float(vertex_cache['v_s_raw'])   # s‑channel weight
        v_d_raw = float(vertex_cache['v_d_raw'])   # d‑channel weight
        vF_mag = np.linalg.norm(vF_arr, axis=1)

        # ---- Derived quantities ----
        lambda_s = g_Delta_s * lambda_lin_max * frac[0]
        lambda_d = g_Delta_d * lambda_lin_max * frac[1]

        # Coherence length — nodal/antinodal decomposition for d-wave validity.
        # For d-wave Δ(k)=Δ_d·(cos kx−cos ky): ξ diverges at nodes, shortest at antinodes.
        # BdG validity (phase coherence) is governed by the NODAL sector (superfluid density ∝ v_F at nodes).
        # ξ_antinodal is diagnostic for BEC–BCS crossover;
        Delta_0 = max(_Ds_mag, 2.0 * _Dd_mag)
        if Delta_0 > 1e-8:
            vF_avg = float(np.mean(vF_mag))

            phi_d_fs  = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
            phi_d_abs = np.abs(phi_d_fs)
            phi_d_max = phi_d_abs.max()

            if phi_d_max > _PHI_D_FLOOR:
                nodal_mask = phi_d_abs < np.percentile(phi_d_abs, _NODAL_REGION_PCTL)
                antinodal_mask = phi_d_abs > np.percentile(phi_d_abs, 100 - _NODAL_REGION_PCTL)

                # Nodal ξ: 25th percentile of |φ_d| as gap scale (conservative).
                if nodal_mask.sum() >= _VERTEX_DIAG_MIN_FS :
                    vF_nodal     = float(np.mean(vF_mag[nodal_mask]))
                    phi_nod_vals = phi_d_abs[nodal_mask]
                    Delta_nodal  = max(_Dd_mag * float(np.percentile(phi_nod_vals, _NODAL_REGION_PCTL)), _MATH_EPS)
                    xi_nodal     = vF_nodal / (np.pi * Delta_nodal)
                else:
                    vF_nodal = vF_avg
                    xi_nodal = vF_avg / (np.pi * max(_Dd_mag * 0.2, _MATH_EPS))

                # Antinodal ξ: gap ≈ Δ_0 at antinodes.
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
            'xi_Gamma6': xi_Gamma6,
            'xi_Gamma7': xi_Gamma7,
            'xi_nodal': xi_nodal,
            'xi_antinodal': xi_antinodal if Delta_0 > 1e-8 else 0.0,
            'valid_BdG': valid_BdG,
            'orbital_selective': orbital_selective,
            'gap_symmetry': gap_symmetry,
        })
        return result
        
    def _compute_F67_singlet(self, ev_all: np.ndarray, ec_all: np.ndarray) -> float:
        """
        Anomalous SC-orbital coherence from off-diagonal BdG amplitudes (u·v)
        This is the anomalous Green's function (Gorkov F-function) in the Γ₆–Γ₇ inter-orbital
        singlet channel.

        Definition:
            F67s = Σ_k (1−2f_n) Re[u*_{6↑} v_{7↓} − u*_{6↓} v_{7↑}]   (per sublattice mean)
        Properties:
            Δ = 0  →  F67s = 0  (selection rule exact in D₄h)
            Δ ≠ 0  →  F67s ≠ 0, SC condensate unlocks B1g JT channel (only Q≠0)
        """
        f_n_all  = _fermi_function(ev_all, self.kT)
        omf_all  = 1.0 - 2.0 * f_n_all
        
        # Sublattice amplitudes: (N_k, _N_ORB, _N_BDG). Local indices 0,1,2,3 within each block still mean Γ6↑,Γ6↓,Γ7a↑,Γ7a↓.
        uA = ec_all[:, 0:6,   :]
        vA = ec_all[:, 12:18, :]
        uB = ec_all[:, 6:12,  :]
        vB = ec_all[:, 18:24, :]

        # Anomalous singlet coherence F_{6↑,7↓} − F_{6↓,7↑} per sublattice.
        # D_on[0,3]=+Δs → F_{6↑,7↓}=uA[:,0,:]*conj(vA[:,3,:])
        # D_on[1,2]=−Δs → F_{6↓,7↑}=uA[:,1,:]*conj(vA[:,2,:]), enters with − (singlet antisymmetry)
        # Spin-triplet terms (same spin: F_{6↑,7↑}, F_{6↓,7↓}) are identically zero for singlet pairing.
        anom_A = np.real(uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :]))
        anom_B = np.real(uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :]))
        F67s = float(
            np.einsum('k,kn,kn->', self.k_weights, omf_all, (anom_A + anom_B)) / 4.0
        )
        return F67s

    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu_guess: float, t_eff: float, g_t: float, g_J: float, F67s_mf: float = 0.0) -> Tuple[float, float]:
        """
        Find the chemical potential μ such that the BdG electron density satisfies n(μ) = 1 − target_doping.
        The cheap estimator: ∂n/∂μ ≈ Σₖ,ₙ wₖ f(Eₙ)[1−f(Eₙ)]/kT
        is the standard BCS approximation. It effectively replaces the exact Hellmann–Feynman slope,
          ∂Eₙ/∂μ = ⟨n|∂H/∂μ|n⟩ = −(Pₙ−Hₙ),  Pₙ+Hₙ=1,
        by −1, assuming purely particle-like BdG states (Pₙ=1). This is exact only for Δ=0, where eigenstates are pure particle or hole branches.
        For finite Δₛ or Δ_d, Bogoliubov mixing gives Pₙ≠1, so ∂Eₙ/∂μ = 1−2Pₙ ≠ −1, and dn/dμ acquires additional eigenvector (Sternheimer-type) contributions omitted by the shortcut.
        Thus, once superconductivity is present, the analytic slope is only approximate: although μ enters H only through the diagonal −μτ_z term, the density's μ-dependence also reflects the evolving u/v coherence factors.
        A plain Newton step can overshoot when the DOS—and hence ∂n/∂μ—changes rapidly (e.g. near a Van Hove singularity). Instead, each step is backtracked (halved) until |n(μ)−target_n| decreases.
        If backtracking fails, the routine switches to a numerical derivative; if convergence is still not achieved within the iteration budget, it falls back to a guaranteed bracketed Brent solve rather than returning an unconverged μ.
        """
        target_n = 1.0 - target_doping
        vbdg = self._get_vbdg()
        _use_numeric_deriv = (abs(Delta_s) + abs(Delta_d)) > _MU_SC_DERIV_THRESH

        def _diag_and_density(mu_val: float) -> Tuple[np.ndarray, np.ndarray, float]:
            ev, ec = np.linalg.eigh(
                vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_n, mu_val, g_t, g_J, F67s_mf, out=vbdg._H_stack)
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
                    f"_find_mu_for_density: failed to bracket the density root after "
                    f"{_MU_NEWTON_MAXIT} Newton iterations (M={_M_repr}, Q={Q:.4f}, "
                    f"|Δ|={abs(Delta_s)+abs(Delta_d):.4f}); returning best estimate μ={mu:.6f} with |n−target|={abs(n_at_mu-target_n):.2e}.",
                    RuntimeWarning,
                )
        return mu, n_at_mu

    def _compute_bdg_free_energy(self, M_channels: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float, Q_Eg2: float, V_s: float = 0.0, V_d: float = 0.0, K_eff_for_free_energy: float = 0.0, K_eff_Eg2_for_free_energy: float = 0.0) -> float:
        """
        Generalized free energy with CHANNEL-PER-SIZE (Γ6, Γ7a, Γ7b) order parameters.
        Grand potential per site computed from the k-space BdG spectrum.

        Ω = (1/2) Σ_{k,n} w_k [E_n f_n − T S(f_n)]
            + |Δ_s|² / V_s    ← condensation correction, s-channel
            + |Δ_d|² / V_d    ← condensation correction, d-channel
            + (K_eff/2) Q²          ← elastic cost, B1g
            + (K_eff_Eg2/2) Q_Eg2²  ← elastic cost, Eg,2 (no B1g-Eg2 cross elastic term is added here;

        Notes on condensation correction:
        - Restores variational stationarity: ∂Ω/∂Δ_ℓ = 0 ↔ Δ_ℓ = g_ℓ · V_ℓ · F_ℓ_BZ (gap equation).
        - V_ℓ > 0 : attractive channel → term positive, energy cost to maintain Δ_ℓ; quasiparticle gain included in Ω_BdG.
        - V_ℓ ≤ 0 : repulsive / absent → Δ_ℓ = 0; term omitted.
        - V_ℓ = None (pre-cache) : fall back to bare JT vertex to allow SCF startup.
        """
        vbdg = self._get_vbdg()
        M_channels = np.asarray(M_channels, dtype=float)
        _tx_b, _ty_b = self.p.effective_hopping_anisotropic(Q)
        _J_A1g_diag, _ = self.p.exchange_channels(Q, n_kspace, _tx_b, _ty_b, g_J)
        _J3 = _channel_J3(_J_A1g_diag)   # (3,) -- csatorna-J-k

        ev_all, ec_all = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M_channels, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, F67s_mf, out=vbdg._H_stack, Q_Eg2=Q_Eg2)
        )

        E_mf_correction = 0.5 * self.p.Z * float(np.dot(_J3, M_channels**2))

        _arg = np.clip(np.abs(ev_all) / self.kT, 0.0, _FERMI_ARG_CLIP)
        Omega_kn = np.minimum(0.0, ev_all) - self.kT * np.log1p(np.exp(-_arg))
        Omega_cell = np.einsum('k,kn->', self.k_weights, Omega_kn)
        Omega_trace = np.einsum('k,k->', self.k_weights, np.real(np.trace(vbdg._H_stack[:, 0:12, 0:12], axis1=1, axis2=2)))

        elastic_energy = 0.5 * K_eff_for_free_energy * Q**2
        if Q_Eg2 != 0.0 or K_eff_Eg2_for_free_energy != 0.0:
            elastic_energy += 0.5 * K_eff_Eg2_for_free_energy * Q_Eg2**2

        condensation = 0.0
        if V_s > 0.0:
            condensation += abs(Delta_s)**2 / V_s
        if V_d > 0.0:
            condensation += abs(Delta_d)**2 / V_d
        return 0.25 * (Omega_cell+Omega_trace) + elastic_energy + condensation + E_mf_correction

    def compute_dF_dM_channels_and_hessian(self, target_doping: float, M_channels: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float, Q_Eg2: float, V_s: float = 0.0, V_d: float = 0.0, K_eff_for_free_energy: float = 0.0, K_eff_Eg2_for_free_energy: float = 0.0, diagonal_only: bool = False, refit_mu: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        3-component gradient (∂F/∂M_Γ6, ∂F/∂M_Γ7a, ∂F/∂M_Γ7b) and Hessian via finite differences.
        diagonal_only=True skips the 3 mixed second-derivative probes (12 F-evals) and returns a diagonal Hessian (Jacobi approximation).
        
        refit_mu=True: Canonical ensemble (fixed particle number). Fits μ and uses the Helmholtz free energy (F = Ω + μN).
        refit_mu=False: Grand Canonical ensemble (fixed μ). Reuses input μ and uses the Grand Potential (Ω).
        """
        M0 = np.asarray(M_channels, dtype=float).copy()
        eps = np.maximum(1e-4, np.abs(M0) * 1e-3)

        def F(Mc: np.ndarray) -> float:
            if refit_mu:
                tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)
                tx, ty = g_t * tx_bare, g_t * ty_bare
                mu_eval, n_k_eval = self._find_mu_for_density(Mc, Q, Delta_s, Delta_d, target_doping, mu, np.sqrt(0.5 * (tx**2 + ty**2)), g_t, g_J, F67s_mf=F67s_mf)
                Omega = self._compute_bdg_free_energy(Mc, Q, Delta_s, Delta_d, n_k_eval, mu_eval, g_t, g_J, F67s_mf, Q_Eg2, V_s, V_d, K_eff_for_free_energy, K_eff_Eg2_for_free_energy)
                return Omega + mu_eval * n_k_eval
            else:
                mu_eval, n_k_eval = mu, n_kspace
                return self._compute_bdg_free_energy(Mc, Q, Delta_s, Delta_d, n_k_eval, mu_eval, g_t, g_J, F67s_mf, Q_Eg2, V_s, V_d, K_eff_for_free_energy, K_eff_Eg2_for_free_energy)

        F0 = F(M0)
        grad = np.zeros(_N_CHANNELS)
        Hess = np.zeros((_N_CHANNELS, _N_CHANNELS))
        Fp = np.zeros(_N_CHANNELS)
        Fm = np.zeros(_N_CHANNELS)

        for c in range(_N_CHANNELS):
            Mp, Mm = M0.copy(), M0.copy()
            Mp[c] += eps[c]; Mm[c] -= eps[c]
            Fp[c], Fm[c] = F(Mp), F(Mm)
            grad[c] = (Fp[c] - Fm[c]) / (2 * eps[c])
            Hess[c, c] = (Fp[c] - 2 * F0 + Fm[c]) / eps[c]**2

        if diagonal_only:
            return grad, Hess

        for a in range(_N_CHANNELS):
            for b in range(a + 1, _N_CHANNELS):
                Mpp, Mmm, Mpm, Mmp = M0.copy(), M0.copy(), M0.copy(), M0.copy()
                Mpp[a] += eps[a]; Mpp[b] += eps[b]
                Mmm[a] -= eps[a]; Mmm[b] -= eps[b]
                Mpm[a] += eps[a]; Mpm[b] -= eps[b]
                Mmp[a] -= eps[a]; Mmp[b] += eps[b]
                val = (F(Mpp) - F(Mpm) - F(Mmp) + F(Mmm)) / (4 * eps[a] * eps[b])
                Hess[a, b] = Hess[b, a] = val
        return grad, Hess

    def compute_cluster_free_energy(self, M_ext: float, Q: float, n_kspace: float, mu: float, tx_bare: float, ty_bare: float, J_A1g_diag: np.ndarray, J_B1g_bare: float, g_J: float, F67s_mf: float, verbose: bool) -> Dict:
        """
        Cluster exact diagonalization for the SC state (Δ≠0) with two-channel (spin + B1g) vertex extraction from a 2x2 plaquette in the full 8x8 site x channel space:

            Γ_ED = χ0_ED^{-1} - χ_ED^{-1},

        and only then projected onto the staggered (spin) / uniform (B1g) subspace.
        This ordering is essential because projection and matrix inversion do not commute.

        Geometry (checkerboard, open plaquette):
            0 --x-- 1
            |       |
            y       y
            |       |
            3 --x-- 2

        Sites 0,2 = sublattice A (sign_M=+1); sites 1,3 = sublattice B (sign_M=-1).
        Bonds: (0,1) x, (1,2) y, (2,3) x, (3,0) y — each carries η=+1 (x) or η=−1 (y) in the B1g channel,
        mirroring the cos(kx)−cos(ky) real-space bond weighting; the A1g (magnetic) channel is direction-independent.

        Every site has exactly 2 intra-cluster NN bonds and 2 external neighbours (Z_eff=Z−2), avoiding
        double-counting of the cluster bonds exactly via H_exch and at mean-field level via the Weiss embedding.

        The local Hamiltonian contains:
        −μ, Δ_CF, AFM Weiss field (J_A1g·sign_M·M_ext, Z_eff neighbours), JT coupling and the anomalous Weiss field (Z_eff·J_B1g·F67s_mf).

        Both the JT and anomalous Weiss terms are multiplied by β_cluster,
        the average downfolding weight, so the cluster sees the same ligand-projected physics as the BdG Hamiltonian.
        """
        # ── 0. Geometry and embedding helpers ─────────────────────────────────
        I6 = np.eye(_N_ORB, dtype=complex)
        _SIGN_M = (+1.0, -1.0, +1.0, -1.0)
        _BONDS  = ((0, 1, +1.0), (1, 2, -1.0), (2, 3, +1.0), (3, 0, -1.0))
        J_bond_M_bare = J_A1g_diag[0]   # single-bond magnetic exchange

        def _embed1(op: np.ndarray, site: int) -> np.ndarray:
            """Embed a 6x6 operator into the 4-site 1296-dim tensor-product space."""
            mats = [I6, I6, I6, I6]
            mats[site] = op.astype(complex)
            return np.kron(np.kron(mats[0], mats[1]), np.kron(mats[2], mats[3]))

        def _embed2(opA: np.ndarray, siteA: int, opB: np.ndarray, siteB: int) -> np.ndarray:
            """Embed a two-site operator opA⊗opB at sites siteA, siteB."""
            mats = [I6, I6, I6, I6]
            mats[siteA] = opA.astype(complex)
            mats[siteB] = opB.astype(complex)
            return np.kron(np.kron(mats[0], mats[1]), np.kron(mats[2], mats[3]))

        def _apply_at_site(evecs_tensor: np.ndarray, op: np.ndarray, site: int) -> np.ndarray:
            """Apply a 6×6 operator to one site-leg of the rank-5 eigenvector tensor."""
            t = np.tensordot(op, evecs_tensor, axes=([1], [site]))
            return np.moveaxis(t, 0, site)

        def _expect_from_applied(evecs_tensor: np.ndarray, applied: np.ndarray) -> np.ndarray:
            """Expectation value per eigenstate from a site-applied operator."""
            return np.einsum('ijkln,ijkln->n', evecs_tensor.conj(), applied).real

        def _site_channel_susceptibility_tensor(evals: np.ndarray, evecs: np.ndarray) -> np.ndarray:
            """
            Exact static Lehmann susceptibility tensor of shape (8,8),
            indexed as k = channel * 4 + site, where channel 0 = S_z, channel 1 = B1g.
            """
            eta = max(_ETA_T_FRAC * self.kT, _ETA_GRID_FLOOR * self.p.t0)
            beta = 1.0 / self.kT
            E = evals - evals[0]
            p = np.exp(np.clip(-beta * E, -700.0, 700.0))
            p /= p.sum()

            # Site-averaged local operators in eigenbasis, with disconnected part removed.
            O_eig = []
            for channel in range(2):
                op_local = np.diag(self.sz_op) if channel == 0 else self.p.B1g_op
                for site in range(4):
                    op_e = evecs.conj().T @ (_embed1(op_local, site) @ evecs)
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
            A = np.asarray(A, dtype=float)
            w, U = np.linalg.eigh(0.5 * (A + A.T))
            scale = max(float(np.max(np.abs(w))), 1.0)
            floor = 1e-12 * scale
            if np.any(np.abs(w) < floor) and verbose:
                print(f"regularising {np.sum(np.abs(w) < floor)} eigenvalues below {floor:.3e}")
            w_inv = np.where(np.abs(w) >= floor, 1.0 / w, 0.0)
            inv_A = (U * w_inv) @ U.T
            return 0.5 * (inv_A + inv_A.T)

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

        def _diag_cluster(Z_val: float, include_trw: bool, include_exchange: bool):
            """Build and diagonalize the cluster Hamiltonian."""
            H = np.zeros((_N_CLUSTER, _N_CLUSTER), dtype=complex)
            M_vec = np.array([M_ext, 0.0, 0.0])

            for site, sign in enumerate(_SIGN_M):
                Hs = self.build_local_hamiltonian_for_bdg(sign, M_vec, J_A1g_diag, mu, Z_val) + H_JT
                if include_trw:
                    Hs -= sign * H_TRW_local
                H += _embed1(0.5 * (Hs + Hs.conj().T), site)

            if include_exchange:
                for i, j, eta in _BONDS:
                    H += (
                        J_bond_M_bare * _embed2(self.multi_op, i, self.multi_op, j)
                        + eta * J_B1g_bare * _embed2(self.p.B1g_op, i, self.p.B1g_op, j)
                    )
            return np.linalg.eigh(0.5 * (H + H.conj().T))
        
        # Local embedding parameters
        Z_eff = self.p.Z - 2   # two intra-cluster neighbours, two external

        # Average downfolding weight: same ligand-projected physics as the BdG Hamiltonian.
        beta_cluster = float(np.mean(
            self.p.wave_function_weight(tx_bare, ty_bare, self.k_points[:, 0], self.k_points[:, 1])
        ))

        # JT coupling and anomalous Weiss field with beta_cluster scaling.
        H_JT = (beta_cluster * (self.p.g_JT * Q) * self.p.B1g_op).astype(complex)
        H_TRW_local = (beta_cluster * (Z_eff * J_B1g_bare * F67s_mf) * self.p.B1g_offdiag).astype(complex)

        # Full cluster spectrum (exchange with Z_eff + anomalous Weiss space)
        evals_full, evecs_full = _diag_cluster(Z_eff, include_trw=True, include_exchange=True)
        # Vertex reference spectra is extracted at F67s=0, so it is a normal-state irreducible interaction appropriate for the lattice Bethe-Salpeter equation.
        evals_vertex_full, evecs_vertex_full = _diag_cluster(self.p.Z, include_trw=False, include_exchange=True)
        # Independent sites: no exact exchange, the site senses the Weiss space of the ENTIRE lattice (Z)
        evals_vertex_0, evecs_vertex_0 = _diag_cluster(self.p.Z, include_trw=False, include_exchange=False)

        # Free energy from the entire cluster
        _bweights = np.exp(np.clip(-(evals_full - evals_full[0]) / self.kT, -700.0, 0.0))
        F_total = evals_full[0] - self.kT * np.log(_bweights.sum())

        # Mean-field double-counting correction
        E_mf_correction = 0.5 * Z_eff * J_bond_M_bare * (M_ext ** 2)

        # Per-site B1g expectation values and fluctuation amplitude
        evecs_tensor = evecs_full.reshape(_N_ORB, _N_ORB, _N_ORB, _N_ORB, evecs_full.shape[1])
        B_applied = [_apply_at_site(evecs_tensor, self.p.B1g_op, s) for s in range(4)]
        b_s = [_expect_from_applied(evecs_tensor, B_applied[s]) for s in range(4)]

        p_full = _bweights / _bweights.sum()
        b_mean = np.array([np.sum(p_full * b_s[s]) for s in range(4)])

        # B1g fluctuation amplitude
        b2_diag = [_expect_from_applied(evecs_tensor, _apply_at_site(B_applied[s], self.p.B1g_op, s)) for s in range(4)]
        b2_mean = np.array([np.sum(p_full * b2_diag[s]) for s in range(4)])
        Q_fluct = float(np.sqrt(max(0.0, np.mean(b2_mean - b_mean**2))))

        # Exact site x channel susceptibilities
        chi0_tensor = _site_channel_susceptibility_tensor(evals_vertex_0, evecs_vertex_0)
        chi_full_tensor = _site_channel_susceptibility_tensor(evals_vertex_full, evecs_vertex_full)

        inv_chi0 = _stable_inverse(chi0_tensor)
        inv_chi_full = _stable_inverse(chi_full_tensor)
        Gamma_ED = inv_chi0 - inv_chi_full
        Gamma_ED = 0.5 * (Gamma_ED + Gamma_ED.T)

        # Only now project onto the uniform q=0 [spin, B1g] channel subspace.
        V_irr = _project_vertex_to_staggered(Gamma_ED)

        # Projected susceptibilities are retained only for diagnostics.
        P = np.zeros((8, 2), dtype=float)
        P[0:4, 0] = 0.5 * np.array([1.0, -1.0, 1.0, -1.0])
        P[4:8, 1] = 0.5
        chi0_loc   = P.T @ chi0_tensor @ P
        chi_loc_full = P.T @ chi_full_tensor @ P
        chi0_loc   = 0.5 * (chi0_loc + chi0_loc.T)
        chi_loc_full = 0.5 * (chi_loc_full + chi_loc_full.T)

        return {
            'F_per_site':   float(F_total / _CLUSTER_SIZE) + E_mf_correction,
            'b_mean':       b_mean,
            'Q_fluct':      Q_fluct,
            'V_irr_QQ':     V_irr[1, 1],
            'chi0_loc':     chi0_loc,
            'chi_loc_full': chi_loc_full,
        }

    def compare_cluster_vs_bdg(self, M: np.ndarray, Q: float, n_kspace: float, tx_bare: float, ty_bare: float, b_mean: np.ndarray, mu: float, g_J: float, ev: np.ndarray, ec: np.ndarray):
        """
        Consistency check: the 2x2 cluster-ED treats the local Hilbert space exactly; the mean-field BdG builds a single-particle spectrum from the SAME local operators.
        If the "well-separated doublet, weak-mixing" premise the multipolar framework rests on is sound, the two independent calculations should predict
        similar site-resolved ⟨B1g⟩ at the SAME (M, Q) point. Large disagreement flags either genuine beyond-mean-field
        correlation physics the BdG treatment misses, or a Weiss-embedding inconsistency between the two calculations that is worth chasing down on its own.
        """
        # De-circularized cluster-ED/BdG comparison: beyond mean-field correlation correction was baked into a 1-particle Hamiltonian parameter, which then re-entered the very cluster-ED Hamiltonian used to measure it next time.
        # g_JT_bare must stay the literal microscopic vertex, so the comparison survives only as a benchmark ratio: how much the correlated (ED) local <B1g> response differs from the mean-field (BdG) one at the same bare coupling.
        _B1g_bdg_benchmark = self.B1g_expectation(tx_bare, ty_bare, (ev, ec))
        _gJT_ED_BdG_benchmark_ratio = float(np.mean(b_mean) / _B1g_bdg_benchmark) if abs(_B1g_bdg_benchmark) > _MATH_EPS else float('nan')
        _scf_log("G_JT-BENCH",
            f"  <B1g>_ED/<B1g>_BdG={_gJT_ED_BdG_benchmark_ratio:.4f} at bare g_JT={self.g_JT_bare:.4f} eV/Å"
            f"  [diagnostic only -- g_JT_bare is NOT rescaled by this ratio]")

        fn = _fermi_function(ev, self.kT)
        fbar = 1.0 - fn
        uA, uB, vA, vB = _get_nambu_spinors(ec)

        def _sandwich(u: np.ndarray) -> np.ndarray:
            return np.real(np.einsum('kan,ab,kbn->kn', u.conj(), self.B1g_op, u))

        bA_per_k = np.sum(_sandwich(uA) * fn + _sandwich(vA) * fbar, axis=1)
        bB_per_k = np.sum(_sandwich(uB) * fn + _sandwich(vB) * fbar, axis=1)
        bA_bdg = float(np.dot(self.k_weights, bA_per_k)) / 2.0
        bB_bdg = float(np.dot(self.k_weights, bB_per_k)) / 2.0

        bA_cl = 0.5 * (b_mean[0] + b_mean[2])
        bB_cl = 0.5 * (b_mean[1] + b_mean[3])

        rel_diff_A = abs(bA_cl - bA_bdg) / max(abs(bA_cl), abs(bA_bdg), 1e-6)
        rel_diff_B = abs(bB_cl - bB_bdg) / max(abs(bB_cl), abs(bB_bdg), 1e-6)
        _scf_log("DOUBLET-CHK",
            f" ⟨B1g⟩_A: cluster={bA_cl:+.4f}  BdG={bA_bdg:+.4f}  rel_A.disagreement={rel_diff_A:.1%} "
            f" ⟨B1g⟩_B: cluster={bB_cl:+.4f}  BdG={bB_bdg:+.4f}  rel_B.disagreement={rel_diff_B:.1%} ")

    def diagnose_doublet_mixing(self, ev: np.ndarray, ec: np.ndarray):
        """
        That low-energy physics lives in well-separated local Γ6/Γ7a/Γ7b Kramers doublets -- by projecting the ACTUAL BdG quasiparticle
        states near the Fermi level onto the local doublet manifolds, rather than assuming the projection is clean.

        (a) Doublet purity:  P_Γ(n,k) = Σ_{a∈Γ} |⟨Γ,a|Ψ_nk⟩|², Gaussian-windowed around E=0 and
            k/band-averaged. purity=1 ⇔ every near-EF state lives entirely in one doublet, as assumed
            throughout; purity<1 signals real leakage into other doublets that the theory neglects.

        (b) B1g_op has BOTH a substantial diagonal ("intra-doublet quadrupolar") part and a substantial off-diagonal (inter-doublet coherence)
            part of COMPARABLE bare magnitude A nonzero ⟨B1g⟩ in a quasiparticle state therefore does NOT by itself prove doublet mixing
            -- it can come entirely from classical doublet-occupation weight with no coherence at all. This isolates the coherence-driven part:
                ⟨B1g⟩_actual    = ⟨Ψ|B1g_op|Ψ⟩                    (full operator, incl. off-diagonal)
                ⟨B1g⟩_diag_only = Σ_Γ P_Γ(n,k)·B1g_op[Γ,Γ]         (same occupations, no coherence)
            residual = ⟨B1g⟩_actual − ⟨B1g⟩_diag_only is attributable ENTIRELY to genuine inter-doublet
            coherence; a residual that is small relative to ⟨B1g⟩_actual means the "well-separated doublet" picture is doing most of the work even where B1g is concerned.
        """
        _therm = _fermi_function(ev, self.kT) * (1.0 - _fermi_function(ev, self.kT)) / self.kT
        w = _therm * self.k_weights[:, None]
        norm = max(float(w.sum()), 1e-300)

        uA, uB, vA, vB = _get_nambu_spinors(ec)
        w_orb = np.abs(uA) ** 2 + np.abs(uB) ** 2 + np.abs(vA) ** 2 + np.abs(vB) ** 2   # (N_k, 6, N_bdg)
        doublets = {'Gamma6': (0, 1), 'Gamma7a': (2, 3), 'Gamma7b': (4, 5)}
        P = {name: w_orb[:, i, :] + w_orb[:, j, :] for name, (i, j) in doublets.items()}
        P_avg = {name: float((w * P[name]).sum()) / norm for name in doublets}
        purity = float((w * np.maximum.reduce(list(P.values()))).sum()) / norm

        def _sandwich(u: np.ndarray) -> np.ndarray:
            return np.real(np.einsum('kan,ab,kbn->kn', u.conj(), self.B1g_op, u))

        B1g_actual_full = _sandwich(uA) + _sandwich(uB) + _sandwich(vA) + _sandwich(vB)   # (N_k, N_bdg)
        B1g_diag_vals = self.B1g_op[[0, 2, 4], [0, 2, 4]]
        B1g_diag_only_full = (P['Gamma6'] * B1g_diag_vals[0] + P['Gamma7a'] * B1g_diag_vals[1]
                               + P['Gamma7b'] * B1g_diag_vals[2])

        B1g_actual = float((w * B1g_actual_full).sum()) / norm
        B1g_diag_only = float((w * B1g_diag_only_full).sum()) / norm

        _scf_log("DOUBLET-MIX",
            f"P_Gamma6'={P_avg['Gamma6']:+.4f}  P_Gamma7a={P_avg['Gamma7a']:+.4f}  P_Gamma7b={P_avg['Gamma7b']:.1%}   "
            f"purity={purity:+.4f}  B1g_actual={B1g_actual:+.4f}  B1g_diag_only={B1g_diag_only:.1%}   "
            f"B1g_coherence_residual={B1g_actual - B1g_diag_only:+.4f}")

    def diagnose_simulated_sc_state(self, M_seed: np.ndarray, Delta_s: float, Delta_d: float, target_doping: float, n_kspace: float, mu: float, g_t: float, g_J: float, chi_SS_q0: float, chi_QQ_q0: float, chi_SS_afm: float, chi_QQ_afm: float, J_eff: float, K_eff_n: float, V_JT_corr: float):
        chi_QQ_n = self._chi_QQ_matrix_elements(M_seed, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J)
        vbdg = self._get_vbdg()
        Q_sc = 6e-3
        F67s = 0.0
        beta_all = self.p.wave_function_weight(self.p.t0, self.p.t0, self.k_points[:, 0], self.k_points[:, 1])
        print("frac_low: ", float(np.mean(beta_all < 0.25)))

        tx_bare_sc, ty_bare_sc = self.p.effective_hopping_anisotropic(Q_sc)
        tx_sc, ty_sc = g_t * tx_bare_sc, g_t * ty_bare_sc
        t_eff_sc = np.sqrt(0.5 * (tx_sc**2 + ty_sc**2))
        M_bdg_sc = M_seed

        for _ in range(10):
            mu, n_kspace = self._find_mu_for_density(M_bdg_sc, Q_sc, Delta_s, Delta_d, target_doping, mu, t_eff_sc, g_t, g_J)
            J_A1g_diag, J_B1g_bare = self.p.exchange_channels(Q_sc, n_kspace, tx_bare_sc, ty_bare_sc, g_J)
            bdg_ev_sc, bdg_ec_sc = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M_bdg_sc, Q_sc, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, F67s, out=vbdg._H_stack))
            M_bdg_sc = vbdg.compute_channel_staggered_magnetizations(Q_sc, Delta_s, Delta_d, mu, bdg_ev_sc, bdg_ec_sc)

        J_eff_sc = self.p.Z * J_A1g_diag[0]
        Gamma_M_sc, V_JT, _ = self._make_vertex_params(target_doping, tx_sc, ty_sc, g_t, J_eff_sc)
        _chi_SS_sc_q0, _chi_SQ_sc_q0, _, _chi_QQ_sc_q0 = self.get_susceptibilities_sc(M_bdg_sc, Q_sc, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M_sc, F67s, (bdg_ev_sc, bdg_ec_sc), apply_diamagnetic_QQ=True)
        _chi_SS_sc_pipi, _chi_SQ_sc_pipi, _, _chi_QQ_sc_pipi = self.get_susceptibilities_sc(M_bdg_sc, Q_sc, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M_sc, F67s, (bdg_ev_sc, bdg_ec_sc), apply_diamagnetic_QQ=True)
        print("M_bdg_sc: ", M_bdg_sc)
        print("stoner-1: ", J_eff * chi_SS_afm - 1)
        print("afm fluct div = ", 1.0 - J_eff_sc * _chi_SS_sc_pipi)
        print("J_eff * chi_SS_q0: ", J_eff * chi_SS_q0)
        print("J_eff_sc * chi_SS_sc_q0: ", J_eff_sc*_chi_SS_sc_q0)

        print("V_JT_corr * chi_QQ_q0 = ", V_JT_corr * chi_QQ_q0)
        F_cluster = self.compute_cluster_free_energy(float(M_bdg_sc[0]), Q_sc, n_kspace, mu, tx_bare_sc, ty_bare_sc, J_A1g_diag, J_B1g_bare, g_J, F67s, verbose=True)
        V_JT_corr_sc = V_JT + F_cluster['V_irr_QQ']
        print("V_JT_corr_sc * chi_QQ_sc_q0 = ", V_JT_corr_sc * _chi_QQ_sc_q0)
        
        chi_QQ_sc = self._chi_QQ_matrix_elements(M_bdg_sc, Q_sc, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, F67s)
        print("chi_QQ_sc_q0 - chi_QQ_q0 = ", _chi_QQ_sc_q0 - chi_QQ_q0)
        print("chi_QQ_sc - chi_QQ_n = ", chi_QQ_sc - chi_QQ_n)
        print("chi_SQ_sc_pipi: ", _chi_SQ_sc_pipi)
        
        print("det_afm: ", self._rpa_det(J_eff, V_JT_corr, chi_SS_afm, 0.0, 0.0, chi_QQ_afm)[0])
        print("det_pomer: ", self._rpa_det(J_eff, V_JT_corr, chi_SS_q0, 0.0, 0.0, chi_QQ_q0)[0])
        print("det_afm_sc: ", self._rpa_det(J_eff_sc, V_JT_corr_sc, _chi_SS_sc_pipi, _chi_SQ_sc_pipi, _chi_SQ_sc_pipi, _chi_QQ_sc_pipi)[0])
        print("det_pomer_sc: ", self._rpa_det(J_eff_sc, V_JT_corr_sc, _chi_SS_sc_q0, _chi_SQ_sc_q0, _chi_SQ_sc_q0, _chi_QQ_sc_q0)[0])

        K_eff_sc, _ = self.compute_K_eff_full(target_doping, M_bdg_sc, Q_sc, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, Gamma_M_sc, F_cluster['V_irr_QQ'], F67s)
        print("K_spont: ", self.g_JT_bare**2 * chi_QQ_q0)
        print("K_eff_n: ", K_eff_n)
        print("K_eff_sc: ", K_eff_sc)
        print("N_EF (≈ χ_SS(q=0)): ", chi_SS_q0)
        print("λ_JT_n  = V_JT * N_EF: ", V_JT_corr * chi_SS_q0)
        chi_tau_val = self._compute_chi_tau(M_bdg_sc, Q_sc, Delta_s, Delta_d, n_kspace, mu, g_t, g_J)['chi_tau_net']
        print("lambda_JT_sc: ", self.g_JT_bare**2 * chi_tau_val / K_eff_sc)

        fs_pts, vF_vec, fs_idx, weights = self._get_fs_points(M_bdg_sc, Q_sc, n_kspace, mu, g_t, g_J)
        vF_mag = np.linalg.norm(vF_vec, axis=1)
        phi_d = np.abs(np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1]))
        mask_nodal = phi_d < np.percentile(phi_d, _NODAL_REGION_PCTL)
        Dd_mag = abs(Delta_d) * float(np.percentile(phi_d[mask_nodal], _NODAL_REGION_PCTL))
        if np.any(mask_nodal):
            vF_nodal = float(np.average(vF_mag[mask_nodal], weights=weights[mask_nodal]))
        else:
            vF_nodal = float(np.average(vF_mag, weights=weights))
        xi_nodal = (vF_nodal / (np.pi * Dd_mag)) if Dd_mag > _MATH_EPS else float('inf')
        print("xi nodal est gap Qsc: ", xi_nodal)

    def refine_M_normal_state(self, target_doping: float, max_iter: int = 35, tol: float = 1e-7, verbose: bool = False) -> Tuple[np.ndarray, float, float]:
        """Fast self-consistent BdG iteration with non-linear Mean-Field acceleration."""
        g_t, g_J, _, _ = self.p.get_gutzwiller_factors(target_doping)
        t_eff = g_t * self.p.t0
        n_kspace = 1.0 - target_doping   # nominal electron density
        mu = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        vbdg = self._get_vbdg()
        kpts = vbdg._kpts
        
        M_current = np.full(_N_CHANNELS, 0.01)
        ev_n, ec_n = np.linalg.eigh(vbdg._build_H_stack(kpts, M_current, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, 0.0, out=vbdg._H_stack))
        J_A1g_diag, _ = self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, g_J)
        J_eff = self.p.Z * J_A1g_diag[0]
        Gamma_M, _, _ = self._make_vertex_params(target_doping, t_eff, t_eff, g_t, J_eff)
        chi_SS_afm, *_ = self.get_susceptibilities_sc(M_current, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, 0.0, (ev_n, ec_n), apply_diamagnetic_QQ=True)
        
        for it in range(max_iter):
            M_old = M_current.copy()
            J_A1g_diag, _ = self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, g_J)
            J_eff = self.p.Z * J_A1g_diag[0]
            M_kick = np.clip(M_current + 0.4 * (np.tanh(J_eff * chi_SS_afm * M_current) - M_current), 0.0, _KICK_M_CLIP_HI)
            mu, n_kspace = self._find_mu_for_density(M_kick, 0.0, 0.0j, 0.0j, target_doping, mu, t_eff, g_t, g_J)
            ev_n, ec_n = np.linalg.eigh(vbdg._build_H_stack(kpts, M_kick, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, 0.0, out=vbdg._H_stack))
            M_current = vbdg.compute_channel_staggered_magnetizations(0.0, 0.0j, 0.0j, mu, ev_n, ec_n)
            Gamma_M, _, _ = self._make_vertex_params(target_doping, t_eff, t_eff, g_t, J_eff)
            chi_SS_afm, *_ = self.get_susceptibilities_sc(M_current, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, 0.0, (ev_n, ec_n), apply_diamagnetic_QQ=True)
            diff = float(np.linalg.norm(M_current - M_old))
            if verbose:
                stoner_crit = J_eff * chi_SS_afm - 1
                print(f"iter {it:2d}: stoner-1 = {stoner_crit:8.6f}, diff = {diff:.2e}, M = {np.array2string(M_current, precision=6)}, mu = {mu:.6f}")

            if diff < tol:
                if verbose: print(f"Converged after {it+1} steps")
                break
        M_current[np.abs(M_current) < 1e-12] = 0.0
        return M_current, n_kspace, mu

    def _scf_jacobi_kick(self, target_doping: float, initial_Delta: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, force_d_wave: bool = False, verbose: bool = False) -> Dict:
        """
        Estimate the dominant Jacobi eigenvalue λ₊ of the two-channel (Δ, Q) SCF map and generate physics-informed seed values for (M, Q, Δ_s, Δ_d).
        Linearised Jacobian of the (Δ, Q) fixed-point map:
            J = [ A   B ]
                [ C   0 ]
        λ₊ = ½[A + √(A²+4BC)]; complex → spectral radius = ½√(A²+|disc|)
        Regimes: λ₊<0.7 subcritical, λ₊∈[0.7,1.4] critical, λ₊>1.4 supercritical.
        """
        vbdg = self._get_vbdg()
        initial_M, n_kspace, mu = self.refine_M_normal_state(target_doping, 35, 1e-7)

        if force_d_wave:
            Delta_s = 0.0j
            Delta_d = complex(initial_Delta)
        else:
            Delta_s = complex(initial_Delta * 0.5)
            Delta_d = complex(initial_Delta * 0.5)

        # --- Anisotropic hopping ---
        _t_eff = g_t * self.p.t0
        bdg_ev_n, bdg_ec_n = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, initial_M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, 0.0, out=vbdg._H_stack))
        J_A1g_diag, J_B1g_bare = self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, g_J)
        M_seed = vbdg.compute_channel_staggered_magnetizations(0.0, 0.0j, 0.0j, mu, bdg_ev_n, bdg_ec_n)

        J_eff = self.p.Z * J_A1g_diag[0]
        Gamma_M, V_JT, _V_cap = self._make_vertex_params(target_doping, _t_eff, _t_eff, g_t, J_eff)
        chi_SS_q0, _, _, chi_QQ_q0 = self.get_susceptibilities_sc(M_seed, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M, 0.0, (bdg_ev_n, bdg_ec_n), apply_diamagnetic_QQ=True)
        chi_SS_afm, _, _, chi_QQ_afm = self.get_susceptibilities_sc(M_seed, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, 0.0, (bdg_ev_n, bdg_ec_n), apply_diamagnetic_QQ=True)
        stoner = J_eff * chi_SS_afm

        # --- chi_tau and Linearised BdG+RPA eigenproblem ---
        Q_probe = _Q_SEED_THR
        chi_tau_val = self._compute_chi_tau(M_seed, Q_probe, Delta_s, Delta_d, n_kspace, mu, g_t, g_J)['chi_tau_net']

        # --- JT stability ---
        # Cluster ED embedding field uses only the leading (Γ6) channel -- it has no separate per-channel Weiss field
        F_cluster = self.compute_cluster_free_energy(float(M_seed[0]), 0.0, n_kspace, mu, self.p.t0, self.p.t0, J_A1g_diag, J_B1g_bare, g_J, 0.0, verbose=True)
        V_JT_corr = V_JT + F_cluster['V_irr_QQ']
        K_eff_n, _ = self.compute_K_eff_full(target_doping, M_seed, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, Gamma_M, F_cluster['V_irr_QQ'])

        if verbose:
            dens = self._compute_orbital_densities(bdg_ev_n, bdg_ec_n)
            total_n = np.sum(dens)
            print(f"Total density from /4 version = {total_n:.6f}, target = { 1.0 - target_doping:.6f}")
            b1g_ratio = float(np.linalg.norm(self.B1g_offdiag)) / max(float(np.linalg.norm(np.diag(self.B1g_op))), _MATH_EPS)
            b1g_weight = float(b1g_ratio / (1.0 + b1g_ratio))
            _scf_log("JACO",
                f"  b1g_weight={b1g_weight:.4f} [{'SC-triggered only' if b1g_weight > 0.90 else 'partial D2h mixing'}]")
            print("V_irr_QQ, V_JT: ", F_cluster['V_irr_QQ'], V_JT)
            print(f"stoner-1 = {stoner - 1:.6f}, initial_M = {np.array2string(initial_M, precision=6)}, M_seed = {np.array2string(M_seed, precision=6)}, mu = {mu:.6f}")
            self.compare_cluster_vs_bdg(M_seed, 0.0, n_kspace, self.p.t0, self.p.t0, F_cluster['b_mean'], mu, g_J, bdg_ev_n, bdg_ec_n)
            self.diagnose_doublet_mixing(bdg_ev_n, bdg_ec_n)
            self.diagnose_simulated_sc_state(M_seed, Delta_s, Delta_d, target_doping, n_kspace, mu, g_t, g_J, chi_SS_q0, chi_QQ_q0, chi_SS_afm, chi_QQ_afm, J_eff, K_eff_n, V_JT_corr)

        # --- Linearised BdG+RPA eigenproblem ---
        _lin_seed = self.compute_pairing_kernel_and_build_cache(M_seed, 0.0, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff, Gamma_M, V_JT, V_JT_corr, _V_cap)

        lambda_lin_max = float(_lin_seed['lambda_lin_max'])
        # ---- Jacobi elements ----
        # The actual pairing is driven by the q≈(π,π) backscattering peak, which is exactly what largest eigenvalue of the full FS kernel captures.
        A = lambda_lin_max
        # Q-mode stiffness based on G₃[2,2] (normal state value, but sufficient in kick)
        D = max(1.0 - V_JT_corr * chi_QQ_q0, _MATH_EPS)  # G22_norm: positive in stable case
        # Coupling estimation from gap-induced B₁g response
        B = math.sqrt(V_JT_corr * abs(chi_tau_val)) * (_t_eff / max(self.p.Delta_CF, 1e-9))
        B = float(np.clip(B, 0.0, 2.0))
        J = np.array([[A,    B],
                      [B,   -D]])

        trace = J[0,0] + J[1,1]
        det   = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        disc  = trace**2 - 4.0*det
        if disc >= 0.0:
            lambda_plus = 0.5 * (trace + math.sqrt(disc))
        else:
            lambda_plus = 0.5 * math.sqrt(trace**2 + abs(disc))
        
        lambda_excess = max(0.0, lambda_plus - 1.0) / lambda_plus

        # --- update Q probe ---
        if lambda_lin_max > 1.0:
            # SC-triggered JT: equilibrium condition K·Q = g_JT·⟨B1g⟩ ≈ g_JT²·χ_τ·Q; gives Q* ≈ g_JT²·χ_τ / K_bare as the natural distortion scale.
            _sign = np.sign(self.p.Delta_B1g_static) if abs(self.p.Delta_B1g_static) > _MATH_EPS else 1.0
            Q_probe = float(np.clip(_sign * _KICK_BOOST_Q * self.g_JT_bare * self.p.lambda_hop * np.sqrt(abs(chi_tau_val / K_eff_n)), -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))
        
        # ---  Early Hessian in the seed neighborhood --- 
        # Use lambda_plus (analytic Jacobi eigenvalue of the (Δ,Q) map) as the pairing-strength indicator for the seed scale
        frac = _lin_seed['frac']
        _hk_early = self.compute_hessian(target_doping, M_seed, Q_probe, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, V_JT, F67s_mf=0.0, Q_Eg2=0.0, vertex_cache=None)
        lambda_min = _hk_early['lambda_min_scaled']

        if lambda_min < 0.0:
            # physical_dir basis: [M_Γ6(0), M_Γ7a(1), M_Γ7b(2), Q(3), Δ(4)]
            step = self._project_kick_from_hessian(
                _hk_early,
                (_EARLY_KICK_BASE + lambda_excess) * min(1.0, abs(lambda_min)),
                sign_ref=Q_probe,
            )

            M_kick = np.clip(M_seed + step[0:3], _KICK_M_CLIP_LO, _KICK_M_CLIP_HI)
            Q_kick = float(np.clip(Q_probe + step[3], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))
            
            Delta_total = abs(Delta_s) + abs(Delta_d)
            new_Delta_total = complex(np.clip(_t_eff * np.exp(-1.0 / max(lambda_plus, 0.1)), _DELTA_ABS_FLOOR, _KICK_DELTA_MAX_FRAC * _t_eff))
            Delta_s_kick = new_Delta_total * frac[0] * (Delta_s / abs(Delta_s) if abs(Delta_s) > _MATH_EPS else 1.0)
            Delta_d_kick = new_Delta_total * frac[1] * (Delta_d / abs(Delta_d) if abs(Delta_d) > _MATH_EPS else 1.0)
        else:
            reduction = _KICK_REDUCTION_AMP * np.maximum(0.0, (M_seed - _KICK_M_EXCESS_CTR)) * max(0.0, (stoner - _KICK_JCHI_EXCESS_CTR)) * lambda_excess
            M_kick = M_seed * (1.0 - reduction)
            Q_kick = Q_probe
            Delta_kick = complex(np.clip( _t_eff * np.exp(-1.0 / max(lambda_plus, 0.1)), _DELTA_ABS_FLOOR, _KICK_DELTA_MAX_FRAC * _t_eff))
            Delta_s_kick = Delta_kick * frac[0]
            Delta_d_kick = Delta_kick * frac[1]

        if force_d_wave:
            Delta_s_kick = 0.0j
        
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q_kick)
        tx, ty = g_t * tx_bare, g_t * ty_bare
        _t_eff = np.sqrt(0.5 * (tx**2 + ty**2))
        mu, n_kspace = self._find_mu_for_density(M_kick, Q_kick, Delta_s_kick, Delta_d_kick, target_doping, mu, _t_eff, g_t, g_J)

        # --- Adaptive mixing ---
        alpha = max(_KICK_MIXING_FLOOR, _MIXING / (1 + _KICK_MIXING_SCALE * np.log1p(lambda_plus)))

        lambda_JT_kernel = float(_lin_seed['lambda_JT_kernel'])
        return {
            'n_kspace':         n_kspace,
            'mu':               mu,
            'M_kick':           M_kick,
            'Q_kick':           Q_kick,
            'Delta_s_kick':     Delta_s_kick,
            'Delta_d_kick':     Delta_d_kick,
            'alpha':            alpha,
            'lambda_plus':      lambda_plus,
            'lambda_lin_max':   lambda_lin_max,
            'J_eff':            J_eff,
            't_eff':            _t_eff,
            'jchi_proxy':       stoner,
            'lambda_JT_kernel': lambda_JT_kernel,
            'V_JT':             V_JT,
            'V_JT_corr':        V_JT_corr,
        }

    def _vertex_matrix_at_Q(self, M: np.ndarray, Qv: float, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, Gamma_M: float, V_JT: float, V_JT_corr: float, V_cap: float, det_afm_sc: float, solve_state: '_SolveState') -> Tuple[float, np.array]:
        """
        Evaluate the linearised pairing eigenvalue and diagonal RPA vertex at a trial JT distortion Qv.

        Used by the ∂λ/∂Q finite-difference scan (SC-triggered JT consistency check).
        Builds the full N_FS × N_FS RPA vertex matrix at Qv from the normal-state (Δ=0) susceptibilities

        Returns
        -------
        lambda_lin_max : float  — largest eigenvalue of the linearised gap kernel at Qv.
        V_diag         : ndarray — diagonal of the N_FS × N_FS vertex matrix (for diagnostics).
        """
        tx_b, ty_b = self.p.effective_hopping_anisotropic(Qv)
        J_A1g_diag, _ = self.p.exchange_channels(Qv, n_kspace, tx_b, ty_b, g_J)
        J_eff_v = self.p.Z * J_A1g_diag[0]

        lin = self.compute_pairing_kernel_and_build_cache(M, Qv, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff_v, Gamma_M, V_JT, V_JT_corr, V_cap, det_afm_sc, solve_state)
        return float(lin['lambda_lin_max']), np.diag(lin['V_ij_full'])

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
            # Large amplitude: first-order-like behaviour.
            # Check if last value is near the high end (jumped and held) or alternating.
            _half = _CYCLE_WINDOW // 2
            _early_mean = float(np.mean(arr[:_half]))
            _late_mean  = float(np.mean(arr[_half:]))
            alternating = abs(_late_mean - _early_mean) < 0.3 * std * _half**0.5
            regime = 'hysteretic' if alternating else 'first_order_jump'
        else:
            regime = 'limit_cycle'

        return {'in_cycle': True, 'regime': regime, 'rel_std': rel_std, 'jump_ratio': jump_ratio}

    def solve_self_consistent(self, target_doping: float, initial_Delta: float, verbose: bool = False, _ic_retry: bool = False, force_d_wave: bool = False, Q_Eg2: float = 0.0, force_delta_zero=False, force_Q_zero=False) -> Dict:
        """
        Coupled (M, Q, Δ_s, Δ_d, μ) SCF via Anderson(5)-accelerated fixed-point + LM Newton.

        Order parameters
        ----------------
        M       : staggered AFM magnetisation (Gutzwiller-renormalised BdG Weiss field)
        Q       : B₁g JT lattice distortion (Å); zero in the AFM normal state by symmetry
        Δ_s     : local (Γ₆⊗Γ₇) singlet pairing amplitude (eV), the orbital part is B₁g (off-diagonal in Γ₆/Γ₇), the k-space form factor is constant; this channel couples primarily to the JT phonon (χ_QQ).
        Δ_d     : inter-sublattice d-wave B₁g singlet pairing amplitude (eV); this channel couples to spin fluctuations (χ_SS)
        μ       : chemical potential enforcing ⟨n⟩ = 1 − δ (Newton + Brentq fallback)

        Per-iteration algorithm:
          1. BdG diagonalisation H(k; M,Q,Δ,μ) → observables (M_BdG, ⟨τ_x⟩, ⟨B₁g⟩, pairs).
          2. SC+JT active (Δ>0, Q>0): inject F67s (Gorkov Γ₆–Γ₇ singlet amplitude) into J_B1g off-diagonal Weiss field,
             rebuild BdG cache.
          3. Update K_eff=K_lattice+∂²F_ex/∂Q².
          4. Gap equation Δ_out = g_Δ·V(q)·F_AA/AB via RPA vertex (always from Δ=0 χ₀).
             After fixed-point gap update, blend in the 2×2 pairing kernel eigenvector direction to ensure the SCF can
             always find a slope toward the dominant instability channel even when |Δ| ≈ 0.
             The 2×2 kernel K_pair is built from (V_s, V_d, V_sd) projections;
             its eigenvector defines the optimal (Δ_s*, Δ_d*) ratio that is mixed in with weight _ALPHA_MIX_2X2, preventing artificial channel locking.
          5. ... computed from a cluster Hamiltonian
          6. Newton step for M (LM-damped, trust-region); blended with the *linearly*-mixed BdG
             fixpoint M_bdg (M is deliberately kept OUT of the Anderson history — see step 8).
          7. Q update (damped Hellmann–Feynman Newton step):
             Full force dHdQ_exp = ⟨∂H/∂Q⟩; denominator is the full stiffness K_eff_full, stabilised by an adaptive LM damping μ_LM_Q.
             The trust-region step is Q_out = Q + clip(−dHdQ_exp/(K_eff_full+μ_LM_Q) − Q, ±step_limit_Q).
             Injected into Anderson only for significant displacement, on iteration 0, or every _Q_UPDATE_PERIOD iterations;
             α is capped at _MIXING×0.3 when a genuine Q displacement is injected.
          8. Anderson(5) mix on (Q, |Δ_s|, |Δ_d|) μ enforcing density is then re-solved at the freshly
             mixed (M,Q,Δ) μ is exactly profiled out as a slave variable at every accepted (M,Q,Δ) point
             rather than being an extra, imperfectly converged "fast" degree of freedom whose residual
             error could otherwise leak into the (Q,Δ_s,Δ_d) Anderson history from one iteration to the next.
          9. Adaptive α via Λ_inst = EMA[max(λ_pair, λ_JT, J·χ_SS)]:
             α_eff = α₀/(1+Λ); halved + history reset on divergence;
             past-QCP (det<0): exponential penalty on α; near-QCP: hard cap.
             FS-resolved ∂λ/∂Q computed at post-convergence to confirm SC-triggered JT consistency on hot-spot level.
             Limit-cycle detector: if |Δ| oscillates with relative std > _CYCLE_THRESHOLD,
             α is reduced by _CYCLE_DAMP_FAC and the Anderson history is reset to escape the cycle.

        Converged when max(|ΔM|,|ΔQ|,|ΔΔ_s|,|ΔΔ_d|)<tol and |n−(1−δ)|<10·tol.
        Post-convergence: 3×3 Hessian, linearised λ_max/gap symmetry, χ₀(q_AFM), χ_τ/λ_JT, Λ_inst.
        """
        converged = False
        g_t, g_J, g_Delta_s, g_Delta_d = self.p.get_gutzwiller_factors(target_doping)
        kick = self._scf_jacobi_kick(target_doping, initial_Delta, g_t, g_J, g_Delta_s, g_Delta_d, force_d_wave, verbose)

        M = kick['M_kick']
        Q = kick['Q_kick']
        Delta_s = kick['Delta_s_kick']   # on-site B₁g singlet
        Delta_d = kick['Delta_d_kick']   # inter-site d-wave B₁g
        n_kspace = kick['n_kspace']
        mu = kick['mu']
        _alpha = kick['alpha']
        _lambda_plus = kick['lambda_plus']
        _lambda_lin_max = kick['lambda_lin_max']
        _lambda_JT_kernel = kick['lambda_JT_kernel']
        _J_eff = kick['J_eff']
        _jchi_proxy = kick['jchi_proxy']
        _t_eff_now = kick['t_eff']
        _V_JT = kick['V_JT']
        _V_JT_corr = kick['V_JT_corr']
        _Gamma_M = 0.0
        _V_cap = 0.0
        
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [], 'F_cluster': [], 'mu': [], 'mixing': [],
        }

        if verbose:
            h_afm_M0 = _J_eff * M[0]   # J_eff is Γ6-only; use the leading (Γ6) channel for this diagnostic estimate.

            # --- Regime classification ---
            if _lambda_plus < 0.7:
                regime = 'subcritical'
            elif _lambda_plus <= 1.4:
                regime = 'critical'
            else:
                regime = 'supercritical'
            _retry_flags = (
                f"{'  [force_d_wave]' if force_d_wave else ''}"
                f"{'  [ic_retry]' if _ic_retry else ''}"
            )
            _scf_log("SCF-INIT", f"δ={target_doping:.4f}  M_kick={np.array2string(M, precision=4)}  Q₀={Q:.5f}  |Δ|₀={abs(Delta_s)+abs(Delta_d):.5f}  g_t={g_t:.4f}  g_J={g_J:.4f}  g_Delta_s={g_Delta_s:.4f}  g_Delta_d={g_Delta_d:.4f}{_retry_flags}")
            _scf_log("SCF-INIT", f"h_afm(M₀)={h_afm_M0:.4f} eV  t_eff={_t_eff_now:.4f} eV  {'✓ metallic AFM' if h_afm_M0 < 4.0 * _t_eff_now else '⚠ marginal/insulating'}")
            _scf_log("SCF-INIT", f"λ_JT_kernel={_lambda_JT_kernel:.3f}  [{regime}]  J_eff/Δ_CF={_J_eff / self.p.Delta_CF:.2f}  λ_lin_max={_lambda_lin_max:.3f}  α={_alpha:.4f}")   # prerequisite of the Schrieffer–Wolff transformation:  J_eff/Δ_CF < 0.5

        scf_x_hist: list = []
        scf_f_hist: list = []

        _vertex_cache: Optional[dict] = None
        _max_diff_prev = float('inf')   # previous iteration's max_diff
        _pairing_strength_proxy = 0.0   # must be initialised before loop; only updated when _vertex_cache is not None

        # Initialise Λ_inst from kick proxies; take the most pessimistic channel.
        _Lambda_inst  = float(np.clip(max(_lambda_plus, _lambda_lin_max, _lambda_JT_kernel, _jchi_proxy), 0.0, 10.0))
        
        max_diff = float('inf')
        _stagnation_count   = 0           # consecutive near-stagnation iterations
        _alpha_freeze_count = 0
        _selection_ratio    = 0.0
        _chi_SS_sc_pipi     = 0.0
        _chi_SQ_sc_pipi     = 0.0
        _chi_QQ_sc_pipi     = 0.0
        _chi_SS_sc_q0       = 0.0
        _chi_SQ_sc_q0       = 0.0
        _chi_QQ_sc_q0       = 0.0
        _det_q0_sc  = _DET_AFM_FLOOR
        _det_q0     = _DET_AFM_FLOOR
        _det_afm    = _DET_AFM_FLOOR
        _det_afm_sc = _DET_AFM_FLOOR
        _F67s_mf    = 0.0
        
        _delta_run_hist: list = []
        _scf_dynamics_regime: str = 'converging'
        _ansatz_unstable_ever: bool = False
        _solve_state = _SolveState()
        _vbdg = self._get_vbdg()

        for iteration in range(_MAX_ITER):
            if force_delta_zero:
                Delta_s = 0.0j
                Delta_d = 0.0j
            if force_Q_zero:
                Q = 0.0
                Q_mixed = 0.0

            _iter_t0 = _time.time()

            tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)
            # Bare exchange couplings (single bond, no Z)
            J_A1g_diag, J_B1g_bare = self.p.exchange_channels(Q, n_kspace, tx_bare, ty_bare, g_J)

            # J_eff comes exclusively from the analytic Gutzwiller/Kotliar–Ruckenstein exchange renormalisation
            _J_eff = self.p.Z * J_A1g_diag[0]
            tx, ty = g_t * tx_bare, g_t * ty_bare
            _t_eff_now = np.sqrt(0.5 * (tx**2 + ty**2))
            _Gamma_M, _V_JT, _V_cap = self._make_vertex_params(target_doping, tx, ty, g_t, _J_eff)
            
            F_cluster = self.compute_cluster_free_energy(float(M[0]), Q, n_kspace, mu, tx_bare, ty_bare, J_A1g_diag, J_B1g_bare, g_J, _F67s_mf, verbose)
            _V_JT_corr = _V_JT + F_cluster['V_irr_QQ']

            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                _vbdg._build_H_stack(_vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _F67s_mf, out=_vbdg._H_stack)
                )
            _B1g_expectation = self.B1g_expectation(tx_bare, ty_bare, (_bdg_ev_sc, _bdg_ec_sc))
            
            # M_bdg is the BdG fixed-point candidate; keep M unchanged until the final mixing step.
            M_bdg = _vbdg.compute_channel_staggered_magnetizations(Q, Delta_s, Delta_d, mu, _bdg_ev_sc, _bdg_ec_sc)
            # V_s/V_d/K_eff are irrelevant here: the elastic and condensation terms have zero M-derivatives.
            grad_M, Hess_M = self.compute_dF_dM_channels_and_hessian(target_doping, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _F67s_mf, Q_Eg2, diagonal_only=True, refit_mu=False)

            # SC+JT active: Gorkov singlet amplitude (u·v), Gutzwiller-renormalised, fed back into J_B1g off-diagonal Weiss field. Zero by symmetry when Δ=0 or Q=0.
            Delta_eff_now = abs(Delta_s) + abs(Delta_d)
            if Delta_eff_now > _QQ_DELTA_THRESH:
                _g_eff = (g_Delta_s * abs(Delta_s) + g_Delta_d * abs(Delta_d)) / Delta_eff_now  
                _F67s_mf = _g_eff * self._compute_F67_singlet(_bdg_ev_sc, _bdg_ec_sc)   # F67s receives contributions from BOTH pairing channels
                bdg_ev_sc, bdg_ec_sc = np.linalg.eigh(
                    _vbdg._build_H_stack(_vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _F67s_mf, out=_vbdg._H_stack)
                    )
                # ── SC-state (Δ≠0) RPA determinant ──────────────
                _chi_SS_sc_pipi, _chi_SQ_sc_pipi, _chi_QS_sc_pipi, _chi_QQ_sc_pipi = self.get_susceptibilities_sc(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, _F67s_mf, (bdg_ev_sc, bdg_ec_sc), apply_diamagnetic_QQ=True)
                _det_afm_sc, *_ = self._rpa_det(_J_eff, _V_JT_corr, _chi_SS_sc_pipi, _chi_SQ_sc_pipi, _chi_QS_sc_pipi, _chi_QQ_sc_pipi)

                _chi_SS_sc_q0, _chi_SQ_sc_q0, _chi_QS_sc_q0, _chi_QQ_sc_q0 = self.get_susceptibilities_sc(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, _F67s_mf, (bdg_ev_sc, bdg_ec_sc), apply_diamagnetic_QQ=True)
                _det_q0_sc, *_ = self._rpa_det(_J_eff, _V_JT_corr, _chi_SS_sc_q0, _chi_SQ_sc_q0, _chi_QS_sc_q0, _chi_QQ_sc_q0)
            
            # Gap equation: V(q) always from Δ=0 χ₀; BdG amplitudes (u,v) from SC state.
            Delta_s_out, Delta_d_out, Delta_s7b_diag, Delta_d7b_diag, _vertex_cache = _vbdg.compute_gap_eq_vectorized(M, Q, Delta_s, Delta_d, n_kspace, mu, _t_eff_now, g_t, g_J, g_Delta_s, g_Delta_d, _J_eff, _Gamma_M, _V_JT, _V_JT_corr, _V_cap, _det_afm_sc, _solve_state, _bdg_ev_sc, _bdg_ec_sc, _vertex_cache, False)
            if force_delta_zero:
                Delta_s_out = 0.0j
                Delta_d_out = 0.0j
            
            # Update Q
            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                _vbdg._build_H_stack(_vbdg._kpts, M, Q, Delta_s_out, Delta_d_out, n_kspace, mu, g_t, g_J, _F67s_mf, out=_vbdg._H_stack)
                )
            
            dHdQ = self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J, _F67s_mf)
            dHdQ_diag = np.einsum('kin,kij,kjn->kn', _bdg_ec_sc.conj(), dHdQ, _bdg_ec_sc).real
            f_k = _fermi_function(_bdg_ev_sc, self.kT)
            dHdQ_exp = np.sum(self.k_weights[:, None] * f_k * dHdQ_diag) / 4.0

            # J_A1g_diag (all 3 channels) depends on Q through effective_hopping_anisotropic(Q)
            _eps_dJA1g = math.sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2)
            _tx_p, _ty_p = self.p.effective_hopping_anisotropic(Q + _eps_dJA1g)
            _tx_m, _ty_m = self.p.effective_hopping_anisotropic(Q - _eps_dJA1g)
            _J_A1g_p, _ = self.p.exchange_channels(Q + _eps_dJA1g, n_kspace, _tx_p, _ty_p, g_J)
            _J_A1g_m, _ = self.p.exchange_channels(Q - _eps_dJA1g, n_kspace, _tx_m, _ty_m, g_J)
            # All Γ6/Γ7 channels contribute via d/dQ[0.5*Z*Σ_c J_c*M_c²] = 0.5*Z*Σ_c M_c²*dJ_c/dQ.
            _dJ3_dQ = (_channel_J3(_J_A1g_p) - _channel_J3(_J_A1g_m)) / (2.0 * _eps_dJA1g)
            dHdQ_exp += 0.5 * self.p.Z * float(np.dot(M ** 2, _dJ3_dQ))

            K_eff_Q, _F_bdg_electronic = self.compute_K_eff_full(target_doping, M, Q, Delta_s_out, Delta_d_out, n_kspace, mu, g_t, g_J, _Gamma_M, F_cluster['V_irr_QQ'], _F67s_mf, Q_Eg2, _vertex_cache)

            V_s = (_vertex_cache['V_s_scalar'] if _vertex_cache else _V_JT) * g_Delta_s
            V_d = (_vertex_cache['V_d_scalar'] if _vertex_cache else _V_JT) * g_Delta_d
            F_bdg = (_F_bdg_electronic + 0.5 * self.p.K_lattice * Q**2 + 0.5 * self.p.K_lattice_Eg2 * Q_Eg2**2 + (abs(Delta_s_out)**2 / V_s if V_s > 0 else 0) + (abs(Delta_d_out)**2 / V_d if V_d > 0 else 0))

            # Adaptive LM floor for the Q Hellmann-Feynman step
            #   K_eff_Q >> 0  (deep JT-stable well)       -> mu_LM_Q small  -> near-bare HF step
            #   K_eff_Q ~  0  (JT QCP, chi_QQ softening)  -> mu_LM_Q = _Q_LM_FRAC * _K_bare -> cautious step
            #   K_eff_Q <  0  (past the QCP, SC-induced)  -> mu_LM_Q = |K_eff_Q| + _Q_LM_FRAC*_K_bare -> guarantees (K_eff_Q + mu_LM_Q) > 0
            _mu_LM_Q_base = _Q_LM_FRAC * self._K_bare
            if K_eff_Q > _MATH_EPS:
                _mu_LM_Q = max(_mu_LM_Q_base / (1.0 + K_eff_Q / self._K_bare), _mu_LM_Q_base * 0.1)
            elif K_eff_Q < -_MATH_EPS:
                _mu_LM_Q = abs(K_eff_Q) + _mu_LM_Q_base
            else:
                _mu_LM_Q = _mu_LM_Q_base

            _lm_denom_Q = max(K_eff_Q + _mu_LM_Q, _MATH_EPS)
            _Q_target = - dHdQ_exp / _lm_denom_Q
            _step_limit_Q = max(_TR_Q_STEP_FRAC * self.p.lambda_hop, _TR_Q_STEP_MIN_FLOOR)
            Q_out_raw = Q + float(np.clip(_Q_target - Q, -_step_limit_Q, _step_limit_Q))

            disp_exceeds_tol = abs(Q_out_raw - Q) > _Q_THR_REL * self.p.lambda_hop
            if force_Q_zero:
                Q_out = 0.0
            elif disp_exceeds_tol or (iteration % _Q_UPDATE_PERIOD == 0):
                # Update the Anderson vector only for significant displacements, keeping the Pulay history clean for (Δ, M) without delaying Q's response.
                Q_out = Q_out_raw
            else:
                Q_out = Q
            
            # Update Λ_inst: Rayleigh λ_pair, λ_JT=(g²/K)·χ_QQ, J·χ_SS
            if _vertex_cache is not None:
                g_t, g_J, g_Delta_s, g_Delta_d = self.estimate_gutzwiller_factors_occupation_based(M, Q, n_kspace, mu, g_t, g_J)
                _N0_now = 1.0 / (np.pi * max(g_t * self.p.t0, 1e-6))
                _pairing_strength_proxy = float(np.clip(max(
                    g_Delta_s * max(_vertex_cache['V_s_scalar'], 0.0) * _N0_now,
                    g_Delta_d * max(_vertex_cache['V_d_scalar'], 0.0) * _N0_now
                ), 0.0, 10.0))
                _chi_QQ_vc = float(_vertex_cache.get('chi_QQ_afm', 0.0))
                _lam_JT_vc = float(np.clip(_V_JT_corr * max(_chi_QQ_vc, 0.0), 0.0, 10.0))
                _chi_SS_vc = float(_vertex_cache.get('chi_SS_afm', 0.0))
                _jchi_vc   = float(np.clip(_J_eff * max(_chi_SS_vc, 0.0), 0.0, _JCHI_HARD_REJECT))
                # Unified instability measure: smooth exponential moving average to avoid single-iteration spikes
                _Lambda_raw = float(np.clip(max(_pairing_strength_proxy, _lam_JT_vc, _jchi_vc), 0.0, 10.0))
                _Lambda_inst = (1-_EMA_NEW_WEIGHT) * _Lambda_inst + _EMA_NEW_WEIGHT * _Lambda_raw

            # Update Δ before Anderson mix so M update sees current SC state.
            # Δ Newton step (2×2 analogue of the M Newton step above): uses the SAME Ω(M,Q,Δ_s,Δ_d) functional as compute_hessian, so its stationary point coincides with the gap equation's fixed point (Delta_s_out/Delta_d_out)
            if force_delta_zero:
                Delta_s_newton, Delta_d_newton = 0.0j, 0.0j
            else:
                _grad_s, _grad_d, _H_delta = self.compute_dF_dDelta_and_d2F(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _V_JT, _F67s_mf, Q_Eg2, _vertex_cache)

                # Adaptive LM floor, 2×2 analogue of the M/Q scalar branches (based on λ_min(H_delta)):
                #   λ_min >> 0  (deep SC minimum)        -> mu_LM_Delta small  -> near-bare Newton step
                #   λ_min ~  0  (near Tc / RPA QCP)       -> mu_LM_Delta = _MU_LM_DELTA -> cautious step
                #   λ_min <  0  (saddle, Δ still growing) -> mu_LM_Delta = |λ_min| + _MU_LM_DELTA -> guarantees a PD damped Hessian
                _lam_min_D = float(np.linalg.eigvalsh(_H_delta)[0])
                if _lam_min_D > _MATH_EPS:
                    _mu_LM_D = max(_MU_LM_DELTA / (1.0 + _lam_min_D / max(_t_eff_now, _MATH_EPS)), _MU_LM_DELTA * 0.1)
                elif _lam_min_D < -_MATH_EPS:
                    _mu_LM_D = abs(_lam_min_D) + _MU_LM_DELTA
                else:
                    _mu_LM_D = _MU_LM_DELTA

                _H_damped = _H_delta + _mu_LM_D * np.eye(2)
                try:
                    _delta_step = -np.linalg.solve(_H_damped, np.array([_grad_s, _grad_d]))
                except np.linalg.LinAlgError:
                    _delta_step = np.zeros(2)
                _delta_step = np.clip(_delta_step, -_TR_DELTA_STEP_MAX, _TR_DELTA_STEP_MAX)

                _phase_s = Delta_s / abs(Delta_s) if abs(Delta_s) > _MATH_EPS else 1.0 + 0j
                _phase_d = Delta_d / abs(Delta_d) if abs(Delta_d) > _MATH_EPS else 1.0 + 0j
                Delta_s_newton = complex(np.clip(abs(Delta_s) + _delta_step[0], 0.0, 1.0)) * _phase_s
                Delta_d_newton = complex(np.clip(abs(Delta_d) + _delta_step[1], 0.0, 1.0)) * _phase_d

            Delta_s_fixpoint = self._mix(Delta_s, Delta_s_out, alpha=_alpha)
            Delta_d_fixpoint = self._mix(Delta_d, Delta_d_out, alpha=_alpha)
            Delta_s_mixed = (1.0 - _ALPHA_HF_DELTA) * Delta_s_fixpoint + _ALPHA_HF_DELTA * Delta_s_newton
            Delta_d_mixed = (1.0 - _ALPHA_HF_DELTA) * Delta_d_fixpoint + _ALPHA_HF_DELTA * Delta_d_newton

            # 3D Anderson vector: (Q/λ_hop, |Δ_s|/t₀, |Δ_d|/t₀) — scaled to O(1) for a balanced least-squares solve.
            x_in_3d  = np.array([Q / self.p.lambda_hop,     abs(Delta_s) / self.p.t0,       abs(Delta_d) / self.p.t0])
            x_out_3d = np.array([Q_out / self.p.lambda_hop, abs(Delta_s_mixed) / self.p.t0, abs(Delta_d_mixed) / self.p.t0])
            scf_x_hist.append(x_in_3d)
            scf_f_hist.append(x_out_3d)
            x_new_3d = self._anderson_mix(scf_x_hist, scf_f_hist, m=5, alpha=_alpha)

            # Anderson updates |Δ| magnitude; phase kept from linear mix (BdG phase convention).
            Delta_s_mixed = complex(float(np.clip(x_new_3d[1] * self.p.t0, 0.0, 1.0))) * Delta_s_mixed / (abs(Delta_s_mixed) + 1e-30)
            Delta_d_mixed = complex(float(np.clip(x_new_3d[2] * self.p.t0, 0.0, 1.0))) * Delta_d_mixed / (abs(Delta_d_mixed) + 1e-30)
            Delta_s_abs = abs(Delta_s_mixed)
            Delta_d_abs = abs(Delta_d_mixed)
            _delta_run_hist.append(Delta_s_abs + Delta_d_abs)   # track |Δ| run history for limit-cycle detection.

            Q_mixed = float(np.clip(x_new_3d[0] * self.p.lambda_hop, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # Adaptive LM floor, generalized to the 3×3 channel Hessian (Hess_M) via its MINIMUM eigenvalue --
            # exactly the same recipe as the 2×2 Δ_s/Δ_d block above (_lam_min_D), just for the M-sector:
            #   λ_min(Hess_M) >> 0  (deep stable minimum) → μ_LM small  → fast, lightly-damped Newton step
            #   λ_min(Hess_M) ≈ 0   (flat / near AFM QCP)  → μ_LM = _MU_LM → cautious step, dominated by 1/μ_LM
            #   λ_min(Hess_M) <  0  (saddle / unstable)    → μ_LM = |λ_min| + _MU_LM → guarantees a PD damped Hessian
            _lam_min_M = float(np.linalg.eigvalsh(Hess_M)[0])
            if _lam_min_M > _MATH_EPS:
                _mu_LM_eff = max(_MU_LM / (1.0 + _lam_min_M / _t_eff_now), _MU_LM * 0.1)
            elif _lam_min_M < -_MATH_EPS:
                _mu_LM_eff = abs(_lam_min_M) + _MU_LM
            else:
                _mu_LM_eff = _MU_LM / (1.0 + (Delta_s_abs + Delta_d_abs) / (2*self.p.t0))  # reduce overdamping when Δ grows (SC–AFM coupling unfreezes).
            
            # 1. Regularization of the LM-damped 3×3 Newton system and J_eff thresholds (against Anderson overshoot)
            _H_damped_M = Hess_M + _mu_LM_eff * np.eye(_N_CHANNELS)
            _j_eff_floor = max(abs(_J_eff), _M_J_EFF_FLOOR_FRAC * _t_eff_now, 1e-4)

            # 2. Trust-region upper bound: J/t stiffness cut + curvature-based penalty near QCP
            _cap_stiff = _TR_M_STEP_MAX / max(1.0, abs(_J_eff) / (2.0 * max(_t_eff_now, 1e-6)))
            _cap_curv = 0.5 + 0.5 * (max(_lam_min_M, 0.0) / max(_lam_min_M + _mu_LM_eff, 1e-6))
            _step_upper = float(np.clip(_cap_stiff * _cap_curv, _TR_M_STEP_MIN_FLOOR, _TR_M_STEP_MAX))

            # 3. Enforcing a dynamic step limit between the lower and upper trust-region boundaries (per channel)
            _step_floor = np.maximum(_M_STEP_FLOOR_REL * np.abs(M), _M_STEP_FLOOR_ABS)
            _step_limit = np.clip(max(self.kT, 0.05 * _t_eff_now) / _j_eff_floor, _step_floor, _step_upper)

            # 4. M update and hybrid mixing (linear BdG fixed point + Newton trajectory), per channel
            try:
                _raw_step_M = np.linalg.solve(_H_damped_M, -grad_M)
            except np.linalg.LinAlgError:
                _raw_step_M = -grad_M / max(_mu_LM_eff, 1e-6)
            M_newton = np.clip(M + np.clip(_raw_step_M, -_step_limit, _step_limit), 0.0, 1.0)
            M_fixpoint = self._mix(M, M_bdg, alpha=_alpha)
            M_mixed = np.clip((1.0 - _ALPHA_HF) * M_fixpoint + _ALPHA_HF * M_newton, 0.0, 1.0)

            # _vertex_cache may be None after a Q sign-flip reset.
            _V_s_now = _vertex_cache['V_s_scalar'] if _vertex_cache is not None else 0.0
            _V_d_now = _vertex_cache['V_d_scalar'] if _vertex_cache is not None else 0.0
            _V_sd_now = _vertex_cache['V_sd'] if _vertex_cache is not None else 0.0

            if len(scf_x_hist) > 1 and (Q * Q_mixed < 0):
                scf_x_hist.clear()
                scf_f_hist.clear()
                _vertex_cache = None         # Q sign flip → FS topology may change
                _solve_state.V_d_ema = None  # EMA from old topology is invalid after Q sign flip
                self._chi0_norm_cache = None # χ₀ eigenvectors keyed on Q → must rebuild
                _Lambda_inst = max(_Lambda_inst, 2.0)
            
            mu_new, n_kspace = self._find_mu_for_density(M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, mu, _t_eff_now, g_t, g_J, _F67s_mf)
            
            # max_diff tracks convergence of all OP components (M_Γ6, M_Γ7a, M_Γ7b, Q, Δ_s, Δ_d).
            max_diff = max(
                float(np.max(np.abs(M_mixed - M))),
                abs(Q_mixed - Q),
                abs(Delta_s_abs - abs(Delta_s)),
                abs(Delta_d_abs - abs(Delta_d)),
            )
            
            # SC-triggered JT selection rule proxy:
            #   |Δ|/Δ_CF   — condensate mixing of Γ₆↔Γ₇ (0=normal state, 1=full mixing)
            #   |F67s_mf|  — ANOMALOUS Gorkov singlet amplitude (u·v* cross term)
            _selection_ratio = float(np.clip(
                (Delta_s_abs + Delta_d_abs) / max(self.p.Delta_CF, 1e-9), 0.0, 1.0
            )) * abs(_F67s_mf)

            if iteration >= 5 and iteration % 5 == 0:
                _diverging  = max_diff > _max_diff_prev * _SCF_DIVERGE_RATIO
                _stagnating = max_diff > _max_diff_prev * _SCF_STAGNATE_RATIO and not _diverging

                # Unified α_eff = α₀ / (1 + 0.5·Λ): Λ_inst measures pairing instability, the 0.5 prefactor suppresses overdamping
                _alpha_base = float(np.clip(
                    _MIXING / (1.0 + 0.5 * _Lambda_inst),
                    _MIXING / 6.0,
                    _MIXING,
                ))

                # --- Limit cycle / first-order dynamics detection ---
                _dyn = self._classify_scf_dynamics(_delta_run_hist)
                _in_cycle = _dyn['in_cycle']
                if _in_cycle:
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
                    if _pairing_strength_proxy < 0.3:
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
                    if disp_exceeds_tol:
                        _alpha = min(_alpha, _MIXING * 0.3)

                # --- Alpha freeze recovery ---
                if _alpha_freeze_count >= _SCF_FREEZE_THR and _alpha < _MIXING * _SCF_ALPHA_FREEZE_LO:
                    _alpha = min(_alpha * _SCF_ALPHA_RECOVER, _MIXING * _SCF_ALPHA_FREEZE_HI)
                    _alpha_freeze_count = 0
                    if verbose:
                        _scf_log("SCF", f"↺ α-recovery after freeze → {_alpha:.5f}")
            _max_diff_prev = max_diff            
                
            if _det_afm_sc < 0.0:
                _ansatz_unstable_ever = True   # physical instability: collinear AFM+SC ansatz broke down
                # Past QCP: exponential α penalty ∝ |det_afm|/det_warn; Λ_inst boosted to keep Anderson conservative.
                _det_penalty = float(np.clip(abs(_det_afm_sc) / max(_RPA_DET_WARN, 1e-6), 0.0, 5.0))
                _alpha = float(np.clip(_alpha * math.exp(-_RPA_QCP_PENALTY * _det_penalty), _MIXING / 16.0, _alpha))
                _Lambda_inst = float(np.clip(_Lambda_inst + 1.5 * _det_penalty, 0.0, 10.0))
                # Count iterations where det-penalty pins alpha at floor → enables recovery
                if _alpha <= _MIXING / 16.0 * 1.1:
                    _alpha_freeze_count += 1
            # Near QCP: belt-and-suspenders α cap
            elif (_det_afm_sc < _RPA_DET_WARN) or (abs(_V_d_now) > _V_CUT):
                _alpha = min(_alpha, _MIXING / (1.0 + _Lambda_inst))
            else:
                # Safe zone (det > 0): fast exponential forgetting of past instability
                if _Lambda_inst > 1.0:
                    _Lambda_inst = float(np.clip((1-_EMA_NEW_WEIGHT) * _Lambda_inst, 0.0, 10.0))

            _iter_s = (_time.time() - _iter_t0)

            history['M'].append(M_mixed.copy())
            history['Q'].append(abs(Q_mixed))
            history['Delta'].append(Delta_s_abs + Delta_d_abs)
            history['density'].append(n_kspace)
            history['F_cluster'].append(F_cluster['F_per_site'])
            history['mu'].append(mu_new)
            history['mixing'].append(_alpha)

            # Store current det_afm so next iteration's sign-flip check can compare the cached value with the live value.
            if _vertex_cache is not None:
                _vertex_cache['det_afm_current'] = _det_afm_sc

            if verbose and _vertex_cache is not None:
                _vmat_flags = ""
                if _vertex_cache.get('vmat_low_var', False):
                    _vmat_flags += " ⚠low-var"
                if _vertex_cache.get('vmat_same_sign', False):
                    _vmat_flags += " ⚠same-sign"
                
                # q-resolved vertex flags: appended only when V_d < 0 to keep normal output compact; V_afm < 0: globally repulsive spin channel (unphysical, cross-term dominates).
                _v_afm = _vertex_cache.get('V_afm_mean', float('nan'))
                _v_fwd = _vertex_cache.get('V_fwd_mean', float('nan'))
                _v_neg = _vertex_cache.get('V_neg_frac', float('nan'))
                _vmat_flags += (f" [V_afm={_v_afm:.3f} V_fwd={_v_fwd:.3f} neg={_v_neg:.2f}]")
            
                _scf_log("SCF-I",
                    f"δ={target_doping:.2f} {iteration+1:3d}/{_MAX_ITER}"
                    f"  conv={max_diff:.1e}  M={np.array2string(M, precision=3)}  Q={Q:+.4f}"
                    f"  |Δ|={(abs(Delta_s)+abs(Delta_d))*1000:.2f} meV"
                    f"  J_eff={_J_eff:.4f} eV  mu={mu_new:.5f}  g_t={g_t:.4f}  g_J={g_J:.4f}"
                    f"  J*χSS(q=0)={_J_eff * _vertex_cache['chi_SS_q0']:.4f}  J*χSS_sc(q=0)={_J_eff * _chi_SS_sc_q0:.4f}  V_JT*χQQ(q=0)={_V_JT_corr * _vertex_cache['chi_QQ_q0']:.4f}  V_JT*χQQ_sc(q=0)={_V_JT_corr * _chi_QQ_sc_q0:.4f}"
                    f"  F_bdg={F_bdg:.4f} eV  F_cluster={F_cluster['F_per_site']:.4f} eV")
                _scf_log("SCF-II",
                    f"  dFM_sc={_det_q0_sc:.4f}  dAFM={_vertex_cache['det_afm']:.4f}  dAFM_sc={_det_afm_sc:.4f}  χ_SQ_sc(q=π,π)={_chi_SQ_sc_pipi:.4f}  χ_SQ_sc(q=0)={_chi_SQ_sc_q0:.4f} "
                    f"  Γ_M={_Gamma_M*1000:.2f}meV  α={_alpha:.4f}"
                    f"  δB1g={_B1g_expectation - self.B1g_expectation(tx_bare, ty_bare, self._get_chi0_norm_cache(M, 0.0, n_kspace, mu, g_t, g_J, _vbdg)):+.4f}  F67s={_F67s_mf:+.4f}"
                    f"  V_s={_V_s_now:.3f}  V_d={_V_d_now:.3f}  V_sd={_V_sd_now:.3f}{_vmat_flags}"
                    f"  Q_fluct={F_cluster['Q_fluct']:.3f}  {_iter_s:3.0f}s/it")
            
            # Save the Anderson-mixed values for post-kick convergence check.
            _M_pre_kick  = M_mixed
            _Q_pre_kick  = Q_mixed
            _Ds_pre_kick = Delta_s_mixed
            _Dd_pre_kick = Delta_d_mixed

            M, Q, Delta_s, Delta_d, mu = M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, mu_new

            # Saddle-escape: |Δ|≈0 + negative Hessian eigenvalue → kick along λ_min eigenvector.
            Delta_total = abs(Delta_s) + abs(Delta_d)
            if Delta_total < 5.0 * self.p.tol and iteration > 3 and iteration % 8 == 0:
                _hk = self.compute_hessian(target_doping, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _V_JT, _F67s_mf, Q_Eg2, _vertex_cache)
                
                _lmin_k = _hk['lambda_min_scaled']
                if np.isfinite(_lmin_k) and _lmin_k < 0:
                    _edir_raw = _hk['physical_dir']

                    _wM, _wQ, _wD = float(np.linalg.norm(_edir_raw[0:3])), abs(_edir_raw[3]), abs(_edir_raw[4])
                    _wsum = max(_wM + _wQ + _wD, 1e-12)
                    _fM, _fQ, _fD = _wM / _wsum, _wQ / _wsum, _wD / _wsum    # modes from component fractions

                    if _fD > _MODE_FRAC_DOMINANT:                           _mode = 'pure-SC'
                    elif _fQ > _MODE_FRAC_DOMINANT:                         _mode = 'pure-JT'
                    elif _fD > _MODE_FRAC_MIXED and _fQ > _MODE_FRAC_MIXED: _mode = 'SC-triggered-JT'
                    elif _fM > _MODE_FRAC_DOMINANT:                         _mode = 'AFM-fluctuation'
                    else:                                                   _mode = 'mixed'

                    _kick_damp = 1.0 / (1.0 + _Lambda_inst)   # Kick magnitude Λ-damped: 1/(1+Λ)
                    _curvature = min(abs(_lmin_k), 1.0)
                    _kick_mag  = _KICK_BASE_FRACTION * _kick_damp * _curvature
                    step = self._project_kick_from_hessian(_hk, _kick_mag)

                    # Gentle pull toward the Γ6-only Stoner estimate: detect overshoot via M[0], then shrink all 3 channels uniformly to preserve their relative weights.
                    _stoner_est = float(_J_eff * _vertex_cache['chi_SS_afm']) if _vertex_cache is not None else None
                    _M_phys_est = self.p.estimate_M0(target_doping, _stoner_est, float(M[0]))
                    _m_was_pulled = False
                    if _mode in ('pure-SC', 'SC-triggered-JT') and M[0] > 3.0 * _M_phys_est:
                        _pull_frac = _MODE_PULL_FRAC * _kick_damp
                        _shrink = 1.0 - _pull_frac * (1.0 - _M_phys_est / max(M[0], 1e-12))
                        _M_kick_component = np.clip(M * _shrink, 0.02, M)
                        _m_was_pulled = True
                    else:
                        _M_kick_component = np.clip(M + step[0:3], 0.0, 1.0)
                    Q_kick = float(np.clip(Q + step[3], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

                    # Δ: signed kick along eigenvector component, preserving s/d ratio.
                    _D_kick_signed = max(0.0, Delta_total + step[4])

                    Delta_s_kick = complex(np.clip(_D_kick_signed * _hk['Delta_s_frac'], 0.0, 0.3))
                    Delta_d_kick = complex(np.clip(_D_kick_signed * (1.0 - _hk['Delta_s_frac']), 0.0, 0.3))
                    
                    # Apply kick only if Δ component is significant (>25%)
                    if _fD > 0.25:
                        M = _M_kick_component
                        Q = Q_kick
                        Delta_s = Delta_s_kick
                        Delta_d = Delta_d_kick
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _vertex_cache = None
                        # Mark the EMA as stale so the vertex loop uses a higher blend weight for the first post-kick iteration
                        _solve_state._ema_kick_pending = True
                        if verbose:
                            _scf_log("SADDLE-ESC",
                                f"δ={target_doping:.3f} ⚡kick iter={iteration}  mode={_mode}  λ_min={_lmin_k:+.4f}  Λ_inst={_Lambda_inst:.3f} "
                                f"  damp={_kick_damp:.3f}  fM={_fM:.2f} fQ={_fQ:.2f} fΔ={_fD:.2f}"
                                f"  → M={np.array2string(M, precision=3)} Q={Q:+.4f} |Δ|={_D_kick_signed:.4f}  {' [M-pulled]' if _m_was_pulled else ''}")

            # Re-evaluate convergence against the current (possibly kicked) state.
            _M_post   = abs(M)          - abs(_M_pre_kick)
            _Q_post   = abs(Q)          - abs(_Q_pre_kick)
            _Ds_post  = abs(Delta_s)    - abs(_Ds_pre_kick)
            _Dd_post  = abs(Delta_d)    - abs(_Dd_pre_kick)
            _kick_fired = (float(np.sum(np.abs(_M_post))) + abs(_Q_post) + abs(_Ds_post) + abs(_Dd_post)) > 1e-10
            
            gc.collect()

            _too_early = (iteration < _MIN_ITER)
            if (not _kick_fired
                    and not _too_early
                    and not _det_afm_sc < 0.0
                    and max_diff < self.p.tol
                    and abs(n_kspace - (1 - target_doping)) < self.p.tol * 10):
                converged = True
                break
        
        if not converged and verbose:
            _scf_log("SCF-RES",
                f"δ={target_doping:.4f} ⚠ no conv after {_MAX_ITER} iters"
                f"  max_diff={max_diff:.2e}"
                f"  dens_err={abs(n_kspace-(1-target_doping)):.2e}"
                f"  M={np.array2string(M, precision=4)}  Q={Q:+.4f}  |Δ|={abs(Delta_s)+abs(Delta_d):.4f}"
                f"  dyn={_scf_dynamics_regime}"
                f"{'  [ansatz unstable]' if _ansatz_unstable_ever else ''}")

        # Post-loop diagnostic: λ_max and Rayleigh JT projection and store converged gap and distortion
        J_A1g_diag, J_B1g_bare = self.p.exchange_channels(Q, n_kspace, tx_bare, ty_bare, g_J)
        _J_eff = self.p.Z * J_A1g_diag[0]
        if _vertex_cache is None:
            _vertex_cache = self.compute_pairing_kernel_and_build_cache(M, Q, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _J_eff, _Gamma_M, _V_JT, _V_JT_corr, _V_cap, _det_afm_sc, _solve_state)
        _vertex_cache = self.scf_gap_diagnostics(Delta_s, Delta_d, g_Delta_s, g_Delta_d, _vertex_cache)
        
        if converged:
            hessian_result = self.compute_hessian(target_doping, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _V_JT, _F67s_mf, Q_Eg2, _vertex_cache)
        else:
            hessian_result = {'Delta_s_frac': 0.0, 'F_bdg': 0.0, 'eigenvectors': None, 'eigenvalues': None}

        _chi_tau_result = self._compute_chi_tau(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _F67s_mf)

        _Delta_s_mag = abs(Delta_s)
        _Delta_d_mag = abs(Delta_d)

        # ── FS-resolved ∂λ/∂Q and gap-channel decomposition (SC state only) ──────
        if converged and (_Delta_s_mag + _Delta_d_mag) > _QQ_DELTA_THRESH:
            # Adaptive finite-difference step: 5 % of |Q| protects against noise at larger distortions.
            _dQ_step = max(_DQ_FS_VERTEX, _DQ_FS_VERTEX_FRAC * abs(Q))
            lmax_p, V_diag_p = self._vertex_matrix_at_Q(M, Q + _dQ_step, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _Gamma_M, _V_JT, _V_JT_corr, _V_cap, _det_afm_sc, _solve_state)
            lmax_m, V_diag_m = self._vertex_matrix_at_Q(M, Q - _dQ_step, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _Gamma_M, _V_JT, _V_JT_corr, _V_cap, _det_afm_sc, _solve_state)
            _dlam_dQ_fs = (lmax_p - lmax_m) / (2.0 * _dQ_step)
            if len(V_diag_p) == len(V_diag_m):
                _dV_dQ_diag = (V_diag_p - V_diag_m) / (2.0 * _dQ_step)
                _hot_spot_frac = float(np.mean(_dV_dQ_diag > 0)) if len(_dV_dQ_diag) > 0 else 0.0
            else:
                # FS point count differs at Q±dQ_step (likely a pocket appearing/merging near pinned Q≈0). Keep _hot_spot_frac = 0.0 rather than aborting over this post-hoc diagnostic.
                _scf_log("SCF-RES", f"δ={target_doping:.4f} ⚠ hot-spot diagnostic skipped: "
                                     f"FS point count differs at Q±dQ ({len(V_diag_p)} vs {len(V_diag_m)})")

            # Channel-resolved pairing strength from the 2×2 (s,d) kernel projection — handles s-d mixing exactly via K12, unlike a single-eigenvector scalar split.
            _lam_max_s = _vertex_cache['lambda_s']
            _lam_max_d = _vertex_cache['lambda_d']
            # Eigenvector vs SCF amplitude mismatch: linearised gap eq says d-wave dominates but SCF converged to s-wave (Δ_s >> Δ_d); this indicates that the non-linear fixpoint (large Δ) has quenched the d-wave channel via spectrum depletion.
            _sym_scf = 'd' if _Dd_mag > _Ds_mag else 's'
            _sym_lin = 'd' if abs(v_d_raw) > abs(v_s_raw) else 's'
            _sym_mismatch = _sym_scf != _sym_lin

            if _vertex_cache['frac'] is not None:
                gap_symmetry = 'B1g (d-wave)' if abs(v_d_raw) >= abs(v_s_raw) else 'A1g (s-wave)'

            if _sym_mismatch:
                gap_symmetry = (
                    gap_symmetry.split(' [')[0]
                    + f' [lin={_sym_lin}, SCF={_sym_scf}: nonlinear quench]'
                )
            else:
                gap_symmetry += f' [lin={_sym_lin}, SCF={_sym_scf}: consistent]'

            if verbose:
                _neg_note = (
                    '  [⚠ λ<0: FS-avg repulsive — instability requires nodal sign change]'
                    if _vertex_cache['lambda_lin_max'] < 0 else ''
                )

                _scf_log("SCF-RES",
                    f"δ={target_doping:.4f}"
                    f"  ∂λ/∂Q(FS)={_dlam_dQ_fs:+.4f} eV⁻¹"
                    f"  {'✓ SC condensate enhances pairing at hot spots' if np.isfinite(_dlam_dQ_fs) and _dlam_dQ_fs > 0 else '⚠ SC-JT FS coupling absent'}"
                    f"  hot_spot_frac={_hot_spot_frac:.2f}"
                    f"  {'[concentrated]' if _hot_spot_frac > 0.6 else '[diffuse]' if _hot_spot_frac > 0.3 else '[anti-nodal suppressed]'}")
                _scf_log("SCF-RES",
                    f"δ={target_doping:.4f}"
                    f"  λ_lin_max={_vertex_cache['lambda_lin_max']:.3f} (K12={_vertex_cache['K12']:+.4f})"
                    f"  λ_s={_lam_max_s:.4f}  λ_d={_lam_max_d:.4f}   [{gap_symmetry}]{_neg_note}"
                    f"  |Δs|={_Delta_s_mag*1000:.3f}meV  |Δd|={_Delta_d_mag*1000:.3f}meV"
                    f"  {'[NL s-d MIXING]' if (_lam_max_d > _lam_max_s) and (_Delta_s_mag > 1e-5) and (_Delta_d_mag > 1e-5) else ''}")

            # d-wave free-energy retry when linear kernel and SCF symmetry disagree.
            if _sym_lin == 'd' and _sym_scf != _sym_lin and not _ic_retry:
                _scf_log("SCF-RES", f"δ={target_doping:.4f} → d-wave enforcing retry (symmetry mismatch)")
                try:
                    _d_result = self.solve_self_consistent(target_doping, _Delta_s_mag + _Delta_d_mag, verbose, True, True)
                    if _d_result.get('converged', False):
                        _F_curr  = hessian_result['F_bdg']
                        _F_dwave = _d_result.get('F_bdg', float('inf'))
                        if _F_dwave < _F_curr - 1e-6:
                            _scf_log("SCF-RES",
                                f"δ={target_doping:.4f} d-wave retry: F_d={_F_dwave:.6f} < F_s={_F_curr:.6f} → adopting d-wave")
                            return _d_result
                        else:
                            _scf_log("SCF-RES",
                                f"δ={target_doping:.4f} d-wave retry: F_d={_F_dwave:.6f} ≥ F_s={_F_curr:.6f} → keeping s-wave")
                except Exception as _d_retry_err:
                    _scf_log("SCF-RES", f"δ={target_doping:.4f} d-wave retry failed: {_d_retry_err}")

        # ── Incommensurate nesting scan ──────────────────────────────────────────
        # The scan uses the normal-state (Δ=0) Lindhard kernel over the BdG eigenstates at the converged (M, Q, μ).
        # This is consistent with the Shubnikov-group argument: χ_SQ=0 in the normal state (odd-in-k integrand cancels over the MBZ), so only χ_SS is evaluated here.
        # The BdG H is commensurate (q_AFM=(π,π)); any incommensurate nesting (δq>0) is detected via the Lindhard sum at shifted momenta q=(π, π−δq), keeping Δ=0 in the BdG so that normal-state kinematics apply.
        _ic_dq_max  = 0.0
        _ic_chi_max = 0.0
        _ic_chi_0   = 0.0
        if converged:
            _ic_dq_max, _ic_chi_max, _ic_chi_0 = self._scan_incommensurate_nesting(M, Q, mu, g_t, g_J, n_kspace)

            _ic_flag = 0.05 * np.pi < _ic_dq_max < 0.10 * np.pi
            _scf_log("SCF-RES",
                f"δ={target_doping:.4f}"
                f"  Incommensurate scan: dq*={_ic_dq_max/np.pi:.3f}π"
                f"  χ(0)={_ic_chi_0:.4f}  χ(max)={_ic_chi_max:.4f}"
                f"  {'⚠ incommensurate — auto-retry' if _ic_flag else '✓ commensurate'}")

            # Auto-retry with softened AFM seed (single recursion; _ic_retry blocks re-entry).
            if _ic_flag and not _ic_retry:
                try:
                    _ic_chi_ratio = _ic_chi_max / max(_ic_chi_0, 1e-12)
                    _ic_chi_ratio_clamped = float(np.clip(_ic_chi_ratio, _IC_RATIO_FLOOR, _IC_RATIO_CAP))
                    _scf_log("SCF-RES",
                        f"δ={target_doping:.4f} IC retry:  (χ_ratio={_ic_chi_ratio_clamped:.2f})")
                    _ic_result = self.solve_self_consistent(target_doping, _Delta_s_mag + _Delta_d_mag, verbose, True)
                    if _ic_result.get('converged', False):
                        _scf_log("SCF-RES",
                            f"δ={target_doping:.4f} IC retry converged:"
                            f"  M={np.array2string(np.asarray(_ic_result['M']), precision=4)}  Q={_ic_result['Q']:+.4f}"
                            f"  |Δ|={_ic_result['Delta_s']+_ic_result['Delta_d']:.4f}")
                        _ic_result['incommensurate_dq']        = _ic_dq_max
                        _ic_result['incommensurate_chi_ratio'] = _ic_chi_max / max(_ic_chi_0, 1e-12)
                        return _ic_result
                    else:
                        _F_comm = hessian_result['F_bdg']
                        _F_ic   = _ic_result.get('F_bdg', float('inf'))
                        if _F_ic < _F_comm - 1e-6:
                            _scf_log("SCF-RES",
                                f"δ={target_doping:.4f} IC retry: lower F but not converged"
                                f" — keeping commensurate result (commensurate AFM ansatz only)")
                        else:
                            _scf_log("SCF-RES",
                                f"δ={target_doping:.4f} IC retry: no F improvement — keeping commensurate")
                except Exception as _ic_retry_err:
                    _scf_log("SCF-RES", f"δ={target_doping:.4f} IC retry failed: {_ic_retry_err}")

        # ── SCF convergence summary ──────────────────────────────────────────────
        if verbose and converged:
            _hstr = "H=n/a"
            if hessian_result['eigenvalues'] is not None:
                _eigs = hessian_result['eigenvalues']
                _hstr = (f"H={np.array2string(_eigs, precision=3)}"
                         f"{'✓MIN' if bool(np.all(_eigs > -1e-6)) else '⚠SADDLE'}")
            _scf_log("SCF-RES",
                f"  F_bdg={hessian_result['F_bdg']:.4f}  F_cluster={F_cluster['F_per_site']:.4f}"
                f"  JT={'✓' if _selection_ratio > _JT_ACT_THR else '✗'}  {_hstr}"
                f"  dyn={_scf_dynamics_regime} {'  ⚠ ANSATZ UNSTABLE' if _ansatz_unstable_ever else ''}")
            _dq_str = 'strong' if abs(_chi_SQ_sc_pipi) > 0.10 else ('moderate' if abs(_chi_SQ_sc_pipi) > 0.02 else 'weak')
            _scf_log("SCF-RES",
                f"δ={target_doping:.4f}"
                f"  χ_SQ^SC(π,π)={_chi_SQ_sc_pipi:+.5f} eV⁻¹"
                f"  χ_SQ/χ_SS={_chi_SQ_sc_pipi / _chi_SS_sc_pipi if abs(_chi_SS_sc_pipi) > 1e-12 else float('nan'):+.4f}"
                f"  [{_dq_str}]")

        # ── Post-SCF Mott filter ──────────────────────────────────────────────
        # g_t<0.10 (δ<0.053): incoherent ZRS band; ξ/a<1.0: BEC limit, Cooper pairs not lattice-coherent.
        _mott_suspect = (g_t < _G_T_COHERENCE_MIN) or (_vertex_cache['xi_nodal'] < 1.0)
        if _mott_suspect:
            Delta_s   = 0.0 + 0.0j
            Delta_d   = 0.0 + 0.0j
            converged = False
            if verbose:
                _reason = 'g_t<min (Mott)' if g_t < _G_T_COHERENCE_MIN else f"ξ/a={_vertex_cache['xi_nodal']:.2f}<1 (BEC)"
                _scf_log("SCF-RES", f"δ={target_doping:.4f} ⚠ MOTT-SUSPECT [{_reason}]  g_t={g_t:.3f}  ξ/a={_vertex_cache['xi_nodal']:.2f}  — gap suppressed")
        
        # ── Post-convergence Eg,2 PHONON STABILITY diagnostic
        _chi_Eg2Eg2_final = float(self._chi_QQ_matrix_elements(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _F67s_mf, Q_Eg2, return_matrix=True)[1, 1])
        _G44_final        = 1.0 - _chi_Eg2Eg2_final * self.p.g_Eg2**2 / self.p.K_lattice_Eg2
        _eg2_exp_final    = self.Eg2_expectation((_bdg_ev_sc, _bdg_ec_sc))
        if verbose:
            _scf_log("SCF-EG2",
                f"δ={target_doping:.4f}  Q_Eg2(fixed)={Q_Eg2:+.4f}  <Eg2>={_eg2_exp_final:+.3e}  "
                f"χ_Eg2,Eg2={_chi_Eg2Eg2_final:+.4f}  G44={_G44_final:+.4f} [{'stable' if _G44_final > 0 else 'SOFT Eg2 PHONON'}]")
                
        _K_eff_sc, _ = self.compute_K_eff_full(target_doping, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _Gamma_M, F_cluster['V_irr_QQ'], _F67s_mf, Q_Eg2, _vertex_cache)
        V_irr_QQ_n = self.compute_cluster_free_energy(float(M[0]), 0.0, n_kspace, mu, self.p.t0, self.p.t0, J_A1g_diag, J_B1g_bare, g_J, 0.0, verbose=True)['V_irr_QQ']
        _K_eff_n, _ = self.compute_K_eff_full(target_doping, M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, _Gamma_M, V_irr_QQ_n, _F67s_mf, Q_Eg2, _vertex_cache)

        result = dict(_vertex_cache)
        result.update({
            'M': M,
            'Q': Q,
            'Eg2_expectation': _eg2_exp_final,     # sanity check: should be ~0 for real Delta_s,Delta_d
            'chi_Eg2Eg2': _chi_Eg2Eg2_final,
            'G44_Eg2_stable': bool(_G44_final > 0) if math.isfinite(_G44_final) else None,
            'Delta_s': _Delta_s_mag,
            'Delta_d': _Delta_d_mag,
            'Delta_s7b_diag': Delta_s7b_diag,   # diagnostic-only, NOT self-consistently iterated
            'Delta_d7b_diag': Delta_d7b_diag,
            'chi_tau_sc': _chi_tau_result['chi_tau_sc'],
            'chi_tau_n': _chi_tau_result['chi_tau_n'],
            'chi_tau_net': _chi_tau_result['chi_tau_net'],
            'richardson_ok': _chi_tau_result['richardson_ok'],
            'chi_tau_weight': _chi_tau_result['chi_tau_weight'],   # 1.0=full, 0.5=halved, 0.0=suppressed
            'density': n_kspace,
            'mu': mu,
            'n_kspace': n_kspace,
            'F_bdg': hessian_result['F_bdg'],
            'tx': tx,
            'ty': ty,
            'J_eff': _J_eff,
            'F67s_mf': _F67s_mf,
            'target_doping': target_doping,
            'afm_unstable': _det_afm_sc <= 0.0,
            'selection_ratio': _selection_ratio,
            'history': history,
            'hessian_result': hessian_result,
            'lambda_plus': _lambda_plus,
            'lambda_JT_sc': self.g_JT_bare**2 * _chi_tau_result['chi_tau_net'] / _K_eff_sc,
            'K_eff_net': _K_eff_sc - _K_eff_n,
            'converged': converged,
            'mott_suspect': _mott_suspect,
            'scf_dynamics_regime': _scf_dynamics_regime,   # 'converging'|'limit_cycle'|'first_order_jump'|'hysteretic'
            'ansatz_unstable': _ansatz_unstable_ever,      # True if det(RPA)<0 at any SCF iteration
            'dlam_dQ_fs': _dlam_dQ_fs,
            'incommensurate_dq': _ic_dq_max,
            'incommensurate_chi_ratio': _ic_chi_max / max(_ic_chi_0, 1e-12) if _ic_chi_0 else float('nan'),
        })
        return result

    def _scan_incommensurate_nesting(self, M: np.ndarray, Q: float, mu: float, g_t: float, g_J: float, n_kspace: float) -> Tuple[float, float, float]:
        """
        Incommensurate nesting scan: q* = (π, π−δq) that maximizes χ_SS. The BdG Hamiltonian is fixed to a commensurate q_AFM = (π, π),
        so any incommensurate nesting is only detected here. The auto-retry (softened M) still uses the same BdG and can help only near the threshold (δq ≳ 0.05π);

        Uses the normal-state (Δ=0) Lindhard kernel evaluated at the converged (M, Q, μ).  Consistent with the Shubnikov-group analysis: in the collinear
        AFM normal state χ_SQ vanishes by k → −k symmetry, so only χ_SS is evaluated.  Δ is set to zero in the BdG Hamiltonian so that normal-state
        kinematics apply; Q enters only through the hopping anisotropy tx(Q)/ty(Q).

        Returns
        -------
        ic_dq_max  : δq (in rad) at the χ_SS maximum
        ic_chi_max : χ_SS value at that maximum
        ic_chi_0   : χ_SS at the commensurate point δq = 0
        """
        sz_bdg24 = np.concatenate([
             self.sz_op,   # particle A  (+)
            -self.sz_op,   # particle B  (stagger)
            -self.sz_op,   # hole A      (p-h conjugate)
             self.sz_op,   # hole B      (double flip)
        ])
        _dq_scan = np.linspace(0.0, 0.15 * np.pi, 7)
        _chi_SS_q_scan = []
        _vbdg = self._get_vbdg()
        _Ek_ic, _Vk_ic = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, _vbdg)
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
                _w_ic   = self.k_weights
            else:
                _kpts_q = (self.k_points + _q_ic[None,:] + np.pi) % (2*np.pi) - np.pi
                _Ekq_ic, _Vkq_ic = np.linalg.eigh(
                    _vbdg._build_H_stack(_kpts_q, M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J)
                    )
                _w_ic   = self.k_weights
            _fk  = _fermi_function(_Ek_ic, self.kT)
            _fkq = _fermi_function(_Ekq_ic, self.kT)
            _SzV = sz_bdg24[None,:,None] * _Vkq_ic
            _Mm  = np.einsum('kin,kim->knm', _Vk_ic.conj(), _SzV)
            _M2  = np.abs(_Mm)**2
            _df  = _fk[:,:,None] - _fkq[:,None,:]
            _dE  = _Ekq_ic[:,None,:] - _Ek_ic[:,:,None]
            _msk = (np.abs(_df) > _FD_MASK_DF) & (np.abs(_dE) > _FD_MASK_DE)
            _sdE = np.where(_msk, _dE, 1.0)
            _r   = np.where(_msk, _w_ic[:,None,None]*_M2*_df/_sdE, 0.0)
            _chi_SS_q_scan.append(float(_r.sum()))
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

    def compute_hessian(self, target_doping: float, M_channels: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, V_JT: float, F67s_mf: float, Q_Eg2: float, vertex_cache: dict = None, refit_mu: bool = True) -> Dict:
        """ 
        (3+2)×(3+2) finite-difference Hessian of F(M_Γ6,M_Γ7a,M_Γ7b,Q,Δ), evaluated with Q_Eg2 HELD FIXED.
        
        If refit_mu=True (recommended for final physical stability checks), the chemical potential 
        is re-optimized at every finite-difference point to maintain strict constant density.
        If refit_mu=False (much faster), the Hessian is evaluated in the Grand Canonical ensemble (fixed mu).
        
        Phase handling: if this Hessian collapsed both channels onto abs(Delta)·Delta_{s,d}_frac
        (a real, phase-stripped split) before probing M/Q/Δ, F would be evaluated off the true
        stationary point whenever the converged state carries a nontrivial relative phase.
        """
        vbdg = self._get_vbdg()
        M_channels = np.asarray(M_channels, dtype=float)
        
        # V_s / V_d: full RPA pairing vertex for the condensation correction (or bare V_JT fallback)
        V_s, V_d = self._pairing_strengths(vertex_cache, g_Delta_s, g_Delta_d, V_JT)

        Delta = abs(Delta_s) + abs(Delta_d)
        Delta_s_frac = (abs(Delta_s) / Delta) if Delta > _QQ_DELTA_THRESH else 0.5
        Delta_d_frac = 1.0 - Delta_s_frac
        # Converged relative phase, held fixed for every probe point below. Falls back to 1.0+0j (real, positive) when a channel's amplitude is too small to define a phase —
        phase_s = (Delta_s / abs(Delta_s)) if abs(Delta_s) > 1e-12 else (1.0 + 0j)
        phase_d = (Delta_d / abs(Delta_d)) if abs(Delta_d) > 1e-12 else (1.0 + 0j)

        eps_M = np.maximum(1e-4, np.abs(M_channels) * 1e-3)
        eps_Q = max(1.5e-4, abs(Q) * 1e-3 * self.p.lambda_hop)
        eps_D = min(max(1e-5, abs(Delta) * 1e-3), max(abs(Delta) / 2.0, 1e-10))
        eps5 = np.array([eps_M[0], eps_M[1], eps_M[2], eps_Q, eps_D])

        def F(x5: np.ndarray) -> float:
            m_vals = x5[0:3]
            q_val = x5[3]
            delta_val = x5[4]
            
            ds = phase_s * abs(delta_val) * Delta_s_frac
            dd = phase_d * abs(delta_val) * Delta_d_frac

            if refit_mu:
                # Thermodynamic consistency: adjust mu to hold particle number constant
                tx_bare, ty_bare = self.p.effective_hopping_anisotropic(q_val)
                tx, ty = g_t * tx_bare, g_t * ty_bare
                t_eff = float(np.sqrt(0.5 * (tx**2 + ty**2)))
                mu_eval, n_eval = self._find_mu_for_density(m_vals, q_val, ds, dd, target_doping, mu, t_eff, g_t, g_J, F67s_mf)
                Omega = self._compute_bdg_free_energy(m_vals, q_val, ds, dd, n_eval, mu_eval, g_t, g_J, F67s_mf, Q_Eg2, V_s, V_d, self.p.K_lattice, K_eff_Eg2_for_free_energy=self.p.K_lattice_Eg2)
                return Omega + mu_eval * n_eval
            else:
                # Fast evaluation: holds mu fixed, density fluctuates
                mu_eval, n_eval = mu, n_kspace
                return self._compute_bdg_free_energy(m_vals, q_val, ds, dd, n_eval, mu_eval, g_t, g_J, F67s_mf, Q_Eg2, V_s, V_d, self.p.K_lattice, K_eff_Eg2_for_free_energy=self.p.K_lattice_Eg2)

        x0 = np.array([M_channels[0], M_channels[1], M_channels[2], Q, Delta])
        F0 = F(x0)
        H = np.zeros((5, 5))

        # Diagonal terms
        for i in range(5):
            xp, xm = x0.copy(), x0.copy()
            xp[i] += eps5[i]; xm[i] -= eps5[i]
            H[i, i] = (F(xp) - 2 * F0 + F(xm)) / eps5[i]**2

        # Off-diagonal (mixed) terms
        for i in range(5):
            for j in range(i + 1, 5):
                xpp, xmm, xpm, xmp = x0.copy(), x0.copy(), x0.copy(), x0.copy()
                xpp[i] += eps5[i]; xpp[j] += eps5[j]
                xmm[i] -= eps5[i]; xmm[j] -= eps5[j]
                xpm[i] += eps5[i]; xpm[j] -= eps5[j]
                xmp[i] -= eps5[i]; xmp[j] += eps5[j]
                val = (F(xpp) - F(xpm) - F(xmp) + F(xmm)) / (4 * eps5[i] * eps5[j])
                H[i, j] = H[j, i] = val

        _evals, _evecs = np.linalg.eigh(H)
        
        _tx_bare_h, _ty_bare_h = self.p.effective_hopping_anisotropic(Q)
        _t_eff_h = float(np.sqrt(0.5 * ((g_t*_tx_bare_h)**2 + (g_t*_ty_bare_h)**2)))
        _scales_h = np.array([_KICK_M_CLIP_HI, _KICK_M_CLIP_HI, _KICK_M_CLIP_HI, self.p.lambda_hop, max(_t_eff_h, _MATH_EPS)])
        _S_h = np.diag(_scales_h)
        _evals_scaled, _evecs_scaled = np.linalg.eigh(_S_h @ H @ _S_h)
        _idx_min_scaled = int(np.argmin(_evals_scaled))
        _physical_dir = _scales_h * _evecs_scaled[:, _idx_min_scaled]

        return {
            'Delta_s_frac': Delta_s_frac,
            'F_bdg': F0,
            'eigenvectors': _evecs,
            'eigenvalues': _evals,
            'physical_dir': _physical_dir,
            'lambda_min_scaled': float(_evals_scaled[_idx_min_scaled]),
        }

    def compute_dF_dDelta_and_d2F(self, M_channels: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, V_JT: float, F67s_mf: float, Q_Eg2: float, vertex_cache: dict = None) -> Tuple[float, float, np.ndarray]:
        """Gradient (∂F/∂|Δ_s|, ∂F/∂|Δ_d|) and 2×2 Hessian of F(M_channels, Q, Δ_s, Δ_d) with respect to the two channel AMPLITUDES, relative phases held fixed at their current values."""
        V_s, V_d = self._pairing_strengths(vertex_cache, g_Delta_s, g_Delta_d, V_JT)

        phase_s = (Delta_s / abs(Delta_s)) if abs(Delta_s) > 1e-12 else (1.0 + 0j)
        phase_d = (Delta_d / abs(Delta_d)) if abs(Delta_d) > 1e-12 else (1.0 + 0j)
        ds0, dd0 = abs(Delta_s), abs(Delta_d)

        eps_s = min(max(1e-5, ds0 * 1e-3), max(ds0 / 2.0, 1e-10)) if ds0 > _MATH_EPS else 1e-5
        eps_d = min(max(1e-5, dd0 * 1e-3), max(dd0 / 2.0, 1e-10)) if dd0 > _MATH_EPS else 1e-5

        def F(ds_val, dd_val):
            ds = phase_s * abs(ds_val)
            dd = phase_d * abs(dd_val)
            return self._compute_bdg_free_energy(M_channels, Q, ds, dd, n_kspace, mu, g_t, g_J, F67s_mf, Q_Eg2, V_s, V_d, self.p.K_lattice, K_eff_Eg2_for_free_energy=self.p.K_lattice_Eg2)

        F0  = F(ds0, dd0)
        Fsp = F(ds0 + eps_s, dd0); Fsm = F(ds0 - eps_s, dd0)
        Fdp = F(ds0, dd0 + eps_d); Fdm = F(ds0, dd0 - eps_d)

        grad_s = (Fsp - Fsm) / (2 * eps_s)
        grad_d = (Fdp - Fdm) / (2 * eps_d)
        H_ss = (Fsp - 2*F0 + Fsm) / eps_s**2
        H_dd = (Fdp - 2*F0 + Fdm) / eps_d**2

        F_pp = F(ds0 + eps_s, dd0 + eps_d); F_mm = F(ds0 - eps_s, dd0 - eps_d)
        F_pm = F(ds0 + eps_s, dd0 - eps_d); F_mp = F(ds0 - eps_s, dd0 + eps_d)
        H_sd = (F_pp - F_pm - F_mp + F_mm) / (4 * eps_s * eps_d)
        return float(grad_s), float(grad_d), np.array([[H_ss, H_sd], [H_sd, H_dd]])

    def _mix(self, old, new, alpha=None):
        """
        Linear interpolation:  result = (1 − α)·old + α·new.
        α defaults to the global _MIXING constant when not supplied.
        """
        alpha = _MIXING if alpha is None else alpha
        return (1 - alpha) * old + alpha * new

    def _project_kick_from_hessian(self, hessian_result: Dict, kick_scale: float, sign_ref: Optional[float] = None) -> np.ndarray:
        """
        Convert a compute_hessian() result into a signed step vector [ΔM_Γ6, ΔM_Γ7a, ΔM_Γ7b, ΔQ, ΔΔ]
        along the most-unstable direction, scaled by kick_scale.
        
        sign_ref: if given, flip physical_dir so its Q-component (index 3) matches sign_ref, preserving the external Q_probe/Q sign convention.
        
        NOTE: step[4] is the linear eigenvector projection of Δ. It is used directly for saddle-escape kicks,
        but NOT for cold-start seed sizing: near weak-coupling SC instabilities, Δ ~ exp(-1/λ), so the linear
        projection can severely underestimate the physical seed magnitude.
        """
        edir = np.array(hessian_result['physical_dir'], dtype=float)
        if sign_ref is not None and sign_ref * edir[3] < 0:
            edir = -edir
        return kick_scale * edir

    def compute_Tc_by_gap_suppression(self, doping: float, sc_result: dict, T_min: float = 1e-4, T_max: float = 0.20, n_bracket: int = 12, n_bisect: int = 16, Delta_tol: float = 1e-5, use_free_energy: bool = False, verbose: bool = False) -> dict:
        """
        Starts every temperature from Δ≈0 (cooling from normal state), finding only the spinodal (2nd-order instability boundary).

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

        def gap_at_T(T: float) -> float:
            s = self._clone_solver_at_T(T)
            try:
                res = s.solve_self_consistent(doping, initial_Delta = 3e-8)
                Ds = res['Delta_s']
                Dd = res['Delta_d']
                D  = (Ds**2 + Dd**2) ** 0.5

                if use_free_energy and D > Delta_tol:
                    s_n = s._clone_solver_at_T(T)
                    res_normal = s_n.solve_self_consistent(doping, initial_Delta = 0.0)
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
        Thermodynamic Tc via warm-start upward temperature scan. For the SC-triggered JT mechanism
        the effective Landau potential F_eff(Δ) = a(T)Δ² + [b−γ²/(2K_eff)]Δ⁴+… can have a negative
        quartic term; the transition is then first-order and Tc* occurs where a(T*)>0.
        Cooling from Δ≈0 misses this and returns Tc ≪ Tc*.

        Algorithm: (1) warm-start heating from the T≈0 SC+JT basin; (2) at each T compare F_SC vs F_NM;
        (3) Tc = max(spinodal collapse, thermodynamic crossing); (4) bisection between last SC-wins and first NM-wins T.
        """
        Delta_s0 = sc_result['Delta_s']
        Delta_d0 = sc_result['Delta_d']
        Delta0   = float(np.sqrt(Delta_s0**2 + Delta_d0**2))
        M0       = float(np.asarray(sc_result['M'])[0])   # leading (Γ6) channel, as elsewhere used as the scalar AFM order-parameter proxy

        if (not sc_result.get('converged', False) or Delta0 < Delta_tol):
            return {
                'Tc': 0.0, 'Tc_spinodal': 0.0, 'T_cross': 0.0, 'T_spinodal_cool': float('nan'),
                'transition_order': 'unknown', 'Delta_at_Tc': 0.0, 'Q_at_Tc': 0.0,
                'ratio_2D': 0.0, 'Delta_jump': 0.0, 'hysteresis': 0.0, 'history': []
            }

        def _eval_sc_basin(solver: 'RMFT_Solver', seed_D: float) -> tuple:
            """Returns (Δ_eff, Q_eff, M_eff, F_sc, converged, collapsed); collapsed if Δ<Delta_tol."""
            try:
                res = solver.solve_self_consistent(doping, initial_Delta = seed_D)
                D_eff = float(np.sqrt(res['Delta_s']**2 + res['Delta_d']**2))
                Q_eff = float(res['Q'])
                M_eff = float(np.asarray(res['M'])[0])
                F_sc = float(res['F_bdg'])
                converged = res['converged']
                collapsed = D_eff < Delta_tol
                return D_eff, Q_eff, M_eff, F_sc, converged, collapsed
            except Exception:
                return 0.0, 0.0, 1e30, False, True

        def _eval_normal_basin(solver: 'RMFT_Solver') -> tuple:
            """Returns (F_nm, converged) for the normal-state basin."""
            try:
                res = solver.solve_self_consistent(doping, initial_Delta = 0.0)
                F_nm = float(res['F_bdg'])
                return F_nm, res['converged']
            except Exception:
                return 1e30, False

        # ── Helper: single upward heating pass — crossing + spinodal in one scan ─
        def _find_crossing_and_spinodal(T_vals: list, seed: dict) -> tuple:
            """
            Single pass over ascending T_vals.  Per T-point: one SC-basin SCF and one
            NM-basin SCF (same cost as before, but only once per point instead of twice).

            Crossing (F_SC = F_NM): interpolated between last sc_wins and first sc_loses.
            After the crossing the seed keeps advancing so hysteresis remains trackable.

            Spinodal (SC basin collapses, Delta < Delta_tol): recorded at first collapse;
            pass terminates immediately after. When collapse precedes crossing (2nd-order
            like), T_cross = 0.
            """
            history      = []
            _seed        = seed.copy()
            T_cross      = 0.0;  D_cross      = 0.0;  Q_cross      = 0.0
            T_spinodal   = 0.0;  D_spinodal   = 0.0;  Q_spinodal   = 0.0
            _cross_found = False

            for i, T in enumerate(T_vals):
                s = self._clone_solver_at_T(T)
                D_eff, Q_eff, M_eff, F_sc, _, collapsed = _eval_sc_basin(s, _seed['Delta'])
                F_nm, _ = _eval_normal_basin(s)

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
                        prev         = history[-2]
                        dF_prev      = prev[3] - prev[4]   # F_sc - F_nm at T_{i-1}
                        dF_curr      = F_sc - F_nm
                        denom_interp = dF_curr - dF_prev
                        T_cross = (T_vals[i-1] - dF_prev * (T - T_vals[i-1]) / denom_interp
                                   if abs(denom_interp) > 1e-30 else T_vals[i-1])
                        D_cross      = prev[1]
                        Q_cross      = prev[2]
                        _cross_found = True

                # advance warm seed while SC basin is alive
                _seed['M']     = M_eff
                _seed['Q']     = Q_eff
                _seed['Delta'] = max(D_eff, Delta_tol * 10)
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
                    res = s.solve_self_consistent(doping, initial_Delta = 0.0)
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
        seed_init = {'M': M0, 'Delta': max(Delta0, 1e-3)}

        # ── 2. Single upward heating pass: crossing + spinodal together ──────────
        (T_cross, D_cross, Q_cross,
         T_spinodal, D_spinodal, Q_spinodal,
         heating_history) = _find_crossing_and_spinodal(T_vals, seed_init)

        cross_history  = heating_history
        spinod_history = heating_history

        # ── 3. Ginzburg-Landau extrapolation for precise Tc near second-order transition ──
        def _gl_extrapolate_Tc(history: list) -> float:
            # h[5] = sc_wins in merged history; use h[1] > _GL_DELTA_MIN
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

        _Tc_GL = _gl_extrapolate_Tc(spinod_history)
        # Apply GL refinement only when transition is near-second-order:
        # for first-order transitions Δ² vanishes discontinuously, making linear extrapolation unreliable.
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

        THRESHOLD_RATIO = 0.02  # 2% difference threshold for first-order classification
        if T_cross > 0.0 and T_spinodal > 0.0:
            if T_cross > T_spinodal * (1.0 + THRESHOLD_RATIO):
                transition_order = 'first-order'
                Tc, Delta_at_Tc, Q_at_Tc, Delta_jump = _snap_to_crossing()
            elif T_spinodal > T_cross * (1.0 + THRESHOLD_RATIO):
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
        ratio = (2.0 * Delta0 / Tc) if Tc > 1e-8 else 0.0

        # heating_history is the single shared scan; label each row for clarity.
        combined_history = [('heating',) + h for h in heating_history]
        # cooling history appended separately if available

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
        Linearised gap eigenvalue λ_max(T) across a temperature range.

        Each point: (1) normal-state SCF at temperature T with Δ≡0, so M(T) and Q(T) relax self-consistently
        without a condensate biasing the bands; (2) compute_pairing_kernel_and_build_cache on that normal-state background.
        This avoids the T=0 AFM Weiss field artefact (artificially split bands → λ never reaches 1)

        Diagnostics:
          λ_max(Tc)=1 by definition; slope |dλ/dT|_Tc measures coupling strength.
          Non-monotone λ(T) flags competing orders or SC-JT fluctuation enhancement.
        """
        T_points = np.geomspace(self.kT * 0.25, self.kT * 4.0, 20)
        lam_arr  = np.zeros(len(T_points))
        sym_list = []

        for i, T in enumerate(T_points):
            s_T = self._clone_solver_at_T(T)
            try:
                # Normal-state SCF: Δ strictly zero so M(T), Q(T) relax without condensate.
                res = s_T.solve_self_consistent(doping, initial_Delta = 0.0)
                lam_arr[i] = res['lambda_lin_max']
            except Exception:
                lam_arr[i] = 0.0
        
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
                _scf_log("TC-LAMBDA", f"⚠ non-monotone λ(T): {len(_crossings)} crossings detected (T={[f'{c[0]*1000:.1f}meV' for c in _crossings]}); using lowest-T crossing as Tc.")

        return {
            'T':              T_points,
            'lambda_lin_max': lam_arr,
            'Tc_lambda':      Tc_lambda,
            'slope_at_Tc':    slope_at_Tc,
            'n_crossings':    len(_crossings),
        }

    def compute_G_instability(self, target_doping: float, M: float) -> dict:
        """
        Normal-state (Δ=0) collective instability matrix and diagnostics.
        M here is a SCALAR: this is a standalone, analytic-2-band diagnostic

        Free energy decomposition: F = F_s + F_d + F_Q + F_sQ + F_dQ, giving the
        3×3 Schur-complement instability matrix in dimensionless units:

            G = | 1 − gVs·χ_pair_s    −√(gVs·gVd)·χ_pair_sd    −c_s·χ_SQ_s |
                | −√(gVs·gVd)·χ_pair_sd    1 − gVd·χ_pair_d    −c_d·χ_SQ_d |
                | −c_s·χ_SQ_s              −c_d·χ_SQ_d        1 − K⁻¹·χ_QQ |

        λ_min(G) < 0 signals an instability; the corresponding eigenvector identifies
        the dominant channel (s-pairing, d-pairing, pure JT, or SC-triggered JT).

        Normal-state selection rule: χ_SQ = 0 (both symmetry and analytic 2-band).
        SC state: χ_SQ finite via Bogoliubov mixing.
        """
        # ── 1. Gutzwiller factors and effective hoppings ──────────────────────────
        g_t, g_J, g_Delta_s, g_Delta_d = self.p.get_gutzwiller_factors(target_doping)

        t_eff_now = g_t * self.p.t0
        mu  = -2.0 * t_eff_now * (1.0 - 2.0 * abs(target_doping))
        mu, n_kspace = self._find_mu_for_density(M, 0.0, 0.0j, 0.0j, target_doping, mu, t_eff_now, g_t, g_J)
        J_A1g_diag, _ = self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, g_J)
        J_eff = self.p.Z * J_A1g_diag[0]
        h_afm = J_eff * M
        
        # ── 2. Normal-state band structure (two-band, analytic) ──────────────────
        kx = self.k_points[:, 0]
        ky = self.k_points[:, 1]
        eps_k   = -2.0 * (t_eff_now * np.cos(kx) + t_eff_now * np.cos(ky)) - mu
        eps_kQ  = -eps_k - 2.0 * mu
        xi_avg  =  0.5 * (eps_k + eps_kQ)
        xi_diff =  0.5 * (eps_k - eps_kQ)
        sq      = np.sqrt(xi_diff**2 + h_afm**2 + 1e-20)
        E_plus  = xi_avg + sq
        E_minus = xi_avg - sq

        def _th2E(E):
            a  = np.clip(E / (2.0 * self.kT), -100, 100)
            se = np.where(np.abs(E) > _MATH_EPS, E, _MATH_EPS)
            return np.tanh(a) / (2.0 * se)

        def _mdf(E):
            f_E = 1.0 / (1.0 + np.exp(np.clip(E / self.kT, -100, 100)))
            return f_E * (1.0 - f_E) / self.kT

        pk    = _th2E(E_plus) + _th2E(E_minus)
        phi_s = np.ones_like(kx)
        phi_d = np.cos(kx) - np.cos(ky)

        chi_pair_s = float(np.dot(self.k_weights, pk * phi_s**2))
        chi_pair_d = float(np.dot(self.k_weights, pk * phi_d**2))
        chi_pair_sd = float(np.dot(self.k_weights, pk * phi_s * phi_d))
        N_eff = float(np.dot(self.k_weights, _mdf(E_plus) + _mdf(E_minus)))

        # ── 3. Orbital susceptibilities ───────────────────────────────────────────
        chi_QQ = self._chi_QQ_matrix_elements(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J)

        # ── 4. χ_SQ at q=0 from full Lindhard tensor ─────────────────────────────
        _Gamma_M, _V_JT, _V_cap = self._make_vertex_params(target_doping, t_eff_now, t_eff_now, g_t, J_eff)
        ev, ec = self._get_chi0_norm_cache(M, 0.0, n_kspace, mu, g_t, g_J, self._get_vbdg())
        _, chi_SQ_q0, _, chi_QQ_q0 = self.get_susceptibilities_sc(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
        chi_SS_afm, *_ = self.get_susceptibilities_sc(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
        
        # ── 5. PSD projection of [[χ_pair_s, χ_SQ_q0], [χ_SQ_q0, χ_QQ]] ──────────
        _chi_pair_s_orig = chi_pair_s
        _chi_QQ_orig = chi_QQ
        _chi_SQ_q0_orig = chi_SQ_q0

        gVs = g_Delta_s * _V_JT
        gVd = g_Delta_d * _V_JT
        
        _chi_mat = np.array([[chi_pair_s, chi_SQ_q0],
                             [chi_SQ_q0,  chi_QQ]], dtype=float)

        _eigv, _evc = np.linalg.eigh(_chi_mat)
        _psd_violated = bool(_eigv[0] < -1e-10)

        if _psd_violated:
            _ev_clipped = np.maximum(_eigv, 0.0)
            _mat_psd    = _evc @ np.diag(_ev_clipped) @ _evc.T
            _mat_psd    = 0.5 * (_mat_psd + _mat_psd.T)

            chi_pair_s = _mat_psd[0, 0]
            chi_QQ     = _mat_psd[1, 1]
            chi_SQ_q0  = _mat_psd[0, 1]
            
            _rel_ss = abs(chi_pair_s - _chi_pair_s_orig) / max(abs(_chi_pair_s_orig), 1e-12)
            _rel_qq = abs(chi_QQ - _chi_QQ_orig) / max(abs(_chi_QQ_orig), 1e-12)
            _rel_sq = abs(chi_SQ_q0 - _chi_SQ_q0_orig)  / max(abs(_chi_SQ_q0_orig),  1e-12)
            _scf_log("G-INST",
                f"⚠ χ-matrix PSD violation at q=0: λ_min={_eigv[0]:.3e}"
                f"  (χ_Δs={_chi_pair_s_orig:.4f} χ_QQ={_chi_QQ_orig:.4f}"
                f"  χ_SQ={_chi_SQ_q0_orig:.4f}) → projecting to nearest PSD" + ("  ⚠ >1% change — numerical instability likely" if max(_rel_ss, _rel_qq, _rel_sq) > 0.01 else "  ✓ <1% — minor numerical noise"))
        
        # Normal-state selection rule: χ_SQ = 0 (symmetry + analytic 2-band).
        chi_SQ_s = 0.0
        chi_SQ_d = 0.0
        
        # ── 6. G3 matrix ──────────────────────────────────────────────────────────
        G3 = np.zeros((3, 3))

        G3[0, 0] = 1.0 - gVs * chi_pair_s
        G3[1, 1] = 1.0 - gVd * chi_pair_d
        G3[2, 2] = 1.0 - chi_QQ * _V_JT
        G3[0, 1] = G3[1, 0] = -np.sqrt(max(gVs * gVd, 0.0)) * chi_pair_sd

        G3[0, 2] = G3[2, 0] = -self.g_JT_bare * math.sqrt(max(gVs / self.p.K_lattice, 0.0)) * chi_SQ_s
        G3[1, 2] = G3[2, 1] = -self.g_JT_bare * math.sqrt(max(gVd / self.p.K_lattice, 0.0)) * chi_SQ_d

        # ── 7. InstabilityInfo ───────────────────────────────────────────────────
        instab = InstabilityInfo.from_G3(G3)

        ws, wd, wq = np.abs(instab.evec_min)
        if wd > ws:
            dominant     = 'd'
            chi_pair_dom = chi_pair_d
            chi_SQ_dom   = chi_SQ_d
            V_dom        = gVd
            G11_sc       = instab.G33
            G12_sc       = instab.G_dJT
        else:
            dominant     = 's'
            chi_pair_dom = chi_pair_s
            chi_SQ_dom   = chi_SQ_s
            V_dom        = gVs
            G11_sc       = instab.G11
            G12_sc       = instab.G_sJT
        if wq > ws and wq > wd:
            dominant = 'JT'

        # ── 8. V_eff Schur complement ─────────────────────────────────────────────
        if dominant != 'JT' and instab.G22 > _MATH_EPS:
            V_eff = V_dom + V_dom * (_V_JT * chi_SQ_dom**2) / (max(chi_pair_dom, 1e-12) * instab.G22)
        else:
            V_eff = V_dom
        lambda_eff = N_eff * V_eff

        lambda_lin_max_Q0 = float(self.compute_pairing_kernel_and_build_cache(M, 0.0, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff, _Gamma_M, _V_JT, _V_JT, _V_cap)['lambda_lin_max'])

        H_afm_mat = self.build_local_hamiltonian_for_bdg(1.0, np.array([M, 0.0, 0.0]), J_A1g_diag, mu, self.p.Z)
        comm = self.B1g_op @ H_afm_mat - H_afm_mat @ self.B1g_op
        blocking_ratio = float(np.linalg.norm(comm, 'fro')) / abs(self.p.Delta_CF)

        # ── 11. Assemble result dict ──────────────────────────────────────────────
        result = {
            'M':                self.p.estimate_M0(target_doping, J_eff*chi_SS_afm, M),
            'chi_pair_s':       chi_pair_s,
            'chi_pair_d':       chi_pair_d,
            'chi_pair_sd':      chi_pair_sd,
            'chi_SS_afm':       chi_SS_afm,
            'chi_SQ_s':         chi_SQ_s,
            'chi_SQ_d':         chi_SQ_d,
            'chi_SQ_q0':        chi_SQ_q0,
            'chi_SQ_q0_orig':   _chi_SQ_q0_orig,
            'psd_projected':    _psd_violated,
            'N_eff':            N_eff,
            'h_afm':            h_afm,
            'mu_n':             mu,
            'chi_QQ':           chi_QQ,
            'chi_pair_dom':     chi_pair_dom,
            'chi_SQ_dom':       chi_SQ_dom,
            'dominant':         dominant,
            'E_plus_mean':      np.mean(E_plus),
            'det_G':            float(np.linalg.det(G3)),
            'V_eff':            float(V_eff),
            'lambda_eff':       lambda_eff,
            'sc_triggered_jt':  False,
            'blocking_ratio':   blocking_ratio,
            'g_t':              float(g_t),
            'g_J':              float(g_J),
            'J_eff':            float(J_eff),
            'lambda_lin_max_q0': lambda_lin_max_Q0,
            'instab_info':      instab,
            'instab_type':      instab.instab_type,
            'instab_dir':       instab.instab_dir,
            'instab_weight':    instab.weight_for_score,
            'instab_severity':  instab.severity,
            'dominant_channel': instab.dominant_channel,
            'eigs3':            instab.eigenvalues,
            'evec_min':         instab.evec_min,
            'lambda_min':       instab.lambda_min,
            'G22':              instab.G22,
            'G11':              G11_sc,
            'G12':              G12_sc,
            'K_eff':            self.p.K_lattice,
            }
        return result

    def _get_fs_points(self, M: np.ndarray, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, store_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normal-state Fermi-surface points, vector Hellmann-Feynman velocities,
        full-grid indices and FS integration weights.

        ```
        The FS is extracted at Δ=0. Disconnected pockets are identified in periodic
        k-space, sampled approximately uniformly in arc length, and weighted as
            w_k ~ dl / (BZ_norm * |v_F|).
        """

        _M_key = tuple(np.asarray(M, dtype=float).ravel())
        _cache_key_vals = (_M_key, float(Q), float(n_kspace), float(mu), float(g_t), float(g_J), int(_N_FS))
        vbdg = self._get_vbdg()

        # ---- Cache lookup ----
        if self._fs_cache_dict is not None:
            for key, val in self._fs_cache_dict.items():
                if (np.allclose(key[0], _M_key, atol=_FS_CACHE_TOL, rtol=0.0)
                        and all(abs(float(key[i]) - float(_cache_key_vals[i])) < _FS_CACHE_TOL for i in range(1, 6))
                        and int(key[6]) == int(_N_FS)):
                    return val

        # ---- 1. Normal-state BdG spectrum ----
        ev_all, ec_all = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=vbdg._H_stack))
        ev_pos = np.where(ev_all > 0.0, ev_all, np.inf)
        min_band_idx = np.argmin(ev_pos, axis=1)
        Emin = ev_pos[np.arange(len(ev_pos)), min_band_idx]

        # ---- 2. MBZ deduplication ----
        # For collinear AFM with Q_AFM=(π,π), k and k+Q_AFM are the same physical state.
        # Keeping both corrupts K_dd because φ_d(k+Q) = -φ_d(k) → negative cross-term.
        half_NK = _NK // 2
        k_all = np.asarray(self.k_points)
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
        deg_tol = max(_MBZ_DEGEN_TOL, 0.01 * self.kT)
        mbz_mask_k = np.where(np.abs(dE_partner) < deg_tol, combinatorial_mask, dE_partner < 0.0)

        # ---- 3. Thermal FS shell ----
        f_all = _fermi_function(Emin, self.kT)
        therm = f_all * (1.0 - f_all) / self.kT
        near_fs = therm > (_FS_THERMAL_THRESHOLD / self.kT)
        fs_idx_all = np.flatnonzero(near_fs & mbz_mask_k)
        if len(fs_idx_all) == 0:
            fs_idx_all = np.flatnonzero(near_fs)
        if len(fs_idx_all) == 0:
            n_fallback = min(max(3 * _N_FS, 16), self.N_k)
            fs_idx_all = np.argsort(Emin)[:n_fallback]

        pts_all = k_all[fs_idx_all]
        bands_all = min_band_idx[fs_idx_all]
        evecs_all = ec_all[fs_idx_all]
        n_all = len(pts_all)

        # ---- 4. Hellmann-Feynman Fermi velocities ----
        dk = min(1e-3, max(1e-5, (2.0 * np.pi / _NK) / 10.0))
        kx, ky = pts_all[:, 0], pts_all[:, 1]
        H_buf_p = np.empty((n_all, _N_BDG, _N_BDG), dtype=complex)
        H_buf_m = np.empty((n_all, _N_BDG, _N_BDG), dtype=complex)

        Hp_x = vbdg._build_H_stack(np.column_stack((kx + dk, ky)), M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=H_buf_p).copy()
        Hm_x = vbdg._build_H_stack(np.column_stack((kx - dk, ky)), M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=H_buf_m).copy()
        dH_dkx = (Hp_x - Hm_x) / (2.0 * dk)

        Hp_y = vbdg._build_H_stack(np.column_stack((kx, ky + dk)), M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=H_buf_p).copy()
        Hm_y = vbdg._build_H_stack(np.column_stack((kx, ky - dk)), M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=H_buf_m).copy()
        dH_dky = (Hp_y - Hm_y) / (2.0 * dk)

        psi_all = np.take_along_axis(evecs_all, bands_all[:, None, None], axis=2)[:, :, 0]
        vF_x = np.real(np.einsum("ni,nij,nj->n", psi_all.conj(), dH_dkx, psi_all, optimize=True))
        vF_y = np.real(np.einsum("ni,nij,nj->n", psi_all.conj(), dH_dky, psi_all, optimize=True))
        vF_vec_all = np.column_stack((vF_x, vF_y))

        # ---- 5. Connected components / pocket identification ----
        grid_scale = 2.0 * np.pi / _NK
        pts_box_all = (pts_all + np.pi) % (2.0 * np.pi)
        tree_all = cKDTree(pts_box_all, boxsize=2.0 * np.pi)

        if n_all == 1:
            labels_all = np.zeros(1, dtype=int)
        else:
            k_query = min(5, n_all)
            nn_dist, nn_idx = tree_all.query(pts_box_all, k=k_query)
            nn1 = nn_dist[:, 1] if k_query > 1 else np.array([])
            nn1 = nn1[np.isfinite(nn1) & (nn1 > 0.0)]
            typical_spacing = float(np.median(nn1)) if len(nn1) > 0 else grid_scale
            r_cut = max(3.0 * typical_spacing, 1.5 * grid_scale)

            pairs = tree_all.query_pairs(r=r_cut, output_type="ndarray")
            if len(pairs) == 0:
                labels_all = np.arange(n_all, dtype=int)
            else:
                rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
                cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
                adj = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_all, n_all))
                _, labels_all = connected_components(adj, directed=False)

        # ---- Helper: periodic contour order from local nearest-neighbour graph ----
        def _periodic_delta(a, b):
            return (a - b + np.pi) % (2.0 * np.pi) - np.pi

        def _order_pocket(idx_lab):
            n_p = len(idx_lab)
            if n_p <= 2:
                return idx_lab.copy()

            pts_p = pts_all[idx_lab]
            box_p = (pts_p + np.pi) % (2.0 * np.pi)
            tree_p = cKDTree(box_p, boxsize=2.0 * np.pi)

            k_nn = min(3, n_p)
            d, neigh = tree_p.query(box_p, k=k_nn)

            # Two closest local neighbours define the contour graph.
            neighbours = [[] for _ in range(n_p)]
            for i in range(n_p):
                for j in np.atleast_1d(neigh[i, 1:]):
                    j = int(j)
                    if j != i and j not in neighbours[i]:
                        neighbours[i].append(j)

            # Start with the left-most graph point and greedily continue without
            # immediately backtracking. This uses local geometry, not centroid angle.
            start = int(np.lexsort((pts_p[:, 1], pts_p[:, 0]))[0])
            order_local = [start]
            visited = {start}
            prev = -1
            curr = start

            while len(order_local) < n_p:
                candidates = [j for j in neighbours[curr] if j != prev and j not in visited]

                if not candidates:
                    candidates = [j for j in range(n_p) if j not in visited]

                if prev < 0:
                    nxt = min(candidates, key=lambda j: np.linalg.norm(_periodic_delta(pts_p[j], pts_p[curr])))
                else:
                    incoming = _periodic_delta(pts_p[curr], pts_p[prev])
                    norm_in = np.linalg.norm(incoming)

                    def _score(j):
                        outgoing = _periodic_delta(pts_p[j], pts_p[curr])
                        norm_out = np.linalg.norm(outgoing)
                        if norm_in <= 1e-14 or norm_out <= 1e-14:
                            return np.linalg.norm(outgoing)
                        return -np.dot(incoming, outgoing) / (norm_in * norm_out)

                    nxt = min(candidates, key=_score)

                prev, curr = curr, int(nxt)
                visited.add(curr)
                order_local.append(curr)

            return idx_lab[np.asarray(order_local, dtype=int)]

        # ---- 6. Pocket geometry and proportional sample allocation ----
        pocket_data = []

        for lab in np.unique(labels_all):
            idx_lab = np.flatnonzero(labels_all == lab)
            n_p = len(idx_lab)
            if n_p == 0:
                continue

            order_p = _order_pocket(idx_lab)

            if n_p == 1:
                point_dl = np.array([grid_scale], dtype=float)
                length = grid_scale
            else:
                ordered_pts = pts_all[order_p]
                delta = _periodic_delta(np.roll(ordered_pts, -1, axis=0), ordered_pts)
                seg = np.linalg.norm(delta, axis=1)
                point_dl = np.maximum(0.5 * (np.roll(seg, 1) + seg), 0.25 * grid_scale)
                length = float(np.sum(point_dl))

            pocket_data.append({"label": int(lab), "order": order_p, "point_dl": point_dl, "length": length})

        if not pocket_data:
            empty_pts = np.empty((0, 2), dtype=float)
            empty_idx = np.empty(0, dtype=int)
            empty_w = np.empty(0, dtype=float)
            return empty_pts, empty_pts.copy(), empty_idx, empty_w

        lengths = np.asarray([p["length"] for p in pocket_data], dtype=float)
        n_target = min(int(_N_FS), n_all)
        n_pockets = len(pocket_data)

        if n_target <= n_pockets:
            keep = np.argsort(lengths)[::-1][:n_target]
            n_alloc = np.zeros(n_pockets, dtype=int)
            n_alloc[keep] = 1
        else:
            raw_alloc = n_target * lengths / max(float(np.sum(lengths)), 1e-14)
            n_alloc = np.maximum(np.floor(raw_alloc).astype(int), 1)

            while n_alloc.sum() < n_target:
                frac = raw_alloc - np.floor(raw_alloc)
                frac[n_alloc >= np.array([len(p["order"]) for p in pocket_data])] = -np.inf
                p = int(np.argmax(frac))
                if not np.isfinite(frac[p]):
                    break
                n_alloc[p] += 1
            while n_alloc.sum() > n_target:
                removable = np.flatnonzero(n_alloc > 1)
                if len(removable) == 0:
                    break
                p = removable[np.argmin(raw_alloc[removable] - n_alloc[removable])]
                n_alloc[p] -= 1

        # ---- 7. Uniform arc-length sampling on each pocket ----
        selected = []

        for p_idx, pocket in enumerate(pocket_data):
            order_p = pocket["order"]
            point_dl = pocket["point_dl"]
            n_avail = len(order_p)
            n_take = min(int(n_alloc[p_idx]), n_avail)

            if n_take <= 0:
                continue
            if n_take == n_avail:
                selected.extend(order_p.tolist())
                continue

            cum = np.concatenate(([0.0], np.cumsum(point_dl)))
            length = float(cum[-1])
            targets = (np.arange(n_take) + 0.5) * length / n_take
            chosen_local = np.searchsorted(cum, targets, side="right") - 1
            chosen_local = np.clip(chosen_local, 0, n_avail - 1)

            chosen_local = list(dict.fromkeys(chosen_local.tolist()))

            if len(chosen_local) < n_take:
                remaining = [j for j in range(n_avail) if j not in chosen_local]
                while len(chosen_local) < n_take and remaining:
                    if not chosen_local:
                        best = remaining[0]
                    else:
                        best = max(
                            remaining,
                            key=lambda j: min(
                                min(abs(cum[j] - cum[c]), length - abs(cum[j] - cum[c]))
                                for c in chosen_local
                            )
                        )
                    chosen_local.append(best)
                    remaining.remove(best)

            selected.extend(order_p[np.asarray(chosen_local[:n_take], dtype=int)].tolist())

        sel = np.asarray(selected, dtype=int)

        # Keep pocket ordering; do not np.unique-sort geometrically.
        _, first = np.unique(sel, return_index=True)
        sel = sel[np.sort(first)]

        # ---- 8. Final data and selected-pocket arc-length weights ----
        fs_pts = pts_all[sel]
        vF_vec = vF_vec_all[sel]
        fs_idx = fs_idx_all[sel]
        labels_sel = labels_all[sel]
        N = len(sel)

        dl = np.empty(N, dtype=float)

        for lab in np.unique(labels_sel):
            loc = np.flatnonzero(labels_sel == lab)
            if len(loc) == 1:
                dl[loc] = grid_scale
                continue
            pts_p = fs_pts[loc]
            box_p = (pts_p + np.pi) % (2.0 * np.pi)
            tree_p = cKDTree(box_p, boxsize=2.0 * np.pi)

            k_q = min(3, len(loc))
            d, _ = tree_p.query(box_p, k=k_q)

            if k_q == 2:
                dl[loc] = np.maximum(d[:, 1], 0.25 * grid_scale)
            else:
                dl[loc] = np.maximum(0.5 * (d[:, 1] + d[:, 2]), 0.25 * grid_scale)

        vF_abs = np.linalg.norm(vF_vec, axis=1)
        finite_v = vF_abs[np.isfinite(vF_abs) & (vF_abs > 0.0)]
        vF_median = float(np.median(finite_v)) if len(finite_v) else 0.0
        vF_floor = max(_VF_FLOOR_TIGHT, _VF_FLOOR_REL_FRAC * vF_median)
        weights = dl / (_BZ_NORM * np.maximum(vF_abs, vF_floor))

        # ---- 9. Cache ----
        if store_cache:
            if self._fs_cache_dict is None:
                self._fs_cache_dict = {}
            if len(self._fs_cache_dict) >= 32:
                self._fs_cache_dict.pop(next(iter(self._fs_cache_dict)))
            self._fs_cache_dict[_cache_key_vals] = (fs_pts, vF_vec, fs_idx, weights)
        return fs_pts, vF_vec, fs_idx, weights

class VectorizedBdG:
    def __init__(self, solver: 'RMFT_Solver'):
        self.solver   = solver
        self._kpts    = solver.k_points        # (N_k, 2)    — SCF / gap grid (endpoint=False)
        self._H_stack = np.zeros((solver.N_k, _N_BDG, _N_BDG), dtype=complex)  # SCF grid buffer
        self.Z        = solver.p.Z
        self.g_Eg2    = solver.p.g_Eg2
        self.Sz_stag_nambu_channels = solver.Sz_stag_nambu_channels
    
    def _build_H_stack(self, kpts: np.ndarray, M: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, F67s_mf: float = 0.0, out: Optional[np.ndarray] = None, Q_Eg2: float = 0.0, Delta_s7b: complex = 0.0j, Delta_d7b: complex = 0.0j) -> np.ndarray:
        """
        Build the (N, 24, 24) BdG Hamiltonian stack for an arbitrary kpts array.
        24×24 Nambu basis: [Part_A(0:6), Part_B(6:12), Hole_A(12:18), Hole_B(18:24)], each sub-block in the
        FULL [6↑, 6↓, 7ₐ↑, 7ₐ↓, 7ᵦ↑, 7ᵦ↓] orbital basis, fully dynamical band throughout (kinetic, crystal field,
        AFM Weiss field, and JT coupling all included exactly). It is therefore not a problem if the Γ6/Γ7a bands rise to, or above, Γ7b's level under renormalisation.

        ┌──────────────┬──────────────────────────────┐
        │  H_A   T_AB  │  D_s        D_d              │  Part_A, Part_B
        │  T_AB† H_B   │  D_d        D_s              │
        ├──────────────┼──────────────────────────────┤
        │  D_s†  D_d†  │  −H_A*     −T_AB†            │  Hole_A, Hole_B
        │  D_d†  D_s†  │  −T_AB*    −H_B*             │
        └──────────────┴──────────────────────────────┘

        D_s (on-site, channel s):  Δ_s · [6↑↔7ₐ↓ singlet, φ=1]
        D_d (inter-site, channel d): Δ_d · φ(k) · [6↑↔7ₐ↓ singlet, φ(k)=cos kx−cos ky]
        D_s7b/D_d7b: OPTIONAL Γ6↔Γ7b analogs of the above (default 0j → identical to the Γ7a-only model when not supplied).
            Symmetry-allowed matrix element only where B1g_op[Γ6,Γ7b]≠0; the gap equation itself decides whether it wants to be nonzero.
        F_AA = u_A·v_A* → feeds Δ_s gap eq.   F_AB = u_A·v_B* → feeds Δ_d gap eq.
        """
        N = len(kpts)
        if out is None:
            H = np.zeros((N, _N_BDG, _N_BDG), dtype=complex)
        else:
            H = out
            H[:] = 0.0 + 0.0j
        
        tx_b, ty_b = self.solver.p.effective_hopping_anisotropic(Q)
        Tx_op, Ty_op = self.solver.p.hopping_matrices(Q)
        J_A1g_diag, J_B1g_bare = self.solver.p.exchange_channels(Q, n_kspace, tx_b, ty_b, g_J)

        # --- Local Hamiltonians (A/B sublattice), full 6×6 (Γ6⊕Γ7a⊕Γ7b) ---
        H_A = self.solver.build_local_hamiltonian_for_bdg(+1.0, M, J_A1g_diag, mu, self.Z)
        H_B = self.solver.build_local_hamiltonian_for_bdg(-1.0, M, J_A1g_diag, mu, self.Z)

        # --- On-site singlet pairing (Delta_s): Γ6-Γ7a channel, plus optional Gamma6-Gamma7b channel ---
        D_on = np.zeros((_N_ORB, _N_ORB), dtype=complex)
        D_on[0, 3] = Delta_s
        D_on[1, 2] = -Delta_s
        D_on[0, 5] = Delta_s7b   # Γ6↑ ↔ Γ7b↓ (same singlet structure as the Γ7a channel, shifted by +2)
        D_on[1, 4] = -Delta_s7b  # Γ6↓ ↔ Γ7b↑
        D_dag = np.conj(D_on).T

        _kx = kpts[:, 0]
        _ky = kpts[:, 1]
        beta_k = self.solver.p.wave_function_weight(tx_b, ty_b, _kx, _ky)  # ZRS spectral weight β²(k)

        # k‑dependent Jahn–Teller coupling
        H_JT_loc = (self.solver.g_JT_bare * Q * self.solver.B1g_op).astype(complex)
        H_JT_k = beta_k[:, None, None] * H_JT_loc[None, :, :]

        # Transverse (off-diagonal) anomalous Weiss field from J_B1g: active only when Q≠0 AND condensate has F67s≠0.
        f67s_loc_matrix = (self.Z * J_B1g_bare * F67s_mf) * self.solver.B1g_offdiag
        H_TRW = beta_k[:, None, None] * f67s_loc_matrix[None, :, :]

        # Eg,2 distortion: UNIFORM across sublattices; treated as global (q=0) structural order parameters, differing only in which symmetry channel/operator they couple to.
        H_Eg2 = (self.g_Eg2 * Q_Eg2) * self.solver.Eg2_op

        H_part_0 = H_A + H_Eg2 + H_JT_k - H_TRW
        H_part_1 = H_B + H_Eg2 + H_JT_k + H_TRW

        # Particle blocks
        H[:, 0:6,   0:6  ] = H_part_0
        H[:, 6:12,  6:12 ] = H_part_1
        # Hole blocks
        H[:, 12:18, 12:18] = -np.conj(H_part_0)
        H[:, 18:24, 18:24] = -np.conj(H_part_1)

        # Particle–hole off-diagonal (Delta_s)
        H[:, 0:6,   12:18] = D_on
        H[:, 6:12,  18:24] = D_on
        H[:, 12:18, 0:6  ] = D_dag
        H[:, 18:24, 6:12 ] = D_dag

        # --- Inter-sublattice hopping: orbital-selective 6×6 matrix (rigorous U6 projection) ---
        H_AB = _build_H_AB_block(_kx, _ky, Tx_op, Ty_op, g_t)     # (N,6,6), Hermitian by construction
        H_AB_T = H_AB.transpose(0, 2, 1)

        # Particle sector: A→B is H_AB, B→A is its conjugate transpose
        H[:, 0:6, 6:12] += H_AB
        H[:, 6:12, 0:6] += np.conj(H_AB_T)

        # Hole sector = -h^T(k) block-by-block (valid since cos(±k)=cos(k) here ⇒ h(-k)=h(k)):
        #   Hole_A→Hole_B = -H_AB*,   Hole_B→Hole_A = -H_AB^T
        # (For this model H_AB is Hermitian by construction, so H_AB* = H_AB^T numerically;
        #  written this way the code stays correct even if that degeneracy is ever lifted.)
        H[:, 12:18, 18:24] += -np.conj(H_AB)
        H[:, 18:24, 12:18] += -H_AB_T

        # --- d-wave pairing Delta_d (Γ6-Γ7a channel only, inter-sublattice) ---
        if N == self.solver.N_k:
            phi_d_k = self.solver.phi_k
        else:
            phi_d_k = np.cos(_kx) - np.cos(_ky)

        phi = phi_d_k * Delta_d
        phi7b = phi_d_k * Delta_d7b   # same B1g d-wave form factor; only the target orbital (Γ7b vs Γ7a) differs

        # Particle ↔ Hole couplings (singlet structure). Global indices: Part_A=0:6, Part_B=6:12, Hole_A=12:18, Hole_B=18:24; within each block local 0,1,2,3,4,5 = Γ6↑,Γ6↓,Γ7a↑,Γ7a↓,Γ7b↑,Γ7b↓.
        H[:, 0,  21] +=  phi   # Γ6↑(A,part) ↔ Γ7a↓(B,hole)
        H[:, 1,  20] -=  phi   # Γ6↓(A,part) ↔ Γ7a↑(B,hole)
        H[:, 6,  15] +=  phi   # Γ6↑(B,part) ↔ Γ7a↓(A,hole)
        H[:, 7,  14] -=  phi   # Γ6↓(B,part) ↔ Γ7a↑(A,hole)

        # Γ6-Γ7b analog (optional; zero unless Delta_d7b is supplied)
        H[:, 0,  23] +=  phi7b   # Γ6↑(A,part) ↔ Γ7b↓(B,hole)   [23 = 18+5]
        H[:, 1,  22] -=  phi7b   # Γ6↓(A,part) ↔ Γ7b↑(B,hole)   [22 = 18+4]
        H[:, 6,  17] +=  phi7b   # Γ6↑(B,part) ↔ Γ7b↓(A,hole)   [17 = 12+5]
        H[:, 7,  16] -=  phi7b   # Γ6↓(B,part) ↔ Γ7b↑(A,hole)   [16 = 12+4]

        phi_c = np.conj(phi)
        H[:, 21,  0] += phi_c
        H[:, 20,  1] -= phi_c
        H[:, 15,  6] += phi_c
        H[:, 14,  7] -= phi_c

        phi7b_c = np.conj(phi7b)
        H[:, 23,  0] += phi7b_c
        H[:, 22,  1] -= phi7b_c
        H[:, 17,  6] += phi7b_c
        H[:, 16,  7] -= phi7b_c

        H[:] = 0.5 * (H + H.conj().transpose(0, 2, 1))
        return H

    def compute_channel_staggered_magnetizations(self, Q: float, Delta_s: complex, Delta_d: complex, mu: float, ev: np.ndarray, ec: np.ndarray) -> np.ndarray:
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

    def compute_gap_eq_vectorized(self, M: np.ndarray, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, t_eff: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, J_eff: float, Gamma_M: float, V_JT: float, V_JT_corr: float, V_cap: float, det_afm_sc: float, solve_state: '_SolveState', ev: np.ndarray, ec: np.ndarray, vertex_cache: dict = None, verbose: bool = False) -> Tuple[complex, complex, complex, complex, dict]:
        """
        Gap equation with q-dependent RPA pairing vertex V(q) built from normal-state (Δ=0) susceptibilities.
        """
        solver = self.solver
        # --- Vertex cache invalidation (Δ-independent for normal-state part!) ---
        staleness = (
            not isinstance(vertex_cache, dict)
            or float(np.max(np.abs(M - vertex_cache.get('M', np.zeros(_N_CHANNELS))))) > _M_THR_REL * float(np.sqrt(abs(det_afm_sc)))
            or abs(Q - vertex_cache.get('Q', 0.0)) > max(_Q_THR_REL * solver.p.lambda_hop, 1e-4)
            or (det_afm_sc * vertex_cache.get('det_afm_current', det_afm_sc)) < 0.0
        )
        if staleness:
            # Compute kernel and obtain base cache
            vertex_cache = solver.compute_pairing_kernel_and_build_cache(M, Q, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff, Gamma_M, V_JT, V_JT_corr, V_cap, det_afm_sc, solve_state)
            # ---- Add normal-state spin/JT determinantal info ----
            ev, ec = solver._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self)

            chi_SS_q0, chi_SQ_q0, chi_QS_q0, chi_QQ_q0 = solver.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
            chi_SS_afm, chi_SQ_afm, _, chi_QQ_afm = solver.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
            
            vertex_cache.update({
                'chi_SS_q0':        chi_SS_q0,
                'chi_SQ_q0':        chi_SQ_q0,
                'chi_QS_q0':        chi_QS_q0,
                'chi_QQ_q0':        chi_QQ_q0,
                'chi_SS_afm':       chi_SS_afm,
                'chi_SQ_afm':       chi_SQ_afm,
                'chi_QQ_afm':       chi_QQ_afm,
                'det_afm':          solver._rpa_det(J_eff, V_JT_corr, chi_SS_afm, chi_SQ_afm, chi_SQ_afm, chi_QQ_afm)[0],  # Bare spin–orbital cross-vertex from the cluster-ED spin–JT cross-coupling J_MQ
                'det_afm_current':  det_afm_sc,
                'ansatz_unstable':  det_afm_sc < 0.0,
            })

        # ---- Anomalous pair amplitudes (always computed with current Δ) ----
        uA, uB, vA, vB = _get_nambu_spinors(ec)
        arg = np.clip(ev / solver.kT, -100, 100)
        f12 = 1.0 - 2.0 / (1.0 + np.exp(arg))   # tanh(E/2kT); (N_k, 24)
        # Per-k pair amplitudes (sum over 24 BdG bands, before BZ integration)
        pair_s_k = np.sum(
            (uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
            + uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])) * f12,
            axis=1)
        pair_d_k = np.sum(
            solver.phi_k[:, None] * (uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
                + uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])) * f12,
            axis=1)

        # Full-BZ anomalous pair amplitudes F_AA/F_AB at the CURRENT (Δ, M, Q).
        # The Bogoliubov u·v* coherence factors here inherit whatever relative phase Delta_s/Delta_d currently carry.
        F_AA_BZ = complex(np.dot(solver.k_weights, pair_s_k)) / 4.0
        F_AB_BZ = complex(np.dot(solver.k_weights, pair_d_k)) / 4.0

        # --- Optional Γ6-Γ7b channel: identical construction, orbital indices 2,3 (Γ7a) -> 4,5 (Γ7b). ---
        # NOTE (scope): this gives a self-consistent-fixed-point-style RAW estimate of Delta_s7b/Delta_d7b, analogous to Delta_s_raw/Delta_d_raw below, using the SAME RPA vertex V_s/V_d (a first-pass choice:
        # the RPA vertex V(q) is a property of the spin/JT collective modes, not of which orbital channel is pairing, so reusing it is a reasonable zeroth-order approximation).
        # It is returned as a DIAGNOSTIC value only: has not wired back yet into the M/Q/Delta_s/Delta_d Anderson-mixed self-consistency loop.
        pair_s7b_k = np.sum(
            (uA[:, 0, :] * np.conj(vA[:, 5, :]) - uA[:, 1, :] * np.conj(vA[:, 4, :])
            + uB[:, 0, :] * np.conj(vB[:, 5, :]) - uB[:, 1, :] * np.conj(vB[:, 4, :])) * f12,
            axis=1)
        pair_d7b_k = np.sum(
            solver.phi_k[:, None] * (uA[:, 0, :] * np.conj(vB[:, 5, :]) - uA[:, 1, :] * np.conj(vB[:, 4, :])
                + uB[:, 0, :] * np.conj(vA[:, 5, :]) - uB[:, 1, :] * np.conj(vA[:, 4, :])) * f12,
            axis=1)
        F_AA7b_BZ = complex(np.dot(solver.k_weights, pair_s7b_k)) / 4.0
        F_AB7b_BZ = complex(np.dot(solver.k_weights, pair_d7b_k)) / 4.0

        # ---- Extract vertex scalars and 2x2 kernel info from cache ----
        V_s_scalar = vertex_cache['V_s_scalar']
        V_d_scalar = vertex_cache['V_d_scalar']
        V_sd       = vertex_cache['V_sd']
        lambda_lin_max = vertex_cache['lambda_lin_max']
        v_s_raw = abs(vertex_cache['v_s_raw'])
        v_d_raw = abs(vertex_cache['v_d_raw'])
        
        # Gap equations + jump limiter: F_AA_BZ / F_AB_BZ already carry BdG saturation via the anomalous Green functions, so a λ_pair-based f_stab would double-count the suppression, channel ratio (s vs d) is preserved during clamping.
        if lambda_lin_max > 0:
            Delta_s_raw = g_Delta_s * (V_s_scalar * F_AA_BZ + V_sd * F_AB_BZ)
            Delta_d_raw = g_Delta_d * (V_sd * F_AA_BZ + V_d_scalar * F_AB_BZ)
            # Γ6-Γ7b gap values
            Delta_s7b_raw = (g_Delta_s * V_s_scalar * F_AA7b_BZ)
            Delta_d7b_raw = (g_Delta_d * V_d_scalar * F_AB7b_BZ)
        else:
            Delta_s_raw = 0.0
            Delta_d_raw = 0.0
            Delta_s7b_raw = 0.0
            Delta_d7b_raw = 0.0

        # Det-proportional jump cap: past the QCP the RPA vertex is unreliable; limit the gap step.
        if det_afm_sc < 0.0:
            _det_depth = float(np.clip(abs(det_afm_sc) / max(_RPA_DET_WARN, 1e-6), 0.0, _DET_DEPTH_CAP))
            effective_jump_cap = max(_JUMP_CAP_FLOOR, _DELTA_JUMP_CAP * math.exp(-_DET_JUMP_HALF_SCALE * _det_depth))
        else:
            effective_jump_cap = _DELTA_JUMP_CAP

        # ---- Blend with 2x2 kernel direction ----
        # L1-normalised eigenvector from cache; align sign with current (Δ_s, Δ_d) direction.
        overlap = (np.conj(Delta_s_raw) * v_s_raw + np.conj(Delta_d_raw) * v_d_raw).real
        sign_align = 1.0 if overlap >= 0 else -1.0
        v_s = v_s_raw * sign_align
        v_d = v_d_raw * sign_align
        v_norm = max(abs(v_s) + abs(v_d), 1e-12)
        v_s /= v_norm
        v_d /= v_norm

        D_fp = abs(Delta_s_raw) + abs(Delta_d_raw)
        # Seed phase: gap is negligible → inject from 2×2 kernel
        if D_fp < _QQ_DELTA_THRESH:
            D_fp = float(max(_BCS_SEED_FRACTION * t_eff * np.exp(-1.0 / max(lambda_lin_max, 0.1)), _DELTA_ABS_FLOOR))
            alpha_blend = 0.5 * _ALPHA_MIX_2X2
        else:
            alpha_blend = _ALPHA_MIX_2X2

        Ds_2x2 = D_fp * v_s
        Dd_2x2 = D_fp * v_d
        
        # Phase-preserving blend: the 2×2 kernel only constrains the REAL relative sign between s and d;
        # Blending magnitudes and then multiplying by a real _v_s/_v_d would silently zero that phase every iteration.
        phase_s_new = Delta_s_raw / max(abs(Delta_s_raw), 1e-30)
        phase_d_new = Delta_d_raw / max(abs(Delta_d_raw), 1e-30)
        
        # Smooth fallback: if fixed-point gives nothing
        Ds_mag = (1.0 - alpha_blend) * max(abs(Delta_s_raw), _KERNEL_DIR_MIN_FRAC * abs(Ds_2x2)) + alpha_blend * abs(Ds_2x2)
        Dd_mag = (1.0 - alpha_blend) * max(abs(Delta_d_raw), _KERNEL_DIR_MIN_FRAC * abs(Dd_2x2)) + alpha_blend * abs(Dd_2x2)
        
        # Below the seed threshold there is no meaningful fixed-point phase yet — use the real 2×2-kernel sign directly.
        # Above threshold, keep whatever complex phase the BdG self-consistency produced.
        if D_fp < _QQ_DELTA_THRESH:
            Ds = complex(Ds_mag * np.sign(v_s) if v_s != 0.0 else Ds_mag)
            Dd = complex(Dd_mag * np.sign(v_d) if v_d != 0.0 else Dd_mag)
        else:
            Ds = Ds_mag * phase_s_new
            Dd = Dd_mag * phase_d_new
        
        # Single, final hard cap: compares the fully-blended/seeded output against the PHYSICAL ceiling
        phys_ceiling = effective_jump_cap * max(abs(Delta_s) + abs(Delta_d), _DELTA_ABS_FLOOR)
        D_new = abs(Ds) + abs(Dd)
        if D_new > phys_ceiling:
            scale = phys_ceiling / max(D_new, 1e-12)
            Ds *= scale
            Dd *= scale

        if verbose:
            pair_intra6_s_k = np.sum(
                (uA[:, 0, :] * np.conj(vA[:, 1, :]) - uA[:, 1, :] * np.conj(vA[:, 0, :])
                + uB[:, 0, :] * np.conj(vB[:, 1, :]) - uB[:, 1, :] * np.conj(vB[:, 0, :])) * f12,
                axis=1)
            pair_intra7_s_k = np.sum(
                (uA[:, 2, :] * np.conj(vA[:, 3, :]) - uA[:, 3, :] * np.conj(vA[:, 2, :])
                + uB[:, 2, :] * np.conj(vB[:, 3, :]) - uB[:, 3, :] * np.conj(vB[:, 2, :])) * f12,
                axis=1)
            F_AA_intra6 = complex(np.dot(solver.k_weights, pair_intra6_s_k)) / 4.0
            F_AA_intra7 = complex(np.dot(solver.k_weights, pair_intra7_s_k)) / 4.0

            max_intra = max(abs(F_AA_intra6), abs(F_AA_intra7))
            inter_intra_ratio = abs(F_AA_BZ) / max_intra if max_intra > 0 else float('inf')
            _scf_log("GAP-DIAG",
                f"Δs7b={Delta_s7b_raw:+.4f}  Δd7b={Delta_d7b_raw:+.4f}"
                f"  F_AA(inter)={abs(F_AA_BZ):.3e}  F_AB(inter)={abs(F_AB_BZ):.3e}"
                f"  F_AA(intra6)={abs(F_AA_intra6):.3e}  F_AA(intra7)={abs(F_AA_intra7):.3e}"
                f"  Inter/Intra={inter_intra_ratio:.2f}  |Δ_s|={abs(Delta_s):.3e}  |Δ_d|={abs(Delta_d):.3e}")
        return complex(Ds), complex(Dd), complex(Delta_s7b_raw), complex(Delta_d7b_raw), vertex_cache

def plot_ground_state_comparison(results: Dict[str, dict], labels: Optional[Dict[str, str]] = None):
    """
    Compare free-energy convergence and the final candidates from __main__'s
    three-way scan:
        "ref"    — full self-consistent solve (SC and Q both free)
        "normal" — Δ forced to 0 (no SC)
        "SC_Q0"  — SC free, Q forced to 0 (no JT relaxation)

    This is the direct numerical check of the hypothesis's central claim:
    F_ref should be the lowest of the three (SC condensation lowers F below
    "normal"; JT relaxation lowers F further below "SC_Q0"), and Q should
    self-consistently settle back to ~0 in "normal" *without* being forced
    there (symmetry-forbidden JT in the normal state, §2/§11), while staying
    finite in "ref".

    Parameters
    ----------
    results : dict of {scenario_name: solve_self_consistent() result}, e.g.
        {"ref": ..., "normal": ..., "SC_Q0": ...} exactly as __main__ builds
        it. Missing or falsy entries (a task that failed/never completed)
        are skipped, not an error — but at least one entry is required.
    labels  : optional {scenario_name: display_label} override for the
        legend/axis text.

    Returns the matplotlib Figure (not saved / shown automatically).
    """
    default_labels = {'ref': 'ref (SC+JT)', 'normal': 'normal (Δ=0)', 'SC_Q0': 'SC, Q=0'}
    default_colors = {'ref': 'tab:blue', 'normal': 'tab:gray', 'SC_Q0': 'tab:orange'}
    labels = {**default_labels, **(labels or {})}

    preferred_order = ['ref', 'normal', 'SC_Q0']
    ordered_keys = ([k for k in preferred_order if results.get(k)]
                     + [k for k in results if k not in preferred_order and results.get(k)])
    scenarios = [(k, results[k]) for k in ordered_keys]
    if not scenarios:
        raise ValueError("plot_ground_state_comparison: no non-empty results to plot.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Ground-state comparison: ref vs. normal vs. SC-only (Q=0)', fontsize=14, fontweight='bold')

    # (0,0) F_cluster per iteration, with each scenario's converged F_bdg as a dotted reference line.
    ax = axes[0, 0]
    for key, res in scenarios:
        c, lbl = default_colors.get(key), labels.get(key, key)
        ok, why = _scf_result_reliability(res)
        ax.plot(res['history']['F_cluster'], '-' if ok else '--', color=c, linewidth=2,
                 label=lbl + ('' if ok else f'  ⚠ {why}'))
        ax.axhline(res.get('F_bdg', np.nan), color=c, linestyle=':', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Free energy (eV)')
    ax.set_title('F_cluster per iteration  (dotted = converged F_bdg)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) Final F_bdg bar comparison -- the actual "who is the ground state" answer.
    ax = axes[0, 1]
    keys = [k for k, _ in scenarios]
    F_vals = [res.get('F_bdg', np.nan) for _, res in scenarios]
    bars = ax.bar([labels.get(k, k) for k in keys], F_vals,
                   color=[default_colors.get(k, 'gray') for k in keys], alpha=0.85)
    for (k, res), bar in zip(scenarios, bars):
        ok, _ = _scf_result_reliability(res)
        if not ok:
            bar.set_hatch('//'); bar.set_edgecolor('red')
    finite = [(k, v) for k, v in zip(keys, F_vals) if np.isfinite(v)]
    if finite:
        best_key = min(finite, key=lambda kv: kv[1])[0]
        ax.set_title(f'Final F_bdg  (lowest = ground state → {labels.get(best_key, best_key)})', fontsize=11)
    ax.set_ylabel('F_bdg (eV)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelrotation=10)

    # (1,0) AFM order parameter (Gamma6 channel) convergence, all scenarios overlaid.
    ax = axes[1, 0]
    for key, res in scenarios:
        m0_hist = [float(m[0]) for m in res['history']['M']]
        ax.plot(m0_hist, color=default_colors.get(key), linewidth=2, label=labels.get(key, key))
    ax.set_xlabel('Iteration'); ax.set_ylabel('Magnetization M[Γ6]')
    ax.set_title('AFM order parameter convergence', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (1,1) |Q| convergence -- the key symmetry-forbiddenness check: "normal" should relax back
    # to ~0 on its own (Q was NOT forced there), while "ref" settles at a finite value.
    ax = axes[1, 1]
    for key, res in scenarios:
        ax.plot(res['history']['Q'], color=default_colors.get(key), linewidth=2, label=labels.get(key, key))
    ax.axhline(0.0, color='k', linewidth=0.7, linestyle=':')
    ax.set_xlabel('Iteration'); ax.set_ylabel('|Q| (Å)')
    ax.set_title('JT distortion: "normal" should self-consistently relax to ≈0', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


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
        t_pd             = 0.470,
        U_dd             = 2.900,
        lambda_soc       = 0.046,
        Delta_tetra      = -0.075,
        g_JT             = 0.330,
        K_lattice        = 2.800,
        lambda_hop       = 1.100,
        g_Eg2            = 0.100,
        K_lattice_Eg2    = 6.500,
        Delta_CT         = 2.700,
        Delta_B1g_static = -0.011,
        hybrid_scale     = 6.000,
        Upp_ratio_bare   = 0.400,
        Z                = 4,
        kT               = 0.005,
        tol              = 1e-4,
        )

    target_doping = 0.127
    doping_margin = 0.20          # scan covers target ± 20 %
    min_doping    = max(target_doping * (1.0 - doping_margin), _G_T_COHERENCE_MIN / (2.0 - _G_T_COHERENCE_MIN))
    max_doping    = target_doping * (1.0 + doping_margin)
    initial_Delta = 8e-3

    _scf_log("INIT",
                f"t_pd={params.t_pd:.3f} eV  U_dd={params.U_dd:.4f} eV  λ_SOC={params.lambda_soc:.3f} eV"
                f"  Δ_tetra={params.Delta_tetra:.3f} eV  bare g_JT={params.g_JT:.3f} eV  K_lattice={params.K_lattice:.4f}"
                f"  lambda_hop={params.lambda_hop:.3f}  Δ_CT={params.Delta_CT:.3f} eV   Δ_ip={params.Delta_B1g_static:.3f} eV"
                f"  kT={params.kT*1000:.2f} meV  Z={params.Z}  N_k={params.N_k}")
    _scf_log("DERIVED",
                f"multi_op (normalised):\n{np.array2string(params.multi_op, precision=4, suppress_small=True)}"
                f"  t0={params.t0:.4f} eV  J_pdct={params.J_pdct:.4f} eV  Δ_CF={params.Delta_CF:.5f} eV "
                f"  Γ₇split={params.g7split:.5f} eV [{'⚠ < 2kT' if abs(params.g7split) < 2.0 * params.kT else '✓'}]")

    bandwidth = _RPA_BW_FACTOR * params.get_gutzwiller_factors(target_doping)[0] * params.t0
    E_G7b = params.Delta_CF + params.g7split
    _scf_log("DERIVED",
        f"Γ7b (E={E_G7b:.3f} eV) vs. estimated Γ6/Γ7a bandwidth ({bandwidth:.3f} eV): "
        + ("within bandwidth — bands may cross/overlap; handled exactly by the full 3-band "
           if E_G7b < bandwidth else
           "well separated from the Γ6/Γ7a manifold."))

    solver_ref    = RMFT_Solver(copy.deepcopy(params))
    solver_normal = RMFT_Solver(copy.deepcopy(params))
    solver_Q0     = RMFT_Solver(copy.deepcopy(params))
    
    # ── Section 1: Reference SCF ─────────────────────────────────────────────
    _scf_log("REF-SCF", "="*60)  # Run SCF first.  All subsequent diagnostics use the self-consistent (M, μ)

    tasks = {
        "ref":    lambda: solver_ref.solve_self_consistent(target_doping, initial_Delta,
                            verbose=True,  force_delta_zero=False, force_Q_zero=False),
        "normal": lambda: solver_normal.solve_self_consistent(target_doping, 0.0,
                            verbose=False, force_delta_zero=True,  force_Q_zero=False),
        "SC_Q0":  lambda: solver_Q0.solve_self_consistent(target_doping, initial_Delta,
                            verbose=False, force_delta_zero=False, force_Q_zero=True)
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

    if results.get("ref") and results.get("normal") and results.get("SC_Q0"):
        is_reliable = _scf_result_reliability
        hess_min = _scf_result_hessian_min

        energies = {}
        reliable = {}

        for key in ("ref", "normal", "SC_Q0"):
            res = results[key]
            energies[key] = res.get("F_bdg", 0.0)
            reliable[key] = is_reliable(res)

            ok, why = reliable[key]
            _scf_log(
                "FREE-ENERGY",
                f"F_{key:<6}= {energies[key]:.6f} eV  "
                f"(Hess min = {hess_min(res):.3e})"
                + ("" if ok else f"  ⚠ UNRELIABLE ({why})")
            )

        if not all(ok for ok, _ in reliable.values()):
            _scf_log("FREE-ENERGY", "Comparison NOT trustworthy")
        elif energies["ref"] < energies["normal"] and energies["ref"] < energies["SC_Q0"]:
            _scf_log("FREE-ENERGY", "SC+JT is the ground state ✓")
        else:
            _scf_log("FREE-ENERGY", "SC+JT is NOT the lowest energy state ✗")

    else:
        _scf_log("FREE-ENERGY", "One or more parallel tasks failed.")

    _ref_result = results.get("ref")
    # ── Sections 2: SCF diagnostics, SC-JT window, Tc pre-estimates ────────
    if _ref_result is not None:
        V_spin_mean  = float(_ref_result['V_spin_mean'])
        V_JT_mean    = float(_ref_result['V_JT_mean'])
        V_rpa_mean   = float(_ref_result['V_rpa_mean'])
        _V_cr        = V_rpa_mean - V_spin_mean - V_JT_mean
        lambda_JT_sc = _ref_result['lambda_JT_sc']
        _lmax_ref    = _ref_result['lambda_lin_max']
        _hessian     = _ref_result['hessian_result']
        _frac        = _ref_result['frac']
        _ref_M       = _ref_result['M']
        _ref_Q       = _ref_result['Q']

        _scf_log("REF-SCF", f"  converged={_ref_result['converged']}  mott_suspect={_ref_result.get('mott_suspect', False)}")
        _scf_log("REF-SCF", f"  M={np.array2string(_ref_M, precision=4)}  Q={_ref_Q:+.5f}  Δs={_ref_result['Delta_s']:.5f} eV  Δd={_ref_result['Delta_d']:.5f} eV  μ={_ref_result['mu']:.4f} eV")
        _scf_log("REF-SCF", f"  Irrep R={_ref_result['selection_ratio']:.4f}  JT {'ALLOWED ✓' if _ref_result['selection_ratio'] > 0.02 else 'BLOCKED ✗'}")

        # — SC-JT window: K_eff path (Δ=0 → Δ≠0) —
        _scf_log("SCF-RES", (
            f"  δK_eff={_ref_result['K_eff_net']:+.4f}"
            f"  → {'✓ gap softens lattice (SC-triggered JT enabled)' if _ref_result['K_eff_net'] < -1e-4 else '⚠ gap stiffens lattice' if _ref_result['K_eff_net'] > 1e-4 else '≈ no K_eff change'}"
        ))

        # ── G-matrix at self-consistent M (normal-state instability) ─
        _scf_log("G-MATRIX", "="*60)
        G_base = solver_ref.compute_G_instability(target_doping, float(_ref_M[0]))

        # Kinetic / exchange scale
        _scf_log("G-MATRIX", f"h_afm={G_base['h_afm']:.4f} eV  N_eff={G_base['N_eff']:.4f} eV⁻¹"
                f"  J_eff={G_base['J_eff']:.4f} eV  blocking_ratio={G_base['blocking_ratio']:.4f}")

        # Pairing susceptibilities (normal-state Lindhard kernel)
        _scf_log("G-MATRIX", f"χ_QQ(Δ=0)={G_base['chi_QQ']:.4f} eV⁻¹  χ_ΔΔ(dom)={G_base['chi_pair_dom']:.4f}"
                f"  χ_Δs={G_base['chi_pair_s']:.4f}  χ_Δd={G_base['chi_pair_d']:.4f}  χ_Δsd={G_base['chi_pair_sd']:.4f}  [eV⁻¹]")
        _scf_log("G-MATRIX", f"χ_SQ(dom)={G_base['chi_SQ_dom']:.4f}  χ_SQ_s={G_base['chi_SQ_s']:.4f}  χ_SQ_d={G_base['chi_SQ_d']:.4f}  [eV⁻¹]")

        # Pairing eigenvalue (normal-state, q=0 reference)
        _lambda_eff = G_base['lambda_eff']
        _leff_status = ("✓ optimal" if 0.3 < _lambda_eff < 1.0
                        else ("⚠ weak — increase J_eff (↓u or ↑t_pd/Δ_CT)" if _lambda_eff <= 0.3
                            else "⚠ too strong — risk of spontaneous JT / AFM QCP"))
        _scf_log("G-MATRIX", f"λ_eff(N_eff·V_eff)={_lambda_eff:.4f}  [{_leff_status}]"
                f"  λ_lin_max(q=0)={G_base['lambda_lin_max_q0']:.4f}"
                f"  G22 (normal JT stability)={G_base['G22']:+.5f} eV/Å²"
                f"  {'✓ Q-stable' if G_base['G22'] > 0 else '✗ spontaneous JT!'}")

        _instab = G_base['instab_info']
        _scf_log("G-MATRIX", _instab.log_summary(verbose=True))

        # — RPA vertex decomposition (appended to G-MATRIX block, uses SCF vertex) —
        # d-wave: negative FS-average is EXPECTED; instability comes from q≈(π,π) backscattering
        if np.isfinite(V_rpa_mean) and abs(V_rpa_mean) > 1e-4:
            _v_note = ('⚠ V_avg<0: d-wave backscattering dominant (normal)' if V_rpa_mean < 0 else '✓ positive avg')
            _scf_log("G-MATRIX", f"V_RPA(FS-avg)={V_rpa_mean:.4f} eV  [{_v_note}]"
                     f"  spin={V_spin_mean/V_rpa_mean*100:.0f}%  JT={V_JT_mean/V_rpa_mean*100:.0f}%  cross={_V_cr/V_rpa_mean*100:.0f}%")
        elif np.isfinite(V_rpa_mean):
            _scf_log("G-MATRIX", f"V_RPA(FS-avg)={V_rpa_mean:.2e} eV"
                     f"  [spin={V_spin_mean:.3f}  V_JT={V_JT_mean:.3f}  cross={_V_cr:.3f} eV]")
        
        # — Linearised gap equation & channel decomposition —
        _scf_log("SCF-RES", f"Gap eq: λ_lin_max={_lmax_ref:.4f} ")
        
        # — Coherence lengths (single line; orbital selectivity shown when relevant) —
        if not _ref_result['valid_BdG']:
            _xi_note = (f"⚠ ξ_nodal/a={_ref_result['xi_nodal']:.2f} < 2 — BdG marginal"
                        f"  ξ_anti/a={_ref_result.get('xi_antinodal', float('nan')):.2f}")
        elif _ref_result['orbital_selective']:
            _xi_note = (f"✓ ξ_nodal/a={_ref_result['xi_nodal']:.2f}"
                        f"  ξ_anti/a={_ref_result.get('xi_antinodal', float('nan')):.2f}"
                        f"  ORBITAL-SEL: ξ_Γ6/a={_ref_result['xi_Gamma6']:.2f}  ξ_Γ7/a={_ref_result['xi_Gamma7']:.2f}")
        else:
            _xi_note = (f"✓ ξ_nodal/a={_ref_result['xi_nodal']:.2f}"
                        f"  ξ_anti/a={_ref_result.get('xi_antinodal', float('nan')):.2f}"
                        f"  orbitally uniform")
        _scf_log("SCF-RES", f"  [{_xi_note}]")

        # — SC Hessian (converged state) and FS-resolved ∂λ/∂Q —
        _hess_lmin_sc = float(_hessian['eigenvalues'][0]) if _hessian.get('eigenvalues') is not None else float('nan')
        _scf_log("SCF-RES", f"  λ_min(H_SC)={_hess_lmin_sc:+.4f}"
                 f"  {'✓ SC-triggered JT CONFIRMED' if np.isfinite(_hess_lmin_sc) and _hess_lmin_sc < 0.0 else '— JT not triggered'}")

        # — Stoner/Moriya: J_eff from the analytic Gutzwiller renormalisation —
        _stoner_r    = _ref_result['J_eff'] * G_base['chi_SS_afm']
        _ston_status = ('✓ near QCP' if 1.0 > _stoner_r > 0.7
            else ('⚠ near/past AFM QCP' if 2.0 > _stoner_r >= 1.0
                else ('safe' if _stoner_r <= 0.7 else '✗ deeply past QCP')))
        _scf_log("SCF-RES", f"J_eff={_ref_result['J_eff']:.4f} eV  χ_SS_AFM(Δ=0)={G_base['chi_SS_afm']:.4f}  J·χ_SS={_stoner_r:.4f} [{_ston_status}]")

        # χ_τ: B1g orbital susceptibility — SC-induced enhancement is the decisive signal
        chi_tau_net_mag =  _ref_result['chi_tau_net']
        _scf_log("SCF-RES", (
            f"  χ_τ: n={_ref_result['chi_tau_n']:+.4f}  sc={_ref_result['chi_tau_sc']:+.4f} eV⁻¹"
            f"  δχ_τ(SC-only)={chi_tau_net_mag:+.4f} eV⁻¹"
            f"  {'| ✓ softens' if chi_tau_net_mag > 0 else '| ⚠ stiffens'}"
        ))

        # SC-JT window bounds
        K_SC = params.g_JT ** 2 * chi_tau_net_mag / _LAMBDA_JT_VIABLE
        K_spont = params.g_JT**2 * G_base['chi_QQ']
        normal_stable = (params.K_lattice > K_spont)
        sc_jt_active  = (params.K_lattice < K_SC)
        window_open = K_SC > K_spont
        jt_viable = window_open and normal_stable and sc_jt_active and _instab.full_stable

        if jt_viable:
            K_opt = float(np.sqrt(K_spont * K_SC))
            window_width = K_SC - K_spont
            frac = (params.K_lattice - K_spont) / max(window_width, 1e-12) if window_width > 0 else 0.0
            note = (f"SC-JT ACTIVE: K_spont={K_spont:.4f}, K_SC={K_SC:.4f}, window={frac*100:.0f}%")
        else:
            if not normal_stable:
                reason = f"spontaneous JT (K_lattice={params.K_lattice:.4f} ≤ K_spont={K_spont:.4f})"
            elif not sc_jt_active:
                reason = f"insufficient softening (K_lattice={params.K_lattice:.4f} ≥ K_SC={K_SC:.4f})"
            elif not _instab.full_stable:
                reason = f"λ_min is negative: {G_base['lambda_min']}"
            else:
                reason = f"window is closed"
            note = f"SC-JT NOT ACTIVE: {reason}"
        
        _chi_tau_w = _ref_result.get('chi_tau_weight', 1.0)
        _chi_tau_w_note = ('  [halved: finer-scale]' if _chi_tau_w == 0.5
                           else '  [suppressed: fully nonlinear]' if _chi_tau_w == 0.0 else '')

        _scf_log("SCF-RES", (
            f"  K_spont_analytic={params.g_JT**2 / max(params.Delta_CF, _MATH_EPS):.4f}  K_spont={K_spont:.4f})"
            f"  λ_JT_sc={lambda_JT_sc:.4f}  λ_JT_opt={float(np.sqrt(_LAMBDA_JT_VIABLE * params.g_JT ** 2 * chi_tau_net_mag / max(K_spont, 1e-12))):.4f}"
            f"  {'✓ Richardson ok' if _ref_result['richardson_ok'] else '⚠ Richardson inconsistent'}"
            f"  χ_τ_weight={_chi_tau_w:.1f}{_chi_tau_w_note}"
        ))
        _scf_log("SCF-RES", (
            f"{'⚠ SC could distort lattice (λ_JT > 1, spontaneous JT regime)' if lambda_JT_sc > 1 else '✓ SC-JT coupling active' if lambda_JT_sc > _LAMBDA_JT_VIABLE else '✗ SC-JT coupling too weak'}"
            f"  → {note}"
        ))

        # — χ_SQ(q) full BZ scan ────────────────────────────────────────────
        solver_ref.estimate_chi_SQ_q_full(target_doping, _ref_M, _ref_Q, _ref_result['Delta_s'], _ref_result['Delta_d'], _ref_result['n_kspace'], _ref_result['mu'], _ref_result['J_eff'], _ref_result['F67s_mf'],  n_q=35)

        # Three independent Tc estimates (no shared label):
        #   Tc₁: McMillan (analytical, ω_SF = J_eff)
        #   Tc₂: λ(T)=1 crossing  (normal-state SCF scan)
        #   Tc₃: thermodynamic free-energy crossing (first-order aware)
        #   Tc_sp: spinodal (metastability limit, companion to Tc₃)
        _pre_Delta_total = float(_ref_result['Delta_s']) + float(_ref_result['Delta_d'])
        _scf_log("TC-PRELIM", f"Pre-BO Tc estimates at |Δ|={_pre_Delta_total*1000:.3f} meV")
        _sc_viable = (not _ref_result.get('mott_suspect', False)) and (float(G_base['g_t']) >= _G_T_COHERENCE_MIN) and bool(_ref_result['converged'])
        if _sc_viable:
            _omega_SF  = float(_ref_result['J_eff'])
            _lmax_safe = max(_lmax_ref, _MATH_EPS)
            _Tc1_eV    = float((_omega_SF / _MAD_DENOM) * np.exp(-_MAD_NUM * (1.0 + _lmax_safe) / _lmax_safe))
            _scf_log("TC-PRELIM",
                    f"  Tc₁(Allen–Dynes-SF): λ_max={_lmax_ref:.4f}"
                    f"  ω_SF(J_eff)={_omega_SF*1000:.1f} meV"
                    f"  → {_Tc1_eV*1000:.2f} meV  ({_Tc1_eV*_EV_TO_K:.1f} K)"
                    f"  [λ_eff(Schur+JT)={G_base['lambda_eff']:.4f}]"
                    f"  [denom={_MAD_DENOM} (Allen–Dynes SF compromise); for phonon Debye use 1.45]")

            _lT = solver_ref.compute_lambda_vs_T(target_doping, _ref_result)
            _scf_log("TC-PRELIM", f"  Tc₂(λ=1 crossing)={_lT['Tc_lambda']*1000:.2f} meV"
                     f"  slope={_lT['slope_at_Tc']*1000:.3f} meV⁻¹"
                     f"  n_crossings={_lT['n_crossings']}")

        # Tc₃: thermodynamic Tc from reference SCF (first-order aware free-energy crossing)
        if _pre_Delta_total > 1e-5:
            try:
                _Tc_ref_thermo = solver_ref.compute_Tc_thermodynamic(
                    target_doping,
                    sc_result = _ref_result,
                    T_min     = 1e-4,
                    T_max     = 0.25,
                    n_scan    = 12,
                    n_bisect  = 8,
                    Delta_tol = 1e-4,
                )
                _Tc_rt   = _Tc_ref_thermo['Tc']
                _Tc_rs   = _Tc_ref_thermo['Tc_spinodal']
                _ord_r   = _Tc_ref_thermo['transition_order']
                _Dj_r    = _Tc_ref_thermo['Delta_jump']
                _scf_log("TC-THERMO",
                         f"  Tc₃(thermo)={_Tc_rt*1000:.2f} meV ({_Tc_rt*_EV_TO_K:.1f} K)"
                         f"  Tc_sp(spinodal)={_Tc_rs*1000:.2f} meV"
                         f"  order={_ord_r}"
                         f"  Δ_jump={_Dj_r*1000:.2f} meV")
                if _Tc_rt > 1e-8:
                    ratio_2D = 2.0 * _pre_Delta_total / _Tc_rt
                    coupling_regime = _bcs_coupling_regime(ratio_2D)
                    _scf_log("TC-THERMO", f"  2Δ₀/kTc={ratio_2D:.3f}  [{coupling_regime}]")
                if _Tc_rs > 1e-6:
                    _uplift_r = (_Tc_rt - _Tc_rs) / _Tc_rs * 100.0
                    _scf_log("TC-THERMO",
                             f"  Tc uplift (Tc₃ vs Tc_sp): {_uplift_r:+.1f}%"
                             f"  [{'first-order dominant' if abs(_uplift_r)>20 else 'weakly first-order' if abs(_uplift_r)>5 else 'effectively second-order'}]")
            except Exception as _Tc_rt_err:
                _scf_log("TC-THERMO", f"  Tc₃(thermo) failed: {_Tc_rt_err}")
        else:
            _scf_log("TC-THERMO",
                     f"  Thermodynamic Tc skipped: |Δ|={_pre_Delta_total*1000:.3f} meV < threshold")
    else:
        _scf_log("TC-THERMO", "  Thermodynamic Tc skipped: reference SCF not converged.")

    # ── Ground-state comparison plot (ref vs. normal vs. SC-only Q=0) ────────
    _scf_log("PLOT", "=" * 60)
    try:
        _gs_fig = plot_ground_state_comparison(results)
        _gs_path = "ground_state_comparison.png"
        _gs_fig.savefig(_gs_path, dpi=150)
        plt.close(_gs_fig)
        _scf_log("PLOT", f"  Ground-state comparison saved -> {_gs_path}")
    except Exception as _gs_err:
        _scf_log("PLOT", f"  Ground-state comparison plot failed: {_gs_err}")