import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import opt_einsum as oe
from scipy.optimize import brentq, differential_evolution
from scipy.stats import norm, t as tdist
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
from threadpoolctl import threadpool_limits as _tpl_ctx
import matplotlib.pyplot as plt

_log_lock = _threading.Lock()

# ── NOT free parameters ───────────────────────────────────────────────────────
_EV_TO_K              : float = 11604.518121        # 1 eV in Kelvin
_GW_G_J_PREFACTOR     : float = 4.0                 # g_J = 4/(1+δ)²  Gutzwiller exchange renormalization prefactor (slave-boson / Kotliar-Ruckenstein derivation, half-filling limit)
_GW_G_T_NUMERATOR     : float = 2.0                 # g_t = 2δ/(1+δ)  Gutzwiller hopping prefactor numerator coefficient
_PI_INT               : int   = 314159              # π in scaled integer units

# ── Fermi surface sampling and k-grid ──────────────────────────────────────────
_NK                   : int   = 64                  # k-grid points per direction (even required for commensurate q_AFM=(π,π))
_N_FS                 : int   = 130                 # FS k-points used in the vertex q-loop; samples the full k-grid, angular resolution need to resolve the d-wave node at (π/2,π/2) and the B₁g anti-nodal hot spots.
_FS_SAMPLING          : float = 4.4                 # integration window around the Fermi level
_FS_THERMAL_THRESHOLD : float = 0.0025              # 1% of peak value of f(E)*(1-f(E)) = 1/4, used as baseline for thermal FS weight
_VF_FLOOR             : float = 1e-4                # Fermi velocity floor (prevents 1/|v_F|→∞ at hot spots), Physical scale: ~0.01·t0·a/ħ in units; Used in geometric FS sampling weight (hypot(dE_dx, dE_dy)).
_VF_FLOOR_TIGHT       : float = _VF_FLOOR * 1e-1    # = 1e-5 : tighter v_F floor in the 1/vF integration kernel (dl/vF arc-length weight); must be  _VF_FLOOR so it never dominates the physical weight;
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

# ── Cluster and regression ────────────────────────────────────────────────────
_CLUSTER_SIZE         : int   = 4                   # 2x2 plaquette
_CLUSTER_Q_REGR_THRESH: float = 8e-5                # below this, the B1g correlation signal is typically too small relative to floating‑point noise.
_REGR_EPS             : float = 1e-12               # zero protection at dividers
_REGR_T_ALPHA         : float = 0.98                # two-sided significance level for the r_J t-test (H0: r_J=1, i.e. no renormalisation)
_REGR_SHRINK_POWER    : float = 2.0                 # exponent for smooth shrinkage (≥1, higher = stronger shrinkage)
_REGR_VAR_MIN         : float = 1e-9                # variance minimum to avoid overshoot instability

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
_ALPHA_MORIYA         : float = 0.02                # Moriya damping floor: α_M ≥ this (numerical safeguard only)
_MORIYA_C             : float = 0.21                # Prefactor in α_M = C·f(δ)·sat(t/J); f(δ)=δ/(δ+DSAT) ∈(0,1), sat=Padé ∈(0,1)
_MORIYA_DSAT          : float = 0.30                # Doping saturation scale: ZRS coherence saturates ~δ=0.3 (g_t~0.46)
_MORIYA_TJ_SAT        : float = 1.0                 # Padé half-saturation at t~J; prevents J_eff↓→t/J↑→Γ_M↑ positive feedback
_RPA_BW_FACTOR        : float = 8.0                 # Bandwidth = 8·t in 2D tight-binding (square lattice, nearest-neighbour only).
_RPA_V_CAP_ALPHA      : float = 2.2                 # Perturbative RPA breaks down when V_pair ~ O(bandwidth); V_cap = α·max(8·max(|tx|,|ty|), J_eff). 2.2× headroom above the BEC-BCS crossover energy while preventing runaway at the AFM QCP
_RPA_DET_WARN         : float = 0.11                # QCP proximity warning threshold for diagnostics and SCF adaptive mixing.
_RPA_QCP_PENALTY      : float = 0.40                # α reduction per unit |det_afm|<0 past QCP (used in SCF near-critical detection, BO near_qcp flag).
_DET_AFM_FLOOR        : float = 0.5                 # default det_afm when vertex cache is absent (normal state, no QCP)
_DET_SIGN_FLIP_SCALE  : float = 0.05                # |det_afm| scale for V_d sign-flip EMA suppression (determines the sigmoid midpoint)
_EMA_SIGN_FLIP_W_MIN  : float = 0.20                # minimum w_factor on V_d sign flip; preserves adaptation even at det≈0
_EMA_SIGN_FLIP_SLOPE  : float = 6.0                 # sigmoid steepness in sign-flip EMA: w=w_min+(1-w_min)/[1+exp(-k·(|det|/floor-0.5))]
_VMAT_LOW_VAR_FRAC    : float = 0.10                # std(V)/|mean(V)| < this → vertex low-variance flag
_V_PREV_SIGN_FLOOR    : float = 1e-6                # |V_d_prev| below this → treat as zero, skip sign-flip check
_V_AFM_Q_MIN          : float = 0.70                # |q|/π > this → counted as AFM region in vertex diagnostics
_V_FWD_Q_MAX          : float = 0.35                # |q|/π < this → counted as forward-scattering region
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
_KICK_M_EXCESS_CTR    : float = 0.70                
_KICK_JCHI_EXCESS_CTR : float = 0.70
_KICK_REDUCTION_AMP   : float = 3.88                # amplitude of M-kick reduction: M_kick × (1 − this × excess)
_KICK_BOOST_Q         : float = 0.006               # Q-kick boost
_KICK_M_CLIP_LO       : float = 0.02                # hard lower clip on M_kick (normal SCF path)
_KICK_M_CLIP_HI       : float = 0.9                 # hard upper clip on M_kick
_KICK_DELTA_MAX_FRAC  : float = 0.4                 # maximum allowed seed gap as a fraction of the effective hopping scale t_eff.
_KICK_MIXING_FLOOR    : float = 0.004               # minimum mixing weight in the kick; prevents α from collapsing to zero when λ_plus is huge.
_KICK_MIXING_SCALE    : float = 4.0                 # damping scale for λ_plus in α = _MIXING / (1 + scale·log1p(λ_plus)).
_M0_S_CLIP_MAX        : float = 5.0                 # upper clip for Stoner (prevents M divergence at large J/W)
_M0_WARMSTART_MIN     : float = 0.1                 # |M| below this is treated as "no real information" (crude/near-zero seed), not a genuine converged warm start

# ── SCF iteration / mixing adaptive control ─────────────────────────────────────
_MAX_ITER             : int   = 700
_MIN_ITER             : int   = 4
_MIXING               : float = 0.06                # base weight of the newly computed residual in the solution update; lower values improve stability at the cost of slower convergence.
_ALPHA_HF             : float = 0.31                # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton)
_Q_UPDATE_PERIOD      : int   = 3                   # update Q every N inner iterations
_Q_THR_REL            : float = 0.016               # fraction of lambda_hop; Q change below this skips vertex rebuild
_Q_SEED_THR           : float = 1e-4                # if initial_Q is already nonzero, trust it as the best current estimate.
_M_THR_REL            : float = 0.01                # absolute M change threshold
_EMA_NEW_WEIGHT       : float = 0.14                # EMA weight for r_Q, r_MQ, V_d, Λ_inst
_EMA_NEW_QRW          : float = 0.38                # EMA weight for r_Q
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
_ENTROPY_CLIP         : float = 1e-12               # lower clip for f in entropy -f·ln(f)
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

# ── BO/scoring: weight floors and soft-constraint values ─────────────────────
_BO_W_HESSIAN_FLOOR   : float = 0.30                # floor for w_hessian / w_lJT_kernel when data missing or over-saturated
_BO_W_LJT_OVR_SAT     : float = 0.10                # w_lJT when λ_JT_kernel ≥ 1 (Rayleigh quotient over-saturation)
_BO_LJT_KERNEL_SIG    : float = 10.0                # sigmoid steepness k in w_lJT_kernel = 1/(1+exp(−k·(x−0.05)))
_BO_JCHI_GAPPED_CAP   : float = 0.98                # J·χ_SS(gapped) must be < this to be accepted as safe (not unconditionally AFM)
_SCORE_SOFTENING_SIG  : float = 0.05                # sigmoid width for w_softening = 1/(1+exp(jt_softening/this))
_BO_MAX_WORKERS       : int   = 6                   # hard ceiling on ThreadPoolExecutor workers
_BO_OPT_JCHI          : float = 0.875               # optimal J·χ_SS for Gauss gate (near-QCP but still metallic)
_BO_SIG_JCHI          : float = 0.15                # Gaussian width σ for J·χ_SS gate
_BO_JCHI_FLOOR        : float = 0.3                 # score floor when J·χ unavailable (jchi≈0)
_BO_JCHI_NOISE        : float = 0.05                # J·χ below this is numerical noise, apply floor
_BO_W_STONER_BAD      : float = 0.20                # score weight when AFM Stoner criterion violated
_BO_G_FALLBACK        : float = 5e-3                # overall scale for G-matrix proxy (no-gap region)
_BO_SIGMOID_W         : float = 0.30                # sigmoid width for g22_f gate (fallback-only)
_BO_SPONT_JT_PEN      : float = 0.05                # penalty floor in g22_f (used only in _g_fallback_score)
_BO_SC_HESS_SIG       : float = 0.05                # eV — sc_hessian_f sigmoid width around lambda_min=0
_BO_G22_MARGIN_CTR    : float = 0.25                # G22 sweet-spot centre for g22_margin_f sigmoid
_BO_G22_MARGIN_W      : float = 0.15                # sigmoid width for g22_margin_f
_BO_ARCH_DENOM        : float = 0.2025              # arch normalisation: (0.45)² so arch peaks at 1 at λ_JT=0.5

# ── Differential evolution (DE) scoring ──────────────────────────────────────
_DE_G22M_SAFE         : float = 0.25                # G22 value considered safely above spontaneous-JT boundary
_FEASIBILITY_THRESHOLD: float = 0.25                # penalty >= this → infeasible regardless of S4
_DE_LAMBDA_MAX_REWARD : float = 4.0                 # λ_max above this → penalised (numerically unstable / past QCP)
_DE_LAMBDA_MIN_OPT    : float = 0.15                # λ_max below this → weak pairing (S2 sigmoid centre)
_DE_LAMBDA_JT_THRESH  : float = 0.05                # Normal-state lam_JT = g²·χ_QQ/K_bare below this → S3 penalised (SC-JT window closed)

# Nambu sector pairs for normal-state (Δ=0) Lindhard sum. Excludes anomalous Part↔Hole pairs, which vanish at Δ=0.
_NORMAL_SECTOR_PAIRS  : tuple = (
    (slice(0,  6),  slice(0,  6)),  # A-A particle
    (slice(6,  12), slice(6,  12)), # B-B particle
    (slice(12, 18), slice(12, 18)), # A-A hole
    (slice(18, 24), slice(18, 24)), # B-B hole
    (slice(0,  6),  slice(6,  12)), # A-B particle
    (slice(6,  12), slice(0,  6)),  # B-A particle
    (slice(12, 18), slice(18, 24)), # A-B hole
    (slice(18, 24), slice(12, 18)), # B-A hole
)

def _scf_log(tag: str, msg: str, verbose: bool = True) -> None:
    """Thread-safe logger.  tag is left-padded to 18 chars so columns stay aligned."""
    if not verbose:
        return
    with _log_lock:
        print(f"[{tag:<18s}] {msg}", flush=True)


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
    u:                float      # —     U/t0 ratio; U = u·t0 = u·t_pd²/Δ_CT (charge-transfer: typ. 6–12)
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
        # L = 1 operators
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
        _Sz_full = np.kron(I3, Sz)   # S_z operator in the full 6D space

        H_SOC = self.lambda_soc * (
            np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)
        )
        _Lx_t2g = np.kron(Lx, I2)
        _Ly_t2g = np.kron(Ly, I2)
        _Lz_t2g = np.kron(Lz, I2)
        H_CF = (self.Delta_tetra * (_Lz_t2g @ _Lz_t2g)
                + self.Delta_B1g_static * (_Lx_t2g @ _Lx_t2g - _Ly_t2g @ _Ly_t2g))

        evals, _evecs_soc = np.linalg.eigh(H_SOC + H_CF)
        LS_op = np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)

        # B1g operator in the t2g manifold (Lx^2 - Ly^2)
        _B1g_t2g_pi = _Lx_t2g @ _Lx_t2g - _Ly_t2g @ _Ly_t2g

        # ── Helper: diagonalise Sz within a 2D Kramers subspace ─────────────────
        def _diagonalize_sz_in_doublet(v1: np.ndarray, v2: np.ndarray) -> tuple:
            """
            Find the z-polarised Kramers partners within the 2D subspace span{v1,v2}.
            Returns (up, dn, sz_up, sz_dn) where eigh gives ascending eigenvalues:
            sz_vals[0] = -|Sz| → spin-down partner,  sz_vals[1] = +|Sz| → spin-up.
            """
            U = np.column_stack((v1, v2))
            sz_vals, sz_vecs = np.linalg.eigh(U.conj().T @ _Sz_full @ U)   # ascending order
            dn    = U @ sz_vecs[:, 0]
            up    = U @ sz_vecs[:, 1]
            sz_dn = float(np.real(sz_vals[0]))
            sz_up = float(np.real(sz_vals[1]))
            return up, dn, sz_up, sz_dn
        
        # ── Identify the three Kramers doublets (sorted by <L·S>) ───────────────
        doublets = []
        for i in [0, 2, 4]:
            v  = _evecs_soc[:, i]
            vp = _evecs_soc[:, i + 1]
            up, dn, sz_up, sz_dn = _diagonalize_sz_in_doublet(v, vp)
            ls_val   = float(np.real(v.conj() @ LS_op @ v))
            doublets.append({
                'idx': i, 'energy': float(evals[i]),
                'ls_val': ls_val,
                'v': v, 'v_p': vp,
                'up': up, 'dn': dn,
                'sz_up': sz_up, 'sz_dn': sz_dn,
            })

        # Γ6: most negative <L·S> → j_eff = 1/2
        doublets.sort(key=lambda x: x['ls_val'])
        G6 = doublets[0]
        G7_candidates = doublets[1:]   # two Γ7 candidates
        
        G7_candidates.sort(key=lambda x: (x['energy'], -abs(x['sz_up'])))
        G7a, G7b = G7_candidates[0], G7_candidates[1]

        # Diagnostic cross-check with μz = Lz + 2Sz norm; in the pure-j limit both agree; disagreement flags the mixed CF/SOC regime.
        _mu_z = _Lz_t2g + 2.0 * np.kron(I3, Sz)
        def _kramers_moment(mu_op, up, dn):
            U = np.column_stack((up, dn))
            return float(np.linalg.norm(U.conj().T @ mu_op @ U))
        
        mu7 = [_kramers_moment(_mu_z, c['up'], c['dn']) for c in G7_candidates]
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
        _evecs_soc = np.column_stack([_v6_up, _v6_dn, _v7_up, _v7_dn, _v7b_up, _v7b_dn])
        new_order  = [G6['idx'], G6['idx']+1, G7a['idx'], G7a['idx']+1, G7b['idx'], G7b['idx']+1]
        evals      = evals[new_order]

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

        # B₁g JT phonon operator: U6†·(Lx²−Ly²)_t2g·U6  (6×6 Γ6⊕Γ7a⊕Γ7b subspace, real, hermitian;
        # Γ7b is kept explicitly, so its own B1g matrix elements, however weak, are carried through exactly rather than discarded; at high energy virtual processes it can no longer be neglected).
        #   D₄h (Δ_B1g_static=0): anti-diagonal real (Γ₆↔Γ₇), diagonal = 0.
        #   D₂h (Δ_B1g_static≠0): both real diagonal AND real off-diagonal elements. The diagonal A₁g component partially lifts the normal-state selection rule for χ_SQ even before superconductivity.
        _U6 = _evecs_soc[:, 0:6]
        self.B1g_op = np.asarray(np.real(_U6.conj().T @ _B1g_t2g_pi @ _U6), dtype=float)
        self.B1g_offdiag = self.B1g_op - np.diag(np.diag(self.B1g_op))

        b1g_ratio = float(np.linalg.norm(self.B1g_offdiag)) / max(float(np.linalg.norm(np.diag(self.B1g_op))), 1e-9)
        self.b1g_weight = float(b1g_ratio / (1.0 + b1g_ratio))

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

        self.U_dd = self.u * self.t0
        # Ligand coordination number: typically hybridizes with two neighboring metal atoms
        z_O = self.Z / 2.0
        # Weak-hybridization limit: recovers the Wannier-derived form U_pp ≈ r0 * U_dd - alpha * t_pd², where r0 encodes the bare p-d orbital extent mismatch and alpha captures delocalization-driven screening.
        self.U_pp = (self.Upp_ratio_bare * self.U_dd) / (1.0 + z_O * self.t_pd**2 / (self.Delta_CT * (self.Delta_CT + self.U_dd)))

        self.J_pdct = 1.0 / self.U_dd + 1.0 / (self.Delta_CT + self.U_pp / 2.0)
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

    def estimate_M0(self, target_doping: float, stoner: float = None, M_seed: float = None) -> float:
        """Warm-start AFM order-parameter estimate."""
        if stoner is None:
            g_t, g_J, _, _ = self.get_gutzwiller_factors(target_doping)
            _J_eff = self.Z * self.exchange_channels(0.0, 1 - target_doping, self.t0, self.t0, g_J)[0][0]
            stoner = float(np.clip(_J_eff / (np.pi * self.t0 * g_t * (1.0 + (self.Z * self.J_pdct * self.t0 / (np.pi * g_t))**2)), _MATH_EPS, _M0_S_CLIP_MAX))
        
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
    
    def exchange_channels(self, Q: float, n_kspace: float, tx_bare: float, ty_bare: float, g_J: float, r_Q: float = 0.0) -> Tuple[np.ndarray, float]:
        """
        Q-dependent multipolar exchange in the full [Γ₆↑, Γ₆↓, Γ₇ₐ↑, Γ₇ₐ↓, Γ₇ᵦ↑, Γ₇ᵦ↓] basis (no downfolding).
            D₄h decomposition: J(Q) = J_A1g(Q)·diag(1,1,η_J7a²,η_J7a²,η_J7b²,η_J7b²) + J_B1g(Q)·B1g_op
        
        Due to hole doping:
            n_kspace → 1 → J_eff → 4J_bare (Mott insulator)
            n_kspace → 0 → J_eff → 0 (empty band)

        No additional ZRS coherence factor: which is already contained in the effective hopping renormalization
        J is now of order t0²~(t_pd²/Delta_CT)² (4th order superexchange), which already includes the suppression of ligand hybridization
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
        J_A1g_diag = n_kspace * g_J * self.J_pdct * (tx_bare**2 + ty_bare**2) * np.array(
            [1.0, 1.0, eta_J7a**2, eta_J7a**2, eta_J7b**2, eta_J7b**2]
        ) / 2.0  # single-bond ZSA superexchange: even, longitudinal scale
        # r_Q is an additive residual beyond the bare (Gutzwiller-renormalised) coupling.
        J_B1g_scalar = n_kspace * g_J * self.J_pdct * (tx_bare**2 - ty_bare**2) * np.sqrt(eta_J7a**2 + eta_J7b**2) * (1.0 + r_Q) / 2.0  # odd, transverse scale
        return J_A1g_diag, J_B1g_scalar

    def moriya_gamma(self, doping: float, t_eff: float, J_eff: float) -> float:
        """
        Moriya spin-fluctuation damping: Γ_M = α_M · t_eff² / J_eff  [eV]

            α_M = max(C · f(δ) · sat(t/J),  _ALPHA_MORIYA)

        f(δ) = δ/(δ+_MORIYA_DSAT): saturating doping factor ∈(0,1). Sub-linear slope
        breaks the positive-feedback loop J_eff↓→t/J↑→α↑→Γ_M↑ at high doping.

        sat(t/J) = (t/J)/(_MORIYA_TJ_SAT+t/J): Padé saturation ∈(0,1).
        Moriya's quasi-local derivation assumes t/J is not a large expansion parameter;
        at moderate doping in the charge-transfer insulator (t/J~O(1)), the linear form is outside
        its validity regime. sat→1 ensures Γ_M is bounded even as J_eff→0.

        Limits: δ→0 → Γ_M→0 (sharp magnons); t→∞ → Γ_M bounded by C·f·t²/J; J→∞ → Γ_M~1/J→0.
        """
        abs_d  = max(abs(doping), 1e-4)
        J_safe = max(abs(J_eff),  1e-9)
        t_safe = max(abs(t_eff),  1e-9)

        f_delta = abs_d / (abs_d + _MORIYA_DSAT)
        r       = t_safe / J_safe
        sat_tj  = r / (_MORIYA_TJ_SAT + r)

        alpha   = _MORIYA_C * f_delta * sat_tj
        Gamma_M = float(max(alpha, _ALPHA_MORIYA)) * t_safe**2 / J_safe   # [eV]
        return min(Gamma_M, 4.0 * t_safe**2 / (np.pi * J_safe) )

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

def _lindhard_bubble(sector_pairs: tuple, E_k_all: np.ndarray, V_k_all: np.ndarray, f_k_all: np.ndarray, shift_idx: np.ndarray, w: np.ndarray, vw_sq: np.ndarray, eta: float, kT: float) -> np.ndarray:
    """Static (ω=0) full-BZ Lindhard bubble for a given q-shift and set of sector pairs. Uses uniform BZ weights w throughout."""
    E_kQ = E_k_all[shift_idx]
    V_kQ = V_k_all[shift_idx]
    f_kQ = _fermi_function(E_kQ, kT)

    df = f_k_all[:, :, None] - f_kQ[:, None, :]
    dE = E_kQ[:, None, :] - E_k_all[:, :, None]
    
    df_dE_k  = -f_k_all * (1.0 - f_k_all) / kT
    df_dE_kQ = -f_kQ * (1.0 - f_kQ) / kT
    df_dE_avg = 0.5 * (df_dE_k[:, :, None] + df_dE_kQ[:, None, :])  # (N_k, 24, 24)
    
    # Unified, continuous Lehmann kernel:
    #   |ΔE| ≫ η  →  Δf·ΔE / ΔE²  (standard particle-hole bubble)
    #   |ΔE| ≪ η  →  -f'          (Fermi surface term in Taylor limit case)
    
    df_safe = np.where(np.abs(df) > _FD_MASK_DF, df, 0.0)
    de_safe = np.where(np.abs(dE.real) > _FD_MASK_DE, dE.real, 0.0)
    
    kernel = (df_safe * de_safe + (-df_dE_avg) * eta**2) / (de_safe**2 + eta**2)
    
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

def _unique_q_pairs(fs_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return i_idx, j_idx, unique_q, inv_idx

def _fs_integration_weights(pts: np.ndarray, vF_arr: np.ndarray) -> np.ndarray:
    """
    Integration weight w_i = dl_i / ( (2π)² · |vF_i| ) for each Fermi surface point.
    The proper FS integration measure for a 2D BZ is:
        ∫_{FS} dS / |vF|  ≈  Σ_i dl_i / |vF_i|
    where dl_i is the arc-length element around the FS contour.
    
    The returned weight includes the BZ area normalisation (2π)² so that:
        Σ_i w_i · f(k_i)  ≈  ∫_{BZ} d²k f(k) δ(ε_k)
    """
    N = len(pts)
    angles     = np.arctan2(pts[:, 1], pts[:, 0])
    sort_idx   = np.argsort(angles)
    sorted_pts = pts[sort_idx]
    diff_prev  = sorted_pts - np.roll(sorted_pts,  1, axis=0)
    diff_next  = sorted_pts - np.roll(sorted_pts, -1, axis=0)
    dl_sorted  = 0.5 * (np.linalg.norm(diff_prev, axis=1)
                        + np.linalg.norm(diff_next, axis=1))
    dl = np.empty(N)
    dl[sort_idx] = dl_sorted
    return dl / (_BZ_NORM * np.maximum(np.abs(vF_arr), _VF_FLOOR_TIGHT))

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
        self.shift_table    = params.shift_table              # (nk, nk, N_k) int32 — cyclic shift index

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

    def _get_vbdg(self) -> 'VectorizedBdG':
        if self._vbdg is None:
            self._vbdg = VectorizedBdG(self)
        return self._vbdg

    def _get_chi0_norm_cache(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, vbdg: 'VectorizedBdG') -> Tuple[float, float]:
        """
        Cached on (M, Q, n_kspace, mu, g_t, g_J)
        Within a single SCF iteration these quantities are constant across the entire q-loop, avoiding O(N_q) redundant eigh calls on the N_k × 24 matrix.
        Return (E_k_all, V_k_all) for the Δ=0 BdG on k_points.
        """
        key = (M, Q, n_kspace, mu, g_t, g_J)
        if self._chi0_norm_cache is not None:
            _E, _V, _M_old, _Q_old, _n_kspace_old, _mu_old, _gt_old, _gJ_old = self._chi0_norm_cache
            if (abs(M - _M_old) < _M_THR_REL * max(abs(M), 1.0) and
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

    def estimate_gutzwiller_factors_occupation_based(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float) -> Tuple[float, float]:
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

    def _clone_solver_at_T(self, T: float) -> 'RMFT_Solver':
        """
        Return a fully independent solver clone with kT = T.

        Performs a shallow copy of self and self.p, then resets all mutable
        per-solve caches via _reset_transient_state().  The immutable bare
        stiffness _K_bare is carried over unchanged.
        """
        s = copy.copy(self)
        s.p = copy.copy(self.p)
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

    def _calc_dHdQ(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float, F67s_mf: float = 0.0) -> np.ndarray:
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
            return self.p.exchange_channels(Qv, n_kspace, tx_v, ty_v, g_J, r_Q)
        
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
        dH_loc = (self.p.g_JT * self.B1g_op).astype(complex)
        dH_loc_Q = (self.p.g_JT * Q * self.B1g_op).astype(complex)

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

    def compute_JT_rigidity_from_exchange(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float = 0.0, F67s_mf: float = 0.0, Q_Eg2: float = 0.0) -> float:
        """
        Exchange contribution to JT stiffness: ∂²F_ex/∂Q², obtained as the numerical second derivative of free energy,
        matching the free-energy functional differentiated analytically in compute_dF_dM_and_d2F.

        The M=0 baseline removes the direct electron-phonon (band-JT) contribution, present even without magnetic order

        K_eff = K_lattice + ∂²F_ex/∂Q²
        (negative = softening, positive = stiffening)

        SC limit: M→0 ⇒ F_ex(0,Q)=0 ∀Q ⇒ ∂²F_ex/∂Q² = 0 exactly.
        """
        eps2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
        eps = np.sqrt(eps2)
        
        def _F_ex(Q_val: float) -> float:
            # V_s=V_d=0 (no condensation term), K_eff_for_free_energy=0 (lattice term added below).
            return self._compute_bdg_free_energy(M, Q_val, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2) - self._compute_bdg_free_energy(0.0, Q_val, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2)
        
        F0 = _F_ex(Q)
        Fp = _F_ex(Q + eps)
        Fm = _F_ex(Q - eps)
        return self._K_bare + (Fp - 2.0 * F0 + Fm) / eps2

    def compute_K_eff_full(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float = 0.0, F67s_mf: float = 0.0, Q_Eg2: float = 0.0) -> float:
        """Total (bare + exchange) stiffness via numerical 2nd derivative of F_total."""
        eps2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
        eps = np.sqrt(eps2)

        F0 = self._compute_bdg_free_energy(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2)
        Fp = self._compute_bdg_free_energy(M, Q + eps, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2)
        Fm = self._compute_bdg_free_energy(M, Q - eps, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2)
        return self._K_bare + (Fp - 2.0 * F0 + Fm) / eps2, F0

    def compute_chi_ss_with_infinitesimal_gap(self, M: float, G_res: dict, target_doping: float, n_kspace: float, delta_test: float = 1e-4) -> float:
        """
        Compute χ_SS at q=(π,π) with a tiny d-wave gap, using the unified Nambu susceptibility.
        
        Physical picture: a small SC gap suppresses low-energy spin fluctuations,
        mimicking the SC-state suppression without solving the full gap equation.
        """
        vbdg = self._get_vbdg()
        t_eff = G_res['g_t'] * self.p.t0
        
        # Fresh buffer to avoid mutating SCF cache
        _H_buf = vbdg._H_stack.copy()
        
        ev, ec = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M, 0.0, 0.0j, complex(delta_test), n_kspace, G_res['mu_n'], G_res['g_t'], G_res['g_J'], out=_H_buf)
            )
        
        # q=(π,π) shift index
        dk = 2.0 * np.pi / _NK
        nx = int(round(np.pi / dk)) % _NK
        ny = int(round(np.pi / dk)) % _NK
        shift_idx = self.shift_table[nx, ny]
        
        # χ_SS = ⟨⟨Sz; Sz⟩⟩: use the full 24×24 Sz_nambu matrix
        eta = max(0.01 * self.p.t0, _ETA_T_FRAC * self.kT, _FD_MASK_DE)
        V_kQ = ec[shift_idx]
        M_A_bands = np.einsum('kan,ab,kbm->knm', ec.conj(), self.Sz_nambu, V_kQ)
        M_B_bands = np.einsum('kam,ab,kbn->kmn', V_kQ.conj(),  self.Sz_nambu, ec)
        chi_ss_raw = self._compute_nambu_susceptibility(ev, M_A_bands, M_B_bands, shift_idx, eta).real
        
        # Moriya damping
        chi_ss_pos = max(chi_ss_raw, 0.0)
        J_A1g_diag, _ = self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, G_res['g_J'], G_res['r_Q'])
        J_eff = self.p.Z * J_A1g_diag[0]
        _Gamma_M = self.p.moriya_gamma(target_doping, t_eff, J_eff)
        return chi_ss_pos / (1.0 + _Gamma_M * chi_ss_pos)

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
        return float(np.dot(self.k_weights, exp_k)) / 4.0

    def Eg2_expectation(self, E_k_cache: tuple) -> float:
        """
        Per-site ⟨Eg2_op⟩ in the BdG ground state. Identical construction to
        B1g_expectation, just contracted against Eg2_24 instead of B1g_24.
        """
        ev, ec = E_k_cache
        f_n = _fermi_function(ev, self.kT)
        diag_qp = np.einsum('kan, ab, kbn -> kn', ec.conj(), self.Eg2_24, ec).real
        exp_k = np.einsum('kn,kn->k', diag_qp, f_n)
        return float(np.dot(self.k_weights, exp_k)) / 4.0

    def _compute_chi_tau(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float = 0.0, F67s_mf: float = 0.0) -> Dict:
        """
        JT orbital susceptibility χ_τ = ∂⟨B1g_op⟩/∂(g_JT·Q) via Richardson-extrapolated
        central finite difference of the per-site ⟨B1g_op⟩ expectation value.

        δχ_τ = χ_τ(Δ≠0) − χ_τ(Δ=0) isolates the condensate contribution.
        In D₄h: ⟨B1g_op⟩=0 exactly in normal state → δχ_τ = χ_τ(Δ≠0).
        In D₂h: a small normal-state baseline can exist; subtraction prevents D₂h
        signal from masquerading as SC-triggered.

        Richardson extrapolation (3 primary step sizes h, h/2, h/4):
          R1 = (4·CD(h/2)−CD(h))/3, R2 = (4·CD(h/4)−CD(h/2))/3, est = mean(R1,R2).
          Converged: |R1−R2|/max(|est|,ε) < 3%  → return est (O(h⁴) accurate).
          Nonlinear: |CD(h)−CD(h/2)|/max(|CD(h/2)|,ε) > 20%
            → try h/8 fallback; if still nonlinear → return 0.0 (conservative).
        """
        vbdg = self._get_vbdg()
        g_JT = max(self.p.g_JT, 1e-12)
        scale = self.p.Delta_CF / g_JT
        h_floor = 1e-4
        h = float(np.clip(1e-3 * max(abs(Q), scale), h_floor, 0.05 * scale))

        def _cd(dq: float, ds: complex, dd: complex) -> float:
            """Central difference d⟨B1g⟩/dQ at step dq."""
            vp = self.B1g_expectation(*self.p.effective_hopping_anisotropic(Q + dq), np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q + dq, ds, dd, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, out=vbdg._H_stack)))
            vm = self.B1g_expectation(*self.p.effective_hopping_anisotropic(Q - dq), np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q - dq, ds, dd, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, out=vbdg._H_stack)))
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
            chi_tau_sc = w_sc * dB1g_sc / g_JT   # sign physical: negative = stiffening

        if w_n == 0.0:
            chi_tau_n = 0.0   # unreliable baseline — skip subtraction
        else:
            chi_tau_n = w_n * dB1g_n / g_JT

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

    def _chi_QQ_matrix_elements(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float = 0.0, F67s_mf: float = 0.0, Q_Eg2: float = 0.0, return_matrix: bool = False):
        """Bare JT orbital susceptibility: χ_QQ = −∂²Ω/∂Q² evaluated at Δ=0. χ_QQ is a normal-state quantity
        return_matrix=True: returns the full 2×2 matrix {χ_QQ[B1g,B1g], χ_QQ[B1g,Eg2]; χ_QQ[Eg2,B1g],
        χ_QQ[Eg2,Eg2]} via mixed finite differences of the SAME grand potential Ω(Q, Q_Eg2), using a
        9-point stencil (the B1g-only diagonal term reuses the original 3-point formula exactly).
        """
        eps2 = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2
        dQ   = np.sqrt(eps2)
        vbdg = self._get_vbdg()

        def omega(Qval, Qeg2val):
            ev = np.linalg.eigvalsh(
                vbdg._build_H_stack(vbdg._kpts, M, Qval, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, vbdg._H_stack, Qeg2val)
                )
            arg = np.clip(np.abs(ev) / self.kT, 0.0, _FERMI_ARG_CLIP)
            Omega_kn = np.minimum(0.0, ev) - self.kT * np.log1p(np.exp(-arg))
            return np.sum(self.k_weights[:, None] * Omega_kn)
        
        Ωp = omega(Q + dQ, Q_Eg2)
        Ω0 = omega(Q, Q_Eg2)
        Ωm = omega(Q - dQ, Q_Eg2)
        
        # −∂²Ω/∂Q²: positive for a stable metal (χ_QQ > 0 convention used in G3[2,2]); division by 4.0 due to 2 (sublattice) * 2 (particle-hole) Nambu doubling
        chi_QQ = -(Ωp - 2.0 * Ω0 + Ωm) / (4.0 * eps2)
        g_JT2 = max(self.p.g_JT**2, _MATH_EPS)
        chi_QQ_n = chi_QQ / g_JT2

        if not return_matrix:
            return chi_QQ_n
        
        eps2_e = _JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q_Eg2**2
        dQe    = np.sqrt(eps2_e)
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
        g_cross  = max(self.p.g_JT * self.p.g_Eg2, _MATH_EPS)
        chi = np.array([[chi_QQ / g_JT2,      chi_QQ_cross / g_cross],
                        [chi_QQ_cross / g_cross, chi_QQ_eg2 / g_Eg2_2]], dtype=float)
        return chi

    def estimate_chi_SQ_q_full(self, target_doping: float, M: float, Q: float, Delta_s: float, Delta_d: float, n_kspace: float, mu: float, J_eff: float, r_Q: float, F67s_mf: float, n_q: int):
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
            vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, out=vbdg._H_stack)
            )
        
        f_k_n  = _fermi_function(E_k_n, self.kT)
        eta_n  = max(_ETA_T_FRAC * self.kT, _ETA_GRID_FLOOR * self.p.t0)    # Normal-state: thermal broadening dominates (bands gapped by h_afm).
        _Gamma_M = self.p.moriya_gamma(target_doping, np.sqrt(0.5 * (tx**2 + ty**2)), J_eff) 
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
            _chi_SS_sc, _chi_SQ_sc, _chi_QQ_sc = self.get_susceptibilities_sc(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, q, _Gamma_M, r_Q, F67s_mf, (E_k_sc, V_k_sc))
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
        The (N_k,24,24) Lehmann-weight kernel shared by every χ_AB(q) contraction at a
        given (E_k_all, shift_idx, eta) -- i.e. it does NOT depend on which vertex
        matrices M_A/M_B are being sandwiched. Factored out of
        _compute_nambu_susceptibility so callers that need χ_SS, χ_SQ, χ_QS, χ_QQ at the
        SAME q (e.g. get_susceptibilities_sc) can build this once and reuse it four times,
        instead of rebuilding the identical (N_k,24,24) arrays four times over.
        """
        E_kQ = E_k_all[shift_idx]                     # (N_k, 24)
        f_k  = _fermi_function(E_k_all, self.kT)      # (N_k, 24)
        f_kQ = _fermi_function(E_kQ, self.kT)         # (N_k, 24)
        
        df = f_k[:, :, None] - f_kQ[:, None, :]       # (N_k, 24, 24)   f_n(k) − f_m(k+q)
        dE = E_kQ[:, None, :] - E_k_all[:, :, None]   # (N_k, 24, 24)   E_m(k+q) − E_n(k)
        
        df_dE_k  = -f_k * (1.0 - f_k) / self.kT
        df_dE_kQ = -f_kQ * (1.0 - f_kQ) / self.kT
        df_dE_avg = 0.5 * (df_dE_k[:, :, None] + df_dE_kQ[:, None, :])  # (N_k, 24, 24)
        
        df_safe = np.where(np.abs(df) > _FD_MASK_DF, df, 0.0)
        de_safe = np.where(np.abs(dE.real) > _FD_MASK_DE, dE.real, 0.0)
        
        # Continuous kernel for ALL ΔE values
        kernel = (df_safe * de_safe + (-df_dE_avg) * eta**2) / (de_safe**2 + eta**2)
        
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
    
    def build_dHdQ_band_basis(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float, F67s_mf: float, V_k_all: np.ndarray, V_kQ: np.ndarray, dHdQ_precomputed: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the ∂H/∂Q vertex matrix in the band basis, suitable for the Nambu Lehmann sum — Ward-identity consistent.
        Returns BOTH bra/ket role matrices built from the SAME local dHdQ(k)

        `dHdQ` depends only on (M, Q, n_kspace, mu, g_t, g_J, r_Q, F67s_mf) — never on q.
        """
        dHdQ = dHdQ_precomputed if dHdQ_precomputed is not None else self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J, r_Q, F67s_mf)

        # Transform to band basis
        M_A_bands_SQ = np.einsum('kan,kab,kbm->knm', V_k_all.conj(), dHdQ, V_kQ, optimize=True)
        M_B_bands_SQ = np.einsum('kam,kab,kbn->kmn', V_kQ.conj(),    dHdQ, V_k_all, optimize=True)
        return M_A_bands_SQ, M_B_bands_SQ

    def _diamagnetic_QQ_term(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float, F67s_mf: float, E_k_all: np.ndarray, V_k_all: np.ndarray) -> float:
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

        The derivative is evaluated at fixed M, μ, g_t, g_J, r_Q and F67s_mf.
        Thus this is the explicit-Hamiltonian second derivative (∂²H/∂Q²)_{M,μ,g_t,g_J,r_Q,F67s_mf},
        not the total second derivative of the fully self-consistent free energy along an SCF solution branch.

        The supplied E_k_all and V_k_all are the eigenvalues/eigenvectors of the ORIGINAL Hamiltonian at Q.
        They are used only to evaluate the expectation value of ∂²H/∂Q² in that state.
        """
        # The outer stencil stays accurate down to a roundoff floor roughly 2-3 orders of magnitude below this step
        h_QQ = np.sqrt(_JT_FD_H2_BASE + _JT_FD_H2_QCOEF * Q**2)

        dHdQ_m2 = self._calc_dHdQ(M, Q - 2.0 * h_QQ, n_kspace, mu, g_t, g_J, r_Q, F67s_mf)
        dHdQ_m1 = self._calc_dHdQ(M, Q - h_QQ, n_kspace, mu, g_t, g_J, r_Q, F67s_mf)
        dHdQ_p1 = self._calc_dHdQ(M, Q + h_QQ, n_kspace, mu, g_t, g_J, r_Q, F67s_mf)
        dHdQ_p2 = self._calc_dHdQ(M, Q + 2.0 * h_QQ, n_kspace, mu, g_t, g_J, r_Q, F67s_mf)
        
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

    def get_susceptibilities_sc(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, q: np.ndarray, Gamma_M: float, r_Q: float, F67s_mf: float, E_k_cache: tuple, apply_diamagnetic_QQ: bool = False, dHdQ_precomputed: np.ndarray = None) -> Tuple[float, float, float]:
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

        # Lehmann kernel depends only on (E_k_all, shift_idx, eta)
        kernel = self._compute_nambu_kernel(E_k_all, shift_idx, eta)

        # Vertex matrices in band basis
        M_A_bands = np.einsum('kan,ab,kbm->knm', V_k_all.conj(), self.Sz_nambu, V_kQ, optimize=True)
        M_B_bands = np.einsum('kam,ab,kbn->kmn', V_kQ.conj(), self.Sz_nambu, V_k_all, optimize=True)
        # χ_SS: spin-spin, both vertices are Sz_nambu
        chi_SS_cplx = self._compute_nambu_susceptibility(E_k_all, M_A_bands, M_B_bands, shift_idx, eta, kernel_precomputed=kernel)

        # ∂H/∂Q vertex, both roles
        M_A_bands_SQ, M_B_bands_SQ = self.build_dHdQ_band_basis(M, Q, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, V_k_all, V_kQ, dHdQ_precomputed=dHdQ_precomputed)

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
            chi_QQ_val -= self._diamagnetic_QQ_term(M, Q, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, E_k_all, V_k_all)   # eV/Å²
        
        # ---- Normalise to common 1/eV units ----
        g_JT = self.p.g_JT
        chi_SS = chi_SS_val                   # 1/eV
        chi_SQ = chi_SQ_val / g_JT            # (1/Å) / (eV/Å) = 1/eV
        chi_QS = chi_QS_val / g_JT            # (1/Å) / (eV/Å) = 1/eV
        chi_QQ = chi_QQ_val / (g_JT * g_JT)   # (eV/Å²) / (eV²/Å²) = 1/eV

        # Symmetric average of χ_SQ/χ_QS
        chi_SQ_sym = 0.5 * (chi_SQ + chi_QS)

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
        
        # ---- Moriya damping on spin channel ----
        chi_SS = chi_SS / max(1.0 + Gamma_M * chi_SS, _MATH_EPS)
        return chi_SS, chi_SQ_sym, chi_QQ
    
    def _rpa_det(self, J: float, V: float, chi_SS_moriya: float, chi_SQ_v: float, chi_QS_v: float, chi_Qr_Q: float, U_SQ: float = 0.0):
        """
        det(I − U·χ₀), with the bare interaction matrix U = [[J, U_SQ], [U_SQ, V]].

        U_SQ is the off-diagonal spin–orbital cross-vertex, U_SQ = r_MQ·√(J·V), with
        r_MQ obtained from exact cluster diagonalisation: the system has a genuine effective
        spin–orbitalcross-interaction, so the bare vertex matrix is NOT diagonal in the
        (spin, JT) channel basis — U_SQ=0 silently throws that interaction away.
        """
        a = 1.0 - (J * chi_SS_moriya + U_SQ * chi_QS_v)
        b = -(J * chi_SQ_v + U_SQ * chi_Qr_Q)
        c = -(U_SQ * chi_SS_moriya + V * chi_QS_v)
        d = 1.0 - (U_SQ * chi_SQ_v + V * chi_Qr_Q)
        return a * d - b * c, a, b, c, d

    def _rpa_vertex(self, J: float, V: float, chi_SS_moriya: float, chi_SQ_v: float, chi_QS_v: float, chi_Qr_Q: float, V_cap: float, U_SQ: float = 0.0) -> float:
        det, a, b, c, d = self._rpa_det(J, V, chi_SS_moriya, chi_SQ_v, chi_QS_v, chi_Qr_Q, U_SQ)

        M_frob    = math.sqrt(a*a + b*b + c*c + d*d)  # Frobenius norm sets natural scale of matrix
        det_floor = max(_MATH_EPS, 1e-4 * M_frob)
        if abs(det) < det_floor:
            det_safe = math.copysign(det_floor, det) if det != 0.0 else det_floor
        else:
            det_safe = det

        i00 =  d / det_safe
        i01 = -b / det_safe
        i10 = -c / det_safe
        i11 =  a / det_safe

        rss = i00 * chi_SS_moriya + i01 * chi_QS_v
        rqq = i10 * chi_SQ_v      + i11 * chi_Qr_Q
        rsq = i00 * chi_SQ_v      + i01 * chi_Qr_Q
        rqs = i10 * chi_SS_moriya + i11 * chi_QS_v
        Vp  = J**2 * rss + V**2 * rqq + J * V * (rsq + rqs)

        if abs(Vp) > V_cap:
            Vp = math.copysign(V_cap, Vp)

        if not math.isfinite(Vp):
            Vp = V_cap
        return float(Vp)

    def _make_vertex_params(self, target_doping: float, tx: float, ty: float, g_t: float, J_eff: float) -> Tuple[float, float, float]:
        Gamma_M = self.p.moriya_gamma(target_doping, np.sqrt(0.5 * (tx**2 + ty**2)), J_eff)  # bare Moriya damping Γ_M at (ω=0, q→Lindhard)
        V_JT    = self.p.g_JT**2 / max(self._K_bare, _MATH_EPS)  # bare JT pairing vertex
        V_cap   = _RPA_V_CAP_ALPHA * max(_RPA_BW_FACTOR * max(abs(tx), abs(ty), 1e-6), J_eff)  # BEC boundary (8t), UV cap on the RPA pairing vertex
        return Gamma_M, V_JT, V_cap

    def _build_fs_vertex_matrices(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, J_eff: float, Gamma_M: float, V_JT: float, V_cap: float, prefer_dwave: bool, dHdQ_precomputed: np.ndarray = None) -> dict:
        """Build the full and JT-only RPA vertex matrices on a given set of Fermi-surface points."""
        # --- Fermi-surface points ---
        fs_pts, vF_arr, fs_idx, ev, ec = self._get_fs_points(M, Q, n_kspace, mu, g_t, g_J, store_cache=True, compute_vF=True, prefer_dwave=prefer_dwave)
        N_fs = len(fs_pts)
        inv_vF = _fs_integration_weights(fs_pts, vF_arr)
        i_idx, j_idx, unique_q, inv_idx = _unique_q_pairs(fs_pts)

        # Build the χ₀ cache once for the normal state
        ev, ec = self._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self._get_vbdg())
        
        if dHdQ_precomputed is None:
            dHdQ_precomputed = self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J, 0.0, 0.0)

        n_q = len(unique_q)
        V_unique = np.empty(n_q, dtype=float)
        V_spin_u = np.empty(n_q, dtype=float)
        V_jt_u   = np.empty(n_q, dtype=float)

        for u_idx, q_u in enumerate(unique_q):
            chi_SS, chi_SQ, chi_QQ = self.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, q_u, Gamma_M, 0.0, 0.0, (ev, ec), apply_diamagnetic_QQ=False, dHdQ_precomputed=dHdQ_precomputed)
            V_unique[u_idx] = self._rpa_vertex(J_eff, V_JT, chi_SS, chi_SQ, chi_SQ, chi_QQ, V_cap)
            V_spin_u[u_idx] = self._rpa_vertex(J_eff, 0.0,  chi_SS, chi_SQ, chi_SQ, chi_QQ, V_cap)
            V_jt_u[u_idx]   = self._rpa_vertex(0.0,   V_JT, chi_SS, chi_SQ, chi_SQ, chi_QQ, V_cap)

        # Build symmetric matrices
        V_ij_full = np.zeros((N_fs, N_fs), dtype=float)
        V_ij_full[i_idx, j_idx] = V_unique[inv_idx]
        V_ij_full = 0.5 * (V_ij_full + V_ij_full.T)

        V_ij_jt = np.zeros((N_fs, N_fs), dtype=float)
        V_ij_jt[i_idx, j_idx] = V_jt_u[inv_idx]
        V_ij_jt = 0.5 * (V_ij_jt + V_ij_jt.T)

        return {
            'fs_pts':    fs_pts,
            'vF_arr':    vF_arr,
            'inv_vF':    inv_vF,
            'fs_idx':    fs_idx,
            'ev':        ev,
            'ec':        ec,
            'V_ij_full': V_ij_full,
            'V_ij_jt':   V_ij_jt,
            'V_unique':  V_unique,
            'V_spin_u':  V_spin_u,
            'V_jt_u':    V_jt_u,
            'unique_q':  unique_q,
            'inv_idx':   inv_idx,
            'i_idx':     i_idx,
            'j_idx':     j_idx,
        }

    def compute_pairing_kernel_and_build_cache(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, J_eff: float, Gamma_M: float, V_JT: float, V_cap: float, det_afm_sc: float = 1.0, solve_state: '_SolveState' = None) -> Dict:
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
        # Build vertex matrices on the two FS sets.
        dHdQ_shared = self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J, 0.0, 0.0)
        vdata_d = self._build_fs_vertex_matrices(M, Q, n_kspace, mu, g_t, g_J, J_eff, Gamma_M, V_JT, V_cap, prefer_dwave=True, dHdQ_precomputed=dHdQ_shared)
        vdata_s = self._build_fs_vertex_matrices(M, Q, n_kspace, mu, g_t, g_J, J_eff, Gamma_M, V_JT, V_cap, prefer_dwave=False, dHdQ_precomputed=dHdQ_shared)

        # --- unpack d-FS data ---
        V_ij_full = vdata_d['V_ij_full']
        V_ij_jt_d = vdata_d['V_ij_jt']
        unique_q  = vdata_d['unique_q']
        V_unique  = vdata_d['V_unique']
        V_spin_u  = vdata_d['V_spin_u']
        V_jt_u    = vdata_d['V_jt_u']
        i_idx     = vdata_d['i_idx']
        j_idx     = vdata_d['j_idx']
        inv_vF_d  = vdata_d['inv_vF']
        fs_pts_d  = vdata_d['fs_pts']
        fs_idx    = vdata_d['fs_idx']
        vF_arr    = vdata_d['vF_arr']
        ev        = vdata_d['ev']
        ec        = vdata_d['ec']

        # ---- unpack s-FS data ----
        inv_vF_s  = vdata_s['inv_vF']
        fs_pts_s  = vdata_s['fs_pts']
        V_ij_jt_s = vdata_s['V_ij_jt']

        # --- basis functions ---
        phi_d   = np.cos(fs_pts_d[:, 0]) - np.cos(fs_pts_d[:, 1])
        phi_s_d = np.ones(len(fs_pts_d), dtype=float)       # constant on d-FS
        phi_s_s = np.ones(len(fs_pts_s), dtype=float)       # constant on s-FS

        sqrt_inv_vF_d = np.sqrt(inv_vF_d)
        sqrt_inv_vF_s = np.sqrt(inv_vF_s)

        # weighted kernel matrices
        W_mat_d = np.outer(sqrt_inv_vF_d, sqrt_inv_vF_d)
        W_mat_s = np.outer(sqrt_inv_vF_s, sqrt_inv_vF_s)

        # --- helper for FS averages ---
        def _fs_avg(phi_a, phi_b, V_ij, W):
            return float(phi_a @ (V_ij * W) @ phi_b)

        # --- normalisation factors ---
        ns = max(float(np.dot(phi_s_s**2, inv_vF_s)), 1e-12)
        nd = max(float(np.dot(phi_d**2,   inv_vF_d)), 1e-12)

        # --- 2x2 pairing kernel (K11, K22, K12) ---
        K11 = g_Delta_s * _fs_avg(phi_s_s, phi_s_s, V_ij_jt_s, W_mat_s) / ns
        K22 = g_Delta_d * _fs_avg(phi_d,   phi_d,   V_ij_full, W_mat_d) / nd
        K12 = (math.sqrt(max(g_Delta_s * g_Delta_d, 0.0))
            * _fs_avg(phi_s_d, phi_d, V_ij_jt_d, W_mat_d)
            / max(math.sqrt(ns * nd), 1e-12))

        K_pair = np.array([[K11, K12], [K12, K22]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(K_pair)
        lambda_lin_max = float(eigvals[-1])
        v_s_raw = float(eigvecs[0, -1])
        v_d_raw = float(eigvecs[1, -1])

        # weighted basis vectors (not normalised) for Rayleigh projections on d-FS
        phi_s_w = phi_s_d * sqrt_inv_vF_d
        phi_d_w = phi_d   * sqrt_inv_vF_d
        phi_s_norm = max(float(np.dot(phi_s_w, phi_s_w)), 1e-12)
        phi_d_norm = max(float(np.dot(phi_d_w, phi_d_w)), 1e-12)

        # channel scalars (Rayleigh quotients) – exactly as they will be used in the gap equation
        V_s_scalar = float(phi_s_w @ V_ij_jt_d @ phi_s_w) / phi_s_norm
        V_s_scalar = float(np.clip(V_s_scalar, -V_cap, V_cap))

        V_d_proj   = phi_d_w @ V_ij_full
        V_d_scalar = float(np.dot(phi_d_w, V_d_proj)) / phi_d_norm
        V_d_scalar = float(np.clip(V_d_scalar, -V_cap, V_cap))

        # --- V_d EMA: only if solve_state is provided (only in non-linear SCF) ---
        if solve_state is not None:
            if solve_state.V_d_ema is None:
                solve_state.V_d_ema = V_d_scalar
            else:
                sign_flipped = (V_d_scalar * solve_state.V_d_ema < 0.0
                                and abs(solve_state.V_d_ema) > _V_PREV_SIGN_FLOOR)
                if sign_flipped:
                    if V_d_scalar > 0.0:
                        _ema_w = _EMA_NEW_WEIGHT
                    else:
                        _ema_w = _EMA_NEW_WEIGHT * (_EMA_SIGN_FLIP_W_MIN + (1.0 - _EMA_SIGN_FLIP_W_MIN)
                                                    / (1.0 + math.exp(-_EMA_SIGN_FLIP_SLOPE * (abs(det_afm_sc) / _DET_SIGN_FLIP_SCALE - 0.5))))
                else:
                    kick_boost = 2.0 if solve_state._ema_kick_pending else 1.0
                    _ema_w = min(_EMA_NEW_WEIGHT * kick_boost, 1.0)
                solve_state._ema_kick_pending = False
                solve_state.V_d_ema = (1.0 - _ema_w) * solve_state.V_d_ema + _ema_w * V_d_scalar
                V_d_scalar = solve_state.V_d_ema

        # --- structure flags for SCF log ---
        V_flat = V_ij_full[i_idx, j_idx]
        vmat_low_var   = float(np.std(V_flat)) < _VMAT_LOW_VAR_FRAC * abs(float(np.mean(V_flat))) + 1e-12
        vmat_same_sign = (float(np.min(V_flat)) > 0.0) or (float(np.max(V_flat)) < 0.0)

        # --- q-resolved vertex diagnostics ---
        q_norms = np.linalg.norm(unique_q, axis=1)
        afm_mask = q_norms > np.pi * _V_AFM_Q_MIN
        fwd_mask = q_norms < np.pi * _V_FWD_Q_MAX
        V_afm_mean = float(np.mean(V_unique[afm_mask])) if afm_mask.any() else float('nan')
        V_fwd_mean = float(np.mean(V_unique[fwd_mask])) if fwd_mask.any() else float('nan')
        V_neg_frac = float(np.mean(V_unique < 0.0))

        # --- Inter‑channel coupling V_sd (d‑FS, same weighting as diagonals) ---
        V_sd = _fs_avg(phi_s_d, phi_d, V_ij_jt_d, W_mat_d) / max(math.sqrt(phi_s_norm * phi_d_norm), 1e-12)

        # --- JT‑only diagonal projection (d‑FS, same weighting) ---
        V_dd_JT = float(phi_d_w @ V_ij_jt_d @ phi_d_w) / phi_d_norm

        # --- linearised 2×2 kernel with full weight (for diagnostic only) ---
        K_sd = math.sqrt(max(g_Delta_s * g_Delta_d, 0.0)) * V_sd
        K2_JT = np.array([[g_Delta_s * V_s_scalar, K_sd],
                         [K_sd, g_Delta_d * V_dd_JT]], dtype=float)
        evec_max = np.array([v_s_raw, v_d_raw])
        lambda_JT_kernel = float(evec_max @ K2_JT @ evec_max)

        # --- gap vector and channel fractions ---
        psi_s = phi_s_d * sqrt_inv_vF_d
        psi_s /= max(np.linalg.norm(psi_s), 1e-12)
        psi_d = phi_d * sqrt_inv_vF_d
        psi_d /= max(np.linalg.norm(psi_d), 1e-12)
        gap_vector = v_s_raw * psi_s + v_d_raw * psi_d
        w = np.abs([v_s_raw, v_d_raw])
        frac = w / max(w.sum(), 1e-12)

        # --- relative gain from inter‑channel mixing ---
        max_diag = max(K11, K22)
        lambda_gain_rel = (lambda_lin_max - max_diag) / max(abs(max_diag), 1e-12) if max_diag > 0 else 0.0

        # --- mean vertex values for diagnostics ---
        V_spin_mean = float(np.mean(V_spin_u))
        V_JT_mean   = float(np.mean(V_jt_u))
        V_rpa_mean  = float(np.mean(V_unique))

        # ---- assemble vertex cache ----
        vertex_cache = {
            'M':               M,
            'Q':               Q,
            'fs_pts':          fs_pts_d,
            'vF_arr':          vF_arr,
            'fs_idx':          fs_idx,
            'ev':              ev,
            'ec':              ec,
            'V_s_scalar':      V_s_scalar,
            'V_d_scalar':      V_d_scalar,
            'V_d_proj':        V_d_proj.copy(),
            'V_sd':            V_sd,
            'vmat_low_var':    vmat_low_var,
            'vmat_same_sign':  vmat_same_sign,
            'V_afm_mean':      V_afm_mean,
            'V_fwd_mean':      V_fwd_mean,
            'V_spin_mean':     V_spin_mean,
            'V_JT_mean':       V_JT_mean,
            'V_rpa_mean':      V_rpa_mean,
            'V_neg_frac':      V_neg_frac,
            'K_pair_v_s':      v_s_raw,
            'K_pair_v_d':      v_d_raw,
            'frac':            frac,
            'gap_vector':      gap_vector,
            'K_pair':          K_pair,
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
        """Diagnostic post-processing of the linearised gap equation results.

        Returns a new dictionary that merges the vertex_cache with additional diagnostics (coherence lengths, channel eigenvalues, etc.).
        """
        # ---- Extract data from cache ----
        fs_pts = vertex_cache['fs_pts']
        vF_arr = vertex_cache['vF_arr']
        fs_idx = vertex_cache['fs_idx']
        ev     = vertex_cache['ev']
        ec     = vertex_cache['ec']
        K_pair = vertex_cache['K_pair']
        frac   = vertex_cache['frac']

        lambda_lin_max = float(vertex_cache['lambda_lin_max'])
        v_s_raw = float(vertex_cache['K_pair_v_s'])   # s‑channel weight
        v_d_raw = float(vertex_cache['K_pair_v_d'])   # d‑channel weight

        # ---- determine dominant symmetry ----
        if abs(v_d_raw) >= abs(v_s_raw):
            gap_symmetry = 'B1g (d-wave)'
            g_delta_dom = g_Delta_d
        else:
            gap_symmetry = 'A1g (s-wave)'
            g_delta_dom = g_Delta_s

        # ---- Derived quantities ----
        lambda_s = g_Delta_s * lambda_lin_max * frac[0]
        lambda_d = g_Delta_d * lambda_lin_max * frac[1]

        # Gap symmetry analysis (check for nonlinear quench)
        _Ds_mag, _Dd_mag = abs(Delta_s), abs(Delta_d)
        if _Ds_mag + _Dd_mag > _DELTA_ABS_FLOOR:
            _scf_dom = 'd' if _Dd_mag > _Ds_mag else 's'
            _lin_dom = 'd' if abs(v_d_raw) > abs(v_s_raw) else 's'
            if _scf_dom != _lin_dom:
                gap_symmetry = gap_symmetry.split(' [')[0]   # strip any old annotation
                gap_symmetry += f' [lin={_lin_dom}, SCF={_scf_dom}: nonlinear quench]'

        # Coherence length — nodal/antinodal decomposition for d-wave validity.
        # For d-wave Δ(k)=Δ_d·(cos kx−cos ky): ξ diverges at nodes, shortest at antinodes.
        # BdG validity (phase coherence) is governed by the NODAL sector (superfluid density ∝ v_F at nodes).
        # ξ_antinodal is diagnostic for BEC–BCS crossover;
        Delta_0 = max(_Ds_mag, 2.0 * _Dd_mag)
        if Delta_0 > 1e-8:
            vF_avg = float(np.mean(vF_arr))

            phi_d_fs  = np.cos(fs_pts[:, 0]) - np.cos(fs_pts[:, 1])
            phi_d_abs = np.abs(phi_d_fs)
            phi_d_max = phi_d_abs.max()

            if phi_d_max > _PHI_D_FLOOR:
                nodal_mask = phi_d_abs < np.percentile(phi_d_abs, _NODAL_REGION_PCTL)
                antinodal_mask = phi_d_abs > np.percentile(phi_d_abs, 100 - _NODAL_REGION_PCTL)

                # Nodal ξ: 25th percentile of |φ_d| as gap scale (conservative).
                if nodal_mask.sum() >= _VERTEX_DIAG_MIN_FS :
                    vF_nodal     = float(np.mean(vF_arr[nodal_mask]))
                    phi_nod_vals = phi_d_abs[nodal_mask]
                    Delta_nodal  = max(_Dd_mag * float(np.percentile(phi_nod_vals, _NODAL_REGION_PCTL)), _MATH_EPS)
                    xi_nodal     = vF_nodal / (np.pi * Delta_nodal)
                else:
                    vF_nodal = vF_avg
                    xi_nodal = vF_avg / (np.pi * max(_Dd_mag * 0.2, _MATH_EPS))

                # Antinodal ξ: gap ≈ Δ_0 at antinodes.
                if antinodal_mask.sum() >= _VERTEX_DIAG_MIN_FS:
                    vF_antinodal = float(np.mean(vF_arr[antinodal_mask]))
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
                vF_G6 = float(np.average(vF_arr, weights=np.array(w6_arr) + 1e-12))
                vF_G7 = float(np.average(vF_arr, weights=np.array(w7_arr) + 1e-12))
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
            'vF_avg': vF_avg,
            'valid_BdG': valid_BdG,
            'orbital_selective': orbital_selective,
            'gap_symmetry': gap_symmetry,
            'g_delta_dom': g_delta_dom,
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

    def build_local_hamiltonian_for_bdg(self, sign_M: float, M: float, J_A1g_diag: np.ndarray, mu: float, Z: float) -> np.ndarray:
        """
        Local 6×6 BdG Hamiltonian for one sublattice, basis [6↑, 6↓, 7ₐ↑, 7ₐ↓, 7ᵦ↑, 7ᵦ↓] — full 3
        Kramers-doublet manifold, no downfolding. sign_M = ±1 for sublattices A/B (staggered AFM).

        Terms:
          1. Chemical potential −μ (all six states — Γ7b is a genuine dynamical band, not a frozen core level,
             so its occupation responds to μ exactly like Γ6/Γ7a; this is what lets Γ6/Γ7a bands rise to, or
             past, Γ7b's level without any special-casing).
          2. Crystal field splitting: Δ_CF on Γ₇ₐ, Δ_CF+g7split on Γ₇ᵦ.
          3. Longitudinal (diagonal) AFM Weiss field from J_A1g: this is purely diagonal (spin-preserving). It
             shifts the orbital energies of all three doublets, weighted by each one's own <Sz> (sz_op) — so
             the large-|μz| Γ7b state always participates in the AFM physics, regardless of how weak its JT
             (B1g) matrix elements happen to be.
        """
        H = np.zeros((_N_ORB, _N_ORB), dtype=complex)

        # 1. Chemical potential
        np.fill_diagonal(H, -mu)

        # 2. Crystal field splitting: Δ_CF on Γ₇ₐ, Δ_CF+g7split on Γ₇ᵦ
        H[2, 2] += self.p.Delta_CF                    # 7a↑
        H[3, 3] += self.p.Delta_CF                    # 7a↓
        H[4, 4] += self.p.Delta_CF + self.p.g7split   # 7b↑
        H[5, 5] += self.p.Delta_CF + self.p.g7split   # 7b↓

        # 3. AFM Weiss field: longitudinal (J_A1g); g_J and spin-dilution factor are now carried inside J_A1g_diag
        O_exp_z = M * self.sz_op
        h_z = sign_M * (J_A1g_diag * O_exp_z) * Z

        H -= np.diag(h_z)   # h_z has _N_ORB=6 entries: 6↑,6↓,7a↑,7a↓,7b↑,7b↓
        return H
    
    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu_guess: float, t_eff: float, g_t: float, g_J: float, r_Q: float = 0.0, F67s_mf: float = 0.0) -> Tuple[float, float]:
        """
        Find the chemical potential μ such that the BdG electron density satisfies n(μ) = 1 − target_doping.
        The cheap estimator: ∂n/∂μ ≈ Σₖ,ₙ wₖ f(Eₙ)[1−f(Eₙ)]/kT
        is the standard BCS approximation. It effectively replaces the exact Hellmann–Feynman slope,
          ∂Eₙ/∂μ = ⟨n|∂H/∂μ|n⟩ = −(Pₙ−Hₙ),  Pₙ+Hₙ=1,
        by −1, assuming purely particle-like BdG states (Pₙ=1). This is exact only for Δ=0, where eigenstates are pure particle or hole branches.
        For finite Δₛ or Δ_d, Bogoliubov mixing gives Pₙ≠1, so ∂Eₙ/∂μ = 1−2Pₙ ≠ −1,
        and dn/dμ acquires additional eigenvector (Sternheimer-type) contributions omitted by the shortcut. Thus, once superconductivity is present, the analytic slope is only approximate: although μ enters H only through the diagonal −μτ_z term, the density's μ-dependence also reflects the evolving u/v coherence factors.
        A plain Newton step can overshoot when the DOS—and hence ∂n/∂μ—changes rapidly (e.g. near a Van Hove singularity). Instead, each step is backtracked (halved) until |n(μ)−target_n| decreases. If backtracking fails, the routine switches to a numerical derivative; if convergence is still not achieved within the iteration budget,
        it falls back to a guaranteed bracketed Brent solve rather than returning an unconverged μ.
        """
        target_n = 1.0 - target_doping
        vbdg = self._get_vbdg()
        _use_numeric_deriv = (abs(Delta_s) + abs(Delta_d)) > _MU_SC_DERIV_THRESH

        def _diag_and_density(mu_val: float) -> Tuple[np.ndarray, np.ndarray, float]:
            ev, ec = np.linalg.eigh(
                vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_n, mu_val, g_t, g_J, r_Q, F67s_mf, out=vbdg._H_stack)
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
                # Backtracking exhausted with no improvement: the local derivative estimate
                # (analytic or numeric) is unreliable here — escalate to the numeric derivative
                # and retry from the same μ rather than accepting a non-improving step.
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
                warnings.warn(
                    f"_find_mu_for_density: failed to bracket the density root after "
                    f"{_MU_NEWTON_MAXIT} Newton iterations (M={M:.4f}, Q={Q:.4f}, "
                    f"|Δ|={abs(Delta_s)+abs(Delta_d):.4f}); returning best estimate "
                    f"μ={mu:.6f} with |n−target|={abs(n_at_mu-target_n):.2e}.",
                    RuntimeWarning,
                )
        return mu, n_at_mu

    def _compute_bdg_free_energy(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float, F67s_mf: float, Q_Eg2: float, V_s: float = 0.0, V_d: float = 0.0, K_eff_for_free_energy: float = 0.0, K_eff_Eg2_for_free_energy: float = 0.0) -> float:
        """
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
        _tx_b, _ty_b = self.p.effective_hopping_anisotropic(Q)
        _J_A1g_diag_h, _ = self.p.exchange_channels(Q, n_kspace, _tx_b, _ty_b, g_J, r_Q)
        ev_all, _ = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, out=vbdg._H_stack, Q_Eg2=Q_Eg2)
            )

        f_n = _fermi_function(ev_all, self.kT)   # (N_k, 24)
        # quasiparticle energy term
        Ef = np.einsum('k,kn,kn->', self.k_weights, ev_all, f_n)

        # entropy contribution
        f_c = np.clip(f_n, _ENTROPY_CLIP, 1 - _ENTROPY_CLIP)
        S_kn = -(f_c*np.log(f_c) + (1-f_c)*np.log(1-f_c))
        S_term = self.kT * np.einsum('k,kn->', self.k_weights, S_kn)

        # Elastic energy uses the full effective spring constant (K_eff = K_lattice + ∂²F_ex/∂Q²).
        elastic_energy = 0.5 * K_eff_for_free_energy * Q**2
        if Q_Eg2 != 0.0 or K_eff_Eg2_for_free_energy != 0.0:
            elastic_energy += 0.5 * K_eff_Eg2_for_free_energy * Q_Eg2**2
        
        condensation = 0.0
        # Condensation correction: |Δ_ℓ|² / (g_ℓ · V_ℓ)
        if V_s > 0.0:
            condensation += abs(Delta_s)**2 / V_s
        if V_d > 0.0:
            condensation += abs(Delta_d)**2 / V_d
        
        # Mean-field AFM double-counting correction (per site); quasiparticle sum double-counts the Weiss field interaction.
        E_mf_correction = 0.5 * self.p.Z * _J_A1g_diag_h[0] * (M**2)
        Omega_cell = Ef - S_term
        # 0.25 corrects for 2 (sublattices) * 2 (Nambu particle-hole doubling) to yield per-site energy
        return 0.25 * Omega_cell + elastic_energy + condensation + E_mf_correction
    
    def compute_cluster_free_energy(self, M_ext: float, Q: float, n_kspace: float, tx_bare: float, ty_bare: float, mu: float, g_J: float, J_A1g_diag: np.ndarray, J_B1g_bare: float, F67s_mf: float, verbose: bool) -> Dict:
        """
        Cluster exact diagonalization for the SC state (Δ≠0) with two-channel (SC-induced JT + spin–JT cross) vertex renormalisation.

        2x2 PLAQUETTE (4 sites, 6^4=1296-dim). Captures both bond directions (x and y) at once, which
        matters because the B1g channel is intrinsically x/y-anisotropic.

        Geometry (checkerboard, open plaquette, no periodic wrap):

              0 --x-- 1
              |       |
              y       y
              |       |
              3 --x-- 2

        Sites 0,2 = sublattice A (sign_M=+1); sites 1,3 = sublattice B (sign_M=-1).
        Bonds: (0,1) x, (1,2) y, (2,3) x, (3,0) y — each carries η=+1 (x) or η=−1 (y) in the B1g
        channel, mirroring the cos(kx)−cos(ky) real-space bond weighting; the A1g (magnetic) channel
        is direction-independent (η≡+1 for all bonds).

        Every site has exactly 2 intra-cluster NN bonds (its two plaquette edges) and 2 remaining
        external neighbours out of the lattice's Z=4 total ⇒ Z_eff = Z−2 uniformly for all 4 sites.
        This avoids double-counting an intra-cluster bond both exactly, via H_exch, and again at
        mean-field level via the Weiss embedding.

            H = Σ_site H_site  +  Σ_bond [J_bond_M_bare·(O_i⊗O_j) + η_bond·J_B1g_bare·(B1g_i⊗B1g_j), Q≠0 only]

        H_site already contains: −μ, Δ_CF, AFM Weiss (J_A1g·sign_M·M_ext, Z_eff neighbours), JT
        (g_JT·Q·B1g), and the anomalous Weiss field (Z_eff·J_B1g·F67s_mf).

        On the A1g (magnetic) channel: J_bond_M_bare is not obtained by regression — corr_M vanishes
        identically by Wick factorisation of the Néel mean-field state — but taken from the analytic
        Kotliar–Ruckenstein/Gutzwiller result.

        2-channel regression model (connected correlators), restricted to the JT sector:
            evals_int ≈ const + J_Q·corr_Q + J_MQ·corr_MQ
        where corr_Q, corr_MQ are the B1g-symmetry-projected connected correlators, averaged (η-weighted)
        over the 4 plaquette bonds:
            corr_Q  = (1/4)·Σ_bond η·[⟨B_i B_j⟩ − ⟨B_i⟩⟨B_j⟩]
            corr_MQ = (1/4)·Σ_bond η·½[⟨O_i B_j+B_i O_j⟩ − ⟨O_i⟩⟨B_j⟩ − ⟨B_i⟩⟨O_j⟩]

        Slopes give: J_Q → SC-induced B1g exchange (r_Q=J_Q/J_B1g_bare); J_MQ → spin–JT cross-coupling
        (r_MQ=J_MQ/√(J_bond_M_bare·J_B1g_bare)).

        Expectation values are computed via reshaped tensor contraction: evecs viewed as a rank-5
        (6,6,6,6,N_states) tensor, contracting the small 6×6 operator onto only the relevant site leg,
        rather than forming full 1296×1296 embedded operators and matrix-multiplying.
        """
        # ── 0. Geometry and embedding helpers (4-site tensor product, 6^4 = 1296-dim) ──
        I6 = np.eye(_N_ORB, dtype=complex)
        _SIGN_M = (+1.0, -1.0, +1.0, -1.0)
        _BONDS  = ((0, 1, +1.0), (1, 2, -1.0), (2, 3, +1.0), (3, 0, -1.0))   # (i, j, eta_B1g)

        def _embed1(op: np.ndarray, site: int) -> np.ndarray:
            mats = [I6, I6, I6, I6]
            mats[site] = op.astype(complex)
            return np.kron(np.kron(mats[0], mats[1]), np.kron(mats[2], mats[3]))

        def _embed2(opA: np.ndarray, siteA: int, opB: np.ndarray, siteB: int) -> np.ndarray:
            mats = [I6, I6, I6, I6]
            mats[siteA] = opA.astype(complex)
            mats[siteB] = opB.astype(complex)
            return np.kron(np.kron(mats[0], mats[1]), np.kron(mats[2], mats[3]))

        def _apply_at_site(evecs_tensor: np.ndarray, op: np.ndarray, site: int) -> np.ndarray:
            """Apply a 6x6 operator to one site-leg of the rank-5 (6,6,6,6,N) eigenvector tensor."""
            t = np.tensordot(op, evecs_tensor, axes=([1], [site]))   # new leg lands at position 0
            return np.moveaxis(t, 0, site)

        def _expect_from_applied(evecs_tensor: np.ndarray, applied: np.ndarray) -> np.ndarray:
            return np.einsum('ijkln,ijkln->n', evecs_tensor.conj(), applied).real

        # ── 1. Single-particle + Weiss + JT + F67s Hamiltonians, per site ────────
        Z_eff = self.p.Z - 2
        H_JT  = (self.p.g_JT * Q) * self.B1g_op
        H_TRW_local = (Z_eff * J_B1g_bare * F67s_mf) * self.B1g_offdiag
        H_sites = []
        for s in range(4):
            Hs = self.build_local_hamiltonian_for_bdg(_SIGN_M[s], M_ext, J_A1g_diag, mu, Z_eff) + H_JT
            Hs = Hs - _SIGN_M[s] * H_TRW_local
            H_sites.append(Hs)

        # ── 2. Magnetic and orbital exchange, summed over the 4 plaquette bonds ──
        J_bond_M_bare = J_A1g_diag[0]   # single-bond A1g magnetic exchange (Gutzwiller-renormalised, no Z factor)

        # ── 3. Full cluster Hamiltonian (4 sites, 6 orbitals/site ⇒ 1296×1296) ──
        H_cluster = np.zeros((_N_ORB**4, _N_ORB**4), dtype=complex)
        for s in range(4):
            H_cluster += _embed1(H_sites[s], s)
        for (i, j, eta) in _BONDS:
            H_cluster += J_bond_M_bare * _embed2(self.multi_op, i, self.multi_op, j)
            H_cluster += eta * J_B1g_bare * _embed2(self.B1g_op, i, self.B1g_op, j)
        H_cluster = 0.5 * (H_cluster + H_cluster.conj().T)   # numerical Hermiticity
        evals, evecs = np.linalg.eigh(H_cluster)

        # ── 4. Free energy ────────────────────────────────────────────────────────
        _bweights = np.exp(-(evals - evals[0]) / self.kT)
        F_total   = evals[0] - self.kT * np.log(_bweights.sum())
        # Mean-field double-counting correction for the Z_eff external bonds each site's Weiss field represents: each site contributes +0.5*Z_eff*J*(order parameter)^2 per site
        # No analogous correction is needed for F67s_mf/H_TRW: F67s_mf is not an independent order parameter with its own gap equation — it is derived from the already self-consistent Delta_s/Delta_d condensate
        E_mf_correction     = 0.5 * Z_eff * J_bond_M_bare * (M_ext ** 2)
        F_total += E_mf_correction

        # ── 5. Per-site operators applied to the reshaped eigenvector tensor ─────
        evecs_tensor = evecs.reshape(_N_ORB, _N_ORB, _N_ORB, _N_ORB, evecs.shape[1])
        O_applied = [_apply_at_site(evecs_tensor, self.multi_op, s) for s in range(4)]
        B_applied = [_apply_at_site(evecs_tensor, self.B1g_op,  s) for s in range(4)]
        o_s = [_expect_from_applied(evecs_tensor, O_applied[s]) for s in range(4)]
        b_s = [_expect_from_applied(evecs_tensor, B_applied[s]) for s in range(4)]

        # ── 6. Connected correlators, B1g-symmetry-projected & averaged over the 4 bonds ──
        corr_Q  = np.zeros_like(evals)
        corr_MQ = np.zeros_like(evals)
        for (i, j, eta) in _BONDS:
            BiBj_applied = _apply_at_site(B_applied[j], self.B1g_op,  i)
            OiBj_applied = _apply_at_site(B_applied[j], self.multi_op, i)   # O_i B_j
            BiOj_applied = _apply_at_site(O_applied[j], self.B1g_op,  i)    # B_i O_j
            Bi_Bj     = _expect_from_applied(evecs_tensor, BiBj_applied)
            OiBj_BiOj = _expect_from_applied(evecs_tensor, 0.5 * (OiBj_applied + BiOj_applied))
            corr_Q  += eta * (Bi_Bj     - b_s[i] * b_s[j])
            corr_MQ += eta * (OiBj_BiOj - 0.5 * (o_s[i]*b_s[j] + b_s[i]*o_s[j]))
        corr_Q  /= 4.0
        corr_MQ /= 4.0

        # ── 7. Weiss-field decontamination (sum over the 4 actual H_exch bonds) ───
        # Subtract the MF (mean-field) energy so the residual evals_int carries only the quantum fluctuation content
        evals_int = evals.copy()
        for (i, j, eta) in _BONDS:
            evals_int -= J_bond_M_bare * o_s[i] * o_s[j]
            evals_int -= eta * J_B1g_bare * b_s[i] * b_s[j]

        # ── 8. Fit temperature and Boltzmann weights ──────────────────────────────
        # T_floor: at least 2×kT and 15% of kinetic scale to prevent a single near-degenerate state from dominating the regression (see _T_floor analysis).
        _kin_scale = np.sqrt(0.5 * (tx_bare**2 + ty_bare**2))
        _T_floor   = max(2.0*self.kT, 0.15*_kin_scale, 1e-3)
        # Use evals_int (not raw evals) so T_fit reflects the fluctuation spectrum, consistent with the regression target — avoids inflating T_fit with large Weiss shifts.
        T_fit      = max(float(np.std(evals_int)), _T_floor)
        raw_w      = np.exp(-(evals_int - evals_int.min()) / T_fit)

        # Drop states with negligible Boltzmann weight; 2-channel regression needs at least 3 points; fall back to full spectrum otherwise.
        valid_mask = raw_w > 1e-6
        if np.count_nonzero(valid_mask) < 3:
            valid_mask = slice(None)

        fit_w     = raw_w[valid_mask];  fit_w /= fit_w.sum()
        corr_Q_f  = corr_Q[valid_mask]
        corr_MQ_f = corr_MQ[valid_mask]
        E_f       = evals_int[valid_mask]

        # ── 9. Two-channel weighted multivariate linear regression (WMLR) ─────────
        cQ  = np.sum(fit_w * corr_Q_f)
        cMQ = np.sum(fit_w * corr_MQ_f)
        cE  = np.sum(fit_w * E_f)

        dx_Q  = corr_Q_f  - cQ
        dx_MQ = corr_MQ_f - cMQ
        dy    = E_f - cE

        S_QQ   = np.sum(fit_w * dx_Q  * dx_Q)
        S_MQMQ = np.sum(fit_w * dx_MQ * dx_MQ)
        S_QMQ  = np.sum(fit_w * dx_Q  * dx_MQ)
        S_Qy   = np.sum(fit_w * dx_Q  * dy)
        S_MQy  = np.sum(fit_w * dx_MQ * dy)

        n_eff   = 1.0 / max(float(np.sum(fit_w**2)), _REGR_EPS)
        min_var = _REGR_VAR_MIN * max(1.0, 2.0 / max(n_eff - 1.0, 0.1))

        J_Q  = 0.0
        J_MQ = 0.0
        _regression_solved = False

        A_mat = np.array([
            [S_QQ,  S_QMQ ],
            [S_QMQ, S_MQMQ],
        ], dtype=float)
        b_vec = np.array([S_Qy, S_MQy], dtype=float)

        if (abs(Q) > _CLUSTER_Q_REGR_THRESH and n_eff > 5.0
                and S_QQ > min_var and S_MQMQ > min_var
                and np.linalg.cond(A_mat) < 1e10):
            try:
                J_Q, J_MQ = np.linalg.solve(A_mat, b_vec)
                _regression_solved = True
            except np.linalg.LinAlgError:
                try:
                    sol, _, _, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-6)
                    if sv[-1] > 1e-10 * sv[0]:   # reject if effective rank < 2
                        J_Q, J_MQ = sol
                        _regression_solved = True
                except Exception:
                    pass

        # ── 10. Renormalisation factors ────────────────────────────────────────────
        # Normalise by J_B1g_bare (the single-bond B1g bare coupling) so that r_Q is dimensionless and directly maps to the (1+r_Q) factor;
        # This is internally consistent: evals_int has J_B1g_bare·B1g⊗B1g subtracted, the regression slope J_Q carries units of energy/[B1g²]
        Q_bare_scale = max(abs(J_B1g_bare), _REGR_EPS)
        M_bare       = max(abs(J_bond_M_bare), _REGR_EPS)

        r_Q  = J_Q  / Q_bare_scale
        # r_MQ couples the spin (O) and orbital (B1g) channels; normalised by the geometric mean √(M_bare · Q_bare_scale) so both channels contribute equally to the scale.
        r_MQ = J_MQ / max(np.sqrt(M_bare * Q_bare_scale), _REGR_EPS)

        r_Q  = float(np.clip(r_Q,  -2.0, 2.0))
        r_MQ = float(np.clip(r_MQ, -2.0, 2.0))

        if not _regression_solved or abs(J_Q) < 1e-6:
            # no JT-sector signal can be reliably extracted
            r_Q = 0.0;  r_MQ = 0.0
        else:
            # ── 11. Independent t-statistic significance tests for r_Q and r_MQ ────
            _n_channels = 2   # J_Q, J_MQ
            df = max(n_eff - _n_channels, 0.5)   # floor at 0.5 to keep t_crit finite
            t_crit = float(tdist.ppf(_REGR_T_ALPHA, df))
            
            res_var = max(np.sum(fit_w * dy * dy) - (J_Q * S_Qy + J_MQ * S_MQy), 0.0)
            # Standard error for each slope uses its own diagonal of the inverse design matrix, which accounts for the Q/MQ collinearity
            try:
                A_inv = np.linalg.inv(A_mat)
                var_J_Q  = max(A_inv[0, 0], 0.0) * res_var / max(n_eff - 1.0, _REGR_EPS)
                var_J_MQ = max(A_inv[1, 1], 0.0) * res_var / max(n_eff - 1.0, _REGR_EPS)
                SE_J_Q  = float(np.sqrt(var_J_Q))
                SE_J_MQ = float(np.sqrt(var_J_MQ))
            except np.linalg.LinAlgError:
                SE_J_Q  = float('inf')
                SE_J_MQ = float('inf')

            def _shrink(value: float, J_val: float, SE_J: float, label: str) -> float:
                if not np.isfinite(SE_J):
                    conf = 0.0
                    t_stat = 0.0
                else:
                    t_stat = abs(J_val) / max(SE_J, np.finfo(float).eps)
                    xp = (max(t_stat, 0.0) / t_crit) ** _REGR_SHRINK_POWER
                    conf = xp / (1.0 + xp)
                shrunk = value * conf

                if verbose and conf < 0.9:
                    _scf_log(
                        "CLUSTER-ED",
                        f"⚠ t={t_stat:.2f} < t_crit={t_crit:.2f} (df={df:.1f}) — "
                        f"{label} significance shrinkage "
                        f"(conf={conf:.4f}): {value:.4f}→{shrunk:.4f}"
                    )
                return shrunk

            r_Q  = _shrink(r_Q,  J_Q,  SE_J_Q,  "r_Q")
            r_MQ = _shrink(r_MQ, J_MQ, SE_J_MQ, "r_MQ")

        # ── 12. Observables (averaged over the 4 sites) ────────────────────────────
        b_mean  = [float(np.sum(fit_w * b_s[s][valid_mask])) for s in range(4)]
        b2_diag = [_expect_from_applied(evecs_tensor, _apply_at_site(B_applied[s], self.B1g_op, s)) for s in range(4)]
        b2_mean = [float(np.sum(fit_w * b2_diag[s][valid_mask])) for s in range(4)]
        Q_fluct = float(np.sqrt(max(0.0, np.mean([b2_mean[s] - b_mean[s]**2 for s in range(4)]))))

        return {
            'F_per_site': float(F_total / _CLUSTER_SIZE),
            'r_Q':        r_Q,
            'r_MQ':       r_MQ,
            'Q_fluct':    Q_fluct,
        }

    def refine_M_normal_state(self, target_doping: float, initial_M: float, max_iter: int = 12, verbose: bool = False) -> Tuple[float, float, float]:
        # ---- preliminary quantities that do not depend on M ----
        g_t, g_J, _, _ = self.p.get_gutzwiller_factors(target_doping)
        t_eff = g_t * self.p.t0
        n_kspace = 1.0 - target_doping   # nominal electron density
        mu = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        M = self.p.estimate_M0(target_doping, M_seed=initial_M)

        vbdg = self._get_vbdg()
        kpts = vbdg._kpts

        for _ in range(max_iter):
            mu, n_kspace = self._find_mu_for_density(M, 0.0, 0.0j, 0.0j, target_doping, mu, t_eff, g_t, g_J)
            ev_n, ec_n = np.linalg.eigh(vbdg._build_H_stack(kpts, M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, 0.0, 0.0, out=vbdg._H_stack))
            J_eff = self.p.Z * self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, g_J)[0][0]
            Gamma_M, _, _ = self._make_vertex_params(target_doping, t_eff, t_eff, g_t, J_eff)
            chi_SS_afm, _, _ = self.get_susceptibilities_sc(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, 0.0, 0.0, (ev_n, ec_n), apply_diamagnetic_QQ=True)
            M = float(np.clip(M + 0.35 * (float(np.tanh(J_eff * chi_SS_afm * M)) - M), 0.0, _KICK_M_CLIP_HI))
            if verbose:
                print(f"stoner-1 = {J_eff * chi_SS_afm - 1:.6f}, M = {M:.6f}")
        return M, n_kspace, mu

    def _scf_jacobi_kick(self, target_doping: float, initial_M: float, initial_Delta: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, force_d_wave: bool = False, verbose: bool = False) -> Dict:
        """
        Estimate the dominant Jacobi eigenvalue λ₊ of the two-channel (Δ, Q) SCF map and generate physics-informed seed values for (M, Q, Δ_s, Δ_d).

        Linearised Jacobian of the (Δ, Q) fixed-point map:
            J = [ A   B ]
                [ C   0 ]
        λ₊ = ½[A + √(A²+4BC)]; complex → spectral radius = ½√(A²+|disc|)
        Regimes: λ₊<0.7 subcritical, λ₊∈[0.7,1.4] critical, λ₊>1.4 supercritical.
        """
        # The seed for the fixed-point solve is the CALLER's own initial_M when it looks like real information
        initial_M, n_kspace, mu = self.refine_M_normal_state(target_doping, initial_M, 12, verbose)

        if force_d_wave:
            Delta_s = 0.0j
            Delta_d = complex(initial_Delta)
        else:
            Delta_s = complex(initial_Delta * 0.5)
            Delta_d = complex(initial_Delta * 0.5)

        # --- Anisotropic hopping ---
        _t_eff = g_t * self.p.t0
        _J_eff = self.p.Z * self.p.exchange_channels(0.0, n_kspace, self.p.t0, self.p.t0, g_J)[0][0]
        _Gamma_M, V_JT, _V_cap = self._make_vertex_params(target_doping, _t_eff, _t_eff, g_t, _J_eff)
        bdg_ev_n, bdg_ec_n = np.linalg.eigh(self._get_vbdg()._build_H_stack(self._get_vbdg()._kpts, initial_M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, 0.0, 0.0, out=self._get_vbdg()._H_stack))
        chi_SS_q0, _, chi_QQ_q0 = self.get_susceptibilities_sc(initial_M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, 0.0, 0.0, (bdg_ev_n, bdg_ec_n), apply_diamagnetic_QQ=True)
        chi_SS_afm, _, chi_QQ_afm = self.get_susceptibilities_sc(initial_M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, 0.0, 0.0, (bdg_ev_n, bdg_ec_n), apply_diamagnetic_QQ=True)
        stoner = _J_eff * chi_SS_afm

        # --- chi_tau and Linearised BdG+RPA eigenproblem ---
        Q_probe = _Q_SEED_THR
        chi_tau_val = self._compute_chi_tau(initial_M, Q_probe, Delta_s, Delta_d, n_kspace, mu, g_t, g_J)['chi_tau_net']

        # --- JT stability ---
        K_eff_ex_n = self.compute_JT_rigidity_from_exchange(initial_M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J)

        # --- Linearised BdG+RPA eigenproblem ---
        _lin_seed = self.compute_pairing_kernel_and_build_cache(initial_M, 0.0, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _J_eff, _Gamma_M, V_JT, _V_cap)

        lambda_lin_max = float(_lin_seed['lambda_lin_max'])
        # ---- Jacobi elements ----
        # The actual pairing is driven by the q≈(π,π) backscattering peak, which is exactly what largest eigenvalue of the full FS kernel captures.
        A = lambda_lin_max
        # Q-mode stiffness based on G₃[2,2] (normal state value, but sufficient in kick)
        D = max(1.0 - V_JT * chi_QQ_q0, _MATH_EPS)  # G22_norm: positive in stable case
        # Coupling estimation from gap-induced B₁g response
        B = math.sqrt(V_JT * abs(chi_tau_val)) * (_t_eff / max(self.p.Delta_CF, 1e-9))
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

        # --- Regime classification ---
        if lambda_plus < 0.7:
            regime = 'subcritical'
        elif lambda_plus <= 1.4:
            regime = 'critical'
        else:
            regime = 'supercritical'
        
        lambda_excess = max(0.0, lambda_plus - 1.0) / lambda_plus

        # --- update Q probe ---
        if lambda_lin_max > 1.0:
            # SC-triggered JT: equilibrium condition K·Q = g_JT·⟨B1g⟩ ≈ g_JT²·χ_τ·Q; gives Q* ≈ g_JT²·χ_τ / K_bare as the natural distortion scale.
            _sign = np.sign(self.p.Delta_B1g_static) if abs(self.p.Delta_B1g_static) > _MATH_EPS else 1.0
            Q_probe = float(np.clip(_sign * _KICK_BOOST_Q * self.p.g_JT * self.p.lambda_hop * np.sqrt(abs(chi_tau_val / K_eff_ex_n)), -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))
        
        # ---  Early Hessian in the seed neighborhood --- 
        _EARLY_KICK_SCALE = 0.05
        _hk_early = self.compute_hessian(initial_M, Q_probe, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, r_Q=0.0, F67s_mf=0.0, Q_Eg2=0.0, V_JT=V_JT, vertex_cache=None)
        evals, evecs = _hk_early['eigenvalues'], _hk_early['eigenvectors']
        idx_min = int(np.argmin(evals))
        lambda_min = float(evals[idx_min])
        evec_min = evecs[:, idx_min].real

        if lambda_min < 0.0:
            edir = evec_min * np.array([1.0, self.p.lambda_hop, 1.0])
            edir /= max(np.linalg.norm(edir), _MATH_EPS)
            if Q_probe * edir[1] < 0:
                edir = -edir
            step = _EARLY_KICK_SCALE * edir

            # --- M kick ---
            M_kick = float(np.clip(initial_M + step[0], _KICK_M_CLIP_LO, _KICK_M_CLIP_HI))
            Q_kick = float(np.clip(Q_probe + step[1], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # --- Δ kick ---
            # Use lambda_plus (analytic Jacobi eigenvalue of the (Δ,Q) map) as the pairing-strength indicator for the seed scale
            frac = _lin_seed['frac']
            Delta_total = abs(Delta_s) + abs(Delta_d)
            new_Delta_total = max(Delta_total + step[2], 0.0)

            Delta_s_kick = new_Delta_total * frac[0] * (Delta_s / abs(Delta_s) if abs(Delta_s) > _MATH_EPS else 1.0)
            Delta_d_kick = new_Delta_total * frac[1] * (Delta_d / abs(Delta_d) if abs(Delta_d) > _MATH_EPS else 1.0)
            if verbose:
                print("Q_probe, Q_kick: ", Q_probe, Q_kick)
                print("initial_M, M_kick: ", initial_M, M_kick)
                print("old_Delta_total, new_Delta_total: ", complex(np.clip(lambda_excess * _t_eff * np.exp(-1.0 / max(lambda_plus, 0.1)), _DELTA_ABS_FLOOR, _KICK_DELTA_MAX_FRAC * _t_eff)), new_Delta_total)
        else:
            reduction = _KICK_REDUCTION_AMP * max(0.0, (initial_M - _KICK_M_EXCESS_CTR)) * max(0.0, (stoner - _KICK_JCHI_EXCESS_CTR)) * lambda_excess
            M_kick = initial_M * (1.0 - reduction)
            Q_kick = Q_probe
            Delta_kick = complex(np.clip(lambda_excess * _t_eff * np.exp(-1.0 / max(lambda_plus, 0.1)), _DELTA_ABS_FLOOR, _KICK_DELTA_MAX_FRAC * _t_eff))
            Delta_s_kick = Delta_kick * frac[0]
            Delta_d_kick = Delta_kick * frac[1]

        if force_d_wave:
            Delta_s_kick = 0.0j
        
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q_kick)
        tx, ty = g_t * tx_bare, g_t * ty_bare
        _t_eff = np.sqrt(0.5 * (tx**2 + ty**2))
        mu, n_kspace = self._find_mu_for_density(M_kick, Q_kick, Delta_s_kick, Delta_d_kick, target_doping, mu, _t_eff, g_t, g_J, 0.0, 0.0)

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
            'regime':           regime,
            'J_eff':            _J_eff,
            't_eff':            _t_eff,
            'jchi_proxy':       stoner,
            'lambda_JT_kernel': lambda_JT_kernel,
            'V_JT':             V_JT,
        }
    
    def _vertex_matrix_at_Q(self, M: float, Qv: float, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, Gamma_M: float, V_JT: float, V_cap: float, det_afm_sc: float, solve_state: '_SolveState') -> Tuple[float, np.array]:
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

        lin = self.compute_pairing_kernel_and_build_cache(M, Qv, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff_v, Gamma_M, V_JT, V_cap, det_afm_sc, solve_state)
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

    def solve_self_consistent(self, target_doping: float, initial_Delta: float, initial_M: float = None, verbose: bool = False, _ic_retry: bool = False, force_d_wave: bool = False, Q_Eg2: float = 0.0, force_delta_zero=False, force_Q_zero=False) -> Dict:
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
          5. Cluster ED → r_Q, r_MQ (EMA-smoothed) for JT-sector vertex renormalisation;
             J_eff itself comes directly from the analytic Gutzwiller factor each iteration.
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
        kick = self._scf_jacobi_kick(target_doping, initial_M, initial_Delta, g_t, g_J, g_Delta_s, g_Delta_d, force_d_wave, verbose)

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
        _Gamma_M = 0.0
        _V_cap = 0.0
        
        tx_bare, ty_bare = self.p.effective_hopping_anisotropic(Q)

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [], 'F_cluster': [], 'mu': [], 'mixing': [],
        }

        if verbose:
            h_afm_M0 = _J_eff * M
            _retry_flags = (
                f"{'  [force_d_wave]' if force_d_wave else ''}"
                f"{'  [ic_retry]' if _ic_retry else ''}"
            )
            _scf_log("SCF-INIT", f"δ={target_doping:.4f}  M_kick={M:.4f}  Q₀={Q:.5f}  |Δ|₀={abs(Delta_s)+abs(Delta_d):.5f}  g_t={g_t:.4f}  g_J={g_J:.4f}  g_Delta_s={g_Delta_s:.4f}  g_Delta_d={g_Delta_d:.4f}{_retry_flags}")
            _scf_log("SCF-INIT", f"h_afm(M₀)={h_afm_M0:.4f} eV  t_eff={_t_eff_now:.4f} eV  {'✓ metallic AFM' if h_afm_M0 < 4.0 * _t_eff_now else '⚠ marginal/insulating'}")
            _scf_log("SCF-INIT", f"λ_JT_kernel={_lambda_JT_kernel:.3f}  [{kick['regime']}]  J_eff/Δ_CF={_J_eff / self.p.Delta_CF:.2f}  λ_lin_max={_lambda_lin_max:.3f}  α={_alpha:.4f}")   # prerequisite of the Schrieffer–Wolff transformation:  J_eff/Δ_CF < 0.5

        scf_x_hist: list = []
        scf_f_hist: list = []

        _vertex_cache: Optional[dict] = None
        _max_diff_prev = float('inf')   # previous iteration's max_diff
        _stagnation_count = 0           # consecutive near-stagnation iterations
        _pairing_strength_proxy = 0.0   # must be initialised before loop; only updated when _vertex_cache is not None

        # Initialise Λ_inst from kick proxies; take the most pessimistic channel.
        _Lambda_inst  = float(np.clip(max(_lambda_plus, _lambda_lin_max, _lambda_JT_kernel, _jchi_proxy), 0.0, 10.0))
        
        max_diff = float('inf')
        _chi_SS_sc_pipi = 0.0
        _chi_SQ_sc_pipi = 0.0
        _chi_QQ_sc_pipi = 0.0
        _chi_SS_sc_q0 = 0.0
        _chi_SQ_sc_q0 = 0.0
        _chi_QQ_sc_q0 = 0.0
        _det_q0_sc  = _DET_AFM_FLOOR
        _F67s_mf    = 0.0
        _r_Q_cur    = 0.0    # orbital vertex renormalization
        _r_MQ_cur   = 0.0    # cross vertex renormalization
        F_cluster   = {'F_per_site': 0.0, 'r_Q': 0.0, 'r_MQ': 0.0, 'Q_fluct': 0.0}
        _det_q0     = _DET_AFM_FLOOR
        _det_pomer  = _DET_AFM_FLOOR
        _det_afm    = _DET_AFM_FLOOR
        _det_afm_sc = _DET_AFM_FLOOR
        _alpha_freeze_count = 0
        selection_ratio = 0.0
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
            J_A1g_diag, J_B1g_bare = self.p.exchange_channels(Q, n_kspace, tx_bare, ty_bare, g_J, _r_Q_cur)

            F_cluster = self.compute_cluster_free_energy(M, Q, n_kspace, tx_bare, ty_bare, mu, g_J, J_A1g_diag, J_B1g_bare, _F67s_mf, verbose)
            _r_Q_cur  = _EMA_NEW_QRW    * F_cluster['r_Q']  + (1.0 - _EMA_NEW_QRW)    * _r_Q_cur
            _r_MQ_cur = _EMA_NEW_WEIGHT * F_cluster['r_MQ'] + (1.0 - _EMA_NEW_WEIGHT) * _r_MQ_cur

            # J_eff comes exclusively from the analytic Gutzwiller/Kotliar–Ruckenstein exchange renormalisation
            _J_eff = self.p.Z * J_A1g_diag[0]
            tx, ty = g_t * tx_bare, g_t * ty_bare
            _t_eff_now = np.sqrt(0.5 * (tx**2 + ty**2))
            _Gamma_M, _V_JT, _V_cap = self._make_vertex_params(target_doping, tx, ty, g_t, _J_eff)

            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                _vbdg._build_H_stack(_vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, out=_vbdg._H_stack)
                )
            M_bdg = _vbdg.compute_observables_vectorized(M, Q, Delta_s, Delta_d, mu, _bdg_ev_sc, _bdg_ec_sc)
            dF_dM_0, d2F_dM2 = self.compute_dF_dM_and_d2F(M, Q, Delta_s, Delta_d, n_kspace, mu, J_A1g_diag, g_J, _F67s_mf, _r_Q_cur, _bdg_ev_sc, _bdg_ec_sc)

            # SC+JT active: Gorkov singlet amplitude (u·v), Gutzwiller-renormalised, fed back into J_B1g off-diagonal Weiss field. Zero by symmetry when Δ=0 or Q=0.
            Delta_eff_now = abs(Delta_s) + abs(Delta_d)
            if Delta_eff_now > _QQ_DELTA_THRESH:
                _g_eff = (g_Delta_s * abs(Delta_s) + g_Delta_d * abs(Delta_d)) / Delta_eff_now  
                _F67s_mf = _g_eff * self._compute_F67_singlet(_bdg_ev_sc, _bdg_ec_sc)   # F67s receives contributions from BOTH pairing channels
                bdg_ev_sc, bdg_ec_sc = np.linalg.eigh(
                    _vbdg._build_H_stack(_vbdg._kpts, M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, out=_vbdg._H_stack)
                    )
                # ── SC-state (Δ≠0) RPA determinant ──────────────
                _U_SQ_sc = _r_MQ_cur * math.sqrt(max(_J_eff * _V_JT, 0.0))
                _chi_SS_sc_pipi, _chi_SQ_sc_pipi, _chi_QQ_sc_pipi = self.get_susceptibilities_sc(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, _r_Q_cur, _F67s_mf, (bdg_ev_sc, bdg_ec_sc), apply_diamagnetic_QQ=True)
                _det_afm_sc, *_ = self._rpa_det(_J_eff, _V_JT, _chi_SS_sc_pipi, _chi_SQ_sc_pipi, _chi_SQ_sc_pipi, _chi_QQ_sc_pipi, _U_SQ_sc)

                _chi_SS_sc_q0, _chi_SQ_sc_q0, _chi_QQ_sc_q0 = self.get_susceptibilities_sc(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, _r_Q_cur, _F67s_mf, (bdg_ev_sc, bdg_ec_sc), apply_diamagnetic_QQ=True)
                _det_q0_sc, *_ = self._rpa_det(_J_eff, _V_JT, _chi_SS_sc_q0, _chi_SQ_sc_q0, _chi_SQ_sc_q0, _chi_QQ_sc_q0, _U_SQ_sc)
            
            # Gap equation: V(q) always from Δ=0 χ₀; BdG amplitudes (u,v) from SC state.
            Delta_s_out, Delta_d_out, Delta_s7b_diag, Delta_d7b_diag, _vertex_cache = _vbdg.compute_gap_eq_vectorized(M, Q, Delta_s, Delta_d, n_kspace, mu, _t_eff_now, g_t, g_J, g_Delta_s, g_Delta_d, _J_eff, _Gamma_M, _V_JT, _V_cap, _det_afm_sc, _r_MQ_cur, _solve_state, _bdg_ev_sc, _bdg_ec_sc, _vertex_cache, False)
            if force_delta_zero:
                Delta_s_out = 0.0j
                Delta_d_out = 0.0j
            
            # Update Q
            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(
                _vbdg._build_H_stack(_vbdg._kpts, M, Q, Delta_s_out, Delta_d_out, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, out=_vbdg._H_stack)
                )
            
            dHdQ = self._calc_dHdQ(M, Q, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf)
            dHdQ_diag = np.einsum('kin,kij,kjn->kn', _bdg_ec_sc.conj(), dHdQ, _bdg_ec_sc).real
            f_k = _fermi_function(_bdg_ev_sc, self.kT)
            dHdQ_exp = np.sum(self.k_weights[:, None] * f_k * dHdQ_diag) / 4.0

            K_eff_Q, _F_bdg_electronic = self.compute_K_eff_full(M, Q, Delta_s_out, Delta_d_out, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, Q_Eg2)

            # compute_K_eff_full's F0 is electronic-only
            V_s = (_vertex_cache['V_s_scalar'] if _vertex_cache else _V_JT) * g_Delta_s
            V_d = (_vertex_cache['V_d_scalar'] if _vertex_cache else _V_JT) * g_Delta_d
            F_bdg = (_F_bdg_electronic + 0.5 * self.p.K_lattice * Q**2 + 0.5 * self.p.K_lattice_Eg2 * Q_Eg2**2 + (abs(Delta_s_out)**2 / V_s if V_s > 0 else 0) + (abs(Delta_d_out)**2 / V_d if V_d > 0 else 0))


            # Adaptive LM floor for the Q Hellmann-Feynman step
            #   K_eff_Q >> 0  (deep JT-stable well)       -> mu_LM_Q small  -> near-bare HF step
            #   K_eff_Q ~  0  (JT QCP, chi_QQ softening)  -> mu_LM_Q = _Q_LM_FRAC * _K_bare -> cautious step
            #   K_eff_Q <  0  (past the QCP, SC-induced)  -> mu_LM_Q = |K_eff_Q| + _Q_LM_FRAC*_K_bare -> guarantees (K_eff_Q + mu_LM_Q) > 0
            _K_ref = max(self._K_bare, _MATH_EPS)
            _mu_LM_Q_base = _Q_LM_FRAC * _K_ref
            if K_eff_Q > _MATH_EPS:
                _mu_LM_Q = max(_mu_LM_Q_base / (1.0 + K_eff_Q / _K_ref), _mu_LM_Q_base * 0.1)
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
                _lam_JT_vc = float(np.clip(_V_JT * max(_chi_QQ_vc, 0.0), 0.0, 10.0))
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
                _grad_s, _grad_d, _H_delta = self.compute_dF_dDelta_and_d2F(
                    M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d,
                    _r_Q_cur, _F67s_mf, Q_Eg2, _V_JT, _vertex_cache)

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

            # Adaptive LM floor
            #   d2F_dM2 >> 0  (deep stable minimum) → μ_LM small  → fast, lightly-damped Newton step
            #   d2F_dM2 ≈ 0   (flat / near AFM QCP)  → μ_LM = _MU_LM → cautious step, dominated by 1/μ_LM
            #   d2F_dM2 < 0   (saddle / unstable)    → μ_LM = |d2F_dM2| + _MU_LM → guarantees (H+μ) > 0
            if d2F_dM2 > _MATH_EPS:
                _mu_LM_eff = max(_MU_LM / (1.0 + d2F_dM2 / _t_eff_now), _MU_LM * 0.1)
            elif d2F_dM2 < -_MATH_EPS:
                _mu_LM_eff = abs(d2F_dM2) + _MU_LM
            else:
                _mu_LM_eff = _MU_LM / (1.0 + (Delta_s_abs + Delta_d_abs) / (2*self.p.t0))  # reduce overdamping when Δ grows (SC–AFM coupling unfreezes).
            
            # 1. Regularization of LM-Newton denominator and J_eff thresholds (against Anderson overshoot)
            _lm_denom = max(d2F_dM2 + _mu_LM_eff, 1e-6)
            _j_eff_floor = max(abs(_J_eff), _M_J_EFF_FLOOR_FRAC * _t_eff_now, 1e-4)

            # 2. Trust-region upper bound: J/t stiffness cut + curvature-based penalty near QCP
            _cap_stiff = _TR_M_STEP_MAX / max(1.0, abs(_J_eff) / (2.0 * max(_t_eff_now, 1e-6)))
            _cap_curv = 0.5 + 0.5 * (max(d2F_dM2, 0.0) / _lm_denom)
            _step_upper = float(np.clip(_cap_stiff * _cap_curv, _TR_M_STEP_MIN_FLOOR, _TR_M_STEP_MAX))

            # 3. Enforcing a dynamic step limit between the lower and upper trust-region boundaries
            _step_floor = max(_M_STEP_FLOOR_REL * abs(M), _M_STEP_FLOOR_ABS)
            _step_limit = float(np.clip(max(self.kT, 0.05 * _t_eff_now) / _j_eff_floor, _step_floor, _step_upper))

            # 4. M update and hybrid mixing (linear BdG fixed point + Newton trajectory)
            M_newton = float(np.clip(M + np.clip(-dF_dM_0 / _lm_denom, -_step_limit, _step_limit), 0.0, 1.0))
            M_fixpoint = self._mix(M, M_bdg, alpha=_alpha)
            M_mixed = float(np.clip((1.0 - _ALPHA_HF) * M_fixpoint + _ALPHA_HF * M_newton, 0.0, 1.0))

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
            
            mu_new, n_kspace = self._find_mu_for_density(M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, mu, _t_eff_now, g_t, g_J, _r_Q_cur, _F67s_mf)
            
            # max_diff tracks order-parameter convergence (M, Q, Δ_s, Δ_d).
            max_diff = max(
                abs(M_mixed - M),
                abs(Q_mixed - Q),
                abs(Delta_s_abs - abs(Delta_s)),
                abs(Delta_d_abs - abs(Delta_d)),
            )
            
            # SC-triggered JT selection rule proxy:
            #   |Δ|/Δ_CF   — condensate mixing of Γ₆↔Γ₇ (0=normal state, 1=full mixing)
            #   |F67s_mf|  — ANOMALOUS Gorkov singlet amplitude (u·v* cross term)
            selection_ratio = float(np.clip(
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
                    if selection_ratio > 0.05 and abs(Q) > 1e-4:
                        # SC+JT active: mild boost, Λ-capped
                        _alpha = min(_alpha_base * _SCF_ALPHA_CONVG_BOOST, _MIXING * _SCF_ALPHA_CONVG_CAP)
                    elif selection_ratio > 0.05:
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
                _alpha = float(np.clip(_alpha * math.exp(-_RPA_QCP_PENALTY * _det_penalty),
                                       _MIXING / 16.0, _alpha))
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

            history['M'].append(abs(M_mixed))
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
                    f"  conv={max_diff:.1e}  M={M:.3f}  Q={Q:+.4f}"
                    f"  |Δ|={(abs(Delta_s)+abs(Delta_d)):.4f}"
                    f"  J_eff={_J_eff:.4f} eV  mu={mu_new:.5f}  g_t={g_t:.4f}  g_J={g_J:.4f}"
                    f"  J*χSS(q=0)={_J_eff * _vertex_cache['chi_SS_q0']:.4f}  J*χSS_sc(q=0)={_J_eff * _chi_SS_sc_q0:.4f}  V_JT*χQQ(q=0)={_V_JT * _vertex_cache['chi_QQ_q0']:.4f}  V_JT*χQQ_sc(q=0)={_V_JT * _chi_QQ_sc_q0:.4f}"
                    f"  F_bdg={F_bdg:.4f} eV  F_cluster={F_cluster['F_per_site']:.4f} eV")
                _scf_log("SCF-II",
                    f"  dFM_sc={_det_q0_sc:.4f}  dAFM={_vertex_cache['det_afm']:.4f}  dAFM_sc={_det_afm_sc:.4f}  χ_SQ_sc(q=π,π)={_chi_SQ_sc_pipi:.4f}  χ_SQ_sc(q=0)={_chi_SQ_sc_q0:.4f} "
                    f"  Γ_M={_Gamma_M*1000:.2f}meV  α={_alpha:.4f}"
                    f"  δB1g={self.B1g_expectation(tx_bare, ty_bare, (_bdg_ev_sc, _bdg_ec_sc)) - self.B1g_expectation(tx_bare, ty_bare, self._get_chi0_norm_cache(M, 0.0, n_kspace, mu, g_t, g_J, _vbdg)):+.4f}  F67s={_F67s_mf:+.4f}"
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
                _hk = self.compute_hessian(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _r_Q_cur, _F67s_mf, Q_Eg2, _V_JT, _vertex_cache)
                
                _lmin_k = float(_hk['eigenvalues'][0])
                if np.isfinite(_lmin_k) and _lmin_k < 0:
                    _edir = _hk['eigenvectors'][:, 0]   # eigenvector of λ_min in Hessian units (M, Q, Δ)

                    # Scale to physical units (Q in Å), normalise to unit vector.
                    _edir_raw  = _edir * np.array([1.0, self.p.lambda_hop, 1.0])
                    _edir_raw /= max(np.linalg.norm(_edir_raw), 1e-12)

                    _wM, _wQ, _wD = abs(_edir_raw[0]), abs(_edir_raw[1]), abs(_edir_raw[2])
                    _wsum = max(_wM + _wQ + _wD, 1e-12)
                    _fM, _fQ, _fD = _wM / _wsum, _wQ / _wsum, _wD / _wsum    # modes from component fractions

                    if _fD > _MODE_FRAC_DOMINANT:                           _mode = 'pure-SC'
                    elif _fQ > _MODE_FRAC_DOMINANT:                         _mode = 'pure-JT'
                    elif _fD > _MODE_FRAC_MIXED and _fQ > _MODE_FRAC_MIXED: _mode = 'SC-triggered-JT'
                    elif _fM > _MODE_FRAC_DOMINANT:                         _mode = 'AFM-fluctuation'
                    else:                                                   _mode = 'mixed'

                    _kick_damp = 1.0 / (1.0 + _Lambda_inst)   # Kick magnitude Λ-damped: 1/(1+Λ)
                    _curvature = min(abs(_lmin_k), 1.0)
                    _kick_mag  = min(2.0 * self.kT, 0.1 * Delta_total) * _kick_damp * _curvature

                    # M: gentle pull toward Stoner estimate when SC mode and M overshoots.
                    _stoner_est = float(_J_eff * _vertex_cache['chi_SS_afm']) if _vertex_cache is not None else None
                    _M_phys_est = self.p.estimate_M0(target_doping, _stoner_est, M)
                    _m_was_pulled = False
                    if _mode in ('pure-SC', 'SC-triggered-JT') and M > 3.0 * _M_phys_est:
                        _pull_frac = _MODE_PULL_FRAC * _kick_damp
                        _M_kick_component = float(np.clip(M - _pull_frac * (M - _M_phys_est), 0.02, M))
                        _m_was_pulled = True
                    else:
                        _M_kick_component = float(np.clip(M + _kick_mag * _edir_raw[0], 0.0, 1.0))
                    Q_kick = float(np.clip(Q + _kick_mag * _edir_raw[1], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

                    # Δ: signed kick along eigenvector component, preserving s/d ratio.
                    _delta_sign = np.sign(_edir_raw[2]) if abs(_edir_raw[2]) > 1e-6 else 1.0
                    _D_kick_signed = max(0.0, Delta_total + _kick_mag * _delta_sign * abs(_edir_raw[2]))

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
                                f"  → M={M:.3f} Q={Q:+.4f} |Δ|={_D_kick_signed:.4f}  {' [M-pulled]' if _m_was_pulled else ''}")

            # Re-evaluate convergence against the current (possibly kicked) state.
            _M_post   = abs(M)          - abs(_M_pre_kick)
            _Q_post   = abs(Q)          - abs(_Q_pre_kick)
            _Ds_post  = abs(Delta_s)    - abs(_Ds_pre_kick)
            _Dd_post  = abs(Delta_d)    - abs(_Dd_pre_kick)
            _kick_fired = (abs(_M_post) + abs(_Q_post) + abs(_Ds_post) + abs(_Dd_post)) > 1e-10
            
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
                f"  M={M:.4f}  Q={Q:+.4f}  |Δ|={abs(Delta_s)+abs(Delta_d):.4f}"
                f"  dyn={_scf_dynamics_regime}"
                f"{'  [ansatz unstable]' if _ansatz_unstable_ever else ''}")

        # Post-loop diagnostic: λ_max and Rayleigh JT projection and store converged gap and distortion
        J_A1g_diag, _ = self.p.exchange_channels(Q, n_kspace, tx_bare, ty_bare, g_J, _r_Q_cur)
        _J_eff = self.p.Z * J_A1g_diag[0]
        if _vertex_cache is None:
            _vertex_cache = self.compute_pairing_kernel_and_build_cache(M, Q, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _J_eff, _Gamma_M, _V_JT, _V_cap, _det_afm_sc, _solve_state)
        _vertex_cache = self.scf_gap_diagnostics(Delta_s, Delta_d, g_Delta_s, g_Delta_d, _vertex_cache)
        
        if converged:
            hessian_result = self.compute_hessian(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _r_Q_cur, _F67s_mf, Q_Eg2, _V_JT, _vertex_cache)
        else:
            hessian_result = {'Delta_s_frac': 0.0, 'F_bdg': 0.0, 'eigenvectors': None, 'eigenvalues': None}

        _chi_tau_result = self._compute_chi_tau(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf)

        _Delta_s_mag = abs(Delta_s)
        _Delta_d_mag = abs(Delta_d)

        # ── FS-resolved ∂λ/∂Q and gap-channel decomposition (SC state only) ──────
        # Per-FS-point ∂V(k,k)/∂Q locates the hot spots where the effect is concentrated; a spatially flat but globally positive signal may hide hot-spot / cold-spot cancellation.
        _dlam_dQ_fs  = 0.0
        _hot_spot_frac = 0.0
        _sym_mismatch  = False
        _sym_lin       = 's'

        if converged and (_Delta_s_mag + _Delta_d_mag) > _QQ_DELTA_THRESH:
            # Adaptive finite-difference step: 5 % of |Q| protects against noise at larger distortions.
            _dQ_step = max(_DQ_FS_VERTEX, _DQ_FS_VERTEX_FRAC * abs(Q))
            lmax_p, V_diag_p = self._vertex_matrix_at_Q(M, Q + _dQ_step, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _Gamma_M, _V_JT, _V_cap, _det_afm_sc, _solve_state)
            lmax_m, V_diag_m = self._vertex_matrix_at_Q(M, Q - _dQ_step, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, _Gamma_M, _V_JT, _V_cap, _det_afm_sc, _solve_state)
            _dlam_dQ_fs = (lmax_p - lmax_m) / (2.0 * _dQ_step)
            _dV_dQ_diag = (V_diag_p - V_diag_m) / (2.0 * _dQ_step)
            _hot_spot_frac = float(np.mean(_dV_dQ_diag > 0)) if len(_dV_dQ_diag) > 0 else 0.0

            # Channel-resolved pairing strength from the 2×2 (s,d) kernel projection — handles s-d mixing exactly via K12, unlike a single-eigenvector scalar split.
            _lam_max_s = _vertex_cache['lambda_s']
            _lam_max_d = _vertex_cache['lambda_d']
            # Eigenvector vs SCF amplitude mismatch: linearised gap eq says d-wave dominates but SCF converged to s-wave (Δ_s >> Δ_d);
            # this indicates that the non-linear fixpoint (large Δ) has quenched the d-wave channel via spectrum depletion.
            _sym_scf      = 'd' if _Delta_d_mag / (_Delta_s_mag + _Delta_d_mag) > 0.5 else 's'
            _sym_lin      = 'd' if abs(_vertex_cache['K_pair_v_d']) > abs(_vertex_cache['K_pair_v_s']) else 's'
            _sym_mismatch = _sym_scf != _sym_lin

            if verbose:
                _scf_log("SCF-RES",
                    f"δ={target_doping:.4f}"
                    f"  ∂λ/∂Q(FS)={_dlam_dQ_fs:+.4f} eV⁻¹"
                    f"  {'✓ SC-JT consistent' if _dlam_dQ_fs > 0 else '⚠ ∂λ/∂Q ≤ 0'}"
                    f"  hot_spot_frac={_hot_spot_frac:.2f}"
                    f"  {'[concentrated]' if _hot_spot_frac > 0.6 else '[diffuse]' if _hot_spot_frac > 0.3 else '[anti-nodal suppressed]'}")
                _scf_log("SCF-RES",
                    f"δ={target_doping:.4f}"
                    f"  λ_lin_max={_vertex_cache['lambda_lin_max']:.3f} (K12={_vertex_cache['K_pair'][0,1]:+.4f})"
                    f"  λ_s={_lam_max_s:.4f}  λ_d={_lam_max_d:.4f}"
                    f"  |Δs|={_Delta_s_mag*1000:.3f}meV  |Δd|={_Delta_d_mag*1000:.3f}meV"
                    f"  SCF={_sym_scf}  lin={_sym_lin}"
                    f"  {'[⚠ SYMMETRY MISMATCH]' if _sym_mismatch else '[consistent]'}"
                    f"  {'[NL s-d MIXING]' if (_lam_max_d > _lam_max_s) and (_Delta_s_mag > 1e-5) and (_Delta_d_mag > 1e-5) else ''}")

            # d-wave free-energy retry when linear kernel and SCF symmetry disagree.
            if _sym_mismatch and _sym_lin == 'd' and not _ic_retry:
                _scf_log("SCF-RES", f"δ={target_doping:.4f} → d-wave enforcing retry (symmetry mismatch)")
                try:
                    _d_result = self.solve_self_consistent(target_doping, _Delta_s_mag + _Delta_d_mag, M, verbose, True, True)
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
                    _M_ic = float(np.clip(M / _ic_chi_ratio_clamped, _KICK_M_CLIP_LO, _KICK_M_CLIP_HI))
                    _scf_log("SCF-RES",
                        f"δ={target_doping:.4f} IC retry: M {M:.4f} → {_M_ic:.4f}"
                        f"  (χ_ratio={_ic_chi_ratio_clamped:.2f})")
                    _ic_result = self.solve_self_consistent(target_doping, _Delta_s_mag + _Delta_d_mag, _M_ic, verbose, True)
                    if _ic_result.get('converged', False):
                        _scf_log("SCF-RES",
                            f"δ={target_doping:.4f} IC retry converged:"
                            f"  M={_ic_result['M']:.4f}  Q={_ic_result['Q']:+.4f}"
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
                _hstr = (f"H=[{_eigs[0]:.3f},{_eigs[1]:.3f},{_eigs[2]:.3f}]"
                         f"{'✓MIN' if bool(np.all(_eigs > -1e-6)) else '⚠SADDLE'}")
            _scf_log("SCF-RES",
                f"  F_bdg={hessian_result['F_bdg']:.4f}  F_cluster={F_cluster['F_per_site']:.4f}"
                f"  JT={'✓' if selection_ratio > _JT_ACT_THR else '✗'}  {_hstr}"
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
        _chi_QQ_mat_final = self._chi_QQ_matrix_elements(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, Q_Eg2, return_matrix=True)
        _chi_Eg2Eg2_final = float(_chi_QQ_mat_final[1, 1])
        _G44_final        = 1.0 - _chi_Eg2Eg2_final * self.p.g_Eg2**2 / self.p.K_lattice_Eg2
        _eg2_exp_final    = self.Eg2_expectation((_bdg_ev_sc, _bdg_ec_sc))
        if verbose:
            _scf_log("SCF-EG2",
                f"δ={target_doping:.4f}  Q_Eg2(fixed)={Q_Eg2:+.4f}  <Eg2>={_eg2_exp_final:+.3e}  "
                f"χ_Eg2,Eg2={_chi_Eg2Eg2_final:+.4f}  G44={_G44_final:+.4f} [{'stable' if _G44_final > 0 else 'SOFT Eg2 PHONON'}]")

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
            'r_Q': _r_Q_cur,
            'r_MQ': _r_MQ_cur,
            'F67s_mf': _F67s_mf,
            'target_doping': target_doping,
            'afm_unstable': _det_afm_sc <= 0.0,
            'selection_ratio': selection_ratio,
            'history': history,
            'hessian_result': hessian_result,
            'lambda_plus': _lambda_plus,
            'regime': kick['regime'],
            'lambda_JT_sc': self.p.g_JT**2 * _chi_tau_result['chi_tau_net'] / self.compute_JT_rigidity_from_exchange(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, Q_Eg2),
            'K_eff_net': self.compute_K_eff_full(M, Q, Delta_s, Delta_d, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, Q_Eg2) - self.compute_K_eff_full(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, _r_Q_cur, _F67s_mf, Q_Eg2),
            'converged': converged,
            'mott_suspect': _mott_suspect,
            'scf_dynamics_regime': _scf_dynamics_regime,   # 'converging'|'limit_cycle'|'first_order_jump'|'hysteretic'
            'ansatz_unstable': _ansatz_unstable_ever,      # True if det(RPA)<0 at any SCF iteration
            'dlam_dQ_fs': _dlam_dQ_fs,
            'hot_spot_frac': _hot_spot_frac,
            'incommensurate_dq': _ic_dq_max,
            'incommensurate_chi_ratio': _ic_chi_max / max(_ic_chi_0, 1e-12) if _ic_chi_0 else float('nan'),
        })
        return result

    def _scan_incommensurate_nesting(self, M: float, Q: float, mu: float, g_t: float, g_J: float, n_kspace: float) -> Tuple[float, float, float]:
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

    def compute_dF_dM_and_d2F(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, J_A1g_diag: np.ndarray, g_J: float, F67s_mf: float, r_Q: float, ev: np.ndarray, ec: np.ndarray) -> Tuple[float, float]:
        
        # ∂H/∂M: only J_A1g (diagonal) contributes; J_B1g (τ_x, off-diagonal) has no diagonal entry, but its Weiss field (F67s_mf) must be included in the full H so eigenvectors ec carry the correct inter-band matrix elements for the Kubo term2.
        h_J_unit = self.p.Z * J_A1g_diag * self.sz_op
        
        dH_diag = np.concatenate([
            -h_J_unit,   # particle A  (sign_M=+1)
            +h_J_unit,   # particle B  (sign_M=−1)
            +h_J_unit,   # hole A      (PH of A)
            -h_J_unit,   # hole B      (PH of B)
        ])

        f_all = _fermi_function(ev, self.kT)
        exp_nn = np.einsum('i,kin->kn', dH_diag, np.abs(ec)**2)

        J_eff = self.p.Z * J_A1g_diag[0]
        grad = float(np.einsum('k,kn,kn->', self.k_weights, f_all, exp_nn)) / 4.0 + J_eff * M

        df_dE = -f_all * (1.0 - f_all) / self.kT   # (N,24)  ≤ 0
        term1 = float(np.einsum('k,kn,kn->', self.k_weights, df_dE, exp_nn**2))

        off = np.einsum('i,kin,kim->knm', dH_diag, ec.conj(), ec)
        off2 = np.abs(off)**2   # |matrix element|²

        dE_nm = ev[:, None, :] - ev[:, :, None]   # E_m − E_n,  (N,24,24)
        df_nm = f_all[:, :, None] - f_all[:, None, :]
        safe  = np.abs(dE_nm) > _FD_MASK_DE8
        ratio = np.where(safe, df_nm / np.where(safe, dE_nm, 1.0), -df_dE[:, :, None])
        np.einsum('knn->kn', ratio)[:] = 0.0
        term2 = float(np.einsum('k,knm,knm->', self.k_weights, ratio, off2))
        d2F = (term1 - term2) / 4.0 + J_eff
        return grad, d2F

    def compute_hessian(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, r_Q: float, F67s_mf: float, Q_Eg2: float, V_JT: float, vertex_cache: dict = None) -> Dict:
        """ 3×3 finite-difference Hessian of F(M,Q,Δ), evaluated with Q_Eg2 HELD FIXED at its current (input) value
        (Q_Eg2 is not one of the three probed directions here -- it enters only as a fixed background parameter, so
        F() is evaluated at the correct total state rather than silently at Q_Eg2=0).
        Must match the vertex used in the SCF gap equation so that ∂F/∂Δ = 0 at the converged point.

        Phase handling: if this Hessian collapsed both channels onto abs(Delta)·Delta_{s,d}_frac
        (a real, phase-stripped split) before probing M/Q/Δ, F would be evaluated off the true
        stationary point whenever the converged state carries a nontrivial relative phase.
        """
        vbdg = self._get_vbdg()
        if vertex_cache is not None:
            # V_s / V_d: full RPA pairing vertex for the condensation correction
            V_s = vertex_cache['V_s_scalar'] * g_Delta_s
            V_d = vertex_cache['V_d_scalar'] * g_Delta_d
        else:
            V_s = V_JT * g_Delta_s
            V_d = V_JT * g_Delta_d

        Delta = abs(Delta_s) + abs(Delta_d)
        Delta_s_frac = (abs(Delta_s) / Delta) if Delta > _QQ_DELTA_THRESH else 0.5
        Delta_d_frac = 1.0 - Delta_s_frac
        # Converged relative phase, held fixed for every probe point below. Falls back to 1.0+0j (real, positive) when a channel's amplitude is too small to define a phase —
        phase_s = (Delta_s / abs(Delta_s)) if abs(Delta_s) > 1e-12 else (1.0 + 0j)
        phase_d = (Delta_d / abs(Delta_d)) if abs(Delta_d) > 1e-12 else (1.0 + 0j)

        eps_M = max(1e-4, abs(M) * 1e-3)
        eps_Q = max(1.5e-4, abs(Q) * 1e-3 * self.p.lambda_hop)
        # eps_D: must keep Δ±eps_D ≥ 0 (negative amplitude biases off-diagonals); clip to Δ/2.
        eps_D = min(max(1e-5, abs(Delta) * 1e-3), max(abs(Delta) / 2.0, 1e-10))

        def F(m_val, q_val, delta_val):
            # abs(delta_val): F is even in the overall amplitude sign; avoids asymmetry at the +/- finite-difference points
            ds = phase_s * abs(delta_val) * Delta_s_frac
            dd = phase_d * abs(delta_val) * Delta_d_frac
            return self._compute_bdg_free_energy(m_val, q_val, ds, dd, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2, V_s, V_d, self.p.K_lattice, K_eff_Eg2_for_free_energy=self.p.K_lattice_Eg2)

        F0 = F(M, Q, Delta)
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

        _evals, _evecs = np.linalg.eigh(H)
        return {
            'Delta_s_frac': Delta_s_frac,
            'F_bdg': F0,
            'eigenvectors': _evecs,
            'eigenvalues': _evals,
        }

    def compute_dF_dDelta_and_d2F(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, r_Q: float, F67s_mf: float, Q_Eg2: float, V_JT: float, vertex_cache: dict = None) -> Tuple[float, float, np.ndarray]:
        """ Gradient (∂F/∂|Δ_s|, ∂F/∂|Δ_d|) and 2×2 Hessian of F(M,Q,Δ_s,Δ_d) with respect to the two
        channel AMPLITUDES, relative phases held fixed at their current values. This is the Δ-sibling of
        compute_dF_dM_and_d2F built the same way compute_hessian's Δ-block already is -- but split into
        two independent radial directions (|Δ_s|, |Δ_d|) instead of one combined direction at a fixed Delta_s_frac ratio
        """
        if vertex_cache is not None:
            V_s = vertex_cache['V_s_scalar'] * g_Delta_s
            V_d = vertex_cache['V_d_scalar'] * g_Delta_d
        else:
            V_s = V_JT * g_Delta_s
            V_d = V_JT * g_Delta_d

        phase_s = (Delta_s / abs(Delta_s)) if abs(Delta_s) > 1e-12 else (1.0 + 0j)
        phase_d = (Delta_d / abs(Delta_d)) if abs(Delta_d) > 1e-12 else (1.0 + 0j)
        ds0, dd0 = abs(Delta_s), abs(Delta_d)

        # eps sizing mirrors compute_hessian's eps_D: keep ds0±eps_s (resp. dd0±eps_d) >= 0, clipped to half the amplitude.
        eps_s = min(max(1e-5, ds0 * 1e-3), max(ds0 / 2.0, 1e-10)) if ds0 > _MATH_EPS else 1e-5
        eps_d = min(max(1e-5, dd0 * 1e-3), max(dd0 / 2.0, 1e-10)) if dd0 > _MATH_EPS else 1e-5

        def F(ds_val, dd_val):
            ds = phase_s * abs(ds_val)
            dd = phase_d * abs(dd_val)
            return self._compute_bdg_free_energy(M, Q, ds, dd, n_kspace, mu, g_t, g_J, r_Q, F67s_mf, Q_Eg2, V_s, V_d, self.p.K_lattice, K_eff_Eg2_for_free_energy=self.p.K_lattice_Eg2)

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
        Works element-wise for scalars, arrays, and complex numbers.
        """
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

        def gap_at_T(T: float) -> float:
            s = self._clone_solver_at_T(T)
            try:
                if sc_result['converged']:
                    _initial_M = float(np.clip(sc_result['M'], _KICK_M_CLIP_LO, _KICK_M_CLIP_HI))
                else:
                    _stoner_ref = float(sc_result.get('J_eff', 0.0) * sc_result.get('chi_SS_afm', 0.0))
                    _initial_M  = self.p.estimate_M0(doping, _stoner_ref, sc_result.get('M', 0.0))
                res = s.solve_self_consistent(
                    target_doping = doping,
                    initial_Delta = 1e-8,   # normal-state seed (below nucleation floor)
                    initial_M     = _initial_M,
                )
                Ds = res['Delta_s']
                Dd = res['Delta_d']
                D  = (Ds**2 + Dd**2) ** 0.5

                if use_free_energy and D > Delta_tol:
                    s_n = s._clone_solver_at_T(T)
                    res_normal = s_n.solve_self_consistent(
                        target_doping = doping,
                        initial_Delta = 0.0,
                        initial_M     = _initial_M,
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

        compute_Tc_by_gap_suppression starts every temperature from Δ≈0 (cooling
        from normal state), finding only the spinodal (2nd-order instability
        boundary).  For the SC-triggered JT mechanism the effective Landau
        potential F_eff(Δ) = a(T)Δ² + [b−γ²/(2K_eff)]Δ⁴+… can have a negative
        quartic term; the transition is then first-order and Tc* occurs where
        a(T*)>0.  Cooling from Δ≈0 misses this and returns Tc ≪ Tc*.

        Algorithm: (1) warm-start heating from the T≈0 SC+JT basin; (2) at each T
        compare F_SC vs F_NM; (3) Tc = max(spinodal collapse, thermodynamic
        crossing); (4) bisection between last SC-wins and first NM-wins T.
        """
        Delta_s0 = sc_result['Delta_s']
        Delta_d0 = sc_result['Delta_d']
        Delta0   = float(np.sqrt(Delta_s0**2 + Delta_d0**2))
        M0       = float(sc_result['M'])

        if (not sc_result.get('converged', False) or Delta0 < Delta_tol):
            return {
                'Tc': 0.0, 'Tc_spinodal': 0.0, 'T_cross': 0.0, 'T_spinodal_cool': float('nan'),
                'transition_order': 'unknown', 'Delta_at_Tc': 0.0, 'Q_at_Tc': 0.0,
                'ratio_2D': 0.0, 'Delta_jump': 0.0, 'hysteresis': 0.0, 'history': []
            }

        def _eval_sc_basin(solver: 'RMFT_Solver', seed_M: float, seed_Q: float, seed_D: float) -> tuple:
            """Returns (Δ_eff, Q_eff, M_eff, F_sc, converged, collapsed); collapsed if Δ<Delta_tol."""
            try:
                res = solver.solve_self_consistent(
                    target_doping = doping,
                    initial_Delta = seed_D,
                    initial_M     = seed_M,
                )
                D_eff = float(np.sqrt(res['Delta_s']**2 + res['Delta_d']**2))
                Q_eff = float(res['Q'])
                M_eff = float(res['M'])
                F_sc = float(res['F_bdg'])
                converged = res['converged']
                collapsed = D_eff < Delta_tol
                return D_eff, Q_eff, M_eff, F_sc, converged, collapsed
            except Exception:
                return 0.0, 0.0, seed_M, 1e30, False, True

        def _eval_normal_basin(solver: 'RMFT_Solver') -> tuple:
            """Returns (F_nm, converged) for the normal-state basin."""
            try:
                res = solver.solve_self_consistent(
                    target_doping = doping,
                    initial_Delta = 0.0,
                )
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
                D_eff, Q_eff, M_eff, F_sc, _, collapsed = _eval_sc_basin(s, _seed['M'], _seed['Delta'])
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
                    res = s.solve_self_consistent(
                        target_doping = doping,
                        initial_Delta = 0.0,
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
        This avoids the T=0 AFM Weiss field artefact (artificially split bands → λ never reaches 1) from the
        previous implementation that passed the converged SC gaps into the linearised kernel.

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
                res = s_T.solve_self_consistent(
                    target_doping = doping,
                    initial_Delta = 0.0,
                )
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
        h_afm = J_eff * M / 2.0
        
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
        _, chi_SQ_q0, chi_QQ_q0 = self.get_susceptibilities_sc(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), _Gamma_M, 0.0, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
        chi_SS_afm, *_ = self.get_susceptibilities_sc(M, 0.0, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), _Gamma_M, 0.0, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
        
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

        G3[0, 2] = G3[2, 0] = -self.p.g_JT * math.sqrt(max(gVs / self.p.K_lattice, 0.0)) * chi_SQ_s
        G3[1, 2] = G3[2, 1] = -self.p.g_JT * math.sqrt(max(gVd / self.p.K_lattice, 0.0)) * chi_SQ_d

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

        lambda_lin_max_Q0 = float(self.compute_pairing_kernel_and_build_cache(M, 0.0, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff, _Gamma_M, _V_JT, _V_cap)['lambda_lin_max'])

        H_afm_mat = self.build_local_hamiltonian_for_bdg(1.0, M, J_A1g_diag, mu, self.p.Z)
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

    def _get_fs_points(self, M: float, Q: float, n_kspace: float, mu: float, g_t: float, g_J: float, store_cache: bool = True, compute_vF: bool = False, prefer_dwave: bool = True):
        """
        Return Fermi-surface k-points with adaptive angular sampling.
        
        The FS geometry (k-locus where ξ_k = 0) and the Fermi velocity |∇_k E_min| are NORMAL-STATE quantities:
        Δ only shifts quasiparticle energies. Both are computed at Δ = 0 so that the cache is Δ-independent
        and consistent with the BCS/BdG convention that the pairing vertex is built from the normal-state Fermi surface.

        Sampling strategy:
        - Angular bins ensure coverage of all directions in [0, π)
        - Within each bin, select the point maximizing the importance weight
        - Weight = (|df/dE| / |v_F|) × w_form(k)

        store_cache: write (fs_pts, vF, fs_idx) to self._fs_cache_dict.
        compute_vF : compute |∇_k E_min| via central FD (4 extra eigh calls, 3×_N_FS points).
                    If False, vF = ones (uniform FS weighting).
        """        
        _cache_key_vals = (
            float(M), float(Q),
            float(n_kspace),
            float(mu), float(g_t), float(g_J),
            int(_N_FS), bool(compute_vF), bool(prefer_dwave),
        )

        vbdg = self._get_vbdg()
        if self._fs_cache_dict is not None:
            for _stored_key, _stored_val in self._fs_cache_dict.items():
                if (int(_stored_key[7]) == int(_N_FS) 
                        and bool(_stored_key[8]) == bool(compute_vF)
                        and bool(_stored_key[9]) == bool(prefer_dwave)
                        and all(abs(float(_stored_key[i]) - float(_cache_key_vals[i])) < _FS_CACHE_TOL
                                for i in range(7))):
                    fs_pts, vF, fs_idx = _stored_val
                    ev_all, ec_all = np.linalg.eigh(
                        vbdg._build_H_stack(vbdg._kpts, M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=vbdg._H_stack)
                    )
                    return fs_pts, vF, fs_idx, ev_all, ec_all

        # ── Step 1: full-BZ BdG at Δ=0 ──────────────────────────────────────────
        ev_all, ec_all = np.linalg.eigh(
            vbdg._build_H_stack(vbdg._kpts, M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=vbdg._H_stack)
        )
        ev_pos = np.where(ev_all > 0, ev_all, np.inf)
        Emin   = ev_pos.min(axis=1)

        # ── Step 2: MBZ deduplication ────────────────────────────────────────────
        # For collinear AFM with Q_AFM=(π,π), k and k+Q_AFM are the same physical state.
        # Keeping both corrupts K_dd because φ_d(k+Q) = -φ_d(k) → negative cross-term.
        _half_NK = _NK // 2
        _kx_all = self.k_points[:, 0]
        _ky_all = self.k_points[:, 1]
        _ix_all = (np.round((_kx_all + np.pi) * (_NK / (2.0 * np.pi)))).astype(int) % _NK
        _iy_all = (np.round((_ky_all + np.pi) * (_NK / (2.0 * np.pi)))).astype(int) % _NK
        _code_all    = (_ix_all * _NK + _iy_all).astype(np.int32)
        _partner_all = ((_ix_all + _half_NK) % _NK * _NK + (_iy_all + _half_NK) % _NK).astype(np.int32)

        sort_idx = np.argsort(_code_all)
        partner_idx = np.empty(len(_code_all), dtype=np.int32)
        partner_idx[sort_idx] = sort_idx[np.searchsorted(_code_all[sort_idx], _partner_all[sort_idx])]
        # At Q=0 (and near it) Emin[k] and Emin[partner_idx[k]] are degenerate to float precision
        _dE_partner = Emin - Emin[partner_idx]
        _combinatorial_mask = _code_all < _partner_all
        mbz_mask_k = np.where(
            np.abs(_dE_partner) < max(_MBZ_DEGEN_TOL, 0.01 * self.kT),
            _combinatorial_mask,
            _dE_partner < 0.0
        )

        # Thermal weight for FS proximity
        _f_all = _fermi_function(Emin, self.kT)
        _therm = _f_all * (1.0 - _f_all) / self.kT
        near_fs = _therm > _FS_THERMAL_THRESHOLD / self.kT
        near_fs_mbz = near_fs & mbz_mask_k
        fs_idx_all = np.where(near_fs_mbz)[0]

        if len(fs_idx_all) == 0:
            fs_idx_all = np.where(near_fs)[0]
            if len(fs_idx_all) == 0:
                fs_idx_all = np.arange(min(3 * _N_FS, self.N_k))

        kxy_all   = self.k_points[fs_idx_all]
        Emin_over = Emin[fs_idx_all]
        therm_over = _therm[fs_idx_all]
        
        # ── Form factor for importance sampling ───────────────────────────────────
        # The d-wave pairing kernel weights each FS point by phi_d(k)^2:
        #   K_dd ~ Σ_{k,k'} phi_d(k) · V(k-k') · phi_d(k')
        # phi_d^2 is large at antinodes (kx≈π, ky≈0, phi_d≈2 → phi_d^2≈4) and zero at nodes (kx≈ky≈π/2, phi_d=0).
        if prefer_dwave:
            phi_d_all = (np.cos(kxy_all[:, 0]) - np.cos(kxy_all[:, 1])) ** 2
            form_weight = np.maximum(phi_d_all, _PHI_D_FLOOR ** 2)
        else:
            form_weight = np.ones(len(kxy_all))

        # ── Step 4: |v_F| on the filtered subset ──────────────────────────────────
        if compute_vF:
            dk = min(1e-2, max(1e-4, 2.0 * np.pi / _NK / 6.0))
            kx, ky = kxy_all[:, 0], kxy_all[:, 1]
            n_near = len(kxy_all)
            _vF_buf = np.zeros((n_near, _N_BDG, _N_BDG), dtype=complex)

            def _Emin_batch(kpts):
                ev_b, _ = np.linalg.eigh(
                    vbdg._build_H_stack(kpts, M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, out=_vF_buf[:len(kpts)])
                )
                return np.where(ev_b > 0, ev_b, np.inf).min(axis=1)

            dE_dx = (_Emin_batch(np.c_[kx + dk, ky]) - _Emin_batch(np.c_[kx - dk, ky])) / (2.0 * dk)
            dE_dy = (_Emin_batch(np.c_[kx, ky + dk]) - _Emin_batch(np.c_[kx, ky - dk])) / (2.0 * dk)
            vF_over = np.maximum(np.hypot(dE_dx, dE_dy), _VF_FLOOR)
        else:
            vF_over = np.ones(len(fs_idx_all), dtype=float)

        # ── Step 4: angular subsampling with importance weighting ────────────────
        # Fold angles to [0, π) to leverage inversion symmetry: E(k) = E(-k)
        angles_folded = np.arctan2(kxy_all[:, 1], kxy_all[:, 0]) % np.pi
        bins = np.linspace(0, np.pi, _N_FS + 1, endpoint=True)
        bin_ids = np.clip(np.digitize(angles_folded, bins) - 1, 0, _N_FS - 1)

        selected = []
        for b in range(_N_FS):
            mask = (bin_ids == b)
            if mask.any():
                score = therm_over[mask] / vF_over[mask] * form_weight[mask]
                selected.append(int(np.flatnonzero(mask)[np.argmax(score)]))

        # Fill empty bins using remaining points sorted by score
        if len(selected) < _N_FS:
            already = set(selected)
            # Compute score for all remaining points
            remaining_scores = therm_over / vF_over * form_weight
            remaining_idx = [i for i in range(len(kxy_all)) if i not in already]
            remaining_idx.sort(key=lambda i: -remaining_scores[i])  # descending score
            for i in remaining_idx:
                if len(selected) >= _N_FS:
                    break
                selected.append(i)

        sel = np.array(selected[:_N_FS], dtype=int)
        fs_pts = kxy_all[sel]
        vF     = vF_over[sel]
        fs_idx = fs_idx_all[sel]

        if store_cache:
            if self._fs_cache_dict is None:
                self._fs_cache_dict = {}
            if len(self._fs_cache_dict) >= 32:
                self._fs_cache_dict.pop(next(iter(self._fs_cache_dict)))
            self._fs_cache_dict[_cache_key_vals] = (fs_pts, vF, fs_idx)
        return fs_pts, vF, fs_idx, ev_all, ec_all


def _build_H_AB_block(kx: np.ndarray, ky: np.ndarray, Tx_op: np.ndarray, Ty_op: np.ndarray, g_t: float) -> np.ndarray:
    """
    Vectorized, orbital-selective inter-sublattice hopping block:
        H_AB(k) = -2·g_t·[cos(kx)·Tx_op + cos(ky)·Ty_op]      shape (N_k, 6, 6)
    """
    cos_kx = np.cos(kx)[:, None, None]
    cos_ky = np.cos(ky)[:, None, None]
    return (-2.0 * g_t * (cos_kx * Tx_op[None, :, :] + cos_ky * Ty_op[None, :, :])).astype(complex)

class VectorizedBdG:
    def __init__(self, solver: 'RMFT_Solver'):
        self.solver   = solver
        self._kpts    = solver.k_points        # (N_k, 2)    — SCF / gap grid (endpoint=False)
        self._H_stack = np.zeros((solver.N_k, _N_BDG, _N_BDG), dtype=complex)  # SCF grid buffer
        self.g_JT     = solver.p.g_JT
        self.Z        = solver.p.Z
        self.g_Eg2    = solver.p.g_Eg2
    
    def _build_H_stack(self, kpts: np.ndarray, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, g_t: float, g_J: float, r_Q: float = 0.0, F67s_mf: float = 0.0, out: Optional[np.ndarray] = None, Q_Eg2: float = 0.0, Delta_s7b: complex = 0.0j, Delta_d7b: complex = 0.0j) -> np.ndarray:
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
        D_s7b/D_d7b: OPTIONAL Γ6↔Γ7b analogs of the above (default 0j → identical to the Γ7a-only model
            when not supplied). Symmetry-allowed matrix element only where B1g_op[Γ6,Γ7b]≠0 (regime-dependent,
            see sec:operatorok discussion); the gap equation itself decides whether it wants to be nonzero.
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
        J_A1g_diag, J_B1g_bare = self.solver.p.exchange_channels(Q, n_kspace, tx_b, ty_b, g_J, r_Q)

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
        H_JT_loc = (self.g_JT * Q * self.solver.B1g_op).astype(complex)
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

    def compute_observables_vectorized(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, mu: float, ev: np.ndarray, ec: np.ndarray) -> float:
        """Returns M_stag: BdG staggered magnetisation (k-weighted, Nambu-corrected)."""
        solver = self.solver
        fn = _fermi_function(ev, solver.p.kT)
        fbar = 1.0 - fn

        uA, uB, vA, vB = _get_nambu_spinors(ec)

        mag_A = np.sum((np.abs(uA)**2 * solver.sz_op[None, :, None]) * fn[:, None, :]
                     + (np.abs(vA)**2 * solver.sz_op[None, :, None]) * fbar[:, None, :], axis=(1, 2))
        mag_B = np.sum((np.abs(uB)**2 * solver.sz_op[None, :, None]) * fn[:, None, :]
                     + (np.abs(vB)**2 * solver.sz_op[None, :, None]) * fbar[:, None, :], axis=(1, 2))

        # k-weighted average; /4 corrects for 2-sublattice × particle-hole Nambu doubling
        M_stag = float(np.dot(solver.k_weights, mag_A - mag_B)) / 4.0
        return M_stag

    def compute_gap_eq_vectorized(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, n_kspace: float, mu: float, t_eff: float, g_t: float, g_J: float, g_Delta_s: float, g_Delta_d: float, J_eff: float, Gamma_M: float, V_JT: float, V_cap: float, det_afm_sc: float, r_MQ: float, solve_state: '_SolveState', ev: np.ndarray, ec: np.ndarray, vertex_cache: dict = None, verbose: bool = False) -> Tuple[complex, complex, complex, complex, dict]:
        """
        Gap equation with q-dependent RPA pairing vertex V(q) built from normal-state (Δ=0) susceptibilities.
        """
        solver = self.solver
        # --- Vertex cache invalidation (Δ-independent for normal-state part!) ---
        staleness = (
            not isinstance(vertex_cache, dict)
            or abs(M - vertex_cache.get('M', 0.0)) > _M_THR_REL * float(np.sqrt(abs(det_afm_sc)))
            or abs(Q - vertex_cache.get('Q', 0.0)) > max(_Q_THR_REL * solver.p.lambda_hop, 1e-4)
            or (det_afm_sc * vertex_cache.get('det_afm_current', det_afm_sc)) < 0.0
        )
        if staleness:
            # Compute kernel and obtain base cache
            vertex_cache = solver.compute_pairing_kernel_and_build_cache(M, Q, n_kspace, mu, g_t, g_J, g_Delta_s, g_Delta_d, J_eff, Gamma_M, V_JT, V_cap, det_afm_sc, solve_state)
            # ---- Add normal-state spin/JT determinantal info ----
            ev, ec = solver._get_chi0_norm_cache(M, Q, n_kspace, mu, g_t, g_J, self)

            chi_SS_q0, chi_SQ_q0, chi_QQ_q0 = solver.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.zeros(2), Gamma_M, 0.0, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
            chi_SS_afm, chi_SQ_afm, chi_QQ_afm = solver.get_susceptibilities_sc(M, Q, 0.0j, 0.0j, n_kspace, mu, g_t, g_J, np.array([np.pi, np.pi]), Gamma_M, 0.0, 0.0, (ev, ec), apply_diamagnetic_QQ=True)
            
            vertex_cache.update({
                'chi_SS_q0':        chi_SS_q0,
                'chi_SQ_q0':        chi_SQ_q0,
                'chi_QQ_q0':        chi_QQ_q0,
                'chi_SS_afm':       chi_SS_afm,
                'chi_SQ_afm':       chi_SQ_afm,
                'chi_QQ_afm':       chi_QQ_afm,
                'det_afm':          solver._rpa_det(J_eff, V_JT, chi_SS_afm, chi_SQ_afm, chi_SQ_afm, chi_QQ_afm, r_MQ * math.sqrt(max(J_eff * V_JT, 0.0)))[0],  # Bare spin–orbital cross-vertex from the cluster-ED spin–JT cross-coupling J_MQ
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
        v_s_raw = abs(vertex_cache['K_pair_v_s'])
        v_d_raw = abs(vertex_cache['K_pair_v_d'])
        
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
        self.lambda_soc  = lambda_soc
        self._exclude_from_gp = False   # set True for G22>0 / spont-JT failures to keep them out of GP training

    def __repr__(self):
        regime = ('SC-trig' if 0.05 < self.lambda_JT < 1.0
                  else ('spont?' if self.lambda_JT >= 1.0 else 'closed'))
        lsoc_str = f", λ_soc={self.lambda_soc:.4f}" if self.lambda_soc is not None else ""
        return (f"OptimPoint(δ={self.doping:.3f}, Δ_tet={self.Delta_tetra:.3f}, "
                f"u={self.u:.2f}, g={self.g_JT:.3f}, t_pd={self.t_pd:.4f}{lsoc_str}, "
                f"Δ={self.Delta_total:.5f}, Tc={self.Tc*1000:.2f}meV, score={self.score:.5f}, "
                f"λ_JT={self.lambda_JT:.3f}[{regime}])")


class UnifiedBayesianOptimizer:
    """
    Unified 5D Bayesian optimiser over (Delta_tetra, lambda_soc, u, g_JT, t_pd).

    Four-phase pipeline:
      Phase 1 — 5D Differential Evolution scout  (analytic G-matrix, no SCF; ~100× faster per point):
        Hard constraints (H1–H2; score=0, excluded from GP):
         H1: G3[2,2] = 1−χ_QQ/K_eff > 0  — JT channel stable, not self-crossing
         H2: J_eff·χ_SS < 1              — below Stoner QCP

        Soft constraints / DE penalty (weights sum to 1.0):
         S1 (0.225): 0 < λ_min(G3) < 0.15  — near-critical, not past QCP
         S2 (0.225): reward λ_max; penalise only λ_max>_DE_LAMBDA_MAX_REWARD and unsolvable cases
         S3 (0.180): λ_JT > 0.05  — SC-JT coupling above viability threshold
         S4 (0.270): λ_JT in SC-triggered window [0.05, 1.0], peak at 0.45 (parabolic arch penalty)
         S5 (0.100): G22-margin > _DE_G22M_SAFE  — rewards distance from spontaneous-JT boundary
        
      Phase 2 — GP seed  (parallel SCF on top-k DE candidates):
        Full SCF seeds ARD Matérn-2.5 GP; _rebuild_orbital_operators called per clone.
      Phase 3 — TuRBO  (trust-region GP-EI, batch parallel):
        Greedy EI inside adaptive hypersphere.
      Phase 4 — Local refinement (optional): dense sampling around global best.

    Post-SCF scoring:
     Tier-1 hard guards (Mott, ξ, jchi, G22/λ_min)
     Tier-2 smooth weights (λ_JT arch, kernel sigmoid, Hessian floor)
     Tier-3 objective (Tc_proxy × conv_f × stoner_f × g22_margin_f × xi_f × lmax_boost × jchi_gate).

    Thread safety: _gp_lock guards GP observations; _tr_lock guards trust-region state.
    """
    _NDIMS   = 5
    _SEED_DE = 42
    _SEED_LHS= 43

    # Soft-constraint weights
    _W_LMIN = 0.225  # S1: lambda_min(G3) near-critical window
    _W_LEFF = 0.225  # S2: lambda_max pairing-vertex window
    _W_LJT  = 0.180  # S3: lambda_JT SC-JT threshold
    _W_DLAM = 0.270  # S4: λ_JT parabolic arch penalty
    _W_G22M = 0.10   # S5: optimal Delta_B1g_static (weight in DE penalty) 

    # Trust-region parameters (normalised [0,1]^5 space)
    _TR_INIT   = 0.80
    _TR_MIN    = 0.10
    _TR_MAX    = 1.00
    _TR_SHRINK = 0.65  # shrink ×0.65 on failure
    _TR_EXPAND = 1.35  # expand ×1.35 on consecutive improvement

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

    def _make_solver(self, dt: float, ls: float, u: float, gJT: float, tpd: float) -> 'RMFT_Solver':
        """Clone solver with all five parameters set; always rebuilds orbital operators."""
        p = copy.copy (self.solver.p)
        p.Delta_tetra = float(dt)
        p.lambda_soc  = float(ls)
        p.u           = float(u)
        p.g_JT        = float(gJT)
        p.t_pd        = float(tpd)
        p.__post_init__()
        return RMFT_Solver(p)

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

    def _eval_constraints(self, solver: 'RMFT_Solver', doping: float) -> Dict:
        """
        Evaluate H1/H2 hard and S1–S5 soft constraints on a solver clone.
        Early exit: if any H fails or partial_penalty(S1+S2+S3+S5) ≥ _S4_SKIP_THRESHOLD,
        return S4 = nan (treated as 0.5 by the DE objective).
        Skip rule: partial_penalty ≥ _FEASIBILITY_THRESHOLD → infeasible regardless of S4.
        """
        # Pre-SCF Mott hard-reject
        _abs_d_pre = max(abs(doping), 1e-6)
        if (2.0 * _abs_d_pre) / (1.0 + _abs_d_pre) < _G_T_COHERENCE_MIN:
            return {
                'hard_fail': True,
                'penalty':   100.0,
                'H1': 0.0, 'H2': 0.0,
                'jchi': 0.0,
                'mott_reject': True,
                'G_res': {},
            }
        
        # ── cheap G-matrix without dlambda ────────────────────────────
        G_res = solver.compute_G_instability(doping, solver.p.estimate_M0(doping))

        H1 = float(G_res['G22'])
        jchi = G_res['J_eff'] * G_res['chi_SS_afm']
        if jchi >= 1.0:
            chi_ss_gapped = solver.compute_chi_ss_with_infinitesimal_gap(G_res['M'], G_res, doping, 1-doping, delta_test=1e-5)
            if chi_ss_gapped < G_res['chi_SS_afm'] and G_res['J_eff'] * chi_ss_gapped < _BO_JCHI_GAPPED_CAP:
                jchi = G_res['J_eff'] * chi_ss_gapped

        H2 = 1.0 - jchi

        if H1 <= 0.0 or H2 <= 0.0:
            return {'hard_fail': True,
                    'penalty': (max(0.0, -H1) + max(0.0, -H2)) * 10.0,
                    'H1': H1, 'H2': H2,
                    'jchi': jchi, 'chi_ss_gapped': chi_ss_gapped, 'G_res': G_res}

        lmin    = float(G_res['lambda_min'])
        V_JT    = solver.p.g_JT**2 / max(solver._K_bare, _MATH_EPS)
        chi_orb = float(G_res['chi_QQ']) / max(solver.p.g_JT**2, 1e-12)
        # lam_JT = g²·χ_QQ(Δ=0)/K_bare: normal-state dimensionless JT coupling. Stable normal state requires lam_JT < 1 (χ_QQ < K_bare).
        lam_JT  = V_JT * chi_orb   # = chi_QQ / K_bare

        S1 = (0.0 if 0.0 < lmin < _DE_LAMBDA_MIN_OPT
              else min(abs(lmin) if lmin <= 0 else max(0.0, lmin - _DE_LAMBDA_MIN_OPT), 1.0))

        # S2: λ_max from the linearised gap equation at Δ=0.
        lmax_s2 = G_res['lambda_lin_max_q0']

        if not np.isfinite(lmax_s2):
            S2 = 1.0   # unknown → maximal penalty
        elif lmax_s2 > _DE_LAMBDA_MAX_REWARD:
            # RPA vertex near-divergent: numerically unreliable
            S2 = float(1.0 / (1.0 + np.exp(20.0 * (lmax_s2 - _DE_LAMBDA_MAX_REWARD))))
        else:
            S2 = float(1.0 - 1.0 / (1.0 + np.exp(-12.0 * (lmax_s2 - _DE_LAMBDA_MIN_OPT))))

        S3 = max(0.0, _DE_LAMBDA_JT_THRESH - lam_JT) / _DE_LAMBDA_JT_THRESH
        
        # ── expensive dlambda only for potentially feasible candidates ─
        S4 = 0.5
        S5 = float(1.0 - np.tanh(H1 / max(_DE_G22M_SAFE, _MATH_EPS)))
        partial_penalty = (self._W_LMIN * S1 + self._W_LEFF * S2 + self._W_LJT * S3 + self._W_G22M * S5)

        if partial_penalty < _FEASIBILITY_THRESHOLD:
            # S4: λ_JT parabolic arch on (0.05, 1.0); 0 penalty at peak (0.45), max at boundaries.
            if lam_JT <= 0.0 or lam_JT >= 1.0:
                S4 = 1.0   # outside window → maximum penalty
            else:
                lJT_c = float(np.clip(lam_JT, 0.0, 1.0))
                arch  = float(np.clip(-lJT_c * (lJT_c - 1.0) / _BO_ARCH_DENOM, 0.0, 1.0))
                S4    = 1.0 - arch   # S4=0: no penalty; S4=1: max penalty

        penalty = partial_penalty + self._W_DLAM * S4
        return {
            'hard_fail': False, 'penalty': float(penalty),
            'feasible':  penalty < _FEASIBILITY_THRESHOLD,
            'H1': H1, 'H2': H2, 'jchi': jchi,
            'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5,
            'lam_JT': lam_JT, 'lmin': lmin, 'G_res': G_res,
        }

    # ── Phase 1: DE scout ────────────────────────────────────────────────────
    def run_de_phase(self, doping: float, param_bounds_5d: Dict[str, tuple], popsize: int = 10, maxiter: int = 50, verbose: bool = True) -> Dict:
        """
        5D Differential Evolution using analytic G-matrix only (no SCF).

        Objective: penalty = H-violations * 10  +  sum(W_i * S_i) when hard-feasible.
        DE minimises penalty in normalised [0,1]^5 space. Returns ranked feasible archive for GP seeding.
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
            _scf_log("DE-SCOUT", f"5D DE scout (analytic G-matrix, no SCF)  pop={popsize*self._NDIMS}  maxiter={maxiter}  doping={doping:.3f}")

        de_res = differential_evolution(
            _obj, bounds=[(0.0, 1.0)] * self._NDIMS,
            popsize=popsize, maxiter=maxiter, seed=self._SEED_DE,
            tol=1e-4, mutation=(0.5, 1.2), recombination=0.85,
            polish=False, workers=1, updating='immediate')

        feasible = sorted([r for r in _archive if r.get('feasible', False)],
                          key=lambda r: r['penalty'])
        if verbose:
            _scf_log("DE-SCOUT", f"Done ({_time.time()-t0:.1f}s)  n_eval={len(_archive)}  n_feasible={len(feasible)}")
            for i, r in enumerate(feasible[:5]):
                _scf_log("DE-SCOUT",
                    f"top-{i+1}: Dt={r['dt']:.3f} ls={r['ls']:.4f}  u={r['u']:.2f}  g={r['gJT']:.4f}"
                    f"  tpd={r['tpd']:.4f}  penalty={r['penalty']:.4f}  S5={r.get('S5', float('nan')):.3f}")
        return {'archive': _archive, 'feasible': feasible, 'de_result': de_res, 'elapsed_s': _time.time() - t0}

    def _make_phase_grid(self, doping_bounds: tuple) -> tuple:
        """Return (dg, fallback_point) shared by all optimisation phases.
        fallback_point: safe zero-score sentinel (high u=8, small g_JT=0.2 → treated as bad by GP).
        """
        d_mid = 0.5 * (doping_bounds[0] + doping_bounds[1])
        dg    = np.linspace(doping_bounds[0], doping_bounds[1], self.n_doping_scan)
        fb    = OptimPoint(d_mid, 0.0, 8.0, 0.2, 0.5, 0.0, False, score=0.0,
                           lambda_soc=self.solver.p.lambda_soc)
        return dg, fb

    # ── Phase 2: GP seed ─────────────────────────────────────────────────────
    def run_gp_seed_phase(self, doping_bounds: tuple, de_feasible: list, top_k: int = 12, verbose: bool = True) -> None:
        """
        full SCF on top_k DE candidates in parallel; results seed the GP.
        Falls back to LHS if de_feasible is empty.
        """
        if not de_feasible:
            _scf_log("GP-SEED", "No feasible DE points — falling back to LHS seeding.")
            lhs_pts = [self._denormalize(x) for x in self._lhs_sample(top_k)]
            candidates = [{'dt': p[0], 'ls': p[1], 'u': p[2],
                           'gJT': p[3], 'tpd': p[4]} for p in lhs_pts]
        else:
            candidates = de_feasible[:top_k]

        dg, fb = self._make_phase_grid(doping_bounds)
        t0     = _time.time()

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
        sigma = np.maximum(sigma, _MATH_EPS)
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
        trust-region GP-EI with parallel batch SCF per iteration.
        TR state is mutated only from the main thread after each batch completes, avoiding any concurrent writes.
        """
        dg, fb = self._make_phase_grid(doping_bounds)
        t0     = _time.time()

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
                    f"[{i+1:3d}/{n_iterations}]  TR={tr_r:.3f}  best Tc={bp_pt.Tc*1000:.2f}meV"
                    f"  score={bp_pt.score:.5f}  {'↑' if improved else '—'}  ({_time.time()-t0:.0f}s)")

            prev_best = cur_best
            with self._tr_lock:
                if self._tr_radius <= self._TR_MIN:
                    _scf_log("TURBO", f"TR min reached — early stop at iter {i+1}.")
                    break

        if verbose:
            _scf_log("TURBO", f"Done ({(_time.time()-t0)/60:.1f} min)  GP obs={len(self._gp_obs)}/{len(self.observations)}")

    # ── Phase 4: Local refinement ────────────────────────────────────────────
    def run_local_refinement(self, doping_bounds: tuple, n_grid: int = 12, margin: float = 0.10, verbose: bool = True) -> None:
        """dense random sampling around the global best ±margin."""
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
        iM0 = solver.p.estimate_M0(doping_grid[0])
        iD0 = 0.02
        for doping in doping_grid:
            iM = prev['M']                                 if prev else iM0
            iD = max(prev['Delta_s']+prev['Delta_d'], iD0) if prev else iD0
            try:
                pt = self._eval_one_doping(solver, doping, Delta_tetra, u, gJT, t_pd, iM, iD, lambda_soc=lsoc)
            except Exception as e:
                _scf_log(f"FULL d={doping:.3f}", f"SCF error: {e}")
                pt = OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)
            if pt.result:
                prev = pt.result
            if best is None or pt.score > best.score:
                best = pt
        return best

    def _eval_one_doping(self, solver, doping, Delta_tetra, u, gJT, t_pd, initial_M, initial_Delta, lambda_soc=None) -> 'OptimPoint':
        """Single-doping SCF evaluation with dual-basin JT probe and parallel multi-seed restart.

        When the SCF dynamics are classified as 'first_order_jump' or 'hysteretic', the system
        is near a first-order transition and a single initial condition may miss the true
        free-energy minimum.  In that case we launch _N_MULTISEED additional seeds in parallel
        (varying M₀, Q₀, Δ₀) and keep the lowest free-energy converged result.
        """
        _N_MULTISEED = 4   # extra seeds for first-order basin search; each uses _tpl_ctx(limits=1)
        tag = f"FULL d={doping:.3f}"
        t0  = _time.time()
        result    = solver.solve_self_consistent(doping, initial_Delta, initial_M)
        Delta     = result['Delta_s'] + result['Delta_d']
        converged = result['converged']

        # ── Parallel multi-seed for first-order / hysteretic dynamics ────────────
        _dyn_regime = result.get('scf_dynamics_regime', 'converging')
        if _dyn_regime in ('first_order_jump', 'hysteretic'):
            _scf_log(tag, f"⚠ {_dyn_regime} dynamics — launching {_N_MULTISEED} parallel seeds for basin search")
            rng = np.random.default_rng(seed=abs(hash((doping, Delta_tetra, u, gJT, t_pd))) % (2**31))
            # Seed grid spans relevant (M, Q, Δ) space near current estimate
            M_est  = result['M']
            D_est  = max(Delta, initial_Delta, 0.02)
            _seeds = []
            for _ in range(_N_MULTISEED):
                _seeds.append({
                    'M': float(np.clip(M_est * rng.uniform(0.5, 1.3), _KICK_M_CLIP_LO, _KICK_M_CLIP_HI)),
                    'D': float(rng.uniform(1e-4, D_est)),
                })

            def _run_seed(seed):
                with _tpl_ctx(limits=1):
                    try:
                        _s = copy.copy(solver)
                        _s._reset_transient_state()
                        r = _s.solve_self_consistent(doping, seed['D'], seed['M'])
                        return r
                    except Exception:
                        return None

            n_w = min(_os.cpu_count() or 1, _N_MULTISEED, _BO_MAX_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_w) as ex:
                _seed_results = list(ex.map(_run_seed, _seeds))

            # Pick the lowest total free energy among all converged seeds (and current result).
            _candidates = [r for r in _seed_results if r is not None and r.get('converged', False)]
            if converged:
                _candidates.append(result)
            if _candidates:
                result = min(_candidates, key=lambda r: r['F_bdg'])
                Delta  = result['Delta_s'] + result['Delta_d']
                converged = result['converged']
                _scf_log(tag, f"Multi-seed best: |Δ|={Delta:.5f}  F={result['F_bdg']:.5f}  dyn={result.get('scf_dynamics_regime','?')}")

        Tc = 0.0
        if converged and Delta > 1e-6:
            try:
                Tc = solver.compute_Tc_by_gap_suppression(doping, sc_result=result)['Tc']
            except Exception:
                pass

        # Dual-basin probe: SC gap exists but Q≈0 → nudge toward JT basin.
        if (converged
                and Delta > solver.p.tol * 5
                and abs(result['Q']) < 1e-3):
            try:
                r2 = solver.solve_self_consistent(doping, result['Delta_s'] + result['Delta_d'], result['M'])
                # Compare total free energy (BdG + cluster) to avoid basin bias from local K_eff differences between the Q=0 and Q≠0 solutions.
                _F1_tot = result['F_bdg']
                _F2_tot = r2['F_bdg']
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

        M_conv = result['M']
        G_n = solver.compute_G_instability(doping, M_conv)
        if Delta < 1e-8 and Tc < 1e-6:
            return self._g_fallback_score(initial_M, doping, Delta_tetra, u, gJT, t_pd)
        
        lambda_JT_sc = result['lambda_JT_sc']
        _Gamma_M = solver.p.moriya_gamma(doping, np.sqrt(0.5 * (result['tx']**2 + result['ty']**2)), result['J_eff']) 
        # G_sc: SC-state quantities for _score (d2F_Q at Δ≠0)
        if converged and Delta > solver.p.tol:
            _vbdg_sc = solver._get_vbdg()
            _E_k_sc, _V_k_sc = np.linalg.eigh(
                _vbdg_sc._build_H_stack(_vbdg_sc._kpts, result['M'], result['Q'], complex(result['Delta_s']), complex(result['Delta_d']), result['n_kspace'], result['mu'], result['g_t'], result['g_J'], result['r_Q'], result['F67s_mf'], out=_vbdg_sc._H_stack)
                )
            _chi_SS_sc, _chi_SQ_sc, _ = solver.get_susceptibilities_sc(result['M'], result['Q'], complex(result['Delta_s']), complex(result['Delta_d']), result['n_kspace'], result['mu'], result['g_t'], result['g_J'], np.array([np.pi, np.pi]), _Gamma_M, result['r_Q'], result['F67s_mf'], (_E_k_sc, _V_k_sc), apply_diamagnetic_QQ=True)
            G_sc = {
                'chi_SQ_s':     _chi_SQ_sc,
                'chi_SS_afm':   _chi_SS_sc,
            }
        else:
            # No SC gap or not converged — _score handles nan gracefully via fallback 0.5
            G_sc: dict = {}

        score = self._score(Delta, converged, result, Tc, G_n, G_sc, lambda_JT_sc)
        lambda_lin_max = result['lambda_lin_max']
        regime  = ('SC-triggered' if 0.05 < lambda_JT_sc < 1.0
                     else ('strong-coupling' if lambda_JT_sc >= 1.0 else 'JT-closed'))
        _hess_eigs_n = result['hessian_result'].get('eigenvalues')
        lmin = float(_hess_eigs_n[0]) if _hess_eigs_n is not None else float('nan')
        stoner_ok = not result['afm_unstable']
        _scf_log(tag,
            f"D={Delta:.5f} Tc={Tc*1000:.2f}meV score={score:.5f}"
            f"  λ_JT_sc={lambda_JT_sc:.3f}[{regime}]"
            f"  λ_lin_max={lambda_lin_max:.3f}({result['gap_symmetry']})"
            f"  lmin(H)={lmin:+.4f}[{G_n['instab_dir']}]"
            f"  {G_n['instab_info'].weight_for_log}"
            f"  {'ok' if converged else 'nc'} ({_time.time()-t0:.1f}s)")
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, Delta, converged, result, lambda_JT_sc, lambda_lin_max, stoner_ok, score, Tc, lambda_soc)

    def _g_fallback_score(self, M0, doping, Delta_tetra, u, gJT, t_pd) -> 'OptimPoint':
        """Cheap G-matrix proxy score when SCF finds no SC gap."""
        try:
            s2 = copy.copy(self.solver)
            s2.p = copy.copy(self.solver.p)
            s2.p.Delta_tetra = float(Delta_tetra)
            s2.p.u    = float(u)
            s2.p.g_JT = float(gJT)
            s2.p.t_pd = float(t_pd)
            s2._full_rebuild()
            G = s2.compute_G_instability(doping, M0)
        except Exception:
            return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)

        ni = G['instab_info']
        if not ni.jt_stable or not ni.full_stable:
            return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=0.0)

        g22_f = _BO_SPONT_JT_PEN + (1.0 - _BO_SPONT_JT_PEN) / (1.0 + np.exp(-ni.G22 / _BO_SIGMOID_W))
        _leff_gate = float(np.clip(G['lambda_eff'] / 0.5, 0.0, 3.0))
        sc = _BO_G_FALLBACK * (1.0 - min(ni.lambda_min, 1.0)) * g22_f * (1.0 + _leff_gate)
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, 0.0, False, score=sc)

    def _score(self, Delta: float, converged: bool, result: dict, Tc: float, G_n: dict, G_sc: dict, lambda_JT: float = float('nan')) -> float:
        """
        Post-SCF scoring: three-tier multiplicative architecture.

        Tier 1 — Hard physical constraints (return 0 immediately):
            mott    : g_t < _G_T_COHERENCE_MIN or ξ/a < 1.0  — incoherent / artefact SC
            jchi    : J·χ_SS > _JCHI_HARD_REJECT — deep AFM, SC impossible
            g22     : G22 ≤ 0 or λ_min(G3) ≤ 0  — spontaneous JT / unstable normal state

        Tier 2 — Smooth mechanism weights (continuous in [0,1]):
            w_lJT        : λ_JT_sc parabola arch on [0,1]; x=(λ_JT−0.05)/0.95,
                           arch = clip(−x(x−1)/_BO_ARCH_DENOM, 0, 1).
                           Mathematical peak of unclipped parabola at x=0.5 → λ_JT≈0.52; w_lJT=0.10 for λ_JT≥1.
            w_lJT_kernel : sigmoid(10·(lJTk − 0.05))
            w_hessian    : sigmoid(−lmin_sc / 0.05), floor 0.30
            w_softening  : sigmoid(−Δ(d²F_Q)/0.05) — SC-induced Q-mode softening
            w_chisq      : sigmoid(|χ_SQ| / 0.1) — spin-orbital cross-channel strength

        Tier 3 — Optimisation objective:
            Tc_proxy · conv_f · stoner_f · g22_margin_f · xi_f · lmax_boost · jchi_gate
        """
        # ── Tier 1: hard guards ───────────────────────────────────────────────
        if result.get('mott_suspect', False):
            return 0.0
        _g_t_sc = float(result.get('g_t', 1.0))
        _xia_sc = float(result.get('xi_nodal', float('nan')))
        if _g_t_sc < _G_T_COHERENCE_MIN or (np.isfinite(_xia_sc) and _xia_sc < 1.0):
            return 0.0

        _jchi = float(np.clip(result['J_eff'] * result['chi_SS_afm'], 0.0, 10.0))
        if _jchi > _JCHI_HARD_REJECT:
            return 0.0

        instab_n = G_n['instab_info']
        if not instab_n.jt_stable or not instab_n.full_stable:
            return 0.0
        G22 = instab_n.G22
        
        # ── Tier 2: smooth mechanism weights (λ_JT arch) ────────────────────────
        if not np.isfinite(lambda_JT):
            w_lJT = 0.5
        elif lambda_JT >= 1.0:
            w_lJT = 0.10
        else:
            lJT_c = float(np.clip(lambda_JT, 0.0, 1.0))
            w_lJT = float(np.clip(-lJT_c * (lJT_c - 1.0) / _BO_ARCH_DENOM, 0.0, 1.0))

        # λ_JT_kernel
        lJTk = float(result.get('lambda_JT_kernel', float('nan')))
        if not np.isfinite(lJTk):
            w_lJT_kernel = 0.5
        elif lJTk >= 1.0:
            w_lJT_kernel = _BO_W_LJT_OVR_SAT  # Rayleigh quotient > 1: numerical over-saturation
        else:
            w_lJT_kernel = float(1.0 / (1.0 + np.exp(-_BO_LJT_KERNEL_SIG * (lJTk - _LAMBDA_JT_VIABLE))))

        # Hessian: sigmoid(−lmin_sc / 0.05). Floor 0.30 for missing/unconverged data.
        _hess_eigs_sc = result.get('hessian_result', {}).get('eigenvalues')
        lmin_sc = float(_hess_eigs_sc[0]) if _hess_eigs_sc is not None else None
        if lmin_sc is not None and np.isfinite(lmin_sc):
            w_hessian = float(1.0 / (1.0 + np.exp(lmin_sc / _BO_SC_HESS_SIG)))
        else:
            w_hessian = _BO_W_HESSIAN_FLOOR

        # SC-induced Q-mode softening (negative = softening)
        jt_softening = result.get('K_eff_net', float('nan'))
        w_softening = float(1.0 / (1.0 + np.exp(jt_softening / _SCORE_SOFTENING_SIG)))  # 1 if strongly negative

        # χ_SQ spin-orbital cross-channel strength
        chi_sq = G_sc['chi_SQ_s']
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
        # apply a strong penalty if J·χ(Δ=0)≥1 and the RPA vertex is singular.
        stoner_f = 1.0 if not result['afm_unstable'] else _BO_W_STONER_BAD

        # softplus(λ_max) grows continuously with pairing strength.
        _softplus = float(np.log1p(np.exp(np.clip(result['lambda_lin_max'], -10.0, 10.0))))
        lmax_boost = float(np.clip(_softplus, 0.0, 2.0))

        # jchi_gate shapes the score *within* the feasible region to prefer the near-QCP sweet spot.
        jchi_gate = float(np.exp(-0.5 * ((_jchi - _BO_OPT_JCHI) / _BO_SIG_JCHI) ** 2))
        jchi_gate = float(np.clip(
            jchi_gate + (_BO_JCHI_FLOOR if _jchi < _BO_JCHI_NOISE else 0.0), 0.0, 1.0))
        # Richardson consistency: if the extrapolation has not converged, lambda_JT_opt is unreliable
        w_richardson = 1.0 if bool(result.get('richardson_ok', True)) else 0.70
        return (Tc_proxy * conv_f * stoner_f * g22_margin_f * w_lJT * w_lJT_kernel * w_hessian * w_softening * w_chisq * xi_f * lmax_boost * jchi_gate * w_richardson)

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
        """ Full four-phase optimisation. """
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
        t_pd             = 0.550,
        u                = 15.000,
        lambda_soc       = 0.039,
        Delta_tetra      = -0.080,
        g_JT             = 0.350,
        K_lattice        = 2.230,
        lambda_hop       = 1.000,
        g_Eg2            = 0.100,
        K_lattice_Eg2    = 6.500,
        Delta_CT         = 2.200,
        Delta_B1g_static = 0.020,
        hybrid_scale     = 6.000,
        Upp_ratio_bare   = 0.400,
        Z                = 4,
        kT               = 0.005,
        tol              = 1e-4,
        )

    target_doping = 0.100
    doping_margin = 0.20          # scan covers target ± 20 %
    min_doping    = max(target_doping * (1.0 - doping_margin), _G_T_COHERENCE_MIN / (2.0 - _G_T_COHERENCE_MIN))
    max_doping    = target_doping * (1.0 + doping_margin)
    initial_Delta = 9e-3

    _scf_log("INIT",
                f"t_pd={params.t_pd:.3f} eV  u={params.u:.3f}  λ_SOC={params.lambda_soc:.3f} eV"
                f"  Δ_tetra={params.Delta_tetra:.3f} eV  g_JT={params.g_JT:.3f} eV  K_lattice={params.K_lattice:.4f}"
                f"  lambda_hop={params.lambda_hop:.3f}  Δ_CT={params.Delta_CT:.3f} eV   Δ_ip={params.Delta_B1g_static:.3f} eV"
                f"  kT={params.kT*1000:.2f} meV  Z={params.Z}  N_k={params.N_k}")
    _scf_log("DERIVED",
                f"multi_op (normalised):\n{np.array2string(params.multi_op, precision=4, suppress_small=True)}"
                f"  t0={params.t0:.4f} eV  U_dd={params.U_dd:.4f} eV  J_pdct={params.J_pdct:.4f} eV  Δ_CF={params.Delta_CF:.5f} eV "
                f"  Γ₇split={params.g7split:.5f} eV [{'⚠ < 2kT' if abs(params.g7split) < 2.0 * params.kT else '✓'}]"
                f"  b1g_weight={params.b1g_weight:.4f} [{'SC-triggered only' if params.b1g_weight > 0.90 else 'partial D2h mixing'}]")

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

    need_optimization = False
    # ── Section 0: Unified 5D optimisation ───────────────────────────────────────────────────
    if need_optimization:
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

        unified_bo = UnifiedBayesianOptimizer(solver_ref, n_doping_scan=7)
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
        is_reliable = lambda res: (
            (False, "not converged")
            if not res or not res.get("converged", False)
            else (False, "no Hessian")
            if res.get("hessian_result", {}).get("eigenvalues") is None
            else (False, "saddle (Hessian has negative eigenvalue)")
            if not all(e > -1e-6 for e in res["hessian_result"]["eigenvalues"])
            else (True, "ok")
        )

        hess_min = lambda res: (
            np.min(res["hessian_result"]["eigenvalues"])
            if res.get("hessian_result") is not None
            and res["hessian_result"].get("eigenvalues") is not None
            else float("nan")
        )

        energies = {}
        reliable = {}

        for key in ("ref", "normal", "SC_Q0"):
            res = results[key]
            energies[key] = res.get("F_bdg", 0.0) + res.get("F_per_site", 0.0)
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
        _scf_log("REF-SCF", f"  M={_ref_M:.4f}  Q={_ref_Q:+.5f}  Δs={_ref_result['Delta_s']:.5f} eV  Δd={_ref_result['Delta_d']:.5f} eV  μ={_ref_result['mu']:.4f} eV")
        _scf_log("REF-SCF", f"  Irrep R={_ref_result['selection_ratio']:.4f}  JT {'ALLOWED ✓' if _ref_result['selection_ratio'] > 0.02 else 'BLOCKED ✗'}")

        # — SC-JT window: K_eff path (Δ=0 → Δ≠0) —
        _scf_log("SCF-RES", (
            f"  δK_eff={_ref_result['K_eff_net']:+.4f}"
            f"  → {'✓ gap softens lattice (SC-triggered JT enabled)' if _ref_result['K_eff_net'] < -1e-4 else '⚠ gap stiffens lattice' if _ref_result['K_eff_net'] > 1e-4 else '≈ no K_eff change'}"
        ))

        # ── G-matrix at self-consistent M (normal-state instability) ─
        _scf_log("G-MATRIX", "="*60)
        G_base = solver_ref.compute_G_instability(target_doping, _ref_M)

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
        _scf_log("SCF-RES", f"Gap eq: λ_lin_max={_lmax_ref:.4f}  g_Δ={_ref_result['g_delta_dom']:.3f}")

        if _frac is not None:
            _ch_note  = _ref_result['gap_symmetry']
            _neg_note = '  [⚠ λ<0: FS-avg repulsive — instability requires nodal sign change]' if _lmax_ref < 0 else ''
            _scf_log("SCF-RES", f"  Channel: λ_s={_lmax_ref * _frac[0]:.4f}"
                     f"  λ_d={_lmax_ref * _frac[1]:.4f}  [{_ch_note}]{_neg_note}")
        
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
        _dlam_fs = _ref_result.get('dlam_dQ_fs', float('nan'))
        _scf_log("SCF-RES", f"  λ_min(H_SC)={_hess_lmin_sc:+.4f}"
                 f"  {'✓ SC-triggered JT CONFIRMED' if np.isfinite(_hess_lmin_sc) and _hess_lmin_sc < 0.0 else '— JT not triggered'}"
                 f"  |  ∂λ/∂Q(FS)={_dlam_fs:+.4f} eV⁻¹"
                 f"  {'✓ SC condensate enhances pairing at hot spots' if np.isfinite(_dlam_fs) and _dlam_fs > 0 else '⚠ SC-JT FS coupling absent'}")

        # — Stoner/Moriya: J_eff from the analytic Gutzwiller renormalisation —
        _stoner_r    = _ref_result['J_eff'] * _ref_result['chi_SS_afm']
        _ston_status = ('✓ near QCP' if 1.0 > _stoner_r > 0.7
            else ('⚠ near/past AFM QCP' if 2.0 > _stoner_r >= 1.0
                else ('safe' if _stoner_r <= 0.7 else '✗ deeply past QCP')))
        _scf_log("SCF-RES", f"J_eff={_ref_result['J_eff']:.4f} eV  χ_SS_AFM(Δ=0)={_ref_result['chi_SS_afm']:.4f}  J·χ_SS={_stoner_r:.4f} [{_ston_status}]"
                 f"  r_Q={_ref_result['r_Q']:.3f}")

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
        solver_ref.estimate_chi_SQ_q_full(target_doping, _ref_M, _ref_Q, _ref_result['Delta_s'], _ref_result['Delta_d'], _ref_result['n_kspace'], _ref_result['mu'], _ref_result['J_eff'], _ref_result['r_Q'], _ref_result['F67s_mf'],  n_q=35)

        # Three independent Tc estimates (no shared label):
        #   Tc₁: McMillan (analytical, ω_SF = J_eff)
        #   Tc₂: λ(T)=1 crossing  (normal-state SCF scan)
        #   Tc₃: thermodynamic free-energy crossing (first-order aware)
        #   Tc_sp: spinodal (metastability limit, companion to Tc₃)
        _pre_Delta_total = float(_ref_result['Delta_s']) + float(_ref_result['Delta_d'])
        _scf_log("TC-PRELIM", f"Pre-BO Tc estimates at |Δ|={_pre_Delta_total*1000:.3f} meV")
        _sc_viable = (not _ref_result.get('mott_suspect', False)) and (float(_ref_result['g_t']) >= _G_T_COHERENCE_MIN) and bool(_ref_result['converged'])
        if _sc_viable:
            _omega_SF  = float(_ref_result.get('J_eff', 2 * params.Z * params.J_pdct * params.t0**2))
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
                    coupling_regime = ('BCS-like' if ratio_2D < _BCS_RATIO_STRONG
                                       else 'strong' if ratio_2D < _BCS_RATIO_VSTRONG
                                       else 'very-strong' if ratio_2D < _BCS_RATIO_EXOTIC
                                       else 'exotic / non-phononic')
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