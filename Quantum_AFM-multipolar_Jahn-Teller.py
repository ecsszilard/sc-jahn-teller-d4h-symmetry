import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings
import copy
import time as _time
import concurrent.futures
import os
import sys

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not available — Bayesian optimisation falls back to Latin Hypercube "
        "random search.  Install with: pip install scikit-learn",
        RuntimeWarning,
    )

# =============================================================================
# 0. PHYSICAL PARAMETERS & MODEL DEFINITION
# =============================================================================

@dataclass
class ModelParams:
    """
    Derived (set in __post_init__):
      Delta_CF  : Γ₆–Γ₇ splitting from SOC+CF Hamiltonian (eV)
      U         : Hubbard U = u·t0 (eV)
      U_mf      : Stoner Weiss-field = 0.5·Δ_CF (eV)
      K_lattice : g_JT²/(alpha_K·Δ_CF) (eV/Å²); alpha_K > 1 → SC-triggered JT only
      omega_JT  : JT phonon frequency (40–80 meV); enters only D_phonon = 2/ω_JT
    """
    # --- Primary inputs ---
    t0:            float      # eV    bare hopping integral
    u:             float      # —     U/t0 ratio (Hubbard U = u·t0 ≈ 3.43 eV; charge-transfer regime, typ. 6–10)
    lambda_soc:    float      # eV    atomic SOC λ (t2g shell, ~0.05–0.15 eV); determines Γ₆–Γ₇ splitting
    Delta_tetra:   float      # eV    tetragonal axial CF Δ_tet·Lz²; negative = z-compression
                              #       Partial cancellation with SOC tunes Γ₆–Γ₇ gap independently of λ
    g_JT:          float      # eV/Å  Jahn–Teller electron–phonon coupling
    alpha_K:       float      # —     spring stiffness ratio; K = g²/(alpha_K·Δ_CF)
                              #       alpha_K > 1 → SC-triggered JT; alpha_K = 1 → marginal boundary
    lambda_hop:    float      # Å     hopping decay length for B₁g anisotropy: t(Q) = t0·exp(±Q/λ_hop)
    eta:           float      # —     Γ₇ AFM asymmetry relative to Γ₆
    doping_0:      float      # —     superexchange regularisation (suppresses g_J→4 divergence near half-filling)
    # --- Charge-transfer / RPA / gap symmetry ---
    Delta_inplane: float      # eV    B2g in-plane anisotropy Δ_ip·(Lx²−Ly²); splits Γ₇ into Γ₇a+Γ₇b
                              #       (preserves Kramers, prevents spontaneous JT from residual Γ₇ degeneracy)
    Delta_CT:      float      # eV    charge-transfer gap (ZSA scale); sets scale for CT-insulator crossover
                              #       Reducing toward ~1.2 eV increases multipolar fluctuations
    omega_JT:      float      # eV    JT phonon frequency (40–80 meV); enters only D_phonon = 2/ω_JT
                              #       All free-energy magnitudes use adiabatic g²/K
    rpa_cutoff:    float      # —     Stoner denominator floor 1/max(sd, cutoff); default 0.12 → max 8.3×
    d_wave:        bool       # —     True → B₁g d-wave φ(k)=cos kx−cos ky; False → s-wave φ(k)=1

    # -----------------------------------------------------------------------
    # TIER 2 — BAYESIAN SEARCH VARIABLES (maximise Δ_total = Δ_s + Δ_d)
    # -----------------------------------------------------------------------
    # target_doping (δ) — doping/filling knob (range 0.08–0.30), passed as arg to solve_self_consistent
    #
    # V_s_scale  — dimensionless scale for the on-site Γ₆⊗Γ₇ orbital singlet channel.
    #              Physical V_s = V_s_scale · g²/K (eV).
    #              Gutzwiller renormalization:
    #                g_Delta_s = sqrt(g_{t,Γ6} · g_{t,Γ7})
    #              where g_{t,Γ7} ≈ 1 because Δ_CF ≫ kT keeps Γ7 nearly empty.
    #              → g_Delta_s ≈ sqrt(g_t)  (intermediate suppression; weaker than g_t).
    #              Search range: [0.1, 2.5] (dimensionless).
    #
    # V_d_scale  — dimensionless scale for the inter-site d-wave B₁g channel.
    #              Physical V_d = V_d_scale · g²/K (eV).
    #              Gutzwiller renormalization: g_Delta_d = g_J = 4/(1+δ)² →
    #              strongest at half-filling, vanishing at large doping.
    #              Search range: [0.1, 2.5] (dimensionless).
    #
    # The two scales are INDEPENDENT: V_s + V_d is NOT constrained to a constant.
    # -----------------------------------------------------------------------
    V_s_scale:     float
    V_d_scale:     float

    # -----------------------------------------------------------------------
    # TIER 3 — SCF NUMERICAL HYPER-PARAMETERS (tune once, do NOT Bayesian-optimise)
    # -----------------------------------------------------------------------
    mu_LM:         float      # Levenberg–Marquardt floor for M Newton step (default 4.0), larger → smaller γ_M → more conservative M update.
    ALPHA_HF:      float      # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton; default 0.2)
    CLUSTER_WEIGHT: float     # cluster ED vs BdG weight in M update (default 0.35)
    ALPHA_D:       float      # Newton vs gap-equation fixpoint blend for Δ update (default 0.3)
    mu_LM_D:       float      # LM floor for Δ Newton step (default 1.0)

    # --- Numerics ---
    Z:             int        # 2D square lattice coordination number
    nk:            int        # k-grid: MUST BE EVEN for commensurate q_AFM=(π,π);
                              # odd nk+1 sub-grid used for chi0 Simpson integration
    kT:            float      # eV  temperature
    a:             float      # Å   lattice constant
    max_iter:      int
    tol:           float
    mixing:        float

    def __post_init__(self):
        """Derive all secondary parameters from the primary inputs."""
        if self.alpha_K <= 1.0:
            raise ValueError(
                f"alpha_K={self.alpha_K:.6f} must be strictly > 1.0 to prevent "
                f"spontaneous JT (K < K_spont requires alpha_K > 1)."
            )
        # Γ₆–Γ₇a splitting from SOC+D4h CF. Pure cubic (Δ_tet=Δ_ip=0) → Δ_CF ≈ (3/2)·λ_SOC.
        # Delta_axial tunes Γ₆–Γ₇ gap; Delta_inplane splits Γ₇ quartet into Γ₇a+Γ₇b.
        self.Delta_CF: float = _gamma_splitting(
            self.lambda_soc, self.Delta_tetra, self.Delta_inplane)

        # Hubbard U from dimensionless ratio
        self.U: float = self.u * self.t0

        # Stoner Weiss-field U_mf = 0.5·Δ_CF (mean-field splitting amplitude ~0.05–0.10 eV in
        # Gutzwiller-projected band model, NOT the bare charge-transfer U ~1–3 eV)
        self.U_mf: float = 0.5 * self.Delta_CF

        # K = g²/(alpha_K·Δ_CF); alpha_K > 1 → K < K_spont = g²/Δ_CF → spontaneous JT blocked
        self.K_lattice: float = self.g_JT**2 / (self.alpha_K * max(self.Delta_CF, 1e-6))

        # omega_JT is direct (40–80 meV); enters only D_phonon = 2/ω (shape of V(k,k')).
        # ZSA: t0_eff = t_pd²/Δ_CT
        self.t_pd: float = np.sqrt(self.t0 * max(self.Delta_CT, 1e-9))

        # Even-nk enforcement: k_i + (π,π) must map to another grid point exactly
        if self.nk % 2 != 0:
            object.__setattr__(self, 'nk', self.nk + 1)

    def summary(self, delta: float = 0.15) -> None:
        """
        Print primary inputs, derived quantities and regime diagnostics.
        Fully consistent with SC-triggered JT hypothesis.
        """

        # ==========================================================
        # 1) Gutzwiller renormalization & magnetic sector
        # ==========================================================
        g_t = 2 * delta / (1 + delta)
        g_J = 4 / (1 + delta) ** 2

        t_eff = g_t * self.t0
        bw2 = 2 * t_eff

        f_d = delta / (delta + self.doping_0)
        J_eff = g_J * 4 * t_eff**2 / self.U * f_d
        h_afm = (g_J * self.U_mf / 2 + J_eff * self.Z / 2) * 0.6 / 2

        # --- AFM metallic check ---
        ok_metal = h_afm < bw2

        # ==========================================================
        # 2) Pairing interaction scales
        # ==========================================================
        V_base  = self.g_JT**2 / self.K_lattice   # bare g²/K scale (JT channel)
        V_eff_s = self.V_s_scale * V_base         # on-site channel
        V_eff_d = self.V_d_scale * V_base         # inter-site channel

        N0 = 1.0 / (np.pi * max(t_eff, 1e-6))     # 2D DOS estimate

        # --- λ_pair: pairing eigenvalue (BCS kernel projection) ---
        #
        # Three pairing channels:
        #   V_JT   = g²/K              (adiabatic JT phonon)
        #   V_spin = U²·χ₀ / (1−Uχ₀)²   (Moriya-RPA spin fluctuations)
        #   V_mix  = U_mf²/Δ_CF        (Γ₆–Γ₇ coherent mixing)
        #
        # Spin channel follows Berk–Schrieffer/paramagnon form.
        # Near AFM criticality (Uχ₀ → 1) V_spin is strongly enhanced.
        # χ₀ is estimated analytically as χ₀ ~ N(0) / [1 + (U_mf / π t_eff)²],
        # where U_mf suppresses χ₀ (Weiss field), while U is the pairing vertex.

        chi0_spin_est = N0 / (1.0 + (self.U_mf / max(np.pi * t_eff, 1e-9))**2)
        stoner_denom  = max(abs(1.0 - self.U * chi0_spin_est), 0.05)
        V_JT_ch       = self.g_JT**2 / max(self.K_lattice, 1e-9)
        V_spin_ch     = (self.U**2 * chi0_spin_est) / stoner_denom**2   # Moriya-RPA
        V_mix_ch      = self.U_mf**2 / max(self.Delta_CF, 1e-9)         # Γ₆–Γ₇ coherence
        # Inter-site (d-wave) and on-site channels weighted separately
        lambda_pair_d = N0 * self.V_d_scale * V_JT_ch + N0 * V_spin_ch + N0 * V_mix_ch
        lambda_pair_s = N0 * self.V_s_scale * V_JT_ch
        lambda_pair   = max(lambda_pair_d, lambda_pair_s)   # dominant channel

        # --- Pairing strength checks ---
        ok_pairing_min = lambda_pair > 0.1         # minimal pairing to matter
        ok_pairing_bcs = lambda_pair < 1.5         # still within BCS (not Eliashberg)

        # ==========================================================
        # 3) Crystal-field & SOC structure
        # ==========================================================
        _Hsoc = _build_soc_cf_hamiltonian(
            self.lambda_soc, self.Delta_tetra, self.Delta_inplane)
        _ev = np.linalg.eigvalsh(_Hsoc)

        g7split = float(_ev[4] - _ev[2])          # Γ7a–Γ7b internal split
        spont_jt_risk = (g7split < 2 * self.kT)   # from residual Γ7 degeneracy

        # ==========================================================
        # 4) JT stability condition (ONLY alpha_K > 1 matters!)
        # ==========================================================
        jt_triggered = self.alpha_K > 1.0

        # ==========================================================
        # 5) Q-driven SC gap estimate: Δ_est ≈ g_JT · Q · λ_pair / Δ_CF
        # Physical interpretation: JT coupling g_JT·Q mixes Γ₆–Γ₇ with amplitude proportional to the pairing strength λ_pair and suppressed by the CF gap.
        # ==========================================================
        Q_est = 0.005  # Å representative small distortion
        Delta_est = self.g_JT * Q_est * lambda_pair / max(self.Delta_CF, 1e-9)
        ok_gap = Delta_est > self.kT

        # ===================== PRINT ==============================

        print("\n================ MODEL SUMMARY ================\n")

        print("Primary inputs:")
        print(f"  t0={self.t0:.3f} eV   u={self.u:.2f}")
        print(f"  λ_SOC={self.lambda_soc:.3f} eV   Δ_tet={self.Delta_tetra:.3f} eV")
        print(f"  g_JT={self.g_JT:.3f} eV/Å   α_K={self.alpha_K:.4f}")
        print(f"  η={self.eta:.3f}   δ₀={self.doping_0:.3f}")

        print("\nDerived quantities:")
        print(f"  U={self.U:.3f} eV   Δ_CF={self.Delta_CF:.4f} eV")
        print(f"  U_mf={self.U_mf:.4f} eV")
        print(f"  K_lattice = {self.K_lattice:.3f} eV/Å²")
        print(f"  Γ7 splitting = {g7split:.4f} eV   "
            f"[{'⚠ residual Γ7 deg.' if spont_jt_risk else '✓ Γ7 split > kT'}]")

        print("\nMagnetic state:")
        print(f"  {'✓' if ok_metal else '✗'} AFM metallic:"
            f"  h_afm={h_afm:.4f} < bw/2={bw2:.4f} eV")

        print("\nPairing diagnostics (at δ={:.3f}):".format(delta))
        print(f"  λ_pair (pairing eigenvalue) = {lambda_pair:.4f}  "
              f"[d-wave: {lambda_pair_d:.4f}, s-wave: {lambda_pair_s:.4f}]")
        print(f"    λ_pair = −N(0)·⟨ϕ_k | V_pair(k,k') | ϕ_k'⟩")
        print(f"    V_pair = g²/K + U²·χ_spin + V_mix  (JT + spin-fluct + Γ₆–Γ₇ mixing)")
        print(f"    {'✓' if ok_pairing_min else '✗'} Minimal pairing: λ_pair > 0.1")
        print(f"    {'✓' if ok_pairing_bcs else '✗'} BCS valid: λ_pair < 1.5")
        print(f"  V_JT  (g²/K)       = {V_JT_ch:.4f} eV  (SC-triggered JT phonon channel)")
        print(f"  V_spin (Moriya-RPA) = {V_spin_ch:.6f} eV  [U²·χ₀/(1−U·χ₀)²; U={self.U:.3f}, χ₀={chi0_spin_est:.4f}, 1−Uχ₀={stoner_denom:.4f}]")
        print(f"  V_mix (U_mf²/Δ_CF) = {V_mix_ch:.4f} eV  (Γ₆–Γ₇ coherent mixing)")
        print(f"  V_s = V_s_scale·V_JT = {V_eff_s:.4f} eV  (on-site Γ₆⊗Γ₇)")
        print(f"  V_d = V_d_scale·V_JT = {V_eff_d:.4f} eV  (inter-site d-wave)")

        K_spont = self.g_JT**2 / max(self.Delta_CF, 1e-9)
        print("\nJT mechanism:  α_K > 1  <=>  K = g²/(α_K*Δ_CF) < K_spont = g²/Δ_CF")
        print(f"  α_K = {self.alpha_K:.4f}  ->  K = {self.K_lattice:.4f} eV/Å²  "
              f"(K_spont(α_K=1) = {K_spont:.4f} eV/Å²,  margin α_K-1 = {self.alpha_K - 1:.4f})")
        print(f"  {'✓ SC-TRIGGERED: α_K > 1, spontaneous JT blocked' if jt_triggered else '✗ SPONTANEOUS JT RISK: alpha_K <= 1'}")

        print("\nEstimated SC gap (linear, Q-driven):")
        print(f"  Δ_est = {Delta_est*1000:.2f} meV  @ Q={Q_est} Å")
        print(f"  {'✓ Δ > kT' if ok_gap else '✗ Δ < kT'}  (kT={self.kT*1000:.2f} meV)")

        print("\n================================================\n")

def _build_soc_cf_hamiltonian(lambda_soc: float, Delta_tetra: float,
                               Delta_inplane: float = 0.0) -> np.ndarray:
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
                   WITHOUT removing Kramers degeneracy (still 2-fold each).
                   Prevents spontaneous JT from the 4-fold degenerate Γ₇ level:
                     Pure SOC alone leaves Γ₇ 4-fold → JT-active in normal state.
                     Δ_axial splits → Γ₇a + Γ₇b (Kramers pairs) by ~0.08 eV.
                     Δ_inplane provides further tuning of this splitting.
                   Default 0 keeps the original tetragonal-only behaviour.

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
    # S = 1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    I2, I3 = np.eye(2, dtype=complex), np.eye(3, dtype=complex)
    # SOC: +λ(L·S) in t2g effective space (Γ₆ is lowest for λ>0)
    H_SOC = lambda_soc * (
        np.kron(Lx, Sx) + np.kron(Ly, Sy) + np.kron(Lz, Sz)
    )
    # Crystal field: axial (D4h tetragonal) + in-plane (B2g)
    Lx_f = np.kron(Lx, I2)
    Ly_f = np.kron(Ly, I2)
    Lz_f = np.kron(Lz, I2)
    H_CF = (Delta_tetra * (Lz_f @ Lz_f)
            + Delta_inplane * (Lx_f @ Lx_f - Ly_f @ Ly_f))
    return H_SOC + H_CF

def _gamma_splitting(lambda_soc: float, Delta_tetra: float,
                     Delta_inplane: float = 0.0) -> float:
    """Γ₆–Γ₇a gap = E[2] − E[0] of the SOC+CF Hamiltonian.
    Delta_tetra=0 gives the pure cubic limit: Δ_CF ≈ (3/2)·λ_SOC.
    A non-zero Delta_tetra allows tuning the splitting independently of λ;
    negative values push the doublets toward quasi-degeneracy.
    Delta_inplane (B2g) splits the Γ₇ quartet without removing Kramers degeneracy."""
    evals = np.linalg.eigvalsh(
        _build_soc_cf_hamiltonian(lambda_soc, Delta_tetra, Delta_inplane))
    return float(evals[2] - evals[0])

# =============================================================================
# 1. SIMPSON INTEGRATION FOR K-SPACE
# =============================================================================

def _simpson_weights_2d(nx: int, ny: int) -> np.ndarray:
    """
    2D composite Simpson weights, normalized to sum = 1.
    
    Parameters:
        nx, ny: Grid points (must be odd)
    
    Returns:
        (nx × ny) weights array
    """
    if nx % 2 == 0 or ny % 2 == 0:
        raise ValueError(f"Simpson requires odd points: got {nx}, {ny}")
    
    def pattern_1d(n):
        p = np.ones(n)
        p[1:-1:2] = 4.0
        p[2:-1:2] = 2.0
        return p / 3.0
    
    wx = pattern_1d(nx)
    wy = pattern_1d(ny)
    weights_2d = np.outer(wx, wy)
    
    dk = 2 * np.pi / (nx - 1)
    weights_2d *= dk * dk
    weights_2d /= np.sum(weights_2d)
    return weights_2d
    
# =========================================================================
# 2. CLUSTER MEAN-FIELD SOLVER
# =========================================================================

class ClusterMF:
    """
    2-site cluster treatment of AFM quantum fluctuations.
    Sites A and B (AFM sublattices); exact diagonalization of O⊗O multipolar exchange
    with mean-field boundary coupling to external magnetization.

    Captures: quantum multipolar correlations, orbital mixing and spin-orbit coupling, thermal fluctuations.
    Does NOT capture: fermionic antisymmetrization, charge-transfer fluctuations.
    Valid when multipolar degrees of freedom dominate over charge fluctuations and system is not deep in the Mott insulator.
    """
    
    def __init__(self, params: ModelParams):
        self.p = params
        self.CLUSTER_SIZE = 2
        self.Z_BOUNDARY = params.Z - 1  # One link is within cluster, Z-1 are boundary
    
    def build_multipolar_operator(self, eta: float) -> np.ndarray:
        """O = (P₆ + η·P₇) ⊗ σ_z in basis [6↑, 6↓, 7↑, 7↓]. Returns 4×4 diagonal matrix."""
        # Orbital projectors in {6,7} basis
        # τz eigenvalues: Γ₆ → +1, Γ₇ → -1
        P6_diag = np.array([1.0, 1.0, 0.0, 0.0])  # Projects to 6↑, 6↓
        P7_diag = np.array([0.0, 0.0, 1.0, 1.0])  # Projects to 7↑, 7↓
        
        # Spin polarization σz
        sz_diag = np.array([1.0, -1.0, 1.0, -1.0])  # ↑=+1, ↓=-1
        
        # Multipolar operator: (P6 + η·P7) × σz
        O_diag = (P6_diag + eta * P7_diag) * sz_diag
        return np.diag(O_diag)
    
    def cluster_afm_exchange(self, J_eff: float, eta: float) -> np.ndarray:
        """Exact intra-cluster AFM exchange H = J · O_A ⊗ O_B. Dimension: 4×4 (site A) ⊗ 4×4 (site B) = 16×16 """
        O = self.build_multipolar_operator(eta)
        return J_eff * np.kron(O, O)
    
    def boundary_afm_field(self, J_eff: float, M_ext: float, eta: float, U_mf: float, g_J: float) -> np.ndarray:
        """
        Mean-field boundary coupling: H_boundary = Z_boundary·(g_J·U_mf/2 + J_eff)·M_ext·O.
        Matches BdG Weiss field (Stoner + Heisenberg, both renormalized by g_J under Gutzwiller).
        Returns 4×4 single-site operator.
        """
        O = self.build_multipolar_operator(eta)
        return self.Z_BOUNDARY * (g_J * U_mf / 2.0 + J_eff) * M_ext * O
    
    def build_cluster_hamiltonian(self, H_sp_A: np.ndarray, H_sp_B: np.ndarray,
                                 J_eff: float, M_ext: float, eta: float, U_mf: float, g_J: float) -> np.ndarray:
        """
        Cluster Hamiltonian in single-particle tensor space (16×16):
        H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)               [single-particle terms]
                  + J·O_A⊗O_B                                [intra-cluster multipolar exchange]
                  + H_boundary(A) ⊗ I + I ⊗ H_boundary(B)   [inter-cluster MF]
        Captures quantum orbital mixing, SOC-coupled multipolar exchange, thermal fluctuations.
        """
        I4 = np.eye(4, dtype=complex)
        
        # Single-particle terms: H(A) ⊗ I + I ⊗ H(B)
        H_cluster = np.kron(H_sp_A, I4) + np.kron(I4, H_sp_B)
        
        # Intra-cluster exact AFM exchange
        H_cluster += self.cluster_afm_exchange(J_eff, eta)
        
        # Boundary mean-field coupling
        H_bound = self.boundary_afm_field(J_eff, M_ext, eta, U_mf, g_J)
        H_cluster += np.kron(H_bound, I4)  # Boundary field on site A
        H_cluster += np.kron(I4, H_bound)  # Boundary field on site B
        return H_cluster
    
    def cluster_expectation(self, evals: np.ndarray, evecs: np.ndarray,
                            Operator: np.ndarray, temperature: float,
                            site_index: int = -1) -> float:
        """
        Thermal expectation ⟨O⟩ = Tr[ρO]/Tr[ρ], ρ = exp(−H_cluster/kT).
        evals, evecs: pre-computed from eigh(H_cluster) — diagonalized once per solve.
        Operator: 16×16, or 4×4 embedded via site_index (0→O⊗I₄, 1→I₄⊗O).
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
        Z = weights.sum()
        # Vectorised: ⟨n|O|n⟩ for each eigenstate, Boltzmann-weighted
        Oevecs = Operator @ evecs          # (16,16)
        diag   = np.einsum('ij,ij->j', evecs.conj(), Oevecs)
        return float(np.real((weights * diag).sum() / Z))

# =============================================================================
# 3. RENORMALIZED MEAN-FIELD THEORY (16x16 BdG) SOLVER WITH CLUSTER MF
# =============================================================================

class RMFT_Solver:
    """
    Self-consistent solver for the SC-activated JT model
    with Gutzwiller renormalization and proper 16x16 double unit cell structure.
    """
    def __init__(self, params: ModelParams):
        self.p = params
        self.cluster_mf = ClusterMF(params)
        
        # ── Dual k-grid setup ────────────────────────────────────────────────
        nk_even = params.nk                    # even  → chi0 requires an EVEN grid so commensurate q_AFM=(π,π) maps k→k+Q exactly onto another grid point: k_i + π ≡ k_{i + nk/2} (mod 2π, endpoint=False).
        nk_odd  = nk_even + 1                  # odd   → Simpson integration requires an ODD grid.

        # SCF / Simpson grid (odd)
        k_odd = np.linspace(-np.pi, np.pi, nk_odd, endpoint=False)
        KX_odd, KY_odd = np.meshgrid(k_odd, k_odd)
        self.k_points = np.column_stack((KX_odd.flatten(), KY_odd.flatten()))
        self.N_k      = len(self.k_points)
        weights_2d    = _simpson_weights_2d(nk_odd, nk_odd)
        self.k_weights = weights_2d.flatten()

        # chi0 grid (even): endpoint=False → [−π,π) with spacing 2π/nk_even.
        # k_i + (π,π) → k_{i + nk_even/2} (mod nk_even) exactly.
        k_even = np.linspace(-np.pi, np.pi, nk_even, endpoint=False)
        KX_ev, KY_ev = np.meshgrid(k_even, k_even)
        self.k_points_even   = np.column_stack((KX_ev.flatten(), KY_ev.flatten()))
        self.N_k_even        = len(self.k_points_even)
        # Uniform weights for the even grid (trapezoidal / rectangular rule)
        self.k_weights_even  = np.full(self.N_k_even, 1.0 / self.N_k_even)
        # Precompute AFM shift index: chi0_Q_idx[i] = index of k_i + Q_AFM in k_points_even
        half = nk_even // 2
        iy_all  = np.arange(self.N_k_even) // nk_even
        ix_all  = np.arange(self.N_k_even) %  nk_even
        self.chi0_Q_idx = ((iy_all + half) % nk_even) * nk_even + (ix_all + half) % nk_even

        print(f"Initialized RMFT solver: {self.N_k} k-points (SCF/Simpson, odd grid nk={nk_odd})")
        print(f"                         {self.N_k_even} k-points (χ₀ even grid nk={nk_even}, commensurate q_AFM)")
        
        # Γ₆/Γ₇ projectors and rank-2 quadrupolar τ_x in [6↑,6↓,7↑,7↓] basis.
        # Core algebraic objects of the SC-activated JT mechanism:
        #   P6 alone → τ_x off-diagonal → ⟨τ_x⟩=0 (B₁g JT forbidden in pure AFM)
        #   P6⊕w·P7  → τ_x acquires diagonal block → ⟨τ_x⟩≠0 (JT unlocked by SC)
        self.P6 = np.diag([1.0, 1.0, 0.0, 0.0])
        self.P7 = np.diag([0.0, 0.0, 1.0, 1.0])
        self.tau_x_op = np.zeros((4, 4), dtype=complex)
        self.tau_x_op[0, 2] = self.tau_x_op[2, 0] = 1.0  # 6↑ ↔ 7↑
        self.tau_x_op[1, 3] = self.tau_x_op[3, 1] = 1.0  # 6↓ ↔ 7↓

        # d-wave / B₁g form factor φ(k) = cos kx − cos ky (B₁g irrep of D₄h),
        # the same irrep as the JT distortion → self-closure condition of SC-activated JT.
        # For s-wave (d_wave=False) φ=1, which loses the irrep alignment.
        if params.d_wave:
            self.phi_k = (np.cos(self.k_points[:, 0] * params.a)
                          - np.cos(self.k_points[:, 1] * params.a))
        else:
            self.phi_k = np.ones(self.N_k)

        # Static Einstein phonon propagator D = 2/ω_JT (static limit).
        # Enters V(k,k') = g²·D·φ(k)·φ(k') shape only; magnitude uses g²/K.
        self._D_phonon: float = 2.0 / max(params.omega_JT, 1e-6)

        # Bare V_eff = g²/K (adiabatic); RPA factor applied per SCF iteration
        V_eff_bare = params.g_JT**2 / params.K_lattice
        
        print(f"Physical parameters: t₀={params.t0:.2f} eV, U={params.U:.2f} eV")
        print(f"Crystal field: Δ_CF={params.Delta_CF:.3f} eV")
        print(f"Electron-phonon: g_JT={params.g_JT:.3f} eV/Å, K={params.K_lattice:.2f} eV/Å²")
        tpd_sq_ratio = params.t_pd**2 / max(params.Delta_CT, 1e-9)
        zsa_ok = abs(tpd_sq_ratio - params.t0) / max(params.t0, 1e-9) < 0.30
        print(f"U_mf = {params.U_mf:.4f} eV  (= 0.5·Δ_CF, Stoner Weiss-field)")
        t0_dev_pct = abs(tpd_sq_ratio - params.t0) / max(params.t0, 1e-9) * 100.0
        print(f"ZSA check: t_pd²/Δ_CT = {tpd_sq_ratio:.4f} eV  "
              f"({'≈' if zsa_ok else '≠ '} t0={params.t0:.4f} eV, Δ={t0_dev_pct:.1f}%)  "
              f"{'✓' if zsa_ok else '⚠ (by construction = t0, check Delta_CT)'}")
        print(f"V_eff(adiabatic g²/K) = {V_eff_bare:.4f} eV  |  ω_JT = {params.omega_JT:.4f} eV  (D_phonon = 2/ω = {self._D_phonon:.2f} eV⁻¹)")
        print(f"λ_eff = V_eff·N(0) ≈ {V_eff_bare/(np.pi*params.t0):.3f}  (BCS valid if 0.1–1.5)")
        print(f"Phonon propagator: D = 2/ω_JT = {self._D_phonon:.2f} eV⁻¹  (shape only, not magnitude)")
        print(f"Gap symmetry: {'d-wave B₁g φ(k)=cos kx−cos ky' if params.d_wave else 's-wave φ(k)=1'}")
        print(f"Method: Variational Free Energy Minimization + RPA spin-fluctuation enhancement")

        # VectorizedBdG is instantiated lazily on first use via _get_vbdg(). Declared here so copy.copy() preserves the attribute.
        self._vbdg: Optional['VectorizedBdG'] = None
        # Transient per-iteration BdG cache shared between observables / gap eq / dF/dM.
        # Set at the top of each SCF iteration, cleared after compute_dF_dM_and_d2F.
        self._scf_bdg_cache: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Lazy VectorizedBdG accessor
    # ------------------------------------------------------------------
    def _get_vbdg(self) -> 'VectorizedBdG':
        """Return (and cache) the VectorizedBdG for this solver instance."""
        if self._vbdg is None:
            self._vbdg = VectorizedBdG(self)
        return self._vbdg

    # =========================================================================
    # 3.1 GUTZWILLER RENORMALIZATION FACTORS
    # =========================================================================
    
    def get_gutzwiller_factors(self, delta: float) -> Tuple[float, float, float, float]:
        """
        Gutzwiller renormalization factors for doping δ = 1 - n.

        g_t       = 2δ/(1+δ)  — kinetic energy; → 0 at half-filling (Mott insulator)
        g_J       = 4/(1+δ)²  — exchange enhancement; → 4 at half-filling (J = 4t²/U)
        g_Delta_s = sqrt(g_{t,Γ6} · g_{t,Γ7})  — on-site inter-orbital Γ₆⊗Γ₇ singlet.
                                  Multi-orbital Gutzwiller: the anomalous amplitude of a
                                  Cooper pair straddling two orbitals renormalizes as the
                                  geometric mean of the two single-orbital factors.
                                  g_{t,Γ6} = g_t = 2δ/(1+δ) (partially occupied band).
                                  g_{t,Γ7} ≈ 1   because Δ_CF ≫ kT → Γ7 nearly empty,
                                  so its Gutzwiller factor approaches the free-electron
                                  limit (no double-occupancy cost in an empty orbital).
                                  Therefore: g_Delta_s = sqrt(g_t · 1) = sqrt(g_t).
                                  Intermediate suppression: g_t < g_Delta_s < 1.
        g_Delta_d = g_J        — inter-site d-wave B₁g renormalization. Superexchange-
                                  mediated: scales with the same vertex as J. Strongest
                                  at half-filling, vanishing at large δ.

        Half-filling floor δ_min = 1e-6 to avoid singularity.
        """
        abs_delta  = max(abs(delta), 1e-6)
        g_t        = (2.0 * abs_delta) / (1.0 + abs_delta)
        g_J        = 4.0 / ((1.0 + abs_delta) ** 2)
        g_Delta_s  = np.sqrt(g_t)
        g_Delta_d  = g_J
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
    
    def effective_superexchange(self, Q: float, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> float:
        """
        J(Q, δ) = g_J · 4·⟨t²(Q)⟩/U · f(δ),  where f(δ) = δ/(δ+δ₀).

        f(δ) → 0 at half-filling (suppresses unphysical g_J→4 divergence where
        coherent spectral weight vanishes in the Gutzwiller approximation).
        tx_bare, ty_bare: raw Q-dependent hoppings (not Gutzwiller-renormalized).
        """
        abs_doping = max(abs(doping), 1e-6)
        f_doping = abs_doping / (abs_doping + self.p.doping_0)
        t_sq_avg = 0.5 * (tx_bare**2 + ty_bare**2)
        return g_J * 4.0 * t_sq_avg / self.p.U * f_doping
    
    def fermi_function(self, E: np.ndarray) -> np.ndarray:
        """The Fermi–Dirac distribution f(E) with μ is already included in the Hamiltonian; clipped to [-100, 100] to prevent overflow."""
        arg = E / self.p.kT
        arg = np.clip(arg, -100, 100)
        return 1.0 / (1.0 + np.exp(arg))
    
    # =========================================================================
    # 3.2  IRREP PROJECTION & MULTIPOLAR ALGEBRA  (symmetry selection rules)
    # =========================================================================

    def build_irrep_selection_projector(self, Delta: complex) -> np.ndarray:
        """
        4×4 projector encoding the SC-activated symmetry lifting of the B₁g JT mode.

        Pure AFM (Δ=0): P_eff = P6 → τ_x strictly off-diagonal → ⟨τ_x⟩=0 (JT blocked)
        SC-condensed (Δ≠0): P_eff = P6 + w·P7, w = |Δ|/Δ_CF
            τ_x acquires diagonal block → ⟨τ_x⟩≠0 → JT unlocked
        w = min(|Δ|/Δ_CF, 1) interpolates smoothly between the two limits.
        """
        Delta_CF = max(self.p.Delta_CF, 1e-9)
        w = float(np.clip(abs(Delta) / Delta_CF, 0.0, 1.0))
        P_eff = self.P6 + w * self.P7
        return P_eff

    def compute_rank2_multipole_expectation(self, Delta: complex,
                                            tau_x_bdg: float) -> Dict:
        """
        Measure the algebraic lifting of the rank-2 B₁g multipole ⟨τ_x⟩ by SC.

        In the AFM-only state: P_eff = P6, ⟨τ_x⟩_P6 = 0 (τ_x off-diagonal).
        In the SC state: w = |Δ|/Δ_CF mixes in Γ₇, ⟨τ_x⟩_eff = w·|τ_x_bdg|.
        Selection ratio R = ⟨τ_x⟩_eff / 1: R≈0 → barrier intact; R→1 → JT allowed.

        Returns dict: 'w', 'selection_ratio', 'jt_algebraically_allowed' (R>0.05),
                      'tau_x_projected', 'tau_x_free_max'
        """
        Delta_CF = max(self.p.Delta_CF, 1e-9)
        w = float(np.clip(abs(Delta) / Delta_CF, 0.0, 1.0))

        tau_x_free_max = 1.0
        # ⟨τx⟩_eff ≈ w · BdG k-space average of ⟨τx⟩  (suppressed linearly by the mixing weight)
        tau_x_projected = w * abs(tau_x_bdg)
        selection_ratio = tau_x_projected / max(tau_x_free_max, 1e-9)

        return {
            'w':                        w,
            'selection_ratio':          selection_ratio,
            'jt_algebraically_allowed': selection_ratio > 0.05,
            'tau_x_projected':          tau_x_projected,
            'tau_x_free_max':           tau_x_free_max,
        }

    def compute_static_chi0_afm(self, M: float, Q: float,
                                 Delta_s: complex, Delta_d: complex, target_doping: float,
                                 mu: float, tx: float, ty: float, g_J: float) -> Dict:
        """
        Static transverse spin susceptibility χ₀(q_AFM) at q = (π,π).

        Uses the even k-grid so k + Q_AFM maps exactly to another grid point via precomputed self.chi0_Q_idx (no interpolation/aliasing).

        Formula: χ₀ = Σ_{k,n,m} |⟨ψ_n(k)|Ŝ_z|ψ_m(k+Q)⟩|² · (f_n − f_m) / (E_m − E_n)
        Ŝ_z in [6↑,6↓,7↑,7↓] = diag(+1,−1,+η,−η) on sublattice A (staggered in BdG).

        Returns:
            dict with keys:
              'chi0'        : float, static susceptibility (eV⁻¹)
              'U_eff_chi'   : float, renormalised magnetic coupling used in Stoner denominator (eV),  NOT the bare Hubbard U. This keeps U_eff_chi · χ₀ ~ O(1) within the ordered AFM phase
              'stoner_denom': float, 1 - U_eff_chi · chi0
              'afm_unstable': bool, True if stoner_denom ≤ 0 (AFM QCP crossed, magnetically unstable)
              'chi_tau'     : float, multipolar susceptibility χ_τx ~ N(0)/(1 + α·M²). Non-zero only when U/t < (U/t)_c ≈ 2.2–2.5 (2D square lattice).
                              Physically: M² < 1 − (U/t)_c/(U/t) required for JT activation. If chi_tau ≈ 0, the JT channel is suppressed by AFM order.
        """
        # Spin operator in 4-orbital BdG Nambu basis (diagonal):
        # Particle A: S_z = diag(+1,-1,+η,-η); B: staggered -1; holes: p-h conjugate signs.
        sz_orb   = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
        sz_diag  = np.concatenate([ sz_orb,   # particle A
                                    -sz_orb,   # particle B (staggered)
                                    -sz_orb,   # hole A (p-h conjugate)
                                     sz_orb])  # hole B

        # Vectorised: diagonalise the even grid ONCE, then reindex for k+Q.
        # chi0_Q_idx is a permutation of [0..N_even) so E(k+Q) and V(k+Q) are
        # simply row-permutations of E(k) and V(k) — no second LAPACK call needed.
        vbdg = self._get_vbdg()
        E_k_all, V_k_all = vbdg.diag_kpts(self.k_points_even, M, Q, Delta_s, Delta_d,
                                            target_doping, mu, tx, ty, g_J)  # (N,16), (N,16,16)

        E_kQ_all = E_k_all[self.chi0_Q_idx]   # (N,16)  — free permutation, no LAPACK
        V_kQ_all = V_k_all[self.chi0_Q_idx]   # (N,16,16)

        f_k_all  = self.fermi_function(E_k_all)    # (N, 16)
        f_kQ_all = self.fermi_function(E_kQ_all)   # (N, 16)

        # Vectorised chi0: M_mat[k,n,m] = ⟨ψ_n(k)|Sz|ψ_m(k+Q)⟩
        # sz_diag is the diagonal of Sz_bdg → no matmul needed.
        SzV_kQ  = sz_diag[None, :, None] * V_kQ_all           # (N,16,16)
        M_mat   = np.einsum('kni,kim->knm', V_k_all.conj(), SzV_kQ)  # (N,16,16)
        M2      = np.abs(M_mat)**2  # (N,16,16)

        df = f_k_all[:, :, None] - f_kQ_all[:, None, :]   # (N,16,16)
        dE = E_kQ_all[:, None, :] - E_k_all[:, :, None]   # (N,16,16)

        # Mask degenerate/near-zero pairs; safe division via np.where avoids NaN
        mask    = (np.abs(df) > 1e-12) & (np.abs(dE) > 1e-6)
        safe_dE = np.where(mask, dE, 1.0)
        ratio   = np.where(mask, self.k_weights_even[:, None, None] * M2 * df / safe_dE, 0.0)
        chi0 = float(ratio.sum())

        # U_eff_chi = g_J·J_eff (Gutzwiller-renormalised exchange, not bare U) → U_eff_chi·chi0 ~ O(1)
        abs_delta  = max(abs(target_doping), 1e-6)
        f_d        = abs_delta / (abs_delta + self.p.doping_0)
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t2         = 0.5 * (tx_bare**2 + ty_bare**2)
        J_eff_now  = g_J * 4.0 * t2 / self.p.U * f_d
        U_eff_chi  = g_J * J_eff_now    # ~ 0.05–0.3 eV

        stoner_denom = 1.0 - U_eff_chi * chi0
        afm_unstable = stoner_denom <= 0.0

        return {
            'chi0':         chi0,
            'U_eff_chi':    U_eff_chi,
            'stoner_denom': stoner_denom,
            'afm_unstable': afm_unstable,
        }

    def _compute_chi_tau(self, M: float, Q: float, target_doping: float) -> Dict:
        """
        Analytic multipolar susceptibility χ_τx and related diagnostics.

        χ_τx ~ N(0) / (1 + α·M²), where α encodes AFM suppression:
          α = max(U/t_eff / (U/t)_c − 1, 0),  (U/t)_c ≈ 2.35 (2D square lattice).

        This is FULLY ANALYTIC — no BdG diagonalisation required.
        Called every SCF iteration (cheap); the expensive chi0(q_AFM) tensor is
        updated only every ~10 iterations for the free-energy rpa_factor.

        Returns dict: 'chi_tau', 'N0', 'Ut_ratio', 'alpha_M'
        """
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t_eff_avg = np.sqrt(0.5 * (tx_bare**2 + ty_bare**2))
        N0        = 1.0 / (np.pi * max(t_eff_avg, 1e-6))
        Ut_ratio  = self.p.U / max(t_eff_avg, 1e-6)
        alpha_M   = max(Ut_ratio / 2.35 - 1.0, 0.0)
        chi_tau   = N0 / (1.0 + alpha_M * M**2)
        return {'chi_tau': chi_tau, 'N0': N0, 'Ut_ratio': Ut_ratio, 'alpha_M': alpha_M}

    def rpa_stoner_factor(self, chi0_result: Dict) -> float:
        """
        RPA Stoner enhancement 1 / (1 − U_eff_chi · χ₀).

        Uses renormalised U_eff_chi = g_J·J_eff (not bare Hubbard U), so the denominator
        stays positive and O(1) in the AFM-ordered phase (10–50% enhancement).
        If stoner_denom ≤ 0 (QCP crossed): returns 1.0 (no enhancement).
        rpa_cutoff clamps denominator from below → max enhancement = 1/rpa_cutoff.
        """
        sd = chi0_result['stoner_denom']
        if sd <= 0.0:
            return 1.0  # AFM instability: suppress enhancement
        return 1.0 / max(sd, self.p.rpa_cutoff)

    # =========================================================================
    # 3.2b  ORBITAL χ₀ TENSOR, RPA VERTEX, LINEARISED GAP EQUATION
    # =========================================================================

    def _u_eff_and_interaction_matrix(self, Q: float, g_J: float,
                                       target_doping: float) -> Tuple[float, np.ndarray]:
        """
        Gutzwiller-renormalised interaction matrices for the RPA pairing vertex.

        Two distinct couplings are returned:

        U_eff  (for AFM diagnostics):
          U_eff = g_J · J_eff = g_J · (4t²/U) · f(δ)  ~ 0.05–0.30 eV
          Used by rpa_stoner_factor() to characterise AFM susceptibility.

        U_mat  (for pairing vertex):
          U_mat = g_J · U · I₄
          The bare local Coulomb U is the correct vertex in the Moriya-RPA spin-fluctuation pairing formula:
            V_spin(q) = U² · χ₀(q) / (1 − U·χ₀(q))²
          Near AFM criticality U·χ₀(q_AFM) → 1, giving the correct divergence.
          Using J_eff here would suppress this divergence by a factor ~(U/J_eff)².
          g_J accounts for reduced quasiparticle weight (Gutzwiller projection).

        Returns
        -------
        U_eff  : float    AFM-diagnostic coupling (eV), g_J·J_eff
        U_mat  : (4,4)   pairing-vertex interaction matrix, g_J·U·I₄
        """
        abs_delta = max(abs(target_doping), 1e-6)
        f_d       = abs_delta / (abs_delta + self.p.doping_0)
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t2        = 0.5 * (tx_bare**2 + ty_bare**2)
        J_eff_now = g_J * 4.0 * t2 / self.p.U * f_d

        # U_eff: AFM/Stoner diagnostic scale (g_J·J_eff, ~ 0.05–0.30 eV)
        # U_pair: RPA pairing vertex = g_J·U (bare U, not J_eff — preserves Stoner divergence near QCP)
        U_eff  = g_J * J_eff_now
        U_pair = g_J * self.p.U

        # 4×4 interaction in [6↑,6↓,7↑,7↓]: U_mat = U_pair·I₄
        # Enters χ_RPA = (I − U_mat @ χ₀)⁻¹ @ χ₀ in compute_gap_eq_vectorized.
        U_mat = U_pair * np.eye(4, dtype=complex)
        return U_eff, U_mat

    def compute_chi0_tensor(self, q: np.ndarray,
                             M: float, Q: float,
                             Delta_s: complex, Delta_d: complex,
                             target_doping: float,
                             mu: float, tx: float, ty: float,
                             g_J: float,
                             _E_k_cache: tuple = None) -> np.ndarray:
        """
        Orbital bare susceptibility tensor chi0^{ab}(q) in [6↑,6↓,7↑,7↓] basis.

        chi0^{ab}(q) = -(1/N) Σ_{k,n,m} V*_{an}(k) V_{am}(k+q) V*_{bm}(k+q) V_{bn}(k)
                       · (f_n(k) - f_m(k+q)) / (E_m(k+q) - E_n(k))

        Sign convention: χ₀ > 0 for stable metal → RPA: χ_RPA = (I − U·χ₀)⁻¹·χ₀.
        Full BdG structure: particle/hole sectors for both sublattices A and B,
        including cross-sublattice pairs (orbital coherence at Q≠0).

        Returns
        -------
        chi0 : (4, 4) complex matrix  (eV⁻¹), positive for stable paramagnet.
        """
        chi0 = np.zeros((4, 4), dtype=complex)

        # BdG sector slices and particle-hole signs.
        # Sublattice staggering is encoded in eigenvectors via H_A/H_B.
        # Cross-sublattice pairs (A↔B) capture inter-sublattice orbital coherence at Q≠0.
        SECTORS = [
            (slice(0,  4),  +1.0),   # particle, sublattice A
            (slice(4,  8),  +1.0),   # particle, sublattice B
            (slice(8,  12), -1.0),   # hole,     sublattice A
            (slice(12, 16), -1.0),   # hole,     sublattice B
        ]
        SECTOR_PAIRS = [
            (SECTORS[0], SECTORS[0]),  # A-part ↔ A-part
            (SECTORS[1], SECTORS[1]),  # B-part ↔ B-part
            (SECTORS[2], SECTORS[2]),  # A-hole ↔ A-hole
            (SECTORS[3], SECTORS[3]),  # B-hole ↔ B-hole
            (SECTORS[0], SECTORS[1]),  # A-part ↔ B-part (cross)
            (SECTORS[1], SECTORS[0]),
            (SECTORS[2], SECTORS[3]),  # A-hole ↔ B-hole (cross)
            (SECTORS[3], SECTORS[2]),
        ]

        vbdg   = self._get_vbdg()
        kpts_k = self.k_points_even                                     # (N, 2)

        # k-grid diagonalisation: reuse cache when available (q-independent)
        if _E_k_cache is not None:
            E_k_all, V_k_all = _E_k_cache
        else:
            E_k_all, V_k_all = vbdg.diag_kpts(kpts_k, M, Q, Delta_s, Delta_d,
                                                target_doping, mu, tx, ty, g_J)

        # k+q grid (q-dependent: always recompute)
        kpts_kq  = (kpts_k + q[None, :] + np.pi) % (2.0 * np.pi) - np.pi
        E_kQ_all, V_kQ_all = vbdg.diag_kpts(kpts_kq, M, Q, Delta_s, Delta_d,
                                              target_doping, mu, tx, ty, g_J)

        f_k_all  = self.fermi_function(E_k_all)
        f_kQ_all = self.fermi_function(E_kQ_all)

        df      = f_k_all[:, :, None] - f_kQ_all[:, None, :]
        dE      = E_kQ_all[:, None, :] - E_k_all[:, :, None]
        mask    = (np.abs(df) > 1e-12) & (np.abs(dE) > 1e-6)
        safe_dE = np.where(mask, dE, 1.0)
        factor  = np.where(mask, self.k_weights_even[:, None, None] * df / safe_dE, 0.0)

        N = len(kpts_k)
        CHUNK = 128
        for (sl_k, sgn_k), (sl_kQ, sgn_kQ) in SECTOR_PAIRS:
            Vk_s  = V_k_all[:,  sl_k,  :]   # (N, 4, 16)
            VkQ_s = V_kQ_all[:, sl_kQ, :]   # (N, 4, 16)
            net_sgn = sgn_k * sgn_kQ
            for k0 in range(0, N, CHUNK):
                k1    = min(k0 + CHUNK, N)
                fac_c = factor[k0:k1]        # (C,16,16)
                Vk_c  = Vk_s[k0:k1]         # (C,4,16)
                VkQ_c = VkQ_s[k0:k1]        # (C,4,16)
                L  = Vk_c.conj()[:, :, None, :] * Vk_c[:, None, :, :]    # (C,4,4,16)
                R  = VkQ_c[:, :, None, :] * VkQ_c.conj()[:, None, :, :]  # (C,4,4,16)
                FR = np.einsum('cnm,cabm->cabn', fac_c, R)                # (C,4,4,16)
                chi0 += net_sgn * np.einsum('cabn,cabn->ab', L, FR)
        return chi0

    def _get_fermi_surface_sample(self, M: float, Q: float,
                                   Delta_s: complex, Delta_d: complex,
                                   target_doping: float,
                                   mu: float, tx: float, ty: float,
                                   g_J: float,
                                   n_fs: int = 48) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample k-points near the Fermi surface and estimate Fermi velocities.

        A k-point is 'near the FS' if at least one quasiparticle band satisfies
        |E_n(k)| < 3kT.  The Fermi velocity proxy is the minimum positive
        quasiparticle energy (a monotone proxy that is zero at a node).

        Returns
        -------
        fs_pts : (N, 2)  k-points on or near the Fermi surface
        vF     : (N,)    |v_F| proxy (eV); proportional to DOS weight
        """
        # Vectorised: diagonalise all k at once, then filter near-FS points
        vbdg   = self._get_vbdg()
        ev_all, _ = vbdg.diag_all_k(M, Q, Delta_s, Delta_d,
                                     target_doping, mu, tx, ty, g_J)  # (N_k, 16)

        # Near-FS mask: at least one quasiparticle band within 3kT of zero
        near_fs = np.any(np.abs(ev_all) < 3.0 * self.p.kT, axis=1)  # (N_k,)

        # Fermi velocity proxy: minimum positive eigenvalue per k
        ev_pos = np.where(ev_all > 0, ev_all, np.inf)
        vF_all = ev_pos.min(axis=1)
        vF_all = np.where(np.isinf(vF_all), self.p.kT, vF_all)
        vF_all = np.maximum(vF_all, 1e-4)
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

    def build_gap_equation_kernel(self, fermi_pts: np.ndarray, vF: np.ndarray,
                                  M: float, Q: float,
                                  Delta_s: complex, Delta_d: complex,
                                  target_doping: float,
                                  mu: float, tx: float, ty: float,
                                  g_J: float) -> np.ndarray:
        """
        Pairing kernel Gamma_{ij} for the linearised gap equation (DIAGNOSTIC only).

            Gamma_{ij} = sqrt(v_F(i)) * V(k_i - k_j) * sqrt(v_F(j))

        where V(q) is the full Gamma6 x Gamma7-projected RPA vertex.
        Vectorised implementation exploits three facts to reduce BdG diagonalisation count:

        1. Symmetry: Gamma is real-symmetric -> compute only upper triangle,
           N(N+1)/2 pairs instead of N^2.
        2. q-deduplication: many (i,j) pairs share the same q = k_i - k_j
           (mod 2pi).  V(q) is computed only for unique q-vectors.
        3. U_mat precomputed once: the RPA denominator (I - U*chi0)^-1 uses
           U_mat that depends only on (Q, g_J, doping), not on q.

        Returns
        -------
        Gamma : (N, N) real symmetric matrix
        """
        N    = len(fermi_pts)
        svF  = np.sqrt(np.abs(vF))   # (N,) -- sqrt(v_F) weights

        # Precompute U_mat once (independent of q)
        _, U_mat = self._u_eff_and_interaction_matrix(Q, g_J, target_doping)
        I4       = np.eye(4, dtype=complex)

        # Build upper-triangle (i,j) list and corresponding q-vectors
        ij_list = []
        q_list  = []
        for i in range(N):
            for j in range(i, N):
                q_raw = fermi_pts[i] - fermi_pts[j]
                q_w   = (q_raw + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-pi, pi)
                ij_list.append((i, j))
                q_list.append(q_w)

        q_arr = np.array(q_list)     # (M, 2), M = N(N+1)/2

        # Deduplicate q-vectors (round to 5 decimal places as hash key)
        q_keys = [f"{r[0]:.5f},{r[1]:.5f}" for r in np.round(q_arr, 5)]
        unique_keys, inv_idx = np.unique(q_keys, return_inverse=True)

        # Representative q-vector for each unique key
        unique_q_map = {}
        for flat_idx, key in enumerate(q_keys):
            if key not in unique_q_map:
                unique_q_map[key] = q_arr[flat_idx]
        unique_q_vecs = np.array([unique_q_map[k] for k in unique_keys])  # (U, 2)

        # Diagonalise the k-grid ONCE (q-independent) and cache for all unique q.
        # This eliminates one full BdG diag per unique-q call inside compute_chi0_tensor.
        vbdg       = self._get_vbdg()
        E_k_cache  = vbdg.diag_kpts(self.k_points_even, M, Q, Delta_s, Delta_d,
                                     target_doping, mu, tx, ty, g_J)   # (N_even,16),(N_even,16,16)

        # Compute V(q) for every unique q, reusing the k-grid cache.
        V_unique = np.empty(len(unique_keys), dtype=float)
        for u_idx, q_u in enumerate(unique_q_vecs):
            chi0_mat = self.compute_chi0_tensor(
                q_u, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                _E_k_cache=E_k_cache)
            denom        = I4 - U_mat @ chi0_mat
            Ud, sd, Vhd  = np.linalg.svd(denom)
            sd_reg       = np.maximum(sd, self.p.rpa_cutoff)
            denom_inv    = (Vhd.conj().T * (1.0 / sd_reg)) @ Ud.conj().T
            chi_rpa      = denom_inv @ chi0_mat
            # Gamma6 x Gamma7 projection: off-diagonal block [0:2, 2:4]
            # V_spin + V_charge = 1.5*V_67 - 0.5*V_67 = V_67
            V_unique[u_idx] = float(np.real(np.trace(chi_rpa[0:2, 2:4])))

        # Reconstruct full symmetric kernel from upper triangle
        Gamma = np.zeros((N, N))
        for flat_idx, (i, j) in enumerate(ij_list):
            val = svF[i] * V_unique[inv_idx[flat_idx]] * svF[j]
            Gamma[i, j] = val
            Gamma[j, i] = val
        return Gamma

    def solve_linearized_gap_equation(self,
                                       M: float, Q: float,
                                       Delta_s: complex, Delta_d: complex,
                                       target_doping: float,
                                       mu: float, tx: float, ty: float,
                                       g_J: float) -> Dict:
        """
        Linearised gap equation as a PURE eigenvalue problem (diagnostic).

        Role in the code
        ----------------
        This method does NOT iterate Δ.  It answers:
          "Given the current BdG state (M, Q, Δ, μ), is there a pairing
           instability, and if so, what is its symmetry?"

        Algorithm
        ---------
        1. Sample Fermi surface → {k_i, v_F(i)}
        2. Build kernel Γ_{ij} = √v_F(i) · V(k_i−k_j) · √v_F(j)
           where V(q) is the Γ₆⊗Γ₇-projected RPA vertex.
        3. Diagonalise Γ: eigenvalues λ_α, eigenvectors Δ_α(k_i).
        4. λ_max > 1  →  pairing instability (Tc criterion in linearised BCS).
        5. Project dominant eigenvector onto φ_s=1 (A1g) and φ_d=coskx−cosky (B1g)
           to identify gap symmetry.

        Returns
        -------
        dict with:
          'lambda_max'     : float   – dominant eigenvalue; >1 means instability
          'gap_vector'     : ndarray – gap on FS at dominant channel
          'fs_pts'         : ndarray – k-points used
          'gap_symmetry'   : str     – 'B1g (d-wave)' or 'A1g (s-wave)'
          'all_eigenvalues': ndarray – full spectrum of Γ
        """
        fs_pts, vF = self._get_fermi_surface_sample(
            M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)

        if len(fs_pts) < 4:
            return {'lambda_max': 0.0, 'gap_vector': np.array([0.0]),
                    'fs_pts': fs_pts,
                    'gap_symmetry': 'undetermined (too few FS points)',
                    'all_eigenvalues': np.array([0.0])}

        Gamma    = self.build_gap_equation_kernel(fs_pts, vF, M, Q, Delta_s, Delta_d,
                                                   target_doping, mu, tx, ty, g_J)
        # eigh guarantees real eigenvalues for symmetric real matrix
        eigvals, eigvecs = np.linalg.eigh(Gamma)
        idx_max          = np.argmax(eigvals)
        lambda_max       = float(eigvals[idx_max])
        gap_vector       = eigvecs[:, idx_max]

        # Symmetry: overlap with B₁g and A₁g form factors on the FS sample
        phi_s = np.ones(len(fs_pts))
        phi_d = np.cos(fs_pts[:, 0] * self.p.a) - np.cos(fs_pts[:, 1] * self.p.a)
        w_s   = abs(gap_vector @ phi_s) / (np.linalg.norm(phi_s) + 1e-12)
        w_d   = abs(gap_vector @ phi_d) / (np.linalg.norm(phi_d) + 1e-12)
        gap_symmetry = 'B1g (d-wave)' if w_d > w_s else 'A1g (s-wave)'

        return {'lambda_max':      lambda_max,
                'gap_vector':      gap_vector,
                'fs_pts':          fs_pts,
                'gap_symmetry':    gap_symmetry,
                'all_eigenvalues': eigvals}

    # =========================================================================
    # 3.3 BdG HAMILTONIAN CONSTRUCTION
    # =========================================================================
    
    def compute_orbital_coherence_from_pairs(self,
                                             M: float, Q: float,
                                             Delta_s: complex, Delta_d: complex,
                                             target_doping: float,
                                             mu: float, tx: float, ty: float,
                                             g_J: float) -> float:
        """
        Orbital coherence ⟨τ_x⟩_anom from the anomalous Green function F(k).

        Correct SC-activated JT feedback:
            Δ → F(k) = ⟨c_{k,6↑}c_{−k,7↓} − c_{k,6↓}c_{−k,7↑}⟩
              → ⟨τ_x⟩_anom = Σ_k (1−2f_n) Re[u*_6(k)v_7(k) + h.c.]
              → Q = (g_JT/K)·⟨τ_x⟩_anom → modifies tx(Q), ty(Q) self-consistently.

        When Δ=0: F(k)=0 → ⟨τ_x⟩_anom=0 → Q=0 (selection rule intact).
        When Δ≠0: F(k)≠0 → ⟨τ_x⟩_anom≠0 → Q≠0 (JT unlocked).
        An explicit off-diagonal H[Γ₆,Γ₇]=f(Δ) would introduce a spurious
        spontaneous-JT channel and double-count the BdG back-action; hence
        ⟨τ_x⟩_anom is computed as a BdG observable, not a Hamiltonian term.

        Returns ⟨τ_x⟩_anom (diagnostic supplement; anomalous part of orbital coherence, exactly zero when Δ=0. SCF uses total ⟨τ_x⟩ from BdG).
        """
        # Vectorised: reuse diag_kpts on self.k_points
        vbdg = self._get_vbdg()
        ev_all, ec_all = vbdg.diag_all_k(M, Q, Delta_s, Delta_d,
                                          target_doping, mu, tx, ty, g_J)
        f_n_all  = self.fermi_function(ev_all)       # (N,16)
        omf_all  = 1.0 - 2.0 * f_n_all              # (N,16)

        # Sublattice amplitudes: (N,4,16)
        uA = ec_all[:, 0:4,   :]
        vA = ec_all[:, 8:12,  :]
        uB = ec_all[:, 4:8,   :]
        vB = ec_all[:, 12:16, :]

        # Anomalous orbital coherence per sublattice and state:
        # Re[u_6↑ * v_7↑* + u_7↑ * v_6↑* + u_6↓ * v_7↓* + u_7↓ * v_6↓*]
        anom_A = (np.real(uA[:, 0, :] * np.conj(vA[:, 2, :]))
                + np.real(uA[:, 2, :] * np.conj(vA[:, 0, :]))
                + np.real(uA[:, 1, :] * np.conj(vA[:, 3, :]))
                + np.real(uA[:, 3, :] * np.conj(vA[:, 1, :])))  # (N,16)
        anom_B = (np.real(uB[:, 0, :] * np.conj(vB[:, 2, :]))
                + np.real(uB[:, 2, :] * np.conj(vB[:, 0, :]))
                + np.real(uB[:, 1, :] * np.conj(vB[:, 3, :]))
                + np.real(uB[:, 3, :] * np.conj(vB[:, 1, :])))  # (N,16)

        # k-weighted sum: Σ_{k,n} w[k] * (1-2f[k,n]) * (anom_A + anom_B)[k,n] / 4
        tau_x_anom = float(
            np.einsum('k,kn,kn->', self.k_weights, omf_all, (anom_A + anom_B)) / 4.0
        )
        return float(tau_x_anom)

    def build_local_hamiltonian_for_bdg(self, sign_M: float, M: float, Q: float,
                                        mu: float, g_J: float,
                                        target_doping: float) -> np.ndarray:
        """
        Local 4×4 BdG Hamiltonian for one sublattice, basis [6↑, 6↓, 7↑, 7↓].
        sign_M = ±1 for sublattices A/B (staggered AFM).

        Terms:
          1. Chemical potential −μ
          2. Crystal field splitting Δ_CF on Γ₇
          3. Stoner-Heisenberg Weiss field h_AFM (spin-diagonal, dipolar only)
          4. JT distortion:  H_JT = g_JT · Q · τ_x # g_JT is the bare Holstein coupling

        SC-activated JT causal chain: Δ≠0 → F(k)≠0 → ⟨τ_x⟩≠0 → Q≠0 → H_JT≠0.
        No explicit Σ_anom=f(Δ) off-diagonal term: adding one would restore
        spontaneous JT in disguise and double-count the BdG pairing back-action.

        Weiss field: h = g_J·(U_mf/2 + Z·2t²/U)·M/2 (both HF and superexchange
        renormalized by g_J under Gutzwiller projection).
        """
        H = np.zeros((4, 4), dtype=complex)

        # 1. Chemical potential
        np.fill_diagonal(H, -mu)

        # 2. Crystal field splitting Δ_CF on Γ₇
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓

        # 3. Mean-field AFM Weiss field: Stoner-Heisenberg combination
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t_sq_avg = 0.5 * (tx_bare**2 + ty_bare**2)
        abs_delta = max(abs(target_doping), 1e-6)
        f_delta   = abs_delta / (abs_delta + self.p.doping_0)
        h_afm_6 = g_J * f_delta * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) * sign_M * M / 2.0
        h_afm_7 = self.p.eta * h_afm_6

        H[0, 0] -= h_afm_6  # 6↑
        H[1, 1] += h_afm_6  # 6↓
        H[2, 2] -= h_afm_7  # 7↑
        H[3, 3] += h_afm_7  # 7↓

        # 4. JT distortion: H_JT = g_JT · Q · τ_x  (orbital mixing, spin-conserving)
        #    Q=0 in normal AFM state  →  no orbital mixing  →  τ_x forbidden (correct).
        #    Q≠0 only when SC condensate has generated ⟨τ_x⟩_anom ≠ 0 (SC-triggered).
        h_jt = self.p.g_JT * Q
        H[0, 2] = h_jt   # 6↑ ↔ 7↑
        H[2, 0] = h_jt
        H[1, 3] = h_jt   # 6↓ ↔ 7↓
        H[3, 1] = h_jt
        return H

    def build_single_particle_hamiltonian(self, Q: float, mu: float) -> np.ndarray:
        """
        Single-particle Hamiltonian for cluster ED (WITHOUT mean-field AFM Zeeman term).
        AFM exchange enters at cluster level via the O⊗O tensor product, not here.
        Basis: [6↑, 6↓, 7↑, 7↓]
        """
        H = np.zeros((4, 4), dtype=complex)
        
        # 1. Chemical potential
        np.fill_diagonal(H, -mu)
        
        # 2. Crystal field splitting: Δ_CF on Γ₇
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓
        
        # 3. JT distortion field, orbital mixing: g_JT·Q·(τ_x ⊗ I_spin)
        h_jt = self.p.g_JT * Q
        H[0, 2] = H[2, 0] = h_jt  # 6↑ ↔ 7↑
        H[1, 3] = H[3, 1] = h_jt  # 6↓ ↔ 7↓
        return H
    
    def build_pairing_block(self, Delta: complex) -> np.ndarray:
        """
        Construct 4×4 pairing matrix for inter-orbital singlet SC.
        Pairing structure: Δ(c₆↑c₇↓ - c₆↓c₇↑) + h.c.

        Irrep content (critical for the SC-activated JT mechanism):
          • The pairing Δ·(c₆↑c₇↓ - c₆↓c₇↑) connects the Γ₆ and Γ₇ Kramers
            doublets.  The Cooper pair lives in the Γ₆⊗Γ₇ tensor product space.
          • Under D₄h, Γ₆⊗Γ₇ contains the B₁g representation — the SAME irrep
            as the JT distortion mode.  This is the self-closure condition:
            the pairing channel and the JT channel are in the same irrep family.
          • The off-diagonal structure D[0,3] and D[1,2] is exactly τx
            (rank-2 quadrupolar operator) restricted to the anomalous sector.
            When Delta → 0 this block vanishes, τx is forbidden; when Delta ≠ 0
            the block opens, τx becomes accessible, JT unlocks.
          • See build_irrep_selection_projector() for the explicit algebraic
            implementation of the Γ₆→(Γ₆⊕Γ₇) space expansion.
        """
        D = np.zeros((4, 4), dtype=complex)
        
        # Singlet pairing: 6↑-7↓ with opposite sign for 6↓-7↑
        D[0, 3] = Delta    # 6↑ pairs with 7↓
        D[1, 2] = -Delta   # 6↓ pairs with 7↑ (opposite sign for antisymmetry)
        return D

    # =========================================================================
    # 3.4 OBSERVABLES FROM BdG SPECTRUM (PRIMARY SOURCE)
    # =========================================================================
    
    def _compute_site_magnetization_and_quadrupole(self, vec: np.ndarray, u_slice: slice, v_slice: slice,
                                                    f: float, f_bar: float) -> Tuple[float, float]:
        """
        Compute magnetization and quadrupole for a single site.
        
        Extracts particle (u) and hole (v) contributions from BdG eigenvector.
        Uses orbital-dependent spin operator: sz = [+1, -1, +η, -η] in [6↑, 6↓, 7↑, 7↓] basis.
        
        Parameters:
            vec: 16-component BdG eigenvector
            u_slice: slice for particle amplitudes
            v_slice: slice for hole amplitudes
            f: Fermi function f(E)
            f_bar: 1 - f(E)
        
        Returns:
            (m, tau): magnetization and quadrupole contributions
        """
        # Sz operator in orbital basis (6↑, 6↓, 7↑, 7↓)
        sz_op = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
        
        # Extract particle (u) and hole (v) parts
        u = vec[u_slice]
        v = vec[v_slice]
        
        # Magnetization contribution
        m = (np.abs(u)**2 @ sz_op) * f + (np.abs(v)**2 @ sz_op) * f_bar
        
        # Quadrupole τ_x mixing between orbitals 6 and 7
        tau_u = 2.0 * np.real(np.vdot(u[[0, 1]], u[[2, 3]]))
        tau_v = 2.0 * np.real(np.vdot(v[[0, 1]], v[[2, 3]]))
        tau = tau_u * f + tau_v * f_bar
        return m, tau
    
    def _compute_density_at_mu(self, mu: float, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float,
                              tx: float, ty: float, g_J: float) -> float:
        """
        Compute total density for given mu and parameters.
        Used for chemical potential root-finding.
        """
        # Vectorised over all k at once via VectorizedBdG (in-place buffer, no alloc)
        vbdg = self._get_vbdg()
        ev_all, ec_all = vbdg.diag_all_k(
            M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        # ev_all: (N_k, 16),  ec_all: (N_k, 16, 16)

        fn     = self.fermi_function(ev_all)          # (N_k, 16)
        fn_bar = 1.0 - fn

        uA = ec_all[:, 0:4,   :]   # (N_k, 4, 16)
        uB = ec_all[:, 4:8,   :]
        vA = ec_all[:, 8:12,  :]
        vB = ec_all[:, 12:16, :]

        dens_A = np.sum(np.abs(uA)**2 * fn[:, None, :]
                      + np.abs(vA)**2 * fn_bar[:, None, :], axis=(1, 2))  # (N_k,)
        dens_B = np.sum(np.abs(uB)**2 * fn[:, None, :]
                      + np.abs(vB)**2 * fn_bar[:, None, :], axis=(1, 2))

        n_avg = (dens_A + dens_B) / 2.0 / 2.0   # BdG doubling correction
        return float(np.dot(self.k_weights, n_avg))
    
    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float,
                             tx: float, ty: float, mu_guess: float, g_J: float) -> float:
        """
        Find chemical potential that yields target_doping via Newton's method.

        The analytic derivative ∂n/∂μ = Σ_{k,n} w_k · (−∂f/∂E) · (|u_A|² + |u_B|² + |v_A|² + |v_B|²)
        is computed from the same BdG eigensystem as n(μ), so each Newton step costs
        one diagonalization instead of the ~15 brentq evaluations.
        Falls back to brentq bracketing if Newton fails to converge.
        """
        target_n = 1.0 - target_doping

        def density_and_deriv(mu_val: float):
            """Return (n − target_n, ∂n/∂μ) from a single BdG diagonalization."""
            vbdg = self._get_vbdg()
            ev, ec = vbdg.diag_all_k(M, Q, Delta_s, Delta_d, target_doping,
                                      mu_val, tx, ty, g_J)
            f   = self.fermi_function(ev)          # (N_k, 16)
            fb  = 1.0 - f

            uA = ec[:, 0:4,   :]
            uB = ec[:, 4:8,   :]
            vA = ec[:, 8:12,  :]
            vB = ec[:, 12:16, :]

            dens_A = np.sum(np.abs(uA)**2 * f[:, None, :]
                          + np.abs(vA)**2 * fb[:, None, :], axis=(1, 2))
            dens_B = np.sum(np.abs(uB)**2 * f[:, None, :]
                          + np.abs(vB)**2 * fb[:, None, :], axis=(1, 2))
            n = float(np.dot(self.k_weights, dens_A + dens_B)) / 4.0

            # ∂n/∂μ: −∂f/∂E = f(1−f)/kT ≥ 0; total weight per (k,n) is
            # |u_A|²+|u_B|²+|v_A|²+|v_B|² = 1 (BdG normalization within each sublattice pair)
            df_dE = f * fb / max(self.p.kT, 1e-10)   # (N_k,16), ≥ 0
            # weight: same orbital sums as density
            w_A = np.sum(np.abs(uA)**2 + np.abs(vA)**2, axis=1)   # (N_k,16)
            w_B = np.sum(np.abs(uB)**2 + np.abs(vB)**2, axis=1)
            dn_dmu = float(np.einsum('k,kn,kn->', self.k_weights, df_dE, w_A + w_B)) / 4.0
            return n - target_n, dn_dmu

        mu = mu_guess
        for _ in range(20):   # Newton iterations (typically converges in 3-6)
            err, deriv = density_and_deriv(mu)
            if abs(err) < 1e-6:
                return mu
            if abs(deriv) < 1e-12:
                break   # flat → fall through to brentq
            # dn/dmu ≥ 0 always; abs() guards against rare numerical noise giving tiny negatives
            step = err / max(abs(deriv), 1e-10)
            # Limit step to bandwidth/4 to avoid overshooting
            step = float(np.clip(step, -self.p.t0, self.p.t0))
            mu -= step

        # Fallback: brentq (rare — only if Newton diverges or lands on flat region)
        def density_error(mu_val):
            vbdg = self._get_vbdg()
            ev, ec = vbdg.diag_all_k(M, Q, Delta_s, Delta_d, target_doping,
                                      mu_val, tx, ty, g_J)
            f  = self.fermi_function(ev)
            fb = 1.0 - f
            uA = ec[:, 0:4,  :];  vA = ec[:, 8:12, :]
            uB = ec[:, 4:8,  :];  vB = ec[:, 12:16,:]
            dA = np.sum(np.abs(uA)**2 * f[:,None,:] + np.abs(vA)**2 * fb[:,None,:], axis=(1,2))
            dB = np.sum(np.abs(uB)**2 * f[:,None,:] + np.abs(vB)**2 * fb[:,None,:], axis=(1,2))
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
                return brentq(density_error, mu_min, mu_max, xtol=1e-5)
        except Exception:
            pass
        return mu
    
    def compute_observables_from_bdg(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> Dict:
        """
        Thermal expectation values from BdG eigensystem. Primary source for M and Q
        (eigenstates carry Γ₆–Γ₇ mixing induced by Δ, implementing SC→JT feedback).

        Spinor ψ_n = [u_A(4), u_B(4), v_A(4), v_B(4)], each [6↑, 6↓, 7↑, 7↓].
        Density:     Σ_n [|u|²f + |v|²(1−f)]
        Magnetization: Σ_n [(u*sz u)f + (v*sz v)(1−f)]
        Quadrupole:  Σ_n [(u*τx u)f + (v*τx v)(1−f)]
        Pairing (inter-site A↔B): ⟨c_i c_j⟩ = Σ_n Re(F_n)·(1−2f(E_n))
        """
        # Fermi occupation factors
        f = self.fermi_function(eigvals)
        one_minus_f = 1.0 - f
        one_minus_2f = 1.0 - 2.0 * f  # For pairing
        
        # Initialize accumulators
        dens_A = dens_B = mag_A = mag_B = quad_A = quad_B = 0.0
        pair_sum_s = pair_sum_d = 0.0

        for n in range(16):
            psi = eigvecs[:, n]
            fn = f[n]
            fn_bar = one_minus_f[n]
            fn_pair = one_minus_2f[n]

            u_A = psi[0:4]    # particle A
            u_B = psi[4:8]    # particle B
            v_A = psi[8:12]   # hole A
            v_B = psi[12:16]  # hole B

            # Density: Σ_n[|u|²f + |v|²(1-f)]; all states counted, divided by 2 at end
            dens_A += np.sum(np.abs(u_A)**2) * fn + np.sum(np.abs(v_A)**2) * fn_bar
            dens_B += np.sum(np.abs(u_B)**2) * fn + np.sum(np.abs(v_B)**2) * fn_bar

            mA, tauA = self._compute_site_magnetization_and_quadrupole(psi, slice(0, 4), slice(8, 12), fn, fn_bar)
            mB, tauB = self._compute_site_magnetization_and_quadrupole(psi, slice(4, 8), slice(12, 16), fn, fn_bar)
            mag_A += mA; mag_B += mB; quad_A += tauA; quad_B += tauB

            # Channel s (on-site B₁g, φ=1): F_AA = u_A·v_A*, (1−2f) factor for finite-T BCS
            pair_onsite  = (u_A[0] * np.conj(v_A[3]) - u_A[1] * np.conj(v_A[2]))
            pair_onsite += (u_B[0] * np.conj(v_B[3]) - u_B[1] * np.conj(v_B[2]))
            # Channel d (inter-site d-wave B₁g, φ(k) in BdG): F_AB = u_A·v_B*
            pair_AB = (u_A[0] * np.conj(v_B[3]) - u_A[1] * np.conj(v_B[2]))
            pair_BA = (u_B[0] * np.conj(v_A[3]) - u_B[1] * np.conj(v_A[2]))
            pair_sum_s += pair_onsite * fn_pair
            pair_sum_d += 0.5 * (pair_AB + pair_BA) * fn_pair
        
        # All observables: /2 sublattice average, /2 BdG particle-hole doubling = /4 total
        n_avg    = (dens_A + dens_B) / 4.0   # average density per site
        M_stag   = (mag_A  - mag_B)  / 4.0   # staggered magnetization
        Q_unif   = (quad_A + quad_B) / 4.0   # uniform quadrupole (JT order)
        Pair_s   = pair_sum_s        / 4.0   # on-site orbital B₁g (channel s)
        Pair_d   = pair_sum_d        / 4.0   # inter-site d-wave B₁g (channel d)

        return {
            'n':      n_avg,
            'M':      M_stag,
            'Q':      Q_unif,
            'Pair_s': Pair_s,    # feeds Δ_s gap equation (no φ weight)
            'Pair_d': Pair_d,    # feeds Δ_d gap equation (φ(k) weight already in BdG)
            'Pair':   Pair_s + Pair_d,   # total pairing amplitude (for backwards compat)
        }
    
    # =========================================================================
    # 3.5 FREE ENERGY CALCULATIONS
    # =========================================================================
    
    def compute_bdg_free_energy(self, M: float, Q: float,
                                Delta_s: complex, Delta_d: complex, target_doping: float,
                                mu: float, tx: float, ty: float, g_J: float,
                                rpa_factor: float = 1.0) -> float:
        """
        Grand potential per site from the k-space BdG spectrum:

        Ω = (1/2) Σ_{k,n} w_k [E_n f_n − T S(f_n)]
              + |Δ_s|²/(g_Δs·V_s) + |Δ_d|²/(g_Δd·V_d) + (K/2)Q²

        The 1/2 accounts for the doubled unit cell.
        Condensation cost per channel uses independent g_Delta_s/g_Delta_d
        (derived internally from target_doping via get_gutzwiller_factors).
        (K/2)Q²: elastic energy; JT gain E_JT = g²/(2K) already in BdG spectrum.
        V_eff = g²/K (adiabatic BCS); RPA renormalizes the electronic gain, not V_eff.
        """
        vbdg = self._get_vbdg()
        ev_all, _ = vbdg.diag_all_k(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        f_n = self.fermi_function(ev_all)   # (N_k, 16)

        Ef = np.einsum('k,kn,kn->', self.k_weights, ev_all, f_n)

        if self.p.kT > 1e-8:
            eps = 1e-12
            f_c = np.clip(f_n, eps, 1.0 - eps)
            S_kn = -(f_c * np.log(f_c) + (1.0 - f_c) * np.log(1.0 - f_c))
            S_term = self.p.kT * np.einsum('k,kn->', self.k_weights, S_kn)
        else:
            S_term = 0.0

        Omega_cell = Ef - S_term

        # V_eff = g²/K (adiabatic, BCS controlled λ~O(1)); g²/ω would give Eliashberg regime
        _V_base_fe  = (self.p.g_JT**2 / max(self.p.K_lattice, 1e-9)) * rpa_factor
        V_s = self.p.V_s_scale * _V_base_fe
        V_d = self.p.V_d_scale * _V_base_fe
        # Condensation cost: |Δ|²/(g_Δ·V) — per channel with independent Gutzwiller factors
        _, _, g_Delta_s_fe, g_Delta_d_fe = self.get_gutzwiller_factors(target_doping)
        cond_s = (abs(Delta_s)**2 / (g_Delta_s_fe * V_s) if (V_s > 1e-12 and abs(Delta_s) > 1e-10) else 0.0)
        cond_d = (abs(Delta_d)**2 / (g_Delta_d_fe * V_d) if (V_d > 1e-12 and abs(Delta_d) > 1e-10) else 0.0)
        condensation_correction = cond_s + cond_d
        elastic_energy = 0.5 * self.p.K_lattice * Q**2  # JT gain already in BdG spectrum

        # Convert per unit cell → per site, then add field costs
        Omega_per_site = Omega_cell / 2.0 + elastic_energy + condensation_correction
        return Omega_per_site
    
    def compute_cluster_free_energy(self, M: float, Q: float, mu: float, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> Dict:
        """
        Cluster free energy from exact diagonalization: F = −T log Z.
        tx_bare, ty_bare: raw Q-dependent hoppings (g_J renormalizes J separately).
        Returns: 'F_per_site', 'M', 'Q_exp', 'Q_rms', 'Q_fluctuation', 'J_eff'.
        """
        H_sp_A = self.build_single_particle_hamiltonian(Q, mu)
        H_sp_B = self.build_single_particle_hamiltonian(Q, mu)

        J_eff = self.effective_superexchange(Q, g_J, tx_bare, ty_bare, doping)
        H_cluster = self.cluster_mf.build_cluster_hamiltonian(
            H_sp_A, H_sp_B, J_eff, M, self.p.eta, self.p.U_mf, g_J
        )

        # ---- Single diagonalization ----
        evals, evecs = eigh(H_cluster)

        # Free energy F = -T log Z
        if self.p.kT < 1e-8:
            F_total = evals[0]
        else:
            E_shifted = evals - evals[0]
            weights   = np.exp(-E_shifted / self.p.kT)
            Z         = weights.sum()
            F_total   = evals[0] - self.p.kT * np.log(Z)

        F_per_site = F_total / self.cluster_mf.CLUSTER_SIZE

        # Magnetization operator
        O_mag = self.cluster_mf.build_multipolar_operator(self.p.eta)
        M_A = self.cluster_mf.cluster_expectation(evals, evecs, O_mag, self.p.kT, site_index=0)
        M_B = self.cluster_mf.cluster_expectation(evals, evecs, O_mag, self.p.kT, site_index=1)
        M_cluster = abs(M_A - M_B) / 2.0

        # Quadrupole operator τ_x  (does NOT commute with H_cluster → ⟨τ_x⟩² ≠ ⟨τ_x²⟩)
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
            'F_per_site':    F_per_site,
            'M':             M_cluster,
            'Q_exp':         Q_exp,
            'Q_rms':         Q_rms,
            'Q_fluctuation': fluctuation,
            'J_eff':         J_eff
        }

# =========================================================================
# 4. SELF-CONSISTENT FIELD SOLVER
# =========================================================================

    def _scf_jacobi_kick(self, target_doping: float,
                         initial_M: float, initial_Q: float,
                         mu_guess: float) -> Dict:
        """
        Estimate the Jacobi spectral radius λ_+ of the (Δ, Q) fixpoint map at the
        given initial point, and return physics-informed seed values for (M, Q, Δ_s, Δ_d).

        The two-channel (Δ, Q) SCF map has Jacobian:

            J = [ A  B ]
                [ C  0 ]

            A = ∂Δ_out/∂Δ  ≈  N(0)·V_pair        (gap-equation self-coupling)
            B = ∂Δ_out/∂Q  ≈  g_JT·N(0)·V_pair/Δ_CF   (JT drives extra pairing)
            C = ∂Q_out/∂Δ  ≈  (g_JT/K)·(∂⟨τx⟩/∂Δ)    (SC unlocks JT)

        The dominant eigenvalue is:
            λ_+ = 0.5·(A + √(A² + 4·B·C))

        Regime classification and seed strategy:
          λ_+ << 1  (subcritical)  : small Δ seed; standard mixing safe
          λ_+ ≈ 1   (critical)     : Δ seed = kT (marginal fluctuation); mixing reduced
          λ_+ >> 1  (supercritical): large Δ seed to land in the correct basin;
                                     mixing must be small to avoid divergence

        Returns dict with:
          'M_kick'      : float  — seed magnetisation
          'Q_kick'      : float  — seed JT distortion (Å)
          'Delta_kick'  : float  — seed gap amplitude (eV)
          'mixing_kick' : float  — recommended initial mixing parameter
          'lambda_plus' : float  — dominant Jacobi eigenvalue
          'regime'      : str    — 'subcritical' | 'critical' | 'supercritical'
          'A', 'B', 'C' : float  — Jacobi matrix elements (diagnostics)
        """
        p   = self.p
        abs_d = max(abs(target_doping), 1e-6)
        g_t   = (2.0 * abs_d) / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d) ** 2
        t_eff = g_t * p.t0
        N0    = 1.0 / (np.pi * max(t_eff, 1e-6))

        # V_pair: JT channel + linearized Moriya-RPA spin fluctuation correction
        # (full q-dependent vertex is in the gap equation; analytic estimate only here)
        _chi0_est = N0 / (1.0 + (p.U_mf / max(np.pi * t_eff, 1e-9))**2)
        _U_pair   = g_J * p.U
        _stoner   = max(abs(1.0 - _U_pair * _chi0_est), 0.05)
        V_spin_est = _U_pair**2 * _chi0_est / _stoner**2   # Moriya-RPA linearised
        V_JT      = p.g_JT**2 / max(p.K_lattice, 1e-9)
        V_pair_d  = p.V_d_scale * (V_JT + V_spin_est)
        V_pair_s  = p.V_s_scale * V_JT
        V_pair    = max(V_pair_d, V_pair_s)

        # Jacobi element A: ∂Δ_out/∂Δ from the linearised BCS gap equation
        # Δ_out = g_Δ · V · Σ_k (1-2f) · Δ/(2E_k)  → ∂/∂Δ = g_Δ · V · N(0)
        g_Delta = np.sqrt(g_t)   # geometric-mean Gutzwiller (on-site channel)
        A = g_Delta * V_pair * N0

        # Jacobi element B: ∂Δ_out/∂Q
        # A JT distortion Q shifts ξ_k by ±g_JT·Q (B₁g), which modifies the pairing
        # kernel: ∂Δ_out/∂Q ≈ g_JT · N(0) · g_Δ · V_pair / Δ_CF
        B = p.g_JT * N0 * g_Delta * V_pair / max(p.Delta_CF, 1e-9)

        # Jacobi element C: ∂Q_out/∂Δ
        # Q_out = (g_JT/K)·⟨τx⟩_anom;  ⟨τx⟩_anom grows linearly with Δ near Δ=0:
        # ∂⟨τx⟩/∂Δ ≈ N(0)/Δ_CF  (first-order perturbation theory in Δ/Δ_CF)
        C = (p.g_JT / max(p.K_lattice, 1e-9)) * N0 / max(p.Delta_CF, 1e-9)

        # Dominant eigenvalue of J
        discriminant = A**2 + 4.0 * B * C
        lambda_plus  = 0.5 * (A + np.sqrt(max(discriminant, 0.0)))

        # --- Regime classification and seed strategy ---
        if lambda_plus < 0.7:
            regime       = 'subcritical'
            Delta_kick   = max(initial_Q * p.g_JT * 0.5, p.kT)   # small seed
            M_kick       = initial_M
            Q_kick       = initial_Q
            mixing_kick  = p.mixing                               # standard mixing

        elif lambda_plus <= 1.4:
            regime       = 'critical'
            # Near λ_+≈1 the SCF map is nearly neutral: seed with marginal fluctuation.
            # A slightly larger Δ seed avoids the trivial Δ=0 fixpoint while staying
            # within the physical basin.
            Delta_kick   = max(3.0 * p.kT, 0.5 * p.g_JT * abs(initial_Q))
            M_kick       = initial_M
            # Seed Q from the self-consistent JT equilibrium at this Δ_kick:
            # Q_eq ≈ (g_JT/K)·(Δ_kick/Δ_CF)·N(0)  (linearised SC→JT response)
            Q_kick_est   = (p.g_JT / max(p.K_lattice, 1e-9)) * (Delta_kick / max(p.Delta_CF, 1e-9)) * N0
            Q_kick       = float(np.clip(Q_kick_est, initial_Q,
                                          0.1 * p.lambda_hop))
            # Reduce mixing to slow down the neutral mode
            mixing_kick  = min(p.mixing * 0.5, 0.02)

        else:
            regime       = 'supercritical'
            # λ_+ > 1: the trivial fixpoint is unstable; the system will find a
            # broken-symmetry state. Seed with a physically motivated large Δ to
            # land directly in the correct basin and avoid oscillations.
            # Estimate: Δ ≈ 2·kT·exp(−1/λ_+) (BCS-like gap equation solution)
            Delta_kick   = float(np.clip(
                2.0 * p.kT * np.exp(min(1.0 / max(lambda_plus - 1.0, 0.05), 10.0)),
                0.01, 0.3))
            M_kick       = initial_M * 0.8   # slight reduction: SC competes with AFM
            Q_kick_est   = (p.g_JT / max(p.K_lattice, 1e-9)) * (Delta_kick / max(p.Delta_CF, 1e-9)) * N0
            Q_kick       = float(np.clip(Q_kick_est, initial_Q, 0.2 * p.lambda_hop))
            # Very small mixing: supercritical regime is prone to overshoot
            mixing_kick  = min(p.mixing * 0.25, 0.01)

        return {
            'M_kick':      M_kick,
            'Q_kick':      Q_kick,
            'Delta_kick':  Delta_kick,
            'mixing_kick': mixing_kick,
            'lambda_plus': lambda_plus,
            'regime':      regime,
            'A': A, 'B': B, 'C': C,
        }

    def solve_self_consistent(self, target_doping: float,
                            initial_M: float = 0.5,
                            initial_Q: float = 1e-4,
                            initial_Delta: float = 0.05,
                            verbose: bool = True) -> Dict:
        """
        Variational SCF: minimizes F_total(M, Q, Δ, μ) subject to ⟨n⟩ = target_density.

        Each iteration:
          1. BdG diagonalization → M_bdg, τ_x, Pair (SC→orbital feedback included)
          2. Irrep selection: compute_rank2_multipole_expectation() quantifies B₁g barrier lifting
          3. RPA χ₀(q_AFM) every ~5 iterations → Stoner rpa_factor for V_eff
          4. Dual-channel gap equations Δ_s, Δ_d
          5. Cluster ED → M_cluster (quantum fluctuation correction, CLUSTER_WEIGHT)
          6. Hellmann-Feynman ∂F/∂M Newton step for M
          7. Q_out = (g_JT/K)·⟨τ_x⟩_total (SC-triggered JT equilibrium)
          8. Anderson mixing on [M,Q] with adaptive mixing rate; simple mixing on Δ
          9. Brent root-finding for μ; post-convergence Hessian test

        Initialization:
          _scf_jacobi_kick() computes the dominant eigenvalue λ_+ of the (Δ,Q) Jacobian
          and classifies the regime (subcritical / critical / supercritical). The seed
          values for (M, Q, Δ) and the initial mixing rate are chosen accordingly:
            subcritical  (λ_+ < 0.7)  : standard seed, standard mixing
            critical     (λ_+ ≈ 1)    : marginal Δ seed, halved mixing
            supercritical(λ_+ > 1.4)  : BCS-estimated large Δ seed, quartered mixing

        Adaptive mixing:
          The mixing rate α is updated every 5 iterations based on the convergence trend:
            diverging  (max_diff increasing)  → halve α (floor: mixing/8)
            stagnating (max_diff barely falls) → increase α by 20% (cap: mixing×2)
            converging normally               → keep α

        Returns converged dict: M, Q, Delta_s, Delta_d, mu, density, free energies,
        Gutzwiller factors, hessian, iteration history, chi0/rpa trajectories,
        lambda_plus (Jacobi spectral radius), regime (SCF regime classification).
        """
        # ------------------------------------------------------------------
        # Jacobi-based initialization: physics-informed seed + mixing rate
        # ------------------------------------------------------------------
        _mu0_est: float
        if abs(target_doping) < 0.01:
            _mu0_est = 0.0
        elif target_doping > 0:
            _mu0_est = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        else:
            _mu0_est = 2.0 * self.p.t0 * np.tanh(abs(target_doping) / 0.1)
        _mu0_est += 0.5 * self.p.Delta_CF

        kick = self._scf_jacobi_kick(target_doping, initial_M, initial_Q, _mu0_est)

        M = kick['M_kick']
        Q = kick['Q_kick']

        # Split kick Delta between channels by V_s_scale / V_d_scale ratio
        _V_ratio_total = max(self.p.V_s_scale + self.p.V_d_scale, 1e-9)
        _frac_s = self.p.V_s_scale / _V_ratio_total
        _frac_d = self.p.V_d_scale / _V_ratio_total
        # Use kick delta if it is larger than the user-supplied initial_Delta to avoid accidentally shrinking a warm-start seed that is already near convergence.
        _Delta_seed = max(kick['Delta_kick'], float(initial_Delta))
        Delta_s = _Delta_seed * _frac_s + 0.0j   # on-site orbital B₁g
        Delta_d = _Delta_seed * _frac_d + 0.0j   # inter-site d-wave B₁g

        # mu: use the analytic estimate (same as before)
        mu = _mu0_est

        # Initial adaptive mixing rate from kick
        _alpha = kick['mixing_kick']

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [],
            'F_bdg': [], 'F_cluster': [],
            'g_t': [], 'g_J': [], 'mu': [],
            'chi0': [], 'rpa_factor': [], 'afm_unstable': [], 'selection_ratio': [],
            'chi_tau': [], 'Ut_ratio': [],
            'lambda_max': [],   # largest eigenvalue of linearised gap equation
            'gap_symmetry': [], # 'B1g (d-wave)' or 'A1g (s-wave)'
            'mixing': [],       # adaptive mixing rate per iteration
        }

        if verbose:
            print(f"\n{'='*80}")
            print("BdG LATTICE-BASED SELF-CONSISTENT CALCULATION")
            print(f"{'='*80}")
            print(f"Target doping δ={target_doping:.3f}")
            print(f"Jacobi kick:  λ_+ = {kick['lambda_plus']:.4f}  "
                  f"regime = {kick['regime']}  "
                  f"(A={kick['A']:.3f}, B={kick['B']:.4f}, C={kick['C']:.4f})")
            print(f"  Seeds:  M={M:.4f}  Q={Q:.5f}  Δ_seed={_Delta_seed:.5f}  "
                  f"α_init={_alpha:.4f}")
            print(f"{'-'*80}")

        # Anderson mixing history for M and Q
        scf_x_hist: list = []
        scf_f_hist: list = []

        # Initialise cached RPA quantities (updated via outer-loop strategy)
        chi0         = 0.0
        rpa_factor   = 1.0
        afm_unstable = False
        chi_tau      = 0.0   # multipolar susceptibility; analytic, updated every iter
        chi0_result  = {'Ut_ratio': 0.0, 'chi_tau': 0.0}  # sentinel until first chi0 update
        _chi0_last_M     = initial_M + 999.0   # force update at iteration 0
        _chi0_last_Delta = initial_Delta + 999.0   # triggers chi0 update at iter 0
        # q-dependent RPA vertex cache: reused when M and Δ change slowly
        _vertex_cache: Optional[dict] = None
        # Initialise irrep_info so it exists even if loop exits before irrep check
        irrep_info = {'w': 0.0, 'selection_ratio': 0.0,
                      'jt_algebraically_allowed': False,
                      'tau_x_projected': 0.0, 'tau_x_free_max': 1.0}

        _scf_t0 = _time.time()
        _tol_use = self.p.tol   # may be relaxed inside loop if lambda_max ≈ 1

        # Adaptive mixing state
        _alpha_min = self.p.mixing / 8.0    # floor: very conservative
        _alpha_max = self.p.mixing * 2.0    # cap: not faster than 2× base
        _max_diff_prev = float('inf')       # previous iteration's max_diff
        _stagnation_count = 0               # consecutive near-stagnation iterations
        _near_critical = False              # updated each iteration from _lambda_max

        for iteration in range(self.p.max_iter):

            # --- Renormalized parameters (use OLD Q for this step) ---
            g_t, g_J, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)
            tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
            tx, ty = g_t * tx_bare, g_t * ty_bare

            # --- Single BdG diagonalisation shared by observables, gap eq, and dF/dM ---
            # Computing it once and caching avoids 3→1 LAPACK calls per iteration.
            _bdg_ev, _bdg_ec = self._get_vbdg().diag_all_k(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            self._scf_bdg_cache = (_bdg_ev, _bdg_ec)   # picked up by dF_dM_and_d2F

            # --- Compute observables with CURRENT mu ---
            obs = self._get_vbdg().compute_observables_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                _bdg_cache=(_bdg_ev, _bdg_ec))
            tau_x      = obs['Q']
            Pair_s_obs = obs['Pair_s']   # on-site pairing amplitude (channel s)
            Pair_d_obs = obs['Pair_d']   # inter-site pairing amplitude (channel d)
            M_bdg      = obs['M']        # BdG response: lattice magnetization

            # --- Irrep selection: has SC lifted the B₁g barrier? ---
            # Core algebraic test of SC-activated JT: |Δ|/Δ_CF → mixing weight (Γ6⊕Γ7 space),
            # and ⟨τx⟩ checks if the JT channel is unlocked. If not allowed, JT drive must vanish.(diagnostic; self-consistently enforced)
            Delta_eff  = abs(Delta_s) + abs(Delta_d)   # combined for irrep mixing weight
            irrep_info = self.compute_rank2_multipole_expectation(Delta_eff, tau_x)

            # --- χ_τx: analytic, every iteration (no LAPACK) ---
            chi_tau_result = self._compute_chi_tau(M, Q, target_doping)
            chi_tau  = chi_tau_result['chi_tau']
            Ut_ratio = chi_tau_result['Ut_ratio']

            # --- RPA spin-fluctuation enhancement (outer-loop) ---
            # χ₀(q_AFM) is expensive (full even-grid LAPACK).  Update only when:
            #   • first iteration, OR
            #   • M or |Δ| changed significantly, AND at least 10 iters have passed.
            # rpa_factor feeds compute_bdg_free_energy; afm_unstable feeds score/diagnostics.
            _chi0_update_needed = (
                iteration == 0
                or (iteration % 10 == 0
                    and (abs(M - _chi0_last_M) > 0.02
                         or abs(Delta_eff - _chi0_last_Delta) > 0.005))
            )
            if _chi0_update_needed:
                chi0_result  = self.compute_static_chi0_afm(
                    M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
                rpa_factor   = self.rpa_stoner_factor(chi0_result)
                chi0         = chi0_result['chi0']
                afm_unstable = chi0_result['afm_unstable']
                _chi0_last_M     = M
                _chi0_last_Delta = Delta_eff

            # --- Dual-channel gap equations with q-dependent RPA vertex ---
            Delta_s_out, Delta_d_out, _vertex_cache = self._get_vbdg().compute_gap_eq_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, g_Delta_s, g_Delta_d,
                _bdg_cache=(_bdg_ev, _bdg_ec),
                _vertex_cache=_vertex_cache)

            # --- Newton–LM correction for Δ: blended with gap-equation fixpoint ---
            # ∂F/∂Δ = Δ/(g_Δ·V) − F_pair = 0; analytic curvature: d²F/dΔ² = 1/denom.
            _V_base = self.p.g_JT**2 / max(self.p.K_lattice, 1e-9)
            V_s_n   = self.p.V_s_scale * _V_base
            V_d_n   = self.p.V_d_scale * _V_base
            mu_LM_D = self.p.mu_LM_D
            ALPHA_D = self.p.ALPHA_D

            def _newton_delta(d_val, V_c, g_Dc, Pair_c):
                if V_c < 1e-12:
                    return 0.0
                denom = max(V_c * g_Dc, 1e-12)
                dF0   = d_val / denom - float(np.real(Pair_c))
                gamma = denom / (1.0 + mu_LM_D * denom)   # analytic d²F = 1/denom
                return d_val - gamma * dF0

            Ds_newton = _newton_delta(abs(Delta_s), V_s_n, g_Delta_s, Pair_s_obs)
            Dd_newton = _newton_delta(abs(Delta_d), V_d_n, g_Delta_d, Pair_d_obs)
            Delta_s_out = float(max((1.0 - ALPHA_D)*Delta_s_out + ALPHA_D*Ds_newton, 0.0))
            Delta_d_out = float(max((1.0 - ALPHA_D)*Delta_d_out + ALPHA_D*Dd_newton, 0.0))

            # --- Hellmann–Feynman Newton update for M ---
            # ∂F/∂M = 0;  → Newton step γ = 1/|∂²F/∂M²|; both gradient and curvature from a single perturbation-theory diag.
            dF_dM_0, d2F_dM2 = self.compute_dF_dM_and_d2F(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            self._scf_bdg_cache = None   # cache consumed; clear to prevent stale reuse
            gamma_M = 1.0 / (abs(d2F_dM2) + self.p.mu_LM)
            ALPHA_HF  = self.p.ALPHA_HF
            F_cluster_early = self.compute_cluster_free_energy(M, Q, mu, g_J, tx_bare, ty_bare, target_doping)
            M_fixpoint = (1.0 - self.p.CLUSTER_WEIGHT) * M_bdg + self.p.CLUSTER_WEIGHT * F_cluster_early['M']
            M_newton   = M - gamma_M * dF_dM_0
            M_out = float(np.clip(
                (1.0 - ALPHA_HF) * M_fixpoint + ALPHA_HF * M_newton,
                0.0, 1.0
            ))

            # --- Q_out: JT equilibrium from total BdG orbital coherence ---
            # Correct SC-activated chain: Δ≠0 → F(k)≠0 → ⟨τ_x⟩≠0 → Q≠0.
            # When Δ=0: ⟨τ_x⟩→0 self-consistently (no spontaneous JT).

            Q_bdg    = -(self.p.g_JT / self.p.K_lattice) * tau_x
            Q_exp_cl = F_cluster_early['Q_exp']   # signed ⟨τ_x⟩ from cluster ED
            Q_min    = 1e-5   # below this the BdG signal is numerical noise
            Q_seed   = float(np.clip(Q_exp_cl,
                                     -0.002 * self.p.lambda_hop,
                                      0.002 * self.p.lambda_hop))
            # Smooth weight: w→1 when |Q_bdg| >> Q_min, w→0 when |Q_bdg| ≈ 0
            w_bdg    = float(np.tanh((abs(Q_bdg) / max(Q_min, 1e-9))**2))
            Q_out    = w_bdg * Q_bdg + (1.0 - w_bdg) * Q_seed
            Q_out    = float(np.clip(Q_out, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # --- Anderson mixing on [M,Q]; reset history on Q sign flip (valley jump) ---
            x_in  = np.array([M,     Q    ])
            x_out = np.array([M_out, Q_out])
            scf_x_hist.append(x_in)
            scf_f_hist.append(x_out)

            x_new = self._anderson_mix(scf_x_hist, scf_f_hist, m=5, alpha=_alpha)
            M_mixed    = float(np.clip(x_new[0], 0.0, 1.0))
            Q_mixed    = float(np.clip(x_new[1], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            if len(scf_x_hist) > 1 and (Q * Q_mixed < 0):
                scf_x_hist.clear()
                scf_f_hist.clear()
                _vertex_cache = None   # Q sign flip → FS topology may change

            # Δ_s and Δ_d: simple mixing with adaptive _alpha
            Delta_s_mixed = self._mix(Delta_s, Delta_s_out, alpha=_alpha)
            Delta_d_mixed = self._mix(Delta_d, Delta_d_out, alpha=_alpha)
            
            # --- Recompute hopping/exchange for mixed parameters ---
            tx_mixed_bare, ty_mixed_bare = self.effective_hopping_anisotropic(Q_mixed)
            tx_mixed, ty_mixed = g_t * tx_mixed_bare, g_t * ty_mixed_bare
            
            # --- Find μ for mixed parameters ---
            mu_new = self._find_mu_for_density(
                M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping,
                tx_mixed, ty_mixed, mu_guess=mu, g_J=g_J
            )
            
            # Verify density at new μ.
            n_kspace_new = self._compute_density_at_mu(
                mu_new, M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping,
                tx_mixed, ty_mixed, g_J
            )

            # --- Free energy with fully consistent converged state ---
            F_bdg = self.compute_bdg_free_energy(
                M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping, mu_new,
                tx_mixed, ty_mixed, g_J, rpa_factor=rpa_factor
            )
            F_cluster = self.compute_cluster_free_energy(M_mixed, Q_mixed, mu_new, g_J, tx_mixed_bare, ty_mixed_bare, target_doping)

            # Convergence: max change over {M, Q, Δ_s, Δ_d, μ}.
            Delta_s_abs = abs(Delta_s_mixed)
            Delta_d_abs = abs(Delta_d_mixed)
            max_diff = max(
                abs(M_mixed - M),
                abs(Q_mixed - Q),
                abs(Delta_s_abs - abs(Delta_s)),
                abs(Delta_d_abs - abs(Delta_d)),
                abs(mu_new - mu)
            )

            # --- Adaptive mixing rate update (every 5 iterations after warm-up) ---
            if iteration >= 5 and iteration % 5 == 0:
                if max_diff > _max_diff_prev * 1.05:
                    # Diverging: halve mixing, reset Anderson history
                    _alpha = max(_alpha * 0.5, _alpha_min)
                    scf_x_hist.clear()
                    scf_f_hist.clear()
                    _stagnation_count = 0
                elif max_diff > _max_diff_prev * 0.85:
                    # Stagnating: Anderson history is stale; reset it (stronger than nudging alpha)
                    _stagnation_count += 1
                    if _stagnation_count >= 2:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _stagnation_count = 0
                else:
                    _stagnation_count = 0
                if _near_critical:
                    _alpha = min(_alpha, self.p.mixing * 0.6)
            _max_diff_prev = max_diff

            # λ_max: full kernel every 10 iters; reuse last value in between.
            if iteration % 10 == 0:
                _lin = self.solve_linearized_gap_equation(
                    M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed,
                    target_doping, mu_new, tx_mixed, ty_mixed, g_J)
                _lambda_max   = _lin['lambda_max']
                _gap_symmetry = _lin['gap_symmetry']
            else:
                _lambda_max   = history['lambda_max'][-1] if history['lambda_max'] else 0.0
                _gap_symmetry = history['gap_symmetry'][-1] if history['gap_symmetry'] else 'unknown'

            # _near_critical and _tol_use set here so adaptive mixing cap sees current lambda.
            _near_critical = 0.8 <= _lambda_max <= 1.8
            _tol_use       = self.p.tol * (5.0 if _near_critical else 1.0)

            # Gap solver can disable vertex reuse when the pairing kernel becomes strongly state-dependent.
            if _vertex_cache is not None:
                _vertex_cache['near_critical'] = _near_critical

            # --- Adaptive mixing rate update (every 5 iterations after warm-up) ---
            if iteration >= 5 and iteration % 5 == 0:
                if max_diff > _max_diff_prev * 1.05:
                    _alpha = max(_alpha * 0.5, _alpha_min)
                    scf_x_hist.clear()
                    scf_f_hist.clear()
                    _stagnation_count = 0
                elif max_diff > _max_diff_prev * 0.85:
                    _stagnation_count += 1
                    if _stagnation_count >= 2:
                        scf_x_hist.clear()
                        scf_f_hist.clear()
                        _stagnation_count = 0
                else:
                    _stagnation_count = 0
                if _near_critical:
                    _alpha = min(_alpha, self.p.mixing * 0.6)
            _max_diff_prev = max_diff

            history['M'].append(abs(M))
            history['Q'].append(abs(Q))
            history['Delta'].append(Delta_s_abs + Delta_d_abs)
            history['density'].append(n_kspace_new)
            history['F_bdg'].append(F_bdg)
            history['F_cluster'].append(F_cluster['F_per_site'])
            history['g_t'].append(g_t)
            history['g_J'].append(g_J)
            history['mu'].append(mu_new)
            history['chi0'].append(chi0)
            history['rpa_factor'].append(rpa_factor)
            history['afm_unstable'].append(afm_unstable)
            history['selection_ratio'].append(irrep_info['selection_ratio'])
            history['chi_tau'].append(chi_tau)
            history['Ut_ratio'].append(Ut_ratio)
            history['mixing'].append(_alpha)
            history['lambda_max'].append(_lambda_max)
            history['gap_symmetry'].append(_gap_symmetry)

            if verbose:
                _elapsed = _time.time() - _scf_t0
                _frac    = (iteration + 1) / self.p.max_iter
                _w       = 38
                _filled  = int(_w * _frac)
                _bar     = "█" * _filled + "░" * (_w - _filled)
                _eta_s   = (_elapsed / max(iteration + 1, 1)) * (self.p.max_iter - iteration - 1)
                _h, _r   = divmod(int(_eta_s), 3600)
                _m, _s   = divmod(_r, 60)
                sys.stdout.write(
                    f"\r  SCF [{_bar}] {iteration+1:3d}/{self.p.max_iter}"
                    f"  {int(100*_frac):3d}%  conv={max_diff:.1e}"
                    f"  ETA {_h}:{_m:02d}:{_s:02d}  delta={target_doping:.3f}  "
                )
                sys.stdout.flush()
                if iteration % 10 == 0 or iteration < 5:
                    print()
                    print(f"  Iter {iteration:3d}: "
                        f"M={M:.4f}  Q={Q:.5f}  "
                        f"Ds={abs(Delta_s):.5f}  Dd={abs(Delta_d):.5f}  "
                        f"n={n_kspace_new:.4f}  mu={mu_new:.4f}  F={F_bdg:.6f}  "
                        f"chi0={chi0:.3f}  rpa={rpa_factor:.2f}  chi_tau={chi_tau:.3f}  "
                        f"α={_alpha:.4f}  "
                        f"JT={'OK' if irrep_info['jt_algebraically_allowed'] else '--'}")

            # Update for next iteration
            M, Q, Delta_s, Delta_d, mu = M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, mu_new

            if max_diff < _tol_use and abs(n_kspace_new - (1 - target_doping)) < 0.01:
                # --- Post-convergence Hessian / curvature test ---
                hessian_result = self.compute_hessian(
                    M, Q, abs(Delta_s) + abs(Delta_d), target_doping, mu_new, g_t, g_J
                )
                if verbose:
                    print(f"\n{'='*80}")
                    _crit_note = "  [near SC critical point — relaxed tol×5]" if _near_critical else ""
                    print(f"✓ CONVERGED after {iteration+1} iterations{_crit_note}")
                    print(f"{'='*80}")
                    print(f"M = {M:.6f}")
                    print(f"Q = {Q:.6f} Å")
                    print(f"|Δ_s| = {abs(Delta_s):.6f} eV  (on-site B₁g, V_s_scale={self.p.V_s_scale:.2f}, g_Δs≈1)")
                    print(f"|Δ_d| = {abs(Delta_d):.6f} eV  (d-wave B₁g,  V_d_scale={self.p.V_d_scale:.2f}, g_Δd=g_J)")
                    print(f"|Δ|   = {abs(Delta_s)+abs(Delta_d):.6f} eV  (total)")
                    print(f"n = {n_kspace_new:.6f}")
                    print(f"μ = {mu:.6f} eV")
                    print(f"F = {F_bdg:.6f} eV")
                    print(f"g_t = {g_t:.4f}, g_J = {g_J:.4f}")
                    print(f"χ₀(q_AFM) = {chi0:.4f}  RPA factor = {rpa_factor:.3f}×")
                    print(f"Irrep selection ratio R = {irrep_info['selection_ratio']:.4f}  "
                          f"(JT algebraically {'ALLOWED ✓' if irrep_info['jt_algebraically_allowed'] else 'BLOCKED ✗'})")
                    print(f"Linearised gap λ_max = {_lambda_max:.4f}  "
                          f"({'UNSTABLE ✓' if _lambda_max > 1 else 'stable'})"
                          f"  symmetry: {_gap_symmetry}")
                    print(f"Jacobi λ_+ = {kick['lambda_plus']:.4f}  "
                          f"regime = {kick['regime']}  final α = {_alpha:.4f}")
                    eigs = hessian_result['eigenvalues']
                    status = "✓ TRUE MINIMUM" if hessian_result['is_minimum'] else "⚠ SADDLE POINT"
                    print(f"Hessian eigenvalues: [{eigs[0]:.4f}, {eigs[1]:.4f}, {eigs[2]:.4f}]  {status}")
                    print(f"{'='*80}\n")
                break

        else:
            if verbose:
                print(f"\n⚠ Warning: Did not converge after {self.p.max_iter} iterations")
                print(f"Final density error: {abs(n_kspace_new - (1 - target_doping)):.6f}")
                print(f"Final μ = {mu:.6f} eV\n")
            hessian_result = {'H': None, 'eigenvalues': None, 'is_minimum': None, 'min_curvature': None}

        return {
            'M': M,
            'Q': Q,
            'Delta_s': abs(Delta_s),
            'Delta_d': abs(Delta_d),
            'chi_tau': chi_tau,       # multipolar susceptibility at convergence
            'Ut_ratio': Ut_ratio,     # U/t_eff at convergence (analytic)
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
            'rpa_factor': rpa_factor,
            'afm_unstable': afm_unstable,
            'irrep_info': irrep_info,
            'history': history,
            'hessian': hessian_result,
            'lambda_max': _lambda_max,      # linearised gap equation: largest eigenvalue
            'gap_symmetry': _gap_symmetry,  # 'B1g (d-wave)' or 'A1g (s-wave)'
            'lambda_plus': kick['lambda_plus'],  # Jacobi spectral radius at initialization
            'regime': kick['regime'],            # SCF regime: subcritical/critical/supercritical
            'converged': (max_diff < _tol_use and abs(n_kspace_new - (1 - target_doping)) < 0.01)
        }

    def _anderson_mix(self, x_history: list, f_history: list,
                      m: int = 5, alpha: float = None) -> np.ndarray:
        """
        Anderson mixing for self-consistent field convergence.
        Computes the minimum-norm linear combination of recent residuals and uses it to generate a new input estimate.
        Equivalent to a quasi-Newton step without explicitly forming the Jacobian.

        Args:
            x_history: list of previous input vectors [x_{n-m}, ..., x_n]
            f_history: list of previous output vectors [F(x_{n-m}), ..., F(x_n)]
            m: mixing history depth (window size)
            alpha: mixing parameter override (uses self.p.mixing if None)
        
        Returns:
            New proposed input vector x_{n+1}
        """
        alpha = self.p.mixing if alpha is None else alpha
        x_last = x_history[-1]
        f_last = f_history[-1]

        n = min(len(x_history), m)
        if n < 2:
            return (1 - alpha) * x_last + alpha * f_last

        # Last n elements only
        X = np.asarray(x_history[-n:])
        F = np.asarray(f_history[-n:])
        R = F - X

        dR = np.diff(R, axis=0)          # (n-1, d)
        r_last = R[-1]
        # Normal equations
        A = dR @ dR.T
        b = dR @ r_last
        # Regularization
        A.flat[::A.shape[0] + 1] += 1e-10

        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return (1 - alpha) * x_last + alpha * f_last

        # Vectorized Anderson update
        dx = np.diff(X, axis=0)
        x_opt = x_last + r_last + theta @ (dx + dR)

        # Safeguarded blend
        x_simple = (1 - alpha) * x_last + alpha * f_last
        return 0.5 * x_opt + 0.5 * x_simple

    def compute_dF_dM_and_d2F(self, M: float, Q: float,
                                Delta_s: complex, Delta_d: complex,
                                target_doping: float,
                                mu: float, tx: float, ty: float,
                                g_J: float) -> Tuple[float, float]:
        """
        Compute ∂F/∂M **and** ∂²F/∂M² analytically from a single BdG diagonalization.

        Hellmann-Feynman first derivative:
            ∂F/∂M = Σ_{k,n} f_n · ⟨ψ_n|∂H/∂M|ψ_n⟩

        Second derivative (∂H/∂M is diagonal, so the curvature has two contributions):
            Term 1 (diagonal): Σ_{k,n} (∂f_n/∂E_n) · ⟨ψ_n|∂H/∂M|ψ_n⟩²
            Term 2 (off-diag): Σ_{k,n≠m} (f_n − f_m)/(E_m − E_n) · |⟨ψ_n|∂H/∂M|ψ_m⟩|²

        Both follow from second-order perturbation theory on the free energy functional.
        Replaces the three-point finite-difference stencil (3 diagonalizations → 1).

        Returns
        -------
        (dF_dM, d2F_dM2) : both in eV per site (double unit cell correction applied).
        """
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t_sq_avg = 0.5 * (tx_bare**2 + ty_bare**2)
        h_prefactor = g_J * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) / 2.0

        sz_orb = np.array([1.0, -1.0, self.p.eta, -self.p.eta]) * h_prefactor
        dH_diag = np.array([
            -sz_orb[0], -sz_orb[1], -sz_orb[2], -sz_orb[3],   # particle A
            +sz_orb[0], +sz_orb[1], +sz_orb[2], +sz_orb[3],   # particle B
            +sz_orb[0], +sz_orb[1], +sz_orb[2], +sz_orb[3],   # hole A
            -sz_orb[0], -sz_orb[1], -sz_orb[2], -sz_orb[3],   # hole B
        ])  # (16,)

        vbdg = self._get_vbdg()
        # Reuse BdG cache from the SCF loop if available
        if hasattr(self, '_scf_bdg_cache') and self._scf_bdg_cache is not None:
            ev, ec = self._scf_bdg_cache
        else:
            ev, ec = vbdg.diag_all_k(M, Q, Delta_s, Delta_d,
                                       target_doping, mu, tx, ty, g_J)   # (N,16), (N,16,16)
        f_all = self.fermi_function(ev)   # (N,16)

        # ⟨ψ_n|∂H/∂M|ψ_n⟩ per (k, n)
        exp_nn = np.einsum('i,kin->kn', dH_diag, np.abs(ec)**2)   # (N,16)

        # --- first derivative ---
        grad = float(np.einsum('k,kn,kn->', self.k_weights, f_all, exp_nn)) / 2.0

        # --- second derivative ---
        # Term 1: diagonal — ∂f/∂E · ⟨∂H/∂M⟩²
        kT = self.p.kT
        df_dE = -f_all * (1.0 - f_all) / max(kT, 1e-10)   # (N,16)  ≤ 0
        term1 = float(np.einsum('k,kn,kn->', self.k_weights, df_dE, exp_nn**2))

        # Term 2: off-diagonal — Σ_{n≠m} (f_n − f_m)/(E_m − E_n) · |⟨ψ_n|∂H|ψ_m⟩|²
        # ∂H/∂M is diagonal, so ⟨ψ_n|∂H|ψ_m⟩ = Σ_i dH_diag[i] · ec[k,i,n]* · ec[k,i,m].
        # Using real(|ec|²) is not enough here since n≠m terms need the full complex product.
        off = np.einsum('i,kin,kim->knm', dH_diag, ec.conj(), ec)   # (N,16,16)
        off2 = np.abs(off)**2   # |matrix element|²

        dE_nm = ev[:, None, :] - ev[:, :, None]   # E_m − E_n,  (N,16,16)
        df_nm = f_all[:, :, None] - f_all[:, None, :]   # f_n − f_m
        safe  = np.abs(dE_nm) > 1e-8
        ratio = np.where(safe, df_nm / np.where(safe, dE_nm, 1.0), df_dE[:, :, None])
        # Zero out diagonal (n==n) to avoid counting self-terms
        np.einsum('knn->kn', ratio)[:] = 0.0
        term2 = float(np.einsum('k,knm,knm->', self.k_weights, ratio, off2))

        d2F = (term1 + term2) / 2.0
        return grad, d2F

    def compute_hessian(self, M: float, Q: float, Delta: float, target_doping: float,
                        mu: float, g_t: float, g_J: float,
                        eps_M: float = 1e-3, eps_Q: float = 1e-4,
                        eps_D: float = 1e-4) -> Dict:
        """
        Post-convergence Hessian of F(M, Q, Δ) via central finite differences.

        Classifies the converged fixpoint:
          - All eigenvalues > 0: true local minimum ✓
          - Any eigenvalue < 0: saddle point (unstable direction) ✗
          - Near-zero eigenvalue: flat/Goldstone direction

        g_t, g_J are passed explicitly to avoid recomputing from stale _last_density.
        g_Delta_s/g_Delta_d are derived internally from target_doping.

        Returns dict with 'H' (3×3), 'eigenvalues', 'is_minimum', 'min_curvature'.
        """
        def F(m, q, d):
            tb_x, tb_y = self.effective_hopping_anisotropic(q)
            return self.compute_bdg_free_energy(
                m, q, complex(d), complex(0), target_doping, mu, g_t * tb_x, g_t * tb_y, g_J
            )

        F0 = F(M, Q, Delta)

        # --- Diagonal second derivatives ---
        H = np.zeros((3, 3))

        F_Mpp = F(M + eps_M, Q, Delta);  F_Mmm = F(M - eps_M, Q, Delta)
        F_Qpp = F(M, Q + eps_Q, Delta);  F_Qmm = F(M, Q - eps_Q, Delta)
        F_Dpp = F(M, Q, Delta + eps_D);  F_Dmm = F(M, Q, Delta - eps_D)

        H[0, 0] = (F_Mpp - 2*F0 + F_Mmm) / eps_M**2
        H[1, 1] = (F_Qpp - 2*F0 + F_Qmm) / eps_Q**2
        H[2, 2] = (F_Dpp - 2*F0 + F_Dmm) / eps_D**2

        # --- Off-diagonal (cross) derivatives ---
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
        α = alpha if alpha is not None else self.p.mixing
        return (1 - α) * old + α * new

    # =========================================================================
    # 4.5  SC–JT INSTABILITY MATRIX
    # =========================================================================
    # Gives the boundary of the normal-state instability (Tc, Q_c).
    #
    #   G = | 1 - V·χ_ΔΔ    -V·χ_ΔQ      |
    #       | -K_inv·χ_ΔQ   1 - K_inv·χ_QQ |
    #
    #   Instability: det(G) = 0  ⟺  min eigenvalue → 0
    #
    # When to use:
    #   • Tc scan / parameter sweep  → G-analysis (fast, analytic)
    #   • Equilibrium Δ, Q, M        → SCF (full nonlinear)
    #   • First-order transitions    → SCF only (G cannot detect these)

    def compute_G_instability(self, target_doping: float, M: float = 0.0) -> dict:
        """
        2×2 SC–JT instability matrix G(Δ→0, Q→0) in the AFM normal state, so M > 0 reconstructs the Fermi surface via the AFM Weiss field ±h_afm·σ_z.

        AFM band reconstruction (Δ=0, Q=0):
          ξ_k    = -2t_eff(cos kx + cos ky) - μ
          ξ_{k+Q} = +2t_eff(cos kx + cos ky) - μ   (nesting vector Q=(π,π))
          ξ_avg  = (ξ_k + ξ_{k+Q}) / 2 = -μ
          ξ_diff = (ξ_k - ξ_{k+Q}) / 2 = -2t_eff(cos kx + cos ky)
          E_k^±  = ξ_avg ± sqrt(ξ_diff² + h_afm²)

        Susceptibilities on E_k^± spectrum:
          χ_ΔΔ = Σ_{k,s=±} tanh(E_k^s/2T) / (2E_k^s)         [eV⁻¹]
          χ_QQ = g_JT² · Σ_{k,s} (-∂f/∂E_k^s)                [eV/Å²]
                 K_inv [Å²/eV] · χ_QQ [eV/Å²] = dimensionless  ✓
                 g_JT² from B₁g coupling: ∂E_k^±/∂Q ≈ ±g_JT
          χ_ΔQ = (g_JT/Δ_CF) · χ_ΔΔ                           [eV⁻¹·Å⁻¹]

        G-matrix (all dimensionless):
          G11 = 1 - V·χ_ΔΔ          [→0: pairing instability]
          G22 = 1 - K_inv·χ_QQ      [>0: SC-triggered JT, not spontaneous]
          G12 = -sqrt(V·K_inv)·χ_ΔQ

        Effective coupling (Schur complement):
          V_eff = V + V·K_inv·χ_ΔQ² / G22   [diverges as G22→0⁺]

        Limit M=0: h_afm=0, reduces to the classical paramagnetic BCS starting point.

        Returns: chi_DD, chi_QQ, chi_DQ, h_afm, E_plus_mean, N_eff,
                 G, det_G, lambda_min, V_eff, lambda_eff,
                 unstable, Tc_estimate, G11, G22, G12
        """
        p        = self.p
        abs_d    = max(abs(target_doping), 1e-6)
        g_t      = (2.0 * abs_d) / (1.0 + abs_d)
        g_J      = 4.0 / (1.0 + abs_d) ** 2
        t_eff    = g_t * p.t0
        kT       = max(p.kT, 1e-8)
        Delta_CF = max(p.Delta_CF, 1e-6)
        K_inv    = 1.0 / max(p.K_lattice, 1e-9)

        # Gutzwiller-renormalised pairing scale
        _, _, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)
        V_base = p.g_JT**2 / max(p.K_lattice, 1e-9)
        V      = max(p.V_s_scale, p.V_d_scale) * V_base * max(g_Delta_s, g_Delta_d)

        # AFM Weiss field (consistent with BdG Hamiltonian)
        f_d   = abs_d / (abs_d + p.doping_0)
        J_eff = g_J * 4.0 * t_eff**2 / max(p.U, 1e-6) * f_d
        h_afm = g_J * (p.U_mf / 2.0 + p.Z * J_eff / 2.0) * M

        # Chemical potential estimate near half-filling
        mu_n = -2.0 * t_eff * (1.0 - 2.0 * abs_d)

        kx = self.k_points[:, 0]
        ky = self.k_points[:, 1]

        # Normal dispersion and nesting partner (Q=(π,π))
        eps_k  = -2.0 * t_eff * (np.cos(kx) + np.cos(ky)) - mu_n
        eps_kQ = +2.0 * t_eff * (np.cos(kx) + np.cos(ky)) - mu_n

        xi_avg  = 0.5 * (eps_k + eps_kQ)
        xi_diff = 0.5 * (eps_k - eps_kQ)

        # AFM-reconstructed quasiparticle bands
        sqrt_term = np.sqrt(xi_diff**2 + h_afm**2)
        E_plus    = xi_avg + sqrt_term   # upper AFM band
        E_minus   = xi_avg - sqrt_term   # lower AFM band

        # χ_ΔΔ: BCS pair susceptibility on both E_k^± branches
        def _chi_dd_branch(E):
            arg    = np.clip(E / (2.0 * kT), -100, 100)
            safe_E = np.where(np.abs(E) > 1e-8, E, 1e-8)
            return np.dot(self.k_weights, np.tanh(arg) / (2.0 * safe_E))

        chi_DD = float(_chi_dd_branch(E_plus) + _chi_dd_branch(E_minus))

        # χ_QQ: dimensionally consistent JT susceptibility
        # χ_QQ = g_JT² · Σ_{k,s} (-∂f/∂E_k^s)  [eV/Å²]
        def _chi_qq_branch(E):
            f_E   = 1.0 / (1.0 + np.exp(np.clip(E / kT, -100, 100)))
            df_dE = -f_E * (1.0 - f_E) / kT
            return np.dot(self.k_weights, -df_dE)

        chi_QQ = p.g_JT**2 * float(_chi_qq_branch(E_plus) + _chi_qq_branch(E_minus)) # g_JT² from ∂E_k^±/∂Q ≈ ±g_JT (conservative upper bound, h→0 limit)
        N_eff  = float(_chi_qq_branch(E_plus) + _chi_qq_branch(E_minus))  # g_JT²-free, for Tc

        # χ_ΔQ: mixed SC–JT response, Cooper-pair hop energy scale Δ_CF
        chi_DQ = (p.g_JT / Delta_CF) * chi_DD

        # G-matrix (all dimensionless)
        G11 = 1.0 - V * chi_DD
        G22 = 1.0 - K_inv * chi_QQ   # K_inv [Å²/eV] · χ_QQ [eV/Å²] = dimensionless  ✓
        G12 = -np.sqrt(V * K_inv) * chi_DQ
        G = np.array([[G11, G12], [G12, G22]])

        det_G   = G11 * G22 - G12**2
        eigs    = np.linalg.eigvalsh(G)
        lam_min = float(eigs[0])

        # Effective coupling via Schur complement
        denom_22 = max(G22, 1e-8)
        V_eff    = V + (V * K_inv * chi_DQ**2) / denom_22

        # BCS Tc estimate using AFM N_eff
        lam_eff = N_eff * V_eff
        Tc_est  = float(1.13 * t_eff * np.exp(-1.0 / lam_eff)) if lam_eff > 1e-3 else 0.0

        return {
            'chi_DD':      chi_DD,
            'chi_QQ':      chi_QQ,
            'chi_DQ':      chi_DQ,
            'h_afm':       float(h_afm),
            'E_plus_mean': float(np.mean(E_plus)),
            'N_eff':       float(N_eff),
            'G':           G,
            'det_G':       float(det_G),
            'lambda_min':  lam_min,
            'V_eff':       float(V_eff),
            'lambda_eff':  float(lam_eff),
            'unstable':    det_G <= 0.0,
            'Tc_estimate': Tc_est,
            'G11': G11, 'G22': G22, 'G12': G12,
        }

def alpha_K_validity_bound(solver) -> float:
    """
    Compute the tight upper bound on alpha_K from BCS+RPA validity conditions.

    Conditions (all must hold):
      BCS validity:    lambda_eff = V_eff · N(0) < 1.5
                           V_eff = g²/K = g²·alpha_K / (alpha_K · Delta_CF)
                                        = g²/(alpha_K·Delta_CF)
                           → alpha_K < 1.5 / (Delta_CF · N(0))
      RPA linearity:   V_eff · chi0_max < 1/rpa_cutoff
                           chi0_max ~ N(0) (worst case)
                           → same order as (A), modulated by rpa_cutoff
      AFM-SC hierarchy: V_eff < h_AFM
                            → alpha_K < h_AFM / Delta_CF

    Returns the tightest bound, clipped to [1.1, 8.0].
    """
    delta_est = 0.16
    abs_delta = max(delta_est, 1e-6)
    g_t   = (2.0 * abs_delta) / (1.0 + abs_delta)
    t_eff = g_t * solver.p.t0
    N0    = 1.0 / (np.pi * max(t_eff, 1e-6))   # 2D tight-binding DOS
    Delta_CF = max(solver.p.Delta_CF, 1e-9)

    # BCS ceiling
    alpha_max_bcs = 1.5 / (Delta_CF * N0)
    # RPA ceiling (stricter by rpa_cutoff factor)
    alpha_max_rpa = 1.0 / (Delta_CF * N0 * max(solver.p.rpa_cutoff, 0.05))
    bound = min(alpha_max_bcs, alpha_max_rpa)

    # AFM ceiling
    f_d   = abs_delta / (abs_delta + solver.p.doping_0)
    g_J   = 4.0 / (1.0 + abs_delta) ** 2
    t2    = t_eff ** 2
    h_afm = g_J * (solver.p.U_mf / 2.0 + solver.p.Z * 2.0 * t2 / solver.p.U) * f_d * 0.5
    if h_afm > 1e-6:
        bound = min(bound, h_afm / Delta_CF)
    return float(np.clip(bound, 1.1, 8.0))

# =============================================================================
# 6. VECTORIZED BdG DIAGONALISATION
# =============================================================================

class VectorizedBdG:
    """
    Fast BdG: diagonalises all k-points simultaneously via numpy batch eigh.

    numpy.linalg.eigh accepts an (N, M, M) stack and returns (N, M) eigenvalues
    and (N, M, M) eigenvectors, exploiting LAPACK batched DSYEVD internally.
    This eliminates the Python-level k-loop and gives ~3–5× speedup on the SCF inner loop.

    Usage:
        vbdg = VectorizedBdG(solver)
        ev_all, ec_all = vbdg.diag_all_k(M, Q, Delta_s, Delta_d, doping, mu, tx, ty, g_J)

    Returns:
        ev_all : (N_k, 16)     eigenvalues
        ec_all : (N_k, 16, 16) eigenvectors (columns)
    """

    def __init__(self, solver: 'RMFT_Solver'):
        self.solver   = solver
        self._kpts    = solver.k_points       # (N_k, 2)
        self._N_k     = solver.N_k
        # Pre-allocated buffer, reused each iteration to avoid GC pressure
        self._H_stack = np.zeros((self._N_k, 16, 16), dtype=complex)

    def _build_H_stack(self,
                       kpts: np.ndarray,
                       M: float, Q: float,
                       Delta_s: complex, Delta_d: complex,
                       target_doping: float, mu: float,
                       tx: float, ty: float,
                       g_J: float,
                       out: Optional[np.ndarray] = None) -> np.ndarray:
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
        if out is None:
            H = np.zeros((N, 16, 16), dtype=complex)
        else:
            H = out
            H[:] = 0.0 + 0.0j

        a = self.solver.p.a

        # --- k-independent blocks ---
        H_A  = self.solver.build_local_hamiltonian_for_bdg(+1.0, M, Q, mu, g_J, target_doping)
        H_B  = self.solver.build_local_hamiltonian_for_bdg(-1.0, M, Q, mu, g_J, target_doping)
        D_on = self.solver.build_pairing_block(Delta_s)    # 4×4 on-site singlet
        D_dag = np.conj(D_on).T

        # Particle/hole diagonal blocks (broadcast to all k)
        H[:, 0:4,   0:4  ] = H_A
        H[:, 4:8,   4:8  ] = H_B
        H[:, 8:12,  8:12 ] = -np.conj(H_A)
        H[:, 12:16, 12:16] = -np.conj(H_B)

        # On-site pairing (k-independent)
        H[:, 0:4,   8:12 ] = D_on
        H[:, 4:8,  12:16 ] = D_on
        H[:, 8:12,  0:4  ] = D_dag
        H[:, 12:16, 4:8  ] = D_dag

        # --- k-dependent: dispersion γ(k) = -2(tx cos kx + ty cos ky) ---
        kx = kpts[:, 0]
        ky = kpts[:, 1]
        gamma_k = -2.0 * (tx * np.cos(kx * a) + ty * np.cos(ky * a))   # (N,)

        # T_AB = γ(k)·I₄ → diagonal sub-blocks via index broadcasting
        di = np.arange(4)
        H[:, di,      di + 4 ] = gamma_k[:, None]   # particle A→B
        H[:, di + 4,  di     ] = gamma_k[:, None]   # particle B→A (Hermitian)
        H[:, di + 8,  di + 12] = -gamma_k[:, None]  # hole sector: −γ*
        H[:, di + 12, di + 8 ] = -gamma_k[:, None]

        # --- k-dependent: inter-site d-wave pairing D_int[k] = Delta_d·φ(k)·pattern ---
        # For the full k-grid (out= path) we reuse the pre-cached form factor;
        # for arbitrary sub-grids (chi0 / k+Q) we recompute from kpts.
        if out is not None:
            phi = self.solver.phi_k * Delta_d   # (N_k,) — cached, no trig call
        else:
            phi = (np.cos(kx * a) - np.cos(ky * a)) * Delta_d   # (N,) fresh
        H[:, 0,  15] +=  phi          # A:6↑ → B:7↓
        H[:, 1,  14] -= phi           # A:6↓ → B:7↑ (singlet sign)
        H[:, 4,  11] +=  phi          # B:6↑ → A:7↓
        H[:, 5,  10] -= phi           # B:6↓ → A:7↑
        phi_c = np.conj(phi)
        H[:, 15,  0] +=  phi_c
        H[:, 14,  1] -= phi_c
        H[:, 11,  4] +=  phi_c
        H[:, 10,  5] -= phi_c
        return H

    def diag_all_k(self, M: float, Q: float,
                   Delta_s: complex, Delta_d: complex,
                   target_doping: float, mu: float,
                   tx: float, ty: float,
                   g_J: float) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalise all k-points in one batched LAPACK call.

        Uses the pre-allocated self._H_stack buffer (no heap allocation).
        """
        self._build_H_stack(self._kpts, M, Q, Delta_s, Delta_d,
                            target_doping, mu, tx, ty, g_J,
                            out=self._H_stack)
        ev_all, ec_all = np.linalg.eigh(self._H_stack)  # (N_k,16), (N_k,16,16)
        return ev_all, ec_all

    def diag_kpts(self,
                  kpts: np.ndarray,
                  M: float, Q: float,
                  Delta_s: complex, Delta_d: complex,
                  target_doping: float, mu: float,
                  tx: float, ty: float,
                  g_J: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalise an **arbitrary** kpts array (N, 2) in one batched LAPACK call.

        Returns (ev, ec) with shapes (N, 16) and (N, 16, 16).
        Used by chi0 routines that need k and k+Q simultaneously.
        Allocates a fresh buffer (kpts may differ in size from self._kpts).
        """
        H = self._build_H_stack(kpts, M, Q, Delta_s, Delta_d,
                                 target_doping, mu, tx, ty, g_J)
        return np.linalg.eigh(H)

    def compute_observables_vectorized(self, M: float, Q: float,
                                       Delta_s: complex, Delta_d: complex,
                                       target_doping: float, mu: float,
                                       tx: float, ty: float,
                                       g_J: float,
                                       _bdg_cache: tuple = None) -> Dict:
        """
        Vectorised observables: M, Q (τ_x), density, Pair_s, Pair_d.

        Computes the same quantities as RMFT_Solver.compute_observables_from_bdg
        but for all k simultaneously via broadcasting, avoiding any Python loop.

        _bdg_cache : optional (ev, ec) tuple from a previous diag_all_k call with
                     the same parameters.  Pass this from the SCF loop to avoid a
                     redundant diagonalisation.

        Returns a dict with the same keys as compute_observables_from_bdg.
        """
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = self.diag_all_k(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        kT = self.solver.p.kT

        # Fermi-Dirac factors: (N_k, 16)
        arg  = np.clip(ev / kT, -100, 100)
        f    = 1.0 / (1.0 + np.exp(arg))
        fbar = 1.0 - f
        f12  = 1.0 - 2.0 * f

        # Spinor slices:  ec[k, component, state_n]
        uA = ec[:, 0:4,   :]    # (N_k, 4, 16) particle sublattice A
        uB = ec[:, 4:8,   :]
        vA = ec[:, 8:12,  :]
        vB = ec[:, 12:16, :]

        # Density: Σ_n [|u|²f + |v|²(1-f)] summed over orbital and eigenstate
        dens_A = np.sum(np.abs(uA)**2 * f[:, None, :] + np.abs(vA)**2 * fbar[:, None, :], axis=(1, 2))
        dens_B = np.sum(np.abs(uB)**2 * f[:, None, :] + np.abs(vB)**2 * fbar[:, None, :], axis=(1, 2))

        # Staggered magnetisation: sz = [+1, -1, η, -η]
        eta = self.solver.p.eta
        sz  = np.array([1.0, -1.0, eta, -eta])

        mag_A = np.sum((np.abs(uA)**2 * sz[None, :, None]) * f[:, None, :]
                     + (np.abs(vA)**2 * sz[None, :, None]) * fbar[:, None, :], axis=(1, 2))
        mag_B = np.sum((np.abs(uB)**2 * sz[None, :, None]) * f[:, None, :]
                     + (np.abs(vB)**2 * sz[None, :, None]) * fbar[:, None, :], axis=(1, 2))

        # Quadrupole τ_x = 2 Re(u₀*u₂ + u₁*u₃)  — orbital mixing indicator
        tau_u_A = 2.0 * np.real(uA[:, 0, :] * np.conj(uA[:, 2, :])
                               + uA[:, 1, :] * np.conj(uA[:, 3, :]))   # (N_k, 16)
        tau_v_A = 2.0 * np.real(vA[:, 0, :] * np.conj(vA[:, 2, :])
                               + vA[:, 1, :] * np.conj(vA[:, 3, :]))
        tau_u_B = 2.0 * np.real(uB[:, 0, :] * np.conj(uB[:, 2, :])
                               + uB[:, 1, :] * np.conj(uB[:, 3, :]))
        tau_v_B = 2.0 * np.real(vB[:, 0, :] * np.conj(vB[:, 2, :])
                               + vB[:, 1, :] * np.conj(vB[:, 3, :]))
        quad_A  = np.sum(tau_u_A * f + tau_v_A * fbar, axis=1)   # (N_k,)
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
        Pair_s = complex(np.dot(w, pair_s))          / 4.0
        Pair_d = complex(np.dot(w, pair_d))          / 4.0

        return {'n': n_avg, 'M': M_stag, 'Q': Q_unif,
                'Pair_s': Pair_s, 'Pair_d': Pair_d, 'Pair': Pair_s + Pair_d}

    def compute_gap_eq_vectorized(self, M: float, Q: float,
                                   Delta_s: complex, Delta_d: complex,
                                   target_doping: float, mu: float,
                                   tx: float, ty: float,
                                   g_J: float, g_Delta_s: float, g_Delta_d: float,
                                   _bdg_cache: tuple = None,
                                   _vertex_cache: dict = None) -> Tuple[float, float, dict]:
        """
        Gap equation with q-dependent RPA pairing vertex.

            Δ_s_new = g_Δs · V_s_scale · V_s(q=0) · Σ_{k'} w(k') · F_AA(k')
            Δ_d_new = g_Δd · V_d_scale · Σ_{k'} w(k') · [Σ_k φ(k)·V(k-k')] · F_AB(k')

        V(q) = Tr[χ_RPA(q)[0:2, 2:4]] — Γ₆⊗Γ₇ off-diagonal block of the orbital RPA.

        • s-wave: single q=0 RPA vertex (k-independent, one χ₀ call).
        • d-wave: full q-dependent FS projection (expensive).
        • If |Δ_d| < 1e-4, skip d-vertex update and reuse the last cached result to speed up early SCF iterations.
        
        • _vertex_cache stores {'V_s','V_d_proj','fs_idx','phi_d'} and the state where it was computed.
        • Reused when |ΔM| < 0.03, |ΔΔ| < 0.008, and iter % 10 ≠ 0.
        • On cache hit, only BdG amplitudes are updated; the vertex is treated as slow-varying.

        Returns (Delta_s_new, Delta_d_new, updated_vertex_cache).
        """
        p   = self.solver.p
        slv = self.solver

        # --- BdG amplitudes on the full k-grid ---
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = self.diag_all_k(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)

        arg = np.clip(ev / p.kT, -100, 100)
        f12 = 1.0 - 2.0 / (1.0 + np.exp(arg))   # (N_k, 16)

        uA = ec[:, 0:4,  :]
        uB = ec[:, 4:8,  :]
        vA = ec[:, 8:12, :]
        vB = ec[:, 12:16,:]

        # Pairing amplitudes on full k-grid: F_AA(k), F_AB(k)
        pair_s_k = np.sum(
            (uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
           + uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])) * f12,
            axis=1)   # (N_k,)
        pair_d_k = np.sum(
            0.5 * (uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
                 + uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])) * f12,
            axis=1)   # (N_k,)

        # --- Fermi-surface sampling (needed for both channels) ---
        near_fs = np.any(np.abs(ev) < 3.0 * p.kT, axis=1)
        fs_idx  = np.where(near_fs)[0]
        if len(fs_idx) == 0:
            fs_idx = np.arange(min(32, slv.N_k))
        fs_idx  = fs_idx[:32]           # Sample FS k-points: 32 for d-channel
        fs_pts  = slv.k_points[fs_idx]  # (N_fs, 2)
        N_fs    = len(fs_pts)
        a       = p.a
        phi_d   = np.cos(fs_pts[:, 0] * a) - np.cos(fs_pts[:, 1] * a)  # (N_fs,)

        F_AA_fs = np.real(pair_s_k[fs_idx])   # (N_fs,)
        F_AB_fs = np.real(pair_d_k[fs_idx])   # (N_fs,)
        w_fs    = slv.k_weights[fs_idx]        # (N_fs,)

        # --- RPA vertex cache invalidation ---
        # The RPA vertex is expensive (full χ₀ loop over FS pairs), so it is reused unless:
        # |ΔM| > 0.03 or |ΔΔ| > 0.008         (absolute drift)
        # |ΔΔ|/Δ > ε_rel                      (large relative change near Δ → 0)
        #  _near_critical flag is set         (λ_max ≈ 1 ⇒ strong Stoner sensitivity)
        # Near criticality small state changes cause O(1) kernel shifts
        # (1 − U·χ₀ small), so caching is disabled to avoid false convergence.
        _near_critical_flag = bool(_vertex_cache.get('near_critical', False)) if _vertex_cache else False
        _cache_M     = _vertex_cache.get('M',     float('nan')) if _vertex_cache else float('nan')
        _cache_Delta = _vertex_cache.get('Delta', float('nan')) if _vertex_cache else float('nan')
        _cache_fs    = _vertex_cache.get('fs_idx', None)        if _vertex_cache else None
        Delta_eff    = abs(Delta_s) + abs(Delta_d)

        _delta_rel_change = (abs(Delta_eff - _cache_Delta) / max(abs(_cache_Delta), 1e-6)
                             if _vertex_cache else float('inf'))

        _vertex_stale = (
            _vertex_cache is None
            or _near_critical_flag                         # near-critical → always recompute
            or abs(M - _cache_M)        > 0.03             # absolute M drift
            or abs(Delta_eff - _cache_Delta) > 0.008       # absolute Δ drift
            or _delta_rel_change        > 0.15             # relative Δ change > 15 %
            or (_cache_fs is None)
            or (len(_cache_fs) != len(fs_idx))
            or not np.array_equal(_cache_fs, fs_idx)
        )

        if _vertex_stale:
            _, U_mat  = slv._u_eff_and_interaction_matrix(Q, g_J, target_doping)
            I4        = np.eye(4, dtype=complex)

            # k-grid BdG cache for chi0_tensor (even grid, q-independent half)
            E_k_cache = self.diag_kpts(slv.k_points_even, M, Q, Delta_s, Delta_d,
                                        target_doping, mu, tx, ty, g_J)

            def _rpa_project(chi0_mat: np.ndarray) -> float:
                """Γ₆⊗Γ₇ off-diagonal trace of χ_RPA = (I − U·χ₀)⁻¹·χ₀."""
                denom       = I4 - U_mat @ chi0_mat
                Ud, sd, Vhd = np.linalg.svd(denom)
                sd_reg      = np.maximum(sd, p.rpa_cutoff)
                denom_inv   = (Vhd.conj().T * (1.0 / sd_reg)) @ Ud.conj().T
                chi_rpa     = denom_inv @ chi0_mat
                return float(np.real(np.trace(chi_rpa[0:2, 2:4])))

            # --- s-channel — single q=0 vertex ---
            chi0_q0  = slv.compute_chi0_tensor(np.zeros(2), M, Q, Delta_s, Delta_d,
                                                target_doping, mu, tx, ty, g_J,
                                                _E_k_cache=E_k_cache)
            V_s_scalar = _rpa_project(chi0_q0)

            # --- d-channel — q-dependent vertex, but only if Delta_d nucleated ---
            if abs(Delta_d) > 1e-4:
                # Unique q = k_i − k_j on FS (upper triangle, deduplicated)
                ij_list, q_list = [], []
                for i in range(N_fs):
                    for j in range(i, N_fs):
                        q_raw = fs_pts[i] - fs_pts[j]
                        ij_list.append((i, j))
                        q_list.append((q_raw + np.pi) % (2.0 * np.pi) - np.pi)
                q_arr  = np.array(q_list)
                q_keys = [f"{r[0]:.5f},{r[1]:.5f}" for r in np.round(q_arr, 5)]
                u_keys, inv_idx = np.unique(q_keys, return_inverse=True)
                u_q_map = {}
                for fi, key in enumerate(q_keys):
                    if key not in u_q_map:
                        u_q_map[key] = q_arr[fi]
                u_q_vecs = np.array([u_q_map[k] for k in u_keys])

                V_rpa = np.empty(len(u_keys), dtype=float)
                for ui, q_u in enumerate(u_q_vecs):
                    chi0_mat  = slv.compute_chi0_tensor(q_u, M, Q, Delta_s, Delta_d,
                                                         target_doping, mu, tx, ty, g_J,
                                                         _E_k_cache=E_k_cache)
                    V_rpa[ui] = _rpa_project(chi0_mat)

                V_mat = np.zeros((N_fs, N_fs))
                for fi, (i, j) in enumerate(ij_list):
                    v = V_rpa[inv_idx[fi]]
                    V_mat[i, j] = v
                    V_mat[j, i] = v

                V_d_proj = phi_d @ V_mat   # (N_fs,)
            else:
                # Delta_d not yet nucleated: use a uniform fallback to avoid expensive loop
                V_d_proj = np.full(N_fs, V_s_scalar)

            _vertex_cache = {
                'M':          M,
                'Delta':      Delta_eff,
                'fs_idx':     fs_idx.copy(),
                'V_s_scalar': V_s_scalar,
                'V_d_proj':   V_d_proj.copy(),
                'phi_d':      phi_d.copy(),
                'near_critical': False,  # reset; SCF loop overwrites via inject_near_critical
            }
        else:
            # Cache hit: reuse vertex, update only FS weights (BdG amplitudes recomputed above)
            V_s_scalar = _vertex_cache['V_s_scalar']
            V_d_proj   = _vertex_cache['V_d_proj']
            phi_d      = _vertex_cache['phi_d']

        # --- Channel s: Δ_s_new = g_Δs · V_s_scale · V_s(q=0) · Σ_{k'} w·F_AA ---
        V_s_eff     = p.V_s_scale * V_s_scalar
        Delta_s_new = abs(g_Delta_s * V_s_eff * float(np.dot(w_fs, F_AA_fs)) / 4.0)

        # --- Channel d: Δ_d_new = g_Δd · V_d_scale · Σ_{k'} w · [Σ_k φ(k)·V(k-k')] · F_AB(k') ---
        integrand   = p.V_d_scale * V_d_proj * F_AB_fs   # (N_fs,)
        Delta_d_new = abs(g_Delta_d * float(np.dot(w_fs, integrand)) / 4.0)
        return Delta_s_new, Delta_d_new, _vertex_cache

# =============================================================================
# # 7. BAYESIAN OPTIMISER
# =============================================================================

def run_scf_material(solver: 'RMFT_Solver',
                     target_doping: float,
                     Delta_tetra:   float,
                     u:             float,
                     g_JT:          float,
                     V_s_scale:     float = 1.0,
                     V_d_scale:     float = 1.0,
                     initial_M:     float = 0.25,
                     initial_Q:     float = 1e-4,
                     initial_Delta: float = 0.04,
                     verbose:       bool  = False) -> Dict:
    """Run SCF for material parameters (Delta_tetra, u, g_JT, doping).

    Calls __post_init__ to recompute Delta_CF, U, K_lattice consistently.
    alpha_K is inherited unchanged from solver.p.alpha_K. It is a fixed design parameter (not Bayesian) enforcing K < K_spont.
    Delta_tetra or g_JT update Delta_CF and K_lattice in __post_init__, but alpha_K = K_spont / K stays fixed.
    """
    s = copy.copy(solver)
    s.p = copy.copy(solver.p)
    s.p.Delta_tetra = float(Delta_tetra)
    s.p.u           = float(u)
    s.p.g_JT        = float(g_JT)
    s.p.V_s_scale   = float(V_s_scale)
    s.p.V_d_scale   = float(V_d_scale)
    s.p.__post_init__()   # recomputes Delta_CF, U, K_lattice = g²/(alpha_K·Delta_CF)
    s._vbdg = None

    return s.solve_self_consistent(
        target_doping=target_doping,
        initial_M=initial_M, initial_Q=initial_Q, initial_Delta=initial_Delta,
        verbose=verbose)

class OptimPoint:
    """Evaluated point in (doping, Delta_tetra, u, g_JT) space with physics diagnostics.

    Diagnostics (all derived analytically from SCF output, no numeric derivatives):
      lambda_JT  : (g_eff^2/K)*chi_tau -- JT feedback strength; viable in (0.05, 1.0)
      lambda_max : largest eigenvalue of linearised gap matrix (>1 -> gap instability)
      stoner_ok  : AFM Stoner criterion not exceeded
    """
    __slots__ = ('doping', 'Delta_tetra', 'u', 'g_JT', 'V_s_scale', 'V_d_scale',
                 'Delta_total', 'converged', 'result',
                 'lambda_JT', 'lambda_max', 'stoner_ok', 'score')

    def __init__(self, doping, Delta_tetra, u, g_JT,
                 Delta_total, converged, result=None,
                 V_s_scale=1.0, V_d_scale=1.0,
                 lambda_JT=0.0, lambda_max=0.0, stoner_ok=True, score=0.0):
        self.doping      = doping
        self.Delta_tetra = Delta_tetra
        self.u           = u
        self.g_JT        = g_JT
        self.V_s_scale   = V_s_scale
        self.V_d_scale   = V_d_scale
        self.Delta_total = Delta_total
        self.converged   = converged
        self.result      = result
        self.lambda_JT   = lambda_JT
        self.lambda_max  = lambda_max
        self.stoner_ok   = stoner_ok
        self.score       = score

    def __repr__(self):
        regime = ('SC-trig' if 0.05 < self.lambda_JT < 1.0
                  else ('spont?' if self.lambda_JT >= 1.0 else 'closed'))
        return (f"OptimPoint(doping={self.doping:.3f}, Dt={self.Delta_tetra:.3f}, "
                f"u={self.u:.2f}, g={self.g_JT:.3f}, "
                f"Delta={self.Delta_total:.5f}, score={self.score:.5f}, "
                f"lJT={self.lambda_JT:.3f}[{regime}])")

class BayesianOptimizer:
    """GP Bayesian optimiser over the 3D material space (Delta_tetra, u, g_JT).

    Material parameters fix the Hamiltonian structure:
      • Γ₆–Γ₇ splitting Δ_CF  → multipolar rigidity          (from Delta_tetra)
      • Hubbard U              → AFM superexchange strength    (from u)
      • e-ph coupling g_JT     → JT pairing channel amplitude  (from g_JT)

    V_s_scale and V_d_scale are held fixed at 1.0 in this phase; channel strengths
    are refined independently by ChannelOptimizer (Phase 2) after the best material point is found.

    Doping δ is an *experimental control variable*: for each material we scan δ
    independently and return the **maximum** Δ over the doping range.

        score = Delta_total
                * (1 if stoner_ok else W_STONER)
                * (1 if converged else 0.1)
    """
    W_STONER_BAD   = 0.20

    def __init__(self,
                 solver,
                 doping_bounds      = (0.08, 0.30),
                 Delta_tetra_bounds = (-0.30, -0.05),
                 u_bounds           = (4.0, 8.0),
                 gJT_bounds         = (0.4, 1.2),
                 n_doping_scan      = 5):
        """
        3D Bayesian optimiser over material space (Delta_tetra, u, g_JT).

        V_s_scale = V_d_scale = 1.0 are held fixed here; channel strengths are
        refined in a separate 2D pass by ChannelOptimizer after the best material
        point is found.  This avoids the alpha_K × V_scale degeneracy that would
        make a 5D GP poorly conditioned with O(200) evaluations.

        Parameters
        ----------
        doping_bounds      : (lo, hi)  range searched *inside* each material trial
        Delta_tetra_bounds : (lo, hi)  BO dimension 1
        u_bounds           : (lo, hi)  BO dimension 2  (U/t0)
        gJT_bounds         : (lo, hi)  BO dimension 3
        n_doping_scan      : doping grid points per material trial
        n_workers is always set to the maximum available CPU count (os.cpu_count()).
        """
        self.solver             = solver
        self.doping_bounds      = doping_bounds
        self.Delta_tetra_bounds = Delta_tetra_bounds
        self.u_bounds           = u_bounds
        self.gJT_bounds         = gJT_bounds
        self.n_doping_scan      = n_doping_scan
        self.n_workers          = os.cpu_count() or 1
        self.observations: List[OptimPoint] = []
        self._gp = None

        print(f"  BayesianOptimizer (3D material space, V_s=V_d=1 fixed):")
        print(f"    Δ_tet ∈ {Delta_tetra_bounds},  u ∈ {u_bounds},  g_JT ∈ {gJT_bounds}")
        print(f"    inner doping scan: {n_doping_scan} points ∈ {doping_bounds}")
        print(f"    parallel workers: {self.n_workers} (all available CPUs)")

        if _SKLEARN_AVAILABLE:
            # 3-dimensional kernel: (Delta_tetra, u, g_JT)
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=[0.1, 0.5, 0.2], nu=2.5)
                      + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 0.1)))
            self._gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6,
                n_restarts_optimizer=5, normalize_y=True)

    # ------------------------------------------------------------------
    # Normalisation helpers — 3D material space only
    # ------------------------------------------------------------------

    def _normalize(self, Delta_tetra, u, gJT):
        """Map (Delta_tetra, u, gJT) → [0,1]³."""
        return np.array([
            (Delta_tetra - self.Delta_tetra_bounds[0]) / (self.Delta_tetra_bounds[1] - self.Delta_tetra_bounds[0]),
            (u           - self.u_bounds[0])           / (self.u_bounds[1]           - self.u_bounds[0]),
            (gJT         - self.gJT_bounds[0])         / (self.gJT_bounds[1]         - self.gJT_bounds[0]),
        ])

    def _denormalize(self, x):
        """Map [0,1]³ → (Delta_tetra, u, gJT)."""
        Delta_tetra = self.Delta_tetra_bounds[0] + x[0] * (self.Delta_tetra_bounds[1] - self.Delta_tetra_bounds[0])
        u           = self.u_bounds[0]           + x[1] * (self.u_bounds[1]           - self.u_bounds[0])
        gJT         = self.gJT_bounds[0]         + x[2] * (self.gJT_bounds[1]         - self.gJT_bounds[0])
        return float(Delta_tetra), float(u), float(gJT)

    def _lhs_sample(self, n):
        """Latin Hypercube sample in [0,1]³ (3D material space)."""
        rng = np.random.default_rng(seed=42)
        samples = np.zeros((n, 3))
        for j in range(3):
            perm = rng.permutation(n)
            samples[:, j] = (perm + rng.uniform(size=n)) / n
        return samples

    # ------------------------------------------------------------------
    # Inner doping scan — physics core
    # ------------------------------------------------------------------

    def _evaluate_material(self, Delta_tetra: float, u: float, gJT: float,
                           Vs: float = 1.0, Vd: float = 1.0) -> 'OptimPoint':
        # Vs=1.0, Vd=1.0 fixed in Phase 1. ChannelOptimizer refines these in Phase 2.
        """
        Evaluate a material defined by (Delta_tetra, u, gJT) by scanning
        n_doping_scan doping values and returning the best OptimPoint.

        Physical rationale
        ------------------
        1. Material params fix the Hamiltonian: Δ_CF, U, K_lattice.
        2. Doping δ only shifts μ and Gutzwiller factors.
        3. For a given material we want the δ that maximises Δ(δ).
        4. The GP sees only the per-material best score — no redundant
           evaluations of the same (Δ_tet, u, g) with different dopings.

        Warm-starting: each doping step initialises from the previous
        converged solution (ordered from low δ to high δ), so the SCF
        spends fewer iterations near convergence.
        """
        doping_grid = np.linspace(
            self.doping_bounds[0], self.doping_bounds[1], self.n_doping_scan)

        print(f"\n  ── Material: Δ_tet={Delta_tetra:.3f}  u={u:.2f}  g={gJT:.3f}  δ∈[{self.doping_bounds[0]:.2f},{self.doping_bounds[1]:.2f}] ({self.n_doping_scan}pts) ──")

        # G-analysis prefilter at mid-doping.
        # If G is stable everywhere (λ_min >> 0), the SCF will not find SC → skip doping scan.
        # Hard exclusion threshold λ_min > 2.0: safely far from instability.
        # Soft exclusion (0 < λ_min ≤ 2.0): proceed — G is analytic/approximate,
        # SCF may still find SC at a different doping within the scan range.
        # Returning score=0 (not NaN/exception) to the GP can learn that this region is unobserved.
        d_mid   = 0.5 * (self.doping_bounds[0] + self.doping_bounds[1])
        scout_g = self._cheap_scout(d_mid, Delta_tetra, u, gJT)
        gamma   = scout_g.get('G', {})
        lam_min = scout_g.get('lambda_min', 1.0)
        Tc_est  = scout_g.get('Tc_estimate', 0.0)
        print(f"     G-analysis: λ_min={lam_min:.3f}  G22={gamma.get('G22',1):.3f}"
              f"  Tc_est={Tc_est*1000:.1f}meV  viable={'✓' if scout_g['viable'] else '✗'}")

        if not scout_g['viable'] and lam_min > 2.0:
            print(f"     ⚡ G-filter: excluded (λ_min={lam_min:.2f} >> 0) — returning score=0")
            return OptimPoint(
                doping=d_mid, Delta_tetra=Delta_tetra, u=u, g_JT=gJT,
                Delta_total=0.0, converged=False, result=None,
                V_s_scale=Vs, V_d_scale=Vd,
                lambda_JT=0.0, lambda_max=0.0, stoner_ok=True, score=0.0)
        best_pt: Optional['OptimPoint'] = None
        prev_result: Optional[Dict] = None

        for i, doping in enumerate(doping_grid):
            # Warm-start from previous doping step
            if prev_result is not None:
                init_M     = prev_result.get('M', 0.1)
                init_Q     = prev_result.get('Q', 1e-4)
                init_Delta = prev_result.get('Delta_s', 0.0) + prev_result.get('Delta_d', 0.0)
                init_Delta = max(init_Delta, 1e-4)
            else:
                init_M, init_Q, init_Delta = 0.1, 1e-4, 1e-4

            pt = self._evaluate_point(
                doping, Delta_tetra, u, gJT, Vs, Vd,
                initial_M=init_M, initial_Q=init_Q, initial_Delta=init_Delta)
            
            # Keep result for warm-starting the next doping
            if pt.result:
                prev_result = pt.result

            if best_pt is None or pt.score > best_pt.score:
                best_pt = pt

        assert best_pt is not None
        print(f"     ↳ best: δ={best_pt.doping:.3f}  Δ={best_pt.Delta_total:.5f}"
              f"  score={best_pt.score:.5f}  λ_JT={best_pt.lambda_JT:.3f}"
              f"  {'✓' if best_pt.converged else '⚠'}"
              f"  [G: λ_min={lam_min:.3f}  Tc≈{Tc_est*1000:.1f}meV]")
        return best_pt

    # --- physics-aware scout (no SCF, one BdG diag + cluster ED) ---

    def _cheap_scout(self, doping: float, Delta_tetra: float,
                     u: float, gJT: float) -> Dict:
        """
        Estimate the critical coupling λ_scout and the SC-JT viability score without a self-consistent loop using a single BdG diagonalization (at M₀, Q=0, Δ=0) plus one cluster ED step

          1. Δ_CF from the full SOC+CF Hamiltonian (not a linear approximation).
          2. Gutzwiller g_t, g_J at the given doping → t_eff, J_eff are
             nonlinear in δ and vanish correctly at half-filling.
          3. N₀ from the actual 2D band structure at (M₀, Q=0) via the real BdG DOS,
             capturing van-Hove singularities and the Fermi-surface reconstruction by M₀.
          4. RPA Stoner factor 1/(1−U_eff·χ₀) computed with the real χ₀ formula (not N₀ as a proxy),
             so the near-instability divergence is correctly included.
          5. Magnetic suppression of the JT channel:
             χ_τ ~ N₀/(1 + α_M·M₀²), where α_M = max(U/t_eff/2.35−1, 0).
             High M₀ (strong AFM) closes the τ_x channel even when λ_BCS looks large.
          6. Cluster ED → self-consistent M₀ at Δ=Q=0, so we use a
             physically motivated M₀ instead of a fixed guess.

        Returns
        -------
        dict with:
          'lambda_scout'  – effective coupling including RPA and τ_x suppression
          'rpa_factor'    – Stoner enhancement
          'chi_tau'       – multipolar susceptibility (JT channel open if > threshold)
          'M0'            – cluster-ED equilibrium magnetisation
          'Delta_CF'      – Γ₆–Γ₇ gap for this (Delta_tetra, gJT) point
          'viable'        – bool: True if λ_scout > 0.15 AND χ_τ > 0.05 AND stoner_ok
        """
        p = self.solver.p

        # --- 1. Build a temporary params copy with the trial material parameters ---
        p2 = copy.copy(p)
        p2.Delta_tetra = Delta_tetra
        p2.u           = u
        p2.g_JT        = gJT
        p2.__post_init__()          # recomputes Delta_CF, U, U_mf, K_lattice

        Delta_CF = max(p2.Delta_CF, 1e-4)

        # --- 2. Gutzwiller factors ---
        abs_d = max(abs(doping), 1e-6)
        g_t   = (2.0 * abs_d) / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d)**2
        t_eff = g_t * p2.t0

        # --- 3. Rough cluster-ED equilibrium M₀ at Q=0, Δ=0 for the normal AFM state ---
        #   We iterate the 16×16 cluster a few times (cheap: no k-space).
        tx_bare = p2.t0          # Q=0 → no anisotropy
        J_eff   = g_J * 4.0 * tx_bare**2 / p2.U * (abs_d / (abs_d + p2.doping_0))
        mu_est  = 0.5 * Delta_CF - 2.0 * g_t * p2.t0 * np.tanh(doping / 0.1)

        # Build a minimal RMFT_Solver-like cluster (no k-grid needed here)
        H_sp  = np.zeros((4, 4), dtype=complex)
        H_sp[2, 2] = Delta_CF; H_sp[3, 3] = Delta_CF   # CF on Γ₇
        np.fill_diagonal(H_sp, H_sp.diagonal() - mu_est)

        I4      = np.eye(4, dtype=complex)
        O_sz    = np.diag(np.array([1.0, -1.0, p2.eta, -p2.eta]))
        O_A_full = np.kron(O_sz, I4)
        O_B_full = np.kron(I4, O_sz)
        f_d  = abs_d / (abs_d + p2.doping_0)   # doping suppression factor
        M0 = 0.35
        for _ in range(12):
            t2  = tx_bare**2
            h6  = g_J * f_d * (p2.U_mf / 2.0 + p2.Z * 2.0 * t2 / p2.U) * M0 / 2.0
            H_A = H_sp.copy()
            H_A[0,0] -= h6;  H_A[1,1] += h6
            H_A[2,2] -= p2.eta*h6;  H_A[3,3] += p2.eta*h6
            H_B = H_sp.copy()
            H_B[0,0] += h6;  H_B[1,1] -= h6
            H_B[2,2] += p2.eta*h6;  H_B[3,3] -= p2.eta*h6
            H_cl = (np.kron(H_A, I4) + np.kron(I4, H_B)
                    + J_eff * np.kron(O_sz, O_sz))
            ev, ec = eigh(H_cl)
            bw  = np.exp(-np.clip(ev - ev[0], 0, 500) / max(p2.kT, 1e-10))
            bw /= bw.sum()
            mA_val = float(np.real(np.einsum('n,in,ij,jn->', bw, ec.conj(), O_A_full, ec)))
            mB_val = float(np.real(np.einsum('n,in,ij,jn->', bw, ec.conj(), O_B_full, ec)))
            M0_new = abs(mA_val - mB_val) / 2.0
            if abs(M0_new - M0) < 1e-4:
                M0 = M0_new; break
            M0 = 0.6 * M0 + 0.4 * M0_new

        # --- 4. χ₀(q_AFM) via analytic 2D tight-binding ---
        #   χ₀ ≈ N₀ / (1 + (h_afm / (π·t_eff))²)
        #   AFM gap (~h_afm) Lorentzian-broadens the q_AFM pole, weak AFM: χ₀→N₀; strong AFM: χ₀→0 (h_afm < bandwidth).
        N0     = 1.0 / (np.pi * max(t_eff, 1e-6))
        h_afm  = g_J * (p2.U_mf / 2.0 + p2.Z * 2.0 * tx_bare**2 / p2.U) * M0 / 2.0
        chi0   = N0 / (1.0 + (h_afm / max(np.pi * t_eff, 1e-6))**2)

        # --- 5. RPA Stoner factor ---
        U_eff  = g_J * J_eff
        stoner_denom = max(1.0 - U_eff * chi0, p2.rpa_cutoff)
        rpa    = 1.0 / stoner_denom
        stoner_ok = (stoner_denom > 0)

        # --- 6. Pairing V_eff and λ_BCS ---
        V_eff  = (p2.g_JT**2 / max(p2.K_lattice, 1e-9)) * rpa
        lambda_bcs = V_eff * N0

        # --- 7. τ_x channel suppression by AFM order ---
        #   α_M = max(U/t_eff / 2.35 − 1, 0)  (2.35 = (U/t)_c on 2D square lattice)
        Ut_ratio = p2.U / max(t_eff, 1e-6)
        alpha_M  = max(Ut_ratio / 2.35 - 1.0, 0.0)
        chi_tau  = N0 / (1.0 + alpha_M * M0**2)

        # --- 8. Composite scout score ---
        #   λ_scout = λ_BCS × (χ_τ / N₀)   → reduces to λ_BCS when AFM is weak,
        #   is suppressed when strong AFM closes the τ_x channel.
        lambda_scout = lambda_bcs * (chi_tau / max(N0, 1e-10))

        viable = (lambda_scout > 0.15) and (chi_tau > 0.05 * N0) and stoner_ok        # --- 9. G-analysis: SC–JT instability matrix for Tc/Q_c boundary testing ---
        try:
            _tmp_solver = copy.copy(self.solver)
            _tmp_solver.p = p2
            _tmp_solver._vbdg = None
            G_res = _tmp_solver.compute_G_instability(doping, M=M0)
        except Exception:
            G_res = {'det_G': 1.0, 'lambda_min': 1.0, 'V_eff': V_eff,
                         'lambda_eff': lambda_bcs, 'unstable': False, 'Tc_estimate': 0.0,
                         'G11': 1.0, 'G22': 1.0}

        # G-viability: good if G₂₂ > 0 (JT is not spontaneous) AND λ_min close to 0
        G_viable = (G_res['G22'] > 0.02) and (G_res['lambda_min'] < 1.5)

        return {
            'lambda_scout': lambda_scout,
            'lambda_bcs':   lambda_bcs,
            'rpa_factor':   rpa,
            'chi_tau':      chi_tau,
            'chi0':         chi0,
            'M0':           M0,
            'Delta_CF':     Delta_CF,
            'h_afm':        h_afm,
            'stoner_ok':    stoner_ok,
            'viable':       viable and G_viable,
            'G':            G_res,
            'G_viable':     G_viable,
            'lambda_min':   G_res['lambda_min'],
            'V_eff_gamma':  G_res['V_eff'],
            'Tc_estimate':  G_res['Tc_estimate'],
        }

    def _adaptive_seed_near_critical(self, n_refine: int,
                                     lambda_target: float = 1.0,
                                     sigma_lambda:  float = 0.40) -> np.ndarray:
        """
        Generate n_refine normalised [0,1]³ material-space points biased biased using _cheap_scout evaluated at mid-doping (no full SCF).
        Acceptance weight: w = w_lambda * w_viable,
           w_lambda = exp(-(λ_scout-λ_target)^2 / (2σ^2)),
           w_viable = 1 (viable) or 0.15 (non-viable).
        Favors λ_scout≈1, open τ_x channel (χ_τ above threshold), and positive Stoner denominator (away from magnetic QCP)
        """
        rng          = np.random.default_rng(seed=7)
        pts          = np.zeros((n_refine, 3))
        accepted     = 0
        max_tries    = n_refine * 800
        # Use midpoint doping for the scout viability estimate
        d_mid = 0.5 * (self.doping_bounds[0] + self.doping_bounds[1])

        for _ in range(max_tries):
            if accepted >= n_refine:
                break
            x = rng.uniform(size=3)
            dt, u, g = self._denormalize(x)
            scout = self._cheap_scout(d_mid, dt, u, g)

            w_lam    = np.exp(-0.5 * ((scout['lambda_scout'] - lambda_target)
                                       / sigma_lambda) ** 2)
            w_viable = 1.0 if scout['viable'] else 0.15
            w_total  = w_lam * w_viable

            if rng.uniform() < w_total:
                pts[accepted] = x
                accepted += 1

        if accepted < n_refine:
            pts[accepted:] = rng.uniform(size=(n_refine - accepted, 3))
        return pts

    # --- BC diagnostics (6 extra BdG calls per point) ---

    def _jt_coupling_strength(self, solver, result) -> dict:
        """
        Analytic JT feedback strength: lambda_JT = (g_eff^2/K) * chi_tau.

        chi_tau is already in the converged SCF result dict (computed analytically
        by _compute_chi_tau every iteration — no extra diagonalisation).

        Physical interpretation
        -----------------------
        lambda_JT < 0.05  : JT channel closed (strong AFM or large Delta_CF)
        lambda_JT in (0.05, 1.0) : SC-triggered regime (viable for cooperative loop)
        lambda_JT >= 1.0  : spontaneous JT risk (alpha_K guard may be insufficient)

        chi_tau already encodes the AFM suppression: it is small when M is large because BdG eigenvectors do not mix Gamma6-Gamma7 under strong spin order.
        """
        chi_tau   = result.get('chi_tau', 0.0)
        K         = solver.p.K_lattice
        lambda_JT = (solver.p.g_JT**2 / max(K, 1e-9)) * chi_tau

        jt_viable = 0.05 < lambda_JT < 1.0
        regime    = ('SC-triggered'     if jt_viable else
                     ('spontaneous risk' if lambda_JT >= 1.0 else 'JT channel closed'))
        return {
            'lambda_JT': float(lambda_JT),
            'chi_tau':   float(chi_tau),
            'jt_viable': jt_viable,
            'regime':    regime,
        }

    def _jt_causality_test(self, solver, result) -> dict:
        """
        Operational test of SC-triggered JT causality.

        Runs a second SCF from initial_Delta=0 (normal-state fixpoint).
        If Q->0 without SC but Q>0 with SC, the JT is SC-triggered (not spontaneous).

        Cost: one extra SCF run (~same as a single BO evaluation).
        Called only on top-N best points AFTER optimisation, not during.

        Returns
        -------
        sc_triggered : bool   -- True if JT is caused by SC condensate
        Q_normal     : float  -- JT distortion in normal-state fixpoint
        Q_sc         : float  -- JT distortion in SC fixpoint
        dQ           : float  -- SC-induced JT amplitude
        Delta_sc     : float  -- SC gap at the tested fixpoint
        """
        doping = result.get('target_doping', 0.15)
        M0     = result.get('M', 0.25)
        Q_sc   = abs(result.get('Q', 0.0))
        D_sc   = abs(result.get('Delta_s', 0.0)) + abs(result.get('Delta_d', 0.0))

        if D_sc < 1e-4 or Q_sc < 1e-5:
            return {'sc_triggered': False, 'Q_normal': 0.0, 'Q_sc': Q_sc,
                    'Delta_sc': D_sc, 'dQ': 0.0, 'note': 'SC gap or Q too small to test'}

        try:
            normal_result = solver.solve_self_consistent(
                target_doping=doping,
                initial_M=M0, initial_Q=1e-5, initial_Delta=0.0,
                verbose=False)
            Q_normal = abs(normal_result.get('Q', 0.0))
            D_normal = abs(normal_result.get('Delta_s', 0.0)) + abs(normal_result.get('Delta_d', 0.0))
        except Exception as e:
            return {'sc_triggered': False, 'Q_normal': float('nan'), 'Q_sc': Q_sc,
                    'Delta_sc': D_sc, 'dQ': float('nan'), 'note': f'normal-state SCF failed: {e}'}

        sc_triggered = (Q_normal < 0.5 * Q_sc) and (Q_sc > 1e-4) and (D_normal < 1e-3)
        return {
            'sc_triggered': sc_triggered,
            'Q_normal':     float(Q_normal),
            'Q_sc':         float(Q_sc),
            'Delta_sc':     float(D_sc),
            'dQ':           float(Q_sc - Q_normal),
            'note':         ('CONFIRMED: JT is SC-triggered' if sc_triggered
                             else 'WARNING: JT may be spontaneous'),
        }

    # --- scored objective ---

    def _score(self, Delta: float, converged: bool, result: dict, solver) -> float:
        """
        Physically motivated score — three analytic factors.

        score = Delta * jt_f * stoner_f * lam_f * conv_f

        jt_f    : lambda_JT = (g²/K)·chi_tau window; SC-triggered regime → peak at 0.65
        stoner_f: AFM stability penalty (0.20 if Stoner unstable)
        lam_f   : gap-equation eigenvalue penalty (lambda_max > 2 penalised)
        RPA enhancement is now internal to the gap equation (q-dependent vertex),
        so rpa_factor is no longer a separate score multiplier.
        """
        if Delta < 1e-8:
            return 0.0
        conv_f = 1.0 if converged else 0.10

        # --- lambda_JT: (g²/K)·chi_tau; SC-triggered window (0.05, 1.0) ---
        chi_tau   = result.get('chi_tau', 0.0) if result else 0.0
        lambda_JT = (solver.p.g_JT**2 / max(solver.p.K_lattice, 1e-9)) * chi_tau

        if lambda_JT < 0.05:
            jt_f = 0.10
        elif lambda_JT >= 1.0:
            jt_f = 0.30
        else:
            jt_f = float(np.clip(1.0 - 0.8 * ((lambda_JT - 0.65) / 0.65)**2, 0.30, 1.0))

        # --- Stoner stability ---
        stoner_ok = not (result.get('afm_unstable', False) if result else False)
        stoner_f  = 1.0 if stoner_ok else self.W_STONER_BAD

        # --- Gap-equation eigenvalue penalty ---
        lambda_max = result.get('lambda_max', 0.0) if result else 0.0
        lam_f      = float(np.clip(1.0 - 0.15 * max(0.0, lambda_max - 2.0), 0.05, 1.0))
        return Delta * conv_f * jt_f * stoner_f * lam_f

    # --- GP ---

    def _fit_gp(self):
        if not _SKLEARN_AVAILABLE or self._gp is None or len(self.observations) < 3:
            return
        X = np.array([self._normalize(o.Delta_tetra, o.u, o.g_JT)
                      for o in self.observations])
        y = np.array([o.score for o in self.observations])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(X, y)

    def _expected_improvement(self, X_cand, xi=0.01):
        if not _SKLEARN_AVAILABLE or self._gp is None:
            return np.random.rand(len(X_cand))
        from scipy.stats import norm
        y_best = max(o.score for o in self.observations)
        mu, sigma = self._gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        z  = (mu - y_best - xi) / sigma
        EI = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(EI, 0.0)

    def _next_point_via_EI(self, n_restarts=50):
        rng  = np.random.default_rng()
        cand = rng.uniform(size=(n_restarts * 200, 3))   # 3D material space
        EI   = self._expected_improvement(cand)
        return self._denormalize(cand[np.argmax(EI)])

    # --- evaluation ---

    def _evaluate_point(self, doping, Delta_tetra, u, gJT, Vs: float = 1.0, Vd: float = 1.0,
                         initial_M: float = 0.1,
                         initial_Q: float = 1e-4,
                         initial_Delta: float = 1e-4):
        t0 = _time.time()
        try:
            result    = run_scf_material(self.solver, doping, Delta_tetra, u, gJT, Vs, Vd,
                                          initial_M=initial_M, initial_Q=initial_Q,
                                          initial_Delta=initial_Delta, verbose=False)
            Delta     = result.get('Delta_s', 0.0) + result.get('Delta_d', 0.0)
            converged = result.get('converged', False)
        except Exception as e:
            print(f"    SCF error: {e}")
            result, Delta, converged = {}, 0.0, False

        # Analytic JT diagnostics from SCF result dict (no extra diagonalisation)
        jt_diag    = self._jt_coupling_strength(self.solver, result) if result else {}
        lambda_JT  = jt_diag.get("lambda_JT", 0.0)
        lambda_max = result.get("lambda_max", 0.0) if result else 0.0
        stoner_ok  = not (result.get("afm_unstable", False) if result else False)

        score   = self._score(Delta, converged, result, self.solver)
        regime  = jt_diag.get("regime", "n/a")
        conv_s  = "✓" if converged else "⚠"
        print(f"      δ={doping:.3f}  Δ={Delta:.5f}  score={score:.5f}"
              f"  λ_JT={lambda_JT:.3f}[{regime}]  {conv_s} ({_time.time()-t0:.1f}s)")
        return OptimPoint(doping, Delta_tetra, u, gJT, Delta, converged, result,
                          V_s_scale=Vs, V_d_scale=Vd,
                          lambda_JT=lambda_JT, lambda_max=lambda_max,
                          stoner_ok=stoner_ok, score=score)

    def _evaluate_batch(self, points):
        """
        Evaluate a list of (Delta_tetra, u, gJT) material points, each with an inner doping scan.
        Parallelism over materials; inner doping scan is always sequential for warm-starting consistency.
        """
        if self.n_workers <= 1 or len(points) == 1:
            return [self._evaluate_material(*p) for p in points]
        # Parallel path: each worker runs the full inner doping scan for one material.
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futs = {ex.submit(self._evaluate_material, *p): p for p in points}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    pt = fut.result()
                except Exception as e:
                    p_err = futs[fut]
                    print(f"    Parallel error for params={p_err}: {e}")
                    pt = OptimPoint(
                        doping=float(0.5*(self.doping_bounds[0]+self.doping_bounds[1])),
                        Delta_tetra=p_err[0], u=p_err[1], g_JT=p_err[2],
                        Delta_total=0.0, converged=False, score=0.0)
                results.append(pt)
        return results

    # --- progress bar ---

    @staticmethod
    def _progress_bar(done, total, elapsed_s, width=40, prefix=""):
        frac   = done / max(total, 1)
        filled = int(width * frac)
        bar    = "█" * filled + "░" * (width - filled)
        pct    = int(100 * frac)
        if done > 0 and elapsed_s > 0:
            eta_s = elapsed_s / done * (total - done)
            h, r  = divmod(int(eta_s), 3600)
            m, s  = divmod(r, 60)
            eta   = f"ETA {h}:{m:02d}:{s:02d}"
        else:
            eta = "ETA --:--:--"
        return f"\r{prefix}[{bar}] {done}/{total} {pct}% {int(elapsed_s//60)}m{int(elapsed_s%60):02d}s {eta}  "

    # --- main loop ---

    def optimize(self, n_initial=12, n_iterations=35, n_refine=-1, verbose=True):
        """
        Three-phase Bayesian optimisation over the 3D material space.

        Phase 1a — LHS seeding (n_initial material points).
        Phase 1b — Adaptive seeding near the SC critical manifold (n_refine points).
        Phase 2  — GP Expected Improvement (n_iterations acquisitions).

        Each material evaluation calls _evaluate_material which runs an inner
        doping scan (n_doping_scan SCF calculations) and returns the best δ.
        The GP is fitted on per-material best scores (3D input, scalar output).
        """
        if n_refine < 0:
            n_refine = max(0, n_initial // 3)
        total_mat  = n_initial + n_refine + n_iterations
        t_start    = _time.time()

        print(f"\n{'='*70}")
        print(f"PHASE 1 — BAYESIAN OPTIMISATION  (3D material space, V_s=V_d=1 fixed)")
        print(f"  seed={n_initial}  adapt={n_refine}  BO={n_iterations}")
        print(f"  Per-material doping scan: {self.n_doping_scan} points")
        print(f"  Total SCF runs (≤): {total_mat * self.n_doping_scan}")
        print(f"  Space: (Delta_tetra, u, g_JT)  |  Objective: max_δ scored Δ")
        print(f"{'='*70}")

        def _tick(done, prefix="BO "):
            sys.stdout.write(
                self._progress_bar(done, total_mat, _time.time() - t_start, prefix=prefix))
            sys.stdout.flush()

        # Phase 1a: LHS over 3D material space
        print(f"\n[Phase 1a] LHS material seeding ({n_initial} materials)")
        lhs_pts = [self._denormalize(x) for x in self._lhs_sample(n_initial)]
        for idx, pt3 in enumerate(lhs_pts):
            _tick(idx, "Seed ")
            self.observations.append(self._evaluate_material(*pt3))
        _tick(n_initial, "Seed ")
        print()

        # Phase 1b: adaptive near critical manifold
        if n_refine > 0:
            print(f"\n[Phase 1b] Adaptive seeding near λ_eff≈1 ({n_refine} materials)")
            adp_pts = [self._denormalize(x)
                       for x in self._adaptive_seed_near_critical(n_refine)]
            for idx, pt3 in enumerate(adp_pts):
                _tick(n_initial + idx, "Adapt")
                self.observations.append(self._evaluate_material(*pt3))
            _tick(n_initial + n_refine, "Adapt")
            print()

        seed_done = n_initial + n_refine

        # Phase 2: GP-guided EI over 3D material space
        print(f"\n[Phase 2] GP EI iterations ({n_iterations} material trials)")
        for i in range(n_iterations):
            self._fit_gp()
            dt, u, g = self._next_point_via_EI(n_restarts=30)
            _tick(seed_done + i, "BO   ")
            print(f"\n  [BO {i+1}/{n_iterations}] Δ_tet={dt:.3f}  u={u:.2f}  g={g:.3f}")
            self.observations.append(self._evaluate_material(dt, u, g))
            _tick(seed_done + i + 1, "BO   ")

            if verbose and (i + 1) % 5 == 0:
                best_s = max(self.observations, key=lambda o: o.score)
                print(f"\n  -- best score so far: {best_s}  --\n")
        print()

        best       = max(self.observations, key=lambda o: o.score)
        best_raw   = max(self.observations, key=lambda o: o.Delta_total)
        valid      = [o for o in self.observations if o.converged and o.lambda_JT > 0.0]
        best_valid = max(valid, key=lambda o: o.score) if valid else best
        elapsed    = _time.time() - t_start

        print(f"\n{'='*70}")
        print(f"DONE ({elapsed/60:.1f} min)  |  Scored best: {best}")
        if best_raw is not best:
            print(f"Raw Δ champion: {best_raw}")
        print(f"Optimal doping: δ = {best.doping:.4f}")
        print(f"{'='*70}\n")
        return {'best_point': best, 'best_valid': best_valid, 'best_raw': best_raw,
                'observations': self.observations, 'gp': self._gp, 'elapsed_s': elapsed}

class ChannelOptimizer:
    """
    Phase 2: 2D grid + GP search over (V_s_scale, V_d_scale) with fixed material
    point (Delta_tetra*, u*, g_JT*) and optimal doping δ*.

    Rationale for two-stage architecture
    ─────────────────────────────────────
    In Phase 1 (BayesianOptimizer) V_s_scale = V_d_scale = 1.0 are held fixed.
    This keeps the GP in a well-conditioned 3D space: with O(200) evaluations a
    3D GP fits accurately, whereas a 5D GP would be severely under-sampled.

    The two scale parameters are intentionally excluded from Phase 1 because:
      1. Degeneracy: λ_eff ∝ V_scale × α_K × Δ_CF, so V_scale and α_K are
         partially collinear → the GP cannot resolve them independently.
      2. Separability: V_scale shifts the BCS coupling strength λ uniformly
         across all (Δ_tet, u, g_JT) combinations.  The optimal *material*
         (phase-boundary topology, optimal-doping position) is V_scale-independent;
         only the amplitude of Δ changes.
      3. Smoothness: Δ(V_s, V_d) is a smooth, nearly monotone 2D surface.
         A coarse 5×5 grid (25 SCF) resolves the optimum; a GP with ~20 points
         suffices for sub-grid refinement.

    Search strategy
    ───────────────
    Step A: 5×5 uniform grid over [Vs_lo, Vs_hi] × [Vd_lo, Vd_hi]
    Step B: GP-guided EI refinement (n_refine ≈ 15 points) around the best grid cell
    Step C: Return argmax (V_s*, V_d*) and corresponding Δ_total, Δ_s, Δ_d

    Typical cost: (25 + 15) × 1 SCF = 40 SCF runs — negligible vs Phase 1.
    """

    def __init__(self, solver: 'RMFT_Solver',
                 best_material: 'OptimPoint',
                 Vs_bounds: Tuple[float, float] = (0.1, 2.5),
                 Vd_bounds: Tuple[float, float] = (0.1, 2.5),
                 n_grid:    int = 5,
                 n_refine:  int = 15):
        """
        Parameters
        ----------
        solver        : base RMFT_Solver (will be shallow-copied per trial)
        best_material : OptimPoint from Phase 1 (carries Delta_tetra, u, g_JT, doping)
        Vs_bounds     : search range for V_s_scale
        Vd_bounds     : search range for V_d_scale
        n_grid        : side of the initial uniform grid (n_grid² evaluations)
        n_refine      : GP EI refinement points after the grid
        """
        self.solver        = solver
        self.mat           = best_material
        self.Vs_bounds     = Vs_bounds
        self.Vd_bounds     = Vd_bounds
        self.n_grid        = n_grid
        self.n_refine      = n_refine
        self.observations: List[Tuple[float, float, float, Dict]] = []
        # (V_s, V_d, score, result_dict)

        self._gp = None
        if _SKLEARN_AVAILABLE:
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=[0.5, 0.5], nu=2.5)
                      + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 0.1)))
            self._gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6,
                n_restarts_optimizer=5, normalize_y=True)

        print(f"  ChannelOptimizer (2D channel space):")
        print(f"    V_s ∈ {Vs_bounds},  V_d ∈ {Vd_bounds}")
        print(f"    grid={n_grid}×{n_grid}={n_grid**2},  GP refine={n_refine}")
        print(f"    fixed: Δ_tet={best_material.Delta_tetra:.3f}  u={best_material.u:.2f}"
              f"  g_JT={best_material.g_JT:.3f}  δ={best_material.doping:.3f}")

    # ------------------------------------------------------------------
    def _normalize_ch(self, vs, vd):
        return np.array([
            (vs - self.Vs_bounds[0]) / (self.Vs_bounds[1] - self.Vs_bounds[0]),
            (vd - self.Vd_bounds[0]) / (self.Vd_bounds[1] - self.Vd_bounds[0]),
        ])

    def _denormalize_ch(self, x):
        vs = self.Vs_bounds[0] + x[0] * (self.Vs_bounds[1] - self.Vs_bounds[0])
        vd = self.Vd_bounds[0] + x[1] * (self.Vd_bounds[1] - self.Vd_bounds[0])
        return float(vs), float(vd)

    # ------------------------------------------------------------------
    def _run_scf(self, vs: float, vd: float) -> Tuple[float, Dict]:
        """Single SCF at fixed material point, given (V_s_scale, V_d_scale)."""
        mat = self.mat
        result = run_scf_material(
            self.solver,
            target_doping = mat.doping,
            Delta_tetra   = mat.Delta_tetra,
            u             = mat.u,
            g_JT          = mat.g_JT,
            V_s_scale     = vs,
            V_d_scale     = vd,
            initial_M     = 0.25,
            initial_Q     = 1e-4,
            initial_Delta = 0.04,
            verbose       = False,
        )
        if result.get('converged', False):
            ds = abs(result.get('Delta_s', 0.0))
            dd = abs(result.get('Delta_d', 0.0))
            score = ds + dd
        else:
            score = 0.0
        return score, result

    # ------------------------------------------------------------------
    def _fit_gp_ch(self):
        if not _SKLEARN_AVAILABLE or self._gp is None or len(self.observations) < 4:
            return
        X = np.array([self._normalize_ch(vs, vd)
                      for vs, vd, _, _ in self.observations])
        y = np.array([s for _, _, s, _ in self.observations])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(X, y)

    def _EI_ch(self, X_cand, xi=0.01):
        if not _SKLEARN_AVAILABLE or self._gp is None or len(self.observations) < 4:
            return np.random.rand(len(X_cand))
        from scipy.stats import norm
        y_best = max(s for _, _, s, _ in self.observations)
        mu, sigma = self._gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        z  = (mu - y_best - xi) / sigma
        return np.maximum((mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z), 0.0)

    def _next_ch(self):
        rng  = np.random.default_rng()
        cand = rng.uniform(size=(5000, 2))
        EI   = self._EI_ch(cand)
        return self._denormalize_ch(cand[np.argmax(EI)])

    # ------------------------------------------------------------------
    def optimize(self) -> Dict:
        """
        Run grid search + GP refinement.

        Returns dict with keys:
          V_s_scale, V_d_scale  : optimal channel scales
          Delta_s, Delta_d      : corresponding gap components
          Delta_total           : Δ_s + Δ_d
          score                 : same as Delta_total for channel opt
          result                : full SCF result dict
          grid_obs              : all (Vs, Vd, score) observations
        """
        t0 = _time.time()
        n  = self.n_grid
        total = n*n + self.n_refine

        print(f"\n{'='*60}")
        print(f"PHASE 2 — CHANNEL STRENGTH OPTIMISATION (2D)")
        print(f"{'='*60}")

        # ── Step A: uniform grid ──────────────────────────────────────
        print(f"\n[Phase 2a] {n}×{n} uniform grid ...")
        vs_grid = np.linspace(*self.Vs_bounds, n)
        vd_grid = np.linspace(*self.Vd_bounds, n)
        done = 0
        for vs in vs_grid:
            for vd in vd_grid:
                score, res = self._run_scf(vs, vd)
                self.observations.append((vs, vd, score, res))
                done += 1
                pct = done / total * 100
                sys.stdout.write(f"\r  grid [{done}/{n*n}]  Vs={vs:.2f}  Vd={vd:.2f}"
                                 f"  Δ={score:.5f}  [{pct:.0f}%]   ")
                sys.stdout.flush()
        print()

        # ── Step B: GP EI refinement ──────────────────────────────────
        print(f"\n[Phase 2b] GP EI refinement ({self.n_refine} points) ...")
        for i in range(self.n_refine):
            self._fit_gp_ch()
            vs, vd = self._next_ch()
            score, res = self._run_scf(vs, vd)
            self.observations.append((vs, vd, score, res))
            done += 1
            pct = done / total * 100
            sys.stdout.write(f"\r  GP [{i+1}/{self.n_refine}]  Vs={vs:.2f}  Vd={vd:.2f}"
                             f"  Δ={score:.5f}  [{pct:.0f}%]   ")
            sys.stdout.flush()
        print()

        # ── Step C: extract best ──────────────────────────────────────
        best_obs = max(self.observations, key=lambda o: o[2])
        vs_opt, vd_opt, score_opt, res_opt = best_obs

        elapsed = _time.time() - t0
        print(f"\n{'='*60}")
        print(f"Channel optimum found ({elapsed:.1f}s):")
        print(f"  V_s_scale = {vs_opt:.4f}   V_d_scale = {vd_opt:.4f}")
        ds = abs(res_opt.get('Delta_s', 0.0)) if res_opt else 0.0
        dd = abs(res_opt.get('Delta_d', 0.0)) if res_opt else 0.0
        print(f"  Δ_s = {ds:.6f} eV   Δ_d = {dd:.6f} eV   |Δ| = {ds+dd:.6f} eV")
        print(f"{'='*60}\n")

        return {
            'V_s_scale':   vs_opt,
            'V_d_scale':   vd_opt,
            'Delta_s':     ds,
            'Delta_d':     dd,
            'Delta_total': ds + dd,
            'score':       score_opt,
            'result':      res_opt,
            'grid_obs':    self.observations,
            'elapsed_s':   elapsed,
        }

def full_bayesian_optimize(solver: 'RMFT_Solver',
                            doping_bounds:      Tuple[float, float] = (0.08, 0.30),
                            Delta_tetra_bounds: Tuple[float, float] = (-0.30, -0.05),
                            u_bounds:           Tuple[float, float] = (4.0, 8.0),
                            gJT_bounds:         Tuple[float, float] = (0.4, 1.2),
                            Vs_bounds:          Tuple[float, float] = (0.1, 2.5),
                            Vd_bounds:          Tuple[float, float] = (0.1, 2.5),
                            n_doping_scan:      int = 5,
                            n_initial:          int = 15,
                            n_refine:           int = -1,
                            n_iterations:       int = 40,
                            n_ch_grid:          int = 5,
                            n_ch_refine:        int = 15) -> Dict:
    """
    Two-stage material + channel optimisation.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  Stage 1 — BayesianOptimizer (3D material space)                     ║
    ║    Parameters: (Δ_tet, u, g_JT)   V_s_scale = V_d_scale = 1          ║
    ║    Inner scan: n_doping_scan dopings per material trial              ║
    ║    Budget:    n_initial LHS + n_refine adaptive + n_iterations EI    ║
    ║                                                                      ║
    ║  Stage 2 — ChannelOptimizer (2D channel space)                       ║
    ║    Parameters: (V_s_scale, V_d_scale)   material fixed to Stage 1    ║
    ║    Grid:       n_ch_grid² uniform + n_ch_refine GP-EI points         ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Two-stage rationale
    ───────────────────
    λ_eff ∝ V_scale × α_K × Δ_CF  →  V_scale and α_K are partially collinear.
    A 5D GP with O(200) points cannot resolve this degeneracy.  Factoring into
    a 3D material search (O(200) evaluations, well-conditioned GP) followed by a
    2D channel search (~40 SCF, smooth surface, grid+GP sufficient) gives:
      • Phase 1: high-quality material space coverage
      • Phase 2: exact channel-strength optimum at negligible extra cost

    Parameters
    ----------
    doping_bounds      : inner doping scan range (NOT a BO dimension)
    Delta_tetra_bounds : Stage 1 BO range for tetragonal CF (eV)
    u_bounds           : Stage 1 BO range for U/t0 ratio
    gJT_bounds         : Stage 1 BO range for e-ph coupling (eV/Å)
    Vs_bounds          : Stage 2 search range for V_s_scale
    Vd_bounds          : Stage 2 search range for V_d_scale
    n_doping_scan      : doping grid points per material (5 recommended)
    n_initial          : Stage 1 LHS seed materials
    n_refine           : Stage 1 adaptive seeds near λ_eff≈1  (−1 → auto)
    n_iterations       : Stage 1 GP EI acquisitions
    n_ch_grid          : Stage 2 grid side (n_ch_grid² evaluations)
    n_ch_refine        : Stage 2 GP EI refinement points
    """
    t0_total = _time.time()

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 1: material search (V_s = V_d = 1 fixed)
    # ─────────────────────────────────────────────────────────────────────
    opt = BayesianOptimizer(solver,
                             doping_bounds=doping_bounds,
                             Delta_tetra_bounds=Delta_tetra_bounds,
                             u_bounds=u_bounds,
                             gJT_bounds=gJT_bounds,
                             n_doping_scan=n_doping_scan)
    res1  = opt.optimize(n_initial=n_initial, n_refine=n_refine,
                         n_iterations=n_iterations, verbose=True)
    best1 = res1['best_valid'] or res1['best_point']

    # SC-triggered JT causality test on top-5 Stage 1 points
    all_obs_sorted = sorted(res1['observations'], key=lambda o: o.score, reverse=True)
    causality_results = []
    print('\nSC-triggered JT causality test (Stage 1 top-5 points)...')
    for top_pt in all_obs_sorted[:5]:
        if top_pt.result and top_pt.converged:
            s = copy.copy(solver)
            s.p = copy.copy(solver.p)
            s.p.Delta_tetra = top_pt.Delta_tetra
            s.p.u           = top_pt.u
            s.p.g_JT        = top_pt.g_JT
            s.p.__post_init__()
            s._vbdg = None
            ctest = opt._jt_causality_test(s, top_pt.result)
            causality_results.append({'point': top_pt, 'causality': ctest})
            print(f"  lJT={top_pt.lambda_JT:.3f}  Δ={top_pt.Delta_total:.5f}  "
                  f"-> {ctest['note']}  (Q_sc={ctest['Q_sc']:.5f}, Q_norm={ctest['Q_normal']:.5f})")

    # ─────────────────────────────────────────────────────────────────────
    # STAGE 2: channel strength refinement at the best material point
    # ─────────────────────────────────────────────────────────────────────
    ch_opt = ChannelOptimizer(solver, best1,
                               Vs_bounds=Vs_bounds,
                               Vd_bounds=Vd_bounds,
                               n_grid=n_ch_grid,
                               n_refine=n_ch_refine)
    res2 = ch_opt.optimize()

    # ─────────────────────────────────────────────────────────────────────
    # Compose final best: material from Stage 1, channel from Stage 2
    # ─────────────────────────────────────────────────────────────────────
    elapsed_total = _time.time() - t0_total
    print(f"\nTotal optimisation time: {elapsed_total/60:.1f} min")
    print(f"  Stage 1 (material): {res1['elapsed_s']/60:.1f} min")
    print(f"  Stage 2 (channel):  {res2['elapsed_s']:.1f} s")

    return {
        # Stage 1 material optimum
        'best_mat':          best1,
        'best_valid':        res1['best_valid'],
        'best_raw':          res1['best_raw'],
        # Stage 2 channel optimum
        'V_s_scale':         res2['V_s_scale'],
        'V_d_scale':         res2['V_d_scale'],
        'Delta_s':           res2['Delta_s'],
        'Delta_d':           res2['Delta_d'],
        # Combined
        'Delta':             res2['Delta_total'],
        'score':             res2['score'],
        'doping':            best1.doping,
        'Delta_tetra':       best1.Delta_tetra,
        'u':                 best1.u,
        'g_JT':              best1.g_JT,
        'result':            res2['result'],
        'all_obs':           res1['observations'],
        'ch_obs':            res2['grid_obs'],
        'causality_results': causality_results,
        'elapsed_s':         elapsed_total,
    }

# =============================================================================
# 8. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_phase_diagrams(solver: RMFT_Solver, initial_M: float, initial_Q: float, initial_Delta: float, doping_range: np.ndarray,
                        cf_min: float = 0.05, cf_max: float = 0.20, N_cf: int = 10,
                        opt_result: Optional[Dict] = None):
    """
    Doping-scan phase diagram with warm-start and crystal-field sweet-spot search.

    opt_result : optional dict returned by full_bayesian_optimize.

    Layout without opt_result (3×3):
      [0,0] Phase diagram   [0,1] CF sweet-spot   [0,2] DOS
      [1,0] M(iter)         [1,1] Q(iter)          [1,2] Δ(iter)
      [2,0] F_bdg/cluster   [2,1] g_t, g_J         [2,2] density

    Layout with opt_result (4×3 — extra bottom row for BO panels):
      [3,0] BO progress     [3,1] δ vs Δ           [3,2] α_K vs Δ
      (V_s vs V_d split shown in [3,2] via a secondary annotation)
    """
    phase_data = {
        'target_doping': [], 'M': [], 'Q': [],
        'Delta_s': [], 'Delta_d': [],
        'mu': [], 'density': [], 'F_bdg': [],
        'chi_tau': [], 'Ut_ratio': []
    }

    print(f"\n{'='*70}")
    print(f"PHASE DIAGRAM SCAN  (n_points={len(doping_range)}, "
          f"δ: {doping_range[0]:.3f}→{doping_range[-1]:.3f})")
    print(f"  V_s_scale={solver.p.V_s_scale:.2f}  V_d_scale={solver.p.V_d_scale:.2f}  "
          f"U/t0={solver.p.u:.2f}  eta={solver.p.eta:.3f}  delta_0={solver.p.doping_0:.3f}")
    print(f"{'='*70}")

    all_results = []   # store every result for convergence history plots
    prev_result = None

    for i, target_doping in enumerate(doping_range):
        # Warm-start from previous converged solution
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

        # Phase classification
        has_afm = result['M'] > 0.15
        has_sc  = result['Delta_d'] > 2 * solver.p.kT
        has_jt  = abs(result['Q']) > 1e-4
        if   has_afm and not has_sc:  phase = 'AFM'
        elif has_sc  and has_jt:      phase = 'SC+JT'
        elif has_afm and has_sc:      phase = 'MIX'
        else:                         phase = 'NM'

        density_error = abs(result['density'] - (1 - target_doping))
        dens_warn = '⚠' if density_error > 0.01 else ' '
        chi_tau = result.get('chi_tau', 0.0)
        Ut_ratio = result.get('Ut_ratio', 0.0)

        print(f"  [{i+1:2d}/{len(doping_range)}] δ={target_doping:.3f}  "
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
    
    # -------------------------------------------------------------------------
    # Crystal field sweet spot search
    # -------------------------------------------------------------------------
    print("\n\nCrystal field sweet spot search:")
    print(f"Target ΔCF window: [{cf_min:.3f}, {cf_max:.3f}] eV  (scanned via Δ_tet)")

    ref_doping_idx = len(doping_range) // 2
    ref_doping = doping_range[ref_doping_idx]
    print(f"Reference doping: δ={ref_doping:.3f}")

    # Build a Delta_tetra grid whose resulting Delta_CF values span [cf_min, cf_max].
    # We sample N_cf candidate Delta_tetra values and keep those whose Delta_CF
    # falls in the requested window. The grid is constructed by inverting the
    # monotone relationship: decreasing |Delta_tetra| increases Delta_CF.
    # A fine pre-scan (200 points) identifies the correct Delta_tetra range.
    _dt_prescan  = np.linspace(-0.60, 0.10, 200)
    _cf_prescan  = np.array([
        _gamma_splitting(solver.p.lambda_soc, dt, solver.p.Delta_inplane)
        for dt in _dt_prescan
    ])
    # Find Delta_tetra values whose Delta_CF lies within [cf_min, cf_max]
    _mask        = (_cf_prescan >= cf_min) & (_cf_prescan <= cf_max)
    if _mask.sum() < 2:
        # Fallback: use full prescan range
        _mask = np.ones(len(_dt_prescan), dtype=bool)
    _dt_lo, _dt_hi = _dt_prescan[_mask][[0, -1]]
    # Build the actual scan grid (N_cf evenly spaced Delta_tetra values)
    dt_scan_grid = np.linspace(_dt_lo, _dt_hi, N_cf)
    # Compute the real Delta_CF for each grid point (used for axis labels)
    cf_scan_actual = np.array([
        _gamma_splitting(solver.p.lambda_soc, dt, solver.p.Delta_inplane)
        for dt in dt_scan_grid
    ])

    cf_gaps, cf_Q_values, cf_M_values, cf_actual_CF = [], [], [], []
    cf_previous = None

    for dt, cf_actual in zip(dt_scan_grid, cf_scan_actual):
        # Deep-copy params, set Delta_tetra, recompute all derived quantities.
        p_cf = copy.copy(solver.p)
        p_cf.Delta_tetra = float(dt)
        p_cf.__post_init__()   # recomputes Delta_CF, U_mf, K_lattice, t_pd, …

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

        cf_result = cf_solver.solve_self_consistent(
            target_doping=ref_doping,
            initial_M=init_M, initial_Q=init_Q, initial_Delta=init_Delta,
            verbose=False
        )
        cf_gaps.append(cf_result['Delta_d'])
        cf_Q_values.append(cf_result['Q'])
        cf_M_values.append(cf_result['M'])
        cf_actual_CF.append(cf_actual)
        cf_previous = {
            'M': cf_result['M'], 'Q': cf_result['Q'],
            'Delta_s': cf_result['Delta_s'], 'Delta_d': cf_result['Delta_d']
        }
        print(f"  Δ_tet={dt:+.4f} eV  ΔCF={cf_actual:.4f} eV → "
              f"Δs={cf_result['Delta_s']:.5f}  Δd={cf_result['Delta_d']:.5f} "
              f"Q={cf_result['Q']:+.5f}  M={cf_result['M']:.4f}")
        print(f"  χ₀(q_AFM) = {cf_result.get('chi0', float('nan')):.4f}"
              f"  |  RPA factor = {cf_result.get('rpa_factor', float('nan')):.3f}×")
        _irr = cf_result.get('irrep_info', {})
        print(f"  Irrep selection R = {_irr.get('selection_ratio', float('nan')):.4f} "
              f"JT {'ALLOWED ✓' if _irr.get('jt_algebraically_allowed', False) else 'BLOCKED ✗'}")

    # Use the actual Delta_CF values as the axis for the sweet-spot plot
    cf_range = np.array(cf_actual_CF)   # replaces the old linspace(cf_min, cf_max)

    max_gap_idx = np.argmax(cf_gaps)
    sweet_spot_cf = cf_range[max_gap_idx]
    max_gap = cf_gaps[max_gap_idx]
    print(f"\n✓ Sweet spot: ΔCF = {sweet_spot_cf:.3f} eV"
          f"  (Δ_tet = {dt_scan_grid[max_gap_idx]:+.4f} eV)"
          f"  Δmax = {max_gap:.4f} eV")

    # -------------------------------------------------------------------------
    # 3×3 Figure layout
    #
    #  [0,0] Phase diagram (M, Q, Δ vs doping)
    #  [0,1] Crystal-field sweet-spot scan (Δ, Q, M vs ΔCF)
    #  [0,2] DOS (last converged doping point)
    #
    #  [1,0] Convergence: M(iter) – one line per target_doping (coloured)
    #  [1,1] Convergence: Q(iter) – one line per target_doping (coloured)
    #  [1,2] Convergence: |Δ|(iter) – one line per target_doping (coloured)
    #
    #  [2,0] Free energy F_bdg & F_cluster (last doping point)
    #  [2,1] Gutzwiller factors g_t, g_J   (last doping point)
    #  [2,2] Density convergence n(iter)   (last doping point)
    # -------------------------------------------------------------------------
    n_rows = 4 if opt_result is not None else 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    fig.suptitle('SC-Activated JT Model – Full Results', fontsize=15, fontweight='bold')

    # Colour cycle for convergence lines (one colour per doping point)
    cmap = plt.cm.plasma
    n_dop = len(doping_range)
    colors = [cmap(i / max(n_dop - 1, 1)) for i in range(n_dop)]

    # --- [0,0] Phase diagram ---
    _plot_phase_data(axes[0, 0], phase_data)

    # --- [0,1] Crystal-field sweet-spot ---
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

    # --- [0,2] DOS (last doping point from doping scan) ---
    _plot_dos(axes[0, 2], solver, all_results[-1])

    # --- [1,0] Convergence: M ---
    ax = axes[1, 0]
    for idx, res in enumerate(all_results):
        ax.plot(res['history']['M'], color=colors[idx],
                linewidth=1.5, label=f'δ={doping_range[idx]:.3f}')
    ax.set_ylabel('Magnetization M', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title('SCF Convergence: M (per target doping)', fontsize=11)
    ax.legend(fontsize=7, ncol=max(1, n_dop // 4), loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- [1,1] Convergence: Q ---
    ax = axes[1, 1]
    for idx, res in enumerate(all_results):
        ax.plot(res['history']['Q'], color=colors[idx],
                linewidth=1.5, label=f'δ={doping_range[idx]:.3f}')
    ax.set_ylabel('JT Distortion Q (Å)', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title('SCF Convergence: Q (per target doping)', fontsize=11)
    ax.legend(fontsize=7, ncol=max(1, n_dop // 4), loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- [1,2] Convergence: |Δ| ---
    ax = axes[1, 2]
    for idx, res in enumerate(all_results):
        ax.plot(res['history']['Delta'], color=colors[idx],
                linewidth=1.5, label=f'δ={doping_range[idx]:.3f}')
    ax.set_ylabel('SC Gap |Δ| (eV)', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title('SCF Convergence: |Δ| (per target doping)', fontsize=11)
    ax.legend(fontsize=7, ncol=max(1, n_dop // 4), loc='upper right')
    ax.grid(True, alpha=0.3)

    # Last-doping history for diagnostic panels
    last_hist = all_results[-1]['history']
    last_label = f'δ={doping_range[-1]:.3f}'

    # --- [2,0] Free energy ---
    ax = axes[2, 0]
    ax.plot(last_hist['F_bdg'],     'k-',  linewidth=2,   label='F_bdg')
    ax.plot(last_hist['F_cluster'], 'r--', linewidth=1.5, alpha=0.8, label='F_cluster')
    ax.set_ylabel('Free Energy (eV)', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title(f'Free Energy (may be non-monotonic) [{last_label}]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- [2,1] Gutzwiller factors ---
    ax = axes[2, 1]
    ax.plot(last_hist['g_t'], 'c-', linewidth=2, label='g_t (kinetic)')
    ax.plot(last_hist['g_J'], 'm-', linewidth=2, label='g_J (exchange)')
    ax.set_ylabel('Renormalization Factor', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title(f'Gutzwiller Factors [{last_label}]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- [2,2] Density convergence ---
    ax = axes[2, 2]
    ax.plot(last_hist['density'], color='darkorange', linewidth=2)
    ax.axhline(1.0 - doping_range[-1], color='k', linestyle='--', linewidth=1,
               label=f'target n={1-doping_range[-1]:.3f}')
    ax.set_ylabel('Electron Density n', fontsize=11)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_title(f'Density Constraint Convergence [{last_label}]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # --- Optional row 3: Bayesian Optimisation panels ---
    if opt_result is not None:
        all_obs = opt_result.get('observations', opt_result.get('all_obs', []))
        if all_obs:
            deltas  = [o.Delta_total for o in all_obs]
            scores  = [getattr(o, 'score', o.Delta_total) for o in all_obs]
            dopings = [o.doping for o in all_obs]
            dt_vals = [o.Delta_tetra for o in all_obs]
            lJT_vals = [getattr(o, 'lambda_JT', 0.0) for o in all_obs]
            # green: SC-triggered JT viable (0.05 < λ_JT < 1.0)
            # orangered: spontaneous JT risk (λ_JT >= 1.0)
            # orange: JT channel closed (λ_JT <= 0.05)
            colours = ['green' if 0.05 < lj < 1.0 else ('orangered' if lj >= 1.0 else 'orange')
                       for lj in lJT_vals]

            running = np.maximum.accumulate(scores)
            best_idx = int(np.argmax(scores))

            ax_p = axes[3, 0]
            ax_p.plot(deltas, 'o', alpha=0.4, color='steelblue', markersize=4, label='Δ')
            ax_p.plot(scores, 's', alpha=0.4, color='darkgreen', markersize=4, label='score')
            ax_p.plot(running, 'g-', linewidth=2, label='best score')
            ax_p.set_xlabel('Evaluation'); ax_p.set_ylabel('eV')
            ax_p.set_title('BO progress (green=SC-trig JT, red=spont. risk, orange=closed)', fontsize=11)
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
    """
    Plot phase diagram from phase_data structure. This function ONLY uses converged final states.
    """
    doping = np.array(phase_data['target_doping'])
    M = np.array(phase_data['M'])
    Q = np.array(phase_data['Q']) 
    
    # Plot order parameters vs doping
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
    
    # Mark expected phase regions (optional)
    if doping[-1] >= 0.03:
        ax.axvspan(0, 0.03, alpha=0.1, color='red', label='AFM dominant')
    if doping[-1] >= 0.15:
        ax.axvspan(0.05, 0.15, alpha=0.1, color='blue', label='SC+JT coexistence')
    
def _plot_dos(ax, solver: 'RMFT_Solver', result: Dict):
    """ Plot the density of states (DOS) to reveal van Hove singularities. """

    M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J = (result['M'], result['Q'], result['Delta_s'], result['Delta_d'], result['target_doping'], result['mu'], result['tx'], result['ty'], result['g_J'])

    vbdg = solver._get_vbdg()
    H_stack = vbdg._build_H_stack(
        solver.k_points, M, Q, Delta_s, Delta_d,
        target_doping, mu, tx, ty, g_J)           # (N_k, 16, 16)
    all_energies = np.linalg.eigvalsh(H_stack).ravel()  # (N_k*16,)

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
    print(f"\nVan Hove singularities detected at energies: {vhs_energies}")
    fermi_distance = np.min(np.abs(vhs_energies)) if len(vhs_energies) > 0 else np.inf
    print(f"Closest VHS to Fermi level: {fermi_distance:.4f} eV")

# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  SC-Activated JT Model - Variational Free Energy Minimization     ║
    ║  Implements: SC → Γ₆–Γ₇ mixing → JT via ∂F/∂M = ∂F/∂Q = 0         ║
    ║  Optimisation: Bayesian GP (Matérn 2.5) + VectorizedBdG           ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # ------------------------------------------------------------------ #
    # Base parameters.
    # BO Stage 1 searches: (Delta_tetra, u, g_JT)  [alpha_K fixed]
    # BO Stage 2 searches: (V_s_scale, V_d_scale)   [material fixed]
    # ------------------------------------------------------------------ #
    params = ModelParams(
        t0=0.326,
        u=4.835,
        lambda_soc=0.112,
        Delta_tetra=-0.084,
        g_JT=0.982,
        alpha_K=1.17,
        lambda_hop=1.2,
        eta=0.09,
        doping_0=0.09,
        Delta_CT=1.234,
        omega_JT=0.060,
        rpa_cutoff=0.09,
        d_wave=True,
        Delta_inplane=0.012,
        mu_LM=6.8,
        ALPHA_HF=0.12,
        CLUSTER_WEIGHT=0.35,
        ALPHA_D=0.18,
        mu_LM_D=2.9,
        V_s_scale=1.0,
        V_d_scale=1.0,
        Z=4,
        nk=84,
        kT=0.011,
        a=1.0,
        max_iter=300,
        tol=1e-4,
        mixing=0.035,
    )
    params.summary()
    solver = RMFT_Solver(params)

    print("\n" + "="*70)
    print("SC–JT INSTABILITY MATRIX (G-analysis, NORMAL-STATE BOUNDARY)")
    print("="*70)
    G_base = solver.compute_G_instability(target_doping=0.15)
    print(f"  h_afm  = {G_base['h_afm']:.4f} eV   (AFM Weiss field; M=0 → paramagnetic limit)")
    print(f"  χ_ΔΔ = {G_base['chi_DD']:.4f} eV⁻¹   (pair susceptibility on E_k^± bands)")
    print(f"  χ_QQ = {G_base['chi_QQ']:.4f} eV/Å²  (JT susceptibility, g_JT²-weighted; K_inv·χ_QQ dimensionless)")
    print(f"  χ_ΔQ = {G_base['chi_DQ']:.4f}         (mixed SC–JT response)")
    print(f"  N_eff= {G_base['N_eff']:.4f} eV⁻¹   (effective DOS on AFM spectrum)")
    print(f"  G11 = {G_base['G11']:.4f}  G22 = {G_base['G22']:.4f}  G12 = {G_base['G12']:.4f}")
    print(f"  det(G) = {G_base['det_G']:.5f}  →  "
          f"{'UNSTABLE ✓ (normal state breaks down)' if G_base['unstable'] else 'stable (SC not yet onset)'}")
    print(f"  λ_min  = {G_base['lambda_min']:.4f}  (→0 marks Tc / Q_c)")
    print(f"  V_eff  = {G_base['V_eff']:.4f} eV   λ_eff = {G_base['lambda_eff']:.4f}")
    print(f"  Tc(BCS estimate) ≈ {G_base['Tc_estimate']*1000:.1f} meV")
    print(f"  G22 > 0: {'✓ SC-triggered JT (not spontaneous)' if G_base['G22'] > 0 else '✗ spontaneous JT risk'}")
    print("="*70)

    # ------------------------------------------------------------------ #
    # Tight alpha_K upper bound from BCS + RPA validity (hard Bayesian constraint)
    # ------------------------------------------------------------------ #
    alpha_bound = alpha_K_validity_bound(solver)
    print(f"\nalpha_K validity bound (BCS+RPA): {alpha_bound:.3f}")

    print("\n" + "="*70)
    print("BAYESIAN MATERIAL PARAMETER OPTIMISATION")
    print("="*70)

    opt = full_bayesian_optimize(
        solver,
        doping_bounds      = (0.06, 0.24),
        Delta_tetra_bounds = (-0.28, -0.05),
        u_bounds           = (3.9, 7.6),
        gJT_bounds         = (0.55, 1.2),
        Vs_bounds          = (0.1, 2.5),
        Vd_bounds          = (0.1, 2.5),
        n_doping_scan      = 7,      # inner doping scan per material
        n_ch_grid          = 5,      # Stage 2: 5×5 channel grid
        n_ch_refine        = 15,     # Stage 2: GP refinement points
        n_initial          = 16,
        n_refine           = -1,     # auto: ~5 adaptive seeds
        n_iterations       = 38,
    )

    winner = opt.get('best_valid') or opt['best_mat']
    print(f"\nOptimal parameters found:")
    print(f"  doping      δ = {opt['doping']:.4f}")
    print(f"  Delta_tetra   = {opt['Delta_tetra']:.4f} eV")
    print(f"  u (U/t0)      = {opt['u']:.4f}")
    print(f"  g_JT          = {opt['g_JT']:.4f} eV/Å")
    print(f"  V_s_scale     = {opt['V_s_scale']:.4f}  (Stage 2)")
    print(f"  V_d_scale     = {opt['V_d_scale']:.4f}  (Stage 2)")
    print(f"  Δ_s           = {opt['Delta_s']:.6f} eV")
    print(f"  Δ_d           = {opt['Delta_d']:.6f} eV")
    print(f"  |Δ| total     = {opt['Delta']:.6f} eV")
    print(f"  score         = {opt['score']:.6f}")
    print(f"  Elapsed       = {opt['elapsed_s']/60:.1f} min")

    # ------------------------------------------------------------------ #
    # Phase diagram at the optimal material parameters.
    # ------------------------------------------------------------------ #
    params.Delta_tetra = winner.Delta_tetra
    params.u           = winner.u
    params.g_JT        = winner.g_JT
    params.V_s_scale   = opt['V_s_scale']   # Stage 2 channel optimum
    params.V_d_scale   = opt['V_d_scale']
    params.__post_init__()   # recompute Delta_CF, U, K_lattice
    solver_opt = RMFT_Solver(params)
    _opt_result_for_plot = {'observations': opt['all_obs']}
    fig = plot_phase_diagrams(
        solver_opt,
        initial_M=0.1, initial_Q=1e-4, initial_Delta=1e-4,
        doping_range=np.linspace(0.06, 0.20, 7),
        opt_result=_opt_result_for_plot,
    )
    plt.show()

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}\n")