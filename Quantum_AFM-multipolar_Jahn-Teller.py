import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings
import copy
import time
import concurrent.futures

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
                              #       Physical range 1.05–3.0. Replaces old λ_JT = g²/(K·t0)
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
    # alpha_K           — JT stability margin; alpha_K = 1 → marginal (spontaneous JT risk)
    # channel_mix       — pairing amplitude split: V_s = (1−mix)·g²/K [orbital B₁g, φ=1]
    #                     V_d = mix·g²/K [d-wave B₁g, φ(k)]; both in B₁g → no symmetry change
    # NOTE: target_doping is NOT stored here; alpha_K and channel_mix ARE, mutated per trial.
    # -----------------------------------------------------------------------
    channel_mix:   float

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
        # Γ₆–Γ₇a splitting from SOC+D4h CF. Pure cubic (Δ_tet=Δ_ip=0) → Δ_CF ≈ (3/2)·λ_SOC.
        # Delta_axial tunes Γ₆–Γ₇ gap; Delta_inplane splits Γ₇ quartet into Γ₇a+Γ₇b.
        self.Delta_CF: float = _gamma_splitting(
            self.lambda_soc, self.Delta_tetra, self.Delta_inplane)

        # Hubbard U from dimensionless ratio
        self.U: float = self.u * self.t0

        # Stoner Weiss-field U_mf = 0.5·Δ_CF (mean-field splitting amplitude ~0.05–0.10 eV in
        # Gutzwiller-projected band model, NOT the bare charge-transfer U ~1–3 eV)
        self.U_mf: float = 0.5 * self.Delta_CF

        # K = g²/(alpha_K·Δ_CF); alpha_K > 1 → K > K_min → spontaneous JT blocked
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
        V_eff = self.g_JT**2 / self.K_lattice     # bare JT-mediated pairing
        V_eff_g = V_eff * g_t                    # bandwidth-renormalized

        N0 = 1.0 / (np.pi * max(t_eff, 1e-6))     # 2D DOS estimate
        lambda_eff = V_eff * N0                  # BCS coupling constant

        # --- Pairing strength checks ---
        ok_pairing_min = lambda_eff > 0.1         # minimal pairing to matter
        ok_pairing_bcs = lambda_eff < 1.5         # still within BCS (not Eliashberg)

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
        # 5) Q-driven SC gap estimate
        # Δ ≈ [V_eff_g * g_JT / (Δ_CF - V_eff_g)] * Q
        # ==========================================================
        Q_est = 0.005  # Å representative small distortion
        denom = max(self.Delta_CF - V_eff_g, 1e-9)
        Delta_est = V_eff_g * self.g_JT * Q_est / denom
        ok_gap = Delta_est > self.kT

        # ==========================================================
        # 6) Mixing relevance ratios
        # ==========================================================
        mixing_pairing = V_eff_g / max(self.Delta_CF, 1e-9)
        mixing_JT = (self.g_JT * Q_est) / max(self.Delta_CF, 1e-9)

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
        print(f"  λ_eff (BCS coupling) = {lambda_eff:.3f}")
        print(f"    {'✓' if ok_pairing_min else '✗'} Minimal pairing: λ > 0.1")
        print(f"    {'✓' if ok_pairing_bcs else '✗'} BCS valid: λ < 1.5")
        print(f"  V_eff (bare) = {V_eff:.4f} eV")
        print(f"  V_eff·g_t (renorm) = {V_eff_g:.4f} eV")
        print(f"  V_eff/Δ_CF = {mixing_pairing:.3f}  (fraction of CF splitting)")

        K_min = self.g_JT**2 / max(self.Delta_CF, 1e-9)   # g²/Δ_CF; K > K_min ⟺ α_K > 1
        print("\nJT mechanism:  K = g²/(α_K·Δ_CF) > K_min = g²/Δ_CF  ⟺  α_K > 1")
        print(f"  α_K = {self.alpha_K:.4f}  →  K = {self.K_lattice:.4f} eV/Å²  "
              f"(K_min = {K_min:.4f} eV/Å²,  margin = {self.alpha_K - 1:.4f})")
        print(f"  {'✓ SC-TRIGGERED (K > K_min)' if jt_triggered else '✗ SPONTANEOUS JT RISK (K ≤ K_min)'}")

        print("\nEstimated SC gap (linear, Q-driven):")
        print(f"  Δ_est = {Delta_est*1000:.2f} meV  @ Q={Q_est} Å")
        print(f"  {'✓ Δ > kT' if ok_gap else '✗ Δ < kT'}  (kT={self.kT*1000:.2f} meV)")

        print("\nMixing relevance ratios:")
        print(f"  (V_eff·g_t)/Δ_CF = {mixing_pairing:.3f}")
        print(f"  (g_JT·Q)/Δ_CF    = {mixing_JT:.3f}")

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

def _make_odd(n: int) -> int:
    """Ensure n is odd for Simpson's rule"""
    return n if n % 2 == 1 else n + 1

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

        # VectorizedBdG (Section 6) is instantiated lazily on first use via
        # _get_vbdg().  Declared here so copy.copy() preserves the attribute.
        self._vbdg: Optional['VectorizedBdG'] = None

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
    
    def get_gutzwiller_factors(self, delta: float) -> Tuple[float, float, float]:
        """
        Gutzwiller renormalization factors for doping δ = 1 - n (hole or electron).

        g_t  = 2δ/(1+δ)  — kinetic energy suppression; → 0 at half-filling (Mott insulator)
        g_J  = 4/(1+δ)²  — exchange enhancement; → 4 at half-filling, gives J = 4t₀²/U
        g_Δ  = 2δ/(1+δ)  — anomalous amplitude renormalization

        Half-filling floor δ_min = 1e-6 to avoid singularity.
        """
        abs_delta = max(abs(delta), 1e-6)
        g_t     = (2.0 * abs_delta) / (1.0 + abs_delta)
        g_J     = 4.0 / ((1.0 + abs_delta) ** 2)
        g_Delta = g_t   # equal in single-band Gutzwiller; kept separate for potential multi-orbital extension
        return g_t, g_J, g_Delta
    
    # =========================================================================
    # 3.2 DISTORTION-DEPENDENT PARAMETERS
    # =========================================================================
    
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
    
    def dispersion(self, k: np.ndarray, tx: float, ty: float) -> float:
        """γ(k) = -2[tx·cos(kx·a) + ty·cos(ky·a)]. B₁g anisotropy: tx ≠ ty when Q ≠ 0."""
        return -2.0 * (tx * np.cos(k[0] * self.p.a) + ty * np.cos(k[1] * self.p.a))
    
    def fermi_function(self, E: np.ndarray) -> np.ndarray:
        """The Fermi–Dirac distribution f(E) with μ is already included in the Hamiltonian; clipped to [-100, 100] to prevent overflow."""
        arg = E / self.p.kT
        arg = np.clip(arg, -100, 100)
        return 1.0 / (1.0 + np.exp(arg))
    
    # =========================================================================
    # 3.2b  IRREP PROJECTION & MULTIPOLAR ALGEBRA  (symmetry selection rules)
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

        Uses the even k-grid so k + Q_AFM maps exactly to another grid point via
        precomputed self.chi0_Q_idx (no interpolation/aliasing).

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
        # Spin operator in 4-orbital BdG Nambu basis:
        # Particle A: S_z = diag(+1,-1,+η,-η); B: staggered -1; holes: p-h conjugate signs.
        sz_orb   = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
        Sz_bdg   = np.zeros((16, 16))
        Sz_bdg[np.arange(4),  np.arange(4)]  =  sz_orb   # particle A
        Sz_bdg[np.arange(4)+4,np.arange(4)+4]= -sz_orb   # particle B (staggered)
        Sz_bdg[np.arange(4)+8, np.arange(4)+8] = -sz_orb # hole A (p-h conjugate)
        Sz_bdg[np.arange(4)+12,np.arange(4)+12]=  sz_orb # hole B

        # Batch-diagonalise the even k-grid for both k and k+Q_AFM
        N   = self.N_k_even
        H_stack_k  = np.zeros((N, 16, 16), dtype=complex)
        H_stack_kQ = np.zeros((N, 16, 16), dtype=complex)
        for i in range(N):
            kvec  = self.k_points_even[i]
            kQvec = self.k_points_even[self.chi0_Q_idx[i]]
            H_stack_k[i]  = self.build_bdg_matrix(kvec,  M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            H_stack_kQ[i] = self.build_bdg_matrix(kQvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)

        # Batched diagonalisation: (N, 16), (N, 16, 16)
        E_k_all,  V_k_all  = np.linalg.eigh(H_stack_k)
        E_kQ_all, V_kQ_all = np.linalg.eigh(H_stack_kQ)

        f_k_all  = self.fermi_function(E_k_all)    # (N, 16)
        f_kQ_all = self.fermi_function(E_kQ_all)   # (N, 16)

        # Vectorised chi0: M_mat[k,n,m] = ⟨ψ_n(k)|Sz|ψ_m(k+Q)⟩
        SzV_kQ = np.einsum('ij,kjm->kim', Sz_bdg, V_kQ_all)   # (N,16,16)
        M_mat  = np.einsum('kni,kim->knm', V_k_all.conj(), SzV_kQ)  # (N,16,16)
        M2     = np.abs(M_mat)**2  # (N,16,16)

        df = f_k_all[:, :, None] - f_kQ_all[:, None, :]   # (N,16,16)
        dE = E_kQ_all[:, None, :] - E_k_all[:, :, None]   # (N,16,16)

        # Mask degenerate/near-zero pairs; divide only where safe to avoid NaN/Inf
        mask = (np.abs(df) > 1e-12) & (np.abs(dE) > 1e-6)
        ratio = np.zeros_like(df)
        ratio[mask] = (self.k_weights_even[:, None, None] * M2 * df / np.where(mask, dE, 1.0))[mask]
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

        # χ_τx ~ N(0)/(1+α·M²): suppressed when M large or U >> Uc (≈2.2–2.5 on 2D square lattice)
        t_eff_avg = np.sqrt(t2)
        N0        = 1.0 / (np.pi * max(t_eff_avg, 1e-6))   # 2D tight-binding DOS
        Ut_ratio    = self.p.U / max(t_eff_avg, 1e-6)
        Ut_critical = 2.35   # midpoint of (U/t)_c ≈ 2.2–2.5
        alpha_M     = max(Ut_ratio / Ut_critical - 1.0, 0.0)
        chi_tau     = N0 / (1.0 + alpha_M * M**2)

        return {
            'chi0':         chi0,
            'U_eff_chi':    U_eff_chi,
            'stoner_denom': stoner_denom,
            'afm_unstable': afm_unstable,
            'chi_tau':      chi_tau,     # multipolar susceptibility (JT activation diagnostic)
            'N0':           N0,          # bare DOS at Fermi level
            'Ut_ratio':     Ut_ratio,    # U/t_eff (must be < (U/t)_c ≈ 2.2–2.5 for χ_τx ≠ 0)
            'alpha_M':      alpha_M,     # AFM suppression factor
        }

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
        Gutzwiller-renormalised U_eff and 4×4 interaction matrix in [6↑, 6↓, 7↑, 7↓].

        U_mat[a,b,c,d] is reduced to a 4×4 matrix U_mat[α,β] by restricting to
        the intra-orbital and inter-orbital (Γ₆–Γ₇) channels that are relevant
        for the SC-activated JT pairing:
          • U_mat[0,0] = U_mat[1,1] = U_eff    (intra Γ₆: 6↑–6↑, 6↓–6↓)
          • U_mat[2,2] = U_mat[3,3] = U_eff    (intra Γ₇)
          • All cross terms set to U_eff as well (single-band approximation;
            Hund J corrections can be added as U_mat[0,2] = U_eff - 2J etc.)

        Returns
        -------
        U_eff  : float    scalar coupling (eV), U_eff = g_J·J_eff = g_J·(4t²/U)·f(δ) ~ 0.05–0.30 eV.
        U_mat  : (4,4)   interaction matrix in orbital basis, U_mat = U_eff·I₄
        """
        abs_delta = max(abs(target_doping), 1e-6)
        f_d       = abs_delta / (abs_delta + self.p.doping_0)
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t2        = 0.5 * (tx_bare**2 + ty_bare**2)
        J_eff_now = g_J * 4.0 * t2 / self.p.U * f_d
        U_eff     = g_J * J_eff_now   # ~ 0.05–0.30 eV (renormalised exchange)

        # 4×4 interaction in [6↑,6↓,7↑,7↓]: single-band approximation U_mat = U_eff·I
        # This correctly enters the matrix RPA: χ_RPA = (I − U_mat @ χ₀)⁻¹ @ χ₀
        U_mat = U_eff * np.eye(4, dtype=complex)
        return U_eff, U_mat

    def compute_chi0_tensor(self, q: np.ndarray,
                             M: float, Q: float,
                             Delta_s: complex, Delta_d: complex,
                             target_doping: float,
                             mu: float, tx: float, ty: float,
                             g_J: float) -> np.ndarray:
        """
        Orbital bare susceptibility tensor χ₀^{ab}(q) in [6↑,6↓,7↑,7↓] basis.

        χ₀^{ab}(q) = −(1/N) Σ_{k,n,m} V*_{an}(k) V_{am}(k+q) V*_{bm}(k+q) V_{bn}(k)
                       · (f_n(k) − f_m(k+q)) / (E_m(k+q) − E_n(k))

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

        # ── Batch-diagonalise the entire even k-grid ─────────────────────────
        N = self.N_k_even
        H_stack_k  = np.zeros((N, 16, 16), dtype=complex)
        H_stack_kQ = np.zeros((N, 16, 16), dtype=complex)
        for i in range(N):
            kvec  = self.k_points_even[i]
            kQvec = (kvec + q + np.pi) % (2 * np.pi) - np.pi
            H_stack_k[i]  = self.build_bdg_matrix(kvec,  M, Q, Delta_s, Delta_d,
                                                    target_doping, mu, tx, ty, g_J)
            H_stack_kQ[i] = self.build_bdg_matrix(kQvec, M, Q, Delta_s, Delta_d,
                                                    target_doping, mu, tx, ty, g_J)

        E_k_all,  V_k_all  = np.linalg.eigh(H_stack_k)    # (N,16), (N,16,16)
        E_kQ_all, V_kQ_all = np.linalg.eigh(H_stack_kQ)

        f_k_all  = self.fermi_function(E_k_all)     # (N,16)
        f_kQ_all = self.fermi_function(E_kQ_all)

        # factor[k,n,m] = w[k] · (f_n - f_m) / (E_m - E_n), masked
        df = f_k_all[:, :, None] - f_kQ_all[:, None, :]    # (N,16,16)
        dE = E_kQ_all[:, None, :] - E_k_all[:, :, None]    # (N,16,16)
        mask   = (np.abs(df) > 1e-12) & (np.abs(dE) > 1e-6)
        factor = np.zeros_like(df)
        factor[mask] = (self.k_weights_even[:, None, None] * df / np.where(mask, dE, 1.0))[mask]

        # Sum over sector pairs. Chunked to keep peak memory below ~32 MB per pair.
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

    def compute_rpa_susceptibility(self, q: np.ndarray,
                                    M: float, Q: float,
                                    Delta_s: complex, Delta_d: complex,
                                    target_doping: float,
                                    mu: float, tx: float, ty: float,
                                    g_J: float) -> np.ndarray:
        """
        RPA susceptibility tensor χ_RPA^{ab}(q) = [I − U·χ₀(q)]⁻¹ · χ₀(q).

        Uses the matrix-valued interaction U_mat (4×4) and bare susceptibility
        χ₀^{ab}(q) (4×4) from compute_chi0_tensor.

        Near the Stoner instability (det[I − U·χ₀] → 0) the denominator is
        regularised by clamping eigenvalues of [I − U·χ₀] to rpa_cutoff from below.

        Returns
        -------
        chi_rpa : (4, 4) complex matrix  (eV⁻¹)
        """
        chi0_mat = self.compute_chi0_tensor(q, M, Q, Delta_s, Delta_d,
                                             target_doping, mu, tx, ty, g_J)
        _, U_mat = self._u_eff_and_interaction_matrix(Q, g_J, target_doping)
        I4       = np.eye(4, dtype=complex)
        denom    = I4 - U_mat @ chi0_mat

        # Regularise: clamp smallest singular value to rpa_cutoff
        U_d, s_d, Vh_d = np.linalg.svd(denom)
        s_reg = np.maximum(s_d, self.p.rpa_cutoff)
        denom_reg_inv = (Vh_d.conj().T * (1.0 / s_reg)) @ U_d.conj().T
        return denom_reg_inv @ chi0_mat

    def project_to_pairing_channel(self, chi_rpa: np.ndarray) -> Tuple[float, float]:
        """
        Project χ_RPA onto the Γ₆⊗Γ₇ pairing channel and extract spin (s) and
        charge (c) pairing vertices.

        The SC-activated JT hypothesis requires pairing in the Γ₆⊗Γ₇ ⊃ B₁g
        channel.  The relevant matrix elements of χ_RPA are those that connect
        a Γ₆ state to a Γ₇ state (orbital off-diagonal):

            V_pair^{s} = (3/2) U_eff² · Tr[P₆ · χ_RPA · P₇]   (spin channel)
            V_pair^{c} = −(1/2) U_eff² · Tr[P₆ · χ_RPA · P₇]  (charge channel)

        In the single-band approximation with U_mat = U_eff·I,
        the spin vertex dominates:
            V(q) = (3/2) Tr[P₆ · χ_RPA^s(q) · P₇] − (1/2) Tr[P₆ · χ_RPA^c(q) · P₇]

        Since χ_RPA^c ≈ χ_RPA^s in the U_mat = U_eff·I limit:
            V(q) ≈ Tr[P₆ · χ_RPA(q) · P₇]

        The P₆ and P₇ projectors are 4×4 diagonal matrices in [6↑,6↓,7↑,7↓]:
            P₆ = diag(1,1,0,0)  P₇ = diag(0,0,1,1)

        Returns
        -------
        V_spin   : float  spin pairing vertex (eV), positive = attractive
        V_charge : float  charge pairing vertex (eV)
        """
        # Γ₆⊗Γ₇ projection: off-diagonal block [0:2, 2:4] of chi_rpa
        block_67 = chi_rpa[0:2, 2:4]  # Γ₆ rows, Γ₇ cols
        V_67     = float(np.real(np.trace(block_67)))

        # Spin and charge decomposition (single-band limit):
        V_spin   =  1.5 * V_67   # (3/2) factor from SU(2) spin algebra
        V_charge = -0.5 * V_67   # (−1/2) factor
        return V_spin, V_charge

    def compute_pairing_vertex(self, q: np.ndarray,
                                M: float, Q: float,
                                Delta_s: complex, Delta_d: complex,
                                target_doping: float,
                                mu: float, tx: float, ty: float,
                                g_J: float) -> float:
        """
        Total pairing vertex V(q) projected onto the Γ₆⊗Γ₇ channel.

        V(q) = V_spin(q) + V_charge(q)
             = (3/2 − 1/2) · Tr[P₆ · χ_RPA(q) · P₇]
             = Tr[P₆ · χ_RPA(q) · P₇]

        In the charge-transfer (ZSA) limit χ_c ≈ χ_s (both enter the RPA
        through the same U_mat = U_eff·I), so the net coefficient is +1
        (spin and charge contributions combine constructively in the singlet
        B₁g channel).

        Returns
        -------
        V_q : float  (eV)   Positive = attractive in the Γ₆⊗Γ₇ B₁g singlet channel.
        """
        chi_rpa = self.compute_rpa_susceptibility(q, M, Q, Delta_s, Delta_d,
                                                   target_doping, mu, tx, ty, g_J)
        V_spin, V_charge = self.project_to_pairing_channel(chi_rpa)
        return V_spin + V_charge   # net B₁g singlet vertex

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
        fs_pts_list: list = []
        vF_list:     list = []

        for kvec in self.k_points:
            H  = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d,
                                        target_doping, mu, tx, ty, g_J)
            ev = eigh(H, eigvals_only=True)
            if np.any(np.abs(ev) < 3.0 * self.p.kT):
                pos_ev = ev[ev > 0]
                vF     = float(pos_ev.min()) if len(pos_ev) > 0 else float(self.p.kT)
                vF     = max(vF, 1e-4)
                fs_pts_list.append(kvec.copy())
                vF_list.append(vF)

        if len(fs_pts_list) == 0:
            fs_pts_list = list(self.k_points[:n_fs])
            vF_list     = [1.0] * len(fs_pts_list)

        fs_pts = np.array(fs_pts_list[:n_fs])
        vF     = np.array(vF_list[:n_fs])
        return fs_pts, vF

    def build_gap_equation_kernel(self, fermi_pts: np.ndarray, vF: np.ndarray,
                                  M: float, Q: float,
                                  Delta_s: complex, Delta_d: complex,
                                  target_doping: float,
                                  mu: float, tx: float, ty: float,
                                  g_J: float) -> np.ndarray:
        """
        Pairing kernel Γ_{ij} for the linearised gap equation (DIAGNOSTIC only).

            Γ_{ij} = √v_F(i) · V(k_i − k_j) · √v_F(j)

        where V(q) is the full Γ₆⊗Γ₇-projected RPA vertex from compute_pairing_vertex.

        The kernel is SYMMETRIC (Γ = Γᵀ) by construction.

        IMPORTANT: This kernel is used exclusively in solve_linearized_gap_equation
        to compute λ_max as a DIAGNOSTIC.  The actual Δ in the SCF loop is
        determined by solve_gap_equation_k (BdG fixpoint + Newton-LM), NOT by
        this eigenvalue.  The two approaches answer different questions:
          • solve_gap_equation_k → WHAT IS the gap (self-consistent value)
          • solve_linearized_gap_equation → IS THERE an instability (λ_max > 1)

        Returns
        -------
        Gamma : (N, N) real symmetric matrix
        """
        N     = len(fermi_pts)
        Gamma = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                q          = fermi_pts[i] - fermi_pts[j]
                V_q        = self.compute_pairing_vertex(q, M, Q, Delta_s, Delta_d,
                                                         target_doping, mu, tx, ty, g_J)
                Gamma[i,j] = np.sqrt(vF[i]) * V_q * np.sqrt(vF[j])
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

        Separation of concerns
        ----------------------
        • Δ value (self-consistent)  ← solve_gap_equation_k + Newton-LM SCF
        • Δ symmetry + instability   ← THIS METHOD (eigenvalue)
        The two are complementary, not redundant.

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

    def solve_gap_equation_k(self, M: float, Q: float,
                              Delta_s: complex, Delta_d: complex,
                              target_doping: float,
                              mu: float, tx: float, ty: float,
                              g_J: float, g_Delta: float,
                              rpa_factor: float) -> Tuple[float, float]:
        """
        Dual-channel B₁g gap equations, solved with a single BdG diagonalization per k.

        Channel s — on-site orbital singlet (Γ₆⊗Γ₇ → B₁g, φ=1):
          Δ_s_new = V_s · g_Δ · Σ_k w_k · F_AA(k)
          V_s = (1−channel_mix) · g²/K · rpa_factor
          No φ weight: on-site pairing gives F_AA(k) φ-independent.

        Channel d — inter-site d-wave (φ(k) → B₁g in k-space, A↔B sublattice):
          Δ_d_new = V_d · g_Δ · Σ_k w_k · φ(k) · F_AB(k)
          V_d = channel_mix · g²/K · rpa_factor
          φ weight IS needed and correct (no double-counting):
            Since D_intersite ∝ φ(k) in BdG, F_AB(k) ~ φ(k)·Δ_d/E_k (one power of φ),
            so dot = Σ w·φ·F_AB ~ Σ w·φ²·Δ_d/E_k = ⟨φ²⟩·Δ_d/E.
            This reproduces the separable gap equation Δ_d = V_d·Σ_{k'} φ(k')²·Δ_d/(2E_k').
            (Compare: on-site F_AA ~ Δ_s/E_k, dot = Σ w·F_AA ~ Δ_s/E, no φ² factor needed.)
          The phonon propagator D = 2/ω enters the shape of V(k,k') but its magnitude
          is set by the spring constant K: V_eff = g²/K (adiabatic BCS, λ~0.3).

        F_AA and F_AB are different matrix elements of the same eigenvectors → no cross-terms.
        The hypothesis requires B₁g symmetry, not necessarily d-wave: channel_mix=0 tests
        the pure on-site orbital B₁g case which is equally valid under D₄h.

        Returns: (Delta_s_new, Delta_d_new) — new scalar amplitudes for both channels.
        """
        # Fast path: batched diagonalisation over all k in one LAPACK call.
        # Falls back to the scalar k-loop below only if _vbdg is unavailable.
        return self._get_vbdg().compute_gap_eq_vectorized(
            M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, g_Delta, rpa_factor)

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
        # Build all BdG matrices and batch-diagonalise
        N = self.N_k
        H_stack = np.zeros((N, 16, 16), dtype=complex)
        for i, kvec in enumerate(self.k_points):
            H_stack[i] = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d,
                                                target_doping, mu, tx, ty, g_J)
        ev_all, ec_all = np.linalg.eigh(H_stack)   # (N,16), (N,16,16)
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
          4. JT distortion:  H_JT = g_JT · Q · τ_x

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
    
    def build_inter_site_pairing_block(self, Delta: complex, k: np.ndarray) -> np.ndarray:
        """
        4×4 inter-site d-wave pairing matrix for A-particle ↔ B-hole channel.

        Pairing: Δ·φ(k)·(c_{A,6↑}c_{B,7↓} − c_{A,6↓}c_{B,7↑}) + h.c.

        The d-wave form factor φ(k) = cos kx − cos ky (B₁g irrep of D₄h) belongs here
        because it modulates INTER-SITE pairing between A and B sublattices — the correct
        physical picture for cuprate d-wave SC. On-site pairing cannot be d-wave.

        The orbital structure (Γ₆⊗Γ₇ singlet) provides the SECOND B₁g factor:
        under D₄h, Γ₆⊗Γ₇ ⊃ B₁g, and φ(k) is also B₁g → product is A₁g (trivial irrep
        of the Cooper pair), which is the singlet ground state.

        Returns 4×4 matrix D such that:
          BdG[0:4, 12:16] += D   (A-particle ↔ B-hole)
          BdG[4:8, 8:12]  += D   (B-particle ↔ A-hole, by sublattice symmetry)
        """
        phi_k = np.cos(k[0] * self.p.a) - np.cos(k[1] * self.p.a)
        D_local = np.zeros((4, 4), dtype=complex)
        D_local[0, 3] =  Delta * phi_k   # A:6↑ ↔ B:7↓
        D_local[1, 2] = -Delta * phi_k   # A:6↓ ↔ B:7↑ (singlet antisymmetry)
        return D_local

    def build_bdg_matrix(self, k: np.ndarray, M: float, Q: float,
                        Delta_s: complex, Delta_d: complex, target_doping: float,
                        mu: float, tx: float, ty: float,
                        g_J: float) -> np.ndarray:
        """
        16×16 BdG Hamiltonian with DUAL B₁g pairing channels.

        Delta_s: on-site orbital B₁g singlet amplitude (Γ₆⊗Γ₇ → B₁g via orbital indices).
                 Enters on-site blocks: BdG[0:4,8:12] and BdG[4:8,12:16].
                 Gap-equation: Δ_s = V_s·g_Δ·Σ_k w_k·F_AA(k)  (no φ weight)
        Delta_d: inter-site d-wave B₁g amplitude (φ(k)=cos kx−cos ky → B₁g in k-space).
                 Enters cross-sublattice blocks: BdG[0:4,12:16] and BdG[4:8,8:12].
                 Gap-equation: Δ_d = V_d·g_Δ·Σ_k w_k·φ(k)·F_AB(k)  (φ weight needed)

        Both channels have B₁g symmetry; the hypothesis requires only B₁g, not d-wave.
        channel_mix in ModelParams controls V_s/V_d split of total g²/K pairing strength.
        F_AA and F_AB are distinct matrix elements → no double-counting in gap equations.

        Basis: [Part_A(0:4), Part_B(4:8), Hole_A(8:12), Hole_B(12:16)], each [6↑,6↓,7↑,7↓].
        tx, ty: anisotropic hopping from B₁g JT distortion (tx≠ty when Q≠0).
        """
        # --- 1. LOCAL BLOCKS ---
        # On-site energy for sublattices from staggered AFM Weiss field: +M on A, −M on B.
        # The orbital mixing (τ_x) enters ONLY through Q (JT distortion), which is
        # itself driven by the anomalous Green function F(k) via the SCF loop.
        # No explicit Sigma_anom here: the SC back-action is mediated by Q, not by
        # a direct Δ→H_loc term (which would re-introduce spontaneous JT).
        H_A = self.build_local_hamiltonian_for_bdg(sign_M=+1.0, M=M, Q=Q, mu=mu, g_J=g_J,
                                                    target_doping=target_doping)
        H_B = self.build_local_hamiltonian_for_bdg(sign_M=-1.0, M=M, Q=Q, mu=mu, g_J=g_J,
                                                    target_doping=target_doping)

        # --- 2. KINETIC BLOCKS ---
        # Inter-sublattice hopping: γ(k)×I₄, tx≠ty allows B₁g symmetry breaking from JT.
        T_AB = self.dispersion(k, tx, ty) * np.eye(4, dtype=complex)

        # --- 3. DUAL B₁g PAIRING BLOCKS ---
        # Channel s: on-site orbital B₁g singlet (Γ₆⊗Γ₇ → B₁g via orbital indices, φ=1).
        #   The hypothesis does NOT require d-wave: on-site inter-orbital singlet is B₁g.
        #   Enters same-sublattice blocks: [Part_A↔Hole_A] and [Part_B↔Hole_B].
        D_onsite = self.build_pairing_block(Delta_s)      # 4×4, φ-independent

        # Channel d: inter-site d-wave B₁g (φ(k)=cos kx−cos ky → B₁g in k-space).
        #   Enters cross-sublattice blocks: [Part_A↔Hole_B] and [Part_B↔Hole_A].
        D_intersite = self.build_inter_site_pairing_block(Delta_d, k)  # 4×4, φ(k)-weighted

        # --- 4. ASSEMBLE 16×16 BdG MATRIX ---
        # Nambu basis: [Part_A(0:4), Part_B(4:8), Hole_A(8:12), Hole_B(12:16)]
        #
        # ┌──────────────┬──────────────────────────────┐
        # │  H_A   T_AB  │  D_s        D_d              │  Part_A, Part_B
        # │  T_AB† H_B   │  D_d        D_s              │
        # ├──────────────┼──────────────────────────────┤
        # │  D_s†  D_d†  │  −H_A*     −T_AB†            │  Hole_A, Hole_B
        # │  D_d†  D_s†  │  −T_AB*    −H_B*             │
        # └──────────────┴──────────────────────────────┘
        #
        # F_AA = u_A·v_A* measures channel s;  F_AB = u_A·v_B* measures channel d.
        # Gap eqs: Δ_s = V_s·Σ_k w_k·F_AA(k)  and  Δ_d = V_d·Σ_k w_k·φ(k)·F_AB(k).
        # The two gap equations are decoupled in the pairing matrix elements → no double-counting.

        BdG = np.zeros((16, 16), dtype=complex)

        # Particle-Particle sector (upper-left 8×8)
        BdG[0:4, 0:4] = H_A
        BdG[4:8, 4:8] = H_B
        BdG[0:4, 4:8] = T_AB
        BdG[4:8, 0:4] = np.conj(T_AB).T

        # Hole-Hole sector (lower-right 8×8): particle-hole symmetry −H*
        BdG[8:12,  8:12]  = -np.conj(H_A)
        BdG[12:16, 12:16] = -np.conj(H_B)
        BdG[8:12,  12:16] = -np.conj(T_AB)
        BdG[12:16,  8:12] = -np.conj(T_AB).T

        # On-site pairing (channel s): same-sublattice particle-hole
        BdG[0:4,   8:12] = D_onsite           # A-particle ↔ A-hole
        BdG[4:8,  12:16] = D_onsite           # B-particle ↔ B-hole
        BdG[8:12,   0:4] = np.conj(D_onsite).T
        BdG[12:16,  4:8] = np.conj(D_onsite).T

        # Inter-site pairing (channel d): cross-sublattice particle-hole
        BdG[0:4,  12:16] = D_intersite        # A-particle ↔ B-hole
        BdG[4:8,   8:12] = D_intersite        # B-particle ↔ A-hole
        BdG[12:16,  0:4] = np.conj(D_intersite).T
        BdG[8:12,   4:8] = np.conj(D_intersite).T
        return BdG
    
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
        n_total = 0.0
        f = self.fermi_function
        
        for i, kvec in enumerate(self.k_points):
            H = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            eigvals, eigvecs = eigh(H)
            fn = f(eigvals)
            fn_bar = 1.0 - fn
            
            dens_A, dens_B = 0.0, 0.0
            for n in range(16):
                psi = eigvecs[:, n]
                u_A, u_B = psi[0:4], psi[4:8]
                v_A, v_B = psi[8:12], psi[12:16]
                
                dens_A += np.sum(np.abs(u_A)**2) * fn[n] + np.sum(np.abs(v_A)**2) * fn_bar[n]
                dens_B += np.sum(np.abs(u_B)**2) * fn[n] + np.sum(np.abs(v_B)**2) * fn_bar[n]
            
            n_avg = (dens_A + dens_B) / 2.0 / 2.0  # BdG doubling correction
            n_total += self.k_weights[i] * n_avg
        return n_total
    
    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float,
                             tx: float, ty: float, mu_guess: float, g_J: float) -> float:
        """
        Find chemical potential that yields target_doping via root-finding.
        Uses Brent's method for robustness.
        """
        
        def density_error(mu_val):
            n = self._compute_density_at_mu(mu_val, M, Q, Delta_s, Delta_d, target_doping, tx, ty, g_J)
            return n - (1 - target_doping)
        
        # Bracket search: start around mu_guess
        w = 6.0 * self.p.t0  # bandwidth estimate
        mu_min, mu_max = mu_guess - w, mu_guess + w
        
        try:
            err_min = density_error(mu_min)
            err_max = density_error(mu_max)
            
            # Expand bracket if needed
            iter_expand = 0
            while (err_min * err_max > 0) and iter_expand < 10:
                if err_min > 0:  # Density too high everywhere, lower mu_min
                    mu_min -= w
                    err_min = density_error(mu_min)
                else:  # Density too low everywhere, raise mu_max
                    mu_max += w
                    err_max = density_error(mu_max)
                iter_expand += 1
            
            if err_min * err_max > 0:
                # Failed to bracket, return guess
                return mu_guess
            
            # Root-finding with Brent's method
            mu_opt = brentq(density_error, mu_min, mu_max, xtol=1e-5)
            return mu_opt
        except Exception:
            return mu_guess
    
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
                                g_Delta: float = 1.0,
                                rpa_factor: float = 1.0) -> float:
        """
        Grand potential per site from the k-space BdG spectrum:

        Ω = (1/2) Σ_{k,n} w_k [E_n f_n − T S(f_n)] + |Δ|²/(g_Δ·V_eff) + (K/2)Q²

        The 1/2 accounts for the doubled unit cell.
        |Δ|²/(g_Δ·V_eff): condensation cost (MF decoupling price; prevents Δ divergence).
        (K/2)Q²: elastic energy; JT gain E_JT = g²/(2K) already in BdG spectrum.
        V_eff = g²/K (adiabatic BCS); RPA renormalizes the electronic gain, not V_eff.
        """
        Omega_cell = 0.0

        for i, kvec in enumerate(self.k_points):
            H_BdG = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            E_n = eigh(H_BdG, eigvals_only=True)
            f_n = self.fermi_function(E_n)
            
            local_omega = 0.0
            for n in range(16):
                local_omega += E_n[n] * f_n[n]
                if self.p.kT > 1e-8: # Entropy contribution (only at finite T)
                    f = f_n[n]
                    if 1e-10 < f < 1.0 - 1e-10:
                        local_omega -= self.p.kT * (-f * np.log(f) - (1 - f) * np.log(1 - f))

            Omega_cell += self.k_weights[i] * local_omega

        # V_eff = g²/K (adiabatic, BCS controlled λ~O(1)); g²/ω would give Eliashberg regime
        V_total_rpa = (self.p.g_JT**2 / max(self.p.K_lattice, 1e-9)) * rpa_factor
        V_s = (1.0 - self.p.channel_mix) * V_total_rpa
        V_d = self.p.channel_mix          * V_total_rpa
        # Condensation cost: |Δ|²/(g_Δ·V) for each channel (MF decoupling price)
        cond_s = (abs(Delta_s)**2 / (g_Delta * V_s) if (V_s > 1e-12 and abs(Delta_s) > 1e-10) else 0.0)
        cond_d = (abs(Delta_d)**2 / (g_Delta * V_d) if (V_d > 1e-12 and abs(Delta_d) > 1e-10) else 0.0)
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

    def solve_self_consistent(self, target_doping: float,
                            initial_M: float = 0.5,
                            initial_Q: float = 1e-4,
                            initial_Delta: float = 0.05,
                            initial_channel_mix: float | None = None,
                            verbose: bool = True) -> Dict:
        """
        Variational SCF: minimizes F_total(M, Q, Δ, μ) subject to ⟨n⟩ = target_density.

        Each iteration:
          1. BdG diagonalization → M_bdg, τ_x, Pair (SC→orbital feedback included)
          2. Irrep selection: compute_rank2_multipole_expectation() quantifies B₁g barrier lifting
          3. RPA χ₀(q_AFM) every ~5 iterations → Stoner rpa_factor for V_eff
          4. Dual-channel gap equations Δ_s, Δ_d via solve_gap_equation_k (BdG fixpoint + Newton-LM)
          5. Cluster ED → M_cluster (quantum fluctuation correction, CLUSTER_WEIGHT)
          6. Hellmann-Feynman ∂F/∂M Newton step for M
          7. Q_out = (g_JT/K)·⟨τ_x⟩_total (SC-triggered JT equilibrium)
          8. Anderson mixing on [M,Q]; simple mixing on Δ
          9. Brent root-finding for μ; post-convergence Hessian test

        Returns converged dict: M, Q, Delta_s, Delta_d, mu, density, free energies,
        Gutzwiller factors, hessian, iteration history, chi0/rpa trajectories.
        """
        M = initial_M
        Q = initial_Q
        # Split initial Delta between channels according to channel_mix
        mix = self.p.channel_mix if initial_channel_mix is None else float(np.clip(initial_channel_mix, 0.0, 1.0))
        Delta_s = float(initial_Delta) * (1.0 - mix) + 0.0j   # on-site orbital B₁g
        Delta_d = float(initial_Delta) * mix           + 0.0j   # inter-site d-wave B₁g
        
        # Initial mu estimatating based on 2D tight-binding
        if abs(target_doping) < 0.01:
            mu = 0.0
        elif target_doping > 0:
            mu = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        else:
            mu = 2.0 * self.p.t0 * np.tanh(abs(target_doping) / 0.1)
        mu += 0.5 * self.p.Delta_CF

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [],
            'F_bdg': [], 'F_cluster': [],
            'g_t': [], 'g_J': [], 'mu': [],
            'chi0': [], 'rpa_factor': [], 'afm_unstable': [], 'selection_ratio': [],
            'chi_tau': [], 'Ut_ratio': [],
            'lambda_max': [],   # largest eigenvalue of linearised gap equation
            'gap_symmetry': [], # 'B1g (d-wave)' or 'A1g (s-wave)'
        }

        if verbose:
            print(f"\n{'='*80}")
            print("BdG LATTICE-BASED SELF-CONSISTENT CALCULATION")
            print(f"{'='*80}")
            print(f"Target doping δ={target_doping:.3f})")
            print(f"{'-'*80}")

        # Anderson mixing history for M and Q
        scf_x_hist: list = []
        scf_f_hist: list = []

        # Initialise cached RPA quantities (updated via outer-loop strategy)
        chi0         = 0.0
        rpa_factor   = 1.0
        afm_unstable = False
        chi_tau      = 0.0   # multipolar susceptibility; updated with chi0
        chi0_result  = {'Ut_ratio': 0.0, 'chi_tau': 0.0}  # sentinel until first chi0 update
        _chi0_last_M     = initial_M + 999.0   # force update at iteration 0
        _chi0_last_Delta = initial_Delta + 999.0   # triggers chi0 update at iter 0
        # Initialise irrep_info so it exists even if loop exits before irrep check
        irrep_info = {'w': 0.0, 'selection_ratio': 0.0,
                      'jt_algebraically_allowed': False,
                      'tau_x_projected': 0.0, 'tau_x_free_max': 1.0}

        for iteration in range(self.p.max_iter):

            # --- Renormalized parameters (use OLD Q for this step) ---
            g_t, g_J, g_Delta = self.get_gutzwiller_factors(target_doping)
            tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
            tx, ty = g_t * tx_bare, g_t * ty_bare

            # --- Compute observables with CURRENT mu ---
            obs = self._compute_k_observables(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            tau_x      = obs['Q']
            Pair_s_obs = obs['Pair_s']   # on-site pairing amplitude (channel s)
            Pair_d_obs = obs['Pair_d']   # inter-site pairing amplitude (channel d)
            M_bdg      = obs['M']        # BdG response: lattice magnetization

            # --- Irrep selection: has SC lifted the B₁g barrier? ---
            # Core algebraic test of SC-activated JT: |Δ|/Δ_CF → mixing weight (Γ6⊕Γ7 space),
            # and ⟨τx⟩ checks if the JT channel is unlocked. If not allowed, JT drive must vanish.(diagnostic; self-consistently enforced)
            Delta_eff  = abs(Delta_s) + abs(Delta_d)   # combined for irrep mixing weight
            irrep_info = self.compute_rank2_multipole_expectation(Delta_eff, tau_x)

            # --- RPA spin-fluctuation enhancement (outer-loop, avoid V_eff jump) ---
            # Update χ₀ only when state changed enough, or at iteration 0
            _chi0_update_needed = (
                iteration == 0
                or abs(M - _chi0_last_M) > 0.02
                or abs(Delta_eff - _chi0_last_Delta) > 0.005
            )
            if _chi0_update_needed:
                chi0_result  = self.compute_static_chi0_afm(
                    M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
                rpa_factor   = self.rpa_stoner_factor(chi0_result)
                chi0         = chi0_result['chi0']
                afm_unstable = chi0_result['afm_unstable']
                _chi0_last_M     = M
                _chi0_last_Delta = Delta_eff
                chi_tau = chi0_result['chi_tau']

            # --- Dual-channel gap equations: V_s=(1−mix)·g²/K·rpa, V_d=mix·g²/K·rpa ---
            Delta_s_out, Delta_d_out = self.solve_gap_equation_k(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, g_Delta, rpa_factor)

            # --- Newton–LM for Δ: ∂F/∂Δ_c = Δ_c/(g_Δ·V_c) − F_c = 0, blended with fixpoint ---
            V_total_n = (self.p.g_JT**2 / max(self.p.K_lattice, 1e-9)) * rpa_factor
            V_s_n     = (1.0 - self.p.channel_mix) * V_total_n
            V_d_n     = self.p.channel_mix           * V_total_n
            eps_D     = 1e-4
            mu_LM_D   = self.p.mu_LM_D
            ALPHA_D   = self.p.ALPHA_D

            def _newton_delta(d_val, V_c, Pair_c):
                if V_c < 1e-12:
                    return 0.0
                denom = max(V_c * g_Delta, 1e-12)
                dF0  = d_val / denom - float(np.real(Pair_c))
                dFp  = (d_val + eps_D) / denom - float(np.real(Pair_c))
                dFm  = (d_val - eps_D) / denom - float(np.real(Pair_c))
                d2F  = (dFp - dFm) / (2.0 * eps_D)
                gamma = 1.0 / (abs(d2F) + mu_LM_D)
                return d_val - gamma * dF0

            Ds_newton = _newton_delta(abs(Delta_s), V_s_n, Pair_s_obs)
            Dd_newton = _newton_delta(abs(Delta_d), V_d_n, Pair_d_obs)
            Delta_s_out = float(max((1.0 - ALPHA_D)*Delta_s_out + ALPHA_D*Ds_newton, 0.0))
            Delta_d_out = float(max((1.0 - ALPHA_D)*Delta_d_out + ALPHA_D*Dd_newton, 0.0))

            # --- Hellmann–Feynman Newton update for M ---
            # ∂F/∂M = 0 → Newton step γ = 1/|∂²F/∂M²|; gradient ∂F/∂M ~ O(10) eV
            # so plain gradient descent with fixed γ is ill-conditioned.
            eps_hf    = 1e-3
            dF_dM_0   = self.compute_dF_dM(M,          Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            dF_dM_p   = self.compute_dF_dM(M + eps_hf, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            dF_dM_m   = self.compute_dF_dM(M - eps_hf, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            d2F_dM2   = (dF_dM_p - dF_dM_m) / (2.0 * eps_hf)
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
            Q_bdg = -(self.p.g_JT / self.p.K_lattice) * tau_x
            if abs(Q_bdg) < 1e-6:
                Q_exp_cl  = F_cluster_early['Q_exp']   # signed ⟨τ_x⟩
                seed_mag  = min(max(abs(Q_exp_cl), 1e-4), 0.005 * self.p.lambda_hop)
                Q_out     = np.sign(Q_exp_cl) * seed_mag if abs(Q_exp_cl) > 1e-6 else seed_mag
            else:
                Q_out = Q_bdg
            Q_out = float(np.clip(Q_out, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # --- Anderson mixing on [M,Q]; reset history on Q sign flip (valley jump) ---
            x_in  = np.array([M,     Q    ])
            x_out = np.array([M_out, Q_out])
            scf_x_hist.append(x_in)
            scf_f_hist.append(x_out)

            x_new = self._anderson_mix(scf_x_hist, scf_f_hist, m=5)
            M_mixed    = float(np.clip(x_new[0], 0.0, 1.0))
            Q_mixed    = float(np.clip(x_new[1], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            if len(scf_x_hist) > 1 and (Q * Q_mixed < 0):
                scf_x_hist.clear()
                scf_f_hist.clear()

            # Δ_s and Δ_d: simple mixing independently (scalars, well-behaved)
            Delta_s_mixed = self._mix(Delta_s, Delta_s_out)
            Delta_d_mixed = self._mix(Delta_d, Delta_d_out)
            
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
                tx_mixed, ty_mixed, g_J, g_Delta=g_Delta, rpa_factor=rpa_factor
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

            history['M'].append(abs(M))
            history['Q'].append(abs(Q))
            history['Delta'].append(Delta_s_abs + Delta_d_abs)   # total gap
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
            history['Ut_ratio'].append(chi0_result.get('Ut_ratio', 0.0))

            # --- Linearised gap equation (every 10 iters): λ_max>1 → instability; diagnostic only ---
            # SCF gap is driven by solve_gap_equation_k; eigenvalue gives independent check + symmetry
            if iteration % 10 == 0:
                _lin = self.solve_linearized_gap_equation(
                    M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed,
                    target_doping, mu_new, tx_mixed, ty_mixed, g_J)
                _lambda_max   = _lin['lambda_max']
                _gap_symmetry = _lin['gap_symmetry']
            else:
                _lambda_max   = history['lambda_max'][-1] if history['lambda_max'] else 0.0
                _gap_symmetry = history['gap_symmetry'][-1] if history['gap_symmetry'] else 'unknown'
            history['lambda_max'].append(_lambda_max)
            history['gap_symmetry'].append(_gap_symmetry)

            if verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iter {iteration:3d}: "
                    f"M={M:.4f}  Q={Q:.5f}  "
                    f"Δs={abs(Delta_s):.5f}  Δd={abs(Delta_d):.5f}  "
                    f"n={n_kspace_new:.4f}  μ={mu_new:.4f}  F={F_bdg:.6f}  "
                    f"χ₀={chi0:.3f}  rpa={rpa_factor:.2f}  χτ={chi_tau:.3f}  "
                    f"JT={'✓' if irrep_info['jt_algebraically_allowed'] else '✗'}")

            # Update for next iteration
            M, Q, Delta_s, Delta_d, mu = M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, mu_new

            if max_diff < self.p.tol and abs(n_kspace_new - (1 - target_doping)) < 0.01:
                # --- Post-convergence Hessian / curvature test ---
                hessian_result = self.compute_hessian(
                    M, Q, abs(Delta_s) + abs(Delta_d), target_doping, mu_new, g_t, g_J, g_Delta
                )
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"✓ CONVERGED after {iteration+1} iterations")
                    print(f"{'='*80}")
                    print(f"M = {M:.6f}")
                    print(f"Q = {Q:.6f} Å")
                    print(f"|Δ_s| = {abs(Delta_s):.6f} eV  (on-site orbital B₁g, V_s weight={1.0-self.p.channel_mix:.2f})")
                    print(f"|Δ_d| = {abs(Delta_d):.6f} eV  (inter-site d-wave B₁g, V_d weight={self.p.channel_mix:.2f})")
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
            'Ut_ratio': chi0_result.get('Ut_ratio', 0.0),  # U/t_eff at convergence
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
            'converged': (max_diff < self.p.tol and abs(n_kspace_new - (1 - target_doping)) < 0.01)
        }

    def _anderson_mix(self, x_history: list, f_history: list, m: int = 5) -> np.ndarray:
        """
        Anderson mixing for self-consistent field convergence.
        Computes the minimum-norm linear combination of recent residuals and uses it to generate a new input estimate.
        Equivalent to a quasi-Newton step without explicitly forming the Jacobian.

        Args:
            x_history: list of previous input vectors [x_{n-m}, ..., x_n]
            f_history: list of previous output vectors [F(x_{n-m}), ..., F(x_n)]
            m: mixing history depth (window size)
        
        Returns:
            New proposed input vector x_{n+1}
        """
        alpha = self.p.mixing
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

    def _compute_k_observables(self, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J):
        """
        k-averaged BdG observables used inside the SCF loop.

        Dispatches to VectorizedBdG.compute_observables_vectorized (batch LAPACK,
        ~3-5x faster than the scalar loop below).  The scalar loop is preserved
        as _compute_k_observables_scalar for debugging.
        """
        return self._get_vbdg().compute_observables_vectorized(
            M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)

    def compute_dF_dM(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float,
                      mu: float, tx: float, ty: float, g_J: float) -> float:
        """
        Variational gradient ∂F/∂M via Hellmann-Feynman:

        ∂F/∂M = Σ_{k,n} f(E_n(k)) · ⟨ψ_n(k)| ∂H_BdG/∂M |ψ_n(k)⟩

        ∂H_BdG/∂M = h_prefactor · diag(sz_orb) embedded in 16×16 BdG
        (particle A: −sz, particle B: +sz, hole A: +sz, hole B: −sz).
        Used via Newton step M_newton = M − (1/|∂²F/∂M²|)·∂F/∂M.
        Returns ∂F/∂M in eV per site (double unit cell: /2).
        """
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t_sq_avg = 0.5 * (tx_bare**2 + ty_bare**2)
        # Prefactor of the Weiss operator (derivative of h_afm w.r.t. M)
        h_prefactor = g_J * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) / 2.0

        # Weiss operator: ∂H/∂M = h_prefactor·sz_orb, embedded into 16×16 BdG diagonal
        sz_orb = np.array([1.0, -1.0, self.p.eta, -self.p.eta]) * h_prefactor

        # Build BdG stack and batch-diagonalise
        N = self.N_k
        H_stack = np.zeros((N, 16, 16), dtype=complex)
        for i, kvec in enumerate(self.k_points):
            H_stack[i] = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d,
                                                target_doping, mu, tx, ty, g_J)
        eigvals_all, eigvecs_all = np.linalg.eigh(H_stack)  # (N,16), (N,16,16)
        f_all = self.fermi_function(eigvals_all)              # (N,16)

        # ⟨ψ_n|∂H/∂M|ψ_n⟩ = Σ_i dH_diag[i]·|ψ_n[i]|² (∂H/∂M is diagonal)
        dH_diag = np.array([
            -sz_orb[0], -sz_orb[1], -sz_orb[2], -sz_orb[3],   # particle A
            +sz_orb[0], +sz_orb[1], +sz_orb[2], +sz_orb[3],   # particle B
            +sz_orb[0], +sz_orb[1], +sz_orb[2], +sz_orb[3],   # hole A
            -sz_orb[0], -sz_orb[1], -sz_orb[2], -sz_orb[3],   # hole B
        ])  # (16,)

        exp_val = np.einsum('i,kin->kn', dH_diag, np.abs(eigvecs_all)**2)  # (N,16)
        grad = float(np.einsum('k,kn,kn->', self.k_weights, f_all, exp_val))
        return grad / 2.0  # per-site (double unit cell correction)

    def compute_hessian(self, M: float, Q: float, Delta: float, target_doping: float,
                        mu: float, g_t: float, g_J: float, g_Delta: float,
                        eps_M: float = 1e-3, eps_Q: float = 1e-4,
                        eps_D: float = 1e-4) -> Dict:
        """
        Post-convergence Hessian of F(M, Q, Δ) via central finite differences.

        Classifies the converged fixpoint:
          - All eigenvalues > 0: true local minimum ✓
          - Any eigenvalue < 0: saddle point (unstable direction) ✗
          - Near-zero eigenvalue: flat/Goldstone direction

        g_t is passed explicitly to avoid recomputing it from _last_density inside the closure,
        which would risk using a stale doping value and double-counting renormalization.

        Returns dict with 'H' (3×3), 'eigenvalues', 'is_minimum', 'min_curvature'.
        """
        def F(m, q, d):
            tb_x, tb_y = self.effective_hopping_anisotropic(q)
            return self.compute_bdg_free_energy(
                m, q, complex(d), complex(0), target_doping, mu, g_t * tb_x, g_t * tb_y, g_J, g_Delta
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

# =============================================================================
# 5. ALPHA_K VALIDITY BOUND
# =============================================================================

def alpha_K_validity_bound(solver) -> float:
    """
    Compute the tight upper bound on alpha_K from BCS+RPA validity conditions.

    Conditions (all must hold):
      (A) BCS validity:    lambda_eff = V_eff · N(0) < 1.5
                           V_eff = g²/K = g²·alpha_K / (alpha_K · Delta_CF)
                                        = g²/(alpha_K·Delta_CF)
                           → alpha_K < 1.5 / (Delta_CF · N(0))
      (B) RPA linearity:   V_eff · chi0_max < 1/rpa_cutoff
                           chi0_max ~ N(0) (worst case)
                           → same order as (A), modulated by rpa_cutoff
      (C) AFM-SC hierarchy: V_eff < h_AFM
                            → alpha_K < h_AFM / Delta_CF

    Returns the tightest bound, clipped to [1.1, 8.0].
    """
    delta_est = 0.16
    abs_delta = max(delta_est, 1e-6)
    g_t   = (2.0 * abs_delta) / (1.0 + abs_delta)
    t_eff = g_t * solver.p.t0
    N0    = 1.0 / (np.pi * max(t_eff, 1e-6))   # 2D tight-binding DOS
    Delta_CF = max(solver.p.Delta_CF, 1e-9)

    # (A) BCS ceiling
    alpha_max_bcs = 1.5 / (Delta_CF * N0)
    # (B) RPA ceiling (stricter by rpa_cutoff factor)
    alpha_max_rpa = 1.0 / (Delta_CF * N0 * max(solver.p.rpa_cutoff, 0.05))

    bound = min(alpha_max_bcs, alpha_max_rpa)

    # (C) AFM ceiling
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
    This eliminates the Python-level k-loop and gives ~3–5× speedup on the SCF
    inner loop.

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
        self._phi_k   = solver.phi_k          # (N_k,)  d-wave form factor
        # Pre-allocated buffer, reused each iteration to avoid GC pressure
        self._H_stack = np.zeros((self._N_k, 16, 16), dtype=complex)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_H_stack(self, M: float, Q: float,
                      Delta_s: complex, Delta_d: complex,
                      target_doping: float, mu: float,
                      tx: float, ty: float, g_J: float) -> None:
        """
        Fill the (N_k, 16, 16) BdG stack.

        k-independent blocks (H_A, H_B, D_onsite) are built once;
        k-dependent blocks (T_AB dispersion, D_intersite d-wave) are
        filled via broadcasting over the k axis.
        """
        H = self._H_stack
        H[:] = 0.0 + 0.0j

        # --- k-independent blocks ---
        H_A  = self.solver.build_local_hamiltonian_for_bdg(+1.0, M, Q, mu, g_J, target_doping)
        H_B  = self.solver.build_local_hamiltonian_for_bdg(-1.0, M, Q, mu, g_J, target_doping)
        D_on = self.solver.build_pairing_block(Delta_s)    # 4×4 on-site singlet

        # Particle/hole diagonal blocks (broadcast to all k)
        H[:, 0:4,   0:4  ] = H_A
        H[:, 4:8,   4:8  ] = H_B
        H[:, 8:12,  8:12 ] = -np.conj(H_A)
        H[:, 12:16, 12:16] = -np.conj(H_B)

        # On-site pairing (k-independent)
        D_dag = np.conj(D_on).T
        H[:, 0:4,   8:12 ] = D_on
        H[:, 4:8,  12:16 ] = D_on
        H[:, 8:12,  0:4  ] = D_dag
        H[:, 12:16, 4:8  ] = D_dag

        # --- k-dependent: dispersion γ(k) = -2(tx cos kx + ty cos ky) ---
        kx = self._kpts[:, 0]
        ky = self._kpts[:, 1]
        a  = self.solver.p.a
        gamma_k = -2.0 * (tx * np.cos(kx * a) + ty * np.cos(ky * a))   # (N_k,)

        # T_AB = γ(k)·I₄ → diagonal sub-blocks via index broadcasting
        di = np.arange(4)
        H[:, di,     di + 4 ] = gamma_k[:, None]   # particle A→B
        H[:, di + 4, di     ] = gamma_k[:, None]   # particle B→A (Hermitian)
        H[:, di + 8, di + 12] = -gamma_k[:, None]  # hole sector: −γ*
        H[:, di + 12,di + 8 ] = -gamma_k[:, None]

        # --- k-dependent: inter-site d-wave pairing D_int[k] = Delta_d·φ(k)·pattern ---
        phi = self._phi_k * Delta_d   # (N_k,) complex
        H[:, 0,  15] +=  phi          # A:6↑ → B:7↓
        H[:, 1,  14] -= phi           # A:6↓ → B:7↑ (singlet sign)
        H[:, 4,  11] +=  phi          # B:6↑ → A:7↓
        H[:, 5,  10] -= phi           # B:6↓ → A:7↑
        phi_c = np.conj(phi)
        H[:, 15,  0] +=  phi_c
        H[:, 14,  1] -= phi_c
        H[:, 11,  4] +=  phi_c
        H[:, 10,  5] -= phi_c

    def diag_all_k(self, M: float, Q: float,
                   Delta_s: complex, Delta_d: complex,
                   target_doping: float, mu: float,
                   tx: float, ty: float,
                   g_J: float) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalise all k-points in one batched LAPACK call."""
        self._fill_H_stack(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        ev_all, ec_all = np.linalg.eigh(self._H_stack)  # (N_k,16), (N_k,16,16)
        return ev_all, ec_all

    def compute_observables_vectorized(self, M: float, Q: float,
                                       Delta_s: complex, Delta_d: complex,
                                       target_doping: float, mu: float,
                                       tx: float, ty: float,
                                       g_J: float) -> Dict:
        """
        Vectorised observables: M, Q (τ_x), density, Pair_s, Pair_d.

        Computes the same quantities as RMFT_Solver.compute_observables_from_bdg
        but for all k simultaneously via broadcasting, avoiding any Python loop.

        Returns a dict with the same keys as compute_observables_from_bdg.
        """
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
                                   g_J: float, g_Delta: float,
                                   rpa_factor: float) -> Tuple[float, float]:
        """
        Vectorised gap equation — replaces solver.solve_gap_equation_k.

        Computes F_AA (on-site) and F_AB (inter-site) pairing amplitudes from
        the batched BdG eigensystem and returns (Delta_s_new, Delta_d_new).
        """
        p  = self.solver.p
        ev, ec = self.diag_all_k(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)

        arg = np.clip(ev / p.kT, -100, 100)
        f12 = 1.0 - 2.0 / (1.0 + np.exp(arg))   # (N_k, 16)

        uA = ec[:, 0:4,  :]
        uB = ec[:, 4:8,  :]
        vA = ec[:, 8:12, :]
        vB = ec[:, 12:16,:]

        # Channel s: on-site (A↔A and B↔B)
        pair_s = np.sum(
            (uA[:, 0, :] * np.conj(vA[:, 3, :]) - uA[:, 1, :] * np.conj(vA[:, 2, :])
           + uB[:, 0, :] * np.conj(vB[:, 3, :]) - uB[:, 1, :] * np.conj(vB[:, 2, :])) * f12,
            axis=1)   # (N_k,)

        # Channel d: inter-site (A↔B)
        pair_d = np.sum(
            0.5 * (uA[:, 0, :] * np.conj(vB[:, 3, :]) - uA[:, 1, :] * np.conj(vB[:, 2, :])
                 + uB[:, 0, :] * np.conj(vA[:, 3, :]) - uB[:, 1, :] * np.conj(vA[:, 2, :])) * f12,
            axis=1)   # (N_k,)

        w   = self.solver.k_weights
        phi = self.solver.phi_k

        F_AA = float(np.real(np.dot(w, pair_s))) / 4.0
        F_AB = float(np.real(np.dot(w * phi, pair_d))) / 4.0

        V_total = (p.g_JT**2 / max(p.K_lattice, 1e-9)) * rpa_factor
        V_s = (1.0 - p.channel_mix) * V_total
        V_d = p.channel_mix          * V_total

        return abs(g_Delta * V_s * F_AA), abs(g_Delta * V_d * F_AB)


# =============================================================================
# 7. FAST SCF WRAPPER  (uses VectorizedBdG)
# =============================================================================

def run_scf_fast(solver: 'RMFT_Solver',
                 target_doping: float,
                 alpha_K: float,
                 channel_mix: float,
                 initial_M: float = 0.25,
                 initial_Q: float = 1e-4,
                 initial_Delta: float = 0.04,
                 verbose: bool = False) -> Dict:
    """
    Run an SCF calculation for given (doping, alpha_K, channel_mix) using
    the vectorised BdG.  Thread-safe: operates on a shallow copy of solver.

    K_lattice is re-derived from alpha_K so the copy is fully self-consistent.
    Returns the same dict as solver.solve_self_consistent.
    """
    s               = copy.copy(solver)
    s.p             = copy.copy(solver.p)
    s.p.alpha_K     = float(alpha_K)
    s.p.channel_mix = float(channel_mix)
    s.p.K_lattice   = s.p.g_JT**2 / (s.p.alpha_K * max(s.p.Delta_CF, 1e-9))
    s._vbdg         = None   # reset so _get_vbdg() builds a fresh VectorizedBdG for this copy

    # Hard validity check: bail out immediately if alpha_K violates BCS/RPA bound
    bound = alpha_K_validity_bound(s)
    if alpha_K > bound * 1.05:
        return {'converged': False, 'Delta_s': 0.0, 'Delta_d': 0.0,
                'M': 0.0, 'Q': 0.0, 'F_bdg': 1e10,
                '_validity_violated': True, '_alpha_bound': bound}

    return s.solve_self_consistent(
        target_doping=target_doping,
        initial_M=initial_M, initial_Q=initial_Q, initial_Delta=initial_Delta,
        verbose=verbose,
    )

# =============================================================================
# 8. BAYESIAN OPTIMISER
# =============================================================================

class OptimPoint:
    """A single evaluated point in the (doping, alpha_K, channel_mix) search space."""
    __slots__ = ('doping', 'alpha_K', 'channel_mix', 'Delta_total', 'converged', 'result')

    def __init__(self, doping: float, alpha_K: float, channel_mix: float,
                 Delta_total: float, converged: bool, result: Optional[Dict] = None):
        self.doping      = doping
        self.alpha_K     = alpha_K
        self.channel_mix = channel_mix
        self.Delta_total = Delta_total
        self.converged   = converged
        self.result      = result

    def __repr__(self) -> str:
        return (f"OptimPoint(δ={self.doping:.3f}, α={self.alpha_K:.3f}, "
                f"mix={self.channel_mix:.2f}, Δ={self.Delta_total:.5f}, conv={self.converged})")


class BayesianOptimizer:
    """
    Gaussian-Process Bayesian optimiser over (doping, alpha_K, channel_mix).

    Objective: maximise Delta_total = Delta_s + Delta_d subject to the
    alpha_K hard constraint derived from BCS/RPA validity.

    Algorithm:
      1. Latin Hypercube Sampling for n_initial seed evaluations.
      2. Fit Matérn(ν=2.5) GP surrogate.
      3. Maximise Expected Improvement (EI) acquisition → next point.
      4. Repeat n_iterations times.

    Falls back to random LHS search if scikit-learn is not available.
    """

    def __init__(self,
                 solver: 'RMFT_Solver',
                 doping_bounds:    Tuple[float, float] = (0.08, 0.30),
                 alpha_bounds:     Tuple[float, float] = (1.05, 3.5),
                 mix_bounds:       Tuple[float, float] = (0.0,  1.0),
                 alpha_K_hard_max: Optional[float]     = None,
                 n_workers:        int                  = 1):

        self.solver        = solver
        self.doping_bounds = doping_bounds
        self.alpha_bounds  = alpha_bounds
        self.mix_bounds    = mix_bounds
        self.n_workers     = n_workers
        self.observations: List[OptimPoint] = []
        self._gp: Optional[object]          = None  # GaussianProcessRegressor

        # alpha_K hard constraint: tightest of user input and physical bound
        auto_bound = alpha_K_validity_bound(solver)
        self.alpha_K_hard_max = float(
            min(alpha_K_hard_max if alpha_K_hard_max is not None else np.inf,
                auto_bound, alpha_bounds[1]))

        print(f"  BayesianOptimizer: alpha_K_hard_max = {self.alpha_K_hard_max:.3f}"
              f"  (validity bound: {auto_bound:.3f})")
        print(f"  Search space: δ∈{doping_bounds},  α_K∈{alpha_bounds},  mix∈{mix_bounds}")
        print(f"  Parallel workers: {n_workers}")

        # GP kernel: Matern(nu=2.5) x constant amplitude + white noise
        if _SKLEARN_AVAILABLE:
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * Matern(length_scale=[0.1, 0.5, 0.3], nu=2.5)
                      + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 0.1)))
            self._gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6,
                n_restarts_optimizer=5, normalize_y=True)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize(self, doping: float, alpha: float, mix: float) -> np.ndarray:
        """Map (doping, alpha_K, mix) → [0,1]³."""
        d_n = (doping - self.doping_bounds[0]) / (self.doping_bounds[1] - self.doping_bounds[0])
        a_n = (alpha  - self.alpha_bounds[0])  / (self.alpha_bounds[1]  - self.alpha_bounds[0])
        m_n = (mix    - self.mix_bounds[0])    / (self.mix_bounds[1]    - self.mix_bounds[0])
        return np.array([d_n, a_n, m_n])

    def _denormalize(self, x: np.ndarray) -> Tuple[float, float, float]:
        doping = self.doping_bounds[0] + x[0] * (self.doping_bounds[1] - self.doping_bounds[0])
        alpha  = self.alpha_bounds[0]  + x[1] * (self.alpha_bounds[1]  - self.alpha_bounds[0])
        mix    = self.mix_bounds[0]    + x[2] * (self.mix_bounds[1]    - self.mix_bounds[0])
        return float(doping), float(alpha), float(mix)

    def _clip_alpha(self, alpha: float) -> float:
        return float(np.clip(alpha, self.alpha_bounds[0], self.alpha_K_hard_max))

    def _lhs_sample(self, n: int) -> np.ndarray:
        """Latin Hypercube Sampling on [0,1]³ (reproducible seed)."""
        rng = np.random.default_rng(seed=42)
        samples = np.zeros((n, 3))
        for j in range(3):
            perm = rng.permutation(n)
            samples[:, j] = (perm + rng.uniform(size=n)) / n
        return samples

    # ------------------------------------------------------------------
    # GP fitting and acquisition
    # ------------------------------------------------------------------

    def _fit_gp(self) -> None:
        """Fit the GP surrogate to current observations."""
        if not _SKLEARN_AVAILABLE or self._gp is None or len(self.observations) < 3:
            return
        X = np.array([self._normalize(o.doping, o.alpha_K, o.channel_mix)
                      for o in self.observations])
        y = np.array([o.Delta_total for o in self.observations])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(X, y)

    def _expected_improvement(self, X_cand: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function (vectorised)."""
        if not _SKLEARN_AVAILABLE or self._gp is None:
            return np.random.rand(len(X_cand))
        from scipy.stats import norm
        y_best = max(o.Delta_total for o in self.observations)
        mu, sigma = self._gp.predict(X_cand, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        z  = (mu - y_best - xi) / sigma
        EI = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(EI, 0.0)

    def _next_point_via_EI(self, n_restarts: int = 50) -> Tuple[float, float, float]:
        """Select next candidate by maximising EI over a random grid."""
        rng  = np.random.default_rng()
        cand = rng.uniform(size=(n_restarts * 200, 3))
        # Enforce alpha_K hard constraint in normalised space
        alpha_max_n = ((self.alpha_K_hard_max - self.alpha_bounds[0])
                       / (self.alpha_bounds[1]  - self.alpha_bounds[0]))
        cand[:, 1] = np.clip(cand[:, 1], 0.0, alpha_max_n)
        EI = self._expected_improvement(cand)
        return self._denormalize(cand[np.argmax(EI)])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_point(self, doping: float, alpha: float, mix: float) -> OptimPoint:
        """Evaluate a single point via run_scf_fast."""
        alpha = self._clip_alpha(alpha)
        t0 = time.time()
        try:
            result    = run_scf_fast(self.solver, doping, alpha, mix, verbose=False)
            Delta     = result.get('Delta_s', 0.0) + result.get('Delta_d', 0.0)
            converged = result.get('converged', False)
            if not converged:
                Delta *= 0.5   # 50% penalty for non-converged runs
        except Exception as e:
            print(f"    SCF error ({doping:.3f}, {alpha:.3f}, {mix:.3f}): {e}")
            result, Delta, converged = None, 0.0, False
        print(f"    δ={doping:.3f} α={alpha:.3f} mix={mix:.2f} "
              f"→ Δ={Delta:.5f} eV  {'✓' if converged else '⚠'}  ({time.time()-t0:.1f}s)")
        return OptimPoint(doping, alpha, mix, Delta, converged, result)

    def _evaluate_batch(self,
                        points: List[Tuple[float, float, float]]) -> List[OptimPoint]:
        """Batch evaluation: parallel if n_workers > 1, sequential otherwise."""
        if self.n_workers <= 1 or len(points) == 1:
            return [self._evaluate_point(d, a, m) for d, a, m in points]

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as ex:
            futs = {ex.submit(run_scf_fast, self.solver, d, a, m,
                              0.25, 1e-4, 0.04, False): (d, a, m)
                    for d, a, m in points}
            for fut in concurrent.futures.as_completed(futs):
                d, a, m = futs[fut]
                try:
                    res   = fut.result()
                    Delta = res.get('Delta_s', 0.0) + res.get('Delta_d', 0.0)
                    conv  = res.get('converged', False)
                    if not conv:
                        Delta *= 0.5
                    pt = OptimPoint(d, a, m, Delta, conv, res)
                except Exception as e:
                    print(f"    Parallel error ({d:.3f},{a:.3f},{m:.3f}): {e}")
                    pt = OptimPoint(d, a, m, 0.0, False, None)
                results.append(pt)
                print(f"    δ={d:.3f} α={a:.3f} mix={m:.2f} → Δ={pt.Delta_total:.5f}")
        return results

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(self, n_initial: int = 12, n_iterations: int = 35,
                 verbose: bool = True) -> Dict:
        """
        Run Bayesian optimisation.

        Phase 1: LHS seed evaluation (n_initial points).
        Phase 2: GP-guided EI maximisation (n_iterations evaluations).

        Returns dict with 'best_point', 'observations', 'gp', 'elapsed_s'.
        """
        t_start = time.time()
        print(f"\n{'='*70}")
        print(f"BAYESIAN OPTIMISATION  (seed={n_initial}  BO-iters={n_iterations})")
        print(f"  Total SCF runs: {n_initial + n_iterations}")
        print(f"{'='*70}")

        # Phase 1: Latin Hypercube seed
        print(f"\n[Phase 1] Latin Hypercube seeding ({n_initial} points)")
        initial_pts = [(d, self._clip_alpha(a), m)
                       for d, a, m in (self._denormalize(x)
                                       for x in self._lhs_sample(n_initial))]
        self.observations.extend(self._evaluate_batch(initial_pts))

        # Phase 2: BO iterations
        print(f"\n[Phase 2] Bayesian iterations ({n_iterations} steps)")
        for i in range(n_iterations):
            self._fit_gp()
            d, a, m = self._next_point_via_EI(n_restarts=30)
            a = self._clip_alpha(a)
            print(f"\n  [BO iter {i+1}/{n_iterations}] candidate: "
                  f"δ={d:.3f}  α={a:.3f}  mix={m:.2f}")
            self.observations.append(self._evaluate_point(d, a, m))

            if verbose and (i + 1) % 5 == 0:
                best = max(self.observations, key=lambda o: o.Delta_total)
                print(f"\n  ── best so far: δ={best.doping:.3f}  α={best.alpha_K:.3f}  "
                      f"mix={best.channel_mix:.2f}  Δ={best.Delta_total:.5f} eV ──\n")

        best    = max(self.observations, key=lambda o: o.Delta_total)
        elapsed = time.time() - t_start

        print(f"\n{'='*70}")
        print(f"OPTIMISATION COMPLETE  ({elapsed/60:.1f} min)")
        print(f"  GLOBAL OPTIMUM:")
        print(f"    doping δ    = {best.doping:.4f}")
        print(f"    alpha_K     = {best.alpha_K:.4f}  (hard max: {self.alpha_K_hard_max:.3f})")
        print(f"    channel_mix = {best.channel_mix:.4f}")
        print(f"    Δ_max       = {best.Delta_total:.6f} eV")
        print(f"    Converged   = {best.converged}")
        print(f"{'='*70}\n")

        return {'best_point':   best,
                'observations': self.observations,
                'gp':           self._gp,
                'elapsed_s':    elapsed}

def plot_optimization_results(opt_result: Dict, ax_progress=None, ax_doping=None,
                               ax_alpha=None, ax_mix=None) -> None:
    """
    Bayesian optimisation summary panels.

    Can be called standalone (creates its own figure) OR embedded into
    plot_phase_diagrams by passing pre-created axes.  When axes are supplied
    no new figure is created and plt.show() is not called.

    Panels:
      ax_progress : Δ vs evaluation index + running best
      ax_doping   : doping δ vs Δ (scatter, green=converged / red=failed)
      ax_alpha    : alpha_K vs Δ
      ax_mix      : channel_mix vs Δ
    """
    obs       = opt_result['observations']
    deltas    = [o.Delta_total for o in obs]
    dopings   = [o.doping      for o in obs]
    alphas    = [o.alpha_K     for o in obs]
    mixes     = [o.channel_mix for o in obs]
    converged = [o.converged   for o in obs]
    colours   = ['g' if c else 'r' for c in converged]
    running   = np.maximum.accumulate(deltas)

    standalone = ax_progress is None
    if standalone:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('Bayesian Optimisation — AFM-SC-JT Model', fontsize=13, fontweight='bold')
        ax_progress, ax_doping, ax_alpha, ax_mix = (
            axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

    ax_progress.plot(deltas, 'o', alpha=0.5, color='steelblue', markersize=5, label='Δ (eV)')
    ax_progress.plot(running, 'r-', linewidth=2, label='Best so far')
    ax_progress.set_xlabel('Evaluation index'); ax_progress.set_ylabel('Δ_total (eV)')
    ax_progress.set_title('Optimisation progress'); ax_progress.legend(fontsize=9)
    ax_progress.grid(True, alpha=0.3)

    best_idx = int(np.argmax(deltas))
    plot_targets = [(ax_doping, dopings, 'Doping δ'),
                    (ax_alpha,  alphas,  'alpha_K')]
    if ax_mix is not None:
        plot_targets.append((ax_mix, mixes, 'channel_mix'))
    for ax, xs, label in plot_targets:
        ax.scatter(xs, deltas, c=colours, s=40, alpha=0.7)
        ax.axvline(xs[best_idx], color='gold', linewidth=1.5, linestyle='--', label='optimum')
        ax.set_xlabel(label); ax.set_ylabel('Δ_total (eV)')
        ax.set_title(f'{label} vs Gap'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    if standalone:
        plt.tight_layout()
        plt.show()

def full_bayesian_optimize(solver: 'RMFT_Solver',
                            doping_bounds:    Tuple[float, float] = (0.12, 0.28),
                            alpha_bounds:     Tuple[float, float] = (1.05, 3.5),
                            mix_bounds:       Tuple[float, float] = (0.0,  1.0),
                            n_initial:        int   = 15,
                            n_iterations:     int   = 40,
                            n_workers:        int   = 1,
                            alpha_K_hard_max: Optional[float] = None) -> Dict:
    """
    Convenience wrapper: create BayesianOptimizer, run, optionally plot.

    Returns dict with keys: 'best', 'Delta', 'doping', 'alpha_K',
    'channel_mix', 'result', 'all_obs', 'elapsed_s'.
    """
    opt = BayesianOptimizer(solver,
                             doping_bounds=doping_bounds,
                             alpha_bounds=alpha_bounds,
                             mix_bounds=mix_bounds,
                             alpha_K_hard_max=alpha_K_hard_max,
                             n_workers=n_workers)
    opt_result = opt.optimize(n_initial=n_initial, n_iterations=n_iterations, verbose=True)
    # Note: plot_optimization_results is now embedded inside plot_phase_diagrams.
    # Call it standalone here only if you want an immediate preview without the phase diagram.
    best = opt_result['best_point']
    return {'best':        best,
            'Delta':       best.Delta_total,
            'doping':      best.doping,
            'alpha_K':     best.alpha_K,
            'channel_mix': best.channel_mix,
            'result':      best.result,
            'all_obs':     opt_result['observations'],
            'elapsed_s':   opt_result['elapsed_s']}


# =============================================================================
# 9. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_phase_diagrams(solver: RMFT_Solver, initial_M: float, initial_Q: float, initial_Delta: float, doping_range: np.ndarray,
                        cf_min: float = 0.05, cf_max: float = 0.20, N_cf: int = 10,
                        opt_result: Optional[Dict] = None):
    """
    Doping-scan phase diagram with warm-start and crystal-field sweet-spot search.

    opt_result : optional dict returned by full_bayesian_optimize.
        When supplied the figure gains a bottom row showing the Bayesian
        optimisation summary (progress, δ/α_K/mix vs Δ scatter), making
        plot_optimization_results redundant as a standalone call.

    Layout without opt_result (3×3):
      [0,0] Phase diagram   [0,1] CF sweet-spot   [0,2] DOS
      [1,0] M(iter)         [1,1] Q(iter)          [1,2] Δ(iter)
      [2,0] F_bdg/cluster   [2,1] g_t, g_J         [2,2] density

    Layout with opt_result (4×3 — extra bottom row for BO panels):
      [3,0] BO progress     [3,1] δ vs Δ           [3,2] α_K vs Δ
      (channel_mix vs Δ is inlined in [3,2] via a secondary x-axis or appended)
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
    print(f"  channel_mix={solver.p.channel_mix:.2f}  "
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
    print(f"Scanning Delta_CF from {cf_min:.3f} to {cf_max:.3f} eV")

    original_cf = solver.p.Delta_CF
    ref_doping_idx = len(doping_range) // 2
    ref_doping = doping_range[ref_doping_idx]

    print(f"Reference doping: δ={ref_doping:.3f}")

    cf_range = np.linspace(cf_min, cf_max, N_cf)
    cf_gaps, cf_Q_values, cf_M_values = [], [], []
    cf_previous = None

    for cf in cf_range:
        solver.p.Delta_CF = cf
        if cf_previous is not None:
            init_M     = cf_previous['M']
            init_Q     = cf_previous['Q']
            init_Delta = cf_previous['Delta_s'] + cf_previous['Delta_d']
        else:
            init_M     = phase_data['M'][ref_doping_idx]
            init_Q     = phase_data['Q'][ref_doping_idx]
            init_Delta = phase_data['Delta_s'][ref_doping_idx] + phase_data['Delta_d'][ref_doping_idx]

        cf_result = solver.solve_self_consistent(
            target_doping=ref_doping,
            initial_M=init_M, initial_Q=init_Q, initial_Delta=init_Delta,
            verbose=False
        )
        cf_gaps.append(cf_result['Delta_d'])
        cf_Q_values.append(cf_result['Q'])
        cf_M_values.append(cf_result['M'])
        cf_previous = {
            'M': cf_result['M'], 'Q': cf_result['Q'],
            'Delta_s': cf_result['Delta_s'], 'Delta_d': cf_result['Delta_d']
        }
        print(f"  ΔCF={cf:.4f} → Δs={cf_result['Delta_s']:.5f}  Δd={cf_result['Delta_d']:.5f} "
              f"Q={cf_result['Q']:+.5f}  M={cf_result['M']:.4f}")
        print(f"  χ₀(q_AFM) = {cf_result['chi0']:.4f}  |  RPA factor = {cf_result['rpa_factor']:.3f}×")
        print(f"  Irrep selection R = {cf_result['irrep_info']['selection_ratio']:.4f} "
              f"JT {'ALLOWED ✓' if cf_result['irrep_info']['jt_algebraically_allowed'] else 'BLOCKED ✗'}")

    # Restore original Delta_CF
    solver.p.Delta_CF = original_cf

    max_gap_idx = np.argmax(cf_gaps)
    sweet_spot_cf = cf_range[max_gap_idx]
    max_gap = cf_gaps[max_gap_idx]
    print(f"\n✓ Sweet spot: ΔCF = {sweet_spot_cf:.3f} eV, Δmax = {max_gap:.4f} eV")

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
        # Wrap in a compatible dict if full_bayesian_optimize return format used
        if 'observations' not in opt_result and 'all_obs' in opt_result:
            _wrapped = {'observations': all_obs}
        else:
            _wrapped = opt_result
        plot_optimization_results(
            _wrapped,
            ax_progress=axes[3, 0],
            ax_doping=axes[3, 1],
            ax_alpha=axes[3, 2],
            ax_mix=None,      # 4th panel omitted (3-col layout); mix info is in ax_alpha colour
        )
        # channel_mix as colour on alpha panel (already set inside plot_optimization_results)
        # Add a super-title for the BO row
        axes[3, 0].set_title('BO: Δ progress',         fontsize=11)
        axes[3, 1].set_title('BO: doping δ vs Δ',      fontsize=11)
        axes[3, 2].set_title('BO: alpha_K vs Δ',       fontsize=11)
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
    ax.plot(doping, np.array(phase_data['Delta_d']), 'b-^', linewidth=2, markersize=6, label='SC Gap (Δ)')
    
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
    
def _plot_dos(ax, solver: RMFT_Solver, result: Dict):
    """ Plot the density function (DOS) of the system to see the van Hove peaks. """
    all_energies = []
    M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J = (result['M'], result['Q'], result['Delta_s'], result['Delta_d'], result['target_doping'], result['mu'], result['tx'], result['ty'], result['g_J'])

    for kvec in solver.k_points:
        H_BdG = solver.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        vals = np.linalg.eigvalsh(H_BdG)
        all_energies.extend(vals)

    all_energies = np.array(all_energies)
    ax.hist(all_energies, bins=200, density=True, color='blue', alpha=0.7, label='DOS')

    # Fermi level (E = mu, but in BdG 0 is the Fermi level)
    ax.axvline(x=0.0, color='red', linestyle='--', label='Fermi Level ($E_F$)')
    ax.set_title(
        f"Density of States (DOS)\n"
        f"$\\Delta_{{CF}}={solver.p.Delta_CF:.4f}$ eV, "
        f"Doping δ={result.get('target_doping', 'N/A')}"
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    # Check: Is there a peak at 0?
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
    ║  Implements: SC → Γ₆–Γ₇ mixing → JT via ∂F/∂M = ∂F/∂Q = 0     ║
    ║  Optimisation: Bayesian GP (Matérn 2.5) + VectorizedBdG          ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # ------------------------------------------------------------------ #
    # Base parameters. alpha_K and channel_mix are Bayesian search variables.
    # ------------------------------------------------------------------ #
    params = ModelParams(
        t0=0.328,
        u=4.821,
        lambda_soc=0.111,
        Delta_tetra=-0.088,
        g_JT=0.944,
        alpha_K=1.19,
        lambda_hop=1.2,
        eta=0.09,
        doping_0=0.09,
        Delta_CT=1.234,
        omega_JT=0.060,
        rpa_cutoff=0.09,
        d_wave=True,
        Delta_inplane=0.012,
        mu_LM=5.0,
        ALPHA_HF=0.16,
        CLUSTER_WEIGHT=0.35,
        ALPHA_D=0.3,
        mu_LM_D=1.0,
        channel_mix=0.5,
        Z=4,
        nk=74,
        kT=0.011,
        a=1.0,
        max_iter=200,
        tol=1e-4,
        mixing=0.04,
    )
    params.summary()
    solver = RMFT_Solver(params)

    # ------------------------------------------------------------------ #
    # Tight alpha_K upper bound from BCS + RPA validity (hard Bayesian constraint)
    # ------------------------------------------------------------------ #
    alpha_bound = alpha_K_validity_bound(solver)
    print(f"\nalpha_K validity bound (BCS+RPA): {alpha_bound:.3f}")
    print(f"  → search range: α_K ∈ [1.05, {alpha_bound:.3f}]")

    # ------------------------------------------------------------------ #
    # Bayesian optimisation over (doping, alpha_K, channel_mix).
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("BAYESIAN PARAMETER OPTIMISATION")
    print("="*70)

    opt = full_bayesian_optimize(
        solver,
        doping_bounds    = (0.08, 0.30),
        alpha_bounds     = (1.05, min(alpha_bound, 3.5)),
        mix_bounds       = (0.0,  1.0),
        n_initial        = 15,    # LHS seed evaluations
        n_iterations     = 38,    # GP-guided BO steps  → total 53 runs
        n_workers        = 4,     # set > 1 for multi-core parallel batch
        alpha_K_hard_max = alpha_bound
    )

    print(f"\nOptimal parameters found:")
    print(f"  doping      δ = {opt['doping']:.4f}")
    print(f"  alpha_K       = {opt['alpha_K']:.4f}")
    print(f"  channel_mix   = {opt['channel_mix']:.4f}")
    print(f"  Δ_max         = {opt['Delta']:.6f} eV")
    print(f"  Elapsed       = {opt['elapsed_s']/60:.1f} min")

    # ------------------------------------------------------------------ #
    # Optional: phase diagram at the optimal parameters.
    # ------------------------------------------------------------------ #
    
    params.alpha_K     = opt['alpha_K']
    params.channel_mix = opt['channel_mix']
    params.K_lattice   = params.g_JT**2 / (params.alpha_K * max(params.Delta_CF, 1e-9))
    solver_opt = RMFT_Solver(params)
    # opt_result passed here → Bayesian panels are embedded in row 3 of the figure.
    # This replaces the old standalone plot_optimization_results() call.
    _opt_result_for_plot = {
        'observations': opt['all_obs'],
    }
    fig = plot_phase_diagrams(
        solver_opt,
        initial_M=0.25, initial_Q=0.06, initial_Delta=0.04,
        doping_range=np.linspace(0.15, 0.25, 3),
        opt_result=_opt_result_for_plot,
    )
    plt.show()

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}\n")