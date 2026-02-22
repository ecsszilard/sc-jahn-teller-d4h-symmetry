import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Tuple, Dict

# =============================================================================
# 0. PHYSICAL PARAMETERS & MODEL DEFINITION
# =============================================================================

@dataclass
class ModelParams:
    """
    Derived quantities (set in __post_init__, read-only in practice)
    ----------------------------------------------------------------
    Delta_CF   : Γ₆–Γ₇ splitting from SOC+CF Hamiltonian (eV)
    U          : Hubbard U = u·t0 (eV)
    U_mf       : Stoner Weiss-field = 0.5·Δ_CF (eV); keeps h_AFM < bandwidth.
    omega_JT   : JT phonon frequency (eV); direct parameter, physical range 40–80 meV.
                 Used only for D_phonon = 2/ω_JT (shape of V(k,k')); all free-energy magnitudes use the adiabatic V_eff = g²/K.
    K_lattice  : phonon spring constant g_JT²/(lambda_jt·t0) (eV/Å²)
    """
    # --- Primary inputs ---
    t0:            float      # eV    bare hopping integral (sets energy scale)
    u:             float      # —     U/t0 → Hubbard U = 3.43 eV (charge-transfer regime, Mott proximity, typ. 6–10)
    lambda_soc:    float      # eV    atomic SOC constant λ (t2g shell); ~0.05–0.15 eV; determines Γ₆–Γ₇ splitting via SOC+CF Hamiltonian
    Delta_tetra:   float      # eV    tetragonal axial CF Δ_axial = Δ_tet·Lz²; negative = z-compression
                              #       → partial cancellation with SOC, tunes Γ₆–Γ₇ gap independently of λ.
    g_JT:          float      # eV/Å  electron–phonon coupling (Jahn–Teller) coupling
    lambda_jt:     float      # —     dimensionless spring-constant ratio; K_lattice = g_JT²/(lambda_jt·t0), SC-triggered (not spontaneous) JT requires lambda_jt < Delta_CF/t0
    lambda_hop:    float      # Å     hopping decay length for B₁g anisotropy: t(Q)=t0·exp(±Q/lambda_hop); physical scale ~1–2 Å (M–O bond), NOT derivable from g_JT/t0
    eta:           float      # —     Γ₇ AFM asymmetry relative to Γ₆
    doping_0:      float      # —     superexchange regularisation (suppresses the unphysical g_J→4 divergence near half-filling in the Gutzwiller approximation, where coherent spectral weight actually vanishes)
    # --- Charge-transfer / RPA / gap symmetry ---
    Delta_inplane: float      # eV    B2g in-plane anisotropy Δ_ip·(Lx²−Ly²). Splits the Γ₇ 4-fold level into two Kramers pairs (Γ₇a, Γ₇b) without removing Kramers degeneracy. Prevents spontaneous JT from the residual 4-fold Γ₇ degeneracy. Default 0 = axial only.
    Delta_CT:      float      # eV    charge-transfer gap (ZSA scale). Not directly used in the band Hamiltonian (see note on t_pd below), but sets the physical scale for the charge-transfer insulator crossover.
                              #       Reducing Delta_CT toward ~1.2 eV increases multipolar fluctuations and pushes the system toward fluktuáló AFM (away from deep Mott).
    omega_JT:      float      # eV    JT phonon frequency (40–80 meV physical range for Cu-O JT modes).
                              #       Direct parameter; replaces the old M_ion → ω = sqrt(K/M) route which gave ω ~ 26 eV
                              #       omega_JT enters only D_phonon = 2/ω_JT (shape of V(k,k')); all free-energy magnitudes use g²/K.
    rpa_cutoff:    float      # —     Stoner denominator floor: rpa_factor = 1/max(sd, cutoff).
                              #       Motivation: BCS is valid for λ_eff ~ 0.3–1.5; at the AFM QCP  the Eliashberg regime (λ>>1) begins. cutoff sets the boundary:
                              #       rpa_max = 1/cutoff. Physical choice: cutoff ~ kT/V_eff (energy resolution limit) ≈ 0.011/0.125 ≈ 0.088 → rpa_max ≈ 11×.
                              #       Conservative default 0.12 → rpa_max = 8.3×. Decreasing rpa_cutoff enhance RPA factor
    d_wave:        bool       # —     True → B₁g d-wave form factor φ(k)=cos kx - cos ky; False → s-wave φ(k)=1. Controls gap symmetry channel.

    # --- Bayesian optimization targets (SCF solver hyper-parameters) ---
    mu_LM:         float      # —     Levenberg–Marquardt regularisation floor for M Newton step.
                              #       Larger → smaller γ_M → more conservative M update (prevents explosion near flat potential).
                              #       Physical range: 1–10 (default 4.0).
    ALPHA_HF:      float      # —     Blend weight for Newton vs BdG fixpoint for M update. (prevents M→1 explosion near flat potential)
                              #       0 → pure BdG fixpoint (safe, slow); 1 → pure Newton (fast, unstable).
                              #       Physical range: 0.05–0.5 (default 0.2).
    CLUSTER_WEIGHT: float     # —     Weight of cluster ED magnetization vs BdG magnetization in M update.
                              #       Higher → more quantum fluctuation correction; lower → more mean-field.
                              #       Physical range: 0.1–0.6 (default 0.35).
    ALPHA_D:       float      # —     Blend weight for Newton vs gap-equation fixpoint for Δ update.
                              #       0 → pure gap equation; 1 → pure Newton.
                              #       Physical range: 0.05–0.6 (default 0.3).
    mu_LM_D:       float      # —     Levenberg–Marquardt regularisation floor for Δ Newton step.
                              #       Softer than mu_LM (Δ landscape smoother than M).
                              #       Physical range: 0.1–5.0 (default 1.0).
    channel_mix:   float      # —     Pairing channel mixing: 0 = pure on-site orbital B₁g (Γ₆⊗Γ₇, φ=1),
                              #       1 = pure inter-site d-wave B₁g (φ(k)=cos kx−cos ky).
                              #       Intermediate values allow both channels simultaneously in BdG:
                              #         V_s = (1−channel_mix)·g²/K  [on-site orbital singlet]
                              #         V_d = channel_mix·g²/K      [inter-site d-wave]
                              #       Both channels have B₁g symmetry (orbital vs momentum sector).

    # --- Numerics ---
    Z:             int        #     2D square lattice coordination number
    nk:            int        #     k-grid: MUST BE EVEN so that q_AFM=(π,π) falls exactly on a grid point (k + π maps to another grid point mod 2π).
                              #     Even nk conflicts with Simpson's rule (needs odd), so we use the even grid for BZ sampling and a separate odd sub-grid for chi0 computation.
                              #     For SCF: nk even → nk+1 odd for Simpson. Rule: nk % 2 == 0 enforced in __post_init__.
    kT:            float      # eV  temperature (~127.7 K)
    a:             float      # Å   lattice constant
    max_iter:      int
    tol:           float
    mixing:        float

    def __post_init__(self):
        """Derive all secondary parameters from the primary inputs."""
        # Γ₆–Γ₇a splitting from the full SOC + D4h CF Hamiltonian.
        # In the cubic limit (Delta_tetra=Delta_inplane=0) this recovers Δ_CF ≈ (3/2)·λ_SOC.
        # Delta_axial (Lz²) tunes the Γ₆–Γ₇ gap; Delta_inplane (Lx²−Ly²) additionally
        # splits the Γ₇ quartet into Γ₇a+Γ₇b without collapsing Kramers degeneracy.
        self.Delta_CF: float = _gamma_splitting(
            self.lambda_soc, self.Delta_tetra, self.Delta_inplane)

        # Hubbard U from dimensionless ratio
        self.U: float = self.u * self.t0

        # Stoner Weiss-field: U_mf = 0.5·Δ_CF.
        # This is the mean-field splitting amplitude (~0.05–0.10 eV) in the
        # Gutzwiller-projected band model.  It is NOT the charge-transfer U
        # (which is ~1–3 eV and would give h_AFM >> bandwidth → instant insulator).
        self.U_mf: float = 0.5 * self.Delta_CF

        # Phonon spring constant: K = g_JT² / (lambda_jt · t0).
        # i.e. K > K_min = g_JT²/Delta_CF.
        self.K_lattice: float = self.g_JT**2 / (self.lambda_jt * self.t0)

        # omega_JT is a direct parameter (40–80 meV physical range).
        # It enters only D_phonon = 2/ω_JT (shape of V(k,k')); all
        # free-energy magnitudes use the ADIABATIC V_eff = g²/K.
        # (The old M_ion → ω=sqrt(K/M) route gave ω~26 eV due to wrong unit
        #  conversion and is removed.)

        # The ZSA second-order perturbation relation gives: t0_eff = t_pd² / Delta_CT
        self.t_pd: float = np.sqrt(self.t0 * max(self.Delta_CT, 1e-9))

        # Even-nk enforcement for commensurate q_AFM=(π,π) grid.
        # If nk is odd, silently round up to even so that adding (π,π) to any
        # k-point maps exactly to another grid point: k_i + π = k_{i + nk/2}.
        if self.nk % 2 != 0:
            object.__setattr__(self, 'nk', self.nk + 1)

    def summary(self, delta: float = 0.15) -> None:
        """Print primary inputs, all derived quantities and coexistence-window checks."""
        g_t = 2*delta / (1+delta)
        g_J = 4 / (1+delta)**2
        t_eff = g_t * self.t0
        f_d = delta / (delta + self.doping_0)
        J_eff = g_J * 4 * t_eff**2 / self.U * f_d
        h_afm = (g_J * self.U_mf/2 + J_eff * self.Z/2) * 0.6 / 2
        bw2   = 2 * t_eff
        # RPA-enhanced effective pairing: V_eff grows toward QCP (χ→large) as
        # V_eff_rpa = (g_JT²/K) / rpa_cutoff  [worst-case; actual χ₀ computed in SCF]
        V_eff_bare = self.g_JT**2 / self.K_lattice
        V_eff_gD = V_eff_bare * g_t   # conservative estimate (rpa_factor=1)

        K_min = self.g_JT**2 / max(self.Delta_CF, 1e-9)
        V_eff_rpa_max = V_eff_bare / self.rpa_cutoff  # upper bound with full RPA

        print("  Primary inputs:")
        print(f"    t0={self.t0:.3f} eV  u={self.u:.2f}  "
              f"λ_SOC={self.lambda_soc:.3f} eV  Δ_tet={self.Delta_tetra:.3f} eV")
        print(f"    g_JT={self.g_JT:.3f} eV/Å  λ_JT={self.lambda_jt:.4f}  "
              f"η={self.eta:.3f}  δ₀={self.doping_0:.3f}")
        # Γ₇ internal split (Γ₇a–Γ₇b): must be > 0 to prevent spontaneous JT
        from scipy.linalg import eigh as _eigh
        _Hsoc = _build_soc_cf_hamiltonian(
            self.lambda_soc, self.Delta_tetra, self.Delta_inplane)
        _ev = np.linalg.eigvalsh(_Hsoc)
        _g7split = float(_ev[4] - _ev[2])   # Γ₇a–Γ₇b gap
        print("  Derived:")
        print(f"    U={self.U:.3f} eV  Δ_CF={self.Delta_CF:.4f} eV  "
              f"U_mf={self.U_mf:.4f} eV")
        spont_jt_risk = '⚠ SPONTÁN JT VESZÉLY' if _g7split < 2*self.kT else '✓ OK'
        print(f"    Γ₇a–Γ₇b split={_g7split:.4f} eV  [{spont_jt_risk}]  "
              f"(Δ_ip={self.Delta_inplane:.4f} eV)")
        print(f"    K={self.K_lattice:.3f} eV/Å²  (K_min={K_min:.3f})  "
              f"λ_hop={self.lambda_hop:.3f} Å")
        print(f"  Coexistence check at δ={delta}  (g_t={g_t:.3f}, g_J={g_J:.3f}):")
        ok_m = h_afm < bw2
        ok_p = V_eff_gD > self.kT
        ok_q = self.Delta_CF < 10*self.kT
        ok_j = self.lambda_jt < self.Delta_CF / self.t0
        # SC gap: V_eff·g_t/Δ_CF < 1 always holds (SC-trig JT ⟹ V_eff < Δ_CF).
        # The gap is Q-driven: Δ ≈ [V_eff·g_t·g_JT / (Δ_CF - V_eff·g_t)] · Q
        # We check at a representative Q = 0.005 Å and require Δ_est > kT.
        sc_suppression = max(self.Delta_CF - V_eff_gD, 1e-9)  # Δ_CF - V_eff·g_t > 0 always
        Q_est    = 0.005   # Å representative small distortion
        Delta_est = V_eff_gD * self.g_JT * Q_est / sc_suppression
        # lambda_eff = V_eff_bare·N(0): BCS coupling constant (should be 0.1–1.5 for BCS validity)
        t_eff = g_t * self.t0
        N0 = 1.0 / (np.pi * max(t_eff, 1e-6))
        lambda_eff = V_eff_bare * N0
        ok_sc = Delta_est > self.kT
        print(f"    {'✓' if ok_m else '✗'} Metallic    h_afm={h_afm:.4f} < bw/2={bw2:.4f} eV")
        print(f"    {'✓' if ok_p else '✗'} Pairing     V_eff·gΔ={V_eff_gD:.4f} eV  kT={self.kT:.4f} eV"
              f"  λ_eff={lambda_eff:.3f}  (RPA max: {V_eff_rpa_max*g_t:.4f} eV)")
        print(f"    {'✓' if ok_sc else '✗'} SC gap (Q-driven)  Δ_est={Delta_est*1000:.2f} meV"
              f" @ Q={Q_est} Å  (suppression={1-V_eff_gD/self.Delta_CF:.3f})")
        print(f"    {'✓' if ok_q else '✗'} Quasi-deg   Δ_CF={self.Delta_CF:.4f} < 10kT={10*self.kT:.4f} eV")
        print(f"    {'✓' if ok_j else '✗'} SC-trig JT  K={self.K_lattice:.3f} > K_min={K_min:.3f} eV/Å²"
              f"  (λ_JT={self.lambda_jt:.4f} < Δ_CF/t0={self.Delta_CF/self.t0:.4f})")

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
    2-site cluster treatment of AFM quantum fluctuations
    
    Physical picture:
    - Cluster contains sites A and B (AFM sublattices)
    - Within cluster: exact diagonalization of MULTIPOLAR exchange O ⊗ O
    - Boundary: mean-field coupling to external magnetization
    - Cluster dimension: 2 sites × 4 orbitals = 8 states per site
    
    What this includes:
    ✓ Quantum multipolar correlations via exchange operator
    ✓ Correct orbital mixing and spin-orbit coupling
    ✓ Finite-temperature thermal fluctuations
    
    What this does NOT include:
    ✗ Fermi statistics (no Pauli exclusion between sites)
    ✗ Charge fluctuations ⟨n_A n_B⟩
    ✗ True many-body quantum fluctuations beyond multipolar sector
    
    This is a controlled approximation valid when:
    - Multipolar degrees of freedom dominate over charge fluctuations
    - System is in weak-coupling limit (not Mott insulator)
    - AFM correlations are captured by effective exchange J
    """
    
    def __init__(self, params: ModelParams):
        self.p = params
        self.CLUSTER_SIZE = 2
        self.Z_BOUNDARY = params.Z - 1  # One link is within cluster, Z-1 are boundary
    
    def build_multipolar_operator(self, eta: float) -> np.ndarray:
        """
        Multipolar operator O = (P₆ + η·P₇) ⊗ σz in basis [6↑, 6↓, 7↑, 7↓].
        P₆ = diag(1,1,0,0), P₇ = diag(0,0,1,1) are orbital projectors; σz gives spin polarization.
        Returns 4×4 diagonal matrix.
        """
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
        Mean-field boundary coupling: H_boundary = Z_boundary · (g_J·U_mf/2 + J_eff) · M_ext · O, where J_eff = g_J·4t²/U
        Matches BdG Weiss field (Stoner + Heisenberg, both renormalized by g_J under Gutzwiller projection).
        U_mf is the ZSA charge-transfer value passed from ModelParams.U_mf.
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

        Captures: quantum orbital mixing (Γ6↔Γ7), SOC-coupled multipolar exchange, thermal fluctuations.
        Does NOT capture: fermionic antisymmetrization between sites, charge-transfer fluctuations.
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
        Thermal expectation value ⟨O⟩ = Tr[ρO]/Tr[ρ] from a pre-computed eigensystem, where ρ = exp(-H_cluster/kT) is the density matrix
        evals, evecs: output of eigh(H_cluster) — called ONCE per cluster solve.
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
        # Vectorised: diag of evecs^† O evecs, Boltzmann weighted average over states
        Oevecs = Operator @ evecs          # (16,16)
        diag   = np.einsum('ij,ij->j', evecs.conj(), Oevecs)   # ⟨n|O|n⟩ for each n
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

        # chi0 grid (even) — SEPARATE from the SCF grid
        # k_even uses endpoint=False so the grid is [−π, π) with spacing 2π/nk_even.
        # Adding (π,π) to grid point i gives grid point i + nk_even/2 (mod nk_even),
        # which is guaranteed to be another grid point.
        k_even = np.linspace(-np.pi, np.pi, nk_even, endpoint=False)
        KX_ev, KY_ev = np.meshgrid(k_even, k_even)
        self.k_points_even   = np.column_stack((KX_ev.flatten(), KY_ev.flatten()))
        self.N_k_even        = len(self.k_points_even)
        # Uniform weights for the even grid (trapezoidal / rectangular rule)
        self.k_weights_even  = np.full(self.N_k_even, 1.0 / self.N_k_even)
        # Precompute the index shift for q_AFM=(π,π):
        # point i → i XOR (nk_even//2 * nk_even + nk_even//2) in row-major layout
        half = nk_even // 2
        # Full index map: for flat index i = iy*nk + ix,
        #   i_Q = ((iy+half)%nk)*nk + ((ix+half)%nk)
        iy_all  = np.arange(self.N_k_even) // nk_even
        ix_all  = np.arange(self.N_k_even) %  nk_even
        self.chi0_Q_idx = ((iy_all + half) % nk_even) * nk_even + (ix_all + half) % nk_even
        # self.chi0_Q_idx[i] = index of k_i + Q_AFM in k_points_even

        print(f"Initialized RMFT solver: {self.N_k} k-points (SCF/Simpson, odd grid nk={nk_odd})")
        print(f"                         {self.N_k_even} k-points (χ₀ even grid nk={nk_even}, commensurate q_AFM)")
        
        # ── Irrep projectors onto the Γ₆ and Γ₇ Kramers doublets ──────────────
        # Basis [6↑, 6↓, 7↑, 7↓].
        # They encode the symmetry selection rule of the SC-activated JT effect:
        #   • Pure AFM: low-energy space = P6·H.  Rank-2 operators (τx, τy)
        #     mix Γ₆↔Γ₇ → off-diagonal → ⟨τx⟩ = 0 (B₁g JT forbidden).
        #   • With Cooper pairing: space enlarges to (P6 ⊕ P7)·H.
        #     τx becomes block-diagonal → ⟨τx⟩ ≠ 0 → JT unlocked.
        #
        # Used in: build_irrep_selection_projector(),
        #          compute_rank2_multipole_expectation(),
        #          build_pairing_block_with_irrep_mixing().
        self.P6 = np.diag([1.0, 1.0, 0.0, 0.0])   # Γ₆ projector (4×4, real)
        self.P7 = np.diag([0.0, 0.0, 1.0, 1.0])   # Γ₇ projector (4×4, real)
        # τx: rank-2 quadrupolar operator that is TILTTED under AFM (off-diagonal
        # in the Γ₆-only basis) but becomes ALLOWED once SC mixes in Γ₇.
        # τx = P6·σ_x(orbital)·P7 + h.c. in [6↑,6↓,7↑,7↓] basis:
        self.tau_x_op = np.zeros((4, 4), dtype=complex)
        self.tau_x_op[0, 2] = self.tau_x_op[2, 0] = 1.0  # 6↑ ↔ 7↑
        self.tau_x_op[1, 3] = self.tau_x_op[3, 1] = 1.0  # 6↓ ↔ 7↓

        # ── d-wave / B₁g form factor φ(k) ───────────────────────────────────
        # φ(k) = cos kx − cos ky  (d_{x²−y²} symmetry, B₁g irrep of D₄h)
        # This is the same irrep as the JT distortion: the gap and the distortion
        # live in the SAME representation channel, implementing the self-closure
        # condition of the SC-activated JT mechanism.
        # For s-wave (d_wave=False) φ=1, which loses the irrep alignment.
        if params.d_wave:
            self.phi_k = (np.cos(self.k_points[:, 0] * params.a)
                          - np.cos(self.k_points[:, 1] * params.a))
        else:
            self.phi_k = np.ones(self.N_k)

        # ── Static Einstein phonon propagator D = 2/ω_JT ─────────────────────
        # D(q) = 2ω_JT / (ω_JT² - ω²)|_{ω→0} = 2/ω_JT (static limit).
        # Used in momentum-dependent V(k,k') = g_JT²·D·φ(k)·φ(k') for the
        # gap equation; does NOT change K_lattice = g_JT²/(lambda_jt·t0).
        self._D_phonon: float = 2.0 / max(params.omega_JT, 1e-6)

        # ── Bare & RPA-enhanced pairing interactions ──────────────────────────
        V_eff_bare = params.g_JT**2 / params.K_lattice   # scalar, q-independent
        
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
        g_Delta = g_t   # equal in single-band Gutzwiller; kept separate for conceptual clarity and future multi-orbital extension
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
        Regularized superexchange J(Q, doping) = g_J * 4*<t²(Q)>/U * f(doping)

        Interpolating factor f(doping) = doping / (doping + doping_0):
          - f -> 0 at half-filling (doping -> 0): J is suppressed, avoiding the unphysical g_J->4 divergence in the bare Gutzwiller approximation,
            where coherent spectral weight actually vanishes near the Mott insulator.
          - f -> 1 at large doping: bare Gutzwiller superexchange is recovered.

        delta: doping (1 - n), must be supplied explicitly by the caller.
        tx_bare, ty_bare: RAW Q-dependent hoppings (not Gutzwiller-renormalized).
        g_J encodes Gutzwiller exchange enhancement; f(delta) provides the Mott cutoff.
        """
        abs_doping = max(abs(doping), 1e-6)
        f_doping = abs_doping / (abs_doping + self.p.doping_0)
        t_sq_avg = 0.5 * (tx_bare**2 + ty_bare**2)
        return g_J * 4.0 * t_sq_avg / self.p.U * f_doping
    
    def dispersion(self, k: np.ndarray, tx: float, ty: float) -> float:
        """γ(k) = -2[tx·cos(kx·a) + ty·cos(ky·a)]. B₁g anisotropy: tx ≠ ty when Q ≠ 0.
        Used for hopping between A and B sublattices"""
        return -2.0 * (tx * np.cos(k[0] * self.p.a) + ty * np.cos(k[1] * self.p.a))
    
    def fermi_function(self, E: np.ndarray) -> np.ndarray:
        """The Fermi–Dirac distribution f(E) with μ is already included in the Hamiltonian; clipped to [-100, 100] to prevent overflow."""
        arg = E / self.p.kT
        arg = np.clip(arg, -100, 100) # to avoid overflow for numerical stability
        return 1.0 / (1.0 + np.exp(arg))
    
    # =========================================================================
    # 3.2b  IRREP PROJECTION & MULTIPOLAR ALGEBRA  (symmetry selection rules)
    # =========================================================================

    def build_irrep_selection_projector(self, Delta: complex) -> np.ndarray:
        """
        Construct the 4×4 effective Hilbert-space projector that encodes the
        SC-activated symmetry lifting of the B₁g JT mode.

        Pure AFM state  (Delta = 0):
            P_eff = P6  → only Γ₆ Kramers doublet is low-energy.
            τx = P6·τx·P7 + h.c. is STRICTLY OFF-DIAGONAL in this subspace.
            ⟨τx⟩ = 0 by symmetry (rank-2 multipole forbidden in P6 space).

        SC-condensed state  (Delta ≠ 0):
            P_eff = P6 ⊕ w·P7, where w = |Δ|/Δ_CF ∈ (0,1].
            τx acquires a DIAGONAL block within the (P6⊕P7) subspace.
            ⟨τx⟩ ≠ 0 is now algebraically allowed → JT distortion unlocked.

        The weight w = min(|Δ|/Δ_CF, 1) interpolates smoothly between the two
        limits and measures how much SC coherence has lifted the symmetry barrier.
        This is the algebraic implementation of the core hypothesis.

        Returns:
            4×4 projector P_eff (real, diagonal in the Γ₆/Γ₇ block structure).
        """
        Delta_CF = max(self.p.Delta_CF, 1e-9)
        # SC mixing weight: how much Γ₇ character is unlocked by Cooper pairs
        w = float(np.clip(abs(Delta) / Delta_CF, 0.0, 1.0))
        # P_eff = P6 + w·P7 (diagonal in [6↑,6↓,7↑,7↓])
        P_eff = self.P6 + w * self.P7
        return P_eff

    def compute_rank2_multipole_expectation(self, Delta: complex,
                                            tau_x_bdg: float) -> Dict:
        """
        Measure the algebraic lifting of the rank-2 B₁g multipole ⟨τx⟩ by SC.

        In the AFM-only ground state the effective projector is P6, and
            ⟨τx⟩_P6 ≡ Tr[P6·τx·ρ·P6] / Tr[P6·ρ·P6] = 0  (algebraic tilt)
        because τx is strictly off-diagonal between Γ₆ and Γ₇.

        When SC mixes in Γ₇ with weight w = |Δ|/Δ_CF, the projected operator
            τx_eff = P_eff · τx · P_eff
        acquires nonzero diagonal elements, and ⟨τx⟩ grows with |Δ|.

        This function returns the 'selection ratio' R = ⟨τx⟩_actual / ⟨τx⟩_free:
          R ≈ 0  → rank-2 multipole still algebraically suppressed (AFM regime)
          R → 1  → full lifting (SC-driven Γ₆⊕Γ₇ space, JT allowed)

        It also returns a boolean flag 'jt_algebraically_allowed' which is True
        only when R exceeds a threshold (default 0.05), i.e. when the symmetry
        barrier has been measurably lifted by SC coherence.

        Args:
            Delta      : current SC gap amplitude
            tau_x_bdg  : BdG k-space average of ⟨τx⟩ (from compute_observables)

        Returns:
            dict with keys: 'w', 'selection_ratio', 'jt_algebraically_allowed',
                            'tau_x_projected', 'tau_x_free_max'
        """
        Delta_CF = max(self.p.Delta_CF, 1e-9)
        w = float(np.clip(abs(Delta) / Delta_CF, 0.0, 1.0))

        # Maximum ⟨τx⟩ achievable in the fully mixed (P6⊕P7) space:
        # τx has eigenvalues ±1 in the mixed basis → max expectation = 1
        tau_x_free_max = 1.0

        # Projected τx amplitude in the SC-extended space:
        # ⟨τx⟩_eff ≈ w · τx_bdg  (suppressed linearly by the mixing weight)
        tau_x_projected = w * abs(tau_x_bdg)

        # Selection ratio: measures fractional lifting of the symmetry barrier
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

        Uses the EVEN k-grid (self.k_points_even) so that k + Q_AFM maps exactly
        to another grid point via the precomputed index self.chi0_Q_idx[i]:
            k_i + (π,π)  →  k_points_even[chi0_Q_idx[i]]
        No interpolation, no aliasing, no wrapping error.

        Formula (BdG coherence-factor weighted, full Nambu basis):
            χ₀ = (1/N) Σ_k Σ_{n,m} |M_{nm}(k,k+Q)|² · (f_n - f_m) / (E_m - E_n + iη)

        Coherence factors M_{nm}(k, k+Q):
            M_{nm} = ⟨ψ_n(k) | Ŝ_z^{4×4} ⊗ I_{p-h} | ψ_m(k+Q)⟩
        where Ŝ_z in the [6↑,6↓,7↑,7↓] basis is diag(+1,-1,+η,-η) on sublattice A
        (the staggered spin operator in the BdG Nambu space).

        Returns:
            dict with keys:
              'chi0'        : float, static susceptibility (eV⁻¹)
              'U_eff_chi'   : float, renormalised magnetic coupling used in Stoner denominator (eV),  NOT the bare Hubbard U. This keeps U_eff_chi · χ₀ ~ O(1) within the ordered AFM phase
              'stoner_denom': float, 1 - U_eff_chi · chi0
              'afm_unstable': bool, True if stoner_denom ≤ 0 (AFM QCP crossed, magnetically unstable)
              'chi_tau'     : float, multipolar susceptibility χ_τx ~ N(0)/(1 + α·M²). Non-zero only when U/t < (U/t)_c ≈ 2.2–2.5 (2D square lattice).
                              Physically: M² < 1 − (U/t)_c/(U/t) required for JT activation. If chi_tau ≈ 0, the JT channel is suppressed by AFM order.
        """
        # Spin operator in 4-orbital BdG basis (particle-particle block only):
        # S_z = diag(+1, -1, +eta, -eta) on sublattice A;
        # in the full 16×16 Nambu basis: S_z appears at [0:4,0:4] with sign +1, at [4:8,4:8] with sign -1 (staggered sublattice B), and at [8:12,8:12] with sign -1, at [12:16,12:16] with sign +1 (hole sector).
        sz_orb   = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
        Sz_bdg   = np.zeros((16, 16))
        Sz_bdg[np.arange(4),  np.arange(4)]  =  sz_orb   # particle A
        Sz_bdg[np.arange(4)+4,np.arange(4)+4]= -sz_orb   # particle B (staggered)
        Sz_bdg[np.arange(4)+8, np.arange(4)+8] = -sz_orb # hole A (p-h conjugate)
        Sz_bdg[np.arange(4)+12,np.arange(4)+12]=  sz_orb # hole B

        chi0 = 0.0

        for i in range(self.N_k_even):
            kvec  = self.k_points_even[i]
            i_Q   = self.chi0_Q_idx[i]            # exact index of k + Q_AFM
            kQvec = self.k_points_even[i_Q]
            w_i   = self.k_weights_even[i]

            # chi0: spin susceptibility uses current pairing state
            H_k  = self.build_bdg_matrix(kvec,  M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            H_kQ = self.build_bdg_matrix(kQvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)

            E_k,  V_k  = eigh(H_k)
            E_kQ, V_kQ = eigh(H_kQ)
            f_k   = self.fermi_function(E_k)
            f_kQ  = self.fermi_function(E_kQ)

            # Matrix elements M_{nm} = ⟨ψ_n(k)|S_z|ψ_m(k+Q)⟩
            # Efficient: Sz·V_kQ first, then dot with V_k†
            SzV_kQ = Sz_bdg @ V_kQ        # (16, 16)
            M_mat  = V_k.conj().T @ SzV_kQ  # (16, 16): M[n,m] = ⟨n,k|Sz|m,k+Q⟩

            for n in range(16):
                for m in range(16):
                    df = f_k[n] - f_kQ[m]
                    dE = E_kQ[m] - E_k[n]
                    if abs(df) < 1e-12 or abs(dE) < 1e-6:
                        continue
                    chi0 += w_i * abs(M_mat[n, m])**2 * df / dE

        # Renormalised magnetic coupling for Stoner denominator: U_eff_chi = g_J · J_eff = g_J · (4t_eff²/U) · f(delta)
        # This is the already-Gutzwiller-renormalised exchange, not the bare U. It ensures U_eff_chi · chi0 ~ O(1) in the ordered phase.
        abs_delta  = max(abs(target_doping), 1e-6)
        f_d        = abs_delta / (abs_delta + self.p.doping_0)
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t2         = 0.5 * (tx_bare**2 + ty_bare**2)
        J_eff_now  = g_J * 4.0 * t2 / self.p.U * f_d
        U_eff_chi  = g_J * J_eff_now    # renormalised coupling ~ 0.05–0.3 eV

        stoner_denom = 1.0 - U_eff_chi * chi0
        afm_unstable = stoner_denom <= 0.0

        # Multipolar susceptibility χ_τx ~ N(0) / (1 + α·M²)
        # Condition for χ_τx ≠ 0: U/t < (U/t)_c, i.e. M is below the AFM saturation limit.
        # M² ≈ 1 − (U/t)_c / (U/t) in the ordered phase; M is too large (ordered phase, not fluctuating), χ_τx ≈ 0 → JT suppressed.
        t_eff_avg = np.sqrt(t2)   # RMS hopping (already Q-dependent via tx_bare, ty_bare)
        N0 = 1.0 / (np.pi * max(t_eff_avg, 1e-6))   # bare DOS for 2D tight-binding (van Hove plateau near half-filling).
        Ut_ratio    = self.p.U / max(t_eff_avg, 1e-6)
        Ut_critical = 2.35   # On the 2D square lattice near half-filling: midpoint of (U/t)_c ≈ 2.2–2.5
        alpha_M     = max(Ut_ratio / Ut_critical - 1.0, 0.0)   # α = U/Uc − 1 ≥ 0
        chi_tau     = N0 / (1.0 + alpha_M * M**2)   # suppressed when M large or U >> Uc

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
        RPA Stoner enhancement factor 1 / (1 - U_eff_chi · χ₀).

        Uses the RENORMALISED coupling U_eff_chi = g_J · J_eff (from the chi0 dict),
        NOT the bare Hubbard U.  This keeps the denominator positive and O(1) within
        the AFM-ordered phase, giving a realistic 10–50% pairing enhancement.

        If stoner_denom ≤ 0 (AFM QCP crossed, magnetically unstable):
            Returns rpa_factor = 1.0 (no enhancement) and logs a warning.
            The caller should treat this as a signal to reduce M or adjust parameters,
            NOT as a reason to clip the denominator to an arbitrary floor.

        The enhancement is applied SMOOTHLY via an outer RPA loop in
        solve_self_consistent, not as a sudden V_eff → V_eff_RPA jump.

        Returns:
            rpa_factor : float ≥ 1.0
        """
        sd = chi0_result['stoner_denom']
        # rpa_cutoff clamps the denominator from below: prevents divergence near AFM QCP while preserving smooth enhancement up to 1/rpa_cutoff ≈ 8.3× (at 0.12).
        # If sd ≤ 0 (Stoner instability), fall back to rpa_cutoff floor, not 1.0:
        # even at the QCP the system still has a finite (if large) pairing enhancement.
        if sd <= 0.0:
            return 1.0  # AFM instability: suppress enhancement, don't amplify
        return 1.0 / max(sd, self.p.rpa_cutoff)

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
        F_AA_kp = np.zeros(self.N_k)   # on-site channel s: u_A·v_A*
        F_AB_kp = np.zeros(self.N_k)   # inter-site channel d: u_A·v_B*

        for i, kvec in enumerate(self.k_points):
            H  = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            ev, ec = eigh(H)
            obs = self.compute_observables_from_bdg(ev, ec)
            F_AA_kp[i] = float(np.real(obs['Pair_s']))
            F_AB_kp[i] = float(np.real(obs['Pair_d']))

        V_total = (self.p.g_JT**2 / max(self.p.K_lattice, 1e-9)) * rpa_factor
        V_s     = (1.0 - self.p.channel_mix) * V_total
        V_d     = self.p.channel_mix          * V_total

        dot_s = float(np.dot(self.k_weights,               F_AA_kp))  # no φ weight
        dot_d = float(np.dot(self.k_weights * self.phi_k,  F_AB_kp))  # φ weight (one power)

        Delta_s_new = abs(g_Delta * V_s * dot_s)
        Delta_d_new = abs(g_Delta * V_d * dot_d)

        return Delta_s_new, Delta_d_new

    # =========================================================================
    # 3.3 BdG HAMILTONIAN CONSTRUCTION
    # =========================================================================
    
    def build_local_hamiltonian_for_bdg(self, sign_M: float, M: float, Q: float,
                                        mu: float, g_J: float, target_doping: float) -> np.ndarray:
        """
        Local 4×4 BdG Hamiltonian for one sublattice, basis [6↑, 6↓, 7↑, 7↓].
        sign_M = ±1 for sublattices A/B (staggered AFM).
        Includes: chemical potential, crystal field, Stoner-Heisenberg Weiss field, JT orbital mixing.
        Weiss field: h = g_J·(U_mf/2 + Z·2t²/U)·M/2; both Hartree-Fock and superexchange terms
        renormalize with g_J under Gutzwiller projection.
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
        
        # 4. JT distortion field (orbital mixing, spin-conserving)
        h_jt = self.p.g_JT * Q
        H[0, 2] = H[2, 0] = h_jt  # 6↑ ↔ 7↑
        H[1, 3] = H[3, 1] = h_jt  # 6↓ ↔ 7↓
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
        # On-site energy for sublattices from staggered AFM Weiss field: +M on A, −M on B
        H_A = self.build_local_hamiltonian_for_bdg(sign_M=+1.0, M=M, Q=Q, mu=mu, g_J=g_J, target_doping=target_doping)
        H_B = self.build_local_hamiltonian_for_bdg(sign_M=-1.0, M=M, Q=Q, mu=mu, g_J=g_J, target_doping=target_doping)

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
        Thermal expectation values from BdG eigensystem. PRIMARY source for M and Q:
        BdG eigenstates carry Γ₆–Γ₇ mixing induced by Δ, implementing SC → JT feedback.

        Spinor: ψ_n = [u_A(4), u_B(4), v_A(4), v_B(4)], each block [6↑, 6↓, 7↑, 7↓].
        u = particle amplitude, v = hole amplitude.

        Formulas (BdG):
        1. Density:      ⟨c†c⟩  = Σ_n [|u_n|² f(E_n) + |v_n|² (1−f(E_n))]
        2. Magnetization: ⟨S_z⟩ = Σ_n [(u*sz*u) f + (v*sz*v) (1−f)]
        3. Quadrupole:   ⟨τ_x⟩  = Σ_n [(u*τx*u) f + (v*τx*v) (1−f)]
        4. Pairing (inter-site A↔B):
             F_n = u_A[0]·v_B[3]* − u_A[1]·v_B[2]*   (6↑-7↓ singlet, A-part ↔ B-hole)
             ⟨c_i c_j⟩ = Σ_n Re(F_n) · (1−2f(E_n))
           The (1−2f) factor is critical for correct finite-T BCS behavior and Tc estimation.
           pair_BA = u_B[0]·v_A[3]* − u_B[1]·v_A[2]* is the h.c. channel; averaging gives Re(pair_AB).
        """
        # Fermi occupation factors
        f = self.fermi_function(eigvals)
        one_minus_f = 1.0 - f
        one_minus_2f = 1.0 - 2.0 * f  # For pairing
        
        # Initialize accumulators for both sublattices
        dens_A = 0.0
        dens_B = 0.0
        mag_A = 0.0
        mag_B = 0.0
        quad_A = 0.0
        quad_B = 0.0
        pair_sum_s = 0.0   # on-site orbital B₁g channel (φ=1)
        pair_sum_d = 0.0   # inter-site d-wave B₁g channel (φ(k) embedded in BdG)
        
        # Sum over all eigenstates
        for n in range(16):
            psi = eigvecs[:, n]
            fn = f[n]
            fn_bar = one_minus_f[n]
            fn_pair = one_minus_2f[n]
            
            # Extract spinor components
            u_A = psi[0:4]    # Particle amplitudes, sublattice A
            u_B = psi[4:8]    # Particle amplitudes, sublattice B
            v_A = psi[8:12]   # Hole amplitudes, sublattice A
            v_B = psi[12:16]  # Hole amplitudes, sublattice B
            
            # --- 1. PARTICLE DENSITY: ⟨c†c⟩ = Σ_n [|u_n|² f(E_n) + |v_n|² (1-f(E_n))] ---
            # CRITICAL: In BdG, only sum over states n where E_n > 0, otherwise we double count! But for simplicity, we count all and divide by 2 at end.
            dens_A += np.sum(np.abs(u_A)**2) * fn + np.sum(np.abs(v_A)**2) * fn_bar
            dens_B += np.sum(np.abs(u_B)**2) * fn + np.sum(np.abs(v_B)**2) * fn_bar
            
            # --- 2. MAGNETIZATION & QUADRUPOLE ---
            # Use class-level helper
            mA, tauA = self._compute_site_magnetization_and_quadrupole(psi, slice(0, 4), slice(8, 12), fn, fn_bar)
            mB, tauB = self._compute_site_magnetization_and_quadrupole(psi, slice(4, 8), slice(12, 16), fn, fn_bar)
            
            mag_A += mA
            mag_B += mB
            quad_A += tauA
            quad_B += tauB
            
            # --- 3. PAIRING AMPLITUDES — both B₁g channels ---
            # Channel s (on-site orbital singlet, φ=1): F_AA = u_A·v_A*
            # Channel d (inter-site d-wave, φ(k) in BdG): F_AB = u_A·v_B*
            # The (1−2f) factor: ⟨c_i c_j⟩ = Σ_n u_i·v_j* (1 − 2f(E_n)).
            # F_AA and F_AB are different matrix elements → no double-counting.
            pair_onsite  = (u_A[0] * np.conj(v_A[3]) -   # A:6↑ ↔ A:7↓ (channel s)
                            u_A[1] * np.conj(v_A[2]))     # A:6↓ ↔ A:7↑
            pair_onsite += (u_B[0] * np.conj(v_B[3]) -   # B:6↑ ↔ B:7↓ (channel s, site B)
                            u_B[1] * np.conj(v_B[2]))
            pair_AB = (u_A[0] * np.conj(v_B[3]) -        # A:6↑ ↔ B:7↓ (channel d)
                       u_A[1] * np.conj(v_B[2]))          # A:6↓ ↔ B:7↑
            pair_BA = (u_B[0] * np.conj(v_A[3]) -        # B:6↑ ↔ A:7↓ (channel d, h.c.)
                       u_B[1] * np.conj(v_A[2]))
            pair_sum_s += pair_onsite * fn_pair           # on-site accumulator
            pair_sum_d += 0.5 * (pair_AB + pair_BA) * fn_pair   # inter-site accumulator
        
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
        Grand potential per site from the k-space BdG spectrum.

        Ω = (1/2) Σ_{k,n} w_k [E_n f_n − T S(f_n)]
            + |Δ|² / (g_Δ·V_eff)
            + (K/2) Q²

        The 1/2 factor reflects the doubled unit cell of the 16×16 BdG matrix.

        The +|Δ|²/(g_Δ·V_eff) term is the condensation cost: mean-field decoupling price..
        The quasiparticle spectrum already contains the condensation gain (gap opening lowers E_n);
        without this term the gap equation ∂F/∂Δ = 0 would be inconsistent and Δ would diverge.

        In the adiabatic JT limit: V_eff = g_JT² / K_lattice.
        This sets the overall pairing strength (λ_eff = V_eff·N(0)).
        The phonon propagator affects the momentum structure V(k,k′), not the magnitude of the condensation-energy cost.

        (K/2)Q² is the elastic energy of the lattice distortion.
        JT stabilization: E_JT = g_JT²/(2K), from Q* = g_JT/K.
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

        # Controlled choice: V_eff = g²/K (adiabatic, Born–Oppenheimer limit).
        #   • JT stabilization E_JT = g²/(2K) follows from the same scale.
        #   • The pairing kernel V(k,k') ∝ g²·D·φφ' is normalized so that
        #     its k-average reproduces g²/K (see solve_gap_equation_k).
        # Using g²/ω (Fröhlich) would overshoot V_eff and push λ ≫ 1 (Eliashberg regime), while g²/K keeps λ ~ O(1) (BCS controlled).
        # RPA renormalizes the electronic gain (BdG spectrum), not the field cost.
        V_total_rpa = (self.p.g_JT**2 / max(self.p.K_lattice, 1e-9)) * rpa_factor
        V_s = (1.0 - self.p.channel_mix) * V_total_rpa
        V_d = self.p.channel_mix          * V_total_rpa
        # Condensation cost: |Δ|²/(g_Δ·V) for each channel (MF decoupling price)
        cond_s = (abs(Delta_s)**2 / (g_Delta * V_s) if (V_s > 1e-12 and abs(Delta_s) > 1e-10) else 0.0)
        cond_d = (abs(Delta_d)**2 / (g_Delta * V_d) if (V_d > 1e-12 and abs(Delta_d) > 1e-10) else 0.0)
        condensation_correction = cond_s + cond_d
        # Elastic restoring cost ½KQ²; gain (−g·Q·⟨τx⟩) is already in the BdG spectrum.
        elastic_energy = 0.5 * self.p.K_lattice * Q**2

        # Convert per unit cell → per site, then add field costs
        Omega_per_site = Omega_cell / 2.0 + elastic_energy + condensation_correction
        return Omega_per_site
    
    def compute_cluster_free_energy(self, M: float, Q: float, mu: float, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> Dict:
        """
        Compute cluster free energy from exact diagonalization
        
        F_cluster = -T log Z = -T log[Σ_i exp(-E_i/T)]
        
        tx_bare, ty_bare: RAW Q-dependent hoppings (not Gutzwiller-renormalized).
        effective_superexchange uses bare t so that g_J alone renormalizes J.
        
        IMPORTANT: This captures quantum AFM fluctuations that mean-field BdG misses!

        Returns:
            Dictionary with:
            - 'F_per_site': Free energy PER SITE
            - 'M': Staggered magnetization from cluster which contribute the final magnetization
            - 'Q_exp': Quadrupole expectation value ⟨τ_x⟩
            - 'Q_rms': RMS quadrupole √⟨τ_x²⟩ (includes fluctuations)
            - 'Q_fluctuation': Fluctuation strength
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
          1. BdG diagonalization → M_bdg, tau_x, Pair_kspace (includes SC→orbital feedback)
          2. Irrep selection check: compute_rank2_multipole_expectation() quantifies how
             much the SC gap has algebraically lifted the B₁g multipole barrier.
             JT is only driven when 'jt_algebraically_allowed' is True.
          3. RPA susceptibility χ₀(q_AFM) every 5 iterations → Stoner factor rpa_factor.
             V_eff → V_eff·rpa_factor enhances pairing near AFM QCP without changing
             Δ_CF or g_JT unrealistically.
          4. k-resolved gap equation Δ(k) = φ(k)·Δ̄ via solve_gap_equation_k()
             (replaces flat Δ_out = V_eff·g_Δ·Pair_kspace from v1).
          5. Cluster ED → M_cluster (quantum fluctuation correction, 20% weight)
          6. Hellmann-Feynman ∂F/∂M correction nudges M toward variational minimum
          7. Newton–LM step on Δ̄: ∂F/∂Δ = 0 treated variationally (same as M)
          8. Q_out from JT equilibrium -(g_JT/K)·⟨τ_x⟩, floored by cluster Q_rms bootstrap
          9. Anderson mixing on [M, Q]; simple mixing on Δ
         10. Brent root-finding for μ; post-convergence Hessian test

        Returns converged dict with M, Q, Delta, mu, density, free energies, Gutzwiller factors,
        hessian result, full iteration history, chi0 and rpa_factor trajectories.
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
            'chi_tau': [], 'Ut_ratio': []
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

            # --- RPA spin-fluctuation enhancement (outer-loop strategy) ---
            # update χ₀(q_AFM) only when state changed enough (avoids V_eff jump).
            # Trigger: update chi0 only if max change since last update > chi0_tol, OR at iteration 0 (initialisation).
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

            # --- Dual-channel gap equations ---
            # V_s = (1−mix)·g²/K·rpa, V_d = mix·g²/K·rpa. Both solved from same BdG eigensystem.
            Delta_s_out, Delta_d_out = self.solve_gap_equation_k(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, g_Delta, rpa_factor)

            # --- Newton–LM for Δ (mathematically corrected curvature use) ---
            # For each channel c ∈ {s, d}: ∂F/∂Δ_c = Δ_c/(g_Δ·V_c) − F_c = 0.
            # V_s = (1−mix)·V_total, V_d = mix·V_total (matching solve_gap_equation_k).
            # LM-regularised Newton blended with gap-equation fixpoint via ALPHA_D.
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
            # SCF fixpoint M = ⟨Ŝ_z⟩_BdG ⇔ ∂F/∂M = 0, so a gradient correction is variationally consistent and helps avoid saddle points.
            # Gradient descent is ill-scaled (∂F/∂M ~ 10 eV vs ΔM ~ 0.1).
            # Use Newton step γ = 1/|∂²F/∂M²|, with ∂²F/∂M² ≈ [dF(M+ε) − dF(M−ε)] / (2ε).
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

            # --- Q_out: JT equilibrium + Bootstrap seeds if |Q_bdg|<1e-6 ---
            # Algebraic selection is enforced self-consistently: (Δ=0 → D=0 → ⟨τx⟩=0 → Q=0 naturally; explicit gating would break BdG–cluster consistency).
            Q_bdg = -(self.p.g_JT / self.p.K_lattice) * tau_x
            if abs(Q_bdg) < 1e-6:
                Q_exp_cl  = F_cluster_early['Q_exp']   # signed ⟨τ_x⟩
                seed_mag  = min(max(abs(Q_exp_cl), 1e-4), 0.005 * self.p.lambda_hop)
                Q_out     = np.sign(Q_exp_cl) * seed_mag if abs(Q_exp_cl) > 1e-6 else seed_mag
            else:
                Q_out = Q_bdg
            Q_out = float(np.clip(Q_out, -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # --- Anderson mixing on [M,Q]; valley jump resets history --- 
            x_in  = np.array([M,     Q    ])
            x_out = np.array([M_out, Q_out])
            scf_x_hist.append(x_in)
            scf_f_hist.append(x_out)

            x_new = self._anderson_mix(scf_x_hist, scf_f_hist, m=5)
            M_mixed    = float(np.clip(x_new[0], 0.0, 1.0))
            Q_mixed    = float(np.clip(x_new[1], -0.5 * self.p.lambda_hop, 0.5 * self.p.lambda_hop))

            # Reset Anderson history on valley jump (Q sign flip) to avoid stale extrapolation; a fresh start lets the system re-settle in the newly selected valley.
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

        totals = dict(M=0.0, Q=0.0, n=0.0, Pair_s=0.0, Pair_d=0.0, Pair=0.0)

        for i, kvec in enumerate(self.k_points):
            H = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            eigvals, eigvecs = eigh(H)
            obs = self.compute_observables_from_bdg(eigvals, eigvecs)

            weight = self.k_weights[i]
            for key in totals:
                totals[key] += weight * obs[key]
        return totals

    def compute_dF_dM(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float,
                      mu: float, tx: float, ty: float, g_J: float) -> float:
        """
        Variational gradient ∂F/∂M via Hellmann-Feynman theorem.

        ∂F/∂M = Σ_k,n f(E_n(k)) · ⟨ψ_n(k)| ∂H_BdG/∂M |ψ_n(k)⟩

        ∂H_BdG/∂M is the analytic Weiss-field operator embedded into 16×16 BdG:
          particle A: diag(-sz_orb),  hole A: diag(+sz_orb)
          particle B: diag(+sz_orb),  hole B: diag(-sz_orb)
        where sz_orb = h_prefactor · [+1, -1, +η, -η].

        Used actively in the SCF loop via a Newton step:
          M_newton = M - (1/|∂²F/∂M²|) · ∂F/∂M
        where ∂²F/∂M² ≈ [dF_dM(M+ε) - dF_dM(M-ε)] / 2ε.
        Plain gradient descent (fixed γ) is ill-conditioned because ∂F/∂M ~ O(10) eV
        (k-sum of Weiss matrix elements); the Newton curvature provides automatic scaling.

        Returns: ∂F/∂M in eV per site (double unit cell correction: /2).
        """
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        t_sq_avg = 0.5 * (tx_bare**2 + ty_bare**2)
        # Prefactor of the Weiss operator (derivative of h_afm w.r.t. M)
        h_prefactor = g_J * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) / 2.0

        # Weiss operator in 4-orbital basis: diag(+1,-1,+η,-η) × h_prefactor
        sz_orb = np.array([1.0, -1.0, self.p.eta, -self.p.eta]) * h_prefactor

        # Embed into 16×16 BdG:
        # particle sector: -sign_M * sz_orb on each sublattice
        # hole sector:     +sign_M * sz_orb* on each sublattice (particle-hole conjugate)
        dH_dM = np.zeros((16, 16), dtype=float)
        # Sublattice A (+1): particle rows 0:4, hole rows 8:12
        dH_dM[0, 0] = -sz_orb[0];  dH_dM[1, 1] = -sz_orb[1]
        dH_dM[2, 2] = -sz_orb[2];  dH_dM[3, 3] = -sz_orb[3]
        dH_dM[8, 8] = +sz_orb[0];  dH_dM[9, 9] = +sz_orb[1]
        dH_dM[10,10] = +sz_orb[2]; dH_dM[11,11] = +sz_orb[3]
        # Sublattice B (-1): particle rows 4:8, hole rows 12:16
        dH_dM[4, 4] = +sz_orb[0];  dH_dM[5, 5] = +sz_orb[1]
        dH_dM[6, 6] = +sz_orb[2];  dH_dM[7, 7] = +sz_orb[3]
        dH_dM[12,12] = -sz_orb[0]; dH_dM[13,13] = -sz_orb[1]
        dH_dM[14,14] = -sz_orb[2]; dH_dM[15,15] = -sz_orb[3]

        grad = 0.0
        for i, kvec in enumerate(self.k_points):
            H = self.build_bdg_matrix(kvec, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            eigvals, eigvecs = eigh(H)
            f_n = self.fermi_function(eigvals)
            for n in range(16):
                psi = eigvecs[:, n]
                grad += self.k_weights[i] * f_n[n] * np.real(psi @ (dH_dM @ psi))

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
# 5. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_phase_diagrams(solver: RMFT_Solver, initial_M: float, initial_Q: float, initial_Delta: float, doping_range: np.ndarray,
                        cf_min: float = 0.05, cf_max: float = 0.20, N_cf: int = 10):
    """
    Doping-scan phase diagram with warm-start and crystal-field sweet-spot search.

    Each doping point:
      - warm-starts from the previous converged solution (M, Q, Δ_s, Δ_d)
      - validates density constraint (error < 0.01 warned)
      - logs: phase tag (AFM/SC+JT/MIX/NM), M, Q, Δ_s, Δ_d, χ_τx, U/t_eff

    Doping is scanned from high to low (overdoped → underdoped) so warm-start
    begins in the well-behaved metallic phase and tracks toward the Mott boundary.

    Crystal-field scan finds the Δ_CF 'sweet spot' maximising total SC gap at
    the midpoint doping, restoring the original Δ_CF afterwards.
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
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
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
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  SC-Activated JT Model - Variational Free Energy Minimization     ║
    ║  Implements: SC → Γ₆–Γ₇ mixing → JT via ∂F/∂M = ∂F/∂Q = 0         ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    params = ModelParams(
        t0=0.328,
        u=4.821,
        lambda_soc=0.144,
        Delta_tetra=-0.13,
        g_JT=0.774,
        lambda_jt=0.417,
        lambda_hop=1.2,
        eta=0.09,
        doping_0=0.09,
        Delta_CT=1.234,
        omega_JT=0.060,
        rpa_cutoff=0.12,
        d_wave=True,
        Delta_inplane=0.02,
        mu_LM=5.0,
        ALPHA_HF=0.16,
        CLUSTER_WEIGHT=0.35,
        ALPHA_D=0.3,
        mu_LM_D=1.0,
        channel_mix=0.5,
        Z=4,
        nk=80,
        kT=0.011,
        a=1.0,
        max_iter=200,
        tol=1e-4,
        mixing=0.04
    )
    params.summary()      # print derived quantities and coexistence checks
    solver = RMFT_Solver(params)
    
    print("\n" + "="*70)
    print("SCANNING FULL PHASE DIAGRAM")
    print("="*70)
    
    fig = plot_phase_diagrams(solver, initial_M=0.25, initial_Q=0.06, initial_Delta=0.04, doping_range=np.linspace(0.13, 0.32, 5))
    plt.show()
    
    print(f"\n{'='*70}")
    print("Simulation complete!")
    print(f"{'='*70}\n")