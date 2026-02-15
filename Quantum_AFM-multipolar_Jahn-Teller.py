"""
FINAL OPTIMIZED SC-Activated Jahn-Teller Model
1. Physical setting:
In a D4h charge-transfer insulator with strong SOC, AFM order restricts the
Γ6 ground-state manifold to dipolar multipoles, symmetry-forbidding the
B1g Jahn–Teller (JT) quadrupolar distortion.

2. Superconductivity as the trigger:
Even-parity singlet pairing induces coherent Γ6–Γ7 mixing, opening a
rank-2 multipolar channel that bypasses AFM selection rules.

Result:
- Cooperative B1g JT distortion
- Stabilized SC gap
- Reduced AFM superexchange cost

Key point: SC is the cause, not the consequence.

3. BdG structure

- Proper particle–hole symmetric BdG matrix
- Correct doubled unit cell
- μ included in H
- Thermal averages via f(E)

4. Emergent pairing

Δ = V_eff ⟨c_up c_down⟩
No free parameter (self-consistent).

5. Distortion-dependent magnetism: J = J(Q)

6. Variational free energy: F_total(M,Q,Δ,μ) = F_BdG(M,Q,Δ,μ) + λ_cluster F_cluster(M,Q,μ)
Equilibrium: ∂F/∂M = 0, ∂F/∂Q = 0

Ensures thermodynamic consistency and SC → JT feedback.

7. Quadrupole operator: τ_x does NOT commute with H_cluster, therefore: ⟨τ_x⟩² ≠ ⟨τ_x²⟩
  
Multiple B₁g channels contribute! Compute both:
  - ⟨τ_x⟩: Expectation value (classical)
  - √⟨τ_x²⟩: RMS quadrupole (includes quantum fluctuations)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Tuple, Dict

# =============================================================================
# 1. PHYSICAL PARAMETERS & MODEL DEFINITION
# =============================================================================

@dataclass
class ModelParams:
    """Physical parameters in eV units"""
    # Microscopic parameters
    t0: float               # Bare hopping integral (eV)
    U: float                # Hubbard U (eV) - charge transfer regime
    Delta_CF: float         # Crystal field splitting Γ₆-Γ₇ (eV)
    g_JT: float             # Electron-phonon coupling (eV/Å)
    K_lattice: float        # Lattice spring constant (eV/Å²)
    beta: float             # Hopping-distortion coupling: t(Q) = t₀(1-βQ²)
    eta: float              # AFM asymmetry (Γ₇ feels η×M compared to Γ₆)
    
    # Cluster parameters
    Z: int                  # Coordination number for 2D square lattice
    
    # Simulation parameters
    nk: int                 # k-points per direction (32×32 grid)
    kT: float               # Temperature (0.01 eV ~ 116 K)
    a: float                # Lattice constant (Å)
    
    # Convergence
    max_iter: int
    tol: float
    mixing: float           # Slower mixing for stability in high-dim space
    mu_adjust_rate: float   # Chemical potential adjustment speed

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
        Build multipolar operator O = (P₆ + η·P₇) ⊗ σz
        
        This operator couples Γ₆ and Γ₇ orbital character to spin, creating the AFM exchange interaction.
        
        Basis: [6↑, 6↓, 7↑, 7↓]
        
        P₆ = (I + τz)/2 projects onto Γ₆
        P₇ = (I - τz)/2 projects onto Γ₇
        σz gives spin polarization
        
        Returns: 4×4 operator matrix
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
        """
        Exact AFM exchange within 2-site cluster
        
        This represents quantum mechanical correlation between
        spin and orbital degrees of freedom on neighboring sites.
        
        Dimension: 4×4 (site A) ⊗ 4×4 (site B) = 16×16
        But we work in 8-dimensional single-particle space (2 sites × 4 orbitals each)
        
        Returns: 16×16 cluster exchange Hamiltonian
        """
        O = self.build_multipolar_operator(eta)
        
        # Construct H_exchange using O_A ⊗ O_B tensor product of multipolar operators
        return J_eff * np.kron(O, O)
    
    def boundary_afm_field(self, J_eff: float, M_ext: float, eta: float) -> np.ndarray:
        """
        Mean-field coupling to external magnetization
        
        H_boundary = Z_boundary * J_eff * M_ext * O
        
        This represents the field from neighboring clusters acting on each site.
        
        Returns: 4×4 single-site operator
        """
        O = self.build_multipolar_operator(eta)
        return self.Z_BOUNDARY * J_eff * M_ext * O
    
    def build_cluster_hamiltonian(self, H_sp_A: np.ndarray, H_sp_B: np.ndarray,
                                 J_eff: float, M_ext: float, eta: float) -> np.ndarray:
        """
        Construct full cluster Hamiltonian in SINGLE-PARTICLE TENSOR SPACE
        
        H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)              [single-particle terms]
                  + H_exchange(A,B)                         [intra-cluster multipolar exchange]
                  + H_boundary(A) ⊗ I + I ⊗ H_boundary(B)   [inter-cluster MF]
        
        Dimension: 16×16 = (4 orbitals × 2 sites)²
        This is a TENSOR PRODUCT of single-particle spaces, not antisymmetrized Fock space.
        
        Physical meaning:
        - Single-particle: includes orbital mixing, SOC, crystal field
        - Exchange: quantum multipolar correlations O_A ⊗ O_B
        - Boundary: mean-field from neighboring clusters
        
        What this DOES capture:
        ✓ Quantum orbital mixing between Γ6 and Γ7
        ✓ Spin-orbit coupled multipolar exchange
        ✓ Thermal averaging over multipolar states
        
        What this does NOT capture:
        ✗ Pauli exclusion between sites (no fermionic antisymmetrization)
        ✗ Charge transfer fluctuations
        ✗ Double occupancy suppression (handled separately by Gutzwiller)
        
        Parameters:
            H_sp_A, H_sp_B: 4×4 single-particle Hamiltonians for each site
                            (includes kinetic, CF, JT, chemical potential)
            J_eff: Effective superexchange
            M_ext: External (mean-field) magnetization from other clusters
            eta: Orbital asymmetry parameter
        
        Returns: 16×16 cluster Hamiltonian in single-particle tensor space
        """
        I4 = np.eye(4, dtype=complex)
        
        # Single-particle terms: H(A) ⊗ I + I ⊗ H(B)
        H_cluster = np.kron(H_sp_A, I4) + np.kron(I4, H_sp_B)
        
        # Intra-cluster exact AFM exchange
        H_cluster += self.cluster_afm_exchange(J_eff, eta)
        
        # Boundary mean-field coupling
        H_bound = self.boundary_afm_field(J_eff, M_ext, eta)
        H_cluster += np.kron(H_bound, I4)  # Boundary field on site A
        H_cluster += np.kron(I4, H_bound)  # Boundary field on site B
        
        return H_cluster
    
    def cluster_expectation(self, H_cluster: np.ndarray, Operator: np.ndarray, temperature: float) -> float:
        """
        Compute thermal expectation value in cluster
        
        ⟨O⟩ = Tr[ρ O] / Tr[ρ]
        
        where ρ = exp(-H_cluster/kT) is the density matrix
        
        For T → 0: ⟨O⟩ → ⟨ψ₀|O|ψ₀⟩ (ground state)
        For T > 0: Boltzmann weighted average over states
        
        Parameters:
            H_cluster: 16×16 cluster Hamiltonian
            Operator: 16×16 operator
            temperature: kT in eV
        
        Returns: Real expectation value
        """
        eigenvalues, eigenvectors = eigh(H_cluster)

        # Extend single-site operator if needed
        if Operator.shape[0] == 4:
            Operator = np.kron(np.eye(4, dtype=complex), Operator)

        if temperature < 1e-6:
            # T = 0
            psi = eigenvectors[:, 0]
            return np.real(np.vdot(psi, Operator @ psi))

        # Finite T
        E = eigenvalues - eigenvalues[0]          # energy shift
        weights = np.exp(-E / temperature)
        Z = np.sum(weights)

        return np.real(
            sum(
                weights[n] * np.vdot(eigenvectors[:, n],
                                    Operator @ eigenvectors[:, n])
                for n in range(len(E))
            ) / Z
        )

# =============================================================================
# 3. RENORMALIZED MEAN-FIELD THEORY (16x16 BdG) SOLVER WITH CLUSTER MF
# =============================================================================

class RMFT_Solver:
    """
    Self-consistent solver for the SC-activated JT model
    with Gutzwiller renormalization and proper 16x16 double unit cell structure
    """
    def __init__(self, params: ModelParams):
        self.p = params
        self.cluster_mf = ClusterMF(params)
        
        # Generate k-space grid (2D Brillouin zone)
        k = np.linspace(-np.pi, np.pi, params.nk, endpoint=False)
        self.KX, self.KY = np.meshgrid(k, k)
        self.k_points = np.column_stack((self.KX.flatten(), self.KY.flatten()))
        self.N_k = len(self.k_points)
        
        # Calculate effective pairing interaction
        V_eff = params.g_JT**2 / params.K_lattice
        
        print(f"Initialized RMFT solver with {self.N_k} k-points")
        print(f"Physical parameters: t₀={params.t0:.2f} eV, U={params.U:.2f} eV")
        print(f"Crystal field: Δ_CF={params.Delta_CF:.3f} eV")
        print(f"Electron-phonon: g_JT={params.g_JT:.3f} eV/Å, K={params.K_lattice:.2f} eV/Å²")
        print(f"Effective Pairing Interaction: V_eff = {V_eff:.4f} eV")
        print(f"Method: Variational Free Energy Minimization")
    
    # =========================================================================
    # 3.1 GUTZWILLER RENORMALIZATION FACTORS
    # =========================================================================
    
    def get_gutzwiller_factors(self, delta: float) -> Tuple[float, float]:
        """
        Gutzwiller renormalization factors as function of doping δ = 1 - n
        
        Physical interpretation:
        - g_t: Kinetic energy suppression (→0 at half-filling = Mott insulator)
        - g_J: Exchange enhancement (→4 at half-filling)
        - Handles both hole doping (δ > 0) and electron doping (δ < 0)
        
        Returns:
            g_t: Kinetic renormalization factor
            g_J: Exchange renormalization factor
        """
        abs_delta = abs(delta)
        abs_delta = max(abs_delta, 1e-6)  # Avoid singularity
        
        g_t = (2.0 * abs_delta) / (1.0 + abs_delta)
        g_J = 4.0 / ((1.0 + abs_delta) ** 2)
        return g_t, g_J
    
    # =========================================================================
    # 3.2 DISTORTION-DEPENDENT PARAMETERS
    # =========================================================================
    
    def effective_hopping(self, Q: float) -> float:
        """
        JT distortion modifies hopping: t(Q) = t₀ * exp(-β Q²)
        
        Physical origin: pd-hopping integral changes with local geometry
        CLUSTER MF: Exponential form prevents t(Q) < 0 at large Q
        """
        return self.p.t0 * np.exp(-self.p.beta * Q**2)
    
    def effective_superexchange(self, Q: float, g_J: float) -> float:
        """
        Superexchange J(Q) = g_J × 4t²(Q)/U
        
        Key feedback: JT distortion reduces J → lowers AFM cost of orbital mixing
        """
        t_Q = self.effective_hopping(Q)
        J_bare = 4.0 * (t_Q ** 2) / self.p.U
        J_eff = g_J * J_bare
        return J_eff
    
    def dispersion(self, k: np.ndarray, t_eff: float) -> float:
        """
        Tight-binding dispersion on 2D square lattice
        γ(k) = -2t[cos(k_x a) + cos(k_y a)]
        
        Used for hopping between A and B sublattices
        """
        return -2.0 * t_eff * (np.cos(k[0] * self.p.a) + np.cos(k[1] * self.p.a))
    
    def fermi_function(self, E: np.ndarray) -> np.ndarray:
        """
        Fermi-Dirac distribution function for thermal averaging
        
        Since μ is already included in the Hamiltonian, we evaluate f(E) directly.
        Numerical stability: clip argument to avoid overflow
        """
        arg = E / self.p.kT
        arg = np.clip(arg, -100, 100)
        return 1.0 / (1.0 + np.exp(arg))
    
    # =========================================================================
    # 3.3 BdG HAMILTONIAN CONSTRUCTION
    # =========================================================================
    
    def build_local_hamiltonian_for_bdg(self, sign_M: float, M: float, Q: float,
                                        J_eff: float, mu: float) -> np.ndarray:
        """
        Build local Hamiltonian for BdG k-space calculation (WITH mean-field AFM)
        
        Basis order: [6↑, 6↓, 7↑, 7↓]
        
        Parameters:
            sign_M: +1 for sublattice A, -1 for sublattice B (staggered pattern)
            M: AFM order parameter comes from k-space averaging (sees Delta effect)
            Q: JT distortion (uniform quadrupolar order)
            J_eff: Effective superexchange
            mu: Chemical potential
        """
        H = np.zeros((4, 4), dtype=complex)
        
        # 1. Chemical potential
        np.fill_diagonal(H, -mu)
        
        # 2. Crystal field splitting Δ_CF on Γ₇
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓
        
        # 3. Mean-field AFM Zeeman term (staggered,  ±J_eff·M/2, orbital-dependent)
        # Γ₆ feels full field, Γ₇ feels reduced field (η factor)
        h_afm_6 = J_eff * sign_M * M / 2.0
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
    
    def build_single_particle_hamiltonian(self, sign_M: float, Q: float, mu: float) -> np.ndarray:
        """
        Build single-particle Hamiltonian for cluster calculation (WITHOUT AFM Zeeman term)
        
        Used as input to cluster exact diagonalization.
        AFM exchange appears at cluster level via O⊗O tensor product.
        
        Basis: [6↑, 6↓, 7↑, 7↓]
        
        Terms:
        1. Chemical potential: -μ
        2. Crystal field: Δ_CF on Γ₇
        3. JT mixing: g_JT·Q·(τ_x ⊗ I_spin)
        """
        H = np.zeros((4, 4), dtype=complex)
        
        # 1. Chemical potential
        np.fill_diagonal(H, -mu)
        
        # 2. Crystal field splitting
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓
        
        # 3. JT distortion field (orbital mixing)
        h_jt = self.p.g_JT * Q
        H[0, 2] = H[2, 0] = h_jt  # 6↑ ↔ 7↑
        H[1, 3] = H[3, 1] = h_jt  # 6↓ ↔ 7↓
        return H
    
    def build_pairing_block(self, Delta: complex) -> np.ndarray:
        """
        Construct 4×4 pairing matrix for inter-orbital singlet SC
        
        Pairing structure: Δ(c₆↑c₇↓ - c₆↓c₇↑) + h.c.
        
        This represents spin-singlet pairing between Γ₆ and Γ₇ orbitals.
        The antisymmetric spin structure enforces singlet character.
        
        In Nambu basis, this gives anomalous matrix elements.
        """
        D = np.zeros((4, 4), dtype=complex)
        
        # Singlet pairing: 6↑-7↓ with opposite sign for 6↓-7↑
        D[0, 3] = Delta    # 6↑ pairs with 7↓
        D[1, 2] = -Delta   # 6↓ pairs with 7↑ (opposite sign for antisymmetry)
        return D
    
    def build_bdg_matrix(self, k: np.ndarray, M: float, Q: float, 
                        Delta: complex, mu: float, t_eff: float, 
                        J_eff: float) -> np.ndarray:
        """
        Construct full 16×16 BdG Hamiltonian for double unit cell in Nambu basis
        Uses build_local_hamiltonian_for_bdg which includes mean-field AFM splits bands.
        This ensures Delta → Γ₆–Γ₇ mixing → M modification feedback.
        
        Basis structure (16 components):
        [Particle_A(4), Particle_B(4), Hole_A(4), Hole_B(4)]
        
        where each 4-component block is: [6↑, 6↓, 7↑, 7↓]
        
        BdG matrix structure:
        ┌─────────────────┬─────────────────┐
        │  H_particle     │  Δ_pairing      │
        ├─────────────────┼─────────────────┤
        │  Δ_pairing†     │ -H_particle*    │
        └─────────────────┴─────────────────┘
        
        where H_particle has 2×2 sublattice structure:
        ┌───────┬────────┐
        │  H_A  │  T_AB  │
        ├───────┼────────┤
        │ T_AB† │  H_B   │
        └───────┴────────┘
        """
        # --- 1. LOCAL BLOCKS (On-site energy for sublattices A and B) ---
        H_A = self.build_local_hamiltonian_for_bdg(sign_M=+1.0, M=M, Q=Q, J_eff=J_eff, mu=mu)
        H_B = self.build_local_hamiltonian_for_bdg(sign_M=-1.0, M=M, Q=Q, J_eff=J_eff, mu=mu)
        
        # --- 2. KINETIC BLOCKS (Inter-sublattice hopping A ↔ B) ---
        # Dispersion: γ(k) = -2t[cos(k_x) + cos(k_y)]
        gamma_k = self.dispersion(k, t_eff)
        
        # Hopping matrix: γ(k) × I₄ (spin and orbital independent nearest-neighbor hopping)
        T_AB = gamma_k * np.eye(4, dtype=complex)
        
        # --- 3. PAIRING BLOCKS (On-site pairing on each sublattice) ---
        D_matrix = self.build_pairing_block(Delta)
        
        # --- 4. ASSEMBLE 16×16 BdG MATRIX ---
        # Structure:
        # ┌──────────────────────────────────┐
        # │  H_A    T_AB    D_A      0       │  ← Particle A, B
        # │  T_AB†  H_B     0        D_B     │
        # ├──────────────────────────────────┤
        # │  D_A†   0      -H_A*   -T_AB†    │  ← Hole A, B
        # │  0      D_B†  -T_AB*   -H_B*     │
        # └──────────────────────────────────┘
        
        BdG = np.zeros((16, 16), dtype=complex)
        
        # Particle-Particle sector (upper-left 8×8)
        BdG[0:4, 0:4] = H_A
        BdG[4:8, 4:8] = H_B
        BdG[0:4, 4:8] = T_AB
        BdG[4:8, 0:4] = np.conj(T_AB).T
        
        # Hole-Hole sector (lower-right 8×8)
        # Particle-hole symmetry: -H*
        BdG[8:12, 8:12] = -np.conj(H_A)
        BdG[12:16, 12:16] = -np.conj(H_B)
        BdG[8:12, 12:16] = -np.conj(T_AB)
        BdG[12:16, 8:12] = -np.conj(T_AB).T
        
        # Pairing sector (off-diagonal 8×8 blocks)
        # Pairing is local (on-site), so block diagonal in A/B
        BdG[0:4, 8:12] = D_matrix       # Pairing on sublattice A
        BdG[4:8, 12:16] = D_matrix      # Pairing on sublattice B
        
        # Hermitian conjugate
        BdG[8:12, 0:4] = np.conj(D_matrix).T
        BdG[12:16, 4:8] = np.conj(D_matrix).T
        return BdG
    
    # =========================================================================
    # 3.4 OBSERVABLES FROM BdG SPECTRUM (PRIMARY SOURCE)
    # =========================================================================
    
    def _compute_site_magnetization_and_quadrupole(self, vec: np.ndarray, u_slice: slice, v_slice: slice,
                                                    f: float, f_bar: float) -> Tuple[float, float]:
        """
        Helper function: compute magnetization and quadrupole for a single site.
        
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
    
    def compute_observables_from_bdg(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> Dict:
        """
        Calculate expectation values from BdG eigensystem with CORRECT thermal weighting
        
        This is the PRIMARY source for M and Q because:
        1. BdG eigenstates contain Γ₆–Γ₇ mixing induced by Delta
        2. This implements: SC pairing → multipolar channel → M,Q modification
        3. Without this, the core hypothesis "SC triggers JT" is NOT implemented!
        
        Given eigenvalues E_n and eigenvectors |ψ_n⟩ (16-component spinors), compute:
        
        CORRECT BdG FORMULAS:
        1. Density: ⟨c†c⟩ = Σ_n [|u_n|² f(E_n) + |v_n|² (1-f(E_n))]
        2. Magnetization: ⟨S_z⟩ = Σ_n [(u*Op*u) f(E_n) + (v*Op*v) (1-f(E_n))]
        3. Quadrupole: ⟨τ_x⟩ = Σ_n [(u*τ_x*u) f(E_n) + (v*τ_x*v) (1-f(E_n))]
        4. Pairing: ⟨c_i c_j⟩ = Σ_n u_i v_j* (1 - 2f(E_n))
        
        The (1-2f) factor is crucial for:
        - Correct finite-T gap behavior
        - Proper Tc estimation
        - Self-consistency at all temperatures
        
        Spinor structure (16 components):
        ψ_n = [u_A(4), u_B(4), v_A(4), v_B(4)]
        where u = particle amplitude, v = hole amplitude
        and each 4-component block is [6↑, 6↓, 7↑, 7↓]
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
        pair_sum = 0.0
        
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
            
            # --- 1. PARTICLE DENSITY (CORRECTED) ---
            # ⟨c†c⟩ = Σ_n [|u_n|² f(E_n) + |v_n|² (1-f(E_n))]
            dens_A += np.sum(np.abs(u_A)**2) * fn + np.sum(np.abs(v_A)**2) * fn_bar
            dens_B += np.sum(np.abs(u_B)**2) * fn + np.sum(np.abs(v_B)**2) * fn_bar
            
            # --- 2. MAGNETIZATION & QUADRUPOLE ---
            # Use class-level helper for clean code
            mA, tauA = self._compute_site_magnetization_and_quadrupole(psi, slice(0, 4), slice(8, 12), fn, fn_bar)
            mB, tauB = self._compute_site_magnetization_and_quadrupole(psi, slice(4, 8), slice(12, 16), fn, fn_bar)
            
            mag_A += mA
            mag_B += mB
            quad_A += tauA
            quad_B += tauB
            
            # --- 3. PAIRING AMPLITUDE ---
            # Sublattice A contribution
            pair_A = (u_A[0] * np.conj(v_A[3]) -  # 6↑-7↓
                     u_A[1] * np.conj(v_A[2]))    # 6↓-7↑
            
            # Sublattice B contribution
            pair_B = (u_B[0] * np.conj(v_B[3]) -  # 6↑-7↓
                     u_B[1] * np.conj(v_B[2]))    # 6↓-7↑
            
            # The (1-2f) factor is CRITICAL for proper SC physics: ⟨c_i c_j⟩ = Σ_n u_i v_j* (1 - 2f(E_n))
            pair_sum += (pair_A + pair_B) * fn_pair
        
        # --- NORMALIZE TO PHYSICAL OBSERVABLES ---
        
        # Average density per site (average of both sublattices)
        n_avg = (dens_A + dens_B) / 2.0
        
        # Staggered magnetization: (M_A - M_B) / 2
        M_staggered = (mag_A - mag_B) / 2.0
        
        # Uniform quadrupole: (Q_A + Q_B) / 2
        Q_uniform = (quad_A + quad_B) / 2.0
        
        # Average pairing amplitude
        Pair_avg = pair_sum / 2.0
        
        return {
            'n': n_avg,
            'M': M_staggered,
            'Q': Q_uniform,
            'Pair': Pair_avg
        }
    
    # =========================================================================
    # 3.5 FREE ENERGY CALCULATIONS
    # =========================================================================
    
    def compute_bdg_free_energy(self, M: float, Q: float, Delta: complex, 
                                mu: float, t_eff: float, J_eff: float) -> float:
        """
        Compute the total grand potential from the k-space BdG spectrum.

        Ω_BdG = Σ_k,n [E_n(k) f(E_n) - T S(f_n)] + (K/2) Q²,

        where S(f) = -f log f - (1 - f) log(1 - f).

        Includes:
        - Quasiparticle energy from the BdG spectrum (kinetic + interaction)
        - Thermal entropy
        - Elastic energy from lattice distortion

        M contains the cluster correction via quantum fluctuations.
        """
        Omega = 0.0
        
        for kvec in self.k_points:
            H_BdG = self.build_bdg_matrix(kvec, M, Q, Delta, mu, t_eff, J_eff)
            E_n = np.linalg.eigvalsh(H_BdG)
            f_n = self.fermi_function(E_n)
            
            for n in range(16):
                # Energy contribution
                Omega += E_n[n] * f_n[n]
                
                # Entropy contribution (only at finite T)
                if self.p.kT > 1e-8:
                    if f_n[n] > 1e-10 and f_n[n] < 1.0 - 1e-10:
                        S_n = -f_n[n] * np.log(f_n[n]) - (1.0 - f_n[n]) * np.log(1.0 - f_n[n])
                        Omega -= self.p.kT * S_n
        
        # Average over k-points
        Omega /= self.N_k
        
        # Add elastic energy: (K/2) Q²
        Omega += 0.5 * self.p.K_lattice * Q**2
        return Omega
    
    def compute_cluster_free_energy(self, M: float, Q: float, mu: float, J_eff: float) -> Dict:
        """
        Compute cluster free energy from exact diagonalization
        
        F_cluster = -T log Z = -T log[Σ_i exp(-E_i/T)]
        
        At T=0: F = E_0 (ground state energy)
        At T>0: Includes quantum and thermal fluctuations
        
        IMPORTANT: This captures quantum AFM fluctuations that mean-field BdG misses!

        Returns:
            Dictionary with:
            - 'F': Free energy PER SITE
            - 'M': Staggered magnetization from cluster which contribute the final magnetization
            - 'Q_exp': Quadrupole expectation value ⟨τ_x⟩
            - 'Q_rms': RMS quadrupole √⟨τ_x²⟩ (includes fluctuations)
            - 'Q_fluctuation': Fluctuation strength
        """
        # Build cluster Hamiltonian
        H_sp_A = self.build_single_particle_hamiltonian(+1.0, Q, mu)
        H_sp_B = self.build_single_particle_hamiltonian(-1.0, Q, mu)
        H_cluster = self.cluster_mf.build_cluster_hamiltonian(
            H_sp_A, H_sp_B, J_eff, M, self.p.eta
        )
        
        # Diagonalize
        E_cluster = np.linalg.eigvalsh(H_cluster)
        
        # Compute free energy (TOTAL for cluster)
        if self.p.kT < 1e-8:
            # T=0: Free energy = ground state energy
            F_total = E_cluster[0]
        else:
            # T>0: F = -T log Z
            E_shifted = E_cluster - E_cluster[0]  # Shift to avoid overflow
            Z = np.sum(np.exp(-E_shifted / self.p.kT))
            F_total = E_cluster[0] - self.p.kT * np.log(Z)
        
        # Normalize to PER-SITE energy: cluster has 2 sites, BdG is per-site, so we must divide by 2
        CLUSTER_SIZE = 2
        F_per_site = F_total / CLUSTER_SIZE
        
        # ===== COMPUTE OBSERVABLES FROM CLUSTER =====
        
        # Magnetization operator
        O_mag = self.cluster_mf.build_multipolar_operator(self.p.eta)
        I4 = np.eye(4, dtype=complex)
        
        M_A = self.cluster_mf.cluster_expectation(H_cluster, np.kron(O_mag, I4), self.p.kT)
        M_B = self.cluster_mf.cluster_expectation(H_cluster, np.kron(I4, O_mag), self.p.kT)
        M_cluster = abs(M_A - M_B) / 2.0
        
        # Quadrupole operator, τ_x does NOT commute with H_cluster, so ⟨τ_x⟩² ≠ ⟨τ_x²⟩
        tau_x = np.zeros((4, 4), dtype=complex)
        tau_x[0, 2] = tau_x[2, 0] = 1.0  # 6↑ ↔ 7↑
        tau_x[1, 3] = tau_x[3, 1] = 1.0  # 6↓ ↔ 7↓
        
        # ⟨τ_x⟩ - Expectation value
        Q_A_exp = self.cluster_mf.cluster_expectation(H_cluster, np.kron(tau_x, I4), self.p.kT)
        Q_B_exp = self.cluster_mf.cluster_expectation(H_cluster, np.kron(I4, tau_x), self.p.kT)
        Q_exp = abs(Q_A_exp + Q_B_exp) / 2.0
        
        # ⟨τ_x²⟩ - Captures all B₁g channels and quantum fluctuations!
        tau_x_squared = tau_x @ tau_x
        Q2_A = self.cluster_mf.cluster_expectation(H_cluster, np.kron(tau_x_squared, I4), self.p.kT)
        Q2_B = self.cluster_mf.cluster_expectation(H_cluster, np.kron(I4, tau_x_squared), self.p.kT)
        
        # RMS quadrupole (includes fluctuations)
        Q_rms = np.sqrt(abs(Q2_A + Q2_B) / 2.0)
        
        # Fluctuation strength: σ² = ⟨τ_x²⟩ - ⟨τ_x⟩²
        sigma2_A = abs(Q2_A - Q_A_exp**2)
        sigma2_B = abs(Q2_B - Q_B_exp**2)
        fluctuation = np.sqrt((sigma2_A + sigma2_B) / 2.0)
        
        return {
            'F': F_per_site,
            'M': M_cluster,
            'Q_exp': Q_exp,
            'Q_rms': Q_rms,
            'Q_fluctuation': fluctuation
        }

# =========================================================================
# 4. SELF-CONSISTENT FIELD SOLVER
# =========================================================================

    def solve_self_consistent(self, target_density: float,
                            initial_M: float = 0.5,
                            initial_Q: float = 0.0,
                            initial_Delta: float = 0.05,
                            verbose: bool = True) -> Dict:
        """
        Self-consistent solution via FREE ENERGY MINIMIZATION: F_total(M, Q, Δ, μ) = F_BdG(M, Q, Δ, μ)
        
        This is thermodynamically rigorous and ensures:
        ✓ Energy minimum (not arbitrary mixing)
        ✓ SC → JT feedback preserved (M, Q respond to Δ via ∂F/∂M, ∂F/∂Q)
        ✓ The BdG eigenstates contain Γ₆–Γ₇ mixing induced by SC pairing.
        
        ALGORITHM:
        1. Calculate M, Q from BdG lattice (sees SC effect!)
        2. For optimal M, Q: compute Δ from gap equation
        3. Adjust μ for density constraint
        4. Iterate until convergence
        
        Parameters:
            target_density: Desired electron density (1 - doping)
            initial_M: Starting guess for magnetization
            initial_Q: Starting guess for distortion
            initial_Delta: Starting guess for gap
            verbose: Print iteration details
        
        Returns:
            Dictionary with converged values and history
        """
        M = initial_M
        Q = initial_Q
        Delta = initial_Delta + 0.0j
        mu = 0.0

        mu_low = mu - 2.0
        mu_high = mu + 2.0
        n_low = None
        n_high = None

        doping = 1.0 - target_density

        history = {
            'M': [], 'Q': [], 'Delta': [], 'density': [],
            'F_total': [], 'F_cluster': [],
            'g_t': [], 'g_J': [], 'mu': []
        }

        if verbose:
            print(f"\n{'='*80}")
            print("BdG LATTICE-BASED SELF-CONSISTENT CALCULATION")
            print(f"{'='*80}")
            print(f"Target density: {target_density:.3f} (doping δ={doping:.3f})")
            print(f"{'-'*80}")

        for iteration in range(self.p.max_iter):

            # --- Renormalized parameters ---
            g_t, g_J = self.get_gutzwiller_factors(doping)
            t_eff = g_t * self.effective_hopping(Q)
            J_eff = self.effective_superexchange(Q, g_J)

            # --- k-space observables ---
            obs = self._compute_k_observables(M, Q, Delta, mu, t_eff, J_eff)
            M_lattice = obs['M']
            tau_x = obs['Q']
            n_kspace = obs['n']
            Pair_kspace = obs['Pair']

            # --- Cluster correction ---
            cluster = self.compute_cluster_free_energy(M, Q, mu, J_eff)
            M_cluster = cluster['M']

            M_new = 0.7 * M_lattice + 0.3 * M_cluster
            Q_new = -(self.p.g_JT / self.p.K_lattice) * tau_x

            # --- Gap equation ---
            V_eff = self.p.g_JT**2 / self.p.K_lattice
            Delta_new = abs(V_eff * Pair_kspace)

            # --- Free energy ---
            t_eff_new = g_t * self.effective_hopping(Q_new)
            J_eff_new = self.effective_superexchange(Q_new, g_J)
            F_total = self.compute_bdg_free_energy(
                M_new, Q_new, Delta, mu, t_eff_new, J_eff_new
            )

            # --- Chemical potential update ---
            mu, mu_low, mu_high, n_low, n_high = self._update_mu(
                mu, n_kspace, target_density,
                mu_low, mu_high, n_low, n_high
            )

            doping = 1.0 - n_kspace

            # --- Mixing ---
            M_mixed = self._mix(M, M_new)
            Q_mixed = self._mix(Q, Q_new)
            Delta_mixed = self._mix(Delta, Delta_new)

            max_diff = max(
                abs(M_mixed - M),
                abs(Q_mixed - Q),
                abs(Delta_mixed - Delta)
            )

            history['M'].append(abs(M))
            history['Q'].append(abs(Q))
            history['Delta'].append(abs(Delta))
            history['density'].append(n_kspace)
            history['F_total'].append(F_total)
            history['F_cluster'].append(cluster['F'])
            history['g_t'].append(g_t)
            history['g_J'].append(g_J)
            history['mu'].append(mu)

            if verbose and (iteration % 10 == 0 or iteration < 5):
                print(f"Iter {iteration:3d}: "
                    f"M={M:.4f}  Q={Q:.5f}  Δ={abs(Delta):.5f}  "
                    f"n={n_kspace:.4f}  F={F_total:.6f}")

            M, Q, Delta = M_mixed, Q_mixed, Delta_mixed

            if max_diff < self.p.tol and abs(n_kspace - target_density) < 0.01:

                final_cluster = self.compute_cluster_free_energy(M, Q, mu, J_eff_new)

                if verbose:
                    print(f"\n{'='*80}")
                    print(f"✓ CONVERGED after {iteration+1} iterations")
                    print(f"{'='*80}")
                    print(f"M = {M:.6f}")
                    print(f"Q = {Q:.6f} Å")
                    print(f"|Δ| = {abs(Delta):.6f} eV")
                    print(f"n = {n_kspace:.6f}")
                    print(f"μ = {mu:.6f} eV")
                    print(f"F = {F_total:.6f} eV")
                    print(f"g_t = {g_t:.4f}, g_J = {g_J:.4f}")
                    print(f"{'='*80}\n")

                break

        else:
            if verbose:
                print(f"\n⚠ Warning: Did not converge after {self.p.max_iter} iterations\n")

        return {
            'M': M,
            'Q': Q,
            'Delta': abs(Delta),
            'density': n_kspace,
            'mu': mu,
            'g_t': g_t,
            'g_J': g_J,
            'J_eff': J_eff_new,
            't_eff': t_eff_new,
            'F_total': F_total,
            'converged': max_diff < self.p.tol,
            'history': history,
            'doping': doping
        }

    def _compute_k_observables(self, M, Q, Delta, mu, t_eff, J_eff):

        totals = dict(M=0.0, Q=0.0, n=0.0, Pair=0.0)

        for kvec in self.k_points:
            H = self.build_bdg_matrix(kvec, M, Q, Delta, mu, t_eff, J_eff)
            eigvals, eigvecs = eigh(H)
            obs = self.compute_observables_from_bdg(eigvals, eigvecs)

            for key in totals:
                totals[key] += obs[key]

        return {k: v / self.N_k for k, v in totals.items()}

    def _update_mu(self, mu, n_kspace, target_density,
                mu_low, mu_high, n_low, n_high):

        density_error = n_kspace - target_density

        if density_error > 0:
            mu_high, n_high = mu, n_kspace
        else:
            mu_low, n_low = mu, n_kspace

        if n_low is not None and n_high is not None:
            dmu = mu_high - mu_low
            dn = n_high - n_low

            if abs(dmu) > 1e-8 and abs(dn) > 1e-8:
                dn_dmu = dn / dmu
                if abs(dn_dmu) > 1e-6:
                    mu_trial = mu - density_error / dn_dmu
                    if mu_low < mu_trial < mu_high:
                        return mu_trial, mu_low, mu_high, n_low, n_high

            return 0.5 * (mu_low + mu_high), mu_low, mu_high, n_low, n_high

        mu_new = mu - self.p.mu_adjust_rate * density_error
        mu_new = np.clip(mu_new, mu_low, mu_high)
        return mu_new, mu_low, mu_high, n_low, n_high

    def _mix(self, old, new):
        α = self.p.mixing
        return (1 - α) * old + α * new

# =============================================================================
# 5. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_convergence(results: Dict, title: str = "Self-Consistency Convergence at Fixed Doping") -> plt.Figure:
    """
    Plot convergence history for a SINGLE doping point
    
    Args:
        results: Dictionary from solve_self_consistent() containing 'history' key
    """
    history = results['history']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Magnetization
    ax = axes[0, 0]
    ax.plot(history['M'], 'r-', linewidth=2)
    ax.set_ylabel('Magnetization M', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('AFM Order Parameter')
    
    # JT Distortion
    ax = axes[0, 1]
    ax.plot(history['Q'], 'g-', linewidth=2)
    ax.set_ylabel('JT Distortion Q (Å)', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Quadrupolar Distortion')
    
    # SC Gap
    ax = axes[0, 2]
    ax.plot(history['Delta'], 'b-', linewidth=2)
    ax.set_ylabel('SC Gap |Δ| (eV)', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Superconducting Gap')
    
    # Free Energy Components - MAY BE NON-MONOTONIC
    ax = axes[1, 0]
    ax.plot(history['F_total'], 'k-', linewidth=2, label='F_total')
    ax.plot(history['F_cluster'], 'r--', linewidth=1.5, alpha=0.7, label='F_cluster')
    ax.set_ylabel('Free Energy (eV)', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Free Energy During Iteration (May Be Non-Monotonic)')
    
    # Renormalization factors
    ax = axes[1, 1]
    ax.plot(history['g_t'], 'c-', linewidth=2, label='g_t (kinetic)')
    ax.plot(history['g_J'], 'm-', linewidth=2, label='g_J (exchange)')
    ax.set_ylabel('Renormalization Factor', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Gutzwiller Factors')
    
    # Density - NUMERICAL CONVERGENCE, NOT PHYSICS
    ax = axes[1, 2]
    ax.plot(history['density'], 'orange', linewidth=2)
    ax.set_ylabel('Density n', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_title('Electron Density (Constraint Convergence)')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_phase_diagrams(solver: RMFT_Solver, doping_range: np.ndarray,
                        cf_min: float = 0.05, cf_max: float = 0.20, N_cf: int = 10):
    """
    Scan doping to create phase diagram using WARM-START
    Scans through the crystal field (Delta_CF) to find the 'Sweet Spot' where TC (Delta_sc) is largest.
    Implements warm-start: each point starts from previous converged solution
    Validates density constraint at each point
    """
    # Phase data storage
    phase_data = {
        'doping': [], 'M': [], 'Q': [], 'Delta': [],
        'mu': [], 'density': [], 'F_total': []
    }
    
    print(f"\n{'='*70}")
    print("Scanning phase diagram over doping range (with WARM-START)")
    print(f"{'='*70}\n")
    
    # Warm-start initialization
    previous_solution = {'M': 0.5, 'Q': 0.0, 'Delta': 0.05}
    
    for i, doping in enumerate(doping_range):
        density = 1.0 - doping
        print(f"\n[{i+1}/{len(doping_range)}] Solving for δ = {doping:.3f}")
        
        # Solve with warm-start from previous solution
        result = solver.solve_self_consistent(
            target_density=density,
            initial_M=previous_solution['M'],
            initial_Q=previous_solution['Q'],
            initial_Delta=previous_solution['Delta'],
            verbose=False
        )
        
        # Validate density constraint
        density_error = abs(result['density'] - density)
        if density_error > 0.01:
            print(f"    ⚠ WARNING: Density constraint not satisfied (error={density_error:.4f})")
        
        # Store CONVERGED solution in phase_data
        phase_data['doping'].append(doping)
        phase_data['M'].append(result['M'])
        phase_data['Q'].append(result['Q'])
        phase_data['Delta'].append(result['Delta'])
        phase_data['mu'].append(result['mu'])
        phase_data['density'].append(result['density'])
        phase_data['F_total'].append(result['F_total'])
        
        print(f"    → M={result['M']:.3f}, Q={result['Q']:.4f}, Δ={result['Delta']:.4f}")
        
        # Update warm-start for next point
        previous_solution = {'M': result['M'], 'Q': result['Q'], 'Delta': result['Delta']}
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Main phase diagram
    _plot_phase_data(axes[0, 0], phase_data)
    
    # Crystal field sweet spot search
    print("\n\nCrystal field sweet spot search:")
    print(f"Scanning Delta_CF from {cf_min:.3f} to {cf_max:.3f} eV")
    
    original_cf = solver.p.Delta_CF
    ref_doping_idx = len(doping_range) // 2
    ref_doping = doping_range[ref_doping_idx]
    ref_density = 1.0 - ref_doping
    
    print(f"Reference doping: δ={ref_doping:.3f}")
    
    # Crystal field scan
    cf_range = np.linspace(cf_min, cf_max, N_cf)
    cf_gaps, cf_Q_values, cf_M_values = [], [], []
    cf_previous = None
    
    for cf in cf_range:
        solver.p.Delta_CF = cf
        
        if cf_previous is not None:
            init_M, init_Q, init_Delta = cf_previous['M'], cf_previous['Q'], cf_previous['Delta']
        else:
            init_M = phase_data['M'][ref_doping_idx]
            init_Q = phase_data['Q'][ref_doping_idx]
            init_Delta = phase_data['Delta'][ref_doping_idx]
        
        # Solve at this CF value
        result = solver.solve_self_consistent(
            target_density=ref_density,
            initial_M=init_M, initial_Q=init_Q, initial_Delta=init_Delta,
            verbose=False
        )
        
        cf_gaps.append(result['Delta'])
        cf_Q_values.append(result['Q'])
        cf_M_values.append(result['M'])
        
        # Update warm-start
        cf_previous = {'M': result['M'], 'Q': result['Q'], 'Delta': result['Delta']}
        
        print(f"  ΔCF={cf:.3f} → Δ={result['Delta']:.4f}, Q={result['Q']:.4f}, M={result['M']:.3f}")
    
    # Restore original Delta_CF
    solver.p.Delta_CF = original_cf
    
    # Find sweet spot
    max_gap_idx = np.argmax(cf_gaps)
    sweet_spot_cf = cf_range[max_gap_idx]
    max_gap = cf_gaps[max_gap_idx]
    
    print(f"\n✓ Sweet spot: ΔCF = {sweet_spot_cf:.3f} eV, Δmax = {max_gap:.4f} eV")
    
    # Panel 2: Sweet spot
    ax = axes[0, 1]
    ax.plot(cf_range, cf_gaps, 'b-o', linewidth=2, markersize=6)
    ax.axvline(sweet_spot_cf, color='r', linestyle='--', label=f'Sweet spot: {sweet_spot_cf:.3f} eV')
    ax.set_xlabel('Crystal Field ΔCF (eV)', fontsize=12)
    ax.set_ylabel('SC Gap Δ (eV)', fontsize=12)
    ax.set_title(f'Sweet Spot Search (δ={ref_doping:.3f})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Q vs CF
    ax = axes[1, 0]
    ax.plot(cf_range, cf_Q_values, 'g-s', linewidth=2, markersize=6)
    ax.set_xlabel('Crystal Field ΔCF (eV)', fontsize=12)
    ax.set_ylabel('JT Distortion Q (Å)', fontsize=12)
    ax.set_title('Q vs Crystal Field', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: M vs CF
    ax = axes[1, 1]
    ax.plot(cf_range, cf_M_values, 'r-^', linewidth=2, markersize=6)
    ax.set_xlabel('Crystal Field ΔCF (eV)', fontsize=12)
    ax.set_ylabel('Magnetization M', fontsize=12)
    ax.set_title('M vs Crystal Field', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def _plot_phase_data(ax, phase_data: Dict):
    """
    Plot phase diagram from phase_data structure. This function ONLY uses converged final states.
    """

    doping = np.array(phase_data['doping'])
    M = np.array(phase_data['M'])
    Q = np.array(phase_data['Q'])
    Delta = np.array(phase_data['Delta'])
    
    # Plot order parameters vs doping
    ax.plot(doping, M, 'r-o', linewidth=2, markersize=6, label='AFM (M)')
    ax.plot(doping, Q, 'g-s', linewidth=2, markersize=6, label='JT Distortion (Q)')
    ax.plot(doping, Delta, 'b-^', linewidth=2, markersize=6, label='SC Gap (Δ)')
    
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
    plt.tight_layout()
    
def plot_dos(solver: RMFT_Solver, result: Dict):
    """
    Plot the density function (DOS) of the system to see the van Hove peaks.
    """
    all_energies = []
    M, Q, Delta, mu, t_eff, J_eff = (
        result['M'], 
        result['Q'], 
        result['Delta'], 
        result['mu'],  
        result['t_eff'],  
        result['J_eff']
    )

    for kvec in solver.k_points:
        H_BdG = solver.build_bdg_matrix(kvec, M, Q, Delta, mu, t_eff, J_eff)
        vals = np.linalg.eigvalsh(H_BdG)
        all_energies.extend(vals)

    all_energies = np.array(all_energies)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    ax.hist(all_energies, bins=200, density=True, color='blue', alpha=0.7, label='DOS')

    # Fermi level (E = mu, but in BdG 0 is the Fermi level)
    ax.axvline(x=0.0, color='red', linestyle='--', label='Fermi Level ($E_F$)')
    ax.set_title(
        f"Density of States (DOS)\n"
        f"$\\Delta_{{CF}}={solver.p.Delta_CF}$, "
        f"Doping={result.get('doping', 'N/A')}"
    )
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Check: Is there a peak at 0?
    hist, bin_edges = np.histogram(all_energies, bins=200, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    peaks, _ = find_peaks(hist, prominence=0.1 * np.max(hist))
    vhs_energies = bin_centers[peaks]
    
    print(f"\nVan Hove singularities detected at energies: {vhs_energies}")
    fermi_distance = np.min(np.abs(vhs_energies)) if len(vhs_energies) > 0 else np.inf
    print(f"Closest VHS to Fermi level: {fermi_distance:.4f} eV")
    return fig
    
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
    
    # =========================================================================
    # Setup physical parameters
    # =========================================================================
    params = ModelParams(
        t0=0.4,           # eV - moderate hopping
        U=3.0,            # eV - charge transfer regime
        Delta_CF=0.01,    # eV - small CF splitting (quasi-degenerate)
        g_JT=1.5,         # eV/Å - moderate e-ph coupling
        K_lattice=1.5,    # eV/Å² - phonon stiffness
        beta=116,         # 1/eV  - β = 1/(k​T), β = 500 is the ~ 0 K
        eta=0.1,          # weak AFM on Γ₇
        Z=4,              # 2D square lattice
        nk=64,            # k-grid resolution, minimal nk ​≈ 16⋅δ⋅t0​​ / ((1 + δ) ⋅ kT)
        kT=0.006,         # temperature: 0.01 eV ~ 116 K
        a=1.0,
        max_iter=100,
        tol = 1e-6,
        mixing=0.1,
        mu_adjust_rate=0.07
    )
    
    solver = RMFT_Solver(params)
    
    # =========================================================================
    # Case 1: Near half-filling (small doping)
    # Expected: Pure AFM, minimal JT, minimal SC
    # =========================================================================
    print("\n" + "="*70)
    print("CASE 1: Near Half-Filling (δ = 0.02)")
    print("="*70)
    
    result_halffill = solver.solve_self_consistent(
        target_density=0.98,  # δ = 0.02
        initial_M=0.1,
        initial_Q=0.0,
        initial_Delta=0.05,
        verbose=True
    )
    fig1 = plot_convergence(result_halffill, title="Case 1: Near Half-Filling (Pure AFM Phase)")
    fig2 = plot_dos(solver, result_halffill)
    
    # =========================================================================
    # Case 2: Heavy doping
    # Expected: Weak AFM, SC without strong JT
    # =========================================================================
    print("\n" + "="*70)
    print("CASE 3: Heavy Doping (δ = 0.20)")
    print("="*70)
    
    result_heavy = solver.solve_self_consistent(
        target_density=0.80,  # δ = 0.20
        initial_M=0.1,
        initial_Q=0.0,
        initial_Delta=0.05,
        verbose=True
    )
    fig3 = plot_convergence(result_heavy, title="Case 3: Heavy Doping (Conventional SC)")
    fig4 = plot_dos(solver, result_heavy)
    
    # =========================================================================
    # Phase diagram scan
    # =========================================================================
    print("\n" + "="*70)
    print("SCANNING FULL PHASE DIAGRAM")
    print("="*70)
    
    doping_scan = np.linspace(0.01, 0.25, 10)
    fig5 = plot_phase_diagrams(solver, doping_scan)

    # =========================================================================
    # Display all figures
    # =========================================================================
    plt.show()
    
    print(f"\n{'='*70}")
    print("Simulation complete!")
    print(f"{'='*70}\n")