"""
Unified SC-Activated Jahn-Teller Model - 16x16 Double Unit Cell Implementation
Based on the rigorous D₄h symmetry specification with Gutzwiller renormalization

Key improvements over original code:
1. Proper symmetry-aware operator construction
2. Correct BdG matrix structure with particle-hole symmetry and proper double unit cell structure (mu inside H, f(E) for thermal average)
3. Emergent Pairing: Delta = V_eff * <c_up c_down> (no free parameter).
4. Distortion-dependent superexchange J(Q)
5. Chemical potential adjustment for fixed density
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import Tuple, Dict

# =============================================================================
# 1. PHYSICAL PARAMETERS & MODEL DEFINITION
# =============================================================================

@dataclass
class ModelParams:
    """Physical parameters in eV units"""
    # Microscopic parameters
    t0: float = 0.5           # Bare hopping integral (eV)
    U: float = 3.0            # Hubbard U (eV) - charge transfer regime
    Delta_CF: float = 0.1     # Crystal field splitting Γ₆-Γ₇ (eV)
    g_JT: float = 0.45        # Electron-phonon coupling (eV/Å)
    K_lattice: float = 4.0    # Lattice spring constant (eV/Å²)
    beta: float = 0.5         # Hopping-distortion coupling: t(Q) = t₀(1-βQ²)
    eta: float = 0.1          # AFM asymmetry (Γ₇ feels η×M compared to Γ₆)
    
    # Simulation parameters
    nk: int = 32              # k-points per direction (32×32 grid)
    kT: float = 0.005          # Temperature (0.01 eV ~ 116 K)
    a: float = 1.0            # Lattice constant (Å)
    
    # Convergence
    max_iter: int = 150
    tol: float = 1e-6
    mixing: float = 0.2       # Slower mixing for stability in high-dim space
    mu_adjust_rate: float = 0.07  # Chemical potential adjustment speed


# =============================================================================
# 2. RENORMALIZED MEAN-FIELD THEORY (16x16 BdG) SOLVER
# =============================================================================

class RMFT_Solver:
    """
    Self-consistent solver for the SC-activated JT model
    with Gutzwiller renormalization and proper 16x16 double unit cell structure
    """
    def __init__(self, params: ModelParams):
        self.p = params
        
        # Generate k-space grid (2D Brillouin zone)
        # We use the full BZ. The folding happens naturally due to the 2-site unit cell structure.
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
    
    # =========================================================================
    # 2.1 GUTZWILLER RENORMALIZATION FACTORS
    # =========================================================================
    
    def get_gutzwiller_factors(self, delta: float) -> Tuple[float, float]:
        """
        Gutzwiller renormalization factors as function of doping δ = 1 - n
        
        Physical interpretation:
        - g_t: Kinetic energy suppression (→0 at half-filling = Mott insulator)
        - g_J: Exchange enhancement (→4 at half-filling)
        
        Returns:
            g_t: Kinetic renormalization factor
            g_J: Exchange renormalization factor
        """
        delta = max(delta, 1e-6)  # Avoid singularity
        
        g_t = (2.0 * delta) / (1.0 + delta)
        g_J = 4.0 / ((1.0 + delta) ** 2)
        return g_t, g_J
    
    # =========================================================================
    # 2.2 DISTORTION-DEPENDENT PARAMETERS
    # =========================================================================
    
    def effective_hopping(self, Q: float) -> float:
        """
        JT distortion modifies hopping: t(Q) = t₀(1 - βQ²)
        
        Physical origin: pd-hopping integral changes with local geometry
        """
        return self.p.t0 * (1.0 - self.p.beta * Q**2)
    
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
    # 2.3 BdG HAMILTONIAN CONSTRUCTION (16×16 MATRIX)
    # =========================================================================
    
    def build_local_hamiltonian(self, sign_M: float, M: float, Q: float, 
                               J_eff: float, mu: float) -> np.ndarray:
        """
        Construct 4×4 local (on-site) Hamiltonian for a single sublattice
        
        Basis order: [6↑, 6↓, 7↑, 7↓]
        
        Parameters:
            sign_M: +1 for sublattice A, -1 for sublattice B (staggered pattern)
            M: AFM order parameter (staggered magnetization)
            Q: JT distortion (uniform quadrupolar order)
            J_eff: Effective superexchange
            mu: Chemical potential
        
        Terms:
        1. Chemical potential: -μ × I₄
        2. Crystal field: Δ_CF on Γ₇ states
        3. AFM Zeeman field: ±J_eff×M/2 (staggered, spin-dependent)
        4. JT mixing: g_JT×Q×τ_x (orbital mixing, spin-conserving)
        """
        H = np.zeros((4, 4), dtype=complex)
        
        # 1. Chemical potential (diagonal, all states)
        np.fill_diagonal(H, -mu)
        
        # 2. Crystal field splitting (Γ₇ higher by Δ_CF)
        H[2, 2] += self.p.Delta_CF  # 7↑
        H[3, 3] += self.p.Delta_CF  # 7↓
        
        # 3. AFM Zeeman-like field (intra-orbital, spin-dependent, staggered)
        # Γ₆ feels full field: ±J_eff×M/2 for ↑/↓
        # Γ₇ feels reduced field: ±η×J_eff×M/2
        h_afm_6 = J_eff * sign_M * M / 2.0
        h_afm_7 = self.p.eta * h_afm_6
        
        H[0, 0] -= h_afm_6  # 6↑
        H[1, 1] += h_afm_6  # 6↓
        H[2, 2] -= h_afm_7  # 7↑
        H[3, 3] += h_afm_7  # 7↓
        
        # 4. JT distortion field (inter-orbital mixing, spin-conserving)
        # τ_x connects Γ₆ ↔ Γ₇ with same spin
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
        Construct full 16×16 BdG Hamiltonian for double unit cell
        
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
        H_A = self.build_local_hamiltonian(sign_M=+1.0, M=M, Q=Q, J_eff=J_eff, mu=mu)
        H_B = self.build_local_hamiltonian(sign_M=-1.0, M=M, Q=Q, J_eff=J_eff, mu=mu)
        
        # --- 2. KINETIC BLOCKS (Inter-sublattice hopping A ↔ B) ---
        # Dispersion: γ(k) = -2t[cos(k_x) + cos(k_y)]
        gamma_k = self.dispersion(k, t_eff)
        
        # Hopping matrix (identity in orbital/spin space)
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
    # 2.4 OBSERVABLES CALCULATION
    # =========================================================================
    
    def compute_observables(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> Dict:
        """
        Calculate expectation values from BdG eigensystem with CORRECT thermal weighting
        
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
            u2_A = np.sum(np.abs(u_A)**2)
            v2_A = np.sum(np.abs(v_A)**2)
            u2_B = np.sum(np.abs(u_B)**2)
            v2_B = np.sum(np.abs(v_B)**2)
            
            dens_A += u2_A * fn + v2_A * fn_bar
            dens_B += u2_B * fn + v2_B * fn_bar
            
            # --- 2. MAGNETIZATION (CORRECTED) ---
            # ⟨S_z⟩ weighted by orbital character
            # Operator in [6↑, 6↓, 7↑, 7↓] basis: diag[+1, -1, +η, -η]
            sz_operator = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
            
            # Particle contribution
            mag_u_A = np.sum(np.abs(u_A)**2 * sz_operator)
            mag_u_B = np.sum(np.abs(u_B)**2 * sz_operator)
            
            # Hole contribution (same operator, different thermal weight)
            mag_v_A = np.sum(np.abs(v_A)**2 * sz_operator)
            mag_v_B = np.sum(np.abs(v_B)**2 * sz_operator)
            
            mag_A += mag_u_A * fn + mag_v_A * fn_bar
            mag_B += mag_u_B * fn + mag_v_B * fn_bar
            
            # --- 3. QUADRUPOLE MOMENT (CORRECTED) ---
            # ⟨τ_x⟩ = ⟨|6⟩⟨7| + |7⟩⟨6|⟩ (orbital mixing)
            def get_quadrupole(amp):
                """Calculate ⟨τ_x⟩ from amplitudes (u or v)"""
                # indices: 0=6↑, 1=6↓, 2=7↑, 3=7↓
                return 2.0 * np.real(np.conj(amp[0])*amp[2] + np.conj(amp[1])*amp[3])
            
            quad_u_A = get_quadrupole(u_A)
            quad_v_A = get_quadrupole(v_A)
            quad_u_B = get_quadrupole(u_B)
            quad_v_B = get_quadrupole(v_B)
            
            quad_A += quad_u_A * fn + quad_v_A * fn_bar
            quad_B += quad_u_B * fn + quad_v_B * fn_bar
            
            # --- 4. PAIRING AMPLITUDE (CORRECTED) ---
            # ⟨c_i c_j⟩ = Σ_n u_i v_j* (1 - 2f(E_n))
            # The (1-2f) factor is CRITICAL for proper SC physics
            
            # Sublattice A contribution
            pair_A = (u_A[0] * np.conj(v_A[3]) -  # 6↑-7↓
                     u_A[1] * np.conj(v_A[2]))    # 6↓-7↑
            
            # Sublattice B contribution
            pair_B = (u_B[0] * np.conj(v_B[3]) -  # 6↑-7↓
                     u_B[1] * np.conj(v_B[2]))    # 6↓-7↑
            
            pair_sum += (pair_A + pair_B) * fn_pair
        
        # --- NORMALIZE TO PHYSICAL OBSERVABLES ---
        
        # Average density per site (average of both sublattices)
        n_avg = (dens_A + dens_B) / 2.0
        
        # Staggered magnetization: (M_A - M_B) / 2
        # This is the AFM order parameter
        M_staggered = (mag_A - mag_B) / 2.0
        
        # Uniform quadrupole: (Q_A + Q_B) / 2
        # JT distortion is ferro-quadrupolar (uniform)
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
    # 3. SELF-CONSISTENT FIELD SOLVER
    # =========================================================================
    
    def solve_self_consistent(self, target_density: float, 
                             initial_M: float = 0.5,
                             initial_Q: float = 0.0,
                             initial_Delta: float = 0.05,
                             verbose: bool = True) -> Dict:
        """
        Self-consistent solution of coupled order parameters
        
        The system has three coupled order parameters:
        1. M (staggered magnetization) - AFM order
        2. Q (quadrupolar distortion) - JT order
        3. Δ (pairing gap) - SC order
        
        Self-consistency equations:
        1. M_new = ⟨M_operator⟩
        2. Q_new = -(g_JT/K) × ⟨τ_x⟩  [from minimizing elastic energy]
        3. Δ_new = V_eff × ⟨pairing⟩  [V_eff = g_JT²/K]
        4. μ adjusted to maintain target density
        
        Parameters:
            target_density: Desired electron density (1 - doping)
            initial_M: Starting guess for magnetization
            initial_Q: Starting guess for distortion
            initial_Delta: Starting guess for gap
            verbose: Print iteration details
        
        Returns:
            Dictionary with converged values and history
        """
        # Initialize order parameters
        M = initial_M
        Q = initial_Q
        Delta = initial_Delta
        mu = 0.0  # Chemical potential (to be adjusted)
        
        # Convergence history
        history = {
            'M': [],
            'Q': [],
            'Delta': [],
            'mu': [],
            'n': [],
            'g_t': [],
            'g_J': []
        }
        
        if verbose:
            print(f"\nStarting self-consistent loop")
            print(f"Target density: n = {target_density:.4f} (doping δ = {1-target_density:.4f})")
            print(f"{'─'*70}")
        
        # Track convergence status
        converged = False
        total_change = float('inf')  # Initialize for first iterations
        
        # Main iteration loop
        for iteration in range(self.p.max_iter):
            # ================================================================
            # STEP 1: UPDATE DERIVED PARAMETERS
            # ================================================================
            
            # Current doping for Gutzwiller factors
            current_doping = abs(1.0 - target_density)
            g_t, g_J = self.get_gutzwiller_factors(current_doping)
            
            # Renormalized hopping and exchange
            t_Q = self.effective_hopping(Q)
            t_eff = g_t * t_Q
            J_eff = self.effective_superexchange(Q, g_J)
            
            # ================================================================
            # STEP 2: K-SPACE LOOP - diagonalize BdG at each k-point
            # ================================================================
            
            # Accumulators for k-space averages
            total_n = 0.0
            total_M = 0.0
            total_Q = 0.0
            total_Pair = 0.0
            
            # Loop over all k-points
            for i in range(self.N_k):
                kvec = self.k_points[i]
                
                # Build and diagonalize BdG Hamiltonian at this k-point
                H_BdG = self.build_bdg_matrix(kvec, M, Q, Delta, mu, t_eff, J_eff)
                eigenvalues, eigenvectors = eigh(H_BdG)
                
                # Compute observables at this k-point
                obs = self.compute_observables(eigenvalues, eigenvectors)
                
                # Accumulate
                total_n += obs['n']
                total_M += obs['M']
                total_Q += obs['Q']
                total_Pair += obs['Pair']
            
            # Average over k-space
            avg_n = total_n / self.N_k
            avg_M = total_M / self.N_k
            avg_Q = total_Q / self.N_k
            avg_Pair = total_Pair / self.N_k
            
            # ================================================================
            # STEP 3: Update order parameters via self-consistency equations
            # ================================================================
            
            # Update Gutzwiller factors based on ACTUAL density, not target, this ensures true self-consistency of renormalization
            current_doping = abs(1.0 - avg_n)  # Use measured density, not target
            g_t_updated, g_J_updated = self.get_gutzwiller_factors(current_doping)
            
            # A) Magnetization: direct from expectation value
            M_new = avg_M
            
            # B) Quadrupole: from elastic free energy minimization
            #    F = (K/2)Q² + g_JT×Q×⟨τ_x⟩
            #    ∂F/∂Q = 0 → Q = -(g_JT/K)×⟨τ_x⟩
            Q_new = -(self.p.g_JT / self.p.K_lattice) * avg_Q
            
            # C) Pairing gap: emergent from phonon-mediated attraction
            #    V_eff = g_JT²/K (effective attractive interaction)
            #    Δ = V_eff × ⟨pairing operator⟩
            V_eff = (self.p.g_JT**2) / self.p.K_lattice
            Delta_new = V_eff * avg_Pair
            
            # D) Chemical potential: adjust to maintain target density
            #    Simple feedback: μ ← μ + α(n_target - n_current)
            mu += self.p.mu_adjust_rate * (target_density - avg_n)
            
            # ================================================================
            # STEP 4: MIX OLD AND NEW VALUES FOR STABILITY
            # ================================================================
            alpha = self.p.mixing
            
            M = (1 - alpha) * M + alpha * M_new
            Q = (1 - alpha) * Q + alpha * Q_new
            Delta = (1 - alpha) * Delta + alpha * Delta_new
            
            # ================================================================
            # STEP 5: RECORD HISTORY AND PRINT PROGRESS
            # ================================================================
            history['M'].append(M)
            history['Q'].append(Q)
            history['Delta'].append(np.abs(Delta))
            history['mu'].append(mu)
            history['n'].append(avg_n)
            history['g_t'].append(g_t_updated)  # Store updated values
            history['g_J'].append(g_J_updated)
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d}: "
                      f"M={M:7.4f}  Q={Q:7.4f}  |Δ|={abs(Delta):7.4f}  "
                      f"n={avg_n:6.4f}  μ={mu:7.4f}  "
                      f"g_t={g_t_updated:5.3f}  g_J={g_J_updated:5.3f}")
            
            # ================================================================
            # STEP 6: CHECK CONVERGENCE
            # ================================================================
            if iteration > 10:
                # Compute change from previous iteration
                dM = abs(M - history['M'][-2])
                dQ = abs(Q - history['Q'][-2])
                dDelta = abs(abs(Delta) - history['Delta'][-2])
                
                total_change = dM + dQ + dDelta
                
                if total_change < self.p.tol:
                    converged = True
                    if verbose:
                        print(f"{'─'*70}")
                        print(f"✓ Converged after {iteration} iterations")
                        print(f"  Total change: {total_change:.2e} < {self.p.tol:.2e}")
                    break
            
            # Track if we're on the last iteration
            if iteration == self.p.max_iter - 1:
                converged = False
                if verbose:
                    print(f"{'─'*70}")
                    print(f"⚠ Warning: Maximum iterations ({self.p.max_iter}) reached without convergence")
                    print(f"  Total change: {total_change:.2e} >= {self.p.tol:.2e}")
        
        # --- FINAL RESULTS ---
        if verbose:
            print(f"{'─'*70}")
            print(f"Final converged values:")
            print(f"  Magnetization:  M = {M:.6f}")
            print(f"  JT Distortion:  Q = {Q:.6f} Å")
            print(f"  SC Gap:        |Δ| = {abs(Delta):.6f} eV")
            print(f"  Density:        n = {avg_n:.6f}")
            print(f"  Chem. Potential: μ = {mu:.6f} eV")
            print(f"  Renormalization: g_t = {g_t_updated:.4f}, g_J = {g_J_updated:.4f}")
            print(f"{'─'*70}\n")
        
        return {
            'M': M,
            'Q': Q,
            'Delta': abs(Delta),
            'mu': mu,
            'n': avg_n,
            'g_t': g_t_updated,
            'g_J': g_J_updated,
            'history': history,
            'converged': converged
        }


# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================

def plot_convergence(results: Dict, title: str = "Convergence History") -> plt.Figure:
    """Plot convergence history of order parameters"""
    history = results['history']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Magnetization
    axes[0, 0].plot(history['M'], 'r-', linewidth=2)
    axes[0, 0].set_ylabel('Magnetization M', fontsize=12)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('AFM Order Parameter')
    
    # JT Distortion
    axes[0, 1].plot(history['Q'], 'g-', linewidth=2)
    axes[0, 1].set_ylabel('JT Distortion Q (Å)', fontsize=12)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Quadrupolar Distortion')
    
    # SC Gap
    axes[1, 0].plot(history['Delta'], 'b-', linewidth=2)
    axes[1, 0].set_ylabel('SC Gap |Δ| (eV)', fontsize=12)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Superconducting Gap')
    
    # Renormalization factors
    axes[1, 1].plot(history['g_t'], 'c-', linewidth=2, label='g_t (kinetic)')
    axes[1, 1].plot(history['g_J'], 'm-', linewidth=2, label='g_J (exchange)')
    axes[1, 1].set_ylabel('Renormalization Factor', fontsize=12)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Gutzwiller Factors')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_phase_diagram(solver: RMFT_Solver, doping_range: np.ndarray):
    """
    Scan doping to create phase diagram
    
    Expected behavior:
    - δ ≈ 0 (half-filling): M large, Q ≈ 0, Δ ≈ 0 (pure AFM)
    - δ ≈ 0.05-0.15: M reduced, Q ≠ 0, Δ ≠ 0 (SC+JT coexistence)
    - δ > 0.2: M → 0, Q → 0, Δ survives (conventional SC)
    """
    M_values = []
    Q_values = []
    Delta_values = []
    
    print(f"\n{'='*70}")
    print("Scanning phase diagram over doping range")
    print(f"{'='*70}\n")
    
    for i, doping in enumerate(doping_range):
        density = 1.0 - doping
        print(f"\n[{i+1}/{len(doping_range)}] Solving for δ = {doping:.3f}")
        
        result = solver.solve_self_consistent(
            target_density=density,
            initial_M=0.5,
            initial_Q=0.0,
            initial_Delta=0.05,
            verbose=False
        )
        
        M_values.append(result['M'])
        Q_values.append(result['Q'])
        Delta_values.append(result['Delta'])
        
        print(f"    → M={result['M']:.3f}, Q={result['Q']:.4f}, Δ={result['Delta']:.4f}")
    
    # Plot phase diagram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(doping_range, M_values, 'r-o', linewidth=2, markersize=6, label='AFM (M)')
    ax.plot(doping_range, Q_values, 'g-s', linewidth=2, markersize=6, label='JT Distortion (Q)')
    ax.plot(doping_range, Delta_values, 'b-^', linewidth=2, markersize=6, label='SC Gap (Δ)')
    
    ax.set_xlabel('Doping δ', fontsize=14)
    ax.set_ylabel('Order Parameters', fontsize=14)
    ax.set_title('Phase Diagram: SC-Activated JT Mechanism', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([doping_range[0], doping_range[-1]])
    
    # Mark expected regions
    ax.axvspan(0, 0.03, alpha=0.1, color='red', label='AFM Phase')
    ax.axvspan(0.05, 0.15, alpha=0.1, color='blue', label='SC+JT Phase')
    
    plt.tight_layout()
    return fig

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  SC-Activated Jahn-Teller Model - 16×16 Implementation           ║
    ║  Based on Unified D₄h Specification with RMFT                     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # Setup physical parameters
    # =========================================================================
    params = ModelParams(
        t0=0.5,           # eV - moderate hopping
        U=3.0,            # eV - charge transfer regime
        Delta_CF=0.1,     # eV - small CF splitting (quasi-degenerate)
        g_JT=0.15,        # eV/Å - moderate e-ph coupling
        K_lattice=5.0,    # eV/Å² - phonon stiffness
        beta=0.5,         # dimensionless
        eta=0.1,          # weak AFM on Γ₇
        nk=32,            # k-grid resolution
        kT=0.01,          # temperature ~ 116 K
        max_iter=100,
        mixing=0.3
    )
    
    solver = RMFT_Solver(params)
    
    # =========================================================================
    # Case 1: Near half-filling (small doping)
    # Expected: Pure AFM, no JT, no SC
    # =========================================================================
    print("\n" + "="*70)
    print("CASE 1: Near Half-Filling (δ = 0.02)")
    print("="*70)
    
    result_halffill = solver.solve_self_consistent(
        target_density=0.98,  # δ = 0.02
        initial_M=0.8,
        initial_Q=0.0,
        initial_Delta=0.05,
        verbose=True
    )
    
    fig1 = plot_convergence(result_halffill, 
                           title="Case 1: Near Half-Filling (Pure AFM Phase)")
    
    # =========================================================================
    # Case 2: Optimal doping
    # Expected: SC+JT coexistence, reduced AFM
    # =========================================================================
    print("\n" + "="*70)
    print("CASE 2: Optimal Doping (δ = 0.10)")
    print("="*70)
    
    result_optimal = solver.solve_self_consistent(
        target_density=0.90,  # δ = 0.10
        initial_M=0.5,
        initial_Q=0.0,
        initial_Delta=0.05,
        verbose=True
    )
    
    fig2 = plot_convergence(result_optimal,
                           title="Case 2: Optimal Doping (SC+JT Coexistence)")
    
    # =========================================================================
    # Case 3: Heavy doping
    # Expected: Weak AFM, SC without strong JT
    # =========================================================================
    print("\n" + "="*70)
    print("CASE 3: Heavy Doping (δ = 0.20)")
    print("="*70)
    
    result_heavy = solver.solve_self_consistent(
        target_density=0.80,  # δ = 0.20
        initial_M=0.3,
        initial_Q=0.0,
        initial_Delta=0.05,
        verbose=True
    )
    
    fig3 = plot_convergence(result_heavy,
                           title="Case 3: Heavy Doping (Conventional SC)")
    
    # =========================================================================
    # Phase diagram scan
    # =========================================================================
    print("\n" + "="*70)
    print("SCANNING FULL PHASE DIAGRAM")
    print("="*70)
    
    doping_scan = np.linspace(0.01, 0.25, 10)
    fig4 = plot_phase_diagram(solver, doping_scan)
    
    # =========================================================================
    # Display all figures
    # =========================================================================
    plt.show()
    
    print(f"\n{'='*70}")
    print("Simulation complete!")
    print(f"{'='*70}\n")