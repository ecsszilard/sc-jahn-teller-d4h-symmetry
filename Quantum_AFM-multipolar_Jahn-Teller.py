import numpy as np
import opt_einsum as oe
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.optimize import brentq
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings
import copy
import time as _time
import concurrent.futures
import os
import sys
import threading as _threading

_log_lock = _threading.Lock()

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
      doping_0 = z_ZRS/(1−z_ZRS)       (ZRS spectral weight scale)
    """
    # --- Primary inputs ---
    t_pd:          float      # eV    pd hybridisation integral (independent of Δ_CT; typ. 0.8–1.5 eV)
    u:             float      # —     U/t0 ratio; U = u·t0 = u·t_pd²/Δ_CT (charge-transfer: typ. 6–12)
    lambda_soc:    float      # eV    atomic SOC λ (t2g shell, ~0.05–0.15 eV); determines Γ₆–Γ₇ splitting
    Delta_tetra:   float      # eV    tetragonal axial CF Δ_tet·Lz²; negative = z-compression
                              #       Partial cancellation with SOC tunes Γ₆–Γ₇ gap independently of λ
    g_JT:          float      # eV/Å  Jahn–Teller electron–phonon coupling
                              #       increasing g_JT beyond the SC-triggered window is risky, because spontaneous JT (G3[2,2] < 0) always precedes the RPA cross-channel divergence.
    K_lattice:     float      # eV/Å² bare lattice spring constant (phonon stiffness, no exchange)
                              #       Physical: ω_JT=60meV, mass~Cu → K~1-2 eV/Å²; K_eff < K_lattice after exchange correction
                              #       K_lattice must satisfy: K_spont = g²/Δ_CF < K < g²/(0.05·π·t0) for SC-JT window (λ_JT > 0.05).
    lambda_hop:    float      # Å     hopping decay length for B₁g anisotropy: t(Q) = t0·exp(±Q/λ_hop)
    eta:           float      # —     Γ₇ AFM asymmetry relative to Γ₆
    # --- Charge-transfer / RPA / gap symmetry ---
    Delta_inplane: float      # eV    B2g in-plane anisotropy Δ_ip·(Lx²−Ly²); splits Γ₇ into Γ₇a+Γ₇b
                              #       (preserves Kramers, prevents spontaneous JT from residual Γ₇ degeneracy)
    Delta_CT:      float      # eV    charge-transfer gap (ZSA scale); sets scale for CT-insulator crossover
    omega_JT:      float      # eV    JT phonon frequency (40–80 meV); enters only D_phonon = 2/ω_JT
                              #       All free-energy magnitudes use adiabatic g²/K
    rpa_cutoff:    float      # —     Determinant floor for 2×2 RPA matrix |det| < rpa_cutoff.
                              #       Acts as a regulariser for near-degenerate denominators
                              #       while PRESERVING sign (sign flip = attractive → repulsive).
                              #       When J_eff*χ_SS ≥ 1 (Stoner / AFM QCP) the vertex returns 0 instead of diverging 

    # --- SCF numerical hyper-parameters (tune once, do NOT Bayesian-optimise) ---
    mu_LM:         float      # Levenberg–Marquardt floor for M Newton step (default 4.0), larger → smaller γ_M → more conservative M update.
    ALPHA_HF:      float      # Newton vs BdG fixpoint blend for M update (0=fixpoint, 1=Newton; default 0.2)

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
        evals, _evecs_soc = np.linalg.eigh(_build_soc_cf_hamiltonian(self.lambda_soc, self.Delta_tetra, self.Delta_inplane))
        self.Delta_CF: float     = float(evals[2] - evals[0])
        self.g7split: float      = float(evals[4] - evals[2])    # Γ₇a–Γ₇b internal split
        self.U_gamma: np.ndarray = _evecs_soc                    # Diagonalise H_SOC+CF; U_gamma columns = eigenvectors (ascending energy):
        self._U4:     np.ndarray = _evecs_soc[:, 0:4]            # _U4 = U_gamma[:, 0:4] is the 4-dim BdG projection (exact when Δ_CF ≫ kT).

        self.t0: float = self.t_pd**2 / max(self.Delta_CT, 1e-9)
        self.U: float = self.u * self.t0
        # ZSA superexchange for a charge-transfer insulator (two virtual-hopping paths):
        #   1/U          : pd→dd excitation (Mott channel, upper Hubbard band)
        #   1/(Δ_CT+U/2) : pd→pp excitation (Zhang-Rice channel, ligand holes)
        _dct  = max(self.Delta_CT, 1e-9)
        _U    = max(self.U, 1e-9)
        _J_ct: float = (2.0 * self.t_pd**4 / _dct**2) * (1.0 / _U + 1.0 / (_dct + _U / 2.0))

        # doping_0: Zhang–Rice singlet spectral weight scale in charge-transfer insulator: z_ZRS ≈ t_pd^2 / (Δ_CT^2 + t_pd^2)
        # Below the crossover doping level: t_pd^2 / Δ_CT^2 , the ZRS band is mostly incoherent (Mott localized), so the g_J→4 divergence must be suppressed
        _z_ZRS = self.t_pd**2 / (_dct**2 + self.t_pd**2)
        self.doping_0: float = float(np.clip(_z_ZRS / max(1.0 - _z_ZRS, 1e-9), 0.01, 0.25))
        self.U_mf: float = self.Z * _J_ct / 2.0  # bare MF amplitude before Gutzwiller renormalisation, BdG applies g_J·f_d at runtime, so U_mf must NOT include g_J here.
        self.J_CT: float = _J_ct
        self._t0_ref: float = self.t0  # store reference hopping for Q-scaling
        if self.nk % 2 != 0:
            self.nk = self.nk + 1

    def summary(self, delta: float = 0.15) -> None:
        g_t   = 2.0 * delta / (1.0 + delta)
        g_J   = 4.0 / (1.0 + delta) ** 2
        t_eff = g_t * self.t0
        f_d   = delta / (delta + self.doping_0)

        _h_prefactor = g_J * f_d * (self.U_mf / 2.0 + self.Z * 2.0 * t_eff**2 / self.U)
        h_afm_M1     = _h_prefactor * 1.00 / 2.0   # M=1: fully-saturated upper bound (Mott insulator limit, unphysical for metals)
        M_phys       = 0.15
        h_afm_Mphys  = _h_prefactor * M_phys / 2.0
        ok_metal     = h_afm_Mphys < 2.0 * t_eff  # Metallic AFM criterion: h_afm(M_phys) < 2·t_eff  (Weiss field < half-bandwidth)

        K_spont = self.g_JT**2 / max(self.Delta_CF, 1e-9)
        nk_odd         = self.nk + 1                  # odd   → Simpson integration requires an ODD grid.
        k_odd          = np.linspace(-np.pi, np.pi, nk_odd, endpoint=False)
        KX_odd, KY_odd = np.meshgrid(k_odd, k_odd)
        self.k_points  = np.column_stack((KX_odd.flatten(), KY_odd.flatten()))
        self.N_k       = len(self.k_points)
        self.k_weights = _simpson_weights_2d(nk_odd, nk_odd)
        
        k_even = np.linspace(-np.pi, np.pi, self.nk, endpoint=False)
        KX_ev, KY_ev = np.meshgrid(k_even, k_even)
        self.k_points_even   = np.column_stack((KX_ev.flatten(), KY_ev.flatten()))
        self.N_k_even        = len(self.k_points_even)        
        self.k_weights_even  = np.full(self.N_k_even, 1.0 / self.N_k_even)  # Uniform weights for the even grid (trapezoidal / rectangular rule)
        self.chi0_Q_idx = ((np.arange(self.N_k_even) // self.nk + self.nk // 2) % self.nk) * self.nk + (np.arange(self.N_k_even) %  self.nk + self.nk // 2) % self.nk  # Precompute AFM shift index: chi0_Q_idx[i] = index of k_i + Q_AFM in k_points_even

        print("\n================ MODEL PARAMS SUMMARY ================\n")
        print("Primary inputs:")
        print(f"  t_pd={self.t_pd:.4f} eV   Δ_CT={self.Delta_CT:.4f} eV   → t0={self.t0:.4f} eV (derived)")
        print(f"  u={self.u:.3f}   U={self.U:.4f} eV")
        print(f"  λ_SOC={self.lambda_soc:.4f} eV   Δ_tet={self.Delta_tetra:.4f} eV"
              f"   Δ_ip={self.Delta_inplane:.4f} eV")
        print(f"  ω_JT={self.omega_JT:.4f} eV")
        print(f"  g_JT={self.g_JT:.4f} eV/Å")
        print(f"  Z={self.Z}   η={self.eta:.4f}   δ₀={self.doping_0:.4f}")
        print(f"  {self.N_k} k-pts (SCF/Simpson odd nk={nk_odd}), {self.N_k_even} k-pts (χ₀ even nk={self.nk})")

        print("\nDerived quantities (from __post_init__):")
        print(f"  Δ_CF   = {self.Delta_CF:.5f} eV   (Γ₆–Γ₇a SOC+CF gap)")
        print(f"  t0     = {self.t0:.5f} eV   (= t_pd²/Δ_CT, ZSA dd hopping)")
        print(f"  J_CT   = {self.J_CT:.5f} eV   (ZSA CT superexchange: 2t_pd⁴/Δ_CT²·(1/U + 1/(Δ_CT+U/2)))")
        print(f"  U_mf   = {self.U_mf:.5f} eV   (= Z·J_CT/2, bare MF Weiss amplitude)")
        print(f"  V_eff_bare = {self.g_JT**2 / max(self.K_lattice, 1e-9):.5f} eV  (= g²/K_lattice, bare adiabatic JT pairing scale)")
        print(f"  Γ₇ split = {self.g7split:.5f} eV"
              f"  [{'⚠ < 2kT — residual Γ₇ degeneracy' if self.g7split < 2.0 * self.kT else '✓ > 2kT'}]")
        print(f"\nMagnetic regime check (δ={delta:.3f}):")
        print(f"  h_afm prefactor = {_h_prefactor:.5f} eV  (= g_J·f_d·(U_mf/2+Z·2t²/U))")
        print(f"  M=1.00 (saturated): h_afm = {h_afm_M1:.5f} eV  vs  2t_eff = {2*t_eff:.5f} eV"
              f"  {'✓' if h_afm_M1 < 2*t_eff else '⚠ insulating at M=1'}")
        print(f"  M={M_phys:.2f} (typical SC+AFM): h_afm = {h_afm_Mphys:.5f} eV  vs  2t_eff = {2*t_eff:.5f} eV"
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
        print(f" kT={self.kT*1000:.2f} meV  mixing={self.mixing:.4f}  rpa_cutoff={self.rpa_cutoff:.4f}")
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
    return weights_2d.flatten()

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

class RMFT_Solver:
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

        # Orbital operators derived from the SOC+CF eigenbasis.
        self._rebuild_orbital_operators(params)

        self.phi_k = (np.cos(self.k_points[:, 0] * params.a)
                      - np.cos(self.k_points[:, 1] * params.a))
        self.phi_k_even = (np.cos(self.k_points_even[:, 0] * params.a)
                           - np.cos(self.k_points_even[:, 1] * params.a))
        self._D_phonon: float = 2.0 / max(params.omega_JT, 1e-6)
        V_eff_bare = params.g_JT**2 / params.K_lattice
        _scf_log("RMFT-INIT",
                 f"t_pd={params.t_pd:.4f} eV  Δ_CT={params.Delta_CT:.4f} eV"
                 f"  t0={params.t0:.4f} eV  U={params.U:.4f} eV"
                 f"  Δ_CF={params.Delta_CF:.4f} eV"
                 f"  g_JT={params.g_JT:.4f} eV/Å  K_lattice={params.K_lattice:.4f} eV/Å²"
                 f"  V_eff_bare={V_eff_bare:.4f} eV"
                 f"  λ_eff≈{V_eff_bare/(np.pi*params.t0):.4f}"
                 f"  K_spont = {params.g_JT**2 / max(params.Delta_CF, 1e-9):.4f} eV/Å²")
        self._vbdg: Optional['VectorizedBdG'] = None
        self._scf_bdg_cache: Optional[tuple] = None
        self._cluster_j_renorm: float = 1.0   # cluster ED vertex correction; 1.0 = bare Gutzwiller
        self._K_bare: float = params.K_lattice # immutable bare lattice spring constant (eV/Å²)

    def _rebuild_orbital_operators(self, params: 'ModelParams') -> None:
        """Rebuild all SOC+CF-basis-dependent operators from params._U4.

        Must be called whenever params.lambda_soc, params.Delta_tetra, or
        params.Delta_inplane changes and params.__post_init__() has been called
        (which regenerates _U4).
        """
        U4 = params._U4  # columns = {Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓}
        P6_t2g    = U4[:, 0:2] @ U4[:, 0:2].conj().T            # (6,6)
        P7_t2g    = U4[:, 2:4] @ U4[:, 2:4].conj().T            # (6,6)
        tau_x_t2g = (U4[:, 0:2] @ U4[:, 2:4].conj().T
                   + U4[:, 2:4] @ U4[:, 0:2].conj().T)           # (6,6)
        self.P6   = np.real(U4.conj().T @ P6_t2g  @ U4)          # (4,4)
        self.P7   = np.real(U4.conj().T @ P7_t2g  @ U4)          # (4,4)
        tau_x_op  = (U4.conj().T @ tau_x_t2g @ U4)               # (4,4), complex
        z4 = np.zeros((4, 4), dtype=complex)
        # BdG particle–hole symmetry requires the hole block to carry −τ_x^T, nambu structure: O_Nambu = block_diag(O_AA, -O_AA^T)
        tau_x_op_T = tau_x_op.T   # transpose (not conjugate-transpose)
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
        self.sz_bdg_op = np.array([1.0, -1.0, params.eta, -params.eta])

    def _get_vbdg(self) -> 'VectorizedBdG':
        if self._vbdg is None:
            self._vbdg = VectorizedBdG(self)
        return self._vbdg

    def _reset_transient_state(self) -> None:
        """Reset all mutable per-solve caches on a solver clone.

        Must be called after copy.copy(solver) to guarantee that:
          - _vbdg          : gets a fresh VectorizedBdG with this clone's k-grids
          - _scf_bdg_cache : previous BdG (ev, ec) from a different solve is not reused
          - _cluster_j_renorm : exchange vertex correction starts at bare value
          - _K_bare        : preserved (immutable per __init__ contract); NOT cleared

        """
        self._vbdg             = None   # re-created on first _get_vbdg()
        self._scf_bdg_cache    = None   # no stale (ev, ec) from parent solve
        self._cluster_j_renorm = 1.0    # bare Gutzwiller vertex

    def get_gutzwiller_factors(self, delta: float) -> Tuple[float, float, float, float]:
        """
        g_t       = 2δ/(1+δ)  — kinetic energy; → 0 at half-filling (Mott insulator)
        g_J       = 4/(1+δ)²  — exchange enhancement; → 4 at half-filling (J = 4t²/U)
        g_Delta_s = g_t        — on-site inter-orbital Γ₆⊗Γ₇ singlet (in a charge-transfer system with strong AFM background)
                                  The spin vertex renormalizes separately as g_s inside the RPA vertex (compute_gap_eq_vectorized)
        g_Delta_d = g_J        — inter-site d-wave B₁g renormalization. Superexchange-mediated: scales with the same vertex as J
        """
        abs_delta  = max(abs(delta), 1e-6) # Half-filling floor
        g_t        = (2.0 * abs_delta) / (1.0 + abs_delta)
        g_J        = 4.0 / ((1.0 + abs_delta) ** 2)
        g_Delta_s  = g_t
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
    
    def effective_superexchange(self, g_J: float, tx_bare: float, ty_bare: float, doping: float, direction: str = 'z') -> float:
        """
        Returns the superexchange coupling J_eff for the cluster Hamiltonian.
        
        This is the PURE superexchange coupling for the cluster Hamiltonian (ZSA charge-transfer mechanism),
        NOT including the kinematic exchange (2t²/U) which is handled separately in the Weiss field.
        
        Formula: J_eff = g_J * f_d * (2·t_pd⁴/Δ_CT²) · (1/U + 1/(Δ_CT + U/2))
        """
        abs_doping = max(abs(doping), 1e-6)
        f_doping = abs_doping / (abs_doping + self.p.doping_0)
        _dct = max(self.p.Delta_CT, 1e-9)
        _U = max(self.p.U, 1e-9)
        
        # tx_bare = t0 = t_pd²/Δ_CT, so t_sq = t_pd⁴/Δ_CT²
        if direction == 'x':
            t_sq = tx_bare**2
        elif direction == 'y':
            t_sq = ty_bare**2
        else:  # 'z' or average (isotropic AFM case)
            t_sq = 0.5 * (tx_bare**2 + ty_bare**2)
        return g_J * f_doping * 2.0 * t_sq * (1.0/_U + 1.0/(_dct + _U/2.0))

    def J_alpha_beta_Q(self, Q: float, lambda_hop: float) -> np.ndarray:
        """
        Q-dependent multipolar exchange matrix in the [Γ₆↑, Γ₆↓, Γ₇↑, Γ₇↓] basis.

        Irrep decomposition (D₄h):
            J(Q) = J_A1g(Q) · P_A1g  +  J_B1g(Q) · P_B1g
            P_A1g = diag(1,1,η²,η²),   P_B1g = η·τ_x   (Γ₆–Γ₇ mixing).

        Microscopic origin (hopping anisotropy):
            t_x = t₀ e^{+Q/λ},  t_y = t₀ e^{-Q/λ},  with J ∝ t²  ⇒
            J_A1g ∝ (J_CT/2) cosh(2Q/λ)   (even in Q),
            J_B1g ∝ (J_CT/2) sinh(2Q/λ)   (odd, vanishes at Q=0).

        MF normalization:
            the 1/2 factor comes from counting each bond once in the
            two-sublattice Heisenberg mean-field equation.

        Physical consequence:
            Q = 0  →  J_B1g = 0, Γ₆–Γ₇ mixing forbidden (AFM selection rule).
            Q ≠ 0  →  B1g channel opens, enabling multipolar/JT response.

        _cluster_j_renorm: scalar vertex correction from cluster ED
        (rescales J but preserves the A1g/B1g structure).
        """
        lam = max(lambda_hop, 1e-9)

        scale_A1g = float(np.cosh(2.0 * Q / lam))
        scale_B1g = float(np.sinh(2.0 * Q / lam))

        eta = self.p.eta
        j_renorm = getattr(self, '_cluster_j_renorm', 1.0)

        J_A1g = j_renorm * (self.p.J_CT / 2.0) * scale_A1g * np.diag([1.0, 1.0, eta**2, eta**2])
        J_B1g = j_renorm * (self.p.J_CT / 2.0) * scale_B1g * eta * self.tau_x_mat
        return J_A1g + J_B1g
    
    def compute_JT_rigidity_from_exchange(self, M: float, Q: float, mu: float, g_J: float, target_doping: float) -> Dict:
        """
        Exchange contribution to the JT stiffness: ∂²F_ex/∂Q².

            F_ex = Σ_{αβ} J_{αβ}(Q) ⟨O_α(Q)⟩⟨O_β(Q)⟩

        Full second derivative via product rule:
            ∂²F_ex/∂Q² = O·(∂²J/∂Q²)·O + 4·(∂O/∂Q)·J·(∂O/∂Q)
                        + 2·O·J·(∂²O/∂Q²) + 4·O·(∂J/∂Q)·(∂O/∂Q)

        All four terms are included. At Q=0 the B1g selection rule forces
        ∂O/∂Q = 0 and ∂²J/∂Q² = 0 (sinh→cosh, leading term ∝ Q²), so only
        the term 2·O·J·(∂²O/∂Q²) survives — but at Q≠0 all terms contribute
        and omitting any would bias the SCF Q-update.

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
        f_d    = abs_d / (abs_d + self.p.doping_0)
        sz_op  = np.array([1.0, -1.0, self.p.eta, -self.p.eta])

        g_t_loc, _, _, _ = self.get_gutzwiller_factors(target_doping)
        vbdg = self._get_vbdg()

        tx_0, ty_0 = self.effective_hopping_anisotropic(Q)
        tx_p, ty_p = self.effective_hopping_anisotropic(Q + eps)
        tx_m, ty_m = self.effective_hopping_anisotropic(Q - eps)

        t_sq_avg  = 0.5 * (tx_0**2 + ty_0**2)
        h_afm_eff = g_J * f_d * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) * M / 2.0

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
            'd2F_ex_dQ2':    d2F_ex_dQ2,
            'K_eff':         K_eff,
            'is_stable':     K_eff > 0.0,
            'h_afm_eff':     h_afm_eff,
            'O_exp':         O_exp_Q,
            'dO_dQ':         dO_dQ,
            'comm_tau_H':    comm,
            'comm_norm':     comm_norm,
            'blocking_ratio': comm_norm / max(abs(self.p.Delta_CF), 1e-9),
        }
    
    def fermi_function(self, E: np.ndarray) -> np.ndarray:
        arg = E / self.p.kT
        arg = np.clip(arg, -100, 100)
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

        Returns:
            dict with keys:
              'chi0'        : float, static susceptibility (eV⁻¹)
              'U_eff_chi'   : float, renormalised magnetic coupling used in Stoner denominator (eV),  NOT the bare Hubbard U. This keeps U_eff_chi · χ₀ ~ O(1) within the ordered AFM phase
              'rpa_factor'  : float, AFM QCP crossed (magnetic instability); returns 1.0 (no enhancement) — the ordered state has broken down and the linear RPA is invalid.
              'afm_unstable': bool, True if stoner_denom ≤ 0 (AFM QCP crossed, magnetically unstable)
        """
        # Spin operator in 4-orbital BdG Nambu basis (diagonal):
        # Particle A: S_z = diag(+1,-1,+η,-η); B: staggered -1; holes: p-h conjugate signs.
        sz_orb   = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
        sz_diag  = np.concatenate([ sz_orb,   # particle A
                                   -sz_orb,   # particle B (staggered)
                                   -sz_orb,   # hole A (p-h conjugate)
                                    sz_orb])  # hole B

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

        mask    = (np.abs(df) > 1e-12) & (np.abs(dE) > 1e-6)
        safe_dE = np.where(mask, dE, 1.0)
        ratio   = np.where(mask, self.k_weights_even[:, None, None] * M2 * df / safe_dE, 0.0)
        chi0 = float(ratio.sum())
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        J_eff_now = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping)
        stoner_denom = 1.0 - J_eff_now * chi0
        return {
            'chi0':         chi0,
            'U_eff_chi':    J_eff_now,
            'rpa_factor':   1.0 / max(stoner_denom, self.p.rpa_cutoff) if stoner_denom > 0.0 else 1.0,
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
            uA = ec[:, 0:4,  :];  uB = ec[:, 4:8,  :]
            vA = ec[:, 8:12, :];  vB = ec[:, 12:16, :]
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

    def compute_chi0_tensor(self, q: np.ndarray, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, _E_k_cache: tuple = None, _E_kq_cache: tuple = None) -> np.ndarray:
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
        """
        chi0 = np.zeros((4, 4), dtype=complex)

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

        vbdg = self._get_vbdg()

        # k-grid: reuse Delta=0 cache when provided (callers must supply one).
        if _E_k_cache is not None:
            E_k_all, V_k_all = _E_k_cache
        else:  # Fallback: build at Delta=0 as safety net.
            E_k_all, V_k_all = np.linalg.eigh(
                vbdg._build_H_stack(vbdg._kpts_ev, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack_ev))

        # k+q grid: use pre-computed cache if available, otherwise diagonalise now.
        if _E_kq_cache is not None:
            E_kQ_all, V_kQ_all = _E_kq_cache
        else:
            kpts_kq = (self.k_points_even + q[None, :] + np.pi) % (2.0 * np.pi) - np.pi
            E_kQ_all, V_kQ_all = np.linalg.eigh(
                vbdg._build_H_stack(kpts_kq, M, Q, 0.0+0j, 0.0+0j,
                                    target_doping, mu, tx, ty, g_J))

        f_k_all  = self.fermi_function(E_k_all)
        f_kQ_all = self.fermi_function(E_kQ_all)

        # Standard Lindhard kernel
        df      = f_k_all[:, :, None] - f_kQ_all[:, None, :]
        dE      = E_kQ_all[:, None, :] - E_k_all[:, :, None]
        mask    = (np.abs(df) > 1e-12) & (np.abs(dE) > 1e-6)
        safe_dE = np.where(mask, dE, 1.0)
        factor  = -np.where(mask, self.k_weights_even[:, None, None] * df / safe_dE, 0.0)

        # Chunked k-loop: avoids building L/R/FR arrays of shape (N_k,4,4,16) all at once.
        N = len(self.k_points_even)
        CHUNK = 128
        for sl_k, sl_kQ in SECTOR_PAIRS:
            Vk_s  = V_k_all[:,  sl_k,  :]   # (N, 4, 16)
            VkQ_s = V_kQ_all[:, sl_kQ, :]   # (N, 4, 16)
            for k0 in range(0, N, CHUNK):
                k1    = min(k0 + CHUNK, N)
                fac_c = factor[k0:k1]       # (C, 16_k, 16_kq)
                Vk_c  = Vk_s[k0:k1]         # (C,  4,   16_k)
                VkQ_c = VkQ_s[k0:k1]        # (C,  4,   16_kq)
                chi0 += oe.contract('cnm,can,cbn,cam,cbm->ab',
                                    fac_c, Vk_c.conj(), Vk_c, VkQ_c, VkQ_c.conj(),
                                    optimize='optimal')
        return chi0

    def _get_fermi_surface_sample(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, n_fs: int = 48) -> Tuple[np.ndarray, np.ndarray]:
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
        ev_all, _ = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack))

        near_fs = np.any(np.abs(ev_all) < 3.0 * self.p.kT, axis=1)

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

    def _orbital_rpa_vertex(self, chi0_mat: np.ndarray, J_eff: float, V_JT: float, chi_QQ_normal: float, rpa_cutoff: float, _log_chi_sq: bool = False, _return_det: bool = False):
        """
        2×2 coupled spin–JT RPA pairing vertex.

        Bare Hamiltonian: H_int = J_eff*S·S + g_JT*Q*τ_x + K*Q²/2
        There is NO bare S–Q cross-vertex in H_int, so V_mix = 0 in Û.
        The spin–JT feedback enters exclusively through chi_SQ/chi_QS
        (opened by SOC + SC condensate), avoiding double-counting.

          Û = diag(J_eff, V_JT)     chi0_tilde = [[chi_SS, chi_SQ  ],
                                                   [chi_QS, chi_QQ/K]]

        KEY PHYSICS — the cross-channel determinant:
            det(RPA) = (1 - J_eff·χ_SS)(1 - V_JT·χ_QQ/K) - J_eff·V_JT·χ_SQ·χ_QS
        where χ_SQ is the SC-condensate-opened Γ₆↔Γ₇ channel.
        Even if the lattice is self-stable on its own (2nd bracket > 0), the
        cross-term −J_eff·V_JT·χ_SQ·χ_QS can flip det → 0, triggering the JT
        distortion.  Near the AFM QCP (J_eff·χ_SS → 1) this cross-channel
        divergence is maximally amplified — this is the SC-triggered JT condition.

        OPERATOR-ALGEBRA NOTE:
            χ_SS = Tr[S_z · χ₀[Γ₆,Γ₆] · S_z]   : spin–spin, Γ₆ block only
            χ_SQ = Tr[S_z · χ₀[Γ₆,Γ₇]]          : spin(Γ₆)–quadrupole(Γ₇) cross
            χ_QS = Tr[χ₀[Γ₇,Γ₆] · S_z]          : quadrupole(Γ₇)–spin(Γ₆) cross

        IMPORTANT: chi_QQ_normal must be the NORMAL-STATE (Δ=0) susceptibility.
        The SC-state χ_QQ is used only for the lattice stability diagnostic
        (compute_G_instability / compute_d2F_dQ2_at_Delta), NOT here.
        Using Δ≠0 χ_QQ in this vertex would conflate the pairing kernel with
        the condensate-driven JT feedback — a double-counting error.

        V_pair = g^T χ^RPA g = J²χ_SS^RPA + V²χ_QQ^RPA + J·V(χ_SQ^RPA + χ_QS^RPA)  [eV]

        Args:
            chi0_mat       : (4,4) orbital bare susceptibility [eV^{-1}]
            J_eff          : superexchange Stoner parameter [eV]
            V_JT           : g_JT^2 / K_bare [eV]
            chi_QQ_normal  : −d²Ω/dQ²|_{Δ=0} [eV/Ang^2]; NORMAL-STATE only
            rpa_cutoff     : regularisation floor for the 2x2 determinant
            _log_chi_sq    : if True, return chi_SQ value via side-channel for
                             diagnostic logging (SC-triggered channel monitoring)
        """
        K_bare = max(self._K_bare, 1e-9)

        # ── Operator-algebraic susceptibility projections ──────────────────
        # Basis ordering: [Γ₆↑(0), Γ₆↓(1), Γ₇↑(2), Γ₇↓(3)]
        # S_z in Γ₆ subspace: diag(+1, -1)  [2×2]
        S_z = np.array([[1.0, 0.0],
                        [0.0, -1.0]])

        chi_66 = chi0_mat[0:2, 0:2]   # Γ₆–Γ₆ block  [2×2]
        chi_67 = chi0_mat[0:2, 2:4]   # Γ₆–Γ₇ block  [2×2]  (off-diagonal)
        chi_76 = chi0_mat[2:4, 0:2]   # Γ₇–Γ₆ block  [2×2]  (off-diagonal)

        # χ_SS = Tr[S_z χ₆₆ S_z]  — spin–spin response in the Γ₆ sector
        chi_SS = float(np.real(np.trace(S_z @ chi_66 @ S_z)))

        # χ_SQ = Tr[S_z χ₆₇]  — spin(Γ₆)–quadrupole(Γ₇) cross-channel, = 0 in the normal state by block-diagonal symmetry
        chi_SQ_raw = float(np.real(np.trace(S_z @ chi_67)))
        chi_QS_raw = float(np.real(np.trace(chi_76 @ S_z)))

        # Enforce normal-state symmetry: clamp numerical noise to zero.
        _eps_sym = 1e-12
        chi_SQ = 0.0 if abs(chi_SQ_raw) < _eps_sym else chi_SQ_raw
        chi_QS = 0.0 if abs(chi_QS_raw) < _eps_sym else chi_QS_raw

        # Store for external diagnostic access (set by caller when _log_chi_sq=True)
        if _log_chi_sq:
            self._last_chi_SQ = chi_SQ

        chi_QQ_tilde = chi_QQ_normal / K_bare   # [eV/Ang^2] / [eV/Ang^2] -> [eV^{-1}]

        # (I - Û @ chi0_tilde), Û = diag(J_eff, V_JT), V_mix = 0
        a = 1.0 - J_eff * chi_SS
        b =     - J_eff * chi_SQ
        c =     - V_JT  * chi_QS          # V_mix = 0: cross comes from chi, not Û
        d = 1.0 - V_JT  * chi_QQ_tilde
        
        # det = (1 - J_eff*chi_SS)*(1 - V_JT*chi_QQ_tilde) - J_eff*V_JT*chi_SQ*chi_QS
        det = a * d - b * c

        # Smooth QCP suppression:
        # When a = 1 - J_eff*chi_SS ≤ margin OR det ≤ margin we are at or beyond the
        # AFM/Stoner QCP where the paramagnon-mediated pairing vertex diverges unphysically.
        # Apply an exponential penalty that:
        #   (a) drives V_pair → 0 smoothly as the instability deepens (det → 0⁻ or a → 0⁻)
        #   (b) preserves the gradient so the GP can navigate back to the stable region
        #   (c) uses the boundary value of det (=margin) to avoid sign flips
        _margin = max(rpa_cutoff, 1e-3)
        if a <= _margin or det <= _margin:
            penalty = float(np.exp(-10.0 * (_margin - min(a, det))))
            _safe = _margin
            _inv00 =  d / _safe;  _inv01 = -b / _safe
            _inv10 = -c / _safe;  _inv11 =  a / _safe
            _chi_SS  = _inv00 * chi_SS  + _inv01 * chi_QS
            _chi_QQ  = _inv10 * chi_SQ  + _inv11 * chi_QQ_tilde
            _chi_SQ  = _inv00 * chi_SQ  + _inv01 * chi_QQ_tilde
            _chi_QS  = _inv10 * chi_SS  + _inv11 * chi_QS
            V_pair = float((
                J_eff**2 * _chi_SS
                + V_JT**2 * _chi_QQ
                + J_eff * V_JT * (_chi_SQ + _chi_QS)
            ) * penalty)
            return (V_pair, det) if _return_det else V_pair

        inv00 =  d / det;  inv01 = -b / det
        inv10 = -c / det;  inv11 =  a / det

        # Full 2×2 RPA susceptibility matrix: χ^RPA = (I - Û χ̃)^{-1} χ̃
        #   g = (J_eff, V_JT)^T  [eV]
        #   χ̃ = [[χ_SS, χ_SQ], [χ_QS, χ_QQ_tilde]]  [eV^{-1}]
        # The correct pairing vertex is the bilinear form:
        #   V_eff = g^T χ^RPA g = J²χ_SS^RPA + V²χ_QQ^RPA + J·V(χ_SQ^RPA + χ_QS^RPA)  [eV]
        # Dimensions: [eV]²·[eV^{-1}] = [eV] ✓
        # The cross-term J·V·χ_SQ^RPA is the SC-triggered JT interference channel.
        chi_rpa_SS = inv00 * chi_SS  + inv01 * chi_QS          # χ_SS^RPA [eV^{-1}]
        chi_rpa_QQ = inv10 * chi_SQ  + inv11 * chi_QQ_tilde    # χ_QQ^RPA [eV^{-1}]
        chi_rpa_SQ = inv00 * chi_SQ  + inv01 * chi_QQ_tilde    # χ_SQ^RPA [eV^{-1}]
        chi_rpa_QS = inv10 * chi_SS  + inv11 * chi_QS          # χ_QS^RPA [eV^{-1}]
        V_pair = float(
            J_eff**2 * chi_rpa_SS
            + V_JT**2 * chi_rpa_QQ
            + J_eff * V_JT * (chi_rpa_SQ + chi_rpa_QS)
        )
        return (V_pair, det) if _return_det else V_pair

    def solve_linearized_gap_equation(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> Dict:
        """
        Linearised gap equation solved as an eigenvalue problem on the Fermi surface.

        λ Δ(k_i) = Σ_j Γ_ij Δ(k_j)

        Γ_ij = g_Δ · V(k_i−k_j) / √(|v_F(i) v_F(j)|)

        with V(q) the full RPA vertex from _orbital_rpa_vertex.
        """

        fermi_pts, vF = self._get_fermi_surface_sample(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
        N = fermi_pts.shape[0]

        # ─────────────────────────────────────────────
        # k_i − k_j vectors
        # ─────────────────────────────────────────────
        i_idx, j_idx = np.triu_indices(N)

        q_raw = fermi_pts[i_idx] - fermi_pts[j_idx]
        q_arr = (q_raw + np.pi) % (2*np.pi) - np.pi

        scale = 1e5
        q_int = np.rint(q_arr * scale).astype(np.int64)
        unique_int, inv_idx = np.unique(q_int, axis=0, return_inverse=True)
        unique_q = unique_int.astype(np.float64) / scale

        # ─────────────────────────────────────────────
        # BdG normal-state cache
        # ─────────────────────────────────────────────
        vbdg = self._get_vbdg()
        E_k_cache, U_k_cache = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts_ev, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx, ty, g_J, out=vbdg._H_stack_ev))

        # ─────────────────────────────────────────────
        # interaction scales
        # ─────────────────────────────────────────────
        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        J_eff = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping)

        chi_QQ_normal = self._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)

        V_JT = self.p.g_JT**2 / max(self._K_bare, 1e-9)

        # ─────────────────────────────────────────────
        # Batched k+q diagonalisation for all unique_q
        # ─────────────────────────────────────────────
        kq_caches: list = []
        for q_u in unique_q:
            kpts_kq = (self.k_points_even + q_u[None, :] + np.pi) % (2.0 * np.pi) - np.pi
            E_kq, V_kq = np.linalg.eigh(
                vbdg._build_H_stack(kpts_kq, M, Q, 0.0+0j, 0.0+0j,
                                    target_doping, mu, tx, ty, g_J))
            kq_caches.append((E_kq, V_kq))

        # ─────────────────────────────────────────────
        # vertex for unique q
        # ─────────────────────────────────────────────
        n_q = len(unique_q)

        V_unique = np.empty(n_q)
        V_spin_u = np.empty(n_q)
        V_JT_u   = np.empty(n_q)

        for u_idx, q_u in enumerate(unique_q):
            chi0 = self.compute_chi0_tensor(q_u, M, Q, 0.0+0j, 0.0+0j, target_doping, mu, tx, ty, g_J, _E_k_cache=(E_k_cache, U_k_cache), _E_kq_cache=kq_caches[u_idx])

            V_spin = self._orbital_rpa_vertex(chi0, J_eff, 0.0, chi_QQ_normal, self.p.rpa_cutoff)
            V_jt   = self._orbital_rpa_vertex(chi0, 0.0, V_JT, chi_QQ_normal, self.p.rpa_cutoff)
            V_full = self._orbital_rpa_vertex(chi0, J_eff, V_JT, chi_QQ_normal, self.p.rpa_cutoff)

            V_spin_u[u_idx] = V_spin
            V_JT_u[u_idx]   = V_jt
            V_unique[u_idx] = V_full

        # ─────────────────────────────────────────────
        # DOS weights
        # ─────────────────────────────────────────────
        vF_safe = np.maximum(np.abs(vF), 1e-8)
        inv_svF = 1.0 / np.sqrt(vF_safe)

        weights = inv_svF[i_idx] * inv_svF[j_idx]
        vals = weights * V_unique[inv_idx]

        # symmetric kernel
        Gamma = np.zeros((N, N), dtype=float)
        Gamma[i_idx, j_idx] = vals
        Gamma += Gamma.T

        # ─────────────────────────────────────────────
        # symmetry detection
        # ─────────────────────────────────────────────
        phi_s = np.ones(N)
        phi_d = np.cos(fermi_pts[:,0]*self.p.a) - np.cos(fermi_pts[:,1]*self.p.a)

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

        # ─────────────────────────────────────────────
        # Gutzwiller factors
        # ─────────────────────────────────────────────
        _, _, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)
        g_Delta_dom = g_Delta_d if w_d > w_s else g_Delta_s
        lambda_max = lambda_raw * g_Delta_dom

        # ─────────────────────────────────────────────
        # vertex diagnostics
        # ─────────────────────────────────────────────
        V_spin_mean = float(np.mean(V_spin_u))
        V_JT_mean   = float(np.mean(V_JT_u))
        V_rpa_mean  = float(np.mean(V_unique))

        V_cross_mean = V_rpa_mean - V_spin_mean - V_JT_mean

        # ─────────────────────────────────────────────
        # G3 matrix
        # ─────────────────────────────────────────────
        K_bare_for_vertex = max(self._K_bare, 1e-9)

        gVs = g_Delta_s * V_rpa_mean
        gVd = g_Delta_d * V_rpa_mean

        chi = self._compute_afm2band_susceptibilities(target_doping, M, Q, Delta_s, Delta_d, mu)
        G3, eigs3, lam3_min, instab_dir = self._build_G3_matrix(chi, gVs, gVd, K_bare_for_vertex)

        return {
            'lambda_max': lambda_max,
            'lambda_max_raw': lambda_raw,
            'g_delta_dom': g_Delta_dom,
            'gap_vector': gap_vector,
            'fs_pts': fermi_pts,
            'gap_symmetry': gap_symmetry,
            'G3': G3,
            'eigs3': eigs3,
            'lambda_min_3x3': lam3_min,
            'instab_dir': instab_dir,
            'V_spin_mean': V_spin_mean,
            'V_JT_mean': V_JT_mean,
            'V_cross_mean': V_cross_mean,
            'V_rpa_mean': V_rpa_mean
        }
    
    def _compute_orbital_coherence_from_pairs(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> float:
        """
        Anomalous orbital coherence ⟨τ_x⟩_anom from off-diagonal BdG amplitudes (u·v).

        Diagnostic only — not used in the SCF Q update (which employs the total ⟨τ_x⟩ from compute_observables_vectorized).

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
               h_α = g_J·f_d·M · [h_kin_scalar · sz_α  +  Σ_β J_{αβ} · sz_β]
        where:
          h_kin_scalar = Z · 2·t_eff²/U    (kinematic dd-exchange, Q-dependent)
          J_{αβ}       = J_A1g_{αβ} + J_B1g_{αβ}  (superexchange tensor, A₁g + B₁g)
          4. JT distortion:  H_JT = g_JT · Q · τ_x

        tx, ty : Gutzwiller-renormalised hoppings g_t·t(Q) (eV).

        O_expectation: optional 4-element array ⟨O_β⟩ for each orbital.
            If None: uses the MF approximation ⟨O_β⟩ = g_J·f_d·M · sz_β.

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
        abs_delta = max(abs(target_doping), 1e-6)
        f_delta   = abs_delta / (abs_delta + self.p.doping_0)

        # 4. Kinematic exchange scalar
        t_sq_avg = 0.5 * (tx**2 + ty**2)
        h_kin_scalar = self.p.Z * 2.0 * t_sq_avg / max(self.p.U, 1e-9) / 2.0

        if O_expectation is None:
            sz_op = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
            O_exp = g_J * f_delta * M * sz_op
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
        sz_op = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
        u = vec[u_slice]
        v = vec[v_slice]
        
        # Magnetization: both terms are positive: |u|²·f gives the particle contribution, |v|²·(1-f) gives the filled-band (hole) electron contribution.
        # sz_op carries the ±1 spin weighting; the minus for spin-down comes from sz_op itself.
        m = (np.abs(u)**2 @ sz_op) * f + (np.abs(v)**2 @ sz_op) * f_bar
        
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

        uA = ec_all[:, 0:4,   :]
        uB = ec_all[:, 4:8,   :]
        vA = ec_all[:, 8:12,  :]
        vB = ec_all[:, 12:16, :]

        dens_A = np.sum(np.abs(uA)**2 * fn[:, None, :]
                      + np.abs(vA)**2 * fn_bar[:, None, :], axis=(1, 2))  # (N_k,)
        dens_B = np.sum(np.abs(uB)**2 * fn[:, None, :]
                      + np.abs(vB)**2 * fn_bar[:, None, :], axis=(1, 2))

        n_avg = (dens_A + dens_B) / 4.0  # BdG doubling correction
        return float(np.dot(self.k_weights, n_avg))
    
    def _find_mu_for_density(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, tx: float, ty: float, mu_guess: float, g_J: float) -> Tuple[float, Optional[tuple]]:
        """Returns (mu, last_bdg_cache) where last_bdg_cache = (ev, ec) at the
        converged mu — so the caller can skip the redundant eigh in
        _compute_density_at_mu for the same (M,Q,Δ,μ) point."""
        target_n = 1.0 - target_doping
        _last_cache: Optional[tuple] = None

        def density_and_deriv(mu_val: float):
            nonlocal _last_cache
            vbdg = self._get_vbdg()
            ev, ec = np.linalg.eigh(vbdg._build_H_stack(vbdg._kpts, M, Q, Delta_s, Delta_d, target_doping, mu_val, tx, ty, g_J, out=vbdg._H_stack))
            _last_cache = (ev, ec)
            f   = self.fermi_function(ev)
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
            if abs(deriv) < 1e-12:
                break   # flat → fall through to brentq
            # dn/dmu ≥ 0 always; abs() guards against rare numerical noise giving tiny negatives
            step = err / max(abs(deriv), 1e-10)
            # Limit step to bandwidth/4 to avoid overshooting
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
                mu = brentq(density_error, mu_min, mu_max, xtol=1e-5)
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

        The condensation correction restores the variational stationarity:
            ∂Ω/∂Δ_s = 0  ↔  Δ_s = g_s · V_s · F_AA_BZ  (gap equation)

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
        if self.p.kT > 1e-8:
            f_c = np.clip(f_n, 1e-12, 1 - 1e-12)
            S_kn = -(f_c*np.log(f_c) + (1-f_c)*np.log(1-f_c))
            S_term = self.p.kT * np.einsum('k,kn->', self.k_weights, S_kn)
        else:
            S_term = 0.0

        Omega_cell = Ef - S_term

        # Elastic energy uses the full effective spring constant (K_eff = K_lattice + ∂²F_ex/∂Q²).
        _K_eff = max(K_eff_for_free_energy if K_eff_for_free_energy is not None else self._K_bare, 1e-9)
        elastic_energy = 0.5 * _K_eff * Q**2

        # Condensation correction: |Δ_ℓ|² / (g_ℓ · V_ℓ)
        #
        # Derived from ∂Ω/∂Δ_ℓ = 0  ↔  Δ_ℓ = g_ℓ · V_ℓ · F_ℓ_BZ (gap equation).
        # The term restores variational stationarity at the converged Δ_ℓ.
        #
        # V_ℓ > 0 : attractive pairing in channel ℓ → term is positive (costs energy
        #           to maintain Δ_ℓ against fluctuations; quasiparticle gain is in Ω_BdG).
        # V_ℓ ≤ 0 : repulsive / absent → no pairing in channel ℓ, Δ_ℓ = 0 by the
        #           gap equation, so the term is absent entirely.
        # V_ℓ = None (pre-cache): fall back to bare JT vertex so the SCF can start.
        _V_JT = self.p.g_JT**2 / max(self._K_bare, 1e-9)
        _, _, g_s, g_d = self.get_gutzwiller_factors(target_doping)

        condensation = 0.0
        _V_s = V_s if V_s is not None else _V_JT
        if _V_s > 0.0:
            condensation += abs(Delta_s)**2 / (g_s * _V_s)
        _V_d = V_d if V_d is not None else _V_JT
        if _V_d > 0.0:
            condensation += abs(Delta_d)**2 / (g_d * _V_d)
        return 0.5 * Omega_cell + elastic_energy + condensation
    
    def compute_cluster_free_energy(self, M: float, Q: float, mu: float, g_J: float, tx_bare: float, ty_bare: float, doping: float) -> Dict:
        H_sp_A = self.build_single_particle_hamiltonian(Q, mu)
        H_sp_B = self.build_single_particle_hamiltonian(Q, mu)

        J_eff = self.effective_superexchange(g_J, tx_bare, ty_bare, doping)
        # U_mf_stoner: the Stoner (on-site Hubbard) contribution to the boundary Weiss field.
        # J_eff = g_J · f_d · (2t²/Δ_CT²) · (...) is the Heisenberg part (two-particle vertex)
        # The BdG Weiss field also includes g_J · f_d · U_mf/2 (is the one-particle Weiss field, magnetic Hartree-Fock, Stoner part).
        abs_d_cl  = max(abs(doping), 1e-6)
        f_d_cl    = abs_d_cl / (abs_d_cl + self.p.doping_0)
        U_mf_stoner = g_J * f_d_cl * self.p.U_mf
        H_cluster = self.cluster_mf.build_cluster_hamiltonian(H_sp_A, H_sp_B, J_eff, M, self.p.eta, U_mf_stoner=U_mf_stoner)

        evals, evecs = eigh(H_cluster)

        if self.p.kT < 1e-8:
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

    def _scf_jacobi_kick(self, target_doping: float, initial_M: float, initial_Q: float) -> Dict:
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

        Returns:
            dict with M_kick, Q_kick, Delta_kick, mixing_kick, lambda_plus, regime, chi_tau
        """
        p = self.p
        abs_d = max(abs(target_doping), 1e-6)
        g_t   = (2.0 * abs_d) / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d) ** 2
        t_eff = g_t * p.t0
        N0 = 1.0 / (np.pi * max(t_eff, 1e-6))

        chi0_est = N0 / (1.0 + (p.U_mf / max(np.pi * t_eff, 1e-9))**2)
        U_pair = g_J * p.U
        stoner = 1.0 - U_pair * chi0_est
        if stoner <= 0.0:
            V_spin_est = 0.0   # QCP crossed: linear RPA invalid
        else:
            stoner_clamped = max(stoner, p.rpa_cutoff)
            V_spin_est = U_pair**2 * chi0_est / stoner_clamped**2   # Moriya-RPA linearised

        V_eff_bare = p.g_JT**2 / max(self._K_bare, 1e-9)
        V_pair = max(V_eff_bare + V_spin_est, V_eff_bare)

        g_Delta = np.sqrt(g_t)
        chi_tau_val = self._compute_chi_tau(initial_M, initial_Q, target_doping)['chi_tau']

        A = g_Delta * V_pair * N0
        B_raw = A * (p.g_JT**2 / (max(p.Delta_CF, 1e-9) * max(self._K_bare, 1e-9))) \
                    * (chi_tau_val / max(N0, 1e-12))
        B = B_raw / (1.0 + B_raw / max(A, 1e-9))
        C = (p.g_JT / max(self._K_bare, 1e-9)) * chi_tau_val

        discriminant = A**2 + 4.0 * B * C
        lambda_plus  = 0.5 * (A + np.sqrt(max(discriminant, 0.0)))

        if lambda_plus < 0.7:
            regime       = 'subcritical'
            Delta_kick   = max(initial_Q * p.g_JT * 0.5, p.kT)   # small seed
            M_kick       = initial_M
            Q_kick       = initial_Q
            mixing_kick  = p.mixing

        elif lambda_plus <= 1.4:
            regime       = 'critical'
            # Near λ₊≈1 the map is nearly neutral: use a thermal-scale Δ seed to avoid the trivial Δ=0 fixpoint while remaining in the physical basin.
            Delta_kick   = max(3.0 * p.kT, 0.5 * p.g_JT * abs(initial_Q))
            M_kick       = initial_M
            # Seed Q from the self-consistent JT equilibrium at this Δ_kick:
            Q_kick_est   = (p.g_JT / max(self._K_bare, 1e-9)) * (Delta_kick / max(p.Delta_CF, 1e-9)) * N0
            Q_kick       = float(np.clip(Q_kick_est, initial_Q, 0.1 * p.lambda_hop))
            mixing_kick  = min(p.mixing * 0.5, 0.02) # Reduce mixing to slow down the neutral mode
        else:
            regime       = 'supercritical'
            Delta_kick   = float(np.clip(
                2.0 * p.kT * np.exp(min(1.0 / max(lambda_plus - 1.0, 0.05), 10.0)),
                0.01, 0.3))
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
        1. Build 16×16 BdG Hamiltonian H(k; M, Q, Δ_s, Δ_d, μ) on the odd Simpson k-grid.
        2. Diagonalise → (E_k, ψ_k); compute observables: M_BdG, ⟨τ_x⟩, Pair_s, Pair_d.
        3. If SC+JT active (Δ>0, Q>0): inject anomalous orbital coherence ⟨τ_x⟩_anom
           into the Weiss field O_expectation, rebuild BdG cache.
        4. Periodically update K_eff = K_lattice + ∂²F_ex/∂Q² (exchange rigidity correction).
        5. Solve gap equations for (Δ_s_out, Δ_d_out) via RPA vertex + LM Newton step.
        6. Update cluster free energy (DMFT-like vertex renormalisation of J_eff).
        7. Newton step for M via ∂F/∂M and ∂²F/∂M² (LM-damped); blend with BdG fixpoint.
        8. Update Q via the adiabatic JT equilibrium: Q_out = −(g_JT/K_eff)·⟨τ_x⟩.
        9. Apply Anderson(5) acceleration to (M, Q); linear mixing to (Δ_s, Δ_d).
        10. Find μ to enforce density; compute F_BdG and F_cluster diagnostics.
        11. Adaptive mixing: halve α on divergence, reset Anderson history on Q sign flip.

        Converged when max(|ΔM|,|ΔQ|,|ΔΔ_s|,|ΔΔ_d|) < tol and |n−(1−δ)| < tol×10.
        Near SC critical point (0.8<λ_max<1.8): tolerance relaxed to 5×tol.

        Post-convergence diagnostics
        ----------------------------
        - 3×3 Hessian of F(M, Q, Δ) (finite-difference); confirms free-energy minimum.
        - Linearised gap equation: largest eigenvalue λ_max and gap symmetry (B₁g / A₁g).
        - Static χ₀(q_AFM) and Stoner denominator (AFM stability check).
        - χ_τ multipolar susceptibility and λ_JT = (g²/K)·χ_τ (SC-triggered JT strength).

        Returns: M, Q, Delta_s, Delta_d, chi_tau, Ut_ratio, density, mu, g_t, g_J,
                 F_bdg, F_cluster, tx, ty, J_eff, target_doping, chi0, rpa_factor,
                 afm_unstable, irrep_info, history, hessian, lambda_max, gap_symmetry,
                 lambda_plus, regime, K_eff_scf, converged.
        """
        ALPHA_HF = self.p.ALPHA_HF
        converged = False
        # K_eff is tracked as a LOCAL variable throughout the SCF loop.
        # All places that previously read self.p.K_lattice for K_eff now receive _K_eff_scf.
        _K_eff_scf: float = self._K_bare   # local; updated by exchange rigidity each ~5 iters
        self._cluster_j_renorm = 1.0
        _mu0_est: float
        if abs(target_doping) < 0.01:
            _mu0_est = 0.0
        elif target_doping > 0:
            _mu0_est = -2.0 * self.p.t0 * np.tanh(target_doping / 0.1)
        else:
            _mu0_est = 2.0 * self.p.t0 * np.tanh(abs(target_doping) / 0.1)
        _mu0_est += 0.5 * self.p.Delta_CF

        kick = self._scf_jacobi_kick(target_doping, initial_M, initial_Q)

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
            'chi0': [], 'rpa_factor': [], 'afm_unstable': [], 'selection_ratio': [],
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
        # QCP detection thresholds:
        #   det_cut: RPA matrix determinant — measures proximity to both AFM-Stoner and SC-JT simultaneous instability
        #   V_cut:   pairing vertex amplitude — V_s catches single-channel (pure spin or pure JT) near-divergence.
        _DET_CUT = 0.05
        _V_CUT   = 20.0
        _gap_symmetry = 'unknown'

        for iteration in range(self.p.max_iter):
            _iter_t0 = _time.time()

            g_t, g_J, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)
            tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
            tx, ty = g_t * tx_bare, g_t * ty_bare

            _vbdg_scf = self._get_vbdg()
            _bdg_ev_sc, _bdg_ec_sc = np.linalg.eigh(_vbdg_scf._build_H_stack(_vbdg_scf._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=_vbdg_scf._H_stack))
            self._scf_bdg_cache = (_bdg_ev_sc, _bdg_ec_sc)

            obs = self._get_vbdg().compute_observables_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J,
                _bdg_cache=(_bdg_ev_sc, _bdg_ec_sc))
            tau_x      = obs['Q']
            Pair_s_obs = obs['Pair_s']   # on-site pairing amplitude (channel s)
            Pair_d_obs = obs['Pair_d']   # inter-site pairing amplitude (channel d)
            M_bdg      = obs['M']        # BdG response: lattice magnetization
            Delta_eff  = abs(Delta_s) + abs(Delta_d)   # combined for irrep mixing weight

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
                f_d_oe       = abs_d_oe / (abs_d_oe + self.p.doping_0)
                sz_op_oe     = np.array([1.0, -1.0, self.p.eta, -self.p.eta])
                O_exp_diag   = g_J * f_d_oe * M * sz_op_oe         # (4,) diagonal part
                # off-diagonal τ_x component: mixes Γ₆↑↔Γ₇↑ and Γ₆↓↔Γ₇↓

                # τ_x couples only the Γ₆↑↔Γ₇↑ (index 0↔2) and Γ₆↓↔Γ₇↓ (index 1↔3)
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
                _rigidity = self.compute_JT_rigidity_from_exchange(M, Q, mu, g_J, target_doping)
                _K_eff_scf       = max(_rigidity['K_eff'], 1e-9)
                _K_eff_last_M    = M
                _K_eff_last_iter = iteration

            # Using the RPA-vertex gap equation result to find fixpoint
            #    The _bdg_cache here feeds ONLY into the anomalous Green function F_AA:
            #      Δ_out = g_Δ · V_s · F_AA(k; Δ, M, Q)
            #    NOT into _orbital_rpa_vertex which uses an independent E_k_cache_normal.
            Delta_s_out, Delta_d_out, _vertex_cache = self._get_vbdg().compute_gap_eq_vectorized(
                M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, g_Delta_s, g_Delta_d,
                _bdg_cache=(_bdg_ev_sc, _bdg_ec_sc),
                _vertex_cache=_vertex_cache)

            # Cluster role: parameter renormalisation via Migdal-Galitski theorem (DMFT-like).
            cluster_result_pre = self.compute_cluster_free_energy(M, Q, mu, g_J, tx_bare, ty_bare, target_doping)

            # Extract renormalised J_eff from cluster and update J_alpha_beta_Q scale. The ratio cluster_J / bare_J measures local vertex corrections (double-occupancy, spin-fluctuation screening) beyond Gutzwiller.
            J_eff_cluster  = cluster_result_pre['J_eff']        # already contains g_J
            J_eff_bare     = self.effective_superexchange(g_J, tx_bare, ty_bare, target_doping)
            # Renormalisation factor: how much the local cluster dresses J
            if abs(J_eff_bare) > 1e-10:
                _j_renorm = float(np.clip(J_eff_cluster / J_eff_bare, 0.5, 2.0))
            else:
                _j_renorm = 1.0
            # Store for use in build_local_hamiltonian_for_bdg on next BdG build.
            # We encode the renorm into a transient attribute read by J_alpha_beta_Q.
            self._cluster_j_renorm = _j_renorm

            dF_dM_0, d2F_dM2 = self.compute_dF_dM_and_d2F(M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J)
            self._scf_bdg_cache = None   # cache consumed; clear to prevent stale reuse
            # LM denominator: d2F_dM2 + mu_LM (positive shift guarantees a positive denominator while preserving sign)
            # When d2F < 0 (saddle/instability), abs() would flip the Newton direction and push M away from the minimum, blocking convergence.
            M_newton = M - dF_dM_0 / (d2F_dM2 + self.p.mu_LM)
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
                _vertex_cache = None         # Q sign flip → FS topology may change
                self._scf_bdg_cache = None   # topology change → stale SC cache unsafe

            Delta_s_mixed = self._mix(Delta_s, Delta_s_out, alpha=_alpha)
            Delta_d_mixed = self._mix(Delta_d, Delta_d_out, alpha=_alpha)

            tx_mixed_bare, ty_mixed_bare = self.effective_hopping_anisotropic(Q_mixed)
            tx_mixed, ty_mixed = g_t * tx_mixed_bare, g_t * ty_mixed_bare
            
            mu_new, _mu_bdg_cache = self._find_mu_for_density(
                M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping,
                tx_mixed, ty_mixed, mu_guess=mu, g_J=g_J
            )

            # _find_mu_for_density already computed (ev,ec) at the converged μ_new.
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
                n_kspace_new = self._compute_density_at_mu(
                    mu_new, M_mixed, Q_mixed, Delta_s_mixed, Delta_d_mixed, target_doping,
                    tx_mixed, ty_mixed, g_J
                )

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
            _mu_shift = abs(mu_new - mu)   # tracked separately for diagnostics

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
            _near_critical = (_det_now < _DET_CUT) or (_Vs_now > _V_CUT)
            _tol_use = self.p.tol * (5.0 if _near_critical else 1.0)
            if _near_critical:
                # Cap alpha near QCP: large steps overshoot the near-singular gap vertex
                _alpha = min(_alpha, self.p.mixing * 0.6)
            if _vertex_cache is not None:
                _vertex_cache['near_critical'] = _near_critical

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

        # ── Post-loop diagnostic: solve linearised gap equation ONCE ──────────────
        # This gives λ_max (normal-state instability) cheaply after the SCF has
        # already converged.  Removed from the inner loop: Anderson acceleration
        # handles critical slowing without needing this expensive call per-iteration.
        _lin: Dict = self.solve_linearized_gap_equation(
            M, Q, Delta_s, Delta_d, target_doping, mu,
            tx_mixed, ty_mixed, g_J)
        _lambda_max   = _lin['lambda_max']
        _gap_symmetry = _lin['gap_symmetry']

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
                _S_z = np.array([[1.0, 0.0], [0.0, -1.0]])

                # Normal-state χ_SQ: Δ=0 → block-diagonal Hamiltonian → should be ≈0
                chi0_normal_q0 = self.compute_chi0_tensor(
                    np.zeros(2), M, Q, 0.0+0j, 0.0+0j,
                    target_doping, mu, tx_mixed, ty_mixed, g_J)
                _chi_SQ_normal = float(np.real(np.trace(_S_z @ chi0_normal_q0[0:2, 2:4])))

                # SC-state χ_SQ: Δ≠0 → condensate mixes Γ₆↔Γ₇ → should be non-zero
                chi0_sc_q0 = self.compute_chi0_tensor(
                    np.zeros(2), M, Q, complex(Delta_s), complex(Delta_d),
                    target_doping, mu, tx_mixed, ty_mixed, g_J)
                _chi_SQ_sc = float(np.real(np.trace(_S_z @ chi0_sc_q0[0:2, 2:4])))

                _chi_SQ_ratio = abs(_chi_SQ_sc) / max(abs(_chi_SQ_normal), 1e-15)
                _channel_opened = abs(_chi_SQ_sc) > 1e-8

                with _log_lock:
                    print(
                        f"[{_tag_chi}] SC-triggered χ_SQ channel:"
                        f"  χ_SQ(Δ=0)={_chi_SQ_normal:.6e}"
                        f"  χ_SQ(Δ≠0)={_chi_SQ_sc:.6e}"
                        f"  ratio={_chi_SQ_ratio:.2f}×"
                        f"  {'✓ CHANNEL OPEN — SC-triggered JT active' if _channel_opened else '✗ channel still closed'}",
                        flush=True
                    )
            except Exception as _chi_sq_err:
                _scf_log(_tag_chi, f"χ_SQ diagnostic failed: {_chi_sq_err}")

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
            'rpa_factor': rpa_factor,
            'afm_unstable': afm_unstable,
            'irrep_info': irrep_info,
            'history': history,
            'hessian': hessian_result,
            'lambda_max': _lambda_max,
            'gap_symmetry': _gap_symmetry,
            'lambda_plus': kick['lambda_plus'],
            'regime': kick['regime'],
            'K_eff_scf': _K_eff_scf,
            'converged': converged,     # use flag set by break, not recomputed expression
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
            scale[1] = 1.0 / max(self.p.lambda_hop, 1e-8)

        R_scaled  = R  * scale
        dR_scaled = dR * scale
        r_scaled  = r_last * scale

        # Solve regularised normal equations
        # min || r_last - dR theta ||² + beta ||theta||²
        A = dR_scaled @ dR_scaled.T
        b = dR_scaled @ r_scaled

        # Adaptive Tikhonov regularisation
        diag_max = max(float(np.max(np.abs(np.diag(A)))), 1e-30)
        beta = 1e-8 * diag_max
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

        if norm_step_opt > 2.5 * max(norm_step_simple, 1e-12):
            # Too aggressive → shrink
            shrink = (2.5 * norm_step_simple) / (norm_step_opt + 1e-12)
            x_opt = x_last + shrink * step_opt

        # Blended final step
        # Conservative near small alpha, more Anderson near full mixing.
        w = float(np.clip(alpha / max(self.p.mixing, 1e-8), 0.4, 0.9))
        x_new = w * x_opt + (1.0 - w) * x_simple

        if not np.all(np.isfinite(x_new)):
            return x_simple
        return x_new

    def compute_dF_dM_and_d2F(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float) -> Tuple[float, float]:
        t_sq_avg    = 0.5 * (tx**2 + ty**2)   # renormalized: consistent with BdG spectrum
        abs_d       = max(abs(target_doping), 1e-6)
        f_d         = abs_d / (abs_d + self.p.doping_0)
        h_prefactor = g_J * f_d * (self.p.U_mf / 2.0 + self.p.Z * 2.0 * t_sq_avg / self.p.U) / 2.0

        sz_orb = np.array([1.0, -1.0, self.p.eta, -self.p.eta]) * h_prefactor
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
        safe  = np.abs(dE_nm) > 1e-8
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
        eps_Q = max(1e-5, abs(Q)     * 1e-3 * self.p.lambda_hop)
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

    def _compute_afm2band_susceptibilities(self, target_doping: float, M: float, Q: float, Delta_s: complex, Delta_d: complex, mu_chi: float = None) -> dict:
        """
        Shared kernel for the analytic 2-band AFM susceptibility tensor.

        Computes χ_ss, χ_dd, χ_sd, χ_sQ, χ_dQ on the full k-grid with proper
        BZ weights, and χ_QQ via the Kubo formula (_chi_QQ_matrix_elements).
        Called by both compute_G_instability (normal-state limit, Q=Δ=0) and
        the 3×3 Hessian in solve_linearized_gap_equation (finite Δ, Q, μ).
        """
        p     = self.p
        abs_d = max(abs(target_doping), 1e-6)
        g_t   = (2.0 * abs_d) / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d) ** 2
        kT    = max(p.kT, 1e-8)

        tx_bare, ty_bare = self.effective_hopping_anisotropic(Q)
        tx_eff = g_t * tx_bare
        ty_eff = g_t * ty_bare
        t_eff  = np.sqrt(0.5 * (tx_eff**2 + ty_eff**2))   # kept for BCS Tc estimate only

        f_d   = abs_d / (abs_d + p.doping_0)
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
            se = np.where(np.abs(E) > 1e-8, E, 1e-8)
            return np.tanh(a) / (2.0 * se)

        def _mdf(E):
            f_E = 1.0 / (1.0 + np.exp(np.clip(E / kT, -100, 100)))
            return f_E * (1.0 - f_E) / kT

        w_k  = self.k_weights
        pk   = _th2E(Ep) + _th2E(Em)
        proj = xi_diff / np.where(sq > 1e-9, sq, 1e-9)
        mix  = _th2E(Ep) * proj - _th2E(Em) * proj   # orbital-mixing kernel (proxy for chi_DQ at Δ≠0)

        phi_s = np.ones_like(kx)
        phi_d = np.cos(kx * p.a) - np.cos(ky * p.a)

        chi_DD_s  = float(np.dot(w_k, pk * phi_s**2))
        chi_DD_d  = float(np.dot(w_k, pk * phi_d**2))
        chi_DD_sd  = float(np.dot(w_k, pk * phi_s * phi_d))

        # SC–JT cross susceptibility χ_DQ (SC response to JT distortion Q).
        # Only the orbital-mixing vertex τ_x (Γ6↔Γ7) contributes; hopping anisotropy is already included in the dispersion used for χ_DD.
        # In the AFM-folded 2-band model ξ_avg = -μ = const ⇒ proj(k)·Δ_k·φ_c(k) is odd under k→k+Q,
        # so the BZ integral vanishes and the analytic 2-band Gorkov formula gives χ_DQ = 0.
        #
        # Physically χ_DQ lives in off-diagonal Nambu sectors (captured by full 16×16 BdG).
        # Enforce: Δ=0 → χ_DQ=0; Δ≠0 → finite but not computable in this reduced basis.
        Delta_mag = np.sqrt(abs(Delta_s)**2 + abs(Delta_d)**2)
        if Delta_mag > 1e-8:
            chi_DQ_s = p.g_JT * float(np.dot(w_k, mix * phi_s))
            chi_dQ_d = p.g_JT * float(np.dot(w_k, mix * phi_d))
        else:
            # Normal state: chi_DQ = 0 enforced (selection rule + the analytic
            # 2-band formula vanishes identically — two independent reasons).
            chi_DQ_s = 0.0
            chi_dQ_d = 0.0
        N_eff   = float(np.dot(w_k, _mdf(Ep) + _mdf(Em)))

        mu_use = mu_n if mu_chi is None else mu_chi
        chi_QQ = self._chi_QQ_matrix_elements(M, Q, target_doping, Delta_s, Delta_d, mu_use)
        return {
            'chi_DD_s': chi_DD_s, 'chi_DD_d': chi_DD_d, 'chi_DD_sd': chi_DD_sd,
            'chi_DQ_s': chi_DQ_s, 'chi_DQ_d': chi_dQ_d, 'chi_QQ': chi_QQ,
            'N_eff': N_eff, 'E_plus': Ep, 'E_minus': Em,
            'h_afm': float(h_afm), 'mu_n': float(mu_n), 't_eff': float(t_eff),
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
                Ds = res.get('Delta_s', 0.0)
                Dd = res.get('Delta_d', 0.0)
                D  = (Ds**2 + Dd**2) ** 0.5

                if use_free_energy and D > Delta_tol:
                    res_normal = s.solve_self_consistent(
                        target_doping = doping,
                        initial_M     = self._estimate_M0(doping, sc_result),
                        initial_Q     = 1e-6,
                        initial_Delta = 1e-8,
                        verbose       = False,
                    )
                    if res_normal.get("F_bdg", 0.0) < res.get("F_bdg", 0.0):
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

    def _background_at_T(self, T: float, doping: float, sc_result: dict) -> tuple:
        """
        Return (M, Q, Delta_s, Delta_d, mu, tx, ty, g_J) at temperature T using a warm-started normal-state SCF solve.
        Used as input to solve_linearized_gap_equation. The SC gap is intentionally set to zero
        so λ_max(T) measures the linearised pairing instability, not the already-condensed state.
        """
        s = copy.copy(self)
        s.p = copy.copy(self.p)
        s.p.kT = T
        s._K_bare = self._K_bare
        s._reset_transient_state()
        try:
            res = s.solve_self_consistent(
                target_doping  = doping,
                initial_M      = self._estimate_M0(doping, sc_result),
                initial_Q      = 1e-6,
                initial_Delta  = 1e-8,   # normal-state background: no SC seed
                verbose        = False,
            )
            M      = res.get('M',  0.1)
            Q      = res.get('Q',  0.0)
            mu     = res.get('mu', 0.0)
            tx     = res.get('tx', self.p.t0)
            ty     = res.get('ty', self.p.t0)
            g_J    = res.get('g_J', 1.0)
            Delta_s = complex(res.get('Delta_s', 0.0))
            Delta_d = complex(res.get('Delta_d', 0.0))
        except Exception:
            g_t = (2.0 * doping) / (1.0 + doping)
            g_J = 4.0 / (1.0 + doping)**2
            M, Q, Delta_s, Delta_d = 0.1, 0.0, 0.0+0j, 0.0+0j
            mu, tx, ty = 0.0, g_t * self.p.t0, g_t * self.p.t0
        return M, Q, Delta_s, Delta_d, mu, tx, ty, g_J

    def compute_lambda_vs_T(self, doping: float, sc_result: dict, T_points: np.ndarray = None) -> Dict:
        """
        Compute the linearised gap eigenvalue λ_max(T) across a temperature range.

        λ_max(T) is the largest eigenvalue of the linearised gap equation kernel at each T.
        It measures the strength of the pairing instability: λ_max(Tc) = 1 by definition.

        Diagnostics available from the curve:
          • Slope |dλ/dT|_Tc  — steeper → stronger coupling, less fluctuation-dominated
          • Strong-coupling signal: λ_max(T) deviates strongly from BCS tanh(T/Tc) form
          • Non-monotonic λ_max(T): indicates competing orders or fluctuation enhancement
          • Asymptotic λ_max(T→0): should saturate; if still rising → not fully converged k-grid

        Parameters
        ----------
        doping    : carrier doping δ
        sc_result : converged SCF result dict (used to warm-start the normal-state solve at each T)
        T_points  : temperature array (eV). Default: 20 log-spaced points from kT/4 to 4·kT.

        Returns
        -------
        dict with keys:
          'T'           : np.ndarray, temperature points (eV)
          'lambda_max'  : np.ndarray, λ_max at each T
          'gap_symmetry': list[str], dominant gap symmetry at each T
          'Tc_lambda'   : float, T where λ_max crosses 1 (linear interpolation), 0 if not found
          'slope_at_Tc' : float, dλ/dT at the crossing (eV⁻¹); large → strong coupling
        """
        kT0 = max(self.p.kT, 1e-4)
        if T_points is None:
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
                bg  = s_T._background_at_T(T, doping, sc_result)
                M, Q, Ds, Dd, mu, tx, ty, g_J = bg
                lin = s_T.solve_linearized_gap_equation(M, Q, Ds, Dd, doping, mu, tx, ty, g_J)
                lam_arr[i] = float(lin.get('lambda_max', 0.0))
                sym_list.append(lin.get('gap_symmetry', 'unknown'))
            except Exception:
                lam_arr[i] = 0.0
                sym_list.append('error')

        # Find Tc: last T where λ_max crosses 1 from above (linear interpolation)
        Tc_lambda  = 0.0
        slope_at_Tc = 0.0
        for i in range(len(T_points) - 1):
            l0, l1 = lam_arr[i], lam_arr[i + 1]
            t0, t1 = T_points[i], T_points[i + 1]
            if l0 >= 1.0 >= l1 and abs(l1 - l0) > 1e-10:
                frac = (1.0 - l0) / (l1 - l0)
                Tc_lambda   = float(t0 + frac * (t1 - t0))
                slope_at_Tc = float((l1 - l0) / (t1 - t0))
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
        
        Parameters
        ----------
        doping    : carrier doping δ
        sc_result : converged T=0 (or base-T) SCF result dict

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
        Tc     = float(tc_res.get('Tc', 0.0))
        if Tc < 1e-8:
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
    
    def _build_G3_matrix(self, chi: dict, gVs: float, gVd: float, K_eff: float) -> tuple:
        """
        Assemble the 3×3 SC–JT instability matrix from susceptibilities and couplings.

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

        Returns (G3, eigs3, lam_min, instab_dir).
        """
        G3 = np.zeros((3, 3))
        G3[0, 0] = 1.0 - gVs * chi['chi_DD_s']
        G3[1, 1] = 1.0 - gVd * chi['chi_DD_d']
        G3[2, 2] = 1.0 - chi['chi_QQ'] / max(K_eff, 1e-9)
        G3[0, 1] = G3[1, 0] = -np.sqrt(max(gVs * gVd, 0.0)) * chi['chi_DD_sd']
        # Off-diagonal SC–JT coupling: c_c = √(g_Δc·V_c / K_eff)
        c_s = np.sqrt(max(gVs / max(K_eff, 1e-9), 0.0))
        c_d = np.sqrt(max(gVd / max(K_eff, 1e-9), 0.0))
        G3[0, 2] = G3[2, 0] = -c_s * chi['chi_DQ_s']
        G3[1, 2] = G3[2, 1] = -c_d * chi['chi_DQ_d']

        eigs3   = np.linalg.eigvalsh(G3)
        lam_min = float(eigs3[0])

        if lam_min < 0.5:
            evec_min = np.linalg.eigh(G3)[1][:, 0]
            nm = np.abs(evec_min)
            # Determine the dominant component
            ws, wd, wq = nm
            sc_weight = ws + wd   # total SC (pairing) weight in the soft mode
            if lam_min < 0:
                # Genuine instability: classify by eigenvector composition
                if wq > 0.6 and sc_weight < 0.3:
                    instab = 'pure JT (spontaneous risk)'
                elif wq > 0.4 and sc_weight > 0.3:
                    instab = 'SC-triggered JT'    # the desired mechanism
                elif ws > 0.6:
                    instab = 's pairing'
                elif wd > 0.6:
                    instab = 'd pairing'
                else:
                    instab = 'mixed SC+JT'
            else:
                # Pre-critical approach zone (0 ≤ lam_min < 0.5): label by dominant direction
                mc = int(np.argmax(nm))
                instab = f"near-critical ({'Δ_s' if mc==0 else 'Δ_d' if mc==1 else 'Q'}-dominant)"
        else:
            instab = 'stable'
        return G3, eigs3, lam_min, instab

    def compute_G_instability(self, target_doping: float, M: float) -> dict:
        """
        Normal-state (Δ=0) collective instability matrix and derived diagnostics.

        Role and limitations
        --------------------
        λ_eff = N_eff · V_eff is a collective instability measure, NOT a pairing
        eigenvalue and NOT a gap.  It counts how close the system is to the QCP
        of the dominant channel (SC or JT) from the normal-state side.

        Correct uses:
          - SCF gatekeeper: λ_eff < λ_min_threshold → pairing mechanism too weak,
            skip the expensive SCF (saves wall time in BO scout).
          - Soft prior in BO: w = exp(−(λ_eff − λ_target)² / σ²) biases sampling
            toward the QCP without hard-rejecting away from it.
          - G22 > 0 hard gate: spontaneous JT (G3[2,2] ≤ 0 at Δ=0) is reliably
            detected here and excluded from GP entirely.

        Tc_estimate uses λ_eff via BCS-McMillan as a cheap diagnostic only.
        The authoritative Tc comes from the SCF converged gap and Hessian.
        """
        p     = self.p
        abs_d = max(abs(target_doping), 1e-6)
        g_J   = 4.0 / (1.0 + abs_d) ** 2

        rigidity = self.compute_JT_rigidity_from_exchange(M, 0.0, 0.0, g_J, target_doping)
        # K_eff (exchange-corrected normal-state stiffness) is used consistently throughout the entire G3 matrix
        # both for the SC-channel couplings (gVs, gVd via V_base) and for the JT diagonal element G3[2,2] = 1 − χ_QQ/K_eff.
        # This guarantees that the Schur complement and the off-diagonal SC–JT coupling constants are built from a single spring constant.
        K_eff_here  = max(rigidity['K_eff'], 1e-9)

        _, _, g_Delta_s, g_Delta_d = self.get_gutzwiller_factors(target_doping)
        # Use K_eff_here for the gap-equation pairing vertex so that the normal-state JT rigidity is accounted for in the SC instability threshold.
        V_base = p.g_JT**2 / K_eff_here
        gVs = g_Delta_s * V_base
        gVd = g_Delta_d * V_base

        chi = self._compute_afm2band_susceptibilities(target_doping, M, 0.0, 0.0, 0.0)
        t_eff = chi['t_eff']
        N_eff = chi['N_eff']

        # G3[2,2] uses K_eff (exchange-corrected) so it correctly reflects the JT stability boundary.
        G3, eigs3, lam_min, instab_dir = self._build_G3_matrix(chi, gVs, gVd, K_eff_here)
        det_G = float(np.linalg.det(G3))

        # If G11_s < G11_d, a large off-diagonal χ_sQ/χ_dQ can rotate the dominant instability direction away from pure Δ_s or Δ_d.
        idx_min = int(np.argmin(eigs3))
        evecs3 = np.linalg.eigh(G3)[1]
        evec_min = evecs3[:, idx_min]

        # Schur-complement effective coupling for the dominant SC channel
        G11_s, G11_d = G3[0, 0], G3[1, 1]
        G22          = G3[2, 2]

        # Dominant channel from the *eigenvector*, not from diagonal alone.
        weights = np.abs(evec_min)
        ws, wd, wq = weights
        if wd > ws:
            dominant   = 'd'
            G11, G12   = G11_d, G3[1, 2]
            chi_DD_dom = chi['chi_DD_d']
            chi_DQ_dom = chi['chi_DQ_d']
            V_dom      = gVd
        else:
            dominant   = 's'
            G11, G12   = G11_s, G3[0, 2]
            chi_DD_dom = chi['chi_DD_s']
            chi_DQ_dom = chi['chi_DQ_s']
            V_dom      = gVs

        if wq > ws and wq > wd:
            dominant   = 'JT'
        
        # V_eff Schur complement: use K_eff_here consistently (not a mix of K_bare and K_eff).
        if dominant != 'JT' and G22 > 1e-8:
            V_eff = V_dom + (V_dom / K_eff_here * chi_DQ_dom**2) / G22
        else:
            V_eff = V_dom   # spontaneous-JT regime: no SC-triggered boost
        lambda_eff = N_eff * V_eff
        Tc_est  = float(1.13 * t_eff * np.exp(-1.0 / lambda_eff)) if lambda_eff > 1e-3 else 0.0
        d2F_Q_normal = K_eff_here - chi['chi_QQ']   # normal-state Q-curvature; adiabatic approx is exact at Δ=0

        # SC-triggered JT criterion — use the full Hessian ∂²F/∂Q²|_{Δ≠0} rather than
        # the adiabatic approximation  K − χ_QQ^SC (is biased in the SC state)
        #
        #     • Δ(Q) shifts because t(Q) changes the pairing kernel
        #     • M(Q) readjusts because the Weiss field depends on t(Q)
        #     • both effects add implicit Q-dependence that can make ∂²F/∂Q² *larger*
        #       than K − χ_QQ^SC, or even positive while χ_QQ^SC > χ_QQ^normal.
        # Probe SC gap for JT–SC coupling test
        _Delta_probe = max(3.0 * p.kT, 1e-3)
        _Delta_probe_c = complex(0.5 * _Delta_probe)

        try:
            _hess_sc  = self.compute_d2F_dQ2_at_Delta(
                target_doping, M,
                Delta_s=_Delta_probe_c, Delta_d=_Delta_probe_c)
            d2F_Q_sc      = _hess_sc['d2F_dQ2']          # full Hessian Q-Q element at Δ≠0
            chi_QQ_sc     = _hess_sc['chi_QQ_finite_D']  # adiabatic back-estimate (diagnostics only)
            sc_triggered_jt = (d2F_Q_normal > 0.0) and (_hess_sc['sc_triggers_JT'])
        except Exception as _hess_err:
            _scf_log("G-INST", f"Warning: Hessian SC-JT test failed ({_hess_err}); "
                     "falling back to linear-χ approximation.")
            try:
                chi_QQ_sc = self._chi_QQ_matrix_elements(M, 0.0, target_doping, _Delta_probe_c, _Delta_probe_c, mu=0.0)
            except Exception as e:
                _scf_log("G-INST", f"Warning: χ_QQ(SC) failed, using normal-state value: {e}")
                chi_QQ_sc = chi['chi_QQ']

            d2F_Q_sc = K_eff_here - chi_QQ_sc
            sc_triggered_jt = (d2F_Q_normal > 0.0) and (d2F_Q_sc < 0.0)

        return {
            'chi_DD_dom':      chi_DD_dom,
            'chi_DD_s':        chi['chi_DD_s'],
            'chi_DD_d':        chi['chi_DD_d'],
            'chi_DD_sd':       chi['chi_DD_sd'],
            'chi_QQ':          chi['chi_QQ'],
            'chi_QQ_sc':       chi_QQ_sc,                   # χ_QQ in SC state (Δ≠0)
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
            'd2F_ex_dQ2':      float(rigidity['d2F_ex_dQ2']),
            'comm_norm':       float(rigidity['comm_norm']),
            'blocking_ratio':  float(rigidity['blocking_ratio']),
            'K_eff':           float(rigidity['K_eff']),
            'd2F_Q_normal':    float(d2F_Q_normal),        # ∂²F/∂Q²|_{Δ=0} (adiabatic, exact at Δ=0)
            'd2F_Q_sc':        float(d2F_Q_sc),            # ∂²F/∂Q²|_{Δ≠0} full Hessian H[Q,Q]
            'sc_triggered_jt': sc_triggered_jt,            # True = genuine SC-triggered JT (Hessian-based)
            'G11': G11, 'G22': G22, 'G12': G12, 'K_spont_blocked': G22 > 0.0,
        }

    def compute_d2F_dQ2_at_Delta(self, target_doping: float, M: float, Delta_s: complex, Delta_d: complex, mu: float = None) -> Dict:
        """
        Second derivative ∂²F/∂Q²|_{Δ} — the Q-curvature at given SC gap.

        Thin wrapper around compute_hessian(): extracts H[1,1] (the Q-Q element
        of the full 3×3 Hessian in (M, Q, Δ) space) and returns it together with
        derived diagnostics.

        This is the key quantity for the two-level SC-triggered JT test:
          ∂²F/∂Q²|_{Δ=0}  > 0  →  normal state Q-stable  (JT symmetry-blocked)
          ∂²F/∂Q²|_{Δ≠0}  < 0  →  SC state Q-unstable    (JT triggered by SC)

        """
        g_t, g_J, _, _ = self.get_gutzwiller_factors(target_doping)

        if mu is None:
            abs_d = max(abs(target_doping), 1e-6)
            mu = -2.0 * g_t * self.p.t0 * (1.0 - 2.0 * abs_d)

        # Determine channel fraction from the input amplitudes
        _Delta_s_amp = float(np.abs(Delta_s))
        _Delta_d_amp = float(np.abs(Delta_d))
        _Delta_total = _Delta_s_amp + _Delta_d_amp
        Delta_real   = _Delta_total   # scalar total gap for compute_hessian
        _Delta_s_frac = (_Delta_s_amp / _Delta_total) if _Delta_total > 1e-10 else 0.5

        hess = self.compute_hessian(
            M, Q=0.0, Delta=Delta_real,
            target_doping=target_doping,
            mu=mu, g_t=g_t, g_J=g_J,
            Delta_s_frac=_Delta_s_frac)

        d2F = float(hess['H'][1, 1])   # Q-Q element of the full Hessian
        K_eff = max(self._K_bare, 1e-9)   # bare phonon stiffness for the adiabatic estimate
        return {
            'd2F_dQ2':         d2F,
            'chi_QQ_finite_D': K_eff - d2F,   # adiabatic estimate: K − ∂²F/∂Q² = χ_QQ
            'stable_in_SC':    d2F > 0.0,
            'sc_triggers_JT':  d2F < 0.0,
            'K_eff':           K_eff,
            'hessian':         hess,
        }

class VectorizedBdG:
    def __init__(self, solver: 'RMFT_Solver'):
        self.solver    = solver
        self._kpts     = solver.k_points        # (N_k_odd, 2)  — SCF / Simpson grid
        self._kpts_ev  = solver.k_points_even   # (N_k_even, 2) — chi0 / commensurate grid
        self._N_k      = solver.N_k
        self._N_k_ev   = solver.N_k_even
        self._H_stack    = np.zeros((self._N_k,    16, 16), dtype=complex)  # odd SCF grid
        self._H_stack_ev = np.zeros((self._N_k_ev, 16, 16), dtype=complex)  # even chi0 grid

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

        a = self.solver.p.a

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
        gamma_k = -2.0 * (tx * np.cos(kx * a) + ty * np.cos(ky * a))

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
            phi_d_k = np.cos(kx * a) - np.cos(ky * a)

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

    def compute_observables_vectorized(self, M: float, Q: float, Delta_s: complex, Delta_d: complex, target_doping: float, mu: float, tx: float, ty: float, g_J: float, _bdg_cache: tuple = None) -> Dict:
        """ Vectorised observables: M, Q (τ_x), density, Pair_s, Pair_d. """
        # optional (ev, ec) tuple from a previous _build_H_stack+eigh call with the same parameters to avoid a redundant diagonalisation.
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = np.linalg.eigh(self._build_H_stack(self._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=self._H_stack))
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

        Two susceptibilities with intentionally different states:
          chi0            : normal-state (Δ=0) — avoids double-counting F_AA/F_AB.
          chi_QQ_normal_v : normal-state (Δ=0) — used in _orbital_rpa_vertex to avoid
                            conflating the pairing kernel with condensate-driven JT feedback.
                            The SC-state χ_QQ is used ONLY in the lattice stability diagnostic
                            (compute_G_instability / compute_d2F_dQ2_at_Delta), NOT here.

        F_AA, F_AB: full-BZ anomalous amplitudes (∝ u·v·tanh(E/2kT)), /4.0 for 16-dim BdG space.
        """
        p   = self.solver.p
        slv = self.solver

        # --- BdG amplitudes on the full k-grid ---
        if _bdg_cache is not None:
            ev, ec = _bdg_cache
        else:
            ev, ec = np.linalg.eigh(self._build_H_stack(self._kpts, M, Q, Delta_s, Delta_d, target_doping, mu, tx, ty, g_J, out=self._H_stack))

        arg = np.clip(ev / p.kT, -100, 100)
        f12 = 1.0 - 2.0 / (1.0 + np.exp(arg))   # tanh(E/2kT); (N_k, 16)

        uA = ec[:, 0:4,  :];  uB = ec[:, 4:8,  :]
        vA = ec[:, 8:12, :];  vB = ec[:, 12:16, :]

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
        # k_weights are Simpson-normalised (Sigma w_k = 1). /4.0 corrects for the 16-dim BdG space: A/B sublattice x particle-hole doubling.
        #
        # F_AA_BZ: on-site s-channel anomalous amplitude (phi_s = 1, no k-factor needed).
        # F_AB_BZ: inter-site d-channel anomalous amplitude.
        #   IMPORTANT: phi_d_full is NOT applied here. The d-wave symmetry projection
        #   is handled entirely on the VERTEX side (V_d_scalar = phi_d @ V_mat @ phi_d / phi2).
        F_AA_BZ = float(np.real(np.dot(slv.k_weights, pair_s_k))) / 4.0
        F_AB_BZ = float(np.real(np.dot(slv.k_weights, pair_d_k))) / 4.0

        # --- Fermi-surface sampling — used ONLY for the q-dependent vertex V(k-k') ---
        near_fs = np.any(np.abs(ev) < 3.0 * p.kT, axis=1)
        fs_idx  = np.where(near_fs)[0]
        if len(fs_idx) == 0:
            fs_idx = np.arange(min(32, slv.N_k))
        fs_idx  = fs_idx[:32]
        fs_pts  = slv.k_points[fs_idx]
        N_fs    = len(fs_pts)
        phi_d   = np.cos(fs_pts[:, 0] * p.a) - np.cos(fs_pts[:, 1] * p.a)

        # --- Vertex cache invalidation ---
        _cache_M     = _vertex_cache.get('M',     float('nan')) if _vertex_cache else float('nan')
        _cache_Delta = _vertex_cache.get('Delta', float('nan')) if _vertex_cache else float('nan')
        _cache_fs    = _vertex_cache.get('fs_idx', None)        if _vertex_cache else None
        _cache_chi_normal = _vertex_cache.get('chi_QQ_from_normal', False) if _vertex_cache else False
        Delta_eff    = abs(Delta_s) + abs(Delta_d)
        _delta_rel   = abs(Delta_eff - _cache_Delta) / max(abs(_cache_Delta), 1e-6) if _vertex_cache else float('inf')

        _vertex_stale = (
            _vertex_cache is None
            or not _cache_chi_normal                  # cache built with wrong Δ≠0 chi_QQ → must rebuild
            or abs(M - _cache_M) > 0.03
            or abs(Delta_eff - _cache_Delta) > 0.008
            or _delta_rel > 0.15
            or _cache_fs is None
            or len(_cache_fs) != len(fs_idx)
            or not np.array_equal(_cache_fs, fs_idx)
        )

        if _vertex_stale:
            I4   = np.eye(4, dtype=complex)
            V_JT = p.g_JT**2 / max(slv._K_bare, 1e-9)   # [eV]

            tx_bare_v, ty_bare_v = slv.effective_hopping_anisotropic(Q)
            J_eff_v = slv.effective_superexchange(g_J, tx_bare_v, ty_bare_v, target_doping)

            # chi0 from NORMAL-STATE (Delta=0): avoids double-counting F_AA / F_AB.
            # chi_QQ_normal (Delta=0): used in the pairing vertex (_orbital_rpa_vertex).
            chi_QQ_normal_v = slv._chi_QQ_matrix_elements(M, Q, target_doping, 0.0+0j, 0.0+0j, mu)
            
            # The SC-state χ_QQ (Δ≠0) captures the condensate-driven JT feedback and is used ONLY in the lattice stability diagnostic (compute_G_instability) compute_d2F_dQ2_at_Delta)
            E_k_cache_normal = np.linalg.eigh(self._build_H_stack(
                self._kpts_ev, M, Q, 0.0+0j, 0.0+0j,
                target_doping, mu, tx, ty, g_J, out=self._H_stack_ev))

            # s-channel: q=0 vertex (normal-state chi0 only)
            chi0_q0    = slv.compute_chi0_tensor(
                np.zeros(2), M, Q, 0.0+0j, 0.0+0j,
                target_doping, mu, tx, ty, g_J,
                _E_k_cache=E_k_cache_normal)
            V_s_scalar, _det_q0 = slv._orbital_rpa_vertex(chi0_q0, J_eff_v, V_JT, chi_QQ_normal_v, p.rpa_cutoff, _return_det=True)

            # d-channel: q-dependent vertex (only if Delta_d has nucleated)
            if abs(Delta_d) > 1e-4:
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
                    chi0_mat   = slv.compute_chi0_tensor(
                        q_u, M, Q, 0.0+0j, 0.0+0j,
                        target_doping, mu, tx, ty, g_J,
                        _E_k_cache=E_k_cache_normal)
                    V_rpa[ui]  = slv._orbital_rpa_vertex(chi0_mat, J_eff_v, V_JT, chi_QQ_normal_v, p.rpa_cutoff)

                V_mat = np.zeros((N_fs, N_fs))
                for fi, (i, j) in enumerate(ij_list):
                    v = V_rpa[inv_idx[fi]]
                    V_mat[i, j] = v;  V_mat[j, i] = v

                V_d_proj = phi_d @ V_mat
            else:
                V_d_proj = np.full(N_fs, V_s_scalar)

            phi2_cache   = float(np.dot(phi_d, phi_d))
            V_d_scalar_c = float(np.dot(phi_d, V_d_proj)) / max(phi2_cache, 1e-12)
            _vertex_cache = {
                'M':                  M,
                'Delta':              Delta_eff,
                'fs_idx':             fs_idx.copy(),
                'V_s_scalar':         V_s_scalar,
                'V_d_scalar':         V_d_scalar_c,
                'V_d_proj':           V_d_proj.copy(),
                'phi_d':              phi_d.copy(),
                'det_q0':             _det_q0,  # RPA det at q=0: near_critical proxy
                'near_critical':      False,    # updated by SCF loop each iteration
                'chi_QQ_from_normal': True,     # True: chi_QQ_normal_v was computed at Δ=0; prevent double-counting the SC-triggered JT feedback in the pairing vertex
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

def run_scf_material(solver: 'RMFT_Solver', target_doping: float, Delta_tetra: float, u: float, g_JT: float, t_pd: float, initial_M: float, initial_Q: float, initial_Delta: float, verbose: bool = False) -> Dict:
    s = copy.copy(solver)
    s.p = copy.copy(solver.p)
    s.p.Delta_tetra = float(Delta_tetra)
    s.p.u           = float(u)
    s.p.g_JT        = float(g_JT)
    s.p.t_pd        = float(t_pd)
    s.p.__post_init__()
    s._K_bare = s.p.K_lattice   # sync _K_bare after param rebuild so SCF uses correct bare stiffness
    s._reset_transient_state()
    return s.solve_self_consistent(target_doping, initial_M, initial_Q, initial_Delta, verbose)

class OptimPoint:
    __slots__ = ('doping', 'Delta_tetra', 'u', 'g_JT', 't_pd',
                 'Delta_total', 'converged', 'result',
                 'lambda_JT', 'lambda_max', 'stoner_ok', 'score', 'Tc',
                 'lambda_soc',
                 '_exclude_from_gp')  # set True on G22>0 / spont-JT failures to keep them out of GP training

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
        self._exclude_from_gp = False   # set True for G22>0 / spont-JT failures

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

class BayesianOptimizer:
    """
    Stage-1 Bayesian optimiser: 2D search over (Delta_tetra, lambda_soc).

    Delta_CF = f(Delta_tetra, lambda_soc) via exact SOC+CF diagonalisation —
    the Gamma_6–Gamma_7 gap, B1g blocking ratio, and SC-JT window all depend
    on both simultaneously. They must be optimised together.

    Fixed in Stage-1: u (correlation), t_pd, g_JT, K_lattice.
    u is fixed here because it primarily controls the AFM Weiss field scale and
    Gutzwiller factors — these set the normal-state chi, not Delta_CF directly.
    u is optimised in Stage-2 together with g_JT and t_pd.

    Every lambda_soc change requires __post_init__ + _rebuild_orbital_operators
    (U_gamma, U4, tau_x_op, Delta_CF must be rebuilt). Handled in _make_solver_stage1.

    Hard constraint: spont-JT (G22 <= 0 at Delta=0) → score=0, excluded from GP.
    SC Hessian scoring: lambda_min(H_3x3) from result['hessian'] enters _score().
    """
    W_STONER_BAD     = 0.20   # score weight when AFM Stoner criterion violated
    SPONT_JT_PENALTY = 0.05   # penalty factor when G3[2,2] ≤ 0 (spontaneous-JT risk)
    G_FALLBACK_SCALE = 5e-3   # overall scale for G-matrix proxy (no-gap region)
    SIGMOID_WIDTH    = 0.30   # sigmoid width for G22 continuous gate
    SC_HESS_SIGMA    = 0.05   # eV — width of sc_hessian_f sigmoid around lambda_min=0

    _NDIMS = 2   # (Delta_tetra, lambda_soc)
    _SEED  = 42

    def __init__(self, solver, n_doping_scan: int):
        self.solver        = solver
        self.n_doping_scan = n_doping_scan
        self.observations:    List[OptimPoint] = []
        self._gp_obs:         List[OptimPoint] = []  # constraint-valid subset for GP
        self._gp_lock         = _threading.Lock()    # guards _gp, _gp_obs, observations
        self._gp   = None
        self._bounds: Optional[Dict] = None

    # GP infrastructure  (shared; subclass overrides only _obs_to_X)
    def _build_gp(self) -> None:
        if not _SKLEARN_AVAILABLE:
            return
        n = self._NDIMS
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(n),
                           length_scale_bounds=[(1e-2, 10.0)] * n, nu=2.5)
                  + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 0.1)))
        self._gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, n_restarts_optimizer=10, normalize_y=True)

    def _obs_to_X(self, obs: 'OptimPoint') -> np.ndarray:
        """Stage-1: (Delta_tetra, lambda_soc). u fixed from solver.p."""
        return self._normalize(obs.Delta_tetra, obs.lambda_soc or 0.0)

    def _fit_gp(self) -> None:
        """Fit GP on constraint-valid subset only.

        Snapshot pattern: copy X, y under lock (fast), then fit outside
        lock (slow LAPACK).  After fitting, swap self._gp back under lock
        so predict in other threads always sees a fully-trained model.
        """
        min_obs = self._NDIMS + 1
        with self._gp_lock:
            if not _SKLEARN_AVAILABLE or self._gp is None or len(self._gp_obs) < min_obs:
                return
            X = np.array([self._obs_to_X(o) for o in self._gp_obs])
            y = np.array([o.score for o in self._gp_obs])
            gp_snapshot = copy.deepcopy(self._gp)   # clone kernel/hyperparams, not fit yet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_snapshot.fit(X, y)                   # slow fit outside lock
        with self._gp_lock:
            self._gp = gp_snapshot                  # atomic swap: threads see complete model

    def _lhs_sample(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(seed=self._SEED)
        s = np.zeros((n, self._NDIMS))
        for j in range(self._NDIMS):
            perm = rng.permutation(n)
            s[:, j] = (perm + rng.uniform(size=n)) / n
        return s

    def _expected_improvement(self, X_cand: np.ndarray, xi: float = 0.01) -> np.ndarray:
        with self._gp_lock:
            if not _SKLEARN_AVAILABLE or self._gp is None or len(self._gp_obs) == 0:
                return np.random.rand(len(X_cand))
            y_best = max(o.score for o in self._gp_obs)
            gp = self._gp   # reference to fully-fitted snapshot
        mu, sigma = gp.predict(X_cand, return_std=True)   # outside lock: read-only on snapshot
        sigma = np.maximum(sigma, 1e-9)
        z  = (mu - y_best - xi) / sigma
        EI = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.maximum(EI, 0.0)

    def _next_point_via_EI(self, n_restarts: int):
        rng  = np.random.default_rng()
        cand = rng.uniform(size=(n_restarts * 300, self._NDIMS))
        return self._denormalize(cand[np.argmax(self._expected_improvement(cand))])

    # ── Normalisation: Stage-1 = (Delta_tetra, lambda_soc, u) ───────────────
    def _normalize(self, Delta_tetra, lsoc) -> np.ndarray:
        b = self._bounds
        return np.array([
            (Delta_tetra - b['dt'][0])   / (b['dt'][1]   - b['dt'][0]),
            (lsoc        - b['lsoc'][0]) / (b['lsoc'][1] - b['lsoc'][0]),
        ])

    def _denormalize(self, x):
        b = self._bounds
        return (float(b['dt'][0]   + x[0] * (b['dt'][1]   - b['dt'][0])),
                float(b['lsoc'][0] + x[1] * (b['lsoc'][1] - b['lsoc'][0])))

    # ── Solver clone for Stage-1 (lambda_soc changes → full Hilbert rebuild) ─
    def _make_solver_stage1(self, Delta_tetra: float, lsoc: float) -> 'RMFT_Solver':
        """
        Clone solver and set (Delta_tetra, lambda_soc). u fixed from solver.p.
        lambda_soc change requires __post_init__ + _rebuild_orbital_operators
        to keep P6, P7, tau_x_op, Delta_CF consistent with new SOC eigenbasis.
        t_pd, g_JT, K_lattice, u remain at solver.p values.
        """
        s = copy.copy(self.solver); s.p = copy.copy(self.solver.p)
        s.p.Delta_tetra  = float(Delta_tetra)
        s.p.lambda_soc   = float(lsoc)
        # u unchanged: solver.p.u is used as-is
        s.p.__post_init__()
        s._K_bare = s.p.K_lattice
        s._rebuild_orbital_operators(s.p)  # lambda_soc or Delta_tetra changed
        s._reset_transient_state()
        return s

    # ── Async batch runner ───────────────────────────────────────────────────
    def _run_batch_async(self, params_list: list, fallback: 'OptimPoint',
                         on_result=None) -> list:
        """
        Evaluate _evaluate_material(*p) for each p with as_completed.
        on_result(opt_point) is called immediately after each result arrives,
        enabling incremental GP updates without a synchronous barrier.
        """
        n_workers = min(os.cpu_count() or 1, len(params_list), 6)
        if n_workers <= 1 or len(params_list) == 1:
            results = []
            for p in params_list:
                r = self._evaluate_material(*p)
                results.append(r)
                if on_result is not None:
                    on_result(r)
            return results
        results = [None] * len(params_list)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(self._evaluate_material, *p): i
                    for i, p in enumerate(params_list)}
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    _scf_log("WORKER-ERR", f"params={params_list[i]}  error={e}")
                    r = fallback
                results[i] = r
                if on_result is not None:
                    on_result(r)
        return results

    def _run_batch(self, params_list, fallback):
        """Sync wrapper for compatibility (Phase-2 uses this)."""
        return self._run_batch_async(params_list, fallback, on_result=None)

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

    def _score(self, Delta: float, converged: bool, result: dict, Tc, _g_post: dict) -> float:
        """
        Score for the superconductivity-triggered Jahn–Teller (JT) hypothesis.

        Physical idea
        -------------
        In the normal AFM state the B1g JT mode is symmetry-blocked (χ_tau^N small by construction).
        The superconducting condensate opens the Γ6 ↔ Γ7 channel and can destabilise the JT mode.

        Consequently the key observable is the curvature of the SC-state
        free-energy Hessian H(M, Q, Δ):

            λ_min(H_3x3) < 0   → SC-induced JT instability (desired)
            λ_min → 0⁻        → instability boundary (typically highest Tc)
            λ_min >> 0        → JT channel remains closed

        Gates (multiplicative factors in (0,1])
        ---------------------------------------
        conv_f        : SCF convergence quality
        stoner_f      : AFM Stoner instability not crossed
        g22_f         : JT mode stable in the normal state (G22 > 0)
        sc_hessian_f  : SC-state JT destabilisation from λ_min(H_3x3)
        ratio_bonus   : strong-coupling fingerprint (2Δ/kTc > 3.52)

        Hard constraint
        ---------------
        If the normal state already has a spontaneous JT distortion
        (G22 <= 0 or λ_min <= 0), the point is excluded (score = 0).
        """
        conv_f = 1.0 if converged else 0.10

        # AFM stability gate
        stoner_f = 1.0 if not result['afm_unstable'] else self.W_STONER_BAD

        # Normal-state JT stability check
        G_res = _g_post
        G22 = G_res['G22']
        lam_min_normal = G_res['lambda_min']

        # Hard reject if JT already unstable in the normal state
        if G22 <= 0.0 or lam_min_normal <= 0.0:
            return 0.0

        # Soft sigmoid gate favouring larger positive G22
        g22_f = self.SPONT_JT_PENALTY + (1.0 - self.SPONT_JT_PENALTY) \
                / (1.0 + np.exp(-G22 / self.SIGMOID_WIDTH))

        # SC-state Hessian gate: JT instability triggered by superconductivity
        hess = result.get('hessian', {})
        lmin_sc = hess.get('min_curvature', None) if hess else None

        if lmin_sc is not None and np.isfinite(lmin_sc):
            sc_hessian_f = float(1.0 / (1.0 + np.exp(lmin_sc / self.SC_HESS_SIGMA)))
            sc_hessian_f = float(np.clip(sc_hessian_f, 0.10, 1.0))
        else:
            # Hessian unavailable → neutral factor
            sc_hessian_f = 0.40

        Tc_proxy = Tc if Tc > 1e-6 else Delta * 0.3

        # Strong-coupling fingerprint: 2Δ₀/kTc > 3.52
        ratio_bonus = 1.0
        if Tc > 1e-6 and Delta > 1e-6:
            ratio_2D = 2.0 * Delta / Tc
            if ratio_2D > 3.52:
                ratio_bonus = 1.0 + 0.15 / (1.0 + np.exp(-(ratio_2D - 5.5) / 0.8))  # sigmoid: 1.0 at ratio=3.52
        return (Tc_proxy * conv_f * stoner_f * g22_f * sc_hessian_f * ratio_bonus)

    def _g_fallback_score(self, M0, doping, Delta_tetra, u, gJT, t_pd) -> float:
        """Cheap G-matrix proximity score when SCF finds no SC gap."""
        try:
            s2 = copy.copy(self.solver); s2.p = copy.copy(self.solver.p)
            s2.p.Delta_tetra = float(Delta_tetra); s2.p.u = float(u)
            s2.p.g_JT = float(gJT); s2.p.t_pd = float(t_pd)
            s2.p.__post_init__()
            s2._K_bare = s2.p.K_lattice
            s2._reset_transient_state()
            G_res = s2.compute_G_instability(doping, M0)
        except Exception:
            return 0.0
        G22 = G_res['G22']
        if G22 <= 0.0:
            return 0.0
        lam_min = G_res['lambda_min']
        Tc_est  = G_res['Tc_estimate']
        if lam_min <= 0.0: return 0.0
        g22_f = self.SPONT_JT_PENALTY + (1.0 - self.SPONT_JT_PENALTY) / (1.0 + np.exp(-G22 / self.SIGMOID_WIDTH))
        return self.G_FALLBACK_SCALE * (1.0 - min(lam_min, 1.0)) * g22_f * (1.0 + min(Tc_est / 0.004, 8.0))

    # JT diagnostics
    def _jt_coupling_strength(self, solver, result: dict, G_post: dict) -> dict:
        """
        Compute λ_JT = (g²/K_bare)·χ_τ, classify coupling regime, and run the
        SC-JT window diagnostic via check_sc_jt_window (uses SC-state chi_tau).

        G_post must be the result of compute_G_instability(doping, M_conv) —
        it is always available at the call site (_eval_one_doping computes it
        unconditionally). Passing it avoids the fallback path in chi_QQ.
        """
        chi_tau   = result['chi_tau']
        g_JT      = solver.p.g_JT
        K_bare    = max(solver._K_bare, 1e-9)
        lam       = (g_JT**2 / K_bare) * chi_tau
        regime    = ('SC-triggered' if 0.05 < lam < 1.0
                     else ('strong-coupling' if lam >= 1.0 else 'JT-closed'))

        # chi_QQ from G_post: g_JT²·χ_orbital; divide out g_JT² to get χ_orbital
        chi_QQ_raw  = G_post['chi_QQ']
        chi0_approx = chi_QQ_raw / max(g_JT**2, 1e-12)
        K_eff_scf   = result['K_eff_scf']
        lam_min_scf = G_post['lambda_min']

        jt_win = check_sc_jt_window(
            g_JT=g_JT, Delta_CF=max(solver.p.Delta_CF, 1e-12),
            chi_tau=chi_tau, chi0=chi0_approx,
            K_lattice=K_bare, K_eff=K_eff_scf,
            lambda_min=max(lam_min_scf, 1e-4))
        return {'lambda_JT': float(lam), 'chi_tau': float(chi_tau),
                'jt_viable': 0.05 < lam < 1.0, 'regime': regime,
                'jt_window': jt_win}

    def _jt_causality_test(self, solver, result) -> dict:
        """
        SC-triggered JT causality test based on G-matrix and Hessian analysis.

        Hypothesis:
          - Normal state: G3[2,2] > 0 (JT mode stable, symmetry-blocked)
          - SC state:     ∂²F/∂Q² < 0 (JT mode unstable, triggered by SC)

        Score = (stability + hess_metric) / 2
          stability  : G22_normal / 0.5,  clips [0,1] — normal-state JT blockade
          hess_metric: −λ_min(H) / 0.1,  clips [0,1] — SC Hessian confirmation

        softening and enhancement are logged as diagnostics but do NOT enter
        the score: both are derivable from d2F_Q_sc and would double-weight
        the same physical signal (χ_QQ^SC) if included in the average.
        """
        if result and 'G_instability' in result:
            G = result['G_instability']
        else:
            try:
                doping = result.get('target_doping', 0.15)
                M      = result.get('M', 0.11)
                G      = solver.compute_G_instability(doping, M)
            except Exception as e:
                return {'sc_triggered': False, 'error': str(e),
                        'note': 'G-matrix computation failed',
                        'Q_sc': 0.0, 'Q_normal': 0.0,
                        'level1_ok': False, 'level2_ok': False}

        G22_normal = G.get('G22', float('nan'))
        d2F_normal = G.get('d2F_Q_normal', float('nan'))
        d2F_sc     = G.get('d2F_Q_sc',     float('nan'))
        chi_ratio  = G.get('chi_QQ_sc', 1.0) / max(G.get('chi_QQ', 1e-12), 1e-12)
        Q_sc       = abs(result.get('Q', 0.0))
        Delta_sc   = abs(result.get('Delta_s', 0.0)) + abs(result.get('Delta_d', 0.0))

        hess     = result.get('hessian', {})
        lmin_sc  = hess.get('min_curvature', float('nan')) if hess else float('nan')

        if Delta_sc < 1e-4 or Q_sc < 1e-5:
            return {'sc_triggered': False, 'G22_normal': G22_normal,
                    'd2F_normal': d2F_normal, 'd2F_sc': d2F_sc,
                    'lmin_sc_hessian': lmin_sc,
                    'note': 'No significant SC or JT order in converged state',
                    'Q_sc': Q_sc, 'Q_normal': 0.0,
                    'level1_ok': False, 'level2_ok': False}

        # Primary metrics (independent signals)
        stability   = float(np.clip(G22_normal / 0.5, 0.0, 1.0)) if np.isfinite(G22_normal) else 0.0
        hess_confirmed = np.isfinite(lmin_sc) and lmin_sc < 0.0
        hess_metric    = float(np.clip(-lmin_sc / 0.1, 0.0, 1.0)) if np.isfinite(lmin_sc) else 0.0

        # Score: only truly independent signals
        score = 0.0 if stability < 0.1 else (stability + hess_metric) / 2.0

        # Diagnostic-only quantities (NOT in score — both derived from d2F_Q_sc)
        if np.isfinite(d2F_normal) and np.isfinite(d2F_sc) and d2F_normal > 1e-6:
            softening = float(np.clip((d2F_normal - d2F_sc) / d2F_normal, 0.0, 1.0))
        else:
            softening = 0.0
        enhancement = float(np.clip((chi_ratio - 1.0), 0.0, 1.0)) if np.isfinite(chi_ratio) else 0.0

        if score > 0.7 and stability > 0.6 and hess_confirmed:
            regime = 'CONFIRMED SC-triggered JT'
        elif score > 0.4 and stability > 0.3:
            regime = 'PARTIAL: SC-JT coupling present'
        elif stability < 0.3:
            regime = 'WARNING: Normal state JT-unstable (spontaneous risk)'
        elif not hess_confirmed:
            regime = 'WARNING: SC does not soften JT mode'
        else:
            regime = 'INCONCLUSIVE: Mixed signals'

        return {
            'sc_triggered': score > 0.5 or hess_confirmed,
            'score': score, 'stability': stability,
            'hess_confirmed': hess_confirmed, 'lmin_sc_hessian': lmin_sc,
            # diagnostics (logged, not scored):
            'softening': softening, 'enhancement': enhancement,
            'G22_normal': G22_normal, 'd2F_normal': d2F_normal, 'd2F_sc': d2F_sc,
            'chi_ratio': chi_ratio, 'regime': regime,
            'note': f"{regime} (score={score:.3f}, λ_min(H)={lmin_sc:.4f})",
            'Q_sc': Q_sc, 'Q_normal': 0.0,
            'level1_ok': stability > 0.3, 'level2_ok': hess_confirmed,
        }

    def _cheap_scout(self, doping, Delta_tetra, lsoc, u, gJT, t_pd, M0) -> Dict:
        """
        Analytic pre-screen for Stage-1 (Delta_tetra, lambda_soc, u).
        Hard constraint: spont_jt=True → viable=False, score=0, not added to GP.
        Critical near the SC instability: χ_DD ∝ ∫tanh(ξ/2T)/(2ξ)dξ depends sensitively on where μ cuts the band.
        μ-dependent chi0: Lorentz DOS shifted by mu_e: χ₀(μ) = N0 / (1 + ((h_afm - μ_e) / π·t_eff)²)   [Stoner-Lorentz, finite doping]
        The (h_afm - mu_e) combination captures both AFM suppression and the doping-induced Fermi level shift simultaneously, without a k-grid.
        """
        p2 = copy.copy(self.solver.p)
        p2.Delta_tetra = Delta_tetra
        p2.lambda_soc  = lsoc
        p2.u           = u
        p2.g_JT        = gJT
        p2.t_pd        = t_pd
        p2.__post_init__()

        abs_d = max(abs(doping), 1e-6)
        g_t   = 2.0 * abs_d / (1.0 + abs_d)
        g_J   = 4.0 / (1.0 + abs_d)**2
        f_d   = abs_d / (abs_d + p2.doping_0)
        tx    = p2.t0

        N0        = 1.0 / (np.pi * max(g_t * tx, 1e-6))
        mu_e      = 0.5 * max(p2.Delta_CF, 1e-4) - 2.0 * g_t * tx * np.tanh(doping / 0.1)  # Chemical potential proxy: Γ₆–Γ₇ midpoint shifted by doping.
        sz_vec    = np.array([1.0, -1.0, p2.eta, -p2.eta])
        h_kin_sc  = p2.Z * 2.0 * tx**2 / max(p2.U, 1e-9) / 2.0
        J_A1g     = np.diag(p2.J_CT / 2.0 * np.array([1.0, 1.0, p2.eta**2, p2.eta**2]))
        O_exp     = g_J * f_d * M0 * sz_vec  # AFM Weiss field proxy from the converged M0 (Γ₆↑ component, sz=+1)
        h_afm     = float((J_A1g @ O_exp + h_kin_sc * O_exp)[0])
        t_eff     = max(g_t * tx, 1e-6)
        chi0_val  = N0 / (1.0 + ((h_afm - mu_e) / (np.pi * t_eff))**2)

        J_super      = self.solver.effective_superexchange(g_J, tx, tx, doping)
        stoner_raw   = 1.0 - J_super * chi0_val
        stoner_denom = max(stoner_raw, p2.rpa_cutoff)
        stoner_ok    = stoner_raw > 0.0
        rpa          = 1.0 / stoner_denom
        V_eff        = (p2.g_JT**2 / max(p2.K_lattice, 1e-9)) * rpa

        s2 = copy.copy(self.solver)
        s2.p = p2
        s2._K_bare = p2.K_lattice
        s2._rebuild_orbital_operators(p2)   # lambda_soc / Delta_tetra changed
        s2._reset_transient_state()
        G_res = s2.compute_G_instability(doping, M=M0)

        alpha_M          = max(p2.U / max(t_eff, 1e-6) / 2.35 - 1.0, 0.0)
        chi_tau_normal   = N0 / (1.0 + alpha_M * M0**2)
        # In the superconducting state the condensate partially lifts B1g blocking through Γ6–Γ7 mixing.
        # Full χ_τ(Δ≠0) is computed after SCF. Here we apply a cheap scout proxy controlled by Δ_CF / kT with conservative cap
        chi_tau_SC_proxy = chi_tau_normal * min(
            3.0, 1.0 + max(p2.Delta_CF, 1e-4) / max(p2.kT, 1e-4))

        lam_min = G_res['lambda_min']
        G22     = G_res['G22']

        # Hard constraint: spontaneous JT = G3[2,2] ≤ 0 at Δ=0.
        spont_jt = G22 <= 0.0
        if spont_jt:
            return {
                'skip_window': False, 'skip_far': False, 'G_score': 0.0,
                'near_critical': False, 'spont_jt': True,
                'M0': M0, 'stoner_ok': stoner_ok,
                'lambda_scout': 0.0, 'chi_tau': chi_tau_normal,
                'chi_tau_SC': chi_tau_SC_proxy,
                'Delta_CF': max(p2.Delta_CF, 1e-4),
                'jt_window': {}, 'G': G_res, 'G22': G22,
            }

        jt_window = check_sc_jt_window(
            g_JT=p2.g_JT, Delta_CF=max(p2.Delta_CF, 1e-12),
            chi_tau=chi_tau_SC_proxy, chi0=chi0_val,
            K_lattice=p2.K_lattice, K_eff=max(p2.K_lattice, 1e-9),
            lambda_min=max(lam_min, 1e-4))

        evec_ok   = abs(G_res['evec_min'][2]) <= 0.6
        lam_scout = V_eff * chi_tau_normal
        g22_f     = self.SPONT_JT_PENALTY + (1.0 - self.SPONT_JT_PENALTY) \
                    / (1.0 + np.exp(-G22 / self.SIGMOID_WIDTH))
        G_score   = g22_f * float(np.clip(1.0 - abs(lam_min) / 3.0, 0.0, 1.0)) \
                    * (1.0 if evec_ok else 0.3)

        skip_window = not jt_window['window_open']
        skip_far    = (lam_min > 2.9 and G_score < 0.1)

        return {
            'skip_window': skip_window, 'skip_far': skip_far,
            'G_score': G_score,
            'near_critical': (0.0 <= lam_min < 0.90),
            'spont_jt': False, 'M0': M0, 'stoner_ok': stoner_ok,
            'lambda_scout': lam_scout, 'chi_tau': chi_tau_normal,
            'chi_tau_SC': chi_tau_SC_proxy,
            'Delta_CF': max(p2.Delta_CF, 1e-4),
            'jt_window': jt_window, 'G': G_res, 'G22': G22,
        }

    def _adaptive_seed_near_critical(self, n_refine, lambda_target=1.0, sigma_lambda=0.40) -> np.ndarray:
        """Importance-weighted LHS biased toward lambda_min ≈ 1 (stable side only)."""
        rng   = np.random.default_rng(seed=7)
        pts   = np.zeros((n_refine, self._NDIMS))
        n_acc = 0
        d_mid = 0.5 * (self._bounds['doping'][0] + self._bounds['doping'][1])
        gJT   = self.solver.p.g_JT
        t_pd  = self.solver.p.t_pd
        for _ in range(n_refine * 800):
            if n_acc >= n_refine: break
            x           = rng.uniform(size=self._NDIMS)
            dt, ls      = self._denormalize(x)
            u_fix       = self.solver.p.u
            m0          = self.solver._estimate_M0(d_mid)
            scout       = self._cheap_scout(d_mid, dt, ls, u_fix, gJT, t_pd, m0)
            if scout['spont_jt']:
                continue                  # hard constraint
            if (scout['skip_window'] or scout['skip_far']) \
                    and rng.uniform() > scout['G_score']:
                continue   # kizárjuk a biztosan unérdekeseket, de G_score-al arányos valószínűséggel átengedjük
            lam = scout['G']['lambda_min']
            if rng.uniform() < np.exp(-0.5 * ((lam - lambda_target) / sigma_lambda)**2):
                pts[n_acc] = x; n_acc += 1
        if n_acc < n_refine:
            pts[n_acc:] = rng.uniform(size=(n_refine - n_acc, self._NDIMS))
        return pts

    # Single-point SCF evaluation (shared between Phase-1 and Phase-2)
    def _eval_one_doping(self, solver, doping, Delta_tetra, u, gJT, t_pd, initial_M, initial_Q, initial_Delta, lsoc=None) -> 'OptimPoint':
        tag = f"FULL δ={doping:.3f}"
        t0  = _time.time()
        try:
            result    = solver.solve_self_consistent(doping, initial_M, initial_Q, initial_Delta, verbose=False)
            Delta     = result.get('Delta_s', 0.0) + result.get('Delta_d', 0.0)
            converged = result.get('converged', False)
        except Exception as e:
            _scf_log(tag, f"SCF error: {e}")
            result, Delta, converged = {}, 0.0, False

        Tc = 0.0
        if converged and Delta > 1e-6:
            try:
                tc_res = solver.compute_Tc_by_gap_suppression(doping, sc_result=result)
                Tc     = float(tc_res.get('Tc', 0.0))
            except Exception:
                pass

        M_conv = result.get('M', initial_M) if result else initial_M
        G_post = solver.compute_G_instability(doping, M=M_conv) if result else {}

        if Delta < 1e-8 and Tc < 1e-6:
            return self._g_fallback_score(initial_M, doping, Delta_tetra, u, gJT, t_pd)

        jt    = self._jt_coupling_strength(solver, result, G_post)
        lJT   = jt['lambda_JT']
        lmax  = result.get('lambda_max', 0.0) if result else 0.0
        s_ok  = not (result.get('afm_unstable', False) if result else False)

        chi_qq_K = G_post.get('chi_QQ', 0.0) / max(G_post.get('K_eff', 1.0), 1e-9)
        score    = self._score(Delta, converged, result, Tc, G_post)

        hess     = result.get('hessian', {}) if result else {}
        lmin_sc  = hess.get('min_curvature', float('nan')) if hess else float('nan')
        hess_dir = G_post.get('instab_dir', '?')
        lmax_log = result.get('lambda_max', float('nan')) if result else float('nan')
        gap_sym  = result.get('gap_symmetry', '?') if result else '?'
        _scf_log(tag, f"Δ={Delta:.5f}  Tc={Tc*1000:.2f}meV  score={score:.5f}"
                      f"  λ_JT={lJT:.3f}[{jt['regime']}]"
                      f"  λ_max={lmax_log:.3f}({gap_sym})  χ_QQ/K={chi_qq_K:.3f}"
                      f"  λ_min(H)={lmin_sc:+.4f}[{hess_dir}]"
                      f"  {'✓' if converged else '⚠'}  ({_time.time()-t0:.1f}s)")
        return OptimPoint(doping, Delta_tetra, u, gJT, t_pd, Delta, converged,
                          result, lambda_JT=lJT, lambda_max=lmax, stoner_ok=s_ok,
                          score=score, Tc=Tc, lambda_soc=lsoc)

    def _scan_doping(self, solver, doping_grid, Delta_tetra, u, gJT, t_pd, lsoc=None) -> 'OptimPoint':
        """
        Warm-started doping scan: each point inherits M, Q, Δ from previous.
        Returns the best-scoring OptimPoint across the grid.
        """
        best: Optional[OptimPoint] = None
        prev: Optional[Dict] = None
        iM0 = self.solver._estimate_M0(doping_grid[0])
        iQ0, iD0 = 1e-4, 0.02
        for doping in doping_grid:
            iM = prev['M']                                                    if prev else iM0
            iQ = prev['Q']                                                    if prev else iQ0
            iD = max(prev['Delta_s'] + prev['Delta_d'], iD0)                  if prev else iD0
            pt = self._eval_one_doping(solver, doping, Delta_tetra, u, gJT, t_pd, iM, iQ, iD, lsoc=lsoc)
            if pt.result: prev = pt.result
            if best is None or pt.score > best.score: best = pt
        return best

    # Phase-1 material evaluation
    def _evaluate_material(self, Delta_tetra, lsoc) -> 'OptimPoint':
        """
        Evaluate one material (Δ_tet, λ_soc) with u, t_pd, g_JT, K_lattice fixed.
        Builds a fresh solver clone with _rebuild_orbital_operators (lambda_soc changed).
        Preceded by cheap scout; spont-JT returns score=0, excluded from GP.
        """
        b     = self._bounds
        u     = self.solver.p.u
        gJT   = self.solver.p.g_JT
        t_pd  = self.solver.p.t_pd
        dg    = np.linspace(b['doping'][0], b['doping'][1], self.n_doping_scan)
        d_mid = 0.5 * (b['doping'][0] + b['doping'][1])
        tag   = f"SCOUT Δt={Delta_tetra:.3f}"

        _scf_log(tag, f"λ_soc={lsoc:.4f}  u={u:.2f}(fix)  g_JT={gJT:.3f}(fix)"
                      f"  t_pd={t_pd:.4f}(fix)  K_latt={self.solver.p.K_lattice:.3f}(fix)"
                      f"  δ∈{b['doping']} ({self.n_doping_scan}pts)")

        sc = self._cheap_scout(d_mid, Delta_tetra, lsoc, u, gJT, t_pd,
                               self.solver._estimate_M0(d_mid))
        G  = sc['G']
        _lmin_str = f"{G['lambda_min']:.3f}" + (" ⚠SPONT" if sc['spont_jt'] else "")
        # Log: mi a skip döntés alapja, nem egy homályos 'viable' flag
        if sc['spont_jt']:
            _skip_str = "SPONT-JT"
        elif sc['skip_window']:
            _skip_str = "SKIP-window"
        elif sc['skip_far']:
            _skip_str = "SKIP-far"
        else:
            _skip_str = "→ SCF"
        _scf_log(tag, f"λ_min={_lmin_str}  G22={G['G22']:.3f}"
                      f"  Δ_CF={sc['Delta_CF']:.4f} eV"
                      f"  dom={G['dominant']}"
                      f"  Tc_est={G['Tc_estimate']*1000:.1f}meV"
                      f"  G_score={sc['G_score']:.3f}  {_skip_str}"
                      + ("  ⚠ near-crit" if sc['near_critical'] else ""))

        # ── Hard constraint: spontaneous JT ────────────────────────────────
        if sc['spont_jt']:
            _scf_log(f"CONST Δt={Delta_tetra:.3f}",
                     f"HARD CONSTRAINT: spont JT (λ_min={G['lambda_min']:.3f}<0)"
                     f"  score=0  [NOT added to GP]")
            pt = OptimPoint(d_mid, Delta_tetra, u, gJT, t_pd, 0.0, False,
                            score=0.0, lambda_soc=lsoc)
            pt._exclude_from_gp = True
            return pt

        # ── Skip: SC-JT window closed (based on chi_tau_SC_proxy) ──────────
        # A scout chi_tau_SC_proxy-ja konzervatív: ha ez is zárva mutat,
        # az SCF sem fog JT instabilitást találni. Fallback score tájékoztatja a GP-t.
        if sc['skip_window']:
            fb = self.G_FALLBACK_SCALE * sc['G_score'] * 0.1
            _scf_log(f"SKIP Δt={Delta_tetra:.3f}",
                     f"SC-JT window closed  Δ_CF={sc['Delta_CF']:.4f} eV"
                     f"  fallback_score={fb:.2e}")
            return OptimPoint(d_mid, Delta_tetra, u, gJT, t_pd, 0.0, False,
                              score=fb, lambda_soc=lsoc)

        # ── Skip: far from being unstable, it's not worth running SCF ────
        if sc['skip_far']:
            fb = self.G_FALLBACK_SCALE * max(0.0, 1.0 - G['lambda_min']) * sc['G_score']
            _scf_log(f"SKIP Δt={Delta_tetra:.3f}",
                     f"far from instability (λ_min={G['lambda_min']:.3f}>2.9)"
                     f"  fallback_score={fb:.2e}")
            return OptimPoint(d_mid, Delta_tetra, u, gJT, t_pd, 0.0, False,
                              score=fb, lambda_soc=lsoc)

        s = self._make_solver_stage1(Delta_tetra, lsoc)
        best = self._scan_doping(s, dg, Delta_tetra, u, gJT, t_pd, lsoc=lsoc)
        _scf_log(tag, f"↳ best δ={best.doping:.3f}  Δ={best.Delta_total:.5f}"
                      f"  Tc={best.Tc*1000:.2f}meV  score={best.score:.5f}"
                      f"  λ_JT={best.lambda_JT:.3f}  {'✓' if best.converged else '⚠'}"
                      f"  [λ_min(scout)={G['lambda_min']:.3f}]")
        return best

    # Helpers
    @staticmethod
    def _pick_best(observations: list) -> tuple:
        best     = max(observations, key=lambda o: o.score)
        best_raw = max(observations, key=lambda o: o.Delta_total)
        valid    = [o for o in observations if o.converged and o.lambda_JT > 0.0]
        return best, (max(valid, key=lambda o: o.score) if valid else best), best_raw

    def _register(self, pt: 'OptimPoint') -> None:
        """Add to observations; add to GP set only if constraint-valid."""
        with self._gp_lock:
            self.observations.append(pt)
            if not getattr(pt, '_exclude_from_gp', False) and pt.score > 0.0:
                self._gp_obs.append(pt)

    # Phase-1 optimisation loop
    def optimize(self,
                 doping_bounds:      Tuple[float, float],
                 Delta_tetra_bounds: Tuple[float, float],
                 lsoc_bounds:        Tuple[float, float],
                 n_initial:          int,
                 n_refine:           int,
                 n_iterations:       int,
                 verbose:            bool = True) -> Dict:
        """
        Stage-1: LHS → adaptive seed → GP-EI in (Δ_tet, λ_soc). u fixed.
        Async: results registered immediately via on_result callback.
        GP fitted on constraint-valid subset only.
        """
        self._bounds = {'doping': doping_bounds, 'dt': Delta_tetra_bounds, 'lsoc': lsoc_bounds}
        self._build_gp()
        total  = n_initial + n_refine + n_iterations
        t0     = _time.time()
        d_mid  = 0.5 * (doping_bounds[0] + doping_bounds[1])
        tp_fix = self.solver.p.t_pd
        gJT_fix= self.solver.p.g_JT
        Kl_fix = self.solver.p.K_lattice
        dct    = self.solver.p.Delta_CT

        _scf_log("BO-P1", "="*60)
        _scf_log("BO-P1", "Stage-1: 2D (Δ_tet, λ_soc)  |  ARD Matérn-2.5")
        _scf_log("BO-P1", f"FIXED: u={self.solver.p.u:.4f}  t_pd={tp_fix:.4f} eV"
                          f"  g_JT={gJT_fix:.4f} eV/Å  K_latt={Kl_fix:.4f} eV/Å²"
                          f"  Δ_CT={dct:.4f} eV  → t0={tp_fix**2/dct:.4f} eV")
        _scf_log("BO-P1", f"Δ_tet∈{Delta_tetra_bounds}  λ_soc∈{lsoc_bounds}  u={self.solver.p.u:.4f}(fix)")
        _scf_log("BO-P1", f"budget: {n_initial} LHS + {n_refine} adaptive + {n_iterations} EI"
                          f" = {total} (~{total*self.n_doping_scan} SCF)")
        _scf_log("BO-P1", "Hard constraint: G22≤0 → score=0, excluded from GP")
        _scf_log("BO-P1", "SC Hessian: λ_min(H_3x3) from result['hessian'] enters score")
        _scf_log("BO-P1", "="*60)

        _fb = OptimPoint(d_mid, 0.0, 5.0, gJT_fix, tp_fix, 0.0, False, score=0.0, lambda_soc=0.5*(lsoc_bounds[0]+lsoc_bounds[1]))
        done_count = [0]

        def tick(pfx="BO "):
            with _log_lock:
                sys.stdout.write(self._progress_bar(
                    done_count[0], total, _time.time()-t0, prefix=pfx))
                sys.stdout.flush()

        def on_result(pt):
            self._register(pt)
            if len(self._gp_obs) >= self._NDIMS + 1:
                self._fit_gp()
            done_count[0] += 1
            tick("Seed ")

        _scf_log("BO-P1", f"[1a] LHS seed ({n_initial} pts, async, {os.cpu_count() or 1} workers)")
        self._run_batch_async(
            [self._denormalize(x) for x in self._lhs_sample(n_initial)],
            _fb, on_result=on_result)

        if n_refine > 0:
            _scf_log("BO-P1", f"[1b] Adaptive seed near λ_eff≈1 ({n_refine} pts, stable side only)")
            self._run_batch_async(
                [self._denormalize(x) for x in self._adaptive_seed_near_critical(n_refine)],
                _fb, on_result=on_result)

        _scf_log("BO-P1", f"[2] GP-EI ({n_iterations} iters, sequential)")
        for i in range(n_iterations):
            self._fit_gp()
            dt, ls = self._next_point_via_EI(n_restarts=60)
            _scf_log("BO-P1", f"[EI {i+1}/{n_iterations}]"
                              f" Δ_tet={dt:.3f}  λ_soc={ls:.4f}")
            pt = self._evaluate_material(dt, ls)
            self._register(pt)
            done_count[0] += 1
            tick("BO   ")
            if verbose and (i + 1) % 10 == 0 and self.observations:
                _scf_log("BO-P1", f"best so far: {max(self.observations, key=lambda o: o.score)}")

        best, best_valid, best_raw = self._pick_best(self.observations)
        elapsed = _time.time() - t0
        n_excl  = sum(1 for o in self.observations if getattr(o, '_exclude_from_gp', False))
        _scf_log("BO-P1", "="*60)
        _scf_log("BO-P1", f"Stage-1 done ({elapsed/60:.1f} min)  |  Best: {best}")
        _scf_log("BO-P1", f"GP set: {len(self._gp_obs)}/{len(self.observations)}"
                          f"  ({n_excl} spont-JT excluded)")
        _scf_log("BO-P1", "="*60)
        return {'best_point': best, 'best_valid': best_valid, 'best_raw': best_raw,
                'observations': self.observations, 'gp': self._gp, 'elapsed_s': elapsed}

class BayesianOptimizerPhase2(BayesianOptimizer):
    """
    Stage-2 BO: 3D search over (u, g_JT, t_pd).
    (Delta_tetra, lambda_soc) fixed from Stage-1. K_lattice fixed throughout.

    u, g_JT, t_pd jointly determine:
      - correlation strength: u sets Weiss field amplitude (J_CT, U_mf),
        Gutzwiller factors g_t, g_J → AFM order and normal-state chi
      - pairing scale:  V_eff = g_JT²/K · RPA  (both s and d channels)
      - hopping:        t0 = t_pd²/Delta_CT → bandwidth, Fermi surface

    No Hilbert-space rebuild needed: lambda_soc and Delta_tetra are fixed,
    so U_gamma, U4, tau_x_op, Delta_CF are identical to Stage-1 best point.
    Only __post_init__ is required when u, t_pd, or g_JT change.

    Inherits: hard constraint, SC Hessian scoring, async pool, _gp_obs subset.
    """
    _NDIMS = 3   # (u, g_JT, t_pd)
    _SEED  = 43

    def __init__(self, solver, n_doping_scan: int,
                 best_Delta_tetra: float, best_lsoc: float, best_u: float):
        super().__init__(solver, n_doping_scan)
        self.best_Delta_tetra = best_Delta_tetra
        self.best_lsoc        = best_lsoc
        self.best_u           = best_u

    def _normalize(self, u, gJT, t_pd) -> np.ndarray:
        b = self._bounds
        return np.array([
            (u    - b['u'][0])   / (b['u'][1]   - b['u'][0]),
            (gJT  - b['g'][0])   / (b['g'][1]   - b['g'][0]),
            (t_pd - b['tpd'][0]) / (b['tpd'][1] - b['tpd'][0]),
        ])

    def _denormalize(self, x):
        b = self._bounds
        return (float(b['u'][0]   + x[0] * (b['u'][1]   - b['u'][0])),
                float(b['g'][0]   + x[1] * (b['g'][1]   - b['g'][0])),
                float(b['tpd'][0] + x[2] * (b['tpd'][1] - b['tpd'][0])))

    def _obs_to_X(self, obs: 'OptimPoint') -> np.ndarray:
        return self._normalize(obs.u, obs.g_JT, obs.t_pd)

    def _make_solver_stage2(self, u: float, gJT: float, t_pd: float) -> 'RMFT_Solver':
        """
        Clone with (Delta_tetra, lambda_soc) fixed from Stage-1.
        u, g_JT, t_pd change → __post_init__ sufficient, no Hilbert rebuild
        (lambda_soc and Delta_tetra are fixed, so orbital operators unchanged).
        """
        s = copy.copy(self.solver); s.p = copy.copy(self.solver.p)
        s.p.Delta_tetra = float(self.best_Delta_tetra)
        s.p.lambda_soc  = float(self.best_lsoc)
        s.p.u           = float(u)
        s.p.g_JT        = float(gJT)
        s.p.t_pd        = float(t_pd)
        s.p.__post_init__()
        s._K_bare = s.p.K_lattice
        # No _rebuild_orbital_operators: lambda_soc and Delta_tetra are fixed
        s._reset_transient_state()
        return s

    def _evaluate_material(self, u, gJT, t_pd) -> 'OptimPoint':
        b     = self._bounds
        dg    = np.linspace(b['doping'][0], b['doping'][1], self.n_doping_scan)
        d_mid = 0.5 * (b['doping'][0] + b['doping'][1])
        tag   = f"P2 u={u:.2f}"

        _scf_log(tag, f"g_JT={gJT:.4f}  t_pd={t_pd:.4f}"
                      f"  [Δ_tet={self.best_Delta_tetra:.3f}"
                      f"  λ_soc={self.best_lsoc:.4f}(fix)"
                      f"  K_latt={self.solver.p.K_lattice:.3f}(fix)]")

        sc = self._cheap_scout(d_mid, self.best_Delta_tetra, self.best_lsoc,
                               u, gJT, t_pd,
                               self.solver._estimate_M0(d_mid))
        if sc['spont_jt']:
            _scf_log(f"CONST P2 u={u:.2f}",
                     "HARD CONSTRAINT: spont JT → score=0 [NOT added to GP]")
            pt = OptimPoint(d_mid, self.best_Delta_tetra, u, gJT, t_pd,
                            0.0, False, score=0.0, lambda_soc=self.best_lsoc)
            pt._exclude_from_gp = True
            return pt

        s    = self._make_solver_stage2(u, gJT, t_pd)
        best = self._scan_doping(s, dg, self.best_Delta_tetra, u,
                                 gJT, t_pd, lsoc=self.best_lsoc)
        _scf_log(tag, f"↳ best δ={best.doping:.3f}  Tc={best.Tc*1000:.2f}meV"
                      f"  score={best.score:.5f}  {'✓' if best.converged else '⚠'}")
        return best

    def optimize(self,
                 doping_bounds: Tuple[float, float],
                 u_bounds:      Tuple[float, float],
                 gJT_bounds:    Tuple[float, float],
                 t_pd_bounds:   Tuple[float, float],
                 n_initial:     int,
                 n_iterations:  int,
                 verbose:       bool = True) -> Dict:
        """Stage-2: LHS → GP-EI in (u, g_JT, t_pd). Async pool with on_result."""
        self._bounds = {'doping': doping_bounds, 'u': u_bounds,
                        'g': gJT_bounds, 'tpd': t_pd_bounds}
        self._build_gp()
        total  = n_initial + n_iterations
        t0     = _time.time()
        dct    = self.solver.p.Delta_CT
        d_mid  = 0.5 * (doping_bounds[0] + doping_bounds[1])
        um     = 0.5 * (u_bounds[0]    + u_bounds[1])
        gm     = 0.5 * (gJT_bounds[0]  + gJT_bounds[1])
        tm     = 0.5 * (t_pd_bounds[0] + t_pd_bounds[1])

        _scf_log("BO-P2", "="*60)
        _scf_log("BO-P2", "Stage-2: 3D (u, g_JT, t_pd)  |  ARD Matérn-2.5")
        _scf_log("BO-P2", f"FIXED: Δ_tet={self.best_Delta_tetra:.4f}"
                          f"  λ_soc={self.best_lsoc:.4f} eV"
                          f"  K_latt={self.solver.p.K_lattice:.4f} eV/Å²"
                          f"  Δ_CT={dct:.4f} eV")
        _scf_log("BO-P2", f"u∈{u_bounds}  g_JT∈{gJT_bounds} eV/Å  t_pd∈{t_pd_bounds} eV"
                          f"  → t0∈({t_pd_bounds[0]**2/dct:.3f},"
                          f"{t_pd_bounds[1]**2/dct:.3f}) eV")
        _scf_log("BO-P2", f"budget: {n_initial} LHS + {n_iterations} EI = {total}")
        _scf_log("BO-P2", "Hard spont-JT constraint + SC Hessian λ_min scoring active")
        _scf_log("BO-P2", "="*60)

        _fb = OptimPoint(d_mid, self.best_Delta_tetra, um, gm, tm,
                         0.0, False, score=0.0, lambda_soc=self.best_lsoc)
        done_count = [0]

        def tick():
            with _log_lock:
                sys.stdout.write(self._progress_bar(
                    done_count[0], total, _time.time()-t0, prefix="P2  "))
                sys.stdout.flush()

        def on_result(pt):
            self._register(pt)
            if len(self._gp_obs) >= self._NDIMS + 1:
                self._fit_gp()
            done_count[0] += 1
            tick()

        _scf_log("BO-P2", f"[2a] LHS seed ({n_initial} pts, async)")
        self._run_batch_async(
            [self._denormalize(x) for x in self._lhs_sample(n_initial)],
            _fb, on_result=on_result)

        _scf_log("BO-P2", f"[2b] GP-EI ({n_iterations} iters)")
        for i in range(n_iterations):
            self._fit_gp()
            u, gJT, t_pd = self._next_point_via_EI(n_restarts=60)
            _scf_log("BO-P2", f"[EI {i+1}/{n_iterations}]"
                              f" u={u:.2f}  g_JT={gJT:.4f}  t_pd={t_pd:.4f}"
                              f"  → t0={t_pd**2/dct:.4f}")
            pt = self._evaluate_material(u, gJT, t_pd)
            self._register(pt)
            done_count[0] += 1
            tick()
            if verbose and (i + 1) % 5 == 0 and self.observations:
                _scf_log("BO-P2", f"best so far: "
                                  f"{max(self.observations, key=lambda o: o.score)}")

        best, best_valid, best_raw = self._pick_best(self.observations)
        elapsed = _time.time() - t0
        n_excl  = sum(1 for o in self.observations
                      if getattr(o, '_exclude_from_gp', False))
        _scf_log("BO-P2", "="*60)
        _scf_log("BO-P2", f"Stage-2 done ({elapsed/60:.1f} min)  |  Best: {best}")
        _scf_log("BO-P2", f"GP set: {len(self._gp_obs)}/{len(self.observations)}"
                          f"  ({n_excl} excluded by hard constraint)")
        _scf_log("BO-P2", "="*60)
        return {'best_point': best, 'best_valid': best_valid, 'best_raw': best_raw,
                'observations': self.observations, 'elapsed_s': elapsed}

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
        chi_tau = result.get('chi_tau', 0.0)
        Ut_ratio = result.get('Ut_ratio', 0.0)

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
            _M_d = float(_res.get('M', 0.1))
            _Gd  = solver.compute_G_instability(_d, M=_M_d)
            _Tcl_res = solver.compute_Tc_by_gap_suppression(_d, sc_result=_res)
            _Tcl = float(_Tcl_res.get('Tc', 0.0))
            _gr_d = solver.compute_gap_ratio(_d, _res)
            _tc_list.append(_Tcl)
            _g22_list.append(_Gd.get('G22', float('nan')))
            _lmin_list.append(_Gd.get('lambda_min', float('nan')))
            _ratio_list.append(_gr_d['ratio_2D'])
            _scf_log("SCAN", f"  {_d:6.3f}  {_Tcl*1000:12.2f}  "
                  f"{_Gd.get('Tc_estimate',0)*1000:12.2f}  "
                  f"{_Gd.get('lambda_min',float('nan')):8.4f}  "
                  f"{_Gd.get('G22',float('nan')):8.4f}  "
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
        _scf_log("CF-SCAN", f"χ₀={cf_result.get('chi0',float('nan')):.4f}"
              f"  |  RPA factor = {cf_result.get('rpa_factor', float('nan')):.3f}×")
        _irr = cf_result.get('irrep_info', {})
        _scf_log("CF-SCAN", f"Irrep R={_irr.get('selection_ratio',float('nan')):.4f} "
              f"JT {'ALLOWED ✓' if _irr.get('jt_algebraically_allowed', False) else 'BLOCKED ✗'}")

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
            scores  = [getattr(o, 'score', o.Delta_total) for o in all_obs]
            dopings = [o.doping for o in all_obs]
            dt_vals = [o.Delta_tetra for o in all_obs]
            lJT_vals = [getattr(o, 'lambda_JT', 0.0) for o in all_obs]
            colours = ['green' if 0.05 < lj < 1.0 else ('orangered' if lj >= 1.0 else 'orange') for lj in lJT_vals]

            conv_mask = [getattr(o, 'converged', False) for o in all_obs]
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
    ║  Stage-1 BO: (Δ_tet, λ_soc, u)  |  Stage-2 BO: (g_JT, t_pd)     ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """, flush=True)

    os.environ.setdefault("OMP_NUM_THREADS",      "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS",      "1")

    params = ModelParams(
        t_pd         = 0.573,
        u            = 4.852,
        lambda_soc   = 0.244,
        Delta_tetra  = -0.94,
        g_JT         = 0.265,
        K_lattice    = 2.971,
        lambda_hop   = 1.18,
        eta          = 0.22,
        Delta_CT     = 1.400,
        omega_JT     = 0.057,
        rpa_cutoff   = 0.09,
        Delta_inplane= 0.03,
        mu_LM        = 6.8,
        ALPHA_HF     = 0.12,
        Z            = 4,
        nk           = 90,
        kT           = 0.015,
        a            = 1.0,
        max_iter     = 300,
        tol          = 1e-4,
        mixing       = 0.035,
    )
    params.summary()
    solver = RMFT_Solver(params)
    target_doping = 0.15
    supposed_M    = solver._estimate_M0(target_doping)
    initial_Q     = 1e-5
    initial_Delta = 1e-5
    min_doping    = 0.06
    max_doping    = 0.24

    # ── Section 1: G-matrix at reference parameters ───────────────────────────
    _scf_log("G-MATRIX", "="*60)
    G_base = solver.compute_G_instability(target_doping=target_doping, M=supposed_M)
    _scf_log("G-MATRIX", f"h_afm={G_base['h_afm']:.4f} eV")
    _scf_log("G-MATRIX", f"χ_ΔΔ (dom)={G_base['chi_DD_dom']:.4f}  χ_DD_s={G_base['chi_DD_s']:.4f}"
             f"  χ_DD_d={G_base['chi_DD_d']:.4f}  χ_DD_sd={G_base['chi_DD_sd']:.4f}  [eV⁻¹]")
    _scf_log("G-MATRIX", f"χ_ΔQ (dom)={G_base['chi_DQ_dom']:.4f}  χ_ΔQ_s={G_base['chi_DQ_s']:.4f}"
             f"  χ_ΔQ_d={G_base['chi_DQ_d']:.4f}  [eV⁻¹]")
    _scf_log("G-MATRIX", f"χ_QQ(normal)={G_base['chi_QQ']:.4f} eV⁻¹"
             f"  χ_QQ(SC probe)/normal={G_base['chi_QQ_sc'] / max(G_base['chi_QQ'], 1e-12):.3f}×")
    _scf_log("G-MATRIX", f"  → {('✓ GENUINE SC-TRIGGERED JT CONFIRMED' if G_base['sc_triggered_jt'] else '— SC-triggered JT not yet confirmed at this Δ_probe')}")
    _scf_log("G-MATRIX", f"N_eff={G_base['N_eff']:.4f} eV⁻¹  K_eff={G_base['K_eff']:.4f} eV/Å²")
    _scf_log("G-MATRIX", f"3×3 eigs: [{G_base['eigs3'][0]:.4f},{G_base['eigs3'][1]:.4f},{G_base['eigs3'][2]:.4f}]")
    _scf_log("G-MATRIX", f"evec_min=[{G_base['evec_min'][0]:.3f},{G_base['evec_min'][1]:.3f},"
             f"{G_base['evec_min'][2]:.3f}]  → instab_dir: {G_base['instab_dir']}")
    _scf_log("G-MATRIX", f"G11={G_base['G11']:.4f}  G3[2,2]={G_base['G22']:.4f}  G12={G_base['G12']:.4f}"
             f"  dom={G_base['dominant']}")
    _lmin_val = G_base['lambda_min']
    _g22_val  = G_base['G22']
    _lmin_note = ("✗ SPONTANEOUS instability — hard-rejected by scorer" if _lmin_val <= 0
                  else ("⚠ near-critical (0 < λ_min < 0.1)" if _lmin_val < 0.1
                        else "✓ normal-state stable"))
    _g22_note  = ("✓ G22>0: spontaneous JT blocked" if _g22_val > 0
                  else "✗ G22≤0: spontaneous JT risk")
    _scf_log("G-MATRIX", f"λ_min={_lmin_val:.4f}  [{_lmin_note}]")
    _scf_log("G-MATRIX", f"G3[2,2]={_g22_val:.4f}  [{_g22_note}]")
    _lambda_eff = G_base['lambda_eff']
    _lambda_eff_status = ("✓ optimal" if 0.3 < _lambda_eff < 1.0
                       else ("⚠ weak — increase J_eff (↓u or ↑t_pd/Δ_CT)" if _lambda_eff <= 0.3
                             else "⚠ too strong — risk of spontaneous JT / AFM QCP"))
    _scf_log("G-MATRIX", f"λ_eff=N_eff·V_eff={_lambda_eff:.4f}  [{_lambda_eff_status}]"
             f"  (target: 0.3 < λ_eff < 1.0; to raise: ↓u or ↑t_pd; to lower: ↑K_lattice)")
    # J_eff * chi_SS < 1 required for stable paramagnon vertex (V_RPA not past Stoner QCP).
    _g_t_log, _g_J_log, _f_d_log, _ = solver.get_gutzwiller_factors(target_doping)
    _tx_log, _ty_log = solver.effective_hopping_anisotropic(initial_Q)
    _J_eff_log = solver.effective_superexchange(_g_J_log, _tx_log, _ty_log, target_doping)
    _stoner_proxy = _J_eff_log * G_base['chi_DD_s']
    _stoner_note  = ("✓ <1 stable" if _stoner_proxy < 1.0
                     else ("⚠ >1 near/past AFM QCP → V_RPA may go negative" if _stoner_proxy < 2.0
                           else "✗ >>1 deeply past AFM QCP"))
    _scf_log("G-MATRIX", f"J_eff={_J_eff_log:.4f} eV  J_eff·χ_SS(proxy)={_stoner_proxy:.3f}  [{_stoner_note}]"
             f"  (to reduce: ↑Δ_CT or ↑u)")
    _scf_log("G-MATRIX", f"||[τ_x,H]||={G_base['comm_norm']:.4f} eV  blocking={G_base['blocking_ratio']:.4f}")

    # Linearised gap equation: λ_max with full RPA vertex + Gutzwiller correction
    _g_t_ref, _g_J_ref, _, _ = solver.get_gutzwiller_factors(target_doping)
    _tx_ref, _ty_ref = solver.effective_hopping_anisotropic(initial_Q)
    _mu_ref = -2.0 * _g_t_ref * solver.p.t0 * (1.0 - 2.0 * max(abs(target_doping), 1e-6))
    _lin = solver.solve_linearized_gap_equation(
        supposed_M, initial_Q, 0.0+0j, 0.0+0j,
        target_doping, _mu_ref,
        _g_t_ref * _tx_ref, _g_t_ref * _ty_ref, _g_J_ref)
    _V_spin = _lin['V_spin_mean']
    _V_JT   = _lin['V_JT_mean']
    _V_cr   = _lin['V_cross_mean']
    _V_tot  = _lin['V_rpa_mean']
    _scf_log("G-MATRIX", "Linearised gap equation (full RPA vertex, Gutzwiller-corrected) [PRE-SCF, Δ=0 seed]:")
    _scf_log("G-MATRIX", f"  λ_max={_lin['lambda_max']:.4f} (raw={_lin['lambda_max_raw']:.4f}"
             f"  × g_Δ={_lin['g_delta_dom']:.3f})  sym={_lin['gap_symmetry']}")
    # V_RPA(FS-avg): average of V(q=k_i−k_j) over FS pairs.
    # For d-wave this is usually NEGATIVE: forward scattering (q≈0) repulsive, back-scattering (q≈(π,π)) attractive.
    # λ_max already accounts for the d-wave sign via φ_d(k), so a negative FS-average is normal for B1g.
    # λ_max indicates the pairing strength, but λ_max absolute value unreliable pre-SCF (log(1/T) divergence + g_Δ amplification
    # Reliable: gap symmetry (B1g/A1g), sign (>0 = attractive channel exists), d vs s competition.
    if abs(_V_tot) > 1e-4:
        _f_spin = _V_spin / _V_tot
        _f_JT   = _V_JT   / _V_tot
        _f_cr   = _V_cr   / _V_tot
        _v_sign = "⚠ negative avg (normal for d-wave)" if _V_tot < 0 else "✓ positive avg"
        _scf_log("G-MATRIX", f"  V_RPA(FS-avg)={_V_tot:.4f} eV  [{_v_sign}]"
                 f"  → spin={_f_spin*100:.0f}%  JT={_f_JT*100:.0f}%  cross={_f_cr*100:.0f}%")
    else:
        _scf_log("G-MATRIX", f"  V_RPA(FS-avg)={_V_tot:.2e} eV  [near RPA cancellation — "
                 f"spin={_V_spin:.3f}  JT={_V_JT:.3f}  cross={_V_cr:.3f} eV (absolute)]")

    _scf_log("G-MATRIX", "Two-level SC-JT test (Hessian):")
    _scf_log("G-MATRIX", f"  ∂²F/∂Q²|Δ=0={G_base['d2F_Q_normal']:+.4f} eV/Å²  "
             f"{'✓ normal-state Q-stable' if G_base['d2F_Q_normal'] > 0 else '✗ spontaneous JT (normal-state Q-soft!)'}")
    _scf_log("G-MATRIX", f"  ∂²F/∂Q²|Δ≠0={G_base['d2F_Q_sc']:+.4f} eV/Å²  "
             f"{'✓ SC softens Q-mode (JT triggered!)' if G_base['d2F_Q_sc'] < 0 else '— Q-mode still stiff in SC state'}")

    # ── Section 2: SC-JT window at REFERENCE parameters ─────────────────────
    # chi_tau is computed at the normal-state (Delta=0) seed; the SC-enhanced chi_tau (from the converged SCF) will be larger and is reported per doping point in the phase diagram scan.
    _chi_tau_ref = solver._compute_chi_tau(supposed_M, initial_Q, target_doping)['chi_tau']
    _lam_min_ref = max(G_base['lambda_min'], 1e-4)   # floor: avoids K_SC → ∞ at criticality
    _jt_win = check_sc_jt_window(
        g_JT      = solver.p.g_JT,
        Delta_CF  = solver.p.Delta_CF,
        chi_tau   = _chi_tau_ref,
        chi0      = G_base['chi_QQ'] / max(solver.p.g_JT**2, 1e-12),
        K_lattice = solver._K_bare,
        K_eff     = G_base['K_eff'],
        lambda_min = _lam_min_ref,
    )
    _scf_log("G-MATRIX", "SC-JT window  [normal-state χ_τ; SC-enhanced χ_τ reported per SCF point]:")
    _scf_log("G-MATRIX", f"  K_spont={_jt_win['K_spont']:.4f}  K_SC={_jt_win['K_SC']:.4f}"
             f"  K_opt(geom)={_jt_win['K_opt']:.4f}  K_lattice={solver._K_bare:.4f}")
    _scf_log("G-MATRIX", f"  λ_JT={_jt_win['lambda_JT']:.4f}  λ_JT_opt={_jt_win['lambda_JT_opt']:.4f}"
             f"  K_dist(log)={_jt_win['K_distance']:+.3f}"
             f"  K_in_window={_jt_win['K_in_window']}  window_open={_jt_win['window_open']}")
    _scf_log("G-MATRIX", f"  → {_jt_win['note']}")
    _scf_log("G-MATRIX", "="*60)

    # ── Section 3: Stage-1 BO: (Δ_tet, λ_soc, u) ─────────────────────────────
    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", "STAGE 1 — (Δ_tet, λ_soc)  |  u, t_pd, g_JT, K_lattice fixed")
    _scf_log("MAIN", "Rationale: Δ_CF=f(Δ_tet,λ_soc) via SOC+CF diag; u fixed here")
    _scf_log("MAIN", "Scoring: Tc × sc_hessian_f[λ_min(H_3x3)] × physics gates")
    _scf_log("MAIN", "Hard constraint: G22≤0 at Δ=0 → score=0, excluded from GP")
    _scf_log("MAIN", "="*60)

    bo = BayesianOptimizer(solver, n_doping_scan=7)
    res1 = bo.optimize(
        doping_bounds      = (min_doping, max_doping),
        Delta_tetra_bounds = (-0.16, -0.05),
        lsoc_bounds        = (0.15,   0.30),
        n_initial          = 30,
        n_refine           = 10,
        n_iterations       = 82,
        verbose            = True,
    )
    best1 = res1['best_valid'] or res1['best_point']
    best_lsoc = best1.lambda_soc or params.lambda_soc

    # SC-triggered JT causality test on top-5
    _scf_log('MAIN', 'SC-triggered JT causality test (top-5 Stage-1 points)...')
    for top_pt in sorted(res1['observations'],
                         key=lambda o: o.score, reverse=True)[:5]:
        if top_pt.result and top_pt.converged:
            s_test = copy.copy(solver); s_test.p = copy.copy(solver.p)
            s_test.p.Delta_tetra = top_pt.Delta_tetra
            s_test.p.lambda_soc  = top_pt.lambda_soc or params.lambda_soc
            s_test.p.u           = top_pt.u
            s_test.p.__post_init__()
            s_test._K_bare = s_test.p.K_lattice
            s_test._rebuild_orbital_operators(s_test.p)
            s_test._reset_transient_state()
            ct = bo._jt_causality_test(s_test, top_pt.result)
            _scf_log("CAUSAL",
                     f"Δ_tet={top_pt.Delta_tetra:.3f}  λ_soc={top_pt.lambda_soc:.4f}"
                     f"  u={top_pt.u:.2f}  Tc={top_pt.Tc*1000:.2f}meV → {ct['note']}")
            _scf_log("CAUSAL",
                     f"  G22_N={ct['G22_normal']:.4f}"
                     f"  λ_min(H)={ct['lmin_sc_hessian']:+.4f}"
                     f"  hess={'✓' if ct['hess_confirmed'] else '✗'}"
                     f"  L1={'✓' if ct.get('level1_ok') else '✗'}"
                     f"  L2={'✓' if ct.get('level2_ok') else '✗'}")

    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", f"STAGE 1 COMPLETE  ({res1['elapsed_s']/60:.1f} min)")
    _scf_log("MAIN", f"  Δ_tet={best1.Delta_tetra:.4f}  λ_soc={best_lsoc:.4f}")
    _scf_log("MAIN", f"  Tc={best1.Tc*1000:.2f} meV  |Δ|={best1.Delta_total:.6f} eV"
                     f"  score={best1.score:.6f}")

    params.Delta_tetra = best1.Delta_tetra
    params.lambda_soc  = best_lsoc
    # u stays at initial value; Stage-2 will optimise it
    params.__post_init__()

    # ── Section 4: Stage-2 BO: (g_JT, t_pd) ──────────────────────────────────
    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", "STAGE 2 — (u, g_JT, t_pd)  |  Δ_tet, λ_soc, K_lattice fixed")
    _scf_log("MAIN", "Rationale: correlation + pairing + hopping; Δ_CF fixed from Stage-1")
    _scf_log("MAIN", "="*60)

    bo2 = BayesianOptimizerPhase2(
        solver           = RMFT_Solver(params),
        n_doping_scan    = 7,
        best_Delta_tetra = best1.Delta_tetra,
        best_lsoc        = best_lsoc,
        best_u           = best1.u,
    )
    res2 = bo2.optimize(
        doping_bounds = (min_doping, max_doping),
        u_bounds      = (3.5,  6.8),
        gJT_bounds    = (0.24, 0.31),
        t_pd_bounds   = (0.54, 0.88),
        n_initial     = 20,
        n_iterations  = 40,
        verbose       = True,
    )
    best2 = res2['best_valid'] or res2['best_point']
    elapsed_total = res1['elapsed_s'] + res2['elapsed_s']

    _scf_log("MAIN", "="*60)
    _scf_log("MAIN", f"OPTIMISATION COMPLETE  ({elapsed_total/60:.1f} min total)")
    _scf_log("MAIN", "Global optimal parameters:")
    _scf_log("MAIN", f"  Δ_tet={best2.Delta_tetra:.4f}  λ_soc={best_lsoc:.4f}"
                     f"  u={best2.u:.4f}  g_JT={best2.g_JT:.4f}"
                     f"  t_pd={best2.t_pd:.4f} eV"
                     f"  K_latt={params.K_lattice:.4f} eV/Å²")
    _scf_log("MAIN", f"  |Δ|={best2.Delta_total:.6f} eV  Tc={best2.Tc*1000:.2f} meV"
                     f"  score={best2.score:.6f}")

    params.u    = best2.u
    params.g_JT = best2.g_JT
    params.t_pd = best2.t_pd
    params.__post_init__()
    solver_opt = RMFT_Solver(params)

    try:
        _sc_opt = solver_opt.solve_self_consistent(best2.doping, initial_M=supposed_M, initial_Q=initial_Q, initial_Delta=initial_Delta, verbose=False)
        _M_opt  = float(_sc_opt.get('M', supposed_M))
        _G_opt  = solver_opt.compute_G_instability(best2.doping, M=_M_opt)
        _Tc_res = solver_opt.compute_Tc_by_gap_suppression(
            best2.doping, sc_result=_sc_opt)
        _Tc_lin = float(_Tc_res.get('Tc', 0.0))
        _hess   = _sc_opt.get('hessian', {})
        _lmin   = _hess.get('min_curvature', float('nan')) if _hess else float('nan')
        _scf_log("MAIN", "── Diagnostics at global optimum ──")
        _scf_log("MAIN",
                 f"Tc(BdG bisect)={_Tc_lin*1000:.2f} meV ({_Tc_lin*11604:.1f} K)")
        _scf_log("MAIN",
                 f"Tc(G-BCS)={_G_opt['Tc_estimate']*1000:.2f} meV"
                 f"  λ_min(G3,Δ=0)={_G_opt['lambda_min']:.4f}"
                 f"  G22={_G_opt['G22']:.4f}")
        _scf_log("MAIN",
                 f"SC Hessian λ_min(H_3x3,Δ≠0)={_lmin:+.4f}"
                 f"  {'✓ SC-triggered JT confirmed' if _lmin < 0 else '— JT not triggered'}")
        _gr = solver_opt.compute_gap_ratio(best2.doping, _sc_opt)
        _scf_log("MAIN", f"2Δ₀/kTc = {_gr['ratio_2D']:.3f}  [{_gr['coupling_regime']}]"
                         f"  Δ₀={_gr['Delta_0']*1000:.2f} meV  Tc={_gr['Tc_K']:.1f} K")

        # λ_max(T) curve
        try:
            _lT = solver_opt.compute_lambda_vs_T(best2.doping, _sc_opt)
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
            _sc_opt.get('Delta_s', 0.0) + _sc_opt.get('Delta_d', 0.0)),
        doping_range  = np.linspace(min_doping, max_doping, 10),
        opt_result    = {'observations': res2['observations']},
    )
    plt.show()