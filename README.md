# SC-Activated Jahn–Teller Model

> **A renormalized mean-field theory simulation of superconductivity-triggered B₁g Jahn–Teller distortion in a D₄h charge-transfer insulator with strong spin–orbit coupling.**

---

## Table of Contents

- [Physical Hypothesis](#physical-hypothesis)
- [Theoretical Framework](#theoretical-framework)
- [Model Architecture](#model-architecture)
- [Key Algorithms](#key-algorithms)
- [Parameters](#parameters)
- [Installation & Usage](#installation--usage)
- [Output & Visualization](#output--visualization)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Physical Hypothesis

In a standard picture, the Jahn–Teller (JT) effect *precedes* superconductivity: orbital degeneracy at the Fermi level drives a lattice distortion, which can then mediate Cooper pairing. This model inverts that logic.

**The central claim:** In a D₄h, charge-transfer-type, strongly correlated system where spin–orbit coupling (SOC) splits the local Hilbert space into Kramers doublets (Γ₆, Γ₇), a collinear AFM ground state stabilizes *only* dipolar (rank-1) multipolar order. This means:

- The Γ₆ ground manifold carries **no orbital quadrupole moment** (Q⁽²⁾ = 0),
- the B₁g Jahn–Teller distortion is **symmetry-forbidden** in the normal AFM state,
- Cooper-pair condensation creates a quasi-degenerate Γ₆–Γ₇ subspace,
- only in this paired subspace does **rank-2 multipolar order become accessible**, and
- the B₁g JT distortion emerges as an **induced response** of the superconducting condensate — not as a primary instability.

The symmetry selection rules that encode this:

| State | Condition | Meaning |
|---|---|---|
| AFM ground state | Γ_JT ⊄ Γ_AFM ⊗ Γ_AFM | JT **forbidden** |
| SC condensate | Γ_JT ⊂ Γ_pair ⊗ Γ_pair | JT **allowed** |

For Eg-symmetry Cooper pairs: Eg ⊗ Eg = A₁g ⊕ A₂g ⊕ **B₁g** ⊕ B₂g — the full JT spectrum is present.

---

## Theoretical Framework

### 1. Local Hilbert Space and SOC+CF Hamiltonian

The system is a Mott/charge-transfer insulator at half-filling (δ = 0). The full SOC + D₄h crystal-field Hamiltonian is constructed and diagonalized explicitly in the t₂g manifold (6×6):

```
H = λ_SOC · L·S  +  Δ_axial · Lz²  +  Δ_inplane · (Lx² − Ly²)
```

This diagonalization yields the Γ₆–Γ₇ splitting `Δ_CF` as a derived quantity (not a free parameter). The four-component local basis is `[6↑, 6↓, 7↑, 7↓]`. The Γ₆ doublet has no intrinsic quadrupole moment; it only develops one when coherently mixed with Γ₇ by the SC condensate.

- `Δ_axial = Δ_tetra · Lz²` — controls the Γ₆–Γ₇ gap independently of λ_SOC; negative values encode tetragonal compression.
- `Δ_inplane = Δ_inplane · (Lx² − Ly²)` — splits the Γ₇ quartet into two Kramers doublets (Γ₇a, Γ₇b) without removing Kramers degeneracy, preventing spontaneous JT in the normal state.

### 2. Stoner–Heisenberg Weiss Field

The mean-field AFM Zeeman splitting combines a Hartree–Fock Stoner term and a superexchange Heisenberg term, both renormalized by the Gutzwiller factor `g_J`:

```
h_AFM = g_J · f(δ) · (U_mf/2 + Z·2t²/U) · M/2
```

where `f(δ) = δ/(δ + δ₀)` is a regularization factor that suppresses the unphysical g_J → 4 divergence near half-filling. `U_mf = 0.5·Δ_CF` is the screened Weiss-field amplitude (ZSA charge-transfer scale, much smaller than the bare U). The Γ₇ orbital feels a reduced splitting `η × h_AFM` controlled by the asymmetry parameter `η`.

### 3. Gutzwiller Renormalization (Mott physics)

Near half-filling, the Mott insulator physics is captured via Gutzwiller factors as a function of doping δ = 1 − n:

```
g_t = 2δ / (1 + δ)      # Kinetic energy suppression → 0 at half-filling
g_J = 4 / (1 + δ)²      # Exchange enhancement → 4 at half-filling
g_Δ = g_t               # Anomalous amplitude renormalization
```

At δ = 0 the hopping is fully suppressed and the system is a Mott insulator with pure AFM order.

### 4. B₁g Jahn–Teller Distortion and Anisotropic Hopping

The B₁g mode breaks the x–y symmetry of the square lattice. Bond-length arguments give an exponential hopping law:

```
tx(Q) = t₀ · exp(+Q / λ_hop)    # x-bonds shorten → larger hopping
ty(Q) = t₀ · exp(−Q / λ_hop)    # y-bonds lengthen → smaller hopping
```

`λ_hop` is the hopping decay length set by M–O bond geometry (~1–2 Å for 3d-oxide perovskites) and is an explicit primary input. The phonon spring constant is:

```
K_lattice = g_JT² / (λ_JT · t₀)
```

where `λ_JT` is a dimensionless ratio. SC-triggered (not spontaneous) JT requires `λ_JT < Δ_CF / t₀`.

The effective superexchange is Q-dependent and regularized near half-filling:

```
J(Q, δ) = g_J · 4⟨t̃²⟩/U · f(δ)     where ⟨t̃²⟩ = (tx² + ty²)/2
```

### 5. Dual B₁g Pairing Channels

Two symmetry-equivalent B₁g pairing channels are treated simultaneously:

- **Channel s** — on-site inter-orbital singlet (Γ₆⊗Γ₇ → B₁g via orbital indices, φ = 1):
  ```
  D_s = Δ_s · (|6↑⟩⟨7↓| − |6↓⟩⟨7↑|)
  ```

- **Channel d** — inter-site d-wave (φ(k) = cos kx − cos ky → B₁g in k-space):
  ```
  D_d = Δ_d · φ(k) · (|A:6↑⟩⟨B:7↓| − |A:6↓⟩⟨B:7↑|)
  ```

The `channel_mix` parameter controls the V_s / V_d split of the total pairing strength `g_JT²/K`. Both channels have B₁g symmetry and are treated by separate gap equations with no double-counting.

### 6. 16×16 BdG Hamiltonian (doubled unit cell)

The particle–hole-symmetric BdG matrix is built in the Nambu basis:

```
Ψ = [Particle_A(4), Particle_B(4), Hole_A(4), Hole_B(4)]
```

where each 4-component block is `[6↑, 6↓, 7↑, 7↓]`. The full 16×16 structure is:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T(k)       │  D_s      D_d        │   ← Particle sector
      │  T†(k)  H_B        │  D_d      D_s        │
      ├────────────────────┼─────────────────────┤
      │  D_s†   D_d†       │  −H_A*   −T*         │   ← Hole sector
      │  D_d†   D_s†       │  −T†*    −H_B*       │
      └────────────────────┴─────────────────────┘
```

The anisotropic hopping `tx ≠ ty` (B₁g JT) enters the kinetic block `T(k) = −2[tx cos kx + ty cos ky] · I₄`.

### 7. Irrep Selection and SC-Activated JT

An algebraic irrep projector tracks how much the SC condensate has lifted the B₁g symmetry barrier:

```
P_eff = P₆ + w · P₇     where w = min(|Δ| / Δ_CF, 1)
```

- `w = 0`: pure AFM state, P_eff = P₆ only; τ_x is strictly off-diagonal → ⟨τ_x⟩ = 0, JT forbidden.
- `w → 1`: SC-mixed state, P_eff = P₆ ⊕ P₇; τ_x acquires diagonal elements → ⟨τ_x⟩ ≠ 0, JT unlocked.

The selection ratio `R = w · |⟨τ_x⟩| / τ_x,max` is tracked throughout the SCF loop as a diagnostic of JT activation.

### 8. RPA Spin-Fluctuation Enhancement

The static transverse spin susceptibility χ₀(q_AFM) is computed at q = (π, π) using BdG coherence factors:

```
χ₀ = (1/N) Σ_{k,n,m} |M_{nm}(k,k+Q)|² · (f_n − f_m) / (E_m − E_n)
```

The RPA Stoner factor enhances the effective pairing interaction near the AFM quantum critical point:

```
rpa_factor = 1 / max(1 − U_eff_chi · χ₀,  rpa_cutoff)
```

where `U_eff_chi = g_J · J_eff` (renormalized coupling, not bare U) keeps the denominator O(1) within the ordered phase. The susceptibility is updated lazily (only when M or |Δ| change by more than a threshold) to avoid abrupt V_eff jumps during iteration.

The multipolar susceptibility `χ_τx ~ N(0) / (1 + α·M²)` is also computed as a diagnostic for JT channel activation: it is suppressed when M is large (deep AFM) and recovers as M decreases toward the AFM QCP.

### 9. Observables: Correct BdG Thermal Averages

From the BdG eigensystem `{E_n, |ψ_n⟩}` with 16-component spinors:

| Observable | Formula |
|---|---|
| Density | ⟨c†c⟩ = Σ_n [\|u_n\|² f(E_n) + \|v_n\|² (1−f(E_n))], divided by 4 (sublattice + BdG doubling) |
| Magnetization | ⟨S_z⟩ using orbital-dependent sz = [+1, −1, +η, −η] |
| Quadrupole ⟨τ_x⟩ | Σ_n [2 Re(u†_{6} u_{7}) f + 2 Re(v†_{6} v_{7})(1−f)] |
| Pairing s | F_AA = u_A[6↑] · v_A[7↓]* − u_A[6↓] · v_A[7↑]* (on-site, no φ weight) |
| Pairing d | F_AB = u_A[6↑] · v_B[7↓]* − u_A[6↓] · v_B[7↑]* (inter-site, φ(k) weight in BdG) |

All observables use the `(1−2f)` factor for pairing amplitudes, which is critical for correct finite-temperature gap behavior and proper Tc estimation.

### 10. Two-Site Cluster: Quantum Multipolar Fluctuations

Beyond the BdG mean field, a 2-site (A–B sublattice) cluster is exactly diagonalized at each iteration. The cluster Hamiltonian lives in the 16×16 tensor product space of the two 4-component sites:

```
H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)
          + J_eff · O_A ⊗ O_B
          + Z_boundary · (g_J · U_mf/2 + J_eff) · M_ext · (O_A ⊗ I + I ⊗ O_B)
```

where the multipolar operator `O = (P₆ + η·P₇) ⊗ σz`. The boundary field matches the BdG Weiss field (both Stoner and Heisenberg terms renormalized by g_J).

The final magnetization blends BdG and cluster contributions via `CLUSTER_WEIGHT`:

```
M_fixpoint = (1 − CLUSTER_WEIGHT) · M_BdG + CLUSTER_WEIGHT · M_cluster
```

The cluster also computes both quadrupole observables:
- `⟨τ_x⟩`: classical expectation value
- `√⟨τ_x²⟩`: RMS quadrupole including quantum fluctuations

Since `[τ_x, H_cluster] ≠ 0`, these are genuinely different.

### 11. Variational Free Energy

The BdG grand potential per site (with 1/2 for doubled unit cell):

```
Ω_BdG = (1/2) Σ_{k,n} w_k [E_n(k) f(E_n) − T S(f_n)]
        + |Δ_s|²/(g_Δ·V_s) + |Δ_d|²/(g_Δ·V_d)
        + (K/2)Q²
```

The condensation correction terms `|Δ|²/(g_Δ·V)` correct for double-counted energy in the BdG mean-field decoupling. Equilibrium is found at `∂F/∂M = 0`, `∂F/∂Q = 0`.

### 12. Chemical Potential and Density Constraint

At each iteration, μ is found by Brent's root-finding method on:

```
n(μ; M, Q, Δ_s, Δ_d) − n_target = 0
```

with automatic bracket expansion up to ±6t₀ around the current guess.

---

## Model Architecture

```
_build_soc_cf_hamiltonian(λ, Δ_tet, Δ_ip)   → 6×6 H_SOC+CF (t2g manifold)
_gamma_splitting(...)                          → Γ₆–Γ₇a gap (derived Δ_CF)

ModelParams (dataclass)
    └── primary inputs → derived: Delta_CF, U, U_mf, K_lattice, t_pd
    └── summary()      → coexistence-window diagnostics

ClusterMF
    ├── build_multipolar_operator(η)            → 4×4 O = (P₆ + η·P₇)⊗σz
    ├── cluster_afm_exchange(J_eff, η)          → 16×16 J·O⊗O
    ├── boundary_afm_field(J_eff, M, η, U_mf, g_J)  → 4×4 Weiss field
    ├── build_cluster_hamiltonian(...)          → 16×16 full cluster H
    └── cluster_expectation(evals, evecs, O, T, site_index)  → thermal ⟨O⟩

RMFT_Solver
    ├── get_gutzwiller_factors(δ)               → g_t, g_J, g_Δ
    ├── effective_hopping_anisotropic(Q)        → tx, ty
    ├── effective_superexchange(Q, g_J, tx, ty, δ) → J(Q,δ)
    ├── dispersion(k, tx, ty)                   → γ(k) = −2[tx cos kx + ty cos ky]
    ├── build_irrep_selection_projector(Δ)      → P_eff (Γ₆ → Γ₆⊕Γ₇ lifting)
    ├── compute_rank2_multipole_expectation(Δ, τx_bdg) → selection ratio R
    ├── compute_static_chi0_afm(...)            → χ₀(q_AFM), Stoner denom, χ_τx
    ├── rpa_stoner_factor(chi0_result)          → rpa_factor ≥ 1
    ├── solve_gap_equation_k(...)               → Δ_s_new, Δ_d_new (dual channel)
    ├── build_local_hamiltonian_for_bdg(...)    → 4×4 H_A or H_B
    ├── build_pairing_block(Δ_s)               → 4×4 D_s (on-site orbital singlet)
    ├── build_inter_site_pairing_block(Δ_d, k) → 4×4 D_d (d-wave inter-site)
    ├── build_bdg_matrix(k, ...)               → 16×16 full BdG H(k)
    ├── compute_observables_from_bdg(...)      → M, Q, n, Pair_s, Pair_d
    ├── compute_bdg_free_energy(...)           → Ω_BdG per site
    ├── compute_cluster_free_energy(...)       → F_cluster + observables
    ├── solve_self_consistent(...)             → main SCF loop
    ├── compute_dF_dM(...)                     → ∂F/∂M via Hellmann–Feynman
    ├── compute_hessian(...)                   → post-convergence 3×3 Hessian
    ├── _anderson_mix(...)                     → quasi-Newton convergence (M, Q)
    ├── _find_mu_for_density(...)              → Brent root-finding for μ
    └── _compute_k_observables(...)            → weighted BZ integration

Visualization
    ├── plot_phase_diagrams(solver, δ_scan)    → 3×3 panel figure
    ├── _plot_phase_data(ax, phase_data)       → phase diagram panel
    └── _plot_dos(ax, solver, result)          → DOS with VHS detection
```

---

## Key Algorithms

### Dual k-Grid Setup

Two separate k-grids are maintained throughout the simulation:

- **SCF / Simpson grid (odd, nk+1 points):** used for BdG diagonalization, observable computation, free energy, and gap equations. Composite 2D Simpson weights give O(h⁴) accuracy.
- **χ₀ grid (even, nk points, endpoint=False):** used exclusively for the static spin susceptibility χ₀(q_AFM). The even grid guarantees that adding q_AFM = (π, π) to any grid point maps exactly to another grid point via the precomputed index `chi0_Q_idx[i]`, avoiding interpolation and aliasing errors.

### Anderson Mixing for Self-Consistency

The order parameters `[M, Q]` are updated via Anderson mixing (quasi-Newton without explicit Jacobian):

1. Compute BdG observables (M_BdG, τ_x, Pair_s, Pair_d) at current parameters.
2. Compute RPA factor from χ₀(q_AFM) (updated lazily when M or |Δ| change significantly).
3. Solve dual-channel gap equations: `Δ_s_new = V_s · g_Δ · Σ_k w_k · F_AA(k)` and `Δ_d_new = V_d · g_Δ · Σ_k w_k · φ(k) · F_AB(k)`.
4. Apply **Hellmann–Feynman Newton step** for M: `∂F/∂M = Σ_{k,n} f(E_n) ⟨ψ_n|∂H_BdG/∂M|ψ_n⟩` computed analytically; Newton step `M_newton = M − γ · ∂F/∂M` with Levenberg–Marquardt curvature `γ = 1/(|∂²F/∂M²| + μ_LM)`.
5. Blend Newton and BdG fixpoint: `M_out = (1−ALPHA_HF)·M_fixpoint + ALPHA_HF·M_newton`.
6. Apply Anderson update to `[M, Q]`; blend with simple mixing for safeguarding.
7. Reset Anderson history on Q sign flip (valley jump protection).
8. Apply **Levenberg–Marquardt Newton step** for Δ channels independently.

After convergence, a **post-convergence Hessian test** is run: if all eigenvalues of the 3×3 `∂²F/∂{M,Q,Δ}²` matrix are positive, the solution is a true thermodynamic minimum; negative eigenvalues indicate a saddle point.

### Brent Root-Finding for Chemical Potential

At each SCF iteration, μ is found by Brent's method on:

```
n(μ; M, Q, Δ_s, Δ_d) − n_target = 0
```

with automatic bracket expansion up to ±6t₀ around the current guess.

---

## Parameters

All energies in **eV**, lengths in **Å**.

> **Parameter design:** `lambda_soc` (atomic SOC, eV) and `Delta_tetra` (tetragonal CF, eV) are primary inputs; `Delta_CF` is derived by exact diagonalization of the 6×6 SOC+CF Hamiltonian. **`Delta_tetra < 0` is required**: it encodes tetragonal compression (c < a, D₄h point group), which is the symmetry prerequisite for the AFM-forbidden/SC-allowed JT asymmetry. `lambda_hop` is an explicit primary input because its physical scale (~1–2 Å) is set by M–O bond geometry. `K_lattice` is derived from `lambda_jt` and `t0` as `K = g_JT²/(lambda_jt·t0)`.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t0` | t₀ | 0.328 eV | Bare hopping integral |
| `u` | u | 4.821 | Dimensionless U/t₀ ratio; Hubbard U = u·t₀ |
| `lambda_soc` | λ_SOC | 0.144 eV | Atomic SOC constant (t₂g shell) |
| `Delta_tetra` | Δ_tet | −0.13 eV | Tetragonal CF (D₄h compression, **required < 0**) |
| `Delta_inplane` | Δ_ip | 0.02 eV | B₂g in-plane anisotropy; splits Γ₇ quartet into Γ₇a+Γ₇b |
| `Delta_CF` | Δ_CF | derived | Γ₆–Γ₇ splitting from SOC+CF diagonalization |
| `U_mf` | U_mf | derived (0.5·Δ_CF) | Screened Stoner Weiss-field amplitude |
| `g_JT` | g_JT | 0.774 eV/Å | Electron–phonon (JT) coupling |
| `lambda_jt` | λ_JT | 0.417 | Dimensionless spring-constant ratio: K = g_JT²/(λ_JT·t₀); must satisfy λ_JT < Δ_CF/t₀ |
| `K_lattice` | K | derived | Phonon spring constant (eV/Å²) |
| `lambda_hop` | λ_hop | 1.2 Å | Hopping decay length for B₁g anisotropy: t(Q) = t₀·exp(±Q/λ_hop) |
| `eta` | η | 0.09 | AFM asymmetry: Γ₇ feels η×M vs Γ₆ |
| `doping_0` | δ₀ | 0.09 | Superexchange regularization floor |
| `Delta_CT` | Δ_CT | 1.234 eV | Charge-transfer gap (ZSA scale); sets t_pd via t_pd² = t₀·Δ_CT |
| `omega_JT` | ω_JT | 0.060 eV | JT phonon frequency (40–80 meV physical range); enters only D_phonon = 2/ω_JT |
| `rpa_cutoff` | — | 0.12 | Stoner denominator floor: rpa_factor = 1/max(sd, cutoff) |
| `d_wave` | — | True | True → d-wave B₁g φ(k) = cos kx − cos ky; False → s-wave φ = 1 |
| `channel_mix` | — | 0.5 | Pairing channel mixing: 0 = pure on-site B₁g, 1 = pure d-wave B₁g |
| `mu_LM` | — | 5.0 | LM regularisation floor for M Newton step |
| `ALPHA_HF` | — | 0.16 | Blend weight: Newton vs BdG fixpoint for M update |
| `CLUSTER_WEIGHT` | — | 0.35 | Weight of cluster ED magnetization in M blend |
| `ALPHA_D` | — | 0.3 | Blend weight: Newton vs gap-equation fixpoint for Δ update |
| `mu_LM_D` | — | 1.0 | LM regularisation floor for Δ Newton step |
| `Z` | Z | 4 | Coordination number (2D square lattice) |
| `nk` | — | 80 | k-points per direction (even; Simpson grid uses nk+1) |
| `kT` | kT | 0.011 eV | Temperature (~127.7 K) |
| `a` | a | 1.0 Å | Lattice constant |
| `max_iter` | — | 200 | Maximum SCF iterations |
| `tol` | — | 1e-4 | Convergence threshold |
| `mixing` | α | 0.04 | Linear mixing weight (Anderson safeguard blend) |

### Analytically Derived Parameter Constraints for SC+JT Coexistence

Three independent conditions must hold simultaneously. Violating any one of them closes the SC+JT window.

**1. Metallicity** — the AFM gap must not swallow the Fermi surface:
```
h_AFM  =  g_J · f(δ) · (U_mf/2 + Z·2t²/U) · M/2  <  2·g_t·t₀
```

**2. Pairing scale** — the SC gap must exceed the thermal energy:
```
V_eff · g_Δ  =  (g_JT² / K) · g_Δ  =  λ_JT · t₀ · g_Δ  >>  kT
```

**3. SC–JT coupling** — SC condensate must drive Γ₆–Γ₇ mixing against the crystal-field gap:
```
V_eff  =  g_JT² / K  >  0.1 · Δ_CF
```

**4. JT stability** — the phonon must not go soft spontaneously before SC onset:
```
K  >  K_min = g_JT² / Δ_CF     equivalently:    λ_JT  <  Δ_CF / t₀
```

All constraints are printed and checked by `params.summary()` at runtime.

---

## Installation & Usage

### Requirements

```
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

### Running the Simulation

```bash
python Quantum_AFM-multipolar_Jahn-Teller.py
```

On startup, the code:

1. Constructs and diagonalizes the SOC+CF Hamiltonian to derive Δ_CF and related quantities.
2. Calls `params.summary()` to print all derived parameters and check coexistence conditions.
3. Initializes the RMFT_Solver with dual k-grids, Simpson weights, irrep projectors, and phonon propagator.
4. Runs `plot_phase_diagrams()` which:
   - Scans the specified doping range with warm-starting,
   - performs a crystal-field sweet-spot search at the midpoint doping,
   - produces a 3×3 figure panel (phase diagram, CF scan, DOS, convergence histories, free energy, Gutzwiller factors, density constraint).
5. Displays all figures via `plt.show()`.

---

## Output & Visualization

### Phase Diagram Figure (3×3 panels)

| Position | Content |
|---|---|
| [0,0] | Phase diagram: M, Q, Δ_d vs. doping δ with phase-region shading |
| [0,1] | Crystal-field sweet-spot: Δ_d, Q, M vs. Δ_CF at fixed doping (twin-axis) |
| [0,2] | Density of States (DOS) with van Hove singularity detection |
| [1,0] | SCF convergence of M — one coloured line per doping point |
| [1,1] | SCF convergence of Q — one coloured line per doping point |
| [1,2] | SCF convergence of \|Δ\| — one coloured line per doping point |
| [2,0] | Free energy F_bdg and F_cluster vs. iteration (last doping point) |
| [2,1] | Gutzwiller factors g_t, g_J vs. iteration (last doping point) |
| [2,2] | Electron density n vs. iteration with target line (last doping point) |

### Iteration Log

Each iteration prints: M, Q, Δ_s, Δ_d, density n, chemical potential μ, free energy F, χ₀(q_AFM), RPA factor, multipolar susceptibility χ_τx, and JT algebraic status (✓/✗).

### Convergence Report

At convergence, the solver prints: all converged order parameters, Hessian eigenvalues, irrep selection ratio R, and whether the fixpoint is a true minimum or a saddle point.

---

## Known Limitations

| Approximation | Scope of validity | Known impact |
|---|---|---|
| No Pauli exclusion between cluster sites | Weak-coupling limit; not deep Mott | Slight overestimate of AFM correlations; controlled by CLUSTER_WEIGHT |
| No charge-transfer fluctuations ⟨n_A n_B⟩ | CT insulator target regime | Charge fluctuations negligible when U_mf ≫ t |
| Static phonon (Q is a mean field) | Adiabatic limit ω_JT ≪ electronic scale | Zero-point quantum lattice fluctuations neglected |
| No spatial fluctuations | Mean-field in space | Cannot describe pseudogap, stripes, or phase separation |
| RPA static (ω = 0) only | Valid near AFM QCP, not deep in ordered phase | Dynamical vertex corrections neglected |

---

## References

The model implements the theoretical framework described in:

- Ecsenyi, S. (2026). *Multipolar superconductivity and coherent orbital mixing* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang et al. (1988). *Supercond. Sci. Technol.* 1, 36.
- ZSA classification: Zaanen, Sawatzky & Allen (1985). *Phys. Rev. Lett.* 55, 418.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.

---

*For questions or contributions, open an issue or pull request.*
