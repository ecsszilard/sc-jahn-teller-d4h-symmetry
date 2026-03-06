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

The "attractive interaction" does not arise from an a priori constant: it comes from the fact that `⟨τ_x⟩ = 0` in the normal state while `⟨τ_x²⟩ ≠ 0`, and the Cooper-pair mixing opens a channel in which the JT distortion is symmetry-allowed but only cooperatively accessible — without the SC condensate, it is symmetry-blocked.

The symmetry selection rules that encode this:

| State | Condition | Meaning |
|---|---|---|
| AFM ground state | Γ_JT ⊄ Γ_AFM ⊗ Γ_AFM | JT **forbidden** |
| SC condensate | Γ_JT ⊂ Γ_pair ⊗ Γ_pair | JT **allowed** |

For B₁g-symmetry Cooper pairs: the SC condensate transfers the order parameter into an irrep channel that is self-closing under the tensor product with the Cooper-pair irrep family, and in which rank-2 multipolar operators — including the B₁g JT mode — are no longer forbidden.

---

## Theoretical Framework

### 1. Local Hilbert Space and SOC+CF Hamiltonian

The full SOC + D₄h crystal-field Hamiltonian is constructed and diagonalized explicitly in the t₂g manifold (6×6):

```
H = λ_SOC · L·S  +  Δ_axial · Lz²  +  Δ_inplane · (Lx² − Ly²)
```

This diagonalization yields the Γ₆–Γ₇ splitting `Δ_CF` as a **derived quantity** (not a free parameter). The SOC eigenbasis `U_gamma` and 4-dim projector `_U4 = U_gamma[:, 0:4]` are precomputed in `__post_init__` so that all orbital operators (P₆, P₇, τ_x) are automatically consistent with the actual diagonalization. The four-component local basis is `[Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓]`.

- `Δ_axial = Δ_tetra · Lz²` — controls the Γ₆–Γ₇ gap; **required < 0** (tetragonal compression, c < a).
- `Δ_inplane = Δ_inplane · (Lx² − Ly²)` — splits the Γ₇ quartet into two Kramers doublets (Γ₇a, Γ₇b) without removing Kramers degeneracy, preventing spontaneous JT in the normal state.

When `lambda_soc`, `Delta_tetra`, or `Delta_inplane` are changed on a solver clone (e.g. in Phase-2 Bayesian optimisation), `p.__post_init__()` must be followed by `solver._rebuild_orbital_operators(p)` to keep P₆, P₇, τ₁₆, and `sz_bdg_op` consistent with the new eigenbasis.

### 2. ZSA Charge-Transfer Superexchange and Weiss Field

The AFM order originates from virtual pd-hopping processes, not from a Stoner Fermi-surface instability. The ZSA charge-transfer superexchange is:

```
J_CT = 2·t_pd⁴/Δ_CT² · (1/U + 1/(Δ_CT + U/2))
```

The two denominator terms represent the Mott channel (pd→dd, cost U) and the Zhang–Rice channel (pd→pp, cost Δ_CT + U/2) respectively. The bare Weiss-field amplitude is:

```
U_mf = Z · J_CT / 2
```

stored without Gutzwiller renormalization; `g_J · f_d` is applied at runtime in `build_local_hamiltonian_for_bdg`. The effective AFM splitting entering the BdG Hamiltonian is:

```
h_AFM = g_J · f(δ) · (U_mf/2 + Z·2t²/U) · M/2
```

where `f(δ) = δ/(δ + δ₀)` suppresses the unphysical g_J → 4 divergence near half-filling, and `Z·2t²/U` is the kinematic dd-exchange (second order in `t₀ = t_pd²/Δ_CT`). These two exchange contributions are physically distinct and do not double-count.

The regularization scale `δ₀` is not a free parameter — it is derived from the Zhang–Rice singlet spectral weight:
```
z_ZRS = t_pd² / (Δ_CT² + t_pd²),    δ₀ = z_ZRS / (1 − z_ZRS)
```

### 3. Primary Parameter: t_pd

The pd hybridisation integral `t_pd` is the primary hopping input. The effective dd hopping is derived as:

```
t₀ = t_pd² / Δ_CT
```

`t₀` is never set directly — it changes consistently whenever `t_pd` or `Δ_CT` changes via `__post_init__`. Phase-2 of the Bayesian optimiser searches over `t_pd`; `Δ_CT` is fixed as a material-class constant controlling the charge-transfer / multipolar fluctuation scale.

### 4. Gutzwiller Renormalization (Mott physics)

Near half-filling, the Mott insulator physics is captured via Gutzwiller factors as a function of doping δ = 1 − n:

```
g_t       = 2δ / (1 + δ)         # Kinetic energy suppression → 0 at half-filling
g_J       = 4 / (1 + δ)²         # Exchange enhancement → 4 at half-filling
g_Delta_s = g_t                   # On-site Γ₆⊗Γ₇ channel: same weight as kinetic
                                  #   (spin vertex renormalized separately inside RPA)
g_Delta_d = g_J                   # Inter-site d-wave B₁g: same vertex as superexchange
```

`g_Delta_s = g_t` avoids double-counting the spin-fluctuation vertex that is already applied inside `compute_gap_eq_vectorized`. `g_Delta_d = g_J` is strongest at half-filling and vanishes at large doping.

### 5. B₁g Jahn–Teller Distortion and Anisotropic Hopping

The B₁g mode breaks the x–y symmetry of the square lattice:

```
tx(Q) = t₀ · exp(+Q / λ_hop)    # x-bonds shorten → larger hopping
ty(Q) = t₀ · exp(−Q / λ_hop)    # y-bonds lengthen → smaller hopping
```

`K_lattice` is the **bare phonon spring constant** (primary input, eV/Å²), representing the lattice stiffness in the absence of any electronic feedback. The physically relevant stiffness during the SCF loop is the effective spring constant:

```
K_eff = K_lattice + ∂²F_ex/∂Q²
```

where `∂²F_ex/∂Q²` is the exchange-rigidity correction computed by `compute_JT_rigidity_from_exchange` via central finite-difference of `⟨O_α(Q)⟩`. This term is negative when the exchange free energy softens the JT mode (SC-triggered regime) and positive when AFM order stiffens it (normal state). `K_lattice` is never mutated; `K_eff` is stored as `_K_eff_scf` in `solve_self_consistent` and used for the adiabatic equilibrium `Q_out = −(g_JT/K_eff)·⟨τ_x⟩`.

The SC-triggered JT coupling strength is characterized by:
```
lambda_JT = (g_JT² / K_lattice) · chi_tau
```
The viable regime is `0.05 < lambda_JT < 1.0`: below 0.05 the JT channel is closed (strong AFM or large Δ_CF), above 1.0 the system enters strong-coupling / Eliashberg regime (BCS invalid, score penalized).

The full multipolar exchange tensor `J_αβ(Q)` is computed by `J_alpha_beta_Q`, which includes the Q-dependent B₁g channel opening via `sinh(2Q/λ)`. Both the Heisenberg A₁g and the SC-unlocked B₁g components enter `compute_JT_rigidity_from_exchange` consistently.

### 6. Dual B₁g Pairing Channels

Two symmetry-equivalent B₁g pairing channels are treated simultaneously with **independent strengths**:

- **Channel s** — on-site inter-orbital singlet (Γ₆⊗Γ₇ → B₁g via orbital indices, φ = 1):
  ```
  D_s = Δ_s · (|6↑⟩⟨7↓| − |6↓⟩⟨7↑|)
  V_s = g_Delta_s · g_JT² / K_lattice   (eV)
  ```

- **Channel d** — inter-site d-wave (φ(k) = cos kx − cos ky → B₁g in k-space):
  ```
  D_d = Δ_d · φ(k) · (|A:6↑⟩⟨B:7↓| − |A:6↓⟩⟨B:7↑|)
  V_d = g_Delta_d · g_JT² / K_lattice   (eV)
  ```

Both channels are treated by separate gap equations with channel-specific Gutzwiller factors and no double-counting.

### 7. 16×16 BdG Hamiltonian (doubled unit cell)

The particle–hole-symmetric BdG matrix is built in the Nambu basis:

```
Ψ = [Particle_A(4), Particle_B(4), Hole_A(4), Hole_B(4)]
```

where each 4-component block is `[Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓]`. The full 16×16 structure is:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T(k)       │  D_s      D_d        │   ← Particle sector
      │  T†(k)  H_B        │  D_d      D_s        │
      ├────────────────────┼─────────────────────┤
      │  D_s†   D_d†       │  −H_A*   −T*         │   ← Hole sector
      │  D_d†   D_s†       │  −T†*    −H_B*       │
      └────────────────────┴─────────────────────┘
```

The anisotropic hopping `tx ≠ ty` (B₁g JT) enters the kinetic block `T(k) = −2[tx cos kx + ty cos ky] · I₄`. An optional `O_expectation` array allows the local Hamiltonian to include the self-consistent JT orbital-mixing term. All BdG construction is handled by `VectorizedBdG._build_H_stack()`, which accepts an arbitrary k-point array and an optional pre-allocated output buffer.

### 8. Irrep Selection and SC-Activated JT

An algebraic irrep projector tracks how much the SC condensate has lifted the B₁g symmetry barrier:

```
P_eff = P₆ + w · P₇     where w = min(|Δ| / Δ_CF, 1)
```

- `w = 0`: pure AFM state, P_eff = P₆ only; τ_x is strictly off-diagonal → ⟨τ_x⟩ = 0, JT forbidden.
- `w → 1`: SC-mixed state, P_eff = P₆ ⊕ P₇; τ_x acquires diagonal elements → ⟨τ_x⟩ ≠ 0, JT unlocked.

The selection ratio `R = w · |⟨τ_x⟩| / τ_x,max` is tracked throughout the SCF loop as a diagnostic of JT activation. The Kubo route (perturbation theory in the 16-dim BdG space) gives zero at Q=0 due to an exact symmetry cancellation; the correct `χ_QQ` is computed via `_chi_QQ_matrix_elements` using the full Kubo response `_kubo_response(ev, ec, Q)`.

### 9. Exchange Rigidity: ∂²F_ex/∂Q²

`compute_JT_rigidity_from_exchange` computes the full second derivative of the exchange free energy with respect to the JT distortion coordinate Q:

```
∂²F_ex/∂Q² = 2·⟨O⟩ @ J @ (∂⟨O⟩/∂Q)  +  ⟨O⟩ @ (∂J/∂Q) @ ⟨O⟩
```

Both the Q-dependence of the multipolar expectation values `⟨O_α(Q)⟩` (via hopping anisotropy and τ_x mixing) and the Q-dependence of the exchange tensor `J_αβ(Q)` (via the B₁g channel opening) are included. The finite-difference step captures both contributions without separate Kubo calculations.

The sign convention is: a positive `∂²F_ex/∂Q²` means exchange stiffens the phonon mode (`K_eff > K_lattice`); a negative value means exchange softens it (`K_eff < K_lattice`), which in the SC condensate can drive `K_eff < 0` — the SC-triggered JT instability.

The commutator diagnostic `[τ_x, H_AFM]` measures how strongly the normal-state exchange blocks the B₁g channel (large commutator norm → strong blocking → JT suppressed in normal state).

### 10. Multipolar Susceptibility χ_τx via Finite-Difference BdG

The multipolar susceptibility entering `lambda_JT` is computed by finite-difference BdG rediagonalization:

```
chi_tau = |∂⟨τ_x⟩/∂(g_JT·Q)|   evaluated at Q ± δQ
```

At each perturbed Q value the full BdG is rediagonalized with both the hopping `t(Q±δQ)` **and** the AFM Weiss field `h_afm(Q±δQ)` recomputed consistently. This ensures the AFM Fermi surface reconstruction is self-consistent with the perturbation. The computation is performed once at post-convergence (not per SCF iteration).

### 11. RPA Spin-Fluctuation Enhancement

The static transverse spin susceptibility χ₀(q_AFM) is computed at q = (π, π) using BdG coherence factors. The even k-grid is exploited so that q_AFM = (π, π) maps each grid point exactly to another via the precomputed permutation `chi0_Q_idx` — eliminating a full second LAPACK call:

```
E(k+Q) = E_k_all[chi0_Q_idx],    V(k+Q) = V_k_all[chi0_Q_idx]
```

The RPA Stoner factor enhances the effective pairing interaction:

```
rpa_factor = 1 / max(1 − U_eff · χ₀,  rpa_cutoff)
```

When the Stoner denominator `1 − U·χ₀ ≤ 0` (AFM QCP crossed), `rpa_factor` is set to 1.0 and the `afm_unstable` flag is raised — the linear RPA is no longer valid in that regime.

### 12. G-Matrix: SC–JT Coupled Instability

The coupled SC–JT instability boundary is tracked via a 3×3 G-matrix (s-channel, d-channel, Q-mode) computed via `_compute_afm2band_susceptibilities` and assembled in `_build_G3_matrix`. Per-channel susceptibilities are computed with the appropriate form factors:

```
χ_ΔΔ^c = Σ_{k,s=±} [tanh(E/2T) / (2E)] · φ_c(k)²   (φ_s=1, φ_d=cos kx−cos ky)
χ_QQ   = computed via _chi_QQ_matrix_elements (Kubo, zone-centre)
χ_ΔQ^c = g_JT  · Σ_{k,s=±} [tanh(E/2T) / (2E)] · (±ξ_diff/√…) · φ_c(k)
```

The 3×3 G-matrix is:
```
G3 = ┌ 1 − gVs·χ_ss     0            −cs·χ_sQ ┐
     │     0         1 − gVd·χ_dd    −cd·χ_dQ │
     └ −cs·χ_Qs    −cd·χ_Qd     1 − χ_QQ/K_eff ┘
```
where `cs = √(gVs/K_eff)`, `cd = √(gVd/K_eff)`.

The key criterion is `G3[2,2] = 1 − χ_QQ/K_eff > 0`: satisfied in the normal state (no spontaneous JT), violated in the SC condensate (SC-triggered JT). The minimum eigenvalue `λ_min` of G3 tracks proximity to the instability boundary. The Schur complement of G22 gives the effective pairing enhancement diverging as `G22 → 0⁺`. Full diagnostics are available via `compute_G_instability()`.

### 13. Tc and Gap Ratio

Two independent Tc estimates are provided:

- `compute_Tc_by_gap_suppression`: bisects in T to find the temperature where `|Δ(T)| < Delta_tol` via full re-SCF at each temperature, using warm-starting from the converged `sc_result`.
- `compute_lambda_vs_T`: tracks the linearized gap eigenvalue `λ_max(T)` across a temperature array; Tc is extracted as the crossing `λ_max = 1`.

`compute_gap_ratio` combines both to report `2Δ₀ / k_B Tc`; values significantly above 3.52 (BCS weak-coupling) indicate SC-JT strong-coupling enhancement.

### 14. Variational Free Energy

The BdG grand potential per site (with 1/2 for doubled unit cell):

```
Ω_BdG = (1/2) Σ_{k,n} w_k [E_n(k) f(E_n) − T S(f_n)]
        + |Δ_s|²/(g_Delta_s · V_s) + |Δ_d|²/(g_Delta_d · V_d)
        + (K_eff/2)Q²
```

The condensation correction terms use **independent** Gutzwiller factors per channel: `g_Delta_s = g_t` for the on-site channel, `g_Delta_d = g_J` for the inter-site d-wave channel.

### 15. Analytic ∂F/∂M and ∂²F/∂M² (Single Diagonalization)

The gradient and curvature of the free energy with respect to the AFM order parameter are computed analytically from a **single BdG diagonalization** using second-order perturbation theory:

```
∂F/∂M  = Σ_{k,n} f_n · ⟨ψ_n|∂H/∂M|ψ_n⟩                                (Hellmann–Feynman)

∂²F/∂M² = Σ_{k,n} (∂f_n/∂E_n) · ⟨ψ_n|∂H|ψ_n⟩²                         (diagonal term)
         + Σ_{k,n≠m} (f_n − f_m)/(E_m − E_n) · |⟨ψ_n|∂H|ψ_m⟩|²        (off-diagonal term)
```

Since `∂H/∂M` is diagonal in the BdG basis, the matrix elements reduce to simple inner products. The Newton step for M uses the analytic curvature with Levenberg–Marquardt regularization.

### 16. Observables: BdG Thermal Averages

From the BdG eigensystem `{E_n, |ψ_n⟩}` with 16-component spinors:

| Observable | Formula |
|---|---|
| Density | ⟨c†c⟩ = Σ_n [\|u_n\|² f(E_n) + \|v_n\|² (1−f(E_n))], divided by 4 |
| Magnetization | ⟨S_z⟩ using orbital-dependent sz = [+1, −1, +η, −η] |
| Quadrupole ⟨τ_x⟩ | Σ_n [2 Re(u†_{Γ₆} u_{Γ₇}) f + 2 Re(v†_{Γ₆} v_{Γ₇})(1−f)] |
| Pairing s | F_AA = u_A[6↑] · v_A[7↓]* − u_A[6↓] · v_A[7↑]* (on-site) |
| Pairing d | F_AB = u_A[6↑] · v_B[7↓]* − u_A[6↓] · v_B[7↑]* (inter-site, φ(k) weight) |

All observables are computed in a single batched LAPACK call via `VectorizedBdG`, with full NumPy broadcasting over the k-axis.

### 17. Two-Site Cluster: Quantum Multipolar Fluctuations

Beyond the BdG mean field, a 2-site (A–B sublattice) cluster is exactly diagonalized at each iteration. The cluster Hamiltonian lives in the 16×16 tensor product space of the two 4-component sites:

```
H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)
          + J_eff · O_A ⊗ O_B
          + Z_boundary · (g_J · U_mf/2 + J_eff) · M_ext · (O_A ⊗ I + I ⊗ O_B)
```

where the multipolar operator `O = (P₆ + η·P₇) ⊗ σz`. The final magnetization blends BdG and cluster contributions via `CLUSTER_WEIGHT` (no separate `CLUSTER_WEIGHT` parameter — the weight is folded into `ALPHA_HF`):

```
M_fixpoint = (1 − ALPHA_HF) · M_BdG_cluster_blend + ALPHA_HF · M_newton
```

The cluster also computes both quadrupole observables: `⟨τ_x⟩` (classical) and `√⟨τ_x²⟩` (RMS including quantum fluctuations). Since `[τ_x, H_cluster] ≠ 0`, these are genuinely different. The cluster-to-BdG J ratio is tracked as `_cluster_j_renorm` and feeds back into `J_alpha_beta_Q` as a vertex renormalization.

### 18. Chemical Potential: Newton's Method with Analytic ∂n/∂μ

At each SCF iteration, μ is found by **Newton's method** using the analytic derivative:

```
∂n/∂μ = Σ_{k,n} w_k · f(E_n)(1−f(E_n)) / kT · (|u_A|² + |u_B|² + |v_A|² + |v_B|²)
```

computed from the same BdG eigensystem as n(μ). Brent's method is retained as a fallback if Newton diverges or lands on a flat region.

---

## Model Architecture

```
ModelParams  (dataclass)
    ├── Primary: t_pd, u, lambda_soc, Delta_tetra, g_JT, K_lattice,
    │            lambda_hop, eta, Delta_inplane, Delta_CT, omega_JT, rpa_cutoff
    ├── Derived by __post_init__: Delta_CF, t0, U, U_mf, J_CT, doping_0, _U4, U_gamma
    └── summary(δ)  — prints parameter diagnostics and K_spont vs K_lattice check

ClusterMF  (2-site exact diagonalization)
    ├── build_multipolar_operator(η)
    ├── build_cluster_hamiltonian(H_sp_A, H_sp_B, J_eff, M_ext, η, U_mf_stoner)
    └── cluster_expectation(evals, evecs, O, T, site_index)

VectorizedBdG  (performance kernel, lives inside RMFT_Solver)
    ├── _build_H_stack(kpts, ..., O_expectation=, out=)  → (N, 16, 16) BdG stack
    ├── compute_observables_vectorized(...)               → M, Q, n, Pair_s, Pair_d
    └── compute_gap_eq_vectorized(...)                    → Δ_s_new, Δ_d_new

RMFT_Solver
    ├── _rebuild_orbital_operators(params)     rebuild P₆, P₇, τ₁₆ after SOC/CF change
    ├── _reset_transient_state()               safe clone reset for parallel workers
    ├── get_gutzwiller_factors(δ)              → g_t, g_J, g_Delta_s, g_Delta_d
    ├── effective_hopping_anisotropic(Q)       → tx, ty
    ├── effective_superexchange(Q, g_J, ...)   → J(Q,δ)  [ZSA CT formula]
    ├── J_alpha_beta_Q(Q, lambda_hop)          → 4×4 multipolar exchange tensor
    ├── compute_JT_rigidity_from_exchange(...) → K_eff, d2F_ex_dQ2, blocking_ratio
    ├── compute_rank2_multipole_expectation(.) → selection ratio R
    ├── compute_static_chi0_afm(...)           → χ₀(q_AFM) via chi0_Q_idx permutation
    ├── _chi_QQ_matrix_elements(...)           → χ_QQ via Kubo route
    ├── _kubo_response(ev, ec, Q)              → complex Kubo kernel
    ├── _compute_afm2band_susceptibilities(...)→ all χ components for G3
    ├── _build_G3_matrix(chi, gVs, gVd, K_eff) → G3, eigenvalues, λ_min, instab_dir
    ├── _build_normal_state_bdg_cache(...)     → shared Δ=0 BdG diagonalisation
    ├── _build_gap_kernel(...)                 → Γ_{ij} pairing kernel
    ├── solve_linearized_gap_equation(...)     → λ_min, dominant channel, G3 diagnostics
    ├── build_local_hamiltonian_for_bdg(...)   → 4×4 H_A or H_B
    ├── compute_dF_dM_and_d2F(...)             → (∂F/∂M, ∂²F/∂M²) from single diag
    ├── compute_bdg_free_energy(...)           → Ω_BdG with per-channel g_Δ factors
    ├── compute_cluster_free_energy(...)       → F_cluster + observables
    ├── compute_G_instability(δ, M)            → G3 diagnostics, λ_min, Tc estimate
    ├── compute_Tc_by_gap_suppression(...)     → Tc via Brent bisection on Δ(T)
    ├── compute_lambda_vs_T(...)               → λ_max(T) curve, Tc at λ_max=1 crossing
    ├── compute_gap_ratio(...)                 → 2Δ₀/kTc strong-coupling diagnostic
    ├── compute_d2F_dQ2_at_Delta(...)          → SC-triggered JT causality check
    ├── _find_mu_for_density(...)              → Newton (analytic ∂n/∂μ) + Brent fallback
    ├── _compute_chi_tau(...)                  → finite-difference BdG χ_τx
    ├── _anderson_mix(...)                     → quasi-Newton convergence (M, Q)
    └── solve_self_consistent(...)             → main SCF loop with K_eff update

Optimisation (two-stage)
    ├── run_scf_material(solver, doping, Δ_tet, u, g_JT, t_pd, ...)
    │       calls __post_init__ → consistent derived params → solve_self_consistent
    ├── BayesianOptimizer  (Stage 1: 3D material space — Δ_tet, u, g_JT)
    │   ├── _cheap_scout(doping, Δ_tet, u, gJT, t_pd)   cheap filter (no full SCF)
    │   ├── _adaptive_seed_near_critical(n_refine)        biased seeding at λ_min ≈ 1
    │   ├── _evaluate_material(Δ_tet, u, gJT)            inner doping scan, warm-started
    │   ├── _g_fallback_score(...)                        G-matrix proximity score for Δ=0
    │   ├── _jt_coupling_strength(solver, result)         lambda_JT = (g²/K)·chi_tau
    │   ├── _jt_causality_test(solver, result)            SC-triggered JT verification
    │   ├── _score(Delta, converged, result, solver)      physics-motivated objective
    │   └── optimize(doping_bounds, Δ_tet_bounds, u_bounds, gJT_bounds, ...)
    │             Phase 1a: LHS seeding (parallel, ThreadPoolExecutor)
    │             Phase 1b: adaptive seeding near λ_min=1 (parallel)
    │             Phase 2:  GP EI acquisition (sequential, ARD Matérn ν=2.5)
    ├── BayesianOptimizerPhase2  (Stage 2: 2D — λ_soc, t_pd; inherits BayesianOptimizer)
    │   ├── _make_solver(lsoc, t_pd)   clone with __post_init__ + _rebuild_orbital_operators
    │   ├── _evaluate_material(lsoc, t_pd)
    │   └── optimize(doping_bounds, lsoc_bounds, t_pd_bounds, ...)
    └── (two-stage pipeline called directly from main())

Visualization
    ├── plot_phase_diagrams(solver, δ_scan, opt_result)   3×3 (or 4×3) panel figure
    ├── _plot_phase_data(ax, phase_data)                  phase diagram panel
    └── _plot_dos(ax, solver, result)                     DOS via vectorized BdG
```

---

## Key Algorithms

### VectorizedBdG: Batched LAPACK and Buffer Reuse

All BdG diagonalization is centralized in the `VectorizedBdG` class. Key optimizations:

- `_build_H_stack(kpts, out=)` accepts an optional pre-allocated `(N, 16, 16)` buffer. On the hot SCF path (full grid, `out=self._H_stack`), no heap allocation occurs per iteration.
- `compute_observables_vectorized` and `compute_gap_eq_vectorized` share the optional `_bdg_cache` tuple `(ev, ec)` from the per-iteration cache to avoid redundant diagonalizations.
- `_build_normal_state_bdg_cache` centralizes the Δ=0 BdG diagonalisation shared by `_build_gap_kernel` and `compute_gap_eq_vectorized`, eliminating a duplicate LAPACK call.

### Per-Iteration BdG Cache

Within each SCF iteration the BdG eigensystem `(ev, ec)` is computed **once** at the top and shared by:
1. Observable computation (`compute_observables_vectorized`)
2. Dual-channel gap equations (`compute_gap_eq_vectorized`)
3. ∂F/∂M and ∂²F/∂M² (`compute_dF_dM_and_d2F`)

The cache is stored in `self._scf_bdg_cache` and explicitly cleared after use to prevent stale reuse in subsequent iterations.

### Thread-Safety and Clone Protocol

`RMFT_Solver` is cloned with `copy.copy()` before each parallel SCF worker. The clone protocol is:
```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s.p.__post_init__()
s._K_bare = s.p.K_lattice
s._rebuild_orbital_operators(s.p)   # if lambda_soc or Delta_tetra changed
s._reset_transient_state()          # clear _vbdg, _scf_bdg_cache, _cluster_j_renorm
```

`_reset_transient_state` ensures each clone owns its own `VectorizedBdG` instance (and thus its own `_H_stack` buffer), preventing inter-worker memory aliasing. The `W` orbital weight matrix in `compute_JT_rigidity_from_exchange` is allocated locally per call for the same reason.

### χ₀(q_AFM): Permutation Trick

The even k-grid is constructed so that adding q_AFM = (π, π) maps each grid point exactly to another grid point. The precomputed index array `chi0_Q_idx` implements this as a free permutation:

```python
E_kQ_all = E_k_all[chi0_Q_idx]   # (N,16)  — no LAPACK, just index reorder
V_kQ_all = V_k_all[chi0_Q_idx]   # (N,16,16)
```

This eliminates an entire second LAPACK call compared to the naive approach.

### Dual k-Grid Setup

Two separate k-grids are maintained:

- **SCF / Simpson grid (odd, nk+1 points):** used for BdG diagonalization, observable computation, free energy, and gap equations. Composite 2D Simpson weights give O(h⁴) accuracy.
- **χ₀ grid (even, nk points, endpoint=False):** used exclusively for χ₀(q_AFM) and the pairing kernel, exploiting the permutation trick above.

### Anderson Mixing for Self-Consistency

The order parameters `[M, Q]` are updated via Anderson mixing (quasi-Newton without explicit Jacobian):

1. Compute BdG eigensystem (shared cache); extract observables (M_BdG, τ_x, Pair_s, Pair_d).
2. Update RPA factor from χ₀(q_AFM) (lazy: only when M or |Δ| change significantly).
3. Solve dual-channel gap equations with Newton–LM correction.
4. Periodically update `K_eff = K_lattice + ∂²F_ex/∂Q²` (every 5 iterations or when M changes by >0.02).
5. Compute `∂F/∂M` and `∂²F/∂M²` analytically from the cached eigensystem (single diag, no finite diff).
6. Blend Newton and BdG fixpoint: `M_out = (1−ALPHA_HF)·M_fixpoint + ALPHA_HF·M_newton`.
7. Apply Anderson update to `[M, Q]`; blend with simple mixing for safeguarding.
8. Reset Anderson history on Q sign flip (valley jump protection).
9. Adaptive mixing rate: if `max_diff` increases between steps, reduce `_alpha`; near SC critical point (`0.8 ≤ λ_max ≤ 1.8`), cap `_alpha` at `0.6·mixing`.

After convergence, a **post-convergence Hessian test** checks that all eigenvalues of the 3×3 `∂²F/∂{M,Q,Δ}²` matrix are positive (true minimum vs. saddle point).

### Two-Stage Bayesian Optimisation

**Stage 1 — BayesianOptimizer (3D: Δ_tetra, u, g_JT):**
`t_pd` and `lambda_soc` are held fixed; `Δ_CT` is a material-class constant. For each candidate, an inner doping scan runs `n_doping_scan` SCF calculations with warm-starting. Three sub-phases:

- **Phase 1a — LHS seeding** (`n_initial` materials): Latin Hypercube over the 3D space. Parallelized over available CPU cores via `ThreadPoolExecutor`.
- **Phase 1b — Adaptive seeding** (`n_refine` materials): biased toward `λ_min ≈ 1` using the G-matrix `lambda_min`; hard-excludes candidates with `G22 ≤ 0`. Parallelized.
- **Phase 2 — GP EI acquisition** (`n_iterations` materials): ARD Matérn(ν=2.5) GP fitted on per-material best scores, Expected Improvement maximized. Sequential (GP must be refitted after each evaluation).

The objective is physics-motivated:
```
score = Tc_proxy × conv_f × jt_f × stoner_f × lam_f × g22_f × evec_f × ratio_bonus
```
where `Tc_proxy` uses Brent-bisected Tc when available, `jt_f` peaks at `lambda_JT ≈ 0.65`, `g22_f` penalizes spontaneous JT (`G22 ≤ 0`), `evec_f` penalizes G3 instability eigenvectors dominated by the Q component, and `ratio_bonus` rewards strong-coupling `2Δ/kTc > 3.52`.

After Stage 1, the top-5 candidates are subjected to a **SC-triggered JT causality test** via `_jt_causality_test`: normal-state and SC fixpoints are compared via `compute_d2F_dQ2_at_Delta`, confirming that `∂²F/∂Q²|_{Δ=0} > 0` (Q-stable without SC) and `∂²F/∂Q²|_{Δ>0} < 0` (Q-unstable with SC).

**Stage 2 — BayesianOptimizerPhase2 (2D: λ_soc, t_pd):**
With `(Δ_tetra, u, g_JT)` fixed from Stage 1, searches `(lambda_soc, t_pd)`. Each clone calls `_rebuild_orbital_operators` after `__post_init__` to keep the SOC eigenbasis consistent. Uses the same GP-EI infrastructure inherited from `BayesianOptimizer`.

---

## Parameters

All energies in **eV**, lengths in **Å**.

> **Parameter design:** `t_pd` is the primary hopping input; `t₀ = t_pd²/Δ_CT` is always derived. `lambda_soc` and `Delta_tetra` are primary inputs; `Delta_CF` is derived by exact diagonalization of the 6×6 SOC+CF Hamiltonian. `Delta_tetra < 0` is required (tetragonal compression, D₄h). `K_lattice` is the bare phonon spring constant (primary input); `K_eff = K_lattice + ∂²F_ex/∂Q²` is computed at runtime. `doping_0` is derived from `t_pd` and `Delta_CT` (not a free parameter).

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | t_pd | 0.562 eV | pd hybridisation integral (primary hopping input) |
| `u` | u | 5.483 | Dimensionless U/t₀ ratio; Hubbard U = u·t₀ |
| `lambda_soc` | λ_SOC | 0.107 eV | Atomic SOC constant (t₂g shell) |
| `Delta_tetra` | Δ_tet | −0.094 eV | Tetragonal CF (D₄h compression, **required < 0**) |
| `Delta_inplane` | Δ_ip | 0.012 eV | B₂g in-plane anisotropy; splits Γ₇ quartet |
| `Delta_CT` | Δ_CT | 1.078 eV | Charge-transfer gap (ZSA scale); fixed during BO |
| `Delta_CF` | Δ_CF | derived | Γ₆–Γ₇ splitting from SOC+CF diagonalization |
| `t0` | t₀ | derived | Effective dd hopping = t_pd²/Δ_CT |
| `J_CT` | J_CT | derived | ZSA CT superexchange = 2t_pd⁴/Δ_CT²·(1/U+1/(Δ_CT+U/2)) |
| `U_mf` | U_mf | derived | Bare Weiss-field amplitude = Z·J_CT/2 |
| `doping_0` | δ₀ | derived | ZRS spectral weight scale = z_ZRS/(1−z_ZRS) |
| `g_JT` | g_JT | 1.048 eV/Å | Electron–phonon (JT) coupling |
| `K_lattice` | K | 3.817 eV/Å² | Bare phonon spring constant (no exchange); K_eff computed at runtime |
| `lambda_hop` | λ_hop | 1.2 Å | Hopping decay length: t(Q) = t₀·exp(±Q/λ_hop) |
| `eta` | η | 0.09 | AFM asymmetry: Γ₇ feels η×M vs Γ₆ |
| `omega_JT` | ω_JT | 0.060 eV | JT phonon frequency (40–80 meV) |
| `rpa_cutoff` | — | 0.09 | Stoner denominator floor |
| `mu_LM` | — | 6.8 | LM regularisation floor for M Newton step |
| `ALPHA_HF` | — | 0.12 | Blend weight: Newton vs BdG fixpoint for M update |
| `ALPHA_D` | — | 0.18 | Blend weight: Newton vs gap-equation fixpoint for Δ update |
| `mu_LM_D` | — | 2.9 | LM regularisation floor for Δ Newton step |
| `Z` | Z | 4 | Coordination number (2D square lattice) |
| `nk` | — | 84 | k-points per direction (must be even; odd nk+1 sub-grid for Simpson) |
| `kT` | kT | 0.011 eV | Temperature (~127.7 K) |
| `a` | a | 1.0 Å | Lattice constant |
| `max_iter` | — | 300 | Maximum SCF iterations |
| `tol` | — | 1e-4 | Convergence threshold |
| `mixing` | α | 0.035 | Linear mixing weight (Anderson safeguard blend) |

### Analytically Derived Parameter Constraints for SC+JT Coexistence

Three independent conditions must hold simultaneously:

**1. Metallicity** — the AFM gap must not swallow the Fermi surface:
```
h_AFM  =  g_J · f(δ) · (U_mf/2 + Z·2t²/U) · M/2  <  2·g_t·t₀
```

**2. Pairing scale** — the SC gap must exceed the thermal energy:
```
V_s · g_Delta_s  >>  kT     and     V_d · g_Delta_d  >>  kT
```

**3. JT stability in normal state** — the phonon must be stable before SC onset:
```
K_eff = K_lattice + ∂²F_ex/∂Q²  >  0    (checked via G3[2,2] > 0)
```
Note: `K_spont = g_JT²/Δ_CF` is the local atomic JT threshold (printed in `summary()` for orientation). The true stability criterion involves the full `K_eff` including exchange rigidity, accessible only after SCF via `compute_G_instability()`.

**4. SC–JT coupling** — the JT feedback must sit in the SC-triggered regime:
```
0.05  <  lambda_JT = (g_JT² / K_lattice) · chi_tau  <  1.0
```

All constraints are printed and checked by `compute_G_instability()` and `compute_JT_rigidity_from_exchange()` at runtime.

---

## Installation & Usage

### Requirements

```
numpy
scipy
matplotlib
scikit-learn   # optional — required for Bayesian optimisation (GP surrogate)
```

Install with:
```bash
pip install numpy scipy matplotlib scikit-learn
```

### Running the Simulation

```bash
python Quantum_AFM-multipolar_Jahn-Teller.py
```

On startup, the code:

1. Constructs and diagonalizes the SOC+CF Hamiltonian to derive Δ_CF, U_gamma, and related quantities.
2. Calls `params.summary()` to print all derived parameters and pre-SCF JT diagnostics.
3. Initializes `RMFT_Solver` with dual k-grids, Simpson weights, irrep projectors, and lazy `VectorizedBdG`.
4. Runs `BayesianOptimizer.optimize()` (Stage 1: 3D, parallel Phase 1a/1b + sequential GP-EI).
5. Runs a **SC-triggered JT causality test** on the top-5 Stage 1 candidates.
6. Runs `BayesianOptimizerPhase2.optimize()` (Stage 2: 2D λ_soc × t_pd).
7. Runs `plot_phase_diagrams()` at the Stage 2 optimal parameters.
8. Reports `compute_Tc_by_gap_suppression`, `compute_lambda_vs_T`, and `compute_gap_ratio` diagnostics.

---

## Output & Visualization

### Phase Diagram Figure (3×3 panels)

| Position | Content |
|---|---|
| [0,0] | Phase diagram: M, Q, Δ_s, Δ_d, \|Δ\| vs. doping δ with phase-region shading |
| [0,1] | Crystal-field sweet-spot: Δ_d, Q, M vs. actual Δ_CF (derived from Δ_tetra scan, twin-axis) |
| [0,2] | Density of States (DOS) via vectorized BdG; van Hove singularity detection |
| [1,0] | SCF convergence of M — one coloured line per doping point |
| [1,1] | SCF convergence of Q — one coloured line per doping point |
| [1,2] | SCF convergence of \|Δ\| — one coloured line per doping point |
| [2,0] | Free energy F_bdg and F_cluster vs. iteration (last doping point) |
| [2,1] | Gutzwiller factors g_t, g_J vs. iteration (last doping point) |
| [2,2] | Electron density n vs. iteration with target line (last doping point) |

If Bayesian optimisation results are provided, a 4th row is added:

| Position | Content |
|---|---|
| [3,0] | BO progress: Δ and score vs. evaluation index; running best; colour by lambda_JT regime |
| [3,1] | Doping δ vs. score scatter (green=SC-triggered JT, red=strong-coupling, orange=JT-closed) |
| [3,2] | Δ_tetra vs. score scatter |

### Iteration Log

Each iteration prints (via `_scf_log` with thread-safe locking): M, Q, Δ_s, Δ_d, density n, chemical potential μ, free energy F, χ₀(q_AFM), RPA factor, K_eff, and JT algebraic status.

### Convergence Report

At convergence: all converged order parameters, Hessian eigenvalues, G3-matrix diagnostics (λ_min, det(G3), dominant channel, Tc estimate), irrep selection ratio R, `compute_gap_ratio` (2Δ₀/kTc), causality test result, and whether the fixpoint is a true minimum or a saddle point.

---

## Known Limitations

| Approximation | Scope of validity | Known impact |
|---|---|---|
| No Pauli exclusion between cluster sites | Weak-coupling limit; not deep Mott | Slight overestimate of AFM correlations; controlled by ALPHA_HF blend |
| No charge-transfer fluctuations ⟨n_A n_B⟩ | CT insulator target regime | Charge fluctuations negligible when U_mf ≫ t |
| Static phonon (Q is a mean field) | Adiabatic limit ω_JT ≪ electronic scale | Zero-point quantum lattice fluctuations neglected |
| No spatial fluctuations | Mean-field in space | Cannot describe pseudogap, stripes, or phase separation |
| RPA static (ω = 0) only | Valid near AFM QCP, not deep in ordered phase | Dynamical vertex corrections neglected |
| K_eff updated every 5 SCF iterations | Near-adiabatic assumption | Back-action of Q on exchange rigidity is approximate during SCF transient |
| chi_tau computed at post-convergence only | Linearized JT response | Neglects self-consistent back-action of Q on chi_tau during SCF |

---

## References

The model implements the theoretical framework described in:

- Ecsenyi, S. (2026). *Multipolar superconductivity and coherent orbital mixing* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang et al. (1988). *Supercond. Sci. Technol.* 1, 36.
- Multi-orbital Gutzwiller: Bünemann, J., Weber, W. & Gebhard, F. (1998). *Phys. Rev. B* 57, 6896.
- ZSA classification: Zaanen, Sawatzky & Allen (1985). *Phys. Rev. Lett.* 55, 418.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.
- Bayesian optimisation / GP: Snoek, J., Larochelle, H. & Adams, R.P. (2012). *NeurIPS.*

---

*For questions or contributions, open an issue or pull request.*
