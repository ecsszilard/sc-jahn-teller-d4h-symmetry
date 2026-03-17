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
- Cooper-pair condensation creates a coherent Γ₆–Γ₇ superposition,
- only in this paired subspace does **rank-2 multipolar order become accessible**, and
- the B₁g JT distortion emerges as an **induced response** of the superconducting condensate — not as a primary instability.

The attractive interaction does not arise from an a priori constant: it comes from the fact that `⟨τ_x⟩ = 0` in the normal state while `⟨τ_x²⟩ ≠ 0`, and the Cooper-pair mixing opens a channel in which the JT distortion is symmetry-allowed but only cooperatively accessible — without the SC condensate, it is symmetry-blocked.

The symmetry selection rules:

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

When `lambda_soc`, `Delta_tetra`, or `Delta_inplane` are changed on a solver clone (e.g. in Bayesian optimisation), `p.__post_init__()` must be followed by `solver._rebuild_orbital_operators(p)` to keep P₆, P₇, τ₁₆, and `sz_bdg_op` consistent with the new eigenbasis.

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

where `f(δ) = δ/(δ + δ₀)` suppresses the unphysical g_J → 4 divergence near half-filling, and `Z·2t²/U` is the kinematic dd-exchange (second order in `t₀ = t_pd²/Δ_CT`). The regularization scale `δ₀` is derived from the Zhang–Rice singlet spectral weight:
```
z_ZRS = t_pd² / (Δ_CT² + t_pd²),    δ₀ = z_ZRS / (1 − z_ZRS)
```

### 3. Primary Parameter: t_pd

`t_pd` is the primary hopping input; the effective dd hopping `t₀ = t_pd² / Δ_CT` is always derived and never set directly. Phase-2 of the optimiser searches over `t_pd`; `Δ_CT` is fixed as a material-class constant.

### 4. Gutzwiller Renormalization (Mott Physics)

```
g_t       = 2δ / (1 + δ)         # kinetic energy suppression → 0 at half-filling
g_J       = 4 / (1 + δ)²         # exchange enhancement → 4 at half-filling
g_Delta_s = g_t                   # on-site Γ₆⊗Γ₇ channel: same weight as kinetic
g_Delta_d = g_J                   # inter-site d-wave B₁g: same vertex as superexchange
```

`g_Delta_s = g_t` avoids double-counting the spin-fluctuation vertex already applied inside `compute_gap_eq_vectorized`. `g_Delta_d = g_J` is strongest at half-filling and vanishes at large doping.

### 5. B₁g Jahn–Teller Distortion and Anisotropic Hopping

The B₁g mode breaks the x–y symmetry of the square lattice:

```
tx(Q) = t₀ · exp(+Q / λ_hop)
ty(Q) = t₀ · exp(−Q / λ_hop)
K_eff = K_lattice + ∂²F_ex/∂Q²
```

`K_lattice` is the **bare phonon spring constant** (primary input, eV/Å²). `∂²F_ex/∂Q²` is computed by `compute_JT_rigidity_from_exchange` via central finite-difference of `⟨O_α(Q)⟩`; negative when the SC condensate softens the JT mode. `K_lattice` is never mutated; `K_eff` is recomputed every 5 SCF iterations.

The SC-triggered JT coupling strength:
```
lambda_JT = (g_JT² / K_lattice) · chi_tau
```
The viable regime is `0.05 < lambda_JT < 1.0`. `chi_tau = |∂⟨τ_x⟩/∂(g_JT·Q)|` requires Δ≠0 and is zero in the normal state — it is the condensate-specific orbital response.

The full multipolar exchange tensor `J_αβ(Q)` includes the Q-dependent B₁g channel opening via `sinh(2Q/λ)`. The commutator diagnostic `‖[τ_x, H_AFM]‖` measures how strongly the normal-state exchange blocks the B₁g channel.

### 6. Dual B₁g Pairing Channels

Two symmetry-equivalent B₁g pairing channels are treated simultaneously with **independent strengths**:

- **Channel s** — on-site inter-orbital singlet (Γ₆⊗Γ₇ → B₁g, φ = 1):
  ```
  D_s = Δ_s · (|6↑⟩⟨7↓| − |6↓⟩⟨7↑|)
  V_s = g_Delta_s · g_JT² / K_lattice
  ```

- **Channel d** — inter-site d-wave (φ(k) = cos kx − cos ky → B₁g in k-space):
  ```
  D_d = Δ_d · φ(k) · (|A:6↑⟩⟨B:7↓| − |A:6↓⟩⟨B:7↑|)
  V_d = g_Delta_d · g_JT² / K_lattice
  ```

### 7. 16×16 BdG Hamiltonian (Doubled Unit Cell)

Nambu basis: `Ψ = [Particle_A(4), Particle_B(4), Hole_A(4), Hole_B(4)]`, each block `[Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓]`. The full 16×16 structure:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T(k)       │  D_s      D_d        │   ← Particle sector
      │  T†(k)  H_B        │  D_d      D_s        │
      ├────────────────────┼─────────────────────┤
      │  D_s†   D_d†       │  −H_A*   −T*         │   ← Hole sector
      │  D_d†   D_s†       │  −T†*    −H_B*       │
      └────────────────────┴─────────────────────┘
```

The particle–hole off-diagonal blocks use the **transposed** (not Hermitian conjugate) pairing operator, consistent with BdG particle–hole symmetry. The anisotropic hopping `T(k) = −2[tx cos kx + ty cos ky] · I₄` encodes the B₁g distortion.

The physical electron density:
```
⟨n_{iσ}⟩ = Σ_n |u_{n,iσ}|² f(E_n) + |v_{n,iσ}|² (1 − f(E_n))
```
Both terms carry a **positive sign**: `|v|²·(1−f)` is the filled-band electron contribution from below the Fermi level. The orbital spin weights `sz = [+1, −1, +η, −η]` handle spin differences without extra minus signs for the v-block.

### 8. Irrep Selection and SC-Activated JT

An algebraic irrep projector tracks how much the SC condensate has lifted the B₁g symmetry barrier:

```
P_eff = P₆ + w · P₇     where w = min(|Δ| / Δ_CF, 1)
```

- `w = 0`: pure AFM state — τ_x strictly off-diagonal → ⟨τ_x⟩ = 0, JT forbidden.
- `w → 1`: SC-mixed state — τ_x acquires diagonal elements → ⟨τ_x⟩ ≠ 0, JT unlocked.

The selection ratio `R = w · |⟨τ_x⟩| / τ_x,max` is tracked throughout the SCF loop.

### 9. Exchange Rigidity: ∂²F_ex/∂Q²

`compute_JT_rigidity_from_exchange` computes:

```
∂²F_ex/∂Q² = 2·⟨O⟩ @ J @ (∂⟨O⟩/∂Q)  +  ⟨O⟩ @ (∂J/∂Q) @ ⟨O⟩
```

Both the Q-dependence of the multipolar expectation values `⟨O_α(Q)⟩` and the Q-dependence of the exchange tensor `J_αβ(Q)` are included. Positive `∂²F_ex/∂Q²` stiffens the phonon; negative softens it, which in the SC condensate can drive `K_eff < 0` — the SC-triggered JT instability.

### 10. Multipolar Susceptibility χ_τx

```
chi_tau = |∂⟨τ_x⟩/∂(g_JT·Q)|   via finite-difference BdG at Q ± δQ
```

At each perturbed Q, the full BdG is rediagonalized with both `t(Q±δQ)` and `h_afm(Q±δQ)` recomputed consistently. Computed once at post-convergence.

### 11. Coupled Spin–JT RPA Vertex and ∂λ_pair/∂Q

The pairing vertex is computed via a 2×2 coupled spin–JT RPA in `[spin, JT-phonon]` channel space. The bare interaction matrix is **diagonal**: `Û = diag(J_eff, V_JT)` — there is no bare S–Q cross-vertex; the spin–JT feedback enters exclusively through the off-diagonal susceptibilities χ_SQ/χ_QS, which are opened by SOC and the SC condensate:

```
V(q) = J_eff² χ_SS^RPA(q) + V_JT² χ_QQ^RPA(q) + J_eff V_JT [χ_SQ^RPA(q) + χ_QS^RPA(q)]
```

The bare susceptibilities χ₀(q) come from the Δ=0 BdG Hamiltonian via the Lindhard formula (4×4 orbital tensor, 8 normal Nambu sector pairs). Projections:

```
χ_SS = Tr[Sz · χ₀[Γ₆,Γ₆] · Sz]      # spin–spin
χ_SQ = Tr[Sz · χ₀[Γ₆,Γ₇]]            # spin–orbital cross
χ_QQ = −∂²Ω/∂Q²  (numerical, q=0)     # orbital JT stiffness
```

The cross-terms χ_SQ and χ_QS are **zero in the normal state** (Γ₆–Γ₇ mixing forbidden at Q=0) and become nonzero when Q > 0 opens the B₁g channel via τ_x. The RPA determinant:

```
det = (1 − J_eff·χ_SS)(1 − V_JT·χ_QQ/K) − J_eff·V_JT·χ_SQ·χ_QS
```

The cross-term `J_eff·V_JT·χ_SQ·χ_QS` **vanishes in the normal state** and is switched on by the SC condensate. Reducing det amplifies both χ_SS^RPA and χ_QQ^RPA, enhancing V(q). Even if the lattice is stable on its own and the spin system is away from the QCP, the cross-term can flip the entire determinant negative once the condensate opens χ_SQ ≠ 0. Near the AFM QCP, `(1 − J_eff·χ_SS) → 0`, which **massively amplifies the cross-channel effect** — a modest χ_QQ is sufficient when spin fluctuations are critical.

**The JT distortion therefore does not pair directly (V_JT is small); it renormalizes the spin-fluctuation vertex upward through the cross-channel structure of the RPA.**

This mechanism is directly quantified by **∂λ_pair/∂Q** — the finite-difference derivative of the largest eigenvalue of the linearized gap equation:

```
∂λ_pair/∂Q = [λ_max(+ε) − λ_max(−ε)] / 2ε
```

**∂λ_pair/∂Q > 0 is the key numerical criterion for the SC-triggered JT hypothesis.** A positive value confirms that an infinitesimal B₁g distortion increases the pairing strength, meaning the distortion renormalizes V(q) upward through the spin-fluctuation channel rather than contributing direct phonon-mediated pairing. It is evaluated at Δ=0 with the converged SCF chemical potential. In the BO scout phase, this quantity requires two full `solve_linearized_gap_equation` calls (~186 s at nk=74) and is therefore only computed for candidates that pass the cheaper H1–H3 and S1–S3 constraints.

**Susceptibility consistency:** χ₀ (normal state, Δ=0) is used for χ_SS, χ_SQ, χ_QS in the pairing vertex — feeding Δ≠0 susceptibilities back into the interaction would double-count the gap. `chi_QQ_bare` (SC state, Δ≠0) is used exclusively for lattice stability diagnostics (G-matrix), because it is the condensate-driven change in χ_QQ that can drive `det(RPA) < 0`. These two roles are kept strictly separated.

**Moriya QCP regularization:** When `J_eff·χ_SS ≥ 1`, the vertex is suppressed via doping-dependent Moriya damping rather than a hard cutoff:
```
α_M = max(C · δ · (t_eff / J_eff),  α_M_floor)
Γ_M = α_M · J_eff · t_eff
```

### 12. Linearized Gap Equation and λ_JT_kernel

The pairing kernel on the Fermi surface:
```
Γ_ij = g_Δ · V(k_i − k_j) / √|v_F(i) v_F(j)|
```

λ_max = largest eigenvalue of Γ, with gap eigenvector φ_max. The **JT-channel Rayleigh projection**:
```
λ_JT_kernel = φ_max^T · Γ_JT · φ_max
```
measures how much of λ_max comes specifically from the JT channel (V_JT component of V(q)), independently of the spin-fluctuation contribution. This is distinct from `lambda_JT = (g²/K)·chi_tau`, which is a scalar q=0 estimate.

### 13. G-Matrix: SC–JT Coupled Instability

The coupled SC–JT instability boundary is tracked via a 3×3 G-matrix (s-channel, d-channel, Q-mode) in `_build_G3_matrix`:

```
G3 = ┌ 1 − gVs·χ_ss     −√(gVs·gVd)·χ_sd   −cs·χ_sQ ┐
     │ −√(gVs·gVd)·χ_sd  1 − gVd·χ_dd       −cd·χ_dQ │
     └ −cs·χ_Qs          −cd·χ_Qd        1 − χ_QQ/K_eff ┘
```

where `cs = √(gVs/K_eff)`, `cd = √(gVd/K_eff)`.

**Interpretation of `λ_min` and `G22` depending on evaluation context:**

| Context | `λ_min` | `G22` | Physical meaning |
|---|---|---|---|
| Normal state (Δ=0, pre-SCF) | > 0 | > 0 | JT mode stable — SC-triggered JT viable |
| Normal state (Δ=0, pre-SCF) | < 0 | < 0 | Spontaneous JT in normal state — wrong mechanism |
| Post-SCF (Δ≠0, converged) | < 0 | > 0 | SC condensate triggered JT — correct and desired |
| Post-SCF (Δ≠0, converged) | < 0 | < 0 | Both modes soft — unphysical |

`G3[2,2] = 1 − χ_QQ/K_eff`. The Schur complement of G22 gives the effective pairing enhancement diverging as `G22 → 0⁺`. Full diagnostics via `compute_G_instability(compute_dlambda=True/False)`.

### 14. SC-JT Coexistence Window

`check_sc_jt_window` verifies that `K_lattice` lies in the cooperative SC–JT window:

```
K_spont = g_JT² / Δ_CF           (spontaneous JT threshold; K_lattice must exceed this)
K_SC    = g_JT² · χ_τ / λ_min    (SC-triggered threshold; K_lattice must be below this)
```

The window condition reduces to `χ_τ · Δ_CF > λ_min` (independent of `g_JT`). `K_opt = √(K_spont · K_SC)` is the geometric midpoint.

### 15. Tc and Gap Ratio

Two independent Tc estimates:

- `compute_Tc_by_gap_suppression`: bisects in T to find where `|Δ(T)| < Delta_tol` via full re-SCF with warm-starting.
- `compute_lambda_vs_T`: tracks linearized gap eigenvalue `λ_max(T)`; Tc at `λ_max = 1` crossing.

`compute_gap_ratio` reports `2Δ₀ / k_B Tc`; values above 3.52 (BCS weak-coupling) indicate SC-JT strong-coupling enhancement.

### 16. Variational Free Energy

```
Ω_BdG = (1/2) Σ_{k,n} w_k [E_n(k) f(E_n) − T S(f_n)]
        + |Δ_s|²/(g_Delta_s · V_s) + |Δ_d|²/(g_Delta_d · V_d)
        + (K_eff/2)Q²
```

The condensation correction terms use **independent** Gutzwiller factors per channel: `g_Delta_s = g_t` (on-site), `g_Delta_d = g_J` (d-wave).

### 17. Analytic ∂F/∂M and ∂²F/∂M² (Single Diagonalization)

```
∂F/∂M   = Σ_{k,n} f_n · ⟨ψ_n|∂H/∂M|ψ_n⟩                              (Hellmann–Feynman)
∂²F/∂M² = Σ_{k,n} (∂f_n/∂E_n) · ⟨ψ_n|∂H|ψ_n⟩²                        (diagonal term)
         + Σ_{k,n≠m} (f_n − f_m)/(E_m − E_n) · |⟨ψ_n|∂H|ψ_m⟩|²       (off-diagonal term)
```

Computed analytically from a single BdG diagonalization via second-order perturbation theory. The Newton step for M uses the analytic curvature with Levenberg–Marquardt regularization.

### 18. Observables: BdG Thermal Averages

| Observable | Formula |
|---|---|
| Density | ⟨c†c⟩ = Σ_n [\|u_n\|² f(E_n) + \|v_n\|² (1−f(E_n))], divided by 4 |
| Magnetization | ⟨S_z⟩ using orbital-dependent sz = [+1, −1, +η, −η] |
| Quadrupole ⟨τ_x⟩ | Σ_n [2 Re(u†_{Γ₆} u_{Γ₇}) f + 2 Re(v†_{Γ₆} v_{Γ₇})(1−f)] |
| Pairing s | F_AA = u_A[6↑] · v_A[7↓]* − u_A[6↓] · v_A[7↑]* (on-site) |
| Pairing d | F_AB = u_A[6↑] · v_B[7↓]* − u_A[6↓] · v_B[7↑]* (inter-site, φ(k) weight) |

All computed in a single batched LAPACK call via `VectorizedBdG`.

### 19. Two-Site Cluster: Quantum Multipolar Fluctuations

Beyond BdG mean field, a 2-site (A–B) cluster is exactly diagonalized at each iteration:

```
H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)
          + J_eff · O_A ⊗ O_B
          + Z_boundary · (J_eff + U_mf_stoner/2) · M_ext · (O_A ⊗ I + I ⊗ O_B)
```

The cluster computes both `⟨τ_x⟩` (classical) and `√⟨τ_x²⟩` (RMS including quantum fluctuations). Since `[τ_x, H_cluster] ≠ 0`, these are genuinely different. The cluster-to-BdG J ratio `_cluster_j_renorm` feeds back into `J_alpha_beta_Q` as a vertex renormalization.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

```
∂n/∂μ = Σ_{k,n} w_k · f(E_n)(1−f(E_n)) / kT · (|u_A|² + |u_B|² + |v_A|² + |v_B|²)
```

Newton's method with the analytic derivative from the same BdG eigensystem; Brent's method as fallback.

### 21. Gap Equations: Full-BZ Integration

`compute_gap_eq_vectorized` evaluates gap equations over the full Brillouin zone using Simpson-weighted k-grids:

```
F_AA_BZ = Σ_k w_k · Pair_s(k) / 4
F_AB_BZ = Σ_k w_k · Pair_d(k) / 4
```

The `/4` corrects for the 16-dimensional BdG space doubling. The d-wave projection is applied via an explicit `φ(k)` projection of `V(k−k')`:

```
V_d_scalar = φ · V_mat · φ / φ²      (φ_k = cos kx − cos ky)
```

---

## Model Architecture

```
ModelParams  (dataclass, __post_init__)
    ├── Primary: t_pd, u, lambda_soc, Delta_tetra, g_JT, K_lattice,
    │            lambda_hop, eta, Delta_inplane, Delta_CT, omega_JT,
    │            mu_LM, ALPHA_HF, Z, nk, kT, a, max_iter, tol, mixing
    ├── Derived: Delta_CF, t0, U, U_mf, J_CT, doping_0, _U4, U_gamma
    └── Grid objects: k_points, k_points_even, k_weights, k_weights_even,
                      chi0_Q_idx, shift_table, N_k, N_k_even

ClusterMF  (2-site exact diagonalization)
    ├── build_multipolar_operator(η)
    ├── build_cluster_hamiltonian(...)
    └── cluster_expectation(evals, evecs, O, T, site_index)

VectorizedBdG  (performance kernel, lives inside RMFT_Solver)
    ├── _build_H_stack(kpts, ..., out=)  → (N, 16, 16) BdG stack
    ├── compute_observables_vectorized(...)
    └── compute_gap_eq_vectorized(...)

RMFT_Solver
    ├── SusceptibilityMixin
    │   ├── get_susceptibilities_fast    analytic 2-band (DE scout, G-matrix pre-SCF)
    │   ├── get_susceptibilities_normal  full Lindhard χ₀(q) tensor (pairing vertex)
    │   └── get_susceptibilities_sc      SC-state χ_QQ via ∂²Ω/∂Q²
    ├── _get_chi0_norm_cache(...)        Δ=0 eigenvector cache across q-loop and iterations
    ├── _rebuild_orbital_operators(p)    rebuild P₆, P₇, τ₁₆ after SOC/CF change
    ├── _reset_transient_state()         safe clone reset for parallel workers
    ├── compute_chi0_tensor(...)         (4,4) orbital susceptibility tensor
    ├── solve_linearized_gap_equation(.) λ_max, gap vector, λ_JT_kernel, gap symmetry
    ├── compute_G_instability(δ, M,
    │       compute_dlambda=True)        G3 matrix, ∂λ_pair/∂Q, Tc estimate
    ├── compute_hessian(...)             post-SCF SC-JT Hessian (3×3 curvature)
    ├── compute_Tc_by_gap_suppression(.) Tc via Brent bisection on Δ(T)
    ├── compute_lambda_vs_T(...)         λ_max(T) curve, Tc at λ_max=1
    ├── compute_gap_ratio(...)           2Δ₀/kTc strong-coupling diagnostic
    ├── _compute_chi_tau(...)            finite-difference BdG χ_τx
    ├── _find_mu_for_density(...)        Newton (analytic ∂n/∂μ) + Brent fallback
    ├── _anderson_mix(...)               quasi-Newton convergence (M, Q)
    └── solve_self_consistent(...)       Anderson-accelerated SCF loop

UnifiedBayesianOptimizer  (5D: Δ_tetra, λ_soc, u, g_JT, t_pd)
    ├── _eval_constraints(s, doping)     two-phase H1–H3 + S1–S4 constraint evaluation
    ├── _eval_one_doping(...)            full SCF + scoring for one parameter point
    ├── _score(..., lambda_JT)           three-tier multiplicative gate scoring
    ├── run_de_phase(...)                Phase 1: DE scout (analytic G-matrix only)
    ├── run_gp_seed_phase(...)           Phase 2: top-k DE → full SCF
    ├── run_turbo_phase(...)             Phase 3: trust-region GP-EI
    ├── run_local_refine(...)            Phase 4: Nelder–Mead polish
    └── optimize(...)                   orchestrates all four phases

check_sc_jt_window(...)                 K_lattice window diagnostic (standalone)

Visualization
    ├── plot_phase_diagrams(solver, δ_scan, opt_result)   3×3 (or 4×3) panel figure
    ├── _plot_phase_data(ax, phase_data)
    └── _plot_dos(ax, solver, result)
```

---

## Key Algorithms

### VectorizedBdG: Batched LAPACK and Buffer Reuse

- `_build_H_stack(kpts, out=)` accepts an optional pre-allocated `(N, 16, 16)` buffer; on the hot SCF path no heap allocation occurs per iteration.
- The per-iteration BdG eigensystem `(ev, ec)` is computed **once** and shared by observable computation, dual-channel gap equations, and ∂F/∂M. Stored in `self._scf_bdg_cache`, cleared after use.

### χ₀(q) Permutation Tricks

The k-grid (endpoint=False) is constructed so that for any `q = (nx, ny) · 2π/nk`, the k+q grid is a cyclic permutation of the k-grid. The precomputed `shift_table[nx, ny]` (built in `ModelParams.__post_init__`, shape `(nk, nk, N_k_even)`, dtype int32) implements this as a free index reorder — eliminating all LAPACK calls for k+q in the pairing kernel loop:

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]   # no LAPACK, just index reorder
```

For q = (π, π), the special `chi0_Q_idx` provides the same trick for the AFM wavevector. The solver-level `_get_chi0_norm_cache` additionally caches the Δ=0 eigenvectors across calls with the same `(M, Q, mu, tx, ty, g_J, target_doping)` within tolerance 1e-5, avoiding redundant diagonalizations across the q-loop and across SCF iterations near convergence.

### Dual k-Grid Setup

Two separate k-grids generated once in `ModelParams.__post_init__`:

- **SCF grid (nk, endpoint=False):** BdG diagonalization, observables, free energy, gap equations. Uniform 1/N weights.
- **χ₀ grid (same nk, endpoint=False):** χ₀(q) and pairing kernel, exploiting the shift_table permutation trick.

### Thread-Safety and Clone Protocol

`RMFT_Solver` is cloned with `copy.copy()` before each parallel SCF worker:
```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s.p.__post_init__()
s._K_bare = s.p.K_lattice
s._rebuild_orbital_operators(s.p)   # if lambda_soc or Delta_tetra changed
s._reset_transient_state()          # clear _vbdg, _scf_bdg_cache, _chi0_norm_cache
```
`_reset_transient_state` ensures each clone owns its own `VectorizedBdG` instance (and thus its own `_H_stack` buffer), preventing inter-worker memory aliasing. `OMP_NUM_THREADS=1` prevents BLAS thread oversubscription inside `ThreadPoolExecutor`.

### Vertex Cache Invalidation

The RPA pairing vertex is recomputed when `|M − M_cached| > 0.03`, `|Δ_eff − Δ_cached| > 0.008` or relative change > 15%, or the Fermi-surface sampling points change. `chi_QQ_bare_v` is evaluated in the SC state (Δ≠0) for the lattice stability branch; the pairing vertex inputs are always from the normal state.

### SCF Loop (`solve_self_consistent`)

Anderson(5)-accelerated iteration over (M, Q, Δ_s, Δ_d, μ):
1. Build and diagonalize the 16×16 BdG stack; cache `(ev, ec)` for the iteration.
2. Update M via Levenberg–Marquardt-regularized Newton step + BdG fixpoint blend `(1−ALPHA_HF)·fixpoint + ALPHA_HF·Newton`.
3. Update Q adiabatically: `Q_out = −(g_JT/K_eff)·⟨τ_x⟩`; invalidate `_chi0_norm_cache` on Q change.
4. Update Δ_s, Δ_d via `compute_gap_eq_vectorized` (full-BZ Simpson integration).
5. Update μ via Newton / Brentq to enforce `⟨n⟩ = 1 − δ`.
6. Periodically update `K_eff` (every 5 iterations or on M jump > 0.02).
7. Anderson acceleration on `[M, Q]`; fall back to linear mixing if Anderson worsens residual; reset history on Q sign flip (valley jump protection).
8. Adaptive mixing: reduce α if max_diff increases; cap α near SC critical point (`0.8 ≤ λ_max ≤ 1.8`).

After convergence: post-convergence Hessian test (3×3 `∂²F/∂{M,Q,Δ}²` all-positive → true minimum), coherence length ξ/a, SC-triggered JT confirmation via `hessian_lmin_sc < 0`, λ_JT_kernel, ∂λ_pair/∂Q, channel decomposition (λ_s vs λ_d).

### Unified Bayesian Optimisation (5D)

`UnifiedBayesianOptimizer` searches `(Δ_tetra, λ_soc, u, g_JT, t_pd)` in four phases:

**Phase 1 — DE scout:** `scipy.differential_evolution` with analytic G-matrix only (no SCF). Two-phase constraint evaluation:
- *Phase 1 (cheap, ~few ms):* `compute_G_instability(compute_dlambda=False)` → H1 (`d²F/dQ² > 0`), H2 (`J·χ_SS < 1`), H3 (`G22 > 0`), S1 (λ_min window), S2 (λ_eff window), S3 (λ_JT > 0.05). Early exit if any H fails or `partial_penalty(S1+S2+S3) ≥ 0.25` (infeasible regardless of S4).
- *Phase 2 (expensive, ~186 s at nk=74, only for promising candidates):* `compute_G_instability(compute_dlambda=True)` → S4 (∂λ_pair/∂Q > 0).

**Phase 2 — GP seed:** top-k DE feasible candidates evaluated with full SCF; results seed the GP surrogate.

**Phase 3 — TuRBO:** trust-region GP-EI acquisition, batch parallel via `ThreadPoolExecutor`.

**Phase 4 — local refine:** Nelder–Mead polish on the best TuRBO candidate.

**Scoring (`_score`)** is a three-tier product of multiplicative gates:
- *Tier 1 (hard guards):* `J·χ_SS > _JCHI_HARD_REJECT → 0`, `G22 ≤ 0 or λ_min ≤ 0 → 0`.
- *Tier 2 (mechanism validation):* `lJT_f` (λ_JT = g²χ_τ/K viability window 0.05–0.90, requires Δ≠0), `lJT_kernel_f` (Rayleigh projection of gap eigenvector on Γ_JT), `sc_hessian_f` (Hessian min_curvature < 0 confirms SC-triggered JT).
- *Tier 3 (optimization objective):* `Tc_proxy`, `xi_f` (coherence length ξ/a ≥ 2), `conv_f` (convergence), `stoner_f` (AFM Stoner penalty), `ratio_bonus` (2Δ/kTc strong-coupling), `jchi_gate` (near-QCP Gaussian).

---

## Parameters

All energies in **eV**, lengths in **Å**.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | t_pd | 0.590 eV | pd hybridisation integral (primary hopping; t₀ derived) |
| `u` | u | 15.0 | U/t₀ ratio; Hubbard U = u·t₀ |
| `lambda_soc` | λ_SOC | 0.144 eV | Atomic SOC constant (t₂g shell) |
| `Delta_tetra` | Δ_tet | −1.500 eV | Tetragonal CF (**required < 0**); Δ_CF derived |
| `g_JT` | g_JT | 0.230 eV/Å | Electron–phonon JT coupling |
| `K_lattice` | K | 1.500 eV/Å² | Bare phonon stiffness; K_eff computed at runtime |
| `lambda_hop` | λ_hop | 1.280 Å | Hopping decay: t(Q) = t₀·exp(±Q/λ) |
| `eta` | η | 0.220 | Γ₇ AFM asymmetry |
| `Delta_CT` | Δ_CT | 2.600 eV | Charge-transfer gap (material-class constant) |
| `Delta_inplane` | Δ_ip | 0.050 eV | B₂g in-plane CF; splits Γ₇ doublet |
| `omega_JT` | ω_JT | 0.057 eV | JT phonon frequency (~46 meV) |
| `mu_LM` | — | 6.8 | LM regularization floor for M Newton step |
| `ALPHA_HF` | — | 0.12 | Newton vs BdG fixpoint blend for M |
| `nk` | — | 74 | k-points per direction (must be even) |
| `kT` | kT | 0.015 eV | Temperature (~174 K) |
| `max_iter` | — | 250 | Maximum SCF iterations |
| `tol` | — | 1e-4 | Convergence threshold |
| `mixing` | α | 0.04 | Anderson mixing weight |

### Derived Parameters (from `__post_init__`)

| Parameter | Formula | Description |
|---|---|---|
| `Delta_CF` | from SOC+CF diag. | Γ₆–Γ₇ splitting |
| `t0` | t_pd²/Δ_CT | Effective dd hopping |
| `J_CT` | 2t_pd⁴/Δ_CT²·(1/U+1/(Δ_CT+U/2)) | ZSA CT superexchange |
| `U_mf` | Z·J_CT/2 | Bare Weiss-field amplitude |
| `doping_0` | z_ZRS/(1−z_ZRS) | ZRS spectral weight scale |

### SC+JT Coexistence Conditions

Four independent conditions checked by `compute_G_instability` and `check_sc_jt_window`:

1. **Metallicity:** `h_AFM < 2·g_t·t₀` — AFM gap does not swallow the Fermi surface.
2. **Pairing scale:** `V_s·g_Delta_s, V_d·g_Delta_d >> kT`.
3. **Normal-state JT stability:** `K_eff > 0`, i.e. `G3[2,2] > 0` at Δ=0.
4. **SC-triggered regime:** `0.05 < lambda_JT = g_JT²·chi_tau/K_lattice < 1.0`.

The SC-triggered JT condition proper:
```
∂²F/∂Q²|_{Δ=0} > 0      (Q-stable without SC)
∂λ_pair/∂Q > 0           (distortion renormalizes V_pair upward)
```

---

## Installation & Usage

### Requirements

```
numpy scipy matplotlib scikit-learn opt_einsum threadpoolctl
```

```bash
pip install numpy scipy matplotlib scikit-learn opt_einsum threadpoolctl
```

### Running

```bash
python Quantum_AFM-multipolar_Jahn-Teller.py
```

On startup:
1. SOC+CF diagonalization → Δ_CF, k-grids, `shift_table`, orbital operators.
2. `params.summary()` — all derived parameters and pre-SCF diagnostics.
3. `compute_G_instability()` + `check_sc_jt_window()` at baseline parameters.
4. Linearized gap equation at Δ=0 — pre-SCF λ_max, gap symmetry, ∂λ_pair/∂Q.
5. `UnifiedBayesianOptimizer.optimize()` — DE scout → GP seed → TuRBO → local refine.
6. Phase-diagram scan and post-SCF diagnostics at optimized parameters.
7. `compute_Tc_by_gap_suppression`, `compute_lambda_vs_T`, `compute_gap_ratio` diagnostics.

---

## Output & Visualization

### Iteration Log

Each SCF step prints (thread-safe): M, Q, Δ_s, Δ_d, density n, μ, F, χ₀(q_AFM), RPA factor, K_eff, JT algebraic status.

### Convergence Report

At convergence: all converged order parameters, Hessian eigenvalues, G3-matrix diagnostics (λ_min, det(G3), dominant channel, Tc estimate), λ_JT, λ_JT_kernel, ∂λ_pair/∂Q, gap symmetry, channel decomposition (λ_s vs λ_d), SC-triggered JT confirmation (hessian_lmin_sc < 0), coherence length ξ/a, 2Δ₀/kTc.

### Phase Diagram (3×3 panels)

| Position | Content |
|---|---|
| [0,0] | M, Q, Δ_s, Δ_d, \|Δ\| vs. doping δ with phase-region shading and Tc overlay |
| [0,1] | Crystal-field sweet-spot: Δ_d, Q, M vs. Δ_CF (Δ_tetra scan, twin-axis) |
| [0,2] | Density of States (BdG) with van Hove singularity detection |
| [1,0–2] | SCF convergence of M, Q, \|Δ\| (one line per doping point) |
| [2,0] | Free energies F_bdg and F_cluster vs. iteration |
| [2,1] | Gutzwiller factors g_t, g_J vs. iteration |
| [2,2] | Tc(δ) and G3[2,2](δ) vs. doping |

With BO results, a 4th row: BO progress (Δ and score vs. evaluation), doping vs. score, parameter scatter — all coloured by λ_JT regime (SC-triggered / strong-coupling / JT-closed).

---

## Known Limitations

| Approximation | Impact |
|---|---|
| No Pauli exclusion between cluster sites | Slight overestimate of AFM correlations; controlled by ALPHA_HF blend |
| No charge-transfer fluctuations ⟨n_A n_B⟩ | Charge fluctuations negligible when U_mf ≫ t |
| Static phonon (Q is a mean field) | Zero-point quantum lattice fluctuations neglected |
| No spatial fluctuations | Cannot describe pseudogap, stripes, or phase separation |
| RPA static (ω = 0) | Dynamical vertex corrections absent |
| K_eff updated every 5 SCF iterations | Back-action of Q on exchange rigidity approximate during SCF transient |
| chi_tau at post-convergence only | Self-consistent Q back-action on chi_tau neglected during SCF |
| `compute_G_instability` at Δ=0 | G-matrix evaluates normal-state only; SC-triggered JT confirmed via post-SCF Hessian λ_min < 0 |
| ∂λ_pair/∂Q at Δ=0, estimated μ | Evaluated at normal-state Fermi surface; SC-state version would require Bogoliubov Lindhard sum |

---

## References

- Ecsenyi, S. (2026). *Multipolar superconductivity and coherent orbital mixing* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang et al. (1988). *Supercond. Sci. Technol.* 1, 36; Bünemann, J., Weber, W. & Gebhard, F. (1998). *Phys. Rev. B* 57, 6896.
- ZSA classification: Zaanen, Sawatzky & Allen (1985). *Phys. Rev. Lett.* 55, 418.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.
- Bayesian optimisation / GP: Snoek, J., Larochelle, H. & Adams, R.P. (2012). *NeurIPS.*

---

*For questions or contributions, open an issue or pull request.*
