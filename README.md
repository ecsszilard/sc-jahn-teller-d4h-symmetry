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
- [Output & Diagnostics](#output--diagnostics)
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

The selection rule is enforced by the B₁g phonon operator `B1g_op = real(U₄†(Lx²−Ly²)U₄)` projected to the active Γ₆⊕Γ₇a subspace. In D₄h (Δ_inplane = 0) this operator is purely anti-diagonal (spin-flip): it couples (Γ₆↑)↔(Γ₇a↓) and (Γ₆↓)↔(Γ₇a↑), so `⟨B1g_op⟩ = 0` exactly in any normal state — the JT distortion is a symmetry singlet that requires Cooper pairs to unlock it. In D₂h (Δ_inplane ≠ 0) the operator gains spin-conserving and diagonal elements, partially activating the JT channel without SC; the SC-triggered excess is then isolated as `δχ_τ = χ_τ(Δ≠0) − χ_τ(Δ=0)`.

SC-triggered JT activation is tracked by a **selection ratio**:

```
selection_ratio = min(|Δ_s| + |Δ_d|, Δ_CF) / Δ_CF · |⟨τ_x⟩_BdG|
```

Values above 0.05 indicate that the condensate has lifted the B₁g symmetry barrier sufficiently for JT to be active. This is computed directly from the converged BdG state.

The symmetry selection rules:

| State | Condition | Meaning |
|---|---|---|
| AFM ground state | Γ_JT ⊄ Γ_AFM ⊗ Γ_AFM | JT **forbidden** |
| SC condensate | Γ_JT ⊂ Γ_pair ⊗ Γ_pair | JT **allowed** |

For B₁g-symmetry Cooper pairs: the SC condensate transfers the order parameter into an irrep channel that is self-closing under the tensor product with the Cooper-pair irrep family, and in which rank-2 multipolar operators — including the B₁g JT mode — are no longer forbidden.

### The JT Distortion as a Thermodynamic Order Parameter

The Jahn–Teller distortion Q in this model is a **macroscopic, thermodynamic order parameter** — not a local, dynamical degree of freedom. Physically it corresponds to an optical Einstein phonon whose dispersion (q-dependence) across the Brillouin zone is negligible (flat band). When the macroscopic free energy Ω is differentiated with respect to Q, the result is the exact thermodynamic softening of this specific mode:

```
λ_JT = −∂²Ω/∂Q²|_{Q=0}   (> 0 → instability, i.e. JT-active)
```

This is a rigorous statement: it is the phonon stiffness of the zone-centre optical mode renormalized by the electron–phonon coupling and the electronic susceptibility of the condensate, without approximations beyond mean field.

An important symmetry constraint follows immediately. The distortion Q has B₁g symmetry; the pairing amplitude squared |Δ|² is A₁g (totally symmetric, whether s-wave or d-wave). Their product Q·|Δ|² transforms as B₁g ⊗ A₁g = B₁g — which is **not** the totally symmetric representation A₁g. Therefore the coefficient of the Q·|Δ|² term in the free energy is strictly zero by symmetry: there is no linear coupling between the JT distortion and the SC condensate at the Landau level. The SC-triggered JT distortion is a **threshold phenomenon**: the condensate renormalizes the stiffness of the B₁g mode until, at λ_JT = 1, the mode goes soft and a spontaneous distortion appears.

### The Mean-Field Back-Action Loop and Its Numerical Stabilization

The fundamental theorem of mean-field theory requires that the expectation value of every order parameter be fed back into the Hamiltonian:

```
H_MF ∝ J ⟨Ô⟩ · Ô
```

When the superconducting condensate — via Γ₆–Γ₇ orbital mixing — creates a macroscopic anomalous coherence `⟨τ_x⟩_anom`, this must be fed back through the B₁g exchange tensor J_B₁g into the Weiss field. Concretely: the condensate generates off-diagonal Γ₆↔Γ₇ orbital coherence in the BdG eigenstates; this coherence modifies the effective exchange field felt by the lattice; the lattice responds by shifting Q; and the shifted Q in turn modifies the electronic structure and the pairing vertex. This feedback loop is physically essential — without it, the model is inconsistent with its own Hamiltonian.

This loop is numerically sensitive and can oscillate or diverge if not treated carefully. Three stabilization mechanisms are in place:

- **Anderson mixing:** the four-dimensional order-parameter vector `[M, Q/λ_hop, |Δ_s|·t₀, |Δ_d|·t₀]` is accelerated jointly via Anderson(5), capturing the cross-coupling ∂M/∂Δ and ∂Δ/∂M in the effective Jacobian.
- **Tikhonov regularization:** the Anderson normal equations use a Tikhonov shift (`_ANDERSON_TIKHONOV = 1e-8 × diag_max`) that prevents the least-squares solve from amplifying noise in the residual history.
- **Jacobi kick (`_scf_jacobi_kick`):** before the main SCF loop, the linearized Jacobian eigenvalue λ₊ of the two-channel (Δ, Q) map is estimated analytically. This determines the initial seed values for (M, Q, Δ) and the initial mixing rate α, so that the SCF starts in the basin of the correct physical fixed point rather than at an arbitrary initialization.

### Hellmann–Feynman Lattice Update

The equilibrium lattice distortion is determined by the Hellmann–Feynman theorem applied to the total free energy:

```
∂F/∂Q = K_eff · Q + g_JT · ⟨B̂₁g⟩ = 0
⟹  Q_eq = −(g_JT / K_eff) · ⟨B̂₁g⟩
```

The lattice update in the SCF loop uses the **full** `⟨B̂₁g⟩ = Tr[B1g_16 · ρ̂]` — not the simpler off-diagonal `τ_x` — because in D₂h (Δ_inplane ≠ 0) the B1g_op gains spin-preserving and diagonal (τ_z) components that are active even in the normal state. Using only `τ_x` would make the lattice blind to those contributions, breaking Hellmann–Feynman consistency with `H_JT = g_JT · Q · B1g_op`. In D₄h the two expressions are exactly equal; in D₂h only `⟨B̂₁g⟩` is correct.

The `⟨B1g_exp⟩` computation correctly accounts for the Nambu structure via a `/4` normalisation factor that corrects for both the particle–hole (Nambu) doubling and the two-sublattice (A–B) doubling simultaneously:

```python
Bdiag_qp = einsum('kna,knb,ab->kn', ec.conj(), ec, B1g_16).real
exp_k    = einsum('kn,kn->k', Bdiag_qp, f)
B1g_exp  = dot(k_weights, exp_k) / 4.0
```

The hole-block sign (`−B1g_op^T`) is already encoded in `B1g_16`, so weighting by `f` alone (not `fbar`) correctly accounts for both particle and hole contributions.

The off-diagonal `⟨τ_x⟩_BdG` is kept separately: it feeds the anomalous orbital coherence back into the Weiss field (the mean-field back-action loop) and enters the `selection_ratio` diagnostic. The two quantities play distinct physical roles and are both returned by `compute_observables_vectorized`.

---

## Theoretical Framework

### 1. Local Hilbert Space and SOC+CF Hamiltonian

The full SOC + D₄h crystal-field Hamiltonian is constructed and diagonalized explicitly in the t₂g manifold (6×6):

```
H = λ_SOC · L·S  +  Δ_axial · Lz²  +  Δ_inplane · (Lx² − Ly²)
```

This diagonalization yields the Γ₆–Γ₇ splitting `Δ_CF` as a **derived quantity** (not a free parameter). The SOC eigenbasis `U_gamma` and 4-dim projector `_U4 = U_gamma[:, 0:4]` are precomputed in `__post_init__` so that all orbital operators (B1g_op, B1g_16, multi_op, sz_op) are automatically consistent with the actual diagonalization. The four-component local basis is `[Γ₆↑, Γ₆↓, Γ₇a↑, Γ₇a↓]`.

- `Δ_axial = Δ_tetra · Lz²` — controls the Γ₆–Γ₇ gap; **required < 0** (tetragonal compression, c < a). Partial cancellation with SOC tunes Δ_CF independently of λ_SOC.
- `Δ_inplane = Δ_inplane · (Lx² − Ly²)` — splits the Γ₇ quartet into two Kramers doublets (Γ₇a, Γ₇b), preventing spontaneous JT from the 4-fold degenerate Γ₇ level. At Δ_ip=0 (D₄h) B1g_op is a pure singlet (spin-flip), so JT is strictly SC-triggered. Finite Δ_ip (D₂h) adds spin-conserving and diagonal elements to B1g_op, partially activating JT without SC.

**Validity of the 4×4 BdG projection:** the truncation to Γ₆⊕Γ₇a is accurate when `Δ_CF ≫ kT` and `Γ₇split/Δ_CF ≪ 1`, where `Γ₇split = evals[4] − evals[2]` is the Γ₇a–Γ₇b internal gap. Virtual Γ₇b contributions scale as `(J_eff/Δ_CF)²` and enter the BO scoring as a smooth projection-quality penalty.

Two quantities are derived directly from the eigenvectors and stored in `__post_init__`:

- **`η` (Γ₇ AFM asymmetry):** `η = |⟨Γ₇|S_z|Γ₇⟩| / |⟨Γ₆|S_z|Γ₆⟩|`, computed from the S_z matrix elements of the first Kramers partners of Γ₆ and Γ₇a. This is not a free parameter — it is fully determined by the SOC+CF eigenbasis. It enters `sz_op = [1, −1, η, −η]` and propagates to all magnetization and Weiss-field calculations via `sz_bdg16`. The corresponding **multipolar operator** `multi_op = diag([1, −1, η, −η])` is pre-built in `ModelParams.__post_init__` and stored as `self.multi_op`; it is shared by both the cluster (2-site ED) and the BdG solver without recomputation.

- **Orbital weights `_w6_xz`, `_w6_yz`, `_w6_xy`, `_w7_xz`, `_w7_yz`, `_w7_xy`:** the d_xz, d_yz, d_xy character of the Γ₆ and Γ₇a Kramers states, used in `_exchange_channels` to compute the Q-dependent exchange asymmetry `η_J(Q)`.

The B₁g phonon coupling operator is constructed as:
```
B1g_op = real(U4† · (Lx²−Ly²)_t2g · U4)    (4×4, real, hermitian)
```
and its 16×16 Nambu extension `B1g_16` is stored with the hole block carrying `−B1g_op^T`, consistent with BdG particle–hole symmetry. Since `B1g_op` is a real symmetric matrix, `−B1g_op^T = −B1g_op`. All JT coupling terms in the Hamiltonian use `H += g_JT · Q · B1g_op` rather than a hand-coded τ_x matrix.

When `lambda_soc`, `Delta_tetra`, or `Delta_inplane` are changed on a solver clone (e.g. in Bayesian optimisation), `p.__post_init__()` must be followed by `solver._rebuild_orbital_operators()` to keep B1g_op, B1g_16, `sz_op`, `sz_bdg16`, and `multi_op` consistent with the new eigenbasis.

### 2. ZSA Charge-Transfer Superexchange and Weiss Field

The AFM order originates from virtual pd-hopping processes, not from a Stoner Fermi-surface instability. The ZSA charge-transfer superexchange (single-bond) is:

```
J_CT = 2·t_pd⁴/Δ_CT² · (1/U + 1/(Δ_CT + U/2))
```

The two denominator terms represent the Mott channel (pd→dd, cost U) and the Zhang–Rice channel (pd→pp, cost Δ_CT + U/2) respectively. The factor of 2 in `J_CT` comes from the two distinct virtual hopping pathways on a single bond — a quantum-mechanical property of the single-bond exchange. The bare Weiss-field amplitude is:

```
U_mf = Z · J_CT / 2
```

stored without Gutzwiller renormalization; `g_J · (1−δ)` is applied at runtime in `build_local_hamiltonian_for_bdg`. The effective AFM Weiss field entering the BdG Hamiltonian is:

```
h_z[α] = sign_M · J_A1g[α,α] · g_J·(1−δ) · M · sz[α] / 2
```

where `J_A1g = J_CT · cosh(2Q/λ) · diag(1, 1, η_J², η_J²)` is the longitudinal (diagonal, spin-preserving) exchange tensor. The ZRS coherence crossover scale `δ₀` is derived from the Zhang–Rice singlet spectral weight:

```
z_ZRS = t_pd² / (Δ_CT² + t_pd²),    δ₀ = z_ZRS / (1 − z_ZRS)
```

`δ₀` appears only as the floor in `f_J(δ)` (see §3); the Weiss field uses `(1−δ)` throughout.

### 3. Primary Parameter: t_pd and Gutzwiller Renormalization

`t_pd` is the primary hopping input; the effective dd hopping `t₀ = t_pd² / Δ_CT` is always derived and never set directly. The optimiser searches over `t_pd`; `Δ_CT` is fixed as a material-class constant.

```
g_t       = 2δ / (1 + δ)         # kinetic energy suppression → 0 at half-filling
g_J       = 4 / (1 + δ)²         # exchange enhancement → 4 at half-filling
g_Delta_s = g_t                   # on-site Γ₆⊗Γ₇ channel: kinetic origin
g_Delta_d = interpolates(g_t, g_J, w_norm)  # d-wave B₁g: weighted by Γ₇ admixture p_7
```

`g_Delta_d` interpolates between `g_t` (Γ₇ decoupled) and `g_J` (full Γ₆–Γ₇ mixing) using `w_norm = p_7 / 0.5`, where `p_7` is the Γ₇ spectral weight in the Γ₆ doublet from the SOC+CF eigenvectors.

**Gutzwiller renormalization of the superexchange:** the superexchange `J ∝ t_bare²/U` arises from doubly-occupied virtual states. It is always computed from the **bare** (non-Gutzwiller) hopping `t_pd` and then multiplied by `g_J`. Computing it from the Gutzwiller-renormalized bands would introduce a spurious `g_t²` suppression — a double-counting error. In the RMFT framework, `g_t` renormalizes the kinetic energy only; `g_J` renormalizes the exchange only. These two factors are orthogonal.

The effective superexchange used in the cluster and pairing vertex is a single-bond quantity:

```
J_bond = g_J · f_J(δ) · (tx² + ty²) · (1/U + 1/(Δ_CT + U/2))
f_J(δ) = max(δ, δ₀) / (max(δ, δ₀) + δ₀)
```

The `tx² + ty²` scaling is exact for a 2D square lattice: on a single bond, one x-direction link and one y-direction link contribute their respective exchange constants, and the energy scale `tx² + ty²` is the correct single-bond sum. The lattice-summed Weiss-field scale is `J_eff = Z · J_bond`, applied consistently at all call sites. `f_J` saturates at 0.5 as δ→0 so that `J_eff → 2·Z·J_CT` at half-filling (Mott limit), rather than vanishing. This is distinct from the Weiss field scaling `(1−δ)`: the Weiss field is maximal at half-filling, while `f_J` prevents `J_eff` from collapsing to zero near the Mott insulator.

A Mott guard suppresses SC at `g_t < 0.10` (δ < 0.053): the Gutzwiller factor encodes the full doping-dependent Mott suppression, and no physical SC gap can exist without a coherent Fermi surface. A secondary guard at `ξ/a < 1.0` filters the BEC/artefact extreme limit.

### 4. B₁g Jahn–Teller Distortion and Anisotropic Hopping

The B₁g mode breaks the x–y symmetry of the square lattice:

```
tx(Q) = t₀ · exp(+Q / λ_hop)
ty(Q) = t₀ · exp(−Q / λ_hop)
K_eff = K_lattice + ∂²F_ex/∂Q²
```

`K_lattice` is the **bare phonon spring constant** (primary input, eV/Å²). `∂²F_ex/∂Q²` is computed by `compute_JT_rigidity_from_exchange` via central finite-difference of `⟨O_α(Q)⟩`; negative when the SC condensate softens the JT mode. `K_lattice` is never mutated; `K_eff` is recomputed every 5 SCF iterations or when `|ΔQ| > _Q_THR_REL·λ_hop` or `|ΔM| > 0.02`, tracked via separate `_K_eff_last_Q` and `_K_eff_last_M` variables.

The SC-triggered JT coupling strength:
```
lambda_JT_sc = (g_JT² / K_lattice) · chi_tau_sc
```
The viable regime is `lambda_JT_sc > _LAMBDA_JT_VIABLE = 0.05`. `chi_tau_sc = ∂⟨B1g_op⟩/∂(g_JT·Q)` (signed) evaluated in the SC state (Δ≠0); it is zero in the normal D₄h state by symmetry — it is the condensate-specific orbital response.

The SC-induced excess susceptibility `δχ_τ = chi_tau_sc − chi_tau_n` isolates the condensate contribution in D₂h, where a small normal-state baseline can exist.

The full multipolar exchange tensor `J_αβ(Q)` includes the Q-dependent B₁g channel opening via `sinh(2Q/λ)`, plus a Q-dependent exchange asymmetry `η_J(Q)` between Γ₆ and Γ₇:

```
η_J(Q) = √(J_Γ₇ / J_Γ₆)    where J_Γ₇/J_Γ₆ comes from orbital-selective hopping
```

Superexchange `J ∝ t²` is orbital-selective: d_xz hops only along x, d_yz only along y, d_xy along both. When `tx ≠ ty` (Q ≠ 0), the Γ₆ (xz-dominant) and Γ₇ (yz-dominant) sectors feel different effective exchanges. `η_J(Q)` is computed from the orbital weights `_w6_xz` etc. stored in `__post_init__`; at Q=0 it equals exactly 1.0.

The commutator diagnostic `‖[B1g_op, H_AFM]‖` measures how strongly the normal-state exchange blocks the B₁g channel.

The anisotropic exchange enters the pairing vertex through separate x- and y-direction superexchange couplings:

```
J_eff_x ∝ tx²,    J_eff_y ∝ ty²
J_eff = ½(J_eff_x + J_eff_y)    (scalar for Stoner denominator and Moriya damping)
```

The scalar `J_eff` is used as the Stoner/Moriya coupling strength (this correctly captures `|J(q_AFM)| = J_x + J_y` at Q=0 where `Jx = Jy`). The full anisotropy enters the pairing vertex through `χ_DD_s(q)` computed from the BdG dispersion with `tx ≠ ty`.

### 5. Dual B₁g Pairing Channels

Two symmetry-equivalent B₁g pairing channels are treated simultaneously with **independent strengths**:

- **Channel s** — on-site inter-orbital singlet (Γ₆⊗Γ₇ → B₁g, φ = 1):
  ```
  D_s = Δ_s · (|6↑⟩⟨7↓| − |6↓⟩⟨7↑|)
  V_s = g_Delta_s · g_JT² / K_lattice
  ```

- **Channel d** — inter-site d-wave (φ(k) = cos kx − cos ky → B₁g in k-space):
  ```
  D_d = Δ_d · φ(k) · (|A:6↑⟩⟨B:7↓| − |A:6↓⟩⟨B:7↑|)
  V_d = g_Delta_d · g_JT² / K_lattice     (clamped ≥ 0; zero when FS too sparse for d-wave projection)
  ```

### 6. 16×16 BdG Hamiltonian (Doubled Unit Cell)

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

The particle–hole off-diagonal blocks use the **transposed** (not Hermitian conjugate) pairing operator, consistent with BdG particle–hole symmetry. The anisotropic hopping `T(k) = −2[tx cos kx + ty cos ky] · I₄` encodes the B₁g distortion. The JT coupling enters as `H += g_JT · Q · B1g_op` in the particle sector and `H += g_JT · Q · (−B1g_op^T)` in the hole sector, consistent with the Nambu convention encoded in `B1g_16`. Exact Hermiticity is enforced after assembly.

The physical electron density:
```
⟨n_{iσ}⟩ = Σ_n |u_{n,iσ}|² f(E_n) + |v_{n,iσ}|² (1 − f(E_n))
```
Both terms carry a **positive sign**: `|v|²·(1−f)` is the filled-band electron contribution from below the Fermi level.

### 7. SC-Activated JT Selection Ratio

An inline selection ratio tracks how much the SC condensate has lifted the B₁g symmetry barrier:

```
selection_ratio = min(|Δ_s| + |Δ_d|, Δ_CF) / Δ_CF · |⟨τ_x⟩_BdG|
```

- `selection_ratio ≈ 0`: pure AFM state — B1g_op strictly off-diagonal → ⟨B1g_op⟩ = 0, JT forbidden (exact in D₄h).
- `selection_ratio > 0.05`: SC-mixed state — condensate has opened the B₁g channel → JT active.

The SC–JT chain: Δ≠0 → anomalous ⟨B1g_op⟩ ≠ 0 → Q≠0 → H_JT≠0. In D₄h B1g_op is a singlet operator (spin-flip off-diagonal), so ⟨B1g_op⟩=0 in any normal state; the condensate is required to unlock it. The selection ratio feeds into the adaptive mixing rate during SCF: when `selection_ratio > 0.05` and `|Q| > 1e-4` (JT-active), α is boosted to accelerate convergence; when below threshold, α is damped to suppress oscillations near Q=0.

### 8. Observables: BdG Thermal Averages

All observables are computed in a single batched LAPACK call via `VectorizedBdG`. Two distinct orbital quantities are returned per SCF iteration:

| Observable | Formula | Role |
|---|---|---|
| τ_x (off-diagonal) | `2 Re(u₀*u₂ + u₁*u₃)` per sublattice; BdG thermal avg | Diagnostic: Γ₆↔Γ₇ coherence; feeds `selection_ratio` and anomalous Weiss-field back-action |
| **B1g_exp** (full) | `Tr[B1g_16 · ρ̂_k]` via einsum over B1g_16; /4 for Nambu+sublattice doubling | **Lattice update:** Hellmann–Feynman force `Q_eq = −(g_JT/K_eff)·B1g_exp` |
| Magnetization | `⟨S_z⟩` via `sz_op = [+1,−1,+η,−η]` where η is derived from SOC+CF eigenvectors | AFM order parameter |
| Anomalous coherence ⟨τ_x⟩_anom | off-diagonal BdG u·v amplitudes probing Γ₆↔Γ₇ coherence | Weiss-field back-action when Δ≠0, Q≠0 |
| Density | `Σ_n [|u|²·f + |v|²·(1−f)]` / 4 | Chemical potential control |
| Pairing s | `u_A[6↑]·v_A[7↓]* − u_A[6↓]·v_A[7↑]*` (on-site) | s-channel gap equation |
| Pairing d | `u_A[6↑]·v_B[7↓]* − u_A[6↓]·v_B[7↑]*` (inter-site, φ(k) weight) | d-channel gap equation |

### 9. Exchange Rigidity: ∂²F_ex/∂Q²

`compute_JT_rigidity_from_exchange` computes:

```
∂²F_ex/∂Q² = O·(∂²J/∂Q²)·O + 4·(∂O/∂Q)·J·(∂O/∂Q)
            + 2·O·J·(∂²O/∂Q²) + 4·O·(∂J/∂Q)·(∂O/∂Q)
```

All four terms are included. The function receives the self-consistent chemical potential `μ` from the susceptibility computation, ensuring the BdG spectrum is evaluated at the correct Fermi level. At Q=0 the B₁g selection rule forces `∂O/∂Q = 0` and `∂²J/∂Q² = 0`, so only `2·O·J·(∂²O/∂Q²)` survives there; at Q≠0 all terms contribute and omitting any would bias the SCF Q-update. Positive `∂²F_ex/∂Q²` stiffens the phonon; negative softens it, which in the SC condensate can drive `K_eff < 0` — the SC-triggered JT instability.

### 10. B₁g Orbital Susceptibility χ_τ

```
chi_tau_sc = ∂⟨B1g_op⟩ / ∂(g_JT · Q)    (signed; evaluated in the SC state, Δ≠0)
```

`⟨B1g_op⟩` is computed via the full 16-component Nambu eigenstates using `B1g_16`, so that anomalous u·v amplitudes — which carry the SC-triggered orbital coherence — are fully included. Three step sizes h, h/2, h/4 provide Richardson-extrapolated central differences (O(h²)→O(h⁴)). The extrapolation additionally checks for nonlinearity: if the response changes by more than 20% between step sizes, the result is flagged as unreliable. The self-consistency flag (`richardson_ok`) requires both Richardson convergence (< 3% disagreement between extrapolation levels) and linear response.

The signed susceptibility means negative values indicate a JT-stiff direction (condensate suppresses rather than enhances orbital response). `δχ_τ = chi_tau_sc − chi_tau_n` isolates the SC-triggered excess:

```
δχ_τ ≡ 0 in D₄h by symmetry; small in D₂h
```

The step size adapts as:
```
h = clip(max(1e-3 · max(|Q|, Δ_CF/g_JT), 1e-4), 1e-4, 0.05·Δ_CF/g_JT)
```
to stay in the linear-response regime.

### 11. χ_QQ from Thermodynamic Finite Differences

The orbital JT susceptibility `χ_QQ = −∂²Ω/∂Q²` is evaluated numerically in the superconducting state (Δ≠0) by central finite-difference of the total free energy. Because the Nambu-basis analytic Lindhard summation for `χ_QQ` in the SC state — especially including anomalous u·v cross-terms — is prohibitively complex, the thermodynamic route via `∂²Ω/∂Q²` is both exact within mean field and numerically stable. This SC-state `χ_QQ` is used exclusively for lattice stability diagnostics (G-matrix); the pairing vertex always uses the normal-state (Δ=0) susceptibilities.

### 12. Coupled Spin–JT RPA Vertex and ∂λ_pair/∂Q

The pairing vertex is computed via a 2×2 coupled spin–JT RPA in `[spin, JT-phonon]` channel space. The bare interaction matrix is **diagonal**: `Û = diag(J_eff, V_JT)` — there is no bare S–Q cross-vertex; the spin–JT feedback enters exclusively through the off-diagonal susceptibilities χ_DQ_s/χ_QD_s, which are opened by SOC and the SC condensate:

```
V(q) = J_eff² χ_DD_s^RPA(q) + V_JT² χ_QQ^RPA(q) + J_eff V_JT [χ_DQ_s^RPA(q) + χ_QD_s^RPA(q)]
```

The bare susceptibilities χ₀(q) come from the Δ=0 BdG Hamiltonian via the Lindhard formula (4×4 orbital tensor, 8 normal Nambu sector pairs). The static Lindhard function is real by time-reversal symmetry — the imaginary part vanishes exactly at ω=0, so taking `chi0_tensor = chi0.real` after enforcing Hermiticity discards only numerical roundoff, not physical information. Projections:

```
χ_DD_s = Tr[Sz · χ₀[Γ₆,Γ₆] · Sz]      # spin–spin (dipole–dipole)
χ_DQ_s = Tr[Sz · χ₀[Γ₆,Γ₇]]            # spin–orbital cross (dipole–quadrupole)
χ_QQ   = −∂²Ω/∂Q²  (numerical, SC state)  # orbital JT stiffness [eV/Å²]
```

The cross-terms χ_DQ_s and χ_QD_s are **zero in the normal state at Q=0** (Γ₆–Γ₇ mixing forbidden) and become nonzero when Q > 0 opens the B₁g channel via B1g_op. A Padé resummation regularizes χ_DQ_s:

```
χ_DQ_s_v = χ_DQ_s / (1 + |χ_DQ_s| / w),    w = _CHI_DQ_S_PADE_W = 0.10
```

This is linear at |χ_DQ_s| ≪ w — continuously suppressing noise — and saturates asymptotically to ±w at large |χ_DQ_s|, with a smooth gradient near the QCP. χ_QQ is regularized via soft Dyson resummation:

```
χ_QQ_eff = χ_QQ / (1 + χ_QQ · V_JT / K_bare)
```

which is continuous and differentiable, saturates at `K_bare/V_JT` as χ_QQ→∞. The RPA determinant:

```
det = (1 − J_eff·χ_DD_s_moriya)(1 − V_JT·χ_QQ_eff/K) − J_eff·V_JT·χ_DQ_s_v·χ_QD_s_v
```

Spin fluctuations are regularised by Moriya damping (doping-dependent) rather than a hard cutoff:

```
Γ_M = α_M · J_eff · t_eff,    α_M = max(C · δ · (t_eff / J_eff), α_M_floor),    C = 0.45
```

This ensures `Γ_M → 0` at half-filling (long-range AFM, no damping) and grows with doping as metallic screening broadens the QCP. The floor `α_M_floor = _ALPHA_MORIYA = 0.05` prevents numerical runaway at very low doping and ensures `det(RPA)` never reaches `_RPA_DET_FLOOR` in the physically relevant near-QCP regime.

**RPA determinant treatment past the QCP:** when `det > 0` a floor `_RPA_DET_REG = 1e-9` guards against exact-zero numerical accidents only. When `det < 0` (past the QCP) the determinant is left intact — applying a soft cap to the vertex in this regime would trap the SCF in the unstable phase. The universal vertex cap `_RPA_V_SOFT_CAP = 50 eV` prevents numerical overflow without altering the sign or divergence character of V(q).

**Correlation correction to Moriya damping:** the quasiparticle weight `z_qp = 1/r_J` (from cluster-ED) introduces an excess `r_J_excess = max(0, r_J − 1)`. Only the overcorrelated part (r_J > 1) boosts Γ_M — at most doubling it — to prevent RPA runaway from vertex corrections that exceed the Gutzwiller-band picture. The `χ₀` bubble is left untouched: Ward identities require that the quasiparticle weight from the bubble (Z²) and the vertex correction (1/Z) cancel to Z, which is already encoded in the Gutzwiller-renormalized BdG bands feeding `χ₀`.

**Separate QCP tracking:** the vertex cache separately monitors the FM instability at q=0 (`det_q0`) and the AFM instability at q=(π,π) (`det_afm`). The SCF adaptive mixing and convergence tolerance respond to `det_afm`; the FM check guards against accidental ferromagnetic divergence. Both determinants are logged at convergence.

**∂λ_pair/∂Q > 0 is the key numerical criterion for the SC-triggered JT hypothesis.** A positive value confirms that an infinitesimal B₁g distortion increases the pairing strength through the spin-fluctuation channel. It is evaluated at Δ=0 with the converged SCF chemical potential.

**Susceptibility consistency:** χ₀ (normal state, Δ=0) is used for χ_DD_s, χ_DQ_s, χ_QD_s in the pairing vertex — feeding Δ≠0 susceptibilities back into the interaction would double-count the gap. `chi_QQ` (SC state, Δ≠0) is used exclusively for lattice stability diagnostics (G-matrix). The vertex is always built from the normal state: `solver._gap_amplitude = 0.0` is set before susceptibility evaluation, preventing the SC-state Padé threshold from inadvertently relaxing the normal-state cross-channel regularization.

### 13. Linearized Gap Equation and λ_JT_kernel

The pairing kernel on the Fermi surface:
```
Γ_ij = g_Δ · √(dl_i/vF_i) · V(k_i − k_j) · √(dl_j/vF_j)
```

Arc-length weights `dl_i` ensure proper Fermi-surface integration measure. `λ_max` = largest eigenvalue of Γ, with gap eigenvector φ_max. C₄ symmetry averaging is applied only when Q ≈ 0 (unbroken D₄h). The **JT-channel Rayleigh projection**:
```
λ_JT_kernel = φ_max^T · Γ_JT · φ_max
```
measures how much of λ_max comes specifically from the JT channel (V_JT component of V(q)), independently of the spin-fluctuation contribution. This is distinct from `lambda_JT_sc = (g²/K)·chi_tau_sc`, which is a scalar q=0 estimate.

The gap vector is projected onto s-wave and d-wave basis functions to determine the dominant pairing symmetry; channel-specific Gutzwiller factors `g_Delta_s` and `g_Delta_d` are applied separately. A 2×2 pairing kernel matrix in (s, d) channel space is also constructed to guide the SCF update direction toward the dominant instability even when `|Δ| ≈ 0`.

### 14. G-Matrix: SC–JT Coupled Instability

The coupled SC–JT instability boundary is tracked via a 3×3 G-matrix (s-channel, d-channel, Q-mode) in `_build_G3_matrix`:

```
G3 = ┌ 1 − gVs·χ_ss     −√(gVs·gVd)·χ_sd   −cs·χ_sQ ┐
     │ −√(gVs·gVd)·χ_sd  1 − gVd·χ_dd       −cd·χ_dQ │
     └ −cs·χ_Qs          −cd·χ_Qd        1 − χ_QQ/K_eff ┘
```

where `cs = √(gVs/K_eff)`, `cd = √(gVd/K_eff)`. The rigidity computation uses `μ_n` from the analytic 2-band model as the chemical potential, ensuring the BdG is evaluated at the correct Fermi level. The Hessian finite-difference step for the Q direction uses `eps_Q = max(5e-3·λ_hop, |Q|·1e-3·λ_hop)` — the floor of ~6.4×10⁻³ Å ensures signal/noise ≫ 1 even at Q=0, where a smaller step would be dominated by LAPACK numerical noise.

**Interpretation of `λ_min` and `G22` depending on evaluation context:**

| Context | `λ_min` | `G22` | Physical meaning |
|---|---|---|---|
| Normal state (Δ=0, pre-SCF) | > 0 | > 0 | JT mode stable — SC-triggered JT viable |
| Normal state (Δ=0, pre-SCF) | < 0 | < 0 | Spontaneous JT in normal state — wrong mechanism |
| Post-SCF (Δ≠0, converged) | < 0 | > 0 | SC condensate triggered JT — correct and desired |
| Post-SCF (Δ≠0, converged) | < 0 | < 0 | Both modes soft — unphysical |

`G3[2,2] = 1 − χ_QQ/K_eff`. The Schur complement of G22 gives the effective pairing enhancement diverging as `G22 → 0⁺`. G3[2,2] captures spontaneous-JT risk from any source, including large Delta_inplane (D₂h spin-conserving elements in B1g_op). Full diagnostics via `compute_G_instability(compute_dlambda=True/False)`.

### 15. SC-JT Coexistence Window

`check_sc_jt_window` verifies that `K_lattice` lies in the cooperative SC–JT window:

```
K_spont = g_JT² / Δ_CF                           (spontaneous JT threshold; K_lattice must exceed this)
K_SC    = g_JT² · chi_tau_sc / _LAMBDA_JT_VIABLE  (SC-triggered threshold; K_lattice must be below this)
```

`_LAMBDA_JT_VIABLE = 0.05` is a fixed physical viability criterion: K_SC is the stiffness above which `λ_JT_sc = g²·χ_τ_sc/K` drops below 5%, independent of the normal-state pairing eigenvalue. The window condition reduces to `χ_τ_sc · Δ_CF > _LAMBDA_JT_VIABLE` (independent of `g_JT`).

Two λ_JT metrics are tracked:
- `lambda_JT_sc = g²·chi_tau_sc/K_lattice` — predictive, based on full SC-state susceptibility
- `lambda_JT = g²·δχ_τ/K_lattice` — post-hoc confirmation, SC-induced excess only

`structural_ok` requires both `g²·χ₀ < K_eff` (G-matrix positivity) and `λ_min > 0` (no normal-state spontaneous instability). If `λ_min ≤ 0`, `normal_unstable = True` is flagged and `viable = False` regardless of the window boundaries.

`K_opt = √(K_spont · K_SC)` is the geometric midpoint.

### 16. Tc and Gap Ratio

Two independent Tc estimates are computed. A preliminary log block (`TC-PRELIM`) is printed before Bayesian optimisation:

- **Tc₁ (G-BCS analytic):** uses `λ_eff = N_eff · V_eff` from the G-matrix (Schur-complement corrected), giving a pre-SCF upper bound.
- **Tc₂ (λ_max-BCS):** `Tc = 1.13 · ω_c · exp(−1/λ_max)` with cutoff `ω_c = max(t_eff, ω_JT)`. In the SC-triggered JT picture the JT phonon energy `ω_JT` sets the relevant boson scale when it exceeds the effective bandwidth.

Post-optimisation:
- `compute_Tc_by_gap_suppression`: bisects in T to find where `|Δ(T)| < Delta_tol` via full re-SCF with warm-starting from a normal-state seed (Δ ≈ 0) at each temperature. Finds only the spinodal (second-order instability boundary).
- `compute_Tc_thermodynamic`: warm-start *upward* temperature scan from the T≈0 SC+JT basin. For the SC-triggered JT mechanism the effective Landau potential can have a negative quartic coefficient, making the transition first-order: the system jumps to finite (Δ, Q) at a Tc* where the normal state is still locally stable. Cooling from Δ≈0 misses this entirely. This method correctly identifies the thermodynamic Tc including such first-order crossings. Returns transition order, Δ_jump, hysteresis, and an SC-JT uplift percentage.
- `compute_lambda_vs_T`: tracks linearized gap eigenvalue `λ_max(T)`; Tc at `λ_max = 1` crossing; detects non-monotone λ(T) and logs a warning with all crossing temperatures.

`compute_gap_ratio` reports `2Δ₀ / k_B Tc`; values above 3.52 (BCS weak-coupling) indicate SC-JT strong-coupling enhancement. The ratio > 3.52 is expected for two reasons: the JT feedback cooperatively enhances Δ₀ beyond the linearised BCS value, and proximity to the AFM QCP suppresses Tc relative to Δ₀ via pair-breaking spin fluctuations.

### 17. Variational Free Energy and DMFT-like Decomposition

The total free energy splits into two physically distinct components without double-counting:

```
F_total = F_BdG + F_cluster
```

This is a Luttinger–Ward / Baym–Kadanoff variational decomposition: **F_BdG** covers itinerant electrons (mean-field BdG spectrum), while **F_cluster** covers local quantum fluctuations (2-site ED capturing AFM correlations, multipolar fluctuations, and vertex renormalization of J_eff). The Gutzwiller factors handle kinematic Mott renormalization; the cluster ED handles irreducible vertex renormalization of J_eff only (not the susceptibility bubble); RPA handles the reducible ladder summation on the full Brillouin zone. These three levels are orthogonal and there is no double-counting, because cluster-ED outputs a renormalized coupling (J_eff), while RPA inputs that coupling to compute the full-BZ susceptibility — two non-overlapping operations.

```
Ω_BdG = (1/2) Σ_{k,n} w_k [E_n(k) f(E_n) − T S(f_n)]
        + |Δ_s|²/(g_Delta_s · V_s) + |Δ_d|²/(g_Delta_d · V_d)
        + (K_eff/2)Q²
```

`V_s` and `V_d` are required positional arguments to `compute_bdg_free_energy`; they must match the vertex used in the SCF gap equation so that `∂F/∂Δ = 0` at the converged point. The condensation correction terms use **independent** Gutzwiller factors per channel: `g_Delta_s = g_t` (on-site), `g_Delta_d` interpolated between `g_t` and `g_J` by Γ₇ admixture.

### 18. Analytic ∂F/∂M and ∂²F/∂M² (Single Diagonalization)

```
∂F/∂M   = Σ_{k,n} f_n · ⟨ψ_n|∂H/∂M|ψ_n⟩                              (Hellmann–Feynman)
∂²F/∂M² = Σ_{k,n} (∂f_n/∂E_n) · ⟨ψ_n|∂H|ψ_n⟩²                        (diagonal term)
         + Σ_{k,n≠m} (f_n − f_m)/(E_m − E_n) · |⟨ψ_n|∂H|ψ_m⟩|²       (off-diagonal term)
```

Computed analytically from a single BdG diagonalization via second-order perturbation theory. The Newton step for M uses the analytic curvature with Levenberg–Marquardt regularization floor `_MU_LM = 3.5`; the LM floor is adaptively reduced as `|Δ|` grows to allow M to relax as SC develops. The Newton vs. BdG fixpoint blend is `_ALPHA_HF = 0.25`.

### 19. Two-Site Cluster: Quantum Multipolar Fluctuations and J_eff Renormalisation

Beyond BdG mean field, a 2-site (A–B) cluster is exactly diagonalized at each iteration:

```
H_cluster = H_sp(A) ⊗ I + I ⊗ H_sp(B)
          + J_bond · O_A ⊗ O_B
          + Z_boundary · J_bond · M_ext · (O_A ⊗ I + I ⊗ O_B)
```

where `O = multi_op` (pre-built in `ModelParams.__post_init__`) and `J_bond = effective_superexchange(g_J, tx_bare, ty_bare, doping)` is the single-bond exchange (Z-factor applied at the call site). `ClusterMF` receives `multi_op` and `Z` as constructor arguments; the operator is not rebuilt per iteration. The boundary Weiss field scales as `g_J·(1−δ)`, consistent with the BdG Weiss field. The cluster computes both `⟨B1g_op⟩` (classical JT order parameter) and `√⟨B1g_op²⟩` (RMS including quantum fluctuations).

**Cluster J_eff renormalisation (Hellmann–Feynman):** after diagonalizing H_cluster, a renormalised exchange is extracted via weighted covariance of the full spectrum:

```
J_eff_cluster = Cov_w(E_n, ⟨O_AB⟩_n) / Var_w(⟨O_AB⟩_n)
```

with Boltzmann weights `w_n = exp(−E_n/kT)/Z`. The result is clipped to `[0.5, 2.0] × J_eff_bare` to prevent runaway corrections and ensure SCF convergence. The ratio `j_renorm = J_eff_cluster / J_eff_bare` feeds back into the Moriya damping correction (`Γ_M` boosted proportionally to `r_J_excess = max(0, 1/j_renorm − 1)`) and is logged at convergence. The τ_x orbital fluctuations appear only at this cluster-ED level; in the BdG, J_eff remains a scalar.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

```
∂n/∂μ = Σ_{k,n} w_k · f(E_n)(1−f(E_n)) / kT · (|u_A|² + |u_B|² + |v_A|² + |v_B|²)
```

Newton's method with the analytic derivative from the same BdG eigensystem; Brent's method as fallback. The `(ev, ec)` from the μ-search is reused directly for subsequent observable computation, avoiding a redundant `eigh` call.

### 21. Gap Equations: Full-BZ Integration

`compute_gap_eq_vectorized` evaluates gap equations over the full Brillouin zone using uniform k-grids:

```
F_AA_BZ = Σ_k w_k · Pair_s(k) / 4    (on-site s-channel)
F_AB_BZ = Σ_k w_k · Pair_d(k) / 4    (inter-site d-channel; d-wave projection on vertex side)
```

The `/4` corrects for the 16-dimensional BdG space doubling. The d-wave projection is applied via an explicit `φ(k)` projection of `V(k−k')`. `V_d_scalar` is clamped to be non-negative; when fewer than 6 Fermi-surface points are available or no d-wave amplitude is present, `V_d_scalar = 0` (d-channel contributes nothing) rather than falling back to the s-channel value.

---

## Model Architecture

```
ModelParams  (dataclass, __post_init__)
    ├── Primary: t_pd, u, lambda_soc, Delta_tetra, g_JT, K_lattice,
    │            lambda_hop, Delta_inplane, Delta_CT, omega_JT, Z, kT, tol
    ├── Derived: Delta_CF, g7split, t0, U, U_mf, J_CT, doping_0, _U4, U_gamma,
    │            eta (from Sz matrix elements), multi_op (pre-built multipolar operator),
    │            _w6_xz/_w6_yz/_w6_xy, _w7_xz/_w7_yz/_w7_xy (orbital weights for η_J(Q))
    └── Grid objects: k_points, k_points_even, k_weights, k_weights_even,
                      shift_table, N_k, N_k_even

Module-level SCF constants (not in ModelParams):
    _NK = 70          # k-points per direction (must be even for commensurate q_AFM = (π,π)); parity is asserted at solver initialization.
    _MAX_ITER = 800   # maximum SCF iterations
    _MIXING = 0.06    # base Anderson mixing weight
    _MU_LM = 3.1      # LM regularization floor for M Newton step
    _ALPHA_HF = 0.31  # Newton vs BdG fixpoint blend for M

ClusterMF  (2-site exact diagonalization)
    ├── __init__(multi_op, Z)           — receives pre-built multi_op from ModelParams
    ├── build_cluster_hamiltonian(...)  — Weiss field scales as g_J·(1−δ)
    └── cluster_expectation(evals, evecs, O, T, site_index)

VectorizedBdG  (performance kernel, lives inside RMFT_Solver)
    ├── _build_H_stack(kpts, ..., out=)  → (N, 16, 16) BdG stack
    ├── compute_observables_vectorized(...) → M, tau_x, B1g_exp, density, Pair_s/d
    └── compute_gap_eq_vectorized(...)  → RPA vertex + full-BZ gap equations

RMFT_Solver
    ├── SusceptibilityMixin
    │   ├── get_susceptibilities_fast    analytic 2-band (DE scout, G-matrix pre-SCF)
    │   ├── get_susceptibilities_normal  full Lindhard χ₀(q) tensor (pairing vertex)
    │   │       returns: chi_DD_s, chi_DQ_s, chi_QD_s, chi_DD_s_moriya, rpa_det, ...
    │   └── get_susceptibilities_sc      SC-state χ_QQ via ∂²Ω/∂Q²
    ├── _get_chi0_norm_cache(...)        Δ=0 eigenvector cache across q-loop and iterations
    ├── _rebuild_orbital_operators()     rebuild B1g_op, B1g_16, multi_op, sz_op after SOC/CF change
    ├── _reset_transient_state()         safe clone reset for parallel workers
    ├── compute_chi0_tensor(...)         (4,4) orbital susceptibility tensor
    ├── compute_static_chi0_afm(...)     q=0 static χ_DD_s in folded 2-sublattice BdG
    │       returns: chi_DD_s, chi_DD_s_moriya, rpa_factor, afm_unstable
    ├── solve_linearized_gap_equation(.) λ_max, gap vector, arc-length weights, λ_JT_kernel, gap symmetry
    ├── compute_G_instability(δ, M,
    │       compute_dlambda=True)        G3 matrix, ∂λ_pair/∂Q, Tc estimate
    ├── compute_hessian(M, Q, Δ, δ, μ,
    │       g_t, g_J, Δ_s_frac,
    │       V_s, V_d, K_eff, [cache])   post-SCF SC-JT Hessian (3×3 curvature)
    ├── compute_bdg_free_energy(...)     variational F_BdG; V_s, V_d required positional args
    ├── compute_cluster_free_energy(...) F_cluster (2-site ED; J_eff renorm via Hellmann–Feynman)
    ├── compute_Tc_by_gap_suppression(.) Tc via bisection on Δ(T); finds spinodal only
    ├── compute_Tc_thermodynamic(...)    warm-start upward scan; first-order aware
    ├── compute_lambda_vs_T(...)         λ_max(T) curve, Tc at λ_max=1; non-monotone detection
    ├── compute_gap_ratio(...)           2Δ₀/kTc strong-coupling diagnostic
    ├── compute_coherence_length(...)    ξ/a; orbital-resolved ξ_Γ6, ξ_Γ7
    ├── _B1g_expectation(...)            per-site ⟨B1g_op⟩ from full 16-component Nambu eigenstates
    ├── _compute_chi_tau(...)            Richardson-extrapolated B1g finite-difference χ_τ
    │       returns: chi_tau_sc, chi_tau_n, delta_chi_tau, richardson_ok, nonlinear flags
    ├── summary(delta, M0)               human-readable parameter and diagnostic summary
    ├── _scf_jacobi_kick(...)            analytic λ₊ estimate → initial (M, Q, Δ) seed and α
    ├── _find_mu_for_density(...)        Newton (analytic ∂n/∂μ) + Brent fallback
    ├── _anderson_mix(...)               quasi-Newton Anderson(5) acceleration
    └── solve_self_consistent(...,
            _ic_retry=False)             Anderson-accelerated SCF loop;
                                         auto-retries with softened AFM seed if incommensurate
                                         tendency detected (single recursion via _ic_retry guard)

UnifiedBayesianOptimizer  (5D: Δ_tetra, λ_soc, u, g_JT, t_pd)
    ├── _eval_constraints(s, doping)     two-phase H1–H4 + S1–S5 constraint evaluation
    ├── _eval_one_doping(...)            full SCF + dual-basin JT probe + scoring
    ├── _score(..., lambda_JT)           three-tier multiplicative gate scoring
    ├── run_de_phase(...)                Phase 1: DE scout (analytic G-matrix only)
    ├── run_gp_seed_phase(...)           Phase 2: top-k DE → full SCF → GP seed
    ├── run_turbo_phase(...)             Phase 3: trust-region GP-EI, batch parallel
    ├── run_local_refinement(...)        Phase 4: dense random sampling around global best
    └── optimize(...)                   orchestrates all four phases

check_sc_jt_window(...)                 K_lattice window diagnostic (standalone function)

Visualization
    ├── plot_phase_diagrams(solver, δ_scan, opt_result)   3×3 (or 4×3) panel figure
    ├── _plot_phase_data(ax, phase_data)
    └── _plot_dos(ax, solver, result)
```

---

## Key Algorithms

### VectorizedBdG: Batched LAPACK and Buffer Reuse

- `_build_H_stack(kpts, out=)` accepts an optional pre-allocated `(N, 16, 16)` buffer; on the hot SCF path no heap allocation occurs per iteration. Hermiticity is enforced after assembly.
- The per-iteration BdG eigensystem `(ev, ec)` is computed **once** and shared by observable computation, dual-channel gap equations, and ∂F/∂M. Stored in `self._scf_bdg_cache`, cleared after use.

### χ₀(q) Permutation Tricks

The k-grid (endpoint=False) is constructed so that for any `q = (nx, ny) · 2π/_NK`, the k+q grid is a cyclic permutation of the k-grid. The precomputed `shift_table[nx, ny]` (built in `ModelParams.__post_init__`, shape `(_NK, _NK, N_k_even)`, dtype int32) implements this as a free index reorder:

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]   # no LAPACK, just index reorder
```

For the static AFM susceptibility χ_DD_s, the computation is performed at q=0 in the folded basis: `E_kQ_all = E_k_all` and `V_kQ_all = V_k_all` — no permutation needed, since the sublattice stagger in `sz_bdg16` already selects the (π,π) channel. The solver-level `_get_chi0_norm_cache` additionally caches the Δ=0 eigenvectors across calls with the same `(M, Q, mu, tx, ty, g_J, target_doping)` within tolerance `_CHI0_CACHE_TOL = 1e-5`.

### Dual k-Grid Setup

Two separate k-grids generated once in `ModelParams.__post_init__`, both endpoint=False with uniform 1/N weights (Σw_k = 1):

- **SCF grid (_NK):** BdG diagonalization, observables, free energy, gap equations.
- **χ₀ grid (same _NK):** χ₀(q) and pairing kernel, exploiting the shift_table permutation trick.


### Thread-Safety and Clone Protocol

`RMFT_Solver` is cloned with `copy.copy()` before each parallel SCF worker:
```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s.p.__post_init__()
s._K_bare = s.p.K_lattice
s._rebuild_orbital_operators()
s._reset_transient_state()
s.cluster_mf = ClusterMF(s.multi_op, s.p.Z)  # re-instantiate with updated operator
```
`_reset_transient_state` ensures each clone owns its own `VectorizedBdG` instance (and thus its own `_H_stack` buffer), preventing inter-worker memory aliasing. `OMP_NUM_THREADS=1` prevents BLAS thread oversubscription inside `ThreadPoolExecutor`.

### Hierarchical Q Update

Q is updated only every `_Q_UPDATE_PERIOD = 4` iterations via the Hellmann–Feynman force. In the intervening inner iterations, (Δ, M, μ) are converged at frozen Q, consistent with the adiabatic lattice timescale. This hierarchical scheme prevents oscillatory Q–Δ coupling that would arise from updating all order parameters at the same rate.

### Vertex Cache Invalidation

The RPA pairing vertex is rebuilt when any of the following thresholds are exceeded:

| Variable | Threshold | Constant |
|---|---|---|
| M | absolute > 0.04 | `_M_THR_REL` |
| Q | absolute > 3% of λ_hop | `_Q_THR_REL` |
| j_renorm | absolute > 0.05 | — |
| doping | absolute > 0.005 | — |
| `chi_QQ_from_normal` flag | False (cache not built from Δ=0) | — |

The vertex is always built from the normal state (Δ=0): there is no Δ-threshold invalidation. `chi_QQ_bare_v` is evaluated in the SC state (Δ≠0) for the lattice stability branch; the pairing vertex inputs are always from the normal state. The cache additionally stores `det_afm`, `chi_DD_s_full`, and `chi_DD_s_moriya_full` from the AFM susceptibility evaluated at q=(π,π).

### Limit-Cycle Detection

The SCF monitors `|Δ|` over the last `_CYCLE_WINDOW = 20` iterations. If the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.30`, an oscillatory regime is diagnosed and α is reduced by `_CYCLE_DAMP_FAC = 0.50` with an Anderson history reset. This prevents the SCF from stalling in a limit cycle near the JT-active onset where the Q–Δ feedback is most nonlinear.

### SCF Loop (`solve_self_consistent`)

Anderson(5)-accelerated iteration over (M, Q, Δ_s, Δ_d, μ):
1. Build and diagonalize the 16×16 BdG stack; cache `(ev, ec)` for the iteration.
2. Compute `tau_x = obs['tau_x']` (off-diagonal Γ₆↔Γ₇ mixing) and `B1g_exp = obs['B1g_exp']` (full Hellmann–Feynman force) from `compute_observables_vectorized`.
3. Update M via Levenberg–Marquardt-regularized Newton step + BdG fixpoint blend `(1−_ALPHA_HF)·fixpoint + _ALPHA_HF·Newton`. LM floor `_MU_LM = 3.5` decreases as `|Δ|` grows.
4. Inject anomalous orbital coherence `⟨τ_x⟩_anom` (from SC condensate, computed via `_compute_orbital_coherence_from_pairs`) into the Weiss field when Δ≠0 and Q≠0, then rebuild BdG cache. This is the mean-field back-action loop.
5. Update `K_eff` on iteration 0 and when `|ΔQ| > _Q_THR_REL·λ_hop` or `|ΔM| > 0.02`, tracked via `_K_eff_last_Q` and `_K_eff_last_M`.
6. Solve gap equations for (Δ_s_out, Δ_d_out) via RPA vertex fixed-point. Blend in 2×2 pairing kernel eigenvector direction (weight `_ALPHA_MIX_2X2 = 0.35`) to prevent channel locking.
7. Update cluster free energy (DMFT-like vertex renormalization of J_eff via Hellmann–Feynman extraction from full cluster spectrum).
8. **Update Q via Hellmann–Feynman every `_Q_UPDATE_PERIOD` iterations:** `Q_out = −(g_JT/K_eff)·B1g_exp` (full B1g operator, not just τ_x).
9. Apply Anderson(5) acceleration to `[M, Q/λ_hop, |Δ_s|·t0, |Δ_d|·t0]` jointly.
10. Find μ to enforce `⟨n⟩ = 1 − δ`; reuse `(ev, ec)` from μ-search; compute F_BdG and F_cluster.
11. Adaptive mixing every 5 iters: halve α on divergence (max_diff > 1.05×prev), boost ×1.35 when `selection_ratio > 0.05` and `|Q| > 1e-4` (JT-active), damp ×0.8 when JT-inactive, cap near AFM QCP (×0.6); reset Anderson history on divergence, stagnation, or Q sign flip. Limit-cycle detector reduces α by `_CYCLE_DAMP_FAC` on oscillation.

After convergence: post-convergence Hessian test (3×3 `∂²F/∂{M,Q,Δ}²` with physical-scale normalisation; mode classification uses scaled eigenvector components), coherence length ξ/a, SC-triggered JT confirmation (hessian_lmin_sc < −kT), λ_JT_kernel, ∂λ_pair/∂Q, channel decomposition (λ_s vs λ_d). A Mott filter suppresses the gap if `g_t < 0.10` or `ξ/a < 1.0`.

**Incommensurate AFM auto-retry:** after convergence a scan over `q = (π, π−δq)` with δq ∈ [0, 0.15π] checks whether the AFM susceptibility χ_DD_s peaks away from (π,π). If `δq_max > 0.05π`, `solve_self_consistent` automatically re-runs with a softened AFM seed (`M → 0.85M`) via a single recursive call guarded by the `_ic_retry` flag.

The result dict includes: all converged order parameters, Hessian eigenvalues, G3-matrix diagnostics, λ_JT, λ_JT_sc, λ_JT_kernel, ∂λ_pair/∂Q, gap symmetry, channel decomposition, coherence length ξ/a, 2Δ₀/kTc, `chi_tau_sc`, `chi_tau_n`, `delta_chi_tau`, `richardson_ok`, `selection_ratio`, `chi_DD_s`, `chi_DD_s_moriya`, `chi_DD_s_full`, `chi_DD_s_moriya_full`, `rpa_factor`, `afm_unstable`, `j_renorm`, `incommensurate_dq`, `incommensurate_chi_ratio`.

### Unified Bayesian Optimisation (5D)

`UnifiedBayesianOptimizer` searches `(Δ_tetra, λ_soc, u, g_JT, t_pd)` in four phases:

**Phase 1 — DE scout:** `scipy.differential_evolution` with analytic G-matrix only (no SCF). Two-phase constraint evaluation:
- *Phase 1 (cheap):* `compute_G_instability(compute_dlambda=False)` → H1–H4, S1–S3, S5. Pre-SCF Mott hard-reject at `g_t < 0.10`. Early exit if partial_penalty ≥ 0.25.
- *Phase 2 (expensive, only for promising candidates):* `compute_G_instability(compute_dlambda=True)` → S4 (∂λ_pair/∂Q > 0).

**Phase 2 — GP seed:** top-k DE feasible candidates evaluated with full SCF; results seed the ARD Matérn-2.5 GP.

**Phase 3 — TuRBO:** trust-region GP-EI acquisition, batch parallel via `ThreadPoolExecutor`. TR shrinks on failure (×0.65), expands on consecutive improvement (×1.35). TR state is mutated only from the main thread after each batch; `_register()` is thread-safe via `_gp_lock`.

**Phase 4 — local refine:** dense random sampling in a ±margin hypercube around the global best.

**Hard constraints (H1–H4):** score = 0, excluded from GP training set:
- H1: `∂²F/∂Q²|_{Δ=0} > 0` — normal-state Q-stability (no spontaneous JT)
- H2: `J_eff · χ_DD_s(Moriya) < 1` — below Stoner QCP (uses Moriya-damped susceptibility; falls back to SC-state gapped χ if past QCP, capped at 0.98)
- H3: `G22 > 0` — JT channel not self-crossing in normal state
- H4: `g_t ≥ 0.10` — coherent Fermi surface (Mott guard)

**Soft constraints / DE penalty (S1–S5, weights sum to 1.0):**
- S1 (w=0.225): `0 < λ_min(G3) < 0.15` — near-critical, not past QCP
- S2 (w=0.225): monotonic reward for larger λ_max; only penalises near-divergence (λ_max > 0.95) and unsolvable cases — small λ_max in the normal state is not penalised, consistent with first-order transition hypothesis
- S3 (w=0.180): `λ_JT > 0.05` — SC-JT coupling above threshold
- S4 (w=0.270): `∂λ_pair/∂Q > 0` — JT renormalises V_pair upward
- S5 (w=0.100): G22-margin > `_DE_G22M_SAFE = 0.25` — distance from the spontaneous-JT boundary; `S5 = 1 − tanh(G22 / _DE_G22M_SAFE)` continuously penalises proximity to G22 = 0

**Scoring (`_score`)** — three-tier multiplicative architecture:
- *Tier 1 (hard guards):* Mott/incoherence guard (`g_t < 0.10` or `ξ/a < 1`), `J·χ_DD_s(Moriya) > 2` → score = 0. Projection-quality penalty for large `(J_eff/Δ_CF)²`.
- *Tier 2 (smooth mechanism weights):*
  - `w_lJT`: parabolic arch on [0,1], peak at λ_JT = 0.45
  - `w_lJT_kernel`: sigmoid(10·(lJTk − 0.05))
  - `w_hessian`: sigmoid(−λ_min_SC / 0.05), floor 0.30
  - `w_softening`: sigmoid on SC-induced Q-mode softening (d²F_Q_sc − d²F_Q_n)
  - `w_chisq`: sigmoid(|χ_SQ| / 0.1) — spin-orbital cross-channel strength
- *Tier 3 (objective):* `Tc_proxy × conv_f × stoner_f × g22_margin_f × xi_f × lmax_boost × jchi_gate`
  - `lmax_boost = 0.6·softplus(λ_max) + 0.4·(∂λ/∂Q)·σ(10·(λ_max−0.70))/0.5`
  - `jchi_gate`: Gaussian reward near optimal `J·χ_DD_s = 0.875` (near-QCP but metallic)
  - `g22_margin_f = sigmoid((G22 − _BO_G22_MARGIN_CTR) / _BO_G22_MARGIN_W)` — continuously rewards distance from the spontaneous-JT boundary

---

## Parameters

All energies in **eV**, lengths in **Å**.

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | t_pd | 0.440 eV | pd hybridisation integral (primary hopping; t₀ = t_pd²/Δ_CT derived) |
| `u` | u | 8.0 | U/t₀ ratio; Hubbard U = u·t₀ |
| `lambda_soc` | λ_SOC | 0.180 eV | Atomic SOC constant (t₂g shell); determines Γ₆–Γ₇ splitting |
| `Delta_tetra` | Δ_tet | −0.300 eV | Tetragonal CF (**required < 0**); Δ_CF derived |
| `g_JT` | g_JT | 0.150 eV/Å | Electron–phonon JT coupling |
| `K_lattice` | K | 0.800 eV/Å² | Bare phonon stiffness; K_eff computed at runtime |
| `lambda_hop` | λ_hop | 1.100 Å | Hopping decay: t(Q) = t₀·exp(±Q/λ) |
| `Delta_CT` | Δ_CT | 1.800 eV | Charge-transfer gap (material-class constant) |
| `Delta_inplane` | Δ_ip | 0.020 eV | B₂g in-plane CF; splits Γ₇ doublet |
| `omega_JT` | ω_JT | 0.057 eV | JT phonon frequency (~46 meV) |
| `kT` | kT | 0.010 eV | Temperature (~116 K) |
| `tol` | — | 1e-4 | Convergence threshold |
| `Z` | Z | 4 | Coordination number |

### Module-level SCF Constants

These are fixed at compile time and not Bayesian-optimised:

| Constant | Value | Description |
|---|---|---|
| `_NK` | 70 | k-points per direction (must be even) |
| `_MAX_ITER` | 800 | Maximum SCF iterations |
| `_MIN_ITER` | 4 | Minimum iterations before convergence check |
| `_MIXING` | 0.05 | Base Anderson mixing weight |
| `_MU_LM` | 3.5 | LM regularization floor for M Newton step |
| `_ALPHA_HF` | 0.25 | Newton vs BdG fixpoint blend for M |
| `_FS_N_VERTEX` | 100 | FS k-points used in the vertex q-loop |
| `_Q_UPDATE_PERIOD` | 4 | Update Q every N inner iterations |
| `_ALPHA_MIX_2X2` | 0.35 | Blend weight: 2×2 eigenvector direction vs fixed-point gap update |
| `_MORIYA_C` | 0.45 | Moriya damping prefactor α_M = C·δ·(t_eff/J_eff) |
| `_ALPHA_MORIYA` | 0.05 | Moriya damping floor |
| `_LAMBDA_JT_VIABLE` | 0.05 | Minimum λ_JT_sc for SC-triggered JT viability |
| `_CHI_DQ_S_PADE_W` | 0.10 | Padé regularisation width for χ_DQ_s |
| `_RPA_V_SOFT_CAP` | 50.0 eV | Universal vertex overflow cap (all det values) |
| `_RPA_DET_REG` | 1e-9 | Det floor applied only when det > 0 |
| `_ANDERSON_TIKHONOV` | 1e-8 | Tikhonov β / diag_max in Anderson normal equations |
| `_ANDERSON_TRUST` | 2.5 | Trust-region step-size limit (multiples of simple step) |
| `_CYCLE_WINDOW` | 20 | Iteration window for limit-cycle detection |
| `_CYCLE_THRESHOLD` | 0.30 | std/mean of |Δ| above this → oscillatory regime |
| `_CYCLE_DAMP_FAC` | 0.50 | α reduction factor on oscillation detection |
| `_G_T_COHERENCE_MIN` | 0.10 | g_t floor for coherent ZRS band (Mott guard) |
| `_BO_MAX_WORKERS` | 6 | ThreadPoolExecutor worker ceiling |

### Derived Parameters (from `__post_init__`)

| Parameter | Formula | Description |
|---|---|---|
| `Delta_CF` | from SOC+CF diag. | Γ₆–Γ₇ splitting (not a free parameter) |
| `g7split` | from SOC+CF diag. | Γ₇a–Γ₇b internal splitting |
| `eta` | `\|⟨Γ₇\|S_z\|Γ₇⟩\| / \|⟨Γ₆\|S_z\|Γ₆⟩\|` | Γ₇ AFM asymmetry (derived from eigenvectors) |
| `multi_op` | `diag([1,−1,η,−η])` | Multipolar spin operator; pre-built and shared by cluster and BdG |
| `_w6_xz` … `_w7_xy` | from eigenvector projections | d_xz/d_yz/d_xy orbital weights of Γ₆, Γ₇a; used for Q-dependent `η_J(Q)` |
| `t0` | t_pd²/Δ_CT | Effective dd hopping |
| `J_CT` | 2t_pd⁴/Δ_CT²·(1/U+1/(Δ_CT+U/2)) | ZSA CT superexchange (single-bond; factor 2 from two virtual pathways) |
| `U_mf` | Z·J_CT/2 | Bare Weiss-field amplitude (g_J·(1−δ) applied at runtime) |
| `doping_0` | z_ZRS/(1−z_ZRS) | ZRS coherence crossover; floor in f_J(δ) only |

### 5D Optimisation Search Bounds

| Parameter | Bounds |
|---|---|
| `Delta_tetra` | (−0.09, −0.03) eV |
| `lambda_soc` | (0.18, 0.34) eV |
| `u` | (10.0, 20.0) |
| `g_JT` | (0.11, 0.24) eV/Å |
| `t_pd` | (0.40, 0.60) eV |

### SC+JT Coexistence Conditions

Four independent conditions checked by `compute_G_instability` and `check_sc_jt_window`:

1. **Metallicity:** `h_AFM < 2·g_t·t₀` — AFM gap does not swallow the Fermi surface.
2. **Mott coherence:** `g_t ≥ 0.10` — ZRS band coherent enough for SC pairing.
3. **Normal-state JT stability:** `K_eff > 0` and `λ_min > 0`, i.e. `G3[2,2] > 0` at Δ=0 and no normal-state spontaneous instability.
4. **SC-triggered regime:** `lambda_JT_sc = g_JT²·chi_tau_sc/K_lattice > _LAMBDA_JT_VIABLE = 0.05`.

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
1. SOC+CF diagonalization → Δ_CF, k-grids, `shift_table`, orbital operators (B1g_op, B1g_16, multi_op).
2. `solver.summary(target_doping)` — all derived parameters and pre-SCF diagnostics.
3. Reference SCF at default parameters → self-consistent (M, Q, Δ, μ) for diagnostics.
4. `compute_G_instability()` at self-consistent M + `check_sc_jt_window()` with χ_τ_sc from post-SCF.
5. Linearized gap equation and channel decomposition from SCF result dict.
6. Preliminary Tc₁/Tc₂ log block (G-BCS analytic and λ_max-BCS with JT phonon cutoff).
7. If `need_optimalization = True`: `UnifiedBayesianOptimizer.optimize()` — DE scout → GP seed → TuRBO → local refine.
8. Post-SCF: Hessian, coherence length, `compute_Tc_thermodynamic`, `compute_lambda_vs_T`, gap ratio, phase-diagram scan.

The flag `need_optimalization` (default `False`) controls whether the Bayesian optimisation pipeline runs.

---

## Output & Diagnostics

### Iteration Log

Each SCF step prints (thread-safe): M, Q, Δ_s, Δ_d, density n, μ, F, det_FM, det_AFM, K_eff, selection_ratio, j_renorm.

### Convergence Report

At convergence, two log lines are printed:

**Line 1:** converged order parameters, Hessian summary (H=[λ₀,λ₁,λ₂] ✓MIN or ⚠SADDLE), gap symmetry, λ_max, JT active flag (selection_ratio > 0.05), coherence length note.

**Line 2:** χ_AFM diagnostics — `χ_DD_s`, `χ_moriya`, `J·χ`, `det_AFM`, near-QCP flag, `j_renorm`.

Additionally: channel decomposition (λ_s vs λ_d), λ_JT, λ_JT_sc, λ_JT_kernel, ∂λ_pair/∂Q, SC-triggered JT confirmation (hessian_lmin_sc < −kT), 2Δ₀/kTc, χ_τ breakdown (chi_tau_sc, chi_tau_n, δχ_τ, richardson_ok). SC-JT window diagnostics (K_eff, K_spont, K_SC, K_opt, K_distance, in_window, lambda_JT_sc, lambda_JT_opt). Incommensurate AFM scan result (dq_max/π, χ ratio, auto-retry outcome if triggered).

**Tc block:** Tc₁/Tc₂ preliminary estimates; thermodynamic Tc with spinodal, transition order, Δ_jump, hysteresis, uplift percentage from SC-JT; `2Δ₀/kTc` with coupling regime label (BCS-like / strong / very-strong / exotic).

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
| No Pauli exclusion between cluster sites | Slight overestimate of AFM correlations; controlled by _ALPHA_HF blend |
| No charge-transfer fluctuations ⟨n_A n_B⟩ | Charge fluctuations negligible when U_mf ≫ t |
| Static phonon (Q is a mean field) | Zero-point quantum lattice fluctuations neglected |
| 4×4 BdG truncation | Valid when Δ_CF ≫ kT and Γ₇split/Δ_CF ≪ 1; monitored via `(J_eff/Δ_CF)²` projection-quality penalty in scoring |
| No spatial fluctuations | Cannot describe pseudogap, stripes, or phase separation |
| RPA static (ω = 0) | Dynamical vertex corrections absent |
| K_eff updated every 5 SCF iterations | Back-action of Q on exchange rigidity approximate during SCF transient |
| χ_τ at post-convergence only | Self-consistent Q back-action on chi_tau neglected during SCF |
| `compute_G_instability` at Δ=0 | G-matrix evaluates normal-state only; SC-triggered JT confirmed via post-SCF Hessian λ_min < −kT |
| ∂λ_pair/∂Q at frozen Fermi surface | FS geometry frozen at middle Q; SC-state version would require Bogoliubov Lindhard sum |
| δχ_τ baseline subtraction approximate in D₂h | Normal-state B1g response at finite Δ_inplane estimated at Δ=0; small D₂h corrections to χ_τ_n neglected |
| Incommensurate AFM auto-retry single recursion | If the softened-seed retry also shows incommensurate tendency, it is not further recursed; result may still be commensurate-biased |
| z_qp = 1/r_J proxy | Cluster r_J is a local (q=0) proxy for the k-dependent quasiparticle weight; unreliable near the Mott boundary |

---

## References

- Ecsenyi, S. (2026). *Multipolar superconductivity and coherent orbital mixing* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang et al. (1988). *Supercond. Sci. Technol.* 1, 36; Bünemann, J., Weber, W. & Gebhard, F. (1998). *Phys. Rev. B* 57, 6896.
- ZSA classification: Zaanen, Sawatzky & Allen (1985). *Phys. Rev. Lett.* 55, 418.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.
- TuRBO / Bayesian optimisation: Eriksson, D. et al. (2019). *NeurIPS.*
- Richardson extrapolation: Richardson, L.F. (1911). *Phil. Trans. R. Soc. A* 210, 307.
- Moriya, T. (1985). *Spin Fluctuations in Itinerant Electron Magnetism.* Springer.

---

*For questions or contributions, open an issue or pull request.*
