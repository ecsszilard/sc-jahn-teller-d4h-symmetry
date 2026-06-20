# SC-Activated Pseudo Jahn–Teller Model

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
selection_ratio = clip(|Δ_s| + |Δ_d|, 0, Δ_CF) / Δ_CF · |F67s_mf|
```

where `F67s_mf` is the self-consistent Gorkov singlet amplitude computed from `_compute_F67_singlet` — the SC-induced Γ₆↔Γ₇ off-diagonal coherence — fed back as an anomalous Weiss field into `build_local_hamiltonian_for_bdg`. This quantity is exactly zero at Δ=0 by construction.

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
λ_JT_norm = χ_QQ / K_eff   (dimensionless; < 1 → stable, = 1 → onset, > 1 → spontaneous JT)
```

This is a rigorous statement: it is the phonon stiffness of the zone-centre optical mode renormalized by the electron–phonon coupling and the electronic susceptibility of the condensate, without approximations beyond mean field. The full SC-state susceptibility-based metric `lambda_JT_sc = g²·|χ_τ_sc|/K_eff` (section §15) is the predictive SC-triggered quantity and shares the same threshold semantics.

An important symmetry constraint follows immediately. The distortion Q has B₁g symmetry; the pairing amplitude squared |Δ|² is A₁g (totally symmetric, whether s-wave or d-wave). Their product Q·|Δ|² transforms as B₁g ⊗ A₁g = B₁g — which is **not** the totally symmetric representation A₁g. Therefore the coefficient of the Q·|Δ|² term in the free energy is strictly zero by symmetry: there is no linear coupling between the JT distortion and the SC condensate at the Landau level. The SC-triggered JT distortion is a **threshold phenomenon**: the condensate renormalizes the stiffness of the B₁g mode (tracked as `λ_JT_norm = χ_QQ/K_eff`) until, at `λ_JT_norm = 1` (equivalently `χ_QQ = K_eff`), the mode goes soft and a spontaneous distortion appears.

### The Mean-Field Back-Action Loop and Its Numerical Stabilization

The fundamental theorem of mean-field theory requires that the expectation value of every order parameter be fed back into the Hamiltonian:

```
H_MF ∝ J ⟨Ô⟩ · Ô
```

When the superconducting condensate — via Γ₆–Γ₇ orbital mixing — creates a macroscopic anomalous coherence `⟨τ_x⟩_anom`, this must be fed back through the B₁g exchange tensor J_B₁g into the Weiss field. Concretely: the condensate generates off-diagonal Γ₆↔Γ₇ orbital coherence in the BdG eigenstates; this coherence modifies the effective exchange field felt by the lattice; the lattice responds by shifting Q; and the shifted Q in turn modifies the electronic structure and the pairing vertex. This feedback loop is physically essential — without it, the model is inconsistent with its own Hamiltonian.

This loop is numerically sensitive and can oscillate or diverge if not treated carefully. Three stabilization mechanisms are in place:

- **Anderson mixing:** the four-dimensional order-parameter vector `[M, Q/λ_hop, |Δ_s|/t₀, |Δ_d|/t₀]` is accelerated jointly via Anderson(5), capturing the cross-coupling ∂M/∂Δ and ∂Δ/∂M in the effective Jacobian.
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
Bdiag_qp = einsum('kan,ab,kbn->kn', ec.conj(), B1g_16, ec).real
exp_k    = einsum('kn,kn->k', Bdiag_qp, f)
B1g_exp  = dot(k_weights, exp_k) / 4.0
```

where `k` indexes k-points, `a,b` index the 16 Nambu components (rows/columns of B1g_16), and `n` indexes BdG bands (columns of `ec`).

The hole-block sign (`−B1g_op^T`) is already encoded in `B1g_16`, so weighting by `f` alone (not `fbar`) correctly accounts for both particle and hole contributions.

The `compute_observables_vectorized` method now returns only `M_stag`. The anomalous Weiss-field back-action is handled exclusively through `F67s_mf = g_eff · _compute_F67_singlet(ev, ec)`, which is injected into `build_local_hamiltonian_for_bdg` each iteration when Δ≠0 and Q≠0.

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

- **`moment_ratio` (Γ₇/Γ₆ magnetic moment ratio):** `moment_ratio = |⟨Γ₇|S_z|Γ₇⟩| / |⟨Γ₆|S_z|Γ₆⟩|`, computed from the S_z matrix elements of the first Kramers partners of Γ₆ and Γ₇a. This is not a free parameter — it is fully determined by the SOC+CF eigenbasis. It enters `sz_op = [1, −1, moment_ratio, −moment_ratio]` and propagates to all magnetization and Weiss-field calculations via `sz_bdg16`. The corresponding **multipolar operator** `multi_op = diag([1, −1, moment_ratio, −moment_ratio])` is pre-built in `ModelParams.__post_init__` and stored as `self.multi_op`; it is shared by both the cluster (2-site ED) and the BdG solver without recomputation.

- **Orbital weights `_w6_xz`, `_w6_yz`, `_w6_xy`, `_w7_xz`, `_w7_yz`, `_w7_xy`:** the d_xz, d_yz, d_xy character of the Γ₆ and Γ₇a Kramers states, used in `_exchange_channels` to compute the Q-dependent exchange asymmetry `η_J(Q)`.

The B₁g phonon coupling operator is constructed as:
```
B1g_op = real(U4† · (Lx²−Ly²)_t2g · U4)    (4×4, real, hermitian)
```
and its 16×16 Nambu extension `B1g_16` is stored with the hole block carrying `−B1g_op^T`, consistent with BdG particle–hole symmetry. Since `B1g_op` is a real symmetric matrix, `−B1g_op^T = −B1g_op`. All JT coupling terms in the Hamiltonian use `H += g_JT · Q · B1g_op` rather than a hand-coded τ_x matrix.

The B₁g operator off-diagonal weight `b1g_weight` is also computed in `__post_init__` from `B1g_op`:

```
b1g_diag_norm = ||diag(B1g_op)||_F
b1g_off_norm  = ||B1g_op − diag(B1g_op)||_F
b1g_ratio     = b1g_off_norm / max(b1g_diag_norm, 1e-9)
b1g_weight    = b1g_ratio / (1 + b1g_ratio)    ∈ (0, 1]
```

where `b1g_off_norm` and `b1g_diag_norm` are the Frobenius norms of the off-diagonal and diagonal parts of `B1g_op = real(U₄†(Lx²−Ly²)U₄)` respectively. In D₄h, `b1g_diag_norm = 0` exactly so `b1g_weight = 1` (SC-triggered only); in D₂h `b1g_weight < 1` (partial normal-state mixing). This quantity scales the anomalous SC→JT feedback: `anom_scale = tanh²(|Δ|/t0) · b1g_weight`.

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

`K_lattice` is the **bare phonon spring constant** (primary input, eV/Å²). `∂²F_ex/∂Q²` is computed by `compute_JT_rigidity_from_exchange` via central finite-difference of `⟨O_α(Q)⟩`; negative when the SC condensate softens the JT mode. `K_lattice` is never mutated;

The SC-triggered JT coupling strength:
```
lambda_JT_sc = (g_JT² / K_eff) · chi_tau_sc
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
selection_ratio = clip(|Δ_s| + |Δ_d|, 0, Δ_CF) / Δ_CF · |F67s_mf|
```

where `F67s_mf` is the self-consistent Gorkov singlet amplitude — the SC-induced off-diagonal Γ₆↔Γ₇ coherence `F₆₇ = Σ_k (1−2f_n) Re[u*_{6↑} v_{7↓} − u*_{6↓} v_{7↑}]` — computed from `_compute_F67_singlet` and fed back into the anomalous Weiss field in `build_local_hamiltonian_for_bdg`. It is exactly zero when Δ=0 or Q=0, enforcing the D₄h selection rule.

- `selection_ratio ≈ 0`: pure AFM state — B1g_op strictly off-diagonal → ⟨B1g_op⟩ = 0, JT forbidden (exact in D₄h).
- `selection_ratio > 0.05`: SC-mixed state — condensate has opened the B₁g channel → JT active.

The SC–JT chain: Δ≠0 → F67s_mf≠0 → H_B1g≠0 → Q≠0. In D₄h B1g_op is a singlet (spin-flip off-diagonal), so ⟨B1g_op⟩=0 in any normal state; the condensate is required to unlock it. The selection ratio feeds into the adaptive mixing rate during SCF: when `selection_ratio > 0.05` and `|Q| > 1e-4` (JT-active), α is boosted to accelerate convergence; when below threshold, α is damped to suppress oscillations near Q=0.

### 8. Observables: BdG Thermal Averages

All observables are computed in a single batched LAPACK call via `VectorizedBdG`. Two distinct orbital quantities are returned per SCF iteration:

| Observable | Formula | Role |
|---|---|---|
| **B1g_exp** (full) | `Tr[B1g_16 · ρ̂_k]` via einsum over B1g_16; /4 for Nambu+sublattice doubling | **Lattice update:** Hellmann–Feynman force `Q_eq = −(g_JT/K_eff)·B1g_exp` |
| Magnetization | `⟨S_z⟩` via `sz_op = [+1,−1,+η,−η]` where η is derived from SOC+CF eigenvectors | AFM order parameter |
| F67s_mf | `Σ_k (1−2f_n) Re[u*_{6↑} v_{7↓} − u*_{6↓} v_{7↑}]` from `_compute_F67_singlet`; fed back via `build_local_hamiltonian_for_bdg` | Anomalous Weiss-field back-action when Δ≠0, Q≠0; enters `selection_ratio` |
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

`⟨B1g_op⟩` is computed via the full 16-component Nambu eigenstates using `B1g_16`, so that anomalous u·v amplitudes — which carry the SC-triggered orbital coherence — are fully included. **Four step sizes h, h/2, h/4, h/8** provide Richardson-extrapolated central differences (O(h²)→O(h⁴) at the primary level). The extrapolation checks for nonlinearity between consecutive step pairs: if the response changes by more than 20% between the first pair (h, h/2), the algorithm descends to the finer pair (h/4, h/8). If the finer pair is linear, the h/8 value is returned at half weight (`chi_tau_weight = 0.5`); if both pairs are nonlinear the derivative is set to zero (`chi_tau_weight = 0.0`). Both cases log a `[CHI-TAU]` warning. The self-consistency flag (`richardson_ok`) requires both Richardson convergence (< 3% disagreement between extrapolation levels) and linear response.

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

The pairing vertex is computed via a 2×2 coupled spin–JT RPA in `[spin, JT-phonon]` channel space. `get_susceptibilities_normal` returns a plain tuple `(chi_DD_s_moriya, chi_DQ_s, chi_QD_s, chi_QQ_tilde, Gamma_M_eff)`; callers destructure it directly. The vertex is then evaluated by calling `self._rpa_vertex(J, V, ...)` explicitly for each required combination (full vertex with J and V_JT; spin-only with V_JT=0; JT-only with J=0) rather than reading named keys from a dict. The bare interaction matrix is **diagonal**: `Û = diag(J_eff, V_JT)` — there is no bare S–Q cross-vertex; the spin–JT feedback enters exclusively through the off-diagonal susceptibilities χ_DQ_s/χ_QD_s, which are opened by SOC and the SC condensate:

```
V(q) = J_eff² χ_DD_s^RPA(q) + V_JT² χ_QQ^RPA(q) + J_eff V_JT [χ_DQ_s^RPA(q) + χ_QD_s^RPA(q)]
```

The bare susceptibilities χ₀(q) come from the Δ=0 BdG Hamiltonian via the Lindhard formula, implemented in the module-level `_lindhard_bubble()` function using `opt_einsum`. A ZRS kinematic form-factor `β²(k)·β²(k+q)` modulates the kernel at each k-point, where `β²(k) = max(1−(cos kx + cos ky)/2, 0)` is the A₁g spectral weight of the ZRS quasiparticle at k. The Lindhard sum for normal-state χ₀(q) uses `_NORMAL_SECTOR_PAIRS` (8 pairs: AA/BB particle and hole blocks, plus AB/BA cross terms). The SC-state χ_SQ is computed separately via `_compute_chi_SQ_sc` using the unified Nambu Lehmann sum `_compute_nambu_susceptibility` with `Sz_nambu` and `Tau_nambu` vertex matrices; this unified approach automatically includes all Gorkov sectors and eliminates the risk of manual sector double-counting. The static Lindhard function is real by time-reversal symmetry — the imaginary part vanishes exactly at ω=0, so taking `chi0_tensor = chi0.real` after enforcing Hermiticity discards only numerical roundoff, not physical information. Projections:

```
χ_DD_s = Tr[Sz · χ₀[Γ₆,Γ₆] · Sz]      # spin–spin (dipole–dipole)
χ_DQ_s = Tr[Sz · χ₀[Γ₆,Γ₇]]            # spin–orbital cross (dipole–quadrupole)
χ_QQ   = −∂²Ω/∂Q²  (numerical, SC state)  # orbital JT stiffness [eV/Å²]
```

**PSD projection of [[χ_SS, χ_SQ], [χ_SQ, χ_QQ]]:** near the QCP the Cauchy–Schwarz condition χ_SQ² ≤ χ_SS · χ_QQ can be violated by numerical noise in the Lindhard sum. After computing (χ_SS, χ_SQ, χ_QQ), the 2×2 matrix is projected to the nearest positive-semidefinite matrix via eigendecomposition and clamping of negative eigenvalues to zero (Higham 1988). This is applied both in `get_susceptibilities_normal` (per q-point in the RPA loop) and in `compute_G_instability` (at q=0 for the G3 matrix). A `PSD-CHI` log line is emitted when the correction exceeds 1%.

The cross-terms χ_DQ_s and χ_QD_s are **zero in the normal state at Q=0** (Γ₆–Γ₇ mixing forbidden) and become nonzero when Q > 0 opens the B₁g channel via B1g_op. A Padé resummation regularizes χ_DQ_s:

```
χ_SQ_v = χ_SQ / (1 + |χ_SQ| / w),    w = _CHI_SQ_S_PADE_W = 0.05
```

This is linear at |χ_DQ_s| ≪ w — continuously suppressing noise — and saturates asymptotically to ±w at large |χ_DQ_s|, with a smooth gradient near the QCP. χ_QQ is regularized via soft Dyson resummation:

```
χ_QQ_eff = χ_QQ / (1 + χ_QQ · V_JT / K_bare)
```

which is continuous and differentiable, saturates at `K_bare/V_JT` as χ_QQ→∞. The RPA determinant:

```
det = (1 − J_eff·χ_DD_s_moriya)(1 − V_JT·χ_QQ_eff/K) − J_eff·V_JT·χ_DQ_s_v·χ_QD_s_v
```

Moriya damping is split across two call sites: `_Gamma_M_bare` is computed once via `_make_vertex_params(target_doping, ...)` and passed to `get_susceptibilities_normal` as the susceptibility damping. Inside `compute_gap_eq_vectorized`, the per-q RPA loop computes `_Gamma_M_act = moriya_gamma(target_doping, ...)`. The earlier draft used `actual_doping` here but that was reverted: the susceptibility bubble is a normal-state quantity evaluated at the lattice filling, which is `target_doping`; using `actual_doping` introduced an inconsistency with the susceptibility call.

Spin fluctuations are regularised by Moriya damping (doping-dependent) rather than a hard cutoff:

```
Γ_M = max(α_M, α_floor) · t_eff² / J_eff

α_M = _MORIYA_C · f(δ) · sat(t/J)
f(δ) = δ / (δ + _MORIYA_DSAT)          ∈ (0, 1)
sat(t/J) = (t/J) / (_MORIYA_TJ_SAT + t/J)  ∈ (0, 1)
```

`_MORIYA_C = 0.21`, `_MORIYA_DSAT = 0.30`, `_MORIYA_TJ_SAT = 1.0`. The product `f(δ)·sat(t/J)` prevents the positive-feedback loop `J_eff↓→t/J↑→Γ_M↑` while ensuring `Γ_M→0` at half-filling (long-range AFM) and growing with doping as metallic screening broadens the QCP. The floor `_ALPHA_MORIYA = 0.02` prevents numerical runaway at very low doping.

**RPA determinant treatment past the QCP:** when `det > 0` a floor `_RPA_DET_REG = 1e-9` guards against exact-zero numerical accidents only. When `det < 0` (past the QCP) the determinant is left intact — applying a soft cap to the vertex in this regime would trap the SCF in the unstable phase. The per-call vertex cap `V_cap = _RPA_V_CAP_ALPHA · max(8·max(|tx|,|ty|), J_eff)` (computed in `_make_vertex_params`) prevents numerical overflow without altering the sign or divergence character of V(q).

**Moriya damping:** `_Gamma_M` is computed once per vertex cache cycle via `_make_vertex_params(target_doping, ...)` and passed to `get_susceptibilities_normal` as the susceptibility damping. The `j_renorm`-based Moriya correction has been removed; the Gutzwiller renormalization of the BdG bands already encodes the relevant quasiparticle weight suppression. The `χ₀` bubble is left untouched: Ward identities require that the quasiparticle weight from the bubble (Z²) and the vertex correction (1/Z) cancel to Z, which is already encoded in the Gutzwiller-renormalized BdG bands feeding `χ₀`.

**χ_SQ(q) full BZ scan:** `estimate_chi_SQ_q_full()` evaluates the spin–quadrupole cross-susceptibility χ_SQ(q) over a **24×24 q-grid**. For the normal state (Δ=0) it uses `_lindhard_bubble` with `_NORMAL_SECTOR_PAIRS`. For the SC state (Δ≠0) it calls `_compute_chi_SQ_sc`, which uses the unified Nambu Lehmann sum `_compute_nambu_susceptibility` with the pre-built 16×16 vertex matrices `Sz_nambu` and `Tau_nambu`. This approach automatically captures all Gorkov sectors (GG and FF Nambu blocks) without manual sector splitting or double-counting — the Nambu vertex structure `M = [[O, 0],[0, −O^T]]` encodes the particle–hole sign for both normal and anomalous propagators. The SC pass makes χ_SQ_sc ≠ 0 even in D₄h, where the normal-state χ_SQ ≡ 0 exactly.

In D₄h (Δ_ip = 0) the normal-state χ_SQ ≡ 0 exactly by the B₂g selection rule Γ(S_z)⊗Γ(τ_x) = B₂g ⊄ A₁g; any finite numerical value indicates grid noise and is checked by `symmetry_ok : |χ_SQ_peak_n| < 1e-3`. In D₂h (Δ_ip ≠ 0) the peak near q=(π,π) controls the RPA cross-term enhancement and the SC-JT window width.

The function accepts `(target_doping, M, Q, mu, g_t, g_J, Delta_s=0, Delta_d=0, n_q=24)` and returns a rich dict:

| Key | Description |
|---|---|
| `chi_SQ_n`, `chi_SQ_sc` | χ_SQ(q) arrays, normal and SC state |
| `chi_SS_n`, `chi_SS_sc` | χ_SS(q) arrays for PSD denominator |
| `phi_d_q` | `\|cos q_x − cos q_y\|` B₁g d-wave form factor |
| `phi_d_overlap_n/sc` | Normalised ∫\|χ_SQ\|·φ_d / (‖χ_SQ‖·‖φ_d‖)·n_pts — overlap with d-wave symmetry |
| `cs_ratio_n/sc` | Cauchy–Schwarz ratio χ_SQ²/(χ_SS·χ_QQ₀) per q-point (conservative lower-bound PSD check) |
| `psd_violations_n/sc` | Count of q-points where cs_ratio > 1+ε |
| `q_peak_n/sc`, `chi_SQ_peak_n/sc` | Peak location and value |
| `peak_region_n/sc` | Region classifier: `afm` / `antinodal` / `gamma` / `nodal` |
| `antinodal_frac_n/sc` | Fraction of \|χ_SQ\| weight at antinodal (π,0)/(0,π) points |
| `phi_d_overlap_n/sc` | d-wave form-factor overlap |
| `local_vertex_ok` | `antinodal_frac_n > 0.5` — confirms that the local q=0 vertex approximation is safe (χ_SQ concentrates where the d-wave gap does) |
| `symmetry_ok` | `\|chi_SQ_peak_n\| < 1e-3` — D₄h selection rule check |
| `sc_computed` | Whether the SC pass was executed |

This is a post-convergence diagnostic called from `compute_G_instability` when `|Δ_inplane| > 0`. The call passes the last converged SC state from the solver's `_last_Delta_s`, `_last_Delta_d`, and `_last_Q` attributes, so both passes reflect the actual converged solution. The `local_vertex_ok` flag is logged with an explicit warning if `antinodal_frac_n ≤ 0.5`: if the χ_SQ peak falls in the `afm` region rather than `antinodal`, the local q=0 vertex approximation may overestimate the RPA cross-term and the SC-JT window could be narrower than predicted. The χ_SS and χ_SQ peak positions are compared to check coincidence (within `π/12` tolerance); a mismatch is flagged.

**q-resolved vertex diagnostics:** after each vertex cache rebuild, four momentum-resolved quantities are stored and logged when `V_d < 0`:

| Key | Description |
|---|---|
| `V_afm_mean` | Mean V(q) at \|q\| > 0.7π; > 0 expected for spin-fluctuation-driven d-wave pairing |
| `V_fwd_mean` | Mean V(q) at \|q\| < 0.35π; typically < 0 (forward scattering repulsion, cancelled by φ_d form factor) |
| `V_neg_frac` | Fraction of q-points with V < 0; > 0.9 indicates globally repulsive vertex (unphysical) |
| `V_dd_fs` | Fermi-surface-projected d-wave vertex `⟨φ_d\|V\|φ_d⟩`; negative → d-wave instability not supported |

These flags are appended to the log line only when `V_d < 0` to keep normal output compact. The `V_dd_fs` sign is the decisive criterion: `V_dd_fs < 0` with `V_afm_mean > 0` signals form-factor sign cancellation (the AFM peak is attractive but the d-wave projection is dominated by repulsive contributions away from (π,π)).

**QCP sign-flip EMA guard:** when `|det_afm| < _DET_SIGN_FLIP_SCALE = 0.05` and `V_d_scalar` would flip sign relative to the cached value, the blend weight is reduced continuously via sigmoid rather than a hard binary switch. This suppresses numerical sign instabilities near the QCP without suppressing genuine physical sign changes.

**Separate QCP tracking:** the vertex cache separately monitors the FM instability at q=0 (`det_q0`) and the AFM instability at q=(π,π) (`det_afm`). The SCF adaptive mixing and convergence tolerance respond to `det_afm`; the FM check guards against accidental ferromagnetic divergence. Both determinants are logged at convergence.

**∂λ_pair/∂Q > 0 is the key numerical criterion for the SC-triggered JT hypothesis.** A positive value confirms that an infinitesimal B₁g distortion increases the pairing strength through the spin-fluctuation channel. It is evaluated at Δ=0 with the converged SCF chemical potential.

**Susceptibility consistency:** χ₀ (normal state, Δ=0) is used for χ_DD_s, χ_DQ_s, χ_QD_s in the pairing vertex — feeding Δ≠0 susceptibilities back into the interaction would double-count the gap. `chi_QQ` (SC state, Δ≠0) is used exclusively for lattice stability diagnostics (G-matrix). The vertex is always built from the normal state (Δ=0 BdG). The `chi_QQ_bare` argument is the bare JT orbital susceptibility: χ_QQ = −∂²Ω/∂Q² (eV/Å²) — not multiplied by g_JT².

### 13. Linearized Gap Equation and λ_JT_kernel

The pairing kernel on the Fermi surface:
```
Γ_ij = g_Δ · √(dl_i/vF_i) · V(k_i − k_j) · √(dl_j/vF_j)
```

The integration weight is the proper FS measure: `inv_vF_i = dl_i / ((2π)² · |vF_i|)` where `dl_i` comes from `_fs_arc_lengths` and `|vF_i|` is floored at `_VF_FLOOR_TIGHT`. This replaces the earlier `1/|vF|` approximation (which omitted the arc-length factor), giving a correctly normalised `∫…dS/|vF|` kernel. `g_t` is now an explicit parameter; when `actual_doping=None`, vertex params use `_eff_doping = actual_doping if actual_doping is not None else target_doping`. `λ_max` = largest eigenvalue of Γ, with gap eigenvector φ_max. C₄ symmetry averaging is applied only when Q ≈ 0 (unbroken D₄h). The **JT-channel Rayleigh projection**:
```
λ_JT_kernel = φ_max^T · Γ_JT · φ_max
```
measures how much of λ_max comes specifically from the JT channel (V_JT component of V(q)). Because `eigh` returns unit-norm eigenvectors, the explicit re-normalisation of `gap_vector` before the Rayleigh quotient is redundant and has been removed. This is distinct from `lambda_JT_sc = (g²/K)·chi_tau_sc`, which is a scalar q=0 estimate.

The gap vector lives in the `√(dl/vF)`-weighted Hilbert space. Projections onto s-wave and d-wave channels use `psi_s = φ_s · √(inv_vF)` and `psi_d = φ_d · √(inv_vF)` (both normalised), ensuring the L2 inner product is correct in the weighted basis. **Signed per-channel Rayleigh quotients** are computed directly:

```
lambda_bare_s = ⟨ψ_s | Γ_bare | ψ_s⟩ / ⟨ψ_s | ψ_s⟩
lambda_bare_d = ⟨ψ_d | Γ_bare | ψ_d⟩ / ⟨ψ_d | ψ_d⟩
```

Channel-specific Gutzwiller factors are applied separately. The optimal s/d hybridisation for the SCF update is taken from the 2×2 pairing kernel (`K_pair_v_s`, `K_pair_v_d`) built inside `compute_gap_eq_vectorized` — see §21 for construction details.

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

**λ_JT quantity disambiguation** — three distinct dimensionless metrics share the λ_JT name:

| Symbol | Formula | State | Location | Threshold |
|---|---|---|---|---|
| `lambda_JT_norm` | χ_QQ / K_eff | normal (Δ=0) | G-MATRIX log | < 1 → stable; ≥ 1 → spontaneous JT |
| `lam_JT` | g²·χ_QQ / K_bare | normal (Δ=0) | DE scout S3/S4 | > 0.05 for viable window |
| `lambda_JT_sc` | g²·\|χ_τ_sc\|·w_sc / K_eff | SC (Δ≠0) | check_sc_jt_window | > 0.05 → SC-triggered JT active |

`lambda_JT_norm` and `lam_JT` use the thermodynamic `χ_QQ = −∂²Ω/∂Q²` (normal state). `lambda_JT_sc` uses `χ_τ_sc = ∂⟨B₁g⟩/∂(g_JT·Q)` from the SC BdG state, which picks up the anomalous u·v coherence.

`check_sc_jt_window` verifies that `K_lattice` lies in the cooperative SC–JT window:

```
K_spont = g_JT² / Δ_CF                           (spontaneous JT threshold; K_lattice must exceed this)
K_SC    = g_JT² · max(−chi_tau_net, 0) / _LAMBDA_JT_VIABLE  (SC-triggered threshold; K_lattice must be below this)
```

`_LAMBDA_JT_VIABLE = 0.05` is a fixed physical viability criterion: K_SC is the stiffness above which `λ_JT_sc = g²·max(−chi_tau_net,0)/K` drops below 5%. The window condition reduces to `|chi_tau_net| · Δ_CF > _LAMBDA_JT_VIABLE` (independent of `g_JT`).

Two λ_JT metrics are tracked:
- `lambda_JT_sc = g²·max(−chi_tau_net, 0)/K_eff` — SC-state metric; the primary viability criterion. `chi_tau_net = chi_tau_sc − chi_tau_n` isolates the SC-triggered softening from any normal-state baseline; only its negative part (lattice softening) counts.
- `lambda_JT_kernel` — Rayleigh projection of the dominant gap eigenvector onto the JT channel of the pairing kernel (see §13); confirms the same mechanism from the pairing side.

`structural_ok` requires both `g²·χ₀ < K_eff` (G-matrix positivity) and `λ_min > 0` (no normal-state spontaneous instability). If `λ_min ≤ 0`, `normal_unstable = True` is flagged and `viable = False` regardless of the window boundaries.

`K_opt = √(K_spont · K_SC)` is the geometric midpoint.

### 16. Richardson-Extrapolated χ_τ

`_compute_chi_tau` computes the B₁g susceptibility `χ_τ = ∂⟨B1g⟩/∂(g_JT·Q)` via 4-level Richardson extrapolation. The finite-difference scheme uses step sizes `h, h/2, h/4, h/8`. At each refinement level the nonlinearity is estimated from the convergence ratio. The function returns:

- `chi_tau_sc`, `chi_tau_n`: SC-state and normal-state susceptibilities
- `chi_tau_net = chi_tau_sc − chi_tau_n`: net SC-induced susceptibility change (signed; used directly for `lambda_JT_sc`)
- `richardson_ok`: True when the Richardson series converged
- `chi_tau_weight` (`w_sc`): reliability weight — `1.0` = full (converged), `0.5` = halved (finer-scale estimate used due to nonlinearity at the primary scale, consistent with being near the first-order SC-JT boundary), `0.0` = suppressed (both scales nonlinear, derivative unresolvable). When `w_sc < 1`, a diagnostic log line reports the weight and the physical interpretation.

`chi_tau_sc` is multiplied by `w_sc` before use so that uncertain estimates contribute proportionally to the SC-JT feedback strength rather than being either fully used or discarded.

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

Computed analytically from a single BdG diagonalization via second-order perturbation theory. The Newton step for M uses the analytic curvature with Levenberg–Marquardt regularization floor `_MU_LM = 3.1`; the LM floor is adaptively reduced as `|Δ|` grows to allow M to relax as SC develops. The Newton vs. BdG fixpoint blend is `_ALPHA_HF = 0.35`.

### 19. Two-Site Cluster: Quantum Multipolar Fluctuations and Two-Channel Vertex Renormalisation

Beyond BdG mean field, a 2-site (A–B) cluster is exactly diagonalized at each iteration. The local single-particle Hamiltonians on each site are built by `build_local_hamiltonian_for_bdg`, which includes `−μ`, Δ_CF, the AFM Weiss field (J_A1g·sign_M·M), the JT coupling (g_JT·Q·B1g_op), and the anomalous Weiss field (J_B1g·F67s_mf, only when Q≠0). The cluster Hamiltonian is:

```
H_cluster = H_A ⊗ I + I ⊗ H_B                      [local SP + Weiss + JT + F67s]
          + J_bond_M_bare · (multi_op ⊗ multi_op)    [A1g magnetic exchange]
          + J_bond_Q_bare · (B1g_op ⊗ B1g_op)         [B1g orbital exchange, Q≠0 only]
```

`J_bond_M_bare` is the A1g Gutzwiller-renormalised superexchange. `J_bond_Q_bare = J_CT · sinh(2Q/λ) · η_J` is the bare B1g orbital exchange (single bond); activated when `|Q| > 1e-8`. Exact Hermiticity is enforced after assembly. The A1g magnetic channel is **not** extracted by regression: in the collinear AFM ground state `corr_M` vanishes identically under Wick factorisation, making the corresponding slope ill-conditioned. The renormalised magnetic exchange is taken from the analytic Gutzwiller result.

**Two-channel JT-sector WMLR:** after diagonalizing H_cluster, two vertex renormalizations are extracted from the JT sector — the only sector with well-conditioned connected correlators — via Boltzmann-weighted multivariate linear regression:

```
evals_int ≈ const + J_Q · corr_Q + J_MQ · corr_MQ
```

where:
- `corr_Q  = ⟨B_A B_B⟩ − ⟨B_A⟩⟨B_B⟩`  (B1g connected correlator)
- `corr_MQ = ½(⟨O_A B_B + B_A O_B⟩ − ⟨O_A⟩⟨B_B⟩ − ⟨B_A⟩⟨O_B⟩)`  (spin–JT cross-correlator)

The mean-field background is subtracted from `evals_int` state-by-state (`J_bond_M_bare · oA·oB` always; `J_bond_Q_bare · bA·bB` when Q≠0), making the regression target the pure quantum fluctuation content. States with negligible Boltzmann weight (< 10⁻⁴ of max) are excluded; the solver requires at least 3 valid points, otherwise falls back to `r_Q = r_MQ = 0`. The 2×2 design matrix is solved by `np.linalg.solve`; `lstsq` is the fallback when the matrix is near-singular. `n_eff < _CLUSTER_N_EFF_FLOOR = 2.0` also forces `r_Q = r_MQ = 0`.

**Independent t-test significance shrinkage (step 12):** each slope is tested independently against H₀: `J = 0` using a two-sided t-statistic with `df = n_eff − 2` degrees of freedom and significance level `_REGR_T_ALPHA = 0.05`. The standard error for each slope is computed from the diagonal of the inverted design matrix scaled by residual variance. A continuous confidence factor `conf = clip(|t| / t_crit, 0, 1)` smoothly shrinks insignificant slopes toward zero, logging the shrinkage when `conf < 0.9`.

The two extracted renormalization factors (both normalised by `J_bond_M_bare`):
- **`q_renorm = r_Q = J_Q / J_bond_M_bare`** — B1g orbital fluctuation vertex; scales `J_B1g` in `_exchange_channels` and propagates through `compute_JT_rigidity_from_exchange`, `_build_H_stack`, `_find_mu_for_density`, `compute_hessian`, and `compute_dF_dM_and_d2F`. Tracked via EMA (weight `_EMA_NEW_QRW = 0.38`).
- **`r_MQ = J_MQ / J_bond_M_bare`** — spin–JT cross-coupling vertex; scales `chi_SQ_v` and `chi_QS_v` in `get_susceptibilities_normal` via explicit multiplication `r_MQ · float(chi_SQ / QS)`. Tracked via EMA (weight `_EMA_NEW_WEIGHT = 0.28`). Passed to `solve_linearized_gap_equation`, `compute_G_instability`, and `_dlambda_dQ_core`.

Both clipped to `[−2.0, +2.0]`. The regression is skipped (`_regression_solved = False`) when: `|Q| < 1e-4`, `n_eff ≤ 5`, variance too small, or design matrix condition number > 10¹⁰.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

```
∂n/∂μ = Σ_{k,n} w_k · f(E_n)(1−f(E_n)) / kT · (|u_A|² + |u_B|² + |v_A|² + |v_B|²)
```

Newton's method with the analytic derivative from the same BdG eigensystem; Brent's method as fallback. The `(ev, ec)` from the μ-search is reused directly for subsequent observable computation, avoiding a redundant `eigh` call.

### 21. Gap Equations, Complex Phase Preservation, and Inline 2×2 Pairing Kernel

`compute_gap_eq_vectorized` evaluates gap equations over the full Brillouin zone:

```
F_AA_BZ = complex(Σ_k w_k · Pair_s(k)) / 4    (on-site s-channel)
F_AB_BZ = complex(Σ_k w_k · Pair_d(k)) / 4    (inter-site d-channel; d-wave projection on vertex side)
```

The `/4` corrects for the 16-dimensional BdG space doubling.

**Why complex-valued F_AA_BZ / F_AB_BZ (not `abs`).** The converged SC state generically carries a nontrivial relative phase between Δ_s and Δ_d. In the previous version both anomalous averages were collapsed to real by taking `abs(...)` before computing the new gap amplitudes. This is incorrect: if the stationary point of the free energy sits at a complex Δ_s/Δ_d pair, stripping the phase shifts the fixed-point equations off the true stationary point and biases the SCF toward real solutions even when the physical minimum is complex. Keeping F_AA_BZ and F_AB_BZ as complex numbers means the gap update `Δ_s_new = g_Delta_s · V_s · F_AA_BZ` inherits the correct phase automatically, so the SCF converges to the true (potentially complex) fixed point without introducing a phase artefact. The phase is also preserved in the Hessian: `compute_hessian` now extracts `phase_s = Δ_s / |Δ_s|` and `phase_d = Δ_d / |Δ_d|` from the converged state and fixes them throughout the finite-difference probes of F(m, q, δ), ensuring that each probe point evaluates the free energy on the correct manifold of the order parameter rather than on a real-valued slice.

**Inline 2×2 pairing kernel (K_pair).** Its logic is now executed directly inside `compute_gap_eq_vectorized` at vertex-cache rebuild time, using the already-available FS grids and RPA vertex loop results. The kernel is built in the `(s, d)` channel basis:

```
K_pair = [[K11, K12],
          [K12, K22]]

K11 = g_Delta_s · ⟨φ_s | V_JT | φ_s⟩_s-FS / ‖φ_s‖²   (on s-FS with uniform φ_s ≡ 1)
K22 = g_Delta_d · ⟨φ_d | V_full| φ_d⟩_d-FS / ‖φ_d‖²   (on d-FS with φ_d = cos kx − cos ky)
K12 = √(g_Delta_s · g_Delta_d) · ⟨φ_s | V_JT | φ_d⟩_d-FS / √(‖φ_s‖²‖φ_d‖²)
```

The s-channel uses a separate Fermi-surface grid (`fs_pts_s`, `vF_arr_s`) with uniform integration weights, while the d-channel uses the standard anisotropic FS grid (`fs_pts`, `vF_arr`). K11 uses `V_JT` (JT-only, J=0) to isolate the phonon-mediated s-wave attraction; K22 uses the full RPA vertex `V_full`; K12 uses `V_JT` on the d-FS to cross-couple the channels via the JT phonon. The integration measure in all cases is `w_ij = √(dl_i/vF_i) · √(dl_j/vF_j)` (proper FS arc-length weight).

The dominant eigenvector `(v_s, v_d)` of K_pair gives the optimal s/d hybridisation; `K_pair_lambda` is the largest eigenvalue. These are cached as `K_pair_v_s`, `K_pair_v_d`, `K_pair_lambda` in the vertex cache and propagated to `solve_linearized_gap_equation` (§13), which reads them to set the initial gap direction without recomputing the FS integrals.

**V_d Sign-Flip Guard.** The d-wave pairing vertex `V_d_scalar` is susceptible to sign flips between SCF iterations when the χ_SQ cross-term phase changes sign due to small changes in M or Q; such flips are always numerical (SCF-scale changes of ~10⁻³ cannot physically reverse the pairing vertex). A sigmoid EMA guard is applied whenever a sign flip is detected:

```
det_x     = |det_afm| / _DET_SIGN_FLIP_SCALE
w_factor  = _EMA_SIGN_FLIP_W_MIN + (1 − _EMA_SIGN_FLIP_W_MIN) / (1 + exp(−k·(det_x − 0.5)))
ema_w     = _EMA_NEW_WEIGHT · w_factor
V_d_new   = (1 − ema_w) · V_d_prev + ema_w · V_d_new
```

Near the QCP (`det_afm → 0`, `w_factor → _EMA_SIGN_FLIP_W_MIN = 0.20`) the blend is light to preserve genuine sign ambiguity; far from QCP (`det_x ≫ 1`, `w_factor → 1`) the blend is full. `V_d_prev` is stored in `_SolveState.V_d_ema` — persistent across vertex cache invalidations (which can occur on Q sign flips or FS topology changes).

Both `V_s_scalar` and `V_d_scalar` are clamped to `[−V_cap, +V_cap]` before being stored in the vertex cache. An early-exit guard returns zeros when the FS has fewer than 4 points, preventing downstream division-by-zero in arc-length weights.

### 22. Temperature-Dependent Tc Estimates

Three independent Tc estimates are computed, each measuring a different aspect of the transition:

**Tc₁ — McMillan strong-coupling formula:**
```
Tc₁ = (ω_SF / _MAD_DENOM) · exp(−_MAD_NUM · (1+λ) / λ)
    = (J_eff / 1.20) · exp(−1.04 · (1+λ_max) / λ_max)
```
The characteristic boson frequency is `ω_SF = J_eff` (paramagnon bandwidth), appropriate for spin-fluctuation-mediated pairing. Uses `λ_max` from the linearized gap equation at the reference doping.

**Tc₂ — λ(T) = 1 crossing from normal-state scan:**
`compute_lambda_vs_T` runs the linearized gap equation at each temperature on a **Δ=0 normal-state SCF background** (M and Q relax self-consistently without a condensate biasing the bands). The initial M seed uses `ModelParams.estimate_M0(doping, lambda_lin_max)` — a warm-start that blends a Stoner estimate (M_stoner ∝ 0.18·(S/max(S,1))·g_J/4) with a doping-weighted prior (M_prior) and a supercritical Curie–Weiss correction when λ_lin > 1 — rather than the converged SC value, avoiding the artefact where the T=0 AFM Weiss field artificially splits bands and prevents λ from reaching 1. Tc₂ is the temperature where the normal-state λ_max(T) = 1. Non-monotone λ(T) is detected and all crossing temperatures are logged.

**Tc₃ — Thermodynamic Tc from free-energy crossing:**
`compute_Tc_thermodynamic` performs a **single** upward temperature scan via `_find_crossing_and_spinodal`, which simultaneously tracks: (a) the free-energy crossing `F_SC = F_NM` (thermodynamic Tc, interpolated between scan points), and (b) the spinodal where the SC basin collapses (`|Δ| < Delta_tol`). The two separate helpers (`_find_crossing` and `_find_spinodal_heating`) have been merged into one pass, halving the number of SCF evaluations per temperature point. For near-second-order transitions, the spinodal temperature is further refined by a Ginzburg-Landau extrapolation: points with `|Δ| > _GL_DELTA_MIN = 2 meV` are fit to `Δ²(T) = a(T − Tc)`, and the zero crossing gives a more precise Tc. The GL refinement is applied only when `D_spinodal / Δ₀ < 0.15` (continuous collapse, consistent with near-second-order). A separate `_find_spinodal_cooling` pass (cold-start from normal state) is still run to detect hysteresis when a crossing was found. The method returns transition order, Δ_jump, hysteresis, and SC-JT uplift percentage relative to a non-JT reference. The `2Δ₀/kTc` gap ratio is computed from Tc₃ (most physically reliable).

Post-SCF:
- `compute_Tc_by_gap_suppression`: bisects in T to find the spinodal only (second-order-like boundary). Retained for cross-check.
- `compute_gap_ratio`: reports `2Δ₀ / k_B Tc₃`; values above 3.52 indicate SC-JT strong-coupling enhancement.

---

## Model Architecture

```
ModelParams  (dataclass, __post_init__)
    ├── Primary: t_pd, u, lambda_soc, Delta_tetra, g_JT, K_lattice,
    │            lambda_hop, Delta_inplane, Delta_CT, Z, kT, tol
    ├── Derived: Delta_CF, g7split, t0, U, U_mf, J_CT, doping_0, _U4, U_gamma,
    │            moment_ratio (from Sz matrix elements), multi_op (pre-built multipolar operator),
    │            b1g_diag_norm, b1g_off_norm, b1g_ratio, b1g_weight (from B1g_op projection),
    │            _w6_xz/_w6_yz/_w6_xy, _w7_xz/_w7_yz/_w7_xy (orbital weights for η_J(Q))
    ├── Grid objects: k_points, k_points_even, k_weights, k_weights_even,
    │            shift_table (_NK×_NK×N_k_even int32), N_k, N_k_even
    ├── ZRS arrays: chi0_kernel_weight (β²(k) = max(1−(cos kx+cos ky)/2, 0), A₁g kinematic form-factor)
    │            zrs_spectral_weight (Z(k) = α²β²/(1+α²β²), Padé quasiparticle weight; diagnostics only)
    └── MBZ arrays: mbz_mask (bool, N_k_even; canonical MBZ representative), k_weights_mbz (2/N inside MBZ, 0 outside)

_SolveState  (dataclass, mutable SCF-run-local state)
    ├── V_d_ema: Optional[float]          # persistent V_d sign-flip EMA; passed explicitly into compute_gap_eq_vectorized; reset each solve_self_consistent call; never stored on self
    └── _ema_kick_pending: bool = False   # True for one iteration after a kick: doubles blend weight so EMA adapts faster

Module-level SCF constants (not in ModelParams):
    _NK = 64          # k-points per direction (must be even for commensurate q_AFM = (π,π)); parity asserted at solver init
    _MAX_ITER = 700   # maximum SCF iterations
    _MIXING = 0.07    # base Anderson mixing weight
    _MU_LM = 3.1      # LM regularization floor for M Newton step
    _ALPHA_HF = 0.35  # Newton vs BdG fixpoint blend for M

Instance methods on RMFT_Solver:
    ├── _rpa_det(self, J, V, chi_DD_s_moriya, chi_DQ_s_v, chi_QD_s_v, chi_QQ_tilde)
    │                             # returns (det, a, b, c, d); det used for QCP tracking
    ├── _rpa_vertex(self, J, V, chi_DD_s_moriya, chi_DQ_s_v, chi_QD_s_v, chi_QQ_tilde, V_cap)
    │                             # full RPA pairing vertex with Frobenius-norm floor and V_cap

Module-level Lindhard infrastructure:
    _NORMAL_SECTOR_PAIRS  # 8 Nambu sector pairs for normal-state χ₀ (particle and hole, AA/BB/AB/BA)
    _lindhard_bubble(sector_pairs, E_k, V_k, f_k, shift_idx, w, eta, fermi_fn, chi0_kernel_weight)
                          # opt_einsum Lindhard sum; ZRS β²(k)·β²(k+q) kinematic form-factor; prefactor 0.25 = 0.5 (Nambu doubling) × 0.5 (hermitisation)

RMFT_Solver
    ├── get_susceptibilities_normal(...)  returns (chi_DD_s_moriya, chi_DQ_s_v, chi_QD_s_v, chi_QQ_renorm, Gamma_M_eff)  [now takes chi_SQ_sc, r_MQ instead of j_renorm]
    ├── _rpa_det(self, J, V, chi_DD_s_moriya, chi_DQ_s_v, chi_QD_s_v, chi_QQ_tilde)
    │                             # returns (det, a, b, c, d); det used for QCP tracking
    ├── _rpa_vertex(self, J, V, chi_DD_s_moriya, chi_DQ_s_v, chi_QD_s_v, chi_QQ_tilde, V_cap)
    │                             # full RPA pairing vertex with Frobenius-norm floor and V_cap
    ├── _get_chi0_norm_cache(...)
    ├── _full_rebuild()                   rebuilds p.__post_init__, K_bare, orbital operators, and all transient state in one call
    ├── _compute_nambu_susceptibility(E_k, V_k, M_A, M_B, shift_idx, eta)  unified Nambu Lehmann sum; auto-includes GG+FF anomalous sectors via vertex structure
    ├── _compute_chi_SQ_sc(q, Delta_s, Delta_d, E_k, V_k)  SC-state χ_SQ via Sz_nambu/Tau_nambu; returns (chi_SS, chi_SQ)
    ├── _classify_scf_dynamics(delta_history)  classifies SCF as: converging / limit_cycle / first_order_jump / hysteretic / stagnating
    ├── _make_vertex_params(doping, tx, ty, g_t, J_eff, K_eff)
    ├── _unique_q_pairs(fs_pts)
    ├── _fs_arc_lengths(pts)
    ├── _rebuild_orbital_operators()     rebuilds B1g_op, B1g_16, multi_op, sz_op, b1g_weight, Sz_nambu, Tau_nambu after SOC/CF change
    ├── _reset_transient_state()
    ├── compute_JT_rigidity_from_exchange(M, Q, mu, g_J, target_doping, g_t_loc)  → exchange-driven K_eff correction
    ├── compute_chi_ss_with_infinitesimal_gap(M, G_res, target_doping)  SC-onset χ_SS via finite-difference gap seeding
    ├── _B1g_expectation(M, Q, Δ_s, Δ_d, doping, mu, g_t, g_J)  → ⟨B₁g⟩ at given SCF state (internal helper)
    ├── _chi_QQ_matrix_elements(M, Q, doping, Δ_s, Δ_d, mu)  → raw χ_QQ Lehmann matrix element sum
    ├── _compute_F67_singlet(M, Q, Δ_s, Δ_d, doping, mu, tx, ty, g_J, ev, ec)  → F₆₇ singlet mixing amplitude
    ├── _dlambda_dQ_core(M, Q, Δ_s, Δ_d, doping, mu, tx, ty, g_t, g_J, K_eff, chi_SQ_sc, r_MQ_cur)  → ∂λ/∂Q core
    ├── compute_cluster_free_energy(...)  2-site ED with two-channel JT-sector WMLR; uses build_local_hamiltonian_for_bdg for site H (inline; no separate ClusterMF class)
    ├── solve_linearized_gap_equation(...)  λ_max, gap vector, λ_s/λ_d, lambda_bare_s/d, V_rpa_mean, …
    ├── compute_G_instability(...)
    ├── estimate_chi_SQ_q_full(...)
    ├── compute_hessian(...)
    ├── compute_bdg_free_energy(...)
    ├── compute_Tc_by_gap_suppression(.)    spinodal only (second-order boundary, cross-check)
    ├── compute_Tc_thermodynamic(...)       Tc₃: single heating pass, GL refinement, first-order aware
    ├── compute_lambda_vs_T(...)            Tc₂: λ(T) on Δ=0 normal-state background
    ├── compute_gap_ratio(...)              2Δ₀/kTc₃
    ├── compute_coherence_length(...)
    ├── _compute_chi_tau(...)               4-level Richardson χ_τ with weighted output (w=1.0/0.5/0.0)
    ├── summary(delta, M0)
    ├── _scf_jacobi_kick(...)
    ├── _find_mu_for_density(...)
    ├── _anderson_mix(...)
    └── solve_self_consistent(...)          Anderson-accelerated SCF; _SolveState passed into gap equations

UnifiedBayesianOptimizer  (5D: Δ_tetra, λ_soc, u, g_JT, t_pd)
    ├── _eval_constraints(s, doping)
    ├── _eval_one_doping(...)
    ├── _score(...)                         three-tier multiplicative gate; fallback uses lambda_eff
    ├── run_de_phase(...)
    ├── run_gp_seed_phase(...)
    ├── run_turbo_phase(...)
    ├── run_local_refinement(...)
    └── optimize(...)

check_sc_jt_window(...)                 K_lattice window diagnostic (standalone function)
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

For the static AFM susceptibility χ_DD_s, the computation is performed at q=0 in the folded basis: `E_kQ_all = E_k_all` and `V_kQ_all = V_k_all` — no permutation needed, since the sublattice stagger in `sz_bdg16` already selects the (π,π) channel. The solver-level `_get_chi0_norm_cache` additionally caches the Δ=0 eigenvectors across calls with the same `(M, Q, mu, tx, ty, g_J, target_doping)` within tolerance `_CHI0_CACHE_TOL = 1e-5`. The FS precomputation (`_fs_precomputed`) is similarly keyed without Δ: the FS locus and Fermi velocities are evaluated at Δ=0, making the cache Δ-independent and consistent with the BCS/BdG convention that the pairing kernel is built from the normal-state Fermi surface.

### Dual k-Grid Setup

Two separate k-grids generated once in `ModelParams.__post_init__`, both endpoint=False with uniform 1/N weights (Σw_k = 1):

- **SCF grid (_NK):** BdG diagonalization, observables, free energy, gap equations.
- **χ₀ grid (same _NK):** χ₀(q) and pairing kernel, exploiting the shift_table permutation trick.


### Thread-Safety and Clone Protocol

`RMFT_Solver` is cloned with `copy.copy()` before each parallel SCF worker:
```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s._full_rebuild()   # calls p.__post_init__(), _K_bare update, _rebuild_orbital_operators(), _reset_transient_state()
```
`_full_rebuild()` is the single canonical method for post-mutation refresh — it calls `p.__post_init__()`, updates `_K_bare`, rebuilds all orbital operators (B1g_op, B1g_16, Sz_nambu, Tau_nambu, sz_op, multi_op), and resets all transient caches. Each clone gets its own `VectorizedBdG` with its own `_H_stack` buffer, preventing inter-worker memory aliasing. `OMP_NUM_THREADS=1` prevents BLAS thread oversubscription inside `ThreadPoolExecutor`.

### Adaptive Q Update

`Q_out_raw = −(g_JT/K_eff)·⟨B̂₁g⟩` is computed at **every** iteration because `⟨B̂₁g⟩` is already available from `compute_observables_vectorized` at zero additional cost. However, it is only **injected into the Anderson vector** when one of the following holds:

- `|Q_out_raw − Q| > _Q_THR_REL · λ_hop` — Q genuinely wants to move
- `iteration == 0` — seed
- `iteration % _Q_UPDATE_PERIOD == 0` — safety heartbeat (every 3 iterations)

When none of the conditions fires, `Q_out = Q`: the Anderson residual for Q is exactly zero and the mixer leaves Q untouched — preserving the adiabatic lattice timescale without a rigid blind period. When a genuine displacement is injected, α is capped at `_MIXING × 0.3` to prevent Q–Δ oscillations.

### Vertex Cache Invalidation

The RPA pairing vertex is rebuilt when any of the following thresholds are exceeded:

| Variable | Threshold | Constant |
|---|---|---|
| M | adaptive: `_M_THR_REL · √max(\|det_afm\|, _DET_AFM_QCP_FLOOR)` ≈ 0.001–0.01 | `_M_THR_REL` |
| Q | absolute > 3% of λ_hop | `_Q_THR_REL` |
| doping | absolute > 0.005 | — |
| `chi_QQ_from_normal` flag | False (cache not built from Δ=0) | triggers unconditional rebuild |

The vertex is always built from the normal state (Δ=0): there is no Δ-threshold invalidation. The cache stores `det_afm`, `det_q0`, `chi_DD_s_afm`, `V_s_scalar`, `V_d_scalar`, `ansatz_unstable` (det_afm < 0), `Gamma_M_eff`, and FS geometry arrays `fs_pts`, `vF_arr`, `fs_pts_s`, `vF_arr_s`, `phi_d`, plus 2×2 pairing kernel results `K_pair_lambda`, `K_pair_v_s`, `K_pair_v_d`. FS geometry is reused from cache when valid, avoiding redundant `eigh` calls.

### Limit-Cycle Detection

The SCF monitors `|Δ|` over the last `_CYCLE_WINDOW = 20` iterations. If the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.25`, an oscillatory regime is diagnosed and α is reduced by `_CYCLE_DAMP_FAC = 0.45` with an Anderson history reset. This prevents the SCF from stalling in a limit cycle near the JT-active onset where the Q–Δ feedback is most nonlinear.

### SCF Loop (`solve_self_consistent`)

Anderson(5)-accelerated iteration over (M, Q, Δ_s, Δ_d, μ):
1. Build and diagonalize the 16×16 BdG stack; cache `(ev, ec)` for the iteration.
2. Compute `B1g_exp` (full Hellmann–Feynman force) from `compute_observables_vectorized` (now returns `M_stag` only; `tau_x` removed). Compute `F67s_mf = g_eff · _compute_F67_singlet(ev, ec)` when Δ≠0 and Q≠0; inject as anomalous Weiss field via `build_local_hamiltonian_for_bdg`. `F67s_mf` drives the SC–JT back-action loop.
3. Update M via Levenberg–Marquardt-regularized Newton step + BdG fixpoint blend `(1−_ALPHA_HF)·fixpoint + _ALPHA_HF·Newton`. LM floor `_MU_LM = 3.1` decreases as `|Δ|` grows.
4. Inject anomalous orbital coherence `⟨τ_x⟩_anom` (from SC condensate, computed via `_compute_orbital_coherence_from_pairs`) into the Weiss field when Δ≠0 and Q≠0, then rebuild BdG cache. This is the mean-field back-action loop.
5. Update `K_eff` on iteration 0 and when `|ΔQ| > _Q_THR_REL·λ_hop` **and** at least 5 iterations have passed since the last update with `|ΔM| > _K_EFF_M_THR = 0.02`, tracked via `_K_eff_last_Q`, `_K_eff_last_M`, and `_K_eff_last_iter`. Pass `q_renorm_cur` to `compute_JT_rigidity_from_exchange`.
6. Solve gap equations for (Δ_s_out, Δ_d_out) via RPA vertex fixed-point; pass `chi_SQ_sc` and `r_MQ_cur` to `compute_gap_eq_vectorized`. F_AA_BZ and F_AB_BZ are kept complex to preserve the converged Δ_s/Δ_d phase (see §21). Blend in 2×2 pairing kernel eigenvector direction `(K_pair_v_s, K_pair_v_d)` (weight `_ALPHA_MIX_2X2 = 0.42`) to prevent channel locking.
7. Update cluster free energy via `compute_cluster_free_energy(M, Q, mu, g_J, q_renorm_cur, F67s_mf, tx_bare, ty_bare, doping)`. Extract `q_renorm` (EMA weight `_EMA_NEW_QRW = 0.38`) and `r_MQ_cur` (EMA weight `_EMA_NEW_WEIGHT = 0.28`) from the returned dict for use in the next iteration.
8. **Update Q via Hellmann–Feynman every `_Q_UPDATE_PERIOD` iterations:** `Q_out = −(g_JT/K_eff)·B1g_exp` (full B1g operator, not just τ_x).
9. Apply Anderson(5) acceleration to `[M, Q/λ_hop, |Δ_s|/t₀, |Δ_d|/t₀]` jointly.
10. Find μ to enforce `⟨n⟩ = 1 − δ`; reuse `(ev, ec)` from μ-search; compute F_BdG and F_cluster.
11. Adaptive mixing every 5 iters: `_classify_scf_dynamics` drives mode detection. α decays by `_SCF_ALPHA_DECAY = 0.95` when SC+JT active and converging; boosts by `_SCF_ALPHA_CONVG_BOOST = 1.15` (capped at `_MIXING × _SCF_ALPHA_CONVG_CAP = 0.75`) when JT-active and improving; recovers by `_SCF_ALPHA_RECOVER = 1.60` when frozen (α_freeze_count ≥ `_SCF_FREEZE_THR = 10`); halves on divergence (`max_diff > _SCF_DIVERGE_RATIO × prev`). Limit-cycle detector reduces α by `_CYCLE_DAMP_FAC = 0.45` on oscillation. Anderson history is reset on divergence, stagnation, or Q sign flip.

After convergence (or early exit): `_classify_scf_dynamics(delta_history)` classifies the SCF trajectory as: `converging`, `limit_cycle`, `first_order_jump`, `hysteretic`, or `stagnating`. The result is stored as `scf_dynamics_regime` in the output dict and drives the multi-seed restart logic: `first_order_jump` and `hysteretic` trigger a 4-seed restart with the lowest-free-energy solution selected; `limit_cycle` only damps α. Post-convergence Hessian test (3×3 `∂²F/∂{M,Q,Δ}²` with physical-scale normalisation; mode classification uses scaled eigenvector components), coherence length ξ/a, SC-triggered JT confirmation (hessian_lmin_sc < −kT), λ_JT_kernel, ∂λ_pair/∂Q, channel decomposition (λ_s vs λ_d). A Mott filter suppresses the gap if `g_t < 0.10` or `ξ/a < 1.0`.

**Incommensurate AFM auto-retry:** after convergence a scan over `q = (π, π−δq)` with δq ∈ [0, 0.15π] checks whether the AFM susceptibility χ_DD_s peaks away from (π,π). If `δq_max > 0.05π`, `solve_self_consistent` automatically re-runs with a softened AFM seed (`M → 0.85M`) via a single recursive call guarded by the `_ic_retry` flag.

The result dict includes: all converged order parameters, Hessian eigenvalues, G3-matrix diagnostics, `lambda_JT_sc`, `lambda_JT_kernel`, `lambda_JT_opt`, ∂λ_pair/∂Q (`dlam_dQ_fs`), gap symmetry, channel decomposition (`lambda_bare_s`, `lambda_bare_d`), coherence length ξ/a (`xi_over_a`, `xi_antinodal`, `xi_Gamma6`, `xi_Gamma7`), `orbital_selective`, `valid_BdG`, 2Δ₀/kTc, `chi_tau_net` (= chi_tau_sc − chi_tau_n), `chi_tau_sc`, `chi_tau_n`, `chi_tau_weight` (`w_sc`), `richardson_ok`, `selection_ratio`, `chi_SS_afm`, `chi_DQ_s`, `chi_QQ`, `rpa_factor`, `J_eff`, `V_spin_mean`, `V_JT_mean`, `V_cross_mean`, `V_rpa_mean`, `g_delta_dom`, `afm_unstable`, `ansatz_unstable`, `q_renorm`, `r_MQ_cur`, `F67s_mf`, `g_t`, `g_J`, `tx`, `ty`, `K_eff_scf`, `rigidity_sc`, `rigidity_n`, `scf_dynamics_regime`, `mott_suspect`, `incommensurate_dq`, `incommensurate_chi_ratio`, `fs_pts`, `gap_vector`. Note: `j_renorm` removed from output; `chi_SQ_sc` renamed to `chi_SQ_sc` (no leading underscore).

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
- H2: `J_eff · χ_DD_s < 1` — below Stoner QCP (χ_DD_s is always Moriya-damped; falls back to SC-state gapped χ if past QCP, capped at 0.98)
- H3: `G22 > 0` — JT channel not self-crossing in normal state
- H4: `g_t ≥ 0.10` — coherent Fermi surface (Mott guard)

**Soft constraints / DE penalty (S1–S5, weights sum to 1.0):**
- S1 (w=0.225): `0 < λ_min(G3) < _DE_LAMBDA_MIN_OPT` (0.15) — near-critical, not past QCP
- S2 (w=0.225): monotonic reward for larger λ_max; only penalises near-divergence (λ_max > `_DE_LAMBDA_MAX_REWARD` = 4.0) and unsolvable cases — small λ_max in the normal state is not penalised, consistent with first-order transition hypothesis
- S3 (w=0.180): normal-state `lam_JT = g²·χ_QQ/K_bare > _DE_LAMBDA_JT_THRESH` (0.05) — normal-state JT orbital susceptibility above viability threshold. This is a **pre-SCF** quantity evaluated at Δ=0; it is distinct from `lambda_JT_sc` (post-SCF, SC state) used in `check_sc_jt_window`.
- S4 (w=0.270): `∂λ_pair/∂Q > 0` — JT renormalises V_pair upward
- S5 (w=0.100): G22-margin > `_DE_G22M_SAFE = 0.25` — distance from the spontaneous-JT boundary; `S5 = 1 − tanh(G22 / _DE_G22M_SAFE)` continuously penalises proximity to G22 = 0

**Scoring (`_score`)** — three-tier multiplicative architecture:
- *Tier 1 (hard guards):* Mott/incoherence guard (`g_t < 0.10` or `ξ/a < 1`), `J·χ_DD_s(Moriya) > 2` → score = 0. Projection-quality penalty `proj_factor = 1 − 0.5·clip((J_eff/Δ_CF)², 0, 1)` accounts for Γ₇b contamination when the BdG truncation to Γ₆⊕Γ₇a is not fully valid; `proj_factor` is applied as a multiplicative factor on the final score.
- *Tier 2 (smooth mechanism weights):*
  - `w_lJT`: parabolic arch on λ_JT_sc ∈ [0, 1]; `x = (λ_JT_sc − 0.05)/0.95`, `arch = clip(−x(x−1)/_BO_ARCH_DENOM, 0, 1)`. Mathematical peak of the unclipped parabola at x=0.5 → **λ_JT_sc ≈ 0.52**; plateau (=1) for λ_JT_sc ∈ [0.32, 0.73]; `w_lJT = 0.10` when λ_JT_sc ≥ 1 (spontaneous/runaway).
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
| `t_pd` | t_pd | 0.460 eV | pd hybridisation integral (primary hopping; t₀ = t_pd²/Δ_CT derived) |
| `u` | u | 19.0 | U/t₀ ratio; Hubbard U = u·t₀ |
| `lambda_soc` | λ_SOC | 0.160 eV | Atomic SOC constant (t₂g shell); determines Γ₆–Γ₇ splitting |
| `Delta_tetra` | Δ_tet | −0.120 eV | Tetragonal CF (**required < 0**); Δ_CF derived |
| `g_JT` | g_JT | 0.160 eV/Å | Electron–phonon JT coupling |
| `K_lattice` | K | 1.100 eV/Å² | Bare phonon stiffness; K_eff computed at runtime |
| `lambda_hop` | λ_hop | 1.100 Å | Hopping decay: t(Q) = t₀·exp(±Q/λ) |
| `Delta_CT` | Δ_CT | 2.000 eV | Charge-transfer gap (material-class constant) |
| `Delta_inplane` | Δ_ip | 0.010 eV | B₂g in-plane CF; splits Γ₇ doublet |
| `kT` | kT | 0.010 eV | Temperature (~116 K) |
| `tol` | — | 1e-4 | Convergence threshold |
| `Z` | Z | 4 | Coordination number |

### Module-level Physical Constants

Fixed unit-conversion and empirical values extracted to named constants:

| Constant | Value | Description |
|---|---|---|
| `_KB_EV` | 8.617333×10⁻⁵ eV/K | Boltzmann constant |
| `_EV_TO_K` | 11604.518 K/eV | 1 eV in Kelvin |
| `_GW_G_J_PREFACTOR` | 4.0 | g_J = 4/(1+δ)²; Gutzwiller exchange renormalization prefactor (slave-boson / Kotliar-Ruckenstein, half-filling limit) |
| `_GW_G_T_NUMERATOR` | 2.0 | g_t = 2δ/(1+δ); Gutzwiller hopping prefactor numerator coefficient |
| `_GW_DOS_PREFACTOR` | 2.0 | N₀ = 2/W; 2D tight-binding bare DOS prefactor (W = bandwidth) |

| `_BCS_RATIO_STRONG` | 3.8 | 2Δ/kTc threshold: strong coupling |
| `_BCS_RATIO_VSTRONG` | 5.0 | 2Δ/kTc threshold: very-strong coupling |
| `_BCS_RATIO_EXOTIC` | 7.0 | 2Δ/kTc threshold: exotic / non-phononic |
| `_GL_DELTA_MIN` | 2e-3 eV | \|Δ\| floor for GL fit points; below this the SC state is numerically unreliable |
| `_GL_MIN_PTS` | 2 | Minimum stable SC points required for GL fit |
| `_GL_MAX_PTS` | 4 | Upper window of recent stable SC points used in GL regression |
| `_GL_TC_MARGIN` | 0.05 | Max relative deviation \|Tc_GL−T_spinodal\|/T_max to accept GL result |
| `_GL_SPINODAL_JUMP` | 0.15 | D_spinodal/Δ₀ < this → GL extrapolation considered reliable (small first-order jump) |
| `_M0_STONER_AMP` | 0.18 | Stoner amplitude in `estimate_M0` |
| `_M0_PRIOR_SLOPE` | 0.40 | Slope of M_prior vs. doping |
| `_M0_PRIOR_REF` | 0.06 | Reference doping for M_prior (AFM dome optimum) |
| `_M0_W_SC_LAMBDA_WIDTH` | 15.0 | λ_lin width for SC-blend weight: w_sc = (λ−1)/this, capped at 0.75 |
| `_M0_W_SC_CAP` | 0.75 | Upper cap on w_sc (SC M₀ blend weight) |
| `_M0_STONER_CLIP_LO` | 0.05 | Lower clip on M_stoner estimate |
| `_M0_STONER_CLIP_HI` | 0.20 | Upper clip on M_stoner estimate |
| `_M0_PRIOR_CLIP_LO` | 0.08 | Lower clip on M_prior |
| `_M0_PRIOR_CLIP_HI` | 0.22 | Upper clip on M_prior |
| `_M0_S_CLIP_MAX` | 5.0 | Upper clip for Stoner S = J·N₀ |
| `_M0_DELTA_C` | 0.23 | Critical doping above which Stoner M→0 linearly |
| `_M0_PRIOR_SLOPE` | 0.40 | Slope of M_prior vs. doping |
| `_M0_PRIOR_REF` | 0.06 | Reference doping for M_prior (AFM dome optimum) |
| `_M0_W_DOPING_SAT` | 0.20 | Doping scale at which blend weight w→1 |
| `_BO_ARCH_DENOM` | 0.2025 | Parabolic arch denominator; arch = −x(x−1)/D where x=(λ_JT−0.05)/0.95. Saturates to clip=1 (plateau) for λ_JT ∈ [0.32, 0.73]; mathematical maximum of the unclipped parabola is at x=0.5 (λ_JT≈0.52) |
| `_RPA_BW_FACTOR` | 8.0 | Bandwidth = 8t for 2D square-lattice tight-binding |
| `_BO_OPT_JCHI` | 0.875 | Optimal J·χ_SS for Gauss gate in scoring (near-QCP but metallic) |
| `_BO_SIG_JCHI` | 0.15 | Gaussian σ for J·χ_SS gate |
| `_BO_JCHI_FLOOR` | 0.3 | Score floor when J·χ unavailable (jchi≈0) |
| `_BO_JCHI_NOISE` | 0.05 | J·χ below this is numerical noise → apply floor |
| `_SCORE_SOFTENING_SIG` | 0.05 | Sigmoid width for w_softening = 1/(1+exp(jt_softening/this)) |
| `_BO_W_HESSIAN_FLOOR` | 0.30 | Floor for w_hessian / w_lJT_kernel when data missing |
| `_BO_W_LJT_OVR_SAT` | 0.10 | w_lJT when λ_JT_kernel ≥ 1 (Rayleigh quotient over-saturation) |
| `_BO_LJT_KERNEL_SIG` | 10.0 | Sigmoid steepness k in w_lJT_kernel = 1/(1+exp(−k·(x−0.05))) |
| `_BO_JCHI_GAPPED_CAP` | 0.98 | J·χ_SS(gapped) must be < this to be accepted as safe |
| `_BO_G22_MARGIN_CTR` | 0.25 | G22 sweet-spot centre for g22_margin_f sigmoid |
| `_BO_W_STONER_BAD` | 0.20 | Score weight applied when AFM Stoner criterion is violated |
| `_BO_G_FALLBACK` | 5e-3 | Overall scale for G-matrix proxy score in the no-gap fallback region |
| `_BO_SIGMOID_W` | 0.30 | Sigmoid width for g22_f gate (fallback-only scoring path) |
| `_BO_SPONT_JT_PEN` | 0.05 | Penalty floor in g22_f (used only in `_g_fallback_score`) |
| `_BO_SC_HESS_SIG` | 0.05 eV | SC Hessian sigmoid width: sc_hessian_f = 1/(1+exp(λ_min_sc/_BO_SC_HESS_SIG)) |
| `_BO_G22_MARGIN_W` | 0.15 | Sigmoid width for g22_margin_f |

### Module-level SCF Constants

These are fixed at compile time and not Bayesian-optimised:

| Constant | Value | Description |
|---|---|---|
| `_NK` | 64 | k-points per direction (must be even) |
| `_MAX_ITER` | 700 | Maximum SCF iterations |
| `_MIN_ITER` | 4 | Minimum iterations before convergence check |
| `_MIXING` | 0.07 | Base Anderson mixing weight |
| `_MU_LM` | 3.1 | LM regularization floor for M Newton step |
| `_ALPHA_HF` | 0.35 | Newton vs BdG fixpoint blend for M |
| `_TR_M_STEP_MAX` | 0.2 | Upper bound on \|ΔM\| (eV); reduced near QCP |
| `_TR_M_STEP_MIN_FLOOR` | 1e-3 | Absolute minimum M step — prevents total freeze near M→0 |
| `_M_STEP_FLOOR_REL` | 0.005 | step_floor = max(_M_STEP_FLOOR_REL×\|M\|, _M_STEP_FLOOR_ABS) |
| `_M_STEP_FLOOR_ABS` | 0.002 | Absolute minimum M step regardless of \|M\| |
| `_M_STEP_FLOOR_M_MIN` | 0.010 | Reference M scale in step floor: max(\|M\|, this) |
| `_M_J_EFF_FLOOR_FRAC` | 0.20 | j_eff_floor = max(\|J_eff\|, this×t_eff, ε) — QCP guard preventing ΔM∝1/J_eff→∞ |
| `_QCP_BLEND_THRESH` | 0.05 | \|det\| < this → near QCP, use damped direction-only M update |
| `_QCP_BLEND_WEIGHT` | 0.18 | Blend weight when near QCP (×_ALPHA_MIX_2X2) |
| `_MOMENT_RATIO_MIN` | 0.01 | Lower bound for \|⟨Γ₇\|S_z\|Γ₇⟩\| / \|⟨Γ₆\|S_z\|Γ₆⟩\|; below this indicates near-degenerate eigenvectors |
| `_N_FS` | 130 | FS k-points used in the vertex q-loop |
| `_Q_UPDATE_PERIOD` | 3 | Update Q every N inner iterations |
| `_ALPHA_MIX_2X2` | 0.42 | Blend weight: 2×2 eigenvector direction vs fixed-point gap update |
| `_MORIYA_C` | 0.21 | Moriya damping prefactor; α_M = C·f(δ)·sat(t/J), where f(δ)=δ/(δ+`_MORIYA_DSAT`) ∈ (0,1) and sat(t/J)=(t/J)/(`_MORIYA_TJ_SAT`+t/J) ∈ (0,1) |
| `_MORIYA_FM_BOOST` | 2.8 | FM/Pomeranchuk channel's Moriya damping factor at q≈0 |
| `_MORIYA_FM_Q0` | 0.3 | Fallback FM crossover scale (π r.l.u.) when J_eff≈0; normally computed as √(Γ_M/J_eff) ∈ [0.05, 0.5] dynamically |
| `_ALPHA_MORIYA` | 0.02 | Moriya damping floor |
| `_LAMBDA_JT_VIABLE` | 0.05 | Minimum λ_JT_sc for SC-triggered JT viability |
| `_CHI_SQ_S_PADE_W` | 0.05 | Padé regularisation width for χ_SQ: χ_SQ_v=χ_SQ/(1+|χ_SQ|/w); linear at |χ_SQ|≪w, saturates to ±w |
| `_ANDERSON_TIKHONOV` | 1e-8 | Tikhonov β / diag_max in Anderson normal equations |
| `_ANDERSON_TRUST` | 2.4 | Trust-region step-size limit (multiples of simple step) |
| `_ANDERSON_W_LO` | 0.3 | Lower blend weight between Anderson and simple mixing |
| `_ANDERSON_W_HI` | 0.8 | Upper blend weight |
| `_CYCLE_WINDOW` | 20 | Iteration window for limit-cycle detection |
| `_CYCLE_THRESHOLD` | 0.25 | std/mean of |Δ| above this → oscillatory regime |
| `_CYCLE_DAMP_FAC` | 0.45 | α reduction factor on oscillation detection |
| `_SCF_DIVERGE_RATIO` | 1.05 | max_diff > prev×this → SCF diverging |
| `_SCF_STAGNATE_RATIO` | 0.95 | max_diff > prev×this (and not diverging) → stagnating |
| `_SCF_ALPHA_DECAY` | 0.95 | α ×= this when SC+JT active and converging (mild damping) |
| `_SCF_ALPHA_RECOVER` | 1.60 | α ×= this on freeze-recovery (restores mobility after stagnation) |
| `_SCF_FREEZE_THR` | 10 | α_freeze_count ≥ this → trigger freeze-recovery boost |
| `_SCF_ALPHA_FREEZE_LO` | 0.15 | α < _MIXING × this → too frozen, trigger recovery |
| `_SCF_ALPHA_FREEZE_HI` | 0.60 | α recovery ceiling: min(α×_RECOVER, _MIXING×this) |
| `_SCF_ALPHA_CONVG_BOOST` | 1.15 | α boosted (×this) when SC+JT active and converging |
| `_SCF_ALPHA_CONVG_CAP` | 0.75 | α ceiling (×_MIXING) during SC+JT converging branch |
| `_MODE_FRAC_DOMINANT` | 0.60 | fX > this → mode dominated by X (pure-SC/JT/AFM) |
| `_MODE_FRAC_MIXED` | 0.30 | fX > this (both Δ and Q) → SC-triggered-JT mode |
| `_VF_FLOOR` | 1e-4 | Fermi velocity floor; prevents 1/|vF|→∞ at hot spots (~0.01·t0·a/ħ); used in FS sampling weight |
| `_VF_FLOOR_TIGHT` | `_VF_FLOOR × 1e-4` = 1e-8 | Tighter floor for `dl/vF` arc-length weight; must be ≪ `_VF_FLOOR` |
| `_RPA_V_CAP_ALPHA` | 2.2 | Dynamic vertex cap: V_cap = α·max(`_RPA_BW_FACTOR`·max(\\|tx\\|,\\|ty\\|), J_eff); 2.2× headroom above BEC-BCS crossover. No static hard cap — cap is computed per call in `_make_vertex_params`. |
| `_DET_SIGN_FLIP_SCALE` | 0.05 | \|det_afm\| sigmoid midpoint for V_d sign-flip EMA suppression; below this the blend weight approaches `_EMA_SIGN_FLIP_W_MIN` |
| `_EMA_SIGN_FLIP_W_MIN` | 0.20 | Minimum w_factor on V_d sign flip; preserves EMA adaptation even at det≈0 |
| `_EMA_SIGN_FLIP_SLOPE` | 6.0 | Sigmoid steepness k in sign-flip EMA: w = w_min + (1−w_min)/[1+exp(−k·(\|det\|/scale−0.5))] |
| `_VMAT_LOW_VAR_FRAC` | 0.10 | std(V)/\|mean(V)\| < this → vertex low-variance flag (`⚠low-var` in log) |
| `_V_PREV_SIGN_FLOOR` | 1e-6 | \|V_d_prev\| below this → treat as zero, skip sign-flip EMA check |
| `_V_AFM_Q_MIN` | 0.70 | \|q\|/π > this → counted in AFM region for vertex diagnostic |
| `_V_FWD_Q_MAX` | 0.35 | \|q\|/π < this → counted in forward-scattering region for vertex diagnostic |
| `_RPA_DET_WARN` | 0.11 | \|det_afm\| < this → QCP proximity warning in diagnostics and adaptive mixing |
| `_RPA_QCP_PENALTY` | 0.40 | α reduction per unit \|det_afm\| < 0 past QCP (adaptive mixing, BO near_qcp flag) |
| `_DET_AFM_FLOOR` | 1.0 | Default det_afm when vertex cache is absent (normal state, no QCP present) |
| `_DET_DEPTH_CAP` | 5.0 | Max det_depth in jump-cap exponential suppression past QCP |
| `_DET_JUMP_HALF_SCALE` | 0.5 | exp(−this·det_depth) decay rate for gap-jump cap past QCP |
| `_JUMP_CAP_FLOOR` | 1.05 | Minimum effective_jump_cap (prevents total gap freeze past QCP) |
| `_CHI_SQ_FLOOR` | 1e-12 | Amplitude floor for χ_SQ before Padé regularisation |
| `_MAX_REGR_CLIP` | 3.6 | Hard upper limit on regression renormalization factors (kept for χ_SQ Padé context) |
| `_CLUSTER_N_EFF_FLOOR` | 2.0 | Effective sample size below which r_Q = r_MQ = 0 (too few states) |
| `_REGR_T_ALPHA` | 0.05 | Two-sided t-test significance level for r_Q and r_MQ shrinkage |
| `_NODAL_REGION_PCTL` | 25 | Percentile cut for nodal/antinodal FS decomposition: nodal = lowest 25% of \|φ_d\|; antinodal = upper 25% |
| `_REGR_EPS` | 1e-12 | Zero-guard at denominators in the WMLR regression solver |
| `_REGR_VAR_MIN` | 1e-9 | Minimum variance threshold in WMLR; Bessel-corrected: min_var × max(1, 2/(n_eff−1)) |
| `_EMA_NEW_WEIGHT` | 0.28 | EMA new-sample weight for r_MQ, V_d, Λ_inst |
| `_EMA_NEW_QRW` | 0.38 | EMA weight for q_renorm; higher than `_EMA_NEW_WEIGHT` to give the orbital channel faster response |
| `_VERTEX_DIAG_MIN_FS` | 10 | Minimum FS points required for V_mat structure diagnostics; below this, std/mean statistics are unreliable |
| `_PHI_D_FLOOR` | 1e-3 | Minimum φ_d max value to enable nodal/antinodal decomposition |
| `_FS_SAMPLING` | 2.8 | Integration window (in units of kT) around the Fermi level for FS k-point selection |
| `_XI_NODAL_MIN` | 2.0 | ξ/a above this required for coherent nodal quasiparticles (BCS-side criterion) |
| `_ORBITAL_SEL_FRAC` | 0.15 | \|ξ_Γ₆ − ξ_Γ₇\|/ξ > this → system classified as orbitally selective |
| `_DOPING_MOTT_FLOOR` | 0.01 | \|δ\| < this → at/near Mott insulator; skip SCF, return metallic=False |
| `_DELTA_ABS_FLOOR` | 1e-3 eV | \|Δ\| below this → jump limiter bypassed (seed-gap free-growth phase; ~0.7–2% of t0) |
| `_BCS_SEED_FRACTION` | 0.09 | BCS seed magnitude as fraction of t_eff; initial Δ seed when starting cold |
| `_DELTA_JUMP_CAP` | 5.0 | Max \|Δ_new\| / \|Δ_current\| ratio per iteration (prevents exponential blow-up) |
| `_DQ_FS_VERTEX` | 0.03 Å | Finite-difference step for ∂V(k,k')/∂Q on FS (≈3–6% of λ_hop) |
| `_JT_ACT_THR` | 0.04 | Γ₆–Γ₇ mixing threshold induced by SC condensate for JT-active classification |
| `_K_EFF_M_THR` | 0.02 | \|ΔM\| threshold to trigger K_eff rigidity recompute |
| `_G_T_COHERENCE_MIN` | 0.10 | g_t floor for coherent ZRS band (Mott guard) |
| `_JCHI_HARD_REJECT` | 2.0 | J·χ_SS > this → score = 0 (deeply AFM, SC impossible) |
| `_V_CUT` | 20.0 | Pairing vertex near-divergence detector threshold |
| `_IC_RATIO_FLOOR` | 1.05 | r < this → negligible inter-channel correction |
| `_IC_RATIO_CAP` | 3.00 | r > this → strong IC, cap reduction to avoid too-small M |
| `_MODE_PULL_FRAC` | 0.30 | Fraction of (M − M_phys_est) used as kick pull in pure-SC/SC-JT mode |
| `_KICK_M_EXCESS_CTR` | 0.70 | M_excess = max(0, (M_kick − this) / _KICK_M_STIFF_WIDTH); overshoot protection |
| `_KICK_M_STIFF_WIDTH` | 0.30 | Width of M-excess sigmoid in Newton kick overshoot protection |
| `_KICK_JCHI_EXCESS_CTR` | 0.70 | jchi_excess = max(0, (jchi − this) / _KICK_JCHI_STIFF_WIDTH) |
| `_KICK_JCHI_STIFF_WIDTH` | 0.30 | Width of jchi-excess sigmoid |
| `_KICK_REDUCTION_AMP` | 0.35 | Amplitude of M-kick reduction: M_kick × (1 − this × excess) |
| `_KICK_LAMBDA_SC_THR` | 5.00 | λ_max above this → supercritical; triggers extra M pull-down |
| `_KICK_LAMBDA_SC_WIDTH` | 15.00 | Denominator for supercritical pull fraction: (λ−thr)/this |
| `_KICK_PULL_CAP` | 0.60 | Max pull fraction |
| `_KICK_BOOST_AMP` | 3.00 | Δ-kick boost: 1 + this × λ_excess/(1+λ_excess) |
| `_KICK_SC_LOG_SCALE` | 5.00 | Supercritical mixing log scale: log10(λ/this) |
| `_KICK_M_CLIP_LO` | 0.02 | Hard lower clip on M_kick (normal SCF path) |
| `_KICK_M_MOTT_CLIP_LO` | 0.05 | M_kick lower clip near Mott boundary (g_t < coherence min) |
| `_KICK_M_CLIP_HI` | 0.45 | Hard upper clip on M_kick |
| `_BO_MAX_WORKERS` | 6 | ThreadPoolExecutor worker ceiling |
| `_FEASIBILITY_THRESHOLD` | 0.25 | Partial penalty ≥ this → infeasible regardless of S4 |
| `_DE_LAMBDA_MAX_REWARD` | 4.0 | λ_max above this → penalised (past QCP / numerically unstable) |
| `_DE_LAMBDA_MIN_OPT` | 0.15 | λ_max below this → weak pairing (S2 sigmoid centre) |
| `_DE_LAMBDA_JT_THRESH` | 0.05 | Normal-state `lam_JT = g²·χ_QQ/K_bare` below this → S3 penalised (JT channel too weak in normal state; pre-SCF DE scout criterion) |
| `_MATH_EPS` | 1e-9 | General protection floor against division by zero |
| `_DEN_DERIV_FLOOR` | 1e-12 | ∂n/∂μ floor in Newton μ-finder |
| `_BRENTQ_TOL` | 1e-5 | Brentq μ-bracketing tolerance |
| `_FERMI_ARG_CLIP` | 100.0 | Clip argument of exp() in Fermi function |
| `_ENTROPY_CLIP` | 1e-12 | Lower clip for f in entropy −f·ln(f) |
| `_FD_MASK_DF` | 1e-12 | \|Δf\| mask threshold in χ₀ Lehmann sums (below → treated as zero) |
| `_FD_MASK_DE` | 1e-6 | \|ΔE\| mask threshold in χ₀ Lehmann sums |
| `_FD_MASK_DE8` | 1e-8 | Tighter \|ΔE\| mask for d²F/dM² off-diagonal term |
| `_LINDHARD_CHUNK` | 128 | k-point batch size in opt_einsum loops (memory vs. speed trade-off) |
| `_BZ_NORM` | (2π)² | BZ area in reduced coordinates (a=1, ħ=1); FS arc-length integration measure dl/((2π)²·vF) |
| `_CLUSTER_SIZE` | 2 | Number of sites in the exact-diagonalization cluster |
| `_ETA_T_FRAC` | 0.10 | Normal-state Lindhard broadening: η = _ETA_T_FRAC · kT (thermal) |
| `_ETA_DELTA_FRAC` | 0.02 | SC-state Lindhard broadening: η += _ETA_DELTA_FRAC · \|Δ\| (gap-scale) |
| `_ETA_GRID_FLOOR` | 0.002 | Broadening floor: η ≥ _ETA_GRID_FLOOR · t₀ (k-grid aliasing) |
| `_FS_CACHE_TOL` | 1e-3 | Parameter-change tolerance for Fermi-surface cache invalidation (fuzzy key comparison) |
| `_CHI0_CACHE_TOL` | 1e-5 | Parameter-change tolerance for χ₀ eigenvector cache invalidation |
| `_CHI_SQ_FLOOR_FRAC` | 1e-4 | Dynamic floor factor for χ_SQ: floor = _CHI_SQ_FLOOR_FRAC · √(χ_DD · χ_QQ) |
| `_MAX_RENORM_SCALE` | 2.9 | Upper clip reference (retained for χ_SQ regularisation context; Moriya j_renorm boost removed) |
| `_Q_FM_CLIP_MIN` | 0.05 | Minimum FM damping momentum extent (π units) |
| `_Q_UNIQUE_SCALE` | 100000 | Integer scaling factor for hashing unique q pairs (prevents floating-point collisions) |
| `_PI_INT` | 314159 | π in scaled integer units (used with `_Q_UNIQUE_SCALE`) |
| `_Q_FM_CLIP_MAX` | 0.50 | Maximum FM damping momentum extent (π units) |
| `_TR_SHRINK` | 0.65 | TuRBO trust-region: shrink ×0.65 on failure |
| `_TR_EXPAND` | 1.35 | TuRBO trust-region: expand ×1.35 on consecutive improvement |

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
3. **Normal-state JT stability:** `K_eff > 0` and `λ_min > 0`, i.e. `G3[2,2] > 0` at Δ=0 and no spontaneous instability.
4. **SC-triggered regime:** `lambda_JT_sc = g_JT²·max(−chi_tau_net,0)·w_sc / K_eff > _LAMBDA_JT_VIABLE = 0.05`  
   where `chi_tau_net = chi_tau_sc − chi_tau_n`, `w_sc` is the Richardson reliability weight (1.0/0.5/0.0), and `K_eff = K_lattice + ∂²F_ex/∂Q²`.

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
1. SOC+CF diagonalization → Δ_CF, `moment_ratio`, `b1g_weight`, k-grids, `shift_table`, orbital operators.
2. `solver.summary(target_doping)` — all derived parameters and pre-SCF diagnostics.
3. Reference SCF at default parameters → self-consistent (M, Q, Δ, μ) for diagnostics.
4. `compute_G_instability(target_doping, M, q_renorm=result['q_renorm'], r_MQ=result['r_MQ_cur'])` at self-consistent M + `check_sc_jt_window()` with `chi_tau_net` from post-SCF.
5. Linearized gap equation and channel decomposition (λ_s vs λ_d) from SCF result dict.
6. Tc block: Tc₁ (McMillan with ω_SF=J_eff); Tc₂ (λ(T)=1 crossing on Δ=0 normal-state scan, `compute_lambda_vs_T`); Tc₃ (thermodynamic, `compute_Tc_thermodynamic`).
7. If `need_optimization = True`: `UnifiedBayesianOptimizer.optimize()` — DE scout → GP seed → TuRBO → local refine.
8. Post-SCF: Hessian, coherence length, gap ratio (2Δ₀/kTc₃), phase-diagram scan.

The flag `need_optimization` (default `False`) controls whether the Bayesian optimisation pipeline runs.

---

## Output & Diagnostics

### Iteration Log

Each SCF step prints (thread-safe, every `_LOG_PERIOD` iterations):

```
[SCF] δ=…  iter/max  conv=…  M=…  Q=…  |Δ|=…  J_eff=… eV  q_renorm=…  r_MQ=…  mu=…
      dFM=…  dAFM=…  V_s=…  V_d=…  [⚠low-var] [⚠same-sign]
      [V_afm=… V_fwd=… neg=… V_dd_fs=…]   ← only when V_d < 0
      Γ_M=…  α=…  B1g=…  F67s=…  [regime]  …s/it
```

`⚠low-var` and `⚠same-sign` appear when the V_mat structure diagnostic flags are active (requires `N_fs > _VERTEX_DIAG_MIN_FS`). The q-resolved vertex diagnostic block `[V_afm=… V_fwd=… neg=… V_dd_fs=…]` is appended only when `V_d < 0`, providing:

- `V_afm`: mean V(q) at `|q| > 0.7π` (AFM region; should be > 0 for spin-fluctuation pairing)
- `V_fwd`: mean V(q) at `|q| < 0.35π` (forward scattering; typically < 0, normal)
- `neg`: fraction of q-points with V < 0 (> 0.9 → globally repulsive, unphysical)
- `V_dd_fs`: FS-projected d-wave vertex; negative → d-wave not supported at this parameter point

At convergence, the SCF-RES line prints: converged order parameters (M, Q, |Δ_s|, |Δ_d|), density, μ, F_bdg, F67s_mf, q_renorm, r_MQ_cur, det_AFM, JT flag, SCF dynamics regime, and symmetry mismatch if linearized and SCF channels disagree. Channel decomposition (λ_s vs λ_d), `lambda_JT_sc`, `lambda_JT_opt`, `lambda_JT_kernel`, ∂λ_pair/∂Q, SC-triggered JT Hessian confirmation, `chi_tau_net` breakdown including `chi_tau_weight` (`1.0` / `0.5` / `0.0`), SC-JT window diagnostics, and incommensurate AFM scan.

### Tc Block

```
[TC-PRELIM]  Tc₁(McMillan-SF): λ_max=…  ω_SF(J_eff)=… meV  → … meV (… K)
[TC-PRELIM]  λ_eff(Schur+JT)=…
[TC-SIMPLE]  Tc₂(λ=1)=… meV  slope=… meV⁻¹  n_crossings=…
[TC-THERMO]  Tc₃(thermo)=… meV (… K)  spinodal=… meV  order=…  Δ_jump=…  JT-uplift=…%
[TC-THERMO]  2Δ₀/kTc=…  [BCS-like | strong | very-strong | exotic / non-phononic]  (from Tc₃)
```

`chi_tau_weight` is logged when < 1.0 to signal reduced reliability of the SC-JT feedback estimate.

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

With BO results, a 4th row: BO progress (Δ and score vs. evaluation), doping vs. score, parameter scatter — all coloured by λ_JT regime.

---

## Known Limitations

| Approximation | Impact |
|---|---|
| No Pauli exclusion between cluster sites | Slight overestimate of AFM correlations; controlled by `_ALPHA_HF` blend |
| No charge-transfer fluctuations ⟨n_A n_B⟩ | Charge fluctuations negligible when U_mf ≫ t |
| Static phonon (Q is a mean field) | Zero-point quantum lattice fluctuations neglected; ω_JT derived from K_eff, not a free parameter |
| 4×4 BdG truncation | Valid when Δ_CF ≫ kT and Γ₇split/Δ_CF ≪ 1; monitored via `(J_eff/Δ_CF)²` projection-quality penalty |
| No spatial fluctuations | Cannot describe pseudogap, stripes, or phase separation |
| RPA static (ω = 0) | Dynamical vertex corrections absent |
| Q update adaptive + heartbeat | Q_out_raw computed every iteration from Hellmann–Feynman; injected into Anderson when `|ΔQ| > _Q_THR_REL·λ_hop` OR every `_Q_UPDATE_PERIOD = 3` iterations (heartbeat). α capped at `_MIXING×0.3` on genuine displacement. Back-action of Q on exchange rigidity approximate during SCF transient |
| χ_τ at post-convergence only | Self-consistent Q back-action on chi_tau neglected during SCF |
| `compute_G_instability` at Δ=0 | G-matrix evaluates normal-state only; SC-triggered JT confirmed via post-SCF Hessian λ_min < −kT |
| ∂λ_pair/∂Q at frozen Fermi surface | FS geometry frozen at middle Q |
| δχ_τ baseline subtraction approximate in D₂h | Normal-state B1g response estimated at Δ=0; small D₂h corrections to χ_τ_n neglected |
| `chi_tau_weight` partial suppression | When `w_sc = 0.5`, the finer Richardson scale is used but the response may still be overestimated near first-order SC-JT boundaries |
| `scf_dynamics_regime` classification | Computed by `_classify_scf_dynamics(delta_history)` at end of SCF. `first_order_jump` and `hysteretic` trigger multi-seed restart (4 seeds, lowest free energy wins); `limit_cycle` only damps α. New classes `converging` / `stagnating` logged but do not alter flow |
| q_renorm / r_MQ WMLR regression | Both are q=0, ω=0 local JT-sector vertices; the A1g magnetic channel is analytically fixed (Gutzwiller). `r_Q ∈ [−2.0, +2.0]`, `r_MQ ∈ [−2.0, +2.0]`. t-test shrinkage may over-conservatively suppress small but physical MQ coupling near Q=0. Regression is skipped entirely when n_eff < 2 or `|Q| < 1e-4` |
| GL Tc extrapolation | Applied only when `D_spinodal/Δ₀ < 0.15` (near-second-order); in the merged `_find_crossing_and_spinodal` history the GL fit uses all points with `|Δ| > _GL_DELTA_MIN` regardless of `sc_wins`, since the collapse shape is needed. May miss Tc refinement for strongly first-order transitions |
| V_d sign-flip EMA | Suppresses numerical oscillation but may slow the vertex response near genuine sign changes (e.g. at doping-driven crossover from d-wave to s-wave dominance). Non-issue for parameter regions where one channel dominates throughout |

---

## References

- Ecsenyi, S. (2026). *Multipolar superconductivity and coherent orbital mixing* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang et al. (1988). *Supercond. Sci. Technol.* 1, 36; Bünemann, J., Weber, W. & Gebhard, F. (1998). *Phys. Rev. B* 57, 6896.
- ZSA classification: Zaanen, Sawatzky & Allen (1985). *Phys. Rev. Lett.* 55, 418.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.
- McMillan strong-coupling formula: McMillan, W.L. (1968). *Phys. Rev.* 167, 331.
- Ginzburg-Landau theory: Ginzburg, V.L. & Landau, L.D. (1950). *Zh. Eksp. Teor. Fiz.* 20, 1064.
- TuRBO / Bayesian optimisation: Eriksson, D. et al. (2019). *NeurIPS.*
- Richardson extrapolation: Richardson, L.F. (1911). *Phil. Trans. R. Soc. A* 210, 307.
- Moriya, T. (1985). *Spin Fluctuations in Itinerant Electron Magnetism.* Springer.

---

*For questions or contributions, open an issue or pull request.*
