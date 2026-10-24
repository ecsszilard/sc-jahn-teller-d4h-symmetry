# Multipolar SC-Triggered Jahn–Teller Model

> **A renormalized mean-field theory (RMFT) simulation of superconductivity-triggered B₁g Jahn–Teller distortion in a D₄h charge-transfer insulator with strong spin–orbit coupling and collinear antiferromagnetic order.**

This repository implements the self-consistent theory described in *"Multipolar superconductivity and Jahn–Teller activation in strongly correlated systems"* (Ecsenyi Szilárd, 2026). The core claim is that in a class of strongly spin–orbit-coupled, charge-transfer Mott insulators, a **B₁g lattice (Jahn–Teller) distortion is symmetry-forbidden in the antiferromagnetic normal state and is unlocked only by Cooper-pair condensation** — the causal arrow runs from superconductivity to the structural instability, not the other way around.

The code carries this mechanism through **both** cross-doublet pairing sectors the symmetry argument allows ($\Gamma_6$–$\Gamma_{7a}$ and $\Gamma_6$–$\Gamma_{7b}$), and — in its current form — through a **general, symmetry-classified pairing basis** rather than a fixed pair of amplitudes: a library of 42 lattice-harmonic form factors (on-site, nearest-neighbour and same-sublattice bonds; even and odd in $k$), each carried as its own complex amplitude in each Γ₇ branch ($2\times42=84$ self-consistent complex gap amplitudes). The pairing kernel is resolved exactly into the eight one-dimensional representations of the magnetic symmetry group $G=\{E,T_Q\}\times D_{2h}$, so that gap functions of different symmetry are never mixed by construction at the linear level. Around this sits an exact multi-orbital cluster diagonalization (including Kanamori Hund's coupling) for the superexchange scale, a Moriya self-consistent-renormalization treatment of the spin channel, a four-site plaquette cluster ED that supplies the local irreducible vertex and the beyond-mean-field stiffness and energy of the lattice channel, and a lattice coordinate $Q$ that is not iterated together with the other order parameters but **profiled out** at every SCF step as the minimum of the total free energy (including the exact cluster energy), then relaxed with its own weight.

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

In the textbook picture, the Jahn–Teller (JT) effect and superconductivity (SC) work against each other: a JT lattice distortion lifts orbital degeneracy at the Fermi level by localizing electrons into small polarons, which competes with the coherent, delocalized electron pairs needed for Cooper pairing. This model inverts that relationship for a specific class of materials.

**The central claim:** in a D₄h, charge-transfer-insulating, strongly correlated system where spin–orbit coupling (SOC) is *not* a perturbation — it reorganizes the local Hilbert space into well-separated Kramers doublets (Γ₆, Γ₇) — a collinear antiferromagnetic (AFM) ground state stabilizes *only* dipolar (rank-1) multipolar order. Concretely:

- The Γ₆ ground-state doublet carries **no electric quadrupole moment**: a pure Γ₆ (or Γ₇) manifold has $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_6\rangle = 0$ exactly, by group theory alone.
- Consequently the B₁g Jahn–Teller distortion is **symmetry-forbidden** in the AFM normal state: $\Gamma_{JT} \not\subset \Gamma_{\mathrm{AFM}} \otimes \Gamma_{\mathrm{AFM}}$.
- Cooper-pair condensation, provided the pairing channel is genuinely **interorbital** (Γ₆↔Γ₇, not Γ₆↔Γ₆ or Γ₇↔Γ₇), builds up a Bogoliubov coherence between the two doublets.
- Only in this paired subspace does rank-2 multipolar order become accessible: $\Gamma_{JT} \subset \Gamma_{\mathrm{pair}} \otimes \Gamma_{\mathrm{pair}}$.
- The B₁g JT distortion therefore emerges as an **induced response of the condensate**, not as a primary instability — superconductivity comes first, the lattice distortion follows.

The three-fold splitting of the $t_{2g}$ shell under SOC actually produces **two** $\Gamma_7$-like doublets once the crystal field is included ($\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}$, see §1). Both are JT-active by the same $\Gamma_6\otimes\Gamma_7\supset B_{1g}$ selection rule (§2), and the code solves **both** cross-doublet sectors self-consistently and on equal footing: every pairing form factor of the basis (§8) carries one complex amplitude for the Γ₆–Γ₇ₐ pair and one for the Γ₆–Γ₇ᵦ pair, and each of them feeds its own anomalous (Gorkov) coherence back into the Hamiltonian as its own B₁g Weiss field. $\Gamma_{7a}$ — identified in `__post_init__` as the doublet with the larger direct $\langle\Gamma_6\vert B_{1g,\mathrm{op}}\vert \Gamma_7\rangle$ matrix element (§1) — is generically the dominant sector, but $\Gamma_{7b}$'s amplitudes are genuine dynamical variables of the SCF fixed point, wired into the 24×24 BdG Hamiltonian, the RPA pairing vertex, the condensation energy and the gap equations identically in structure, differing only in the (self-consistently distinct) Gutzwiller factor, exchange coupling, and Bogoliubov coherence each branch carries.

### Material class and the three viability conditions

The theory targets systems with:

- D₄h point-group symmetry **and** a global inversion center (square-lattice materials: cuprates, pnictides, selected layered transition-metal oxides). Inversion symmetry is what allows the Cooper pair to have a well-defined parity (pure singlet or pure triplet); if it is broken locally, Rashba/Dresselhaus terms mix the two and the clean tensor-product selection rule below no longer applies unambiguously.
- Strong electron correlation with a charge-transfer (ZSA-type) insulating parent state.
- Strong SOC that reorganizes the $t_{2g}$ manifold into $\Gamma_6 \oplus \Gamma_{7a} \oplus \Gamma_{7b}$ Kramers doublets (typical of 4d/5d transition-metal ions, though the framework is agnostic to the microscopic origin of the SOC scale).
- Superexchange-stabilized collinear AFM order in the parent compound, including a finite intra-atomic Hund's coupling in the virtual $d^2$ superexchange intermediate state (§4). The Néel axis is selectable (`M_axis`, §1).
- Finite hole doping ($\delta > 0$), which restores the itinerancy needed for coherent Cooper pairing.

Three conditions must hold simultaneously for the mechanism to operate:

1. **Charge-transfer insulating character** — the ZSA charge-transfer gap and the on-site $U_{dd}$ (both primary inputs, §4) are large enough that genuine Mott physics is in play.
2. **Non-Mott-localized coherence** — the Cooper pairs must actually be mobile; this requires $\delta > 0$ so the Gutzwiller kinetic factor $g_t$ does not collapse to zero.
3. **Moderate AFM order** — AFM correlations must be present (they are what forbids the JT channel in the normal state) but not so strong that spin fluctuations kill superconductivity outright (Stoner criterion: the largest eigenvalue of $\mathrm{diag}(J_c(q))\,\chi_{cc'}$ stays below 1).

### Why the pairing must be interorbital, and why turning on Δ alone is not enough

A tempting but incorrect picture is that the condensate simply mixes $\vert \Gamma_6\rangle$ and $\vert \Gamma_{7}\rangle$ into a coherent superposition $\alpha\vert \Gamma_6\rangle+\beta\vert \Gamma_{7}\rangle$, giving a quadrupole moment linear in the mixing amplitude $\beta\propto\Delta$. This cannot be right: it would make $\langle B_{1g,\mathrm{op}}\rangle$ depend on the arbitrary global $U(1)$ phase of $\Delta$, violating gauge invariance. The correct microscopic statement is that the condensate modifies the **normal** (charge-conserving) density matrix through the Bogoliubov coherence factors: $\langle c^{\dagger}_{6\sigma}c_{7\sigma'}\rangle$ picks up a $v^{*}v \sim \vert \Delta\vert ^2$ contribution, which is explicitly gauge-invariant and appears at *quadratic*, not linear, order in $\Delta$.

For this to happen at all, the pairing must directly connect $\Gamma_6$ and a $\Gamma_7$ doublet — a purely intraorbital singlet (Γ₆–Γ₆ or Γ₇–Γ₇) generates no $\Gamma_6$–$\Gamma_7$ Bogoliubov mixing and leaves the JT channel closed even in the superconducting state. This is exactly the structure built into every pairing operator of the code's basis (§8): all of them are Γ₆↔Γ₇ₐ or Γ₆↔Γ₇ᵦ by construction.

There is a further, more subtle point verified directly in the group-theoretic analysis: **at $Q=0$, even with $\Delta\neq0$, a purely singlet-paired BdG ground state still gives $\langle B_{1g,\mathrm{op}}\rangle \equiv 0$ band-pair by band-pair.** The reason is that $B_{1g,\mathrm{op}}$ in the numerical SOC eigenbasis turns out to be **spin-conserving** ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7}{\uparrow}$), while the singlet pairing operator only ever connects opposite pseudospin sectors; $S_z$ stays block-diagonal in those same sectors. The actual switch is thrown by the *self-consistently induced anomalous coherence* — numerically the same operator as $B_{1g,\mathrm{op}}$ — which is exactly zero at $Q=0$ and becomes nonzero only once **both** $\Delta\neq0$ **and** $Q\neq0$. In the code this anomalous coherence is the complex vector `F67_vec` (one entry per pairing amplitude, §2), fed back into the local Hamiltonian's transverse Weiss field (§9, §19); the loop is genuinely self-consistent rather than a one-way "SC turns on JT."

### The JT distortion as a thermodynamic order parameter, not a dynamical mode

$Q$ is treated as a **macroscopic, thermodynamic order parameter** — physically the amplitude of a flat (dispersionless) optical Einstein phonon — rather than a fluctuating dynamical degree of freedom. Differentiating the free energy with respect to $Q$ gives the equilibrium condition and the softening criterion

$$\lambda_{JT}^{\mathrm{norm}} = \chi_{QQ}/K_{\mathrm{eff}} \qquad (<1\text{ stable},\ =1\text{ onset},\ >1\text{ spontaneous JT}).$$

In the SCF solver this equilibrium is not reached by iterating a damped Newton step on $Q$: at every iteration $Q$ is **profiled out** as a slave variable, $Q^*=\arg\min_Q F_{\mathrm{tot}}(Q)$ with $F_{\mathrm{tot}}=F_{\mathrm{can}}(Q;M,\Delta,F_{67},\mu)$ plus the exact beyond-mean-field plaquette energy (§12), so that first-order jumps and $\pm Q$ symmetry breaking are handled by a global scan rather than a local Newton step. $Q$ is relaxed toward $Q^*$ with its own weight and kept out of the Anderson history of the other order parameters.

A natural question is whether the free energy contains a **linear** coupling $Q\cdot\vert \Delta\vert ^2$ between the lattice and the condensate. The theoretical analysis finds that in the clean D₄h limit this cross-coupling vanishes — an exact but non-generic consequence of the purely interorbital structure of the pairing operators, not a trivial symmetry accident (the naive argument "$\vert \Delta\vert ^2$ is always $A_{1g}$" is incomplete, because different pairing channels need not share one point-group label and their cross terms could in principle transform as $B_{1g}$). The code does **not** impose this cancellation by hand: it is left to the free energy itself. The Δ-Hessian (`compute_dF_dDelta_and_d2F`, §21) is written so that a bilinear term $Q\,\Delta_i^{*}\Delta_j$ would be picked up (inactive channels are fitted with a two-point stencil precisely because such terms make $F$ linear in $\vert\Delta_i\vert$), and the $Q$ scan of §12 would find a finite $Q^*$ in a purely linear-coupling scenario. (If the lattice already sits in D₂h because of a finite static crystal field `Delta_B1g_static` ≠ 0, the cancellation is lifted and a genuine linear coupling appears — see §3.) The upshot: in the clean D₄h limit, SC-triggered JT is a **threshold phenomenon**. The condensate progressively softens the B₁g mode's effective stiffness $K_{\mathrm{eff}}$ until the lattice snaps into a finite distortion.

While $Q$ itself is a static mean field, its *coupling* to the electronic condensate is not treated as instantaneous everywhere in the code: the phonon-mediated piece of the pairing vertex carries an explicit adiabatic retardation factor tied to a physical bare-phonon energy scale derived from the lattice spring constant and an input reduced mass (§14) — a first step beyond the purely static-propagator treatment, though $Q$'s own equation of motion remains a thermodynamic equilibrium condition, not a dynamical one.

### First-order character

Numerically, this transition is expected to be **first-order**, not a simple second-order spin-fluctuation instability: spin fluctuations alone are not sufficient to drive the lattice unstable, and the system tips into the $B_{1g}$-distorted configuration only cooperatively, together with the superconducting condensate. This is reflected in the solver's SCF-dynamics classifier (see "SCF Loop" under [Key Algorithms](#key-algorithms)), in the `__main__`'s explicit three-way free-energy comparison against the normal and $Q$-pinned states (see [Installation & Usage](#installation--usage)), and in the thermodynamic, first-order-aware Tc estimate of §23.

---

## Theoretical Framework

### 1. Local Hilbert Space: SOC + Crystal-Field Diagonalization

The full SOC + D₄h crystal-field Hamiltonian is built and diagonalized explicitly on the 6-dimensional $t_{2g}\otimes\mathrm{spin}$ manifold, directly inside `ModelParams.__post_init__` (via the module-level helper `_t2g_soc_cf_operators`):

```
H = λ_SOC · L·S  +  Δ_tetra · Lz²  +  Delta_B1g_static · (Lx² − Ly²)
```

This yields the Γ₆–Γ₇ₐ splitting `Delta_CF` as a **derived quantity**, never a free input. `Delta_tetra` (negative = tetragonal z-compression) sets the axial crystal field; `Delta_B1g_static` is a static, in-plane crystal-field term with the same $(L_x^2-L_y^2)$ functional form as the dynamical JT operator, logged as `Δ_ip`. Its role is to split the four-dimensional $\Gamma_7$ manifold into two Kramers doublets, $\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}$, which prevents a spurious spontaneous JT instability from the residual $\Gamma_7$ degeneracy while leaving $\Delta_{CF}$ tunable independently of $\lambda_{SOC}$.

**Kramers doublet identification** (`_get_G6_G7a_G7b`) proceeds in four steps:
1. The three Kramers doublets of $H_{SOC}+H_{CF}$ are sorted by the expectation value $\langle L\!\cdot\!S\rangle$; the doublet with the most negative value is assigned $\Gamma_6$ ($j_{\mathrm{eff}}=1/2$-like), and the two remaining candidates are the $\Gamma_7$ pair.
2. Within each of the three doublets, a 2×2 diagonalization of $S_z$ selects the exact $z$-polarized Kramers partners (`up`, `dn`) and their eigenvalues (`sz_up`, `sz_dn`).
3. **Time-reversal gauge.** The partner is then *defined* as $|\!\downarrow\rangle:=\Theta|\!\uparrow\rangle$ (with $\Theta$ the time-reversal matrix `_U_THETA_T2G` on the $t_{2g}\otimes$spin space), so that the pair operator $c^\dagger_{a\uparrow}c^\dagger_{b\downarrow}-c^\dagger_{a\downarrow}c^\dagger_{b\uparrow}$ is a true pseudospin singlet with the same time-reversal convention for every orbital. The code checks that $\Theta|\!\uparrow\rangle$ really is the $S_z$-down state and raises a `RuntimeError` otherwise.
4. Between the two $\Gamma_7$ candidates, the one carrying the **larger direct $\langle\Gamma_6\vert B_{1g,\mathrm{op}}\vert \Gamma_7\rangle$ matrix-element norm** is assigned $\Gamma_{7a}$ (the dominant JT-pairing partner) — this is the defining property, evaluated directly rather than inferred from a proxy. The total moment $\mu_z=\langle L_z+2S_z\rangle$ is computed only as an independent **cross-check** and triggers a warning log if it disagrees with the $B_{1g}$-coupling criterion (`⚠ Γ7a (B1g-coupling-sorted) has smaller |μz| than Γ7b`) — this can happen in strongly mixed CF/SOC regimes, and the log line reports both the $\vert S_z\vert $ and $\mu_z$ values for each candidate alongside the $B_{1g}$-coupling norms that actually decided the assignment.

**All three doublets enter the solver exactly — there is no downfolding, and neither is diagnostic-only.** The resulting basis is $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$, and all six components enter every operator (`B1g_op`, `Eg2_op`, the spin operators, the hopping matrices, and the BdG Hamiltonian itself). $\Gamma_{7b}$ disperses as a genuine band through the same rigorous orbital-projected hopping as $\Gamma_6/\Gamma_{7a}$ (§6, §9) and pairs with $\Gamma_6$ through its own self-consistent amplitudes (§8). Several derived quantities are cached on `ModelParams` at this point:

- `sz_op = [sz6_up, sz6_dn, sz7_up, sz7_dn, sz7b_up, sz7b_dn]` — **exact** $\langle S_z\rangle$ eigenvalues from the doublet diagonalization; used as the spin vertex for the default Néel axis.
- **Néel-axis spin operators.** `S_ops = {'x','y','z'}` are the doublet-projected spin operators in the ordered basis, **block-diagonal per Kramers doublet** (only the intra-doublet 2×2 blocks enter the Weiss field and observables, consistent with the channel-diagonal exchange); the dropped inter-doublet pieces are kept as `S_inter` for diagnostics. For $S_z$ the projection equals `diag(sz_op)` exactly (checked; the exact diagonal is then used). `S_axis_op = S_ops[M_axis]` is the operator along the ordering axis chosen by the primary input **`M_axis`** (`'x' | 'y' | 'z'`; `x, y` ∥ the two metal–ligand bond directions). For `M_axis='z'` every vertex stays diagonal and the cheap diagonal-vector code paths are used; for `'x'`/`'y'` the Weiss field, the staggered/uniform spin vertices of the susceptibilities, the magnetization observables, the incommensurate-nesting scan and the cluster exchange switch to dense 24×24 (resp. 6×6) matrices built from `S_axis_op`.
- `p_7` — the average $\Gamma_7$ (either doublet) orbital-weight admixture in the $\Gamma_6$ eigenvectors (weight of Γ₆ in the $|L_z|=1$ sector $[0,1,4,5]$ subtracted from 1); used by the static admixture-based Gutzwiller estimate (§5).
- `Delta_CF = evals[2] − evals[0]` (Γ₇ₐ–Γ₆ gap, JT-active) and `g7split = evals[4] − evals[2]` (Γ₇ᵦ–Γ₇ₐ internal splitting).
- Orbital-character weights `_w_orb` (3×3: rows Γ6/Γ7a/Γ7b, columns $d_{xz}/d_{yz}/d_{xy}$) and the orbital projectors `Tx_A_xz/_yz/_xy` — used to build both the Q-dependent exchange anisotropy (§6) and the orbital-selective hopping matrices (§6, §9).
- `kappa_doublets` (3 components: Γ6, Γ7a, Γ7b) and `kappa_cross` (2 components: Γ6–Γ7a, Γ6–Γ7b) — the exact-diagonalization superexchange coefficients (§4), used by `exchange_channels()` (§6).
- `U_pp` and `J_H` (the ligand hole–hole repulsion and the absolute Hund's coupling, §4) are computed in `__post_init__` and **are stored** on the instance (`__main__` logs them in the `DERIVED` line).

If `lambda_soc`, `Delta_tetra`, `Delta_B1g_static` or `M_axis` are mutated on a live solver, `RMFT_Solver._full_rebuild()` is the single method that re-runs `p.__post_init__()`, updates `_K_bare` and the bare phonon energy `_omega_0_JT`, rebuilds every orbital operator (`B1g_op`, `B1g_24`, `Eg2_op`, `Eg2_24`, `Sz_nambu`, the per-channel staggered/uniform spin vertices) and resets all transient caches.

### 2. Symmetry Protection of the JT Channel

**Selection rule in a pure doublet.** A rank-$k$ irreducible tensor operator has a nonzero diagonal matrix element in $\Gamma_6$ only if $\Gamma^{(k)}\subset\Gamma_6\otimes\Gamma_6$. Since $\bar D_{4h}$ character theory gives $\Gamma_6\otimes\Gamma_6=\Gamma_7\otimes\Gamma_7=A_{1g}\oplus A_{2g}\oplus E_g$ — containing neither $B_{1g}$ nor $B_{2g}$ — the quadrupole operator has $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_6\rangle=0$ **exactly**. A pure $\Gamma_6$ (or $\Gamma_7$) manifold carries no electric quadrupole moment and does not couple to a $B_{1g}$ lattice shear. Because the collinear AFM state stabilizes only this kind of dipolar (rank-1), spin–orbitally mixed order, the JT channel is symmetry-blocked in the normal state.

**Cross-product opens the channel.** $\Gamma_6\otimes\Gamma_7 = B_{1g}\oplus B_{2g}\oplus E_g$ *does* contain $B_{1g}$, so the **off-diagonal** element $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_{7a}\rangle\neq0$ is allowed — and, by the identical argument, so is $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_{7b}\rangle$, generally with a smaller (but finite) matrix element, since $\Gamma_{7a}$ is defined precisely as the doublet carrying the larger of the two (§1). Both channels are realized by the code's self-consistent pairing structure (§8): realizing either one requires a genuinely interorbital pairing operator and, as emphasized above, requires $Q\neq0$ as well as $\Delta\neq0$ — the self-consistent loop, not a one-shot symmetry argument, is what actually opens the channel.

**Θ symmetry and the global BZ cancellation.** In the collinear AFM state, time reversal $\mathcal T$ is broken, but the combined Shubnikov element $\Theta=\mathcal T\cdot\tau_{AB}$ (time reversal composed with the $A\leftrightarrow B$ sublattice translation) survives; since $\tau_{AB}^2=+1$ and $\mathcal T^2=-1$, $\Theta^2=-1$. This does **not** give pointwise Kramers degeneracy — $\Theta$ maps crystal momentum $k\to -k$, and in the magnetic Brillouin zone $k$ and $-k$ are generally inequivalent — but it does impose a **global** constraint on Brillouin-zone integrals. The spin–quadrupole susceptibility integrand is odd under $\Theta$ at $\Delta=0$, so the full-BZ Lindhard sum cancels identically:

$$\chi_{SQ}(q) = \int_{\mathrm{BZ}} d^2k\; \mathcal I_{SQ}(k,q) = 0 \qquad (\Delta=0).$$

This BZ-wide cancellation — not a pointwise Kramers argument — is a structural property of every $\Delta=0$ evaluation of `_chi0_channels_raw`/`_rpa_closure` (§14, built on the module-level `_lehmann_kernel`): for the collinear $S_z$ order the spin–JT cross entries of the returned $4\times4$ matrix vanish at $Q=0,\Delta=0$ by the odd-in-$k$ structure of the normal-state Lindhard kernel, not by a runtime numerical check.

In the superconducting state the full Nambu–Lehmann sum includes the anomalous (Gorkov) sectors, whose Bogoliubov coherence factors can break the odd-$\Theta$ structure — but as detailed above, this alone is not sufficient at $Q=0$: purely singlet, interorbital pairing keeps $S_z$ and $B_{1g,\mathrm{op}}$ acting within disjoint pseudospin sectors band-pair by band-pair, so the product stays exactly zero until the self-consistently generated $Q\neq0$ genuinely couples the two sectors.

**The anomalous amplitudes, explicitly.** The quantities that actually cross the symmetry boundary are the Γ₆–Γ₇ anomalous (Gorkov) singlet amplitudes, computed directly from the converged BdG eigensystem by the module-level `_compute_F67_pair_amplitudes`, one per entry of `Delta_vec`. For the Γ₇ₐ branch the elementary pair correlator is

```math
F^{(7a)}_{\mathrm{raw}} = \sum_k w_k\,(1-2f_n)\,\big[u_{6\uparrow}v^{*}_{7a\downarrow} - u_{6\downarrow}v^{*}_{7a\uparrow}\big]/4
```

(summed over both sublattices for the same-sublattice correlator and over A–B/B–A for the inter-sublattice one), and an identical expression with $7a\to7b$ gives the Γ₇ᵦ branch. For every basis channel $i$ the amplitude is weighted by that channel's form factor, $F_i=\sum_k w_k\,\varphi_i(k)\,(1-2f_n)\,[\ldots]/4$ (with $\varphi_s\equiv1$ for the on-site channel; on-site and same-sublattice channels use the same-sublattice correlator, nearest-neighbour bond channels the inter-sublattice one). Two independent operations turn the raw complex BZ sum into the mean-field quantity fed back into the local Hamiltonian: (i) a **global gauge fixing** — every component is multiplied by $e^{-i\varphi_{\mathrm{ref}}}$, with $\varphi_{\mathrm{ref}}$ the phase of the *largest* $\vert\Delta_i\vert$ (zero if $\Delta=0$), so `F67_vec` is a **complex**, gauge-invariant vector that keeps the relative phases of the channels (an $s+id$ relative phase, for instance, survives), and (ii) Gutzwiller weighting by the **same per-channel vector the pairing vertex uses** (`_gutzwiller_irrep_vector`, §5: `g_Delta_s` for the on-site channel; `g_Delta_ad` / `g_Delta_bd` for the nearest-neighbour A–B bond channels of Γ₇ₐ / Γ₇ᵦ; `g_t` for the same-sublattice bond channels). All entries vanish whenever $\Delta=0$ (exact D₄h selection rule).

The scalar **selection ratio** logged in `__main__` ("Irrep R") is `selection_ratio = clip(Σ_i|Δ_i| / Δ_CF, 0, 1) · |F67_vec[0]|` — the total gap amplitude relative to $\Delta_{CF}$, weighted by the on-site Γ₇ₐ anomalous amplitude. It is used purely as a diagnostic (the SCF-RES summary compares it to `_JT_ACT_THR`; the `__main__` "JT ALLOWED/BLOCKED" line uses a literal threshold of 0.02) and as a gate for the mild mixing-rate boost in the SCF loop, never as an input to the fixed point itself.

### 3. The B₁g Operator and the D₄h/D₂h Crossover

The B₁g phonon coupling operator is constructed from the same $t_{2g}$ operators used for $H_{CF}$ and projected into the full 6-dimensional $\Gamma_6\oplus\Gamma_{7a}\oplus\Gamma_{7b}$ subspace — no downfolding:

```
B1g_op = real(U6† · (Lx² − Ly²)_t2g · U6)     # 6×6, real, Hermitian
```

where `U6` is the 6×6 change-of-basis matrix from the bare $t_{2g}\otimes\mathrm{spin}$ manifold to the Kramers-doublet (SOC+CF) eigenbasis built in §1.

- **D₄h (`Delta_B1g_static = 0`):** the $\Gamma_6$–$\Gamma_6$ and $\Gamma_{7a}$–$\Gamma_{7a}$ (and $\Gamma_{7b}$–$\Gamma_{7b}$) diagonal blocks of `B1g_op` vanish exactly and it is purely off-diagonal and **spin-conserving** — it connects $\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\uparrow}$ and $\Gamma_6\leftrightarrow\Gamma_{7b}$, generally with different weight for each. Hence $\langle B_{1g,\mathrm{op}}\rangle=0$ in *any* normal state.
- **D₂h (`Delta_B1g_static ≠ 0`):** the operator picks up real diagonal ($A_{1g}$-like) components that renormalize $\Delta_{CF}$ but do not by themselves drive JT, plus additional off-diagonal weight. The fraction of the operator that remains purely off-diagonal — `b1g_ratio = ‖B1g_offdiag‖/‖diag(B1g_op)‖`, converted to a weight $b_{1g,\mathrm{weight}} = b_{1g,\mathrm{ratio}}/(1+b_{1g,\mathrm{ratio}})$ — is computed on demand (inside the verbose branch of `_scf_kick`) rather than cached as a permanent `ModelParams` field; it quantifies how much of the operator remains genuinely SC-triggered versus normal-state-active. `b1g_weight ≈ 1` in the clean D₄h limit.

The 24×24 Nambu extension `B1g_24` (built in `_rebuild_orbital_operators`) carries the hole block as $-B_{1g,\mathrm{op}}^{T}$ (real, so $=-B_{1g,\mathrm{op}}$), consistent with BdG particle–hole symmetry; every JT coupling term in the Hamiltonian is written `H += β²(k) · g_JT · Q · B1g_op` rather than a hand-built matrix (§9). `Eg2_24` mirrors this construction for the Eg,2 operator (§7).

### 4. Multi-Orbital Cluster Superexchange and the Weiss Field

AFM order originates from virtual $p$–$d$ hopping (ZSA charge-transfer superexchange), not from a bare Stoner Fermi-surface instability, and the superexchange scale is obtained from an **exact diagonalization of the full multi-orbital $d$–$p$–$d$ two-hole cluster**, resolved separately for all three Kramers doublets and both cross-doublet channels rather than a single scalar.

`_multiorbital_2hole_hamiltonian` builds the full two-hole problem on the $\binom{18}{2}=153$-dimensional Fock space of an 18-spin-orbital, 3-site chain (metal $d_1$ – ligand $p$ – metal $d_2$, each site carrying the full 3-orbital $\times$ 2-spin $t_{2g}$ manifold, SOC and crystal field included on both metal sites; fermion signs handled by `_apply_hop`). The ligand is modeled as **three independent modes**, one per $L_z$ channel, each hybridizing with equal bare $t_{pd}$ — this preserves $J_z=L_z+S_z$ through the SOC mixing and matches the orbital-blind bare hopping $T_x(Q{=}0)=t_0 I_6$ used elsewhere in the model. The two-hole interaction is the density–density Kanamori form,

```
same channel, opposite spin   → U_dd
different channel, opposite spin → U_dd − 2·J_H
different channel, same spin     → U_dd − 3·J_H
```

so that `J_H_ratio = 0` recovers the original single-parameter interaction exactly, and a finite Hund's coupling $J_H = $ `J_H_ratio` $\times\,U_{dd}$ lowers the effective two-hole triplet energy relative to the naive $U_{dd}$-only estimate, as it must physically.

From the resulting exact spectrum, two downfolding routines extract the physical exchange couplings via a rigorous **Löwdin/Bloch–des Cloizeaux quasi-degenerate projection** onto the relevant two-ion reference subspace — not a leading-order perturbative expansion in $t_{pd}$:

- `_intra_doublet_kappa` computes, for each of the three doublets independently, the **energy-dependent** (Feshbach) effective Hamiltonian on the reference subspace $\{\vert \text{doublet}\,a\rangle_1\otimes\vert \text{doublet}\,b\rangle_2\}$ via `_lowdin_branch_energies` (an exact, seeded fixed-point search for every branch energy — not a single shared trial energy), reads off the isotropic (Heisenberg) coupling from the triplet–singlet splitting in the normalized pseudospin basis, and **rescales by the doublet's own effective spin** $s_{\mathrm{eff}}$ (from its exact $\langle S_z\rangle$, §1): $\kappa_c=(E_T-E_{S_0})/(4\,s_{\mathrm{eff},c}^2)$. This rescaling matters: without it, $J_{A1g}$ would be over-strengthened by up to two orders of magnitude for $\Gamma_{7b}$ in the strong-SOC regime, where $s_{\mathrm{eff}}$ can be $\ll 1/2$. The denominator is now guarded by a **floor on the effective spin**, $s_{\mathrm{safe}}=\max(s_{\mathrm{eff}},$ `_S_EFF_FLOOR`$=0.15)$, because a Kramers doublet with $s_{\mathrm{eff}}\to0$ (the spin-switch regime $\Delta_{\mathrm{tetra}}/\lambda_{SOC}\gtrsim0.5$, where Γ₆ turns mostly $xy$-like) would make the Wigner–Eckart factor diverge; a warning is logged whenever the floor is active. A non-Heisenberg (Ising/DM-like) component is separately diagnosed via a Pauli-tensor decomposition of the effective Hamiltonian and flagged with a warning if it exceeds `_ANISO_WARN_TRESH`, though only the isotropic part is used downstream.
- `_cross_doublet_kappas` extracts the $\Gamma_6$–$\Gamma_{7a}$ and $\Gamma_6$–$\Gamma_{7b}$ cross-exchange coefficients $J_{B1g}$ from the same exact spectrum, this time via the **energy-independent** Bloch–des Cloizeaux effective Hamiltonian (found ≈7% more accurate for this cluster than the energy-dependent Feshbach construction used for the intra-doublet case) on the reference subspace $\{\vert \Gamma_6\rangle_1\vert \Gamma_7\rangle_2\}\cup\{\vert \Gamma_7\rangle_1\vert \Gamma_6\rangle_2\}$, converting the raw transition-block amplitude to the operator coefficient that actually multiplies $B_{1g,\mathrm{op}}(i)\cdot B_{1g,\mathrm{op}}(j)$ downstream. The routine returns zero (with a warning) if the reference subspace is near-singular, the projected overlap is singular, or the $\Gamma_6$↔$\Gamma_7$ block carries less than `_B1G_BLOCK_MIN_FRAC` (now $0.02$) of the operator's total weight — all are ill-conditioned regimes for the selection-rule-suppressed channel. A soft warning is issued when the exact eigenstates are not well isolated in the reference space.

`ModelParams.__post_init__` stores the results as `kappa_doublets` (3-vector: Γ6, Γ7a, Γ7b) and `kappa_cross` (2-vector: Γ6–Γ7a, Γ6–Γ7b); `exchange_channels()` (§6) turns these, together with the orbital-character weights of §1, into the actual $Q$-dependent exchange tensors used everywhere else. `U_dd` is a **primary input** (the on-site Hubbard repulsion); `U_pp`, the ligand (2p) hole–hole repulsion, is obtained from a second-order downfolding estimate that also depends on the `Upp_ratio_bare` primary input (the bare, unhybridized $U_{pp}/U_{dd}$ ratio) and the coordination number:

```
U_pp = (Upp_ratio_bare · U_dd) / (1 + (Z/2) · t_pd² / (Delta_CT · (Delta_CT + U_dd)))
```

with `Z/2` the ligand coordination number (each ligand typically bridges two metal sites). `hybrid_scale` does **not** enter this calculation; its role is instead the $k$-dependent quasiparticle downfolding weight $\beta^2(k)$ used in the BdG Hamiltonian and the cluster embedding (§6, §9).

The effective AFM Weiss field entering the BdG Hamiltonian (built by `build_local_hamiltonian_for_bdg`) is block-diagonal in the full 6-component $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$ basis. For the default Néel axis `M_axis='z'` it is diagonal,

```
H_Weiss = − sign_M · Z_afm_eff · diag( J_A1g[α] · M_c · sz_op[α] )          (channel c matching orbital α)
```

and for `M_axis = 'x' | 'y'` the diagonal $\mathrm{sz\_op}$ is replaced by the doublet-projected spin operator along that axis, $-\,\mathrm{sign}_M\,Z_{\mathrm{afm,eff}}\;\mathrm{diag}(J_c M_c)\,S_{\mathrm{axis}}$ (which stays Hermitian because $J_cM_c$ is uniform inside each Kramers pair and $S_{\mathrm{axis}}$ is block-diagonal per doublet). Here `J_A1g_diag` is the longitudinal (spin-preserving) exchange tensor from `exchange_channels()` (§6), $M_c$ are the three channel-resolved staggered magnetizations (carried and converged independently — with one $M_c$ per doublet the Hellmann–Feynman identity is exact because $J_c$ is uniform inside a Kramers pair), and the doping renormalization is carried by the itinerant carrier density `n_kspace` returned from the chemical-potential solve.

### 5. Gutzwiller Renormalization and the Mott Guard

`t_pd` is the single primary hopping input; the effective $dd$ hopping `t0 = t_pd² / Delta_CT` is always derived, never set directly.

```
g_t       = 2δ/(1+δ)         # kinetic-energy suppression → 0 at half-filling
g_J       = 4/(1+δ)²         # exchange enhancement → 4 at half-filling
```

`ModelParams.get_gutzwiller_factors` also returns a **static**, admixture-based estimate `g_Delta_s = g_t`, `g_Delta_ad = g_t + (g_J−g_t)·p_7` (interpolated by the doping-independent $\Gamma_7$ admixture `p_7` of §1) — used only for the pre-SCF bookkeeping. Inside the SCF loop itself, `RMFT_Solver.estimate_gutzwiller_factors_occupation_based(M, Q, n_kspace, mu, g_t, g_J)` replaces this with a **dynamic**, occupation-based estimate, re-evaluated every iteration (and at every probe point of every finite-difference free-energy evaluation) from the actual normal-state ($\Delta=0$) BdG orbital densities $n_6, n_{7a}, n_{7b}$ on sublattice A:

```
g_Delta_s  = g_t
g_Delta_ad = g_t + (g_J − g_t) · clip(2·n7a / n_total, 0, 1)
g_Delta_bd = g_t + (g_J − g_t) · clip(2·n7b / n_total, 0, 1)
```

so the bond (inter-site) renormalization of **each** cross-doublet branch tracks that branch's own actual orbital occupation, rather than a single fixed admixture shared by both. `g_Delta_s` (the on-site, kinetic-origin channel) is common to both branches.

Because the pairing basis now contains many channels, the per-channel Gutzwiller vector used in the pairing-strength matrices and the condensation energy is assembled by `_gutzwiller_irrep_vector(g_Delta_s, g_Delta_bond, g_Delta_nnn)`: the on-site channel gets `g_Delta_s`, every nearest-neighbour (inter-sublattice) bond channel gets the branch's `g_Delta_ad` / `g_Delta_bd`, and every same-sublattice (NNN / 3NN) bond channel gets `g_t` — the renormalization depends on the real-space range of the pair, not on which symmetry channel a bond amplitude belongs to. (The anomalous amplitudes `F67_vec` of §2 use the very same vector, so that $F_{67}$, the gap equation $V\sqrt{g_i/g_j}$, the condensation energy and the Hellmann–Feynman identity are mutually consistent; earlier versions used `g_Delta_ad` / `g_Delta_bd` also for the same-sublattice channels.)

The superexchange is always computed from the **bare** hopping `t_pd` (through the cluster diagonalization of §4), then multiplied by `g_J` — computing it from Gutzwiller-*renormalized* bands would double-count the suppression (a spurious `g_t²` factor), since `g_t` renormalizes only the kinetic energy while `g_J` renormalizes only the exchange; the two are orthogonal channels in this RMFT scheme. The lattice-summed Weiss-field/superexchange scale is $J_{\mathrm{eff}} = \sum_c M_c^2\, J_c(q_{AFM})\big/\sum_c M_c^2$, an **$M^2$-weighted** average over the three doublet channels' own $J_c(q_{AFM})=Z_{\mathrm{afm,eff}}\,J_{A1g}[c]$ (§6), evaluated by `_exchange_J_q` through the helper `_m2_weighted_J`, which falls back to the Γ₆ value when $\sum_cM_c^2$ vanishes or is not finite (at $M=0$ the weights are all zero and a plain weighted average would fail).

A **Mott guard** acts on the post-SCF result: if `g_t < _G_T_COHERENCE_MIN = 0.10` (i.e. $\delta \lesssim 0.053$) the Gutzwiller factor signals that the Zhang–Rice-singlet band is no longer coherent enough to support a physical SC gap; the same guard fires on a nodal coherence length $\xi_{\mathrm{nodal}}/a < 1$ (BEC-side breakdown of BdG mean-field validity). In either case the result is flagged `mott_suspect`, **all 84 gap amplitudes are zeroed** and `converged` is set to `False`, rather than returning a spuriously converged superconducting state. `_G_T_COHERENCE_MIN` is also used by `__main__` for its doping floor and for the Tc-viability test.

### 6. B₁g Jahn–Teller Distortion, Orbital-Selective Hopping, and Further-Neighbor Terms

The B₁g mode breaks the $x$–$y$ symmetry of the square lattice through an exponential (Harrison-type) hopping law:

```
tx(Q) = t0 · exp(+Q / lambda_hop)      # elongation along x → shorter bond → larger hopping
ty(Q) = t0 · exp(−Q / lambda_hop)      # compression along y → longer bond → smaller hopping
K_eff = K_lattice + ∂²F_el/∂Q²  (+ capped cluster correction)
```

`K_lattice` is the bare phonon spring constant (primary input, never mutated; cached immutably as `_K_bare`); the second-derivative term is the electronic/exchange contribution to the stiffness (§11), negative when the condensate softens the mode.

Unlike a simple scalar $t(Q)$ applied uniformly across all orbitals, the inter-sublattice hopping is **orbital-selective**: `hopping_matrices(Q)` builds rigorous $6\times6$ matrices

```
T_x(Q) = t_x(Q)·A_xz + t_y(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy
T_y(Q) = t_y(Q)·A_xz + t_x(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy   (x↔y swap)
```

from the exact $d_{xz}/d_{yz}/d_{xy}$ orbital-character projectors (`Tx_A_xz`, `Tx_A_yz`, `Tx_A_xy`, built once in `__post_init__` and satisfying $A_{xz}+A_{yz}+A_{xy}=I_6$, checked with a warning), so that $\Gamma_6$, $\Gamma_{7a}$, and $\Gamma_{7b}$ each disperse according to their own $d_{xz}/d_{yz}/d_{xy}$ admixture rather than sharing one isotropic band. `hopping_matrices_dQ(Q)` gives the companion exact analytic $\partial T_{x,y}/\partial Q$.

The full multipolar exchange tensor is likewise Q-dependent through both the overall B₁g channel opening and an orbital-selective asymmetry. With $j_{xz}=e^{+2Q/\lambda_{hop}}$, $j_{yz}=e^{-2Q/\lambda_{hop}}$, $j_{xy}=(j_{xz}+j_{yz})/2$ (exchange scales as $t^2$), `exchange_channels(Q, g_J)` returns a 3-tuple:

```
J_A1g_diag[c]        = g_J · κ_c · Σ_orb w_c,orb · j_orb(Q)                           (longitudinal, even; 6-component, Kramers-doubled)
J_B1g_scalar_7a, _7b = g_J · κ_cross,c · [ j_xz(Q)·√(w6,xz·w7c,xz) − j_yz(Q)·√(w6,yz·w7c,yz) ]     (transverse, odd; ONE scalar per cross-doublet branch c = 7a, 7b)
```

i.e. **two independent transverse couplings**, $J_{B1g}^{(7a)}(Q)$ and $J_{B1g}^{(7b)}(Q)$ — one for each paired branch (§8) — each built from its own `kappa_cross` entry (§4) and its own $\Gamma_6$–$\Gamma_7$ geometric-mean orbital weight over the $xz/yz$ orbitals ($xy$ is $A_{1g}$-like and cannot enter the $B_{1g}$ channel). Both vanish at $Q=0$ for equal $xz/yz$ weights; with `Delta_B1g_static` $\neq 0$ a residual static B₁g exchange survives at $Q=0$ in both channels.

**Momentum dependence.** `_exchange_J_q(J_A1g_diag, qx, qy)` evaluates the longitudinal exchange at arbitrary $q$ via an NN + 2nd-NN + 3rd-NN sum tied to the same single-bond value:

```
J_c(q) = J_A1g_diag[c] · [ −2·(cos qx + cos qy) − 4·(J2/J1)·cos qx·cos qy − 2·(J3/J1)·(cos 2qx + cos 2qy) ]
```

with $J_2/J_1=(t'/t)^2$, $J_3/J_1=(t''/t)^2$; at $q=(\pi,\pi)$ this gives $J_c(q_{AFM}) = Z_{\mathrm{afm,eff}}\cdot J_{A1g,\mathrm{diag}}[c]$, the AFM-peaked value that enters the Weiss field (§4), the mean-field double-counting correction of the free energy (§18) and the RPA denominator (§14); at $q=0$ it is negative (the uniform channel is exchange-suppressed).

**Further-neighbor hopping.** Two additional primary inputs, `t_prime_ratio` and `t_dprime_ratio`, set a same-sublattice, diagonal dispersion on top of the orbital-selective nearest-neighbor terms above:

```
disp_nnn(k) = −4·g_t·t_prime·cos(kx)·cos(ky) − 2·g_t·t_dprime·[cos(2kx)+cos(2ky)]
t_prime  = t_prime_ratio  · t0     # 2nd-neighbor, diagonal (1,1)-type hopping
t_dprime = t_dprime_ratio · t0     # 3rd-neighbor, axial (2,0)-type hopping
```

added as `disp_nnn(k)·I₆` to each sublattice's diagonal block (2nd/3rd-neighbor bonds stay within one checkerboard sublattice, unlike the nearest-neighbor $T_{x,y}$ terms, which connect A↔B). The same ratios set $J_2/J_1$, $J_3/J_1$ above and the frustration-reduced coordination factor $Z_{\mathrm{afm,eff}}=Z\cdot(1-J_2/J_1-J_3/J_1)$.

### 7. The Eg,2 Phonon Channel

Alongside the B₁g mode, the model carries an independent second vibronic channel of Eg,2 symmetry, built from the operator $L_yL_z+L_zL_y$ and projected into the same full $\Gamma_6\oplus\Gamma_{7a}\oplus\Gamma_{7b}$ subspace exactly like `B1g_op`:

```
Eg2_op  = U6† · (Ly·Lz + Lz·Ly)_t2g · U6      # 6×6, Hermitian (complex in general)
```

with its own coupling constant `g_Eg2` (eV/Å), bare stiffness `K_lattice_Eg2` (eV/Å²), and a distortion amplitude `Q_Eg2`, entering the BdG Hamiltonian (`H += g_Eg2·Q_Eg2·Eg2_op`, uniformly in $k$, i.e. a spatially uniform, $q=0$ structural order parameter, **without** the $\beta^2(k)$ weight), the free energy (elastic term $\tfrac12K_{Eg2}Q_{Eg2}^2$) and the Hessian via `Eg2_24` (the 24×24 Nambu lift, mirroring `B1g_24`) and `Eg2_expectation()`. Unlike `B1g_op`, `Eg2_op` connects Kramers partners with an actual **spin-flip** structure ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7}{\downarrow}$) rather than the spin-conserving structure of `B1g_op`.

**In the current code `Q_Eg2` is an input of `solve_self_consistent` that is held fixed (default 0) — it is not iterated.** The channel is therefore wired through the Hamiltonian, the free energy and the observables, but not (yet) a self-consistent degree of freedom; no exchange-driven Eg,2 stiffness correction and no B₁g–Eg,2 cross-rigidity are computed (the elastic term uses the bare `K_lattice_Eg2`). What *is* evaluated is a **post-convergence Eg,2 stability diagnostic** (`SCF-EG2` log line): the Eg,2 susceptibility $\chi_{Eg2,Eg2}$ from `_chi_QQ_matrix_elements(..., return_matrix=True)` (a 9-point mixed stencil of the same grand potential $\Omega(Q,Q_{Eg2})$, returning the full $2\times2$ B₁g/Eg,2 matrix), the stability factor $G_{44}=1-\chi_{Eg2,Eg2}\,g_{Eg2}^2/K_{Eg2}$ (`G44_Eg2_stable` in the result), and the expectation $\langle Eg2\rangle$ as a sanity check (it should vanish for the converged B₁g-only state).

### 8. The Pairing Basis: a Symmetry-Classified Harmonic Library, Both Γ₇ Branches

The pairing sector is no longer a fixed set of four amplitudes. It is defined by **one registry**, `_HARMONIC_LIBRARY`, in which every entry is a lattice-harmonic form factor $\varphi_i(k)$ with a declared D₄h label, a bond *kind*, and a parity under $k\to-k$:

```
'name': (D4h irrep, kind, parity, φ(kx, ky), scale)
```

Every library entry is an **active pairing channel** — there is no filter and no hand-selected subset (`_PAIR_CHANNELS`). Currently the library holds **42 harmonics** (`_N_IRREP = 42`), grouped by the kind of bond they live on:

| Kind | Bond / BdG block | $T_Q$ parity | Channels (D₄h label) |
|---|---|---|---|
| `onsite` | same sublattice (A–A / B–B), $k$-independent | even | `s` (A₁g) |
| `nn_AB` | inter-sublattice (A–B / B–A), odd $\lvert\delta x\rvert+\lvert\delta y\rvert$ | odd | `es`, `es3` (A₁g); `gA2`, `gA3` (A₂g); `d`, `d3` (B₁g); `gB2`, `gB3` (B₂g); `px, px4…px8`, `py, py4…py8` (E_u) |
| `nnn_diag` / `nnn_axis` | same sublattice, even $\lvert\delta x\rvert+\lvert\delta y\rvert$ | even | `es2`, `es4`, `cc`, `cc2`, `cc3` (A₁g); `d2` (B₁g); `g, g2…g5` (B₂g); `px2, px3, px9, pxA, pxB`, `py2, py3, py9, pyA, pyB` (E_u) |

(`d` is exactly the original $\varphi_d=\cos k_x-\cos k_y$, `es` the extended-$s$ $\cos k_x+\cos k_y$, `g` the $\sin k_x\sin k_y$ $d_{xy}$-type channel, `px/py` $\sin k_x$/$\sin k_y$.) Each channel carries **two complex amplitudes**, one for the Γ₆–Γ₇ₐ pair and one for the Γ₆–Γ₇ᵦ pair, packed into one flat vector

```
Delta_vec = [ Δ_s7a, Δ_es7a, …, Δ_pyB7a | Δ_s7b, Δ_es7b, …, Δ_pyB7b ]     (_N_DELTA = 2·_N_IRREP = 84 complex)
```

with names in `_DELTA_NAMES`, the index lookup `_DELTA_IDX['d7a']`, and the branch index arrays `_IDX_7A_ARR`/`_IDX_7B_ARR` (canonical `_IRREP_NAMES` order). `F67_vec` has the same layout. The original four amplitudes survive as the entries `s7a, d7a, s7b, d7b`; the headline result entries `Delta_s`, `Delta_d` are $\vert\Delta_{s,7a}\vert$ and $\vert\Delta_{d,7a}\vert$, and `Delta_irrep` gives $\vert\Delta_{7a}\vert$ for every channel. The legacy 2- and 4-component forms are **no longer accepted** by `VectorizedBdG._build_H_stack`.

**Symmetry labels are measured, not assigned.** The magnetic normal state is invariant under $G=\{E,T_Q\}\times D_{2h}$ (abelian, order 8; $T_Q$: $k\to k+(\pi,\pi)$; C₄ and diagonal mirrors are *not* in $G$ once $M\neq0$ or $Q\neq0$, so e.g. `es` (A₁g) and `d` (B₁g) share the same $G$-character). Every gap function of definite symmetry is a one-dimensional character $\chi=(t,m_x,m_y)\in\{\pm1\}^3$ of $G$. `_irrep_character` **measures** this character of each form factor numerically from $\varphi(g\cdot k)/\varphi(k)$ at generic points and raises an error if a form factor is not a 1-D character; `_check_registry()` runs at import time and verifies unique names, that the declared parity equals the measured $m_xm_y$, and that the $T_Q$ parity is consistent with the bond kind (A–B bonds ⇔ $T_Q$-odd, same-sublattice bonds ⇔ $T_Q$-even). The 8 characters define the **sectors** (`_G_OPS`, `_SECTOR_MEMBERS`): the channels whose character matches. Sectors with no member in the basis are *uncovered* — a "basis gap", reported (together with the library harmonics that would close it) by `_sector_report_lines`. Adding a channel therefore means adding one library line; every downstream object (`Delta_vec` layout, BdG blocks, Weiss fields, kernels, diagnostics) follows automatically.

**How each kind enters the 24×24 BdG pairing block** (all use the same interorbital singlet pattern $\Gamma_6{\uparrow}\Gamma_{7}{\downarrow}-\Gamma_6{\downarrow}\Gamma_{7}{\uparrow}$ for both branches):

- **On-site (`s`)**: $D_{on}[6{\uparrow},7a{\downarrow}]=\Delta_{s,7a}$, $D_{on}[6{\downarrow},7a{\uparrow}]=-\Delta_{s,7a}$ (and the same with $7b$), in the A–A and B–B blocks.
- **Even-parity inter-sublattice bonds** (`es, es3, gA2, gA3, d, d3, gB2, gB3`): $\sum_i\varphi_i(k)\Delta_i$ placed in the A–B and B–A pairing blocks, then **antisymmetrized**, $\Delta_{ij}(k)=-\Delta_{ji}(k)$.
- **Even-parity same-sublattice bonds** (`es2, es4, cc*, d2, g*`): same pattern in the A–A and B–B blocks with the $k$-dependent form factor, identical on both sublattices ($T_Q$-even), antisymmetrized.
- **Odd-parity channels** (all `px*`, `py*`, in both inter- and same-sublattice versions): the form factor is odd in $k$, so the index structure must instead be **symmetric**, $\Delta_{ij}=+\Delta_{ji}$ — these are filled separately (`Delta_full_odd`) and *never* antisymmetrized, in accordance with fermion antisymmetry $\Delta_{ij}(k)=-\Delta_{ji}(-k)$.

All branches feed into the **same** gap-equation infrastructure (§21) and the **same** FS-projected pairing matrix $V_{\mathrm{irrep}}$ (§15) — there is only one momentum-space pairing vertex — but each branch converts that shared vertex into its own gap amplitudes using its own Gutzwiller vector (`g_Delta_ad` vs. `g_Delta_bd`, §5) and its own Bogoliubov anomalous amplitudes (`F67_vec[_IDX_7A_ARR]` vs. `F67_vec[_IDX_7B_ARR]`, §2), which differ because $\Gamma_{7a}$ and $\Gamma_{7b}$ sit at different energies ($\Delta_{CF}$ vs. $\Delta_{CF}+g7split$) and generally carry different direct $B_{1g}$ coupling to $\Gamma_6$ (§1). Concretely, for each branch $b\in\{7a,7b\}$, inside `compute_gap_eq_vectorized`:

```
Δ_raw[b, i] = Σ_j V_irrep[i, j] · √(g_i / g_j) · F67_raw,g[b, j]          i, j ∈ _IRREP_NAMES
```

which reduces term-for-term to the original $(V_s,V_d,V_{sd})$ formulas of the two-channel code when only `s` and `d` are populated.

**Which quantities are still branch-specific headline numbers.** $\Gamma_{7a}$ is generically the dominant sector (it couples more strongly to essentially everything downstream); `Delta_s`, `Delta_d` and `selection_ratio` are built from $\Gamma_{7a}$ entries, and the Anderson acceleration tracks $(\vert\Delta_{s,7a}\vert,\vert\Delta_{d,7a}\vert)$ as its primary block (§ Key Algorithms). By contrast the **coherence-length and spectral-gap diagnostics** treat the two branches symmetrically: the order-parameter magnitude on the Fermi surface is $\vert\Delta(k)\vert=\sqrt{\vert\Phi(k)\cdot\Delta_{7a}\vert^2+\vert\Phi(k)\cdot\Delta_{7b}\vert^2}$ — channels add **coherently within a branch** (their relative phases matter: $s+id$ is nodeless) and the two branches add **in quadrature** (they pair $\Gamma_6$ with orthogonal doublets). The full 84-component `Delta_vec` and `F67_vec` are always in the result dictionary.

---

### 9. The 24×24 BdG Hamiltonian (Doubled Unit Cell, Full 3-Doublet, 84-Amplitude Pairing)

Nambu basis $\Psi=[\text{Particle}_A(6),\ \text{Particle}_B(6),\ \text{Hole}_A(6),\ \text{Hole}_B(6)]$, each block ordered $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T_AB(k)    │  D(k)  (A–A,A–B,    │   ← Particle sector
      │  T_AB†(k)  H_B     │         B–A,B–B)    │
      ├────────────────────┼─────────────────────┤
      │  D(k)†             │  −H_A*   −T_AB*      │   ← Hole sector
      │                    │  −T_AB†* −H_B*       │
      └────────────────────┴─────────────────────┘
```

`H_A`, `H_B` are the $6\times6$ local (AFM Weiss field + crystal field + JT + $t',t''$ diagonal dispersion + same-sublattice transverse Weiss field) sublattice Hamiltonians; $T_{AB}(k)=-2g_t[\cos k_x\,T_x(Q)+\cos k_y\,T_y(Q)]$ is the **orbital-selective** inter-sublattice hopping block of §6 (not a scalar times the identity), plus the inter-sublattice transverse-Weiss-field contribution described below. `D(k)` is the $12\times12$ particle–hole pairing block assembled from the 84 amplitudes exactly as described in §8 (on-site and same-sublattice channels in the diagonal sublattice blocks, nearest-neighbour bond channels in the off-diagonal sublattice blocks, even channels antisymmetrized, odd channels symmetric); the particle–hole off-diagonal blocks use the **transposed** (not Hermitian-conjugate) pairing operator, consistent with BdG particle–hole symmetry.

**The JT coupling itself is $k$-dependent**, not a rigid on-site term: both the JT coupling and the on-site anomalous Weiss field are modulated by the same $k$-dependent quasiparticle spectral weight $\beta^2(k)$ (`beta_k` in the code) used for the charge-transfer downfolding:

```
β²(k) = sigmoid[ k_s·(1 − hybrid_scale·[t̄·(cos kx+cos ky) + δt·(cos kx−cos ky)]/Delta_CT − 0.5) ]
        t̄ = (tx+ty)/2,  δt = (tx−ty)/2,  k_s=10                # wave_function_weight(tx, ty, kx, ky)

H += β²(k) · g_JT · Q · B1g_op                                                         # k-weighted JT coupling
H_TRW (on-site, s)        += β²(k) · Z · [ F67a_s·J_B1g^(7a)·B1g_offdiag_{6,7a} + F67b_s·J_B1g^(7b)·B1g_offdiag_{6,7b} ]
H_TRW (NN bond i)         += s_i · φ_i(k) · [ F67a_i·J_B1g^(7a)·B1g_offdiag_{6,7a} + F67b_i·J_B1g^(7b)·B1g_offdiag_{6,7b} ]    (A–B block, every nn_AB channel)
H_TRW (same-sublattice i) += s_i · φ_i(k) · [ … same structure … ]                                                              (A–A / B–B blocks, every nnn_* channel; sign − on A, + on B)
s_i = trw_irrep_scale[i]                                                                                              (bond-class weight of channel i, below)
```

so the transverse Weiss field is fed back **channel by channel** from the complex anomalous amplitudes `F67_vec`. **Bond-class weights.** The B₁g⊗B₁g exchange on a bond $\delta$ scales like $(t_\delta/t)^2$, the same bridge counting that gives $J_2/J_1=(t'/t)^2$ and $J_3/J_1=(t''/t)^2$ (§6); a pairing channel of longer range therefore feeds back a proportionally weaker transverse field. `ModelParams.trw_irrep_scale` holds one weight $s_i$ per channel: the squared-amplitude fractions of its form factor on the bond classes (on-site, NN $(1,0)$, NNN $(1,1)$, 3NN $(2,0)$, longer — obtained by an FFT of $\varphi_i$ on a `_BOND_GRID`×`_BOND_GRID` periodic grid, `_bond_weights` / `_bond_class_fractions` / `_TRW_BOND_FRAC`) multiplied by the class weights $(1,\ 1,\ J_2/J_1\cdot f,\ J_3/J_1\cdot f,\ 0)$, where $f=$ `trw_far_scale` (default 1: the $(t_\delta/t)^2$ estimate; 0 switches the far-bond transverse field off; $>1$ is a stress test). Bonds that carry no hopping in the model get weight 0. The identical weight is applied in `_build_H_stack` and in `_calc_dHdQ`, so the Hamiltonian and the Hellmann–Feynman force stay consistent. At import time `_check_bond_content` verifies that every harmonic's declared bond kind (on-site / `nn_AB` / `nnn_diag` / `nnn_axis`) matches the real-space support of its form factor. `wave_function_weight` additionally $T_Q$-symmetrizes this sigmoid, averaging its value at $k$ and at $k+(\pi,\pi)$, so the downfolding weight itself respects the AFM-translation parity used elsewhere. $\beta^2(k)$ is the fraction of the quasiparticle wavefunction that remains on the metal ion rather than the ligands at each $k$-point, so the channels that couple through the metal-ion orbital angular momentum operators ($B_{1g,\mathrm{op}}$) are naturally suppressed where the quasiparticle is more ligand-like. By contrast, the Eg,2 term (§7) is added **without** this $k$-weighting.

**Parity bookkeeping in the hole sector.** The hole blocks are $-h(-k)^{*}$. Even-in-$k$ parts keep their sign under $k\to-k$, odd-in-$k$ parts flip it, so every term that can be odd (the `px*/py*` transverse Weiss contributions and hopping-type blocks) is **split into an even and an odd part** and added to the hole blocks with opposite signs (`H_AB_even`/`H_AB_odd`, `H_TRW_nnn_even`/`_odd`), rather than being routed through a plain $-\mathrm{conj}$ of the particle block. Exact Hermiticity is enforced after assembly.

`VectorizedBdG._build_H_stack` builds and diagonalizes this 24×24 matrix for the entire k-grid in one batched `numpy.linalg.eigh` call (the SCF grid is `_NK`×`_NK`=72×72=5184 points), reusing a pre-allocated buffer (`out=`) across SCF iterations to avoid repeated allocation.

The physical electron density is $\langle n_{i\sigma}\rangle=\sum_n \vert u_{n,i\sigma}\vert ^2 f(E_n) + \vert v_{n,i\sigma}\vert ^2(1-f(E_n))$ — both terms carry a positive sign, since $\vert v\vert ^2(1-f)$ is the filled-band contribution from below the Fermi level.

### 10. Observables via `VectorizedBdG`

All thermal-average observables are extracted from a single batched diagonalization:

| Observable | Formula (schematic) | Role |
|---|---|---|
| **⟨B1g⟩** (full) | $\mathrm{Tr}[B_{1g,24}\cdot\rho_k]$, `/4` for Nambu+sublattice doubling, $\beta^2(k)$-weighted | Hellmann–Feynman lattice force (the explicit-$Q$ force used by the Q solver is the full $\langle\partial H/\partial Q\rangle$, §12) |
| **⟨Eg2⟩** | same construction against `Eg2_24`, **without** the $\beta^2(k)$ weight | Eg,2 sanity check / stability diagnostic (§7) |
| Magnetization | $\langle S_{\mathrm{axis}}\rangle$ per doublet via `Sz_stag_nambu_channels` (`compute_channel_staggered_magnetizations`) | AFM order parameter `M` (channel-resolved: Γ₆, Γ₇ₐ, Γ₇ᵦ) |
| `F67_vec` | Gorkov Γ₆–Γ₇ₐ and Γ₆–Γ₇ᵦ amplitudes for **every** channel (84 complex components), `_compute_F67_pair_amplitudes` | Anomalous Weiss-field back-action (§2, §8, §19) and gap-equation input |
| Density | $\sum_n[\vert u\vert ^2 f + \vert v\vert ^2(1-f)]$ / 4 | Chemical-potential control; per-orbital densities for the dynamic Gutzwiller factors (§5) |

The lattice force uses the **full** $\langle\hat B_{1g}\rangle=\mathrm{Tr}[B_{1g,24}\cdot\rho]$, not a bare off-diagonal piece, because in D₂h `B1g_op` gains diagonal and spin-preserving components that are active even without SC. Concretely, `_eval_B1g_and_kweights_at_Q(target_doping, M, Q, Δ_vec, F67_vec, μ_guess, g_t, g_J)` (which replaced the former `B1g_expectation`; it re-solves $\mu$ at fixed density and returns the scalar $\langle B_{1g}\rangle$ together with its $k$-resolved array) contracts the diagonalized Nambu eigenvectors against `B1g_24` and weights by the occupation and the same $\beta^2(k)$ ZRS factor used in the Hamiltonian (§9):

```python
diag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), B1g_24, ec).real     # a,b: 24 Nambu components; n: band index
exp_k   = np.einsum('kn,kn->k', diag_qp, f_n) * beta2_k               # per-k thermal average, β²(k)-weighted
B1g_exp = np.dot(k_weights, exp_k) / 4.0                              # /4: Nambu (particle–hole) × sublattice (A–B)
```

Summing over all 24 Nambu bands with plain $f(E_n)$ (not $1-f$) automatically covers the hole contributions too, since the hole-sector sign is already built into `B1g_24` itself (§3) and $f(-E)=1-f(E)$. `Eg2_expectation` mirrors this exactly but contracts against `Eg2_24` and omits the $\beta^2(k)$ weighting.

### 11. Rigidity K_eff: ∂²F/∂Q² of the Total Free Energy

All free-energy derivatives are taken of the **canonical** (fixed-particle-number) free energy $F=\Omega+\mu N$ (`_canonical_free_energy`): at every probed point it refits $\mu$ (and the density-dependent Gutzwiller factors), evaluates the BdG grand potential of §18 and Legendre-transforms it; its elastic term is the bare `K_lattice`.

$K_{\mathrm{eff}}$ is computed **inside `compute_cluster_free_energy`** (§19), which `_make_vertex_params` calls — i.e. in every SCF iteration, in the seeding kick and in the Tc routines. There is no separate `compute_K_eff_full` any more (nor the old `_delta_relaxed_K_eff_correction` helper); the function returns the ingredients and their sum:

```
K_bulk  = 5-point (O(h⁴)) second derivative of F_can(Q) at fixed (M, Δ_vec, F67_vec, μ-refit), step h(Q) = _q_fd_step(Q), with V_JT_corr = V_JT_eff + V_irr_QQ of THIS call
K_cl    = Delta_K_cluster = (K_full − K_MF)/_CLUSTER_SIZE, sign-preserving capped at _DK_CORR_CAP_MULT × K_bare     (§19)
K_schur = H_QΔ² / H_ΔΔ  (capped likewise): Schur complement of the 2×2 finite-difference Hessian of F_can(Q, Δ_tot) along the
          current Δ direction and phases; 0 in the normal state (Σ|Δ| ≤ _QQ_DELTA_THRESH) or if H_ΔΔ ≤ 0
K_eff   = K_bulk + K_cl − K_schur
```

$K_{\mathrm{eff}}$ is the curvature of the **reduced** free energy $F_{\mathrm{eff}}(Q)=\min_\Delta F(Q,\Delta)$, i.e. the quantity whose sign says whether the condensate has softened the lattice; it is a **diagnostic** (iteration log `SCF-A`, result keys below) and is *not* the denominator of any Newton step. In particular the Schur term is deliberately **not** part of the $Q$ equation of §12: at a stationary $\Delta$ the force on $Q$ is $\partial F/\partial Q$ at fixed $\Delta$ (envelope theorem), and subtracting the adiabatic Δ-relaxation there would count it twice. (A toy model with the alternating structure of the SCF confirms the bias: with the Schur term inside the $Q$ step the converged $Q$ is wrong, even spuriously non-zero for $K_{\mathrm{eff}}>0$.)

Two reference numbers are reported after convergence:

- **`K_eff_net`** $=K_{\mathrm{eff}}(\text{SC})-K_{\mathrm{eff}}(\Delta{=}0,F_{67}{=}0)$, both recomputed (two extra `_make_vertex_params` calls) at the **returned** state with the **same** $(M,Q,\mu)$ — it isolates the direct effect of the condensate on the stiffness. Negative: the gap softens the lattice ("SC-triggered JT enabled"); the threshold in `__main__` is $\mp10^{-4}$.
- **`K_eff_normal_Q0`** $=K_{\mathrm{eff}}$ of the normal state at $Q=0$ with its **own** refined $(M,\mu)$, from the seeding kick (`kick['K_eff_n']`; it is a normal-state quantity only if the Δ passed into the solve was zero).

As noted in §7, there is no Eg,2 or B₁g–Eg,2 cross-rigidity computation: the Eg,2 elastic term uses the bare `K_lattice_Eg2`.

### 12. Profiling Out Q: `solve_Q_slave`

Earlier versions iterated $Q$ with a damped Hellmann–Feynman Newton step $Q_{\mathrm{out}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ (`_compute_Q_newton_step`, `_Q_UPDATE_PERIOD`) and a Moriya-regularized lattice susceptibility with Richardson-extrapolated $\chi_\tau$ (`_compute_lambda_JT_sc`, `lambda_JT_sc`, `K_eff_reg`). **None of that exists any more.** At every SCF iteration

$$Q^{*}=\arg\min_Q\,F_{\mathrm{tot}}(Q),\qquad F_{\mathrm{tot}}(Q)=F_{\mathrm{can}}(Q;\,M,\Delta,F_{67},\mu)+\Delta F_{\mathrm{cl}}(Q)-\Delta F_{\mathrm{cl}}(0),$$

where $\Delta F_{\mathrm{cl}}(Q)=[F_{\mathrm{full}}(Q)-F_{\mathrm{MF}}(Q)]/N_{\mathrm{cluster}}$ is the **exact** beyond-mean-field energy of the intra-plaquette bonds from the four-site ED (the `dF_cluster` closure returned by `compute_cluster_free_energy`, §19), evaluated with the chemical potential $\mu(Q)$ of the bulk fixed-density solve at the same $Q$. (A constant-curvature quadratic $\tfrac12 K_{\mathrm{cl}}Q^2$ is *not* used: $\Delta K_{\mathrm{cl}}(Q)$ changes strongly and even changes sign within the scan range — in a test state it went from −0.96 at $Q=0$ to +0.07 at 0.05 Å and +0.15 at 0.10 Å per site — so a frozen curvature misrepresents the energy by up to a factor ≈3 at 0.1 Å.)

`solve_Q_slave` proceeds as follows:

1. **Coarse scan** of $F_{\mathrm{tot}}$ on 13 points over $[-Q_{\max},Q_{\max}]$, $Q_{\max}=0.15$ Å, to find the global basin (no parabolic assumption: first-order jumps and $\pm Q$ breaking are handled). If the minimum lies on the box edge, the range is doubled adaptively (at most three times, capped at $\max(2Q_{\max},\lambda_{hop}/2)$, only while the minimum improves); if it still sits on the edge, the grid value is returned.
2. Inside the bracketing interval the exact stationary point is located by a **Brent root** of the full force
   ```
   F_Q(Q) = ⟨∂H_BdG/∂Q⟩  +  ½ Σ_c M_c² ∂J_c(q_AFM)/∂Q  +  K_lattice·Q  +  ∂E_cond/∂Q  +  d[ΔF_cl]/dQ
   ```
   with the complete explicit $Q$-vertex of `_calc_dHdQ` (§14), $\partial E_{\mathrm{cond}}/\partial Q$ by a central difference of the condensation energy with the **frozen vertex cache** that `solve_self_consistent` now passes in (a diagonal $V_{JT,\mathrm{corr}}\,\mathrm{diag}(g)$ fallback if there is none), and $d\Delta F_{\mathrm{cl}}/dQ$ by a central difference ($h=10^{-3}$ Å) of the cached exact ED function. $\mu$ is refit at every force evaluation; the Eg,2 amplitude is held at 0 in the $Q$ search.
3. The root's curvature is verified to be positive; otherwise the grid minimum is kept. Force and energy are consistent by construction — a numerical check of the root against an independent fine-grid minimum of $F_{\mathrm{tot}}$ agreed (0.0504 vs 0.0500 Å, $dF_{\mathrm{tot}}/dQ=3\times10^{-5}$ at $Q^*$ vs $1.5\times10^{-2}$ at $Q^*+0.01$ Å).

The Schur term and $K_{\mathrm{eff}}$ itself never enter (§11). In `solve_self_consistent` the solved $Q_{\mathrm{out}}$ is **relaxed with its own weight**, $Q_{\mathrm{mixed}}=Q+\alpha_Q\,(Q_{\mathrm{out}}-Q)$ with `_ALPHA_Q = 0.5` (clipped to $\pm\lambda_{hop}/2$), instead of the small $\Delta/M$ mixing weight, and **$Q$ is not part of any Anderson vector**; convergence is judged on the $Q$ *residual* $\lvert Q_{\mathrm{out}}-Q\rvert$ (a step $\alpha_Q\cdot$residual would declare convergence too early). With `force_Q_zero` no $Q$ solve is performed at all. A sign flip $Q\cdot Q_{\mathrm{mixed}}<0$ clears the Anderson history, the RPA vertex cache, the $V_{\mathrm{irrep}}$ EMA state and the cached normal-state eigensystem (the FS topology may change). A displacement larger than `_Q_THR_REL·lambda_hop` caps the mixing rate at $0.3\,\alpha_0$ for that iteration. **Cost:** every scan, root and curvature point triggers the cluster ED (two $1296\times1296$ diagonalizations) and a μ solve, which makes `solve_Q_slave` the most expensive single call of an iteration.

### 13. χ_QQ from Thermodynamic Finite Differences

The orbital JT susceptibility $\chi_{QQ}=-\partial^2\Omega/\partial Q^2$ is evaluated by `_chi_QQ_matrix_elements` as a central finite difference of the BdG grand potential **at the state that is passed in** — $(M,Q,\Delta_{\mathrm{vec}},F_{67},\mu)$ — divided by 4 for the Nambu×sublattice doubling and normalized by $g_{JT}^2$ (1/eV). It is therefore state-dependent: the normal-state quantity needs $\Delta=0$, $F_{67}=0$, $Q=0$ and the normal-state $\mu$, the SC-state quantity the converged SC values, and the two must not be interchanged. With `return_matrix=True` it returns the full $2\times2$ matrix over $(Q_{B_{1g}},Q_{Eg2})$ using a 9-point mixed stencil. `__main__` evaluates it twice: at the converged SC state (`K_spont(SC state)=g_JT²χ_QQ(SC)`) and at $\Delta=0$, $Q=0$ with the SC run's $M$ and $\mu$ re-solved at the target density (`K_spont(Δ=0)`, the spontaneous-JT criterion of §16/§17). The Eg,2 element is evaluated in the converged SC state for the stability diagnostic of §7. The pairing vertex itself always uses the normal-state ($\Delta=0$) susceptibilities, to avoid feeding the gap back into its own interaction.

### 14. Channel-Resolved RPA Vertex: [Γ₆, Γ₇ₐ, Γ₇ᵦ, JT] in One 4×4 Closure

The bare local interaction is **diagonal** in a 4-channel basis $[S_{\Gamma_6},S_{\Gamma_{7a}},S_{\Gamma_{7b}},\,\mathrm{JT}]$ — there is no separate bare spin–JT cross-vertex constant:

```
U(q) = diag( J_Γ6(q), J_Γ7a(q), J_Γ7b(q), V_JT_corr )
```

with $J_c(q)$ the per-doublet momentum-dependent exchange of §6 and `V_JT_corr = V_JT_eff + V_irr_QQ`, where `V_JT = g_JT_bare² / K_bare` is the bare JT pairing vertex, `V_JT_eff = V_JT · retardation_factor` applies the adiabatic phonon-retardation reduction described below, and `V_irr_QQ` is the B₁g–B₁g component of the **irreducible vertex extracted from the four-site plaquette cluster ED** (§19). All spin–JT **mixing** in the RPA matrix comes from the off-diagonal bare susceptibilities themselves, not from a bare cross-interaction. The quantities `V_JT_eff`, `V_JT_corr`, $V_{\mathrm{cap}}$ and $\Gamma_M$ are assembled by `_make_vertex_params(target_doping, M, Q, Δ_vec, F67_vec, n_kspace, μ, g_t, g_J, Q_Eg2, vertex_cache)`, which returns `(Γ_M, F_cluster, V_JT_eff, V_JT_corr, V_cap)`: it evaluates the retardation factor, calls `compute_cluster_free_energy` (§19) — which supplies `V_irr_QQ` and hence `V_JT_corr`, and in the same pass $K_{\mathrm{eff}}$ (§11) and the $Q$-energy closure (§12) — and computes $J_{\mathrm{eff}}$ (the $M^2$-weighted average of §5), $V_{\mathrm{cap}}$ and the Landau-derived $\Gamma_M$.

**Phonon retardation.** Because `M_eff` (an input reduced mass, amu) is a primary parameter, the code derives a genuine bare JT phonon energy $\hbar\omega_0 = 64.6528\,\mathrm{meV}\cdot\sqrt{K_{\mathrm{bare}}[\mathrm{eV/\mathring A^2}]/M_{\mathrm{eff}}[\mathrm{amu}]}$ (`_omega_0_JT`, set in `RMFT_Solver.__init__` and `_full_rebuild`), and uses it, inside `_make_vertex_params`, to evaluate the static-limit reduction of a single-Einstein-mode phonon propagator at the characteristic gap scale. $\Delta_{\mathrm{typ}}$ is now the **maximum of the full order-parameter magnitude on the Fermi surface**,

```
Δ_typ = max_{k ∈ FS} √( |Φ(k)·Δ_7a|² + |Φ(k)·Δ_7b|² ),          retardation_factor = x² / (x² + 1),   x = ω₀ / Δ_typ
```

— a one-point evaluation of $D(i\Omega)/D(0)=1/(1+(\Omega/\omega_0)^2)$ at $\Omega=\Delta_{\mathrm{typ}}$, not a full Matsubara average (there is no separate `_jt_retardation_factor` method any more). It is 1 (no suppression) in the static/anti-adiabatic limit $\omega_0\gg\Delta_{\mathrm{typ}}$ and at $\Delta_{\mathrm{typ}}\to0$, and suppresses the phonon-mediated coupling once the gap grows comparable to or past the bare phonon energy. It applies **only** to the phonon-mediated `V_JT`, never to `V_irr_QQ` (a purely electronic irreducible vertex with no phonon propagator of its own).

**Bare susceptibilities.** $\chi_0(q)$ comes from the $\Delta=0$ BdG Hamiltonian via the static Lehmann kernel `_lehmann_kernel` (continuous $\omega\to0$ limit: $\Delta f\,\Delta E/(\Delta E^2+\eta^2)$ away from degeneracy, $-f'$ on the Fermi surface), with broadening $\eta=\max(\text{ETA\_DELTA\_FRAC}\cdot|\Delta|,\ \text{ETA\_GRID\_FLOOR}\cdot t_0)$. `_chi0_channels_raw` builds the full raw $4\times4$ matrix $\chi_{ab}(q)=\langle\!\langle O_a;O_b\rangle\!\rangle$ as **one Gram product** of four band-basis vertex tensors against the shared kernel; the vertices are the per-doublet uniform spin $S_{\mathrm{axis}}$ for $c\in\{\Gamma_6,\Gamma_{7a},\Gamma_{7b}\}$ (diagonal Nambu vectors for `M_axis='z'`, dense 24×24 matrices for `'x'`/`'y'`) and the JT vertex. The JT vertex is the **mechanical** one: the *full* explicit $\partial H/\partial Q$ of `_calc_dHdQ` (local JT, hopping, Weiss field and transverse-Weiss channels, §12), not just $g_{JT}B_{1g}$; the contact (diamagnetic) term $\langle\partial^2H/\partial Q^2\rangle$ — the five-point derivative of the complete first-derivative vertex, including $\partial_Q^2[\beta_k Q]$ and the second derivatives of the exchange and transverse-Weiss terms — is subtracted from $\chi_{QQ}$ when `apply_diamagnetic_QQ=True` (`_diamagnetic_QQ_term`). Entries are normalized to a common 1/eV scale (spin–Q by $g_{JT}$, QQ by $g_{JT}^2$). The $k+q$ partner eigensystem is obtained from the pre-built cyclic `shift_table` (§ Key Algorithms), not a second diagonalization. The static Lindhard function is real by time-reversal symmetry, so only the real part is kept after Hermitization. `_rpa_closure` then Moriya-damps only the $3\times3$ spin sub-block (eigen-decomposition, $w\to w/(1+\Gamma_Mw)$, via the model-derived $\Gamma_M$ below), PSD-projects the full $4\times4$ matrix by eigenvalue clamping (guarding against Cauchy–Schwarz violations from numerical noise near the QCP), and returns the full matrix (`chi0`) with legacy-compatible scalar aggregates:

```
chi_SS = Σ_cc' χ0[c,c']       (sum over the 3 spin channels)
chi_SQ = chi_QS = Σ_c χ0[c, JT]
chi_QQ = χ0[JT, JT]
```

Off-diagonal spin–spin entries $\chi_{cc'}$ ($c\neq c'$) vanish for orbitally-isotropic hopping at $Q=0,\Delta=0$, but not for $Q\neq0$ (JT-induced Γ6–Γ7 mixing) or in the SC state.

**Vertex assembly (`_rpa_closure`, `_rpa_vertex_tables`).** The closure builds the $4\times4$ RPA denominator $M(q) = I - \chi_0(q)\,U(q)$ and its floor-regularized determinant (floored in magnitude only, at $\max(\varepsilon,10^{-4}\Vert M(q)\Vert_F)$, never in sign, so a genuine QCP crossing is never masked), and a **Stoner factor** (`stoner`) $=$ the largest eigenvalue of $\mathrm{diag}(J_c(q))\,\chi_{cc'}$. (The earlier $2\times2$ scalar `_rpa_det` no longer exists; the 4×4 closure is the only entry point.) For every unique $q$ needed by the pairing kernel, `_rpa_vertex_tables` forms $\chi_{\mathrm{RPA}}(q)=\mathrm{adj}(M(q))\,\chi_0(q)/\det M(q)$ — the adjugate (computed through an SVD, `_adjugate`) stays finite even as $\det M(q)\to0$ — and tabulates the spin block $\Xi=-J_cJ_{c'}\chi_{cc'}^{\mathrm{RPA}}$, the cross vector $2J_cV_{JT,\mathrm{eff}}\chi^{\mathrm{RPA}}_{c,JT}$ and the JT element $V_{JT,\mathrm{eff}}^2\chi^{\mathrm{RPA}}_{JT,JT}$. The summed pairing-channel contribution of every FS pair is hard-clamped to $\pm V_{\mathrm{cap}}$ with `V_cap = _RPA_V_CAP_ALPHA · max(_RPA_BW_FACTOR·max(|tx|,|ty|), J_eff)` (`_RPA_V_CAP_ALPHA = 2.2`, `_RPA_BW_FACTOR = 8`) — a numerical overflow guard only; the sign and divergence character of $V(q)$ near the QCP are never altered, and a negative determinant (past the QCP) is deliberately left untouched rather than capped.

**Two separately tracked determinants.** The vertex cache stores `det_q0` (the $q=0$, ferromagnetic-channel determinant) and `det_afm` (the $q=(\pi,\pi)$, AFM-channel determinant) independently, together with `stoner_q0`/`stoner_afm` and the scalar susceptibilities `chi_SS/SQ/QQ_{q0,afm}`. SCF adaptive mixing and convergence respond to the **SC-state** AFM determinant `det_afm_sc`; the normal-state ones guard separately against an accidental ferromagnetic divergence. Both are logged (`dFM_sc=`, `dAFM=`, `dAFM_sc=`).

**Sign-flip EMA guard.** The diagonal of the pairing matrix $V_{\mathrm{irrep}}$ is smoothed channel by channel. Every channel whose form factor **changes sign under the AFM shift** $k\to k+(\pi,\pi)$ (`_afm_shift_character = −1`: `es`, `d`, `px`, `py`, …) sees the AFM-peaked spin-fluctuation interaction as *attraction* and can flip sign together with $\det_{\mathrm{afm}}$ near the QCP; for these, when the raw diagonal element would flip sign relative to its EMA state (`SolveState.V_irrep_ema`), the update is blended with a weight that depends on $\vert\det_{\mathrm{afm,sc}}\vert/$`_DET_SIGN_FLIP_SCALE`$(=0.05)$ through a sigmoid (steepness `_EMA_SIGN_FLIP_SLOPE = 6.0`, floor `_EMA_SIGN_FLIP_W_MIN = 0.20`, base weight `_EMA_NEW_WEIGHT`), so the blend weight shrinks toward its floor near the QCP (preserving genuine sign ambiguity there) and grows away from it (suppressing numerical noise); a transition to a positive diagonal element is blended with the plain base weight. After a saddle-escape kick the base weight is doubled for one iteration (`_ema_kick_pending`). Channels with $\sigma=+1$ ($s$, $g$, `px2`, `py2`, …) are not smoothed.

**Moriya damping** ($\Gamma_M$) is obtained self-consistently from the model's own Landau free-energy expansion rather than an empirical closed-form fit: `_moriya_gamma_landau` probes $F(M)$ at $M=0,\pm h,\pm2h$ (`_MORIYA_LANDAU_M_STEP = 0.06`, a 5-point stencil) on the $\Gamma_6$-only channel via `_compute_bdg_free_energy` (with $\mu$ refit and the Legendre correction), extracts the quartic Landau coefficients $a,b$, and solves the self-consistent fluctuation equation $\Gamma_M=b\langle\delta M^2\rangle$, $\langle\delta M^2\rangle=kT/(a+b\langle\delta M^2\rangle)$ in closed form: $\Gamma_M=({-}a+\sqrt{a^2+4bkT})/2$ (reducing to the classical $bkT/a$ limit when $a^2\gg4bkT$, and staying finite as $a\to0$). The result is capped at $\Gamma_{M,\max}=4t_{\mathrm{eff}}^2/(\pi J_{\mathrm{eff}})$ and cached per (doping, $g_t$, $g_J$); the cache is dropped on cloning because $\Gamma_M$ is $kT$-dependent.

### 15. FS-Resolved Pairing Kernel: Orbital-Character-Weighted, Exactly G-Covariant

`compute_pairing_kernel_and_build_cache` builds the actual momentum-space pairing interaction seen by the gap equation. It is a genuine sum over the doublet-channel content of the Fermi-surface quasiparticle at **each** of the two scattering momenta. `_fs_channel_weights` first extracts, at every sampled FS point $k$, the doublet content $w_c(k)$ ($\sum_c w_c=1$, from the squared overlap of the lowest positive-energy BdG band with each channel's orbital block). The spin-mediated, JT, and cross contributions to the pairing interaction between two FS points $k,k'$ (momentum transfer $q=k-k'$) are

```
V_spin (k,k') = − Σ_{c,c'} √(w_c(k) w_c(k'))·√(w_{c'}(k) w_{c'}(k'))· J_c(q) J_{c'}(q) · [χ_RPA(q)]_{cc'}
V_cross(k,k') =   2 Σ_c √(w_c(k) w_c(k'))· J_c(q) V_JT_eff · [χ_RPA(q)]_{c,JT}
V_JT   (k,k') =   V_JT_eff² · [χ_RPA(q)]_{JT,JT}
```

A local B₁g-character weight $c_B(k)=\vert \varphi_d(k)\vert /\max_{k'}\vert \varphi_d(k')\vert \in[0,1]$ symmetry-gates the spin and cross channels — `V_w(k,k') = V_JT(k,k') + c_B(k)c_B(k')·[V_spin(k,k')+V_cross(k,k')]` — so that both electrons antinodal recovers the full RPA-dressed interaction, either leg nodal keeps only the always-on JT piece, and pure d-wave symmetry is never accidentally broken by a spin contribution surviving at one nodal leg. The gate stays tied to the B₁g/antinodal structure and weights the same kernel from which every channel is projected.

**Exact $G$-covariant sector kernels.** `_get_fs_points` returns the *fundamental domain* of $G=\{E,T_Q\}\times D_{2h}$ (the diamond magnetic BZ, closed under $D_{2h}$ but not under $T_Q$), so the plain kernel $V(k_i-k_j)$ on it is not $G$-covariant: the full-BZ gap equation also contains the image terms $V(k_i-T_Q k_j)$, which enter with the $T_Q$ parity of the gap function. Every gap function of definite symmetry is a 1-D character $\chi$ of $G$ (§8), and the exact projected kernel is

```
K_χ(k_i, k_j) = (1/|G|) Σ_{g∈G} χ(g) · V_w(k_i, g·k_j),        q_g = k_i − g·k_j,   V_cap applied to every image pair
```

evaluated on the canonical set of unique $q$ (folded to $\lvert q_x\rvert,\lvert q_y\rvert$ — the same $D_{2h}$-Shubnikov assumption — so the number of RPA evaluations stays bounded by the $(N_K/2+1)^2$ canonical grid however many images are used). Consequences:

- The **8 sector kernels** are diagonalized separately (with the FS integration weights as $\sqrt{w}$ factors). `lambda_lin_max` is the largest top eigenvalue among the sectors that are **covered** (contain at least one basis function; if the odd-channel FS-closure check below fails, the odd-parity sectors — those with $m_xm_y<0$ — are excluded as well, and if nothing remains all non-empty sectors count). The boolean array `sector_drives` in the cache is the single source of truth for "covered": the leading-sector choice, the kick seeds and the Tc routines all read it; `lambda_lin_max_all` is the largest of all 8 — a larger *uncovered* eigenvalue signals a **basis gap**, reported by `_sector_report_lines` together with the library harmonics that would close it (a greedy scan; a candidate is suggested only if it raises the resolved weight by at least `_SECTOR_SCAN_MIN_GAIN`).
- The pairing matrices `V_irrep`, `W_irrep`, `S_irrep` ($N_{\mathrm{irrep}}\times N_{\mathrm{irrep}}$, FS-weighted projections of the sector kernels onto the basis form factors, with $S$ the Gram matrix) are **block-diagonal in $\chi$**: functions of different $T_Q$ parity cannot mix at the linear level (the nonlinear BdG equations still couple them). `v_raw` / `frac` / `resolved_weight` come from the Löwdin decomposition (`_irrep_decomposition`) of the leading covered sector's eigenvector onto that sector's basis functions; `unresolved_weight = 1 − resolved_weight` quantifies how much of the dominant eigenvector lies outside the basis.
- `frac_seed` additionally mixes in the near-degenerate covered sectors ($\lambda_\chi\ge$ `_SECTOR_NEAR_FRAC`$(=0.8)\cdot\lambda_{\mathrm{lead}}$, weighted by $\lambda_\chi/\lambda_{\mathrm{lead}}$ times their resolved weight), so that `_scf_kick` seeds all of them and the nonlinear gap equation can hybridize them.
- Per-channel decompositions `V_parts_diag` (spin / JT / cross projections onto each form factor, which sum to the diagonal of `V_irrep` before the clamp), `K_d_spin` and `W22_JT` (the $d$–$d$ JT-only element) are kept for the diagnostics.
- **Odd-channel closure check.** The odd-parity channels (`px`, `py`, `px2`, `py2`, …) are only well-defined if the FS sample is inversion-closed with inversion-symmetric weights; `_check_fs_closure` tests this directly on the $N_K\times N_K$ grid (closure under $k\to-k$, $k_x\to-k_x$, $k_y\to-k_y$ and weight symmetry). If it fails, the odd rows and columns of `V_irrep`, `W_irrep` and `V_parts_diag` are zeroed with a `PAIR-DIAG` warning. The kernel-level $D_{2h}$ selection rules are not re-checked — they follow from character orthogonality.

The FS integration weight $w_k$ of `_get_fs_points` is a **symmetry-adapted thermal 2D element** $w_k=\mathrm{therm}_{\mathrm{sym}}(k)\,dA/\text{BZ\_NORM}\cdot\text{scale}$ (the finite-$T$ regularized form of $dl/\vert v_F\vert$; no $\vert v_F\vert$ floor), computed on whole symmetry orbits of the diamond MBZ so that non-$A_{1g}$ functions cancel exactly by summation. The symmetrization group is chosen from the **exact symmetry of the normal-state BdG spectrum**: $D_{2h}$ if $\Delta_{B_{1g,\mathrm{static}}}\neq0$, $Q\neq0$, or $\max_{k,n}\vert E_n(k)-E_n(Rk)\vert>$ `_C4_SPECTRAL_TOL` (with $R$ the C₄ rotation), otherwise $D_{4h}$. The staggered Weiss field alone is **not** a C₄ breaker for the collinear $S_z$ order (it is invariant under a site-centred C₄ rotation), whereas an in-plane Néel axis is caught by the spectral test. The number of sampled points is the adaptive `thermal_fs_nat_size`.

The FS-weighted symmetric kernel $K_{\mathrm{fs}}(k,k')=\sqrt{w_k}\,V_w(k,k')\,\sqrt{w_{k'}}$ of each sector sets the largest eigenvalue `lambda_lin_max` and the eigenvector decomposition `(v_raw, frac)` that blend into the SCF gap update (§21). `lambda_JT_kernel` (the Rayleigh projection of `v_raw` onto the pairing matrix with its $d$–$d$ element replaced by the JT-only one, `W22_JT`) is a diagnostic companion.

### 16. Normal-State JT Stability and the Post-Convergence Hessian Mode Classification

The spontaneous-JT criterion of the normal state is a direct threshold comparison: `K_spont = g_JT² · χ_QQ` evaluated at **Δ = 0, Q = 0** with the `ref` run's $M$ and the chemical potential **re-solved at the target density** (`__main__`, §13), and `K_lattice > K_spont` is logged as the verdict ("✓ normal state JT-stable" / "⚠ normal state already JT-unstable"). A second line, `K_spont(SC state)`, evaluates the same expression at the converged SC state and is labelled as such.

Which mode of the system is actually soft is read off the **finite-difference Hessian** `compute_hessian`: a $(3{+}1{+}1)$-dimensional probe over $(M_{\Gamma_6},M_{\Gamma_{7a}},M_{\Gamma_{7b}},Q,\Delta_{\mathrm{tot}})$ of the canonical free energy, with all 84 gap channels probed as **one combined amplitude** $\Delta_{\mathrm{tot}}=\sum_i\vert\Delta_i\vert$, rescaling them proportionally and holding their converged relative phases fixed (so the probe stays on the stationary point even for a state with a nontrivial relative phase). The step in the Δ direction is $\max(\text{\_HESS\_DELTA\_EPS\_MIN},\,10^{-3}\vert\Delta_{\mathrm{tot}}\vert)$ (the former guard that shrank the step to $10^{-10}$ at $\Delta=0$ is gone). The Hessian is rescaled by the natural physical scales of the five directions ($M$ clip, $\lambda_{hop}$, $t_{\mathrm{eff}}$) before its smallest eigenvector is extracted; the eigenvector's fractional weight on the $M$-block, the $Q$-component and the $\Delta$-component classifies the soft mode against `_MODE_FRAC_DOMINANT = 0.60` as **`pure-SC`**, **`pure-JT`**, **`AFM-fluctuation`**, or **`mixed`** (the fifth label `SC-triggered-JT` and its threshold `_MODE_FRAC_MIXED` no longer exist). The result dictionary also carries `frac_D`, the Δ-fraction of the soft mode. `lambda_min` is the scaled smallest eigenvalue (used for the kick logic), while the unscaled eigenvalues are stored as `eigenvalues` and feed the reliability check (`_scf_result_reliability`: a result is reliable only if converged, not past the QCP, and the Hessian has no eigenvalue below $-10^{-6}$). The classification drives the saddle-escape "kick" inside `solve_self_consistent`: **the kick is applied whenever the soft mode has a significant gap component, `frac_D > _KICK_MIN_DELTA_FRAC` (0.25)** — so a `pure-SC` *or* a mixed Δ–Q mode — while `M` is pulled toward the Γ₆-only Stoner estimate only in `pure-SC` mode (§ Key Algorithms, step 11); the same Hessian also seeds `_scf_kick`.

### 17. SC-JT Coexistence: What the Code Actually Evaluates

In the current code the SC-triggered-JT verdict is assembled from the following independently logged quantities, all computed from the converged `ref` result:

| Quantity | Definition | Where | Reading |
|---|---|---|---|
| normal-state stability | $K_{\mathrm{lattice}}$ vs. $K_{\mathrm{spont}}=g_{JT}^2\chi_{QQ}(\Delta{=}0,Q{=}0)$ | `__main__` | $K_{\mathrm{lattice}}>K_{\mathrm{spont}}$ required (else spontaneous, SC-independent JT) |
| `K_eff_net` | $K_{\mathrm{eff}}(\mathrm{SC})-K_{\mathrm{eff}}(\Delta{=}0)$ at the **returned** $(M,Q,\mu)$ (§11) | result dict / `__main__` | $<-10^{-4}$: the gap softens the lattice (SC-triggered JT enabled); $>+10^{-4}$: the gap stiffens it |
| `K_eff_normal_Q0` | $K_{\mathrm{eff}}$ of the normal state at $Q=0$ with its own $(M,\mu)$ (from the kick, §11) | result dict | the sign says whether the normal state is itself lattice-stable |
| Hessian $\lambda_{\min}$ | smallest eigenvalue of the (3+1+1)-dim Hessian (§16) | result dict / `__main__` | $<0$: "SC-triggered JT CONFIRMED" (state is a saddle along the soft mode) |
| $\partial\lambda_{\mathrm{pair}}/\partial Q$ (FS) | central difference of `lambda_lin_max` at $Q\pm\max(\text{DQ\_FS\_VERTEX},\ \text{FRAC}\cdot\lvert Q\rvert)$, and the fraction of FS pairs whose vertex grows with $Q$ (`hot_spot_frac`) | post-convergence | $>0$: the condensate-induced lattice response enhances pairing at hot spots |
| `selection_ratio` | §2 | `__main__` | $>$ `_JT_ACT_THR` (log) / $>0.02$ (`__main__` line): JT channel judged open |

There is **no** separate $\lambda_{JT,\mathrm{sc}}=g_{JT}^2\chi_\tau^{\mathrm{net}}/K^{\mathrm{reg}}_{\mathrm{eff}}$ viability threshold; the constant `_LAMBDA_JT_VIABLE` is still defined in the module but is not referenced by any routine. The normal-state stability check and the quantities above are logged directly in `__main__`, not assembled by a dedicated solver method or plotting routine. (The earlier "AFM Stoner/Moriya proximity" log line of `__main__` has been dropped; the Stoner factors are still logged per iteration in `SCF-C`.)

---

### 18. Variational Free Energy and the Role of the Cluster

The mean-field free energy used throughout is the BdG grand potential per site (`_compute_bdg_free_energy`)

```
Ω = ¼ Σ_{k,n} w_k [ min(0,E_n) − kT·ln(1+e^{−|E_n|/kT}) ]  +  ¼ Tr[H_particle]
    + E_cond  +  ½ K Q²  (+ ½ K_Eg2 Q_Eg2²)  +  ½ Σ_c J_c(q_AFM) M_c²
```

where the last term is the mean-field double-counting correction of the Weiss field, $J_c(q_{AFM})=Z_{\mathrm{afm,eff}}J_{A1g}[c]$, and the condensation energy runs over **all 84 amplitudes**:

```
E_cond = Re[ Δ_7a† · V_a⁻¹ · Δ_7a ] + Re[ Δ_7b† · V_b⁻¹ · Δ_7b ]
V_{a,b}[i,j] = V_irrep[i,j] · √(g_i g_j)             (_pairing_strength_matrix; g from _gutzwiller_irrep_vector, §5)
```

$V^{-1}$ is taken on the **attractive spectrum only** (`_pairing_condensation_energy_Nch`): the symmetry-resolved $V_{\mathrm{irrep}}$ has repulsive diagonal entries (e.g. `px`, `px2`) and an indefinite $(es,d)$ block, and a plain inverse would give a repulsive eigenmode the *negative* cost $1/\lambda<0$ — a free energy unbounded from below along a gap in a repulsive channel. Eigenvalues below `_V_REPULSIVE_FLOOR_FRAC`$(=10^{-2})\cdot\lambda_{\max}$ are replaced by that floor (a stiff but finite penalty); for an all-attractive matrix this equals $(V+10^{-6}I)^{-1}$. The $\Gamma_{7a}\leftrightarrow\Gamma_{7b}$ cross term is omitted, consistently with the block-diagonal pairing structure. All derivatives of §11–§12, §16 and §21 use the **canonical** form $F=\Omega+\mu N$ with $\mu$ refit at each probe (`_canonical_free_energy`; the elastic term inside it is the bare `K_lattice`). The vertex cache that supplies $V_{\mathrm{irrep}}$ is passed to `_canonical_free_energy` and to the $Q$ search; without one a diagonal $V_{JT,\mathrm{corr}}\,\mathrm{diag}(g)$ fallback is used.

**The cluster is not added to this free energy.** There is no `F_total = F_bdg + F_cluster`: the four-site ED (§19) reports its own free energy per site (`F_per_site`, logged as `F_cluster`) purely as a diagnostic, and it enters the physics in exactly three places — the irreducible B₁g vertex `V_irr_QQ` that renormalizes the JT pairing vertex, $V_{JT,\mathrm{corr}}=V_{JT,\mathrm{eff}}+V_{irr,QQ}$ (§14); the beyond-mean-field correction $\Delta K_{\mathrm{cl}}$ to the stiffness $K_{\mathrm{eff}}$ (§11); and the beyond-mean-field energy $\Delta F_{\mathrm{cl}}(Q)$ of the intra-plaquette bonds in the $Q$ profile (§12). Gutzwiller factors handle kinematic Mott renormalization; the cluster ED handles the local beyond-mean-field effects of the B₁g channel (never the itinerant susceptibility bubble itself); RPA handles the reducible ladder summation over the full BZ. Because the bulk BdG already treats **every** bond at mean field, the cluster only contributes the *difference* between exact and mean-field treatment of its own intra-cluster bonds (§19) — otherwise the mean-field part of those bonds would be counted twice.

### 19. Four-Site Plaquette Cluster: Beyond-Mean-Field Stiffness, Q-Energy and Exact Vertex Extraction

Beyond the BdG mean field, a **four-site (2×2 open plaquette)** cluster, each site carrying the full 6-orbital Kramers-doublet basis (a $6^4=1296$-dimensional Hilbert space), is exactly diagonalized by `compute_cluster_free_energy` — every SCF iteration (through `_make_vertex_params`), in the seeding kick and in the Tc routines. The checkerboard geometry is

```
0 --x-- 1
|       |
y       y
|       |
3 --x-- 2
```

with sites 0, 2 on sublattice A and 1, 3 on sublattice B. Each site has $n_{\mathrm{intra}}=2\,\mathrm{len}(\_BONDS)/N_{\mathrm{cluster}}=2$ intra-cluster nearest-neighbor bonds (derived from the bond list, not hard-coded; bond sign $\eta=+1$ for $x$-bonds, $\eta=-1$ for $y$-bonds in the B₁g channel, mirroring the real-space $\cos k_x-\cos k_y$ bond weighting; the A₁g/magnetic channel is bond-direction-independent) and $Z_{\mathrm{eff}}=Z-n_{\mathrm{intra}}$ external neighbours absorbed into the mean-field Weiss embedding. The local single-site Hamiltonians (`build_local_hamiltonian_for_bdg`) include $-\mu$ (the chemical potential **of the bulk fixed-density solve at the probed $Q$**, cached per $Q$), $\Delta_{CF}$ (and the $\Gamma_{7b}$ internal splitting `g7split`), the AFM Weiss field (leading Γ₆ channel only: the embedding uses the single $M_{\Gamma_6}$ and $J_{A1g}[0]$), the JT coupling $g_{JT}^{\mathrm{cluster}}\,Q\,B_{1g,\mathrm{op}}$ with $g^{\mathrm{cluster}}_{JT}=\beta_{\mathrm{cluster}}\,g_{JT}$, and — once the on-site anomalous amplitudes are non-zero — the transverse Weiss field built from `F67a_s` and `F67b_s`, each weighted by its own cross-doublet exchange constant `J_B1g^(7a)`/`J_B1g^(7b)` (§6) and scaled by the same $\beta_{\mathrm{cluster}}$ — the thermally weighted ($f(1-f)/kT$) $k$-average of the ZRS spectral weight over the normal-state spectrum. (The bond-channel amplitudes of the 84-component basis do not enter the cluster embedding.) The cluster Hamiltonian is

```
H_cluster = Σ_site H_site  +  Σ_bonds [ J_A1g[0] · (S_axis ⊗ S_axis) + η · J_B1g^(7a) · (B1g_op ⊗ B1g_op) ]
```

with $S_{\mathrm{axis}}$ the doublet-projected spin operator along the Néel axis (§1); the former `multi_op` operator no longer exists.

**Beyond-mean-field stiffness and energy.** The bulk BdG treats every bond at mean field, so the cluster must only supply the *difference* between exact and mean-field treatment of its own intra-plaquette bonds. Three cluster free energies are formed, each as $F=E_0-k_BT\ln Z+N_{\mathrm{cluster}}\cdot\tfrac12 Z_{\mathrm{Weiss}}\,J_{A1g}[0](Q)\,M^2$ — the last term being the Weiss double-counting counter-term that belongs to *its own* Weiss coordination, evaluated at the probed $Q$:

```
F_full(Q): external Weiss field (Z_eff = Z − n_intra)  +  exact intra-plaquette exchange
F_MF(Q)  : ALL NN bonds at mean field (Z_MF = Z), no explicit exchange
F_bare(Q): external Weiss field only, no explicit exchange        (diagnostic, see V_irr_QQ_curv)
ΔF_cl(Q) = [F_full(Q) − F_MF(Q)] / N_cluster                      beyond-mean-field energy per site
ΔK_cl    = (K_full − K_MF) / N_cluster                            its curvature (5-point stencils in Q)
```

Subtracting $F_{\mathrm{MF}}$ (not $F_{\mathrm{bare}}$) removes exactly the mean-field content of the intra-plaquette bonds, which the bulk already contains. (In a test state, using $F_{\mathrm{bare}}$ gave a stiffness correction of −0.31 per site, whereas the proper beyond-mean-field part was −1.19: the mean-field part is *not* small, because the Weiss field changes the Γ₆/Γ₇ energy denominators that drive the B₁g curvature.) $K_{\mathrm{MF}}$ is built with $Z_{\mathrm{MF}}=Z$ — the cluster has no $J_2/J_3$ bonds — whereas the bulk longitudinal Weiss field uses $Z_{\mathrm{afm,eff}}=Z(1-J_2/J_1-J_3/J_1)$ (≈5% lower; see Known Limitations). `compute_cluster_free_energy` also returns the callable **`dF_cluster(q, mu_q)`** that evaluates $\Delta F_{\mathrm{cl}}$ at an arbitrary $q$ (two $1296\times1296$ diagonalizations, μ taken from the caller's bulk solve at that $q$); `solve_Q_slave` uses it for the exact $Q$ profile (§12). $K_{\mathrm{eff}}$ and its pieces (§11) are assembled in the same routine.

**Vertex extraction via cluster inverse susceptibilities.** Rather than a linear regression against connected correlators, the irreducible vertex is extracted the standard cluster-ED way: an exact static Lehmann susceptibility tensor of shape $(8,8)$ (2 channels — $S_{\mathrm{axis}}$ and $B_{1g}$ (one aggregate B₁g channel at this level, not separately per cross-doublet branch) — × 4 sites) is computed for two reference spectra — `chi0_tensor` (four independent sites, no intra-cluster exchange) and `chi_full_tensor` (the same cluster **with** intra-cluster exchange switched on) — both at $M=0$, $Q_{\mathrm{ref}}=0$, no anomalous Weiss field, no JT, with $J_{A1g},J_{B1g}$ evaluated at $Q=0$: the appropriate *normal-state* reference for a lattice Bethe–Salpeter equation. At $M=0$ the Weiss fields vanish, so the vertex does not depend on the choice of $Z$. The irreducible vertex follows from the inverse-susceptibility difference,

```
Γ_ED = χ0⁻¹ − χ_full⁻¹                    (both inverses stabilized: only numerically null eigenvalues regularized)
```

computed in the full 8×8 site×channel space and only *then* projected onto the staggered (spin, weight $\tfrac12(1,-1,1,-1)$ across the 4 sites) and uniform (B₁g, weight $\tfrac12$ on all 4 sites) subspaces — projection and matrix inversion do not commute, so this ordering is essential. The resulting $2\times2$ projected vertex `V_irr` has a single quantity fed back into the RPA vertex construction (§14): `V_irr_QQ = V_irr[1,1]`, which additively renormalizes the (retardation-reduced) bare JT pairing vertex, `V_JT_corr = V_JT_eff + V_irr_QQ`.

**Returned dictionary.** `b_mean` (per-site $\langle B_{1g}\rangle$), `Q_fluct` (intra-cluster B₁g fluctuation $\sqrt{\langle B_{1g}^2\rangle-\langle B_{1g}\rangle^2}$ averaged over the 4 sites), `V_irr_QQ`, `V_JT_corr`, `V_irr_QQ_curv` ($=4g_{\mathrm{eff}}^2(1/K_{\mathrm{full}}-1/K_{\mathrm{bare}})$, a curvature-based cross-check, diagnostic only), `chi0_loc`/`chi_loc_full` (projected 2×2 susceptibilities, diagnostic), `K_eff` and `F_can` (§11), `K_corr_const` (= $K_{\mathrm{cl}}-K_{\mathrm{schur}}$; kept in the dictionary but no longer used by the $Q$ equation), `dF_cluster`, and the cluster free energy per site `F_per_site`. The benchmark ratio $\langle B_{1g}\rangle_{ED}/\langle B_{1g}\rangle_{BdG}$ at the bare $g_{JT}$ (`G_JT-BENCH`) and the per-sublattice cluster-vs-BdG disagreement (`DOUBLET-CHK`) are logged by `_scf_kick` (the former `compare_cluster_vs_bdg` method no longer exists); $g_{JT}$ is never rescaled by them.

**Cost.** Each call does three $Q$ stencils of five diagonalizations of the $1296\times1296$ Hamiltonian ($F_{\mathrm{full}}$, $F_{\mathrm{MF}}$, $F_{\mathrm{bare}}$; only the central point of the "full" stencil keeps its eigenvectors, supplying `F_per_site`, `b_mean` and `Q_fluct`), two reference diagonalizations for the susceptibility tensors, and — for $K_{\mathrm{eff}}$ — five canonical-free-energy evaluations (plus nine more for the Schur complement once $\Delta\neq0$), each with a $\mu$ solve. With the exact $Q$ profile of §12 this routine and `solve_Q_slave` dominate the cost of an SCF iteration.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

`_find_mu_for_density` solves $\langle n\rangle = 1-\delta$ by Newton's method using the analytic derivative $\partial n/\partial\mu=\sum_{k,n} w_k f(1-f)/kT\cdot(\vert u\vert ^2+\vert v\vert ^2)$ from the same BdG eigensystem, backtracking on step failure, with Brent's method as a guaranteed fallback bracket-and-bisect. Above `_MU_SC_DERIV_THRESH` total gap amplitude ($\sum_i\vert\Delta_i\vert$) the analytic derivative (exact only for the pure normal-state branches) is replaced by a centered numerical derivative. The `(ev, ec)` pair from the μ-search is reused directly for the subsequent observable computation, avoiding a redundant diagonalization. Because the canonical free energy refits $\mu$ at every probe, this routine is also called inside every finite-difference stencil and every $Q$-scan point.

### 21. Gap Equations: the 84-Amplitude Fixed Point, Complex Phases, and the Irrep Kernel

`VectorizedBdG.compute_gap_eq_vectorized` evaluates the gap equations over the full BZ for **both** cross-doublet branches at once (it returns `(Delta_out, F67_vec, vertex_cache)`: the anomalous amplitudes come from the very eigensystem used for the gap equation), keeping the Fock sums **complex** rather than taking `abs(·)` before forming the new gap — because the BdG Hamiltonian is genuinely complex (SOC enters through $L_y\propto(L_{+}-L_{-})/2i$ and $S_y$), the Nambu eigenvectors are complex at every $k$-point, and forcing a real magnitude at every iteration would erase the physical relative phase between the channels and destabilize convergence. Per iteration:

1. **Raw fixed point.** For each branch, $\Delta_{\mathrm{raw}}[b,i]=\sum_jV_{\mathrm{irrep}}[i,j]\sqrt{g_i/g_j}\,F_{\mathrm{raw},g}[b,j]$ with the branch's Gutzwiller vector (§8). The matrix $V_{\mathrm{irrep}}$ is the sector-block-diagonal projection of §15, with the diagonal of the sign-changing-under-AFM-shift channels replaced by its EMA (§14). (If `lambda_lin_max` $\le0$ the raw update is zero.)
2. **Kernel direction.** The eigenvector shapes `sector_vraw` of the leading covered sector and of every near-degenerate covered sector (weights `seed_w`) are superposed — each sign-aligned with the current candidate $\Delta_{\mathrm{raw}}[7a]$ since a sector eigenvector's sign is arbitrary — and L1-normalized to a direction $v_{\mathrm{irr}}$ over all 42 channels. This keeps the iteration pulled toward whichever symmetry the FS kernel's dominant eigenvector actually carries, even when $\vert\Delta\vert\approx0$, without suppressing a sector that is almost as unstable as the leading one.
3. **Seed threshold and blend.** A branch whose raw amplitude is below `_QQ_DELTA_THRESH` is *seeded*: its magnitude becomes $\max(\text{BCS\_SEED\_FRACTION}\cdot t_{\mathrm{eff}}\,e^{-1/\max(\lambda_{\mathrm{lin,max}},0.1)},\ \text{DELTA\_ABS\_FLOOR})$ with the real kernel signs, and the kernel blend weight is halved. Otherwise the magnitudes are blended, $\vert\Delta\vert=(1-\alpha)\max(\vert\Delta_{\mathrm{raw}}\vert,\ \text{KERNEL\_DIR\_MIN\_FRAC}\cdot\vert\Delta_{\mathrm{kernel}}\vert)+\alpha\vert\Delta_{\mathrm{kernel}}\vert$ with $\alpha=$ `_ALPHA_MIX_2X2` $=0.56$ (the constant name is historical: it now weights the full irrep-kernel direction), and the **complex phases of $\Delta_{\mathrm{raw}}$ are re-applied** — the kernel eigenvector only constrains the real relative signs between channels, so the self-consistent phase is never silently overwritten.
4. **Jump caps.** The per-branch total amplitude is capped at `_DELTA_JUMP_CAP` $\times\Vert\Delta_{\mathrm{vec}}\Vert$ (floor `_DELTA_ABS_FLOOR`), with an exponential penalty $\exp(-\text{DET\_JUMP\_HALF\_SCALE}\cdot\text{depth})$ (depth capped at `_DET_DEPTH_CAP`, cap floored at `_JUMP_CAP_FLOOR`) when the SC-state AFM determinant is negative (past the QCP). Channel ratios are preserved during clamping. The result is the 84-vector `Delta_out`.
5. **Newton step in the full amplitude space.** `compute_dF_dDelta_and_d2F` returns the gradient and Hessian of $F_{\mathrm{bdg}}$ with respect to the 84 amplitude **moduli**, phases held fixed. Active channels use the standard finite-difference gradient/Hessian; **inactive** channels ($\Delta_i=0$) are fitted with a two-point ($\varepsilon,2\varepsilon$) stencil with a larger step (`_HESS_INACTIVE_EPS`), because $F$ need not be even/smooth at $x_i=0$ — bilinear terms $\lvert\Delta_i\rvert\lvert\Delta_j\rvert\cos(\varphi_i-\varphi_j)$ allowed by the global $U(1)$ make $F$ linear in $\lvert x_i\rvert$ as soon as a coupled channel is active, and a single one-sided evaluation would return a bogus $2g/\varepsilon$ curvature. Mixed derivatives to active channels are evaluated only for linearly coupled inactive channels ($\lvert\partial F/\partial\lvert\Delta_i\rvert\rvert>$ `_HESS_LINEAR_TOL`); inactive–inactive cross terms are not evaluated. The step is Levenberg–Marquardt damped through `_lm_damping` with the minimum Hessian eigenvalue (`_MU_LM_DELTA`), clipped to `_TR_DELTA_STEP_MAX`, and combined with the fixed-point update through `_blend_update_candidates`: $\Delta_{\mathrm{new}}=\Delta_{\mathrm{old}}+\alpha\,[(1-\beta)(\Delta_{\mathrm{fix}}-\Delta_{\mathrm{old}})+\beta(\Delta_{\mathrm{newton}}-\Delta_{\mathrm{old}})]$ with $\beta=$ `_ALPHA_HF_DELTA` $=0.75$, i.e. the Newton step is damped by the adaptive mixing rate $\alpha$ as well. The helper is complex-safe: the **modulus** is clipped to $[0,1]$ and the phase kept (a plain `np.clip` on complex numbers orders them lexicographically and would zero every component with a negative real part). The same helper mixes $M$ with $\beta=$ `_ALPHA_HF`.
6. **Anderson acceleration (two blocks, no $Q$).** The primary block $(\vert\Delta_{s,7a}\vert,\vert\Delta_{d,7a}\vert)/t_0$ and the **extended block** of all other 82 amplitude moduli (`_IDX_ANDERSON_EXT`) each get their own Anderson step (history depth 5, shared history list); phases always come from the blended vector. $Q$ is not in the Anderson vector (§12), and $M$ is excluded as well (updated by its own Newton/blend rule).

The same phase-freezing logic is used in `compute_hessian`, which fixes the converged phases and relative amplitudes before taking finite-difference probes.

### 22. Incommensurate AFM Nesting Check and Gap-Symmetry Consistency (Diagnostics Only)

Because the BdG Hamiltonian is fixed to commensurate AFM ordering at $Q_{AFM}=(\pi,\pi)$, `_scan_incommensurate_nesting` separately checks whether the normal-state spin susceptibility $\chi_{SS}$ — evaluated with the staggered Néel-axis spin vertex, which is exact only at $q=(\pi,\pi)$ itself (off-commensurate points are explicitly a "staggered-approximation" scan, not the exact physical $\chi_{SS}(q)$), on a magnetic-BZ-deduplicated FS-weighted sum (`_mbz_deduplication`, `_NAMBU_MBZ_NORMFACTOR`) — would actually prefer a nearby incommensurate wavevector $q^{*}=(\pi,\pi-\delta q)$, scanning 7 points $\delta q\in[0,0.15\pi]$ at the converged $(M,Q,\mu)$. If the maximum lies in the window $0.05\pi<\delta q^{*}<0.10\pi$ the result is flagged ("⚠ incommensurate") and returned (`incommensurate_dq`, `incommensurate_chi_ratio`). It does not change the ordering wavevector used in the Hamiltonian, and **it triggers no re-solve**.

Likewise, the linear-kernel leading channel (`argmax |v_raw|`) is compared with the dominant converged channel (`argmax |Δ_7a|`) only to *label* the result: the `gap_symmetry` string records `[lin=…, SCF=…: consistent]` or `[… nonlinear quench]`, and the log adds `[NL SYMMETRY MIXING]` when both $\Delta_s$ and $\Delta_d$ are finite despite a mismatch.

**The automatic restarts of earlier versions have been removed** (the incommensurate re-solve with a scalar seed, and the d-wave-forced re-solve with `force_d_wave` / `_ic_retry`): an SCF run can take a very long time, so whether to restart from a different seed is left to the user (`solve_self_consistent` accepts an explicit initial $(\Delta,F_{67})$ pair for that purpose; `seed_complex_phase` still rotates the `d7a` seed relative to `s7a`). `_IC_RATIO_FLOOR`/`_IC_RATIO_CAP` remain defined but unused.

### 23. Temperature-Dependent Tc Estimates

Three independent estimates target different aspects of the transition, deliberately not sharing a single label:

- **Tc₁ — Allen–Dynes/McMillan-type spin-fluctuation formula** (`__main__`): $T_{c1}=(\omega_{SF}/D)\cdot\exp(-N\cdot(1+\lambda_{\max})/\lambda_{\max})$, with constants $D=$ `_MAD_DENOM` = 1.13, $N=$ `_MAD_NUM` = 1.04, $\omega_{SF}=J_{\mathrm{eff}}$ (paramagnon bandwidth), and $\lambda_{\max}$ = `lambda_lin_max` of the reference run — a fast analytic estimate, not a full temperature scan; evaluated only if the reference state is SC-viable (not Mott-suspect, $g_t\ge$ `_G_T_COHERENCE_MIN`, converged).
- **Tc₂ — λ(T)=1 crossing:** `compute_lambda_vs_T` re-runs `solve_self_consistent` at each of 20 log-spaced temperatures between $0.25\,kT$ and $4\,kT$ of the reference run, with $\Delta$ **strictly forced to zero** (`initial_Delta = 0`) so $M(T)$ and $Q(T)$ relax self-consistently on a genuinely normal-state background rather than a fixed $T{=}0$ SC-biased one (this avoids the artifact of the AFM Weiss field being frozen at its condensed-state value, which would otherwise keep $\lambda_{\max}$ from ever reaching 1); $T_{c2}$ is where `lambda_lin_max(T)` $=1$, taken as the lowest-temperature crossing if the (generically non-monotone, in a first-order-like regime) curve crosses more than once (a warning is logged). A refined companion, **`compute_Tc_by_gap_suppression`**, is also available (it is **not** called by `__main__`): it locates the linearized instability $\max_\chi\lambda_\chi(T)=1$ over the *covered* $G$-sectors by bracketing on a geometric $T$ grid (extending upward by ×1.6 up to six times if the top of the scan is still unstable) and bisecting; each temperature uses a normal-state `refine_M_state` at $Q=0$, the plaquette vertex and the full sector kernel; it flags near-degenerate leading sectors (the nonlinear BdG decides which symmetry survives, not $\lambda_{\max}$), optionally performs a single SC-vs-normal free-energy comparison at the upper bracket as a first-order flag, and returns the spectral-gap information described below.
- **Tc₃ — thermodynamic, first-order-aware** (`compute_Tc_thermodynamic`): a single upward-heating scan on a geometric temperature grid, warm-started from the converged $T\approx0$ SC+JT basin (the seed advances with the SC basin: $M$, $Q$, $\Delta_{\mathrm{vec}}$), comparing $F_{SC}$ (SC-basin solve at each $T$) against a separately relaxed normal-state free energy at every point. Because the effective Landau potential for the condensate can have a negative effective quartic coefficient here, the transition can be genuinely first-order, and a naive cooling-from-$\Delta{\approx}0$ scan (which only finds the spinodal) can badly underestimate $T_c$. The thermodynamic crossing `T_cross` is located by **linear interpolation** of $F_{SC}-F_{NM}$ between the last SC-wins and first SC-loses temperatures; the spinodal collapse `Tc_spinodal` is the first temperature at which the SC basin vanishes ($\lvert\Delta\rvert<$ `_DELTA_ABS_FLOOR`); for near-second-order cases ($D_{\mathrm{spinodal}}/\Delta_0<$ `_GL_SPINODAL_JUMP`) a Ginzburg–Landau-refined spinodal from fitting $\Delta^2(T)=a(T-T_c)$ to the last 2–4 points with $\lvert\Delta\rvert>$ `_GL_DELTA_MIN` replaces the coarse grid value if it agrees within `_GL_TC_MARGIN`. If a crossing was found, a **cooling pass** from the normal state on a finer grid up to $1.2\,T_{\mathrm{cross}}$ finds where SC first appears (`T_spinodal_cool`) and `hysteresis` $=T_{\mathrm{spinodal,cool}}-T_{\mathrm{spinodal}}$ is reported. The transition order is classified with the relative tolerance `_THRESHOLD_RATIO` ($=0.02$) between `T_cross` and `Tc_spinodal` as `first-order`, `weakly-first-order`, `second-order` or `unknown`, and the returned `Tc` is the thermodynamic crossing for first-order cases (the spinodal for second-order ones). The routine also returns `Delta_jump`, `Q_at_Tc`, the full heating `history` and the spectral-gap record. (`n_bisect` is accepted for API compatibility but the crossing is interpolated, not bisected.) `__main__` calls it with `T_min=1e-4`, `T_max=0.25`, `n_scan=12`.

**Spectral FS gap instead of the amplitude ratio.** The coupling-strength diagnostic no longer divides a pairing amplitude by $T_c$: the pairing amplitudes multiply form factors $\Phi_i(k)$ (up to 2), pair *interorbital* states, and live in an AFM-reconstructed 24×24 problem, so they are not the observable gap. `_bcs_ratio_2D` instead measures the **quasiparticle gap on the normal-state Fermi surface**, $\Delta_{\mathrm{gap}}(k_F)=\min_n\lvert E_n(k_F)\rvert$, with the full 84-component `Delta_vec`: the FS sample (a thermal shell, not the exact $\xi=0$ surface) is reduced to 60 farthest-point-sampled points (uniform coverage of the contour whatever its topology), each point is extrapolated to its own $\xi=0$ crossing ($s_0=-\xi/v_F$) and a 1201-point line search along the FS normal (half-window $\max(0.03\,\mathrm{eV},3\max\lvert\Delta\rvert)/v_F$, capped) finds the minimum; points with $\lvert v_F\rvert$ below `_GAP_VF_MIN` are excluded. It returns `Delta_max`, `Delta_min`, a robust reference gap `Delta_ref` (FS-weighted mean of the top 10% of the spectral gap), the dominant irrep (branches combined in quadrature), a `nodal` flag ($\Delta_{\min}<0.25\,\Delta_{\max}$), the `coupling_regime` label, `T_state` and a `gap_saturated` flag ($kT\le0.25\,T_c$); the fraction of FS points with $\min\lvert E\rvert<$ `_GAP_BFS_TOL` (a Bogoliubov Fermi surface) is used internally to classify the regime. The coupling ratio reported with it is the **measured** one, $2\Delta_{\mathrm{ref}}/k_BT_c$, with $T_c$ the value passed in (`__main__` passes $T_{c3}$). It is classified against thresholds calibrated to the weak-coupling s-wave value $2\pi/e^{\gamma}\approx3.53$ — `BCS-like` ($<$ `_BCS_RATIO_STRONG`=3.8), `very-strong` ($<$ `_BCS_RATIO_EXOTIC`=6.0) or `exotic` — with **no** gap-shape correction (a pure d-wave state therefore lands in the `very-strong` band although it is weak-coupling). The regime is `gapless (…% BFS)` when more than half the FS has a Bogoliubov Fermi surface, and `undetermined (T not << Tc)` when the SCF state is not saturated ($kT>0.25\,T_c$, so $\Delta_{\mathrm{ref}}$ is still below its $T=0$ value). The shape-based baseline of an earlier version is gone (`_BCS_RATIO_S_WAVE` is defined but unused); the antinodal gap is no longer computed (`_GAP_ANTI_PCTL` is unused). The log label "weak-coupling BCS for this gap shape" of the `TC-THERMO` line is legacy wording for this regime.

---

## Model Architecture

```
Pairing registry (module level; import-time checked by _check_registry and _check_bond_content)
    ├── _HARMONIC_LIBRARY / _PAIR_CHANNELS / _IRREP_NAMES     42 form factors, each (D4h irrep, bond kind, parity, φ(kx,ky), scale)
    ├── _IRREP_CHAR, _G_OPS, _G_CHI, _SECTOR_MEMBERS          measured G = {E,T_Q}×D2h characters; 8 sectors; covered/uncovered
    ├── _N_IRREP = 42, _N_DELTA = 2·_N_IRREP = 84             Delta_vec / F67_vec layout; _DELTA_NAMES, _DELTA_IDX, _IDX_7A_ARR, _IDX_7B_ARR
    ├── bond content:    _BOND_GRID, _bond_weights, _bond_class_fractions, _BOND_CLASSES, _TRW_BOND_FRAC   (real-space bond support of every form factor → TRW weights)
    └── helpers: _pairing_basis_matrix, _is_onsite/_is_nn_bond/_is_nnn_bond, _afm_shift_character, _mirror_characters,
                 _gutzwiller_irrep_vector, _pairing_strength_matrix, _pairing_condensation_energy_Nch, _m2_weighted_J,
                 _irrep_decomposition, _sector_label, _sector_report_lines, _check_fs_closure, _grid_sym_permutation

ModelParams  (dataclass, __post_init__ runs the SOC+CF diagonalization and the superexchange cluster ED)
    ├── Primary inputs:  t_pd, t_prime_ratio, t_dprime_ratio, U_dd, lambda_soc, Delta_tetra,
    │                    g_JT, K_lattice, lambda_hop, g_Eg2, K_lattice_Eg2, Delta_CT, Delta_B1g_static,
    │                    hybrid_scale, Upp_ratio_bare, J_H_ratio, M_eff, Z, kT, M_axis, tol, trw_far_scale (default 1.0)
    ├── Derived scalars: Delta_CF, g7split, t0, t_prime, t_dprime, p_7, thermal_fs_nat_size,
    │                    J2_over_J1, J3_over_J1, Z_afm_eff, N_k, U_pp, J_H
    ├── Derived arrays:  sz_op (exact ⟨Sz⟩ per Kramers partner, 6 components),
    │                    S_ops{'x','y','z'} / S_inter / S_axis_op (doublet-projected spin operators; Néel axis),
    │                    B1g_op, B1g_offdiag, Eg2_op, _w_orb (3×3: Γ6/Γ7a/Γ7b rows × xz/yz/xy columns),
    │                    kappa_doublets (3), kappa_cross (2), Tx_A_xz/_yz/_xy (orbital projectors, sum to I₆),
    │                    trw_irrep_scale (42 bond-class weights of the transverse Weiss field, §9)
    ├── Grid objects:    k_points, k_weights, shift_table (_NK×_NK×N_k int32 cyclic shift table for arbitrary q)
    └── Methods:         get_gutzwiller_factors() (static admixture-based estimate),
                         exchange_channels() (returns J_A1g_diag, J_B1g_7a, J_B1g_7b),
                         _exchange_J_q(), effective_hopping_anisotropic(), hopping_matrices(),
                         hopping_matrices_dQ(), wave_function_weight() (T_Q-symmetrized)

_SolveState  (dataclass, mutable per-SCF-run state — never stored on self)
    ├── V_irrep_ema: Optional[ndarray]   # per-channel sign-flip EMA of diag(V_irrep)
    └── _ema_kick_pending: bool          # doubles the EMA blend weight for one iter after a kick

RMFT_Solver
    ├── Initialization: __init__ (derives _omega_0_JT from M_eff and K_bare, Phi_irrep_k on the SCF grid),
    │                   _rebuild_orbital_operators (incl. Néel-axis spin vertices), _get_vbdg,
    │                   _get_chi0_norm_cache, _reset_transient_state, _shallow_clone,
    │                   _clone_solver_at_T, _full_rebuild
    ├── Local H / μ:    build_local_hamiltonian_for_bdg, _find_mu_for_density,
    │                   estimate_gutzwiller_factors_occupation_based (dynamic Γ7a/Γ7b factors, §5),
    │                   _compute_orbital_densities
    ├── JT / Q:         _calc_dHdQ (full explicit ∂H/∂Q, 24×24, with even/odd hole parity and TRW weights),
    │                   solve_Q_slave (total-free-energy profile of Q, §12),
    │                   _eval_B1g_and_kweights_at_Q (scalar ⟨B1g⟩ + k-resolved array at fixed density)
    ├── Observables:    Eg2_expectation, _chi_QQ_matrix_elements (3-point, or 2×2 B1g/Eg2),
    │                   _diamagnetic_QQ_term
    ├── RPA vertex:     _chi0_channels_raw (4×4 channel-resolved raw susceptibility, §14),
    │                   _rpa_closure (Moriya damping + PSD projection + determinant + Stoner factor),
    │                   _rpa_vertex_tables (Ξ, Ξ_cross, V_JT per unique q), _fs_channel_weights,
    │                   _moriya_gamma_landau,
    │                   _make_vertex_params → (Γ_M, F_cluster, V_JT_eff, V_JT_corr, V_cap)
    ├── Pairing kernel: compute_pairing_kernel_and_build_cache (exact G-sector kernels, §15),
    │                   scf_gap_diagnostics (coherence lengths from the exact FS gap shape),
    │                   _vertex_matrix_at_Q
    ├── Free energy:    _compute_bdg_free_energy, _canonical_free_energy (F = Ω+μN, μ refit),
    │                   compute_cluster_free_energy (four-site plaquette ED: Γ_ED vertex, K_eff, dF_cluster(Q), §11/§19),
    │                   _bdg_b1g_per_sublattice,
    │                   compute_dF_dM_channels_and_hessian (3-component FD gradient/Hessian)
    ├── SCF machinery:  _scf_kick (seeding via the early Hessian and the sector kernels),
    │                   refine_M_state (Anderson-accelerated M/μ at fixed Q, Δ_vec, F67_vec; returns a 10-tuple),
    │                   _classify_scf_dynamics, _anderson_mix, _blend_update_candidates (complex-safe Newton/fixed-point blend),
    │                   _relative_phase_change, _project_kick_from_hessian
    ├── Main solve:     solve_self_consistent   ← fixed point over (M[3], Δ_vec[84], μ) with Anderson on the Δ moduli,
    │                   Q profiled out each iteration and relaxed with its own weight
    ├── Post-hoc:       _scan_incommensurate_nesting, compute_hessian ((3+1+1)-dim),
    │                   compute_dF_dDelta_and_d2F (84×84, one entry per amplitude modulus)
    ├── Tc / gap:       _bcs_ratio_2D (spectral FS gap), compute_Tc_by_gap_suppression,
    │                   compute_Tc_thermodynamic, compute_lambda_vs_T
    └── Diagnostics:    _get_fs_points

VectorizedBdG   (thin batched-LAPACK wrapper bound to one RMFT_Solver)
    ├── _build_H_stack                     builds & Hermitizes the (N_k, 24, 24) BdG stack from an
    │                                       84-component Delta_vec and an 84-component F67_vec
    │                                       (even channels antisymmetrized, odd channels symmetric)
    ├── compute_channel_staggered_magnetizations   → per-channel (Γ6, Γ7a, Γ7b) M_stag
    └── compute_gap_eq_vectorized           → (Delta_out (84,), F67_vec (84,), vertex_cache) — both branches through
                                               the shared V_irrep, each with its own Gutzwiller vector
                                               and anomalous amplitudes (§8, §15, §21)

plot_ground_state_comparison(results, labels=None)   — standalone function, not a class method;
    builds the 2×2 "ref vs. normal vs. SC_Q0" free-energy/order-parameter comparison figure
    described in Installation & Usage / Output & Diagnostics. It is a plotting routine only.
```

Module-level functions supporting the superexchange cluster ED (§4) — `_apply_hop`, `_multiorbital_site_of`, `_multiorbital_2hole_hamiltonian`, `_multiorbital_reference_vector`, `_lowdin_branch_energies`, `_intra_doublet_kappa`, `_cross_doublet_kappas` — the module-level `_compute_F67_pair_amplitudes` (all 84 anomalous amplitudes and their raw Gutzwiller-weighted counterparts in one call, §2), the numerical helpers `_fd_grad_hessian` (the single finite-difference gradient/Hessian core shared by the $M$-, $(M,Q,\Delta)$-, $\Delta$- and $(Q,\Delta)$-Hessians), `_lm_damping` (Levenberg–Marquardt damping shared by the $M$ and $\Delta$ Newton steps), `_q_fd_step`, `_adjugate`, `_lehmann_kernel`, `_fermi_function`, `_get_nambu_spinors`, `_build_H_AB_block`, `_expand_M_channels`, `_wave_project`, `_mbz_deduplication`, `_scf_result_reliability`/`_scf_result_hessian_min`, and the unused-by-default `_gamma_splitting` (a fast standalone $\Delta_{CF}$ pre-scan) and `_unique_q_pairs` (a $q$-sampling helper that can write the debug plot `unique_q_hist.png` when called with `verbose=True`; the pairing kernel now builds its own unique-$q$ table over the eight $G$-images, so a normal run never produces that plot) round out the module. `compute_cluster_free_energy` contains the closures `_cluster_stiffness`, `_build_H_cluster`, `_mu_state_at` and `_dF_cluster` (§19).

---

## Key Algorithms

### SCF Loop (`solve_self_consistent`)

A fixed point over $(M_{\Gamma_6},M_{\Gamma_{7a}},M_{\Gamma_{7b}},\ \Delta_{\mathrm{vec}}[84],\ \mu)$ with Anderson(5) acceleration on the gap moduli, the anomalous amplitudes $F_{67}$ taken from the gap-equation pass, and the lattice coordinate $Q$ profiled out at every step and relaxed with its own weight. The signature is `solve_self_consistent(target_doping, initial_Delta, initial_F67, Q_Eg2=0.0, force_delta_zero, force_Q_zero, seed_complex_phase, verbose)`; the initial $(\Delta,F_{67})$ pair is supplied by the caller (all zeros in `__main__`; the kick then seeds the gap). After the seeding stage (`_scf_kick`, below) the iteration starts at $Q=$ `_Q_SEED_THR` ($Q=0$ with `force_Q_zero`) and, per iteration:

1. Recompute the occupation-based Gutzwiller factors `g_Delta_s`, `g_Delta_ad`, `g_Delta_bd` (§5) and the exchange tensors $J_{A1g}$, $J_{B1g}^{(7a)}$, $J_{B1g}^{(7b)}$ (§6); then `_make_vertex_params` — the retardation factor (§14), the four-site cluster ED with its $K_{\mathrm{eff}}$ stencils and the $Q$-energy closure (§11, §19), $J_{\mathrm{eff}}$, $\Gamma_M$, `V_JT_eff`, `V_JT_corr`, `V_cap`.
2. Evaluate the SC-state RPA closure at $q=(\pi,\pi)$ and $q=0$ (`_rpa_closure`) for the determinants, susceptibilities and Stoner factors, and update the instability indicator $\Lambda_{\mathrm{inst}}$ (an EMA, weight `_EMA_NEW_WEIGHT`, of the worst of: the largest entry of the Gutzwiller-scaled pairing-strength matrix $V_{\mathrm{irrep}}\circ\sqrt{g_ig_j}$, $V_{JT,\mathrm{corr}}\chi_{QQ}$, and the Stoner factor, each from the cached normal-state vertex).
3. **Gap sector.** The 84-dimensional Newton step (`compute_dF_dDelta_and_d2F`, §21 step 5) and the gap-equation update `compute_gap_eq_vectorized` (§21 steps 1–4), which also returns the new anomalous amplitudes `F67_vec_new`; the candidate $\Delta$ is formed by `_blend_update_candidates` with $\beta=$ `_ALPHA_HF_DELTA` $=0.75$ (modulus clipped, phase kept).
4. **Profile out $Q$:** $Q_{\mathrm{out}}=Q^*$ from `solve_Q_slave` (§12), unless `force_Q_zero`; $Q_{\mathrm{mixed}}=Q+\alpha_Q(Q_{\mathrm{out}}-Q)$ with `_ALPHA_Q = 0.5`.
5. **Anderson on the gap moduli** (§21 step 6): the primary block $(\vert\Delta_{s,7a}\vert,\vert\Delta_{d,7a}\vert)$ and the extended block of the remaining 82 moduli each get their own Anderson(5) step; phases are kept from the blended vector. Neither $Q$ nor $M$ is in any Anderson vector.
6. **Newton step for $M$** (three channels): the finite-difference gradient/Hessian of the canonical free energy (`compute_dF_dM_channels_and_hessian`), Levenberg–Marquardt-damped through the softest eigenvalue (with a near-critical $\mu$ that shrinks as $\vert\Delta\vert$ grows), trust-region-limited by a stiffness cut $\propto J_{\mathrm{eff}}/t_{\mathrm{eff}}$ and a curvature factor, with per-channel step floors, combined with the BdG fixed point $M_{\mathrm{BdG}}$ from the staggered-magnetization observable by `_blend_update_candidates` with $\beta=$ `_ALPHA_HF` $=0.85$ (so the Newton step is damped by $\alpha$ too).
7. $\mu$ is re-solved exactly (Newton+Brent) at the freshly mixed $(M,Q_{\mathrm{mixed}},\Delta,F_{67})$ rather than carried as an imperfectly converged fast variable.
8. **Adaptive mixing rate** (every 5th iteration from iteration 5): $\alpha_{\mathrm{base}}=\mathrm{clip}\big(\alpha_0/(1+0.5\Lambda_{\mathrm{inst}}),\ \alpha_0/6,\ \alpha_0\big)$; a diverging step halves $\alpha$ and clears the history, stagnation clears it after two hits (and mildly boosts $\alpha$ in the subcritical plateau), converging steps apply a mild SC+JT boost/decay; a **limit-cycle detector** (relative std of the total $\sum_i\vert\Delta_i\vert$ over the last `_CYCLE_WINDOW` iterations $>$ `_CYCLE_THRESHOLD`) damps $\alpha$ by `_CYCLE_DAMP_FAC`, clears the history and classifies the regime (`limit_cycle`, `first_order_jump`, `hysteretic`; otherwise `converging`). A $Q$ displacement larger than `_Q_THR_REL·lambda_hop` caps $\alpha$ at $0.3\,\alpha_0$. A negative SC-state AFM determinant (past the QCP) applies an exponential penalty to $\alpha$ and boosts $\Lambda_{\mathrm{inst}}$; near the QCP ($\det<$ `_RPA_DET_WARN` or $\vert V_d\vert>$ `_V_CUT`) $\alpha$ is capped at $\alpha_0/(1+\Lambda)$; a freeze counter triggers an $\alpha$-recovery boost. A sign flip of $Q$ clears the Anderson history and the vertex cache (§12).
9. **Saddle-escape kick** (iteration $>3$, every 8th iteration, only while the total gap is still below $5\times$`tol`): `compute_hessian` is evaluated at the current point; a negative scaled smallest eigenvalue triggers a step along its eigenvector, scaled by `_KICK_BASE_FRACTION`, the curvature and the damping $1/(1+\Lambda_{\mathrm{inst}})$. **The step is applied if the soft mode has a significant gap component, `frac_D > _KICK_MIN_DELTA_FRAC` (0.25)** — `pure-SC` or a mixed Δ–Q mode (§16): $M$, $Q$ and the gap amplitudes are modified (the gap kick keeps the per-channel ratios and phases), the Anderson history and the vertex cache are cleared and the EMA is marked stale. In `pure-SC` mode a cheap Γ₆-only Stoner estimate (a normal-state `refine_M_state` at $Q=0$, $\Delta=0$) is computed, and when $M_{\Gamma_6}$ exceeds twice that estimate all three $M$ channels are pulled back toward it uniformly instead of adding the eigenvector step.

Convergence requires $\max\big(\vert\Delta M\vert,\ \vert Q_{\mathrm{out}}-Q\vert,\ \max_i\big\vert\vert\Delta_i\vert_{\mathrm{new}}-\vert\Delta_i\vert_{\mathrm{old}}\big\vert,\ \text{relative phase change},\ \max_i\big\vert\vert F_{67,i}\vert_{\mathrm{new}}-\vert F_{67,i}\vert_{\mathrm{old}}\big\vert\big)<$ `tol` — all 84 gap and 84 anomalous amplitudes; the $Q$ term is the *residual* of the $Q$ profile, not the relaxed step, and the phase term is gauge-invariant (`_relative_phase_change`, sensitive to drifts of e.g. the $s+id$ phase that the moduli alone miss) —, density error $<10\cdot$`tol`, at least `_MIN_ITER` iterations, a non-negative SC-state AFM determinant, and no kick applied on the same iteration. After convergence (or exhausting `_MAX_ITER = 500` iterations) the solver runs the post-convergence stage: the Hessian, the FS-resolved $\partial\lambda/\partial Q$ and gap-symmetry diagnostics, the incommensurate-nesting scan (diagnostic only, §22), the Mott filter (§5), **$K_{\mathrm{eff}}$ of the returned state** (two further `_make_vertex_params` calls: SC and $\Delta=0$ at the same $(M,Q,\mu)$, §11), the Eg,2 stability diagnostic (§7) and the QCP flag — `past_qcp` is the sign of the SC-state AFM determinant of the last loop evaluation (the state at the *start* of the final iteration); if it is negative, `converged` is reported as `False` even if the iteration tolerance was met (`scf_iteration_converged` keeps the raw flag). The result dictionary is consumed by the diagnostics block described in [Output & Diagnostics](#output--diagnostics). There are **no automatic restarts**: a long run is never silently re-solved from another seed.

### Vectorized BdG, Buffer Reuse, and the χ₀(q) Permutation Trick

`VectorizedBdG._build_H_stack` assembles the entire $(N_k,24,24)$ Hamiltonian stack with vectorized NumPy operations and diagonalizes it in a single `np.linalg.eigh` call per SCF iteration, reusing a pre-allocated `out=` buffer to avoid repeated allocation across hundreds of iterations; Hermiticity is enforced after assembly. The per-iteration eigensystem `(ev, ec)` is computed once and shared by the observables and the gap equations of both branches.

The $q$-loop inside the RPA vertex construction never re-diagonalizes: the uniform k-grid (`endpoint=False`) is built in `ModelParams.__post_init__` so that for any $q=(n_x,n_y)\cdot2\pi/\mathrm{\_NK}$, the $k+q$ grid is exactly a cyclic permutation of the $k$-grid. A precomputed `shift_table[nx, ny]` (shape `(_NK, _NK, N_k)`, `int32`) turns "shift by $q$" into a free index reorder,

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]     # index reorder — no extra LAPACK call
```

reusing the *same* $\Delta=0$ eigensystem for every $q$-point in one vertex-cache rebuild. `_get_chi0_norm_cache` additionally memoizes this normal-state $(E_k,V_k)$ across separate calls (susceptibilities, rigidity, incommensurate-nesting scan, Gutzwiller occupations) that fall within the same iteration, keyed on $(M,Q,\mu,g_t,g_J,n_{\mathrm{kspace}})$: the $M$ tolerance is `_M_THR_REL` relative to $\max(\Vert M\Vert,1)$, the $Q$ tolerance is `_Q_THR_REL`$\cdot\lambda_{hop}$, and $\mu$, $g_t$, $g_J$ and the density are checked at $10^{-4}$ (relative for $\mu$), $10^{-4}$, $10^{-4}$ and $10^{-6}$ respectively.

### Vertex Cache Invalidation

The RPA vertex cache is rebuilt when $\max_c\vert\Delta M_c\vert$ exceeds `_M_THR_REL`$\cdot\sqrt{\vert\det_{\mathrm{afm,sc}}\vert}$ (a threshold that tightens toward the QCP, where the vertex is most sensitive), when $Q$ moves by more than $\max($`_Q_THR_REL`$\cdot\lambda_{hop},10^{-4})$, or unconditionally if the cached determinant sign disagrees with the freshly computed SC-state determinant sign (a proxy for having crossed the QCP since the cache was built). A sign flip of $Q$ or a saddle-escape kick also discards the cache. There is no Δ-based invalidation — the vertex is *always* built from $\Delta=0$ by construction (§14). The cache stores the RPA determinants (`det_q0`, `det_afm`, `det_afm_current`), the susceptibilities and Stoner factors, FS geometry (`fs_pts`, `vF_arr`, `fs_idx`, `fs_weights`), and the pairing-kernel products of §15 (`V_irrep`, `W_irrep`, `S_irrep`, `V_parts_diag`, `sector_*`, `frac_seed`, …), so repeated calls within one iteration reuse the same Fermi-surface sampling.

### Limit-Cycle Detection

Independent of the adaptive-$\alpha$ rules, a dedicated oscillation check monitors the total $\sum_i\vert\Delta_i\vert$ (over all 84 channels) over a rolling window of `_CYCLE_WINDOW = 20` iterations; when the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.25`, the mixing rate is cut by `_CYCLE_DAMP_FAC = 0.45` (floored at $\alpha_0/16$) and the Anderson history reset, which specifically targets the strongly nonlinear regime near the JT-activation onset where the $(Q,\Delta)$ feedback is most prone to overshoot. The regime is classified by the max/min ratio over the window (`limit_cycle` if $<2$; otherwise `first_order_jump` or `hysteretic` depending on whether the late-window mean has moved relative to the early one) and reported as `scf_dynamics_regime`; it is a diagnostic label and a mixing-rate control — it does not by itself launch a multi-seed restart.

### Initial-Condition Seeding (`_scf_kick`)

Before the main SCF loop, `_scf_kick(target_doping, Delta_vec, F67_vec, seed_complex_phase, verbose)` first calls `refine_M_state(target_doping, Q, Δ_vec, F67_vec, …)` at $Q=0$ — a normal-state AFM seed when the passed $\Delta$ is zero (it returns `(M, Δ_vec, F67_vec, n_kspace, μ, g_t, g_J, g_Δs, g_Δad, g_Δbd)`) — a fixed number `_MAX_ITER_FOR_REFINE_M` of Anderson-accelerated iterations at damping 0.45, starting from $M=0.01$ on every channel, with $\mu$ refit and the anomalous amplitudes recomputed each pass; no early exit). It then builds the cheap cluster vertex and RPA vertex parameters and the linearized pairing eigenproblem (`lambda_lin_max`, `frac_seed`, from `compute_pairing_kernel_and_build_cache`), and evaluates the (3+1+1)-dimensional Hessian (`compute_hessian`) at that normal-state seed. If its smallest scaled eigenvalue is negative, the seed is stepped along its eigenvector (`_project_kick_from_hessian`, scaled by `_EARLY_KICK_BASE` plus the linear-eigenvalue excess) and the **full 84-component gap seed is the kernel eigenvector shape** `frac_seed` (all channels and both branches, near-degenerate sectors included) times a BCS-like total amplitude $\mathrm{clip}\big(0.5\,t_{\mathrm{eff}}/\sinh(1/\max(\lambda_{\mathrm{lin,max}},0.1)),\ \text{DELTA\_ABS\_FLOOR},\ \text{KICK\_DELTA\_MAX\_FRAC}\cdot t_{\mathrm{eff}}\big)$ (the kick does not seed $Q$: `solve_self_consistent` starts at `_Q_SEED_THR`; `_KICK_Q_SEED` is defined but unused); otherwise a more conservative Stoner-proximity heuristic shrinks $M$ toward the RPA-safe region (`_KICK_REDUCTION_AMP`, centers `_KICK_M_EXCESS_CTR`/`_KICK_JCHI_EXCESS_CTR`) and seeds the same shape directly. `seed_complex_phase` rotates the `d7a` seed relative to `s7a` (seeding the weaker one at 50% if it is missing). The kick returns `M_kick`, `Delta_vec`, `F67_vec`, `n_kspace`, `mu`, `alpha`, `lambda_lin_max`, `B1g_exp` (⟨B1g⟩ of the seed state) and `K_eff_n` ($K_{\mathrm{eff}}$ of the seed state; a normal-state value only if the passed $\Delta$ was zero, §11); with `verbose` it also logs `KICK-INIT`, `KICK-STATE`, the `G_JT-BENCH` ratio and `DOUBLET-CHK` (the cluster-vs-BdG ⟨B1g⟩ checks formerly done by `compare_cluster_vs_bdg`). Either way this lands the iteration in the basin of the physically correct fixed point, and sets the starting mixing rate $\alpha=\max(\text{KICK\_MIXING\_FLOOR},\ \text{MIXING}/(1+\text{KICK\_MIXING\_SCALE}\cdot\log1p(\lambda_{\mathrm{lin,max}})))$. The Anderson solver itself (`_anderson_mix`) is a scale-normalized, Tikhonov-regularized (`_ANDERSON_TIKHONOV`) normal-equation solve with a condition-number guard, a residual-acceptance test against simple mixing, a trust-region cap (`_ANDERSON_TRUST`$\times$ the simple step) and a blend weight between `_ANDERSON_W_LO` and `_ANDERSON_W_HI` set by $\alpha/\alpha_0$.

### Thread-Safety and Clone Protocol

The current `__main__` runs its three parallel SCF tasks (§ Installation & Usage) as three independently constructed `RMFT_Solver` instances (each built from a deep copy of `params`) rather than clones of a single solver, but the underlying clone protocol remains available for any code that mutates parameters mid-run (and is used internally by the Tc routines' temperature scans, via `_clone_solver_at_T`, which also sets `kT`, the temperature-scaled FS sample budget, and resets the transient caches):

```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s._full_rebuild()
```

`_full_rebuild()` is the single canonical post-mutation refresh: it re-runs `p.__post_init__()` (SOC+CF diagonalization and the superexchange cluster ED), updates the bare stiffness `_K_bare` and the derived bare phonon energy `_omega_0_JT` (§14), rebuilds every orbital operator, and resets all transient caches (`_reset_transient_state`). Each clone owns its own `VectorizedBdG` and its own `_H_stack` buffer, so concurrent workers never alias each other's memory. At import time the module pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to 1 (via `setdefault`, so an explicit user setting wins) to prevent the BLAS backend from oversubscribing threads underneath an outer `ThreadPoolExecutor`.

### Finite-Difference ∂F/∂M and ∂²F/∂M²

The Newton step for $M$ (SCF Loop step 6) needs the free-energy gradient and curvature with respect to the three channel-resolved $M$ components. `compute_dF_dM_channels_and_hessian` obtains them by **finite differences of the canonical free energy** through the shared `_fd_grad_hessian` core (step $\varepsilon_c=\max(10^{-4},\,10^{-3}\vert M_c\vert)$; 3-point diagonal and 4-point cross stencils; the gradient reuses the diagonal's $F(x\pm\varepsilon)$), with $\mu$ refit and the dynamic Gutzwiller factors re-evaluated at every probe — so the curvature is that of the *fixed-density* problem and includes the density-dependent renormalizations. (Earlier versions obtained these analytically from first- and second-order perturbation theory on the already-computed eigensystem; that code path has been replaced.) The same core serves `compute_hessian` ($n=5$) and `compute_dF_dDelta_and_d2F` ($n=84$, with the active/inactive treatment of §21).

---

## Parameters

All energies in **eV**, lengths in **Å**. Defaults below are the values set in the current `__main__` block, verified directly against the source.

### Primary Inputs (`ModelParams`)

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | $t_{pd}$ | 0.335 eV | $p$–$d$ hybridization integral; the single primary hopping input ($t_0=t_{pd}^2/\Delta_{CT}$ is derived) |
| `t_prime_ratio` | — | −0.20 | 2nd-neighbor (diagonal) hopping ratio; $t'=$ this $\times\,t_0$ |
| `t_dprime_ratio` | — | 0.10 | 3rd-neighbor (axial) hopping ratio; $t''=$ this $\times\,t_0$ |
| `U_dd` | $U_{dd}$ | 2.300 eV | On-site Hubbard repulsion — a **primary** input, not derived from a dimensionless ratio |
| `lambda_soc` | $\lambda_{SOC}$ | 0.020 eV | Atomic SOC constant on the $t_{2g}$ shell; sets the Γ₆–Γ₇ splitting together with `Delta_tetra` |
| `Delta_tetra` | $\Delta_{\mathrm{tetra}}$ | −0.035 eV | Axial (tetragonal) crystal field, $\Delta_{\mathrm{tetra}}\cdot L_z^2$; negative = $z$-axis compression |
| `g_JT` | $g_{JT}$ | 0.254 eV/Å | B₁g electron–phonon (JT) coupling |
| `K_lattice` | $K$ | 3.500 eV/Å² | Bare B₁g phonon spring constant; `K_eff` is computed at runtime (§11) |
| `lambda_hop` | $\lambda_{\mathrm{hop}}$ | 1.200 Å | Hopping-anisotropy decay length: $t(Q)=t_0\exp(\pm Q/\lambda_{\mathrm{hop}})$ |
| `g_Eg2` | $g_{Eg2}$ | 0.100 eV/Å | Eg,2-channel electron–phonon coupling (§7) |
| `K_lattice_Eg2` | $K_{Eg2}$ | 6.500 eV/Å² | Bare Eg,2 phonon spring constant |
| `Delta_CT` | $\Delta_{CT}$ | 2.000 eV | Charge-transfer gap |
| `Delta_B1g_static` | $\Delta_{\mathrm{ip}}$ | +1×10⁻⁵ eV | Static in-plane crystal field, $(L_x^2-L_y^2)$; drives the D₄h→D₂h crossover (§1, §3); any non-zero value lifts the exact D₄h cancellation of the linear $Q\vert\Delta\vert^2$ coupling |
| `hybrid_scale` | — | 5.000 | Downfolding coordination factor entering the ZRS spectral weight $\beta^2(k)$ (§6, §9) |
| `Upp_ratio_bare` | — | 0.400 | Bare (unhybridized) $U_{pp}/U_{dd}$ ratio entering the internally-computed $U_{pp}$ (§4) |
| `J_H_ratio` | — | 0.10 | Kanamori Hund's coupling for the virtual $d^2$ superexchange intermediate state, as a fraction of $U_{dd}$: $J_H = $ this $\times\,U_{dd}$ (§4) |
| `M_eff` | — | 16.0 amu | Normal-mode reduced mass for the B₁g phonon; sets the bare phonon energy $\hbar\omega_0$ used by the adiabatic retardation factor (§14). For this octahedral B₁g-type distortion the metal ion sits close to a symmetry-imposed node, so the mode is usually ligand-dominated |
| `Z` | $Z$ | 4 | Coordination number |
| `kT` | $k_BT$ | 0.005 eV | Temperature ($\approx$ 58 K) |
| `M_axis` | — | `'z'` | Direction of the staggered (Néel) moment: `'x'`, `'y'` (∥ the two metal–ligand bond directions) or `'z'`. The AFM Weiss field, the $M$ observables, the spin vertices of the susceptibilities and the cluster exchange use the doublet-projected spin $S_{\mathrm{axis}}$ (§1) |
| `tol` | — | $10^{-4}$ | SCF convergence threshold |
| `trw_far_scale` | — | 1.0 | Multiplier of the transverse-Weiss-field weight of every **non-NN** bond class (NNN, 3NN): 1 = the $(t_\delta/t)^2$ estimate, 0 = far-bond TRW off, $>1$ = stress test (§9). A field with a default, so it is not set in `__main__` |

The `__main__` block uses a single target doping `target_doping = 0.149`.

### Derived Quantities (from `__post_init__`)

| Quantity | Origin | Description |
|---|---|---|
| `Delta_CF`, `g7split` | SOC+CF diagonalization | Γ₆–Γ₇ₐ gap and Γ₇ₐ–Γ₇ᵦ internal splitting — **not** free parameters |
| `sz_op` | exact $S_z$ diagonalization in each Kramers doublet | 6 components $[sz_{6\uparrow},sz_{6\downarrow},sz_{7a\uparrow},sz_{7a\downarrow},sz_{7b\uparrow},sz_{7b\downarrow}]$ |
| `S_ops`, `S_inter`, `S_axis_op` | doublet projection of $S_x,S_y,S_z$ | Néel-axis spin operators, block-diagonal per doublet (`S_axis_op` $=$ `diag(sz_op)` for `M_axis='z'`) |
| `p_7` | Γ₇ admixture in the Γ₆ eigenvectors | Interpolates the *static* estimate `g_Delta_ad` in `get_gutzwiller_factors`; the SCF loop instead uses the dynamic, occupation-based estimate of §5 |
| `_w_orb` (3×3) | eigenvector projections | $d_{xz}/d_{yz}/d_{xy}$ orbital weights (rows Γ6/Γ7a/Γ7b) feeding the exchange anisotropy and the `Tx_A_*` hopping projectors |
| `kappa_doublets` (3), `kappa_cross` (2) | multi-orbital cluster ED, §4 | Intra-doublet (Γ6, Γ7a, Γ7b) and cross-doublet (Γ6–Γ7a, Γ6–Γ7b) exact-diagonalization exchange coefficients |
| `Tx_A_xz/_yz/_xy` | orbital-character projectors | Rigorous orbital-selective hopping building blocks (§6); sum to $I_6$ |
| `t0`, `t_prime`, `t_dprime` | $t_{pd}^2/\Delta_{CT}$ and the `*_ratio` inputs | Effective nearest/2nd/3rd-neighbor $dd$ hopping |
| `U_pp`, `J_H` | `Upp_ratio_bare`, `J_H_ratio` (§4) | Ligand hole–hole repulsion and absolute Hund's coupling; used to build the superexchange cluster Hamiltonian and **stored** as attributes |
| `J2_over_J1`, `J3_over_J1`, `Z_afm_eff` | $(t'/t)^2$, $(t''/t)^2$, $Z(1-J_2/J_1-J_3/J_1)$ | Further-neighbor exchange ratios and frustration-reduced AFM coordination (§6) |
| `trw_irrep_scale` (42) | `_TRW_BOND_FRAC` · class weights $(1,1,J_2/J_1\,f,J_3/J_1\,f,0)$ | Bond-class weight of the transverse Weiss field of every pairing channel (§9) |
| `thermal_fs_nat_size` | $\max(\lfloor kT/t_0\cdot$`_NK`$^2\rfloor,\,32)$ | Adaptive Fermi-surface sample-point budget (clones at other temperatures use $\lfloor T/t_0\cdot N_K^2\rfloor$) |
| `k_points`, `k_weights`, `N_k`, `shift_table` | $N_k=$ `_NK`² uniform grid | k-space infrastructure, including the cyclic shift table used for arbitrary-$q$ Lindhard sums |

`B1g_op`, `B1g_offdiag`, `Eg2_op` are also set on `ModelParams` in `__post_init__` (§3, §7); the corresponding 24×24 Nambu lifts `B1g_24`, `Eg2_24`, plus `Sz_nambu`, the per-channel staggered `Sz_stag_nambu_channels` and uniform `Sz_nambu_channels`/`_sz_nambu_diag_channels` vertices, are built on the `RMFT_Solver` instance by `_rebuild_orbital_operators`, along with `phi_k = cos(kx) − cos(ky)`, the full form-factor matrix `Phi_irrep_k` ($N_k\times42$) on the SCF grid, and the bare phonon energy `_omega_0_JT` (§14).

### Module-Level Constants

The source file documents essentially every numerical-methods constant inline, each with its own physical or numerical justification — that block remains the single source of truth. The tables below reproduce every value relevant to interpreting solver behavior and output, transcribed directly from the current module-level constant block. Constants marked † are still defined but no longer referenced by any routine (left over from machinery that has been replaced).

**Grid, iteration budget, pairing basis, general safety**

| Constant | Value | Role |
|---|---|---|
| `_NK` | 72 | k-points per direction (even, for commensurate $q_{AFM}=(\pi,\pi)$); $N_k=72^2=5184$ |
| `_MAX_ITER` / `_MIN_ITER` | 500 / 4 | SCF iteration ceiling / floor before a convergence check is even attempted |
| `_MAX_ITER_FOR_REFINE_M` | 100 | Fixed iteration count for the `refine_M_state` Anderson loop (no early-exit check) |
| `_MIXING` | 0.15 | Base mixing weight (the adaptive rate $\alpha$ lies between $\alpha_0/6$ and $\alpha_0$, further reduced near the QCP / on limit cycles) |
| `_N_ORB` / `_N_BDG` | 6 / 24 | Orbital flavors (full Γ₆⊕Γ₇ₐ⊕Γ₇ᵦ manifold) / BdG Nambu dimension ($4\times$`_N_ORB`) |
| `_N_CHANNELS` / `_CLUSTER_SIZE` / `_N_CLUSTER` | 3 / 4 / $6^4{=}1296$ | Channel resolution (Γ₆,Γ₇ₐ,Γ₇ᵦ) / sites in the plaquette cluster ED / cluster Hilbert-space dimension |
| `_N_IRREP` / `_N_CH67` / `_N_DELTA` | 42 / 2 / 84 | Pairing form factors in the library / Γ₇ branches / length of `Delta_vec`, `F67_vec` |
| `_MATH_EPS` | $10^{-9}$ | General division-by-zero guard |
| `_BZ_NORM` | $(2\pi)^2$ | BZ-area normalization in the FS integration measure |
| `_Q_UNIQUE_SCALE` | $10^5$ | Integer scaling used to hash unique $q$ vectors without floating-point collisions |
| `_BOND_GRID` | 16 | Periodic grid of the FFT that extracts the real-space bond content of every pairing form factor (exact for bonds with $\lvert\delta_{x,y}\rvert<8$; the library reaches 5) |
| `_NAMBU_MBZ_NORMFACTOR` | 0.125 | Normalization of the magnetic-BZ nesting sum (½ sublattice × ½ Nambu × ½ Hermitization) |
| `_NORMAL_SECTOR_PAIRS` † | 8 slice pairs | Nambu sector pairs of the normal-state Lindhard sum |

**Unit conversion & Gutzwiller prefactors**

| Constant | Value | Role |
|---|---|---|
| `_EV_TO_K` | 11604.518121 K/eV | $1/k_B$ |
| `_GW_G_J_PREFACTOR` | 4.0 | Numerator in $g_J=4/(1+\delta)^2$ (slave-boson / Kotliar–Ruckenstein, half-filling limit) |
| `_GW_G_T_NUMERATOR` | 2.0 | Numerator in $g_t=2\delta/(1+\delta)$ |
| `_JT_PHONON_MEV_CONST` | 64.6528 | $\hbar\omega_0[\mathrm{meV}]=$ this $\times\sqrt{K[\mathrm{eV/\mathring A^2}]/M[\mathrm{amu}]}$ — exact unit-conversion constant, not a fitted number (§14) |

**AFM Newton solver ($M$-step control)**

| Constant | Value | Role |
|---|---|---|
| `_MU_LM` | 3.0 | Levenberg–Marquardt floor for the $M$ Newton step |
| `_ALPHA_HF` | 0.85 | Newton weight $\beta$ of `_blend_update_candidates` for $M$ (the Newton step is additionally damped by $\alpha$) |
| `_TR_M_STEP_MAX` / `_TR_M_STEP_MIN_FLOOR` | 0.1 / $10^{-3}$ | Trust-region cap / absolute floor on $\vert \Delta M\vert $ per step |
| `_M_STEP_FLOOR_REL` / `_M_STEP_FLOOR_ABS` | 0.005 / 0.002 | Step floor $=\max($ `_M_STEP_FLOOR_REL` $\cdot\vert M\vert ,$ `_M_STEP_FLOOR_ABS`$)$ per channel |
| `_M_J_EFF_FLOOR_FRAC` | 0.20 | QCP guard: $J_{\mathrm{eff}}$ floored at this fraction of $t_{\mathrm{eff}}$ to prevent $\Delta M\propto1/J_{\mathrm{eff}}\to\infty$ |

**Q handling.** $Q$ is profiled out by `solve_Q_slave` (13-point scan over $\pm0.15$ Å with adaptive extension, §12); those numbers are arguments of the function, not module constants. `_ALPHA_Q` (below) is the relaxation weight of $Q$ toward its profile. `_Q_THR_REL` (below) controls the cache invalidation and the mixing-rate cap after a large $Q$ displacement. The former Newton-step constants `_TR_Q_STEP_FRAC` and `_TR_Q_STEP_MIN_FLOOR` have been removed; `_Q_LM_FRAC` (0.08) † is left defined but unused.

**Δ Newton solver step control**

| Constant | Value | Role |
|---|---|---|
| `_MU_LM_DELTA` | 3.0 | Levenberg–Marquardt floor for the 84-channel Δ Newton step |
| `_ALPHA_HF_DELTA` | 0.75 | Newton weight $\beta$ of `_blend_update_candidates` for the Δ update (likewise damped by $\alpha$) |
| `_TR_DELTA_STEP_MAX` | 0.1 eV | Upper bound on the Δ Newton step per channel per iteration |
| `_HESS_DELTA_EPS_MIN` | $10^{-5}$ | Floor of the combined-amplitude finite-difference step of `compute_hessian` (step $=\max(\text{this},10^{-3}\vert\Delta_{\mathrm{tot}}\vert)$) |
| `_HESS_INACTIVE_EPS` | $4\times10^{-4}$ | Minimum finite-difference step for exactly-zero amplitudes (two-point gradient+curvature fit) |
| `_HESS_LINEAR_TOL` | $2\times10^{-5}$ | $\vert\partial F/\partial\lvert\Delta_i\rvert\vert$ above which a zero-amplitude channel counts as linearly coupled to an active one |

**Saddle-escape / initial-condition seeding**

| Constant | Value | Role |
|---|---|---|
| `_MODE_FRAC_DOMINANT` | 0.60 | Fractional-weight threshold for a pure-channel soft mode (§16) |
| `_KICK_MIN_DELTA_FRAC` | 0.25 | The saddle-escape kick is applied only if the Δ fraction (`frac_D`) of the soft mode exceeds this (`pure-SC` and mixed Δ–Q modes) |
| `_MODE_PULL_FRAC` | 0.30 | Fraction of $(M-M_{\mathrm{phys,est}})$ used as the kick pull in `pure-SC` mode |
| `_KICK_BASE_FRACTION` | 0.05 | Base trust-region step-size fraction for the mid-run saddle-escape kick |
| `_KICK_M_EXCESS_CTR` / `_KICK_JCHI_EXCESS_CTR` | 0.70 / 0.70 | Onset centers for $M$-excess / Stoner-excess overshoot suppression in the seed kick |
| `_KICK_REDUCTION_AMP` | 3.88 | $M_{\mathrm{kick}}=M\times(1-\text{this}\times\text{excess})$ |
| `_KICK_M_CLIP_LO` / `_KICK_M_CLIP_HI` | $10^{-9}$ (`_MATH_EPS`) / 0.9 | Hard clips on $M_{\mathrm{kick}}$ |
| `_KICK_DELTA_MAX_FRAC` | 0.4 | Maximum seed gap as a fraction of the effective hopping scale $t_{\mathrm{eff}}$ |
| `_KICK_MIXING_FLOOR` / `_KICK_MIXING_SCALE` | 0.004 / 4.0 | Minimum kick mixing weight / damping scale in $\alpha=$`_MIXING`$/(1+\text{scale}\cdot\log1p(\lambda_{+}))$ |
| `_Q_SEED_THR` | $2\times10^{-4}$ Å | Initial $Q$ of the SCF loop (0 with `force_Q_zero`) |
| `_KICK_Q_SEED` † | $10^{-2}$ Å | Former initial-$Q$ seed of a `pure-JT` kick; no longer referenced |
| `_EARLY_KICK_BASE` | 0.01 | Base step fraction of the early (seed-time) Hessian kick |

There is no `estimate_M0`/`_M0_*` empirical-prior warm start: the normal-state AFM seed comes from a plain Anderson-accelerated fixed point (`refine_M_state`).

**Chemical potential (Newton + Brent)**

| Constant | Value | Role |
|---|---|---|
| `_DEN_DERIV_FLOOR` | $10^{-12}$ | Floor on $\partial n/\partial\mu$ |
| `_BRENTQ_TOL` | $10^{-6}$ | Brent bracketing tolerance |
| `_MU_NEWTON_MAXIT` / `_MU_BACKTRACK_MAX` / `_MU_BACKTRACK_FLOOR` | 20 / 6 / 0.05 | Newton iteration budget / max step-halvings / minimum backtrack damping before falling back to Brent |
| `_MU_DENSITY_TOL` | $10^{-8}$ | $\vert n(\mu)-n_{\mathrm{target}}\vert$ convergence tolerance |
| `_MU_SC_DERIV_THRESH` | $10^{-4}$ eV | Total gap amplitude above which the analytic $\partial n/\partial\mu$ (exact only at $\Delta=0$) is replaced by a centered numeric derivative |

**Lindhard broadening & Fermi-surface sampling**

| Constant | Value | Role |
|---|---|---|
| `_ETA_T_FRAC` | 0.10 | Thermal broadening $\eta=$ this $\times\,kT$ (cluster susceptibilities and the nesting scan) |
| `_ETA_DELTA_FRAC` | 0.02 | SC-state broadening increment $\propto\sum_i\vert\Delta_i\vert$ |
| `_ETA_GRID_FLOOR` | 0.001 | Broadening floor (units of $t_0$), guards k-grid aliasing |
| `_FERMI_ARG_CLIP` | 100.0 | Numerical clip in $f(E)$ |
| `_FD_MASK_DF` / `_FD_MASK_DE` | $10^{-12}$ / $10^{-6}$ | Degenerate-denominator masks in the $\chi_0$ Lehmann kernel |
| `_FS_SAMPLING` / `_FS_THERMAL_THRESHOLD` | 4.4 / 0.0025 | Thermal window (in units of $kT$) around $E_F$ / minimum relative thermal weight kept for the FS shell |
| `_FS_CACHE_TOL` | $10^{-3}$ | Parameter-change tolerance for FS-point cache invalidation |
| `_C4_SPECTRAL_TOL` | $10^{-9}$ eV | C₄ is an exact symmetry of the normal-state spectrum if $\max\vert E_n(k)-E_n(Rk)\vert$ is below this (selects the $D_{4h}$ vs $D_{2h}$ FS sampling) |
| `_SYM_THRESH` | $10^{-8}$ | Symmetry-breaking floor for `Delta_B1g_static` / $\vert Q\vert$ classification |
| `_ODD_VTX_TOL` | $10^{-9}$ | Relative tolerance of the odd-parity (p-channel) FS-closure/weight-symmetry validation |
| `_MBZ_DEGEN_FRAC` | $2\times10^{-3}$ | Energy tie-break scale (fraction of $kT$) for magnetic-BZ Fermi-surface point deduplication |
| `_NODAL_REGION_PCTL` | 25 | Percentile split (upper/lower 25%) for nodal/antinodal FS decomposition of the gap shape |
| `_PHI_D_FLOOR` | $10^{-3}$ | Minimum gap-shape maximum to enable the nodal/antinodal split (legacy $\varphi_d$ path) |
| `_VERTEX_DIAG_MIN_FS` | 10 | Minimum FS points required before vertex/nodal diagnostics are considered reliable |

The Fermi-surface sample budget is the dynamically computed `thermal_fs_nat_size` (see Derived Quantities); the FS integration weight is a symmetry-adapted thermal 2D element with no separate Fermi-velocity floor constant (§15).

**Pairing-kernel construction**

| Constant | Value | Role |
|---|---|---|
| `_SECTOR_NEAR_FRAC` | 0.80 | Covered sectors with $\lambda_{\mathrm{top}}\ge$ this fraction of the leading one are "near-degenerate": their eigenvector shapes also seed the kick (`frac_seed`) |
| `_V_REPULSIVE_FLOOR_FRAC` | $10^{-2}$ | Eigenvalues of the pairing matrix below this fraction of its largest count as "no attraction" in the condensation energy (stiff, finite penalty) |
| `_SECTOR_SCAN_MIN_GAIN` | 0.02 | Library scan: a candidate harmonic is suggested only if it raises the resolved weight of the sector eigenvector by at least this much |

**RPA vertex & QCP tracking**

| Constant | Value | Role |
|---|---|---|
| `_RPA_BW_FACTOR` | 8.0 | Tight-binding bandwidth estimate $=8t$ |
| `_RPA_V_CAP_ALPHA` | 2.2 | Headroom multiplier for the dynamic vertex cap $V_{\mathrm{cap}}$ |
| `_RPA_DET_WARN` | 0.11 | $\vert \det_{\mathrm{afm}}\vert $ below this ⇒ QCP-proximity warning, feeds adaptive mixing |
| `_RPA_QCP_PENALTY` | 0.40 | Mixing-rate reduction per unit $\vert \det_{\mathrm{afm}}\vert <0$ past the QCP |
| `_DET_AFM_FLOOR` | 0.5 | Default `det_afm` when no vertex cache exists yet |
| `_DK_CORR_CAP_MULT` | 1.0 | Cap on the cluster stiffness correction (§11) (and on the unused Δ-relaxation correction), relative to the bare stiffness $K_{\mathrm{bare}}$ |
| `_DET_DEPTH_CAP` / `_DET_JUMP_HALF_SCALE` / `_JUMP_CAP_FLOOR` | 5.0 / 0.5 / 1.05 | Past-QCP gap-jump cap: exponential suppression depth cap / decay rate / minimum allowed cap |
| `_DET_SIGN_FLIP_SCALE` | 0.05 | $\vert \det_{\mathrm{afm}}\vert $ sigmoid midpoint for the $V_{\mathrm{irrep}}$ sign-flip EMA guard |
| `_EMA_SIGN_FLIP_W_MIN` / `_EMA_SIGN_FLIP_SLOPE` | 0.20 / 6.0 | Minimum blend weight / sigmoid steepness in the sign-flip guard |
| `_V_PREV_SIGN_FLOOR` | $10^{-6}$ | EMA diagonal below this is treated as zero (sign-flip check skipped) |
| `_V_CUT` | 20.0 | Pairing-vertex near-divergence detector threshold |
| `_JCHI_HARD_REJECT` | 2.0 | Cap on the Stoner factor entering the instability indicator |
| `_QQ_DELTA_THRESH` | $10^{-8}$ | $\sum\vert\Delta\vert$ threshold below which a branch counts as unseeded (seed-magnitude branch of the gap update); also the normal-state threshold of the Δ-Hessian |

**Moriya damping, JT diagnostics, finite-difference steps**

| Constant | Value | Role |
|---|---|---|
| `_MORIYA_LANDAU_M_STEP` | 0.06 | $M$-probe step for the self-consistent, Landau-expansion-derived $\Gamma_M$ (§14) |
| `_LAMBDA_JT_VIABLE` † | 0.05 | Former $\lambda_{JT,\mathrm{sc}}$ viability threshold (§17); no longer referenced |
| `_JT_ACT_THR` | 0.04 | Threshold on the condensate-induced selection ratio for the "JT-active" flag in the `SCF-RES` summary |
| `_DQ_FS_VERTEX` / `_DQ_FS_VERTEX_FRAC` | 0.03 Å / 0.05 | Minimum / adaptive-fraction finite-difference step for $\partial\lambda/\partial Q$ on the FS |
| `_JT_FD_H2_BASE` / `_JT_FD_H2_QCOEF` | $3\times10^{-8}$ / $6\times10^{-7}$ | $Q$-derivative FD step schedule, $h(Q)=\sqrt{\text{base}+\text{qcoef}\cdot Q^2}$, shared by every $Q$-derivative (`_q_fd_step`) |

**Limit-cycle detection & Anderson mixing**

| Constant | Value | Role |
|---|---|---|
| `_CYCLE_WINDOW` / `_CYCLE_THRESHOLD` / `_CYCLE_DAMP_FAC` | 20 / 0.25 / 0.45 | Rolling window / relative-std trigger / mixing-rate cut on a detected limit cycle |
| `_ALPHA_Q` | 0.5 | Relaxation weight of $Q$ toward its profile $Q_{\mathrm{out}}$ ($Q$ is kept out of the Anderson vectors) |
| `_ANDERSON_TIKHONOV` | $10^{-8}$ | Tikhonov regularization in the Anderson normal equations |
| `_ANDERSON_TRUST` | 2.4 | Trust-region cap on the Anderson step (multiples of the simple-mixing step) |
| `_ANDERSON_W_LO` / `_ANDERSON_W_HI` | 0.3 / 0.8 | Blend-weight bounds between Anderson and simple mixing |

**SCF regime classification (freeze / recover / diverge / stagnate)**

| Constant | Value | Role |
|---|---|---|
| `_SCF_DIVERGE_RATIO` / `_SCF_STAGNATE_RATIO` | 1.05 / 0.95 | $\max\vert \Delta\vert$ growth ratio thresholds classifying the step as diverging / stagnating |
| `_SCF_ALPHA_DECAY` / `_SCF_ALPHA_RECOVER` | 0.95 / 1.60 | Mixing-rate multiplier while converging (mild damping) / on freeze-recovery |
| `_SCF_FREEZE_THR` | 10 | Consecutive frozen iterations that trigger freeze-recovery |
| `_SCF_ALPHA_FREEZE_LO` / `_SCF_ALPHA_FREEZE_HI` | 0.15 / 0.60 | $\alpha/$`_MIXING` bounds defining "too frozen" / the recovery ceiling |
| `_SCF_ALPHA_CONVG_BOOST` / `_SCF_ALPHA_CONVG_CAP` | 1.09 / 0.75 | Mixing-rate boost / ceiling while SC+JT active and converging |
| `_EMA_NEW_WEIGHT` | 0.12 | EMA weight for $\Lambda_{\mathrm{inst}}$ and the $V_{\mathrm{irrep}}$ sign-flip guard |
| `_Q_THR_REL` | $2\times10^{-5}$ | Fraction of `lambda_hop`; a $Q$ change above this invalidates the $\chi_0$ cache and caps the mixing rate at $0.3\,\alpha_0$ (the vertex cache itself uses $\max(\text{this}\cdot\lambda_{hop},10^{-4})$). Because it is far below the finite-difference steps, the $\chi_0$ cache is effectively reused only for identical $Q$ |
| `_M_THR_REL` | 0.01 | $M$-change threshold (scaled by $\sqrt{\vert\det_{\mathrm{afm,sc}}\vert}$ for vertex-cache invalidation; relative to $\max(\Vert M\Vert,1)$ for the $\chi_0$ cache) |
| `_ALPHA_MIX_2X2` | 0.56 | Blend weight of the irrep-kernel eigenvector direction vs. the fixed-point gap update, both branches (§21; historical name) |

**Gap (Δ) update and coherence-length classification**

| Constant | Value | Role |
|---|---|---|
| `_BCS_SEED_FRACTION` | 0.1 | Cold-start $\Delta$ seed in the gap equation, as a fraction of $t_{\mathrm{eff}}$ (times $e^{-1/\lambda}$) |
| `_DELTA_JUMP_CAP` | 5.0 | Maximum $\Vert\Delta_{\mathrm{new}}\Vert/\Vert\Delta_{\mathrm{old}}\Vert$ ratio per iteration (per branch, total amplitude) |
| `_DELTA_ABS_FLOOR` | $10^{-4}$ eV | Amplitude below this bypasses the jump limiter (free seed-growth phase); also the "SC collapsed" threshold of the Tc routines |
| `_KERNEL_DIR_MIN_FRAC` | 0.5 | The kernel-direction amplitude gives a floor (this fraction) for each channel's magnitude in the blend |
| `_XI_NODAL_MIN` | 2.0 | Minimum $\xi/a$ (nodal) for BCS-side quasiparticle coherence |
| `_ORBITAL_SEL_FRAC` | 0.15 | $\vert \xi_{\Gamma_6}-\xi_{\Gamma_7}\vert /\xi$ threshold for "orbitally selective" pairing |
| `_IC_RATIO_FLOOR` / `_IC_RATIO_CAP` † | 1.05 / 3.00 | Bounds of the former incommensurate-nesting retry diagnostic; no longer referenced |

**Superexchange cluster ED validity guards**

| Constant | Value | Role |
|---|---|---|
| `_WIGNER_ECKART_DIV_FLOOR` | $10^{-6}$ | Floor under the $4s_{\mathrm{eff}}^2$ / $\Vert M\Vert_F^4$ denominators in the κ extraction (§4) |
| `_S_EFF_FLOOR` | 0.15 | Floor on the effective spin $s_{\mathrm{eff}}$ in $\kappa_c/(4s_{\mathrm{eff}}^2)$ (spin-switch regime protection) |
| `_ANISO_WARN_TRESH` | $5\times10^{-4}$ | Non-Heisenberg component magnitude above which `_intra_doublet_kappa` logs a warning |
| `_B1G_BLOCK_MIN_FRAC` | 0.02 | Minimum share of $\Vert B_{1g,\mathrm{op}}\Vert_F^2$ a Γ6↔Γ7 block must carry for `_cross_doublet_kappas` to resolve it |

**Tc / spectral gap / Ginzburg–Landau**

| Constant | Value | Role |
|---|---|---|
| `_BCS_RATIO_S_WAVE` † | 3.528 | Weak-coupling BCS ratio $2\pi/e^{\gamma}$ for an isotropic gap — the baseline the two thresholds below were calibrated to; no longer referenced |
| `_BCS_RATIO_STRONG` / `_BCS_RATIO_EXOTIC` | 3.8 / 6.0 | Weak-coupling-ratio thresholds separating `BCS-like` / `very-strong` / `exotic / non-phononic` |
| `_MAD_DENOM` / `_MAD_NUM` | 1.13 / 1.04 | Allen–Dynes-type strong-coupling denominator (Millis–Monien–Pines spin-fluctuation value) / exponent prefactor |
| `_THRESHOLD_RATIO` | 0.02 | Relative difference between `T_cross` and `Tc_spinodal` for the first-order classification |
| `_GL_DELTA_MIN` | $2\times10^{-3}$ eV | $\vert \Delta\vert $ floor for points admitted to the GL fit |
| `_GL_MIN_PTS` / `_GL_MAX_PTS` | 2 / 4 | Minimum / maximum recent stable-SC points used in the GL regression |
| `_GL_TC_MARGIN` | 0.05 | Maximum relative deviation $\vert T_{c,GL}-T_{\mathrm{spinodal}}\vert /T_{\max}$ to accept the GL result |
| `_GL_SPINODAL_JUMP` | 0.15 | $D_{\mathrm{spinodal}}/\Delta_0$ below this ⇒ GL extrapolation treated as reliable (small first-order jump) |
| `_GAP_N_SUB` | 60 | FS points (farthest-point sampled) on which the spectral gap is searched |
| `_GAP_NLINE` | 1201 | Points of the line search per FS point |
| `_GAP_WIN` / `_GAP_WIN_KMAX` | 0.030 eV / 2.0 rad | Minimum line-search half-window (widened to $3\max\vert\Delta\vert$, converted through $v_F$) / cap of the half-window in $k$ |
| `_GAP_SCENTER_MAX` | 2.0 rad | Cap on the first-order shift $s_0=-\xi/v_F$ from a sample point to the $\xi=0$ crossing |
| `_GAP_VF_MIN` | $10^{-5}$ | FS points with $\vert v_F\vert$ below this are excluded from the gap statistics |
| `_GAP_REF_TOP_FRAC` | 0.10 | $\Delta_{\mathrm{ref}}=$ FS-weighted mean of the largest this-fraction of the spectral gap |
| `_GAP_NODAL_ANISO` | 0.25 | $\min/\max$ spectral gap below this ⇒ gap labelled (near-)nodal |
| `_GAP_BFS_TOL` | $10^{-4}$ eV | FS points with $\min\vert E\vert$ below this count as gapless (Bogoliubov Fermi surface) |
| `_GAP_ANTI_PCTL` † | 75 | Percentile of the legacy "antinodal" region of the gap statistics; no longer referenced |

**Physical thresholds (SC viability / Mott)**

| Constant | Value | Role |
|---|---|---|
| `_G_T_COHERENCE_MIN` | 0.10 | Mott guard: minimum coherent $g_t$ ($\delta\gtrsim0.053$), used in the post-SCF Mott filter and in the `__main__` doping floor and Tc-viability test |

Constants listed in earlier versions of this document that **no longer exist** in the source: `_LINDHARD_CHUNK`, `_PI_INT`, `_M_STEP_FLOOR_M_MIN`, `_KICK_BOOST_Q`, `_FD_MASK_DE8`, `_CHI_TAU_ABS_FLOOR`, `_Q_UPDATE_PERIOD`, `_BCS_RATIO_VSTRONG`.

---

## Installation & Usage

### Requirements

```bash
pip install numpy scipy matplotlib opt_einsum
```

(`opt_einsum` is imported at module load; the heavy contractions currently use `numpy.einsum` with `optimize=True`.)

### Running

```bash
python Quantum_AFM-multipolar_Jahn-Teller.py
```

The module is CPU-heavy: per SCF iteration the plaquette ED diagonalizes $1296\times1296$ matrices for three $Q$ stencils plus the susceptibility references, the $K_{\mathrm{eff}}$ stencils run several canonical free-energy evaluations, and the exact $Q$ profile (§12) repeats the cluster ED and a $\mu$ solve at every scan, root and curvature point — on the production $72\times72$ grid a single SCF run can take days to weeks, which is why no automatic restarts are performed. The BLAS thread count is pinned to 1 per worker and the three comparison runs execute concurrently.

On startup, the current `__main__` block does the following, in order:

1. **Parameter setup and SOC+CF diagonalization.** `ModelParams(...)` is constructed with the defaults listed in [Parameters](#parameters) and `__post_init__` runs the SOC+CF diagonalization (Γ₆/Γ₇ₐ/Γ₇ᵦ identification, `Delta_CF`, `sz_op`, spin operators, `p_7`, k-grids, orbital operators) and the multi-orbital superexchange cluster ED (`kappa_doublets`, `kappa_cross`, §4). The `INIT` and `DERIVED` log lines report the inputs, $U_{pp}$, $J_H$, $t_0$, $\Delta_{CF}$, $\kappa_{\Gamma_6}$, the $\Gamma_7$ splitting (flagged if below $2kT$) and whether $\Gamma_{7b}$ lies inside the estimated $\Gamma_6/\Gamma_{7a}$ bandwidth.
2. **Doping setup.** `target_doping = 0.149`, with a symmetric ±20% scan margin (`doping_margin = 0.20`) defining `min_doping`/`max_doping`, floored to stay above the `_G_T_COHERENCE_MIN` Mott-incoherence boundary. These bounds are computed but, in the current `__main__`, not otherwise consumed — all three solves below run at the single point `target_doping`, not a scan. The initial gap and anomalous amplitudes are `np.zeros(_N_DELTA)` (`Delta_vec_ans`, `F67_vec_ans`) — there is no manually chosen cold-start amplitude: `_scf_kick` seeds the 84-component gap from the leading sector eigenvector(s) of the pairing kernel (§ Key Algorithms).
3. **Three-way self-consistent comparison, run in parallel.** Three independent `RMFT_Solver` instances are built from deep copies of `params`, and `solve_self_consistent` is run on each concurrently via a 3-worker `ThreadPoolExecutor`:
   - **`ref`** — the full self-consistent solve, SC (all 84 channels) and $Q$ both free (the model's actual prediction); the only run with verbose logging.
   - **`normal`** — $\Delta$ pinned to zero throughout (`force_delta_zero=True`); the normal (non-superconducting) AFM state.
   - **`SC_Q0`** — SC free but $Q$ pinned to zero (`force_Q_zero=True`); superconductivity without the JT relaxation.

   This is the direct numerical test of the central hypothesis: if the theory is right, `F_bdg` should order `ref < normal` (condensation lowers the free energy) and `ref < SC_Q0` (JT relaxation lowers it further), and `Q` should self-consistently relax back toward 0 in `normal` *without being forced there*. The code logs `"SC+JT is the ground state ✓"` or `"✗"` accordingly — but only when `_scf_result_reliability` passes for all three (each must be converged, **not past the AFM QCP**, and have a Hessian with no negative eigenvalue); otherwise it logs `"Comparison NOT trustworthy"`.
4. **Post-SCF diagnostics** (only if the reference run returned a result): the `REF-SCF` summary (converged flag, raw iteration-convergence flag, Mott-suspect, past-QCP, final AFM determinant, $M$, $Q$, $\Delta_s$, $\Delta_d$, $\mu$), the selection ratio ("Irrep R") and its JT ALLOWED/BLOCKED verdict, `K_eff_net` (§11), two stability lines — $K_{\mathrm{lattice}}$ vs. $K_{\mathrm{spont}}=g_{JT}^2\chi_{QQ}$ at the converged **SC state**, and at $\Delta=0$, $Q=0$ with $\mu$ re-solved (the normal-state criterion, §13/§16/§17) — `lambda_lin_max`, the coherence-length summary, the post-convergence Hessian's smallest eigenvalue ("SC-triggered JT CONFIRMED" if negative), and the three Tc estimates (§23) with the spectral-gap record of Tc₃.
5. **One diagnostic plot.** `plot_ground_state_comparison(results)` produces the `ref`/`normal`/`SC_Q0` comparison figure, saved to disk (see [Output & Diagnostics](#output--diagnostics)).

---

## Output & Diagnostics

All output is a structured, thread-safe log stream (`_scf_log`, tagged by stage — `INIT`/`DERIVED`, `REF-SCF`, `PARALLEL`, `KICK-INIT`/`KICK-STATE` (the seeding stage), `SCF-A/B/C` (the per-iteration blocks), `SCF` (α-recovery), `SCF-RES`, `SCF-EG2`, `Q-SLAVE` (edge/extension/curvature notices of the $Q$ search), `REFINE-M`, `SADDLE-ESC`, `LIMIT-CYCLE`, `FREE-ENERGY`, `GAP-DIAG`, `PAIR-DIAG`, `SECTOR`, `DOUBLET-CHK`, `G_JT-BENCH`, `TC-PRELIM`, `TC-LAMBDA`, `TC-LIN`, `TC-THERMO`, `TC-COOL`, `RMFT-WARN`, `PLOT`) rather than a GUI; the graphical output is the one `matplotlib` figure described below. (The startup banner text of `__main__` is legacy wording from an earlier optimizer-based pipeline and does not describe the current flow.)

### Iteration Log

Each logged SCF step (verbose run) is split into three lines:

```
[SCF-A] iter/max  conv=…  M=[…]  Q=…  J_eff=… eV  mu=…  g_Delta_ad=…  g_Delta_bd=…  Γ_M=…meV  F_bdg=…  F_cluster=…  K_eff=…  Q_fluct=…  |Δ|₂=…
[SCF-B]   Delta_vec: top-4 amplitudes (of 84), nact=…   F67_vec: top-4   V_diag: top-4 signed diag of V_irrep   V_sd=…  lead=(sector)
[SCF-C]   dFM_sc=…  dAFM=…  dAFM_sc=…  χSQ_sc(π,π)=…  χSQ_sc(q=0)=…  stoner_q0(N)=…  stoner_q0(SC)=…  V_JT*χQQ(q=0)=…  δB1g=…   α=…  …s/it
```

with `KICK-INIT` reporting the kicked seed ($M$, $Q_0$, $\vert\Delta\vert_0$, Gutzwiller factors, the regime `subcritical/critical/supercritical` from `lambda_lin_max`, the metallic-AFM check, the Fermi-level orbital weights of the three doublets and `K_eff_n`) and `PAIR-DIAG`/`GAP-DIAG`/`SECTOR` blocks reporting the spin/JT/cross decomposition of the pairing diagonal, the irrep-projection capture of the dominant FS eigenvector (with an explicit warning if more than 25% lies outside the 42-function basis), the leading-channel weights, and a table of the eight $G$-sector eigenvalues with the library harmonics that would close any basis gap.

At convergence, `SCF-RES` lines report: $\partial\lambda_{\mathrm{pair}}/\partial Q$(FS) and the hot-spot fraction, `lambda_lin_max`, the per-channel $\lambda_{\mathrm{irrep}}$ and $\vert\Delta_{7a}\vert$ (meV) with the gap-symmetry label and the linear-vs-SCF consistency, the unresolved-weight fraction, the incommensurate-nesting scan result (a flag only, §22), the free energies, the JT flag, the Hessian eigenvalues (✓MIN or ⚠SADDLE), the SCF-dynamics regime and the ⚠ ANSATZ UNSTABLE / PAST-QCP flags, and $\chi_{SQ}^{SC}(\pi,\pi)$ with its strength label; `SCF-EG2` reports the Eg,2 stability diagnostic (§7).

### Free-Energy Ground-State Check

Logged under the `FREE-ENERGY` tag right after the three parallel tasks (`ref`, `normal`, `SC_Q0`) complete: `F_bdg` and the post-convergence Hessian minimum eigenvalue for each of the three, flagged `⚠ UNRELIABLE (reason)` individually when `_scf_result_reliability` fails (reasons: not converged, past the AFM QCP, no Hessian, saddle), and a final one-line verdict — `"SC+JT is the ground state ✓"`, `"SC+JT is NOT the lowest energy state ✗"`, or `"Comparison NOT trustworthy"` if any of the three results is unreliable.

### RPA Vertex, Coherence, and SC-JT Diagnostics

If the reference SCF produced a result: the channel-resolved pairing diagnostics above; the coherence-length summary (flagging orbital-selective pairing when $\Gamma_6$ and $\Gamma_7$ channels have meaningfully different $\xi$ — computed from the exact FS gap magnitude $\vert\Delta(k)\vert$ and the nodal/antinodal regions of the gap *shape*); the post-convergence Hessian's smallest eigenvalue; `K_eff_net`; and the two `K_spont` stability lines (§13, §17). The Stoner factors are logged per iteration (`SCF-C`), not as a separate `__main__` line.

### Ground-State Comparison Plot

`plot_ground_state_comparison(results)`, a standalone module-level function (not a solver method), takes the `{"ref", "normal", "SC_Q0"}` result dictionary produced in [Installation & Usage](#installation--usage) step 3 and saves a **2×2** figure to `ground_state_comparison.png`: (top-left) `F_cluster` per SCF iteration for all three scenarios, each with its converged `F_bdg` as a dotted reference line, unreliable trajectories drawn dashed with the reason in the legend; (top-right) a bar chart of the three final `F_bdg` values, titled with whichever scenario is lowest (unreliable bars hatched in red); (bottom-left) the Γ₆-channel AFM order parameter $M[\Gamma_6]$ versus iteration, all three overlaid; (bottom-right) $\vert Q\vert$ versus iteration, the key visual check that `normal`'s $Q$ relaxes back toward 0 on its own (it is never forced there) while `ref` settles at a finite value. Missing scenario entries are skipped rather than erroring, provided at least one entry is present. The debug-only hexbin plot of the old unique-$q$ sampling (`unique_q_hist.png`, from `_unique_q_pairs(verbose=True)`) is not produced by the current pipeline.

---

## Known Limitations

The framework makes a number of physically motivated approximations. The table below reflects the current code, not an earlier design pass.

| Approximation | Impact |
|---|---|
| No Pauli exclusion between plaquette sites | The four-site cluster's external bonds are closed with a mean-field Weiss embedding rather than genuine Pauli-respecting hopping to the rest of the lattice; mild overestimate of AFM correlations |
| No charge-transfer fluctuations $\langle n_An_B\rangle$ | Negligible when the mean-field exchange scale is large compared to the hopping |
| Phonon still adiabatic/quasi-static | $Q$'s own equation of motion is a thermodynamic equilibrium condition, not a dynamical one, and zero-point lattice fluctuations are neglected; the phonon-mediated pairing vertex carries an adiabatic retardation factor tied to a physical $\hbar\omega_0$ derived from `M_eff` (§14), but this is a one-point static-limit evaluation at the FS-maximum gap scale, not a full frequency-dependent (Matsubara) treatment |
| No thermal/quantum fluctuation correction to the lattice stiffness | The Moriya-SCR regularization of $K_{\mathrm{eff}}$ that earlier versions applied to the lattice channel is gone; $Q$ is the exact minimizer of the mean-field canonical free energy (§12), so thermal fluctuations of $Q$ itself are not renormalized (the spin channel keeps its Landau-derived $\Gamma_M$) |
| Γ₇ₐ remains the headline channel | Both Γ₆–Γ₇ₐ and Γ₆–Γ₇ᵦ pairing are fully self-consistent (§8), but several reported summary quantities (`Delta_s`, `Delta_d`, `selection_ratio`, `Delta_irrep`, the primary Anderson block) are built from the Γ₇ₐ amplitudes; the Γ₇ᵦ channel's own physics requires the second half of `Delta_vec`/`F67_vec` (`Delta_vec[_IDX_7B_ARR]`) |
| Basis completeness is checked, not guaranteed | The 42-harmonic library covers all eight $G$-sectors only up to its truncation order; an uncovered sector, or a covered one whose leading eigenvector has a large `unresolved_weight`, signals a basis gap (reported by `SECTOR`/`GAP-DIAG`); `lambda_lin_max_all` can exceed `lambda_lin_max` when an *uncovered* sector is the more unstable one. Odd-parity ($p$-type) channels are additionally gated by an FS-closure check and are disabled in the kernel if it fails |
| Linear kernel vs. nonlinear outcome | The leading linear sector need not be the symmetry the nonlinear BdG selects (near-degenerate sectors are seeded together, and a mismatch is reported in the `gap_symmetry` label). No automatic re-seeding is performed for any mismatch; restarting from another seed is left to the user |
| Resolution mismatch between the RPA and cluster-ED layers | The itinerant RPA vertex (§14) resolves three separate spin channels (Γ6, Γ7a, Γ7b); the local cluster-ED irreducible-vertex extraction (§19) works in a 2-channel (aggregate spin, B₁g) space, uses only the leading Γ₆ channel's $M$ and $J_{A1g}[0]$ for its embedding and exchange, and feeds back only the on-site anomalous amplitudes into its Weiss field, so `V_irr_QQ` does not resolve Γ7a- vs. Γ7b-specific or bond-channel-specific local corrections |
| No spatial fluctuations | Cannot describe a pseudogap, stripe order, or phase separation |
| RPA static ($\omega=0$) | Dynamical vertex corrections beyond the JT-channel retardation factor above are absent; the spin channels have no analogous retardation treatment |
| Cluster stiffness correction is capped and uses a NN-only mean-field reference | The beyond-mean-field stiffness correction $K_{\mathrm{cl}}$ is clipped at `_DK_CORR_CAP_MULT`$\times K_{\mathrm{bare}}$ in $K_{\mathrm{eff}}$ (the exact $\Delta F_{\mathrm{cl}}(Q)$ of the $Q$ profile is not capped). $K_{\mathrm{MF}}$ is built with $Z_{\mathrm{MF}}=Z$ (the cluster has no $J_2/J_3$ bonds), whereas the bulk longitudinal Weiss field uses $Z_{\mathrm{afm,eff}}=Z(1-J_2/J_1-J_3/J_1)$ — a ≈5% mismatch of the mean-field level that is not propagated into the cluster (it shifts $\Delta K$ by a few percent in a test state). The cluster also uses a single Γ₆-channel $M$ and only the on-site anomalous amplitudes |
| $Q$ profile is a coarse scan + bracketed root with an expensive objective | The 13-point scan over $\pm0.15$ Å (adaptively extended, at most three times) may miss a narrow basin; if the minimum stays on the edge the grid value is returned, and if the force has no sign change in the bracketing interval the scan point (resolution $\approx0.025$ Å) is returned. The objective includes the exact cluster energy, so every probe costs a cluster ED and a $\mu$ solve; the Eg,2 amplitude is held at 0 in the search; without a vertex cache the condensation energy uses a diagonal fallback matrix |
| Simplified normal-state stability check | `K_lattice` vs. `K_spont` is exact for the pure-JT instability direction but does not itself diagnose a genuinely *mixed* (e.g. AFM–JT coupled) normal-state instability; the post-convergence Hessian mode classification (§16) carries that role, but only after a solve has converged near the relevant point. The `K_spont(Δ=0)` evaluation uses the SC run's $M$ (not the normal state's own self-consistent $M$) with $\mu$ re-solved |
| $\partial\lambda_{\mathrm{pair}}/\partial Q$ at a frozen Fermi surface | FS geometry is evaluated at a fixed $Q$ rather than self-consistently re-resolved; a fully SC-state Bogoliubov–Lindhard version would be more expensive |
| SCF-dynamics regime classification | `first_order_jump`, `hysteretic` and `limit_cycle` are heuristic labels from the shape of the $\sum\vert\Delta_i\vert$ history; they only damp the mixing rate and clear the Anderson history — no multi-seed restart is launched |
| Eg,2 channel not self-consistent | `Q_Eg2` is a fixed input (default 0); the Eg,2 phonon is wired into the Hamiltonian, the free energy and a post-convergence stability diagnostic, but its exchange-driven rigidity correction and cross-rigidity with the B₁g channel are not computed |
| Incommensurate AFM handled only as a diagnostic | `_scan_incommensurate_nesting` (a staggered-vertex approximation off $(\pi,\pi)$) detects a preference for $q^{*}\neq(\pi,\pi)$ and flags it; the BdG Hamiltonian itself remains fixed at commensurate $(\pi,\pi)$ ordering and no re-solve is triggered — a genuinely incommensurate spiral solve is not implemented |
| Sign-flip EMA on the pairing diagonal | Suppresses numerical oscillation in the diagonal of $V_{\mathrm{irrep}}$ for the channels that change sign under the AFM shift, but may slow the genuine response near a doping-driven crossover between channels |
| Superexchange cluster is a 3-site chain with independent ligand modes | `_multiorbital_2hole_hamiltonian` (§4) models the ligand as three independent per-$L_z$ modes rather than a single fully hybridized ligand orbital; this preserves $J_z$ through the SOC mixing but is itself an approximation to the true ligand electronic structure; the intra-doublet $\kappa$ is additionally floored through `_S_EFF_FLOOR` in the spin-switch regime |
| Cluster-ED vertex is a $q=0,\omega=0$ local estimate | `V_irr_QQ` from the four-site plaquette (§19) is a local quantity standing in for a genuinely $k$-dependent, dynamical vertex; together with the $K_{\mathrm{eff}}$ stencils and the exact $Q$ profile the cluster evaluation is the dominant cost of an SCF iteration |
| Tc diagnostics | The reported coupling ratio is $2\Delta_{\mathrm{ref}}/k_BT_c$ with the thresholds of the weak-coupling s-wave value and no gap-shape correction (a pure d-wave state lands in the `very-strong` band) and is marked `undetermined` unless the SCF state is saturated; Tc₃'s crossing is interpolated on a coarse temperature grid, and `compute_Tc_by_gap_suppression` is not called by `__main__` (its optional free-energy check passes the Δ seed also as the $F_{67}$ seed) |
| TRW bond weights are a hopping-counting estimate | The weight of the transverse Weiss field of a longer-range pairing channel follows $(t_\delta/t)^2$ (§9); the bulk transverse field uses the bare $Z$ while the longitudinal field uses $Z_{\mathrm{afm,eff}}$. `trw_far_scale` allows stress-testing the far-bond weights |
| No automatic restarts | The incommensurate re-solve and the d-wave-forced re-solve were removed because a single SCF run can take very long; the user decides whether to restart from a different seed |

---

## References

- Ecsenyi, S. (2026). *Multipolar superconductivity and Jahn–Teller activation in strongly correlated systems: a self-consistent theoretical framework* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang, F.C. et al. (1988). *Supercond. Sci. Technol.* 1, 36; Bünemann, J., Weber, W. & Gebhard, F. (1998). *Phys. Rev. B* 57, 6896.
- ZSA classification: Zaanen, J., Sawatzky, G.A. & Allen, J.W. (1985). *Phys. Rev. Lett.* 55, 418.
- Kanamori multi-orbital interaction: Kanamori, J. (1963). *Prog. Theor. Phys.* 30, 275.
- Quasi-degenerate perturbation theory / effective Hamiltonians: Löwdin, P.-O. (1951). *J. Chem. Phys.* 19, 1396; des Cloizeaux, J. (1960). *Nucl. Phys.* 20, 321; Bloch, C. (1958). *Nucl. Phys.* 6, 329.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge University Press.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.
- Allen–Dynes strong-coupling formula: Allen, P.B. & Dynes, R.C. (1975). *Phys. Rev. B* 12, 905; basis: McMillan, W.L. (1968). *Phys. Rev.* 167, 331.
- Ginzburg–Landau theory: Ginzburg, V.L. & Landau, L.D. (1950). *Zh. Eksp. Teor. Fiz.* 20, 1064.
- Cluster-DMFT-style irreducible-vertex extraction (χ0⁻¹−χ⁻¹): Maier, T. et al. (2005). *Rev. Mod. Phys.* 77, 1027.
- Moriya spin fluctuations and self-consistent renormalization: Moriya, T. (1985). *Spin Fluctuations in Itinerant Electron Magnetism.* Springer.
- Nearest positive-semidefinite matrix projection: Higham, N.J. (1988). *Linear Algebra Appl.* 103, 103.

---

*For questions or contributions, open an issue or pull request.*
