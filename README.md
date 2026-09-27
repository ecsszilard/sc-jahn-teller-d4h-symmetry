# Multipolar SC-Triggered Jahn–Teller Model

> **A renormalized mean-field theory (RMFT) simulation of superconductivity-triggered B₁g Jahn–Teller distortion in a D₄h charge-transfer insulator with strong spin–orbit coupling and collinear antiferromagnetic order.**

This repository implements the self-consistent theory described in *"Multipolar superconductivity and Jahn–Teller activation in strongly correlated systems"* (Ecsenyi Szilárd, 2026). The core claim is that in a class of strongly spin–orbit-coupled, charge-transfer Mott insulators, a **B₁g lattice (Jahn–Teller) distortion is symmetry-forbidden in the antiferromagnetic normal state and is unlocked only by Cooper-pair condensation** — the causal arrow runs from superconductivity to the structural instability, not the other way around. The code now carries this mechanism through **both** cross-doublet pairing channels the symmetry argument allows ($\Gamma_6$–$\Gamma_{7a}$ and $\Gamma_6$–$\Gamma_{7b}$) as fully coupled, self-consistent degrees of freedom, together with an exact multi-orbital cluster diagonalization (including Kanamori Hund's coupling) for the superexchange scale and a Moriya self-consistent-renormalization treatment of both the spin and the lattice channel.

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

The three-fold splitting of the $t_{2g}$ shell under SOC actually produces **two** $\Gamma_7$-like doublets once the crystal field is included ($\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}$, see §1). Both are JT-active by the same $\Gamma_6\otimes\Gamma_7\supset B_{1g}$ selection rule (§2), and the code now solves **both** cross-doublet channels self-consistently and on equal footing: the interorbital singlet and d-wave pairing amplitudes $(\Delta_{s,7a},\Delta_{d,7a})$ and $(\Delta_{s,7b},\Delta_{d,7b})$ are all part of the same coupled fixed point, each feeding its own anomalous (Gorkov) coherence back into the Hamiltonian as its own B₁g Weiss field. $\Gamma_{7a}$ — identified in `__post_init__` as the doublet with the larger direct $\langle\Gamma_6\vert B_{1g,\mathrm{op}}\vert \Gamma_7\rangle$ matrix element (§1) — is generically the dominant channel because that matrix element enters the pairing/Weiss-field machinery directly, but $\Gamma_{7b}$'s amplitudes are genuine dynamical variables of the SCF fixed point, not a fixed-zero diagnostic overlay: both are wired into the 24×24 BdG Hamiltonian, the RPA pairing vertex, the four-site cluster free energy, and the gap equations identically in structure, differing only in the (self-consistently distinct) Gutzwiller factor, exchange coupling, and Bogoliubov coherence each channel carries.

### Material class and the three viability conditions

The theory targets systems with:

- D₄h point-group symmetry **and** a global inversion center (square-lattice materials: cuprates, pnictides, selected layered transition-metal oxides). Inversion symmetry is what allows the Cooper pair to have a well-defined parity (pure singlet or pure triplet); if it is broken locally, Rashba/Dresselhaus terms mix the two and the clean tensor-product selection rule below no longer applies unambiguously.
- Strong electron correlation with a charge-transfer (ZSA-type) insulating parent state.
- Strong SOC that reorganizes the $t_{2g}$ manifold into $\Gamma_6 \oplus \Gamma_{7a} \oplus \Gamma_{7b}$ Kramers doublets (typical of 4d/5d transition-metal ions, though the framework is agnostic to the microscopic origin of the SOC scale).
- Superexchange-stabilized collinear AFM order in the parent compound, including a finite intra-atomic Hund's coupling in the virtual $d^2$ superexchange intermediate state (§4).
- Finite hole doping ($\delta > 0$), which restores the itinerancy needed for coherent Cooper pairing.

Three conditions must hold simultaneously for the mechanism to operate:

1. **Charge-transfer insulating character** — the ZSA charge-transfer gap and the on-site $U_{dd}$ (both primary inputs, §4) are large enough that genuine Mott physics is in play.
2. **Non-Mott-localized coherence** — the Cooper pairs must actually be mobile; this requires $\delta > 0$ so the Gutzwiller kinetic factor $g_t$ does not collapse to zero.
3. **Moderate AFM order** — AFM correlations must be present (they are what forbids the JT channel in the normal state) but not so strong that spin fluctuations kill superconductivity outright (Stoner criterion $J_{\mathrm{eff}}\cdot\chi_{SS} < 1$).

### Why the pairing must be interorbital, and why turning on Δ alone is not enough

A tempting but incorrect picture is that the condensate simply mixes $\vert \Gamma_6\rangle$ and $\vert \Gamma_{7}\rangle$ into a coherent superposition $\alpha\vert \Gamma_6\rangle+\beta\vert \Gamma_{7}\rangle$, giving a quadrupole moment linear in the mixing amplitude $\beta\propto\Delta$. This cannot be right: it would make $\langle B_{1g,\mathrm{op}}\rangle$ depend on the arbitrary global $U(1)$ phase of $\Delta$, violating gauge invariance. The correct microscopic statement is that the condensate modifies the **normal** (charge-conserving) density matrix through the Bogoliubov coherence factors: $\langle c^{\dagger}_{6\sigma}c_{7\sigma'}\rangle$ picks up a $v^{*}v \sim \vert \Delta\vert ^2$ contribution, which is explicitly gauge-invariant and appears at *quadratic*, not linear, order in $\Delta$.

For this to happen at all, the pairing must directly connect $\Gamma_6$ and a $\Gamma_7$ doublet — a purely intraorbital singlet (Γ₆–Γ₆ or Γ₇–Γ₇) generates no $\Gamma_6$–$\Gamma_7$ Bogoliubov mixing and leaves the JT channel closed even in the superconducting state. This is exactly the structure built into the code's four self-consistent pairing operators — the on-site singlet and inter-sublattice d-wave amplitude for each of the $\Gamma_6$–$\Gamma_{7a}$ and $\Gamma_6$–$\Gamma_{7b}$ branches (§8 below) — all interorbital by construction.

There is a further, more subtle point verified directly in the group-theoretic analysis: **at $Q=0$, even with $\Delta\neq0$, a purely singlet/d-wave-paired BdG ground state still gives $\langle B_{1g,\mathrm{op}}\rangle \equiv 0$ band-pair by band-pair.** The reason is that $B_{1g,\mathrm{op}}$ in the numerical SOC eigenbasis turns out to be **spin-conserving** ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7}{\uparrow}$), while the singlet pairing operator only ever connects opposite pseudospin sectors; $S_z$ stays block-diagonal in those same sectors. The actual switch is thrown by the *self-consistently induced anomalous coherence* — numerically the same operator as $B_{1g,\mathrm{op}}$ — which is exactly zero at $Q=0$ and becomes nonzero only once **both** $\Delta\neq0$ **and** $Q\neq0$. In the code this anomalous coherence is the pair of quantities `F67a_s`/`F67b_s` (and their d-wave counterparts `F67a_d`/`F67b_d`), fed back into the local Hamiltonian's transverse Weiss field (§19 below) only once a nonzero JT distortion has already appeared; the loop is genuinely self-consistent rather than a one-way "SC turns on JT."

### The JT distortion as a thermodynamic order parameter, not a dynamical mode

$Q$ is treated as a **macroscopic, thermodynamic order parameter** — physically the amplitude of a flat (dispersionless) optical Einstein phonon — rather than a fluctuating dynamical degree of freedom. Differentiating the free energy with respect to $Q$ gives the equilibrium condition and the softening criterion

$$\lambda_{JT}^{\mathrm{norm}} = \chi_{QQ}/K_{\mathrm{eff}} \qquad (<1\text{ stable},\ =1\text{ onset},\ >1\text{ spontaneous JT}).$$

A natural question is whether the free energy contains a **linear** coupling $Q\cdot\vert \Delta\vert ^2$ between the lattice and the condensate. Naively one might argue this is forbidden simply because "$\vert \Delta\vert ^2$ is always $A_{1g}$" — but that argument is incomplete, because the model's pairing channels do not necessarily share one point-group label, and their cross terms could in principle transform as $B_{1g}$. The theory derives, and the code verifies numerically, that this cross-coupling is *exactly* zero in D₄h — an exact but non-generic consequence of the purely interorbital structure of the pairing operators, not a trivial symmetry accident. (If the lattice already sits in D₂h because of a finite static crystal field `Delta_B1g_static` ≠ 0, this exact cancellation is lifted and a genuine linear coupling appears — see §3 below.) The upshot: in the clean D₄h limit, SC-triggered JT is a **threshold phenomenon**. The condensate progressively softens the B₁g mode's effective stiffness $K_{\mathrm{eff}}$ until it collapses toward the fluctuation-corrected instability boundary (§12), at which point the lattice snaps into a finite distortion.

While $Q$ itself is a static mean field, its *coupling* to the electronic condensate is no longer treated as instantaneous everywhere in the code: the phonon-mediated piece of the pairing vertex now carries an explicit adiabatic retardation factor tied to a physical bare-phonon energy scale derived from the lattice spring constant and an input reduced mass (§12 / §14) — a first step beyond the purely static-propagator treatment, though $Q$'s own equation of motion remains a thermodynamic equilibrium condition, not a dynamical one.

### First-order character

Numerically, this transition is expected to be **first-order**, not a simple second-order spin-fluctuation instability: spin fluctuations alone are not sufficient to drive the lattice unstable, and the system tips into the $B_{1g}$-distorted configuration only cooperatively, together with the superconducting condensate. This is reflected in the solver's SCF-dynamics classifier (see the "SCF Loop" description under [Key Algorithms](#key-algorithms)), in the current `__main__`'s explicit three-way free-energy comparison against the normal and $Q$-pinned states (see [Installation & Usage](#installation--usage)), and in the thermodynamic, first-order-aware Tc estimate of §23 below.

---

## Theoretical Framework

### 1. Local Hilbert Space: SOC + Crystal-Field Diagonalization

The full SOC + D₄h crystal-field Hamiltonian is built and diagonalized explicitly on the 6-dimensional $t_{2g}\otimes\mathrm{spin}$ manifold, directly inside `ModelParams.__post_init__`:

```
H = λ_SOC · L·S  +  Δ_tetra · Lz²  +  Delta_B1g_static · (Lx² − Ly²)
```

This yields the Γ₆–Γ₇ₐ splitting `Delta_CF` as a **derived quantity**, never a free input. `Delta_tetra` (negative = tetragonal z-compression) sets the axial crystal field; `Delta_B1g_static` is a static, in-plane crystal-field term with the same $(L_x^2-L_y^2)$ functional form as the dynamical JT operator, logged as `Δ_ip`. Its role is to split the four-dimensional $\Gamma_7$ manifold into two Kramers doublets, $\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}$, which prevents a spurious spontaneous JT instability from the residual $\Gamma_7$ degeneracy while leaving $\Delta_{CF}$ tunable independently of $\lambda_{SOC}$.

**Kramers doublet identification** proceeds in three steps:
1. The three Kramers doublets of $H_{SOC}+H_{CF}$ are sorted by the expectation value $\langle L\!\cdot\!S\rangle$; the doublet with the most negative value is assigned $\Gamma_6$ ($j_{\mathrm{eff}}=1/2$-like), and the two remaining candidates are the $\Gamma_7$ pair.
2. Within each of the three doublets, a 2×2 diagonalization of $S_z$ selects the exact $z$-polarized Kramers partners (`up`, `dn`) and their eigenvalues (`sz_up`, `sz_dn`).
3. Between the two $\Gamma_7$ candidates, the one carrying the **larger direct $\langle\Gamma_6\vert B_{1g,\mathrm{op}}\vert \Gamma_7\rangle$ matrix-element norm** is assigned $\Gamma_{7a}$ (the dominant JT-pairing partner) — this is the defining property, evaluated directly rather than inferred from a proxy. The total moment $\mu_z=\langle L_z+2S_z\rangle$ is computed only as an independent **cross-check** and triggers a warning log if it disagrees with the $B_{1g}$-coupling criterion (`⚠ Γ7a (B1g-coupling-sorted) has smaller |μz| than Γ7b`) — this can happen in strongly mixed CF/SOC regimes, and the log line reports both the $\vert S_z\vert $ and $\mu_z$ values for each candidate alongside the $B_{1g}$-coupling norms that actually decided the assignment.

**All three doublets enter the solver exactly — there is no downfolding, and neither is diagnostic-only.** The resulting basis is $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$, and all six components enter every operator (`B1g_op`, `Eg2_op`, the hopping matrices, and the BdG Hamiltonian itself). $\Gamma_{7b}$ disperses as a genuine band through the same rigorous orbital-projected hopping as $\Gamma_6/\Gamma_{7a}$ (§6, §9), and — as of the current code — pairs through its own self-consistent $\Gamma_6$–$\Gamma_{7b}$ singlet and d-wave amplitudes on exactly the same footing as $\Gamma_{7a}$ (§8), rather than being projected out or treated as a perturbative/diagnostic correction. Several derived quantities are cached on `ModelParams` at this point:

- `sz_op = [sz6_up, sz6_dn, sz7_up, sz7_dn, sz7b_up, sz7b_dn]` — **exact** $\langle S_z\rangle$ eigenvalues from the doublet diagonalization (not an approximate moment-ratio model); used directly as the AFM Weiss-field weights and as the spin vertex in every susceptibility calculation.
- `multi_op` — the effective multipolar spin operator entering the cluster exchange $H_{\mathrm{exch}} = J\cdot(\mathrm{multi\_op}\otimes\mathrm{multi\_op})$, built as $\mathrm{diag}((\vert sz_6\vert \cdot P_6+\vert sz_{7a}\vert \cdot P_{7a}+\vert sz_{7b}\vert \cdot P_{7b})\cdot sz_{\mathrm{diag}})$.
- `p_7` — the average $\Gamma_7$ (either doublet) orbital-weight admixture in the $\Gamma_6$ eigenvectors; used by the static admixture-based Gutzwiller estimate (§5).
- `Delta_CF = evals[2] − evals[0]` (Γ₇ₐ–Γ₆ gap, JT-active) and `g7split = evals[4] − evals[2]` (Γ₇ᵦ–Γ₇ₐ internal splitting).
- Orbital-character weights `_w6_xz/_yz/_xy`, `_w7_xz/_yz/_xy`, `_w7b_xz/_yz/_xy` (assembled as the `_w_orb` array, rows Γ6/Γ7a/Γ7b) — the $d_{xz}/d_{yz}/d_{xy}$ character of each doublet, used to build both the Q-dependent exchange anisotropy (§6) and the orbital-selective hopping matrices `Tx_A_xz/_yz/_xy` (§6, §9).
- `kappa_doublets` (3 components: Γ6, Γ7a, Γ7b) and `kappa_cross` (2 components: Γ6–Γ7a, Γ6–Γ7b) — the exact-diagonalization superexchange coefficients (§4), used by `exchange_channels()` (§6) every time the Q-dependent exchange tensor is evaluated.

If `lambda_soc`, `Delta_tetra`, or `Delta_B1g_static` are mutated on a live solver, `params.__post_init__()` must be followed by `solver._rebuild_orbital_operators()` so that `B1g_op`, `B1g_24`, `Eg2_op`, `Eg2_24`, `sz_op`, `multi_op`, `Sz_nambu`, and `Sz_stag_nambu_channels` stay consistent with the new eigenbasis; `RMFT_Solver._full_rebuild()` (§ Key Algorithms) is the single method that does this along with everything else that depends on it.

### 2. Symmetry Protection of the JT Channel

**Selection rule in a pure doublet.** A rank-$k$ irreducible tensor operator has a nonzero diagonal matrix element in $\Gamma_6$ only if $\Gamma^{(k)}\subset\Gamma_6\otimes\Gamma_6$. Since $\bar D_{4h}$ character theory gives $\Gamma_6\otimes\Gamma_6=\Gamma_7\otimes\Gamma_7=A_{1g}\oplus A_{2g}\oplus E_g$ — containing neither $B_{1g}$ nor $B_{2g}$ — the quadrupole operator has $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_6\rangle=0$ **exactly**. A pure $\Gamma_6$ (or $\Gamma_7$) manifold carries no electric quadrupole moment and does not couple to a $B_{1g}$ lattice shear. Because the collinear AFM state stabilizes only this kind of dipolar (rank-1), spin–orbitally mixed order, the JT channel is symmetry-blocked in the normal state.

**Cross-product opens the channel.** $\Gamma_6\otimes\Gamma_7 = B_{1g}\oplus B_{2g}\oplus E_g$ *does* contain $B_{1g}$, so the **off-diagonal** element $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_{7a}\rangle\neq0$ is allowed — and, by the identical argument, so is $\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_{7b}\rangle$, generally with a smaller (but finite) matrix element, since $\Gamma_{7a}$ is defined precisely as the doublet carrying the larger of the two (§1). Both channels are realized by the code's self-consistent pairing structure (§8): realizing either one requires a genuinely interorbital pairing operator and, as emphasized above, requires $Q\neq0$ as well as $\Delta\neq0$ — the self-consistent loop, not a one-shot symmetry argument, is what actually opens the channel.

**Θ symmetry and the global BZ cancellation.** In the collinear AFM state, time reversal $\mathcal T$ is broken, but the combined Shubnikov element $\Theta=\mathcal T\cdot\tau_{AB}$ (time reversal composed with the $A\leftrightarrow B$ sublattice translation) survives; since $\tau_{AB}^2=+1$ and $\mathcal T^2=-1$, $\Theta^2=-1$. This does **not** give pointwise Kramers degeneracy — $\Theta$ maps crystal momentum $k\to -k$, and in the magnetic Brillouin zone $k$ and $-k$ are generally inequivalent — but it does impose a **global** constraint on Brillouin-zone integrals. The spin–quadrupole susceptibility integrand is odd under $\Theta$ at $\Delta=0$, so the full-BZ Lindhard sum cancels identically:

$$\chi_{SQ}(q) = \int_{\mathrm{BZ}} d^2k\; \mathcal I_{SQ}(k,q) = 0 \qquad (\Delta=0).$$

This BZ-wide cancellation — not a pointwise Kramers argument — is enforced in the code by the odd-in-$k$ structure of the normal-state Lindhard kernel (`_lindhard_bubble` with `_NORMAL_SECTOR_PAIRS`), and is a structural property of every $\Delta=0$ evaluation of `_chi0_channels_raw`/`_rpa_closure` (§14): the cross-channel entries of the returned $4\times4$ matrix vanish identically at $Q=0,\Delta=0$ by construction, not by a runtime numerical check.

In the superconducting state the full Nambu–Lehmann sum includes the anomalous (Gorkov) sectors, whose Bogoliubov coherence factors can break the odd-$\Theta$ structure — but as detailed above, this alone is not sufficient at $Q=0$: purely singlet/d-wave, interorbital pairing keeps $S_z$ and $B_{1g,\mathrm{op}}$ acting within disjoint pseudospin sectors band-pair by band-pair, so the product stays exactly zero until the self-consistently generated $Q\neq0$ genuinely couples the two sectors.

**The selection ratio, explicitly.** The quantity that actually crosses the symmetry boundary is the Γ₆–Γ₇ₐ anomalous (Gorkov) singlet amplitude, computed directly from the converged BdG eigensystem as

```math
F_{67a,s} = \sum_k (1-2f_n)\,\mathrm{Re}[u^{*}_{6\uparrow}v_{7a\downarrow} - u^{*}_{6\downarrow}v_{7a\uparrow}]
```

(a mean over sublattices is implied). An identical expression gives $F_{67b,s}$ for the $\Gamma_6$–$\Gamma_{7b}$ branch, and companion d-wave amplitudes $F_{67a,d}$, $F_{67b,d}$ come from the inter-sublattice pairing structure. Each of the four is exactly zero whenever $\Delta=0$ (exact D₄h selection rule) — the code's own inline invariant, computed in one shot for all four by `_compute_F67_pair_amplitudes`. Two independent operations turn the raw complex BZ sum into the mean-field quantity actually fed back into the local Hamiltonian: (i) it is **phase-locked** against its own gap component, $F=\mathrm{Re}[F_{\mathrm{raw}}\cdot\Delta_i^{*}/\vert \Delta_i\vert ]$ — projecting onto the condensate's own phase so the returned amplitude is real and every one of the four channels is treated symmetrically, referenced only to itself — and then (ii) Gutzwiller-weighted by that branch's own factor (`g_Delta_s` for both on-site amplitudes, `g_Delta_ad` or `g_Delta_bd` for the respective d-wave one, §5). The selection ratio logged in `__main__` is a single scalar built from the total gap amplitude relative to $\Delta_{CF}$, weighted by the leading ($\Gamma_{7a}$) anomalous amplitude, and is used purely as a threshold diagnostic (`_JT_ACT_THR`) for whether the JT channel is judged "active," not as an input to the SCF loop itself.

### 3. The B₁g Operator and the D₄h/D₂h Crossover

The B₁g phonon coupling operator is constructed from the same $t_{2g}$ operators used for $H_{CF}$ and projected into the full 6-dimensional $\Gamma_6\oplus\Gamma_{7a}\oplus\Gamma_{7b}$ subspace — no downfolding:

```
B1g_op = real(U6† · (Lx² − Ly²)_t2g · U6)     # 6×6, real, Hermitian
```

where `U6` is the 6×6 change-of-basis matrix from the bare $t_{2g}\otimes\mathrm{spin}$ manifold to the Kramers-doublet (SOC+CF) eigenbasis built in §1.

- **D₄h (`Delta_B1g_static = 0`):** the $\Gamma_6$–$\Gamma_6$ and $\Gamma_{7a}$–$\Gamma_{7a}$ (and $\Gamma_{7b}$–$\Gamma_{7b}$) diagonal blocks of `B1g_op` vanish exactly and it is purely off-diagonal and **spin-conserving** — it connects $\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\uparrow}$ and $\Gamma_6\leftrightarrow\Gamma_{7b}$, generally with different weight for each. Hence $\langle B_{1g,\mathrm{op}}\rangle=0$ in *any* normal state.
- **D₂h (`Delta_B1g_static ≠ 0`):** the operator picks up real diagonal ($A_{1g}$-like) components that renormalize $\Delta_{CF}$ but do not by themselves drive JT, plus additional off-diagonal weight. The fraction of the operator that remains purely off-diagonal — `b1g_ratio = ‖B1g_offdiag‖/‖diag(B1g_op)‖`, converted to a weight $b_{1g,\mathrm{weight}} = b_{1g,\mathrm{ratio}}/(1+b_{1g,\mathrm{ratio}})$ — is computed on demand (e.g. inside the verbose branch of `_scf_kick`) rather than cached as a permanent `ModelParams` field; it quantifies how much of the operator remains genuinely SC-triggered versus normal-state-active. `b1g_weight ≈ 1` in the clean D₄h limit.

The 24×24 Nambu extension `B1g_24` (built in `_rebuild_orbital_operators`) carries the hole block as $-B_{1g,\mathrm{op}}^{T}$ (real, so $=-B_{1g,\mathrm{op}}$), consistent with BdG particle–hole symmetry; every JT coupling term in the Hamiltonian is written `H += β²(k) · g_JT · Q · B1g_op` rather than a hand-built matrix (§9). `Eg2_24` mirrors this construction for the Eg,2 operator (§7).

### 4. Multi-Orbital Cluster Superexchange and the Weiss Field

AFM order originates from virtual $p$–$d$ hopping (ZSA charge-transfer superexchange), not from a bare Stoner Fermi-surface instability, and — as of the current code — the superexchange scale is obtained from an **exact diagonalization of the full multi-orbital $d$–$p$–$d$ two-hole cluster**, resolved separately for all three Kramers doublets and both cross-doublet channels rather than a single scalar.

`_multiorbital_2hole_hamiltonian` builds the full two-hole problem on the $\binom{18}{2}=153$-dimensional Fock space of an 18-spin-orbital, 3-site chain (metal $d_1$ – ligand $p$ – metal $d_2$, each site carrying the full 3-orbital $\times$ 2-spin $t_{2g}$ manifold, SOC and crystal field included on both metal sites). The ligand is modeled as **three independent modes**, one per $L_z$ channel, each hybridizing with equal bare $t_{pd}$ — this preserves $J_z=L_z+S_z$ through the SOC mixing and matches the orbital-blind bare hopping $T_x(Q{=}0)=t_0 I_6$ used elsewhere in the model. The two-hole interaction is the density–density Kanamori form,

```
same channel, opposite spin   → U_dd
different channel, opposite spin → U_dd − 2·J_H
different channel, same spin     → U_dd − 3·J_H
```

so that `J_H_ratio = 0` recovers the original single-parameter interaction exactly, and a finite Hund's coupling $J_H = $ `J_H_ratio` $\times\,U_{dd}$ lowers the effective two-hole triplet energy relative to the naive $U_{dd}$-only estimate, as it must physically.

From the resulting exact spectrum, two downfolding routines extract the physical exchange couplings via a rigorous **Löwdin/Bloch–des Cloizeaux quasi-degenerate projection** onto the relevant two-ion reference subspace — not a leading-order perturbative expansion in $t_{pd}$:

- `_intra_doublet_kappa` computes, for each of the three doublets independently, the **energy-dependent** (Feshbach) effective Hamiltonian on the reference subspace $\{\vert \text{doublet}\,a\rangle_1\otimes\vert \text{doublet}\,b\rangle_2\}$ via `_lowdin_branch_energies` (an exact, seeded fixed-point search for every branch energy — not a single shared trial energy), reads off the isotropic (Heisenberg) coupling from the triplet–singlet splitting in the normalized pseudospin basis, and **rescales by the doublet's own effective spin** $s_{\mathrm{eff}}$ (from its exact $\langle S_z\rangle$, §1): $\kappa_c=(E_T-E_{S_0})/(4\,s_{\mathrm{eff},c}^2)$. This rescaling matters: without it, $J_{A1g}$ would be over-strengthened by up to two orders of magnitude for $\Gamma_{7b}$ in the strong-SOC regime, where $s_{\mathrm{eff}}$ can be $\ll 1/2$. A non-Heisenberg (Ising/DM-like) component is separately diagnosed via a Pauli-tensor decomposition of the effective Hamiltonian and flagged with a warning if it exceeds `_ANISO_WARN_TRESH`, though only the isotropic part is used downstream.
- `_cross_doublet_kappas` extracts the $\Gamma_6$–$\Gamma_{7a}$ and $\Gamma_6$–$\Gamma_{7b}$ cross-exchange coefficients $J_{B1g}$ from the same exact spectrum, this time via the **energy-independent** Bloch–des Cloizeaux effective Hamiltonian (found ≈7% more accurate for this cluster than the energy-dependent Feshbach construction used for the intra-doublet case) on the reference subspace $\{\vert \Gamma_6\rangle_1\vert \Gamma_7\rangle_2\}\cup\{\vert \Gamma_7\rangle_1\vert \Gamma_6\rangle_2\}$, converting the raw transition-block amplitude to the operator coefficient that actually multiplies $B_{1g,\mathrm{op}}(i)\cdot B_{1g,\mathrm{op}}(j)$ downstream. The routine returns zero (with a warning) if the reference subspace is near-singular, the exact eigenstates are not well isolated in it, or the $\Gamma_6$↔$\Gamma_7$ block carries less than `_B1G_BLOCK_MIN_FRAC` of the operator's total weight — all three are ill-conditioned regimes for the selection-rule-suppressed channel.

`ModelParams.__post_init__` stores the results as `kappa_doublets` (3-vector: Γ6, Γ7a, Γ7b) and `kappa_cross` (2-vector: Γ6–Γ7a, Γ6–Γ7b); `exchange_channels()` (§6) turns these, together with the orbital-character weights of §1, into the actual $Q$-dependent exchange tensors used everywhere else. `U_dd` is a **primary input** (the on-site Hubbard repulsion); `U_pp`, the ligand (2p) hole–hole repulsion, is obtained from a second-order downfolding estimate that also depends on the `Upp_ratio_bare` primary input (the bare, unhybridized $U_{pp}/U_{dd}$ ratio) and the coordination number:

```
U_pp = (Upp_ratio_bare · U_dd) / (1 + (Z/2) · t_pd² / (Delta_CT · (Delta_CT + U_dd)))
```

with `Z/2` the ligand coordination number (each ligand typically bridges two metal sites). `hybrid_scale` does **not** enter this calculation; its role is instead the $k$-dependent quasiparticle downfolding weight $\beta^2(k)$ used in the BdG Hamiltonian and the cluster embedding (§6, §9).

The effective AFM Weiss field entering the BdG Hamiltonian is diagonal in the full 6-component $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$ basis, proportional to `sign_M · Z_afm_eff · J_A1g[α,α] · M_c · sz_op[α]` (channel $c$ matching orbital $\alpha$), where `J_A1g_diag` is the longitudinal (spin-preserving) exchange tensor from `exchange_channels()` (§6) and the doping renormalization is carried by the itinerant carrier density `n_kspace` returned from the chemical-potential solve.

### 5. Gutzwiller Renormalization and the Mott Guard

`t_pd` is the single primary hopping input; the effective $dd$ hopping `t0 = t_pd² / Delta_CT` is always derived, never set directly.

```
g_t       = 2δ/(1+δ)         # kinetic-energy suppression → 0 at half-filling
g_J       = 4/(1+δ)²         # exchange enhancement → 4 at half-filling
```

`ModelParams.get_gutzwiller_factors` also returns a **static**, admixture-based estimate `g_Delta_s = g_t`, `g_Delta_ad = g_t + (g_J−g_t)·p_7` (interpolated by the doping-independent $\Gamma_7$ admixture `p_7` of §1) — used only for the cheap pre-SCF seed (`_scf_kick`). Inside the SCF loop itself, `RMFT_Solver.estimate_gutzwiller_factors_occupation_based(M, Q, n_kspace, mu, g_t, g_J)` replaces this with a **dynamic**, occupation-based estimate, re-evaluated every iteration from the actual normal-state ($\Delta=0$) BdG orbital densities $n_6, n_{7a}, n_{7b}$ on sublattice A:

```
g_Delta_s  = g_t
g_Delta_ad = g_t + (g_J − g_t) · clip(2·n7a / n_total, 0, 1)
g_Delta_bd = g_t + (g_J − g_t) · clip(2·n7b / n_total, 0, 1)
```

so the d-wave (inter-sublattice) renormalization of **each** cross-doublet channel tracks that channel's own actual orbital occupation, rather than a single fixed admixture shared by both. `g_Delta_s` (the on-site, kinetic-origin channel) is common to both branches.

The superexchange is always computed from the **bare** hopping `t_pd` (through the cluster diagonalization of §4), then multiplied by `g_J` — computing it from Gutzwiller-*renormalized* bands would double-count the suppression (a spurious `g_t²` factor), since `g_t` renormalizes only the kinetic energy while `g_J` renormalizes only the exchange; the two are orthogonal channels in this RMFT scheme. The lattice-summed Weiss-field/superexchange scale for the leading ($\Gamma_6$) channel is $J_{\mathrm{eff}} = Z_{\mathrm{afm,eff}} \cdot J_{A1g}(q_{AFM})$, an $M$-weighted average over the three doublet channels' own $J_c(q_{AFM})$ (§6), evaluated by `_exchange_J_q`.

A **Mott guard** suppresses superconductivity when `g_t < _G_T_COHERENCE_MIN = 0.10` (i.e. $\delta \lesssim 0.053$): below this the Gutzwiller factor signals that the Zhang–Rice-singlet band is no longer coherent enough to support a physical SC gap, and the post-SCF result is flagged `mott_suspect` (all four gap channels zeroed) rather than returned as a spuriously converged superconducting state; the same guard also fires on a nodal coherence length $\xi_{\mathrm{nodal}}/a < 1$ (BEC-side breakdown of BdG mean-field validity).

### 6. B₁g Jahn–Teller Distortion, Orbital-Selective Hopping, and Further-Neighbor Terms

The B₁g mode breaks the $x$–$y$ symmetry of the square lattice through an exponential (Harrison-type) hopping law:

```
tx(Q) = t0 · exp(+Q / lambda_hop)      # elongation along x → shorter bond → larger hopping
ty(Q) = t0 · exp(−Q / lambda_hop)      # compression along y → longer bond → smaller hopping
K_eff = K_lattice + ∂²F_ex/∂Q²
```

`K_lattice` is the bare phonon spring constant (primary input, never mutated); `∂²F_ex/∂Q²` is the exchange contribution to the stiffness (§11), negative when the condensate softens the mode.

Unlike a simple scalar $t(Q)$ applied uniformly across all orbitals, the inter-sublattice hopping is **orbital-selective**: `hopping_matrices(Q)` builds rigorous $6\times6$ matrices

```
T_x(Q) = t_x(Q)·A_xz + t_y(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy
T_y(Q) = t_y(Q)·A_xz + t_x(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy   (x↔y swap)
```

from the exact $d_{xz}/d_{yz}/d_{xy}$ orbital-character projectors (`Tx_A_xz`, `Tx_A_yz`, `Tx_A_xy`, built once in `__post_init__` and satisfying $A_{xz}+A_{yz}+A_{xy}=I_6$), so that $\Gamma_6$, $\Gamma_{7a}$, and $\Gamma_{7b}$ each disperse according to their own $d_{xz}/d_{yz}/d_{xy}$ admixture rather than sharing one isotropic band. `hopping_matrices_dQ(Q)` gives the companion exact analytic $\partial T_{x,y}/\partial Q$.

The full multipolar exchange tensor is likewise Q-dependent through both the overall B₁g channel opening and an orbital-selective asymmetry. `exchange_channels(Q, g_J)` returns:

```
J_A1g_diag[c]      = g_J · κ_c · ⟨orbital weight⟩_c(Q)                     (longitudinal, even; 6-component, Kramers-doubled)
J_B1g_scalar_7a, _7b = g_J · κ_cross,{7a,7b} · [√(f6,orb f7,orb)]·(j_xz(Q) − j_yz(Q))   (transverse, odd; ONE scalar per cross-doublet branch)
```

i.e. the code now returns **two independent transverse couplings**, $J_{B1g}^{(7a)}(Q)$ and $J_{B1g}^{(7b)}(Q)$ — one for each self-consistently paired branch (§8) — rather than a single shared scalar, each built from its own `kappa_cross` entry (§4) and its own $\Gamma_6$–$\Gamma_7$ orbital-weight geometric mean. Both vanish at $Q=0$ for equal $xz/yz$ weights; with `Delta_B1g_static` $\neq 0$ a residual static B₁g exchange survives at $Q=0$ in both channels.

**Momentum dependence.** `_exchange_J_q(J_A1g_diag, qx, qy)` evaluates the longitudinal exchange at arbitrary $q$ via an NN + 2nd-NN + 3rd-NN sum tied to the same single-bond value:

```
J_c(q) = J_A1g_diag[c] · [ −2·(cos qx + cos qy) − 4·(J2/J1)·cos qx·cos qy − 2·(J3/J1)·(cos 2qx + cos 2qy) ]
```

with $J_2/J_1=(t'/t)^2$, $J_3/J_1=(t''/t)^2$; at $q=(\pi,\pi)$ this gives $J_c(q_{AFM}) = Z_{\mathrm{afm,eff}}\cdot J_{A1g,\mathrm{diag}}[c]$, the AFM-peaked value that enters the Weiss field (§4) and the RPA denominator (§14).

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

with its own coupling constant `g_Eg2` (eV/Å), bare stiffness `K_lattice_Eg2` (eV/Å²), and distortion amplitude `Q_Eg2`, entering the BdG Hamiltonian, the free energy, and the Hessian on the same footing as the B₁g channel via `Eg2_24` (the 24×24 Nambu lift, mirroring `B1g_24`) and `Eg2_expectation()`. Unlike `B1g_op`, `Eg2_op` connects Kramers partners with an actual **spin-flip** structure ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7}{\downarrow}$) rather than the spin-conserving structure of `B1g_op` — the two channels probe genuinely different multipolar sectors of the same $\Gamma_6\otimes\Gamma_7$ manifold.

At the current stage of the implementation, the exchange contribution to the Eg,2 stiffness and the B₁g–Eg,2 cross term vanish identically by Kramers symmetry, so `K_eff_Eg2` is left at its bare value `K_lattice_Eg2` (no exchange-driven softening is computed for this channel yet, in contrast to the fully renormalized `K_eff` for B₁g). The Eg,2 channel is therefore best read, in the current code, as a genuine second JT-active degree of freedom already wired through the Hamiltonian and observables, whose own self-consistent back-action on the lattice stiffness is not yet as developed as the B₁g channel's.

### 8. Dual Cross-Doublet Pairing Channels — Both Fully Self-Consistent

Four interorbital B₁g pairing amplitudes are carried self-consistently, exactly as required by the symmetry argument in §2, packed into one complex 4-vector `Delta_vec = [Δ_s7a, Δ_d7a, Δ_s7b, Δ_d7b]`:

- **On-site singlet** ($\Gamma_6\otimes\Gamma_7\to B_{1g}$, constant $k$-space form factor), one amplitude per branch:
  ```
  D_on[6↑,7a↓] =  Δ_s7a ,  D_on[6↓,7a↑] = −Δ_s7a       D_on[6↑,7b↓] =  Δ_s7b ,  D_on[6↓,7b↑] = −Δ_s7b
  ```
- **Inter-sublattice d-wave** ($\varphi(k)=\cos k_x-\cos k_y \to B_{1g}$ in $k$-space), one amplitude per branch:
  ```
  D_d(k)[A:6↑,B:7a↓] = φ(k)·Δ_d7a       D_d(k)[A:6↑,B:7b↓] = φ(k)·Δ_d7b
  ```

Both branches feed into the **same** gap-equation infrastructure (§21) and the **same** FS-averaged pairing kernel $(V_s,V_d,V_{sd})$ (§15) — there is only one momentum-space pairing vertex — but each branch converts that shared vertex into its own gap amplitude using its own Gutzwiller factor (`g_Delta_ad` vs. `g_Delta_bd`, §5) and its own Bogoliubov anomalous amplitude (`F67a_{s,d}` vs. `F67b_{s,d}`, §2), which differ because $\Gamma_{7a}$ and $\Gamma_{7b}$ sit at different energies ($\Delta_{CF}$ vs. $\Delta_{CF}+g7split$) and generally carry different direct $B_{1g}$ coupling to $\Gamma_6$ (§1). Concretely, inside `compute_gap_eq_vectorized` the same raw fixed-point construction is evaluated for both branches,

```
Δ_s7a_raw = V_s·F67a_s + √(g_Delta_s/g_Delta_ad)·V_sd·F67a_d      Δ_s7b_raw = V_s·F67b_s + √(g_Delta_s/g_Delta_bd)·V_sd·F67b_d
Δ_d7a_raw = √(g_Delta_ad/g_Delta_s)·V_sd·F67a_s + V_d·F67a_d      Δ_d7b_raw = √(g_Delta_bd/g_Delta_s)·V_sd·F67b_s + V_d·F67b_d
```

and both pairs are carried through the identical jump-cap / phase-preserving blend logic (§21) to produce the 4-component output. The 2-component legacy form `[Δ_s7a, Δ_d7a]` is still accepted by `VectorizedBdG._build_H_stack` for backward compatibility (it reproduces the $\Gamma_{7b}$-decoupled model exactly, with $\Delta_{7b}\equiv0$), but the code's own default path — every `solve_self_consistent` call in `__main__` — runs the full four-channel solve.

In practice $\Gamma_{7a}$ is generically the dominant branch (it is defined in §1 as the doublet with the larger direct $B_{1g}$ matrix element to $\Gamma_6$, so it couples more strongly to essentially everything downstream), and several headline quantities in the result dictionary (`Delta_s`, `Delta_d`, the Tc estimates of §23, the coherence-length diagnostics of §15) are still built from the $\Gamma_{7a}$ amplitudes specifically — with the exception of the coherence-length gap magnitude in `scf_gap_diagnostics`, which sums both branches, $\Delta_s=\vert \Delta_{s7a}+\Delta_{s7b}\vert $, $\Delta_d=\vert \Delta_{d7a}+\Delta_{d7b}\vert $. The full four-component `Delta_vec` (and the matching four-component anomalous-amplitude vector `F67_vec`) is always available in the result dictionary for anyone who needs the $\Gamma_{7b}$ channel's own converged amplitude directly.

---

### 9. The 24×24 BdG Hamiltonian (Doubled Unit Cell, Full 3-Doublet, 4-Channel Pairing)

Nambu basis $\Psi=[\text{Particle}_A(6),\ \text{Particle}_B(6),\ \text{Hole}_A(6),\ \text{Hole}_B(6)]$, each block ordered $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T_AB(k)    │  D_on(k)             │   ← Particle sector
      │  T_AB†(k)  H_B     │  D_on(k)             │
      ├────────────────────┼─────────────────────┤
      │  D_on(k)†          │  −H_A*   −T_AB*      │   ← Hole sector
      │                    │  −T_AB†* −H_B*       │
      └────────────────────┴─────────────────────┘
```

`H_A`, `H_B` are the $6\times6$ local (AFM Weiss field + crystal field + JT + $t',t''$ diagonal dispersion) sublattice Hamiltonians; $T_{AB}(k)=-2g_t[\cos k_x\,T_x(Q)+\cos k_y\,T_y(Q)]$ is the **orbital-selective** inter-sublattice hopping block of §6 (not a scalar times the identity), plus an inter-sublattice transverse-Weiss-field contribution described below. `D_on(k)` packs the on-site singlet ($\Gamma_6\leftrightarrow\Gamma_{7a}$ **and** $\Gamma_6\leftrightarrow\Gamma_{7b}$, weighted independently by $\Delta_{s,7a}$ and $\Delta_{s,7b}$) and the $\varphi(k)$-modulated inter-sublattice d-wave piece (again both branches, weighted by $\Delta_{d,7a}$, $\Delta_{d,7b}$) into one antisymmetrized $12\times12$ particle–hole block (§8); the particle–hole off-diagonal blocks use the **transposed** (not Hermitian-conjugate) pairing operator, consistent with BdG particle–hole symmetry.

**The JT coupling itself is $k$-dependent**, not a rigid on-site term: both the JT coupling and the anomalous Weiss field are modulated by the same $k$-dependent quasiparticle spectral weight $\beta^2(k)$ used for the charge-transfer downfolding:

```
β²(k) = sigmoid[ k_s·(1 − hybrid_scale·[t̄·(cos kx+cos ky) + δt·(cos kx−cos ky)]/Delta_CT − 0.5) ]
        t̄ = (tx+ty)/2,  δt = (tx−ty)/2,  k_s=10                # wave_function_weight(tx, ty, kx, ky)

H += β²(k) · g_JT · Q · B1g_op                                                        # k-weighted JT coupling
H_TRW_on-site  += β²(k) · Z · [ F67a_s·J_B1g^(7a)·B1g_offdiag_{6,7a}  +  F67b_s·J_B1g^(7b)·B1g_offdiag_{6,7b} ]   # per-branch, k-weighted
H_TRW_inter-sublattice += φ(k) · [ F67a_d·J_B1g^(7a)·B1g_offdiag_{6,7a}  +  F67b_d·J_B1g^(7b)·B1g_offdiag_{6,7b} ]
```

`wave_function_weight` additionally $T_Q$-symmetrizes this sigmoid, averaging its value at $k$ and at $k+(\pi,\pi)$, so the downfolding weight itself respects the AFM-translation parity used elsewhere in the code. $\beta^2(k)$ is the fraction of the quasiparticle wavefunction that remains on the metal ion rather than the ligands at each $k$-point, so both channels that couple through the metal-ion orbital angular momentum operators ($B_{1g,\mathrm{op}}$) are naturally suppressed where the quasiparticle is more ligand-like. By contrast, the Eg,2 term (§7) is added **without** this $k$-weighting — `H += g_Eg2·Q_Eg2·Eg2_op` uniformly — reflecting its treatment, at the current stage of the implementation, as a spatially uniform (q=0) structural order parameter. In the particle sector this all enters as shown above; the hole sector carries the corresponding $-(\cdot)^{*}$, and exact Hermiticity is enforced after assembly.

`VectorizedBdG._build_H_stack` builds and diagonalizes this 24×24 matrix for the entire k-grid in one batched `numpy.linalg.eigh` call, reusing a pre-allocated buffer (`out=`) across SCF iterations to avoid repeated allocation.

The physical electron density is $\langle n_{i\sigma}\rangle=\sum_n \vert u_{n,i\sigma}\vert ^2 f(E_n) + \vert v_{n,i\sigma}\vert ^2(1-f(E_n))$ — both terms carry a positive sign, since $\vert v\vert ^2(1-f)$ is the filled-band contribution from below the Fermi level.

### 10. Observables via `VectorizedBdG`

All thermal-average observables are extracted from a single batched diagonalization:

| Observable | Formula (schematic) | Role |
|---|---|---|
| **⟨B1g⟩** (full) | $\mathrm{Tr}[B_{1g,24}\cdot\rho_k]$, `/4` for Nambu+sublattice doubling, $\beta^2(k)$-weighted | Hellmann–Feynman lattice force: $Q_{\mathrm{eq}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ |
| **⟨Eg2⟩** | same construction against `Eg2_24`, **without** the $\beta^2(k)$ weight | Hellmann–Feynman force for the Eg,2 channel |
| Magnetization | $\langle S_z\rangle$ via the exact `sz_op` weights, per-channel via `Sz_stag_nambu_channels` | AFM order parameter `M` (channel-resolved: Γ₆, Γ₇ₐ, Γ₇ᵦ) |
| `F67_vec` | Gorkov Γ₆–Γ₇ₐ **and** Γ₆–Γ₇ᵦ singlet/d-wave amplitudes (4 components), `_compute_F67_pair_amplitudes` | Anomalous Weiss-field back-action, both branches (§2, §8, §19) |
| Density | $\sum_n[\vert u\vert ^2 f + \vert v\vert ^2(1-f)]$ / 4 | Chemical-potential control |
| Pairing s / d (×2 branches) | on-site / inter-site $u^{*}v$ combinations, Γ₆–Γ₇ₐ and Γ₆–Γ₇ᵦ | s-/d-channel gap-equation inputs, both branches |

The lattice update in the SCF loop uses the **full** $\langle\hat B_{1g}\rangle=\mathrm{Tr}[B_{1g,24}\cdot\rho]$, not a bare off-diagonal piece, because in D₂h `B1g_op` gains diagonal and spin-preserving components that are active even without SC — using only the off-diagonal piece would break Hellmann–Feynman consistency with `H_{JT}=\beta^2(k)\,g_{JT}\,Q\,B_{1g,\mathrm{op}}`. In D₄h the two expressions coincide exactly. Concretely, `B1g_expectation` contracts the already-diagonalized Nambu eigenvectors against `B1g_24` and weights by the occupation and the same $\beta^2(k)$ ZRS factor used in the Hamiltonian (§9), so the observable used to drive $Q$ stays consistent with what was actually put into $H$:

```python
diag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), B1g_24, ec).real     # a,b: 24 Nambu components; n: band index
exp_k   = np.einsum('kn,kn->k', diag_qp, f_n) * beta2_k               # per-k thermal average, β²(k)-weighted
B1g_exp = np.dot(k_weights, exp_k) / 4.0                              # /4: Nambu (particle–hole) × sublattice (A–B)
```

Summing over all 24 Nambu bands with plain $f(E_n)$ (not $1-f$) automatically covers the hole contributions too, since the hole-sector sign is already built into `B1g_24` itself (§3) and $f(-E)=1-f(E)$. `Eg2_expectation` mirrors this exactly but contracts against `Eg2_24` instead, and — consistent with §7/§9 — omits the $\beta^2(k)$ weighting.

### 11. Exchange Rigidity: ∂²F_ex/∂Q² and the Adiabatic Δ-Relaxation Correction

`compute_K_eff_full` evaluates the *mechanical* (frozen-Δ) contribution to the B₁g stiffness by a central second difference of the canonical (fixed-density) free energy at $Q\pm\varepsilon$, $\varepsilon=\max(10^{-4}, 0.01\vert Q\vert +10^{-4})$, plus the cluster-ED stiffness correction `Delta_K_cluster` of §19 (sign-preserving-capped at the bare stiffness scale):

```
K_can = (F(Q+ε) − 2F(Q) + F(Q−ε)) / ε²  +  dK_corr        (Delta_vec, F67_vec held fixed at their current SCF values)
```

Separately, `_delta_relaxed_K_eff_correction` computes an **adiabatic condensate-relaxation** softening: if $\Delta$ is allowed to follow $Q$ adiabatically rather than being frozen, the effective stiffness is reduced by the Schur-complement term $H_{Q\Delta}^2/H_{\Delta\Delta}$ of the local $(Q,\Delta)$ Hessian of the same canonical free energy (evaluated with all four phases fixed and only the total amplitude probed), capped at `_DK_CORR_CAP_MULT`$\times K_{\mathrm{bare}}$ and returned as exactly zero in the normal state or whenever $H_{\Delta\Delta}\le0$. This correction is subtracted from `compute_K_eff_full`'s result at the two call sites that need the fully relaxed stiffness — the Q Newton step (§ Key Algorithms) and the JT-viability evaluation of §12 — but **not** inside `compute_K_eff_full` itself, which always returns the purely mechanical curvature.

`K_eff = K_lattice + ∂²F_ex/∂Q²` (negative correction = exchange softens the mode; this is what can drive `K_eff` toward the JT-triggering threshold in the SC state). As noted in §7, the analogous Eg,2 and B₁g–Eg,2 cross-rigidity terms currently vanish identically by Kramers symmetry, so `K_eff_Eg2` stays at its bare `K_lattice_Eg2` value.

### 12. B₁g Orbital Susceptibility χ_τ, Richardson Extrapolation, and Moriya-SCR-Regularized K_eff

```
chi_tau = ∂⟨B1g_op⟩ / ∂(g_JT · Q)      (signed; evaluated separately at Δ≠0 and Δ=0)
```

`_compute_lambda_JT_sc` — the single routine that now owns this entire diagnostic, in place of a separate `chi_tau` routine — uses an **adaptive** Richardson extrapolation: it first computes central differences of $\langle B_{1g}\rangle$ at three step sizes $h, h/2, h/4$ and forms the two Richardson estimates $R_1=(4\,CD(h/2)-CD(h))/3$, $R_2=(4\,CD(h/4)-CD(h/2))/3$. If $\vert R_1-R_2\vert /\max(\vert {\cdot}\vert ,\varepsilon)<3\%$ the mean of $R_1,R_2$ is returned at full weight (`chi_tau_weight = 1.0`). If the raw central differences disagree by more than 20% between successive step sizes (nonlinear regime), the weight is halved (`chi_tau_weight = 0.5`); if the disagreement persists, the derivative is judged unresolvable at this $Q$ and returned as **zero** (`chi_tau_weight = 0.0`). Both the SC-state (`chi_tau_sc`) and normal-state (`chi_tau_n`) susceptibilities are computed this way (the normal-state baseline matters because a finite `Delta_B1g_static` gives D₂h a small nonzero response even at $\Delta=0$); the signed excess $\chi_{\tau}^{\mathrm{net}}=\max(\chi_{\tau,n}-\chi_{\tau,sc},\,0)$ isolates the SC-triggered softening (equal to $\vert \chi_{\tau,sc}\vert -\vert \chi_{\tau,n}\vert $ when both share the same negative sign, but well-defined even if $\chi_{\tau,sc}$ changes sign) and is what enters `lambda_JT_sc` below.

**Moriya self-consistent renormalization of the lattice channel.** In the same call, `K_eff(Q)` is evaluated on a local 5-point stencil (using `compute_K_eff_full` **plus** the adiabatic correction of §11) to extract the quartic Landau coefficient of the JT free energy, $b_Q=\tfrac16\,\partial^2K_{\mathrm{eff}}/\partial Q^2$ (floored at a small positive value and validated by a stencil-smoothness diagnostic, `K_eff_spread`, before being trusted). This closely parallels the existing self-consistent Moriya damping of the *spin* channel (§14): the same fluctuation self-consistency equation is solved in closed form for the *lattice* channel,

```
Γ_Q      = ½·( −K_eff + √(K_eff² + 4·b_Q·kT) )
K_eff_reg = K_eff + Γ_Q
lambda_JT_sc = g_JT² · chi_tau_net / K_eff_reg          (only when chi_tau_net > 0 and K_eff_reg is numerically trustworthy; else 0)
```

so the JT-viability parameter used downstream (§17) is evaluated against the **thermally fluctuation-corrected** stiffness $K_{\mathrm{eff}}^{\mathrm{reg}}$, not the bare mechanical $K_{\mathrm{eff}}$ — mirroring, for the lattice order parameter, exactly the same physical idea (self-consistent renormalization of a soft mode's stiffness by its own thermal fluctuations) that `_moriya_gamma_landau` already applies to the AFM order parameter.

### 13. χ_QQ from Thermodynamic Finite Differences

The orbital JT susceptibility $\chi_{QQ}=-\partial^2\Omega/\partial Q^2$ is evaluated in the SC state by central finite difference of the total free energy, divided by 4 to correct for the combined Nambu (particle–hole) and sublattice (A–B) doubling in the 24×24 BdG matrix. This SC-state $\chi_{QQ}$ is used **exclusively** for lattice-stability diagnostics (the normal-state JT-stability check of §17); the pairing vertex itself always uses the normal-state ($\Delta=0$) susceptibilities, to avoid feeding the gap back into its own interaction.

### 14. Channel-Resolved RPA Vertex: [Γ₆, Γ₇ₐ, Γ₇ᵦ, JT] in One 4×4 Closure

The bare local interaction is **diagonal** in a 4-channel basis $[S_{\Gamma_6},S_{\Gamma_{7a}},S_{\Gamma_{7b}},\,\mathrm{JT}]$ — there is no separate bare spin–JT cross-vertex constant:

```
U(q) = diag( J_Γ6(q), J_Γ7a(q), J_Γ7b(q), V_JT_corr )
```

with $J_c(q)$ the per-doublet momentum-dependent exchange of §6 and `V_JT_corr = V_JT_eff + V_irr_QQ`, where `V_JT = g_JT_bare² / K_bare` is the bare JT pairing vertex, `V_JT_eff = V_JT · retardation_factor` applies the adiabatic phonon-retardation reduction described below, and `V_irr_QQ` is the B₁g–B₁g component of the **irreducible vertex extracted from the four-site plaquette cluster ED** (§19) — the local, non-perturbative renormalization of the JT self-interaction. All spin–JT **mixing** in the RPA matrix comes from the off-diagonal bare susceptibilities themselves (below), not from a bare cross-interaction.

**Phonon retardation.** Because `M_eff` (an input reduced mass, amu) is now a primary parameter, the code derives a genuine bare JT phonon energy $\hbar\omega_0 = 64.6528\,\mathrm{meV}\cdot\sqrt{K_{\mathrm{bare}}[\mathrm{eV/\mathring A^2}]/M_{\mathrm{eff}}[\mathrm{amu}]}$ (`_full_rebuild`/`RMFT_Solver.__init__`), and uses it to evaluate the static-limit reduction of a single-Einstein-mode phonon propagator at the characteristic gap scale $\Delta_{\mathrm{typ}}=\max(\vert \Delta_{s7a}\vert ,2\vert \Delta_{d7a}\vert ,\vert \Delta_{s7b}\vert ,2\vert \Delta_{d7b}\vert )$:

```
retardation_factor = x² / (x² + 1),   x = ω₀ / Δ_typ
```

— a one-point evaluation of $D(i\Omega)/D(0)=1/(1+(\Omega/\omega_0)^2)$ at $\Omega=\Delta_{\mathrm{typ}}$, not a full Matsubara average. It is 1 (no suppression) in the static/anti-adiabatic limit $\omega_0\gg\Delta_{\mathrm{typ}}$ and at $\Delta_{\mathrm{typ}}\to0$ (nothing to retard against yet, e.g. the very first linearized kick), and suppresses the phonon-mediated coupling once the gap grows comparable to or past the bare phonon energy. It applies **only** to the phonon-mediated `V_JT`, never to `V_irr_QQ` (a purely electronic irreducible vertex with no phonon propagator of its own to retard).

**Bare susceptibilities.** $\chi_0(q)$ comes from the $\Delta=0$ BdG Hamiltonian via the static Lindhard formula, `_lindhard_bubble`/`_chi0_channels_raw`, accelerated with `opt_einsum`. `_chi0_channels_raw` builds the full raw $4\times4$ matrix $\chi_{ab}(q)=\langle\!\langle O_a;O_b\rangle\!\rangle$ (vertices $O_c$ = per-doublet uniform $S_z$ for $c\in\{\Gamma_6,\Gamma_{7a},\Gamma_{7b}\}$, $O_{\mathrm{JT}}=\partial H/\partial Q$) as **one Gram product** of four band-basis vertex tensors against a shared Lehmann kernel — the same asymptotic cost as the earlier scalar (spin, JT) routine, now resolving all three spin channels individually. The normal-state sum runs over the pre-built cyclic `shift_table` (§ Key Algorithms) rather than a second diagonalization, and is additionally weighted by the same ZRS spectral weight $\beta^2(k)\beta^2(k+q)$ that modulates the JT coupling itself (§9). The static Lindhard function is real by time-reversal symmetry, so its imaginary part is discarded as roundoff, not physical information, after Hermiticity is enforced. `_rpa_closure` then Moriya-damps only the $3\times3$ spin sub-block (via the model-derived $\Gamma_M$ of the next paragraph), PSD-projects the full $4\times4$ matrix by eigenvalue clamping (Higham nearest-PSD, guarding against Cauchy–Schwarz violations from numerical noise near the QCP), and returns both the full matrix (`chi0`) and legacy-compatible scalar aggregates:

```
chi_SS = Σ_cc' χ0[c,c']       (sum over the 3 spin channels — reduces to a single Γ6 entry if only Γ6 is populated)
chi_SQ = chi_QS = Σ_c χ0[c, JT]
chi_QQ = χ0[JT, JT]
```

Off-diagonal spin–spin entries $\chi_{cc'}$ ($c\neq c'$) vanish for orbitally-isotropic hopping at $Q=0,\Delta=0$, but not for $Q\neq0$ (JT-induced Γ6–Γ7 mixing) or in the SC state.

**Vertex assembly (`_rpa_closure`/`_rpa_det`).** The full closure builds the $4\times4$ RPA denominator $M(q) = I - \chi_0(q)\,U(q)$ and its floor-regularized determinant (floored in magnitude only, at $\max(\varepsilon,10^{-4}\vert M(q)\vert _F)$, never in sign, so a genuine QCP crossing is never masked); `_rpa_det` survives as the simpler **2×2** scalar reduction of the same construction (equivalent when only the $\Gamma_6$ channel is populated), still used by a few lighter-weight callers such as `_moriya_gamma_landau`. The channel-space inverse enters the FS pairing vertex (§15) as $\chi_{\mathrm{RPA}}(q) = \mathrm{adj}(M(q))\,\chi_0(q)/\det M(q)$ (the adjugate form stays finite even as $\det M(q)\to0$), hard-clamped in the summed pairing-channel contribution to $\pm V_{\mathrm{cap}}$, with `V_cap = _RPA_V_CAP_ALPHA · max(_RPA_BW_FACTOR·max(|tx|,|ty|), J_eff)` (`_RPA_V_CAP_ALPHA = 2.2`, `_RPA_BW_FACTOR = 8` for the tight-binding bandwidth estimate) — a numerical overflow guard only; the sign and divergence character of $V(q)$ near the QCP are never altered, and a negative determinant (past the QCP) is deliberately left untouched rather than capped, so the SCF is not artificially trapped away from a genuinely unstable regime.

**Two separately tracked determinants.** The vertex cache stores `det_q0` (the $q=0$, ferromagnetic-channel determinant) and `det_afm` (the $q=(\pi,\pi)$, AFM-channel determinant) independently. SCF adaptive mixing and convergence behavior respond to `det_afm`; `det_q0` guards separately against an accidental ferromagnetic divergence. Both are logged at convergence (`dFM=`, `dAFM=` in the iteration log).

**Sign-flip EMA guard.** When $\vert \mathrm{det\_afm}\vert <$ `_DET_SIGN_FLIP_SCALE = 0.05` and the d-wave vertex `V_d_scalar` would flip sign relative to its cached value, the update is blended continuously rather than switched, via a sigmoid in $\vert \mathrm{det\_afm}\vert /0.05$ (steepness `_EMA_SIGN_FLIP_SLOPE = 6.0`, floor `_EMA_SIGN_FLIP_W_MIN = 0.20`) — so the blend weight shrinks toward its floor near the QCP (preserving genuine sign ambiguity there, where a real physical crossover may be in progress) and grows toward 1 away from it (suppressing pure numerical noise).

**Moriya damping** ($\Gamma_M$, the spin-channel analogue of §12's $\Gamma_Q$) is obtained self-consistently from the model's own Landau free-energy expansion rather than an empirical closed-form fit: `_moriya_gamma_landau` probes $F(M)$ at $M=0,\pm h,\pm2h$ (`_MORIYA_LANDAU_M_STEP = 0.06`, a 5-point stencil) on the $\Gamma_6$-only channel via `_compute_bdg_free_energy`, extracts the quartic Landau coefficients $a,b$, and solves the self-consistent fluctuation equation $\Gamma_M=b\langle\delta M^2\rangle$, $\langle\delta M^2\rangle=kT/(a+b\langle\delta M^2\rangle)$ in closed form: $\Gamma_M=({-}a+\sqrt{a^2+4bkT})/2$ (reducing to the classical $bkT/a$ limit when $a^2\gg4bkT$, and staying finite as $a\to0$). The result is capped at $\Gamma_{M,\max}=4t_{\mathrm{eff}}^2/(\pi J_{\mathrm{eff}})$ and cached per doping.

### 15. FS-Resolved Pairing Kernel: Orbital-Character-Weighted, Full Channel Matrix

`compute_pairing_kernel_and_build_cache` builds the actual momentum-space pairing interaction seen by the gap equation, and it is **not** simply the scalar $J_{\mathrm{eff}}^2\chi_{SS}^{\mathrm{RPA}}+V_{JT}^2\chi_{QQ}^{\mathrm{RPA}}+\ldots$ of a single aggregate spin channel: it is a genuine sum over the doublet-channel content of the Fermi-surface quasiparticle at **each** of the two scattering momenta. `_fs_channel_weights` first extracts, at every sampled FS point $k$, the doublet content $w_c(k)$ ($\sum_c w_c=1$, from the squared overlap of the lowest positive-energy BdG band with each channel's orbital block). The spin-mediated and cross (spin–JT) contributions to the pairing interaction between two FS points $k,k'$ (momentum transfer $q=k-k'$) are then

```
V_spin (k,k') = − Σ_{c,c'} √(w_c(k) w_c(k'))·√(w_{c'}(k) w_{c'}(k'))· J_c(q) J_{c'}(q) · [χ_RPA(q)]_{cc'}
V_cross(k,k') =   2 Σ_c √(w_c(k) w_c(k'))· J_c(q) V_JT_eff · [χ_RPA(q)]_{c,JT}
V_JT   (k,k') =   V_JT_eff² · [χ_RPA(q)]_{JT,JT}
```

i.e. the spin-fluctuation pairing kernel is a full $3\times3$ contraction over the doublet channel indices, weighted at each leg by that leg's own orbital character — a k-point sitting purely on the $\Gamma_6$ Fermi surface pocket "sees" a different effective $J\chi_{SS}^{\mathrm{RPA}}$ than one with substantial $\Gamma_7$ admixture, and the two legs of the same scattering process can differ. A local B₁g-character weight $c_B(k)=\vert \varphi_d(k)\vert /\max_{k'}\vert \varphi_d(k')\vert \in[0,1]$ then symmetry-gates the spin and cross channels — `V_w(k,k') = V_JT(k,k') + c_B(k)c_B(k')\cdot[V_{\mathrm{spin}}(k,k')+V_{\mathrm{cross}}(k,k')]` — so that both electrons antinodal recovers the full RPA-dressed interaction, either leg nodal keeps only the always-on JT piece, and pure d-wave symmetry is never accidentally broken by a spin contribution surviving at one nodal leg. This construction is used consistently for **both** the eigenvalue diagonalization and the $(s,d)$ channel projections feeding the gap equation and the condensation energy.

The FS-weighted symmetric kernel $K_{\mathrm{fs}}(k,k')=\sqrt{w_k}\,V_w(k,k')\,\sqrt{w_{k'}}$ (with $w_k$ the thermal 2D integration weight of `_get_fs_points`, a symmetry-adapted D₄h/D₂h thermal element with no $\vert v_F\vert $-floor singularity, rather than a floored $dl/v_F$ arc-length measure) is diagonalized; its largest eigenvalue `lambda_lin_max` and eigenvector `(v_s_raw, v_d_raw)` set the s/d channel ratio blended into the SCF gap update (§21). The scalar projections `V_s_scalar`, `V_d_scalar`, `V_sd` used by both cross-doublet branches (§8) are the same $\varphi_s$/$\varphi_d$ projections of this one weighted kernel. `lambda_JT_kernel` (the Rayleigh projection of the eigenvector onto the JT-only sub-block `W22_JT`) is the FS-resolved companion to the $q{=}0$ estimate `lambda_JT_sc` of §12.

### 16. Normal-State JT Stability and the Post-Convergence Hessian Mode Classification

The lattice stability check in the normal state is now a direct threshold comparison rather than a dedicated instability-matrix data structure: `K_spont = g_JT² · χ_QQ(Δ=0, q=0)` is evaluated once at the self-consistent $M$, and `K_lattice > K_spont` is logged as the normal-state JT-stability verdict (`__main__`, "✓ normal state JT-stable" / "⚠ normal state already JT-unstable"). Separately, whether the *SC-triggered* mechanism is active at all is the `lambda_JT_sc > _LAMBDA_JT_VIABLE` threshold of §12/§17.

Which mode of the system is actually unstable is instead read off the **post-convergence finite-difference Hessian** (`compute_hessian`, §18): a $(3{+}1{+}1)$-dimensional probe over $(M_{\Gamma_6},M_{\Gamma_{7a}},M_{\Gamma_{7b}},Q,\Delta_{\mathrm{tot}})$ of the same canonical free energy used for §11's mechanical stiffness, with the four gap channels probed as one combined amplitude along their converged relative phases. Its eigenvector with the smallest (trust-region-rescaled) eigenvalue is decomposed into fractional weight on the $M$-block, the $Q$-component, and the $\Delta$-component (`_fM, _fQ, _fD`), and classified against `_MODE_FRAC_DOMINANT = 0.60`/`_MODE_FRAC_MIXED = 0.30` as `pure-SC`, `pure-JT`, `SC-triggered-JT` (both $\Delta$ and $Q$ above the mixed threshold — the theory's target signature), `AFM-fluctuation`, or `mixed`. This classification drives the saddle-escape "kick" mechanism inside `solve_self_consistent` (§ Key Algorithms) rather than being reported as a separate normal-state diagnostic block.

### 17. SC-JT Coexistence: Two Independent Thresholds

Two dimensionless quantities, evaluated at different stages of the pipeline, together determine whether the mechanism is operative for a given `K_lattice`:

| Symbol | Formula | Evaluated at | Threshold |
|---|---|---|---|
| — (normal-state stability) | $K_{\mathrm{lattice}}$ vs. $K_{\mathrm{spont}}=g_{JT}^2\chi_{QQ}(\Delta=0)$ | normal state | $K_{\mathrm{lattice}}>K_{\mathrm{spont}}$ required (else spontaneous, SC-independent JT) |
| `lambda_JT_sc` | $g_{JT}^2\cdot\chi_{\tau}^{\mathrm{net}}/K_{\mathrm{eff}}^{\mathrm{reg}}$ (§12, Moriya-SCR-regularized) | SC state, post-SCF | $>$ `_LAMBDA_JT_VIABLE` $=0.05$ ⇒ SC-triggered JT active |

Both checks are logged directly in `__main__` from the converged `ref` result and a matching $\Delta=0$ reference solve at the same $M$, rather than assembled by a dedicated solver method or plotting routine. There is no longer a separate upper stiffness bound `K_SC` or geometric-midpoint `K_opt` computed as its own quantity: the SC-side viability is entirely carried by the `lambda_JT_sc` threshold above, which already folds in the fluctuation-regularized $K_{\mathrm{eff}}^{\mathrm{reg}}$.

---

### 18. Variational Free Energy and the Cluster Decomposition

The total free energy splits, without double-counting, into an itinerant and a local piece:

```
F_total = F_bdg + F_cluster
```

This is a Luttinger–Ward/Baym–Kadanoff-style variational decomposition: `F_bdg` (`_compute_bdg_free_energy`) covers the itinerant mean-field BdG spectrum plus the condensation-energy terms summed over **all four** gap channels, $\sum_{i\in\{s7a,d7a,s7b,d7b\}}\vert \Delta_i\vert ^2/(g_{\Delta,i}V_i) + (K_{\mathrm{eff}}/2)Q^2$ (plus the analogous Eg,2 term); `F_cluster` (`compute_cluster_free_energy`) covers local quantum fluctuations from an exactly diagonalized four-site (2×2 plaquette) cluster (§19). Gutzwiller factors handle kinematic Mott renormalization; the cluster ED handles irreducible-vertex renormalization of the local B₁g self-interaction only (never the itinerant susceptibility bubble itself); RPA handles the reducible ladder summation over the full BZ. These three levels are orthogonal — cluster-ED outputs a renormalized coupling (`V_irr_QQ`) that RPA then uses as an *input* to `V_JT_corr` (§14), so there is no overlap between what each layer computes.

### 19. Four-Site Plaquette Cluster: Quantum Fluctuations and Exact Vertex Extraction

Beyond the BdG mean field, a **four-site (2×2 open plaquette)** cluster, each site carrying the full 6-orbital Kramers-doublet basis (a $6^4=1296$-dimensional Hilbert space), is exactly diagonalized every SCF iteration by `compute_cluster_free_energy`. The checkerboard geometry is

```
0 --x-- 1
|       |
y       y
|       |
3 --x-- 2
```

with sites 0, 2 on sublattice A and 1, 3 on sublattice B; each site has exactly 2 intra-cluster nearest-neighbor bonds (bond sign $\eta=+1$ for $x$-bonds, $\eta=-1$ for $y$-bonds in the B₁g channel, mirroring the real-space $\cos k_x-\cos k_y$ bond weighting; the A₁g/magnetic channel is bond-direction-independent) and $Z_{\mathrm{eff}}=Z-2$ external neighbours absorbed into the mean-field Weiss embedding, so intra-cluster bonds are never double-counted. The local single-site Hamiltonians (`build_local_hamiltonian_for_bdg`) include $-\mu$, $\Delta_{CF}$ (and the $\Gamma_{7b}$ internal splitting `g7split`), the AFM Weiss field, the JT coupling $g_{JT}^{\mathrm{cluster}}\,Q\,B_{1g,\mathrm{op}}$, and — only once $Q\neq0$ — the anomalous Weiss field built from **both** `F67a_s` and `F67b_s`, each weighted by its own cross-doublet exchange constant `J_B1g^(7a)`/`J_B1g^(7b)` (§6, §8), all scaled by the same cluster-averaged downfolding weight $\beta_{\mathrm{cluster}}$ used for the ZRS spectral weight, so the cluster sees the same ligand-projected physics as the BdG Hamiltonian. The cluster Hamiltonian is

```
H_cluster = Σ_site H_site  +  Σ_bonds [ J_bond_A1g · (multi_op ⊗ multi_op) + η · J_bond_B1g · (B1g_op ⊗ B1g_op) ]
```

**Vertex extraction via cluster inverse susceptibilities.** Rather than a linear regression against connected correlators, the irreducible vertex is extracted the standard cluster-ED way: an exact static Lehmann susceptibility tensor of shape $(8,8)$ (2 channels — $S_z$ and $B_{1g}$ (still one aggregate B₁g channel at this level, not separately per cross-doublet branch) — × 4 sites) is computed for two reference spectra — `chi0_tensor` (four independent sites, no intra-cluster exchange, each still seeing the full-lattice Weiss space) and `chi_full_tensor` (the same cluster **with** intra-cluster exchange switched on, at $M=0$, no anomalous Weiss field, no JT — the appropriate *normal-state* reference for a lattice Bethe–Salpeter equation). The irreducible vertex follows from the inverse-susceptibility difference,

```
Γ_ED = χ0⁻¹ − χ_full⁻¹                    (both inverses stabilized: null eigenvalues regularized, not the physical ones)
```

computed in the full 8×8 site×channel space and only *then* projected onto the staggered (spin, weight $\tfrac12(1,-1,1,-1)$ across the 4 sites) and uniform (B₁g, weight $\tfrac12$ on all 4 sites) subspaces — projection and matrix inversion do not commute, so this ordering is essential. The resulting $2\times2$ projected vertex `V_irr` has a single quantity fed back into the RPA vertex construction (§14): `V_irr_QQ = V_irr[1,1]`, the B₁g–B₁g irreducible coupling, which additively renormalizes the (retardation-reduced) bare JT pairing vertex, `V_JT_corr = V_JT_eff + V_irr_QQ`. The same routine also returns the per-site $\langle B_{1g}\rangle$ expectation values (`b_mean`), an intra-cluster B₁g fluctuation amplitude `Q_fluct` $=\sqrt{\langle B_{1g}^2\rangle-\langle B_{1g}\rangle^2}$ averaged over the 4 sites, the mechanical cluster stiffness correction `Delta_K_cluster` (§11), an alternative curvature-based cross-check `V_irr_QQ_curvature` $=4g_{\mathrm{eff}}^2(1/K_{\mathrm{full}}-1/K_{\mathrm{bare}})$ (a diagnostic, not fed back), and the cluster free energy per site `F_per_site` (with the mean-field double-counting correction $\tfrac12 Z_{\mathrm{eff}}J_{\mathrm{bond}}M^2$ added back).

Including the physical (JT- and anomalous-Weiss-including) cluster ground state used for `F_per_site`/`b_mean`/`Q_fluct`, and the two normal-state reference spectra used to build the susceptibility tensors, **three** $1296\times1296$ exact diagonalizations run per call — plus a 5-point-stencil sweep of the mechanical stiffness itself, each point its own diagonalization, making this cluster-ED evaluation the single most expensive step in the SCF loop.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

`_find_mu_for_density` solves $\langle n\rangle = 1-\delta$ by Newton's method using the analytic derivative $\partial n/\partial\mu=\sum_{k,n} w_k f(1-f)/kT\cdot(\vert u\vert ^2+\vert v\vert ^2)$ from the same BdG eigensystem, backtracking on step failure, with Brent's method as a guaranteed fallback bracket-and-bisect. Above `_MU_SC_DERIV_THRESH` total gap amplitude the analytic derivative (exact only for the pure normal-state branches) is replaced by a centered numerical derivative. The `(ev, ec)` pair from the μ-search is reused directly for the subsequent observable computation, avoiding a redundant diagonalization.

### 21. Gap Equations: 4-Channel Fixed Point, Complex Phase, and the Shared 2×2 Pairing Kernel

`VectorizedBdG.compute_gap_eq_vectorized` evaluates the gap equations over the full BZ for **both** cross-doublet branches at once, keeping the Fock sums **complex** rather than taking `abs(·)` before forming the new gap — because the BdG Hamiltonian is genuinely complex (SOC enters through $L_y\propto(L_{+}-L_{-})/2i$ and $S_y$), the Nambu eigenvectors are complex at every $k$-point, and forcing a real magnitude at every iteration would erase the physical relative phase between the channels and destabilize convergence. The (single, shared) real, FS-averaged 2×2 pairing kernel

```
K_pair = [[ K11, K12 ], [ K12, K22 ]]      (s/d basis; K11 uses the JT-only vertex, K22 the full weighted vertex, §15)
```

is built inline from the already-available FS grids at vertex-cache rebuild time, with `K11`/`K22` scaled by whichever branch's own Gutzwiller factor is being evaluated (`g_Delta_s`/`g_Delta_ad` for the a-branch, `g_Delta_s`/`g_Delta_bd` for the b-branch); its dominant eigenvector `(v_s, v_d)` gives the SCF the optimal s/d hybridization direction for **both** branches simultaneously, blended into each branch's fixed-point gap update with weight `_ALPHA_MIX_2X2 = 0.56` — but only the real relative *magnitude* ratio is taken from `K_pair`; the actual complex phases of each of the four channels are always re-applied after blending, so the self-consistent phase is never silently overwritten. Both branches then pass through the identical det-proportional jump cap (`_DELTA_JUMP_CAP` with an exponential penalty past the AFM QCP) and per-branch seed threshold before being reshaped into the 4-vector `Delta_out = [Δ_s7a, Δ_d7a, Δ_s7b, Δ_d7b]`. The same phase-freezing logic is used in `compute_hessian`, which fixes the converged phases before taking finite-difference probes.

The **Anderson-accelerated** part of the mixing (§ Key Algorithms) only tracks the leading $\Gamma_{7a}$ branch magnitudes explicitly, as part of a shared 3D vector with $Q$; the $\Gamma_{7b}$ branch (and the phases of all four channels) still passes through the linear fixed-point/Newton blend every iteration, just without being folded into the same Pulay history — a deliberate asymmetry that keeps the Anderson history low-dimensional while still leaving $\Gamma_{7b}$'s amplitudes as genuine converging degrees of freedom of the overall fixed point.

### 22. Incommensurate AFM Nesting Check

Because the BdG Hamiltonian is fixed to commensurate AFM ordering at $Q_{AFM}=(\pi,\pi)$, `_scan_incommensurate_nesting` separately checks whether the normal-state spin susceptibility $\chi_{SS}$ (evaluated with the staggered $S_z$ vertex, exact only at $q=(\pi,\pi)$ itself — off-commensurate points are explicitly flagged in the code as a "staggered-approximation" scan rather than the exact physical $\chi_{SS}(q)$) would actually prefer a nearby incommensurate wavevector $q^{*}=(\pi,\pi-\delta q)$, scanning $\delta q\in[0,0.15\pi]$ at the converged $(M,Q,\mu)$. If the scan finds the maximum away from $\delta q=0$ within the window $0.05\pi<\delta q^{*}<0.10\pi$, `solve_self_consistent` automatically retries once with a softened AFM seed ($M\to0.85\,M$, applied by re-solving with the current total gap amplitude as the seed), guarded by the `_ic_retry` flag to prevent infinite recursion. This does not change the ordering wavevector used in the Hamiltonian itself — it only flags, and mildly compensates for, the possibility that the true instability sits away from $(\pi,\pi)$.

### 23. Temperature-Dependent Tc Estimates

Three independent estimates target different aspects of the transition, deliberately not sharing a single label:

- **Tc₁ — Allen–Dynes/McMillan-type spin-fluctuation formula:** $T_{c1}=(\omega_{SF}/D)\cdot\exp(-N\cdot(1+\lambda_{\max})/\lambda_{\max})$, with constants $D=$ `_MAD_DENOM` = 1.13, $N=$ `_MAD_NUM` = 1.04, $\omega_{SF}=J_{\mathrm{eff}}$ (paramagnon bandwidth), and $\lambda_{\max}$ from the linearized gap equation at the reference doping — a fast analytic estimate, not a full temperature scan.
- **Tc₂ — λ(T)=1 crossing:** `compute_lambda_vs_T` re-runs `solve_self_consistent` at each of 20 log-spaced temperatures between $0.25\,kT$ and $4\,kT$ of the reference run, with $\Delta$ **strictly forced to zero** so $M(T)$ and $Q(T)$ relax self-consistently on a genuinely normal-state background rather than a fixed $T{=}0$ SC-biased one (this avoids the artifact of the AFM Weiss field being frozen at its condensed-state value, which would otherwise keep $\lambda_{\max}$ from ever reaching 1); $T_{c2}$ is where $\lambda_{\max}(T)=1$, taken as the lowest-temperature crossing if the (generically non-monotone, in a first-order-like regime) curve crosses more than once.
- **Tc₃ — thermodynamic, first-order-aware:** `compute_Tc_thermodynamic` performs a single upward-heating temperature scan, warm-started from the converged $T\approx0$ SC+JT basin, comparing $F_{SC}$ (from a warm-started SC-basin solve at each $T$) against a separately relaxed normal-state free energy at every point. Because the effective Landau potential for the condensate can have a negative effective quartic coefficient here, the transition can be genuinely first-order, and a naive cooling-from-$\Delta{\approx}0$ scan (which only finds the spinodal) can badly underestimate $T_c$. The routine returns both the thermodynamic crossing `Tc` and the spinodal collapse `Tc_spinodal`, the transition order, the gap jump `Delta_jump`, and — for near-second-order cases (`D_spinodal/Δ₀ < 0.15`) — a Ginzburg–Landau-refined spinodal from fitting $\Delta^2(T)=a(T-T_c)$ to points with $\vert \Delta\vert >2\,\mathrm{meV}$.

`compute_Tc_by_gap_suppression` (cooling-only spinodal search) is retained as an independent cross-check. The $2\Delta_0/k_BT_c$ ratio reported in the Tc block is computed against $T_{c3}$, the most physically complete of the three. All three routines seed and warm-start through ordinary `solve_self_consistent` calls at cloned temperatures (`_clone_solver_at_T`); there is no separate specialized warm-start routine — a normal-state point is simply `solve_self_consistent(doping, initial_Delta=0.0)`.

---

## Model Architecture

```
ModelParams  (dataclass, __post_init__ runs the SOC+CF diagonalization and the superexchange cluster ED)
    ├── Primary inputs:  t_pd, t_prime_ratio, t_dprime_ratio, U_dd, lambda_soc, Delta_tetra,
    │                    g_JT, K_lattice, lambda_hop, g_Eg2, K_lattice_Eg2, Delta_B1g_static,
    │                    hybrid_scale, Upp_ratio_bare, J_H_ratio, Delta_CT, M_eff, Z, kT, tol
    ├── Derived scalars: Delta_CF, g7split, t0, t_prime, t_dprime, p_7, thermal_fs_nat_size,
    │                    J2_over_J1, J3_over_J1, Z_afm_eff, N_k
    │                    (U_pp and J_H are computed and consumed internally during __post_init__
    │                     to build the superexchange cluster Hamiltonian; neither is stored on
    │                     the instance. The old scalar J_pdct no longer exists — see below.)
    ├── Derived arrays:  sz_op (exact ⟨Sz⟩ per Kramers partner, 6 components), multi_op, B1g_op,
    │                    B1g_offdiag, Eg2_op, _w_orb (3×3: Γ6/Γ7a/Γ7b rows × xz/yz/xy columns),
    │                    kappa_doublets (3: Γ6, Γ7a, Γ7b intra-doublet exchange, §4),
    │                    kappa_cross (2: Γ6–Γ7a, Γ6–Γ7b cross-doublet exchange, §4),
    │                    Tx_A_xz/_yz/_xy (orbital-selective hopping projectors, sum to I₆)
    ├── Grid objects:    k_points, k_weights, shift_table (_NK×_NK×N_k int32 cyclic shift
    │                    table for arbitrary q)
    └── Methods:         get_gutzwiller_factors() (static admixture-based estimate),
                         exchange_channels() (now returns 2 J_B1g scalars: 7a, 7b),
                         _exchange_J_q(), effective_hopping_anisotropic(), hopping_matrices(),
                         hopping_matrices_dQ(), wave_function_weight() (T_Q-symmetrized)

_SolveState  (dataclass, mutable per-SCF-run state — never stored on self)
    ├── V_d_ema: Optional[float]         # persistent V_d sign-flip EMA
    └── _ema_kick_pending: bool          # doubles blend weight for one iter after a kick

RMFT_Solver
    ├── Initialization: __init__ (also derives _omega_0_JT, the bare JT phonon energy, from
    │                   M_eff and K_bare), _rebuild_orbital_operators, _get_vbdg,
    │                   _get_chi0_norm_cache, _reset_transient_state, _shallow_clone,
    │                   _clone_solver_at_T, _full_rebuild
    ├── JT rigidity:    _calc_dHdQ (∂H/∂Q in the band basis), compute_K_eff_full (mechanical
    │                   exchange contribution to the B₁g/Eg,2 stiffness, §11),
    │                   _delta_relaxed_K_eff_correction (adiabatic Δ-relaxation softening, §11),
    │                   _jt_retardation_factor (adiabatic phonon-propagator reduction, §14)
    ├── Susceptibilities: B1g_expectation, Eg2_expectation, _compute_lambda_JT_sc (χ_τ,
    │                   Richardson extrapolation, and the Moriya-SCR-regularized K_eff_reg /
    │                   lambda_JT_sc, all in one call, §12), _chi_QQ_matrix_elements,
    │                   _compute_nambu_kernel, _compute_nambu_susceptibility, _diamagnetic_QQ_term
    ├── RPA vertex:     _rpa_det (2×2 scalar reduction), _chi0_channels_raw (4×4 channel-
    │                   resolved raw susceptibility, §14), _rpa_closure (4×4 closure: Moriya
    │                   damping + PSD projection + determinant, the primary entry point),
    │                   _moriya_gamma_landau, _make_vertex_params
    ├── Gap equation:   compute_pairing_kernel_and_build_cache (orbital-character-weighted FS
    │                   pairing kernel shared by both cross-doublet branches, §15, using the
    │                   module-level _unique_q_pairs helper and FS integration weights from
    │                   _get_fs_points), scf_gap_diagnostics (coherence lengths; combines both
    │                   branches' s/d amplitudes for the gap-magnitude-dependent diagnostics)
    ├── Local H / μ:    build_local_hamiltonian_for_bdg, _find_mu_for_density,
    │                   estimate_gutzwiller_factors_occupation_based (dynamic, occupation-based
    │                   Γ7a/Γ7b Gutzwiller factors, §5)
    ├── Free energy:    _compute_bdg_free_energy, compute_cluster_free_energy
    │                   (four-site plaquette ED + Γ_ED vertex extraction, both cross-doublet
    │                   branches wired into the physical-state Hamiltonian, §19)
    ├── SCF machinery:  _scf_kick (initial-condition seeding via the linearized pairing
    │                   eigenvalue and an early finite-difference Hessian probe), refine_M_state
    │                   (Anderson-accelerated M/μ refinement at fixed Q, Δ_vec — generalizes the
    │                   normal-state-only warm start), _vertex_matrix_at_Q, _classify_scf_dynamics,
    │                   _anderson_mix, _mix, _project_kick_from_hessian
    ├── Main solve:     solve_self_consistent   ← the ~700-line Anderson-accelerated fixed point
    │                   over (M, Q, Δ_s7a, Δ_d7a, Δ_s7b, Δ_d7b, μ)
    ├── Post-hoc:       _scan_incommensurate_nesting, compute_dF_dM_channels_and_hessian,
    │                   compute_dF_dDelta_and_d2F (4×4, one channel per Δ amplitude),
    │                   compute_hessian ((3+1+1)-dim, Δ probed as one combined amplitude)
    ├── Tc:             compute_Tc_by_gap_suppression, compute_Tc_thermodynamic,
    │                   compute_lambda_vs_T
    ├── Diagnostics:    _get_fs_points, compare_cluster_vs_bdg, _fs_channel_weights
    └── Occupation:     _compute_orbital_densities

VectorizedBdG   (thin batched-LAPACK wrapper bound to one RMFT_Solver)
    ├── _build_H_stack                     builds & Hermitizes the (N_k, 24, 24) BdG stack from a
    │                                       4-component Delta_vec and a 4-component F67_vec,
    │                                       covering the Γ6–Γ7a AND Γ6–Γ7b pairing/Weiss terms
    │                                       (a legacy 2-component Delta_vec is still accepted,
    │                                       reproducing the Γ7b-decoupled model with Δ_7b≡0)
    ├── compute_channel_staggered_magnetizations   → per-channel (Γ6, Γ7a, Γ7b) M_stag
    └── compute_gap_eq_vectorized           → (Delta_out (4,), vertex_cache) — both cross-doublet
                                               branches solved through the same shared 2×2 pairing
                                               kernel, each with its own Gutzwiller factor and
                                               anomalous amplitude (§8, §15, §21)

plot_ground_state_comparison(results, labels=None)   — standalone function, not a class method;
    builds the 2×2 "ref vs. normal vs. SC_Q0" free-energy/order-parameter comparison figure
    described in Installation & Usage / Output & Diagnostics. It is a plotting routine only —
    it does not itself assemble any stability verdict (contrast §17).
```

Module-level functions supporting the superexchange cluster ED (§4) — `_apply_hop`, `_multiorbital_site_of`, `_multiorbital_2hole_hamiltonian`, `_multiorbital_reference_vector`, `_lowdin_branch_energies`, `_intra_doublet_kappa`, `_cross_doublet_kappas` — and the module-level `_compute_F67_pair_amplitudes` (all four anomalous amplitudes in one call, §2) and `_pairing_strengths` (a small free function, not an `RMFT_Solver` method, used by both the SCF loop's instability-EMA update and the free-energy condensation term to turn a vertex cache into the five scalars `(V_s, V_d_a, V_sd_a, V_d_b, V_sd_b)`) round out the physics layer.

---

## Key Algorithms

### SCF Loop (`solve_self_consistent`)

An Anderson(5)-accelerated fixed point over $(M, Q, \Delta_{s7a}, \Delta_{d7a}, \Delta_{s7b}, \Delta_{d7b}, \mu)$, per iteration:

1. Build and diagonalize the 24×24 BdG stack for the current $(M,Q,\Delta_{\mathrm{vec}},\mu)$.
2. If SC+JT are both active ($\sum\vert \Delta_i\vert $ above a small threshold), compute all four Gorkov anomalous amplitudes (`F67a_s`, `F67a_d`, `F67b_s`, `F67b_d`, via `_compute_F67_pair_amplitudes`) and inject them into the transverse (off-diagonal) Weiss field — this is the joint-$(\Delta,Q)$ activation loop of §2 — then rebuild the BdG cache.
3. Recompute the occupation-based Gutzwiller factors `g_Delta_s`, `g_Delta_ad`, `g_Delta_bd` (§5) and the exchange tensor $J_{A1g}$, $J_{B1g}^{(7a)}$, $J_{B1g}^{(7b)}$ (§6), then the four-site cluster free energy and its stiffness correction (§19), then `_make_vertex_params` — including the current adiabatic phonon-retardation factor (§14) — to get $\Gamma_M$, `V_JT_eff`, `V_JT_corr`, `V_cap`.
4. Evaluate the SC-state RPA determinant at $q=(\pi,\pi)$ and $q=0$ (`_rpa_closure`), and the Q Newton step (`_compute_Q_newton_step`, using the full $K_{\mathrm{eff}}$ of §11).
5. Solve the gap equations for **both** cross-doublet branches via the shared RPA pairing kernel (§15, §21); blend in the 2×2 pairing-kernel eigenvector direction with weight `_ALPHA_MIX_2X2 = 0.56` for each branch, and take a 4-channel Newton step (`compute_dF_dDelta_and_d2F`) blended in with weight `_ALPHA_HF_DELTA = 0.20`.
6. Newton step for `M` (Levenberg–Marquardt-damped, trust-region-limited), blended with the *linearly*-mixed BdG fixed point — `M` is deliberately **excluded** from the Anderson history and updated by this separate Newton/blend rule.
7. Adaptive Hellmann–Feynman update for `Q`: $Q_{\mathrm{out}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$, injected into the Anderson vector when the implied displacement is significant, on iteration 0, or on a periodic safety heartbeat (`_Q_UPDATE_PERIOD = 4` iterations); otherwise left untouched, consistent with the lattice's adiabatic timescale.
8. Anderson(5) mixing is applied jointly to $(Q,\ \vert \Delta_{s7a}\vert ,\ \vert \Delta_{d7a}\vert )$ only — a Tikhonov-regularized least-squares solve over the last 5 residuals; the $\Gamma_{7b}$ amplitudes and all four phases pass through the linear fixed-point/Newton blend of step 5 every iteration without entering this reduced Anderson history (§21). $\mu$ is then re-solved exactly (Newton+Brent) at the freshly mixed point rather than carried as an imperfectly converged fast variable, so its residual error cannot leak into the Anderson history.
9. Adaptive mixing rate: $\alpha_{\mathrm{eff}}=\alpha_0/(1+\Lambda_{\mathrm{inst}})$, where $\Lambda_{\mathrm{inst}}$ is an EMA of the worst current instability indicator (the five `_pairing_strengths` outputs, $\lambda_{JT}$, or $J\chi_{SS}$); $\alpha$ is halved and the Anderson history reset on divergence, and a **limit-cycle detector** (relative std of the total $\vert \Delta\vert $ over the last `_CYCLE_WINDOW` iterations $>$ `_CYCLE_THRESHOLD`) damps $\alpha$ and resets history if the SCF is oscillating rather than converging.

Convergence requires $\max(\vert \Delta M\vert ,\vert \Delta Q\vert ,\vert \Delta\vert \Delta_i\vert \vert ,\vert \Delta F_{67}\vert )<$ `tol` (over all four gap channels and all four anomalous amplitudes) and density error $<10\cdot$`tol`, with no pending saddle-escape kick on the same iteration. After convergence (or exhausting `_MAX_ITER = 500` iterations), `_classify_scf_dynamics` labels the trajectory as `converging`, `limit_cycle`, `first_order_jump`, `hysteretic`, or `stagnating` from the total-$\vert \Delta\vert $ history; oscillatory/first-order-like classes trigger mixing-rate adjustments (and, for a linear/symmetry mismatch between the converged state and the FS eigenvector, a one-shot d-wave-forced retry that adopts the lower-free-energy result). A saddle-escape kick (step 10 below) can fire mid-run whenever the total gap is still near zero and the post-convergence-style Hessian at the current point has a negative eigenvalue. Post-convergence the solver runs the FS-resolved $\partial\lambda/\partial Q$ and channel-decomposition diagnostics, the coherence-length/gap-symmetry diagnostics, the incommensurate-nesting check (§22), the Mott filter (§5), and assembles the full result dictionary consumed by the diagnostics block described in [Output & Diagnostics](#output--diagnostics).

10. **Saddle-escape kick.** When the total gap is still below $5\times$`tol` (iteration $>3$, every 8th iteration), `compute_hessian` is evaluated at the current point; a negative smallest eigenvalue triggers a kick along its eigenvector, classified into `pure-SC`/`pure-JT`/`SC-triggered-JT`/`AFM-fluctuation`/`mixed` by the fractional weight on each block (§16), with the kick magnitude Λ-damped and the Anderson history reset. In the `pure-SC`/`SC-triggered-JT` modes, if $M_{\Gamma_6}$ has overshot a cheap Stoner-only estimate (`refine_M_state`), the kick instead gently pulls all three $M$ channels back toward it rather than adding to the eigenvector step directly.

### Vectorized BdG, Buffer Reuse, and the χ₀(q) Permutation Trick

`VectorizedBdG._build_H_stack` assembles the entire $(N_k,24,24)$ Hamiltonian stack with vectorized NumPy operations and diagonalizes it in a single `np.linalg.eigh` call per SCF iteration, reusing a pre-allocated `out=` buffer to avoid repeated allocation across hundreds of iterations; Hermiticity is enforced after assembly. The per-iteration eigensystem `(ev, ec)` is computed once and shared by observable computation, both cross-doublet branches' gap equations, and the analytic $\partial F/\partial M$ (below).

The $q$-loop inside the RPA vertex construction never re-diagonalizes: the uniform k-grid (`endpoint=False`) is built in `ModelParams.__post_init__` so that for any $q=(n_x,n_y)\cdot2\pi/\mathrm{\_NK}$, the $k+q$ grid is exactly a cyclic permutation of the $k$-grid. A precomputed `shift_table[nx, ny]` (shape `(_NK, _NK, N_k)`, `int32`) turns "shift by $q$" into a free index reorder,

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]     # index reorder — no extra LAPACK call
```

reusing the *same* $\Delta=0$ eigensystem for every $q$-point in one vertex-cache rebuild. `_get_chi0_norm_cache` additionally memoizes this normal-state $(E_k,V_k)$ across separate calls (susceptibilities, rigidity, incommensurate-nesting scan) that fall within the same iteration, keyed on $(M,Q,\mu,g_t,g_J,\delta)$ with independent tolerances tightened around the physically sensitive ones — the $M$ and $Q$ tolerances are the *same* `_M_THR_REL`/`_Q_THR_REL` thresholds used for RPA vertex-cache invalidation below, while $\mu$, $g_t$, $g_J$, and the doping are checked at $10^{-4}$, $10^{-4}$, $10^{-4}$, and $10^{-6}$ respectively.

### Vertex Cache Invalidation

The RPA vertex cache is rebuilt when $M$ moves by more than an adaptive threshold scaled to `_M_THR_REL = 0.01` (finer near the QCP, where the vertex is most sensitive), when $Q$ moves by more than `_Q_THR_REL = 0.02` (2%) of `lambda_hop`, or unconditionally if the cached determinant sign disagrees with the freshly computed SC-state determinant sign (a proxy for having crossed the QCP since the cache was built). There is no Δ-based invalidation — the vertex is *always* built from $\Delta=0$ by construction (§14). The cache stores the RPA determinant (both `det_q0` and `det_afm`), FS geometry (`fs_pts`, `vF_arr`, and the FS-point index array), the per-channel vertex components of §15 (`V_channels`, `V_ij`, `V_weighted`), and the 2×2 pairing-kernel results, so repeated calls within one iteration reuse the same Fermi-surface sampling.

### Limit-Cycle Detection

Independent of the adaptive-$\alpha$ mechanism below, a dedicated oscillation check monitors the total $\vert \Delta\vert $ (summed over all four channels) over a rolling window of `_CYCLE_WINDOW = 20` iterations; when the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.25`, the mixing rate is cut by `_CYCLE_DAMP_FAC = 0.45` and the Anderson history reset, which specifically targets the strongly nonlinear regime near the JT-activation onset where the $(Q,\Delta)$ feedback is most prone to overshoot.

### Initial-Condition Seeding (`_scf_kick`)

Before the main SCF loop, `_scf_kick` first calls `refine_M_state` to get a normal-state ($\Delta=Q=0$) AFM seed, builds the cheap linearized pairing eigenvalue `lambda_lin_max` from `compute_pairing_kernel_and_build_cache` at that seed, and evaluates a full finite-difference Hessian (`compute_hessian`, §16) at a small probe distortion `Q_kick = _KICK_Q_SEED`. If that Hessian's smallest eigenvalue is negative, the seed for $(M,Q,\Delta)$ is stepped along its eigenvector (`_project_kick_from_hessian`, scaled by the linear-eigenvalue excess above 1); otherwise a more conservative Stoner-proximity heuristic shrinks $M$ toward the RPA-safe region and seeds a small BCS-like $\Delta$ directly. Either way this lands the iteration in the basin of the physically correct fixed point rather than an arbitrary starting guess, and sets the starting mixing rate $\alpha=\max($`_KICK_MIXING_FLOOR`$,\ $`_MIXING`$/(1+$`_KICK_MIXING_SCALE`$\cdot\log1p(\lambda_{\mathrm{lin,max}})))$. The Anderson solve itself (used throughout the main loop, not in the seed step) uses a Tikhonov-regularized (`_ANDERSON_TIKHONOV = 1e-8`) normal-equation solve with a trust-region cap (`_ANDERSON_TRUST = 2.4`×) on the step size relative to simple mixing.

### Adaptive Q Update

$Q_{\mathrm{out}}^{\mathrm{raw}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ is evaluated at **every** iteration via a damped Newton step on the Hellmann–Feynman force (`_compute_Q_newton_step`), since $\langle B_{1g}\rangle$ is already available at zero extra cost from the same observable pass. It is only **injected into the Anderson vector**, however, when at least one of two conditions holds: the implied displacement $\vert Q_{\mathrm{out}}^{\mathrm{raw}}-Q\vert $ exceeds `_Q_THR_REL·lambda_hop`; or a periodic safety heartbeat fires (`iteration % _Q_UPDATE_PERIOD == 0`, every 4 iterations). Otherwise $Q_{\mathrm{out}}=Q$ exactly — the Anderson residual for $Q$ is zero and the mixer leaves it untouched, respecting the lattice's slower (adiabatic) timescale relative to the electronic degrees of freedom without imposing a rigid blind period.

### Thread-Safety and Clone Protocol

The current `__main__` runs its three parallel SCF tasks (§ Installation & Usage) as three independently constructed `RMFT_Solver` instances rather than clones of a single solver, but the underlying clone protocol remains available for any code that mutates parameters mid-run (and is used internally by the Tc routines' temperature scans, via `_clone_solver_at_T`):

```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s._full_rebuild()
```

`_full_rebuild()` is the single canonical post-mutation refresh: it re-runs `p.__post_init__()` (SOC+CF diagonalization and the superexchange cluster ED), updates the bare stiffness `_K_bare` and the derived bare phonon energy `_omega_0_JT` (§14), rebuilds every orbital operator (`B1g_op`, `B1g_24`, `Eg2_op`, `Eg2_24`, `sz_op`, `multi_op`), and resets all transient caches (`_reset_transient_state`). Each clone owns its own `VectorizedBdG` and its own `_H_stack` buffer, so concurrent workers never alias each other's memory. At import time the module pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to prevent the BLAS backend from oversubscribing threads underneath an outer `ThreadPoolExecutor`.

### Analytic ∂F/∂M and ∂²F/∂M² from a Single Diagonalization

The Newton step for $M$ (SCF Loop step 6) needs both the free-energy gradient and curvature with respect to the three channel-resolved $M$ components, but these are obtained **without any extra diagonalization**: `compute_dF_dM_channels_and_hessian` builds $\partial H/\partial M_c$ analytically (only the diagonal $J_{A1g}$ Weiss-field term contributes; the off-diagonal transverse-Weiss term has no direct diagonal piece with respect to $M$, though the anomalous amplitudes it carries must still be present in $H$ so the eigenvectors reflect the correct inter-band structure) and then applies first- and second-order perturbation theory directly to the **already-computed** eigenvalues/eigenvectors $(ev,ec)$ of the converged BdG stack:

```
∂F/∂M_c   =  ⟨n|∂H/∂M_c|n⟩ weighted by f(E_n)                                    (Hellmann–Feynman, diagonal)
∂²F/∂M_c∂M_c' =  −Σ_{n} f'(E_n)·⟨n|∂H/∂M_c|n⟩⟨n|∂H/∂M_c'|n⟩  +  Σ_{n≠m} [f(E_n)−f(E_m)]/(E_m−E_n) · ⟨n|∂H/∂M_c|m⟩⟨m|∂H/∂M_c'|n⟩
```

the second (off-diagonal, Kubo-like) term using a numerically safe $\tfrac{\Delta f}{\Delta E}\to -f'(E)$ limit at near-degenerate $E_n\approx E_m$. Because both derivatives come from the one BdG eigensystem already sitting in memory, this replaces what would otherwise be several additional full diagonalizations per SCF iteration with O(1) extra tensor contractions.

---

## Parameters

All energies in **eV**, lengths in **Å**. Defaults below are the values set in the current `__main__` block, verified directly against the source.

### Primary Inputs (`ModelParams`)

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | $t_{pd}$ | 0.405 eV | $p$–$d$ hybridization integral; the single primary hopping input ($t_0=t_{pd}^2/\Delta_{CT}$ is derived) |
| `t_prime_ratio` | — | −0.18 | 2nd-neighbor (diagonal) hopping ratio; $t'=$ this $\times\,t_0$ |
| `t_dprime_ratio` | — | 0.10 | 3rd-neighbor (axial) hopping ratio; $t''=$ this $\times\,t_0$ |
| `U_dd` | $U_{dd}$ | 2.720 eV | On-site Hubbard repulsion — a **primary** input, not derived from a dimensionless ratio |
| `lambda_soc` | $\lambda_{SOC}$ | 0.036 eV | Atomic SOC constant on the $t_{2g}$ shell; sets the Γ₆–Γ₇ splitting together with `Delta_tetra` |
| `Delta_tetra` | $\Delta_{\mathrm{tetra}}$ | −0.040 eV | Axial (tetragonal) crystal field, $\Delta_{\mathrm{tetra}}\cdot L_z^2$; negative = $z$-axis compression |
| `g_JT` | $g_{JT}$ | 0.312 eV/Å | B₁g electron–phonon (JT) coupling |
| `K_lattice` | $K$ | 2.890 eV/Å² | Bare B₁g phonon spring constant; `K_eff` is computed at runtime |
| `lambda_hop` | $\lambda_{\mathrm{hop}}$ | 1.100 Å | Hopping-anisotropy decay length: $t(Q)=t_0\exp(\pm Q/\lambda_{\mathrm{hop}})$ |
| `g_Eg2` | $g_{Eg2}$ | 0.100 eV/Å | Eg,2-channel electron–phonon coupling (§7) |
| `K_lattice_Eg2` | $K_{Eg2}$ | 6.500 eV/Å² | Bare Eg,2 phonon spring constant |
| `Delta_CT` | $\Delta_{CT}$ | 2.430 eV | Charge-transfer gap |
| `Delta_B1g_static` | $\Delta_{\mathrm{ip}}$ | −0.005 eV | Static in-plane crystal field, $(L_x^2-L_y^2)$; drives the D₄h→D₂h crossover (§1, §3) |
| `hybrid_scale` | — | 6.000 | Downfolding coordination factor entering the ZRS spectral weight $\beta^2(k)$ (§6, §9) |
| `Upp_ratio_bare` | — | 0.400 | Bare (unhybridized) $U_{pp}/U_{dd}$ ratio entering the internally-computed $U_{pp}$ (§4) |
| `J_H_ratio` | — | 0.08 | Kanamori Hund's coupling for the virtual $d^2$ superexchange intermediate state, as a fraction of $U_{dd}$: $J_H = $ this $\times\,U_{dd}$ (§4) |
| `M_eff` | — | 16.0 amu | Normal-mode reduced mass for the B₁g phonon; sets the bare phonon energy $\hbar\omega_0$ used by the adiabatic retardation factor (§14). For this octahedral B₁g-type distortion the metal ion sits close to a symmetry-imposed node, so the mode is usually ligand-dominated |
| `Z` | $Z$ | 4 | Coordination number |
| `kT` | $k_BT$ | 0.005 eV | Temperature ($\approx$ 58 K) |
| `tol` | — | $10^{-4}$ | SCF convergence threshold |

### Derived Quantities (from `__post_init__`)

| Quantity | Origin | Description |
|---|---|---|
| `Delta_CF`, `g7split` | SOC+CF diagonalization | Γ₆–Γ₇ₐ gap and Γ₇ₐ–Γ₇ᵦ internal splitting — **not** free parameters |
| `sz_op` | exact $S_z$ diagonalization in each Kramers doublet | AFM/spin-vertex weights, 6 components $[sz_{6\uparrow},sz_{6\downarrow},sz_{7a\uparrow},sz_{7a\downarrow},sz_{7b\uparrow},sz_{7b\downarrow}]$ |
| `multi_op` | built from `sz_op` | Effective multipolar spin operator shared by the cluster and BdG solvers |
| `p_7` | Γ₇ admixture in the Γ₆ eigenvectors | Interpolates the *static* estimate `g_Delta_ad` in `get_gutzwiller_factors`; the SCF loop instead uses the dynamic, occupation-based estimate of §5 |
| `_w_orb` (3×3) | eigenvector projections | $d_{xz}/d_{yz}/d_{xy}$ orbital weights (rows Γ6/Γ7a/Γ7b) feeding the exchange anisotropy and the `Tx_A_*` hopping projectors |
| `kappa_doublets` (3), `kappa_cross` (2) | multi-orbital cluster ED, §4 | Intra-doublet (Γ6, Γ7a, Γ7b) and cross-doublet (Γ6–Γ7a, Γ6–Γ7b) exact-diagonalization exchange coefficients — replace the earlier single scalar `J_pdct`, which no longer exists |
| `Tx_A_xz/_yz/_xy` | orbital-character projectors | Rigorous orbital-selective hopping building blocks (§6); sum to $I_6$ |
| `t0`, `t_prime`, `t_dprime` | $t_{pd}^2/\Delta_{CT}$ and the `*_ratio` inputs | Effective nearest/2nd/3rd-neighbor $dd$ hopping |
| `thermal_fs_nat_size` | $\max(\lfloor kT/t_0\cdot$`_NK`$^2\rfloor,\,32)$ | Adaptive Fermi-surface sample-point budget (§15/§ Key Algorithms), replacing an earlier fixed sample count |
| `k_points`, `k_weights`, `N_k`, `shift_table` | $N_k=$ `_NK`² uniform grid | k-space infrastructure, including the cyclic shift table used for arbitrary-$q$ Lindhard sums |

`U_pp` (ligand hole–hole repulsion) and `J_H` (the absolute Hund's coupling) are computed from `Upp_ratio_bare`/`J_H_ratio` and consumed immediately inside `__post_init__` to build the superexchange cluster Hamiltonian (§4); neither is stored as an accessible `ModelParams` attribute afterward. `B1g_op`, `B1g_offdiag`, `Eg2_op` are also set on `ModelParams` in `__post_init__` (§3, §7); the corresponding 24×24 Nambu lifts `B1g_24`, `Eg2_24`, plus `Sz_nambu` and the per-channel `Sz_stag_nambu_channels`/`_sz_nambu_diag_channels`, are built on the `RMFT_Solver` instance by `_rebuild_orbital_operators`, along with `phi_k = cos(kx) − cos(ky)` on the SCF grid and the bare phonon energy `_omega_0_JT` (§14).

### Module-Level Constants

The source file documents essentially every numerical-methods constant inline, each with its own physical or numerical justification — that block remains the single source of truth. The tables below reproduce every value relevant to interpreting solver behavior and output, transcribed directly from the current module-level constant block (not from any earlier documentation pass).

**Grid, iteration budget, general safety**

| Constant | Value | Role |
|---|---|---|
| `_NK` | 64 | k-points per direction (even, for commensurate $q_{AFM}=(\pi,\pi)$) |
| `_MAX_ITER` / `_MIN_ITER` | 500 / 4 | SCF iteration ceiling / floor before a convergence check is even attempted |
| `_MAX_ITER_FOR_REFINE_M` | 120 | Fixed iteration count for the `refine_M_state` Anderson loop (no early-exit check) |
| `_MIXING` | 0.05 | Base Anderson mixing weight |
| `_N_ORB` / `_N_BDG` | 6 / 24 | Orbital flavors (full Γ₆⊕Γ₇ₐ⊕Γ₇ᵦ manifold) / BdG Nambu dimension ($4\times$`_N_ORB`) |
| `_N_CHANNELS` / `_CLUSTER_SIZE` / `_N_CLUSTER` | 3 / 4 / $6^4{=}1296$ | Channel resolution (Γ₆,Γ₇ₐ,Γ₇ᵦ) / sites in the plaquette cluster ED / cluster Hilbert-space dimension |
| `_MATH_EPS` | $10^{-9}$ | General division-by-zero guard |
| `_LINDHARD_CHUNK` | 128 | k-point batch size in the `opt_einsum` Lindhard loops |
| `_BZ_NORM` | $(2\pi)^2$ | BZ-area normalization in the FS integration measure |
| `_Q_UNIQUE_SCALE` / `_PI_INT` | $10^5$ / 314159 | Integer scaling used to hash unique $q$-pairs without floating-point collisions |

**Unit conversion & Gutzwiller prefactors**

| Constant | Value | Role |
|---|---|---|
| `_EV_TO_K` | 11604.518 K/eV | $1/k_B$ |
| `_GW_G_J_PREFACTOR` | 4.0 | Numerator in $g_J=4/(1+\delta)^2$ (slave-boson / Kotliar–Ruckenstein, half-filling limit) |
| `_GW_G_T_NUMERATOR` | 2.0 | Numerator in $g_t=2\delta/(1+\delta)$ |
| `_JT_PHONON_MEV_CONST` | 64.6528 | $\hbar\omega_0[\mathrm{meV}]=$ this $\times\sqrt{K[\mathrm{eV/\mathring A^2}]/M[\mathrm{amu}]}$ — exact unit-conversion constant, not a fitted number (§14) |

**AFM Newton solver ($M$-step control)**

| Constant | Value | Role |
|---|---|---|
| `_MU_LM` | 3.0 | Levenberg–Marquardt floor for the $M$ Newton step |
| `_ALPHA_HF` | 0.28 | Newton-vs-BdG-fixpoint blend weight for $M$ |
| `_TR_M_STEP_MAX` / `_TR_M_STEP_MIN_FLOOR` | 0.1 / $10^{-3}$ | Trust-region cap / absolute floor on $\vert \Delta M\vert $ per step |
| `_M_STEP_FLOOR_REL` / `_M_STEP_FLOOR_ABS` / `_M_STEP_FLOOR_M_MIN` | 0.005 / 0.002 / 0.010 | Step floor $=\max($ `_M_STEP_FLOOR_REL` $\cdot\vert M\vert ,$ `_M_STEP_FLOOR_ABS`$)$, referenced against $\max(\vert M\vert ,$ `_M_STEP_FLOOR_M_MIN`$)$ |
| `_M_J_EFF_FLOOR_FRAC` | 0.20 | QCP guard: $J_{\mathrm{eff}}$ floored at this fraction of $t_{\mathrm{eff}}$ to prevent $\Delta M\propto1/J_{\mathrm{eff}}\to\infty$ |

**Q Newton solver step control**

| Constant | Value | Role |
|---|---|---|
| `_Q_LM_FRAC` | 0.08 | Q-channel LM floor, as a fraction of the bare stiffness `_K_bare` |
| `_TR_Q_STEP_FRAC` | 0.10 | Trust-region cap on $\vert Q_{\mathrm{out,raw}}-Q\vert $, as a fraction of `lambda_hop` |
| `_TR_Q_STEP_MIN_FLOOR` | $10^{-4}$ Å | Absolute minimum $Q$ step, preventing total freeze near the JT QCP |

**Δ Newton solver step control**

| Constant | Value | Role |
|---|---|---|
| `_MU_LM_DELTA` | 3.0 | Levenberg–Marquardt floor for the 4-channel Δ Newton step |
| `_ALPHA_HF_DELTA` | 0.20 | Newton-vs-BdG-fixpoint blend for the Δ update |
| `_TR_DELTA_STEP_MAX` | 0.1 eV | Upper bound on the Δ Newton step per channel per iteration |

**Saddle-escape / initial-condition seeding**

| Constant | Value | Role |
|---|---|---|
| `_MODE_FRAC_DOMINANT` / `_MODE_FRAC_MIXED` | 0.60 / 0.30 | Thresholds classifying a pure-channel vs. mixed SC-triggered-JT mode (§16) |
| `_MODE_PULL_FRAC` | 0.30 | Fraction of $(M-M_{\mathrm{phys,est}})$ used as the kick pull in pure-SC/SC-JT mode |
| `_KICK_BASE_FRACTION` | 0.05 | Base trust-region step-size fraction for the mid-run saddle-escape kick |
| `_KICK_M_EXCESS_CTR` / `_KICK_JCHI_EXCESS_CTR` | 0.70 / 0.70 | Sigmoid centers for $M$-kick / $J\chi_{SS}$-excess overshoot suppression in the seed kick |
| `_KICK_REDUCTION_AMP` | 3.88 | $M_{\mathrm{kick}}\times(1-\text{this}\times\text{excess})$ |
| `_KICK_BOOST_Q` | 0.01 | $Q$-kick boost (seed path) |
| `_KICK_M_CLIP_LO` / `_KICK_M_CLIP_HI` | 0.02 / 0.9 | Hard clips on $M_{\mathrm{kick}}$ |
| `_KICK_DELTA_MAX_FRAC` | 0.4 | Maximum seed gap as a fraction of the effective hopping scale $t_{\mathrm{eff}}$ |
| `_KICK_MIXING_FLOOR` / `_KICK_MIXING_SCALE` | 0.004 / 4.0 | Minimum kick mixing weight / damping scale in $\alpha=$`_MIXING`$/(1+\text{scale}\cdot\log1p(\lambda_{+}))$ |
| `_Q_SEED_THR` | $10^{-4}$ | If the initial $Q$ seed is already nonzero, it is trusted as the current best estimate |
| `_KICK_Q_SEED` | $10^{-2}$ Å | Probe distortion used for the seed-time Hessian evaluation |
| `_EARLY_KICK_BASE` | 0.01 | Base step fraction in the coupled seeding space |

There is no longer a separate `estimate_M0`/`_M0_*` empirical-prior warm start: the normal-state AFM seed comes from a plain Anderson-accelerated fixed point (`refine_M_state`, §5/§ Key Algorithms).

**Chemical potential (Newton + Brent)**

| Constant | Value | Role |
|---|---|---|
| `_DEN_DERIV_FLOOR` | $10^{-12}$ | Floor on $\partial n/\partial\mu$ |
| `_BRENTQ_TOL` | $10^{-6}$ | Brent bracketing tolerance |
| `_MU_NEWTON_MAXIT` / `_MU_BACKTRACK_MAX` / `_MU_BACKTRACK_FLOOR` | 20 / 6 / 0.05 | Newton iteration budget / max step-halvings / minimum backtrack damping before falling back to Brent |
| `_MU_DENSITY_TOL` | $10^{-8}$ | $\vert n(\mu)-n_{\mathrm{target}}\vert $ convergence tolerance |
| `_MU_SC_DERIV_THRESH` | $10^{-4}$ eV | Gap amplitude above which the analytic $\partial n/\partial\mu$ (exact only at $\Delta=0$) is replaced by a centered numeric derivative |

**Lindhard broadening & Fermi-surface sampling**

| Constant | Value | Role |
|---|---|---|
| `_ETA_T_FRAC` | 0.10 | Normal-state broadening $\eta=$ this $\times\,kT$ |
| `_ETA_DELTA_FRAC` | 0.02 | SC-state broadening increment $\propto\vert \Delta\vert $ |
| `_ETA_GRID_FLOOR` | 0.001 | Broadening floor (units of $t_0$), guards k-grid aliasing |
| `_FERMI_ARG_CLIP` | 100.0 | Numerical clip in $f(E)$ |
| `_FD_MASK_DF` / `_FD_MASK_DE` / `_FD_MASK_DE8` | $10^{-12}$ / $10^{-6}$ / $10^{-8}$ | Degenerate-denominator masks in the $\chi_0$ Lehmann sums (the tightest, `_FD_MASK_DE8`, is used in the $\partial^2F/\partial M^2$ off-diagonal term) |
| `_FS_SAMPLING` / `_FS_THERMAL_THRESHOLD` | 4.4 / 0.0025 | Thermal window (in units of $kT$) around $E_F$ for FS selection / minimum relative thermal weight kept |
| `_FS_CACHE_TOL` | $10^{-3}$ | Parameter-change tolerance for FS-point cache invalidation |
| `_NODAL_REGION_PCTL` | 25 | Percentile split (upper/lower 25%) for nodal/antinodal FS decomposition |
| `_PHI_D_FLOOR` | $10^{-3}$ | Minimum $\varphi_d^{\max}$ to enable the nodal/antinodal split at all |
| `_VERTEX_DIAG_MIN_FS` | 10 | Minimum FS points required before vertex-structure diagnostics are considered reliable |

The Fermi-surface sample budget itself is no longer a fixed module constant: it is the dynamically computed `thermal_fs_nat_size` (see Derived Quantities above), and the FS integration weight is a symmetry-adapted (D₄h/D₂h) thermal 2D element with no separate Fermi-velocity floor constant (§15).

**RPA vertex & QCP tracking**

| Constant | Value | Role |
|---|---|---|
| `_RPA_BW_FACTOR` | 8.0 | Tight-binding bandwidth estimate $=8t$ |
| `_RPA_V_CAP_ALPHA` | 2.2 | Headroom multiplier for the dynamic vertex cap $V_{\mathrm{cap}}$ |
| `_RPA_DET_WARN` | 0.11 | $\vert \det_{\mathrm{afm}}\vert $ below this ⇒ QCP-proximity warning, feeds adaptive mixing |
| `_RPA_QCP_PENALTY` | 0.40 | Mixing-rate reduction per unit $\vert \det_{\mathrm{afm}}\vert <0$ past the QCP |
| `_DET_AFM_FLOOR` | 0.5 | Default `det_afm` when no vertex cache exists yet |
| `_DK_CORR_CAP_MULT` | 1.0 | Cap on both the mechanical cluster stiffness correction (§11) and the adiabatic Δ-relaxation correction (§11), relative to the bare stiffness $K_{\mathrm{bare}}$ |
| `_DET_DEPTH_CAP` / `_DET_JUMP_HALF_SCALE` / `_JUMP_CAP_FLOOR` | 5.0 / 0.5 / 1.05 | Past-QCP gap-jump cap: exponential suppression depth cap / decay rate / minimum allowed cap |
| `_DET_SIGN_FLIP_SCALE` | 0.05 | $\vert \det_{\mathrm{afm}}\vert $ sigmoid midpoint for the $V_d$ sign-flip EMA guard |
| `_EMA_SIGN_FLIP_W_MIN` / `_EMA_SIGN_FLIP_SLOPE` | 0.20 / 6.0 | Minimum blend weight / sigmoid steepness in the sign-flip guard |
| `_V_PREV_SIGN_FLOOR` | $10^{-6}$ | $\vert V_{d,\mathrm{prev}}\vert $ below this is treated as zero (sign-flip check skipped) |
| `_V_CUT` | 20.0 | Pairing-vertex near-divergence detector threshold |
| `_JCHI_HARD_REJECT` | 2.0 | $J\chi_{SS}$ above this ⇒ hard-rejected (deeply AFM, SC impossible) |
| `_QQ_DELTA_THRESH` | $10^{-8}$ | $\vert \Delta\vert $ threshold below which the seed-magnitude branch of the gap update is used instead of the fixed-point one, per branch |

**Moriya-SCR damping (spin and lattice channels), JT viability, finite-difference steps**

| Constant | Value | Role |
|---|---|---|
| `_MORIYA_LANDAU_M_STEP` | 0.06 | $M$-probe step for the self-consistent, Landau-expansion-derived $\Gamma_M$ (§14) |
| `_CHI_TAU_ABS_FLOOR` | $10^{-5}$ eV$^{-1}$ | Absolute floor under the relative-error denominator in the $\chi_{\tau}$ Richardson/nonlinearity tests (§12), preventing a genuinely small signal from having its relative error spuriously amplified |
| `_LAMBDA_JT_VIABLE` | 0.05 | Minimum $\lambda_{JT,\mathrm{sc}}$ for SC-triggered-JT viability (§17) |
| `_JT_ACT_THR` | 0.04 | Threshold on the condensate-induced selection ratio for the "JT-active" classification |
| `_DQ_FS_VERTEX` / `_DQ_FS_VERTEX_FRAC` | 0.03 Å / 0.05 | Minimum / adaptive-fraction finite-difference step for $\partial\lambda/\partial Q$ on the FS |
| `_JT_FD_H2_BASE` / `_JT_FD_H2_QCOEF` | $3\times10^{-8}$ / $6\times10^{-7}$ | $Q$-derivative FD step schedule for the cluster mechanical stiffness, $h(Q)=\sqrt{\text{base}+\text{qcoef}\cdot Q^2}$ |

**Limit-cycle detection & Anderson mixing**

| Constant | Value | Role |
|---|---|---|
| `_CYCLE_WINDOW` / `_CYCLE_THRESHOLD` / `_CYCLE_DAMP_FAC` | 20 / 0.25 / 0.45 | Rolling window / relative-std trigger / mixing-rate cut on a detected limit cycle |
| `_ANDERSON_TIKHONOV` | $10^{-8}$ | Tikhonov regularization in the Anderson normal equations |
| `_ANDERSON_TRUST` | 2.4 | Trust-region cap on the Anderson step (multiples of the simple-mixing step) |
| `_ANDERSON_W_LO` / `_ANDERSON_W_HI` | 0.3 / 0.8 | Blend-weight bounds between Anderson and simple mixing |

**SCF regime classification (freeze / recover / diverge / stagnate)**

| Constant | Value | Role |
|---|---|---|
| `_SCF_DIVERGE_RATIO` / `_SCF_STAGNATE_RATIO` | 1.05 / 0.95 | $\max\vert \Delta\vert $ growth ratio thresholds classifying the step as diverging / stagnating |
| `_SCF_ALPHA_DECAY` / `_SCF_ALPHA_RECOVER` | 0.95 / 1.60 | Mixing-rate multiplier while converging (mild damping) / on freeze-recovery |
| `_SCF_FREEZE_THR` | 10 | Consecutive frozen iterations that trigger freeze-recovery |
| `_SCF_ALPHA_FREEZE_LO` / `_SCF_ALPHA_FREEZE_HI` | 0.15 / 0.60 | $\alpha/$`_MIXING` bounds defining "too frozen" / the recovery ceiling |
| `_SCF_ALPHA_CONVG_BOOST` / `_SCF_ALPHA_CONVG_CAP` | 1.09 / 0.75 | Mixing-rate boost / ceiling while SC+JT active and converging |
| `_EMA_NEW_WEIGHT` | 0.12 | EMA weight for $\Lambda_{\mathrm{inst}}$ and the $V_d$ sign-flip guard |
| `_Q_UPDATE_PERIOD` | 4 | Heartbeat period (iterations) for the Hellmann–Feynman $Q$ update |
| `_Q_THR_REL` | 0.02 | Fraction of `lambda_hop`; $Q$ change below this skips the Anderson injection |
| `_M_THR_REL` | 0.01 | Absolute $M$-change threshold for vertex-cache invalidation |
| `_ALPHA_MIX_2X2` | 0.56 | Blend weight: 2×2 pairing-kernel eigenvector vs. fixed-point gap update, both branches (§15, §21) |

**Gap (Δ) update and coherence-length classification**

| Constant | Value | Role |
|---|---|---|
| `_BCS_SEED_FRACTION` | 0.1 | Initial cold-start $\Delta$ seed, as a fraction of $t_{\mathrm{eff}}$ |
| `_DELTA_JUMP_CAP` | 5.0 | Maximum $\vert \Delta_{\mathrm{new}}\vert /\vert \Delta_{\mathrm{old}}\vert $ ratio per iteration (per branch) |
| `_DELTA_ABS_FLOOR` | $10^{-4}$ eV | $\vert \Delta\vert $ below this bypasses the jump limiter (free seed-growth phase) |
| `_KERNEL_DIR_MIN_FRAC` | 0.5 | 2×2-kernel eigenvector allowed to dominate the mixing below this fraction of fixed-point amplitude |
| `_XI_NODAL_MIN` | 2.0 | Minimum $\xi/a$ (nodal) for BCS-side quasiparticle coherence |
| `_ORBITAL_SEL_FRAC` | 0.15 | $\vert \xi_{\Gamma_6}-\xi_{\Gamma_7}\vert /\xi$ threshold for "orbitally selective" pairing |
| `_IC_RATIO_FLOOR` / `_IC_RATIO_CAP` | 1.05 / 3.00 | Bounds used to clamp the incommensurate-nesting $\chi$-ratio diagnostic |
| `_MBZ_DEGEN_FRAC` | $2\times10^{-3}$ | Energy tie-break scale (fraction of $kT$) for magnetic-BZ Fermi-surface point deduplication |

**Superexchange cluster ED validity guards**

| Constant | Value | Role |
|---|---|---|
| `_WIGNER_ECKART_DIV_FLOOR` | $10^{-6}$ | Floor under the effective-spin-squared / $\Vert M\Vert_F^4$ denominators in the κ extraction (§4) |
| `_ANISO_WARN_TRESH` | $5\times10^{-4}$ | Non-Heisenberg component magnitude above which `_intra_doublet_kappa` logs a warning |
| `_B1G_BLOCK_MIN_FRAC` | 0.05 | Minimum share of $\Vert B_{1g,\mathrm{op}}\Vert_F^2$ a Γ6↔Γ7 block must carry for `_cross_doublet_kappas` to resolve it |
| `_SYM_THRESH` | $10^{-8}$ | General symmetrization/Hermiticity tolerance |

**Tc / Ginzburg–Landau / BCS ratio**

| Constant | Value | Role |
|---|---|---|
| `_BCS_RATIO_STRONG` / `_VSTRONG` / `_EXOTIC` | 3.8 / 5.0 / 7.0 | $2\Delta_0/k_BT_c$ thresholds for strong / very-strong / exotic coupling |
| `_MAD_DENOM` / `_MAD_NUM` | 1.13 / 1.04 | Allen–Dynes-type strong-coupling denominator (Millis–Monien–Pines spin-fluctuation value) / exponent prefactor |
| `_GL_DELTA_MIN` | 2 meV | $\vert \Delta\vert $ floor for points admitted to the GL fit |
| `_GL_MIN_PTS` / `_GL_MAX_PTS` | 2 / 4 | Minimum / maximum recent stable-SC points used in the GL regression |
| `_GL_TC_MARGIN` | 0.05 | Maximum relative deviation $\vert T_{c,GL}-T_{\mathrm{spinodal}}\vert /T_{\max}$ to accept the GL result |
| `_GL_SPINODAL_JUMP` | 0.15 | $D_{\mathrm{spinodal}}/\Delta_0$ below this ⇒ GL extrapolation treated as reliable (small first-order jump) |

**Physical thresholds (SC viability / Mott)**

| Constant | Value | Role |
|---|---|---|
| `_G_T_COHERENCE_MIN` | 0.10 | Mott guard: minimum coherent $g_t$ ($\delta\gtrsim0.053$), used in `_scf_kick`, the Mott filter, and the `__main__` doping floor |

---

## Installation & Usage

### Requirements

```bash
pip install numpy scipy matplotlib opt_einsum
```

### Running

```bash
python Quantum_AFM-multipolar_Jahn-Teller.py
```

On startup, the current `__main__` block does the following, in order:

1. **Parameter setup and SOC+CF diagonalization.** `ModelParams(...)` is constructed with the defaults listed in [Parameters](#parameters) and `__post_init__` runs the SOC+CF diagonalization (Γ₆/Γ₇ₐ/Γ₇ᵦ identification, `Delta_CF`, `sz_op`, `p_7`, k-grids, orbital operators) and the multi-orbital superexchange cluster ED (`kappa_doublets`, `kappa_cross`, §4).
2. **Doping setup.** `target_doping = 0.139`, with a symmetric ±20% scan margin (`doping_margin = 0.20`) defining `min_doping`/`max_doping`, floored to stay above the `_G_T_COHERENCE_MIN` Mott-incoherence boundary. These bounds are computed but, in the current `__main__`, not otherwise consumed — all three solves below run at the single point `target_doping`, not a scan. `initial_Delta = 8×10⁻³` eV is the cold-start gap seed.
3. **Three-way self-consistent comparison, run in parallel.** Three independent `RMFT_Solver` instances are built from deep copies of `params`, and `solve_self_consistent` is run on each concurrently via a 3-worker `ThreadPoolExecutor`:
   - **`ref`** — the full self-consistent solve, SC (all four channels) and $Q$ both free (the model's actual prediction).
   - **`normal`** — $\Delta$ pinned to zero throughout (`force_delta_zero=True`); the normal (non-superconducting) AFM state.
   - **`SC_Q0`** — SC free but $Q$ pinned to zero (`force_Q_zero=True`); superconductivity without the JT relaxation.

   This is the direct numerical test of the central hypothesis: if the theory is right, `F_bdg` should order `ref < normal` (condensation lowers the free energy) and `ref < SC_Q0` (JT relaxation lowers it further), and `Q` should self-consistently relax back toward 0 in `normal` *without being forced there* — the code checks this explicitly and logs `"SC+JT is the ground state ✓"` or `"✗"` accordingly (guarded by the `_scf_result_reliability` check on all three results).
4. **Normal-state JT-stability check.** At the self-consistent $M$ from `ref`, $K_{\mathrm{spont}}=g_{JT}^2\chi_{QQ}(\Delta{=}0)$ is evaluated and compared directly against `K_lattice` (§16, §17) — a direct threshold comparison rather than a dedicated instability-matrix data structure.
5. **Post-SCF diagnostics** (only if the reference SCF succeeded): the channel-resolved RPA vertex decomposition, the linearized-gap-equation channel decomposition, coherence lengths, the post-convergence Hessian's SC-triggered-JT mode classification (§16), the Stoner ratio, the Moriya-SCR-regularized $\chi_{\tau}$/`lambda_JT_sc` check (§12), the SC-JT viability verdict (§17), and the three Tc estimates (§23).
6. **One diagnostic plot.** `plot_ground_state_comparison(results)` produces the `ref`/`normal`/`SC_Q0` comparison figure, saved to disk (see [Output & Diagnostics](#output--diagnostics)). There is no longer a separate full-BZ $\chi_{SQ}(q)$ scan plot in this version of the script.

---

## Output & Diagnostics

All output is a structured, thread-safe log stream (`_scf_log`, tagged by stage — `INIT`/`RMFT-INIT`/`SCF-INIT`/`DERIVED`, `REF-SCF`, `PARALLEL`, `SCF`/`SCF-I`/`SCF-II`, `SCF-RES`, `SCF-EG2`, `KICK`, `REFINE-M`, `SADDLE-ESC`, `LIMIT-CYCLE`, `FREE-ENERGY`, `GAP-DIAG`, `PAIR-DIAG`, `DOUBLET-CHK`, `G_JT-BENCH`, `TC-PRELIM`, `TC-LAMBDA`, `TC-THERMO`, `RMFT-WARN`, `PLOT`, …) rather than a GUI; the graphical output is the one `matplotlib` figure described below.

### Iteration Log

Each logged SCF step reports the current order parameters, the effective exchange and mixing rate, and warning flags for numerically marginal vertex structure:

```
[SCF] δ=…  iter/max  conv=…  M=…  Q=…  |Δ|=…  J_eff=… eV  mu=…
      dFM=…  dAFM=…  V_s=…  V_d=…  [⚠same-sign]
      Γ_M=…  α=…  B1g=…  F67s=…  [regime]  …s/it
```

At convergence, an `SCF-RES` block reports the converged order parameters (all four gap channels), density, $\mu$, free energies, `F67s_mf`, the AFM/RPA determinant, the JT-active flag, the SCF-dynamics regime classification (§ Key Algorithms), the s-/d-channel decomposition of $\lambda_{\max}$, `lambda_JT_sc`, `lambda_JT_kernel`, $\partial\lambda_{\mathrm{pair}}/\partial Q$, the post-convergence Hessian's SC-triggered-JT confirmation, coherence lengths, the $\chi_{\tau}$ breakdown (including its reliability weight), the SC-JT window verdict, and the incommensurate-nesting scan result.

### Free-Energy Ground-State Check

Logged under the `FREE-ENERGY` tag right after the three parallel tasks (`ref`, `normal`, `SC_Q0`) complete: `F_bdg` and the post-convergence Hessian minimum eigenvalue for each of the three, flagged `⚠ UNRELIABLE` individually when `_scf_result_reliability` fails, and a final one-line verdict — `"SC+JT is the ground state ✓"`, `"SC+JT is NOT the lowest energy state ✗"`, or `"Comparison NOT trustworthy"` if any of the three results is unreliable.

### RPA Vertex, Coherence, and SC-JT Window Diagnostics

If the reference SCF converged: the FS-averaged, channel-resolved RPA vertex decomposition into spin / JT / cross contributions (§14, §15); the linearized-gap-equation channel decomposition; the coherence-length summary (flagging orbital-selective pairing when $\Gamma_6$ and $\Gamma_7$ channels have meaningfully different $\xi$, and reporting the combined-branch $\xi_{\Gamma_7}$ of §15); the post-convergence Hessian's smallest eigenvalue and mode classification (§16); the Stoner ratio $J_{\mathrm{eff}}\chi_{SS}$ with a QCP-proximity classification; the `K_eff`/`K_eff_reg` path from the normal to the SC state, including the Moriya-SCR lattice-fluctuation correction $\Gamma_Q$ (§12); the $\chi_{\tau}$ breakdown; and the two-threshold SC-JT viability verdict (§17).

### Ground-State Comparison Plot

`plot_ground_state_comparison(results)`, a standalone module-level function (not a solver method), takes the `{"ref", "normal", "SC_Q0"}` result dictionary produced in [Installation & Usage](#installation--usage) step 3 and saves a **2×2** figure to `ground_state_comparison.png`: (top-left) `F_cluster` per SCF iteration for all three scenarios, each with its converged `F_bdg` as a dotted reference line, unreliable trajectories drawn dashed; (top-right) a bar chart of the three final `F_bdg` values, titled with whichever scenario is lowest (unreliable bars hatched); (bottom-left) the Γ₆-channel AFM order parameter $M[\Gamma_6]$ versus iteration, all three overlaid; (bottom-right) $\vert Q\vert $ versus iteration, the key visual check that `normal`'s $Q$ relaxes back toward 0 on its own (it is never forced there) while `ref` settles at a finite value. Missing or unreliable scenario entries are skipped rather than erroring, provided at least one entry is present. (A second, debug-only hexbin plot of the unique-$q$-pair sampling can be produced by `_unique_q_pairs` when called with `verbose=True`, saved as `unique_q_hist.png`; the default pipeline never passes that flag, so this plot is not produced by a normal run.)

---

## Known Limitations

The framework makes a number of physically motivated approximations. The table below reflects the current code, not an earlier design pass.

| Approximation | Impact |
|---|---|
| No Pauli exclusion between plaquette sites | The four-site cluster's external bonds are closed with a mean-field Weiss embedding rather than genuine Pauli-respecting hopping to the rest of the lattice; mild overestimate of AFM correlations |
| No charge-transfer fluctuations $\langle n_An_B\rangle$ | Negligible when the mean-field exchange scale is large compared to the hopping |
| Phonon still adiabatic/quasi-static | $Q$'s own equation of motion is a thermodynamic equilibrium condition, not a dynamical one, and zero-point lattice fluctuations are neglected; the phonon-mediated pairing vertex now carries a genuine adiabatic retardation factor tied to a physical $\hbar\omega_0$ derived from `M_eff` (§14), but this is a one-point static-limit evaluation at the characteristic gap scale, not a full frequency-dependent (Matsubara) treatment |
| $\Gamma_{7a}$ remains the headline channel | Both $\Gamma_6$–$\Gamma_{7a}$ and $\Gamma_6$–$\Gamma_{7b}$ pairing are now fully self-consistent (§8) — this is no longer a diagnostic-only limitation — but several reported summary quantities (`Delta_s`, `Delta_d`, all three Tc estimates) are still built from the $\Gamma_{7a}$ amplitudes specifically; reading off the $\Gamma_{7b}$ channel's own physics requires pulling `Delta_vec[2:]`/`F67_vec[2:]` from the result dictionary directly |
| Resolution mismatch between the RPA and cluster-ED layers | The itinerant RPA vertex (§14) resolves three separate spin channels (Γ6, Γ7a, Γ7b); the local cluster-ED irreducible-vertex extraction (§19) still works in a 2-channel (aggregate spin, B₁g) space, so `V_irr_QQ` does not separately resolve a Γ7a- vs. Γ7b-specific local correction |
| No spatial fluctuations | Cannot describe a pseudogap, stripe order, or phase separation |
| RPA static ($\omega=0$) | Dynamical vertex corrections beyond the JT-channel retardation factor above are absent; the spin channels have no analogous retardation treatment |
| `K_eff`/cluster-ED update conditional | The expensive four-site cluster diagonalization and the associated stiffness correction are recomputed only when the SCF state has moved enough to warrant it, not literally every iteration; $Q$'s back-action on the exchange rigidity is therefore approximate during the SCF transient, though exact at convergence |
| Moriya-SCR lattice correction requires a smoothness gate | $\Gamma_Q$ (§12) depends on a quartic Landau coefficient extracted from a local 5-point $K_{\mathrm{eff}}(Q)$ stencil; if the stencil is judged numerically untrustworthy (`K_eff_spread` too large), the correction silently falls back toward the uncorrected `K_eff`, which can understate the fluctuation suppression in numerically noisy regimes |
| Simplified normal-state stability check | The earlier full $3\times3$ instability-matrix eigendecomposition has been replaced by a direct `K_lattice` vs. `K_spont` comparison (§16); this is exact for the pure-JT instability direction but does not itself diagnose a genuinely *mixed* (e.g. AFM–JT coupled) normal-state instability the way a full coupled-channel matrix treatment would — the post-convergence Hessian mode classification (§16) is what now carries that role, but only after a solve has already converged near the relevant point |
| $\partial\lambda_{\mathrm{pair}}/\partial Q$ at a frozen Fermi surface | FS geometry is evaluated at a fixed $Q$ rather than self-consistently re-resolved; a fully SC-state Bogoliubov–Lindhard version would be more expensive |
| $\delta\chi_{\tau}$ baseline subtraction approximate in D₂h | The normal-state B₁g response at finite `Delta_B1g_static` is estimated at $\Delta=0$; small residual D₂h corrections to the baseline are neglected |
| `chi_tau_weight` partial suppression | When the Richardson extrapolation only agrees at the finer step pair (`chi_tau_weight = 0.5`) or fails outright (`= 0.0`), the SC-JT feedback used in `lambda_JT_sc` may be under-resolved near a first-order boundary |
| SCF-dynamics regime classification | `first_order_jump` and `hysteretic` trigger a multi-seed restart (lowest free energy wins); `limit_cycle` only damps the mixing rate; the classification is heuristic, based on the shape of the $\vert \Delta\vert $ history |
| Eg,2 channel partially self-consistent | The Eg,2 phonon is fully wired into the Hamiltonian, free energy, and Hessian, but its exchange-driven rigidity correction and its cross-rigidity with the B₁g channel are currently left at zero (they vanish by Kramers symmetry at the level presently implemented); its own SC-triggering diagnostics are therefore less developed than the B₁g channel's |
| Incommensurate AFM handled only as a diagnostic + soft retry | `_scan_incommensurate_nesting` detects a preference for $q^{*}\neq(\pi,\pi)$ and triggers one softened-$M$ retry, but the BdG Hamiltonian itself remains fixed at commensurate $(\pi,\pi)$ ordering throughout — a genuinely incommensurate spiral solve is not implemented |
| $V_d$ sign-flip EMA | Suppresses numerical oscillation in the d-wave vertex but may slow the genuine response near a doping-driven crossover between d-wave and s-wave dominance |
| Superexchange cluster is a 3-site chain with independent ligand modes | `_multiorbital_2hole_hamiltonian` (§4) models the ligand as three independent per-$L_z$ modes rather than a single fully hybridized ligand orbital; this preserves $J_z$ through the SOC mixing but is itself an approximation to the true ligand electronic structure |
| Cluster-ED vertex is a $q=0,\omega=0$ local estimate | `V_irr_QQ` from the four-site plaquette (§19) is a local quantity standing in for a genuinely $k$-dependent, dynamical vertex; the physical-state diagonalization plus the two reference-spectra diagonalizations, together with the 5-point stiffness stencil (each point its own diagonalization), make this cluster-ED evaluation the single most expensive step in the SCF loop |

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
- Moriya spin fluctuations, and its self-consistent-renormalization extension to a soft lattice mode (§12): Moriya, T. (1985). *Spin Fluctuations in Itinerant Electron Magnetism.* Springer.
- Nearest positive-semidefinite matrix projection: Higham, N.J. (1988). *Linear Algebra Appl.* 103, 103.

---

*For questions or contributions, open an issue or pull request.*
