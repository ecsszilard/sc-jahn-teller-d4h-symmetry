# Multipolar SC-Triggered Jahn–Teller Model

> **A renormalized mean-field theory (RMFT) simulation of superconductivity-triggered B₁g Jahn–Teller distortion in a D₄h charge-transfer insulator with strong spin–orbit coupling and collinear antiferromagnetic order.**

This repository implements the self-consistent theory described in *"Multipolar superconductivity and Jahn–Teller activation in strongly correlated systems"* (Ecsenyi Szilárd, 2026). The core claim is that in a class of strongly spin–orbit-coupled, charge-transfer Mott insulators, a **B₁g lattice (Jahn–Teller) distortion is symmetry-forbidden in the antiferromagnetic normal state and is unlocked only by Cooper-pair condensation** — the causal arrow runs from superconductivity to the structural instability, not the other way around.

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

- The Γ₆ ground-state doublet carries **no electric quadrupole moment**: a pure Γ₆ (or Γ₇) manifold has $\langle\Gamma_6|Q^{(2)}_{B_{1g}}|\Gamma_6\rangle = 0$ exactly, by group theory alone.
- Consequently the B₁g Jahn–Teller distortion is **symmetry-forbidden** in the AFM normal state: $\Gamma_{JT} \not\subset \Gamma_{\mathrm{AFM}} \otimes \Gamma_{\mathrm{AFM}}$.
- Cooper-pair condensation, provided the pairing channel is genuinely **interorbital** (Γ₆↔Γ₇, not Γ₆↔Γ₆ or Γ₇↔Γ₇), builds up a Bogoliubov coherence between the two doublets.
- Only in this paired subspace does rank-2 multipolar order become accessible: $\Gamma_{JT} \subset \Gamma_{\mathrm{pair}} \otimes \Gamma_{\mathrm{pair}}$.
- The B₁g JT distortion therefore emerges as an **induced response of the condensate**, not as a primary instability — superconductivity comes first, the lattice distortion follows.

The three-fold splitting of the $t_{2g}$ shell under SOC actually produces **two** $\Gamma_7$-like doublets once the crystal field is included ($\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}$, see §1). The code carries all three Kramers doublets ($\Gamma_6$, $\Gamma_{7a}$, $\Gamma_{7b}$) exactly, with no downfolding; the SC-triggered-JT mechanism above is about $\Gamma_6\leftrightarrow\Gamma_{7a}$ specifically, since $\Gamma_{7a}$ is (by construction, §1) the JT-active partner. $\Gamma_6$–$\Gamma_{7b}$ coupling is symmetry-allowed by the same $\Gamma_6\otimes\Gamma_7\supset B_{1g}$ argument, and the code tracks it as a diagnostic (see §8), but the self-consistent solution does not pair through it.

### Material class and the three viability conditions

The theory targets systems with:

- D₄h point-group symmetry **and** a global inversion center (square-lattice materials: cuprates, pnictides, selected layered transition-metal oxides). Inversion symmetry is what allows the Cooper pair to have a well-defined parity (pure singlet or pure triplet); if it is broken locally, Rashba/Dresselhaus terms mix the two and the clean tensor-product selection rule below no longer applies unambiguously.
- Strong electron correlation with a charge-transfer (ZSA-type) insulating parent state.
- Strong SOC that reorganizes the $t_{2g}$ manifold into $\Gamma_6 \oplus \Gamma_{7a} \oplus \Gamma_{7b}$ Kramers doublets (typical of 4d/5d transition-metal ions, though the framework is agnostic to the microscopic origin of the SOC scale).
- Superexchange-stabilized collinear AFM order in the parent compound.
- Finite hole doping ($\delta > 0$), which restores the itinerancy needed for coherent Cooper pairing.

Three conditions must hold simultaneously for the mechanism to operate:

1. **Charge-transfer insulating character** — the ZSA charge-transfer gap and the on-site $U_{dd}$ (both primary inputs, §4) are large enough that genuine Mott physics is in play.
2. **Non-Mott-localized coherence** — the Cooper pairs must actually be mobile; this requires $\delta > 0$ so the Gutzwiller kinetic factor $g_t$ does not collapse to zero.
3. **Moderate AFM order** — AFM correlations must be present (they are what forbids the JT channel in the normal state) but not so strong that spin fluctuations kill superconductivity outright (Stoner criterion $J_{\mathrm{eff}}\cdot\chi_{SS} < 1$).

### Why the pairing must be interorbital, and why turning on Δ alone is not enough

A tempting but incorrect picture is that the condensate simply mixes $|\Gamma_6\rangle$ and $|\Gamma_{7a}\rangle$ into a coherent superposition $\alpha|\Gamma_6\rangle+\beta|\Gamma_{7a}\rangle$, giving a quadrupole moment linear in the mixing amplitude $\beta\propto\Delta$. This cannot be right: it would make $\langle B_{1g,\mathrm{op}}\rangle$ depend on the arbitrary global $U(1)$ phase of $\Delta$, violating gauge invariance. The correct microscopic statement is that the condensate modifies the **normal** (charge-conserving) density matrix through the Bogoliubov coherence factors: $\langle c^\dagger_{6\sigma}c_{7a\sigma'}\rangle$ picks up a $v^*v \sim |\Delta|^2$ contribution, which is explicitly gauge-invariant and appears at *quadratic*, not linear, order in $\Delta$.

For this to happen at all, the pairing must directly connect $\Gamma_6$ and $\Gamma_{7a}$ — a purely intraorbital singlet (Γ₆–Γ₆ or Γ₇ₐ–Γ₇ₐ) generates no $\Gamma_6$–$\Gamma_{7a}$ Bogoliubov mixing and leaves the JT channel closed even in the superconducting state. This is exactly the structure built into the code's two self-consistent pairing operators $D_s$, $D_d$ (§8 below), which are both interorbital by construction.

There is a further, more subtle point verified directly in the group-theoretic analysis: **at $Q=0$, even with $\Delta\neq0$, a purely $D_s/D_d$-paired BdG ground state still gives $\langle B_{1g,\mathrm{op}}\rangle \equiv 0$ band-pair by band-pair.** The reason is that $B_{1g,\mathrm{op}}$ in the numerical SOC eigenbasis turns out to be **spin-conserving** ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\uparrow}$), while the singlet pairing operator only ever connects opposite pseudospin sectors; $S_z$ stays block-diagonal in those same sectors. The actual switch is thrown by the *self-consistently induced anomalous coherence* $\langle\tau_x\rangle_{\mathrm{anom}}$ — numerically the same operator as $B_{1g,\mathrm{op}}$ — which is exactly zero at $Q=0$ and becomes nonzero only once **both** $\Delta\neq0$ **and** $Q\neq0$. In the code this anomalous coherence is the quantity `F67s_mf`, fed back into the local Hamiltonian (§19 below) only once a nonzero JT distortion has already appeared; the loop is genuinely self-consistent rather than a one-way "SC turns on JT."

### The JT distortion as a thermodynamic order parameter, not a dynamical mode

$Q$ is treated as a **macroscopic, thermodynamic order parameter** — physically the amplitude of a flat (dispersionless) optical Einstein phonon — rather than a fluctuating dynamical degree of freedom. Differentiating the free energy with respect to $Q$ gives the equilibrium condition and the softening criterion

$$\lambda_{JT}^{\mathrm{norm}} = \chi_{QQ}/K_{\mathrm{eff}} \qquad (<1\text{ stable},\ =1\text{ onset},\ >1\text{ spontaneous JT}).$$

A natural question is whether the free energy contains a **linear** coupling $Q\cdot|\Delta|^2$ between the lattice and the condensate. Naively one might argue this is forbidden simply because "$|\Delta|^2$ is always $A_{1g}$" — but that argument is incomplete, because the model's two pairing channels ($D_s$, $D_d$) do not necessarily share one point-group label, and their cross term could in principle transform as $B_{1g}$. The theory derives, and the code verifies numerically, that this cross-coupling is *exactly* zero in D₄h — an exact but non-generic consequence of the purely interorbital structure of the pairing operators, not a trivial symmetry accident. (If the lattice already sits in D₂h because of a finite static crystal field `Delta_B1g_static` ≠ 0, this exact cancellation is lifted and a genuine linear coupling appears — see §3 below.) The upshot: in the clean D₄h limit, SC-triggered JT is a **threshold phenomenon**. The condensate progressively softens the B₁g mode's effective stiffness $K_{\mathrm{eff}}$ until $\chi_{QQ} = K_{\mathrm{eff}}$, at which point the lattice snaps into a finite distortion.

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

**Kramers doublet identification** proceeds in two steps:
1. The three Kramers doublets of $H_{SOC}+H_{CF}$ are sorted by the expectation value $\langle L\!\cdot\!S\rangle$; the doublet with the most negative value is assigned $\Gamma_6$ ($j_{\mathrm{eff}}=1/2$-like), and the two remaining candidates are the $\Gamma_7$ pair.
2. Within each doublet a 2×2 diagonalization of $S_z$ selects the exact $z$-polarized Kramers partners (`up`, `dn`) and their eigenvalues (`sz_up`, `sz_dn`). Between the two $\Gamma_7$ candidates, the one with the **larger $|\langle S_z\rangle|$** is assigned $\Gamma_{7a}$ (the JT-active partner); the total moment $\mu_z=\langle L_z+2S_z\rangle$ is computed only as an independent **cross-check** and triggers a warning log if it disagrees with the spin-polarization criterion — this can happen in strongly mixed CF/SOC regimes.

**All three doublets enter the solver exactly — there is no downfolding.** The resulting basis is $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$, and all six components enter every operator (`B1g_op`, `Eg2_op`, the hopping matrices, and the BdG Hamiltonian itself). $\Gamma_{7b}$ disperses as a genuine band through the same rigorous orbital-projected hopping as $\Gamma_6/\Gamma_{7a}$ (§6, §9), rather than being projected out and reintroduced only as a perturbative correction. Several derived quantities are cached on `ModelParams` at this point:

- `sz_op = [sz6_up, sz6_dn, sz7_up, sz7_dn, sz7b_up, sz7b_dn]` — **exact** $\langle S_z\rangle$ eigenvalues from the doublet diagonalization (not an approximate moment-ratio model); used directly as the AFM Weiss-field weights and as the spin vertex in every susceptibility calculation.
- `multi_op` — the effective multipolar spin operator entering the cluster exchange $H_{\mathrm{exch}} = J\cdot(\mathrm{multi\_op}\otimes\mathrm{multi\_op})$, built as $\mathrm{diag}\big((|sz_6|\cdot P_6+|sz_{7a}|\cdot P_{7a}+|sz_{7b}|\cdot P_{7b})\cdot sz_{\mathrm{diag}}\big)$.
- `p_7` — the average $\Gamma_7$ (either doublet) orbital-weight admixture in the $\Gamma_6$ eigenvectors; interpolates the d-wave Gutzwiller factor (§5).
- `Delta_CF = evals[2] − evals[0]` (Γ₇ₐ–Γ₆ gap, JT-active) and `g7split = evals[4] − evals[2]` (Γ₇ᵦ–Γ₇ₐ internal splitting).
- Orbital-character weights `_w6_xz/_yz/_xy`, `_w7_xz/_yz/_xy`, `_w7b_xz/_yz/_xy` — the $d_{xz}/d_{yz}/d_{xy}$ character of $\Gamma_6$, $\Gamma_{7a}$, and $\Gamma_{7b}$, used to build both the Q-dependent exchange anisotropy $\eta_J(Q)$ (§6) and the orbital-selective hopping matrices `Tx_A_xz/_yz/_xy` (§6, §9).

If `lambda_soc`, `Delta_tetra`, or `Delta_B1g_static` are mutated on a live solver, `params.__post_init__()` must be followed by `solver._rebuild_orbital_operators()` so that `B1g_op`, `B1g_24`, `Eg2_op`, `Eg2_24`, `sz_op`, `multi_op`, `Sz_nambu`, and `Sz_stag_nambu_channels` stay consistent with the new eigenbasis.

### 2. Symmetry Protection of the JT Channel

**Selection rule in a pure doublet.** A rank-$k$ irreducible tensor operator has a nonzero diagonal matrix element in $\Gamma_6$ only if $\Gamma^{(k)}\subset\Gamma_6\otimes\Gamma_6$. Since $\bar D_{4h}$ character theory gives $\Gamma_6\otimes\Gamma_6=\Gamma_7\otimes\Gamma_7=A_{1g}\oplus A_{2g}\oplus E_g$ — containing neither $B_{1g}$ nor $B_{2g}$ — the quadrupole operator has $\langle\Gamma_6|Q^{(2)}_{B_{1g}}|\Gamma_6\rangle=0$ **exactly**. A pure $\Gamma_6$ (or $\Gamma_7$) manifold carries no electric quadrupole moment and does not couple to a $B_{1g}$ lattice shear. Because the collinear AFM state stabilizes only this kind of dipolar (rank-1), spin–orbitally mixed order, the JT channel is symmetry-blocked in the normal state.

**Cross-product opens the channel.** $\Gamma_6\otimes\Gamma_7 = B_{1g}\oplus B_{2g}\oplus E_g$ *does* contain $B_{1g}$, so the **off-diagonal** element $\langle\Gamma_6|Q^{(2)}_{B_{1g}}|\Gamma_{7a}\rangle\neq0$ is allowed — and, by the identical argument, so in principle is $\langle\Gamma_6|Q^{(2)}_{B_{1g}}|\Gamma_{7b}\rangle$, which is why the code tracks a diagnostic $\Gamma_6$–$\Gamma_{7b}$ channel (§8) even though it is not fed back self-consistently. Realizing either channel requires a genuinely interorbital pairing operator and, as emphasized above, requires $Q\neq0$ as well as $\Delta\neq0$ — the self-consistent loop, not a one-shot symmetry argument, is what actually opens the channel.

**Θ symmetry and the global BZ cancellation.** In the collinear AFM state, time reversal $\mathcal T$ is broken, but the combined Shubnikov element $\Theta=\mathcal T\cdot\tau_{AB}$ (time reversal composed with the $A\leftrightarrow B$ sublattice translation) survives; since $\tau_{AB}^2=+1$ and $\mathcal T^2=-1$, $\Theta^2=-1$. This does **not** give pointwise Kramers degeneracy — $\Theta$ maps crystal momentum $k\to -k$, and in the magnetic Brillouin zone $k$ and $-k$ are generally inequivalent — but it does impose a **global** constraint on Brillouin-zone integrals. The spin–quadrupole susceptibility integrand is odd under $\Theta$ at $\Delta=0$, so the full-BZ Lindhard sum cancels identically:

$$\chi_{SQ}(q) = \int_{\mathrm{BZ}} d^2k\; \mathcal I_{SQ}(k,q) = 0 \qquad (\Delta=0).$$

This BZ-wide cancellation — not a pointwise Kramers argument — is enforced in the code by the odd-in-$k$ structure of the normal-state Lindhard kernel (`_lindhard_bubble` with `_NORMAL_SECTOR_PAIRS`), and is checked at runtime by `estimate_chi_SQ_q_full`, which logs the normal-state $\chi_{SQ}(q)$ peak location and amplitude directly — it should be numerically indistinguishable from zero — rather than returning a boolean pass/fail flag; the same call also logs a local-vertex-validity warning and a normal-vs-SC peak-shift warning as related diagnostics.

In the superconducting state the full Nambu–Lehmann sum includes the anomalous (Gorkov) sectors, whose Bogoliubov coherence factors can break the odd-$\Theta$ structure — but as detailed above, this alone is not sufficient at $Q=0$: a purely singlet, interorbital pairing keeps $S_z$ and $B_{1g,\mathrm{op}}$ acting within disjoint pseudospin sectors band-pair by band-pair, so the product stays exactly zero until the self-consistently generated $Q\neq0$ genuinely couples the two sectors.

**The selection ratio, explicitly.** The quantity that actually crosses the symmetry boundary is the Γ₆–Γ₇ₐ anomalous (Gorkov) singlet amplitude, computed directly from the converged BdG eigensystem as

$$F_{67s} = \sum_k (1-2f_n)\,\mathrm{Re}\big[u^*_{6\uparrow}v_{7a\downarrow} - u^*_{6\downarrow}v_{7a\uparrow}\big] \quad\text{(mean over sublattices)},$$

with $F_{67s}\equiv0$ whenever $\Delta=0$ (exact D₄h selection rule) — the code's own inline invariant. The mean-field quantity actually fed back into the local Hamiltonian is a Gutzwiller-weighted average over both pairing channels, `F67s_mf = g_eff * F_67s`, with `g_eff = (g_Delta_s * |Δ_s| + g_Delta_d * |Δ_d|)/(|Δ_s|+|Δ_d|)` — so a channel that carries more of the total gap amplitude also carries proportionally more weight in how strongly the condensate talks to the lattice.

### 3. The B₁g Operator and the D₄h/D₂h Crossover

The B₁g phonon coupling operator is constructed from the same $t_{2g}$ operators used for $H_{CF}$ and projected into the full 6-dimensional $\Gamma_6\oplus\Gamma_{7a}\oplus\Gamma_{7b}$ subspace — no downfolding:

```
B1g_op = real(U6† · (Lx² − Ly²)_t2g · U6)     # 6×6, real, Hermitian
```

where `U6` is the 6×6 change-of-basis matrix from the bare $t_{2g}\otimes\mathrm{spin}$ manifold to the Kramers-doublet (SOC+CF) eigenbasis built in §1.

- **D₄h (`Delta_B1g_static = 0`):** the $\Gamma_6$–$\Gamma_6$ and $\Gamma_{7a}$–$\Gamma_{7a}$ (and $\Gamma_{7b}$–$\Gamma_{7b}$) diagonal blocks of `B1g_op` vanish exactly and it is purely off-diagonal and **spin-conserving** — it connects $\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\uparrow}$ (and, with generally smaller weight, $\Gamma_6\leftrightarrow\Gamma_{7b}$). Hence $\langle B_{1g,\mathrm{op}}\rangle=0$ in *any* normal state.
- **D₂h (`Delta_B1g_static ≠ 0`):** the operator picks up real diagonal ($A_{1g}$-like) components that renormalize $\Delta_{CF}$ but do not by themselves drive JT, plus additional off-diagonal weight. The fraction of the operator that remains purely off-diagonal — `b1g_weight = ‖B1g_offdiag‖/(‖diag(B1g_op)‖+‖B1g_offdiag‖)` — is computed on demand (e.g. inside the diagnostic branch of the Jacobi-kick, §14) rather than cached as a permanent `ModelParams` field; it quantifies how much of the operator remains genuinely SC-triggered versus normal-state-active. `b1g_weight ≈ 1` in the clean D₄h limit.

The 24×24 Nambu extension `B1g_24` (built in `_rebuild_orbital_operators`) carries the hole block as $-B_{1g,\mathrm{op}}^{T}$ (real, so $=-B_{1g,\mathrm{op}}$), consistent with BdG particle–hole symmetry; every JT coupling term in the Hamiltonian is written `H += g_JT · Q · B1g_op` rather than a hand-built matrix. `Eg2_24` mirrors this construction for the Eg,2 operator (§7).

### 4. ZSA Charge-Transfer Superexchange and the Weiss Field

AFM order originates from virtual $p$–$d$ hopping (ZSA charge-transfer superexchange), not from a bare Stoner Fermi-surface instability. Rather than a leading-order analytic formula, the superexchange scale $J_{pdct}$ is obtained from an **exact diagonalization of the minimal $d$–$p$–$d$ two-hole cluster**: `_kappa_superexchange` builds the two-hole Hilbert space of a 3-orbital (metal–ligand–metal), 2-spin chain (`_build_block`/`_apply_hop`, a small Slater-determinant exact-diagonalization routine, not a perturbative expansion) at the given $(t_{pd},\Delta_{CT},U_{dd},U_{pp})$, and returns half the singlet–triplet splitting $\kappa=(E_T-E_{S_0})/2$ of that cluster. `ModelParams.__post_init__` then sets

```
J_pdct = _kappa_superexchange(t_pd, Delta_CT, U_dd, U_pp) / t0²
```

`U_dd` is now itself a **primary input** (the on-site Hubbard repulsion, no longer parameterized through a dimensionless $U/t_0$ ratio); `U_pp`, the ligand (2p) hole–hole repulsion, is obtained from a second-order downfolding estimate that also depends on the new `Upp_ratio_bare` primary input (the bare, unhybridized $U_{pp}/U_{dd}$ ratio) and the coordination number:

```
U_pp = (Upp_ratio_bare · U_dd) / (1 + (Z/2) · t_pd² / (Delta_CT · (Delta_CT + U_dd)))
```

with `Z/2` the ligand coordination number (each ligand typically bridges two metal sites). `hybrid_scale` does **not** enter this formula; its role is instead the $k$-dependent quasiparticle downfolding weight $\beta^2(k)$ used in the BdG Hamiltonian and the cluster embedding (§6, §9).

The effective AFM Weiss field entering the BdG Hamiltonian is diagonal in the full 6-component $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$ basis, proportional to `sign_M · J_A1g[α,α] · M · sz_op[α]`, where `J_A1g` is the longitudinal (spin-preserving) exchange tensor from `exchange_channels()` (§6) and the doping renormalization is carried by the itinerant carrier density `n_kspace` returned from the chemical-potential solve.

### 5. Gutzwiller Renormalization and the Mott Guard

`t_pd` is the single primary hopping input; the effective $dd$ hopping `t0 = t_pd² / Delta_CT` is always derived, never set directly.

```
g_t       = 2δ/(1+δ)         # kinetic-energy suppression → 0 at half-filling
g_J       = 4/(1+δ)²         # exchange enhancement → 4 at half-filling
g_Delta_s = g_t                                        # on-site Γ6⊗Γ7 channel: kinetic in origin
g_Delta_d = g_t + (g_J − g_t) · p_7                    # d-wave B1g channel: interpolated by Γ7 admixture
```

The superexchange is always computed from the **bare** hopping `t_pd`, then multiplied by `g_J` — computing it from Gutzwiller-*renormalized* bands would double-count the suppression (a spurious `g_t²` factor), since `g_t` renormalizes only the kinetic energy while `g_J` renormalizes only the exchange; the two are orthogonal channels in this RMFT scheme. The lattice-summed Weiss-field/superexchange scale is $J_{\mathrm{eff}} = Z \cdot J_{\mathrm{bond}}$, where the single-bond quantity $J_{\mathrm{bond}} \sim g_J \cdot J_{pdct} \cdot (t_x^2+t_y^2)$ is exactly the per-component value `exchange_channels()` (§6) returns as `J_A1g_diag`, itinerancy-weighted by the self-consistent carrier density.

A **Mott guard** suppresses superconductivity when `g_t < _G_T_COHERENCE_MIN = 0.10` (i.e. $\delta \lesssim 0.053$): below this the Gutzwiller factor signals that the Zhang–Rice-singlet band is no longer coherent enough to support a physical SC gap, and the solver returns a non-metallic/non-SC result rather than a spuriously converged one.

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
T_y(Q) = t_y(Q)·A_xz + t_x(Q)·A_yz + [t_x(Q)+t_y(Q)]/2·A_xy
```

from the exact $d_{xz}/d_{yz}/d_{xy}$ orbital-character projectors (`Tx_A_xz`, `Tx_A_yz`, `Tx_A_xy`, built once in `__post_init__` and satisfying $A_{xz}+A_{yz}+A_{xy}=I_6$), so that $\Gamma_6$, $\Gamma_{7a}$, and $\Gamma_{7b}$ each disperse according to their own $d_{xz}/d_{yz}/d_{xy}$ admixture rather than sharing one isotropic band. `hopping_matrices_dQ(Q)` gives the companion exact analytic $\partial T_{x,y}/\partial Q$.

The full multipolar exchange tensor is likewise Q-dependent through both the overall B₁g channel opening (`exchange_channels()` returns `J_A1g_diag`, now a 6-component array covering $\Gamma_6,\Gamma_{7a},\Gamma_{7b}$, and the scalar `J_B1g_scalar`, the latter proportional to `(tx²−ty²)`) and an orbital-selective asymmetry — separate ratios $\eta_{J,7a}(Q)=\sqrt{J_{\Gamma_{7a}}/J_{\Gamma_6}}$ and $\eta_{J,7b}(Q)=\sqrt{J_{\Gamma_{7b}}/J_{\Gamma_6}}$ — since $d_{xz}$ hops preferentially along $x$ and $d_{yz}$ along $y$; both ratios equal 1 exactly at $Q=0$. The commutator $\|[B_{1g,\mathrm{op}}, H_{AFM}]\|/|\Delta_{CF}|$ (`blocking_ratio`) diagnoses how strongly the normal-state exchange field blocks the B₁g channel.

**Further-neighbor hopping.** Two additional primary inputs, `t_prime_ratio` and `t_dprime_ratio`, set a same-sublattice, diagonal dispersion on top of the orbital-selective nearest-neighbor terms above:

```
disp_nnn(k) = −4·g_t·t_prime·cos(kx)·cos(ky) − 2·g_t·t_dprime·[cos(2kx)+cos(2ky)]
t_prime  = t_prime_ratio  · t0     # 2nd-neighbor, diagonal (1,1)-type hopping
t_dprime = t_dprime_ratio · t0     # 3rd-neighbor, axial (2,0)-type hopping
```

added as `disp_nnn(k)·I₆` to each sublattice's diagonal block (2nd/3rd-neighbor bonds stay within one checkerboard sublattice, unlike the nearest-neighbor $T_{x,y}$ terms, which connect A↔B).

### 7. The Eg,2 Phonon Channel

Alongside the B₁g mode, the model carries an independent second vibronic channel of Eg,2 symmetry, built from the operator $L_yL_z+L_zL_y$ and projected into the same full $\Gamma_6\oplus\Gamma_{7a}\oplus\Gamma_{7b}$ subspace exactly like `B1g_op`:

```
Eg2_op  = U6† · (Ly·Lz + Lz·Ly)_t2g · U6      # 6×6, Hermitian (complex in general)
```

with its own coupling constant `g_Eg2` (eV/Å), bare stiffness `K_lattice_Eg2` (eV/Å²), and distortion amplitude `Q_Eg2`, entering the BdG Hamiltonian, the free energy, and the Hessian on the same footing as the B₁g channel via `Eg2_24` (the 24×24 Nambu lift, mirroring `B1g_24`) and `Eg2_expectation()`. Unlike `B1g_op`, `Eg2_op` connects Kramers partners with an actual **spin-flip** structure ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\downarrow}$) rather than the spin-conserving structure of `B1g_op` — the two channels probe genuinely different multipolar sectors of the same $\Gamma_6\otimes\Gamma_7$ manifold.

At the current stage of the implementation, the exchange contribution to the Eg,2 stiffness and the B₁g–Eg,2 cross term vanish identically by Kramers symmetry, so `K_eff_Eg2` is left at its bare value `K_lattice_Eg2` (no exchange-driven softening is computed for this channel yet, in contrast to the fully renormalized `K_eff` for B₁g). The Eg,2 channel is therefore best read, in the current code, as a genuine second JT-active degree of freedom already wired through the Hamiltonian and observables, whose own self-consistent back-action on the lattice stiffness is not yet as developed as the B₁g channel's.

### 8. Dual B₁g Pairing Channels (and a Diagnostic Γ₆–Γ₇ᵦ Channel)

Two symmetry-equivalent, **interorbital** B₁g pairing channels are carried self-consistently, exactly as required by the symmetry argument in §2:

- **Channel s** — on-site inter-orbital singlet ($\Gamma_6\otimes\Gamma_{7a}\to B_{1g}$, constant $k$-space form factor):
  ```
  D_s = Δ_s · (|6↑⟩⟨7a↓| − |6↓⟩⟨7a↑|)
  ```
- **Channel d** — inter-sublattice d-wave ($\varphi(k)=\cos k_x-\cos k_y \to B_{1g}$ in $k$-space):
  ```
  D_d = Δ_d · φ(k) · (|A:6↑⟩⟨B:7a↓| − |A:6↓⟩⟨B:7a↑|)
  ```

Both channels feed into the same gap-equation infrastructure (§21) with independent Gutzwiller factors `g_Delta_s`, `g_Delta_d` (§5) and independent RPA-vertex projections; the dominant channel is identified post-convergence by the largest eigenvalue of the linearized 2×2 pairing kernel (§21).

**The optional Γ₆–Γ₇ᵦ analog.** `VectorizedBdG._build_H_stack` accepts two further arguments, `Delta_s7b` and `Delta_d7b`, that build the same singlet/d-wave structure between $\Gamma_6$ and $\Gamma_{7b}$ (symmetry-allowed by the identical $\Gamma_6\otimes\Gamma_7\supset B_{1g}$ argument, §2); they default to `0j`, which reproduces the $\Gamma_{7a}$-only model exactly. `compute_gap_eq_vectorized` computes a **raw, one-shot RPA estimate** of these amplitudes (`Delta_s7b_diag`, `Delta_d7b_diag`) alongside the real gap-equation solution, using the same s-/d-channel RPA vertex; these are returned purely as diagnostics — the SCF loop never feeds them back into the Hamiltonian — so the self-consistent solution always pairs through $\Gamma_6$–$\Gamma_{7a}$ only, and the diagnostic tells you whether the $\Gamma_6$–$\Gamma_{7b}$ channel would also want to activate if it were allowed to.

### 9. The 24×24 BdG Hamiltonian (Doubled Unit Cell, Full 3-Doublet Basis)

Nambu basis $\Psi=[\text{Particle}_A(6),\ \text{Particle}_B(6),\ \text{Hole}_A(6),\ \text{Hole}_B(6)]$, each block ordered $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T_AB(k)    │  D_s      D_d        │   ← Particle sector
      │  T_AB†(k)  H_B     │  D_d      D_s        │
      ├────────────────────┼─────────────────────┤
      │  D_s†   D_d†       │  −H_A*   −T_AB*      │   ← Hole sector
      │  D_d†   D_s†       │  −T_AB†* −H_B*       │
      └────────────────────┴─────────────────────┘
```

with `D_s`, `D_d` as in §8 (plus the optional, always-zero-by-default $\Gamma_6$–$\Gamma_{7b}$ terms). $H_A$, $H_B$ are the $6\times6$ local (AFM Weiss field + crystal field + JT + $t',t''$ diagonal dispersion) sublattice Hamiltonians; $T_{AB}(k)=-2g_t\big[\cos k_x\,T_x(Q)+\cos k_y\,T_y(Q)\big]$ is the **orbital-selective** inter-sublattice hopping block of §6 (not a scalar times the identity). The particle–hole off-diagonal blocks use the **transposed** (not Hermitian-conjugate) pairing operator, consistent with BdG particle–hole symmetry.

**The JT coupling itself is $k$-dependent**, not a rigid on-site term: both the JT coupling and the anomalous Weiss field are modulated by the same $k$-dependent quasiparticle spectral weight $\beta^2(k)$ used for the charge-transfer downfolding:

```
β²(k) = sigmoid[ k_s·(1 − hybrid_scale·[t̄·(cos kx+cos ky) + δt·(cos kx−cos ky)]/Delta_CT − 0.5) ]
        t̄ = (tx+ty)/2,  δt = (tx−ty)/2                      # wave_function_weight(tx, ty, kx, ky), k_s=10

H += β²(k) · g_JT · Q · B1g_op                                # k-weighted JT coupling
H += β²(k) · Z · J_B1g_bare · F67s_mf · B1g_offdiag           # k-weighted anomalous Weiss field
```

$\beta^2(k)$ is the fraction of the quasiparticle wavefunction that remains on the metal ion rather than the ligands at each $k$-point (a sigmoid-regularized downfolding factor, `wave_function_weight`), so both channels that couple through the metal-ion orbital angular momentum operators ($B_{1g,\mathrm{op}}$) are naturally suppressed where the quasiparticle is more ligand-like. By contrast, the Eg,2 term (§7) is added **without** this $k$-weighting — `H += g_Eg2·Q_Eg2·Eg2_op` uniformly — reflecting its treatment, at the current stage of the implementation, as a spatially uniform (q=0) structural order parameter. In the particle sector this all enters as shown above; the hole sector carries the corresponding $-(\cdot)^T$, and exact Hermiticity is enforced after assembly.

`VectorizedBdG._build_H_stack` builds and diagonalizes this 24×24 matrix for the entire k-grid in one batched `numpy.linalg.eigh` call, reusing a pre-allocated buffer (`out=`) across SCF iterations to avoid repeated allocation.

The physical electron density is $\langle n_{i\sigma}\rangle=\sum_n |u_{n,i\sigma}|^2 f(E_n) + |v_{n,i\sigma}|^2(1-f(E_n))$ — both terms carry a positive sign, since $|v|^2(1-f)$ is the filled-band contribution from below the Fermi level.

### 10. Observables via `VectorizedBdG`

All thermal-average observables are extracted from a single batched diagonalization:

| Observable | Formula (schematic) | Role |
|---|---|---|
| **⟨B1g⟩** (full) | $\mathrm{Tr}[B_{1g,24}\cdot\rho_k]$, `/4` for Nambu+sublattice doubling | Hellmann–Feynman lattice force: $Q_{\mathrm{eq}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ |
| **⟨Eg2⟩** | same construction against `Eg2_24` | Hellmann–Feynman force for the Eg,2 channel |
| Magnetization | $\langle S_z\rangle$ via the exact `sz_op` weights, per-channel via `Sz_stag_nambu_channels` | AFM order parameter `M` (channel-resolved: Γ₆, Γ₇ₐ, Γ₇ᵦ) |
| `F67s_mf` | Gorkov Γ₆–Γ₇ₐ singlet amplitude, `_compute_F67_singlet` | Anomalous Weiss-field back-action (§2, §19) |
| Density | $\sum_n[|u|^2 f + |v|^2(1-f)]$ / 4 | Chemical-potential control |
| Pairing s / d | on-site / inter-site $u^*v$ combinations (Γ₆–Γ₇ₐ and diagnostic Γ₆–Γ₇ᵦ) | s-/d-channel gap-equation inputs |

The lattice update in the SCF loop uses the **full** $\langle\hat B_{1g}\rangle=\mathrm{Tr}[B_{1g,24}\cdot\rho]$, not a bare off-diagonal $\tau_x$ piece, because in D₂h `B1g_op` gains diagonal and spin-preserving components that are active even without SC — using only the off-diagonal piece would break Hellmann–Feynman consistency with `H_{JT}=g_{JT}\,Q\,B_{1g,\mathrm{op}}`. In D₄h the two expressions coincide exactly. Concretely, `B1g_expectation` contracts the already-diagonalized Nambu eigenvectors against `B1g_24` and weights by the occupation and the same $\beta^2(k)$ ZRS factor used in the Hamiltonian (§9), so the observable used to drive $Q$ stays consistent with what was actually put into $H$:

```python
diag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), B1g_24, ec).real     # a,b: 24 Nambu components; n: band index
exp_k   = np.einsum('kn,kn->k', diag_qp, f_n) * beta2_k               # per-k thermal average, β²(k)-weighted
B1g_exp = np.dot(k_weights, exp_k) / 4.0                              # /4: Nambu (particle–hole) × sublattice (A–B)
```

Summing over all 24 Nambu bands with plain $f(E_n)$ (not $1-f$) automatically covers the hole contributions too, since the hole-sector sign is already built into `B1g_24` itself (§3) and $f(-E)=1-f(E)$. `Eg2_expectation` mirrors this exactly but contracts against `Eg2_24` instead, and — consistent with §7/§9 — omits the $\beta^2(k)$ weighting.

### 11. Exchange Rigidity: ∂²F_ex/∂Q²

`compute_K_eff_full` evaluates the exchange contribution to the B₁g stiffness from $F_{\mathrm{ex}}=\sum_{\alpha\beta}J_{\alpha\beta}(Q)\langle O_\alpha(Q)\rangle\langle O_\beta(Q)\rangle$ via the full product rule:

```
∂²F_ex/∂Q² = O·(∂²J/∂Q²)·O + 2·(∂O/∂Q)·J·(∂O/∂Q) + 2·O·J·(∂²O/∂Q²) + 4·O·(∂J/∂Q)·(∂O/∂Q)
```

using central finite differences at step $\varepsilon=\max(10^{-4}, 0.01|Q|+10^{-4})$. At $Q=0$ the B₁g selection rule forces $\partial O/\partial Q=0$ and $\partial^2 J/\partial Q^2=0$, so only the $O\!\cdot\!J\!\cdot\!\partial^2O/\partial Q^2$ term survives; away from $Q=0$ all four terms contribute. `K_eff = K_lattice + ∂²F_ex/∂Q²` (negative correction = exchange softens the mode; this is what can drive `K_eff` toward the JT-triggering threshold in the SC state). As noted in §7, the analogous Eg,2 and B₁g–Eg,2 cross-rigidity terms currently vanish identically by Kramers symmetry, so `K_eff_Eg2` stays at its bare `K_lattice_Eg2` value.

### 12. B₁g Orbital Susceptibility χ_τ (Richardson-Extrapolated)

```
chi_tau = ∂⟨B1g_op⟩ / ∂(g_JT · Q)      (signed; evaluated separately at Δ≠0 and Δ=0)
```

`_compute_chi_tau` uses an **adaptive** Richardson extrapolation: it first computes central differences at three step sizes $h, h/2, h/4$ and forms the two Richardson estimates $R_1=(4\,CD(h/2)-CD(h))/3$, $R_2=(4\,CD(h/4)-CD(h/2))/3$. If $|R_1-R_2|/|{\cdot}|<3\%$ the mean of $R_1,R_2$ is returned at full weight. If the raw central differences at the coarsest pair disagree by more than 20% (nonlinear regime), a fourth, finer step $h/8$ is tried; if that pair then agrees, the $h/8$-based value is returned at **half weight** (`chi_tau_weight = 0.5`); if it still disagrees, the derivative is judged unresolvable at this $Q$ and returned as **zero** (`chi_tau_weight = 0.0`), with a `[CHI-TAU]` warning either way. `richardson_ok` requires convergence at both scales.

Both the SC-state (`chi_tau_sc`) and normal-state (`chi_tau_n`) susceptibilities are computed (the normal-state baseline matters because a finite `Delta_B1g_static` gives D₂h a small nonzero response even at $\Delta=0$); `chi_tau_net = chi_tau_sc − chi_tau_n` isolates the SC-triggered excess and is what enters `lambda_JT_sc` (§17).

### 13. χ_QQ from Thermodynamic Finite Differences

The orbital JT susceptibility $\chi_{QQ}=-\partial^2\Omega/\partial Q^2$ is evaluated in the SC state by central finite difference of the total free energy, divided by 4 to correct for the combined Nambu (particle–hole) and sublattice (A–B) doubling in the 24×24 BdG matrix. This SC-state $\chi_{QQ}$ is used **exclusively** for lattice-stability diagnostics (the G-matrix, §16); the pairing vertex itself always uses the normal-state ($\Delta=0$) susceptibilities, to avoid feeding the gap back into its own interaction.

### 14. Coupled Spin–JT RPA Vertex

The pairing vertex is built from a 2×2 coupled spin–orbital RPA in $[\mathrm{spin},\ \mathrm{JT{-}phonon}]$ channel space. The bare local interaction is **diagonal** in this channel basis — there is no separate bare spin–JT cross-vertex constant:

```
U = [[ J_eff,    0     ],
     [ 0,        V_JT_corr ]]
```

with `V_JT_corr = V_JT + V_irr_QQ`, where `V_JT = g_JT_bare² / K_bare` is the bare JT pairing vertex and `V_irr_QQ` is the B₁g–B₁g component of the **irreducible vertex extracted from the four-site plaquette cluster ED** (§19) — the local, non-perturbative renormalization of the JT self-interaction. All spin–JT **mixing** in the RPA matrix comes from the off-diagonal bare susceptibilities $\chi_{SQ},\chi_{QS}$ themselves (below), not from a bare cross-interaction.

**Bare susceptibilities.** $\chi_0(q)$ comes from the $\Delta=0$ BdG Hamiltonian via the static Lindhard formula, `_lindhard_bubble(sector_pairs, E_k, V_k, f_k, shift_idx, weights, η, kT)`, accelerated with `opt_einsum`. The normal-state sum runs over `_NORMAL_SECTOR_PAIRS` — 8 sector pairs covering the AA/BB (intra-sublattice) and AB/BA (inter-sublattice) particle and hole blocks — evaluated at $k$ and $k+q$ using the pre-built cyclic `shift_table` (§ Key Algorithms) rather than a second diagonalization. Each Lindhard term is additionally weighted by the same ZRS spectral weight $\beta^2(k)\beta^2(k+q)$ that modulates the JT coupling itself (§9), since the susceptibility bubble should only "see" the fraction of the quasiparticle that actually carries $B_{1g,\mathrm{op}}$/$S_z$ character. The static Lindhard function is real by time-reversal symmetry, so its imaginary part is discarded as roundoff, not physical information, after Hermiticity is enforced. Projections onto the physical channels:

```
χ_SS = Tr[Sz · χ₀[Γ6,Γ6] · Sz]     # spin–spin (dipole–dipole)
χ_SQ = Tr[Sz · χ₀[Γ6,Γ7a]]         # spin–orbital cross (dipole–quadrupole), then divided by g_JT
χ_QQ = −∂²Ω/∂Q² / 4                # orbital JT stiffness [eV/Å²]
```

The cross-terms $\chi_{SQ},\chi_{QS}$ are exactly zero in the normal state at $Q=0$ (§2) and become nonzero once $Q>0$ opens the B₁g channel. A single routine, `get_susceptibilities_sc`, covers both cases via the unified Nambu Lehmann sum against the pre-built `Sz_nambu`/`B1g_24`-derived vertex matrices — the normal state is simply its $\Delta_s=\Delta_d=0$ call. It applies a **PSD projection** of the symmetric 2×2 matrix with diagonal entries $\chi_{SS}$, $\chi_{QQ}$ and shared off-diagonal $\chi_{SQ}$ (Higham nearest-PSD via eigenvalue clamping) to guard against Cauchy–Schwarz violations from numerical noise near the QCP.

**Vertex assembly (`_rpa_det`/`_rpa_vertex`).** Writing $I-\hat U\chi_0$ in the 2×2 channel basis with entries $a,b,c,d$ below,

```
a = 1 − J_eff·χ_SS          b = −V_JT_corr·χ_QS
c = −J_eff·χ_SQ             d = 1 − V_JT_corr·χ_QQ
det = a·d − b·c
```

(note `V_JT_corr`, the cluster-ED-renormalized JT self-interaction, appears rather than the bare `V_JT` in $b$ and $d$ — the pairing vertex still uses the bare `V_JT` in its final assembly below, but the RPA screening matrix itself is built from the renormalized value). `det` is floored in magnitude — not in sign — at `max(_MATH_EPS, 1e-4·‖(a,b,c,d)‖)`, guarding only against an exact-zero numerical accident without ever masking a genuine sign change. The channel-space inverse $(a,b,c,d)^{-1}$ is then contracted into the physical pairing vertex

```
V(q) = J_eff²·χ_SS^RPA(q) + V_JT²·χ_QQ^RPA(q) + J_eff·V_JT·[χ_SQ^RPA(q) + χ_QS^RPA(q)]
```

and finally hard-clamped to $\pm V_{\mathrm{cap}}$, with `V_cap = _RPA_V_CAP_ALPHA · max(_RPA_BW_FACTOR·max(|tx|,|ty|), J_eff)` (`_RPA_V_CAP_ALPHA = 2.2`, `_RPA_BW_FACTOR = 8` for the tight-binding bandwidth estimate) — a numerical overflow guard only; the sign and divergence character of $V(q)$ near the QCP are never altered, and `det<0` (past the QCP) is deliberately left untouched rather than capped, so the SCF is not artificially trapped away from a genuinely unstable regime.

**Two separately tracked determinants.** The vertex cache stores `det_q0` (the $q=0$, ferromagnetic-channel determinant) and `det_afm` (the $q=(\pi,\pi)$, AFM-channel determinant) independently. SCF adaptive mixing and convergence behavior respond to `det_afm`; `det_q0` guards separately against an accidental ferromagnetic divergence. Both are logged at convergence (`dFM=`, `dAFM=` in the iteration log).

**Sign-flip EMA guard.** When $|\mathrm{det\_afm}|<$ `_DET_SIGN_FLIP_SCALE = 0.05` and the d-wave vertex `V_d_scalar` would flip sign relative to its cached value, the update is blended continuously rather than switched, via a sigmoid in $|\mathrm{det\_afm}|/0.05$ (steepness `_EMA_SIGN_FLIP_SLOPE = 6.0`, floor `_EMA_SIGN_FLIP_W_MIN = 0.20`) — so the blend weight shrinks toward its floor near the QCP (preserving genuine sign ambiguity there, where a real physical crossover may be in progress) and grows toward 1 away from it (suppressing pure numerical noise).

**q-resolved vertex diagnostics**, stored in the cache and logged whenever `V_d < 0`: `V_afm_mean` (mean $V(q)$ in the AFM region of the sampled Fermi surface, $>0$ expected for spin-fluctuation-driven d-wave pairing), `V_fwd_mean` (mean $V(q)$ in the forward-scattering region, typically $<0$, cancelled by the d-wave form factor), and `V_neg_frac` (fraction of sampled $q$-points with $V(q)<0$; a large value flags a globally repulsive, unphysical vertex). Diagnostics of this kind are gated on having at least `_VERTEX_DIAG_MIN_FS = 10` sampled Fermi-surface points, below which the std/mean statistics are treated as unreliable.

**Moriya damping** is obtained self-consistently from the model's own Landau free-energy expansion rather than an empirical closed-form fit: `_moriya_gamma_landau` probes $F(M)$ at $M=0, h, 2h$ (`_MORIYA_LANDAU_M_STEP = 0.06`) via `_compute_bdg_free_energy`, extracts the quartic Landau coefficients $a,b$ from $F(M)=F_0+\tfrac{a}{2}M^2+\tfrac{b}{4}M^4$, and solves the self-consistent fluctuation equation $\Gamma_M=b\langle\delta M^2\rangle$, $\langle\delta M^2\rangle=kT/(a+b\langle\delta M^2\rangle)$ in closed form: $\Gamma_M=\big({-}a+\sqrt{a^2+4bkT}\big)/2$ (reducing to the classical $bkT/a$ limit when $a^2\gg4bkT$, and staying finite as $a\to0$). The result is capped at $\Gamma_{M,\max}=4t_{\mathrm{eff}}^2/(\pi J_{\mathrm{eff}})$ and cached per doping.

The full spin–quadrupole cross-susceptibility is also scanned over the whole BZ by `estimate_chi_SQ_q_full` (a 35×35 $q$-grid, called with `n_q=35` in the current `__main__`), producing one of the two diagnostic plots described in [Output & Diagnostics](#output--diagnostics).

### 15. Linearized Gap Equation and λ_JT_kernel

The pairing kernel on the Fermi surface, $\Gamma_{ij}=g_\Delta\cdot\sqrt{dl_i/v_{F,i}}\cdot V(k_i-k_j)\cdot\sqrt{dl_j/v_{F,j}}$, is diagonalized in `compute_pairing_kernel_and_build_cache`, which builds this together with the full vertex cache consumed elsewhere in the SCF loop; `lambda_lin_max` ($\lambda_{\max}$) is its largest eigenvalue, with eigenvector components `v_s_raw`, `v_d_raw` ($\varphi_{\max}$). The FS integration weights — the proper $dl/((2\pi)^2 v_F)$ measure with a floored $|v_F|$ — are built by `_get_fs_points` (an `RMFT_Solver` instance method, not a standalone static one) as its fourth return value, consumed throughout as `inv_vF`. The **JT-channel Rayleigh projection** $\lambda_{JT}^{\mathrm{kernel}}=\varphi_{\max}^T\,\Gamma_{JT}\,\varphi_{\max}$ (`lambda_JT_kernel`) measures how much of $\lambda_{\max}$ is carried specifically by the JT (as opposed to spin-fluctuation) component of $V(q)$ — a scalar, FS-resolved companion to the $q{=}0$ estimate `lambda_JT_sc` (§17). The same call also forms the signed bare Rayleigh quotients `lambda_s_bare = W11/ns` and `lambda_d_bare = W22/nd` on that *same* FS grid (not separate s-/d-projected grids); these are internal-only — not part of the returned cache — and feed a single diagnostic, `lambda_gain_rel`, the relative gain of the mixed $\lambda_{\max}$ over the better of the two bare channels.

### 16. The G-Matrix (`InstabilityInfo`)

The coupled SC–JT instability boundary is tracked by a 3×3 matrix in the ordered basis $[s,\ d,\ \mathrm{JT}]$, computed in `compute_G_instability` and wrapped by the `InstabilityInfo` dataclass:

```
G3 = ┌ 1 − gV_s·χ_pair,s        −√(gV_s·gV_d)·χ_pair,sd    −g_JT·√(gV_s/K_eff)·χ_SQ,s ┐
     │ −√(gV_s·gV_d)·χ_pair,sd   1 − gV_d·χ_pair,d          −g_JT·√(gV_d/K_eff)·χ_SQ,d │
     └ −g_JT·√(gV_s/K_eff)·χ_SQ,s  −g_JT·√(gV_d/K_eff)·χ_SQ,d   1 − χ_QQ/K_eff          ┘
```

evaluated in the **normal** ($\Delta=0$) state, using the self-consistent AFM magnetization $M$ as input. `InstabilityInfo.from_G3` diagonalizes this matrix and classifies the result:

| `instab_type` | Condition | Meaning |
|---|---|---|
| `stable` | $\lambda_{\min}>0$ | No instability |
| `spontaneous_JT` | $G_{22}\le 0$ | JT mode itself unstable — independent of pairing, undesired |
| `s_pairing` / `d_pairing` | $G_{11}\le0$ / $G_{33}\le0$ (JT stable) | Conventional s- or d-wave pairing instability |
| `both_pairing` | both $G_{11},G_{33}\le0$ | Both pairing channels unstable |
| `cross_channel` | $\lambda_{\min}\le0$ but all diagonals $>0$ | **SC-triggered JT** signature — the desired scenario |

Because $\chi_{SQ}\equiv0$ in the normal state by the selection rule of §2, the off-diagonal $s$–JT and $d$–JT blocks of $G_3$ are set to zero at this evaluation point — the G-matrix at $\Delta=0$ diagnoses whether the *lattice itself* and the *pairing channels* are separately stable, while the genuine SC-triggered-JT signature (a negative post-convergence Hessian eigenvalue with $G_{22}>0$ still holding) is confirmed only **post-SCF**, in the Hessian test described under the "SCF Loop" heading in [Key Algorithms](#key-algorithms). `G3[2,2] = 1 − χ_QQ/K_eff` is the quantity whose Schur complement diverges as $G_{22}\to0^+$, and it captures spontaneous-JT risk from any source, including a large `Delta_B1g_static`.

### 17. SC-JT Coexistence Window

Three distinct dimensionless λ_JT-type quantities appear at different stages of the pipeline:

| Symbol | Formula | Evaluated at | Threshold |
|---|---|---|---|
| $\lambda_{JT}^{\mathrm{norm}}$ | $\chi_{QQ}/K_{\mathrm{eff}}$ (`= 1 − InstabilityInfo.G22`) | normal state | $<1$ stable, $\ge1$ spontaneous JT |
| $\lambda_{JT}$ (bare) | $g_{JT}^2\chi_{QQ}/K_{\mathrm{bare}}$ | normal state, cheap pre-convergence estimate | $>0.05$ for a non-trivially open window |
| `lambda_JT_sc` | $g_{JT}^2\cdot\max(-\chi_\tau^{\mathrm{net}},0)/K_{\mathrm{eff}}$ | SC state, post-SCF | $>0.05$ ⇒ SC-triggered JT active |

The SC–JT window itself is bounded by

```
K_spont = g_JT² · chi_QQ                                           (must have K_lattice above this; g_JT²/Delta_CF is also logged alongside as a simpler analytic estimate, `K_spont_analytic`)
K_SC    = g_JT² · max(−chi_tau_net, 0) / _LAMBDA_JT_VIABLE          (must have K_lattice below this)
```

so viability requires $K_{\mathrm{spont}} < K_{\mathrm{lattice}} < K_{SC}$, with `K_opt = √(K_spont·K_SC)` the geometric midpoint of the window. If `K_lattice ≤ K_spont` the lattice is already spontaneously unstable regardless of SC; if `K_lattice ≥ K_SC` it is too stiff for the condensate to ever soften it into the JT regime. This window analysis is assembled in `plot_ground_state_comparison`'s reporting logic (not a dedicated `RMFT_Solver` method): `normal_stable = (K_lattice > K_spont)` and `sc_jt_active = (K_lattice < K_SC)` combine with `window_open = (K_SC > K_spont)` and the post-SCF `InstabilityInfo.full_stable` check into the overall `jt_viable` verdict — so the window is reported non-viable whenever the lattice is already unstable in the normal state, regardless of where `K_lattice` sits relative to the `K_spont`/`K_SC` bounds.

### 18. Variational Free Energy and the Cluster Decomposition

The total free energy splits, without double-counting, into an itinerant and a local piece:

```
F_total = F_bdg + F_cluster
```

This is a Luttinger–Ward/Baym–Kadanoff-style variational decomposition: `F_bdg` (`_compute_bdg_free_energy`) covers the itinerant mean-field BdG spectrum plus the condensation-energy terms $|\Delta_s|^2/(g_{\Delta,s}V_s) + |\Delta_d|^2/(g_{\Delta,d}V_d) + (K_{\mathrm{eff}}/2)Q^2$; `F_cluster` (`compute_cluster_free_energy`) covers local quantum fluctuations from an exactly diagonalized four-site (2×2 plaquette) cluster (§19). Gutzwiller factors handle kinematic Mott renormalization; the cluster ED handles irreducible-vertex renormalization of the local B₁g self-interaction only (never the itinerant susceptibility bubble itself); RPA handles the reducible ladder summation over the full BZ. These three levels are orthogonal — cluster-ED outputs a renormalized coupling (`V_irr_QQ`) that RPA then uses as an *input* to `V_JT_corr` (§14), so there is no overlap between what each layer computes.

### 19. Four-Site Plaquette Cluster: Quantum Fluctuations and Exact Vertex Extraction

Beyond the BdG mean field, a **four-site (2×2 open plaquette)** cluster, each site carrying the full 6-orbital Kramers-doublet basis (a $6^4=1296$-dimensional Hilbert space), is exactly diagonalized every SCF iteration by `compute_cluster_free_energy`. The checkerboard geometry is

```
0 --x-- 1
|       |
y       y
|       |
3 --x-- 2
```

with sites 0, 2 on sublattice A and 1, 3 on sublattice B; each site has exactly 2 intra-cluster nearest-neighbor bonds (bond sign $\eta=+1$ for $x$-bonds, $\eta=-1$ for $y$-bonds in the B₁g channel, mirroring the real-space $\cos k_x-\cos k_y$ bond weighting; the A₁g/magnetic channel is bond-direction-independent) and $Z_{\mathrm{eff}}=Z-2$ external neighbours absorbed into the mean-field Weiss embedding, so intra-cluster bonds are never double-counted. The local single-site Hamiltonians (`build_local_hamiltonian_for_bdg`) include $-\mu$, $\Delta_{CF}$, the AFM Weiss field, the JT coupling $g_{JT}\,Q\,B_{1g,\mathrm{op}}$, and — only once $Q\neq0$ — the anomalous Weiss field from `F67s_mf`, all scaled by the same cluster-averaged downfolding weight $\beta_{\mathrm{cluster}}$ used for the ZRS spectral weight, so the cluster sees the same ligand-projected physics as the BdG Hamiltonian. The cluster Hamiltonian is

```
H_cluster = Σ_site H_site  +  Σ_bonds [ J_bond_M_bare · (multi_op ⊗ multi_op) + η · J_bond_Q_bare · (B1g_op ⊗ B1g_op) ]
```

**Vertex extraction via cluster inverse susceptibilities.** Rather than a linear regression against connected correlators, the irreducible vertex is extracted the standard cluster-ED way: an exact static Lehmann susceptibility tensor of shape $(8,8)$ (2 channels — $S_z$ and $B_{1g}$ — × 4 sites) is computed for two reference spectra — `chi0_tensor` (four independent sites, no intra-cluster exchange, each still seeing the full-lattice Weiss space) and `chi_full_tensor` (the same cluster **with** intra-cluster exchange switched on, at $M=0$, no anomalous Weiss field, no JT — the appropriate *normal-state* reference for a lattice Bethe–Salpeter equation). The irreducible vertex follows from the inverse-susceptibility difference,

```
Γ_ED = χ0⁻¹ − χ_full⁻¹                    (both inverses stabilized: null eigenvalues regularized, not the physical ones)
```

computed in the full 8×8 site×channel space and only *then* projected onto the staggered (spin, weight $\tfrac12(1,-1,1,-1)$ across the 4 sites) and uniform (B₁g, weight $\tfrac12$ on all 4 sites) subspaces — projection and matrix inversion do not commute, so this ordering is essential. The resulting $2\times2$ projected vertex `V_irr` has a single quantity fed back into the RPA vertex construction (§14): `V_irr_QQ = V_irr[1,1]`, the B₁g–B₁g irreducible coupling, which additively renormalizes the bare JT pairing vertex, `V_JT_corr = V_JT + V_irr_QQ`. The same routine also returns the per-site $\langle B_{1g}\rangle$ expectation values (`b_mean`), an intra-cluster B₁g fluctuation amplitude `Q_fluct` $=\sqrt{\langle B_{1g}^2\rangle-\langle B_{1g}\rangle^2}$ averaged over the 4 sites, and the cluster free energy per site `F_per_site` (with the mean-field double-counting correction $\tfrac12 Z_{\mathrm{eff}}J_{\mathrm{bond}}M^2$ added back).

Two spectra are diagonalized in addition to the physical (JT- and anomalous-Weiss-including) cluster ground state used for the free energy itself: `evals_vertex_full` (exchange on, $M=0$, no anomalous/JT terms — the interacting vertex reference) and `evals_vertex_0` (exchange off — the bare/independent-site reference), so three $1296\times1296$ exact diagonalizations run per call.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

`_find_mu_for_density` solves $\langle n\rangle = 1-\delta$ by Newton's method using the analytic derivative $\partial n/\partial\mu=\sum_{k,n} w_k f(1-f)/kT\cdot(|u|^2+|v|^2)$ from the same BdG eigensystem, backtracking on step failure, with Brent's method as a guaranteed fallback bracket-and-bisect. Above `_MU_SC_DERIV_THRESH` total gap amplitude the analytic derivative (exact only for the pure normal-state branches) is replaced by a centered numerical derivative. The `(ev, ec)` pair from the μ-search is reused directly for the subsequent observable computation, avoiding a redundant diagonalization.

### 21. Gap Equations, Complex Phase, and the 2×2 Pairing Kernel

`VectorizedBdG.compute_gap_eq_vectorized` evaluates the gap equations over the full BZ, keeping the Fock sums **complex** rather than taking `abs(·)` before forming the new gap — because the BdG Hamiltonian is genuinely complex (SOC enters through $L_y\propto(L_+-L_-)/2i$ and $S_y$), the Nambu eigenvectors are complex at every $k$-point, and forcing a real magnitude at every iteration would erase the physical relative phase between $\Delta_s$ and $\Delta_d$ and destabilize convergence. A real, FS-averaged 2×2 pairing kernel

```
K_pair = [[ K11, K12 ], [ K12, K22 ]]      (s/d basis; K11 uses the JT-only vertex, K22 the full RPA vertex)
```

is built inline from the already-available FS grids at vertex-cache rebuild time; its dominant eigenvector `(v_s, v_d)` gives the SCF the optimal s/d hybridization direction, blended into the fixed-point gap update with weight `_ALPHA_MIX_2X2` — but only the real relative *magnitude* ratio is taken from `K_pair`; the actual complex phases `Delta_s_new/|Delta_s_new|`, `Delta_d_new/|Delta_d_new|` are always re-applied after blending, so the self-consistent phase is never silently overwritten. The same phase-freezing logic is used in `compute_hessian`, which fixes the converged phases before taking finite-difference probes. Alongside `(Delta_s_out, Delta_d_out)` the routine also returns the diagnostic, non-fed-back $\Gamma_6$–$\Gamma_{7b}$ raw estimates of §8 in the same call.

### 22. Incommensurate AFM Nesting Check

Because the BdG Hamiltonian is fixed to commensurate AFM ordering at $Q_{AFM}=(\pi,\pi)$, `_scan_incommensurate_nesting` separately checks whether the normal-state spin susceptibility $\chi_{SS}$ would actually prefer a nearby incommensurate wavevector $q^*=(\pi,\pi-\delta q)$, scanning $\delta q\in[0,0.15\pi]$ at the converged $(M,Q,\mu)$. If the scan finds $\chi_{SS}(q^*)/\chi_{SS}(\pi,\pi)$ meaningfully above 1 (specifically, a maximum away from $\delta q=0$ beyond a small tolerance), `solve_self_consistent` automatically retries once with a softened AFM seed ($M\to0.85\,M$), guarded by the `_ic_retry` flag to prevent infinite recursion. This does not change the ordering wavevector used in the Hamiltonian itself — it only flags, and mildly compensates for, the possibility that the true instability sits away from $(\pi,\pi)$.

### 23. Temperature-Dependent Tc Estimates

Three independent estimates target different aspects of the transition, deliberately not sharing a single label:

- **Tc₁ — Allen–Dynes/McMillan-type spin-fluctuation formula:** $T_{c1}=(\omega_{SF}/D)\cdot\exp(-N\cdot(1+\lambda_{\max})/\lambda_{\max})$, with constants $D=$ `_MAD_DENOM` = 1.13, $N=$ `_MAD_NUM` = 1.04, $\omega_{SF}=J_{\mathrm{eff}}$ (paramagnon bandwidth), and $\lambda_{\max}$ from the linearized gap equation at the reference doping — a fast analytic estimate, not a full temperature scan.
- **Tc₂ — λ(T)=1 crossing:** `compute_lambda_vs_T` re-runs the linearized gap equation at each temperature on a **Δ=0, self-consistently relaxed normal-state** background (using `estimate_M0` as a warm-start rather than the converged $T{=}0$ SC value, which would otherwise artificially bias the bands away from the crossing); $T_{c2}$ is where $\lambda_{\max}(T)=1$. Non-monotone $\lambda(T)$ is detected and all crossings are logged.
- **Tc₃ — thermodynamic, first-order-aware:** `compute_Tc_thermodynamic` performs a single upward-heating temperature scan, warm-started from the converged $T\approx0$ SC+JT basin, comparing $F_{SC}$ against a separately relaxed normal-state free energy at every point. Because the effective Landau potential $F_{\mathrm{eff}}(\Delta) = a(T)\Delta^2 + [b - \gamma^2/(2K_{\mathrm{eff}})]\Delta^4 + \dots$ can have a negative quartic coefficient here, the transition can be genuinely first-order, and a naive cooling-from-$\Delta{\approx}0$ scan (which only finds the spinodal) can badly underestimate $T_c$. The routine returns both the thermodynamic crossing `Tc` and the spinodal collapse `Tc_spinodal`, the transition order, the gap jump `Delta_jump`, and — for near-second-order cases (`D_spinodal/Δ₀ < 0.15`) — a Ginzburg–Landau-refined spinodal from fitting $\Delta^2(T)=a(T-T_c)$ to points with $|\Delta|>2\,\mathrm{meV}$.

`compute_Tc_by_gap_suppression` (cooling-only spinodal search) is retained as an independent cross-check. The $2\Delta_0/k_BT_c$ ratio reported in the Tc block is computed against $T_{c3}$, the most physically complete of the three.

---

## Model Architecture

```
ModelParams  (dataclass, __post_init__ runs the SOC+CF diagonalization)
    ├── Primary inputs:  t_pd, t_prime_ratio, t_dprime_ratio, U_dd, lambda_soc, Delta_tetra,
    │                    g_JT, K_lattice, lambda_hop, g_Eg2, K_lattice_Eg2, Delta_B1g_static,
    │                    hybrid_scale, Upp_ratio_bare, Delta_CT, Z, kT, tol
    ├── Derived scalars: Delta_CF, g7split, t0, t_prime, t_dprime, U_pp, J_pdct, p_7
    ├── Derived arrays:  sz_op (exact ⟨Sz⟩ per Kramers partner, 6 components), multi_op, B1g_op,
    │                    B1g_offdiag, Eg2_op, _w6_xz/_yz/_xy, _w7_xz/_yz/_xy, _w7b_xz/_yz/_xy,
    │                    Tx_A_xz/_yz/_xy (orbital-selective hopping projectors, sum to I₆)
    ├── Grid objects:    k_points, k_weights, N_k, shift_table (_NK×_NK×N_k int32 cyclic shift
    │                    table for arbitrary q)
    └── Methods:         estimate_M0(), get_gutzwiller_factors(), exchange_channels(),
                         effective_hopping_anisotropic(), hopping_matrices(),
                         hopping_matrices_dQ(), wave_function_weight()

InstabilityInfo  (dataclass wrapping a 3×3 G-matrix eigendecomposition)
    ├── Fields:   G11, G33, G22, G_sd, G_sJT, G_dJT, eigenvalues, eigenvectors, lambda_min, evec_min
    ├── Booleans: jt_stable, s_stable, d_stable, full_stable
    └── Classifiers: instab_type, instab_dir, dominant_channel, severity,
                     weight_for_score, weight_for_log, log_summary()
                     from_G3(G3) — classmethod constructor from a raw 3×3 array

_SolveState  (dataclass, mutable per-SCF-run state — never stored on self)
    ├── V_d_ema: Optional[float]         # persistent V_d sign-flip EMA
    └── _ema_kick_pending: bool          # doubles blend weight for one iter after a kick

RMFT_Solver
    ├── Initialization: __init__, _rebuild_orbital_operators, _get_vbdg, _get_chi0_norm_cache,
    │                   _reset_transient_state, _shallow_clone, _clone_solver_at_T, _full_rebuild
    ├── JT rigidity:    _calc_dHdQ (∂H/∂Q in the band basis), compute_K_eff_full (exchange
    │                   contribution to the B₁g/Eg,2 stiffness, §11)
    ├── Susceptibilities: B1g_expectation, Eg2_expectation, _compute_chi_tau,
    │                   _chi_QQ_matrix_elements, estimate_chi_SQ_q_full, _compute_nambu_kernel,
    │                   _compute_nambu_susceptibility, _diamagnetic_QQ_term,
    │                   get_susceptibilities_sc (Δ=0 call also covers the normal state)
    ├── RPA vertex:     _rpa_det, _rpa_vertex, _moriya_gamma_landau, _make_vertex_params
    ├── Gap equation:   compute_pairing_kernel_and_build_cache (linearized 2×2 kernel + FS
    │                   vertex cache, using the module-level _unique_q_pairs helper and FS
    │                   integration weights from _get_fs_points),
    │                   scf_gap_diagnostics (coherence lengths, gap-ratio-relevant quantities)
    ├── Local H / μ:    build_local_hamiltonian_for_bdg, _find_mu_for_density, _compute_F67_singlet
    ├── Free energy:    compute_bdg_free_energy (_compute_bdg_free_energy), compute_cluster_free_energy
    │                   (four-site plaquette ED + Γ_ED vertex extraction, §19)
    ├── SCF machinery:  _scf_jacobi_kick, _vertex_matrix_at_Q, _pairing_strengths,
    │                   _classify_scf_dynamics, _anderson_mix, _mix, _project_kick_from_hessian
    ├── Main solve:     solve_self_consistent   ← the ~800-line Anderson-accelerated fixed point
    ├── Post-hoc:       _scan_incommensurate_nesting, compute_dF_dM_and_d2F (compute_dF_dM_channels_and_hessian),
    │                   compute_hessian
    ├── Tc:             compute_Tc_by_gap_suppression, compute_Tc_thermodynamic,
    │                   compute_lambda_vs_T
    ├── Diagnostics:    compute_G_instability, _get_fs_points, compare_cluster_vs_bdg,
    │                   diagnose_doublet_mixing, diagnose_simulated_sc_state,
    │                   refine_M_normal_state, estimate_gutzwiller_factors_occupation_based
    └── Occupation:     _compute_orbital_densities

VectorizedBdG   (thin batched-LAPACK wrapper bound to one RMFT_Solver)
    ├── _build_H_stack                     builds & Hermitizes the (N_k, 24, 24) BdG stack, including
    │                                       the optional (default-zero) Γ6–Γ7b pairing terms
    ├── compute_channel_staggered_magnetizations   → per-channel (Γ6, Γ7a, Γ7b) M_stag
    └── compute_gap_eq_vectorized           → (Delta_s_out, Delta_d_out, Delta_s7b_diag,
                                               Delta_d7b_diag, vertex_cache)

plot_ground_state_comparison(results, labels=None)   — standalone function, not a class method;
    builds the 2×2 "ref vs. normal vs. SC_Q0" free-energy/order-parameter comparison figure
    described in Installation & Usage / Output & Diagnostics.
```

---

## Key Algorithms

### SCF Loop (`solve_self_consistent`)

An Anderson(5)-accelerated fixed point over $(M, Q, \Delta_s, \Delta_d, \mu)$, per iteration:

1. Build and diagonalize the 24×24 BdG stack for the current $(M,Q,\Delta_s,\Delta_d,\mu)$.
2. If SC+JT are both active ($|\Delta_s|+|\Delta_d|$ above a small threshold), compute the Gorkov Γ₆–Γ₇ₐ singlet amplitude `F67s_mf` and inject it as an anomalous Weiss field, then rebuild the BdG cache — this is the joint-$(\Delta,Q)$ activation loop of §2.
3. Update `K_eff` (and `K_eff_Eg2`) via `compute_K_eff_full` — on iteration 0, or when $|\Delta Q|$ exceeds threshold, or every few iterations once $|\Delta M|$ has moved enough — rather than every single step, since the rigidity computation is comparatively expensive.
4. Solve the gap equations via the RPA vertex fixed point (§21 above); blend in the 2×2 pairing-kernel eigenvector direction with weight `_ALPHA_MIX_2X2 = 0.56` to prevent one channel from artificially locking out the other.
5. Diagonalize the four-site plaquette cluster Hamiltonian (§19) and extract the irreducible B₁g vertex `V_irr_QQ`, feeding `V_JT_corr = V_JT + V_irr_QQ` into the next iteration's RPA vertex construction (§14); `J_eff` itself always comes directly from the analytic Gutzwiller/exact-diagonalization superexchange result (§4–5), not from the cluster vertex.
6. Newton step for `M` (Levenberg–Marquardt-damped, trust-region-limited), blended with the linearly mixed BdG fixed point — `M` is deliberately **excluded** from the Anderson history and updated by this separate Newton/blend rule.
7. Adaptive Hellmann–Feynman update for `Q`: $Q_{\mathrm{out}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$, injected into the Anderson vector when the implied displacement is significant, on iteration 0, or on a periodic safety heartbeat (`_Q_UPDATE_PERIOD`); otherwise left untouched, consistent with the lattice's adiabatic timescale.
8. Anderson(5) mixing is applied jointly to $(Q,\ |\Delta_s|,\ |\Delta_d|)$ — a Tikhonov-regularized least-squares solve over the last 5 residuals; $\mu$ is then re-solved exactly (Newton+Brent) at the freshly mixed point rather than carried as an imperfectly converged fast variable, so its residual error cannot leak into the Anderson history.
9. Adaptive mixing rate: $\alpha_{\mathrm{eff}}=\alpha_0/(1+\Lambda_{\mathrm{inst}})$, where $\Lambda_{\mathrm{inst}}$ is an EMA of the worst current instability indicator ($\lambda_{\mathrm{pair}}$, $\lambda_{JT}$, $J\chi_{SS}$); $\alpha$ is halved and the Anderson history reset on divergence, and a **limit-cycle detector** (relative std of $|\Delta|$ over the last `_CYCLE_WINDOW` iterations $> $ `_CYCLE_THRESHOLD`) damps $\alpha$ and resets history if the SCF is oscillating rather than converging.

Convergence requires $\max(|\Delta M|,|\Delta Q|,|\Delta\Delta_s|,|\Delta\Delta_d|)<$ `tol` and density error $<10\cdot$`tol`. After convergence (or exhausting `_MAX_ITER = 700`), `_classify_scf_dynamics` labels the trajectory as `converging`, `limit_cycle`, `first_order_jump`, `hysteretic`, or `stagnating` from the $|\Delta|$ history; the first-order-like classes trigger a multi-seed restart (several initial conditions, lowest free energy wins), consistent with the theory's first-order-transition expectation (see Physical Hypothesis). Post-convergence the solver runs the 3×3 Hessian test, the coherence-length/gap-symmetry diagnostics, the incommensurate-nesting check (§22), and assembles the full result dictionary consumed by the diagnostics block described in [Output & Diagnostics](#output--diagnostics).

### Vectorized BdG, Buffer Reuse, and the χ₀(q) Permutation Trick

`VectorizedBdG._build_H_stack` assembles the entire $(N_k,24,24)$ Hamiltonian stack with vectorized NumPy operations and diagonalizes it in a single `np.linalg.eigh` call per SCF iteration, reusing a pre-allocated `out=` buffer to avoid repeated allocation across hundreds of iterations; Hermiticity is enforced after assembly. The per-iteration eigensystem `(ev, ec)` is computed once and shared by observable computation, both pairing-channel gap equations, and the analytic $\partial F/\partial M$ (below).

The $q$-loop inside the RPA vertex construction never re-diagonalizes: the uniform k-grid (`endpoint=False`) is built in `ModelParams.__post_init__` so that for any $q=(n_x,n_y)\cdot2\pi/\mathrm{\_NK}$, the $k+q$ grid is exactly a cyclic permutation of the $k$-grid. A precomputed `shift_table[nx, ny]` (shape `(_NK, _NK, N_k)`, `int32`) turns "shift by $q$" into a free index reorder,

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]     # index reorder — no extra LAPACK call
```

reusing the *same* $\Delta=0$ eigensystem for every $q$-point in one vertex-cache rebuild. `_get_chi0_norm_cache` additionally memoizes this normal-state $(E_k,V_k)$ across separate calls (susceptibilities, rigidity, incommensurate-nesting scan) that fall within the same iteration, keyed on $(M,Q,\mu,g_t,g_J,\delta)$ with independent tolerances tightened around the physically sensitive ones — the $M$ and $Q$ tolerances are the *same* `_M_THR_REL`/`_Q_THR_REL` thresholds used for RPA vertex-cache invalidation below, while $\mu$, $g_t$, $g_J$, and the doping are checked at $10^{-4}$, $10^{-4}$, $10^{-4}$, and $10^{-6}$ respectively.

### Vertex Cache Invalidation

The RPA vertex cache is rebuilt when $M$ moves by more than an adaptive threshold scaled to `_M_THR_REL = 0.01` (finer near the QCP, where the vertex is most sensitive), when $Q$ moves by more than `_Q_THR_REL = 0.016` (1.6%) of `lambda_hop`, when doping changes by more than 0.005, or unconditionally if the cache was not built from the normal state. There is no Δ-based invalidation — the vertex is *always* built from $\Delta=0$ by construction (§14). The cache stores the RPA determinant (both `det_q0` and `det_afm`), FS geometry (`fs_pts`, `vF_arr`, and separate s-channel FS arrays), the $q$-resolved diagnostics of §14, and the 2×2 pairing-kernel results, so repeated calls within one iteration reuse the same Fermi-surface sampling.

### Limit-Cycle Detection

Independent of the adaptive-$\alpha$ mechanism below, a dedicated oscillation check monitors $|\Delta|$ over a rolling window of `_CYCLE_WINDOW = 20` iterations; when the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.25`, the mixing rate is cut by `_CYCLE_DAMP_FAC = 0.45` and the Anderson history reset, which specifically targets the strongly nonlinear regime near the JT-activation onset where the $(Q,\Delta)$ feedback is most prone to overshoot.

### Anderson Mixing and the Jacobi Kick

Before the main SCF loop, `_scf_jacobi_kick` linearizes the coupled $(Q,\Delta)$ map analytically — via a $2\times2$ Jacobian $J$ with entries $J_{11}=A$, $J_{12}=B$, $J_{21}=C$, $J_{22}=0$, built from the linearized pairing eigenvalue, the normal-state $G_{22}$-type JT stiffness, and a gap-induced $B_{1g}$-response coupling — to estimate its leading eigenvalue $\lambda_+=\tfrac12\big[A+\sqrt{A^2+4BC}\big]$ (classified subcritical/critical/supercritical against thresholds 0.7/1.4), and uses this to choose the initial seed for $(M,Q,\Delta_s,\Delta_d)$ and the starting mixing rate $\alpha$ — this lands the iteration in the basin of the physically correct fixed point rather than an arbitrary starting guess. The Anderson solve itself uses a Tikhonov-regularized (`_ANDERSON_TIKHONOV = 1e-8`) normal-equation solve with a trust-region cap (`_ANDERSON_TRUST = 2.4`×) on the step size relative to simple mixing.

### Adaptive Q Update

$Q_{\mathrm{out}}^{\mathrm{raw}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ is evaluated at **every** iteration, since $\langle B_{1g}\rangle$ is already available at zero extra cost from the same observable pass. It is only **injected into the Anderson vector**, however, when at least one of three conditions holds: the implied displacement $|Q_{\mathrm{out}}^{\mathrm{raw}}-Q|$ exceeds `_Q_THR_REL·lambda_hop`; it is the first iteration (seed); or a periodic safety heartbeat fires (`iteration % _Q_UPDATE_PERIOD == 0`, every 3 iterations). Otherwise $Q_{\mathrm{out}}=Q$ exactly — the Anderson residual for $Q$ is zero and the mixer leaves it untouched, respecting the lattice's slower (adiabatic) timescale relative to the electronic degrees of freedom without imposing a rigid blind period.

### Thread-Safety and Clone Protocol

The current `__main__` runs its three parallel SCF tasks (§ Installation & Usage) as three independently constructed `RMFT_Solver` instances rather than clones of a single solver, but the underlying clone protocol remains available for any code that mutates parameters mid-run:

```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s._full_rebuild()
```

`_full_rebuild()` is the single canonical post-mutation refresh: it re-runs `p.__post_init__()` (SOC+CF diagonalization), updates the bare stiffness `_K_bare`, rebuilds every orbital operator (`B1g_op`, `B1g_24`, `Eg2_op`, `Eg2_24`, `sz_op`, `multi_op`), and resets all transient caches (`_reset_transient_state`). Each clone owns its own `VectorizedBdG` and its own `_H_stack` buffer, so concurrent workers never alias each other's memory. At import time the module pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to prevent the BLAS backend from oversubscribing threads underneath an outer `ThreadPoolExecutor`.

### Analytic ∂F/∂M and ∂²F/∂M² from a Single Diagonalization

The Newton step for $M$ (SCF Loop step 6) needs both the free-energy gradient and curvature with respect to $M$, but these are obtained **without any extra diagonalization**: `compute_dF_dM_channels_and_hessian` builds $\partial H/\partial M$ analytically (only the diagonal $J_{A1g}$ Weiss-field term contributes; the off-diagonal $J_{B1g}$ term has no direct diagonal piece with respect to $M$, though the anomalous `F67s_mf` term it carries must still be present in $H$ so the eigenvectors reflect the correct inter-band structure) and then applies first- and second-order perturbation theory directly to the **already-computed** eigenvalues/eigenvectors $(ev,ec)$ of the converged BdG stack:

```
∂F/∂M   =  ⟨n|∂H/∂M|n⟩ weighted by f(E_n)                                    (Hellmann–Feynman, diagonal)
∂²F/∂M² =  −Σ_{n} f'(E_n)·|⟨n|∂H/∂M|n⟩|²  +  Σ_{n≠m} [f(E_n)−f(E_m)]/(E_m−E_n) · |⟨n|∂H/∂M|m⟩|²
```

the second (off-diagonal, Kubo-like) term using a numerically safe $\tfrac{\Delta f}{\Delta E}\to -f'(E)$ limit at near-degenerate $E_n\approx E_m$. Because both derivatives come from the one BdG eigensystem already sitting in memory, this replaces what would otherwise be 2–3 additional full diagonalizations per SCF iteration with O(1) extra tensor contractions.

### Four-Site Plaquette Cluster ED (Vertex Renormalization)

See §19 for the full construction. Algorithmically, three exact diagonalizations of the $1296\times1296$ ($6^4$) cluster Hamiltonian are performed per call to `compute_cluster_free_energy` — the physical (JT- and anomalous-Weiss-including) state used for `F_per_site`/`b_mean`/`Q_fluct`, and two normal-state reference spectra (exchange-on and exchange-off) used purely to build the $8\times8$ site×channel susceptibility tensors from which the irreducible vertex `Γ_ED = χ0_ED⁻¹ − χ_full_ED⁻¹` is extracted by a numerically stabilized symmetric matrix inverse (eigenvalues below a floor relative to the largest are treated as null and zeroed in the inverse, rather than blown up).

---

## Parameters

All energies in **eV**, lengths in **Å**. Defaults below are the values set in the `__main__` block.

### Primary Inputs (`ModelParams`)

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | $t_{pd}$ | 0.490 eV | $p$–$d$ hybridization integral; the single primary hopping input ($t_0=t_{pd}^2/\Delta_{CT}$ is derived) |
| `t_prime_ratio` | — | −0.06 | 2nd-neighbor (diagonal) hopping ratio; $t'=$ this $\times\,t_0$ |
| `t_dprime_ratio` | — | 0.03 | 3rd-neighbor (axial) hopping ratio; $t''=$ this $\times\,t_0$ |
| `U_dd` | $U_{dd}$ | 3.000 eV | On-site Hubbard repulsion — now a **primary** input, not derived from a dimensionless ratio |
| `lambda_soc` | $\lambda_{SOC}$ | 0.042 eV | Atomic SOC constant on the $t_{2g}$ shell; sets the Γ₆–Γ₇ splitting together with `Delta_tetra` |
| `Delta_tetra` | $\Delta_{\mathrm{tetra}}$ | −0.072 eV | Axial (tetragonal) crystal field, $\Delta_{\mathrm{tetra}}\cdot L_z^2$; negative = $z$-axis compression |
| `g_JT` | $g_{JT}$ | 0.330 eV/Å | B₁g electron–phonon (JT) coupling |
| `K_lattice` | $K$ | 2.800 eV/Å² | Bare B₁g phonon spring constant; `K_eff` is computed at runtime |
| `lambda_hop` | $\lambda_{\mathrm{hop}}$ | 1.100 Å | Hopping-anisotropy decay length: $t(Q)=t_0\exp(\pm Q/\lambda_{\mathrm{hop}})$ |
| `g_Eg2` | $g_{Eg2}$ | 0.100 eV/Å | Eg,2-channel electron–phonon coupling (§7) |
| `K_lattice_Eg2` | $K_{Eg2}$ | 6.500 eV/Å² | Bare Eg,2 phonon spring constant |
| `Delta_CT` | $\Delta_{CT}$ | 2.900 eV | Charge-transfer gap |
| `Delta_B1g_static` | $\Delta_{\mathrm{ip}}$ | −0.011 eV | Static in-plane crystal field, $(L_x^2-L_y^2)$; drives the D₄h→D₂h crossover (§1, §3) |
| `hybrid_scale` | — | 6.000 | Downfolding coordination factor entering the ZRS spectral weight $\beta^2(k)$ (§6, §9) |
| `Upp_ratio_bare` | — | 0.400 | Bare (unhybridized) $U_{pp}/U_{dd}$ ratio entering $U_{pp}$ (§4) |
| `Z` | $Z$ | 4 | Coordination number |
| `kT` | $k_BT$ | 0.005 eV | Temperature ($\approx$ 58 K) |
| `tol` | — | $10^{-4}$ | SCF convergence threshold |

### Derived Quantities (from `__post_init__`)

| Quantity | Origin | Description |
|---|---|---|
| `Delta_CF`, `g7split` | SOC+CF diagonalization | Γ₆–Γ₇ₐ gap and Γ₇ₐ–Γ₇ᵦ internal splitting — **not** free parameters |
| `sz_op` | exact $S_z$ diagonalization in each Kramers doublet | AFM/spin-vertex weights, 6 components $[sz_{6\uparrow},sz_{6\downarrow},sz_{7a\uparrow},sz_{7a\downarrow},sz_{7b\uparrow},sz_{7b\downarrow}]$ |
| `multi_op` | built from `sz_op` | Effective multipolar spin operator shared by the cluster and BdG solvers |
| `p_7` | Γ₇ admixture in the Γ₆ eigenvectors | Interpolates `g_Delta_d` between `g_t` and `g_J` |
| `_w6_xz/_yz/_xy`, `_w7_...`, `_w7b_...` | eigenvector projections | $d_{xz}/d_{yz}/d_{xy}$ orbital weights feeding $\eta_J(Q)$ and the `Tx_A_*`/`Ty_A_*` hopping projectors |
| `Tx_A_xz/_yz/_xy` | orbital-character projectors | Rigorous orbital-selective hopping building blocks (§6); sum to $I_6$ |
| `t0`, `t_prime`, `t_dprime` | $t_{pd}^2/\Delta_{CT}$ and the `*_ratio` inputs | Effective nearest/2nd/3rd-neighbor $dd$ hopping |
| `U_pp`, `J_pdct` | §4 | Ligand hole–hole repulsion and the exact-diagonalization ZSA superexchange scale |
| `k_points`, `k_weights`, `N_k`, `shift_table` | $N_k=$ `_NK`² uniform grid | k-space infrastructure, including the cyclic shift table used for arbitrary-$q$ Lindhard sums |

`B1g_op`, `B1g_offdiag`, `Eg2_op` are also set on `ModelParams` in `__post_init__` (§3, §7); the corresponding 24×24 Nambu lifts `B1g_24`, `Eg2_24`, plus `Sz_nambu` and the per-channel `Sz_stag_nambu_channels`, are built on the `RMFT_Solver` instance by `_rebuild_orbital_operators`, along with `phi_k = cos(kx) − cos(ky)` on the SCF grid.

### Module-Level Constants

The source file documents essentially every numerical-methods constant inline, each with its own physical or numerical justification — that block remains the single source of truth. The tables below reproduce the values relevant to interpreting solver behavior and output, organized by function, checked directly against the current module-level constant block.

**Grid, iteration budget, general safety**

| Constant | Value | Role |
|---|---|---|
| `_NK` | 52 | k-points per direction (even, for commensurate $q_{AFM}=(\pi,\pi)$) |
| `_MAX_ITER` / `_MIN_ITER` | 700 / 4 | SCF iteration ceiling / floor before a convergence check is even attempted |
| `_MIXING` | 0.06 | Base Anderson mixing weight |
| `_N_ORB` / `_N_BDG` | 6 / 24 | Orbital flavors (full Γ₆⊕Γ₇ₐ⊕Γ₇ᵦ manifold) / BdG Nambu dimension ($4\times$`_N_ORB`) |
| `_N_CHANNELS` / `_CLUSTER_SIZE` / `_N_CLUSTER` | 3 / 4 / $6^4{=}1296$ | Channel resolution (Γ₆,Γ₇ₐ,Γ₇ᵦ) / sites in the plaquette cluster ED / cluster Hilbert-space dimension |
| `_MATH_EPS` | $10^{-9}$ | General division-by-zero guard |
| `_LINDHARD_CHUNK` | 128 | k-point batch size in the `opt_einsum` Lindhard loops |
| `_BZ_NORM` | $(2\pi)^2$ | BZ-area normalization in the FS arc-length measure $dl/((2\pi)^2 v_F)$ |
| `_Q_UNIQUE_SCALE` / `_PI_INT` | $10^5$ / 314159 | Integer scaling used to hash unique $q$-pairs without floating-point collisions |

**Unit conversion & Gutzwiller prefactors**

| Constant | Value | Role |
|---|---|---|
| `_EV_TO_K` | 11604.518 K/eV | $1/k_B$ |
| `_GW_G_J_PREFACTOR` | 4.0 | Numerator in $g_J=4/(1+\delta)^2$ (slave-boson / Kotliar–Ruckenstein, half-filling limit) |
| `_GW_G_T_NUMERATOR` | 2.0 | Numerator in $g_t=2\delta/(1+\delta)$ |

**AFM Newton solver ($M$-step control)**

| Constant | Value | Role |
|---|---|---|
| `_MU_LM` | 3.0 | Levenberg–Marquardt floor for the $M$ Newton step |
| `_ALPHA_HF` | 0.31 | Newton-vs-BdG-fixpoint blend weight for $M$ |
| `_TR_M_STEP_MAX` / `_TR_M_STEP_MIN_FLOOR` | 0.1 / $10^{-3}$ | Trust-region cap / absolute floor on $|\Delta M|$ per step |
| `_M_STEP_FLOOR_REL` / `_M_STEP_FLOOR_ABS` / `_M_STEP_FLOOR_M_MIN` | 0.005 / 0.002 / 0.010 | Step floor $=\max($ `_M_STEP_FLOOR_REL` $\cdot|M|,$ `_M_STEP_FLOOR_ABS`$)$, referenced against $\max(|M|,$ `_M_STEP_FLOOR_M_MIN`$)$ |
| `_M_J_EFF_FLOOR_FRAC` | 0.20 | QCP guard: $J_{\mathrm{eff}}$ floored at this fraction of $t_{\mathrm{eff}}$ to prevent $\Delta M\propto1/J_{\mathrm{eff}}\to\infty$ |

**Q Newton solver step control**

| Constant | Value | Role |
|---|---|---|
| `_Q_LM_FRAC` | 0.08 | Q-channel LM floor, as a fraction of the bare stiffness `_K_bare` |
| `_TR_Q_STEP_FRAC` | 0.10 | Trust-region cap on $|Q_{\mathrm{out,raw}}-Q|$, as a fraction of `lambda_hop` |
| `_TR_Q_STEP_MIN_FLOOR` | $10^{-4}$ Å | Absolute minimum $Q$ step, preventing total freeze near the JT QCP |

**Saddle-escape / Jacobi-kick seeding**

| Constant | Value | Role |
|---|---|---|
| `_MODE_FRAC_DOMINANT` / `_MODE_FRAC_MIXED` | 0.60 / 0.30 | Thresholds classifying a pure-channel vs. mixed SC-triggered-JT mode |
| `_MODE_PULL_FRAC` | 0.30 | Fraction of $(M-M_{\mathrm{phys,est}})$ used as the kick pull in pure-SC/SC-JT mode |
| `_KICK_BASE_FRACTION` | 0.05 | Base trust-region step-size fraction for the seed kick |
| `_KICK_M_EXCESS_CTR` / `_KICK_JCHI_EXCESS_CTR` | 0.70 / 0.70 | Sigmoid centers for $M$-kick / $J\chi_{SS}$-excess overshoot suppression |
| `_KICK_REDUCTION_AMP` | 3.88 | $M_{\mathrm{kick}}\times(1-\text{this}\times\text{excess})$ |
| `_KICK_BOOST_Q` | 0.01 | $Q$-kick boost |
| `_KICK_M_CLIP_LO` / `_KICK_M_CLIP_HI` | 0.02 / 0.9 | Hard clips on $M_{\mathrm{kick}}$ |
| `_KICK_DELTA_MAX_FRAC` | 0.4 | Maximum seed gap as a fraction of the effective hopping scale $t_{\mathrm{eff}}$ |
| `_KICK_MIXING_FLOOR` / `_KICK_MIXING_SCALE` | 0.004 / 4.0 | Minimum kick mixing weight / damping scale in $\alpha=$`_MIXING`$/(1+\text{scale}\cdot\log1p(\lambda_+))$ |
| `_M0_WARMSTART_MIN` | 0.1 | $|M|$ below this is treated as no genuine warm start |
| `_EARLY_KICK_BASE` | 0.01 | Base step fraction in the coupled seeding space |

`estimate_M0` itself (the AFM order-parameter warm start) is now a plain self-consistent Stoner iteration, $M\leftarrow\tanh(\mathrm{stoner}\cdot M)$ to fixed point, clipped to $[0,$`_KICK_M_CLIP_HI`$]$ — there is no separate empirical-prior blend or the associated `_M0_*` constant family.

**Chemical potential (Newton + Brent)**

| Constant | Value | Role |
|---|---|---|
| `_DEN_DERIV_FLOOR` | $10^{-12}$ | Floor on $\partial n/\partial\mu$ |
| `_BRENTQ_TOL` | $10^{-6}$ | Brent bracketing tolerance |
| `_MU_NEWTON_MAXIT` / `_MU_BACKTRACK_MAX` / `_MU_BACKTRACK_FLOOR` | 20 / 6 / 0.05 | Newton iteration budget / max step-halvings / minimum backtrack damping before falling back to Brent |
| `_MU_DENSITY_TOL` | $10^{-8}$ | $|n(\mu)-n_{\mathrm{target}}|$ convergence tolerance |
| `_MU_SC_DERIV_THRESH` | $10^{-4}$ eV | Gap amplitude above which the analytic $\partial n/\partial\mu$ (exact only at $\Delta=0$) is replaced by a centered numeric derivative |

**Lindhard broadening & Fermi-surface sampling**

| Constant | Value | Role |
|---|---|---|
| `_ETA_T_FRAC` | 0.10 | Normal-state broadening $\eta=$ this $\times\,kT$ |
| `_ETA_DELTA_FRAC` | 0.02 | SC-state broadening increment $\propto|\Delta|$ |
| `_ETA_GRID_FLOOR` | 0.001 | Broadening floor (units of $t_0$), guards k-grid aliasing |
| `_FERMI_ARG_CLIP` | 100.0 | Numerical clip in $f(E)$ |
| `_FD_MASK_DF` / `_FD_MASK_DE` / `_FD_MASK_DE8` | $10^{-12}$ / $10^{-6}$ / $10^{-8}$ | Degenerate-denominator masks in the $\chi_0$ Lehmann sums (the tightest, `_FD_MASK_DE8`, is used in the $\partial^2F/\partial M^2$ off-diagonal term) |
| `_VF_FLOOR` / `_VF_FLOOR_TIGHT` / `_VF_FLOOR_REL_FRAC` | $10^{-4}$ / $10^{-5}$ / 0.05 | Fermi-velocity floors (the tighter one guards the $dl/v_F$ arc-length weight specifically; the relative floor scales with the FS-median $|v_F|$) |
| `_N_FS` | 130 | Fermi-surface k-points sampled in the vertex $q$-loop |
| `_FS_SAMPLING` / `_FS_THERMAL_THRESHOLD` | 4.4 / 0.0025 | Thermal window (in units of $kT$) around $E_F$ for FS selection / minimum relative thermal weight kept |
| `_FS_CACHE_TOL` | $10^{-3}$ | Parameter-change tolerance for FS-point cache invalidation |
| `_NODAL_REGION_PCTL` | 25 | Percentile split (upper/lower 25%) for nodal/antinodal FS decomposition |
| `_PHI_D_FLOOR` | $10^{-3}$ | Minimum $\varphi_d^{\max}$ to enable the nodal/antinodal split at all |

**RPA vertex & QCP tracking**

| Constant | Value | Role |
|---|---|---|
| `_RPA_BW_FACTOR` | 8.0 | Tight-binding bandwidth estimate $=8t$ |
| `_RPA_V_CAP_ALPHA` | 2.2 | Headroom multiplier for the dynamic vertex cap $V_{\mathrm{cap}}$ |
| `_RPA_DET_WARN` | 0.11 | $|\det_{\mathrm{afm}}|$ below this ⇒ QCP-proximity warning, feeds adaptive mixing |
| `_RPA_QCP_PENALTY` | 0.40 | Mixing-rate reduction per unit $|\det_{\mathrm{afm}}|<0$ past the QCP |
| `_DET_AFM_FLOOR` | 0.5 | Default `det_afm` when no vertex cache exists yet |
| `_DK_CORR_CAP_MULT` | 1.0 | Direct cap on $|dK_{\mathrm{corr}}|$ relative to the bare stiffness $K_{\mathrm{bare}}$ |
| `_DET_DEPTH_CAP` / `_DET_JUMP_HALF_SCALE` / `_JUMP_CAP_FLOOR` | 5.0 / 0.5 / 1.05 | Past-QCP gap-jump cap: exponential suppression depth cap / decay rate / minimum allowed cap |
| `_DET_SIGN_FLIP_SCALE` | 0.05 | $|\det_{\mathrm{afm}}|$ sigmoid midpoint for the $V_d$ sign-flip EMA guard |
| `_EMA_SIGN_FLIP_W_MIN` / `_EMA_SIGN_FLIP_SLOPE` | 0.20 / 6.0 | Minimum blend weight / sigmoid steepness in the sign-flip guard |
| `_V_PREV_SIGN_FLOOR` | $10^{-6}$ | $|V_{d,\mathrm{prev}}|$ below this is treated as zero (sign-flip check skipped) |
| `_VMAT_LOW_VAR_FRAC` | 0.10 | $\mathrm{std}(V)/|\mathrm{mean}(V)|$ below this ⇒ `⚠low-var` flag |
| `_VERTEX_DIAG_MIN_FS` | 10 | Minimum FS points required before vertex-structure diagnostics are considered reliable |
| `_V_CUT` | 20.0 | Pairing-vertex near-divergence detector threshold |
| `_JCHI_HARD_REJECT` | 2.0 | $J\chi_{SS}$ above this ⇒ hard-rejected (deeply AFM, SC impossible) |
| `_QQ_DELTA_THRESH` | $10^{-8}$ | $|\Delta|$ threshold below which $\chi_{SQ}$ is enforced to be exactly zero |

**Moriya damping, JT viability, finite-difference steps**

| Constant | Value | Role |
|---|---|---|
| `_MORIYA_LANDAU_M_STEP` | 0.06 | $M$-probe step for the self-consistent, Landau-expansion-derived $\Gamma_M$ (§14) — replaces the older empirical $\alpha_M\cdot f(\delta)\cdot\mathrm{sat}(t/J)$ formula entirely; there is no `_MORIYA_C`/`_ALPHA_MORIYA`-type constant family in the current code |
| `_LAMBDA_JT_VIABLE` | 0.05 | Minimum $\lambda_{JT,\mathrm{sc}}$ for SC-triggered-JT viability (§17) |
| `_JT_ACT_THR` | 0.04 | Condensate-induced Γ₆–Γ₇ mixing threshold for the "JT-active" classification |
| `_DQ_FS_VERTEX` / `_DQ_FS_VERTEX_FRAC` | 0.03 Å / 0.05 | Minimum / adaptive-fraction finite-difference step for $\partial\lambda/\partial Q$ on the FS |
| `_JT_FD_H2_BASE` / `_JT_FD_H2_QCOEF` | $3\times10^{-8}$ / $6\times10^{-7}$ | $Q$-derivative FD step schedule, $h(Q)=\sqrt{\text{base}+\text{qcoef}\cdot Q^2}$ |

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
| `_SCF_DIVERGE_RATIO` / `_SCF_STAGNATE_RATIO` | 1.05 / 0.95 | $\max|\Delta|$ growth ratio thresholds classifying the step as diverging / stagnating |
| `_SCF_ALPHA_DECAY` / `_SCF_ALPHA_RECOVER` | 0.95 / 1.60 | Mixing-rate multiplier while converging (mild damping) / on freeze-recovery |
| `_SCF_FREEZE_THR` | 10 | Consecutive frozen iterations that trigger freeze-recovery |
| `_SCF_ALPHA_FREEZE_LO` / `_SCF_ALPHA_FREEZE_HI` | 0.15 / 0.60 | $\alpha/$`_MIXING` bounds defining "too frozen" / the recovery ceiling |
| `_SCF_ALPHA_CONVG_BOOST` / `_SCF_ALPHA_CONVG_CAP` | 1.15 / 0.75 | Mixing-rate boost / ceiling while SC+JT active and converging |
| `_EMA_NEW_WEIGHT` | 0.14 | EMA weight for the $V_d$ sign-flip guard and for $\Lambda_{\mathrm{inst}}$ — **not** a cluster-regression EMA (that scheme no longer exists, §19) |
| `_Q_UPDATE_PERIOD` | 3 | Heartbeat period (iterations) for the Hellmann–Feynman $Q$ update |
| `_Q_THR_REL` | 0.016 | Fraction of `lambda_hop`; $Q$ change below this skips the vertex-cache rebuild / Anderson injection |
| `_Q_SEED_THR` | $10^{-4}$ | If the initial $Q$ seed is already nonzero, it is trusted as the current best estimate |
| `_M_THR_REL` | 0.01 | Absolute $M$-change threshold for vertex-cache invalidation |
| `_ALPHA_MIX_2X2` | 0.56 | Blend weight: 2×2 pairing-kernel eigenvector vs. fixed-point gap update |

**Gap (Δ) update and coherence-length classification**

| Constant | Value | Role |
|---|---|---|
| `_BCS_SEED_FRACTION` | 0.1 | Initial cold-start $\Delta$ seed, as a fraction of $t_{\mathrm{eff}}$ |
| `_DELTA_JUMP_CAP` | 5.0 | Maximum $|\Delta_{\mathrm{new}}|/|\Delta_{\mathrm{old}}|$ ratio per iteration |
| `_DELTA_ABS_FLOOR` | $10^{-4}$ eV | $|\Delta|$ below this bypasses the jump limiter (free seed-growth phase) |
| `_PHI_D_FLOOR` | $10^{-3}$ | Minimum $\varphi_d^{\max}$ to enable nodal/antinodal decomposition |
| `_KERNEL_DIR_MIN_FRAC` | 0.5 | 2×2-kernel eigenvector allowed to dominate the mixing below this fraction of fixed-point amplitude |
| `_MU_LM_DELTA` | 3.0 | Levenberg–Marquardt floor for the Δ Newton step (2×2 analogue of `_MU_LM`) |
| `_ALPHA_HF_DELTA` | 0.20 | Newton-vs-BdG-fixpoint blend for the Δ update |
| `_TR_DELTA_STEP_MAX` | 0.1 | Upper bound on the Δ Newton step per channel per iteration (eV) |
| `_XI_NODAL_MIN` | 2.0 | Minimum $\xi/a$ (nodal) for BCS-side quasiparticle coherence |
| `_ORBITAL_SEL_FRAC` | 0.15 | $|\xi_{\Gamma_6}-\xi_{\Gamma_7}|/\xi$ threshold for "orbitally selective" pairing |
| `_IC_RATIO_FLOOR` / `_IC_RATIO_CAP` | 1.05 / 3.00 | Bounds on the cluster-ED inter-channel correction ratio |
| `_MBZ_DEGEN_TOL` | $10^{-8}$ eV | Energy-based tie-break tolerance for magnetic-BZ Fermi-surface point selection |

**Tc / Ginzburg–Landau / BCS ratio**

| Constant | Value | Role |
|---|---|---|
| `_BCS_RATIO_STRONG` / `_VSTRONG` / `_EXOTIC` | 3.8 / 5.0 / 7.0 | $2\Delta_0/k_BT_c$ thresholds for strong / very-strong / exotic coupling |
| `_MAD_DENOM` / `_MAD_NUM` | 1.13 / 1.04 | Allen–Dynes-type strong-coupling denominator (Millis–Monien–Pines spin-fluctuation value) / exponent prefactor |
| `_GL_DELTA_MIN` | 2 meV | $|\Delta|$ floor for points admitted to the GL fit |
| `_GL_MIN_PTS` / `_GL_MAX_PTS` | 2 / 4 | Minimum / maximum recent stable-SC points used in the GL regression |
| `_GL_TC_MARGIN` | 0.05 | Maximum relative deviation $|T_{c,GL}-T_{\mathrm{spinodal}}|/T_{\max}$ to accept the GL result |
| `_GL_SPINODAL_JUMP` | 0.15 | $D_{\mathrm{spinodal}}/\Delta_0$ below this ⇒ GL extrapolation treated as reliable (small first-order jump) |

**Physical thresholds (SC viability / Mott)**

| Constant | Value | Role |
|---|---|---|
| `_G_T_COHERENCE_MIN` | 0.10 | Mott guard: minimum coherent $g_t$ |

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

1. **Parameter setup and SOC+CF diagonalization.** `ModelParams(...)` is constructed with the defaults listed in [Parameters](#parameters) and `__post_init__` runs the SOC+CF diagonalization (Γ₆/Γ₇ₐ/Γ₇ᵦ identification, `Delta_CF`, `sz_op`, `p_7`, k-grids, orbital operators).
2. **Doping setup.** `target_doping = 0.11`, with a symmetric ±20% scan margin (`doping_margin`) defining `min_doping`/`max_doping`, floored to stay above the `_G_T_COHERENCE_MIN` Mott-incoherence boundary; `initial_Delta = 8×10⁻³` eV is the cold-start gap seed.
3. **Three-way self-consistent comparison, run in parallel.** Three independent `RMFT_Solver` instances are built from deep copies of `params`, and `solve_self_consistent` is run on each concurrently via a 3-worker `ThreadPoolExecutor`:
   - **`ref`** — the full self-consistent solve, SC and $Q$ both free (the model's actual prediction).
   - **`normal`** — $\Delta$ pinned to zero throughout (`force_delta_zero=True`); the normal (non-superconducting) AFM state.
   - **`SC_Q0`** — SC free but $Q$ pinned to zero (`force_Q_zero=True`); superconductivity without the JT relaxation.

   This is the direct numerical test of the central hypothesis: if the theory is right, `F_bdg` should order `ref < normal` (condensation lowers the free energy) and `ref < SC_Q0` (JT relaxation lowers it further), and `Q` should self-consistently relax back toward 0 in `normal` *without being forced there* — the code checks this explicitly and logs `"SC+JT is the ground state ✓"` or `"✗"` accordingly (guarded by the `_scf_result_reliability` check on all three results).
4. **G-matrix diagnostics.** `compute_G_instability(target_doping, M)` at the self-consistent $M$ from `ref`, logging the normal-state instability classification (§16).
5. **Post-SCF diagnostics** (only if the reference SCF succeeded): RPA vertex decomposition, linearized-gap-equation channel decomposition, coherence lengths, the SC Hessian and SC-JT-triggering confirmation, the Stoner ratio, the $\chi_\tau$ softening check, the SC-JT coexistence window (§17), and the three Tc estimates (§23).
6. **Two diagnostic plots.** `estimate_chi_SQ_q_full` (`n_q=35`) produces the spin–quadrupole full-BZ scan figure; `plot_ground_state_comparison(results)` produces the `ref`/`normal`/`SC_Q0` comparison figure. Both are saved to disk (see [Output & Diagnostics](#output--diagnostics)).

---

## Output & Diagnostics

All output is a structured, thread-safe log stream (`_scf_log`, tagged by stage: `RMFT-INIT`/`INIT`/`DERIVED`, `REF-SCF`, `SCF`, `SCF-RES`, `FREE-ENERGY`, `G-MATRIX`, `TC-PRELIM`, `TC-THERMO`, `PLOT`, …) rather than a GUI; the graphical output is the two `matplotlib` figures described below.

### Iteration Log

Each logged SCF step reports the current order parameters, the effective exchange and mixing rate, and warning flags for degenerate or numerically marginal vertex structure:

```
[SCF] δ=…  iter/max  conv=…  M=…  Q=…  |Δ|=…  J_eff=… eV  mu=…
      dFM=…  dAFM=…  V_s=…  V_d=…  [⚠low-var] [⚠same-sign]
      Γ_M=…  α=…  B1g=…  F67s=…  [regime]  …s/it
```

At convergence, an `SCF-RES` block reports the converged order parameters, density, $\mu$, free energies, `F67s_mf`, the AFM/RPA determinant, the JT-active flag, the SCF-dynamics regime classification (§ Key Algorithms), the s-/d-channel decomposition of $\lambda_{\max}$, `lambda_JT_sc`, `lambda_JT_kernel`, $\partial\lambda_{\mathrm{pair}}/\partial Q$, the post-convergence Hessian's SC-triggered-JT confirmation, coherence lengths, the $\chi_\tau$ breakdown (including its reliability weight), the SC-JT window verdict, and the incommensurate-nesting scan result.

### Free-Energy Ground-State Check

Logged under the `FREE-ENERGY` tag right after the three parallel tasks (`ref`, `normal`, `SC_Q0`) complete: `F_bdg` and the post-convergence Hessian minimum eigenvalue for each of the three, flagged `⚠ UNRELIABLE` individually when `_scf_result_reliability` fails, and a final one-line verdict — `"SC+JT is the ground state ✓"`, `"SC+JT is NOT the lowest energy state ✗"`, or `"Comparison NOT trustworthy"` if any of the three results is unreliable.

### G-Matrix Block

Logged separately from the SCF result (evaluated at the self-consistent $M$ from `ref` but in the normal, $\Delta=0$ state): the exchange scale $h_{AFM}$, the pairing susceptibilities $\chi_{\Delta\Delta}$ in each channel, the normal-state pairing eigenvalue `lambda_eff`, the normal-state lattice stability ($K_{\mathrm{eff}}$, $\chi_{QQ}$, $\lambda_{JT}^{\mathrm{norm}}$, $\partial^2F/\partial Q^2|_{\Delta=0}$), and the full `InstabilityInfo.log_summary()` classification (§16).

### RPA Vertex and SC-JT Window

Following the G-matrix block, if the reference SCF converged: the FS-averaged RPA vertex decomposition into spin / JT / cross contributions; the linearized-gap-equation channel decomposition; the coherence-length summary (flagging orbital-selective pairing when $\Gamma_6$ and $\Gamma_7$ channels have meaningfully different $\xi$); the SC Hessian's smallest eigenvalue; the Stoner ratio $J_{\mathrm{eff}}\chi_{SS}$ with a QCP-proximity classification; the `K_eff` path from the normal to the SC state (with a term-by-term breakdown of the four contributions in §11); the $\chi_\tau$ breakdown; and the SC-JT window verdict (§17), including the current `K_lattice`'s position within the viable window as a percentage.

### χ_SQ(q) Full-BZ Scan Plot

`estimate_chi_SQ_q_full` (called with `n_q=35` in the current `__main__`) produces a `matplotlib` figure with a **2×3** grid of panels (`plt.subplots(2, 3, ...)`) showing the spin–quadrupole cross-susceptibility $\chi_{SQ}(q)$ over the Brillouin zone — separately for the normal and SC states — together with the symmetry-consistency check described in §2 — the logged normal-state peak, which should be numerically indistinguishable from zero — plus the local-vertex-validity and normal-vs-SC peak-shift warnings the same call logs.

### Ground-State Comparison Plot

`plot_ground_state_comparison(results)`, a standalone module-level function (not a solver method), takes the `{"ref", "normal", "SC_Q0"}` result dictionary produced in [Installation & Usage](#installation--usage) step 3 and saves a **2×2** figure to `ground_state_comparison.png`: (top-left) `F_cluster` per SCF iteration for all three scenarios, each with its converged `F_bdg` as a dotted reference line, unreliable trajectories drawn dashed; (top-right) a bar chart of the three final `F_bdg` values, titled with whichever scenario is lowest (unreliable bars hatched); (bottom-left) the Γ₆-channel AFM order parameter $M[\Gamma_6]$ versus iteration, all three overlaid; (bottom-right) $|Q|$ versus iteration, the key visual check that `normal`'s $Q$ relaxes back toward 0 on its own (it is never forced there) while `ref` settles at a finite value. Missing or unreliable scenario entries are skipped rather than erroring, provided at least one entry is present.

---

## Known Limitations

The framework makes a number of physically motivated approximations. The table below merges the limitations identified in the theoretical write-up with implementation-level caveats found in the current code.

| Approximation | Impact |
|---|---|
| No Pauli exclusion between plaquette sites | Mild overestimate of AFM correlations; controlled by the Newton/BdG-fixpoint blend weight `_ALPHA_HF` |
| No charge-transfer fluctuations $\langle n_An_B\rangle$ | Negligible when the mean-field exchange scale is large compared to the hopping |
| Static phonon ($Q$ is a mean field) | Zero-point quantum lattice fluctuations are neglected; the JT frequency is derived from $K_{\mathrm{eff}}$, not an independent input |
| $\Gamma_{7b}$ pairing is diagnostic-only, not self-consistent | The full 6-orbital BdG basis includes $\Gamma_{7b}$'s band structure exactly (§1, §9), and a raw RPA estimate of a $\Gamma_6$–$\Gamma_{7b}$ pairing amplitude is computed every iteration (§8), but it is never fed back into the Hamiltonian — the converged solution always pairs through $\Gamma_6$–$\Gamma_{7a}$ only. If the diagnostic amplitude is not small, the converged state may not be the true minimum in the full 3-doublet pairing space |
| No spatial fluctuations | Cannot describe a pseudogap, stripe order, or phase separation |
| RPA static ($\omega=0$) | Dynamical vertex corrections are absent |
| `K_eff` update conditional | Recomputed only when $M$ or $Q$ have moved enough (§ Key Algorithms); $Q$'s back-action on the exchange rigidity is approximate during the SCF transient, though exact at convergence |
| $\chi_\tau$ evaluated only post-convergence in some diagnostics | The fully self-consistent back-action of $Q$ on $\chi_\tau$ within the SCF loop itself is not iterated to convergence at every step |
| G-matrix evaluated at $\Delta=0$ only | Diagnoses normal-state stability; the actual SC-triggered-JT scenario is confirmed independently via the post-SCF Hessian ($\lambda_{\min}<-kT$) and, in the current `__main__`, via the explicit three-way free-energy comparison (`ref` vs. `normal` vs. `SC_Q0`) |
| $\partial\lambda_{\mathrm{pair}}/\partial Q$ at a frozen Fermi surface | FS geometry is evaluated at a fixed $Q$ rather than self-consistently re-resolved; a fully SC-state Bogoliubov–Lindhard version would be more expensive |
| $\delta\chi_\tau$ baseline subtraction approximate in D₂h | The normal-state B₁g response at finite `Delta_B1g_static` is estimated at $\Delta=0$; small residual D₂h corrections to the baseline are neglected |
| `chi_tau_weight` partial suppression | When the Richardson extrapolation only agrees at the finer step pair (`chi_tau_weight = 0.5`), the SC-JT feedback may still be over- or under-estimated near a first-order boundary |
| SCF-dynamics regime classification | `first_order_jump` and `hysteretic` trigger a multi-seed restart (lowest free energy wins); `limit_cycle` only damps the mixing rate; the classification is heuristic, based on the shape of the $|\Delta|$ history |
| Eg,2 channel partially self-consistent | The Eg,2 phonon is fully wired into the Hamiltonian, free energy, and Hessian, but its exchange-driven rigidity correction and its cross-rigidity with the B₁g channel are currently left at zero (they vanish by Kramers symmetry at the level presently implemented); its own SC-triggering diagnostics are therefore less developed than the B₁g channel's |
| Incommensurate AFM handled only as a diagnostic + soft retry | `_scan_incommensurate_nesting` detects a preference for $q^*\neq(\pi,\pi)$ and triggers one softened-$M$ retry, but the BdG Hamiltonian itself remains fixed at commensurate $(\pi,\pi)$ ordering throughout — a genuinely incommensurate spiral solve is not implemented |
| $V_d$ sign-flip EMA | Suppresses numerical oscillation in the d-wave vertex but may slow the genuine response near a doping-driven crossover between d-wave and s-wave dominance |
| Cluster-ED vertex is a $q=0,\omega=0$ local estimate | `V_irr_QQ` from the four-site plaquette (§19) is a local quantity standing in for a genuinely $k$-dependent, dynamical vertex; three exact diagonalizations of a $1296$-dimensional Hilbert space per call also make it the single most expensive step in the SCF loop |

---

## References

- Ecsenyi, S. (2026). *Multipolar superconductivity and Jahn–Teller activation in strongly correlated systems: a self-consistent theoretical framework* (preprint).
- Anderson mixing: Pulay, P. (1980). *Chem. Phys. Lett.* 73, 393.
- Gutzwiller renormalization: Zhang, F.C. et al. (1988). *Supercond. Sci. Technol.* 1, 36; Bünemann, J., Weber, W. & Gebhard, F. (1998). *Phys. Rev. B* 57, 6896.
- ZSA classification: Zaanen, J., Sawatzky, G.A. & Allen, J.W. (1985). *Phys. Rev. Lett.* 55, 418.
- BdG formalism: de Gennes, P.G. (1966). *Superconductivity of Metals and Alloys.*
- Jahn–Teller effect: Bersuker, I.B. (2006). *The Jahn–Teller Effect.* Cambridge University Press.
- RPA spin fluctuations: Scalapino, D.J. (1995). *Phys. Rep.* 250, 329.
- Allen–Dynes strong-coupling formula: Allen, P.B. & Dynes, R.C. (1975). *Phys. Rev. B* 12, 905; basis: McMillan, W.L. (1968). *Phys. Rev.* 167, 331.
- Ginzburg–Landau theory: Ginzburg, V.L. & Landau, L.D. (1950). *Zh. Eksp. Teor. Fiz.* 20, 1064.
- Cluster-DMFT-style irreducible-vertex extraction (χ0⁻¹−χ⁻¹): Maier, T. et al. (2005). *Rev. Mod. Phys.* 77, 1027.
- Moriya spin fluctuations: Moriya, T. (1985). *Spin Fluctuations in Itinerant Electron Magnetism.* Springer.
- Nearest positive-semidefinite matrix projection: Higham, N.J. (1988). *Linear Algebra Appl.* 103, 103.

---

*For questions or contributions, open an issue or pull request.*
