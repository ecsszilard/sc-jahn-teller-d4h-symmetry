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

### Material class and the three viability conditions

The theory targets systems with:

- D₄h point-group symmetry **and** a global inversion center (square-lattice materials: cuprates, pnictides, selected layered transition-metal oxides). Inversion symmetry is what allows the Cooper pair to have a well-defined parity (pure singlet or pure triplet); if it is broken locally, Rashba/Dresselhaus terms mix the two and the clean tensor-product selection rule below no longer applies unambiguously.
- Strong electron correlation with a charge-transfer (ZSA-type) insulating parent state.
- Strong SOC that reorganizes the $t_{2g}$ manifold into $\Gamma_6 \oplus \Gamma_7$ Kramers doublets (typical of 4d/5d transition-metal ions, though the framework is agnostic to the microscopic origin of the SOC scale).
- Superexchange-stabilized collinear AFM order in the parent compound.
- Finite hole doping ($\delta > 0$), which restores the itinerancy needed for coherent Cooper pairing.

Three conditions must hold simultaneously for the mechanism to operate:

1. **Charge-transfer insulating character** — $U$ (or more precisely the ZSA charge-transfer gap) is large enough that genuine Mott physics is in play.
2. **Non-Mott-localized coherence** — the Cooper pairs must actually be mobile; this requires $\delta > 0$ so the Gutzwiller kinetic factor $g_t$ does not collapse to zero.
3. **Moderate AFM order** — AFM correlations must be present (they are what forbids the JT channel in the normal state) but not so strong that spin fluctuations kill superconductivity outright (Stoner criterion $J_{\mathrm{eff}}\cdot\chi_{SS} < 1$).

### Why the pairing must be interorbital, and why turning on Δ alone is not enough

A tempting but incorrect picture is that the condensate simply mixes $|\Gamma_6\rangle$ and $|\Gamma_7\rangle$ into a coherent superposition $\alpha|\Gamma_6\rangle+\beta|\Gamma_7\rangle$, giving a quadrupole moment linear in the mixing amplitude $\beta\propto\Delta$. This cannot be right: it would make $\langle B_{1g,\mathrm{op}}\rangle$ depend on the arbitrary global $U(1)$ phase of $\Delta$, violating gauge invariance. The correct microscopic statement is that the condensate modifies the **normal** (charge-conserving) density matrix through the Bogoliubov coherence factors: $\langle c^\dagger_{6\sigma}c_{7\sigma'}\rangle$ picks up a $v^*v \sim |\Delta|^2$ contribution, which is explicitly gauge-invariant and appears at *quadratic*, not linear, order in $\Delta$.

For this to happen at all, the pairing must directly connect $\Gamma_6$ and $\Gamma_7$ — a purely intraorbital singlet (Γ₆–Γ₆ or Γ₇–Γ₇) generates no $\Gamma_6$–$\Gamma_7$ Bogoliubov mixing and leaves the JT channel closed even in the superconducting state. This is exactly the structure built into the code's two pairing operators $D_s$, $D_d$ (§8 below), which are both interorbital by construction.

There is a further, more subtle point verified directly in the group-theoretic analysis: **at $Q=0$, even with $\Delta\neq0$, a purely $D_s/D_d$-paired BdG ground state still gives $\langle B_{1g,\mathrm{op}}\rangle \equiv 0$ band-pair by band-pair.** The reason is that $B_{1g,\mathrm{op}}$ in the numerical SOC eigenbasis turns out to be **spin-conserving** ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\uparrow}$), while the singlet pairing operator only ever connects opposite pseudospin sectors; $S_z$ stays block-diagonal in those same sectors. The actual switch is thrown by the *self-consistently induced anomalous coherence* $\langle\tau_x\rangle_{\mathrm{anom}}$ — numerically the same operator as $B_{1g,\mathrm{op}}$ — which is exactly zero at $Q=0$ and becomes nonzero only once **both** $\Delta\neq0$ **and** $Q\neq0$. In the code this anomalous coherence is the quantity `F67s_mf`, fed back into the local Hamiltonian (§19 below) only once a nonzero JT distortion has already appeared; the loop is genuinely self-consistent rather than a one-way "SC turns on JT."

### The JT distortion as a thermodynamic order parameter, not a dynamical mode

$Q$ is treated as a **macroscopic, thermodynamic order parameter** — physically the amplitude of a flat (dispersionless) optical Einstein phonon — rather than a fluctuating dynamical degree of freedom. Differentiating the free energy with respect to $Q$ gives the equilibrium condition and the softening criterion

$$\lambda_{JT}^{\mathrm{norm}} = \chi_{QQ}/K_{\mathrm{eff}} \qquad (<1\text{ stable},\ =1\text{ onset},\ >1\text{ spontaneous JT}).$$

A natural question is whether the free energy contains a **linear** coupling $Q\cdot|\Delta|^2$ between the lattice and the condensate. Naively one might argue this is forbidden simply because "$|\Delta|^2$ is always $A_{1g}$" — but that argument is incomplete, because the model's two pairing channels ($D_s$, $D_d$) do not necessarily share one point-group label, and their cross term could in principle transform as $B_{1g}$. The theory derives, and the code verifies numerically, that this cross-coupling is *exactly* zero in D₄h — an exact but non-generic consequence of the purely interorbital structure of the pairing operators, not a trivial symmetry accident. (If the lattice already sits in D₂h because of a finite static crystal field `Delta_B1g_static` ≠ 0, this exact cancellation is lifted and a genuine linear coupling appears — see §3 below.) The upshot: in the clean D₄h limit, SC-triggered JT is a **threshold phenomenon**. The condensate progressively softens the B₁g mode's effective stiffness $K_{\mathrm{eff}}$ until $\chi_{QQ} = K_{\mathrm{eff}}$, at which point the lattice snaps into a finite distortion.

### First-order character

Numerically, this transition is expected to be **first-order**, not a simple second-order spin-fluctuation instability: spin fluctuations alone are not sufficient to drive the lattice unstable, and the system tips into the $B_{1g}$-distorted configuration only cooperatively, together with the superconducting condensate. This is reflected in the solver's SCF-dynamics classifier (see the "SCF Loop" description under [Key Algorithms](#key-algorithms)) and in the thermodynamic, first-order-aware Tc estimate of §23 below.

---

## Theoretical Framework

### 1. Local Hilbert Space: SOC + Crystal-Field Diagonalization

The full SOC + D₄h crystal-field Hamiltonian is built and diagonalized explicitly on the 6-dimensional $t_{2g}\otimes\mathrm{spin}$ manifold, directly inside `ModelParams.__post_init__`:

```
H = λ_SOC · L·S  +  Δ_tetra · Lz²  +  Delta_B1g_static · (Lx² − Ly²)
```

This yields the Γ₆–Γ₇ splitting `Delta_CF` as a **derived quantity**, never a free input. `Delta_tetra` (negative = tetragonal z-compression) sets the axial crystal field; `Delta_B1g_static` is a static, in-plane crystal-field term with the same $(L_x^2-L_y^2)$ functional form as the dynamical JT operator, logged as `Δ_ip` and referred to as $\Delta_{\mathrm{inplane}}$ in the theoretical write-up. Its role is to split the four-dimensional $\Gamma_7$ manifold into two Kramers doublets, $\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}$, which prevents a spurious spontaneous JT instability from the residual $\Gamma_7$ degeneracy while leaving $\Delta_{CF}$ tunable independently of $\lambda_{SOC}$.

**Kramers doublet identification** proceeds in two steps:
1. The three Kramers doublets of $H_{SOC}+H_{CF}$ are sorted by the expectation value $\langle L\!\cdot\!S\rangle$; the doublet with the most negative value is assigned $\Gamma_6$ ($j_{\mathrm{eff}}=1/2$-like), and the two remaining candidates are the $\Gamma_7$ pair.
2. Within each doublet a 2×2 diagonalization of $S_z$ selects the exact $z$-polarized Kramers partners (`up`, `dn`) and their eigenvalues (`sz_up`, `sz_dn`). Between the two $\Gamma_7$ candidates, the one with the **larger $|\langle S_z\rangle|$** is assigned $\Gamma_{7a}$ (the JT-active partner); the total moment $\mu_z=\langle L_z+2S_z\rangle$ is computed only as an independent **cross-check** and triggers a warning log if it disagrees with the spin-polarization criterion — this can happen in strongly mixed CF/SOC regimes.

The resulting basis is $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow},\Gamma_{7b}{\uparrow},\Gamma_{7b}{\downarrow}]$; only the first four components (the $\Gamma_6\oplus\Gamma_{7a}$ subspace) enter the BdG solver. Several derived quantities are cached on `ModelParams` at this point:

- `sz_op = [sz6_up, sz6_dn, sz7_up, sz7_dn]` — **exact** $\langle S_z\rangle$ eigenvalues from the doublet diagonalization (not an approximate moment-ratio model); used directly as the AFM Weiss-field weights and as the spin vertex in every susceptibility calculation.
- `multi_op` — the effective multipolar spin operator entering the cluster exchange $H_{\mathrm{exch}} = J\cdot(\mathrm{multi\_op}\otimes\mathrm{multi\_op})$, built as $\mathrm{diag}\big((|sz_6|\cdot P_6+|sz_7|\cdot P_7)\cdot sz_{\mathrm{diag}}\big)$.
- `p_7` — the average $\Gamma_7$ orbital-weight admixture in the $\Gamma_6$ eigenvectors; interpolates the d-wave Gutzwiller factor (§5).
- `Delta_CF = evals[2] − evals[0]` (Γ₇a–Γ₆ gap, JT-active) and `g7split = evals[4] − evals[2]` (Γ₇a–Γ₇b internal splitting).
- Orbital-character weights `_w6_xz, _w6_yz, _w6_xy, _w7_xz, _w7_yz, _w7_xy` — the $d_{xz}/d_{yz}/d_{xy}$ character of $\Gamma_6$ and $\Gamma_{7a}$, used to build the Q-dependent exchange anisotropy $\eta_J(Q)$ (§6).

**Validity of the 4×4 truncation:** dropping $\Gamma_{7b}$ is controlled — it is accurate when $\Delta_{CF}\gg kT$ and $\Gamma_{7\mathrm{split}}/\Delta_{CF}\ll1$; virtual $\Gamma_{7b}$ contributions scale as $(J_{\mathrm{eff}}/\Delta_{CF})^2$ and enter the Bayesian-optimizer scoring as a smooth projection-quality penalty.

If `lambda_soc`, `Delta_tetra`, or `Delta_B1g_static` are mutated on a live solver (e.g. inside the optimizer's parameter search), `params.__post_init__()` must be followed by `solver._rebuild_orbital_operators()` so that `B1g_op`, `B1g_16`, `Eg2_op`, `Eg2_16`, `sz_op`, `multi_op` and the Nambu vertex matrices stay consistent with the new eigenbasis.

### 2. Symmetry Protection of the JT Channel

**Selection rule in a pure doublet.** A rank-$k$ irreducible tensor operator has a nonzero diagonal matrix element in $\Gamma_6$ only if $\Gamma^{(k)}\subset\Gamma_6\otimes\Gamma_6$. Since $\bar D_{4h}$ character theory gives $\Gamma_6\otimes\Gamma_6=\Gamma_7\otimes\Gamma_7=A_{1g}\oplus A_{2g}\oplus E_g$ — containing neither $B_{1g}$ nor $B_{2g}$ — the quadrupole operator has $\langle\Gamma_6|Q^{(2)}_{B_{1g}}|\Gamma_6\rangle=0$ **exactly**. A pure $\Gamma_6$ (or $\Gamma_7$) manifold carries no electric quadrupole moment and does not couple to a $B_{1g}$ lattice shear. Because the collinear AFM state stabilizes only this kind of dipolar (rank-1), spin–orbitally mixed order, the JT channel is symmetry-blocked in the normal state.

**Cross-product opens the channel.** $\Gamma_6\otimes\Gamma_7 = B_{1g}\oplus B_{2g}\oplus E_g$ *does* contain $B_{1g}$, so the **off-diagonal** element $\langle\Gamma_6|Q^{(2)}_{B_{1g}}|\Gamma_7\rangle\neq0$ is allowed. Realizing this requires a genuinely interorbital pairing channel (see the Physical Hypothesis section above) and, as emphasized there, requires $Q\neq0$ as well as $\Delta\neq0$ — the self-consistent loop, not a one-shot symmetry argument, is what actually opens the channel.

**Θ symmetry and the global BZ cancellation.** In the collinear AFM state, time reversal $\mathcal T$ is broken, but the combined Shubnikov element $\Theta=\mathcal T\cdot\tau_{AB}$ (time reversal composed with the $A\leftrightarrow B$ sublattice translation) survives; since $\tau_{AB}^2=+1$ and $\mathcal T^2=-1$, $\Theta^2=-1$. This does **not** give pointwise Kramers degeneracy — $\Theta$ maps crystal momentum $k\to -k$, and in the magnetic Brillouin zone $k$ and $-k$ are generally inequivalent — but it does impose a **global** constraint on Brillouin-zone integrals. The spin–quadrupole susceptibility integrand is odd under $\Theta$ at $\Delta=0$, so the full-BZ Lindhard sum cancels identically:

$$\chi_{SQ}(q) = \int_{\mathrm{BZ}} d^2k\; \mathcal I_{SQ}(k,q) = 0 \qquad (\Delta=0).$$

This BZ-wide cancellation — not a pointwise Kramers argument — is enforced in the code by the odd-in-$k$ structure of the normal-state Lindhard kernel (`_lindhard_bubble` with `_NORMAL_SECTOR_PAIRS`), and is checked at runtime: `estimate_chi_SQ_q_full` flags `symmetry_ok` when the residual normal-state $\chi_{SQ}$ peak is below $10^{-3}$.

In the superconducting state the full Nambu–Lehmann sum includes the anomalous (Gorkov) sectors, whose Bogoliubov coherence factors can break the odd-$\Theta$ structure — but as detailed above, this alone is not sufficient at $Q=0$: a purely singlet, interorbital pairing keeps $S_z$ and $B_{1g,\mathrm{op}}$ acting within disjoint pseudospin sectors band-pair by band-pair, so the product stays exactly zero until the self-consistently generated $Q\neq0$ genuinely couples the two sectors.

**The selection ratio, explicitly.** The quantity that actually crosses the symmetry boundary is the Γ₆–Γ₇ anomalous (Gorkov) singlet amplitude, computed directly from the converged BdG eigensystem as

$$F_{67s} = \sum_k (1-2f_n)\,\mathrm{Re}\big[u^*_{6\uparrow}v_{7\downarrow} - u^*_{6\downarrow}v_{7\uparrow}\big] \quad\text{(mean over sublattices)},$$

with $F_{67s}\equiv0$ whenever $\Delta=0$ (exact D₄h selection rule) — the code's own inline invariant. The mean-field quantity actually fed back into the local Hamiltonian is a Gutzwiller-weighted average over both pairing channels, `F67s_mf = g_eff * F_67s`, with `g_eff = (g_Delta_s * |Δ_s| + g_Delta_d * |Δ_d|)/(|Δ_s|+|Δ_d|)` — so a channel that carries more of the total gap amplitude also carries proportionally more weight in how strongly the condensate talks to the lattice.

### 3. The B₁g Operator and the D₄h/D₂h Crossover

The B₁g phonon coupling operator is constructed from the same $t_{2g}$ operators used for $H_{CF}$ and projected into the 4-dimensional $\Gamma_6\oplus\Gamma_{7a}$ subspace:

```
B1g_op = real(U4† · (Lx² − Ly²)_t2g · U4)     # 4×4, real, Hermitian
```

where `U4` is the projector onto the low-energy manifold built during the SOC+CF diagonalization.

- **D₄h (`Delta_B1g_static = 0`):** `B1g_op` is purely off-diagonal and **spin-conserving** — it connects $\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\uparrow}$ and $\Gamma_6{\downarrow}\leftrightarrow\Gamma_{7a}{\downarrow}$, with an exactly zero diagonal. Hence $\langle B_{1g,\mathrm{op}}\rangle=0$ in *any* normal state, and the diagonal-norm-to-off-diagonal-norm ratio gives `b1g_weight ≈ 1`.
- **D₂h (`Delta_B1g_static ≠ 0`):** the operator picks up real diagonal ($A_{1g}$-like) components that renormalize $\Delta_{CF}$ but do not by themselves drive JT, plus additional off-diagonal weight. `b1g_weight = b1g_off_norm/(b1g_diag_norm+b1g_off_norm) < 1` in this regime, quantifying how much of the operator remains genuinely SC-triggered versus normal-state-active.

The 16×16 Nambu extension `B1g_16` carries the hole block as $-B_{1g,\mathrm{op}}^{T}$ (real, so $=-B_{1g,\mathrm{op}}$), consistent with BdG particle–hole symmetry; every JT coupling term in the Hamiltonian is written `H += g_JT · Q · B1g_op` rather than a hand-built matrix.

### 4. ZSA Charge-Transfer Superexchange and the Weiss Field

AFM order originates from virtual $p$–$d$ hopping (ZSA charge-transfer superexchange), not from a bare Stoner Fermi-surface instability:

```
J_pdct = 1/U_dd + 1/(Delta_CT + U_pp/2)
```

`U_dd = u · t0` is the on-site Hubbard repulsion (`u` is the dimensionless $U/t_0$ ratio, a primary input); `U_pp` is the ligand (2p) hole–hole repulsion, obtained from a second-order downfolding estimate rather than a fixed ratio:

```
U_pp = U_dd / (1 + hybrid_scale · t_pd / Delta_CT)
```

`hybrid_scale` represents the geometric coordination factor in this downfolding; larger values correspond to stronger covalency (for typical `t_pd`, `Delta_CT` this pushes `U_pp` below roughly half of `U_dd`, i.e. quasiparticle weight shifting off the metal ion). The two `J_pdct` terms are, respectively, the Mott channel ($pd\to dd$, cost $U_dd$) and the Zhang–Rice channel ($pd\to pp$, cost $\Delta_{CT}+U_{pp}/2$).

The effective AFM Weiss field entering the BdG Hamiltonian is diagonal in the $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow}]$ basis, proportional to `sign_M · J_A1g[α,α] · M · sz_op[α]`, where `J_A1g` is the longitudinal (spin-preserving) exchange tensor from `exchange_channels()` (§6) and the doping renormalization is carried by `spin_dil = max(1 − δ, 0)`, the fraction of singly occupied sites.

### 5. Gutzwiller Renormalization and the Mott Guard

`t_pd` is the single primary hopping input; the effective $dd$ hopping `t0 = t_pd² / Delta_CT` is always derived, never set directly — the optimizer searches over `t_pd` while `Delta_CT` is held fixed as a material-class constant.

```
g_t       = 2δ/(1+δ)         # kinetic-energy suppression → 0 at half-filling
g_J       = 4/(1+δ)²         # exchange enhancement → 4 at half-filling
g_Delta_s = g_t                                        # on-site Γ6⊗Γ7 channel: kinetic in origin
g_Delta_d = g_t + (g_J − g_t) · p_7                    # d-wave B1g channel: interpolated by Γ7 admixture
```

The superexchange is always computed from the **bare** hopping `t_pd`, then multiplied by `g_J` — computing it from Gutzwiller-*renormalized* bands would double-count the suppression (a spurious `g_t²` factor), since `g_t` renormalizes only the kinetic energy while `g_J` renormalizes only the exchange; the two are orthogonal channels in this RMFT scheme. The lattice-summed Weiss-field/superexchange scale is `J_eff = Z · J_bond`, with `J_bond` a single-bond quantity scaling as `spin_dil · g_J · J_pdct · (tx² + ty²)`.

A **Mott guard** suppresses superconductivity when `g_t < _G_T_COHERENCE_MIN = 0.10` (i.e. $\delta \lesssim 0.053$): below this the Gutzwiller factor signals that the Zhang–Rice-singlet band is no longer coherent enough to support a physical SC gap, and the solver returns a non-metallic/non-SC result rather than a spuriously converged one.

### 6. B₁g Jahn–Teller Distortion and Anisotropic Hopping

The B₁g mode breaks the $x$–$y$ symmetry of the square lattice through an exponential (Harrison-type) hopping law:

```
tx(Q) = t0 · exp(+Q / lambda_hop)      # elongation along x → shorter bond → larger hopping
ty(Q) = t0 · exp(−Q / lambda_hop)      # compression along y → longer bond → smaller hopping
K_eff = K_lattice + ∂²F_ex/∂Q²
```

`K_lattice` is the bare phonon spring constant (primary input, never mutated); `∂²F_ex/∂Q²` is the exchange contribution to the stiffness (§11), negative when the condensate softens the mode. The effective hopping scale used for bandwidth estimates and Moriya damping is the RMS form `t_eff = √(0.5·(tx²+ty²))`.

The full multipolar exchange tensor is Q-dependent through both the overall B₁g channel opening (`exchange_channels()` returns `J_A1g_diag` and `J_B1g_scalar`, the latter proportional to `(tx²−ty²)`) and an orbital-selective asymmetry $\eta_J(Q)=\sqrt{J_{\Gamma_7}/J_{\Gamma_6}}$ between the two doublets, since $d_{xz}$ hops preferentially along $x$ and $d_{yz}$ along $y$; $\eta_J(0)=1$ exactly. The commutator $\|[B_{1g,\mathrm{op}}, H_{AFM}]\|/|\Delta_{CF}|$ (`blocking_ratio`) diagnoses how strongly the normal-state exchange field blocks the B₁g channel.

### 7. The Eg,2 Phonon Channel

Alongside the B₁g mode, the model carries an independent second vibronic channel of Eg,2 symmetry, built from the operator $L_yL_z+L_zL_y$ and projected into the same $\Gamma_6\oplus\Gamma_{7a}$ subspace exactly like `B1g_op`:

```
Eg2_op  = U4† · (Ly·Lz + Lz·Ly)_t2g · U4      # 4×4, Hermitian (complex in general)
```

with its own coupling constant `g_Eg2` (eV/Å), bare stiffness `K_lattice_Eg2` (eV/Å²), and distortion amplitude `Q_Eg2`, entering the BdG Hamiltonian, the free energy, and the Hessian on the same footing as the B₁g channel via `Eg2_16` (the 16×16 Nambu lift, mirroring `B1g_16`) and `Eg2_expectation()`. Unlike `B1g_op`, `Eg2_op` connects Kramers partners with an actual **spin-flip** structure ($\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7a}{\downarrow}$) rather than the spin-conserving structure of `B1g_op` — the two channels probe genuinely different multipolar sectors of the same $\Gamma_6\otimes\Gamma_7$ manifold.

At the current stage of the implementation, the exchange contribution to the Eg,2 stiffness and the B₁g–Eg,2 cross term vanish identically by Kramers symmetry, so `K_eff_Eg2` is left at its bare value `K_lattice_Eg2` (no exchange-driven softening is computed for this channel yet, in contrast to the fully renormalized `K_eff` for B₁g). The Eg,2 channel is therefore best read, in the current code, as a genuine second JT-active degree of freedom already wired through the Hamiltonian and observables, whose own self-consistent back-action on the lattice stiffness is not yet as developed as the B₁g channel's.

### 8. Dual B₁g Pairing Channels

Two symmetry-equivalent, **interorbital** B₁g pairing channels are carried simultaneously with independent strengths, exactly as required by the symmetry argument in §2:

- **Channel s** — on-site inter-orbital singlet ($\Gamma_6\otimes\Gamma_7\to B_{1g}$, constant $k$-space form factor):
  ```
  D_s = Δ_s · (|6↑⟩⟨7↓| − |6↓⟩⟨7↑|)
  ```
- **Channel d** — inter-sublattice d-wave ($\varphi(k)=\cos k_x-\cos k_y \to B_{1g}$ in $k$-space):
  ```
  D_d = Δ_d · φ(k) · (|A:6↑⟩⟨B:7↓| − |A:6↓⟩⟨B:7↑|)
  ```

Both channels feed into the same gap-equation infrastructure (§21) with independent Gutzwiller factors `g_Delta_s`, `g_Delta_d` (§5) and independent RPA-vertex projections; the dominant channel is identified post-convergence by the largest eigenvalue of the linearized 2×2 pairing kernel (§21).

### 9. The 16×16 BdG Hamiltonian (Doubled Unit Cell)

Nambu basis $\Psi=[\text{Particle}_A(4),\ \text{Particle}_B(4),\ \text{Hole}_A(4),\ \text{Hole}_B(4)]$, each block ordered $[\Gamma_6{\uparrow},\Gamma_6{\downarrow},\Gamma_{7a}{\uparrow},\Gamma_{7a}{\downarrow}]$:

```
BdG = ┌────────────────────┬─────────────────────┐
      │  H_A    T(k)       │  D_s      D_d        │   ← Particle sector
      │  T†(k)  H_B        │  D_d      D_s        │
      ├────────────────────┼─────────────────────┤
      │  D_s†   D_d†       │  −H_A*   −T*         │   ← Hole sector
      │  D_d†   D_s†       │  −T†*    −H_B*       │
      └────────────────────┴─────────────────────┘
```

The particle–hole off-diagonal blocks use the **transposed** (not Hermitian-conjugate) pairing operator, consistent with BdG particle–hole symmetry. Anisotropic hopping $T(k)=-2[t_x\cos k_x + t_y\cos k_y]\cdot I_4$ encodes the B₁g distortion.

**The JT coupling itself is $k$-dependent**, not a rigid on-site term: both the JT coupling and the anomalous Weiss field are modulated by the same $k$-dependent quasiparticle spectral weight $\beta^2(k)$ used for the charge-transfer downfolding (§4):

```
β²(k) = clip(1 − hybrid_scale·[t̄·(cos kx+cos ky) + δt·(cos kx−cos ky)]/Delta_CT,  0, 1)
        t̄ = (tx+ty)/2,  δt = (tx−ty)/2                      # wave_function_weight(tx, ty, kx, ky)

H += β²(k) · g_JT · Q · B1g_op                                # k-weighted JT coupling
H += β²(k) · Z · J_B1g_bare · F67s_mf · B1g_offdiag           # k-weighted anomalous Weiss field
```

$\beta^2(k)$ is the same second-order downfolding quantity that renormalizes $U_{pp}$ (§4): it is the fraction of the quasiparticle wavefunction that remains on the metal ion rather than the ligands at each $k$-point, so both channels that couple through the metal-ion orbital angular momentum operators ($B_{1g,\mathrm{op}}$) are naturally suppressed where the quasiparticle is more ligand-like. By contrast, the Eg,2 term (§7) is added **without** this $k$-weighting — `H += g_Eg2·Q_Eg2·Eg2_op` uniformly — reflecting its treatment, at the current stage of the implementation, as a spatially uniform (q=0) structural order parameter rather than one downfolded through the same multiband perturbation theory as the B₁g channel. In the particle sector this all enters as shown above; the hole sector carries the corresponding $-(\cdot)^T$, and exact Hermiticity is enforced after assembly.

`VectorizedBdG._build_H_stack` builds and diagonalizes this 16×16 matrix for the entire k-grid in one batched `numpy.linalg.eigh` call, reusing a pre-allocated buffer (`out=`) across SCF iterations to avoid repeated allocation.

The physical electron density is $\langle n_{i\sigma}\rangle=\sum_n |u_{n,i\sigma}|^2 f(E_n) + |v_{n,i\sigma}|^2(1-f(E_n))$ — both terms carry a positive sign, since $|v|^2(1-f)$ is the filled-band contribution from below the Fermi level.

### 10. Observables via `VectorizedBdG`

All thermal-average observables are extracted from a single batched diagonalization:

| Observable | Formula (schematic) | Role |
|---|---|---|
| **⟨B1g⟩** (full) | $\mathrm{Tr}[B_{1g,16}\cdot\rho_k]$, `/4` for Nambu+sublattice doubling | Hellmann–Feynman lattice force: $Q_{\mathrm{eq}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ |
| **⟨Eg2⟩** | same construction against `Eg2_16` | Hellmann–Feynman force for the Eg,2 channel |
| Magnetization | $\langle S_z\rangle$ via the exact `sz_op` weights | AFM order parameter `M` |
| `F67s_mf` | Gorkov Γ₆–Γ₇ singlet amplitude, `_compute_F67_singlet` | Anomalous Weiss-field back-action (§2, §19) |
| Density | $\sum_n[|u|^2 f + |v|^2(1-f)]$ / 4 | Chemical-potential control |
| Pairing s / d | on-site / inter-site $u^*v$ combinations | s-/d-channel gap-equation inputs |

The lattice update in the SCF loop uses the **full** $\langle\hat B_{1g}\rangle=\mathrm{Tr}[B_{1g,16}\cdot\rho]$, not a bare off-diagonal $\tau_x$ piece, because in D₂h `B1g_op` gains diagonal and spin-preserving components that are active even without SC — using only the off-diagonal piece would break Hellmann–Feynman consistency with `H_{JT}=g_{JT}\,Q\,B_{1g,\mathrm{op}}`. In D₄h the two expressions coincide exactly. Concretely, `B1g_expectation` contracts the already-diagonalized Nambu eigenvectors against `B1g_16` and weights by the occupation and the same $\beta^2(k)$ ZRS factor used in the Hamiltonian (§9), so the observable used to drive $Q$ stays consistent with what was actually put into $H$:

```python
diag_qp = np.einsum('kan,ab,kbn->kn', ec.conj(), B1g_16, ec).real     # a,b: 16 Nambu components; n: band index
exp_k   = np.einsum('kn,kn->k', diag_qp, f_n) * beta2_k               # per-k thermal average, β²(k)-weighted
B1g_exp = np.dot(k_weights, exp_k) / 4.0                              # /4: Nambu (particle–hole) × sublattice (A–B)
```

Summing over all 16 Nambu bands with plain $f(E_n)$ (not $1-f$) automatically covers the hole contributions too, since the hole-sector sign is already built into `B1g_16` itself (§3) and $f(-E)=1-f(E)$. `Eg2_expectation` mirrors this exactly but contracts against `Eg2_16` instead, and — consistent with §7/§9 — omits the $\beta^2(k)$ weighting, since the Eg,2 channel is currently treated as spatially uniform rather than downfolded through the same multiband perturbation theory.

### 11. Exchange Rigidity: ∂²F_ex/∂Q²

`compute_JT_rigidity_from_exchange` evaluates the exchange contribution to the B₁g stiffness from $F_{\mathrm{ex}}=\sum_{\alpha\beta}J_{\alpha\beta}(Q)\langle O_\alpha(Q)\rangle\langle O_\beta(Q)\rangle$ via the full product rule:

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

The orbital JT susceptibility $\chi_{QQ}=-\partial^2\Omega/\partial Q^2$ is evaluated in the SC state by central finite difference of the total free energy, divided by 4 to correct for the combined Nambu (particle–hole) and sublattice (A–B) doubling in the 16×16 BdG matrix. This SC-state $\chi_{QQ}$ is used **exclusively** for lattice-stability diagnostics (the G-matrix, §16); the pairing vertex itself always uses the normal-state ($\Delta=0$) susceptibilities, to avoid feeding the gap back into its own interaction.

### 14. Coupled Spin–JT RPA Vertex

The pairing vertex is built from a 2×2 coupled spin–orbital RPA in $[\mathrm{spin},\ \mathrm{JT{-}phonon}]$ channel space. The bare interaction matrix is **not diagonal**:

```
U = [[ J_eff,  U_SQ ],
     [ U_SQ,   V_JT ]],      U_SQ = r_MQ · √(J_eff · V_JT)
```

where `r_MQ` is the spin–JT cross-vertex extracted from the two-site cluster ED (§19) — the system genuinely has an effective spin–orbital cross-interaction at the local level, and setting `U_SQ = 0` would silently discard it.

**Bare susceptibilities.** $\chi_0(q)$ comes from the $\Delta=0$ BdG Hamiltonian via the static Lindhard formula, `_lindhard_bubble(sector_pairs, E_k, V_k, f_k, shift_idx, weights, η, kT)`, accelerated with `opt_einsum`. The normal-state sum runs over `_NORMAL_SECTOR_PAIRS` — 8 sector pairs covering the AA/BB (intra-sublattice) and AB/BA (inter-sublattice) particle and hole blocks — evaluated at $k$ and $k+q$ using the pre-built cyclic `shift_table` (§ Key Algorithms) rather than a second diagonalization. Each Lindhard term is additionally weighted by the same ZRS spectral weight $\beta^2(k)\beta^2(k+q)$ that modulates the JT coupling itself (§9), since the susceptibility bubble should only "see" the fraction of the quasiparticle that actually carries $B_{1g,\mathrm{op}}$/$S_z$ character. The static Lindhard function is real by time-reversal symmetry, so its imaginary part is discarded as roundoff, not physical information, after Hermiticity is enforced. Projections onto the physical channels:

```
χ_SS = Tr[Sz · χ₀[Γ6,Γ6] · Sz]     # spin–spin (dipole–dipole)
χ_SQ = Tr[Sz · χ₀[Γ6,Γ7]]          # spin–orbital cross (dipole–quadrupole), then divided by g_JT
χ_QQ = −∂²Ω/∂Q² / 4                # orbital JT stiffness [eV/Å²]
```

The cross-terms $\chi_{SQ},\chi_{QS}$ are exactly zero in the normal state at $Q=0$ (§2) and become nonzero once $Q>0$ opens the B₁g channel. `get_susceptibilities_normal` (normal state) and `get_susceptibilities_sc` (SC state, via the unified Nambu Lehmann sum against the pre-built `Sz_nambu`/`B1g_16`-derived vertex matrices) both apply a **PSD projection** of the $\big[\begin{smallmatrix}\chi_{SS}&\chi_{SQ}\\\chi_{SQ}&\chi_{QQ}\end{smallmatrix}\big]$ matrix (Higham nearest-PSD via eigenvalue clamping) to guard against Cauchy–Schwarz violations from numerical noise near the QCP.

**Vertex assembly (`_rpa_det`/`_rpa_vertex`).** Writing $\hat U\chi_0$ in the 2×2 channel basis as $\big[\begin{smallmatrix}a&b\\c&d\end{smallmatrix}\big]=I-\hat U\chi_0$,

```
a = 1 − (J·χ_SS + U_SQ·χ_QS)          b = −(J·χ_SQ + U_SQ·χ_QQ)
c = −(U_SQ·χ_SS + V_JT·χ_QS)         d = 1 − (U_SQ·χ_SQ + V_JT·χ_QQ)
det = a·d − b·c
```

`det` is floored in magnitude — not in sign — at `max(_MATH_EPS, 1e-4·‖(a,b,c,d)‖)`, guarding only against an exact-zero numerical accident without ever masking a genuine sign change. The channel-space inverse $(a,b,c,d)^{-1}$ is then contracted into the physical pairing vertex

```
V(q) = J_eff²·χ_SS^RPA(q) + V_JT²·χ_QQ^RPA(q) + J_eff·V_JT·[χ_SQ^RPA(q) + χ_QS^RPA(q)]
```

and finally hard-clamped to $\pm V_{\mathrm{cap}}$, with `V_cap = _RPA_V_CAP_ALPHA · max(8·max(|tx|,|ty|), J_eff)` (`_RPA_V_CAP_ALPHA = 2.2`, `_RPA_BW_FACTOR = 8` for the tight-binding bandwidth estimate) — a numerical overflow guard only; the sign and divergence character of $V(q)$ near the QCP are never altered, and `det<0` (past the QCP) is deliberately left untouched rather than capped, so the SCF is not artificially trapped away from a genuinely unstable regime.

**Two separately tracked determinants.** The vertex cache stores `det_q0` (the $q=0$, ferromagnetic-channel determinant) and `det_afm` (the $q=(\pi,\pi)$, AFM-channel determinant) independently. SCF adaptive mixing and convergence behavior respond to `det_afm`; `det_q0` guards separately against an accidental ferromagnetic divergence. Both are logged at convergence (`dFM=`, `dAFM=` in the iteration log).

**Sign-flip EMA guard.** When $|{\rm det\_afm}|<$ `_DET_SIGN_FLIP_SCALE = 0.05` and the d-wave vertex `V_d_scalar` would flip sign relative to its cached value, the update is blended continuously rather than switched:
$$w = \frac{1}{1+\exp\!\big(-k\,(|{\rm det\_afm}|/0.05 - 0.5)\big)}$$
so the blend weight shrinks toward 0 near the QCP (preserving genuine sign ambiguity there, where a real physical crossover may be in progress) and grows toward 1 away from it (suppressing pure numerical noise).

**q-resolved vertex diagnostics**, stored in the cache and logged whenever $V_d<0$:

| Key | Region ($|q|$ relative to $\pi$) | Role |
|---|---|---|
| `V_afm_mean` | $>0.70$ (`_V_AFM_Q_MIN`) | Mean $V(q)$ in the AFM region; $>0$ expected for spin-fluctuation-driven d-wave pairing |
| `V_fwd_mean` | $<0.35$ (`_V_FWD_Q_MAX`) | Mean $V(q)$ in the forward-scattering region; typically $<0$, cancelled by the d-wave form factor |
| `V_neg_frac` | — | Fraction of sampled $q$-points with $V(q)<0$; $>0.9$ flags a globally repulsive (unphysical) vertex |

**Moriya damping** regularizes the spin channel (doping- and $t/J$-dependent, saturating so it neither diverges at half-filling nor runs away at large $t/J$):

```
Γ_M = max(α_M, _ALPHA_MORIYA) · t_eff² / J_eff,   α_M = _MORIYA_C · f(δ) · sat(t/J)
f(δ) = δ/(δ+_MORIYA_DSAT) ∈ (0,1),   sat(t/J) = (t/J)/(_MORIYA_TJ_SAT + t/J) ∈ (0,1)
```

The full spin–quadrupole cross-susceptibility is also scanned over the whole BZ by `estimate_chi_SQ_q_full` (a 35×35 $q$-grid, called with `n_q=35` in the current `__main__`), producing the diagnostic plot described in [Output & Diagnostics](#output--diagnostics).

### 15. Linearized Gap Equation and λ_JT_kernel

The pairing kernel on the Fermi surface, $\Gamma_{ij}=g_\Delta\cdot\sqrt{dl_i/v_{F,i}}\cdot V(k_i-k_j)\cdot\sqrt{dl_j/v_{F,j}}$, is diagonalized in `solve_linearized_gap_equation`; `λ_max` is its largest eigenvalue with eigenvector $\varphi_{\max}$ (FS integration weights from `_fs_integration_weights`, a static method building the proper $dl/((2\pi)^2 v_F)$ measure with a floored $|v_F|$). C₄ symmetry-averaging is applied only near $Q\approx0$. The **JT-channel Rayleigh projection** $\lambda_{JT}^{\mathrm{kernel}}=\varphi_{\max}^T\,\Gamma_{JT}\,\varphi_{\max}$ measures how much of $\lambda_{\max}$ is carried specifically by the JT (as opposed to spin-fluctuation) component of $V(q)$ — a scalar, FS-resolved companion to the $q{=}0$ estimate `lambda_JT_sc` (§17). Signed s-/d-channel Rayleigh quotients (`lambda_bare_s`, `lambda_bare_d`) are computed on separate s- and d-projected FS grids with their respective channel-specific Gutzwiller factors.

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
| `lambda_JT_norm` | $\chi_{QQ}/K_{\mathrm{eff}}$ | normal state | $<1$ stable, $\ge1$ spontaneous JT |
| `lam_JT` | $g_{JT}^2\chi_{QQ}/K_{\mathrm{bare}}$ | normal state, pre-SCF (DE scout) | $>0.05$ for a non-trivially open window |
| `lambda_JT_sc` | $g_{JT}^2\cdot\max(-\chi_\tau^{\mathrm{net}},0)/K_{\mathrm{eff}}$ | SC state, post-SCF | $>0.05$ ⇒ SC-triggered JT active |

The SC–JT window itself is bounded by

```
K_spont = g_JT² / Delta_CF                                         (must have K_lattice above this)
K_SC    = g_JT² · max(−chi_tau_net, 0) / _LAMBDA_JT_VIABLE          (must have K_lattice below this)
```

so viability requires $K_{\mathrm{spont}} < K_{\mathrm{lattice}} < K_{SC}$, with `K_opt = √(K_spont·K_SC)` the geometric midpoint of the window. If `K_lattice ≤ K_spont` the lattice is already spontaneously unstable regardless of SC; if `K_lattice ≥ K_SC` it is too stiff for the condensate to ever soften it into the JT regime. `structural_ok` additionally requires $\lambda_{\min}(G_3)>0$ in the normal state — if that fails (`normal_unstable = True`), the window is reported non-viable regardless of the `K` bounds.

### 18. Variational Free Energy and the Cluster Decomposition

The total free energy splits, without double-counting, into an itinerant and a local piece:

```
F_total = F_BdG + F_cluster
```

This is a Luttinger–Ward/Baym–Kadanoff-style variational decomposition: `F_BdG` (`compute_bdg_free_energy`) covers the itinerant mean-field BdG spectrum plus the condensation-energy terms $|\Delta_s|^2/(g_{\Delta,s}V_s) + |\Delta_d|^2/(g_{\Delta,d}V_d) + (K_{\mathrm{eff}}/2)Q^2$; `F_cluster` (`compute_cluster_free_energy`) covers local quantum fluctuations from an exactly diagonalized two-site cluster. Gutzwiller factors handle kinematic Mott renormalization; the cluster ED handles irreducible vertex renormalization of $J_{\mathrm{eff}}$ only (never the susceptibility bubble itself); RPA handles the reducible ladder summation over the full BZ. These three levels are orthogonal — cluster-ED outputs a renormalized coupling that RPA then uses as an *input*, so there is no overlap between what each layer computes.

### 19. Two-Site Cluster: Quantum Fluctuations and Vertex Renormalization

Beyond the BdG mean field, a 2-site (A–B) cluster is exactly diagonalized every SCF iteration. The local single-particle Hamiltonians (`build_local_hamiltonian_for_bdg`) include $-\mu$, $\Delta_{CF}$, the AFM Weiss field, the JT coupling $g_{JT}\,Q\,B_{1g,\mathrm{op}}$, and — only once $Q\neq0$ — the anomalous Weiss field from `F67s_mf`. The cluster Hamiltonian is

```
H_cluster = H_A⊗I + I⊗H_B
          + J_bond_M_bare · (multi_op ⊗ multi_op)     [A1g magnetic exchange]
          + J_bond_Q_bare · (B1g_op ⊗ B1g_op)          [B1g orbital exchange, |Q| > 1e-8 only]
```

Two vertex renormalizations are extracted from the JT sector — the only sector with well-conditioned connected correlators, since the A1g magnetic channel's connected correlator vanishes identically in the collinear AFM ground state under Wick factorization and is instead taken from the analytic Gutzwiller result. A Boltzmann-weighted multivariate linear regression fits

```
evals_int ≈ const + J_Q · corr_Q + J_MQ · corr_MQ
corr_Q  = ⟨B_A B_B⟩ − ⟨B_A⟩⟨B_B⟩                                             (B1g connected correlator)
corr_MQ = ½[⟨O_A B_B + B_A O_B⟩ − ⟨O_A⟩⟨B_B⟩ − ⟨B_A⟩⟨O_B⟩]                    (spin–JT cross-correlator)
```

after subtracting the mean-field background state-by-state, excluding states with negligible Boltzmann weight, and requiring at least 3 valid points (otherwise `r_Q = r_MQ = 0`). Each slope is independently significance-tested against $H_0\!:\!J=0$ with a two-sided $t$-test (`_REGR_T_ALPHA = 0.05`); a continuous confidence factor smoothly shrinks insignificant slopes toward zero rather than hard-thresholding. The two resulting renormalization factors, both normalized by `J_bond_M_bare` and clipped to $[-2,+2]$:

- **`q_renorm` (`r_Q`)** — B₁g orbital-fluctuation vertex; rescales `J_B1g` in `exchange_channels()` and propagates through the rigidity, BdG-stack, μ-solver, and Hessian calls. Tracked via EMA (`_EMA_NEW_QRW = 0.38`).
- **`r_MQ`** — spin–JT cross-coupling vertex; this is exactly the microscopic origin of the bare RPA cross-vertex `U_SQ` in §14. Tracked via EMA (`_EMA_NEW_WEIGHT = 0.28`).

The regression is skipped when $|Q|<10^{-4}$, too few effective points remain, the sample variance is too small, or the design-matrix condition number exceeds $10^{10}$.

### 20. Chemical Potential: Newton with Analytic ∂n/∂μ

`_find_mu_for_density` solves $\langle n\rangle = 1-\delta$ by Newton's method using the analytic derivative $\partial n/\partial\mu=\sum_{k,n} w_k f(1-f)/kT\cdot(|u|^2+|v|^2)$ from the same BdG eigensystem, backtracking on step failure, with Brent's method as a guaranteed fallback bracket-and-bisect. Above `_MU_SC_DERIV_THRESH` total gap amplitude the analytic derivative (exact only for the pure normal-state branches) is replaced by a centered numerical derivative. The `(ev, ec)` pair from the μ-search is reused directly for the subsequent observable computation, avoiding a redundant diagonalization.

### 21. Gap Equations, Complex Phase, and the 2×2 Pairing Kernel

`VectorizedBdG.compute_gap_eq_vectorized` evaluates the gap equations over the full BZ, keeping the Fock sums **complex** rather than taking `abs(·)` before forming the new gap — because the BdG Hamiltonian is genuinely complex (SOC enters through $L_y\propto(L_+-L_-)/2i$ and $S_y$), the Nambu eigenvectors are complex at every $k$-point, and forcing a real magnitude at every iteration would erase the physical relative phase between $\Delta_s$ and $\Delta_d$ and destabilize convergence. A real, FS-averaged 2×2 pairing kernel

```
K_pair = [[ K11, K12 ], [ K12, K22 ]]      (s/d basis; K11 uses the JT-only vertex, K22 the full RPA vertex)
```

is built inline from the already-available FS grids at vertex-cache rebuild time; its dominant eigenvector `(v_s, v_d)` gives the SCF the optimal s/d hybridization direction, blended into the fixed-point gap update with weight `_ALPHA_MIX_2X2` — but only the real relative *magnitude* ratio is taken from `K_pair`; the actual complex phases `Delta_s_new/|Delta_s_new|`, `Delta_d_new/|Delta_d_new|` are always re-applied after blending, so the self-consistent phase is never silently overwritten. The same phase-freezing logic is used in `compute_hessian`, which fixes the converged phases before taking finite-difference probes.

A **V_d sign-flip guard** applies a sigmoid-weighted EMA to the d-wave vertex `V_d_scalar` whenever its sign flips between iterations (these flips are numerical — a genuine sign reversal cannot occur from an SCF-scale parameter change) — the blend weight is small near the QCP (preserving genuine sign ambiguity there) and grows toward a full update far from it. Both `V_s_scalar` and `V_d_scalar` are clamped to $[-V_{\mathrm{cap}}, +V_{\mathrm{cap}}]$ before caching.

### 22. Incommensurate AFM Nesting Check

Because the BdG Hamiltonian is fixed to commensurate AFM ordering at $Q_{AFM}=(\pi,\pi)$, `_scan_incommensurate_nesting` separately checks whether the normal-state spin susceptibility $\chi_{SS}$ would actually prefer a nearby incommensurate wavevector $q^*=(\pi,\pi-\delta q)$, scanning $\delta q\in[0,0.15\pi]$ at the converged $(M,Q,\mu)$. If the scan finds $\chi_{SS}(q^*)/\chi_{SS}(\pi,\pi)$ meaningfully above 1 (specifically, a maximum away from $\delta q=0$ beyond a small tolerance), `solve_self_consistent` automatically retries once with a softened AFM seed ($M\to0.85\,M$), guarded by the `_ic_retry` flag to prevent infinite recursion. This does not change the ordering wavevector used in the Hamiltonian itself — it only flags, and mildly compensates for, the possibility that the true instability sits away from $(\pi,\pi)$.

### 23. Temperature-Dependent Tc Estimates

Three independent estimates target different aspects of the transition, deliberately not sharing a single label:

- **Tc₁ — Allen–Dynes/McMillan-type spin-fluctuation formula:** $T_{c1}=(\omega_{SF}/D)\cdot\exp(-N\cdot(1+\lambda_{\max})/\lambda_{\max})$, with constants $D=$ `_MAD_DENOM`, $N=$ `_MAD_NUM`, $\omega_{SF}=J_{\mathrm{eff}}$ (paramagnon bandwidth), and $\lambda_{\max}$ from the linearized gap equation at the reference doping — a fast analytic estimate, not a full temperature scan.
- **Tc₂ — λ(T)=1 crossing:** `compute_lambda_vs_T` re-runs the linearized gap equation at each temperature on a **Δ=0, self-consistently relaxed normal-state** background (using `estimate_M0` as a warm-start rather than the converged $T{=}0$ SC value, which would otherwise artificially bias the bands away from the crossing); $T_{c2}$ is where $\lambda_{\max}(T)=1$. Non-monotone $\lambda(T)$ is detected and all crossings are logged.
- **Tc₃ — thermodynamic, first-order-aware:** `compute_Tc_thermodynamic` performs a single upward-heating temperature scan, warm-started from the converged $T\approx0$ SC+JT basin, comparing $F_{SC}$ against a separately relaxed normal-state free energy at every point. Because the effective Landau potential $F_{\mathrm{eff}}(\Delta) = a(T)\Delta^2 + [b - \gamma^2/(2K_{\mathrm{eff}})]\Delta^4 + \dots$ can have a negative quartic coefficient here, the transition can be genuinely first-order, and a naive cooling-from-$\Delta{\approx}0$ scan (which only finds the spinodal) can badly underestimate $T_c$. The routine returns both the thermodynamic crossing `Tc` and the spinodal collapse `Tc_spinodal`, the transition order, the gap jump `Delta_jump`, and — for near-second-order cases (`D_spinodal/Δ₀ < 0.15`) — a Ginzburg–Landau-refined spinodal from fitting $\Delta^2(T)=a(T-T_c)$ to points with $|\Delta|>2\,\mathrm{meV}$.

`compute_Tc_by_gap_suppression` (cooling-only spinodal search) is retained as an independent cross-check. The $2\Delta_0/k_BT_c$ ratio reported in the Tc block is computed against $T_{c3}$, the most physically complete of the three.

---

## Model Architecture

```
ModelParams  (dataclass, __post_init__ runs the SOC+CF diagonalization)
    ├── Primary inputs:  t_pd, u, lambda_soc, Delta_tetra, g_JT, K_lattice, lambda_hop,
    │                    g_Eg2, K_lattice_Eg2, Delta_CT, Delta_B1g_static, hybrid_scale,
    │                    Z, kT, tol
    ├── Derived scalars: Delta_CF, g7split, t0, U_dd, J_pdct, p_7, b1g_weight, b1g_diag_norm,
    │                    b1g_off_norm
    ├── Derived arrays:  sz_op (exact ⟨Sz⟩ per Kramers partner), multi_op, B1g_op, B1g_offdiag,
    │                    Eg2_op, _w6_xz/_w6_yz/_w6_xy, _w7_xz/_w7_yz/_w7_xy
    ├── Grid objects:    k_points, k_weights, N_k, shift_table (_NK×_NK×N_k int32 cyclic shift
    │                    table for arbitrary q), mbz_mask, k_weights_mbz
    └── Methods:         estimate_M0(), get_gutzwiller_factors(), exchange_channels(),
                         moriya_gamma(), effective_hopping_anisotropic(), wave_function_weight()

InstabilityInfo  (dataclass wrapping a 3×3 G-matrix eigendecomposition)
    ├── Fields:   G11, G22, G33, G_sJT, G_dJT, eigenvalues, evec_min, lambda_min
    ├── Booleans: jt_stable, s_stable, d_stable, full_stable
    └── Classifiers: instab_type, instab_dir, dominant_channel, severity, log_summary()
                     from_G3(G3) — classmethod constructor from a raw 3×3 array

_SolveState  (dataclass, mutable per-SCF-run state — never stored on self)
    ├── V_d_ema: Optional[float]         # persistent V_d sign-flip EMA
    └── _ema_kick_pending: bool          # doubles blend weight for one iter after a kick

RMFT_Solver
    ├── Initialization: __init__, _rebuild_orbital_operators, _get_vbdg, _get_chi0_norm_cache,
    │                   _reset_transient_state, _clone_solver_at_T, _full_rebuild
    ├── JT rigidity:    build_dHdQ_band_basis, compute_JT_rigidity_from_exchange
    ├── Susceptibilities: compute_chi_ss_with_infinitesimal_gap, B1g_expectation, Eg2_expectation,
    │                   _compute_chi_tau, _chi_QQ_matrix_elements, estimate_chi_SQ_q_full,
    │                   _compute_nambu_susceptibility, _diamagnetic_QQ_term,
    │                   get_susceptibilities_sc, get_susceptibilities_normal
    ├── RPA vertex:     _rpa_det, _rpa_vertex, _make_vertex_params
    ├── Gap equation:   _fs_integration_weights (static), solve_linearized_gap_equation,
    │                   scf_gap_diagnostics (coherence lengths, gap-ratio-relevant quantities)
    ├── Local H / μ:    build_local_hamiltonian_for_bdg, _find_mu_for_density, _compute_F67_singlet
    ├── Free energy:    compute_bdg_free_energy, compute_cluster_free_energy
    ├── SCF machinery:  _lambda_at_Q, _scf_jacobi_kick, _vertex_matrix_at_Q,
    │                   _classify_scf_dynamics, _anderson_mix, _mix
    ├── Main solve:     solve_self_consistent   ← the ~750-line Anderson-accelerated fixed point
    ├── Post-hoc:       _scan_incommensurate_nesting, compute_dF_dM_and_d2F, compute_hessian
    ├── Tc:             compute_Tc_by_gap_suppression, compute_Tc_thermodynamic,
    │                   compute_lambda_vs_T
    └── Diagnostics:    compute_G_instability, _unique_q_pairs (static), _get_fs_points

VectorizedBdG   (thin batched-LAPACK wrapper bound to one RMFT_Solver)
    ├── _build_H_stack               builds & Hermitizes the (N_k, 16, 16) BdG stack
    ├── compute_observables_vectorized      → M_stag
    └── compute_gap_eq_vectorized           → (Delta_s_out, Delta_d_out, v_s, v_d, vertex_cache)

OptimPoint      (plain result container: doping, 5D params, Delta_total, converged, score, …)

UnifiedBayesianOptimizer  (5D: Delta_tetra, lambda_soc, u, g_JT, t_pd)
    ├── GP infrastructure: _build_gp, _fit_gp, _obs_to_X, _register, _normalize/_denormalize
    ├── Sampling:          _lhs_sample, _make_phase_grid
    ├── Constraints:       _eval_constraints  (H1–H3 hard, S1–S5 soft — see Key Algorithms)
    ├── Phase 1 (DE):      run_de_phase
    ├── Phase 2 (GP seed): run_gp_seed_phase
    ├── Phase 3 (TuRBO):   _update_tr_center, _update_trust_region,
    │                      _expected_improvement_tr, run_turbo_phase
    ├── Phase 4 (refine):  run_local_refinement
    ├── Evaluation:        _scan_doping, _eval_one_doping, _g_fallback_score, _score
    └── Orchestrator:      optimize()   — runs all four phases, returns best_point/best_valid
```

---

## Key Algorithms

### SCF Loop (`solve_self_consistent`)

An Anderson(5)-accelerated fixed point over $(M, Q, \Delta_s, \Delta_d, \mu)$, per iteration:

1. Build and diagonalize the 16×16 BdG stack for the current $(M,Q,\Delta_s,\Delta_d,\mu)$.
2. If SC+JT are both active ($|\Delta_s|+|\Delta_d|$ above a small threshold), compute the Gorkov Γ₆–Γ₇ singlet amplitude `F67s_mf` and inject it as an anomalous Weiss field, then rebuild the BdG cache — this is the joint-$(\Delta,Q)$ activation loop of §2.
3. Update `K_eff` (and `K_eff_Eg2`) via `compute_JT_rigidity_from_exchange` — on iteration 0, or when $|\Delta Q|$ exceeds threshold, or every few iterations once $|\Delta M|$ has moved enough — rather than every single step, since the rigidity computation is comparatively expensive.
4. Solve the gap equations via the RPA vertex fixed point (§21 above); blend in the 2×2 pairing-kernel eigenvector direction with weight `_ALPHA_MIX_2X2 = 0.56` to prevent one channel from artificially locking out the other.
5. Update the two-site cluster free energy; extract `r_Q` and `r_MQ` (EMA-smoothed, §19) for the next iteration's JT-sector vertex renormalization; `J_eff` itself always comes directly from the analytic Gutzwiller factor.
6. Newton step for `M` (Levenberg–Marquardt-damped, trust-region-limited), blended with the linearly mixed BdG fixed point — `M` is deliberately **excluded** from the Anderson history and updated by this separate Newton/blend rule.
7. Adaptive Hellmann–Feynman update for `Q`: $Q_{\mathrm{out}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$, injected into the Anderson vector when the implied displacement is significant, on iteration 0, or on a periodic safety heartbeat (`_Q_UPDATE_PERIOD`); otherwise left untouched, consistent with the lattice's adiabatic timescale.
8. Anderson(5) mixing is applied jointly to $(Q,\ |\Delta_s|,\ |\Delta_d|)$ — a Tikhonov-regularized least-squares solve over the last 5 residuals; $\mu$ is then re-solved exactly (Newton+Brent) at the freshly mixed point rather than carried as an imperfectly converged fast variable, so its residual error cannot leak into the Anderson history.
9. Adaptive mixing rate: $\alpha_{\mathrm{eff}}=\alpha_0/(1+\Lambda_{\mathrm{inst}})$, where $\Lambda_{\mathrm{inst}}$ is an EMA of the worst current instability indicator ($\lambda_{\mathrm{pair}}$, $\lambda_{JT}$, $J\chi_{SS}$); $\alpha$ is halved and the Anderson history reset on divergence, and a **limit-cycle detector** (relative std of $|\Delta|$ over the last `_CYCLE_WINDOW` iterations $> $ `_CYCLE_THRESHOLD`) damps $\alpha$ and resets history if the SCF is oscillating rather than converging.

Convergence requires $\max(|\Delta M|,|\Delta Q|,|\Delta\Delta_s|,|\Delta\Delta_d|)<$ `tol` and density error $<10\cdot$`tol`. After convergence (or exhausting `_MAX_ITER`), `_classify_scf_dynamics` labels the trajectory as `converging`, `limit_cycle`, `first_order_jump`, `hysteretic`, or `stagnating` from the $|\Delta|$ history; the first-order-like classes trigger a multi-seed restart (several initial conditions, lowest free energy wins), consistent with the theory's first-order-transition expectation (see Physical Hypothesis). Post-convergence the solver runs the 3×3 Hessian test, the coherence-length/gap-symmetry diagnostics, the incommensurate-nesting check (§22), and assembles the full result dictionary consumed by the diagnostics block described in [Output & Diagnostics](#output--diagnostics).

### Vectorized BdG, Buffer Reuse, and the χ₀(q) Permutation Trick

`VectorizedBdG._build_H_stack` assembles the entire $(N_k,16,16)$ Hamiltonian stack with vectorized NumPy operations and diagonalizes it in a single `np.linalg.eigh` call per SCF iteration, reusing a pre-allocated `out=` buffer to avoid repeated allocation across hundreds of iterations; Hermiticity is enforced after assembly. The per-iteration eigensystem `(ev, ec)` is computed once and shared by observable computation, both pairing-channel gap equations, and the analytic $\partial F/\partial M$ (below).

The $q$-loop inside the RPA vertex construction never re-diagonalizes: the uniform k-grid (`endpoint=False`) is built in `ModelParams.__post_init__` so that for any $q=(n_x,n_y)\cdot2\pi/{\it \_NK}$, the $k+q$ grid is exactly a cyclic permutation of the $k$-grid. A precomputed `shift_table[nx, ny]` (shape `(_NK, _NK, N_k)`, `int32`) turns "shift by $q$" into a free index reorder,

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]     # index reorder — no extra LAPACK call
```

reusing the *same* $\Delta=0$ eigensystem for every $q$-point in one vertex-cache rebuild. `_get_chi0_norm_cache` additionally memoizes this normal-state $(E_k,V_k)$ across separate calls (susceptibilities, rigidity, incommensurate-nesting scan) that fall within the same iteration, keyed on $(M,Q,\mu,g_t,g_J,\delta)$ with independent tolerances tightened around the physically sensitive ones — e.g. the $M$ and $Q$ tolerances are the *same* `_M_THR_REL`/`_Q_THR_REL` thresholds used for RPA vertex-cache invalidation below, while $\mu$, $g_t$, $g_J$, and the doping are checked at $10^{-4}$, $10^{-4}$, $10^{-4}$, and $10^{-6}$ respectively — rather than one independent cache tolerance.

### Vertex Cache Invalidation

The RPA vertex cache is rebuilt when $M$ moves by more than an adaptive threshold scaled to $\sqrt{\max(|\det_{\mathrm{AFM}}|, \mathrm{floor})}$ (finer near the QCP, where the vertex is most sensitive), when $Q$ moves by more than 0.5% of `lambda_hop` (`_Q_THR_REL = 0.005`), when doping changes by more than 0.005, or unconditionally if the cache was not built from the normal state. There is no Δ-based invalidation — the vertex is *always* built from $\Delta=0$ by construction (§14). The cache stores the RPA determinant (both `det_q0` and `det_afm`), FS geometry (`fs_pts`, `vF_arr`, and separate s-channel FS arrays), the $q$-resolved diagnostics of §14, and the 2×2 pairing-kernel results, so repeated calls within one iteration reuse the same Fermi-surface sampling.

### Limit-Cycle Detection

Independent of the adaptive-$\alpha$ mechanism below, a dedicated oscillation check monitors $|\Delta|$ over a rolling window of `_CYCLE_WINDOW = 20` iterations; when the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.25`, the mixing rate is cut by `_CYCLE_DAMP_FAC = 0.45` and the Anderson history reset, which specifically targets the strongly nonlinear regime near the JT-activation onset where the $(Q,\Delta)$ feedback is most prone to overshoot.

### Anderson Mixing and the Jacobi Kick

Before the main SCF loop, `_scf_jacobi_kick` linearizes the coupled $(Q,\Delta)$ map analytically to estimate its leading eigenvalue $\lambda_+$, and uses this to choose the initial seed for $(M,Q,\Delta_s,\Delta_d)$ and the starting mixing rate $\alpha$ — this lands the iteration in the basin of the physically correct fixed point rather than an arbitrary starting guess, and adapts (with overshoot protection) as the system approaches or passes a supercritical regime. The Anderson solve itself uses a Tikhonov-regularized (`_ANDERSON_TIKHONOV = 1e-8`) normal-equation solve with a trust-region cap (`_ANDERSON_TRUST = 2.4`×) on the step size relative to simple mixing.

### Adaptive Q Update

$Q_{\mathrm{out}}^{\mathrm{raw}}=-(g_{JT}/K_{\mathrm{eff}})\langle B_{1g}\rangle$ is evaluated at **every** iteration, since $\langle B_{1g}\rangle$ is already available at zero extra cost from the same observable pass. It is only **injected into the Anderson vector**, however, when at least one of three conditions holds: the implied displacement $|Q_{\mathrm{out}}^{\mathrm{raw}}-Q|$ exceeds `_Q_THR_REL·lambda_hop`; it is the first iteration (seed); or a periodic safety heartbeat fires (`iteration % _Q_UPDATE_PERIOD == 0`, every 3 iterations). Otherwise $Q_{\mathrm{out}}=Q$ exactly — the Anderson residual for $Q$ is zero and the mixer leaves it untouched, respecting the lattice's slower (adiabatic) timescale relative to the electronic degrees of freedom without imposing a rigid blind period.

### Thread-Safety and Clone Protocol

Parallel evaluations (the optimizer's TuRBO batches, doping scans) clone the solver rather than share it:

```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s._full_rebuild()
```

`_full_rebuild()` is the single canonical post-mutation refresh: it re-runs `p.__post_init__()` (SOC+CF diagonalization), updates the bare stiffness `_K_bare`, rebuilds every orbital operator (`B1g_op`, `B1g_16`, `Eg2_op`, `Eg2_16`, `sz_op`, `multi_op`), and resets all transient caches (`_reset_transient_state`). Each clone owns its own `VectorizedBdG` and its own `_H_stack` buffer, so concurrent workers never alias each other's memory. At import time the module pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to prevent the BLAS backend from oversubscribing threads underneath an outer `ThreadPoolExecutor`.

### Analytic ∂F/∂M and ∂²F/∂M² from a Single Diagonalization

The Newton step for $M$ (SCF Loop step 6) needs both the free-energy gradient and curvature with respect to $M$, but these are obtained **without any extra diagonalization**: `compute_dF_dM_and_d2F` builds $\partial H/\partial M$ analytically (only the diagonal $J_{A1g}$ Weiss-field term contributes; the off-diagonal $J_{B1g}$ term has no direct diagonal piece with respect to $M$, though the anomalous `F67s_mf` term it carries must still be present in $H$ so the eigenvectors reflect the correct inter-band structure) and then applies first- and second-order perturbation theory directly to the **already-computed** eigenvalues/eigenvectors $(ev,ec)$ of the converged BdG stack:

```
∂F/∂M   =  ⟨n|∂H/∂M|n⟩ weighted by f(E_n)                                    (Hellmann–Feynman, diagonal)
∂²F/∂M² =  −Σ_{n} f'(E_n)·|⟨n|∂H/∂M|n⟩|²  +  Σ_{n≠m} [f(E_n)−f(E_m)]/(E_m−E_n) · |⟨n|∂H/∂M|m⟩|²
```

the second (off-diagonal, Kubo-like) term using a numerically safe $\tfrac{\Delta f}{\Delta E}\to -f'(E)$ limit at near-degenerate $E_n\approx E_m$. Because both derivatives come from the one BdG eigensystem already sitting in memory, this replaces what would otherwise be 2–3 additional full diagonalizations per SCF iteration with O(1) extra tensor contractions.

### Unified Bayesian Optimization (5D)

`UnifiedBayesianOptimizer` searches $(\Delta_{\mathrm{tetra}}, \lambda_{\mathrm{soc}}, u, g_{JT}, t_{pd})$ in four sequential phases, run by `optimize()`:

**Phase 1 — DE scout (`run_de_phase`).** `scipy.differential_evolution` over the normalized 5D cube, using **only** the cheap analytic G-matrix (`compute_G_instability`) — no full SCF is run in this phase. Each candidate is scored by `_eval_constraints`:

- **Hard constraints (H1–H3)**, any violation → `hard_fail=True`, penalty $\propto$ magnitude of violation, excluded from GP training:
  - H1: $\partial^2F/\partial Q^2|_{\Delta=0} > 0$ — no spontaneous normal-state JT.
  - H2: $J_{\mathrm{eff}}\cdot\chi_{SS} < 1$ — below the Stoner QCP (if past it, the gapped-susceptibility fallback `compute_chi_ss_with_infinitesimal_gap` is tried once before rejecting).
  - H3: $G_{22}>0$ — JT channel not self-crossing in the normal state.

  A cheaper pre-check rejects Mott-incoherent doping ($g_t<0.10$) before any G-matrix evaluation is attempted at all.

- **Soft constraints (S1–S5)**, weighted sum forms the DE penalty when hard-feasible:
  - S1 ($w{=}0.225$): $0<\lambda_{\min}(G_3)<$ `_DE_LAMBDA_MIN_OPT` — near-critical but not past the QCP.
  - S2 ($w{=}0.225$): monotonic reward for larger normal-state $\lambda_{\max,q=0}$, penalizing only near-divergence.
  - S3 ($w{=}0.180$): normal-state `lam_JT = g²·χ_QQ/K_bare` above `_DE_LAMBDA_JT_THRESH` — JT orbital response not vanishingly weak.
  - S5 ($w{=}0.10$): a $\tanh$ penalty on proximity to the $G_{22}=0$ spontaneous-JT boundary.
  - S4 ($w{=}0.270$): a parabolic-arch penalty on `lam_JT` computed **directly and cheaply** from the same G-matrix call, peaking (zero penalty) near the interior of the $(0,1)$ window and rising toward both boundaries — evaluated only when the partial penalty from S1/S2/S3/S5 is already below `_FEASIBILITY_THRESHOLD`, but with no separate expensive SCF or Fermi-surface-gradient sub-phase.

**Phase 2 — GP seed (`run_gp_seed_phase`).** The top-$k$ DE-feasible candidates are evaluated with a **full** SCF run each; the results seed an ARD Matérn-5/2 Gaussian process surrogate over the 5D space.

**Phase 3 — TuRBO (`run_turbo_phase`).** Trust-region GP-expected-improvement acquisition, batched via a thread pool; the trust region shrinks by `_TR_SHRINK = 0.65` on a failed batch and expands by `_TR_EXPAND = 1.35` on consecutive improvement. Trust-region state is mutated only from the main thread after each batch; observation registration is thread-safe.

**Phase 4 — local refinement (`run_local_refinement`, optional).** Dense random sampling in a small hypercube around the current global best, for final polishing.

**Scoring (`_score`)** is a three-tier multiplicative construction: hard Mott/incoherence/Stoner guards zero the score outright; a projection-quality factor discounts candidates where the $(J_{\mathrm{eff}}/\Delta_{CF})^2$ truncation error is non-negligible; smooth mechanism weights (a parabolic arch on `lambda_JT_sc` peaking near its interior, sigmoids on the post-convergence Hessian eigenvalue, on the SC-induced Q-mode softening, and on the cross-susceptibility $|\chi_{SQ}|$) multiply a `Tc`-proxy objective that itself rewards $\lambda_{\max}$, the sign of $\partial\lambda/\partial Q$, and proximity to an empirically chosen optimal $J\cdot\chi_{SS}\approx0.875$ (near-QCP but still metallic).

---

## Parameters

All energies in **eV**, lengths in **Å**. Defaults below are the values set in the `__main__` block.

### Primary Inputs (`ModelParams`)

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | $t_{pd}$ | 0.470 eV | $p$–$d$ hybridization integral; the single primary hopping input ($t_0=t_{pd}^2/\Delta_{CT}$ is derived) |
| `u` | $u$ | 24.0 | Dimensionless $U/t_0$ ratio; $U_{dd}=u\cdot t_0$ |
| `lambda_soc` | $\lambda_{SOC}$ | 0.180 eV | Atomic SOC constant on the $t_{2g}$ shell; sets the Γ₆–Γ₇ splitting together with `Delta_tetra` |
| `Delta_tetra` | $\Delta_{\mathrm{tetra}}$ | −0.050 eV | Axial (tetragonal) crystal field, $\Delta_{\mathrm{tetra}}\cdot L_z^2$; negative = $z$-axis compression |
| `g_JT` | $g_{JT}$ | 0.200 eV/Å | B₁g electron–phonon (JT) coupling |
| `K_lattice` | $K$ | 1.200 eV/Å² | Bare B₁g phonon spring constant; `K_eff` is computed at runtime |
| `lambda_hop` | $\lambda_{\mathrm{hop}}$ | 0.900 Å | Hopping-anisotropy decay length: $t(Q)=t_0\exp(\pm Q/\lambda_{\mathrm{hop}})$ |
| `g_Eg2` | $g_{Eg2}$ | 0.200 eV/Å | Eg,2-channel electron–phonon coupling (§7) |
| `K_lattice_Eg2` | $K_{Eg2}$ | 1.200 eV/Å² | Bare Eg,2 phonon spring constant |
| `Delta_CT` | $\Delta_{CT}$ | 2.100 eV | Charge-transfer gap; held fixed as a material-class constant |
| `Delta_B1g_static` | $\Delta_{\mathrm{ip}}$ | −0.014 eV | Static in-plane crystal field, $(L_x^2-L_y^2)$; drives the D₄h→D₂h crossover (§1, §3) |
| `hybrid_scale` | — | 4.000 | Downfolding coordination factor entering $U_{pp}$ (§4) |
| `Z` | $Z$ | 4 | Coordination number |
| `kT` | $k_BT$ | 0.007 eV | Temperature ($\approx$ 81 K) |
| `tol` | — | $10^{-4}$ | SCF convergence threshold |

### Derived Quantities (from `__post_init__`)

| Quantity | Origin | Description |
|---|---|---|
| `Delta_CF`, `g7split` | SOC+CF diagonalization | Γ₆–Γ₇ₐ gap and Γ₇ₐ–Γ₇ᵦ internal splitting — **not** free parameters |
| `sz_op` | exact $S_z$ diagonalization in each Kramers doublet | AFM/spin-vertex weights $[sz_{6\uparrow},sz_{6\downarrow},sz_{7\uparrow},sz_{7\downarrow}]$ |
| `multi_op` | built from `sz_op` | Effective multipolar spin operator shared by the cluster and BdG solvers |
| `p_7` | Γ₇ admixture in the Γ₆ eigenvectors | Interpolates `g_Delta_d` between `g_t` and `g_J` |
| `b1g_weight`, `b1g_diag_norm`, `b1g_off_norm` | `B1g_op` projection | Fraction of the B₁g operator that is off-diagonal (SC-triggered-only when $\to1$) |
| `_w6_xz/_w6_yz/_w6_xy`, `_w7_...` | eigenvector projections | $d_{xz}/d_{yz}/d_{xy}$ orbital weights feeding $\eta_J(Q)$ |
| `t0` | $t_{pd}^2/\Delta_{CT}$ | Effective $dd$ hopping |
| `U_dd`, `J_pdct` | §4 | Hubbard repulsion and ZSA charge-transfer superexchange |
| `k_points`, `k_weights`, `shift_table`, `mbz_mask` | $N_k=$ `_NK`² uniform grid | k-space infrastructure, including the cyclic shift table used for arbitrary-$q$ Lindhard sums |

### Module-Level Constants

The source file documents essentially every numerical-methods constant inline, each with its own physical or numerical justification — that block remains the single source of truth. The tables below reproduce the values relevant to interpreting solver behavior and output, organized by function; all were checked directly against the current source rather than carried over from any earlier documentation.

**Grid, iteration budget, general safety**

| Constant | Value | Role |
|---|---|---|
| `_NK` | 48 | k-points per direction (even, for commensurate $q_{AFM}=(\pi,\pi)$) |
| `_MAX_ITER` / `_MIN_ITER` | 700 / 4 | SCF iteration ceiling / floor before a convergence check is even attempted |
| `_MIXING` | 0.07 | Base Anderson mixing weight |
| `_CLUSTER_SIZE` | 2 | Sites in the exact-diagonalization cluster |
| `_MATH_EPS` | $10^{-9}$ | General division-by-zero guard |
| `_LINDHARD_CHUNK` | 128 | k-point batch size in the `opt_einsum` Lindhard loops |
| `_BZ_NORM` | $(2\pi)^2$ | BZ-area normalization in the FS arc-length measure $dl/((2\pi)^2 v_F)$ |
| `_Q_UNIQUE_SCALE` / `_PI_INT` | $10^5$ / 314159 | Integer scaling used to hash unique $q$-pairs without floating-point collisions |

**Unit conversion & Gutzwiller prefactors**

| Constant | Value | Role |
|---|---|---|
| `_KB_EV` | $8.617333\times10^{-5}$ eV/K | Boltzmann constant |
| `_EV_TO_K` | 11604.518 K/eV | $1/k_B$ |
| `_GW_G_J_PREFACTOR` | 4.0 | Numerator in $g_J=4/(1+\delta)^2$ (slave-boson / Kotliar–Ruckenstein, half-filling limit) |
| `_GW_G_T_NUMERATOR` | 2.0 | Numerator in $g_t=2\delta/(1+\delta)$ |

**AFM Newton solver ($M$-step control)**

| Constant | Value | Role |
|---|---|---|
| `_MU_LM` | 3.1 | Levenberg–Marquardt floor for the $M$ Newton step |
| `_ALPHA_HF` | 0.35 | Newton-vs-BdG-fixpoint blend weight for $M$ |
| `_TR_M_STEP_MAX` / `_TR_M_STEP_MIN_FLOOR` | 0.2 / $10^{-3}$ | Trust-region cap / absolute floor on $|\Delta M|$ per step |
| `_M_STEP_FLOOR_REL` / `_M_STEP_FLOOR_ABS` / `_M_STEP_FLOOR_M_MIN` | 0.005 / 0.002 / 0.010 | Step floor $=\max($ `_M_STEP_FLOOR_REL` $\cdot|M|,$ `_M_STEP_FLOOR_ABS`$)$, referenced against $\max(|M|,$ `_M_STEP_FLOOR_M_MIN`$)$ |
| `_M_J_EFF_FLOOR_FRAC` | 0.20 | QCP guard: $J_{\mathrm{eff}}$ floored at this fraction of $t_{\mathrm{eff}}$ to prevent $\Delta M\propto1/J_{\mathrm{eff}}\to\infty$ |

**Newton-kick overshoot protection** (used when re-seeding $M$/$\Delta$ mid-SCF)

| Constant | Value | Role |
|---|---|---|
| `_MODE_PULL_FRAC` | 0.30 | Fraction of $(M-M_{\mathrm{phys,est}})$ used as the kick pull in pure-SC/SC-JT mode |
| `_KICK_M_EXCESS_CTR` / `_KICK_M_STIFF_WIDTH` | 0.70 / 0.30 | Sigmoid center/width for $M$-kick overshoot suppression |
| `_KICK_JCHI_EXCESS_CTR` / `_KICK_JCHI_STIFF_WIDTH` | 0.70 / 0.30 | Sigmoid center/width for $J\chi_{SS}$-excess overshoot suppression |
| `_KICK_REDUCTION_AMP` | 0.35 | $M_{\mathrm{kick}}\times(1-\text{this}\times\text{excess})$ |
| `_KICK_LAMBDA_SC_THR` / `_KICK_LAMBDA_SC_WIDTH` | 5.00 / 15.00 | Supercritical-$\lambda_{\max}$ threshold / pull-fraction denominator |
| `_KICK_PULL_CAP` | 0.60 | Maximum pull fraction |
| `_KICK_BOOST_AMP` | 3.00 | $\Delta$-kick boost $=1+\text{this}\times\lambda_{\mathrm{excess}}/(1+\lambda_{\mathrm{excess}})$ |
| `_KICK_SC_LOG_SCALE` | 5.00 | Supercritical mixing log-scale, $\log_{10}(\lambda/\text{this})$ |
| `_KICK_M_CLIP_LO` / `_KICK_M_MOTT_CLIP_LO` / `_KICK_M_CLIP_HI` | 0.02 / 0.05 / 0.45 | Hard clips on $M_{\mathrm{kick}}$ (the Mott-boundary floor is higher, to avoid $M\to0$ collapse) |

**Chemical potential (Newton + Brent)**

| Constant | Value | Role |
|---|---|---|
| `_DEN_DERIV_FLOOR` | $10^{-12}$ | Floor on $\partial n/\partial\mu$ |
| `_BRENTQ_TOL` | $10^{-5}$ | Brent bracketing tolerance |
| `_MU_NEWTON_MAXIT` / `_MU_BACKTRACK_MAX` / `_MU_BACKTRACK_FLOOR` | 20 / 6 / 0.05 | Newton iteration budget / max step-halvings / minimum backtrack damping before falling back to Brent |
| `_MU_DENSITY_TOL` | $10^{-6}$ | $|n(\mu)-n_{\mathrm{target}}|$ convergence tolerance |
| `_MU_SC_DERIV_THRESH` | $10^{-4}$ eV | Gap amplitude above which the analytic $\partial n/\partial\mu$ (exact only at $\Delta=0$) is replaced by a centered numeric derivative |

**Lindhard broadening & Fermi-surface sampling**

| Constant | Value | Role |
|---|---|---|
| `_ETA_T_FRAC` | 0.10 | Normal-state broadening $\eta=$ this $\times\,kT$ |
| `_ETA_DELTA_FRAC` | 0.03 | SC-state broadening increment $\propto|\Delta|$ |
| `_ETA_GRID_FLOOR` | 0.002 | Broadening floor (units of $t_0$), guards k-grid aliasing |
| `_FERMI_ARG_CLIP` / `_ENTROPY_CLIP` | 100.0 / $10^{-12}$ | Numerical clips in $f(E)$ and $-f\ln f$ |
| `_FD_MASK_DF` / `_FD_MASK_DE` / `_FD_MASK_DE8` | $10^{-12}$ / $10^{-6}$ / $10^{-8}$ | Degenerate-denominator masks in the $\chi_0$ Lehmann sums (the tightest, `_FD_MASK_DE8`, is used in the $\partial^2F/\partial M^2$ off-diagonal term) |
| `_VF_FLOOR` / `_VF_FLOOR_TIGHT` | $10^{-4}$ / $10^{-5}$ | Fermi-velocity floors (the tighter one guards the $dl/v_F$ arc-length weight specifically) |
| `_N_FS` | 130 | Fermi-surface k-points sampled in the vertex $q$-loop |
| `_FS_SAMPLING` / `_FS_WEIGHT_THR` | 2.8 / 0.01 | Thermal window (in units of $kT$) around $E_F$ for FS selection / minimum relative thermal weight kept |
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
| `_DET_DEPTH_CAP` / `_DET_JUMP_HALF_SCALE` / `_JUMP_CAP_FLOOR` | 5.0 / 0.5 / 1.05 | Past-QCP gap-jump cap: exponential suppression depth cap / decay rate / minimum allowed cap |
| `_DET_SIGN_FLIP_SCALE` | 0.05 | $|\det_{\mathrm{afm}}|$ sigmoid midpoint for the $V_d$ sign-flip EMA guard |
| `_EMA_SIGN_FLIP_W_MIN` / `_EMA_SIGN_FLIP_SLOPE` | 0.20 / 6.0 | Minimum blend weight / sigmoid steepness in the sign-flip guard |
| `_V_PREV_SIGN_FLOOR` | $10^{-6}$ | $|V_{d,\mathrm{prev}}|$ below this is treated as zero (sign-flip check skipped) |
| `_VMAT_LOW_VAR_FRAC` | 0.10 | $\mathrm{std}(V)/|\mathrm{mean}(V)|$ below this ⇒ `⚠low-var` flag |
| `_VERTEX_DIAG_MIN_FS` | 10 | Minimum FS points required before vertex-structure diagnostics are considered reliable |
| `_V_AFM_Q_MIN` / `_V_FWD_Q_MAX` | 0.70 / 0.35 | $|q|/\pi$ cutoffs defining the AFM / forward-scattering regions in `V_afm_mean`/`V_fwd_mean` |
| `_V_CUT` | 20.0 | Pairing-vertex near-divergence detector threshold |
| `_JCHI_HARD_REJECT` | 2.0 | $J\chi_{SS}$ above this ⇒ hard-rejected (deeply AFM, SC impossible) |

**Moriya damping**

| Constant | Value | Role |
|---|---|---|
| `_MORIYA_C` | 0.21 | Prefactor in $\alpha_M=C\cdot f(\delta)\cdot\mathrm{sat}(t/J)$ |
| `_MORIYA_DSAT` | 0.30 | Doping saturation scale in $f(\delta)$ |
| `_MORIYA_TJ_SAT` | 1.0 | Padé half-saturation scale for $t/J$ |
| `_ALPHA_MORIYA` | 0.02 | Hard floor on the damping prefactor |

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
| `_MODE_FRAC_DOMINANT` / `_MODE_FRAC_MIXED` | 0.60 / 0.30 | Thresholds classifying a pure-channel vs. mixed SC-triggered-JT regime |
| `_Q_UPDATE_PERIOD` | 3 | Heartbeat period (iterations) for the Hellmann–Feynman $Q$ update |
| `_ALPHA_MIX_2X2` | 0.56 | Blend weight: 2×2 pairing-kernel eigenvector vs. fixed-point gap update |
| `_JT_ACT_THR` | 0.04 | Condensate-induced Γ₆–Γ₇ mixing threshold for the "JT-active" classification |
| `_K_EFF_M_THR` | 0.02 | $|\Delta M|$ threshold triggering a rigidity (`K_eff`) recompute |
| `_G_T_COHERENCE_MIN` | 0.10 | Mott guard: minimum coherent $g_t$ |
| `_DELTA_ABS_FLOOR` | $10^{-3}$ eV | $|\Delta|$ below this bypasses the jump limiter (free seed-growth phase) |
| `_BCS_SEED_FRACTION` | 0.09 | Initial cold-start $\Delta$ seed, as a fraction of $t_{\mathrm{eff}}$ |
| `_DELTA_JUMP_CAP` | 5.0 | Maximum $|\Delta_{\mathrm{new}}|/|\Delta_{\mathrm{old}}|$ ratio per iteration |
| `_DQ_FS_VERTEX` | 0.03 Å | Minimum finite-difference step for $\partial\lambda/\partial Q$ on the FS |
| `_IC_RATIO_FLOOR` / `_IC_RATIO_CAP` | 1.05 / 3.00 | Bounds on the cluster-ED inter-channel correction ratio |

**Coherence length / gap classification**

| Constant | Value | Role |
|---|---|---|
| `_XI_NODAL_MIN` | 2.0 | Minimum $\xi/a$ (nodal) for BCS-side quasiparticle coherence |
| `_ORBITAL_SEL_FRAC` | 0.15 | $|\xi_{\Gamma_6}-\xi_{\Gamma_7}|/\xi$ threshold for "orbitally selective" pairing |
| `_DOPING_MOTT_FLOOR` | 0.01 | $|\delta|$ below this ⇒ at/near the Mott insulator; SCF skipped |

**Two-site cluster WMLR regression**

| Constant | Value | Role |
|---|---|---|
| `_REGR_EPS` / `_REGR_VAR_MIN` | $10^{-12}$ / $10^{-9}$ | Zero-guard at denominators / minimum variance before the regression is trusted |
| `_REGR_T_ALPHA` | 0.05 | Two-sided significance level for the `r_Q`/`r_MQ` $t$-test |
| `_EMA_NEW_WEIGHT` / `_EMA_NEW_QRW` | 0.28 / 0.38 | EMA new-sample weights for `r_MQ` / `q_renorm` (the orbital channel is tracked with a faster-responding EMA) |

**Tc / Ginzburg–Landau / BCS ratio**

| Constant | Value | Role |
|---|---|---|
| `_BCS_RATIO_STRONG` / `_VSTRONG` / `_EXOTIC` | 3.8 / 5.0 / 7.0 | $2\Delta_0/k_BT_c$ thresholds for strong / very-strong / exotic coupling |
| `_GL_DELTA_MIN` | 2 meV | $|\Delta|$ floor for points admitted to the GL fit |
| `_GL_MIN_PTS` / `_GL_MAX_PTS` | 2 / 4 | Minimum / maximum recent stable-SC points used in the GL regression |
| `_GL_TC_MARGIN` | 0.05 | Maximum relative deviation $|T_{c,GL}-T_{\mathrm{spinodal}}|/T_{\max}$ to accept the GL result |
| `_GL_SPINODAL_JUMP` | 0.15 | $D_{\mathrm{spinodal}}/\Delta_0$ below this ⇒ GL extrapolation treated as reliable (small first-order jump) |

**AFM warm-start (`estimate_M0`)**

| Constant | Value | Role |
|---|---|---|
| `_M0_STONER_AMP` | 0.18 | Stoner amplitude in the analytic $M_{\mathrm{stoner}}$ estimate |
| `_M0_PRIOR_SLOPE` / `_M0_PRIOR_REF` | 0.40 / 0.06 | Slope and reference doping (AFM-dome optimum) of the empirical $M_{\mathrm{prior}}$ |
| `_M0_DELTA_C` | 0.23 | Critical doping above which the Stoner estimate decays linearly to zero |
| `_M0_W_SC_LAMBDA_WIDTH` / `_M0_W_SC_CAP` | 15.0 / 0.75 | Width / cap of the SC-state blend weight between the two estimates |
| `_M0_W_DOPING_SAT` | 0.20 | Doping scale at which the prior-dominant blend saturates |
| `_M0_STONER_CLIP_LO/HI`, `_M0_PRIOR_CLIP_LO/HI` | 0.05–0.20, 0.08–0.22 | Physical clipping ranges on the two component estimates |
| `_M0_S_CLIP_MAX` | 5.0 | Upper clip on the Stoner parameter $S=J\cdot N_0$ |

**Bayesian optimizer — DE-scout scoring**

| Constant | Value | Role |
|---|---|---|
| `_W_LMIN / _W_LEFF / _W_LJT / _W_DLAM / _W_G22M` | 0.225 / 0.225 / 0.180 / 0.270 / 0.10 | Soft-constraint weights S1–S5 (sum to 1.0) |
| `_FEASIBILITY_THRESHOLD` | 0.25 | Partial-penalty ceiling for "feasible" classification |
| `_DE_LAMBDA_MIN_OPT` | 0.15 | S2 sigmoid center (weak-pairing boundary) |
| `_DE_LAMBDA_MAX_REWARD` | 4.0 | $\lambda_{\max}$ above this is penalized (near-divergent) |
| `_DE_LAMBDA_JT_THRESH` | 0.05 | S3 threshold on the normal-state `lam_JT` |
| `_BO_ARCH_DENOM` | 0.2025 $(=0.45^2)$ | Parabolic-arch normalization so S4 peaks at 1 for `lam_JT` = 0.5 |
| `_BO_MAX_WORKERS` | 6 | Thread-pool ceiling for parallel GP-seed / TuRBO evaluation |

**Bayesian optimizer — full scoring function**

| Constant | Value | Role |
|---|---|---|
| `_BO_OPT_JCHI` / `_BO_SIG_JCHI` | 0.875 / 0.15 | Center / width of the Gaussian gate rewarding near-QCP-but-metallic $J\chi_{SS}$ |
| `_BO_JCHI_FLOOR` / `_BO_JCHI_NOISE` | 0.3 / 0.05 | Score floor when $J\chi$ is unavailable / noise threshold below which the floor applies |
| `_BO_JCHI_GAPPED_CAP` | 0.98 | Gapped-state $J\chi_{SS}$ must stay below this to count as safely metallic |
| `_BO_W_STONER_BAD` | 0.20 | Score weight applied when the Stoner criterion is violated |
| `_SCORE_SOFTENING_SIG` | 0.05 | Sigmoid width for the JT-softening reward |
| `_BO_W_HESSIAN_FLOOR` | 0.30 | Floor for the Hessian/kernel weight when data is missing |
| `_BO_W_LJT_OVR_SAT` | 0.10 | Weight applied when `lambda_JT_kernel` $\ge1$ (Rayleigh-quotient over-saturation) |
| `_BO_LJT_KERNEL_SIG` | 10.0 | Sigmoid steepness for the kernel-based JT weight |
| `_BO_G22_MARGIN_CTR` / `_BO_G22_MARGIN_W` | 0.25 / 0.15 | Center / width of the $G_{22}$-margin sweet-spot sigmoid |
| `_BO_SC_HESS_SIG` | 0.05 eV | Sigmoid width for the post-convergence Hessian eigenvalue reward |
| `_BO_G_FALLBACK` / `_BO_SIGMOID_W` / `_BO_SPONT_JT_PEN` | $5\times10^{-3}$ / 0.30 / 0.05 | Overall scale / sigmoid width / penalty floor in the no-gap G-matrix-proxy fallback score |

**TuRBO**

| Constant | Value | Role |
|---|---|---|
| `_TR_SHRINK` / `_TR_EXPAND` | 0.65 / 1.35 | Trust-region contraction on a failed batch / expansion on consecutive improvement |

### 5D Optimization Search Bounds


| Parameter | Bounds |
|---|---|
| `Delta_tetra` | (−0.09, −0.03) eV |
| `lambda_soc` | (0.18, 0.34) eV |
| `u` | (10.0, 20.0) |
| `g_JT` | (0.11, 0.24) eV/Å |
| `t_pd` | (0.40, 0.60) eV |

### SC+JT Coexistence Conditions

Four conditions, checked by `compute_G_instability` together with the SC-JT window logic of §17, must hold simultaneously for the mechanism to be viable at a given parameter point:

1. **Metallicity:** the AFM gap does not swallow the Fermi surface ($h_{AFM}$ small compared to the effective bandwidth).
2. **Mott coherence:** $g_t \ge 0.10$ — the Zhang–Rice-singlet band is coherent enough to support pairing.
3. **Normal-state JT stability:** $K_{\mathrm{eff}}>0$ and $\lambda_{\min}(G_3)>0$ at $\Delta=0$ — no spontaneous instability of any kind.
4. **SC-triggered regime:** `lambda_JT_sc` $=g_{JT}^2\cdot\max(-\chi_\tau^{\mathrm{net}},0)/K_{\mathrm{eff}} >$ `_LAMBDA_JT_VIABLE` $=0.05$, i.e. the condensate genuinely softens the B₁g mode past the viability threshold.

---

## Installation & Usage

### Requirements

```bash
pip install numpy scipy matplotlib scikit-learn opt_einsum threadpoolctl
```

### Running

```bash
python Quantum_AFM-multipolar_Jahn-Teller.py
```

On startup, the current `__main__` block does the following, in order:

1. **Parameter setup and SOC+CF diagonalization.** `ModelParams(...)` is constructed with the defaults listed in [Parameters](#parameters) and `__post_init__` runs the SOC+CF diagonalization (Γ₆/Γ₇ₐ/Γ₇ᵦ identification, `Delta_CF`, `sz_op`, `p_7`, `b1g_weight`, k-grids, orbital operators). `RMFT_Solver(params)` is then built from those parameters.
2. **Doping setup.** `target_doping = 0.31`, with a symmetric ±20% scan margin (`doping_margin`) defining `min_doping`/`max_doping` — these bounds are only used if the optimizer runs (step 3); a floor tied to `_G_T_COHERENCE_MIN` prevents the lower bound from entering the Mott-incoherent region.
3. **Optional 5D Bayesian optimization.** Controlled by the boolean `need_optimization` (**default `False`**). When `True`, `UnifiedBayesianOptimizer.optimize()` runs the full DE→GP-seed→TuRBO→local-refine pipeline (§ Key Algorithms) over the bounds in [Parameters](#parameters), and the best-found $(\Delta_{\mathrm{tetra}},\lambda_{SOC},u,g_{JT},t_{pd})$ are written back into `params` before continuing.
4. **Reference SCF.** `solver.solve_self_consistent(target_doping, ...)` is run once at `target_doping`, producing the converged $(M,Q,\Delta_s,\Delta_d,\mu)$ used by every diagnostic that follows.
5. **G-matrix diagnostics.** `compute_G_instability(target_doping, M)` at the self-consistent $M$, logging the normal-state instability classification (§16).
6. **Post-SCF diagnostics** (only if the reference SCF succeeded): RPA vertex decomposition, linearized-gap-equation channel decomposition, coherence lengths, the SC Hessian and SC-JT-triggering confirmation, the Stoner ratio, a full-BZ $\chi_{SQ}(q)$ scan (`estimate_chi_SQ_q_full`, which also produces a diagnostic plot — see below), the SC-JT coexistence window (§17), and the three Tc estimates (§23).

`need_optimization` is the single flag controlling whether the optimizer runs before the reference SCF; leave it `False` for a fast single-point evaluation at fixed parameters, or set it `True` to search the 5D space first.

---

## Output & Diagnostics

All output is a structured, thread-safe log stream (`_scf_log`, tagged by stage: `RMFT-INIT`, `SCF-INIT`, `SCF`, `SCF-RES`, `G-MATRIX`, `TC-PRELIM`, `TC-THERMO`, …) rather than a GUI; the only graphical output produced by the current script is the diagnostic plot described below.

### Iteration Log

Each logged SCF step reports the current order parameters, the effective exchange and mixing rate, and warning flags for degenerate or numerically marginal vertex structure:

```
[SCF] δ=…  iter/max  conv=…  M=…  Q=…  |Δ|=…  J_eff=… eV  r_Q=…  r_MQ=…  mu=…
      dFM=…  dAFM=…  V_s=…  V_d=…  [⚠low-var] [⚠same-sign]
      Γ_M=…  α=…  B1g=…  F67s=…  [regime]  …s/it
```

At convergence, an `SCF-RES` block reports the converged order parameters, density, $\mu$, free energies, `F67s_mf`, `q_renorm`/`r_MQ`, the AFM/RPA determinant, the JT-active flag, the SCF-dynamics regime classification (§ Key Algorithms), the s-/d-channel decomposition of $\lambda_{\max}$, `lambda_JT_sc`, `lambda_JT_kernel`, $\partial\lambda_{\mathrm{pair}}/\partial Q$, the post-convergence Hessian's SC-triggered-JT confirmation, coherence lengths, the $\chi_\tau$ breakdown (including its reliability weight), the SC-JT window verdict, and the incommensurate-nesting scan result.

### G-Matrix Block

Logged separately from the SCF result (evaluated at the self-consistent $M$ but in the normal, $\Delta=0$ state): the exchange scale $h_{AFM}$, the pairing susceptibilities $\chi_{\Delta\Delta}$ in each channel, the normal-state pairing eigenvalue `lambda_eff`, the normal-state lattice stability ($K_{\mathrm{eff}}$, $\chi_{QQ}$, `lambda_JT_norm`, $\partial^2F/\partial Q^2|_{\Delta=0}$), and the full `InstabilityInfo.log_summary()` classification (§16).

### RPA Vertex and SC-JT Window

Following the G-matrix block, if the reference SCF converged: the FS-averaged RPA vertex decomposition into spin / JT / cross contributions; the linearized-gap-equation channel decomposition; the coherence-length summary (flagging orbital-selective pairing when $\Gamma_6$ and $\Gamma_7$ channels have meaningfully different $\xi$); the SC Hessian's smallest eigenvalue and the FS-resolved $\partial\lambda/\partial Q$; the Stoner ratio $J_{\mathrm{eff}}\chi_{SS}$ with a QCP-proximity classification; the `K_eff` path from the normal to the SC state (with a term-by-term breakdown of the four contributions in §11); the $\chi_\tau$ breakdown; and the SC-JT window verdict (§17), including the current `K_lattice`'s position within the viable window as a percentage.

### χ_SQ(q) Full-BZ Scan Plot

`estimate_chi_SQ_q_full` (called with `n_q=35` in the current `__main__`) produces the one graphical output of the script: a `matplotlib` figure with a **2×3** grid of panels (`plt.subplots(2, 3, ...)`) showing the spin–quadrupole cross-susceptibility $\chi_{SQ}(q)$ over the Brillouin zone — separately for the normal and SC states — together with the symmetry-consistency check described in §2 (`symmetry_ok`: the normal-state peak should be numerically indistinguishable from zero). This replaces an older, much larger doping-sweep phase-diagram figure that is no longer part of the current pipeline (see below).

### Tc Block

```
[TC-PRELIM]  Tc₁(Allen–Dynes-SF): λ_max=…  ω_SF(J_eff)=… meV  → … meV (… K)
[TC-PRELIM]  Tc₂(λ=1 crossing)=… meV  slope=… meV⁻¹  n_crossings=…
[TC-THERMO]  Tc₃(thermo)=… meV (… K)  Tc_sp(spinodal)=… meV  order=…  Δ_jump=… meV
[TC-THERMO]  2Δ₀/kTc=…  [BCS-like | strong | very-strong | exotic / non-phononic]  (from Tc₃)
[TC-THERMO]  Tc uplift (Tc₃ vs Tc_sp): …%  [first-order dominant | weakly first-order | effectively second-order]
```

### What is *not* currently produced

Earlier versions of this pipeline generated a multi-row doping-sweep phase diagram (order parameters, density of states, SCF convergence traces, free-energy and Gutzwiller-factor histories, and $T_c(\delta)$, optionally extended with Bayesian-optimization progress panels when the optimizer had run). **The current `__main__` block does not perform a doping sweep and does not produce this figure** — it evaluates a single reference doping point (plus, optionally, the 5D optimizer, which internally scans doping only as part of its own scoring and does not plot it). Reconstructing a doping-swept phase diagram is straightforward by looping `solve_self_consistent` over a `doping` array and collecting the fields already present in its result dictionary, but this is not currently wired up as part of the script.

---

## Known Limitations

The framework makes a number of physically motivated approximations. The table below merges the limitations identified in the theoretical write-up with implementation-level caveats found in the current code.

| Approximation | Impact |
|---|---|
| No Pauli exclusion between cluster sites | Mild overestimate of AFM correlations; controlled by the Newton/BdG-fixpoint blend weight `_ALPHA_HF` |
| No charge-transfer fluctuations $\langle n_An_B\rangle$ | Negligible when the mean-field exchange scale is large compared to the hopping |
| Static phonon ($Q$ is a mean field) | Zero-point quantum lattice fluctuations are neglected; the JT frequency is derived from $K_{\mathrm{eff}}$, not an independent input |
| 4×4 BdG truncation (Γ₆⊕Γ₇ₐ only) | Valid when $\Delta_{CF}\gg kT$ and $\Gamma_{7\mathrm{split}}/\Delta_{CF}\ll1$; monitored via the $(J_{\mathrm{eff}}/\Delta_{CF})^2$ projection-quality penalty in the optimizer scoring |
| No spatial fluctuations | Cannot describe a pseudogap, stripe order, or phase separation |
| RPA static ($\omega=0$) | Dynamical vertex corrections are absent |
| `K_eff` update conditional | Recomputed only when $M$ or $Q$ have moved enough (§ Key Algorithms); $Q$'s back-action on the exchange rigidity is approximate during the SCF transient, though exact at convergence |
| $\chi_\tau$ evaluated only post-convergence in some diagnostics | The fully self-consistent back-action of $Q$ on $\chi_\tau$ within the SCF loop itself is not iterated to convergence at every step |
| G-matrix evaluated at $\Delta=0$ only | Diagnoses normal-state stability; the actual SC-triggered-JT scenario is confirmed independently via the post-SCF Hessian ($\lambda_{\min}<-kT$) |
| $\partial\lambda_{\mathrm{pair}}/\partial Q$ at a frozen Fermi surface | FS geometry is evaluated at a fixed $Q$ rather than self-consistently re-resolved; a fully SC-state Bogoliubov–Lindhard version would be more expensive |
| $\delta\chi_\tau$ baseline subtraction approximate in D₂h | The normal-state B₁g response at finite `Delta_B1g_static` is estimated at $\Delta=0$; small residual D₂h corrections to the baseline are neglected |
| `chi_tau_weight` partial suppression | When the Richardson extrapolation only agrees at the finer step pair (`chi_tau_weight = 0.5`), the SC-JT feedback may still be over- or under-estimated near a first-order boundary |
| SCF-dynamics regime classification | `first_order_jump` and `hysteretic` trigger a multi-seed restart (lowest free energy wins); `limit_cycle` only damps the mixing rate; the classification is heuristic, based on the shape of the $|\Delta|$ history |
| Quasiparticle-weight proxy $z_{qp}\approx1/r_J$ | The cluster-ED vertex renormalization `r_Q`/`r_MQ` is a local ($q=0,\ \omega=0$) estimate standing in for a genuinely $k$-dependent quasiparticle weight $Z(k)$; this is a good approximation near the Mott boundary ($r_J\to2$) and in the weakly correlated limit ($r_J\approx1$, where $Z\approx1$ anyway by the Ward identity), but is least controlled in the intermediate, strongly correlated range $r_J\sim1.3$–$1.7$ |
| `hybrid_scale` fixed during optimization | The ligand-repulsion downfolding parameter entering `U_pp` (§4) is not part of the 5D Bayesian search; scanning it separately is advisable when benchmarking against a specific compound |
| Eg,2 channel partially self-consistent | The Eg,2 phonon is fully wired into the Hamiltonian, free energy, and Hessian, but its exchange-driven rigidity correction and its cross-rigidity with the B₁g channel are currently left at zero (they vanish by Kramers symmetry at the level presently implemented); its own SC-triggering diagnostics are therefore less developed than the B₁g channel's |
| Bare spin–JT cross-vertex `U_SQ` from a local cluster estimate | `U_SQ = r_MQ·√(J_eff·V_JT)` inherits the same $q=0,\ \omega=0$ locality caveat as `r_Q`/`r_MQ` above |
| Incommensurate AFM handled only as a diagnostic + soft retry | `_scan_incommensurate_nesting` detects a preference for $q^*\neq(\pi,\pi)$ and triggers one softened-$M$ retry, but the BdG Hamiltonian itself remains fixed at commensurate $(\pi,\pi)$ ordering throughout — a genuinely incommensurate spiral solve is not implemented |
| $V_d$ sign-flip EMA | Suppresses numerical oscillation in the d-wave vertex but may slow the genuine response near a doping-driven crossover between d-wave and s-wave dominance |
| No built-in doping-sweep visualization | The current `__main__` block evaluates one reference doping point; a phase-diagram-style sweep over $\delta$ (order parameters, $T_c(\delta)$, etc.) is not wired up as a script feature at present, though every quantity needed for one is already returned in the per-doping result dictionaries |

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
- TuRBO / trust-region Bayesian optimization: Eriksson, D. et al. (2019). *NeurIPS.*
- Richardson extrapolation: Richardson, L.F. (1911). *Phil. Trans. R. Soc. A* 210, 307.
- Moriya spin fluctuations: Moriya, T. (1985). *Spin Fluctuations in Itinerant Electron Magnetism.* Springer.
- Nearest positive-semidefinite matrix projection: Higham, N.J. (1988). *Linear Algebra Appl.* 103, 103.

---

*For questions or contributions, open an issue or pull request.*