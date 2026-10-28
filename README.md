# Multipolar SC-Triggered Jahn–Teller Model

> **A renormalized mean-field theory (RMFT) simulation of superconductivity-triggered B₁g Jahn–Teller distortion in a D₄h charge-transfer insulator with strong spin–orbit coupling and collinear antiferromagnetic order.**

This repository implements the self-consistent theory described in *"Multipolar superconductivity and Jahn–Teller activation in strongly correlated systems"* (Ecsenyi Szilárd, 2026). The core claim is that in a class of strongly spin–orbit-coupled, charge-transfer Mott insulators, a **B₁g lattice (Jahn–Teller) distortion is symmetry-forbidden in the antiferromagnetic normal state and is unlocked only by Cooper-pair condensation** — the causal arrow runs from superconductivity to the structural instability, not the other way around.

The code carries this mechanism through **both** cross-doublet pairing sectors the symmetry argument allows ($`\Gamma_6`$–$`\Gamma_{7a}`$ and $`\Gamma_6`$–$`\Gamma_{7b}`$), and — in its current form — through a **general, symmetry-classified pairing basis** rather than a fixed pair of amplitudes: a library of 42 lattice-harmonic form factors (on-site, nearest-neighbour and same-sublattice bonds; even and odd in $`k`$), each carried as its own complex amplitude in each Γ₇ branch ($`2\times42=84`$ self-consistent complex gap amplitudes). The pairing kernel is resolved exactly into the eight one-dimensional characters of the reduced eight-element group $`G=\{E,T_Q\}\times\{E,\,k_x\to-k_x,\,k_y\to-k_y,\,\mathbf k\to-\mathbf k\}`$ (the second factor is the four-element $`k`$-space image of $`D_{2h}`$ — the full $`D_{2h}`$ has eight elements and acts on in-plane momenta only through this quotient; the code and this document abbreviate it $`D_{2h}`$. $`G`$ is adapted to the diamond-MBZ sample and to the $`D_{2h}`$ working point and is a subgroup of, not the full, magnetic symmetry group of the crystal, §8), so that gap functions of different $`G`$-character are never mixed by construction at the linear level. Around this sits an exact multi-orbital cluster diagonalization (including Kanamori Hund's coupling) for the superexchange scale, a Moriya self-consistent-renormalization treatment of the spin channel, a four-site plaquette cluster ED that supplies the local irreducible vertex and the beyond-mean-field stiffness and energy of the lattice channel, and a lattice coordinate $`Q`$ that is not iterated together with the other order parameters but **profiled out** at every SCF step as the minimum of the total free energy (including the exact cluster energy) inside a local trust window around its current value, then relaxed with its own adaptive weight.

---

## Table of Contents

- [Physical Hypothesis](#physical-hypothesis)
- [Theoretical Framework](#theoretical-framework) — three detail pages: [Part 1 (§1–§9)](docs/theory-1-local-physics.md), [Part 2 (§10–§17)](docs/theory-2-q-vertex-kernel.md), [Part 3 (§18–§23)](docs/theory-3-free-energy-tc.md)
- [Model Architecture](#model-architecture)
- [Key Algorithms](#key-algorithms)
- [Parameters](#parameters) — constants tables: [docs/constants.md](docs/constants.md)
- [Installation & Usage](#installation--usage)
- [Output & Diagnostics](#output--diagnostics)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Physical Hypothesis

In the textbook picture, the Jahn–Teller (JT) effect and superconductivity (SC) work against each other: a JT lattice distortion lifts orbital degeneracy at the Fermi level by localizing electrons into small polarons, which competes with the coherent, delocalized electron pairs needed for Cooper pairing. This model inverts that relationship for a specific class of materials.

**The central claim:** in a D₄h, charge-transfer-insulating, strongly correlated system where spin–orbit coupling (SOC) is *not* a perturbation — it reorganizes the local Hilbert space into well-separated Kramers doublets (Γ₆, Γ₇) — a collinear antiferromagnetic (AFM) ground state stabilizes *only* dipolar (rank-1) multipolar order. Concretely:

- The Γ₆ ground-state doublet carries **no electric quadrupole moment**: a pure Γ₆ (or Γ₇) manifold has $`\langle\Gamma_6\vert Q^{(2)}_{B_{1g}}\vert \Gamma_6\rangle = 0`$ exactly, by group theory alone.
- Consequently the B₁g Jahn–Teller distortion is **symmetry-forbidden** in the AFM normal state: $`\Gamma_{JT} \not\subset \Gamma_{\mathrm{AFM}} \otimes \Gamma_{\mathrm{AFM}}`$. Here $`\Gamma_{\mathrm{AFM}}`$ is the ground-state multiplet ($`\Gamma_6`$) classified in $`\bar D_{4h}`$, the group of the single-ion Hamiltonian; at the level of the lattice, the closure of the dynamical spin–quadrupole channel $`\chi_{SQ}`$ in the normal state rests on the antitranslation $`\Theta=\mathcal T\tau_{AB}`$ together with the inversion (§2), not on the $`D_{4h}`$ labels.
- Cooper-pair condensation, provided the pairing channel is genuinely **interorbital** (Γ₆↔Γ₇, not Γ₆↔Γ₆ or Γ₇↔Γ₇), builds up a Bogoliubov coherence between the two doublets.
- Only in this paired subspace does rank-2 multipolar order become accessible: $`\Gamma_{JT} \subset \Gamma_{\mathrm{pair}} \otimes \Gamma_{\mathrm{pair}}`$.
- The B₁g JT distortion therefore emerges as an **induced response of the condensate**, not as a primary instability — superconductivity comes first, the lattice distortion follows.

The three-fold splitting of the $`t_{2g}`$ shell under SOC actually produces **two** $`\Gamma_7`$-like doublets once the crystal field is included ($`\Gamma_7 \to \Gamma_{7a}\oplus\Gamma_{7b}`$, see §1). Both are JT-active by the same $`\Gamma_6\otimes\Gamma_7\supset B_{1g}`$ selection rule (§2), and the code solves **both** cross-doublet sectors self-consistently and on equal footing: every pairing form factor of the basis (§8) carries one complex amplitude for the Γ₆–Γ₇ₐ pair and one for the Γ₆–Γ₇ᵦ pair, and each of them feeds its own anomalous (Gorkov) coherence back into the Hamiltonian as its own B₁g Weiss field. $`\Gamma_{7a}`$ — identified in `__post_init__` as the doublet with the larger direct $`\langle\Gamma_6\vert B_{1g,\mathrm{op}}\vert \Gamma_7\rangle`$ matrix element (§1) — is generically the dominant sector, but $`\Gamma_{7b}`$'s amplitudes are genuine dynamical variables of the SCF fixed point, wired into the 24×24 BdG Hamiltonian, the RPA pairing vertex, the condensation energy and the gap equations identically in structure, differing only in the (self-consistently distinct) Gutzwiller factor, exchange coupling, and Bogoliubov coherence each branch carries.

### Material class and the three viability conditions

The theory targets systems with:

- D₄h point-group symmetry **and** a global inversion center (square-lattice materials: cuprates, pnictides, selected layered transition-metal oxides). The D₄h label refers to the lattice and to the single-ion crystal field; the AFM-ordered state itself has the magnetic point group 4/mm′m′ (for $`M\parallel z`$), reduced to m′m′m by a static B₁g field, and the AFM order alone does not lower D₄h to D₂h (§3). Inversion symmetry is what allows the Cooper pair to have a well-defined parity (pure singlet or pure triplet); if it is broken locally, Rashba/Dresselhaus terms mix the two and the clean tensor-product selection rule below no longer applies unambiguously.
- Strong electron correlation with a charge-transfer (ZSA-type) insulating parent state.
- Strong SOC that reorganizes the $`t_{2g}`$ manifold into $`\Gamma_6 \oplus \Gamma_{7a} \oplus \Gamma_{7b}`$ Kramers doublets (typical of 4d/5d transition-metal ions, though the framework is agnostic to the microscopic origin of the SOC scale).
- Superexchange-stabilized collinear AFM order in the parent compound, including a finite intra-atomic Hund's coupling in the virtual $`d^2`$ superexchange intermediate state (§4). The Néel axis is selectable (`M_axis`, §1); the paper's symmetry theorems are derived for `'z'`.
- Finite hole doping ($`\delta`$ > 0), which restores the itinerancy needed for coherent Cooper pairing.

Three conditions must hold simultaneously for the mechanism to operate:

1. **Charge-transfer insulating character** — the ZSA charge-transfer gap and the on-site $`U_{dd}`$ (both primary inputs, §4) are large enough that genuine Mott physics is in play.
2. **Non-Mott-localized coherence** — the Cooper pairs must actually be mobile; this requires $`\delta`$ > 0 so the Gutzwiller kinetic factor $`g_t`$ does not collapse to zero.
3. **Moderate AFM order** — AFM correlations must be present (they are what forbids the JT channel in the normal state) but not so strong that spin fluctuations kill superconductivity outright (Stoner criterion: the largest eigenvalue of $`\mathrm{diag}(J_c(q))\,\chi_{cc'}`$ stays below 1).

### Why the pairing must be interorbital, and why turning on Δ alone is not enough

A tempting but incorrect picture is that the condensate simply mixes $`\vert \Gamma_6\rangle`$ and $`\vert \Gamma_{7}\rangle`$ into a coherent superposition $`\alpha\vert \Gamma_6\rangle+\beta\vert \Gamma_{7}\rangle`$, giving a quadrupole moment linear in the mixing amplitude $`\beta\propto\Delta`$. This cannot be right: it would make $`\langle B_{1g,\mathrm{op}}\rangle`$ depend on the arbitrary global $`U(1)`$ phase of $`\Delta`$, violating gauge invariance. The correct microscopic statement is that the condensate modifies the **normal** (charge-conserving) density matrix through the Bogoliubov coherence factors: $`\langle c^{\dagger}_{6\sigma}c_{7\sigma'}\rangle`$ picks up a $`v^{*}v \sim \vert \Delta\vert ^2`$ contribution, which is explicitly gauge-invariant and appears at *quadratic*, not linear, order in $`\Delta`$.

For this to happen at all, the pairing must directly connect $`\Gamma_6`$ and a $`\Gamma_7`$ doublet — a purely intraorbital singlet (Γ₆–Γ₆ or Γ₇–Γ₇) generates no $`\Gamma_6`$–$`\Gamma_7`$ Bogoliubov mixing and leaves the JT channel closed even in the superconducting state. This is exactly the structure built into every pairing operator of the code's basis (§8): all of them are Γ₆↔Γ₇ₐ or Γ₆↔Γ₇ᵦ by construction.

There is a further, more subtle point verified directly in the group-theoretic analysis: **at $`Q=0`$, even with $`\Delta\neq0`$, a purely singlet-paired BdG ground state still gives $`\langle B_{1g,\mathrm{op}}\rangle \equiv 0`$ band-pair by band-pair.** The reason is that $`B_{1g,\mathrm{op}}`$ in the numerical SOC eigenbasis turns out to be **spin-conserving** in the $`S_z`$-labelled Kramers basis of the reference parameter regime ($`\Gamma_6{\uparrow}\leftrightarrow\Gamma_{7}{\uparrow}`$; it is a pure orbital operator, so it conserves the true $`S_z`$, and for $`\Delta_{\mathrm{tetra}}/\lambda_{\mathrm{SOC}}`$ < $`1/2`$ the $`S_z`$-up partner of $`\Gamma_6`$ is the $`\vert L_z\vert=1`$-dominated state), while the singlet pairing operator only ever connects opposite pseudospin sectors; $`S_z`$ stays block-diagonal in those same sectors. The actual switch is thrown by the *self-consistently induced anomalous coherence* — numerically the same operator as $`B_{1g,\mathrm{op}}`$ — which is exactly zero at $`Q=0`$ and becomes nonzero only once **both** $`\Delta\neq0`$ **and** $`Q\neq0`$. In the code this anomalous coherence is the complex vector `F67_vec` (one entry per pairing amplitude, §2), fed back into the local Hamiltonian's transverse Weiss field (§9, §19); the loop is genuinely self-consistent rather than a one-way "SC turns on JT."

### The JT distortion as a thermodynamic order parameter, not a dynamical mode

$`Q`$ is treated as a **macroscopic, thermodynamic order parameter** — physically the amplitude of a flat (dispersionless) optical Einstein phonon — rather than a fluctuating dynamical degree of freedom. Differentiating the free energy with respect to $`Q`$ gives the equilibrium condition and the softening criterion

```math
\lambda_{JT}^{\mathrm{norm}} = \frac{g_{JT}^{2}\,\chi_{QQ}}{K_{\mathrm{lattice}}} \qquad (<1\text{ stable},\ =1\text{ onset},\ >1\text{ spontaneous JT}),
```

where $`\chi_{QQ}=-g_{JT}^{-2}\,\partial^2\Omega_{\mathrm{el}}/\partial Q^2`$ is the $`g_{JT}^{2}`$-normalised electronic JT susceptibility (§13), so that $`K_{\mathrm{spont}}=g_{JT}^{2}\chi_{QQ}`$ is the electronic softening and, in the quadratic (Landau) approximation, $`K_{\mathrm{eff}}=K_{\mathrm{lattice}}-g_{JT}^{2}\chi_{QQ}=K_{\mathrm{lattice}}\,(1-\lambda_{JT}^{\mathrm{norm}})`$: the onset $`\lambda_{JT}^{\mathrm{norm}}=1`$ is the point where $`K_{\mathrm{eff}}`$ vanishes (it is *not* $`\chi_{QQ}/K_{\mathrm{eff}}`$, which would diverge there).

In the SCF solver this equilibrium is not reached by iterating a damped Newton step on $`Q`$: at every iteration $`Q`$ is **profiled out** as a slave variable, $`Q^*=\arg\min_Q F_{\mathrm{tot}}(Q)`$ with $`F_{\mathrm{tot}}=F_{\mathrm{can}}(Q;M,\Delta,F_{67},\mu)`$ plus the exact beyond-mean-field plaquette energy (§12), evaluated on a small grid inside a **local trust window around the current $`Q`$** and polished by a bracketed force root. The window is deliberate: the two $`\pm Q`$ wells of the B₁g profile are $`Z_2`$ partners that differ only by a few $`\mu`$eV, and a global scan would flip between them from iteration to iteration; the local profile is branch-continuous, so the sign of $`Q`$ is selected by the seed and by the dynamics (§22) and a first-order jump is a property of the converged branches (compared in §17 and §23), not of the profile search. $`Q`$ is relaxed toward $`Q^*`$ with its own adaptive weight and kept out of the Anderson history of the other order parameters.

A natural question is whether the free energy contains a **linear** coupling $`Q\cdot\vert \Delta\vert ^2`$ between the lattice and the condensate. The theoretical analysis finds that in the clean D₄h limit this cross-coupling vanishes — an exact but non-generic consequence of the purely interorbital structure of the pairing operators, not a trivial symmetry accident (the naive argument "$`\vert \Delta\vert ^2`$ is always $`A_{1g}`$" is incomplete, because different pairing channels need not share one point-group label and their cross terms could in principle transform as $`B_{1g}`$). The code does **not** impose this cancellation by hand: it is left to the free energy itself. The Δ-Hessian (`compute_dF_dDelta_and_d2F`, §21) is written so that a bilinear term $`Q\,\Delta_i^{*}\Delta_j`$ would be picked up (inactive channels are fitted with a two-point stencil precisely because such terms make $`F`$ linear in $`\vert\Delta_i\vert`$), and the $`Q`$ profile of §12 would find a finite $`Q^*`$ in a purely linear-coupling scenario. (If the lattice already sits in D₂h because of a finite static crystal field `Delta_B1g_static` ≠ 0, the cancellation is lifted and a genuine linear coupling appears — see §3.) The upshot: in the clean D₄h limit, SC-triggered JT is a **threshold phenomenon**. The condensate progressively softens the B₁g mode's effective stiffness $`K_{\mathrm{eff}}`$ until the lattice snaps into a finite distortion.

While $`Q`$ itself is a static mean field, its *coupling* to the electronic condensate is not treated as instantaneous everywhere in the code: the phonon-mediated piece of the pairing vertex carries an explicit adiabatic retardation factor tied to a phonon energy scale derived from the lattice spring constant, the input reduced mass and the electronic softening of the mode itself (§14) — a first step beyond the purely static-propagator treatment, though $`Q`$'s own equation of motion remains a thermodynamic equilibrium condition, not a dynamical one.

### First-order character

Numerically, this transition is expected to be **first-order**, not a simple second-order spin-fluctuation instability: spin fluctuations alone are not sufficient to drive the lattice unstable, and the system tips into the $`B_{1g}`$-distorted configuration only cooperatively, together with the superconducting condensate. This is reflected in the solver's SCF-dynamics classifier (see "SCF Loop" under [Key Algorithms](#key-algorithms)), in the `__main__`'s explicit three-way free-energy comparison against the normal and $`Q`$-pinned states (see [Installation & Usage](#installation--usage)), and in the thermodynamic, first-order-aware Tc estimate of §23.

---

## Theoretical Framework

The derivation is long and formula-heavy, so it is split over three pages (GitHub renders a single page with well over a thousand formulas unreliably). Section numbers **§N are global**: every “§N” in this README and in the pages below refers to the numbering listed here.

- **[Part 1 — local physics, symmetry and the BdG Hamiltonian (§1–§9)](docs/theory-1-local-physics.md)**
  - §1 Local Hilbert Space: SOC + Crystal-Field Diagonalization
  - §2 Symmetry Protection of the JT Channel
  - §3 The B₁g Operator and the D₄h/D₂h Crossover
  - §4 Multi-Orbital Cluster Superexchange and the Weiss Field
  - §5 Gutzwiller Renormalization and the Mott Guard
  - §6 B₁g Jahn–Teller Distortion, Orbital-Selective Hopping, and Further-Neighbor Terms
  - §7 The Eg,2 Phonon Channel
  - §8 The Pairing Basis: a Symmetry-Classified Harmonic Library, Both Γ₇ Branches
  - §9 The 24×24 BdG Hamiltonian (Doubled Unit Cell, Full 3-Doublet, 84-Amplitude Pairing)
- **[Part 2 — observables, the Q coordinate, the RPA vertex and the pairing kernel (§10–§17)](docs/theory-2-q-vertex-kernel.md)**
  - §10 Observables via `VectorizedBdG`
  - §11 Rigidity K_eff: ∂²F/∂Q² of the Total Free Energy
  - §12 Profiling Out Q: `solve_Q_slave`
  - §13 χ_QQ from Thermodynamic Finite Differences
  - §14 Channel-Resolved RPA Vertex: [Γ₆, Γ₇ₐ, Γ₇ᵦ, JT] in One 4×4 Closure
  - §15 FS-Resolved Pairing Kernel: Orbital-Character-Weighted, Exactly G-Covariant
  - §16 Normal-State JT Stability and the Post-Convergence Hessian Mode Classification
  - §17 SC-JT Coexistence: What the Code Actually Evaluates
- **[Part 3 — free energy, cluster ED, gap equations and Tc (§18–§23)](docs/theory-3-free-energy-tc.md)**
  - §18 Variational Free Energy and the Role of the Cluster
  - §19 Four-Site Plaquette Cluster: Beyond-Mean-Field Stiffness, Q-Energy and Exact Vertex Extraction
  - §20 Chemical Potential: Newton with Analytic ∂n/∂μ
  - §21 Gap Equations: the 84-Amplitude Fixed Point, Complex Phases, and the Irrep Kernel
  - §22 Incommensurate AFM Nesting Check and Gap-Symmetry Consistency (Diagnostics Only)
  - §23 Temperature-Dependent Tc Estimates

---

## Model Architecture

```
Pairing registry (module level; import-time checked by _check_registry and _check_bond_content)
    ├── _HARMONIC_LIBRARY / _PAIR_CHANNELS / _IRREP_NAMES     42 form factors, each (D4h irrep, bond kind, parity, phi(kx,ky), scale)
    ├── _IRREP_CHAR, _G_OPS, _G_CHI, _SECTOR_MEMBERS          measured G = {E,T_Q} x {E,kx->-kx,ky->-ky,k->-k} characters; 8 sectors; covered/uncovered
    ├── _LIB_CHAR, _LIB_SWAP, _IRREP_SWAP, _D4H_BY_MX_MY_SWAP measured (T_Q, mx, my, sigma_d) of every harmonic -> D4h irrep inside a sector
    ├── _N_IRREP = 42, _N_DELTA = 2*_N_IRREP = 84             Delta_vec / F67_vec layout; _DELTA_NAMES, _DELTA_IDX, _IDX_7A_ARR, _IDX_7B_ARR
    ├── bond content:    _BOND_GRID, _bond_weights, _bond_class_fractions, _BOND_CLASSES, _TRW_BOND_FRAC   (real-space bond support of every form factor -> TRW weights)
    └── helpers: _pairing_basis_matrix, _is_onsite/_is_nn_bond/_is_nnn_bond, _afm_shift_character, _mirror_characters,
                 _gutzwiller_irrep_vector, _gram_sqrt, _pairing_spectral_decomposition, _sector_condensation_energy, _m2_weighted_J,
                 _irrep_decomposition, _sector_label, _sector_report_lines, _check_fs_closure, _grid_sym_permutation,
                 _on_tq_boundary, _tq_partner, _swap_resolved_top, lambda_channel_decomposition

ModelParams  (dataclass, __post_init__ runs the SOC+CF diagonalization and the superexchange cluster ED)
    ├── Primary inputs:  t_pd, t_prime_ratio, t_dprime_ratio, U_dd, lambda_soc, Delta_tetra,
    │                    g_JT, K_lattice, lambda_hop, g_Eg2, K_lattice_Eg2, Delta_CT, Delta_B1g_static,
    │                    hybrid_scale, Upp_ratio_bare, J_H_ratio, M_eff, kT, M_axis, tol, trw_far_scale (default 1.0)
    │                    (the coordination number is the module constant _Z = 4, not a field)
    ├── Derived scalars: Delta_CF, g7split, t0, t_prime, t_dprime, thermal_fs_nat_size,
    │                    J2_over_J1, J3_over_J1, Z_afm_eff, N_k, U_pp, J_H
    ├── Derived arrays:  sz_op (exact <Sz> per Kramers partner, 6 components),
    │                    S_ops{'x','y','z'} / S_inter / S_axis_op (doublet-projected spin operators; Neel axis),
    │                    B1g_op, B1g_offdiag, Eg2_op, _w_orb (3x3: G6/G7a/G7b rows x xz/yz/xy columns),
    │                    kappa_doublets (3), kappa_cross (2), Tx_A_xz/_yz/_xy (orbital projectors, sum to I_6),
    │                    trw_irrep_scale (42 bond-class weights of the transverse Weiss field, §9)
    ├── Grid objects:    k_points, k_weights, shift_table (_NK x _NK x N_k int32 cyclic shift table for arbitrary q)
    └── Methods:         get_gutzwiller_factors() (returns (g_t, g_J) only),
                         exchange_channels() (returns J_A1g_diag, J_B1g_7a, J_B1g_7b),
                         _exchange_J_q(), effective_hopping_anisotropic(), hopping_matrices(),
                         hopping_matrices_dQ(), wave_function_weight() (T_Q-symmetrized)

_SolveState  (dataclass, mutable per-SCF-run state - never stored on self)
    ├── V_irrep_ema: Optional[ndarray]   # per-channel sign-flip EMA of diag(V_irrep)
    └── _ema_kick_pending: bool          # doubles the EMA blend weight for one iter after a kick

RMFT_Solver
    ├── Initialization: __init__ (g_JT_bare, _K_bare, Phi_irrep_k on the SCF grid),
    │                   _rebuild_orbital_operators (incl. Neel-axis spin vertices), _get_vbdg,
    │                   _get_chi0_norm_cache, _reset_transient_state, _shallow_clone,
    │                   _clone_solver_at_T, _full_rebuild
    ├── Local H / mu:   build_local_hamiltonian_for_bdg, _find_mu_for_density,
    │                   estimate_gutzwiller_factors_occupation_based (dynamic G7a/G7b factors, §5),
    │                   _compute_orbital_densities
    ├── JT / Q:         _calc_dHdQ (full explicit dH/dQ, 24x24, with even/odd hole parity and TRW weights),
    │                   _make_Q_profile (F_total(Q) and its force), solve_Q_slave (local trust-window profile, §12),
    │                   _eval_B1g_and_kweights_at_Q (scalar <B1g> + k-resolved array at fixed density)
    ├── Observables:    Eg2_expectation, _chi_QQ_matrix_elements (3-point, or 2x2 B1g/Eg2),
    │                   _diamagnetic_QQ_term
    ├── RPA vertex:     _chi0_channels_raw (4x4 channel-resolved raw susceptibility, §14),
    │                   _rpa_closure (Moriya damping + PSD projection + determinant + Stoner factor),
    │                   _rpa_vertex_tables (Xi, Xi_cross, V_JT per unique q), _fs_channel_weights,
    │                   _moriya_gamma_landau,
    │                   _make_vertex_params -> (Gamma_M, F_cluster, V_JT_eff, V_JT_corr, V_cap),
    │                   preflight_report (PREFLIGHT diagnostic, called from _make_vertex_params(verbose=True))
    ├── Pairing kernel: compute_pairing_kernel_and_build_cache (exact G-sector kernels, §15),
    │                   scf_gap_diagnostics (coherence lengths from the exact FS gap shape),
    │                   _vertex_matrix_at_Q
    ├── Free energy:    _compute_bdg_free_energy, _canonical_free_energy (F = Omega+mu*N, mu refit),
    │                   compute_cluster_free_energy (four-site plaquette ED: Gamma_ED vertex, K_eff and its parts, §11/§19),
    │                   _build_H_cluster, _dF_cluster (beyond-MF energy at (M,Q,Delta)), _dFc_reference,
    │                   _bdg_b1g_per_sublattice,
    │                   compute_dF_dM_channels_and_hessian (3-component FD gradient/Hessian of F_bulk + dF_cl)
    ├── SCF machinery:  _scf_kick (seeding via the early Hessian and the sector kernels),
    │                   refine_M_state (Anderson-accelerated M/mu at fixed Q, Delta_vec, F67_vec; returns a 10-tuple),
    │                   _classify_scf_dynamics, _classify_q_dynamics, _anderson_mix,
    │                   _blend_update_candidates (complex-safe Newton/fixed-point blend),
    │                   _relative_phase_change, _project_kick_from_hessian
    ├── Main solve:     solve_self_consistent   <- fixed point over (M[3], Delta_vec[84], mu) with Anderson on the Delta moduli,
    │                   Q profiled out each iteration (local window) and relaxed with its own adaptive weight
    ├── Post-hoc:       _scan_incommensurate_nesting, compute_hessian ((3+1+1)-dim, beyond-MF F),
    │                   compute_dF_dDelta_and_d2F (84x84, one entry per amplitude modulus)
    ├── Tc / gap:       _bcs_ratio_2D (spectral FS gap), compute_Tc_linearized_normal (Tc2),
    │                   compute_Tc_thermodynamic (Tc3)
    └── Diagnostics:    _get_fs_points

VectorizedBdG   (thin batched-LAPACK wrapper bound to one RMFT_Solver)
    ├── _build_H_stack                     builds & Hermitizes the (N_k, 24, 24) BdG stack from an
    │                                       84-component Delta_vec and an 84-component F67_vec
    │                                       (even channels antisymmetrized, odd channels symmetric)
    ├── compute_channel_staggered_magnetizations   -> per-channel (G6, G7a, G7b) M_stag
    └── compute_gap_eq_vectorized           -> (Delta_out (84,), F67_vec (84,), vertex_cache) - both branches through
                                               the shared V_irrep, each with its own Gutzwiller vector
                                               and anomalous amplitudes (§8, §15, §21)

Module-level drivers (after the classes):
    compute_and_log_hessian_diagnostics    HESS-DIAG / FREE-ENERGY report for ref / normal / SC_Q0 (§16, §17)
    _solve_normal_state                    the Delta = 0 reference state: normal-state M, mu, Q profile, F_bdg (no SCF history)
    _sc_gap_amplitude                      amplitude vector of a result used by the Tc routines
    plot_ground_state_comparison(results, labels=None)   - standalone function, not a class method;
        builds the 2x2 "ref vs. normal vs. SC_Q0" free-energy/order-parameter comparison figure
        described in Installation & Usage / Output & Diagnostics. It is a plotting routine only.
```

Module-level functions supporting the superexchange cluster ED (§4) — `_apply_hop`, `_multiorbital_site_of`, `_multiorbital_2hole_hamiltonian`, `_multiorbital_reference_vector`, `_lowdin_branch_energies`, `_intra_doublet_kappa`, `_cross_doublet_kappas` — the module-level `_compute_F67_pair_amplitudes` (all 84 anomalous amplitudes and their raw Gutzwiller-weighted counterparts in one call, §2), the numerical helpers `_fd_grad_hessian` (the single finite-difference gradient/Hessian core shared by the $`M`$-, $`(M,Q,\Delta)`$-, $`\Delta`$- and $`(Q,\Delta)`$-Hessians), `_lm_damping` (Levenberg–Marquardt damping shared by the $`M`$ and $`\Delta`$ Newton steps), `_q_fd_step`, `_adjugate`, `_lehmann_kernel`, `_fermi_function`, `_get_nambu_spinors`, `_build_H_AB_block`, `_expand_M_channels`, `_wave_project`, `_mbz_deduplication`, `_fmt_amplitude_line` (log formatting of the amplitude vectors), and the unused-by-default `_gamma_splitting` (a fast standalone $`\Delta_{CF}`$ pre-scan) and `_unique_q_pairs` (a $`q`$-sampling helper that can write the debug plot `unique_q_hist.png` when called with `verbose=True`; the pairing kernel now builds its own unique-$`q`$ table over the eight $`G`$-images, so a normal run never produces that plot) round out the module. `compute_cluster_free_energy` contains the closures `_cluster_stiffness`, `_mu_state_at`, `_mu_state_at_normal`, `_stable_inverse` and `_F_can` (§19); `_build_H_cluster` and `_dF_cluster` are methods of `RMFT_Solver`.

---

## Key Algorithms

### SCF Loop (`solve_self_consistent`)

A fixed point over $`(M_{\Gamma_6},M_{\Gamma_{7a}},M_{\Gamma_{7b}},\ \Delta_{\mathrm{vec}}[84],\ \mu)`$ with Anderson(5) acceleration on the gap moduli, the anomalous amplitudes $`F_{67}`$ taken from the gap-equation pass, and the lattice coordinate $`Q`$ profiled out at every step and relaxed with its own weight. The signature is `solve_self_consistent(target_doping, seed_complex_phase, initial_Delta, initial_F67, Q_Eg2=0.0, force_delta_zero=False, force_Q_zero=False, max_iter=_MAX_ITER, verbose=False)` (the seed phases and the initial $`(\Delta,F_{67})`$ pair are supplied by the caller; all zeros in `__main__`, where the kick then seeds the gap). After the seeding stage (`_scf_kick`, below) the iteration starts at $`Q=`$ `_Q_SEED_THR` ($`Q=0`$ with `force_Q_zero`) and, per iteration:

1. Recompute the occupation-based Gutzwiller factors `g_Delta_s`, `g_Delta_ad`, `g_Delta_bd` (§5) and the exchange tensors $`J_{A1g}`$, $`J_{B1g}^{(7a)}`$, $`J_{B1g}^{(7b)}`$ (§6); then `_make_vertex_params` — the retardation factor (§14), the four-site cluster ED with its $`K_{\mathrm{eff}}`$ stencils and the $`Q`$-energy closure (§11, §19), $`J_{\mathrm{eff}}`$, $`\Gamma_M`$, `V_JT_eff`, `V_JT_corr`, `V_cap`.
2. Evaluate the SC-state RPA closure at $`q=(\pi,\pi)`$ and $`q=0`$ (`_rpa_closure`) for the determinants, susceptibilities and Stoner factors, and update the instability indicator $`\Lambda_{\mathrm{inst}}`$ (`_lambda_lin_max`: an EMA, weight `_EMA_NEW_WEIGHT`, of the linearized pairing eigenvalue `lambda_lin_max` of the cached normal-state vertex, clipped to $`[0,10]`$; the Stoner factors are logged but no longer enter it).
3. **Gap sector.** The 84-dimensional Newton step (`compute_dF_dDelta_and_d2F`, §21 step 5) and the gap-equation update `compute_gap_eq_vectorized` (§21 steps 1–4), which also returns the new anomalous amplitudes `F67_vec_new`; the candidate $`\Delta`$ is formed by `_blend_update_candidates` with $`\beta=`$ `_ALPHA_HF_DELTA` $`=0.75`$ (modulus clipped, phase kept).
4. **Profile out $`Q`$:** $`Q_{\mathrm{out}}=Q^*`$ from the local trust-window profile `solve_Q_slave` (§12), unless `force_Q_zero`; $`Q_{\mathrm{mixed}}=Q+\alpha_Q(Q_{\mathrm{out}}-Q)`$ with the adaptive $`\alpha_Q\le`$ `_ALPHA_Q` $`=0.5`$ and the adaptive window scale $`s_Q`$, both reduced by the Q-oscillation guard (§12).
5. **Anderson on the gap moduli** (§21 step 6): the primary block $`(\vert\Delta_{s,7a}\vert,\vert\Delta_{d,7a}\vert)`$ and the extended block of the remaining 82 moduli each get their own Anderson(5) step; phases are kept from the blended vector. Neither $`Q`$ nor $`M`$ is in any Anderson vector.
6. **Newton step for $`M`$** (three channels): the finite-difference gradient/Hessian of the beyond-mean-field canonical free energy $`F_{\mathrm{bulk}}(M,Q)+\Delta F_{\mathrm{cl}}(M,Q)-\Delta F_{\mathrm{cl}}(0,Q)`$ (`compute_dF_dM_channels_and_hessian`), Levenberg–Marquardt-damped through the softest eigenvalue (with a near-critical $`\mu`$ that shrinks as $`\vert\Delta\vert`$ grows), trust-region-limited by a stiffness cut $`\propto J_{\mathrm{eff}}/t_{\mathrm{eff}}`$ and a curvature factor, with per-channel step floors, combined with the BdG fixed point $`M_{\mathrm{BdG}}`$ from the staggered-magnetization observable by `_blend_update_candidates` with $`\beta=`$ `_ALPHA_HF` $`=0.85`$ (so the Newton step is damped by $`\alpha`$ too).
7. $`\mu`$ is re-solved exactly (Newton+Brent) at the freshly mixed $`(M,Q_{\mathrm{mixed}},\Delta,F_{67})`$ rather than carried as an imperfectly converged fast variable.
8. **Adaptive mixing rate** (every 5th iteration from iteration 5): $`\alpha_{\mathrm{base}}=\mathrm{clip}\big(\alpha_0/(1+0.5\Lambda_{\mathrm{inst}}),\ \alpha_0/6,\ \alpha_0\big)`$; a diverging step halves $`\alpha`$ and clears the history, stagnation clears it after two hits (and mildly boosts $`\alpha`$ in the subcritical plateau), converging steps apply a mild SC+JT boost/decay; a **limit-cycle detector** (relative std of the total $`\sum_i\vert\Delta_i\vert`$ over the last `_CYCLE_WINDOW` iterations > `_CYCLE_THRESHOLD`) damps $`\alpha`$ by `_CYCLE_DAMP_FAC`, clears the history and classifies the regime (`limit_cycle`, `first_order_jump`, `hysteretic`; otherwise `converging`). A $`Q`$ displacement larger than `_Q_THR_REL·lambda_hop` caps $`\alpha`$ at $`0.3\,\alpha_0`$ (`_SCF_ALPHA_QMOVE_CAP`), and a detected $`Q`$ zig-zag additionally damps $`\alpha`$ together with $`\alpha_Q`$ and the $`Q`$ window (`Q-CYCLE`, §12). A negative SC-state AFM determinant (past the QCP) applies an exponential penalty to $`\alpha`$ and boosts $`\Lambda_{\mathrm{inst}}`$; near the QCP ($`\det`$ < `_RPA_DET_WARN` or $`\vert V_d\vert`$ > `_V_CUT`) $`\alpha`$ is capped at $`\alpha_0/(1+\Lambda)`$; a freeze counter triggers an $`\alpha`$-recovery boost. A sign flip of $`Q`$ clears the Anderson history and the vertex cache (§12).
9. **Saddle-escape kick** (iteration > 3, every 8th iteration, only while the total gap is still below $`5\times`$ `tol`): `compute_hessian` is evaluated at the current point; a negative scaled smallest eigenvalue triggers a step along its eigenvector, scaled by `_KICK_BASE_FRACTION`, the curvature and the damping $`1/(1+\Lambda_{\mathrm{inst}})`$. **The step is applied if the soft mode has a significant gap component, `frac_D > _KICK_MIN_DELTA_FRAC` (0.25)** — `pure-SC` or a mixed Δ–Q mode (§16): $`M`$, $`Q`$ and the gap amplitudes are modified (the gap kick keeps the per-channel ratios and phases), the Anderson history and the vertex cache are cleared and the EMA is marked stale. In `pure-SC` mode a cheap Γ₆-only Stoner estimate (a normal-state `refine_M_state` at $`Q=0`$, $`\Delta=0`$) is computed, and when $`M_{\Gamma_6}`$ exceeds twice that estimate all three $`M`$ channels are pulled back toward it uniformly instead of adding the eigenvector step.

Convergence requires $`\max\big(\vert\Delta M\vert,\ \vert Q_{\mathrm{out}}-Q\vert,\ \max_i\big\vert\vert\Delta_i\vert_{\mathrm{new}}-\vert\Delta_i\vert_{\mathrm{old}}\big\vert,\ \text{relative phase change},\ \max_i\big\vert\vert F_{67,i}\vert_{\mathrm{new}}-\vert F_{67,i}\vert_{\mathrm{old}}\big\vert\big)`$ < `tol` — all 84 gap and 84 anomalous amplitudes; the $`Q`$ term is the *residual* of the $`Q`$ profile, not the relaxed step, and the phase term is gauge-invariant (`_relative_phase_change`, sensitive to drifts of e.g. the $`s+id`$ phase that the moduli alone miss) —, density error < $`10\cdot`$ `tol`, at least `_MIN_ITER` iterations, a non-negative SC-state AFM determinant, and no kick applied on the same iteration. After convergence (or exhausting `_MAX_ITER = 500` iterations) the solver runs the post-convergence stage: the Hessian, the FS-resolved $`\partial\lambda/\partial Q`$ and gap-symmetry diagnostics, the incommensurate-nesting scan (diagnostic only, §22), the Mott filter (§5), the cluster/$`K_{\mathrm{eff}}`$ evaluation **of the returned state** (one further `_make_vertex_params` call at the returned $`(M,Q,\Delta,F_{67},\mu)`$, §11, which also yields `F_cluster`, `V_JT_eff` and `V_JT_corr`; `F_bdg` is then $`F_{\mathrm{can}}+\Delta F_{\mathrm{cl}}-\Delta F_{\mathrm{cl,ref}}`$, §18), the Eg,2 stability diagnostic (§7) and the QCP flag — `past_qcp` is the sign of the SC-state AFM determinant of the last loop evaluation (the state at the *start* of the final iteration); if it is negative, `converged` is reported as `False` even if the iteration tolerance was met (`scf_iteration_converged` keeps the raw flag). The result dictionary is consumed by the diagnostics block described in [Output & Diagnostics](#output--diagnostics). There are **no automatic restarts**: a long run is never silently re-solved from another seed.

### Vectorized BdG, Buffer Reuse, and the χ₀(q) Permutation Trick

`VectorizedBdG._build_H_stack` assembles the entire $`(N_k,24,24)`$ Hamiltonian stack with vectorized NumPy operations and diagonalizes it in a single `np.linalg.eigh` call per SCF iteration, reusing a pre-allocated `out=` buffer to avoid repeated allocation across hundreds of iterations; Hermiticity is enforced after assembly. The per-iteration eigensystem `(ev, ec)` is computed once and shared by the observables and the gap equations of both branches.

The $`q`$-loop inside the RPA vertex construction never re-diagonalizes: the uniform k-grid (`endpoint=False`) is built in `ModelParams.__post_init__` so that for any $`q=(n_x,n_y)\cdot2\pi/N_K`$ ($`N_K=`$ `_NK`), the $`k+q`$ grid is exactly a cyclic permutation of the $`k`$-grid. A precomputed `shift_table[nx, ny]` (shape `(_NK, _NK, N_k)`, `int32`) turns "shift by $`q`$" into a free index reorder,

```python
E_kQ_all = E_k_all[shift_table[nx, ny]]     # index reorder — no extra LAPACK call
```

reusing the *same* $`\Delta=0`$ eigensystem for every $`q`$-point in one vertex-cache rebuild. `_get_chi0_norm_cache` additionally memoizes this normal-state $`(E_k,V_k)`$ across separate calls (susceptibilities, rigidity, incommensurate-nesting scan, Gutzwiller occupations) that fall within the same iteration, keyed on $`(M,Q,\mu,g_t,g_J,n_{\mathrm{kspace}})`$: the $`M`$ tolerance is `_M_THR_REL` relative to $`\max(\Vert M\Vert,1)`$, the $`Q`$ tolerance is `_Q_THR_REL` $`\cdot\lambda_{\mathrm{hop}}`$, and $`\mu`$, $`g_t`$, $`g_J`$ and the density are checked at $`10^{-4}`$ (relative for $`\mu`$), $`10^{-4}`$, $`10^{-4}`$ and $`10^{-6}`$ respectively.

### Vertex Cache Invalidation

The RPA vertex cache is rebuilt when $`\max_c\vert\Delta M_c\vert`$ exceeds `_M_THR_REL` $`\cdot\sqrt{\vert\det_{\mathrm{afm,sc}}\vert}`$ (a threshold that tightens toward the QCP, where the vertex is most sensitive), when $`Q`$ moves by more than $`\max(f_Q\,\lambda_{\mathrm{hop}},10^{-4})`$ (with $`f_Q=`$ `_Q_THR_REL`), or unconditionally if the cached determinant sign disagrees with the freshly computed SC-state determinant sign (a proxy for having crossed the QCP since the cache was built). or when the retarded JT vertex `V_JT_eff` or the corrected `V_JT_corr` has drifted by more than $`10^{-3}`$ relative to the cached values (these depend on $`\Delta`$ — and on $`\chi_{QQ}`$ — through the phonon retardation factor, §14, which is the only way the gap enters the cache logic). A sign flip of $`Q`$ or a saddle-escape kick also discards the cache. Apart from that there is no Δ-based invalidation — the vertex *itself* is always built from $`\Delta=0`$ by construction (§14). The cache stores the live AFM determinant `det_afm_current`, the scalar susceptibilities `chi_SS/SQ/QQ_{q0,afm}`, FS geometry (`fs_pts`, `vF_arr`, `fs_idx`, `fs_weights`), and the pairing-kernel products of §15 (`V_irrep`, `W_irrep`, `S_irrep`, `V_parts_diag`, `sector_*`, `frac_seed`, …), so repeated calls within one iteration reuse the same Fermi-surface sampling.

### Limit-Cycle Detection

Independent of the adaptive-$`\alpha`$ rules, a dedicated oscillation check monitors the total $`\sum_i\vert\Delta_i\vert`$ (over all 84 channels) over a rolling window of `_CYCLE_WINDOW = 20` iterations; when the relative standard deviation exceeds `_CYCLE_THRESHOLD = 0.25`, the mixing rate is cut by `_CYCLE_DAMP_FAC = 0.45` (floored at $`\alpha_0/16`$) and the Anderson history reset, which specifically targets the strongly nonlinear regime near the JT-activation onset where the $`(Q,\Delta)`$ feedback is most prone to overshoot. The regime is classified by the max/min ratio over the window (`limit_cycle` if < 2; otherwise `first_order_jump` or `hysteretic` depending on whether the late-window mean has moved relative to the early one) and reported as `scf_dynamics_regime`; it is a diagnostic label and a mixing-rate control — it does not by itself launch a multi-seed restart. The same classifier is applied a second time to the history of $`\lvert Q\rvert`$ (`Q dynamics`, `scf_dynamics_regime = 'Q-<regime>'`), which damps $`\alpha`$ when $`\lvert\Delta\rvert`$ itself is not cycling; the signed-$`Q`$ zig-zag guard of §12 (`Q-CYCLE`) is a third, independent check.

### Initial-Condition Seeding (`_scf_kick`)

Before the main SCF loop, `_scf_kick(target_doping, seed_complex_phase, Delta_vec, F67_vec, Q_Eg2, verbose)` first calls `refine_M_state(target_doping, Q, Δ_vec, F67_vec, …)` at $`Q=0`$ — a normal-state AFM seed when the passed $`\Delta`$ is zero (it returns `(M, Δ_vec, F67_vec, n_kspace, μ, g_t, g_J, g_Δs, g_Δad, g_Δbd)`) — a fixed number `_MAX_ITER_FOR_REFINE_M` of Anderson-accelerated iterations at damping 0.45, starting from $`M=0.01`$ on every channel, with $`\mu`$ refit and the anomalous amplitudes recomputed each pass; no early exit). It then builds the cheap cluster vertex and RPA vertex parameters and the linearized pairing eigenproblem (`lambda_lin_max`, `frac_seed`, from `compute_pairing_kernel_and_build_cache`), and evaluates the (3+1+1)-dimensional Hessian (`compute_hessian`) at that normal-state seed. If its smallest scaled eigenvalue is negative, the seed is stepped along its eigenvector (`_project_kick_from_hessian`, scaled by `_EARLY_KICK_BASE` plus the linear-eigenvalue excess) and the **full 84-component gap seed is the kernel eigenvector shape** `frac_seed` (all channels and both branches, near-degenerate sectors included) times a BCS-like total amplitude $`\mathrm{clip}\big(0.5\,t_{\mathrm{eff}}/\sinh(1/\max(\lambda_{\mathrm{lin,max}},0.1)),\ \Delta_{\mathrm{floor}},\ c_{\max}\,t_{\mathrm{eff}}\big)`$ (with $`\Delta_{\mathrm{floor}}=`$ `_DELTA_ABS_FLOOR` and $`c_{\max}=`$ `_KICK_DELTA_MAX_FRAC`; the kick does not seed $`Q`$: `solve_self_consistent` starts at `_Q_SEED_THR`; the former `_KICK_Q_SEED` constant has been removed); otherwise a more conservative Stoner-proximity heuristic shrinks $`M`$ toward the RPA-safe region (`_KICK_REDUCTION_AMP`, centers `_KICK_M_EXCESS_CTR`/`_KICK_JCHI_EXCESS_CTR`) and seeds the same shape directly. `seed_complex_phase` (a dictionary `{channel: phase}`, §22) then sets the phase of the named channels in both Γ₇ branches, the anomalous amplitudes and $`\mu`$ are recomputed for the phased seed, and the gap-equation pass is run once. The kick returns `n_kspace`, `mu`, `M_kick`, `Delta_vec`, `F67_vec`, `alpha`, `lambda_lin_max`, `V_cap`, `Gamma_M` and `chi_QQ` (the $`q=0`$ RPA $`\chi_{QQ}`$ that seeds the phonon softening of §14); the former `B1g_exp` / `K_eff_n` entries and the `KICK-INIT`, `KICK-STATE`, `G_JT-BENCH` and `DOUBLET-CHK` log lines no longer exist (the cluster-vs-BdG $`\langle B_{1g}\rangle`$ checks are part of `preflight_report`). Either way this lands the iteration in the basin of the physically correct fixed point, and sets the starting mixing rate $`\alpha=\max(\alpha_{\mathrm{kick}},\ \alpha_0/(1+s_{\mathrm{kick}}\cdot\log1p(\lambda_{\mathrm{lin,max}})))`$, with $`\alpha_{\mathrm{kick}}=`$ `_KICK_MIXING_FLOOR`, $`\alpha_0=`$ `_MIXING` and $`s_{\mathrm{kick}}=`$ `_KICK_MIXING_SCALE`. The Anderson solver itself (`_anderson_mix`) is a scale-normalized, Tikhonov-regularized (`_ANDERSON_TIKHONOV`) normal-equation solve with a condition-number guard, a residual-acceptance test against simple mixing, a trust-region cap (`_ANDERSON_TRUST` $`\times`$ the simple step) and a blend weight between `_ANDERSON_W_LO` and `_ANDERSON_W_HI` set by $`\alpha/\alpha_0`$.

### Thread-Safety and Clone Protocol

The current `__main__` runs its three parallel SCF tasks (§ Installation & Usage) as three independently constructed `RMFT_Solver` instances (each built from a deep copy of `params`) rather than clones of a single solver, but the underlying clone protocol remains available for any code that mutates parameters mid-run (and is used internally by the Tc routines' temperature scans, via `_clone_solver_at_T`, which also sets `kT`, the temperature-scaled FS sample budget, and resets the transient caches):

```python
s = copy.copy(solver);  s.p = copy.copy(solver.p)
s.p.some_param = new_value
s._full_rebuild()
```

`_full_rebuild()` is the single canonical post-mutation refresh: it re-runs `p.__post_init__()` (SOC+CF diagonalization and the superexchange cluster ED), updates the bare stiffness `_K_bare`, rebuilds every orbital operator, and resets all transient caches (`_reset_transient_state`). Each clone owns its own `VectorizedBdG` and its own `_H_stack` buffer, so concurrent workers never alias each other's memory. At import time the module pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to 1 (via `setdefault`, so an explicit user setting wins) to prevent the BLAS backend from oversubscribing threads underneath an outer `ThreadPoolExecutor`.

### Finite-Difference ∂F/∂M and ∂²F/∂M²

The Newton step for $`M`$ (SCF Loop step 6) needs the free-energy gradient and curvature with respect to the three channel-resolved $`M`$ components. `compute_dF_dM_channels_and_hessian` obtains them by **finite differences of the beyond-mean-field canonical free energy** $`F_{\mathrm{bulk}}+\Delta F_{\mathrm{cl}}(M,Q)-\Delta F_{\mathrm{cl}}(0,Q)`$ through the shared `_fd_grad_hessian` core (step $`\varepsilon_c=\max(10^{-4},\,10^{-3}\vert M_c\vert)`$; 3-point diagonal and 4-point cross stencils; the gradient reuses the diagonal's $`F(x\pm\varepsilon)`$), with $`\mu`$ refit and the dynamic Gutzwiller factors re-evaluated at every probe — so the curvature is that of the *fixed-density* problem and includes the density-dependent renormalizations. (Earlier versions obtained these analytically from first- and second-order perturbation theory on the already-computed eigensystem; that code path has been replaced.) The same core serves `compute_hessian` ($`n=5`$) and `compute_dF_dDelta_and_d2F` ($`n=84`$, with the active/inactive treatment of §21).

---

## Parameters

All energies in **eV**, lengths in **Å**. Defaults below are the values set in the current `__main__` block, verified directly against the source.

### Primary Inputs (`ModelParams`)

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `t_pd` | $`t_{pd}`$ | 0.370 eV | $`p`$–$`d`$ hybridization integral; the single primary hopping input ($`t_0=t_{pd}^2/\Delta_{CT}`$ is derived) |
| `t_prime_ratio` | — | −0.17 | 2nd-neighbor (diagonal) hopping ratio; $`t'=`$ this $`\times\,t_0`$ |
| `t_dprime_ratio` | — | 0.08 | 3rd-neighbor (axial) hopping ratio; $`t''=`$ this $`\times\,t_0`$ |
| `U_dd` | $`U_{dd}`$ | 2.200 eV | On-site Hubbard repulsion — a **primary** input, not derived from a dimensionless ratio |
| `lambda_soc` | $`\lambda_{SOC}`$ | 0.024 eV | Atomic SOC constant on the $`t_{2g}`$ shell; sets the Γ₆–Γ₇ splitting together with `Delta_tetra` |
| `Delta_tetra` | $`\Delta_{\mathrm{tetra}}`$ | −0.057 eV | Axial (tetragonal) crystal field, $`\Delta_{\mathrm{tetra}}\cdot L_z^2`$; negative = $`z`$-axis compression |
| `g_JT` | $`g_{JT}`$ | 0.315 eV/Å | B₁g electron–phonon (JT) coupling |
| `K_lattice` | $`K`$ | 2.300 eV/Å² | Bare B₁g phonon spring constant; `K_eff` is computed at runtime (§11). The source comment gives the window for an SC-triggered JT: $`K_{\mathrm{spont}}\approx g_{JT}^2/\Delta_{CF}`$ < $`K`$ < $`g_{JT}^2/(0.05\,\pi\,t_0)`$ |
| `lambda_hop` | $`\lambda_{\mathrm{hop}}`$ | 1.200 Å | Hopping-anisotropy decay length: $`t(Q)=t_0\exp(\pm Q/\lambda_{\mathrm{hop}})`$ |
| `g_Eg2` | $`g_{Eg2}`$ | 0.100 eV/Å | Eg,2-channel electron–phonon coupling (§7) |
| `K_lattice_Eg2` | $`K_{Eg2}`$ | 6.500 eV/Å² | Bare Eg,2 phonon spring constant |
| `Delta_CT` | $`\Delta_{CT}`$ | 1.750 eV | Charge-transfer gap |
| `Delta_B1g_static` | $`\Delta_{\mathrm{ip}}`$ | +5×10⁻⁵ eV | Static in-plane crystal field, $`(L_x^2-L_y^2)`$; drives the D₄h→D₂h crossover (§1, §3); any non-zero value lifts the exact D₄h cancellation of the linear $`Q\vert\Delta\vert^2`$ coupling |
| `hybrid_scale` | — | 5.000 | Downfolding coordination factor entering the ZRS spectral weight $`\beta^2(k)`$ (§6, §9) |
| `Upp_ratio_bare` | — | 0.400 | Bare (unhybridized) $`U_{pp}/U_{dd}`$ ratio entering the internally-computed $`U_{pp}`$ (§4) |
| `J_H_ratio` | — | 0.09 | Kanamori Hund's coupling for the virtual $`d^2`$ superexchange intermediate state, as a fraction of $`U_{dd}`$, i.e. $`J_H=r_J\,U_{dd}`$ with $`r_J`$ the value of this parameter (§4) |
| `M_eff` | — | 8.0 amu | Normal-mode reduced mass for the B₁g phonon; sets the phonon energy $`\hbar\omega=64.6528\,\mathrm{meV}\sqrt{K/M_{\mathrm{eff}}}`$ (softened by the electronic JT susceptibility) used by the adiabatic retardation factor (§14). For this octahedral B₁g-type distortion the metal ion sits close to a symmetry-imposed node, so the mode is usually ligand-dominated |
| `kT` | $`k_BT`$ | 0.005 eV | Temperature ($`\approx`$ 58 K) |
| `M_axis` | — | `'z'` | Direction of the staggered (Néel) moment: `'x'`, `'y'` (∥ the two metal–ligand bond directions) or `'z'`. The AFM Weiss field, the $`M`$ observables, the spin vertices of the susceptibilities and the cluster exchange use the doublet-projected spin $`S_{\mathrm{axis}}`$ (§1); the paper's quenching theorems are derived for `'z'` only |
| `tol` | — | $`10^{-4}`$ | SCF convergence threshold |
| `trw_far_scale` | — | 1.0 | Multiplier of the transverse-Weiss-field weight of every **non-NN** bond class (NNN, 3NN): 1 = the $`(t_\delta/t)^2`$ estimate, 0 = far-bond TRW off, > 1 = stress test (§9). A field with a default, so it is not set in `__main__` |

The `__main__` block uses a single target doping `target_doping = 0.160`. The square-lattice coordination number is the module constant `_Z = 4` (it is no longer a `ModelParams` field and not passed to the constructor); the plaquette cluster uses `_Z_EFF = _Z − n_intra = 2` external neighbours.

### Derived Quantities (from `__post_init__`)

| Quantity | Origin | Description |
|---|---|---|
| `Delta_CF`, `g7split` | SOC+CF diagonalization | Γ₆–Γ₇ₐ gap and (signed) Γ₇ᵦ–Γ₇ₐ internal splitting — **not** free parameters |
| `sz_op` | exact $`S_z`$ diagonalization in each Kramers doublet | 6 components $`[sz_{6\uparrow},sz_{6\downarrow},sz_{7a\uparrow},sz_{7a\downarrow},sz_{7b\uparrow},sz_{7b\downarrow}]`$ |
| `S_ops`, `S_inter`, `S_axis_op` | doublet projection of $`S_x,S_y,S_z`$ | Néel-axis spin operators, block-diagonal per doublet (`S_axis_op` $`=`$ `diag(sz_op)` for `M_axis='z'`) |
| `_w_orb` (3×3) | eigenvector projections | $`d_{xz}/d_{yz}/d_{xy}`$ orbital weights (rows Γ6/Γ7a/Γ7b) feeding the exchange anisotropy and the `Tx_A_*` hopping projectors |
| `kappa_doublets` (3), `kappa_cross` (2) | multi-orbital cluster ED, §4 | Intra-doublet (Γ6, Γ7a, Γ7b) and cross-doublet (Γ6–Γ7a, Γ6–Γ7b) exact-diagonalization exchange coefficients |
| `Tx_A_xz/_yz/_xy` | orbital-character projectors | Rigorous orbital-selective hopping building blocks (§6); sum to $`I_6`$ |
| `t0`, `t_prime`, `t_dprime` | $`t_{pd}^2/\Delta_{CT}`$ and the `*_ratio` inputs | Effective nearest/2nd/3rd-neighbor $`dd`$ hopping |
| `U_pp`, `J_H` | `Upp_ratio_bare`, `J_H_ratio` (§4) | Ligand hole–hole repulsion and absolute Hund's coupling; used to build the superexchange cluster Hamiltonian and **stored** as attributes |
| `J2_over_J1`, `J3_over_J1`, `Z_afm_eff` | $`(t'/t)^2`$, $`(t''/t)^2`$, $`Z(1-J_2/J_1-J_3/J_1)`$ | Further-neighbor exchange ratios and frustration-reduced AFM coordination (§6) |
| `trw_irrep_scale` (42) | `_TRW_BOND_FRAC` · class weights $`(1,1,J_2/J_1\,f,J_3/J_1\,f,0)`$ | Bond-class weight of the transverse Weiss field of every pairing channel (§9) |
| `thermal_fs_nat_size` | $`\max(\lfloor kT/t_0\cdot N_K^2\rfloor,\,32)`$ | Adaptive Fermi-surface sample-point budget, with $`N_K=`$ `_NK` (clones at other temperatures use $`\lfloor T/t_0\cdot N_K^2\rfloor`$) |
| `k_points`, `k_weights`, `N_k`, `shift_table` | $`N_k=`$ `_NK`² uniform grid | k-space infrastructure, including the cyclic shift table used for arbitrary-$`q`$ Lindhard sums |

`B1g_op`, `B1g_offdiag`, `Eg2_op` are also set on `ModelParams` in `__post_init__` (§3, §7); the corresponding 24×24 Nambu lifts `B1g_24`, `Eg2_24`, plus `Sz_nambu`, the per-channel staggered `Sz_stag_nambu_channels` and uniform `Sz_nambu_channels`/`_sz_nambu_diag_channels` vertices, are built on the `RMFT_Solver` instance by `_rebuild_orbital_operators`, along with `phi_k = cos(kx) − cos(ky)`, the full form-factor matrix `Phi_irrep_k` ($`N_k\times42`$) on the SCF grid and the bare coupling `g_JT_bare` (§14); the phonon energy is no longer stored but computed inside `_make_vertex_params` (§14).

### Module-Level Constants

The constants tables (grid and iteration budget, pairing basis, Newton/trust-region steps, saddle-escape, chemical potential, Lindhard/FS sampling, pairing kernel, RPA/QCP guards, limit-cycle and Anderson mixing, gap update, superexchange, $`T_c`$ estimation, thresholds, and the list of constants that no longer exist) are on their own page: **[docs/constants.md](docs/constants.md)**. The source file remains the single source of truth for every value.

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

The module is CPU-heavy: per SCF iteration the plaquette ED diagonalizes $`1296\times1296`$ matrices for three $`Q`$ stencils plus the susceptibility references, the $`K_{\mathrm{eff}}`$ stencils run several canonical free-energy evaluations, and the exact $`Q`$ profile (§12) repeats the cluster ED and a $`\mu`$ solve at every scan, root and curvature point — on the production $`60\times60`$ grid (`_NK = 60`) a single SCF run is a very long computation, which is why no automatic restarts are performed. The BLAS thread count is pinned to 1 per worker and the three comparison runs execute concurrently.

On startup, the current `__main__` block does the following, in order:

1. **Parameter setup and SOC+CF diagonalization.** `ModelParams(...)` is constructed with the defaults listed in [Parameters](#parameters) and `__post_init__` runs the SOC+CF diagonalization (Γ₆/Γ₇ₐ/Γ₇ᵦ identification, `Delta_CF`, `sz_op`, spin operators, k-grids, orbital operators) and the multi-orbital superexchange cluster ED (`kappa_doublets`, `kappa_cross`, §4). The `INIT` and `DERIVED` log lines report the inputs (including $`Z=`$ `_Z` and $`N_k`$), $`U_{pp}`$, $`J_H`$, $`t_0`$, $`\Delta_{CF}`$, $`\kappa_{\Gamma_6}`$, the $`\Gamma_7`$ splitting (flagged if its modulus is below $`2kT`$), the **J-chain** of the leading exchange ($`\kappa_{\Gamma_6}`$ from the two-hole ED $`\to\times g_J\to`$ single-bond $`J_{A1g}\to\times Z_{\mathrm{afm,eff}}\to J_q(\pi,\pi)=J_{\mathrm{eff}}`$, the Weiss-field scale) and whether $`\Gamma_{7b}`$ lies inside the estimated $`\Gamma_6/\Gamma_{7a}`$ bandwidth ($`8\,g_t\,t_0`$).
2. **Doping setup.** `target_doping = 0.160`, with a symmetric ±20% scan margin (`doping_margin = 0.20`) defining `min_doping`/`max_doping`, floored to stay above the `_G_T_COHERENCE_MIN` Mott-incoherence boundary. These bounds are computed but, in the current `__main__`, not otherwise consumed — all three solves below run at the single point `target_doping`, not a scan. The initial gap and anomalous amplitudes are `np.zeros(_N_DELTA)` (`Delta_vec_ans`, `F67_vec_ans`) — there is no manually chosen cold-start amplitude: `_scf_kick` seeds the 84-component gap from the leading sector eigenvector(s) of the pairing kernel (§ Key Algorithms). The seed phases are set by `seed_complex_phase = {'s': 0.0, 'd': 0.0}` (a dictionary of channel phases; the commented alternative `{'s': 0.0, 'd': np.pi/2}` seeds $`s+id`$).
3. **Three-way self-consistent comparison, run in parallel.** Three independent `RMFT_Solver` instances are built from deep copies of `params` and run concurrently via a 3-worker `ThreadPoolExecutor`:
   - **`ref`** — the full self-consistent solve, SC (all 84 channels) and $`Q`$ both free (the model's actual prediction); the only run with verbose logging.
   - **`normal`** — the normal (non-superconducting) AFM state, built by `_solve_normal_state` rather than by a full SCF: the normal-state $`M`$, $`\mu`$ and Gutzwiller factors from `refine_M_state` at $`Q=0`$, then up to 12 relaxations of $`Q`$ with `solve_Q_slave` starting from $`Q=0`$ (no vertex cache, $`\Delta\equiv0`$), and $`F_{\mathrm{bdg}}=F_{\mathrm{can}}+\Delta F_{\mathrm{cl}}-\Delta F_{\mathrm{cl,ref}}`$ (§18). It returns no SCF history beyond a single `F_cluster` point.
   - **`SC_Q0`** — SC free but $`Q`$ pinned to zero (`force_Q_zero=True`); superconductivity without the JT relaxation.

   This is the direct numerical test of the central hypothesis: if the theory is right, `F_bdg` should order `ref < normal` (condensation lowers the free energy) and `ref < SC_Q0` (JT relaxation lowers it further), and `Q` should relax back toward 0 in `normal` *without being forced there* (up to the linear response to the small static `Delta_B1g_static`, §17). The code logs `"SC+JT is the ground state ✓"` or `"✗"` accordingly — but only when the reliability check of `compute_and_log_hessian_diagnostics` passes for all three (each must be converged, **not past the AFM QCP**, and have a Hessian without negative eigenvalue); otherwise it logs `"comparison NOT trustworthy"`.
4. **Post-SCF diagnostics** (only if the reference run returned a result): the `REF-SCF` summary (converged flag, raw iteration-convergence flag, Mott-suspect, past-QCP, final AFM determinant, $`M`$, $`Q`$, $`\mu`$, $`\Delta_s`$, $`\Delta_d`$), the selection ratio (`IrrepR`) with its `JT=ALLOWED/BLOCKED` tag; the **`JT (F_total)`** verdict from the normal and SC $`Q`$ profiles and the **`K_eff (diag, Q=0)`** line with its parts (§11, §17); the **`STATIC-FIELD`** test (§17); the **Hessian / free-energy report** `compute_and_log_hessian_diagnostics` (`HESS-DIAG`, `FREE-ENERGY`; §16, §17); `lambda_lin_max` and the coherence-length summary; and the three $`T_c`$ estimates (§23) with the spectral-gap record of $`T_{c3}`$, compared with the 230 K target.
5. **One diagnostic plot.** `plot_ground_state_comparison(results)` is meant to produce the `ref`/`normal`/`SC_Q0` comparison figure, saved to disk (see [Output & Diagnostics](#output--diagnostics) and the corresponding Known-Limitations row).

---

## Output & Diagnostics

All output is a structured, thread-safe log stream (`_scf_log`, tagged by stage — `INIT`/`DERIVED`, `PREFLIGHT` (the pre-run regime and state report of `preflight_report`), `REF-SCF`, `PARALLEL`, `SCF-A/B/C/D` (the per-iteration blocks), `SCF` (α-recovery), `SCF-RES`, `SCF-EG2`, `Q-SLAVE` (the $`Q`$ window of every verbose iteration), `Q-CYCLE` (the $`Q`$-oscillation guard), `REFINE-M`, `SADDLE-ESC`, `LIMIT-CYCLE`, `HESS-DIAG` and `FREE-ENERGY` (the post-run Hessian / ground-state report), `GAP-DIAG`, `PAIR-DIAG`, `PAIR-WARN`, `SECTOR`, `FS-DEBUG`, `TC-PRELIM`, `TC-THERMO`, `TC-COOL`, `RMFT-WARN`, `CLUSTER-WARN`, `PLOT`) rather than a GUI; the graphical output is the one `matplotlib` figure described below. (The startup banner text of `__main__` is legacy wording from an earlier optimizer-based pipeline and does not describe the current flow.)

### Iteration Log

Each logged SCF step (verbose run) is split into four lines:

```
[SCF-A] iter/max  conv=...  M=[...]  Q=... A  mu=...  J_eff=... eV  g_Delta_ad=...  g_Delta_bd=...  F_can=... eV  F_cluster=... eV  alpha=...  ...s/it
[SCF-B]   Q_fluct=...  Gamma_M=... meV  K_eff=...  [K_bulk=... (E_cond''=...)  + K_cluster=...  K_schur=...]
[SCF-C]   |Delta|_2=...  Delta: top-4 amplitudes (of 84)   F67: top-4 amplitudes
[SCF-D]   Stoner: FM=...  SDW=...  V_JT_corr*chiQQ(q): (0,0)=...  (pi,pi)=...   chiSQ_sc(pi,pi)=...  chiSQ_sc(q=0)=...   V_diag: top-4 signed diag of V_irrep   V_sd=...  lead=(sector)
```

preceded, in a verbose run, by a `Q-SLAVE` line with the $`Q`$ window and the $`\pm Q`$ degeneracy flag (§12) and, when the guards fire, `Q-CYCLE`, `LIMIT-CYCLE`, `SADDLE-ESC` and `SCF` (α-recovery) notices. The seeding stage no longer has its own tags; its information is part of the `PREFLIGHT` report, which `_make_vertex_params(..., verbose=True)` prints: the energy scales ($`t_{\mathrm{eff}}`$, $`\hbar\omega`$, $`\Delta_{CF}`$, the $`\Gamma_7`$ splitting) and one flag per regime condition ($`\hbar\omega/\Delta_{CF}`$ < 1, $`U_{dd}/t_0`$, $`J_{\mathrm{eff}}/t_{\mathrm{eff}}`$, the adiabaticity ratio $`\hbar\omega/t_{\mathrm{eff}}`$, $`\Delta_{CF}/t_{\mathrm{eff}}`$, $`\Gamma_7`$ splitting against $`2k_BT`$, $`\Delta_{B_{1g}}/\Delta_{CF}`$, $`k_BT_c^{\mathrm{target}}/t_{\mathrm{eff}}`$ with the 230 K target) with a regime flag for each, the state content (doublet occupations, $`B_{1g}`$ ratio and weight, the cluster-vs-BdG $`\langle B_{1g}\rangle`$ benchmark), the Weiss-field amplitude $`h_{\mathrm{afm}}`$ against the bandwidth ($`Z\,t_{\mathrm{eff}}`$) and against $`\Delta_{CF}`$ together with the perturbativity of $`\kappa_{B_{1g}}/\Delta_{CF}`$, and the SCF state summary (`M`, `Q`, `|Δ|`, $`K_{\mathrm{eff}}`$, the regime `subcritical`/`critical`/`supercritical` of `lambda_lin_max`) and the Gutzwiller factors. (The $`J`$-chain of the leading exchange is logged by the `DERIVED` lines of `__main__`.) `PAIR-DIAG`/`GAP-DIAG`/`SECTOR` blocks report the spin/JT/cross decomposition of the pairing diagonal (with $`\lambda_{\mathrm{JT}}+\lambda_{\mathrm{spin}}+\lambda_{\mathrm{cross}}=\lambda_{FS}`$ from `lambda_channel_decomposition`), the irrep-projection capture of the dominant FS eigenvector (with an explicit warning if more than 25% lies outside the 42-function basis), the leading-channel weights, the FS-sample properties (boundary weight, $`T_Q`$ consistency, closure under the mirror maps, validation of the odd-$`k`$ sectors), and a table of the eight $`G`$-sector eigenvalues with their $`\sigma_d`$-resolved irreps and the library harmonics that would close any basis gap (including irreps that share a sector with a stronger one and are not followed by the sector seed).

At convergence, `SCF-RES` lines report: $`\partial\lambda_{\mathrm{pair}}/\partial Q`$(FS) and the hot-spot fraction, `lambda_lin_max`, the per-channel $`\lambda_{\mathrm{irrep}}`$ and $`\vert\Delta_{7a}\vert`$ (meV) with the gap-symmetry label and the linear-vs-SCF consistency, the unresolved-weight fraction, the incommensurate-nesting scan result (a flag only, §22), the SCF-dynamics regime and the ⚠ ANSATZ UNSTABLE / PAST-QCP flags, $`\chi_{SQ}^{SC}(\pi,\pi)`$ with its strength label, and the Mott-suspect notice; `SCF-EG2` reports the Eg,2 stability diagnostic (§7). The $`K_{\mathrm{eff}}`$, `JT (F_total)` and `STATIC-FIELD` lines of §17 are added by `__main__`.

### Free-Energy Ground-State Check

Logged by `compute_and_log_hessian_diagnostics` right after the three parallel tasks (`ref`, `normal`, `SC_Q0`) complete: the `HESS-DIAG` table lists, for each scenario, the beyond-mean-field $`F_{\mathrm{bdg}}`$, the smallest Hessian eigenvalue and the reliability status (`✓ ok`, or `✗` with the reason: not converged, past QCP, Hessian failed, saddle), and warns if the bulk-only $`F`$ from the Hessian and from the cluster routine differ by more than $`10^{-4}`$ eV (they should agree). The `FREE-ENERGY` block then prints $`F_{\mathrm{ref}}`$, $`F_{\mathrm{normal}}`$ and $`F_{\mathrm{SC,Q0}}`$ with $`\Delta E`$ in meV relative to `ref` and the one-line verdict — `"SC+JT is the ground state ✓"`, `"SC+JT is NOT the lowest energy state ✗"`, or `"comparison NOT trustworthy"` if any of the three results is unreliable — followed by the line with $`Q(SC)`$, $`Q(N)`$, $`\lvert Q_{SC}/Q_N\rvert`$, $`K_{\mathrm{eff}}(N,Q=0)`$ and its verdict (§17).

### RPA Vertex, Coherence, and SC-JT Diagnostics

If the reference SCF produced a result: the channel-resolved pairing diagnostics above; the coherence-length summary (flagging orbital-selective pairing when $`\Gamma_6`$ and $`\Gamma_7`$ channels have meaningfully different $`\xi`$ — computed from the exact FS gap magnitude $`\vert\Delta(k)\vert`$ and the nodal/antinodal regions of the gap *shape*); the `JT (F_total)`, `K_eff (diag, Q=0)` and `STATIC-FIELD` lines and the Hessian/free-energy report (§17). The Stoner factors are logged per iteration (`SCF-D`), not as a separate `__main__` line.

### Ground-State Comparison Plot

`plot_ground_state_comparison(results)`, a standalone module-level function (not a solver method), takes the `{"ref", "normal", "SC_Q0"}` result dictionary produced in [Installation & Usage](#installation--usage) step 3 and saves a **2×2** figure to `ground_state_comparison.png`: (top-left) `F_cluster` per SCF iteration for all three scenarios, each with its converged `F_bdg` as a dotted reference line, unreliable trajectories drawn dashed with the reason in the legend; (top-right) a bar chart of the three final `F_bdg` values, titled with whichever scenario is lowest (unreliable bars hatched in red); (bottom-left) the Γ₆-channel AFM order parameter $`M[\Gamma_6]`$ versus iteration, all three overlaid; (bottom-right) $`\vert Q\vert`$ versus iteration, the key visual check that `normal`'s $`Q`$ relaxes back toward 0 on its own (it is never forced there) while `ref` settles at a finite value. Missing scenario entries are skipped rather than erroring, provided at least one entry is present. (The panels that need the iteration traces read `history['M']` and `history['Q']` of every scenario; see the corresponding row of Known Limitations for the `normal` entry.) The debug-only hexbin plot of the old unique-$`q`$ sampling (`unique_q_hist.png`, from `_unique_q_pairs(verbose=True)`) is not produced by the current pipeline.

---

## Known Limitations

The framework makes a number of physically motivated approximations. The table below reflects the current code, not an earlier design pass.

| Approximation | Impact |
|---|---|
| No Pauli exclusion between plaquette sites | The four-site cluster's external bonds are closed with a mean-field Weiss embedding rather than genuine Pauli-respecting hopping to the rest of the lattice; mild overestimate of AFM correlations |
| No charge-transfer fluctuations $`\langle n_An_B\rangle`$ | Negligible when the mean-field exchange scale is large compared to the hopping |
| Phonon still adiabatic/quasi-static | $`Q`$'s own equation of motion is a thermodynamic equilibrium condition, not a dynamical one, and zero-point lattice fluctuations are neglected; the phonon-mediated pairing vertex carries an adiabatic retardation factor tied to a phonon energy $`\hbar\omega`$ derived from `M_eff` and the electronically softened stiffness (§14), but this is a one-point static-limit evaluation at the FS-maximum gap scale, not a full frequency-dependent (Matsubara) treatment |
| No thermal/quantum fluctuation correction to the lattice stiffness | The Moriya-SCR regularization of $`K_{\mathrm{eff}}`$ that earlier versions applied to the lattice channel is gone; $`Q`$ is the exact minimizer of the mean-field canonical free energy (§12), so thermal fluctuations of $`Q`$ itself are not renormalized (the spin channel keeps its Landau-derived $`\Gamma_M`$) |
| Γ₇ₐ remains the headline channel | Both Γ₆–Γ₇ₐ and Γ₆–Γ₇ᵦ pairing are fully self-consistent (§8), but several reported summary quantities (`Delta_s`, `Delta_d`, `selection_ratio`, `Delta_irrep`, the primary Anderson block) are built from the Γ₇ₐ amplitudes; the Γ₇ᵦ channel's own physics requires the second half of `Delta_vec`/`F67_vec` (`Delta_vec[_IDX_7B_ARR]`) |
| Basis completeness is checked, not guaranteed | The 42-harmonic library covers all eight $`G`$-sectors only up to its truncation order; an uncovered sector, or a covered one whose leading eigenvector has a large `unresolved_weight`, signals a basis gap (reported by `SECTOR`/`GAP-DIAG`); `lambda_lin_max_all` can exceed `lambda_lin_max` when an *uncovered* sector is the more unstable one. Odd-parity ($`p`$-type) channels are additionally gated by an FS-closure check and are disabled in the kernel if it fails |
| Linear kernel vs. nonlinear outcome | The leading linear sector need not be the symmetry the nonlinear BdG selects (near-degenerate sectors are seeded together, and a mismatch is reported in the `gap_symmetry` label). No automatic re-seeding is performed for any mismatch; restarting from another seed is left to the user |
| Resolution mismatch between the RPA and cluster-ED layers | The itinerant RPA vertex (§14) resolves three separate spin channels (Γ6, Γ7a, Γ7b); the local cluster-ED irreducible-vertex extraction (§19) works in a 2-channel (aggregate spin, B₁g) space, embeds only the leading Γ₆ channel's magnetization in the stiffness stencils and the vertex extraction, and feeds back only the on-site anomalous amplitudes into its Weiss field, so `V_irr_QQ` does not resolve Γ7a- vs. Γ7b-specific or bond-channel-specific local corrections |
| No spatial fluctuations | Cannot describe a pseudogap, stripe order, or phase separation |
| RPA static ($`\omega=0`$) | Dynamical vertex corrections beyond the JT-channel retardation factor above are absent; the spin channels have no analogous retardation treatment |
| Cluster stiffness uses a NN-only mean-field reference and a capped Schur term | $`K_{\mathrm{MF}}`$ is built with $`Z_{\mathrm{MF}}=Z`$ (the cluster has no $`J_2/J_3`$ bonds), whereas the bulk longitudinal Weiss field uses $`Z_{\mathrm{afm,eff}}=Z(1-J_2/J_1-J_3/J_1)`$ — a ≈5% mismatch of the mean-field level that is not propagated into the cluster (it shifts $`\Delta K`$ by a few percent in a test state). The beyond-mean-field correction $`\Delta K_{\mathrm{cl}}`$ itself is **not** capped any more; only the Schur term is limited, to $`\lvert K_{\mathrm{bare}}\rvert`$, and set to 0 when $`H_{\Delta\Delta}`$ is not positive. The stiffness stencils embed only the Γ₆ magnetization and the cluster feeds back only the on-site anomalous amplitudes |
| $`Q`$ profile is a local window search with an expensive objective | The profile is minimized only inside a trust window of half-width $`s_Q(0.006+0.5\lvert Q_{\mathrm{prev}}\rvert)`$ Å on 7 points (§12): a distant, deeper basin is reached only by the dynamics (seeds, kicks, $`\alpha_Q`$ relaxation) and never by the search itself, and when the minimum sits on the window edge the edge value is returned (flagged `edge`). The sign of $`Q^*`$ is meaningful only if the $`\pm Q`$ splitting exceeds the window depth (the `Q-SLAVE` ✓/⚠ flag). The objective includes the exact cluster energy, so every probe costs a cluster ED and a $`\mu`$ solve; the Eg,2 amplitude is held at 0 in the search; without a vertex cache the condensation energy uses a diagonal fallback matrix |
| Simplified normal-state stability check | The normal-state JT stability is read off the normal state's own $`Q`$ profile and $`K_{\mathrm{eff}}(N,Q=0)`$ (§16, §17); it is exact for the pure-JT instability direction but does not itself diagnose a genuinely *mixed* (e.g. AFM–JT coupled) normal-state instability — the post-convergence Hessian mode classification (§16) carries that role, but only after a solve has converged near the relevant point. The `normal` reference is relaxed in $`Q`$ only (normal-state $`M`$ from `refine_M_state` at $`Q=0`$ is kept fixed) |
| $`\partial\lambda_{\mathrm{pair}}/\partial Q`$ at a frozen Fermi surface | FS geometry is evaluated at a fixed $`Q`$ rather than self-consistently re-resolved; a fully SC-state Bogoliubov–Lindhard version would be more expensive |
| SCF-dynamics regime classification | `first_order_jump`, `hysteretic` and `limit_cycle` are heuristic labels from the shape of the $`\sum\vert\Delta_i\vert`$ (and, separately, $`\vert Q\vert`$) history, and the $`Q`$ zig-zag guard is a heuristic of the signed $`Q`$ iterates; they only damp the mixing rate (and shrink $`\alpha_Q`$ and the $`Q`$ window) and clear the Anderson history — no multi-seed restart is launched |
| Eg,2 channel not self-consistent | `Q_Eg2` is a fixed input (default 0); the Eg,2 phonon is wired into the Hamiltonian, the free energy and a post-convergence stability diagnostic, but its exchange-driven rigidity correction and cross-rigidity with the B₁g channel are not computed; holding it at 0 is the symmetric stationary point ($`E_g`$ has no $`A_g`$ component in $`D_{4h}`$ or $`D_{2h}`$) on the real, $`\Theta`$-invariant branch of the condensates (see the flat-plane row below) |
| Only the B₁g tensor of the Γ₆–Γ₇ cross-exchange block is kept | `_cross_doublet_kappas` returns the coefficient of $`B_{1g,\mathrm{op}}(i)\,B_{1g,\mathrm{op}}(j)`$ only; the B₂g and E_g tensors of the block (≈70 % of its squared Frobenius norm for the paper's reference parameters; the B₁g term carries ≈30 %) are not in the Hamiltonian. This is a controlled truncation justified by a selection rule: a term linear in the B₁g coordinate $`Q`$ needs a B₁g ($`A_g`$ in $`D_{2h}`$) partner, so the B₂g/E_g tensors can enter only at $`\mathcal O(Q^2)`$, as a correction to $`K_{\mathrm{eff}}`$ (a quantitative threshold shift, not a change of the linear mechanism). The 30 % measures the scope of the approximation, not a physical prediction; in the cluster $`J_{B_{1g}}=J_{B_{2g}}`$ ($`J_z`$ conservation), so the omission rests on the selection rule, not on a smaller coupling |
| Flat, $`\sigma_h`$-symmetric plane; $`Q_{Eg2}=0`$; `M_axis='z'` and the $`S_z`$ vertex | The paper's decoupling theorems for B₂g and E_g and the premise of the hypothesis (the AFM order alone cannot induce a B₁g strain) are derived for this case: exact $`\sigma_h`$, the real ($`\Theta`$-invariant) branch of the condensates, a Néel axis along $`z`$. A static $`\sigma_h`$-odd distortion (tilts about in-plane axes, buckling: $`C_{2h}`$ or lower), a chiral $`7a`$–$`7b`$ relative phase of $`\pm\pi/2`$ (which couples linearly to $`Q_{Eg2}`$), or an in-plane Néel axis (where $`M_x^2-M_y^2`$ couples linearly to $`Q_{B_{1g}}`$ and the dropped inter-doublet spin pieces are its natural carriers) define different regimes in which these statements must be re-derived, not refuted |
| The pairing-kernel symmetry decomposition uses a reduced group | Sectors are the characters of $`G=\{E,T_Q\}\times D_{2h}`$, not irreps of the full magnetic group: a sector can contain several D₄h irreps (e.g. `es` (A₁g) and `d` (B₁g)), and $`\Theta`$ together with all internal (spin, orbital, sublattice) structure enters through the normal-state propagators, not through the sector decomposition. The FS sample (`_get_fs_points`) holds one representative of each pair $`k\leftrightarrow k+(\pi,\pi)`$ (the diamond boundary with weight ½ per $`T_Q`$-paired point) and is not $`G`$-closed; the $`\sigma_d`$ split of a sector (`sector_swap_lambda`) is an exact irrep resolution only if the kernel is $`C_4`$-covariant (`sector_swap_defect` = 0): sector decisions use the exact $`G`$-closed kernels (§15), while irrep projections of eigenvectors computed on the sample can mix the two $`T_Q`$ sectors numerically |
| Incommensurate AFM handled only as a diagnostic | `_scan_incommensurate_nesting` (a staggered-vertex approximation off $`(\pi,\pi)`$) detects a preference for $`q^{*}\neq(\pi,\pi)`$ and flags it; the BdG Hamiltonian itself remains fixed at commensurate $`(\pi,\pi)`$ ordering and no re-solve is triggered — a genuinely incommensurate spiral solve is not implemented |
| Sign-flip EMA on the pairing diagonal | Suppresses numerical oscillation in the diagonal of $`V_{\mathrm{irrep}}`$ for the channels that change sign under the AFM shift, but may slow the genuine response near a doping-driven crossover between channels |
| Superexchange cluster is a 3-site chain with independent ligand modes | `_multiorbital_2hole_hamiltonian` (§4) models the ligand as three independent per-$`L_z`$ modes rather than a single fully hybridized ligand orbital; this preserves $`J_z`$ through the SOC mixing but is itself an approximation to the true ligand electronic structure; the intra-doublet $`\kappa`$ is additionally floored through `_S_EFF_FLOOR` in the spin-switch regime |
| Cluster-ED vertex is a $`q=0,\omega=0`$ local estimate | `V_irr_QQ` from the four-site plaquette (§19) is a local quantity standing in for a genuinely $`k`$-dependent, dynamical vertex; together with the $`K_{\mathrm{eff}}`$ stencils and the exact $`Q`$ profile the cluster evaluation is the dominant cost of an SCF iteration |
| Tc diagnostics | The reported coupling ratio is $`2\Delta_{\mathrm{ref}}/k_BT_c`$ with the thresholds of the weak-coupling s-wave value and no gap-shape correction (a pure d-wave state lands in the `very-strong` band) and is marked `undetermined` unless the SCF state is saturated. $`T_{c2}`$ (`compute_Tc_linearized_normal`) freezes $`M`$ and $`\mu`$ at their normal-state call values, uses a linearly interpolated crossing on a 14-point grid, and its un-projected map need not agree with the SCF onset when the leading eigenvector is not in the attractive subspace; $`T_{c3}`$'s crossing is interpolated on a coarse temperature grid |
| TRW bond weights are a hopping-counting estimate | The weight of the transverse Weiss field of a longer-range pairing channel follows $`(t_\delta/t)^2`$ (§9); the bulk transverse field uses the bare $`Z`$ while the longitudinal field uses $`Z_{\mathrm{afm,eff}}`$. `trw_far_scale` allows stress-testing the far-bond weights |
| No automatic restarts | The incommensurate re-solve and the d-wave-forced re-solve were removed because a single SCF run can take very long; the user decides whether to restart from a different seed |
| Current code state: the `normal` result has no iteration history | `_solve_normal_state` returns a one-point `history` (`F_cluster` only), whereas `plot_ground_state_comparison` indexes `history['M']` and `history['Q']` of every scenario; as the code stands, the bottom two panels — and with them the final figure — cannot be drawn for the `normal` entry (a `KeyError`) until those traces are supplied or skipped |
| Current code state: the SCF calls inside $`T_{c3}`$ | The `solve_self_consistent(...)` calls of `compute_Tc_thermodynamic` (SC basin, normal basin, cooling pass) pass `initial_Delta`/`initial_F67` by keyword but omit the required positional argument `seed_complex_phase`; the resulting `TypeError` is caught and logged (`TC-THERMO` / `TC-COOL`: "… raised TypeError …"), each point is scored `F=1e30` (SC loses) or $`\Delta=0`$, so $`T_{c3}`$ carries no information until a `seed_complex_phase` is passed |

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
