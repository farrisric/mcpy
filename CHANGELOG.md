# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-07-28

### Fixed
- **Batched replica exchange accepted every chemical-potential swap.** `BatchedReplicaExchange._accept_swap` implemented only the temperature-ladder criterion `(beta_j - beta_i)(Phi_j - Phi_i)`, which is identically zero when the replicas share a temperature: every mu-ladder swap was accepted with p = 1, configurations random-walked freely across the ladder, and no replica sampled its own mu. **Results produced by `BatchedReplicaExchange(..., mus=[...])` in 1.3.0 or earlier are invalid and need re-running**; temperature ladders were always correct, as was the MPI `ReplicaExchange`, which selected its mu criterion all along. The replacement carries the cross terms, `beta_i Phi_X^(i) + beta_j Phi_Y^(j) - beta_i Phi_Y^(i) - beta_j Phi_X^(j)`, where `Phi_Z^(k)` is a configuration scored with slot `k`'s chemical potentials; it reduces to the previous expression for a shared mu, to `beta (mu_i - mu_j)(N_j - N_i)` for a shared temperature, and additionally covers a joint (T, mu) ladder. The failure was not a statistical degradation: an EMT Ag(111)/Au ladder gives `N_Au = [0, 0, 0, 14]` with the fix and `[0, 0, 0, 0]` without it, because unconditional swapping averages the rungs together and erases the mu dependence entirely.
- **`PermutationMove` could mutate the configuration and then report that it had not.** With `n_swaps > 1`, drawing a species absent from the system in a later iteration returned the "could not propose" sentinel after earlier swaps had already been applied; both GCMC loops read that sentinel as "the atoms were not touched" and skipped the rollback, leaving the configuration out of step with the stored energy for the rest of the run. The usable species are now resolved once, before any mutation (species counts are swap-invariant, so a species absent at the start is absent for the whole trial), and both ensembles restore their pre-trial snapshot on the sentinel path rather than trusting the contract.
- Molecules whose center of mass drifted above a `CustomCell`'s top stopped being deletion candidates and dropped out of the per-species de Broglie count, inflating `V/((N+1)Lambda^3)` into the runaway insertion mode documented in `docs/gcmc_acceptance_convention.rst`. Molecule candidacy now goes through the new `is_point_exchangeable` predicate, which applies the same dropped z upper bound that `get_atoms_specie_inside_cell` already applied to single atoms; molecules below the cell floor stay excluded, as atoms do.
- `AlchemiCalculator(energy_only=True)` discarded `'forces'` from a `model_config` that a pre-loaded `MACEWrapper` shares with every calculator built from it, silently disabling FIRE relaxation in an `AlchemiFCalculator` depending on construction order. The combination now raises with the workaround in the message.
- `GrandCanonicalEnsemble.set_state` and `CanonicalEnsemble.set_state` restored the step count and exchange statistics from the incoming state. `ReplicaExchange` passes a full `get_state()` dict on every accepted swap, so the two ranks traded their swap tallies and the per-rank "Accepted Exchange (%)" column was meaningless. Only the configuration travels now.
- Per-interval acceptance ratios were never cleared when the outfile was disabled, silently degrading `interval_ratios()` into `total_ratios()` for the rest of the run.
- A missing `species_radii` entry raised a bare `KeyError` from inside the free-volume sampler; the error now names the cell and the missing species, and notes that species inserted during the run need radii too.

### Added
- `Cell.is_point_exchangeable(point)`: the point counterpart of `get_atoms_specie_inside_cell`, used by the molecule moves to decide which molecules the reservoir may take back. It defaults to `is_point_inside`, so the box, spherical and dome cells are unchanged; `CustomCell` overrides it. `MoleculeDisplacementMove` tests the same predicate for its region guard, so candidacy and displacement always agree on the region.
- `BatchedReplicaExchange` warns at the end of a run when the whole-run swap tally is all-accept or all-reject, the two ways a ladder stops being a ladder. Checked on the whole-run tally rather than live, because replicas that have not differentiated yet legitimately accept every early swap.
- `--chunk-size` on `examples/re_gcmc_co_cupd_batched.py`: peak GPU memory follows the largest chunk rather than the replica count, which is what lets a correctly spaced ladder run at all (an acceptance-equalized ladder for that system needs ~29 rungs, i.e. ~13k atoms in the relax batch, several times the whole-batch ceiling).
- `docs/replica_exchange_ladder_spacing.rst`: how to detect a dead ladder (read second-half swap acceptance, never the cumulative column, which early free swaps inflate permanently), why uniform mu spacing cannot work across a coverage range (`dN/dmu` grows with coverage while `dmu` does not), and the `dmu = 1/sqrt(beta dN/dmu)` spacing rule with a reference implementation. Worked example on CO/Cu375Pd30 at 400 K: 29 rungs instead of 5 gives second-half acceptance of 18-65% (median 40%) with no dead pair, against three of four pairs at exactly zero accepted swaps on the uniform ladder, and GPU utilisation of 94-98% instead of 43%.
- CI runs `flake8 mcpy/`, the lint command the contributor docs already specified but the workflow never invoked.

### Changed
- The outfile acceptance-ratio header abbreviates the molecule moves distinctly (`MolIns`, `MolDel`, `MolDis`) instead of collapsing all three to an indistinguishable `Mol`. Single-word move labels (`Ins`, `Del`, `Dis`, `Per`, `Sha`, `Bro`) are unchanged, so outfiles from atomic runs stay comparable across versions.

## [1.3.0] - 2026-07-08

### Added
- **Molecular GCMC**: `MoleculeInsertionMove` and `MoleculeDeletionMove` exchange whole rigid molecules (any ASE-buildable template) with the reservoir, using the textbook rigid-molecule acceptance (per-species in-cell molecule count, de Broglie wavelength from the total molecular mass, full molecular chemical potential with orientations sampled uniformly). Atomic moves keep their documented convention; the two coexist (`docs/gcmc_acceptance_convention.rst`).
- `MoleculeDisplacementMove`: rigid translate+rotate of one molecule, with an optional `max_angle` rotation cap for strongly anchored adsorbates (measured on CO/CuPd: acceptance 5% -> 42.5%, roughly 2x faster convergence than exchange-only sampling).
- Molecule bookkeeping via a per-atom `molecule_id` array: rolls back with the existing rejection snapshot, shrinks correctly on deletion, and round-trips through extxyz trajectories (declared in `Properties=`), making molecular runs restartable.
- Atomic and molecular species can coexist in one simulation (e.g. dissociative O at `mu_O = mu_O2/2` alongside molecular O2): atomic insertions tag their atoms as free, atomic deletions never touch molecule members.
- `SetUnits` accepts a `molecules` dict (name -> ASE template) for molecular masses and wavelengths, rejecting isomer compositions and atomic/molecular name collisions.
- `plot_phase_diagram`: `adsorbate_label` and `atoms_per_reservoir_molecule` for molecular adsorbates (correct pressure-axis exponent and axis labels; molecule-aware structure-thumbnail formulas such as `(CO)_n`).
- LAMMPS cross-validation benchmarks (`benchmark/lammps_gcmc_parity.py`, `benchmark/mace_gcmc_parity.py`) and a public `benchmark/README.md`: mcpy matches LAMMPS `fix gcmc` on Lennard-Jones (atomic and rigid-dimer, all stages within 1.4 sigma) and matches `pair_style mace` energies pointwise; the one disagreement was isolated to a trial-insertion defect in the LAMMPS-MACE fork.
- Examples: `examples/gcmc_molecule_mace.py` (molecular adsorption on Ag(111) with MACE) and `examples/re_gcmc_co_cupd_batched.py` (CO on a CuPd nanoparticle: batched replica exchange over a mu ladder, trajectory-seeded restarts, coverage isotherm, snapshots, and the library phase diagram).
- Teaching notebooks: GCMC basics, molecular GCMC, and the CO/CuPd replica-exchange workflow (executed outputs included).
- Consolidated console logging for `BatchedReplicaExchange`: one status line per write interval covering all replicas (per-replica detail stays in the per-replica outfiles); disable with `consolidate_logging=False`.
- `min_atoms` / `max_atoms` limits on the atomic GCMC moves (per-species population floors and caps).
- `torch.compile` support for locally stored Alchemi checkpoints (~2x faster forwards) with an on/off A/B benchmark, and batched FIRE now always retires converged graphs from the batch (compaction: ~1.9x on mixed-convergence batches).

### Fixed
- `MoleculeDisplacementMove` rejects displacements that would carry a molecule's center of mass out of the region cell: the boundary was a one-way door that stranded molecules outside the grand-canonical bookkeeping (detailed-balance violation caught in review with an end-to-end reproduction).
- Atomic `InsertionMove` next to molecular species: ASE's `extend` zero-pads missing arrays, silently attaching the inserted atom to molecule id 0; inserted atoms are now explicitly tagged free.
- `BatchedReplicaExchange`'s swap criterion now counts molecular species in the grand potential (it delegated to a symbol count that is always zero for molecular names).
- The MPI `ReplicaExchange` raises `NotImplementedError` for molecular species instead of silently accepting every mu-ladder swap; `BatchedReplicaExchange` is the supported path (and units-less ensembles still pass the guard).
- The extxyz trajectory writer declares `molecule_id` in `Properties=` so `ase.io.read` recovers it (it was silently dropped).
- Audit of the core package (sampling correctness):
  - Rejected deletion moves corrupted `FixAtoms` constraints (ASE remaps indices in place; the rollback restored arrays but not constraints), silently freezing the wrong atoms for the rest of the run.
  - Deleting the last atom of a species was treated as a failed proposal instead of a real trial move (`Atoms` truthiness vs an explicit `False` sentinel) across both GCMC loops, `MoveSelector`, and `CanonicalEnsemble`.
  - `AlchemiBrownianMove` mutated a copy, so the ensembles scored the unchanged original and accepted no-ops.
  - LJ units did not define `beta`; LJ GCMC crashed on the first uphill move.
  - `set_state` did not recalculate cell free volumes, so acceptance after a replica-exchange swap used the previous configuration's volume.
  - `min_insert` distance checks ignored periodic images, and `CustomCell.calculate_volume` missed exclusion spheres straddling a box face.
  - `overlap_volume` returned twice the analytic sphere-sphere lens volume.
  - `MoveSelector` crashed on float weights (`n_moves` now explicit); `NullCell.calculate_volume` raised `TypeError` inside an ensemble.
  - GCMC now raises if a move returns a different `Atoms` object instead of mutating in place (copy-based moves are Canonical-only).
- Example scripts: the gas-cell `DeletionMove` in `examples/gcmc.py` targeted the metal species, so inserted gas atoms could never be removed (a silent detailed-balance violation); the '110'/'211' surface types built fcc100 slabs; bare `mace_mp` example usage returned unrelaxed energies.
- Documentation audit: install instructions, stale claims, wrong API signatures, and version drift.

### Changed
- The phase-diagram pressure twin-axis exponent is configurable per reservoir stoichiometry (`atoms_per_reservoir_molecule`); the historical dissociative-diatomic behaviour remains the default.

## [1.2.0] - 2026-06-22

### Added
- Chunked batched evaluation: `chunk_size` on `AlchemiCalculator` (energy-only batched eval) and on `AlchemiFCalculator` (batched FIRE relaxation), plus a `chunk_ranges` helper. Peak GPU memory is decoupled from the replica count, enabling larger batched replica-exchange runs.
- `energy_only` flag on `AlchemiCalculator` to skip force autograd when only energies are needed.
- Flexible Alchemi model loading: local `.model` paths, head selection, and cuEq support.
- `plot_phase_diagram`: `adsorbate_count_fn` for custom adsorbate counting (e.g. an adsorbate symbol shared with an inert sublattice) and `gamma_in_ev` to report the unnormalized formation energy in eV.
- Energy-only GCMC equivalence test (ASE/MACE vs Alchemi).
- GPU-memory benchmark harness and findings for batched replica exchange.

### Changed
- Alchemi calculator module split into energy, FIRE, and shared components.
- GCMC acceptance convention: the de Broglie count reverts to the total atom count, and the convention is now documented.

### Docs
- JOSS paper (`paper/`), validated against LAMMPS for two Lennard-Jones reference systems.
- Calculators and API Reference documentation, plus a units section.
- Domain glossary and ADRs for GCMC conventions.

## [1.1.0] - 2026-06-03

### Added
- `DomeCell`: hemispherical insertion region for supported nanoparticles, with a dome-region GCMC example.
- `AlchemiBrownianMove`: GPU-native Langevin Brownian move (NVIDIA Alchemi backend).
- `CanonicalEnsemble` now plugs into `ReplicaExchange` for NVT replicas via `get_state`/`set_state`; `ReplicaExchange` teardown is chemical-potential-optional.
- Compound perturbation moves: `n_swaps` / `n_steps` trial moves per step.
- Minima trajectory output for basin-hopping-style sampling.
- `plot_phase_diagram` utility for building phase diagrams from multiple trajectories.
- Per-step wall-time logging in `AlchemiFCalculator`.

### Changed
- `CanonicalEnsemble` is now routed through the mcpy `MoveSelector` (legacy multi-mutation loop removed); NVT move statistics are logged via the public `move_selector` attribute. Existing `CanonicalEnsemble` usage remains compatible.

### Fixed
- Replica-exchange swap acceptance now compares the grand potential for grand-canonical replicas — the correct GCMC parallel-tempering criterion.
- `BatchedReplicaExchange` now performs `n_moves` trial moves per step.
- `AlchemiFCalculator` honors `FixAtoms` constraints during relaxation.

### Docs
- JOSS paper draft (now on the `paper` branch), cluster Alchemi tutorial, phase-diagram and `DomeCell` documentation, plus supported-NP and batched RE-GCMC examples.

## [1.0.0] - 2026-05-26

Initial public release.

[1.4.0]: https://github.com/farrisric/mcpy/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/farrisric/mcpy/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/farrisric/mcpy/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/farrisric/mcpy/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/farrisric/mcpy/releases/tag/v1.0.0
