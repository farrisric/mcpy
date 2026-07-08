# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.2.0]: https://github.com/farrisric/mcpy/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/farrisric/mcpy/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/farrisric/mcpy/releases/tag/v1.0.0
