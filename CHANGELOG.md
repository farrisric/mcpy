# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
