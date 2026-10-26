# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e .

# Install with uv
uv pip install -e .

# Lint
flake8 mcpy/

# Run tests
pytest tests/

# Install an optional backend (extras: mace, alchemi, test)
pip install -e '.[mace]'
pip install -e '.[alchemi]'

# Run a simulation script (CPU-only starters, ASE potentials, no GPU)
python examples/00_hello_gcmc.py

# Production GCMC on a surface (MACE/nvalchemi, GPU)
python examples/03_gcmc_surface_mace.py

# Single-GPU replica exchange (no MPI; batched energy evaluation)
python examples/04_replica_exchange_batched.py
```

Tests live in `tests/` and run with `pytest tests/` (CI runs them on Python 3.11–3.14). They cover the cells, moves, logging, acceptance criteria, molecule moves, constraint rollback, and both replica-exchange swap rules, and do not require `mace-torch`, `torch`, or `mpi4py`. End-to-end verification of simulations is still done by running example scripts and inspecting `outfile.out` and `trajectory.xyz`.

GPU runs: see the `run-gpu` skill (`.claude/skills/run-gpu/SKILL.md`).

## Architecture

`mcpy` is a Grand Canonical Monte Carlo (GCMC) simulation library for atomistic systems, built on top of ASE (Atomic Simulation Environment).

### Core abstractions

**Ensembles** (`mcpy/ensembles/`) — orchestrate the MC loop:
- `BaseEnsemble` — abstract base; owns atoms, calculator, file I/O, step counter
- `GrandCanonicalEnsemble` — GCMC: inserts/deletes particles, evaluates acceptance via the de Broglie criterion
- `CanonicalEnsemble` — NVT: constant particle number
- `ReplicaExchange` - parallel-tempering wrapper around GCMC; requires `mpi4py`, one MPI rank per replica. The factory gets T **or** μ, never both, but the swap uses the same general cross-term rule as `BatchedReplicaExchange` (exact for a per-rung μ(T), de Broglie term included). No molecular species (it counts atoms by symbol)
- `BatchedReplicaExchange` — single-process, single-GPU replica exchange: all replicas live in one process and every replica's trial energy is evaluated in one batched forward pass. Needs a calculator with `get_potential_energies(atoms_list)`. Same general swap rule as `ReplicaExchange`, plus molecular species; warns on a degenerate or dead ladder

Every ensemble can emit a `minima_file` of strictly improving configurations; replica-exchange runs also write a final `global_minimum.xyz` (one frame per rung whenever the rungs' μ differ). GCMC scores minima by the grand potential Ω = E − Σ μᵢNᵢ, not bare energy.

**Moves** (`mcpy/moves/`) — propose trial configurations:
- `BaseMove` — abstract; each move holds a reference to a `Cell` and a `RandomNumberGenerator`
- `InsertionMove`, `DeletionMove` — add/remove atoms in the cell's free volume
- `MoleculeInsertionMove`, `MoleculeDeletionMove` — exchange whole rigid
  molecules (ASE template + name; per-atom `molecule_id` bookkeeping;
  textbook per-species acceptance N — see `docs/gcmc_acceptance_convention.rst`)
- `MoleculeDisplacementMove` — rigid translate+rotate of one molecule
  (`max_displacement`, optional `max_angle` cap for anchored adsorbates)
- `molecule_utils` — shared helpers: `find_molecules`, `molecule_com`,
  `random_rotation_matrix`
- `DisplacementMove`, `ShakeMove`, `BrownianMove` — displace existing atoms
  (`DisplacementMove` takes `n_steps` to bundle several displacements into one trial)
- `AlchemiBrownianMove` — short GPU Langevin MD as the proposal, reusing the
  Alchemi calculator's already-loaded model
- `PermutationMove` — swap species identities (`n_swaps` for compound trials)
- `MoveSelector` — weighted sampler over a list of moves; tracks per-interval and
  cumulative acceptance counters, and excludes failed proposals from the denominator

GCMC moves mutate the passed `atoms` **in place**; the ensemble snapshots
`atoms.arrays` plus the constraints before each trial and restores them on
rejection. A move that cannot propose returns `False` in place of the atoms.
Copy-returning moves are for `CanonicalEnsemble` only.

**Cells** (`mcpy/cell/`) — define the spatial region for MC moves and compute free volume:
- `BaseCell` — abstract interface: `calculate_volume`, `get_random_point`, `get_volume`
- `SphericalCell` — sphere around a nanoparticle; free volume estimated by MC sampling (excludes overlaps with atomic radii)
- `Cell` — periodic box cell
- `DomeCell` — hemisphere on a supported particle: centered on a chosen particle
  species, clipped at the support surface, does not move the atoms
- `CustomCell` — rectangular sub-slab of the box; its exchangeable region drops
  the z upper bound so desorbed species stay deletable
- `NullCell` — placeholder for moves that sample no spatial region

Cells expose two predicates: `is_point_inside` (where insertions are proposed)
and `is_point_exchangeable` (what may be deleted or counted). They differ on
purpose for `CustomCell`.

**Calculators** (`mcpy/calculators/`) — thin wrappers around ASE calculators:
- `BaseCalculator` — calls `LBFGS` geometry relaxation before returning energy
  (see `docs/adr/0002-trial-energy-is-lbfgs-relaxed.md`)
- `MACECalculator`, `MACE_F_Calculator` — wrappers for MACE-torch potentials;
  `MACE_F_Calculator` relaxes (`optimizer='lbfgs'|'fire'`) and counts relax steps
- `AlchemiCalculator`, `AlchemiFCalculator` — optional GPU-native nvalchemi
  backend. Energy-only and FIRE-relaxing respectively; both honor `FixAtoms`,
  expose batched `get_potential_energies(atoms_list)` with `chunk_size`, and
  offer `run_md` for Langevin proposals. Shared internals in `_AlchemiBase` (`mcpy/calculators/alchemi.py`)

**Utils** (`mcpy/utils/`):
- `SetUnits` — computes thermodynamic constants (β, de Broglie wavelengths) for `"metal"` or `"LJ"` unit systems
- `RandomNumberGenerator` — seeded uniform/normal sampler
- `reference_mu` — `derive_mu_bulk` / `derive_mu_gas`: reference chemical potentials
  computed from the running potential instead of hardcoded numbers
- `chunking` — `chunk_ranges`, torch-free index partitioning for batched evaluation
- `logging` — `configure(...)`; library modules only ever call `getLogger`, so
  importing mcpy never touches the host application's logging config
- `phase_diagram` — post-processing into surface and nanoparticle phase diagrams.
  `plot_phase_diagram` is the current entry point; `analyze_phase_diagram_results`
  is the older oxidation-specific one. Imported lazily so the MC core never pulls
  in matplotlib
- `scripts/compute_radii.py` — helper for species exclusion radii (a script, not a module)

### Data flow

```
GrandCanonicalEnsemble.run(steps)
  └─ _run() each step
       └─ do_gcmc_step()
            └─ MoveSelector.do_trial_move(atoms) → atoms_new, delta_particles, species
            └─ compute_energy(atoms_new)           ← calls ASE calculator
            └─ _acceptance_condition(...)           ← uses SetUnits.de_broglie_{insertion,deletion}
            └─ if accepted: update atoms, recalculate cell volumes
       └─ write_outfile() / write_coordinates()
```

### Output files

- `outfile.out` — step-by-step log: step, N_atoms, energy, per-move acceptance
  ratios (per write interval; cumulative totals go to the `finalize_run` log)
- `trajectory.xyz` — extended XYZ trajectory with `energy=` and `Lattice=` in the
  comment line, plus `molecule_id` when molecule moves are active
- `minima_file` / `global_minimum.xyz` — optional basin-hopping output (off by default)

### Linting

`flake8` config is in `.flake8`: max line length 100, `E203` ignored, single quotes enforced.

### mpi4py note

`mpi4py` is the optional `mpi` extra (`pip install -e '.[mpi]'`). A conda-installed mpi4py is often better, because it links a matching MPI:
```bash
conda install mpi4py
```
Launch with `mpirun -n N python -m mpi4py my_run.py`. Every rank must be seeded independently (`ValueError` otherwise).
`BatchedReplicaExchange` needs no MPI and is the better choice on a single GPU.

### Conventions worth knowing before editing

- Acceptance particle counts: atomic moves pass the pre-move total atom count,
  molecule moves pass their pre-move in-cell molecule count. See
  `docs/gcmc_acceptance_convention.rst` and
  `docs/adr/0001-gcmc-de-broglie-total-atom-count.md`.
- Replica-exchange ladder spacing has its own guide:
  `docs/replica_exchange_ladder_spacing.rst`. Uniform μ spacing can kill the ladder.
- Reproducibility: seeds must be passed explicitly to the cells, each
  move, the `MoveSelector`, and the ensemble (`random_seed=`). Any one of them left at
  `None` makes a run unrepeatable.
- When you add, rename or remove a module, update the Architecture section of
  this file in the same PR.

## Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Agent skills

### Issue tracker

Issues are tracked in the `farrisric/mcpy` GitHub repo via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Pull requests

Every change reaches `main` through a PR that the other maintainer reviews.
To review a PR, check out its branch and run the `verify-gcmc` skill on it.
