# CLAUDE.md

`mcpy` is a Grand Canonical Monte Carlo (GCMC) library for atomistic systems, built on ASE.

## Commands

```bash
# Install (editable); extras: mace, alchemi, mpi, plot, test
pip install -e .
pip install -e '.[mace]'

# Lint (config in .flake8: max line length 100, single quotes)
flake8 mcpy/

# Tests (no mace-torch, torch or mpi4py needed)
pytest tests/
pytest tests/test_moves.py   # one file

# CPU example (ASE potentials, no GPU)
python examples/00_hello_gcmc.py
```

Unit tests do not prove the physics.
Verify a simulation change by running the nearest example and reading `outfile.out` and `trajectory.xyz`; the `verify-gcmc` skill has the full procedure.
GPU runs (examples 03-05, MACE and Alchemi calculators) need the `run-gpu` skill first.

## Contracts the code does not make obvious

- GCMC moves mutate the passed `atoms` in place.
  The ensemble snapshots `atoms.arrays` plus the constraints before each trial and restores both on rejection.
  A move that cannot propose returns `False` in place of the atoms.
  Copy-returning moves are for `CanonicalEnsemble` only.
- Cells have two predicates: `is_point_inside` (where insertions are proposed) and `is_point_exchangeable` (what may be deleted or counted).
  They differ on purpose for `CustomCell`, whose exchangeable region has no z upper bound so desorbed species stay deletable.
- Trial energies are relaxed energies, not single points (`docs/adr/0002-trial-energy-is-lbfgs-relaxed.md`).
- GCMC scores minima by the grand potential Omega = E - sum(mu_i N_i), not bare energy.
- `mcpy.utils.phase_diagram` is imported lazily so the MC core never pulls in matplotlib.
  Library modules only call `logging.getLogger`, so importing mcpy never touches the host application's logging config; `mcpy.utils.logging.configure(...)` is the opt-in.

## Conventions

- Acceptance particle counts: atomic moves pass the pre-move total atom count, molecule moves pass their pre-move in-cell molecule count.
  The two coexist in one run.
  See `docs/gcmc_acceptance_convention.rst` and `docs/adr/0001-gcmc-de-broglie-total-atom-count.md`.
- Replica-exchange ladders: uniform mu spacing can kill the ladder.
  See `docs/replica_exchange_ladder_spacing.rst`.
- Reproducibility: pass seeds explicitly to the cells, each move, the `MoveSelector`, and the ensemble (`random_seed=`).
  One `None` makes a run unrepeatable.
- Reference chemical potentials come from the running calculator (`mcpy.utils.reference_mu`), never from a hardcoded number.

## MPI

`mpi4py` is the optional `mpi` extra; a conda-installed `mpi4py` is often better, because it links a matching MPI (`conda install mpi4py`).
Launch with `mpirun -n N python -m mpi4py my_run.py`.
Every rank must be seeded independently (`ValueError` otherwise).
MPI `ReplicaExchange` supports no molecular species; `BatchedReplicaExchange` needs no MPI and is the better choice on a single GPU.

## Issues and pull requests

- Issues are tracked in the `farrisric/mcpy` GitHub repo via the `gh` CLI; external PRs are not a triage surface.
  See `docs/agents/issue-tracker.md`.
- Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
  See `docs/agents/triage-labels.md`.
- Domain docs: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
- Every change reaches `main` through a PR that the other maintainer reviews.
  To review a PR, check out its branch and run the `verify-gcmc` skill on it.
