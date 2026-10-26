---
name: verify-gcmc
description: Verify a change to mcpy before calling it done - moves, cells, ensembles, acceptance, calculators. Use after any code change to mcpy/ and before opening a PR.
---

# Verify a change to mcpy

Unit tests alone do not prove the physics. A GCMC bug often shows only as wrong coverage or a drifting energy in a real run.

## Steps

1. Lint: `flake8 mcpy/` gives no output.
2. Tests: `pytest tests/` passes.
   Fix every failure or flaky test you see, also ones your change did not cause.
3. For a bug fix: reproduce the bug first in a real run (an example script or a small script that uses the public API).
   The fix is done when that same run goes from wrong to right.
4. Run the example closest to the change:
   - CPU, quick: `python examples/00_hello_gcmc.py`, `01_canonical_mc.py`, `02_molecule_gcmc.py`.
   - GPU (see the `run-gpu` skill): `03_gcmc_surface_mace.py`, `04_replica_exchange_batched.py`, `05_pt_nanoparticle_h_che.py`.
5. Read the output, not only the exit code:
   - `outfile.out`: N_atoms settles to a plateau; energy has no jumps or NaN; each move has an acceptance ratio that is not stuck at 0 % or 100 %.
   - `trajectory.xyz`: open a few frames (for example `ase gui trajectory.xyz`); atoms stay inside the cell, fixed atoms stay fixed, no atoms overlap.
6. When the change can move results (acceptance rule, free volume, move proposal, calculator energy), compare a short run before and after on the same seeds.
   Seed the cells, the `MoveSelector` and the ensemble (`random_seed=`); one unseeded part makes the comparison meaningless.

Done when: lint is clean, all tests pass, and an example run shows physically sane output for the code you changed.

## Rules to check against

- Acceptance counts: `docs/gcmc_acceptance_convention.rst`, `docs/adr/0001-gcmc-de-broglie-total-atom-count.md`.
- Trial energies are relaxed: `docs/adr/0002-trial-energy-is-lbfgs-relaxed.md`.
- Moves change `atoms` in place; on rejection the ensemble restores `atoms.arrays` and the constraints.
