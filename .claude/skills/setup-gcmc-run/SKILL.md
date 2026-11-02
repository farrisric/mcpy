---
name: setup-gcmc-run
description: Use when writing a new GCMC or canonical MC run script - choosing a cell, moves and weights, calculator, seeds, and chemical potential for a surface, nanoparticle, or box system.
---

# Set up a GCMC run

Copy the nearest example and edit it; do not write from scratch.
`examples/00_hello_gcmc.py` (LJ box, CPU), `01_canonical_mc.py`, `02_molecule_gcmc.py`,
`03_gcmc_surface_mace.py` (surface, GPU), `04_replica_exchange_batched.py`,
`05_pt_nanoparticle_h_che.py`. Examples take no CLI flags: constants at the top of the file.

## Steps

1. Pick the cell by geometry:
   - periodic box: `Cell`
   - slab / surface window: `CustomCell` (its exchangeable region has no z upper bound, so
     desorbed species stay deletable - keep that)
   - free nanoparticle: `SphericalCell`
   - supported nanoparticle: `DomeCell`
2. Set `species_radii` on the cell with the `species-radii` skill. Not optional:
   free volume enters the acceptance rule.
3. Pick moves. Minimum GCMC set: `InsertionMove` + `DeletionMove` with equal weights,
   plus `DisplacementMove`. Molecules: use the `molecule-gcmc` skill.
   Wrap in `MoveSelector(probabilities, move_list, seed=...)`.
4. Get mu with the `choose-mu` skill. Never hardcode a number from another potential.
5. Seed EVERYTHING: each cell, each move, the `MoveSelector`, and the ensemble
   (`random_seed=`). One `None` makes the run unrepeatable.
6. Calculator: ASE/EMT for toy runs only (EMT has no oxygen chemistry);
   production uses MACE or Alchemi wrappers - launch per the `run-gpu` skill.
7. Acceptance convention is handled by the moves; do not pass your own N.
   Background: `docs/gcmc_acceptance_convention.rst`, `docs/adr/0001`, `docs/adr/0002`
   (trial energy is the relaxed energy).

Done when: the script runs a few hundred steps, `outfile.out` shows nonzero acceptance
for every move, and rerunning with the same seeds reproduces it.

## Traps

- Missing seed anywhere = silent irreproducibility (MPI ranks raise `ValueError`, single runs do not).
- `FixAtoms` on the support bottom is respected by the calculators; set it, or the slab drifts.
- Zero insertions after many steps is often physics (mu too low), not a bug - see `troubleshoot-run`.
