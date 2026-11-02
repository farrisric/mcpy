---
name: molecule-gcmc
description: Use when the GCMC adsorbate is a molecule (CO, O2, H2O) - setting up molecule insertion/deletion/displacement moves, molecule_id bookkeeping, molecular chemical potential, or mixing atomic and molecular species in one run.
---

# GCMC with molecular adsorbates

Full guide: `docs/molecular_adsorbates.rst`. Working script: `examples/02_molecule_gcmc.py`.
This card holds only the parts people get wrong.

## Steps

1. Moves: `MoleculeInsertionMove` / `MoleculeDeletionMove` / `MoleculeDisplacementMove`,
   each built from an ASE template molecule plus a name. The template is rigid at proposal
   time; a relaxing calculator still relaxes it after placement.
2. `molecule_id`: every atom carries one; molecule members share a value, free atoms AND
   all host atoms carry `-1`. Forgetting `-1` on the host makes the whole slab one fake
   molecule (`ValueError` when the group matches no template).
3. mu is molecular: compute `E(molecule)` in a box with the production calculator and
   reference against it. No rotational factor in the acceptance rule; orientations are
   drawn uniformly.
4. Acceptance N: molecule moves count in-cell molecules of that species; atomic moves keep
   the total-atom-count convention. The two coexist in one run - do not "fix" either.
   Background: `docs/gcmc_acceptance_convention.rst`.
5. `MoleculeDisplacementMove(max_displacement=..., max_angle=...)`: cap `max_angle` for
   anchored adsorbates. Rigid CO displacement converged ~2x faster than atomic shakes even
   at ~5 % acceptance; tune the caps toward 20-40 %.
6. Replica exchange with molecules: `BatchedReplicaExchange` only; MPI `ReplicaExchange`
   counts atoms by symbol and supports no molecular species.

Done when: `trajectory.xyz` carries a `molecule_id` column and molecule counts in
`outfile.out` respond to mu.

## Traps

- LAMMPS fix gcmc comparison: its molecule rotation sampling is non-uniform; composition
  gaps vs LAMMPS dimer runs are LAMMPS's issue, not mcpy's.
- Deleting one atom of a molecule with an atomic `DeletionMove`: never mix atomic exchange
  moves with molecular species of the same elements in one cell.
