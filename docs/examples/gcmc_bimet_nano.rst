GCMC of H on a Pt/Au bimetallic nanoparticle with metal permutations
====================================================================

Hydrogen GCMC on a Pt/Au truncated-octahedral nanoparticle, combined with
Pt↔Au :class:`PermutationMove` on the metal framework. Demonstrates how to
mix variable-particle-number moves with chemical-ordering moves in a
single ensemble.

Mirrors ``examples/gcmc_bimet_nano.py``.

Goal
----

At fixed :math:`T = 500\,\mathrm{K}` and :math:`\mu_{\mathrm{H}}` derived
from :math:`\tfrac{1}{2}E(\mathrm{H}_2)`, sample equilibrium H coverage on
the cluster while letting the Pt/Au shell composition equilibrate.

Prerequisites
-------------

- An ASE calculator that can evaluate H\ :sub:`2` (MACE in the script,
  but any ASE calculator works).

Code
----

.. code-block:: python

   import numpy as np
   from ase.build import molecule
   from ase.cluster import Octahedron
   from mace.calculators import mace_mp

   from mcpy.calculators import BaseCalculator
   from mcpy.cell import SphericalCell
   from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
   from mcpy.moves import DeletionMove, InsertionMove, PermutationMove
   from mcpy.moves.move_selector import MoveSelector

   T = 500.0
   delta_mu_H = -0.5

   ss = np.random.SeedSequence(0)
   seed_del, seed_ins, seed_perm = (
       int(s) for s in ss.generate_state(3, dtype=np.uint32)
   )

   atoms = Octahedron('Pt', 7, 2)
   half = len(atoms) // 2
   atoms.symbols = ['Pt'] * half + ['Au'] * half + ['Pt']
   atoms.set_pbc(False)

   scell = SphericalCell(
       atoms,
       vacuum=3,
       species_radii={'Pt': 2, 'Au': 2.5, 'H': 0},
       mc_sample_points=100_000,
   )

   # BaseCalculator relaxes with LBFGS before each energy evaluation;
   # a bare mace_mp would return unrelaxed energies.
   calculator = BaseCalculator(mace_mp(device='cuda'), steps=100, fmax=0.05)

   move_selector = MoveSelector(
       [1, 1, 1],
       [DeletionMove(scell, species=['H'], seed=seed_del),
        InsertionMove(scell, species=['H'], min_insert=0.5, seed=seed_ins),
        PermutationMove(species=['Au', 'Pt'], seed=seed_perm)],
   )

   # Reference mu_H from molecular H2.
   e_h2 = calculator.get_potential_energy(molecule('H2'))
   mus = {'H': e_h2 / 2 + delta_mu_H}

   tag = f'{atoms.get_chemical_formula()}_dmu_{delta_mu_H}'
   gcmc = GrandCanonicalEnsemble(
       atoms=atoms,
       cells=[scell],
       calculator=calculator,
       mu=mus,
       units_type='metal',
       species=['H'],
       temperature=T,
       move_selector=move_selector,
       outfile=f'gcmc_{tag}.out',
       trajectory_write_interval=1,
       outfile_write_interval=1,
       traj_file=f'gcmc_{tag}.xyz',
   )
   gcmc.run(1_000_000)

Outputs
-------

- ``gcmc_<formula>_dmu_<x>.out`` — step log with per-move acceptance for
  H insertion, H deletion, and Pt↔Au permutation.
- ``gcmc_<formula>_dmu_<x>.xyz`` — extended XYZ trajectory.

Interpretation
--------------

- ``species=['H']`` declares the *variable-number* species. Pt and Au
  counts are conserved by :class:`PermutationMove`, which only swaps
  identities of existing atoms.
- The ``species_radii`` entry for ``'H'`` is 0 so that hydrogen never
  blocks itself or the metal scaffold during the free-volume estimate.
- Reference energies use a stricter relaxation (``steps=100``,
  ``fmax=0.05``) than the production loop; restore the production
  tolerance before calling ``gcmc.run`` if you reuse the same calculator
  object.

Next steps
----------

- Drop the permutation move to recover a pure GCMC run: see
  :doc:`gcmc_nano`.
- Replace H with O and the metal symbols with Ag to study oxidation on
  a single-metal cluster.
