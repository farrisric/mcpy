GCMC of O on an Ag octahedral nanoparticle
==========================================

Grand-canonical sampling of oxygen on a truncated-octahedral Ag
nanoparticle in vacuum. A :class:`SphericalCell` confines insertions and
deletions to a shell around the cluster.

Mirrors ``examples/gcmc_nano.py``.

Goal
----

At fixed :math:`T = 500\,\mathrm{K}` and a chosen :math:`\mu_{\mathrm{O}}`,
sample equilibrium O coverage on the nanoparticle. Repeat across
:math:`\Delta\mu_{\mathrm{O}}` to construct a coverage–chemical-potential
curve.

Prerequisites
-------------

- A MACE checkpoint, or any ASE calculator (the script defaults to
  ``mace_mp`` on CUDA — swap for EMT or another calculator for a quick
  smoke test).
- See :doc:`../first_simulation` for the minimal API walkthrough.

Code
----

.. code-block:: python

   import numpy as np
   from ase.cluster import Octahedron
   from mace.calculators import mace_mp

   from mcpy.cell import SphericalCell
   from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
   from mcpy.moves import DeletionMove, InsertionMove
   from mcpy.moves.move_selector import MoveSelector

   T = 500.0
   delta_mu_O = -0.5
   mus = {'Ag': -2.99, 'O': -4.91 + delta_mu_O}

   ss = np.random.SeedSequence(0)
   seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

   atoms = Octahedron('Ag', 6, 1)

   scell = SphericalCell(
       atoms,
       vacuum=3,
       species_radii={'Ag': 2.947, 'O': 0},
       mc_sample_points=100_000,
   )

   calculator = mace_mp(device='cuda')

   move_selector = MoveSelector(
       [1, 1],
       [DeletionMove(scell, species=['O'], seed=seed_del),
        InsertionMove(scell, species=['O'], min_insert=0.5, seed=seed_ins)],
   )

   tag = f'{atoms.get_chemical_formula()}_dmu_{delta_mu_O}'
   gcmc = GrandCanonicalEnsemble(
       atoms=atoms,
       cells=[scell],
       calculator=calculator,
       mu=mus,
       units_type='metal',
       species=['O'],
       temperature=T,
       move_selector=move_selector,
       outfile=f'gcmc_{tag}.out',
       traj_file=f'gcmc_{tag}.xyz',
   )
   gcmc.run(1_000_000)

Outputs
-------

- ``gcmc_<formula>_dmu_<x>.out`` — step log with per-move acceptance.
- ``gcmc_<formula>_dmu_<x>.xyz`` — extended XYZ trajectory.

Interpretation
--------------

- :class:`SphericalCell` recomputes its free volume after every accepted
  move; ``species_radii`` set per-element exclusion radii that prevent
  trial insertions from overlapping existing atoms.
- Healthy insertion/deletion acceptance is typically 5–30 %. Adjust
  ``min_insert`` and the ``vacuum`` padding if acceptance collapses.
- Run a short warm-up and discard the leading frames before computing
  averages; the trajectory is written from step 0.

Next steps
----------

- Sweep :math:`\Delta\mu_{\mathrm{O}}` and post-process the trajectories
  with :doc:`phase_diagram_analysis`.
- Replace the single sphere with a supported-particle setup in
  :doc:`gcmc_nano_supported`.
- Add a metal permutation move on a bimetallic cluster in
  :doc:`gcmc_bimet_nano`.
