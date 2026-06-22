GCMC of O on a supported Ag nanoparticle (dome region)
======================================================

Grand-canonical sampling of oxygen on an Ag truncated-octahedral
nanoparticle placed on an Al\ :sub:`2`\ O\ :sub:`3` support, using a
:class:`DomeCell` to confine insertion/deletion to a hemispherical region
centred on the particle. Compared with :doc:`gcmc_nano_supported` (which
uses a slab-spanning :class:`CustomCell`), the dome keeps trial insertions
on and around the particle and its metal-support contact rim instead of
spreading them across the bare support.

Adapted from ``examples/gcmc_nano_supported.py`` by swapping
:class:`CustomCell` for :class:`DomeCell`.

Goal
----

At fixed :math:`T = 500\,\mathrm{K}` and chosen
:math:`\Delta\mu_{\mathrm{O}}`, sample equilibrium O coverage on the
supported cluster, with insertions localised on the nanoparticle.

Prerequisites
-------------

- A POSCAR of the support (``Al2O3.poscar`` in the script).
- A MACE checkpoint or other ASE calculator that can describe Ag, Al,
  and O.

Code
----

.. code-block:: python

   import numpy as np
   from ase.cluster import Octahedron
   from ase.constraints import FixAtoms
   from ase.io import read
   from mace.calculators import mace_mp

   from mcpy.cell import DomeCell
   from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
   from mcpy.moves import DeletionMove, InsertionMove
   from mcpy.moves.move_selector import MoveSelector
   from mcpy.utils.utils import get_p_at_support

   T = 500.0
   delta_mu_O = -0.5
   mus = {'Ag': -2.99, 'O': -4.91 + delta_mu_O}

   ss = np.random.SeedSequence(0)
   seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

   support = read('Al2O3.poscar').repeat((4, 4, 1))
   support.center(vacuum=10.0, axis=2)
   z = support.positions[:, 2]
   z_half = z.min() + 0.5 * (z.max() - z.min())
   support.set_constraint(FixAtoms(mask=(z < z_half)))

   surface_z = float(np.max(support.positions[:, 2]))
   particle = Octahedron('Ag', 5, 2)
   atoms = get_p_at_support(support, particle, contact_surface='100', gap=2.0)

   scell = DomeCell(
       atoms,
       particle_species='Ag',
       bottom_z=surface_z,
       vacuum=3.0,
       species_radii={'Ag': 2.068, 'O': 0, 'Al': 3},
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

- ``gcmc_<formula>_dmu_<x>.out`` -- step log with per-move acceptance.
- ``gcmc_<formula>_dmu_<x>.xyz`` -- extended XYZ trajectory.

Interpretation
--------------

- ``particle_species='Ag'`` tells the dome which atoms are the
  nanoparticle: the region is centred on the Ag centroid and its radius
  is the farthest Ag atom plus ``vacuum``. The support is never used to
  place or size the dome.
- ``bottom_z=surface_z`` clips the dome at the topmost support atom, so
  sampled points stay above the surface; the part of the ball that would
  dip into the support is discarded.
- Only O is exchanged. Al and Ag enter ``species_radii`` solely so the
  free-volume estimator excludes overlaps with them.
- Because insertions are confined to the dome, O accumulates on the
  particle far more efficiently than with a slab-spanning cell, where
  many trial insertions land on the bare support.

Next steps
----------

- Compare against the slab-cell version: :doc:`gcmc_nano_supported`.
- Visualise the loaded particle with
  ``examples/plot_supported_np.py``.
- Sweep :math:`\Delta\mu_{\mathrm{O}}` and post-process with
  :doc:`phase_diagram_analysis`.
