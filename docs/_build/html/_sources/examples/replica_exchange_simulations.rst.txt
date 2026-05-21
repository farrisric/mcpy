Replica-Exchange GCMC across a temperature ladder
=================================================

Parallel tempering over six replicas of an Ag(111) + O system, exchanging
configurations between neighbouring temperatures every ``exchange_interval``
GCMC steps. One MPI rank corresponds to one replica.

Mirrors ``examples/re_gcmc.py``.

Goal
----

Run RE-GCMC at :math:`T \in \{250, 300, 350, 400, 450, 500\}\,\mathrm{K}`
with shared :math:`\mu_{\mathrm{Ag}}, \mu_{\mathrm{O}}` to escape local
basins that a single GCMC trajectory would not leave.

Prerequisites
-------------

- ``mpi4py`` and a working system MPI (see :doc:`../installation`).
- A MACE checkpoint, or any ASE calculator that you wrap in
  ``BaseCalculator``.

Code
----

.. code-block:: python

   import numpy as np
   from ase.build import fcc111
   from ase.constraints import FixAtoms
   from mace.calculators import mace_mp

   from mcpy.cell import CustomCell
   from mcpy.ensembles import ReplicaExchange
   from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
   from mcpy.moves import InsertionMove, DeletionMove
   from mcpy.moves.move_selector import MoveSelector

   temperatures = [250, 300, 350, 400, 450, 500]
   gcmc_steps = 200
   exchange_interval = 10
   delta_mu_O = -0.5

   ss = np.random.SeedSequence(0)
   seeds = [int(s) for s in ss.generate_state(5, dtype=np.uint32)]

   atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8.0)
   atoms.set_constraint(FixAtoms(indices=[a.index for a in atoms if a.tag == 3]))

   cell_ag = CustomCell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                        species_radii={'Ag': 2.75, 'O': 0.0})
   cell_o  = CustomCell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                        species_radii={'Ag': 2.11, 'O': 0.0})

   calculator = mace_mp(device='cuda')
   calculator.steps = 40
   calculator.fmax = 0.1

   mus = {'Ag': -2.99, 'O': -4.91 + delta_mu_O}

   move_selector = MoveSelector(
       [25, 25, 25, 25],
       [DeletionMove(cell_ag, species=['Ag'], seed=seeds[0]),
        DeletionMove(cell_o,  species=['O'],  seed=seeds[1]),
        InsertionMove(cell_ag, species=['Ag'], min_insert=0.5, seed=seeds[2]),
        InsertionMove(cell_o,  species=['O'],  min_insert=0.5, seed=seeds[3])],
   )

   def gcmc_factory(T, rank=0):
       tag = f'rank{rank}_T{int(T)}'
       return GrandCanonicalEnsemble(
           atoms=atoms,
           cells=[cell_ag, cell_o],
           calculator=calculator,
           mu=mus,
           units_type='metal',
           species=['Ag', 'O'],
           temperature=T,
           move_selector=move_selector,
           outfile=f'gcmc_{tag}.out',
           traj_file=f'gcmc_{tag}.xyz',
           trajectory_write_interval=1,
           outfile_write_interval=1,
       )

   rex = ReplicaExchange(
       gcmc_factory,
       temperatures=temperatures,
       gcmc_steps=gcmc_steps,
       exchange_interval=exchange_interval,
       seed=seeds[4],
   )
   rex.run()

Run it
------

.. code-block:: bash

   mpirun -n 6 python re_gcmc.py

The number of MPI ranks must equal ``len(temperatures)``.

Outputs
-------

- One ``gcmc_rank*_T*.out`` log and matching ``.xyz`` trajectory per rank.
- ``replica_exchange.log`` — the cross-replica exchange log.

Interpretation
--------------

- ``gcmc_factory`` must produce *per-rank* output filenames, otherwise all
  ranks race on the same files.
- A healthy exchange acceptance is typically 20–40 %. If it is too low,
  reduce the temperature gap between neighbouring replicas; if too high,
  widen it.
- For a chemical-potential ladder, pass ``mus=[mu_0, mu_1, ...]`` instead
  of ``temperatures``; the corresponding acceptance criterion is selected
  automatically.

Next steps
----------

- See ``ReplicaExchange`` in :doc:`../ensembles` for the parallel-tempering
  acceptance rule.
- Combine the per-rank trajectories into a phase diagram with
  :doc:`phase_diagram_analysis`.
