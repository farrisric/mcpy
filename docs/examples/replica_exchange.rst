Replica exchange on one GPU or over MPI
=======================================

Replica exchange runs one GCMC replica per rung of a temperature ladder or a
chemical-potential ladder, and swaps neighbouring configurations from time to
time.
mcpy has two drivers for it (:doc:`../ensembles`).
:class:`~mcpy.ensembles.BatchedReplicaExchange` keeps every replica in one
process and evaluates all their trial energies in one batched GPU call.
:class:`~mcpy.ensembles.ReplicaExchange` runs one replica per MPI rank.

The ladder spacing decides whether swaps happen at all.
Read :doc:`../replica_exchange_ladder_spacing` before you choose a ladder.


One GPU: ``examples/04_replica_exchange_batched.py``
----------------------------------------------------

The script runs six replicas of a small Ag(111) slab, from 300 K to 800 K,
with O and Ag exchange in every replica.
It shares the calibration constants of
``examples/03_gcmc_surface_mace.py`` (see :doc:`gcmc_custom_cell`).
``gcmc_factory`` builds one replica per call, each with its own cells, moves
and seeds, and one shared ``AlchemiFCalculator`` makes the batched forward
pass possible.

.. literalinclude:: ../../examples/04_replica_exchange_batched.py
   :language: python

Each replica writes its own ``.out`` and ``.xyz`` files, tagged with its
rank, and ``replica_exchange_batched.log`` records the swap attempts and
acceptances.


Several processes: ``ReplicaExchange`` with MPI
-----------------------------------------------

The MPI driver needs ``mpi4py`` (``conda install mpi4py``) and exactly one
MPI rank per replica.
It cannot exchange molecular species; use the batched driver for those.
The factory has the same contract as above: it takes ``T=`` (or ``mu=``)
and ``rank=``, and it must give each rank its own output file names.
The sketch below reuses the LJ box of ``examples/00_hello_gcmc.py`` on a
four-rung temperature ladder.

.. code-block:: python

   # re_lj.py
   from ase import Atoms
   from ase.calculators.lj import LennardJones

   from mcpy.cell import Cell
   from mcpy.ensembles import GrandCanonicalEnsemble, ReplicaExchange
   from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector
   from mcpy.utils.logging import configure as configure_logging

   TEMPERATURES = [1.6, 1.8, 2.0, 2.2]   # one rank per entry


   def gcmc_factory(T, rank):
       atoms = Atoms(cell=[9.0] * 3, pbc=True)
       cell = Cell(atoms, seed=100 + rank)
       moves = MoveSelector(
           [1, 1, 1],
           [InsertionMove(cell, species=['H'], seed=200 + rank),
            DeletionMove(cell, species=['H'], seed=300 + rank),
            DisplacementMove(species=['H'], max_displacement=0.3, seed=400 + rank)],
           seed=500 + rank)
       return GrandCanonicalEnsemble(
           atoms=atoms, cells=[cell],
           calculator=LennardJones(sigma=1.0, epsilon=1.0, rc=3.0, smooth=False),
           mu={'H': -3.0}, species=['H'], units_type='LJ', temperature=T,
           move_selector=moves, random_seed=600 + rank,
           outfile=f're_rank{rank}.out', traj_file=f're_rank{rank}.xyz',
           outfile_write_interval=100, trajectory_write_interval=100)


   configure_logging()   # without a handler, a rank's error is not shown
   ReplicaExchange(gcmc_factory, temperatures=TEMPERATURES, gcmc_steps=2000,
                   exchange_interval=50, seed=7).run()

Launch it with one rank per temperature:

.. code-block:: bash

   mpirun -n 4 python re_lj.py

Rank 0 writes ``replica_exchange.log`` and, at the end, ``global_minimum.xyz``.
An error on any rank aborts the whole job instead of leaving the other ranks
waiting.
