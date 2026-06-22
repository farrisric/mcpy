Ensembles
=========

``mcpy.ensembles`` holds the Monte Carlo loop drivers. Import a concrete class
from its module, for example
``from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble``.
Conceptual background is in :doc:`../ensembles`.

All concrete ensembles share the file and output parameters listed under
:ref:`BaseEnsemble <ref-base-ensemble>`. Each one runs through ``run(steps)``.


.. _ref-base-ensemble:

BaseEnsemble
------------

.. code-block:: python

   BaseEnsemble(atoms, cells, units_type, calculator, user_tag=None,
                random_seed=None, traj_file='trajectory.xyz', traj_mode='w',
                trajectory_write_interval=1, outfile='outfile.out',
                outfile_mode='w', outfile_write_interval=1,
                minima_file=None, minima_mode='a')

Abstract base for every ensemble. Owns the atoms, cells, calculator, file
handles, and step counter. Not instantiated directly.

Shared parameters (inherited by the concrete ensembles):

- ``atoms`` (ase.Atoms): starting configuration.
- ``cells`` (list): cell objects providing free volume and insertion points.
- ``units_type`` (str): ``'metal'`` or ``'LJ'`` (see :doc:`../ensembles`).
- ``calculator``: an mcpy calculator wrapper or any object exposing
  ``get_potential_energy(atoms)``.
- ``random_seed`` (int, optional): seed source for derived generators.
- ``traj_file`` / ``traj_mode`` / ``trajectory_write_interval``: sampling
  trajectory path, file mode, and write stride. ``traj_file=None`` disables it.
- ``outfile`` / ``outfile_mode`` / ``outfile_write_interval``: log path, mode,
  and write stride. ``outfile=None`` disables it.
- ``minima_file`` / ``minima_mode``: basin-hopping output, off by default.
  ``'a'`` appends every new minimum; ``'w'`` keeps only the running best.

Methods:

- ``run(steps)``: open files, run ``steps`` ensemble steps, finalize.
- ``initialize_run()`` / ``finalize_run()``: open and close handles. Idempotent.
  Also available through the context-manager protocol.
- ``atoms`` / ``cells`` / ``step``: properties.

Abstract: ``_run()`` performs one ensemble step and is implemented by
subclasses.


CanonicalEnsemble
-----------------

.. code-block:: python

   CanonicalEnsemble(atoms, calculator, cells=None, units_type='metal',
                     random_seed=None, optimizer=None, fmax=0.1,
                     temperature=300, move_selector=None, constraints=None,
                     traj_file='trajectory.xyz', traj_mode='w',
                     outfile='outfile.out', outfile_mode='w',
                     outfile_write_interval=10, trajectory_write_interval=1,
                     minima_file=None, minima_mode='a')

Fixed-composition (NVT) Metropolis sampler. Each step applies one move from
``move_selector``, relaxes it with ``optimizer`` to ``fmax``, and accepts by the
Metropolis rule at ``temperature``.

Parameters beyond the shared set:

- ``optimizer``: an ASE optimizer class (e.g. ``ase.optimize.LBFGS``) used to
  relax each trial.
- ``fmax`` (float): force tolerance for the relaxation.
- ``temperature`` (float): canonical temperature in K.
- ``move_selector`` (MoveSelector): the trial-move sampler.
- ``constraints``: ASE constraint applied to each mutated structure.

Methods: ``run(steps)``, plus ``get_state()`` / ``set_state(state)`` for use as a
temperature replica inside ``ReplicaExchange``.


GrandCanonicalEnsemble
----------------------

.. code-block:: python

   GrandCanonicalEnsemble(atoms, cells, units_type, calculator, mu, species,
                          temperature, move_selector, random_seed=None,
                          traj_file='trajectory.xyz', traj_mode='w',
                          trajectory_write_interval=1, outfile='outfile.out',
                          outfile_mode='w', outfile_write_interval=1,
                          minima_file=None, minima_mode='a')

Variable-composition (μVT) sampler. Insertion and deletion moves exchange
``species`` with a reservoir at chemical potentials ``mu``; acceptance uses the
de Broglie criterion (see :doc:`../ensembles`).

Parameters beyond the shared set:

- ``mu`` (dict): species to reservoir chemical potential in eV.
- ``species`` (list[str]): symbols the move set may insert or delete.
- ``temperature`` (float): temperature in K.
- ``move_selector`` (MoveSelector): the trial-move sampler.

Methods: ``run(steps)``, plus ``get_state()`` / ``set_state(state)`` used by the
replica-exchange drivers. The running-minimum score is the grand potential
``E - Σ μ_i N_i``.


ReplicaExchange
---------------

.. code-block:: python

   ReplicaExchange(gcmc_factory, temperatures=None, mus=None, gcmc_steps=100,
                   exchange_interval=10, outfile='replica_exchange.log',
                   write_out_interval=20, seed=31,
                   global_minimum_file='global_minimum.xyz')

MPI parallel-tempering wrapper, one replica per rank. Requires ``mpi4py``;
constructing it without ``mpi4py`` installed raises ``ImportError``.

Parameters:

- ``gcmc_factory`` (callable): builds the rank-local ensemble. Called as
  ``gcmc_factory(T=..., rank=i)`` for a temperature ladder or
  ``gcmc_factory(mu=..., rank=i)`` for a chemical-potential ladder. Must produce
  per-rank output paths so ranks do not race on the same files.
- ``temperatures`` (list, optional): one temperature per rank.
- ``mus`` (list, optional): one ``mu`` dict per rank. Pass exactly one of
  ``temperatures`` or ``mus``.
- ``gcmc_steps`` (int): GCMC steps per replica.
- ``exchange_interval`` (int): steps between exchange attempts.
- ``write_out_interval`` (int): steps between summary writes.
- ``seed`` (int): RNG seed for exchange decisions.
- ``global_minimum_file`` (str): path for the lowest-score frame across ranks.
  ``None`` disables it.

Methods: ``run()`` (no step argument; the step count is fixed at construction).


BatchedReplicaExchange
----------------------

.. code-block:: python

   BatchedReplicaExchange(gcmc_factory, calculator, temperatures=None, mus=None,
                          gcmc_steps=100, exchange_interval=10,
                          outfile='replica_exchange.log', write_out_interval=20,
                          seed=31, global_minimum_file='global_minimum.xyz')

Single-process replica exchange that batches every replica's trial energy into
one calculator call. The right choice on a single GPU.

Parameters beyond those of ``ReplicaExchange``:

- ``calculator``: shared calculator. Must implement
  ``get_potential_energies(atoms_list)``. Passing one without it raises
  ``TypeError``. ``AlchemiCalculator`` provides the method.

The factory must return a fresh ensemble per replica with its own cells and
move selector; sharing them across replicas corrupts per-replica state. Methods:
``run()``.
