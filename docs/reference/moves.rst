Moves
=====

``mcpy.moves`` holds the trial-move classes and the weighted sampler that mixes
them. Conceptual background is in :doc:`../moves`.

Every move returns a tuple ``(atoms, delta_particles, species)`` from
``do_trial_move(atoms)``, where ``atoms`` is the mutated structure (or a falsy
value when the move cannot be proposed), ``delta_particles`` is ``+1``, ``-1``,
or ``0``, and ``species`` is the affected symbol. Moves mutate ``atoms`` in
place; the ensemble rolls back on rejection.


MoveSelector
------------

.. code-block:: python

   MoveSelector(probabilities, move_list, seed=None)

Samples one move per step from ``move_list`` with weights ``probabilities``
(weights need not sum to one). Tracks per-interval and cumulative acceptance
counters, excluding non-viable proposals from the denominator.

Parameters:

- ``probabilities`` (list): relative weight of each move. Their sum sets
  ``n_moves``, the number of trial moves per GCMC step.
- ``move_list`` (list): the move instances to sample from.
- ``seed`` (int, optional): RNG seed.

Methods: ``do_trial_move(atoms)``, ``acceptance_counter()``, ``get_volume()``,
``interval_ratios()``, ``total_ratios()``, ``reset_counters()``.


BaseMove
--------

.. code-block:: python

   BaseMove(cell, species, seed)

Abstract base. Holds the attached ``cell``, the ``species`` list, and a seeded
RNG. Provides ``get_volume()`` and ``calculate_volume(atoms)`` over the cell.
Subclasses implement ``do_trial_move(atoms)``.


InsertionMove
-------------

.. code-block:: python

   InsertionMove(cell, species, seed, min_insert=None)

Inserts one atom of a random selected species at a random point in ``cell``.
Sets ``delta_particles = +1``.

- ``min_insert`` (float, optional): minimum distance to existing cell atoms. The
  move retries up to an internal cap and reports a failed move if it cannot
  place the atom without a closer contact.


DeletionMove
------------

.. code-block:: python

   DeletionMove(cell, species, seed)

Deletes a random atom of the selected species lying inside ``cell``. Sets
``delta_particles = -1``. Returns a falsy result when no candidate atom exists,
recorded as a failed move rather than a rejection.


DisplacementMove
----------------

.. code-block:: python

   DisplacementMove(species, seed, constraints=None, max_displacement=0.1,
                    n_steps=1)

Displaces ``n_steps`` distinct atoms by random vectors of magnitude up to
``max_displacement``. Particle count is unchanged. Uses a ``NullCell``
internally (no insertion region).

- ``constraints`` (list, optional): indices held fixed.
- ``max_displacement`` (float): maximum per-atom step.
- ``n_steps`` (int): atoms moved per trial. Exceeding the movable-atom count
  raises ``ValueError``.


PermutationMove
---------------

.. code-block:: python

   PermutationMove(species, seed, n_swaps=1)

Swaps the chemical identities of ``n_swaps`` atom pairs drawn from different
species groups, in a single trial. Particle count is unchanged. Returns a falsy
result if a requested species is absent.


ShakeMove
---------

.. code-block:: python

   ShakeMove(r_max, seed)

Displaces every atom by an independent vector drawn uniformly inside a ball of
radius ``r_max``. A global perturbation, usually paired with a relaxing
calculator.


BrownianMove
------------

.. code-block:: python

   BrownianMove(temperature, calculator, steps, d_t, seed)

Runs ``steps`` of Velocity-Verlet MD at ``temperature`` (timestep ``d_t`` in fs)
as the trial move, from a Maxwell-Boltzmann velocity draw.


AlchemiBrownianMove
-------------------

.. code-block:: python

   AlchemiBrownianMove(calculator, temperature, friction=0.01, steps=100,
                       dt=2.0, seed=0)

GPU-native Langevin Brownian move. Runs a short NVT Langevin trajectory through
``calculator.run_md`` (an ``AlchemiCalculator`` or ``AlchemiFCalculator``),
reusing its model. Honors ``FixAtoms``.


Experimental moves
------------------

``mcpy.moves.go_moves`` contains specialised exploratory moves (``BallMove``,
``ShellMove``, ``BondMove``, ``HighEnergyAtomsMove``, and variant
permutation/shake/Brownian moves). They are not part of the ``mcpy.moves``
export list and their interfaces may change.
