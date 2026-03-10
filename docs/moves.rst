Moves and Operations
====================

In `mcpy`, a *move* (operation) proposes a new atomic configuration.
The ensemble then accepts or rejects that proposal with the appropriate acceptance rule.


Core move interface
-------------------

All moves follow the `BaseMove` interface and implement:

- `do_trial_move(atoms)`: propose a new structure,
- optional volume hooks through the attached cell (`get_volume`, `calculate_volume`).

Moves are typically combined using `MoveSelector`.


`MoveSelector`
--------------

`MoveSelector` chooses moves stochastically from a weighted list and tracks acceptance counters.
This is the standard way to mix insertion/deletion/displacement (or other) operations in a run.

.. code-block:: python

   from mcpy.moves import MoveSelector, InsertionMove, DeletionMove

   selector = MoveSelector(
       [1, 1],  # weights
       [InsertionMove(cell, ["O"], seed=1, min_insert=0.5),
        DeletionMove(cell, ["O"], seed=2)],
   )


Main move classes
-----------------

The following move classes are available from `mcpy.moves`:

`InsertionMove`
~~~~~~~~~~~~~~~

Attempts insertion of a species at a random point sampled from the configured cell.
Returns `delta_particles = +1` on success.


`DeletionMove`
~~~~~~~~~~~~~~

Attempts deletion of a randomly chosen atom of selected species within the active cell.
Returns `delta_particles = -1` on success.


`DisplacementMove`
~~~~~~~~~~~~~~~~~~

Displaces one randomly selected atom by a random vector up to `max_displacement`.
Particle count does not change (`delta_particles = 0`).


`PermutationMove`
~~~~~~~~~~~~~~~~~

Swaps species labels of two atoms from different species groups.
Useful for compositional ordering in multicomponent systems.


`ShakeMove`
~~~~~~~~~~~

Randomly displaces all atoms by small random vectors (bounded by `r_max`).
Useful as a global perturbation move.


`BrownianMove`
~~~~~~~~~~~~~~

Runs a short MD segment (Velocity Verlet) at a target temperature as a trial move.
Useful for larger relaxation-like perturbations.


Advanced/experimental moves
---------------------------

`mcpy.moves.go_moves` includes additional specialized operations such as:

- `BallMove`,
- `ShellMove`,
- `BondMove`,
- `HighEnergyAtomsMove`,
- alternative `PermutationMove` / `ShakeMove` / `BrownianMove` variants.

These are useful for custom workflows but are not part of the default `mcpy.moves` export list.
