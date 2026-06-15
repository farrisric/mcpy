Moves
=====

A *move* in `mcpy` proposes a new atomic configuration. The ensemble then evaluates the
corresponding acceptance rule -- Metropolis for number-conserving moves, de-Broglie for
insertions and deletions -- and either commits the new configuration or rolls back.

Every move follows the `BaseMove` interface and exposes:

- `do_trial_move(atoms)` -- propose a trial structure (and, for number-changing moves,
  signal the change via `delta_particles`),
- volume hooks via the attached cell (`get_volume`, `calculate_volume`).

A run typically mixes several move types using a `MoveSelector`.


`MoveSelector`
--------------

`MoveSelector` samples one move per step from a weighted list and tracks acceptance
statistics on two levels:

- **per-interval counters** -- reset after each log write; reported in `outfile` as the
  recent acceptance ratio of each move,
- **cumulative counters** -- never reset; reported in the final summary at the end of the
  run.

Trial moves that cannot be proposed (for example, a deletion when no atom of the requested
species lies inside the cell) are tracked separately and excluded from the acceptance-ratio
denominator, so the reported ratios reflect *viable* attempts only.

.. code-block:: python

   from mcpy.moves import MoveSelector, InsertionMove, DeletionMove

   selector = MoveSelector(
       probabilities=[1, 1],
       move_list=[
           InsertionMove(cell, ["O"], seed=1, min_insert=0.5),
           DeletionMove(cell, ["O"], seed=2),
       ],
   )

Weights need not sum to one -- they are interpreted as relative frequencies.


Built-in moves
--------------

The following move classes are available from `mcpy.moves`.

`InsertionMove`
~~~~~~~~~~~~~~~

Attempts to insert an atom of one of the selected species at a random point drawn from the
attached cell. Sets `delta_particles = +1` on a successful proposal. Optional ``max_atoms``
skips the trial (no mutation, no energy evaluation) when the structure already contains at
least that many atoms of the selected species.

`DeletionMove`
~~~~~~~~~~~~~~

Attempts to delete a randomly chosen atom of one of the selected species lying inside the
attached cell. Sets `delta_particles = -1` on a successful proposal; returns falsy when no
candidate atom exists or when optional ``min_atoms`` would be violated, which the
`MoveSelector` records as a failure rather than as a rejection.

`DisplacementMove`
~~~~~~~~~~~~~~~~~~

Displaces one randomly selected atom by a random vector with magnitude up to
`max_displacement`. Particle count is preserved (`delta_particles = 0`).

The optional ``n_steps`` parameter (default ``1``) bundles ``n_steps`` *distinct* atom
displacements into a single trial -- the energy is evaluated once, after all displacements,
and the whole compound perturbation is accepted or rejected together. This is the building
block of basin-hopping with displacement moves: a one-atom step usually relaxes back into the
same basin, so a meaningful escape requires several atoms to move at once. Setting
``n_steps`` larger than the number of movable atoms raises ``ValueError``.

`PermutationMove`
~~~~~~~~~~~~~~~~~

Swaps the species labels of two atoms drawn from different species groups. Useful for
sampling compositional ordering (homotop space) in multi-component systems at fixed
stoichiometry.

The optional ``n_swaps`` parameter (default ``1``) applies ``n_swaps`` independent pair swaps
in a single trial -- one energy evaluation, one accept/reject. For basin-hopping on alloy
ordering, a single swap is typically too gentle (relaxation returns to a near-identical
basin); a perturbation of order :math:`\sim 1\text{--}5\%` of the atom count per trial is a
sensible default. In a replica-exchange run, scaling ``n_swaps`` with replica temperature --
small ``n_swaps`` for cold refiners, large ``n_swaps`` for hot explorers -- keeps acceptance
ratios balanced across the ladder.

`ShakeMove`
~~~~~~~~~~~

Displaces *all* atoms by independent random vectors bounded by `r_max`. Useful as a global
perturbation, typically combined with a local relaxation in the calculator wrapper.

`BrownianMove`
~~~~~~~~~~~~~~

Runs a short Velocity-Verlet trajectory at a target temperature as the trial move. Useful
as a relaxation-like perturbation that explores the local basin more thoroughly than a
single random displacement.


Experimental moves
------------------

`mcpy.moves.go_moves` contains additional specialised moves used in exploratory workflows:

- `BallMove`, `ShellMove`,
- `BondMove`,
- `HighEnergyAtomsMove`,
- alternative `PermutationMove`, `ShakeMove`, and `BrownianMove` variants.

These are not part of the default `mcpy.moves` export list and their interfaces may change.
