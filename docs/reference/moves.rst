Moves
=====

``mcpy.moves`` holds the trial-move classes and the weighted sampler that mixes
them. Conceptual background is in :doc:`../moves`.

Every move returns a tuple ``(atoms, delta_particles, species)`` from
``do_trial_move(atoms)``, where ``atoms`` is the mutated structure (or a falsy
value when the move cannot be proposed), ``delta_particles`` is ``+1``, ``-1``,
or ``0``, and ``species`` is the affected symbol. Moves mutate ``atoms`` in
place; the ensemble rolls back on rejection.


.. autoclass:: mcpy.moves.MoveSelector
   :members:

.. autoclass:: mcpy.moves.base_move.BaseMove
   :members:

Particle exchange
-----------------

.. autoclass:: mcpy.moves.InsertionMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.DeletionMove
   :members:
   :show-inheritance:

Molecular exchange
------------------

.. autoclass:: mcpy.moves.MoleculeInsertionMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.MoleculeDeletionMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.MoleculeDisplacementMove
   :members:
   :show-inheritance:

.. automodule:: mcpy.moves.molecule_utils
   :members:

Displacement and reordering
---------------------------

.. autoclass:: mcpy.moves.DisplacementMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.PermutationMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.ShakeMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.BrownianMove
   :members:
   :show-inheritance:

.. autoclass:: mcpy.moves.AlchemiBrownianMove
   :members:
   :show-inheritance:
