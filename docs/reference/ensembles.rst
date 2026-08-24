Ensembles
=========

``mcpy.ensembles`` holds the Monte Carlo loop drivers. Import a concrete class
from its module, for example
``from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble``.
Conceptual background is in :doc:`../ensembles`.

All concrete ensembles share the file and output parameters of
:class:`~mcpy.ensembles.base_ensemble.BaseEnsemble`. Each one runs through
``run(steps)``.

``GrandCanonicalEnsemble`` takes ``units_type`` (``'metal'`` or ``'LJ'``), which
selects the constants its de Broglie factors are built from; see
:doc:`../ensembles`.


.. autoclass:: mcpy.ensembles.base_ensemble.BaseEnsemble
   :members:

.. autoclass:: mcpy.ensembles.grand_canonical_ensemble.GrandCanonicalEnsemble
   :members:
   :show-inheritance:

.. autoclass:: mcpy.ensembles.canonical_ensemble.CanonicalEnsemble
   :members:
   :show-inheritance:

.. autoclass:: mcpy.ensembles.replica_exchange.ReplicaExchange
   :members:

.. autoclass:: mcpy.ensembles.batched_replica_exchange.BatchedReplicaExchange
   :members:

.. autofunction:: mcpy.ensembles.base_ensemble.write_xyz
