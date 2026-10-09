Ensembles
=========

``mcpy.ensembles`` holds the Monte Carlo loop drivers. Import them from the
package, for example ``from mcpy.ensembles import GrandCanonicalEnsemble``.
Conceptual background is in :doc:`../ensembles`.

All concrete ensembles share the file and output parameters of
:class:`~mcpy.ensembles.BaseEnsemble`. Each one runs through
``run(steps)``.

``GrandCanonicalEnsemble`` takes ``units_type`` (``'metal'`` or ``'LJ'``), which
selects the constants its de Broglie factors are built from; see
:doc:`../ensembles`.


.. autoclass:: mcpy.ensembles.BaseEnsemble
   :members:

.. autoclass:: mcpy.ensembles.GrandCanonicalEnsemble
   :members:
   :show-inheritance:

.. autoclass:: mcpy.ensembles.CanonicalEnsemble
   :members:
   :show-inheritance:

.. autoclass:: mcpy.ensembles.ReplicaExchange
   :members:

.. autoclass:: mcpy.ensembles.BatchedReplicaExchange
   :members:

.. autofunction:: mcpy.ensembles.base_ensemble.write_xyz
