Utilities
=========

``mcpy.utils`` holds the thermodynamic constants, the seeded RNG, logging setup,
reference chemical potentials, and phase-diagram post-processing.

``phase_diagram`` is imported lazily, so the Monte Carlo core never pulls in
matplotlib.


.. autoclass:: mcpy.utils.SetUnits
   :members:

.. autoclass:: mcpy.utils.RandomNumberGenerator
   :members:

.. automodule:: mcpy.utils.reference_mu
   :members:

.. automodule:: mcpy.utils.chunking
   :members:

.. autofunction:: mcpy.utils.logging.configure

Phase diagrams
--------------

A worked example is in :doc:`../examples/phase_diagram_analysis`.

.. autofunction:: mcpy.utils.plot_phase_diagram

.. autofunction:: mcpy.utils.analyze_phase_diagram_results
