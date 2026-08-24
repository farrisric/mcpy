Utilities
=========

``mcpy.utils`` holds the thermodynamic constants, the seeded RNG, logging setup,
reference chemical potentials, and phase-diagram post-processing.

``phase_diagram`` is imported lazily, so the Monte Carlo core never pulls in
matplotlib.


.. autoclass:: mcpy.utils.set_unit_constant.SetUnits
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

.. autofunction:: mcpy.utils.phase_diagram.plot_phase_diagram

.. autofunction:: mcpy.utils.phase_diagram.analyze_phase_diagram_results
