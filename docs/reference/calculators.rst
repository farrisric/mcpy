Calculators
===========

``mcpy.calculators`` wraps energy backends for the ensembles. Conceptual
background, including the relaxation-inside-energy design, is in
:doc:`../calculators`.

Each wrapper exposes ``get_potential_energy(atoms) -> float``. The Alchemi
classes add ``get_potential_energies(atoms_list, chunk_size=None) -> ndarray``
for batched evaluation, where ``chunk_size`` caps peak GPU memory at one chunk.
``BaseCalculator`` imports unconditionally; ``MACE_F_Calculator`` needs
``mace-torch`` and the Alchemi classes need ``nvalchemi-toolkit``, and each is
exported only when its backend is installed.


.. autoclass:: mcpy.calculators.BaseCalculator
   :members:

.. autoclass:: mcpy.calculators.mace_f_calculator.MACE_F_Calculator
   :members:

.. autoclass:: mcpy.calculators.alchemi_calculator.AlchemiCalculator
   :members:

.. autoclass:: mcpy.calculators.alchemi_f_calculator.AlchemiFCalculator
   :members:
