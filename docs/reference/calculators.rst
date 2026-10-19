Calculators
===========

``mcpy.calculators`` wraps energy backends for the ensembles.
``BaseCalculator``, ``MACE_F_Calculator`` and ``AlchemiFCalculator`` relax the trial structure in place before they return the energy; ``steps=0`` gives a pure single point.
``AlchemiCalculator`` always returns a single-point energy.
Each relaxing wrapper records ``last_relax_steps`` and ``total_relax_steps``, and logs a warning when the step cap stops a relaxation before it converges.

Each wrapper exposes ``get_potential_energy(atoms) -> float``.
The Alchemi classes add ``get_potential_energies(atoms_list, chunk_size=None) -> ndarray`` for batched evaluation, where ``chunk_size`` caps peak GPU memory at one chunk.
An empty structure (0 atoms) gets the energy 0.0 and is not sent to the model.
The Alchemi relaxation and MD honour ``FixAtoms`` only; any other ASE constraint raises ``NotImplementedError``.

``BaseCalculator`` imports without torch.
``MACE_F_Calculator`` needs ``mace-torch`` and the Alchemi classes need ``nvalchemi-toolkit``.
They import lazily: ``import mcpy.calculators`` does not import torch, and the backend loads the first time you access one of these names.
If the backend is not installed, that access raises the ``ImportError``.

``AlchemiCalculator`` is energy-only by default when it loads its own model.
Pass ``energy_only=False`` to keep forces, for example for ``run_md``.
With ``enable_cueq=True``, mcpy repairs a checkpoint that MACE's cuEq conversion would otherwise corrupt (mh-1, ACEsuit/mace#1298).
It sets the missing ``use_edge_irreps_first`` flag before the conversion, so cuEq gives the same energies as e3nn.
A checkpoint that it cannot repair loads with the e3nn kernels, and a warning is logged.
With cuEq on, ``torch.compile`` gives wrong forces for models with ``RealAgnosticResidualNonLinearInteractionBlock`` layers (mh-1).
The energies stay right, but the forces are off by up to about 1 eV/Å, because of a miscompile in the torch inductor backend.
For these models mcpy keeps cuEq on, turns ``compile_model`` off and logs a warning.
``examples/05_pt_nanoparticle_h_che.py`` sets ``COMPILE_MODEL = False`` for this reason.
A checkpoint string that looks like a path (it contains a path separator or ends in ``.model`` or ``.pt``) but does not exist raises ``FileNotFoundError``.

The Alchemi classes sum the per-atom energies of each structure in float64.
The model's own per-structure sum runs in the model dtype, and in float32 it rounds a total of about 1000 eV to the meV.
``AlchemiFCalculator(steps=0)`` is a pure single point on every supported nvalchemi version: it does not build the optimizer and does not move the atoms.


.. autoclass:: mcpy.calculators.BaseCalculator
   :members:

.. autoclass:: mcpy.calculators.MACE_F_Calculator
   :members:

.. autoclass:: mcpy.calculators.alchemi._AlchemiBase
   :members:

.. autoclass:: mcpy.calculators.AlchemiCalculator
   :members:
   :show-inheritance:

.. autoclass:: mcpy.calculators.AlchemiFCalculator
   :members:
   :show-inheritance:
