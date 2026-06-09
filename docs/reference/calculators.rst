Calculators
===========

``mcpy.calculators`` wraps energy backends for the ensembles. Conceptual
background, including the relaxation-inside-energy design, is in
:doc:`../calculators`.

Each wrapper exposes ``get_potential_energy(atoms) -> float``. The Alchemi
classes add ``get_potential_energies(atoms_list, chunk_size=None) -> ndarray``
for batched evaluation, where ``chunk_size`` caps peak GPU memory at one chunk.
``MACECalculator``, ``MACE_F_Calculator``, and ``BaseCalculator`` import
unconditionally; the Alchemi classes import only when ``nvalchemi-toolkit`` is
installed.


BaseCalculator
--------------

.. code-block:: python

   BaseCalculator(calculator, steps, fmax)

Adapter for any ASE calculator. ``get_potential_energy(atoms)`` relaxes
``atoms`` with LBFGS up to ``steps`` iterations or ``fmax``, then returns the
relaxed energy.

- ``calculator``: a constructed ASE calculator.
- ``steps`` (int): maximum LBFGS steps.
- ``fmax`` (float): force tolerance in eV/Å.


MACECalculator
--------------

.. code-block:: python

   MACECalculator(model_paths, device='cpu')

Single-point MACE evaluation with no relaxation.
``get_potential_energy(atoms)`` returns one forward pass.

- ``model_paths`` (str): path to a MACE model file.
- ``device`` (str): ``'cpu'`` or ``'cuda'``.


MACE_F_Calculator
-----------------

.. code-block:: python

   MACE_F_Calculator(model_paths, steps, fmax, device='cpu', cueq=False,
                     optimizer='lbfgs')

Relax-then-energy MACE wrapper used in most GCMC runs. Records
``last_relax_steps`` and ``total_relax_steps`` after each call.

- ``model_paths`` (str | MACECalculator): model file path, or a built
  ``MACECalculator`` to reuse.
- ``steps`` (int): maximum relaxation steps.
- ``fmax`` (float): force tolerance in eV/Å.
- ``device`` (str): ``'cpu'`` or ``'cuda'``.
- ``cueq`` (bool): enable cuEquivariance kernels.
- ``optimizer`` (str): ``'lbfgs'`` or ``'fire'``. Raises ``ValueError`` for any
  other value.


AlchemiCalculator
-----------------

.. code-block:: python

   AlchemiCalculator(checkpoint='medium-mpa-0', device='cuda',
                     dtype=torch.float32, enable_cueq=True, compile_model=True,
                     max_neighbors=None, chunk_size=None, energy_only=False)

GPU-native MACE evaluation with no relaxation (optional backend). Provides
``get_potential_energy(atoms)``, ``get_potential_energies(atoms_list,
chunk_size=None)``, and ``run_md(...)``.

- ``checkpoint`` (str | MACEWrapper): named checkpoint, local ``.pt`` path, or a
  shared ``MACEWrapper``.
- ``enable_cueq`` (bool): fused equivariance kernels.
- ``compile_model`` (bool): ``torch.compile`` the model. Set ``False`` for GCMC,
  where the atom count changes between trials.
- ``max_neighbors`` (int, optional): neighbor-list cap.
- ``chunk_size`` (int, optional): default sub-batch size for
  ``get_potential_energies``; caps peak GPU memory at one chunk. ``None``
  evaluates the whole batch in one pass. See :doc:`../calculators`.
- ``energy_only`` (bool): drop force computation (no autograd graph) for a
  ~12% memory saving. Energy is unchanged; forces are unavailable.


AlchemiFCalculator
------------------

.. code-block:: python

   AlchemiFCalculator(checkpoint='medium-mpa-0', steps=500, fmax=0.05,
                      device='cuda', dtype=torch.float32, enable_cueq=True,
                      compile_model=True, dt=1.0, optimizer='fire',
                      max_neighbors=None, chunk_size=None)

GPU counterpart of ``MACE_F_Calculator``: FIRE relaxation then energy, honoring
``FixAtoms`` (optional backend). Provides ``get_potential_energy(atoms)``,
``get_potential_energies(atoms_list, chunk_size=None)``, and ``run_md(...)``.

- ``steps`` (int): maximum FIRE steps.
- ``fmax`` (float): force tolerance in eV/Å.
- ``dt`` (float): FIRE initial timestep.
- ``optimizer`` (str): ``'fire'`` or ``'fire2'``. Raises ``ValueError``
  otherwise.
- ``compile_model`` (bool): set ``False`` for GCMC.
- ``chunk_size`` (int, optional): default sub-batch size for
  ``get_potential_energies``; caps peak GPU memory at one chunk. Chunked
  relaxation reaches the same minimum to within ``fmax`` but is not bit
  identical. See :doc:`../calculators`.


run_md
------

.. code-block:: python

   run_md(atoms, temperature, friction=0.01, dt=2.0, steps=100, seed=42)

Method on both Alchemi classes. Runs NVT Langevin MD in place on ``atoms``,
reusing the loaded model. ``temperature`` in K, ``friction`` in 1/fs, ``dt`` in
fs. Honors ``FixAtoms``. Backs ``AlchemiBrownianMove``.
