Installation
============

Requirements
------------

- Python >= 3.11
- ASE >= 3.23 (pulls in NumPy, SciPy, matplotlib)
- (Optional) An MLIP backend such as MACE (``.[mace]``) or NVIDIA Alchemi (``.[alchemi]``)
- (Optional) `mpi4py` for MPI replica-exchange runs

From source
-----------

``mcpy`` is not published on PyPI (the ``mcpy`` name there is an unrelated
macro library) — install from the repository:

.. code-block:: bash

   git clone https://github.com/farrisric/mcpy.git
   cd mcpy
   pip install -e .

Add the ``test`` extra to run the test suite (``pip install -e .[test]``).

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import mcpy, ase; print('mcpy', mcpy.__version__)"

MLIP backends
-------------

Install only the backends you intend to use. For MACE
(``MACECalculator`` / ``MACE_F_Calculator`` and the bundled examples):

.. code-block:: bash

   pip install -e .[mace]

GPU support follows each backend's own installation guide.

NVIDIA Alchemi backend (optional)
---------------------------------

For GPU-native MACE evaluation, ``mcpy`` ships an optional ``AlchemiCalculator``
and ``AlchemiFCalculator`` backed by `nvalchemi-toolkit
<https://github.com/NVIDIA/nvalchemi-toolkit>`_. Recommended for systems
with **≥500 atoms** on CUDA — the repository benchmark measured a **3.08x
speedup on a 586-atom GCMC run** (20 steps, MACE+LBFGS vs Alchemi+FIRE,
RTX 5090; see the README).

Install via the ``alchemi`` extra:

.. code-block:: bash

   pip install -e .[alchemi]

This pulls in ``nvalchemi-toolkit[mace]``. Requires a CUDA-enabled PyTorch
build matching your driver.

Usage (drop-in replacement for ``MACE_F_Calculator``):

.. code-block:: python

   from mcpy.calculators import AlchemiFCalculator

   calc = AlchemiFCalculator(
       checkpoint='medium-mpa-0',
       steps=500,
       fmax=0.05,
       device='cuda',
       enable_cueq=True,
       compile_model=True,    # one-time warmup, then faster even at varying N
       dt=1.0,                # tuned default; not 0.1
   )

See ``NVALCHEMI_NOTES.md`` in the repository root for tuning details and
known pitfalls.

MPI for Replica Exchange
------------------------

`mpi4py` is not pulled in automatically, since it depends on a system MPI implementation.
Install it with conda before running RE-GCMC:

.. code-block:: bash

   conda install mpi4py

Then launch a replica-exchange simulation with one MPI rank per replica:

.. code-block:: bash

   mpirun -n <N> python examples/re_gcmc.py
