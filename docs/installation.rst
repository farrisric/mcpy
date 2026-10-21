Installation
============

Requirements
------------

- Python >= 3.11
- ASE >= 3.23, NumPy and SciPy (installed with mcpy)
- Optional: matplotlib for the phase-diagram utilities (``.[plot]``)
- Optional: an MLIP backend, MACE (``.[mace]``) or NVIDIA Alchemi (``.[alchemi]``)
- Optional: ``mpi4py`` for MPI replica exchange (``.[mpi]`` or conda)

From source
-----------

``mcpy`` is not published on PyPI (the ``mcpy`` name there belongs to an
unrelated macro library), so install it from the repository:

.. code-block:: bash

   git clone https://github.com/farrisric/mcpy.git
   cd mcpy
   pip install -e .

The distribution is named ``mcpy-gcmc``, because the ``mcpy`` name on PyPI belongs to an unrelated package.
The import name stays ``mcpy``.

This is enough for the CPU examples ``examples/00`` to ``examples/02``.
Add the ``test`` extra to run the test suite (``pip install -e .[test]``).

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import mcpy; print('mcpy', mcpy.__version__)"
   python examples/00_hello_gcmc.py   # about three minutes on a CPU

MACE backend
------------

``MACE_F_Calculator`` wraps a MACE model in a relaxation and needs
``mace-torch``:

.. code-block:: bash

   pip install -e .[mace]

GPU support follows the MACE installation guide.

NVIDIA Alchemi backend (optional)
---------------------------------

``AlchemiCalculator`` and ``AlchemiFCalculator`` evaluate MACE on the GPU
through `nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_,
and ``AlchemiFCalculator`` also relaxes each trial with a GPU-native
optimizer: FIRE by default, or FIRE2 or LBFGS with ``optimizer=``
(LBFGS needs nvalchemi newer than 0.2.0).
They are the backend of ``examples/03``, ``examples/04`` and ``examples/05``.
Install them with the ``alchemi`` extra:

.. code-block:: bash

   pip install -e .[alchemi]

This pulls in ``nvalchemi-toolkit[mace]``.
It needs a CUDA-enabled PyTorch build that matches your driver.

.. code-block:: python

   from mcpy.calculators import AlchemiFCalculator

   calc = AlchemiFCalculator(
       checkpoint='medium-mpa-0',
       steps=300,
       fmax=0.05,
       device='cuda',
       enable_cueq=True,
       compile_model=True,    # one-time warmup, then faster even at varying N
   )

:doc:`reference/calculators` documents every parameter, the batched
evaluation and the GPU-memory controls.

GCMC changes the atom count on every accepted insertion or deletion, so the
PyTorch CUDA allocator caches blocks of many sizes and its reserved memory
grows during a long run.
Start long runs with expandable segments to keep it near the real peak:

.. code-block:: bash

   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/04_replica_exchange_batched.py

MPI for replica exchange
------------------------

``ReplicaExchange`` runs one replica per MPI rank and needs ``mpi4py``.
Install it with conda, which links a matching MPI library:

.. code-block:: bash

   conda install mpi4py

Then launch the script with exactly one rank per rung of the ladder, for
example four ranks for four temperatures:

.. code-block:: bash

   mpirun -n 4 python -m mpi4py re_lj.py

``python -m mpi4py`` makes an uncaught exception on one rank abort all ranks.
:doc:`examples/replica_exchange` gives a complete ``re_lj.py``.
On a single GPU, ``BatchedReplicaExchange`` is the better choice: it needs no
MPI and runs as a plain Python process
(``python examples/04_replica_exchange_batched.py``).
