Installation
============

Requirements
------------

- Python >= 3.9
- NumPy, SciPy, ASE
- (Optional) An MLIP backend such as MACE, NequIP, or ACE
- (Optional) `mpi4py` for replica-exchange runs

From PyPI
---------

.. code-block:: bash

   pip install mcpy

From source
-----------

.. code-block:: bash

   git clone https://github.com/farrisric/mcpy.git
   cd mcpy
   pip install -e .[dev]

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import mcpy, ase; print('mcpy', mcpy.__version__)"

MLIP backends
-------------

Install only the backends you intend to use. For example, MACE:

.. code-block:: bash

   pip install mace-torch

GPU support follows each backend's own installation guide.

MPI for Replica Exchange
------------------------

`mpi4py` is not pulled in automatically, since it depends on a system MPI implementation.
Install it with conda before running RE-GCMC:

.. code-block:: bash

   conda install mpi4py

Then launch a replica-exchange simulation with one MPI rank per replica:

.. code-block:: bash

   mpirun -n <N> python examples/re_gcmc.py
