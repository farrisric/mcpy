Installation
============


Requirements
------------
- Python >= 3.9
- NumPy, SciPy, ASE
- (Optional) One or more MLIP backends (e.g., MACE, NequIP, ACE)


From PyPI (recommended)
-----------------------
.. code-block:: bash


pip install mcpy


From source
-----------
.. code-block:: bash


git clone https://github.com/your-org/mcpy.git
cd mcpy
pip install -e .[dev]


Verify installation
-------------------
.. code-block:: bash


python -c "import mcpy, ase; print('mcpy', mcpy.__version__)"


Optional backends
-----------------
Install MLIP backends you intend to use. For example, for MACE:


.. code-block:: bash


pip install mace-torch


GPU support depends on each backend’s installation instructions.
