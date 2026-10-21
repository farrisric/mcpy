Examples
========

The ``examples/`` directory of the repository holds five runnable scripts.
Run them in order.
None of them take command-line flags: every setting is a named constant at the
top of the file, so you edit the file and run it again.

.. list-table::
   :header-rows: 1
   :widths: 30 46 24

   * - Script
     - What it shows
     - Needs, runtime
   * - `00_hello_gcmc.py <https://github.com/farrisric/mcpy/blob/main/examples/00_hello_gcmc.py>`_
     - The GCMC loop on a Lennard-Jones fluid, checked against LAMMPS
       ``fix gcmc``.
     - ASE only, about 3 min on a CPU
   * - `01_canonical_mc.py <https://github.com/farrisric/mcpy/blob/main/examples/01_canonical_mc.py>`_
     - Canonical Monte Carlo and basin hopping: chemical ordering in a CuAu
       nanoparticle with EMT.
     - ASE only, about 10 s on a CPU
   * - `02_molecule_gcmc.py <https://github.com/farrisric/mcpy/blob/main/examples/02_molecule_gcmc.py>`_
     - Molecular GCMC: rigid Lennard-Jones dimers, checked against LAMMPS
       ``fix gcmc mol``.
     - ASE only, about 2 min on a CPU
   * - `03_gcmc_surface_mace.py <https://github.com/farrisric/mcpy/blob/main/examples/03_gcmc_surface_mace.py>`_
     - GCMC: O and Ag exchanged on Ag(111) with the small AGNESI MACE
       potential, as in the Ag oxidation study.
     - ``.[mace]``, CPU or GPU, hours
   * - `04_replica_exchange_batched.py <https://github.com/farrisric/mcpy/blob/main/examples/04_replica_exchange_batched.py>`_
     - Replica exchange on one GPU with batched energy evaluation.
     - ``.[alchemi]`` and a CUDA GPU, hours

The first three use simple potentials on purpose: they run anywhere, and their
results can be checked against known values.
The last two use a machine-learned potential for real chemistry.
The pages below show each script in full and explain what to look for in its
output.
The acceptance rules that every script relies on are derived in
:doc:`ensembles`.

.. toctree::
   :maxdepth: 1

   examples/gcmc_simulations
   examples/mc_simulations
   examples/gcmc_custom_cell
   examples/replica_exchange
   examples/phase_diagram_analysis
