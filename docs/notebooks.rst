Notebooks
=========

The repository ships Jupyter notebooks under `notebooks/
<https://github.com/farrisric/mcpy/tree/main/notebooks>`_.
Four of them are saved with every cell executed, so they read as worked case
studies without running anything.
The fifth needs a GPU and ships without output.

Executed, CPU only (they run again in a few minutes):

- `gcmc_basics_lj.ipynb
  <https://github.com/farrisric/mcpy/blob/main/notebooks/gcmc_basics_lj.ipynb>`_:
  GCMC fundamentals on a Lennard-Jones system: the ideal gas against its
  analytic result, then an interacting fluid against LAMMPS.
- `molecular_gcmc.ipynb
  <https://github.com/farrisric/mcpy/blob/main/notebooks/molecular_gcmc.ipynb>`_:
  whole-molecule insertion, deletion, and bookkeeping step by step
  (see :doc:`molecular_adsorbates`).

Executed on a GPU with MACE:

- `phase_diagram.ipynb
  <https://github.com/farrisric/mcpy/blob/main/notebooks/phase_diagram.ipynb>`_:
  building a phase diagram from swept trajectories
  (see :doc:`phase_diagrams`).
- `co_on_cupd_replica_exchange.ipynb
  <https://github.com/farrisric/mcpy/blob/main/notebooks/co_on_cupd_replica_exchange.ipynb>`_:
  CO on a CuPd nanoparticle with batched replica exchange, ending in the
  molecular phase diagram.

Not executed (needs ``mace-torch`` and a CUDA GPU):

- `grand_canonical_simulation_ag.ipynb
  <https://github.com/farrisric/mcpy/blob/main/notebooks/grand_canonical_simulation_ag.ipynb>`_:
  oxygen on a 140-atom Ag nanoparticle with MACE-MPA-0, one setup step per
  cell.
  :doc:`getting_started` covers the same workflow on a slab.
