mcpy
====

Grand Canonical Monte Carlo for atomistic systems — with native Replica
Exchange and machine-learning interatomic potentials.

`mcpy` predicts the composition and stability of surfaces and nanoparticles
under realistic temperature and chemical-potential conditions. It is built on
the Atomic Simulation Environment (ASE), so any ASE-compatible calculator —
DFT, classical potentials, or MLIPs such as MACE — can drive the sampling.

Highlights
----------

- **GCMC, NVT, and Replica-Exchange** in a single, modular run loop,
  including a single-process ``BatchedReplicaExchange`` for single-GPU runs.
- **Hybrid scheme of Senftle et al.** — every trial insertion/deletion is
  followed by a short local relaxation, which makes acceptance realistic in
  densely packed metallic systems.
- **Basin-hopping output for free** — every ensemble can emit a running
  ``minima_file`` of strictly improving configurations alongside the
  sampling trajectory, and replica-exchange runs add a global
  ``global_minimum.xyz`` at the end.
- **Calibratable free volume** via Monte Carlo sampling with element-wise
  exclusion radii.
- **Cell geometries out of the box** — periodic box, rectangular sub-slab,
  spherical region around a nanoparticle, and user-defined custom cells.
- **Modular trial moves** — insertion, deletion, displacement, permutation,
  shake, and Brownian moves, mixed through a weighted ``MoveSelector``;
  permutation and displacement support compound ``n_swaps`` / ``n_steps``
  perturbations per trial for basin-hopping sampling.
- **MLIP-ready** — dedicated MACE wrappers, plus any ASE-compatible potential
  (NequIP, ACE, classical force fields) through the generic calculator adapter,
  and an optional GPU-native NVIDIA Alchemi backend for large systems.
- **Phase-diagram utilities** for post-processing GCMC ensembles into
  surface and nanoparticle phase diagrams.


Citing mcpy
-----------

If you use `mcpy` in a publication, please cite the project repository and
the hybrid GCMC method it implements (see :doc:`bibliography`).


.. toctree::
   :caption: Get started
   :maxdepth: 1

   installation
   getting_started

.. toctree::
   :caption: Diving deeper
   :maxdepth: 1

   ensembles
   cells
   species_radii
   moves
   molecular_adsorbates
   calculators
   phase_diagrams
   gcmc_acceptance_convention

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/cluster_install_iqtc

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples
   notebooks

.. toctree::
   :caption: API reference
   :maxdepth: 1

   reference/ensembles
   reference/moves
   reference/cells
   reference/calculators
   reference/utils

.. toctree::
   :caption: Reference
   :maxdepth: 1

   glossary
   bibliography
