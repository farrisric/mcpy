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

- **GCMC and Replica-Exchange GCMC** in a single, modular run loop.
- **Hybrid scheme of Senftle et al.** — every trial insertion/deletion is
  followed by a short local relaxation, which makes acceptance realistic in
  densely packed metallic systems.
- **Calibratable free volume** via Monte Carlo sampling with element-wise
  exclusion radii.
- **Cell geometries out of the box** — periodic box, rectangular sub-slab,
  spherical region around a nanoparticle, and user-defined custom cells.
- **Modular trial moves** — insertion, deletion, displacement, permutation,
  shake, and Brownian moves, mixed through a weighted ``MoveSelector``.
- **MLIP-ready** — MACE, NequIP, ACE, and an optional GPU-native
  NVIDIA Alchemi backend for large systems.
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
   first_simulation

.. toctree::
   :caption: Background
   :maxdepth: 1

   ensembles
   cells
   species_radii
   moves

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/oxidation_phase_diagram

.. toctree::
   :caption: Examples
   :maxdepth: 1

   examples

.. toctree::
   :caption: Reference
   :maxdepth: 1

   glossary
   bibliography
