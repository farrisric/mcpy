mcpy
====

Overview
--------

`mcpy` is a Python library for **Grand Canonical Monte Carlo (GCMC)** simulations of atomistic
systems, with native support for **Replica-Exchange GCMC (RE-GCMC)** and modern
**machine-learning interatomic potentials (MLIPs)** such as MACE.

It targets computational materials science and heterogeneous catalysis, where the goal is often
to predict the thermodynamic stability and composition of surfaces and nanoparticles under
realistic temperature and chemical-potential conditions. The library is built on the
**Atomic Simulation Environment (ASE)**, so any ASE-compatible calculator -- DFT, classical
potentials, or MLIPs -- can drive the sampling.

The implementation follows the hybrid GCMC scheme of Senftle et al., in which trial
insertions/deletions are paired with a short local relaxation. This makes acceptance
realistic in densely packed metallic systems and is a prerequisite for the workflow used to
build oxidation phase diagrams of Ag surfaces and nanoparticles in our reference application.

Features
--------

- GCMC and Replica-Exchange GCMC in a single, modular run loop.
- Free-volume estimation via Monte Carlo sampling, with element-wise exclusion radii
  (`species_radii`) that can be calibrated from relaxed structures.
- Canonical (NVT) sampling for fixed-composition runs.
- Multiple cell geometries: full periodic box, rectangular sub-slab, and spherical region
  around a nanoparticle.
- Trial-move framework with insertion, deletion, displacement, permutation, shake, and short
  Brownian (Velocity-Verlet) moves, mixed through a weighted `MoveSelector`.
- Per-step output and XYZ trajectory writers with per-move acceptance statistics.
- Phase-diagram utilities for post-processing GCMC ensembles.

.. toctree::
   :caption: Contents:
   :maxdepth: 2

   installation
   cells
   species_radii
   ensembles
   moves
   examples
