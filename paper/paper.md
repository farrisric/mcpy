---
title: 'mcpy: Grand Canonical Monte Carlo for atomistic systems with machine-learning potentials'
tags:
  - Python
  - computational chemistry
  - materials science
  - Monte Carlo
  - grand canonical ensemble
  - machine-learning interatomic potentials
  - nanoparticles
  - heterogeneous catalysis
authors:
  - name: Riccardo Farris
    orcid: 0000-0002-8157-6786
    affiliation: "1, 2"
  - name: Emanuele Telari
    orcid: 0009-0009-3296-959X
    affiliation: 1
  - name: Albert Bruix
    orcid: 0000-0003-2585-5542
    affiliation: 1
affiliations:
  - name: Institut de Química Teòrica i Computacional (IQTCUB), Universitat de Barcelona, Barcelona, Spain
    index: 1
  - name: LEITAT Technological Center, Terrassa, Spain
    index: 2
date: 27 May 2026
bibliography: paper.bib
---

# Summary

`mcpy` is a Python package for grand canonical Monte Carlo (GCMC) simulation of
atomistic systems of any kind (surfaces, nanoparticles, fluids, and bulk) at
controlled temperature and chemical potential. It is built on the Atomic Simulation
Environment (ASE) [@ase2017], so any ASE-compatible calculator can drive the
sampling: density-functional theory codes, classical force fields, or
machine-learning interatomic potentials (MLIPs) such as MACE [@mace2022].
`mcpy` implements both GCMC and replica-exchange GCMC [@swendsen1986] in a
single, modular run loop, and runs with or without local relaxation of trial
moves. Following the hybrid scheme of @senftle2014, insertions and deletions
can be relaxed locally before the acceptance test, which makes sampling
efficient in densely packed metallic systems where rigid insertions would
almost always be rejected; relaxation can equally be switched off for standard
GCMC of fluids and other systems. The same loop runs on CPUs via MPI or on
GPUs through the NVIDIA Alchemi backend, which relaxes a batch of replicas in a
single evaluation. The package provides modular trial
moves (insertion, deletion, displacement, permutation, shake, and Brownian;
permutation enables chemical-ordering optimization in multi-component systems),
configurable cell geometries (periodic box, rectangular sub-slab, spherical
region around a nanoparticle, and user-defined cells), and post-processing
utilities that turn raw ensembles into surface and nanoparticle phase diagrams.

# Statement of need

Predicting the equilibrium composition of surfaces and nanoparticles under
reactive atmospheres requires sampling over particle number (the grand
canonical ensemble) at near–first-principles accuracy. MLIPs now deliver
close-to-DFT accuracy at a small fraction of the cost, making large-scale GCMC
of realistic structures increasingly practical. Yet no open-source package
combines relaxation-coupled GCMC, replica exchange, an ASE-native MLIP backend,
and the cell geometries needed for finite nanoparticles in a single workflow.

`mcpy` is aimed at computational chemists and materials scientists studying
heterogeneous catalysis, oxidation, and gas–surface equilibria, where the
stable composition, not just a fixed structure, is the quantity of interest.
Although it was developed for these applications, the grand-canonical sampling
is system-agnostic: it runs on any system ASE can represent and any
ASE-compatible calculator, from molecular fluids to bulk solids, so the same
workflow extends well beyond the metallic surfaces and nanoparticles that
motivated it.

![Hydrogenation phase diagram of a 201-atom palladium–silver nanoparticle
(Pd$_{151}$Ag$_{50}$) obtained with replica-exchange GCMC in `mcpy`, driven by
the small AGNESI MACE potential [@mace_foundation]. The free energy of formation per atom $\Delta G$ (in meV/atom) is
shown as a function of the hydrogen chemical potential $\Delta\mu_\mathrm{H}$
(equivalently $\log_{10}(p_\mathrm{H}/p_0)$ at $T = 300$ K); the lower band shows
the most stable Pd$_{151}$Ag$_{50}$H$_z$ structure sampled in each
chemical-potential window, from the bare particle at low $\mu_\mathrm{H}$ to a
hydrogen-loaded Pd$_{151}$Ag$_{50}$H$_{111}$ at high pressure. The chemical
ordering of the bare Pd$_{151}$Ag$_{50}$ particle was first optimized with a
single basin-hopping, replica-exchange run over six temperatures; hydrogenated
structures were then sampled by replica-exchange GCMC at hydrogen chemical
potentials from $\Delta\mu_\mathrm{H} = -1$ to $-0.35$ eV with six temperatures
per chemical-potential window.\label{fig:phase}](phase_diagram.png)

# State of the field

Several established codes perform Monte Carlo sampling of atomistic systems, but
each covers only part of the space `mcpy` targets. RASPA2 [@raspa2016] and its
recent C++ rewrite RASPA3 [@raspa3] are specialized for adsorption and diffusion
of molecules in nanoporous materials with classical force fields. DL_MONTE [@dlmonte2021] is a general-purpose Monte
Carlo code that supports GCMC of solids and fluids, but with classical
interaction models and its own input ecosystem rather than native integration
with MLIPs or ASE. LAMMPS [@lammps2022] provides a GCMC fix, parallel tempering,
and machine-learning pair styles, but it neither couples replica exchange to
grand-canonical sampling nor exposes a relaxation-coupled GCMC in which an
arbitrary ASE calculator drives each acceptance test. ASE
itself [@ase2017] supplies the `Atoms` object and a unified calculator interface
spanning DFT codes, classical force fields, and MLIPs, but no grand-canonical
ensemble built on top of it.

We built `mcpy` as a new package rather than extending any one of these because
the capabilities it unifies live in different ecosystems: the
calculator-agnostic accuracy of ASE, the grand-canonical and replica-exchange
machinery of dedicated Monte Carlo codes, and the geometric flexibility needed
for finite nanoparticles. Adding MLIP-native, relaxation-coupled GCMC to RASPA2
or DL_MONTE would require reimplementing ASE's calculator abstraction, while
wrapping LAMMPS in a relaxation-coupled, ASE-native grand-canonical loop with
free-volume cell geometries and *ab initio* thermodynamics post-processing would
reproduce much of `mcpy`'s design around an MD-centric engine. Building directly on ASE instead
lets `mcpy` inherit the entire ASE calculator ecosystem and contribute the
missing grand-canonical layer. That layer is the package's distinct
contribution: ASE-native, MLIP-driven, relaxation-coupled GCMC with replica
exchange and nanoparticle/surface geometries.

# Software design

`mcpy` is organized around four extensible abstractions: *ensembles* that drive
the Monte Carlo loop (`GrandCanonicalEnsemble`, `CanonicalEnsemble`,
`ReplicaExchange`), *moves* that propose trial configurations (insertion,
deletion, displacement, permutation, shake, and Brownian), *cells* that define
the sampling region and estimate its free volume (periodic box, rectangular
sub-slab, spherical region around a nanoparticle, and user-defined cells), and
*calculators* that wrap ASE energy evaluations. This separation reflects the
central design choice: by expressing the physics through ASE's `Atoms` and
calculator interfaces, `mcpy` trades the raw speed of a monolithic, hand-tuned
Monte Carlo kernel for the ability to drive identical sampling code with any
ASE-compatible backend, from a DFT code to a MACE potential [@mace2022]. Because
the dominant cost in the target application is the energy evaluation rather than
the Monte Carlo bookkeeping, that trade-off is strongly favorable, and it lets a
researcher exchange accuracy for cost without rewriting the simulation.

The second design choice is an optional relaxation-coupled acceptance criterion.
Following the hybrid scheme of @senftle2014, each trial insertion or deletion can
be relaxed locally (a short LBFGS minimization) before the acceptance test. This adds cost
to each step, but in densely packed metallic surfaces and nanoparticles a rigid
insertion overlaps neighboring atoms and is almost always rejected; the local
relaxation is what makes grand-canonical sampling converge at all for these
systems, so the per-step cost is the enabling trade-off rather than an overhead.

The third choice concerns ergodicity and parallelism. Replica exchange
[@swendsen1986] across temperatures or chemical potentials improves mixing, and
`mcpy` exposes two complementary parallelization paths: over MPI ranks on CPUs,
one replica per rank, or on a GPU through the NVIDIA Alchemi backend, which
relaxes a batch of replicas together in a single evaluation. The MPI path is
portable and dependency-light (`mpi4py` is optional); the GPU path trades that
portability for throughput on large systems where batched relaxation dominates.
Because the ensemble loop is decoupled from the calculator, the same
replica-exchange code runs under either path.

Finally, the modular loop is reused beyond equilibrium sampling: every ensemble
doubles as a basin-hopping global optimizer, emitting a running trajectory of
strictly improving (minimum-energy) configurations, and replica-exchange runs
additionally report the global minimum found across replicas. Compound moves
that perturb several atoms per trial make these basin-hopping escapes effective
in densely packed systems. Simulations log per-step energies, particle counts,
and per-move acceptance ratios, and write extended-XYZ trajectories alongside an
optional `minima` trajectory of strictly improving structures. The bundled
phase-diagram utilities map sampled ensembles onto *ab initio* thermodynamics
phase diagrams [@reuter2001], connecting the simulated chemical potential to
experimental temperature and partial pressure.

To validate the grand-canonical sampling independently of any MLIP, we
reproduced two Lennard-Jones reference systems against LAMMPS [@lammps2022]
(\autoref{fig:validation}): a purely repulsive fluid in reduced units and real
argon in physical (eV) units. The average particle number and density as a
function of chemical potential agree between the two codes across both unit
systems, including the sharp gas–liquid condensation of argon. Although `mcpy`
has so far been applied mainly to metal surfaces and nanoparticles, this
benchmark confirms that its grand-canonical machinery is correct and general,
applicable to atomistic materials beyond the metallic systems that motivated
it.

![Validation of `mcpy`'s grand-canonical sampling against LAMMPS [@lammps2022]
for two Lennard-Jones reference systems. (a) A purely repulsive Lennard-Jones
fluid in reduced units: chemical potential $\mu$ (in units of $\epsilon$)
versus reduced density $\sigma^3\rho$ and average particle number. (b) Real
argon in physical units: $\mu$ (eV) versus average particle number and number
density, capturing the sharp gas–liquid condensation. `mcpy` (red) reproduces
LAMMPS (blue) across both unit systems.\label{fig:validation}](argon_lammps_validation.png)

# Research impact statement

`mcpy` was developed to study the equilibrium oxidation of metal nanoparticles
and surfaces under reactive atmospheres at near-DFT accuracy. The
replica-exchange GCMC workflow was built during doctoral research at the
Universitat de Barcelona and underpins a forthcoming manuscript on the oxidation
thermodynamics of silver nanoparticles, with further application to additional
catalytic metal/oxide systems within the IQTCUB group. \autoref{fig:phase} shows
a representative result from this workflow: the hydrogenation phase diagram of a
201-atom Pd–Ag nanoparticle, in which `mcpy` recovers the progressive hydrogen
loading of the particle as the chemical potential increases. By coupling
MLIP-driven energetics with grand-canonical sampling and *ab initio*
thermodynamics post-processing, `mcpy` predicts stable surface and nanoparticle
compositions as a function of temperature and chemical potential, quantities
that are otherwise inaccessible to fixed-composition relaxations. The package is
open-source, documented with worked examples covering GCMC, replica exchange,
and phase-diagram construction, and continuously tested across Python 3.11–3.13,
so that the published workflows can be reproduced and extended by other groups.

# Acknowledgements

The authors received no specific funding for this work.

# AI usage disclosure

Claude (Opus) assisted with code refactoring, test scaffolding, packaging
configuration, and editing of the manuscript and documentation. All AI-assisted
outputs were reviewed, edited, and validated by the human authors, who designed
the overall code architecture and take full responsibility for the accuracy and
originality of the software and all submitted materials.

# References
