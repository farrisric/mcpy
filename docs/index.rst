mcpy
===========================

Overview
--------

`mcpy` is a Python library designed for performing **Replica-Exchange Grand Canonical Monte Carlo (RE-GCMC)** simulations using **state-of-the-art machine learning interatomic potentials (MLIPs)**.  
It provides a flexible and modular framework to explore the **thermodynamic stability and compositional phase space** of nanoparticles and surfaces under realistic reaction conditions.

Developed for computational materials science and catalysis research, `mcpy` enables users to efficiently sample atomic configurations and construct phase diagrams based on chemical potential, temperature, and pressure conditions.  
The library integrates seamlessly with existing tools in the **Atomic Simulation Environment (ASE)** ecosystem and supports potentials from modern frameworks such as **MACE**.

Features
--------

- **Grand Canonical Monte Carlo (GCMC)** and **Replica-Exchange** algorithms for enhanced sampling.  
- Full compatibility with **machine learning interatomic potentials** (MACE, NequIP, ACE, etc.).  
- Support for **adsorption–desorption processes** on surfaces and nanoparticles.  
- **Flexible ensemble definitions** (canonical, grand canonical, and replica-exchange).  
- Efficient **trial move selection** framework (e.g., adsorption, desorption, translation, exchange).  
- Integration with **ASE** for structure management, input/output, and visualization.  
- Built-in tools for **phase diagram generation**, **thermodynamic analysis**, and **logging**.  
- Modular architecture for easy extension and reproducible workflows.  

.. toctree::
   :caption: Contents:
   :maxdepth: 2

   installation
   cells
   species_radii
   ensembles
   moves
   examples


   


