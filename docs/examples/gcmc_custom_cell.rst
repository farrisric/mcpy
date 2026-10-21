GCMC on Ag(111) with a machine-learned potential
================================================

``examples/03_gcmc_surface_mace.py`` exchanges O and Ag with their
reservoirs on a 48-atom Ag(111) slab.
Every trial is relaxed with the small AGNESI MACE foundation model
(``mace-small-density-agnesi-stress``).
The settings are those of one replica of the Ag oxidation study.
It needs the ``mace`` extra and runs on a CPU (see :doc:`../installation`).
``mace_mp`` downloads the model on the first run.
This page explains its three design choices: the exchange region, the
relaxed trial energy, and the reference chemical potentials.


The exchange region
-------------------

A ``CustomCell`` limits insertions and deletions to a slab-wide window.
The window spans the full :math:`x` and :math:`y` extent of the box and is
``CELL_HEIGHT`` = 7 Å high.
Its floor, ``bottom_z``, sits one O-Ag bond length (``R_O_AG``) *below* the
top Ag layer, so an O atom can also be inserted under the first layer.
Restricting the exchange to a region near the surface keeps trials out of
the vacuum and out of the bulk, as in the ab initio GCMC surface studies of
Wexler et al. (2019).

Each species gets its own cell over the same window.
The cells differ only in ``species_radii``, the clearance that an incoming
atom needs from the atoms already present.
An incoming O needs ``R_O_AG`` = 2.11 Å from Ag, and an incoming Ag needs
``R_AG_AG`` = 2.75 Å.
Both cells keep 0.5 Å from existing O atoms.
The free volume of each cell, and so its insertion and deletion acceptance,
follows from its own radii (:doc:`../species_radii`).


The relaxed trial energy
------------------------

``MACE_F_Calculator`` relaxes each trial structure with LBFGS before it
returns the energy, and it honours the ``FixAtoms`` constraint on the bottom
layer.
The relaxation stops at 0.5 eV/Å or after 40 steps.
The study uses 0.1 eV/Å; the looser value keeps a CPU run short.
A random insertion in a dense metal surface rarely lands at a binding site,
and its unrelaxed energy would reject almost every trial.
The relaxation lets the acceptance test see the nearby minimum instead.
This is the hybrid GCMC scheme of Senftle et al. (2014), and
``docs/adr/0002-trial-energy-is-lbfgs-relaxed.md`` records its trade-off:
the run samples relaxed basins, not the plain configurational ensemble.


Reference chemical potentials
-----------------------------

The script uses fixed DFT (PBE) references, as the study does.
They do not change when you change the potential.
:math:`\mu_\mathrm{Ag}` = -2.99 eV is the bulk Ag energy, -2.82 eV, plus a
thermal free-energy correction of -0.163 eV from thermochemical tables.
The reference for O is :math:`E(\mathrm{O_2})/2` = -4.91 eV.
It comes from
:math:`E(\mathrm{Ag_2O}) - 2E(\mathrm{Ag}) + \Delta H^f_{\mathrm{Ag_2O}}`,
which avoids the PBE error of the O\ :sub:`2` molecule.

``DELTA_MU_O`` sets the O\ :sub:`2` gas conditions relative to
:math:`E(\mathrm{O_2})/2`.
The study uses -0.40, -0.60 and -0.90 eV.
It runs six replicas from 250 to 500 K for each value, with replica exchange
(:doc:`replica_exchange`).

To derive the references from the running potential instead, use
:func:`~mcpy.utils.reference_mu.derive_mu_gas` and
:func:`~mcpy.utils.reference_mu.derive_mu_bulk`, as
``examples/04_replica_exchange_batched.py`` does.
Measure the radii again when you change the potential.


The script
----------

.. literalinclude:: ../../examples/03_gcmc_surface_mace.py
   :language: python


Reading the output
------------------

The log ``gcmc_<tag>.out`` gets one line every ``WRITE_INTERVAL`` steps with
the atom count, the energy, and the acceptance ratio of each of the four
moves.
If one move almost never accepts, check its cell first.
A radius that is too large closes the free volume, and a window that misses
the surface proposes only into vacuum.
Scan ``DELTA_MU_O`` over several runs to build a coverage curve or a phase
diagram (:doc:`../phase_diagrams`).
