GCMC on Ag(111) with a machine-learned potential
================================================

``examples/03_gcmc_surface_mace.py`` is a production setup: O and Ag are both
exchanged with their reservoirs on a 1024-atom Ag(111) slab, and every trial
is relaxed on the GPU with MACE-MPA-0.
It needs the ``alchemi`` extra and a CUDA GPU (see :doc:`../installation`).
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
An incoming O needs ``R_O_AG`` = 2.14 Å from Ag, and an incoming Ag needs
``R_AG_AG`` = 2.75 Å.
Both cells keep 0.5 Å from existing O atoms.
The free volume of each cell, and so its insertion and deletion acceptance,
follows from its own radii (:doc:`../species_radii`).


The relaxed trial energy
------------------------

``AlchemiFCalculator`` relaxes each trial structure with FIRE before it
returns the energy, and it honours the ``FixAtoms`` constraint on the two
bottom layers.
A random insertion in a dense metal surface rarely lands at a binding site,
and its unrelaxed energy would reject almost every trial.
The relaxation lets the acceptance test see the nearby minimum instead.
This is the hybrid GCMC scheme of Senftle et al. (2014), and
``docs/adr/0002-trial-energy-is-lbfgs-relaxed.md`` records its trade-off:
the run samples relaxed basins, not the plain configurational ensemble.


Reference chemical potentials
-----------------------------

The script derives both references from the running potential.
:func:`~mcpy.utils.reference_mu.derive_mu_gas` returns
:math:`E(\mathrm{O_2})/2` for an O\ :sub:`2` molecule relaxed in a box, and
:func:`~mcpy.utils.reference_mu.derive_mu_bulk` returns the bulk Ag energy per
atom at the potential's own lattice constant.
Hardcoded reference energies from another checkpoint silently shift every
:math:`\Delta\mu`, because each potential has its own energy zero.

Only two numbers stay physical inputs.
``DELTA_MU_O`` sets the O\ :sub:`2` gas conditions relative to
:math:`E(\mathrm{O_2})/2`.
``MU_AG_CORRECTION`` = -0.176 eV shifts the Ag reservoir below bulk Ag.
It comes from the group's earlier Ag-oxidation setup; set it to 0.0 for a
reservoir in equilibrium with bulk Ag.

The calibration constants at the top of the script (``AG_LATTICE``,
``R_O_AG``, ``R_AG_AG``) were measured with MACE-MPA-0.
Measure them again when you change the potential.
``examples/04_replica_exchange_batched.py`` uses the same values.


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
