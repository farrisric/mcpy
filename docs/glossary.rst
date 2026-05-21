Glossary
========

Single source of truth for the terms used across the documentation. Every
page that mentions one of these terms links back here on first use.

.. glossary::

   Activity
       The reservoir activity of a chemical species,
       :math:`z = e^{\beta \mu}`, with :math:`\beta = 1/(k_B T)` and
       :math:`\mu` the species' chemical potential. Appears in the GCMC
       insertion and deletion acceptance criteria.

   Cell
       An object that defines the spatial region in which trial moves are
       proposed and returns the accessible free volume to the ensemble. See
       :doc:`cells` for the available geometries.

   Chemical potential
       The thermodynamic driving force the reservoir applies to a species.
       In `mcpy` it is supplied per species through the ``mu`` argument of
       :class:`GrandCanonicalEnsemble`.

   de Broglie wavelength
       The thermal de Broglie wavelength
       :math:`\Lambda = h / \sqrt{2\pi m\, k_B T}`. Enters GCMC acceptance
       through the per-species :math:`\Lambda^3` factor.

   Ensemble
       The object that owns the Monte Carlo loop: it stores the atoms and
       calculator, evaluates acceptance, and writes outputs. See
       :doc:`ensembles`.

   Exclusion radius
       The element-wise radius used by a cell to decide whether a random
       sample point is occupied by an existing atom. Set through
       ``species_radii`` on each cell; calibrated as described in
       :doc:`species_radii`.

   Free volume
       The accessible volume :math:`V_{\mathrm{free}}` returned by a cell,
       estimated by Monte Carlo sampling with exclusion radii. Replaces the
       geometric :math:`V` of the textbook GCMC acceptance criteria.

   GCMC
       Grand Canonical Monte Carlo. Samples the
       :math:`(\mu, V, T)` ensemble: temperature and chemical potentials
       are fixed, the number of selected species fluctuates.

   Hybrid GCMC
       The variant of GCMC introduced by Senftle et al. in which each
       trial move is followed by a short local relaxation before
       acceptance is evaluated. This is the default workflow in `mcpy`.

   Move
       A proposed change to the configuration: insertion, deletion,
       displacement, permutation, shake, or Brownian. Moves are mixed by
       a :class:`MoveSelector`. See :doc:`moves`.

   MoveSelector
       The weighted sampler that draws a move from a list of registered
       moves on every MC step and tracks per-move acceptance counters.

   Relaxation
       Short local geometry optimisation (LBFGS by default) applied to a
       trial configuration before its energy is used in the acceptance
       criterion. Mandatory for the hybrid GCMC workflow.

   Replica
       One independent GCMC trajectory in a Replica-Exchange run, mapped
       to one MPI rank. Replicas differ in temperature, chemical
       potential, or both.

   Replica Exchange
       A wrapper around several GCMC replicas that periodically attempts
       configuration swaps between neighbouring replicas with the
       parallel-tempering acceptance criterion.

   Trajectory
       The extended-XYZ file written by the ensemble. Each frame stores
       atomic positions, the simulation cell, and the current energy on
       its comment line.

   Units type
       ``'metal'`` (eV/Å/amu/K) or ``'LJ'`` — selects the set of
       thermodynamic constants used to compute :math:`\beta` and the
       de Broglie wavelengths. Configured through ``units_type`` on the
       ensemble.
