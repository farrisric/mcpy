Examples
========


This section presents minimal, self-contained examples for three workflows:

Equations at a glance
-----------------------
The examples rely on a shared set of theoretical ingredients (acceptance criteria, grand-canonical moves, and phase-diagram thermodynamics). In this thesis, the key move acceptance rules are written as follows.

Monte Carlo acceptance (Basin Hopping and Metropolis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a generic move from state :math:`i` to :math:`j`, the Basin Hopping (BH) acceptance probability is

.. math::

   P_{ij}^{\\text{acc}} = \\min\\left(1, e^{-\\beta (E_j - E_i)}\\right),

with :math:`\beta = 1/(k_B T)` where :math:`T` is an *effective* optimization temperature (not a physical thermodynamic temperature).

The Metropolis Monte Carlo (MC) acceptance rule has the same functional form:

.. math::

   P_{ij}^{\\text{acc}} = \\min\\left(1, e^{-\\beta (E_j - E_i)}\\right),

but now :math:`T` is a physical temperature that controls sampling of the canonical ensemble.

Grand Canonical Monte Carlo (GCMC) insertion and deletion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a grand-canonical ensemble, number-changing moves are accepted with different acceptance probabilities for deletion and insertion:

Deletion :math:`(N \rightarrow N-1)`:

.. math::

   P_{ij}^{N \\rightarrow N-1} =
   \\min\\left(1, \\frac{N \\Lambda^3}{z V}
   \\exp\\left[-\\beta (E_j - E_i)\\right]\\right),

Insertion :math:`(N \rightarrow N+1)`:

.. math::

   P_{ij}^{N \\rightarrow N+1} =
   \\min\\left(1, \\frac{z V}{(N+1) \\Lambda^3}
   \\exp\\left[-\\beta (E_j - E_i)\\right]\\right),

where :math:`z = \exp(\beta \mu)`, :math:`\Lambda = h / \sqrt{2\pi m k_B T}` is the thermal de Broglie wavelength, and :math:`V` is replaced by the computed accessible/free volume for insertion (see :doc:`species_radii`).

Phase-diagram thermodynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For oxygen-dependent surface phase stability, the surface Gibbs free energy is written as

.. math::

   \\gamma = \\frac{E - n_{\\mathrm{Ag}}\\mu_{\\mathrm{Ag}} - n_{\\mathrm{O}}\\mu_{\\mathrm{O}}}{A}
   - \\gamma_{\\mathrm{ref}},

with chemical potential :math:`\mu_{\mathrm{O}}(T,p)` expressed as

.. math::

   \\mu_{\\mathrm{O}}(T,p) =
   \\frac{1}{2} E^{\\mathrm{DFT}}_{\\mathrm{O}_2}
   + \\Delta \\mu_{\\mathrm{O}}(T,p)
   + \\frac{1}{2} \\ln\\left(\\frac{p}{p_0}\\right).


Canonical Monte Carlo (MC)
--------------------------

.. toctree::
   :maxdepth: 1

   examples/mc_simulations.rst

Grand Canonical Monte Carlo (GCMC)
----------------------------------

.. toctree::
   :maxdepth: 1

   examples/gcmc_custom_cell.rst
   examples/gcmc_nano.rst
   examples/gcmc_bimet_nano.rst
   examples/gcmc_nano_supported.rst
   examples/gcmc_simulations.rst

Replica-Exchange Grand Canonical Monte Carlo (RE-GCMC)
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   examples/replica_exchange_simulations.rst
   examples/phase_diagram_analysis.rst