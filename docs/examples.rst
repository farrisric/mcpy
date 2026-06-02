Examples
========

This section collects minimal, self-contained scripts for the three standard `mcpy`
workflows: canonical Monte Carlo, grand-canonical Monte Carlo, and replica-exchange GCMC.

Before browsing the individual examples, it is useful to recall the acceptance rules each
workflow relies on and the thermodynamic definitions used downstream to build phase
diagrams. The full derivations live in :doc:`ensembles` and :doc:`species_radii`; what
follows is a quick reference.


Acceptance rules
----------------

Basin Hopping and Metropolis MC share the same functional form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a trial move from state :math:`i` to :math:`j`, both Basin Hopping (BH) and Metropolis
Monte Carlo (MC) accept with

.. math::

   P_{ij}^{\mathrm{acc}} = \min\!\left(1,\; e^{-\beta (E_j - E_i)}\right),

with :math:`\beta = 1/(k_B T)`. The interpretation of :math:`T` differs: in BH it is an
*effective* control parameter that governs how aggressively the search escapes basins, while
in Metropolis MC it is a *physical* temperature that sets the canonical distribution.

GCMC insertion and deletion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number-changing moves carry an extra de-Broglie factor relative to the Metropolis form:

Deletion :math:`(N \rightarrow N-1)`:

.. math::

   P_{ij}^{\,N \rightarrow N-1} =
   \min\!\left(1,\; \frac{N\, \Lambda^3}{z\, V_{\mathrm{free}}}
   \,e^{-\beta (E_j - E_i)}\right),

Insertion :math:`(N \rightarrow N+1)`:

.. math::

   P_{ij}^{\,N \rightarrow N+1} =
   \min\!\left(1,\; \frac{z\, V_{\mathrm{free}}}{(N+1)\, \Lambda^3}
   \,e^{-\beta (E_j - E_i)}\right),

with activity :math:`z = e^{\beta \mu}` and thermal de Broglie wavelength
:math:`\Lambda = h/\sqrt{2\pi m k_B T}`. The accessible free volume
:math:`V_{\mathrm{free}}` -- estimated by the cell, with element-wise exclusion radii --
replaces the geometric :math:`V` of the textbook expressions (see :doc:`species_radii`).


Phase diagrams from GCMC ensembles
----------------------------------

For oxidation studies, the relevant thermodynamic potential is the oxygen-dependent
formation Gibbs energy. For an Ag-O configuration of energy :math:`E` containing
:math:`n_{\mathrm{Ag}}` silver and :math:`n_{\mathrm{O}}` oxygen atoms,

.. math::

   \Delta G = \bigl(E - n_{\mathrm{Ag}}\,\mu_{\mathrm{Ag}} - n_{\mathrm{O}}\,\mu_{\mathrm{O}}\bigr)
              - E_{\mathrm{ref}},

where :math:`E_{\mathrm{ref}}` is the energy of a reference clean Ag system (slab or
nanoparticle). For surfaces, normalising by the slab area :math:`A` yields the surface
Gibbs energy

.. math::

   \gamma = \frac{E - n_{\mathrm{Ag}}\,\mu_{\mathrm{Ag}} - n_{\mathrm{O}}\,\mu_{\mathrm{O}}}{A}
             - \gamma_{\mathrm{ref}},
   \qquad
   \gamma_{\mathrm{ref}} = E_{\mathrm{ref}}/A.

The oxygen chemical potential is varied to mimic a range of environmental conditions:

.. math::

   \mu_{\mathrm{O}}(T, p) =
   \frac{1}{2}\, E^{\mathrm{DFT}}_{\mathrm{O}_2}
   + \Delta \mu_{\mathrm{O}}(T)
   + \frac{1}{2} k_B T \ln\!\left(\frac{p}{p_0}\right),

with :math:`\Delta \mu_{\mathrm{O}}(T)` a tabulated temperature-dependent correction and
:math:`p_0` the reference pressure (typically 1 atm).

Plotting :math:`\Delta G` (or :math:`\gamma`) against :math:`\mu_{\mathrm{O}}` for each
candidate configuration produces the phase diagram; the lower envelope (convex hull) is the
set of thermodynamically stable phases, and crossings between curves locate the phase
transitions.


Examples by workflow
--------------------

Canonical Monte Carlo (NVT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/mc_simulations.rst

Grand Canonical Monte Carlo (GCMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/gcmc_custom_cell.rst
   examples/gcmc_nano.rst
   examples/gcmc_bimet_nano.rst
   examples/gcmc_nano_supported.rst
   examples/gcmc_dome_supported.rst
   examples/gcmc_simulations.rst

Replica-Exchange GCMC (RE-GCMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   examples/replica_exchange_simulations.rst
   examples/phase_diagram_analysis.rst
