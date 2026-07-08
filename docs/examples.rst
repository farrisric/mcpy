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

The formation-energy construction, the lower-envelope logic, and the
post-processing utilities are described in :doc:`phase_diagrams`.
The replica-exchange examples below end in exactly that analysis.


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
