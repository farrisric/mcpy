Ensemble Classes
================

`mcpy` provides ensemble classes to run Monte Carlo simulations in different thermodynamic settings.


Implemented ensembles
---------------------

`BaseEnsemble`
~~~~~~~~~~~~~~

Provides shared Monte Carlo infrastructure used by all concrete ensembles.
It provides shared infrastructure such as:

- atom/cell/calculator storage,
- energy evaluation,
- trajectory and output writing,
- common run initialization/finalization helpers.

You usually do not instantiate this class directly.


`CanonicalEnsemble`
~~~~~~~~~~~~~~~~~~~

Samples the canonical (NVT) distribution at fixed composition and fixed physical temperature.
It performs trial mutations and accepts/rejects them with a Metropolis criterion based on energy
change and temperature.

Canonical ensemble equations (NVT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a configuration :math:`i` with energy :math:`E_i`, the canonical Boltzmann probability is

.. math::

   P_i = \frac{e^{-\beta E_i}}{Z},

with inverse temperature :math:`\beta = 1/(k_B T)` and partition function

.. math::

   Z = \sum_j e^{-\beta E_j}.

When generating a trial move from state :math:`i` to :math:`j`, the Metropolis acceptance probability is

.. math::

   P_{ij}^{\mathrm{acc}} = \min\left(1, e^{-\beta (E_j - E_i)}\right).

Typical use case: structural optimization/sampling at fixed stoichiometry.


`GrandCanonicalEnsemble`
~~~~~~~~~~~~~~~~~~~~~~~~

Samples adsorption/desorption equilibria where the number of selected species fluctuates under reservoir chemical potentials.
This is the core class for insertion/deletion sampling controlled by:

- chemical potentials `mu`,
- temperature,
- selected species,
- and a `MoveSelector` with trial moves.

Typical use case: adsorption/desorption equilibria and composition changes under given `mu, T`.

Grand canonical ensemble equations (GCMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In GCMC, the number of atoms can fluctuate while sampling a reservoir with fixed chemical potentials
:math:`\mu`. Number-changing moves are accepted with different probabilities for deletion and insertion.

Deletion :math:`(N \rightarrow N-1)`:

.. math::

   P_{ij}^{N \rightarrow N-1} =
   \min\left(1, \frac{N\Lambda^3}{zV}
   e^{-\beta (E_j - E_i)}\right).

Insertion :math:`(N \rightarrow N+1)`:

.. math::

   P_{ij}^{N \rightarrow N+1} =
   \min\left(1, \frac{zV}{(N+1)\Lambda^3}
   e^{-\beta (E_j - E_i)}\right).

Here :math:`\beta = 1/(k_B T)`, :math:`z = e^{\beta \mu}`, and :math:`\Lambda` is the thermal de Broglie wavelength:

.. math::

   \Lambda = \frac{h}{\sqrt{2\pi m k_B T}}.

In ``mcpy``, the geometric :math:`V` is replaced by the accessible/free volume estimated from the
configured insertion cell (via `species_radii` and Monte Carlo sampling in the cell object).

.. code-block:: python

   from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
   from mcpy.moves import MoveSelector

   gcmc = GrandCanonicalEnsemble(
       atoms=atoms,
       cells=[cell],
       calculator=calculator,
       mu={"O": -5.0, "Ag": -2.9},
       units_type="metal",
       species=["O"],
       temperature=500,
       move_selector=move_selector,
   )
   gcmc.run(steps=10000)

Inputs / Outputs (statistical meaning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Inputs:

- `mu`: chemical potentials for each species. They bias insertion vs deletion so the
  selected species number fluctuates in equilibrium.
- `temperature`: physical temperature. It sets :math:`\\beta = 1/(k_B T)` in the
  Metropolis factors.
- `cells`: cell objects that define the insertion/deletion region and provide the accessible
  free volume estimate ``V_free`` used in the acceptance rules.
- `species`: which chemical symbols are allowed to change via the move set.
- `move_selector`: collection of trial moves (insertion, deletion, lateral moves, …) and their
  relative proposal weights.

Outputs:

- The ensemble evolves the internal configuration through accept/reject decisions.
- A trajectory file is written to `traj_file` on `trajectory_write_interval` steps.
- A text output file is written to `outfile` on `outfile_write_interval` steps (including
  energy and acceptance ratios per move type).


`ReplicaExchange`
~~~~~~~~~~~~~~~~~

Improves sampling by exchanging GCMC states between replicas across MPI ranks.
It manages multiple replicas (typically at different temperatures or chemical potentials) and
attempts periodic exchanges between neighboring replicas to improve sampling.

Notes:

- requires `mpi4py`,
- designed to wrap GCMC instances created by a user-provided factory.


Simulations without and with relaxation
---------------------------------------

In practice, GCMC workflows are often run in two styles:

**Without relaxation (pure MC acceptance on trial structures)**

- Propose a trial configuration (insertion/deletion/displacement, …).
- Evaluate the trial energy and accept/reject using the ensemble criterion.

This is the fastest per-step workflow and is often used for broad screening.
For insertion/deletion moves, free volume still matters because it enters the acceptance terms.

**With relaxation (MC + local structural equilibration)**

- After proposing a trial configuration, relax it briefly
  (for example with a local optimizer or a short MD-like perturbation),
- then accept/reject using the relaxed energy.

Relaxation reduces sensitivity to unphysical trial overlaps, but each step costs extra compute.

Free volume, `species_radii`, and `mc_sample_points`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insertion/deletion acceptance needs an estimate of how much of the insertion region is
actually available (not excluded by nearby atoms).

In `mcpy`, cell objects compute that free volume with a Monte Carlo estimator. Two parameters
are especially important:

- `species_radii`: defines atomic exclusion spheres used to classify sampled points as occupied/free.
  Realistic radii improve free-volume estimates and reduce unphysical insertion statistics.
- `mc_sample_points`: number of random points used in the free-volume Monte Carlo estimate.
  More points reduce statistical noise, but increase computational cost.

With relaxation-heavy move sets, better free-volume estimates reduce wasted insertions and help
the simulation reach stable acceptance behavior faster.

Free-volume Monte Carlo estimate (using `species_radii`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For GCMC insertion/deletion, the accessible/free volume :math:`V_{\mathrm{free}}` used in acceptance criteria is
estimated by Monte Carlo sampling inside the configured insertion cell.

Let :math:`V_{\mathrm{cell}}` be the geometric volume of the insertion cell and
:math:`N_{\mathrm{MC}}` the number of random sample points :math:`\mathbf{x}_k`.
For each sample point, define an occupancy indicator based on the exclusion sphere radii
:math:`r_{\mathrm{species}(a)}`:

.. math::

   I_k =
   \begin{cases}
   1, & \exists\, a\ \text{such that}\ \|\mathbf{x}_k-\mathbf{r}_a\|^2 \le r_{\mathrm{species}(a)}^2,\\
   0, & \text{otherwise.}
   \end{cases}

The occupied fraction and free volume are then

.. math::

   f_{\mathrm{occ}} = \frac{1}{N_{\mathrm{MC}}}\sum_{k=1}^{N_{\mathrm{MC}}} I_k,
   \qquad
   V_{\mathrm{free}} = V_{\mathrm{cell}}\,(1 - f_{\mathrm{occ}}).

In practice, this :math:`V_{\mathrm{free}}` replaces the geometric :math:`V` in the GCMC insertion/deletion
acceptance equations.


Quick selection guide
---------------------

- Use `CanonicalEnsemble` for fixed-composition sampling.
- Use `GrandCanonicalEnsemble` for variable-composition GCMC.
- Use `ReplicaExchange` to accelerate sampling by exchanging states between replicas.
