Ensemble Classes
================

`mcpy` provides ensemble classes to run Monte Carlo simulations in different thermodynamic settings.


Implemented ensembles
---------------------

`BaseEnsemble`
~~~~~~~~~~~~~~

Abstract parent class used by concrete ensembles.
It provides shared infrastructure such as:

- atom/cell/calculator storage,
- energy evaluation,
- trajectory and output writing,
- common run initialization/finalization helpers.

You usually do not instantiate this class directly.


`CanonicalEnsemble`
~~~~~~~~~~~~~~~~~~~

Canonical Monte Carlo ensemble (fixed composition, fixed temperature).
It performs trial mutations and accepts/rejects them with a Metropolis criterion based on energy
change and temperature.

Canonical ensemble equations (NVT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a configuration ``i`` with energy ``E_i``, the canonical Boltzmann probability is

.. math::

   P_i = \frac{e^{-\beta E_i}}{Z},

with inverse temperature ``\beta = 1/(k_B T)`` and partition function

.. math::

   Z = \sum_j e^{-\beta E_j}.

When generating a trial move from state ``i`` to ``j``, the Metropolis acceptance probability is

.. math::

   P_{ij}^{\mathrm{acc}} = \min\left(1, e^{-\beta (E_j - E_i)}\right).

Typical use case: structural optimization/sampling at fixed stoichiometry.


`GrandCanonicalEnsemble`
~~~~~~~~~~~~~~~~~~~~~~~~

Grand Canonical Monte Carlo (GCMC) ensemble (variable composition).
This is the core class for insertion/deletion sampling controlled by:

- chemical potentials `mu`,
- temperature,
- selected species,
- and a `MoveSelector` with trial moves.

Typical use case: adsorption/desorption equilibria and composition changes under given `mu, T`.

Grand canonical ensemble equations (GCMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In GCMC, the number of atoms can fluctuate while sampling a reservoir with fixed chemical potentials
``\mu``. Number-changing moves are accepted with different probabilities for deletion and insertion.

Deletion ``(N \rightarrow N-1)``:

.. math::

   P_{ij}^{N \rightarrow N-1} =
   \min\left(1, \frac{N\Lambda^3}{zV}
   e^{-\beta (E_j - E_i)}\right).

Insertion ``(N \rightarrow N+1)``:

.. math::

   P_{ij}^{N \rightarrow N+1} =
   \min\left(1, \frac{zV}{(N+1)\Lambda^3}
   e^{-\beta (E_j - E_i)}\right).

Here ``\beta = 1/(k_B T)``, ``z = e^{\beta \mu}``, and ``\Lambda`` is the thermal de Broglie wavelength:

.. math::

   \Lambda = \frac{h}{\sqrt{2\pi m k_B T}}.

In ``mcpy``, the geometric ``V`` is replaced by the accessible/free volume estimated from the
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


`ReplicaExchange`
~~~~~~~~~~~~~~~~~

Replica-exchange driver for GCMC simulations across MPI ranks.
It manages multiple replicas (typically at different temperatures or chemical potentials) and
attempts periodic exchanges between neighboring replicas to improve sampling.

Notes:

- requires `mpi4py`,
- designed to wrap GCMC instances created by a user-provided factory.


Simulations without and with relaxation
---------------------------------------

In practice, GCMC workflows are often run in two styles:

**Without relaxation (pure MC acceptance on trial structures)**

- You attempt moves (for example insertion/deletion/displacement),
- evaluate the trial energy directly,
- and accept/reject from the ensemble criterion.

This is the fastest per-step workflow and is often used for broad screening.
For insertion/deletion moves, free volume is still important because it enters acceptance terms.

**With relaxation (MC + local structural equilibration)**

- After a trial move (or after selected intervals), you perform a short relaxation step
  (for example a local optimizer run or short MD-like perturbation),
- then use the relaxed trial structure for acceptance/rejection.

This usually gives more physically equilibrated configurations and can improve convergence quality,
but each step is more expensive.

Free volume, `species_radii`, and `mc_sample_points`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GCMC insertion/deletion, cell objects estimate accessible free volume using Monte Carlo sampling.
Two parameters are especially important:

- `species_radii`: defines atomic exclusion spheres used to classify sampled points as occupied/free.
  Realistic radii improve free-volume estimates and reduce unphysical insertion statistics.
- `mc_sample_points`: number of random points used in the free-volume Monte Carlo estimate.
  More points increase accuracy (lower statistical noise), but increase computational cost.

In relaxation-heavy runs, accurate free-volume estimates are especially useful because they reduce
wasted trial attempts and help the simulation reach stable acceptance behavior faster.

Free-volume Monte Carlo estimate (using `species_radii`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For GCMC insertion/deletion, the accessible/free volume ``V_free`` used in acceptance criteria is
estimated by Monte Carlo sampling inside the configured insertion cell.

Let ``V_cell`` be the geometric volume of the insertion cell and ``N_MC`` the number of random
sample points ``\mathbf{x}_k``. For each sample point, define an occupancy indicator based on the
exclusion sphere radii ``r_{\mathrm{species}(a)}``:

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

In practice, this ``V_free`` replaces the geometric ``V`` in the GCMC insertion/deletion
acceptance equations.


Quick selection guide
---------------------

- Use `CanonicalEnsemble` for fixed-composition sampling.
- Use `GrandCanonicalEnsemble` for variable-composition GCMC.
- Use `ReplicaExchange` to accelerate sampling by exchanging states between replicas.
