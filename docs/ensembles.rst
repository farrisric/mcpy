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


Quick selection guide
---------------------

- Use `CanonicalEnsemble` for fixed-composition sampling.
- Use `GrandCanonicalEnsemble` for variable-composition GCMC.
- Use `ReplicaExchange` to accelerate sampling by exchanging states between replicas.
