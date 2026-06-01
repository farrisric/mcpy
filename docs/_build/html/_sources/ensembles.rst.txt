Ensembles
=========

An *ensemble* in `mcpy` is the object that owns the Monte Carlo loop. It holds the atomic
configuration, the energy calculator, the set of cells used for free-volume estimation, and
the `MoveSelector` that proposes trial configurations. It then evaluates the appropriate
acceptance rule and writes the trajectory and log.

Four ensembles are available:

- :class:`CanonicalEnsemble` -- fixed composition (NVT) Metropolis sampling.
- :class:`GrandCanonicalEnsemble` -- variable composition at fixed :math:`(\mu, V, T)`.
- :class:`ReplicaExchange` -- an MPI wrapper that runs several GCMC replicas at different
  temperatures or chemical potentials and periodically attempts exchanges between them.
- :class:`BatchedReplicaExchange` -- single-process variant that batches every replica's
  trial-move energy into a single forward pass (one GPU context, one kernel launch per layer).

All ensembles share a basin-hopping-style *running minimum* output: whenever a strictly
lower-score configuration is observed, it is appended to (or overwrites) a dedicated
``minima_file`` independent of the per-step sampling trajectory.

The remainder of this page describes each one, the acceptance criteria they implement, the
free-volume estimator used by GCMC insertion/deletion, and the running-minimum output.


`BaseEnsemble`
--------------

Shared infrastructure used by every concrete ensemble:

- storage of `atoms`, `cells`, and the ASE-compatible calculator,
- single-point energy evaluation (optionally preceded by a local relaxation),
- trajectory and log writers driven by `trajectory_write_interval` and
  `outfile_write_interval`,
- initialization, finalization, and step-counter management,
- a running-minimum hook (``_record_minimum``) that snapshots the best-so-far configuration
  and optionally writes it to ``minima_file`` -- see :ref:`running-minima`.

`BaseEnsemble` is abstract -- you instantiate one of the concrete subclasses below.


`CanonicalEnsemble`
-------------------

Samples the canonical (NVT) distribution at fixed composition. Trial moves are proposed by
the configured `MoveSelector` and accepted with the Metropolis criterion.

For a configuration :math:`i` with energy :math:`E_i`, the canonical Boltzmann weight is

.. math::

   P_i = \frac{e^{-\beta E_i}}{Z},
   \qquad
   Z = \sum_j e^{-\beta E_j},

with :math:`\beta = 1/(k_B T)`. The acceptance probability for a trial move
:math:`i \rightarrow j` reduces to the symmetric Metropolis form

.. math::

   P_{ij}^{\mathrm{acc}} = \min\!\left(1,\; e^{-\beta (E_j - E_i)}\right).

Typical use: structural optimization or thermal sampling at fixed stoichiometry.


`GrandCanonicalEnsemble`
------------------------

Samples adsorption/desorption equilibria where the number of selected species fluctuates in
contact with a reservoir at fixed chemical potentials.

In a grand-canonical run, the user fixes:

- `temperature` -- sets :math:`\beta = 1/(k_B T)`,
- `mu` -- a dictionary mapping each chemical species to its reservoir chemical potential,
- `species` -- the chemical symbols that the move set is allowed to insert or delete,
- `cells` -- one or more cell objects defining the insertion region and providing the
  accessible (free) volume,
- `move_selector` -- the weighted collection of trial moves.

Acceptance rules
~~~~~~~~~~~~~~~~

For a number-conserving move (e.g. displacement, permutation), `GrandCanonicalEnsemble` falls
back to the Metropolis criterion above. For number-changing moves, the de-Broglie factors
appear:

Deletion :math:`(N \rightarrow N-1)`:

.. math::

   P_{ij}^{\,N \rightarrow N-1} =
   \min\!\left(1,\; \frac{N\,\Lambda^3}{z\,V_{\mathrm{free}}}
   \,e^{-\beta (E_j - E_i)}\right).

Insertion :math:`(N \rightarrow N+1)`:

.. math::

   P_{ij}^{\,N \rightarrow N+1} =
   \min\!\left(1,\; \frac{z\,V_{\mathrm{free}}}{(N+1)\,\Lambda^3}
   \,e^{-\beta (E_j - E_i)}\right).

Here :math:`z = e^{\beta \mu}` is the species activity, and

.. math::

   \Lambda = \frac{h}{\sqrt{2\pi m\, k_B T}}

is the thermal de Broglie wavelength. The geometric volume :math:`V` of the textbook
formulation is replaced by the accessible free volume :math:`V_{\mathrm{free}}` returned by
the configured cell -- see :ref:`free-volume`.

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

Outputs
~~~~~~~

- A trajectory is appended to `traj_file` (extended XYZ, with `energy=` and
  `Lattice=` on the comment line) every `trajectory_write_interval` steps.
- A human-readable log is written to `outfile` every `outfile_write_interval` steps,
  containing the step index, particle count, current energy, and the per-interval
  acceptance ratio of each registered move. The interval counters are reset after every
  write; cumulative ratios are reported at the end of the run.


.. _running-minima:

Running minima and basin-hopping output
---------------------------------------

When the trial move is followed by a local relaxation, the Metropolis loop is, by
construction, a basin-hopping sampler: each accepted move corresponds to a local minimum of
the potential energy surface. Two output channels coexist:

- ``traj_file`` (extended XYZ) -- the **sampling** trajectory, written every
  ``trajectory_write_interval`` accepted steps. Use this for thermodynamic averages and
  visualisation.
- ``minima_file`` (extended XYZ, off by default) -- the **basin-hopping** output. A frame is
  written only when the score of the just-accepted configuration is strictly lower than every
  previous score observed in the run.

Either channel can be disabled independently by passing ``None``. The four useful combinations:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - ``traj_file``
     - ``minima_file``
     - What you get
   * - ``'traj.xyz'``
     - ``None``
     - Standard sampling (default).
   * - ``None``
     - ``'minima.xyz'``
     - Pure basin hopping -- only improving minima are persisted.
   * - ``'traj.xyz'``
     - ``'minima.xyz'``
     - Both sampling and the basin-hopping history.
   * - ``None``
     - ``None``
     - No XYZ output (energies still go to ``outfile``).

``minima_mode`` controls the policy of the minima file:

- ``'a'`` (default) -- append every new minimum, building a history of improving structures.
  The last frame of the file is always the running best.
- ``'w'`` -- overwrite, keeping a single frame with the current best configuration.

Each minima frame carries ``step=`` and ``score=`` in its comment line, so the improvement
trace can be reconstructed from the file alone.

Score function
~~~~~~~~~~~~~~

The quantity compared across configurations is the *score*, returned by ``_minimum_score``.
Defaults:

- :class:`CanonicalEnsemble` -- the potential energy :math:`E`.
- :class:`GrandCanonicalEnsemble` -- the grand potential

  .. math::

     \Omega = E - \sum_i \mu_i N_i,

  which is the meaningful comparison across moves that change particle counts. Subclasses
  may override ``_minimum_score`` to introduce custom criteria (e.g. order parameters).

Global minimum across replicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both :class:`ReplicaExchange` and :class:`BatchedReplicaExchange` accept a
``global_minimum_file`` argument. After all replicas finalise, the lowest-score
configuration observed across the whole ladder is written as a single XYZ frame to that
file. The MPI variant gathers per-rank bests to rank 0; the batched variant minimises
over the in-process replicas. The per-replica ``minima_file`` history is independent of
this and continues to live in the per-rank outputs (e.g. ``minima_rank_3.xyz``).


.. _free-volume:

Free volume and `species_radii`
-------------------------------

In the textbook GCMC equations, :math:`V` is the geometric volume of the simulation box. In
practice, dense atomistic systems leave only a fraction of that volume actually accessible
to a new particle. `mcpy` follows the hybrid scheme of Senftle et al.: every cell object
estimates an accessible volume :math:`V_{\mathrm{free}}` by Monte Carlo sampling and that
quantity is used in the acceptance rules above.

Let :math:`V_{\mathrm{cell}}` be the geometric volume of the insertion region and
:math:`N_{\mathrm{MC}}` the number of random sample points :math:`\mathbf{x}_k`. Each sample
point is classified by an indicator that tests whether it falls inside the exclusion sphere
of any atom :math:`a`:

.. math::

   I_k =
   \begin{cases}
   1, & \exists\, a\ \text{such that}\ \|\mathbf{x}_k-\mathbf{r}_a\|^2 \le r_{\mathrm{species}(a)}^2,\\
   0, & \text{otherwise.}
   \end{cases}

The occupied fraction and the resulting free volume are

.. math::

   f_{\mathrm{occ}} = \frac{1}{N_{\mathrm{MC}}}\sum_{k=1}^{N_{\mathrm{MC}}} I_k,
   \qquad
   V_{\mathrm{free}} = V_{\mathrm{cell}}\,(1 - f_{\mathrm{occ}}).

Two parameters control this estimator on every cell:

- `species_radii` -- an element-wise mapping from chemical species to an exclusion radius.
  Physically meaningful values are critical: if the radii are too small, GCMC will spend
  most of its time proposing overlapping insertions; if they are too large, the accessible
  volume collapses and insertions become impossible. See :doc:`species_radii` for a
  reproducible workflow to calibrate them from short relaxation trials.
- `mc_sample_points` -- number of random points used in the estimate. More points reduce
  statistical noise on :math:`V_{\mathrm{free}}` at the cost of a single up-front sweep per
  configuration change.

With or without relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~

GCMC workflows in `mcpy` are typically run in one of two styles:

- **No relaxation** -- propose, evaluate the trial energy, accept/reject. Fast per step,
  but acceptance is dominated by overlap rejections in dense regions.
- **With relaxation** -- propose, run a short local optimization on the trial structure,
  then accept/reject using the relaxed energy. This is the workflow used in the
  reference application: relaxation absorbs unphysical overlaps and lets the system reach
  stable acceptance behaviour in roughly an order of magnitude fewer steps.

In both cases the free-volume estimate enters the insertion/deletion acceptance, so good
`species_radii` calibration pays off either way.


`ReplicaExchange`
-----------------

`ReplicaExchange` parallelises sampling by running several GCMC replicas at different
temperatures *or* different chemical potentials, one per MPI rank, and periodically
attempting to swap their configurations. It requires `mpi4py` and a user-supplied factory
that constructs the rank-local `GrandCanonicalEnsemble`.

The exchange between two neighbouring replicas in temperature ladders is accepted with the
standard parallel-tempering criterion

.. math::

   P_{\mathrm{exch}} = \min\!\left(1,\; e^{(\beta_2 - \beta_1)(E_2 - E_1)}\right),

and an analogous expression involving :math:`\beta\mu \Delta N` is used when replicas
differ in chemical potentials. After acceptance, replicas swap their configurations and
energies; partner selection alternates between even and odd ranks on consecutive exchange
attempts to ensure every replica eventually mixes with both neighbours.

Notes:

- the ensemble factory must produce per-rank output filenames (e.g. by including `rank` in
  the trajectory, log, and minima paths) so that ranks do not race on the same files;
- one MPI rank corresponds to one replica;
- at the end of the run, rank 0 collects each replica's running best (via ``comm.gather``)
  and writes the global minimum as a single XYZ frame to ``global_minimum_file``
  (default ``"global_minimum.xyz"``; pass ``None`` to disable). See :ref:`running-minima`.


`BatchedReplicaExchange`
------------------------

`BatchedReplicaExchange` runs every replica in a single Python process and evaluates the
trial-move energy of all replicas in one batched calculator call. The accept/reject step is
still sequential per replica -- the batching is over replicas at the same logical step. This
is the right choice on a single GPU, where MPI-based replica exchange would either
oversubscribe one CUDA context or serialise through it.

Requirements:

- the calculator must implement ``get_potential_energies(atoms_list) -> ndarray`` -- the
  optional ``AlchemiCalculator`` does;
- ``gcmc_factory(T=..., rank=i)`` (or ``mu=..., rank=i`` for chemical-potential ladders) must
  return a fresh, independently-seeded :class:`GrandCanonicalEnsemble` with its own cells and
  move selector.

Output layout mirrors :class:`ReplicaExchange`: per-replica ``outfile``/``traj_file``/
``minima_file`` (typically tagged with ``rank``) plus a single ``global_minimum_file``
written from ``min(replicas, key=best_score)`` after the run completes.


Choosing the right ensemble
---------------------------

- :class:`CanonicalEnsemble` -- fixed-composition sampling, structural relaxation, or
  thermal annealing.
- :class:`GrandCanonicalEnsemble` -- variable-composition adsorption/desorption studies at
  fixed :math:`(\mu, T)`, e.g. oxidation phase diagrams.
- :class:`ReplicaExchange` -- multi-node MPI runs, when a single GCMC trajectory is prone to
  getting trapped in a single basin and broader thermodynamic coverage is needed.
- :class:`BatchedReplicaExchange` -- single-GPU runs, where batching trial energies across
  replicas is faster than launching one CUDA context per replica.
