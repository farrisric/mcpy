Oxidation phase diagram of a metallic surface
=============================================

This tutorial reproduces the canonical `mcpy` workflow: sweep the oxygen
chemical potential :math:`\mu_{\mathrm{O}}` across a range of environmental
conditions, run an independent GCMC trajectory at each value, then assemble
the resulting configurations into a surface oxidation phase diagram.

It is intentionally longer than :doc:`../first_simulation` — by the end you
will have a publication-style figure, not just a log file.

.. contents::
   :local:
   :depth: 2


Goal
----

For an Ag(111) slab in contact with an oxygen reservoir, determine which
oxide coverages are thermodynamically stable as a function of
:math:`\mu_{\mathrm{O}}`, and locate the transitions between them.

The deliverable is a plot of the surface Gibbs energy :math:`\gamma` vs.
:math:`\mu_{\mathrm{O}}`, with the lower envelope (convex hull) marking the
stable phases.


Prerequisites
-------------

- A working `mcpy` installation with a MACE checkpoint
  (see :doc:`../installation`).
- A successful run of :doc:`../first_simulation` — confirms that the
  calculator and cell are wired correctly.
- Calibrated ``species_radii`` for Ag and O on your potential
  (see :doc:`../species_radii`).


Inputs
------

============================  =================================================
Setting                       Value
============================  =================================================
Surface                       fcc(111), 4×4×3 Ag slab, 8 Å vacuum
Reservoir species             O
Temperature                   500 K
:math:`\mu_{\mathrm{O}}`      sweep: −6.0, −5.5, −5.0, −4.5, −4.0 eV
GCMC steps per condition      2 000 (production), 500 (equilibration)
Move set                      Insertion + Deletion + Displacement of O
Cell                          ``CustomCell`` 5 Å above the top Ag layer
============================  =================================================


Workflow
--------

1. **Per-condition GCMC.** Loop over the target :math:`\mu_{\mathrm{O}}`
   values; for each one start from the same clean slab, run equilibration
   followed by production, and store the trajectory and log under a
   condition-specific filename.
2. **Reference energies.** Compute the energy of the clean Ag slab
   :math:`E_{\mathrm{ref}}` once, separately, and record the bulk Ag
   chemical potential :math:`\mu_{\mathrm{Ag}}`.
3. **Phase-diagram assembly.** For every frame written during production,
   compute the surface Gibbs energy

   .. math::

      \gamma(\mu_{\mathrm{O}}) =
      \frac{E - n_{\mathrm{Ag}}\,\mu_{\mathrm{Ag}}
              - n_{\mathrm{O}}\,\mu_{\mathrm{O}}}{A}
      - \gamma_{\mathrm{ref}},

   then take the lower envelope across all conditions.


Code
----

Per-condition driver script:

.. code-block:: python

   for delta_mu in (-1.0, -0.5, 0.0, 0.5, 1.0):
       run_gcmc(
           delta_mu_gas=delta_mu,
           gcmc_steps=2000,
           outdir=f'sweep/dmu_{delta_mu:+.1f}',
       )

The body of ``run_gcmc`` is the same as in :doc:`../first_simulation`,
parameterised on ``delta_mu_gas`` so the only difference between runs is
:math:`\mu_{\mathrm{O}} = \mu_{\mathrm{O}}^{\mathrm{ref}} + \Delta\mu`.

Phase-diagram post-processing with
:func:`mcpy.utils.phase_diagram.analyze_phase_diagram_results` (concatenate
the per-condition trajectories into one file first; the reference frame is
selected by index):

.. code-block:: python

   from mcpy.utils.phase_diagram import analyze_phase_diagram_results

   results = analyze_phase_diagram_results(
       trajectory_path='sweep/all_relaxed_structures.xyz',
       host_symbol='Ag',
       oxygen_symbol='O',
       idx_ref=0,                    # frame index of the clean reference slab
       e_host=-2.829,                # bulk Ag energy per atom (eV)
       delta_mu_o_min=-1.0,
       delta_mu_o_max=0.0,
       T=500,
       output_plot_path='phase_diagram.png',
   )

The returned dictionary carries the :math:`\gamma(\Delta\mu_{\mathrm{O}})`
curves, the stable-phase transitions, and per-phase oxide ratios; the lower
envelope and the phase tinting are drawn into ``output_plot_path``.


Expected output
---------------

You should obtain:

- One ``gcmc_*.xyz`` and ``gcmc_*.out`` pair per :math:`\mu_{\mathrm{O}}`
  value, under ``sweep/dmu_*/``.
- A phase-diagram figure ``phase_diagram.png``, showing one curve per
  condition and a highlighted convex hull marking the stable phases.

A representative log line at the end of a converged run looks like
(columns: step, N_atoms, energy, per-move acceptance ratios in move order):

.. code-block:: text

   2000       62         -174.550000     22.3%, 10.1%, 41.7%


Interpretation
--------------

- Each :math:`\mu_{\mathrm{O}}` value should converge to a roughly constant
  :math:`N_{\mathrm{O}}` in the second half of its run. If it does not,
  extend the production stage or add displacement moves.
- Intersections between :math:`\gamma(\mu)` curves locate the
  :math:`\mu_{\mathrm{O}}` values at which the surface switches between
  oxide phases. Convert those values to :math:`(T, p_{\mathrm{O}_2})` with
  the standard relation given in :doc:`../examples`.
- A single curve sitting above all others across the whole range usually
  means the structure is metastable and was never visited at equilibrium
  — discard it.


Common pitfalls
---------------

- **Mis-calibrated radii** — insertions all overlap; acceptance is near
  zero. Re-run the calibration in :doc:`../species_radii`.
- **Too-short equilibration** — early frames bias :math:`\gamma`. Drop the
  first 25–50 % of every trajectory before assembling the phase diagram.
- **Reference energy mismatch** — if :math:`E_{\mathrm{ref}}` is computed
  with a different relaxation tolerance than the production runs, the
  curves all shift by a constant. Use the same calculator settings
  everywhere.


Next steps
----------

- Run the sweep in parallel with Replica Exchange: see the
  ``ReplicaExchange`` section of :doc:`../ensembles`.
- Apply the same recipe to a supported nanoparticle: see
  ``examples/gcmc_nano_supported.py``.
- Extend to a bimetallic reservoir by adding a second species to ``mu``
  and registering the corresponding insertion/deletion moves.
