Phase diagram analysis from relaxed trajectories
================================================

This example shows how to analyze relaxed structures and construct a phase-diagram plot
using the utility function in :mod:`mcpy.utils.phase_diagram`.

The script evaluates the phase stability of each relaxed frame as a function of the oxygen
chemical potential :math:`\Delta\mu_O` and selects the lowest-energy structure across the sweep.

Example output
--------------

The figure below was produced by running a short MACE-MP sweep across
:math:`\Delta\mu_O \in \{-0.6, -0.3, 0.0, +0.3\}\,\mathrm{eV}` on an
Ag(111) 3×3×3 slab at :math:`T = 500\,\mathrm{K}` (248 GCMC frames total),
then feeding the merged trajectory into ``analyze_phase_diagram_results``.

.. figure:: ../_static/phase_diagram_quick.png
   :alt: Surface phase diagram from a quick MACE-MP GCMC sweep.
   :width: 80%
   :align: center

   Surface Gibbs energy :math:`\gamma(\Delta\mu_O)` for every relaxed
   configuration in the sweep (gray lines). The colored envelope marks
   the stable phase at each :math:`\Delta\mu_O`; vertical bands separate
   distinct stable phases. The hatched dark-red region indicates
   :math:`\Delta\mu_O > -\Delta H^f_{\mathrm{Ag}_2\mathrm{O}}`, where
   bulk Ag\ :sub:`2`\ O is more stable than any surface phase.

What to read off the plot:

- **Phase transitions** appear as kinks in the lower envelope; the
  :math:`\Delta\mu_O` values are returned in ``transitions_delta_mu_o``.
- **Color intensity** encodes the surface oxide ratio
  (O atoms divided by top-layer Ag atoms above ``z_threshold``).
- The leftmost band corresponds to the clean reference slab
  (``idx_ref``); rightward bands carry increasing O coverage.

Thermodynamic model (from the thesis)
--------------------------------------
For an oxidized configuration at given oxygen chemical potential, the thesis uses the oxygen-dependent formation Gibbs energy.
For surfaces, this is written in terms of the surface Gibbs energy :math:`\gamma`:

.. math::

   \gamma =
   \frac{E - n_{\mathrm{Ag}}\mu_{\mathrm{Ag}} - n_{\mathrm{O}}\mu_{\mathrm{O}}}{A}
   - \gamma_{\mathrm{ref}},

where :math:`E` is the DFT energy of the relaxed configuration, :math:`n_{\mathrm{Ag}}` and
:math:`n_{\mathrm{O}}` are atom counts, and :math:`A` is the surface area.

The oxygen chemical potential :math:`\mu_{\mathrm{O}}(T,p)` is expressed as

.. math::

   \mu_{\mathrm{O}}(T,p) =
   \frac{1}{2} E^{\mathrm{DFT}}_{\mathrm{O}_2}
   + \Delta \mu_{\mathrm{O}}(T,p)
   + \frac{1}{2} \ln\left(\frac{p}{p_0}\right).

To correct DFT overbinding of O$_2$, the thesis replaces :math:`\tfrac{1}{2}E^{\mathrm{DFT}}_{\mathrm{O}_2}` using the experimental formation enthalpy of the corresponding oxide:

.. math::

   \frac{1}{2} E^{\mathrm{DFT}}_{\mathrm{O}_2} =
   E^{\mathrm{DFT}}_{\mathrm{Ag}_2\mathrm{O}}
   - 2E^{\mathrm{DFT}}_{\mathrm{Ag}}
   + \Delta H^{f}_{\mathrm{Ag}_2\mathrm{O}}.

In the code, this correction is embedded in ``e_o2`` (see ``mcpy/utils/phase_diagram.py``).

How `utils/phase_diagram.py` finds stable phases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The function computes :math:`\gamma(\Delta\mu_O)` for each relaxed structure and selects the minimum at each chemical potential value.
Concretely, it performs the following loop:

1. Load all frames from ``relaxed_structures.xyz``.
2. Define a grid of oxygen chemical-potential offsets :math:`\Delta\mu_O` (``delta_mu_o`` in the code).
3. For each structure ``conf`` and each :math:`\Delta\mu_O`, compute the surface Gibbs energy density :math:`\gamma` using the DFT energy, atom counts, the corrected oxygen reference energy ``e_o2``, and the surface area :math:`A`.
   In the implementation, this is done by ``free_en(...)`` and then shifted so that the reference structure at ``idx_ref`` defines :math:`\gamma_{\mathrm{ref}}`.
4. For each :math:`\Delta\mu_O` bin, select the index of the stable phase as ``argmin`` of :math:`\gamma` over all configurations.
5. Detect transition points when the stable phase index changes between neighboring bins; these are returned as ``transitions_delta_mu_o``.

Importing the analysis helper
-----------------------------

.. code-block:: python

   from mcpy.utils.phase_diagram import analyze_phase_diagram_results

   result = analyze_phase_diagram_results(
       trajectory_path="relaxed_structures.xyz",
       idx_ref=2400,
       output_plot_path="lines_phases_mace.png",
       show_plot=False,
   )

   print("Transition points (delta mu_O):", result["transitions_delta_mu_o"])
   print("Stable configuration indices:", result["stable_conf_idx"])
   print("Saved plot:", result["plot_path"])

Inputs and outputs
~~~~~~~~~~~~~~~~~~~
Inputs:

- `trajectory_path`: path to an ASE-readable trajectory (e.g. ``.xyz``) containing the
  relaxed structures you want to classify.
- `idx_ref`: reference frame index used to define ``\\gamma_{\\mathrm{ref}}`` (a constant energy
  offset that shifts the whole phase diagram line).
- `output_plot_path`: where to save the generated phase-diagram plot.
- `show_plot`: whether to display the plot interactively.

Outputs:

- `transitions_delta_mu_o`: oxygen chemical-potential offsets where the stable phase changes.
- `stable_conf_idx`: the index of the lowest-energy configuration at each ``\\Delta\\mu_O`` bin.
- `phase_oxide_ratios`: a per-phase oxide ratio used for coloring.
- `plot_path`: path to the saved figure.

What this function does
-----------------------

- Loads all frames from `relaxed_structures.xyz`.
- Computes free-energy lines as a function of :math:`\Delta\mu_O`.
- Finds the lowest-energy structure at each :math:`\Delta\mu_O`.
- Detects transition points between stable phases.
- Highlights phase regions with background color bands and a color-changing stable-energy line.
- Saves the resulting plot (`lines_phases_mace.png` by default).

Command-line usage
------------------

You can still run the same analysis script directly:

.. code-block:: bash

   python -m mcpy.utils.phase_diagram relaxed_structures.xyz \
       --idx-ref 2400 --output lines_phases_mace.png

