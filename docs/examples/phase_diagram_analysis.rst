Phase diagrams from many trajectories
=====================================

A chemical-potential sweep or a replica-exchange ladder leaves many
trajectory files for one system.
:func:`mcpy.utils.plot_phase_diagram` merges them, takes the lowest-energy
adsorbate-free frame as the reference, and selects the stable phase at each
:math:`\Delta\mu`.
:doc:`../phase_diagrams` explains the construction and every parameter; this
page shows the batch workflow.

One call builds the diagram for one configuration, for example one surface
facet or one nanoparticle size.
Extended surfaces are normalised per surface area (meV/Å²) and
nanoparticles per metal atom (meV/atom).

.. code-block:: python

   import glob

   from ase.io import read

   from mcpy.utils import plot_phase_diagram

   # All trajectory files for ONE configuration (here a PdAg(111) surface,
   # sweeping the H chemical potential across several runs and ranks).
   files = sorted(glob.glob('results/regcmc_surface_111/PdAg_111_dmu_*_rank_*.xyz'))
   frames = [read(f, ':') for f in files]   # a list of trajectories is flattened

   result = plot_phase_diagram(
       frames,
       adsorbate='H',
       metal_symbols=('Pd', 'Ag'),
       mu_ref=mu_h_ref,         # 1/2 E(H2) on your calculator's scale
       kind='surface',          # 'surface' -> meV/Å²; 'nano' -> meV/atom
       T=300.0,
       dmu_range=(-1.0, 0.0),
       system_label='surface_111',
       outfile='figures/phase_surface_111.png',
   )

   print('Transitions (delta mu):', result['transitions'])
   print('Stable phases (H counts):', [result['stoich'][i] for i in result['phase_order']])

Here ``mu_h_ref`` comes from
``mcpy.utils.derive_mu_gas(calculator, 'H2')`` with the calculator of the
production runs.

To process several configurations, call the function once per configuration
with the arguments that differ between systems:

.. code-block:: python

   systems = {
       'surface_111': dict(glob='results/regcmc_surface_111/PdAg_111_dmu_*_rank_*.xyz',
                           kind='surface'),
       'nano_small': dict(glob='results/regcmc_nano_small/PdAg_nano_small_dmu_*_rank_*.xyz',
                          kind='nano'),
   }
   for name, cfg in systems.items():
       frames = [read(f, ':') for f in sorted(glob.glob(cfg['glob']))]
       plot_phase_diagram(frames, adsorbate='H', metal_symbols=('Pd', 'Ag'),
                          mu_ref=mu_h_ref, kind=cfg['kind'], system_label=name,
                          outfile=f'figures/phase_{name}.png')

The function returns a dictionary with ``dmu_grid``, ``free``, ``stoich``,
``stable_idx``, ``min_gamma``, ``transitions``, ``phase_order``,
``phase_ratios``, ``unit`` and ``plot_path``.


The surface free energy
-----------------------

For a configuration with :math:`n_{\mathrm{M}}` host atoms and
:math:`n_{\mathrm{A}}` adsorbate atoms, the surface free energy relative to
the clean reference is

.. math::

   \gamma(\Delta\mu) =
   \frac{E - n_{\mathrm{M}}\mu_{\mathrm{M}} - n_{\mathrm{A}}\mu_{\mathrm{A}}}{A}
   - \gamma_{\mathrm{ref}},

where :math:`E` is the energy of the relaxed configuration and :math:`A` the
surface area.
For oxygen from an O\ :sub:`2` reservoir at temperature :math:`T` and
pressure :math:`p`,

.. math::

   \mu_{\mathrm{O}}(T, p) =
   \tfrac{1}{2} E_{\mathrm{O_2}}
   + \Delta\mu_{\mathrm{O}}(T, p^0)
   + \tfrac{1}{2} k_B T \ln\frac{p}{p^0}.

The twin pressure axis of the figure inverts the last term at the given
``T``, with :math:`\Delta\mu_{\mathrm{O}}(T, p^0)` set to zero.


The older oxidation-specific analyzer
-------------------------------------

``analyze_phase_diagram_results`` predates ``plot_phase_diagram`` and is kept
for the scripts that use it.
It reads one concatenated trajectory, takes the reference from a given frame
index (``idx_ref``), and works against tabulated Ag and Ag\ :sub:`2`\ O
reference energies.
It also replaces :math:`\tfrac{1}{2} E_{\mathrm{O_2}}` by an oxide-based
reference that corrects the O\ :sub:`2` overbinding of DFT:

.. math::

   \tfrac{1}{2} E_{\mathrm{O_2}} =
   E_{\mathrm{Ag_2O}} - 2E_{\mathrm{Ag}} + \Delta H^{f}_{\mathrm{Ag_2O}}.

Prefer ``plot_phase_diagram`` for new work.

.. code-block:: python

   from mcpy.utils import analyze_phase_diagram_results

   result = analyze_phase_diagram_results(
       trajectory_path='relaxed_structures.xyz',
       idx_ref=2400,
       output_plot_path='lines_phases_mace.png',
       show_plot=False,
   )
   print(result['transitions_delta_mu_o'], result['stable_conf_idx'])

The same analysis runs from the command line:

.. code-block:: bash

   python -m mcpy.utils.phase_diagram relaxed_structures.xyz \
       --idx-ref 2400 --output lines_phases_mace.png
