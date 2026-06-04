Utilities
=========

``mcpy.utils`` holds the thermodynamic constants, the seeded RNG, logging setup,
and phase-diagram post-processing.


SetUnits
--------

.. code-block:: python

   SetUnits(unit_type, temperature, species)

Computes the constants the GCMC acceptance rules need. Built automatically by
``GrandCanonicalEnsemble`` from its ``units_type``, ``temperature``, and
``species``. Conceptual background is in :doc:`../ensembles` (section
"Setting a unit system").

- ``unit_type`` (str): ``'metal'`` or ``'LJ'``. Any other value raises
  ``ValueError``.
- ``temperature`` (float): temperature in K.
- ``species`` (list[str]): symbols to precompute masses and wavelengths for.

Attributes: ``beta``, ``lambda_dbs`` (per-species thermal de Broglie
wavelength), ``masses``, ``BOLTZMANN_CONSTANT``, ``PLANCK_CONSTANT``.

Methods:

- ``de_broglie_insertion(volume, n_atoms, specie)``: the insertion prefactor
  ``V / ((n_atoms + 1) Λ³)``. Raises ``ValueError`` if ``n_atoms < 0``.
- ``de_broglie_deletion(volume, n_atoms, specie)``: the deletion prefactor
  ``n_atoms Λ³ / V``.


RandomNumberGenerator
---------------------

.. code-block:: python

   RandomNumberGenerator(seed=None, warm_up=0)

Seeded wrapper over Python's ``random.Random``. Each move and the acceptance
test carries its own instance, so the global ``random`` state is never touched.

- ``seed`` (int, optional): seed for reproducibility.
- ``warm_up`` (int): discarded draws after seeding. The default ``0`` is
  sufficient for the Mersenne Twister.

Methods: ``get_uniform(a=0.0, b=1.0)``, ``get_gaussian(mu=0.0, sigma=1.0)``.


configure
---------

.. code-block:: python

   configure(level=logging.INFO, file=None, mpi_rank=None, stream=True)

Attaches handlers to the top-level ``mcpy`` logger. Importing `mcpy` attaches
no handlers and mutates no global state; call this once at startup to opt in to
log output. Importable from ``mcpy.utils.logging``.

- ``level`` (int): log level for the ``mcpy`` logger.
- ``file`` (str, optional): log file path. Pass ``f"mcpy_rank_{rank}.log"`` for
  per-rank files.
- ``mpi_rank`` (int, optional): included in the format string when set.
- ``stream`` (bool): also emit to stderr.

Returns the configured ``mcpy`` logger. Re-calling is idempotent: it drops the
handlers it previously attached.


plot_phase_diagram
------------------

.. code-block:: python

   plot_phase_diagram(frames, adsorbate='H', metal_symbols=('Pd', 'Ag'),
                      mu_ref=-3.05, kind='surface', T=300.0,
                      dmu_range=(-1.0, 0.0), n_bins=400, ...)

Builds a grand-canonical adsorbate phase diagram for one system configuration
from a list of relaxed frames (or a list of trajectories). The lowest-energy
adsorbate-free frame is the reference. Returns a dict of arrays and metadata and
optionally saves the figure. A worked example is in
:doc:`../examples/phase_diagram_analysis`.

Key parameters:

- ``frames`` (list[ase.Atoms] | list[list]): frames for one configuration.
- ``adsorbate`` (str): species whose count sets the stoichiometry.
- ``kind`` (str): ``'surface'`` normalizes by cell area (meV/Å²); ``'nano'`` by
  metal-atom count (meV/atom). Any other value raises ``ValueError``.
- ``mu_ref`` (float): reference chemical potential of the adsorbate in eV.
- ``adsorbate_count_fn`` (callable, optional): custom ``atoms -> int`` count for
  adsorbates that also occur in an inert sublattice.

See the function docstring for the full plotting and normalization parameters.


analyze_phase_diagram_results
-----------------------------

.. code-block:: python

   analyze_phase_diagram_results(trajectory_path='relaxed_structures.xyz',
                                 host_symbol='Ag', oxygen_symbol='O',
                                 idx_ref=2400, ...)

Reads a relaxed-structure trajectory from disk and builds a surface oxidation
phase diagram against the oxygen chemical potential, returning a dict of arrays
and metadata. The module is also runnable as a script through ``main()``. See
the function docstring for the full set of thermodynamic-reference parameters.


compute_radii
-------------

``mcpy/utils/compute_radii.py`` is a calibration script, not a library API. It
automates the ``species_radii`` workflow for FCC(111) hosts. Configuration and
usage are documented in :doc:`../species_radii`.
