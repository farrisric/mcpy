Cells
=====

``mcpy.cell`` defines the insertion region and estimates its free volume.
Conceptual background, including the free-volume estimator and ``species_radii``
calibration, is in :doc:`../cells` and :doc:`../species_radii`.

Every non-trivial cell exposes ``calculate_volume(atoms)`` (refresh the cached
free volume), ``get_volume()``, ``get_random_point()``, and
``get_atoms_specie_inside_cell(atoms, species)``.


BaseCell
--------

.. code-block:: python

   BaseCell()

Abstract interface. Declares ``calculate_volume``, ``get_random_point``, and
``get_volume``. Not instantiated directly.


Cell
----

.. code-block:: python

   Cell(atoms, species_radii=None, seed=None)

The full ASE simulation box as the active region. Random points are sampled
uniformly in fractional coordinates. Its volume is the fixed box volume, so
``calculate_volume`` does no free-volume sampling.

- ``species_radii`` (dict, optional): per-element exclusion radii.
- ``seed`` (int, optional): cell-local RNG seed.


CustomCell
----------

.. code-block:: python

   CustomCell(atoms, custom_height=None, bottom_z=None, species_radii=None,
              mc_sample_points=100_000, seed=None)

A rectangular sub-slab spanning the full x and y extent with finite thickness
along z. Standard choice for surface slabs.

- ``custom_height`` (float): thickness along z. Must not exceed the box height.
- ``bottom_z`` (float): z of the lower face.
- ``species_radii`` (dict): per-element exclusion radii.
- ``mc_sample_points`` (int): random points for the free-volume estimate.


SphericalCell
-------------

.. code-block:: python

   SphericalCell(atoms, vacuum, species_radii, mc_sample_points=100_000,
                 seed=None)

A sphere around an isolated cluster. On construction it translates the atoms so
their center of mass sits at the origin, then sets the radius to the outermost
atom distance plus ``vacuum``.

- ``vacuum`` (float): padding added to the bounding radius.
- ``species_radii`` (dict): per-element exclusion radii.
- ``mc_sample_points`` (int): random points for the free-volume estimate.


DomeCell
--------

.. code-block:: python

   DomeCell(atoms, particle_species, bottom_z, vacuum, species_radii,
            mc_sample_points=100_000, seed=None)

A hemispherical region for a supported nanoparticle. Centers on the
``particle_species`` centroid and clips the ball at the support surface
(``z >= bottom_z``). Unlike ``SphericalCell`` it does not translate the atoms.

- ``particle_species`` (str | list[str]): symbol(s) identifying the particle.
  Raises ``ValueError`` if none are present.
- ``bottom_z`` (float): z of the support surface.
- ``vacuum`` (float): padding added to the particle bounding radius.
- ``species_radii`` (dict): per-element exclusion radii.
- ``mc_sample_points`` (int): random points for the free-volume estimate.


NullCell
--------

.. code-block:: python

   NullCell()

A no-op cell. ``get_volume()`` returns ``0`` and it provides no insertion
points. Use it as a placeholder where the API needs a cell but no insertion
region exists, such as a pure-displacement canonical run.
