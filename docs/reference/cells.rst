Cells
=====

``mcpy.cell`` defines the insertion region and estimates its free volume.
Conceptual background, including the free-volume estimator and ``species_radii``
calibration, is in :doc:`../cells` and :doc:`../species_radii`.

Every non-trivial cell exposes ``calculate_volume(atoms)`` (refresh the cached
free volume), ``get_volume()``, ``get_random_point()``, and
``get_atoms_specie_inside_cell(atoms, species)``.

Every cell also exposes two point-membership predicates.
``is_point_inside(point)`` bounds the *proposal* region -- the one
``get_random_point`` samples. ``is_point_exchangeable(point)`` bounds the region
the reservoir may take molecules back from, and is what the molecule moves use
to pick candidates. They coincide for every cell except :class:`CustomCell`;
see :doc:`../gcmc_acceptance_convention`.

Every ``species_radii`` mapping must cover every species the system can hold,
including species inserted during the run; a missing entry raises
``ValueError`` from ``calculate_volume``.


.. autoclass:: mcpy.cell.base_cell.BaseCell
   :members:

.. autoclass:: mcpy.cell.Cell
   :members:
   :show-inheritance:

.. autoclass:: mcpy.cell.CustomCell
   :members:
   :show-inheritance:

.. autoclass:: mcpy.cell.SphericalCell
   :members:
   :show-inheritance:

.. autoclass:: mcpy.cell.DomeCell
   :members:
   :show-inheritance:

.. autoclass:: mcpy.cell.NullCell
   :members:
   :show-inheritance:
