Cell Types
==========

`mcpy` uses *cell objects* to define the geometric region where Monte Carlo moves are sampled and where free volume is estimated.

Available cell types are exposed from `mcpy.cell`:

.. code-block:: python

   from mcpy.cell import Cell, CustomCell, SphericalCell, NullCell


Why cells matter
----------------

In GCMC moves (especially insertion/deletion), the cell controls:

- where random trial points are generated,
- which atoms are considered "inside" the active MC region,
- and the accessible/free volume used in acceptance terms.

Importance of `species_radii` for convergence
---------------------------------------------

Setting `species_radii` is important because free volume is estimated by excluding points that fall inside atomic spheres.
If radii are missing or unrealistic, the free-volume term can be poorly estimated, which leads to less efficient insertion/deletion acceptance behavior.

During relaxation phases of a grand-canonical run, a good free-volume estimate helps the simulation spend less time on unphysical trial moves and improves sampling efficiency.
In practice, this usually speeds up convergence toward the equilibrium composition/structure.

For this reason, always provide physically meaningful radii for the species involved in your GCMC moves.
See :doc:`species_radii` for a practical workflow to compute these values from relaxed structures (for example, O on Ag surfaces).

Importance of `mc_sample_points` for free-volume accuracy
---------------------------------------------------------

`mc_sample_points` sets how many random points are used in the Monte Carlo estimate of free volume.
The algorithm samples points in the active cell and classifies each one as free or occupied.

As a rule of thumb:

- more `mc_sample_points` -> lower statistical noise and higher free-volume accuracy,
- fewer `mc_sample_points` -> faster evaluation but noisier free-volume estimates.

Choose this value by balancing accuracy and runtime cost for your system size and workflow.


`Cell` (base periodic simulation box)
-------------------------------------

`Cell` uses the full ASE simulation cell (`atoms.cell`) as the MC region.
Random points are generated uniformly in fractional coordinates and mapped to Cartesian coordinates.

Use this when your active region is simply the full periodic box.

.. code-block:: python

   from mcpy.cell import Cell

   cell = Cell(atoms, species_radii={"O": 0.0, "Ag": 2.947})
   cell.calculate_volume(atoms)
   V = cell.get_volume()


`CustomCell` (rectangular z-slab inside the box)
------------------------------------------------

`CustomCell` defines a sub-cell with custom height and bottom position along `z`.
This is useful for surfaces or slabs where only part of the simulation box should be sampled.

Key parameters:

- `custom_height`: thickness of the active region,
- `bottom_z`: starting `z` coordinate of that region,
- `species_radii`: per-element exclusion radii for free-volume estimation.

.. code-block:: python

   from mcpy.cell import CustomCell

   custom_cell = CustomCell(
       atoms,
       custom_height=8.0,
       bottom_z=5.0,
       species_radii={"O": 0.0, "Ag": 2.947},
       mc_sample_points=100_000,
   )
   custom_cell.calculate_volume(atoms)


`SphericalCell` (nanoparticle-centered spherical region)
--------------------------------------------------------

`SphericalCell` is designed for nanoparticles.
It recenters the structure at the origin and defines a spherical MC region with:

- radius = max atom distance from center + `vacuum`.

This is typically the most natural region for finite clusters.

.. code-block:: python

   from mcpy.cell import SphericalCell

   spherical_cell = SphericalCell(
       atoms,
       vacuum=3.0,
       species_radii={"Ag": 2.947, "O": 0.0},
       mc_sample_points=100_000,
   )
   spherical_cell.calculate_volume(atoms)


`NullCell` (no active region)
-----------------------------

`NullCell` is a no-op cell object that returns zero volume.
Use it when an algorithm requires a cell-like object but you intentionally do not want insertion-region behavior.

.. code-block:: python

   from mcpy.cell import NullCell

   null_cell = NullCell()
   V = null_cell.get_volume()  # 0


Choosing the right cell
-----------------------

- Use `Cell` for full periodic bulk-like regions.
- Use `CustomCell` for slab/surface windows inside a periodic cell.
- Use `SphericalCell` for nanoparticles and finite clusters.
- Use `NullCell` as a placeholder when no physical insertion region is needed.
