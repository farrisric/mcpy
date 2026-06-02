Cells
=====

Cell objects in `mcpy` define **where** Monte Carlo moves act and **how much free volume** is
available for insertions. They are the geometric counterpart of the move set: every cell
provides random points for insertion, classifies atoms as "inside" the active region, and
returns the accessible volume :math:`V_{\mathrm{free}}` used by the GCMC acceptance rules
(see :ref:`free-volume`).

Five cell types are exported from `mcpy.cell`:

.. code-block:: python

   from mcpy.cell import Cell, CustomCell, SphericalCell, DomeCell, NullCell


Choosing exclusion radii (`species_radii`)
------------------------------------------

Every non-trivial cell takes a `species_radii` argument: an element-wise dictionary of
exclusion radii. These radii control how the free-volume Monte Carlo estimator labels each
sampled point as occupied or free, and therefore directly determine the
:math:`V_{\mathrm{free}}` that appears in the GCMC acceptance terms.

The thesis underlying this library calls this quantity :math:`r_{\mathrm{relax}}` and defines
it operationally as the position of the first maximum in the radial distribution function of
the adsorbate around the host atoms, calibrated on relaxed structures with the *same*
calculator used in production. Using values that are too small inflates insertion attempts
into atomic cores; using values that are too large collapses :math:`V_{\mathrm{free}}` and
freezes the chemistry. See :doc:`species_radii` for a complete, reproducible workflow
(including a ready-to-use script for O on Ag) and the recommended conventions for mapping
pair distances onto element-wise radii.


Sampling resolution (`mc_sample_points`)
----------------------------------------

`mc_sample_points` sets the number of random points used in the free-volume estimate.
Larger values reduce statistical noise on :math:`V_{\mathrm{free}}` at the cost of one
up-front sweep per configuration change. In practice, :math:`10^5` points is a good default
for nanoparticle- and slab-sized systems. The cached estimate is reused until the atom count
changes, so the cost is amortised over many trial moves of equal :math:`N`.


`Cell` -- full periodic simulation box
--------------------------------------

`Cell` uses the full ASE simulation cell (`atoms.cell`) as the active MC region. Random
points are sampled uniformly in fractional coordinates and mapped to Cartesian space.

Use this when the active region is simply the entire periodic box.

.. code-block:: python

   from mcpy.cell import Cell

   cell = Cell(atoms, species_radii={"O": 0.0, "Ag": 2.947})
   cell.calculate_volume(atoms)
   V = cell.get_volume()


`CustomCell` -- rectangular sub-slab
------------------------------------

`CustomCell` carves out a sub-volume that spans the full :math:`x` and :math:`y` extent of
the simulation box but has a finite thickness along :math:`z`. This is the standard
construction for surface slabs, where insertions should be limited to the topmost layers
and a slice of vacuum above them.

Key parameters:

- `custom_height` -- thickness of the active region along :math:`z`,
- `bottom_z` -- :math:`z`-coordinate of its lower face,
- `species_radii` -- per-element exclusion radii,
- `mc_sample_points` -- free-volume sampling resolution.

.. code-block:: python

   from mcpy.cell import CustomCell

   custom_cell = CustomCell(
       atoms,
       custom_height=5.5,
       bottom_z=5.0,
       species_radii={"O": 0.0, "Ag": 2.947},
       mc_sample_points=100_000,
   )
   custom_cell.calculate_volume(atoms)


`SphericalCell` -- nanoparticle-centred region
----------------------------------------------

`SphericalCell` is designed for finite clusters. On construction it centres the structure on
the origin and defines a spherical active region of radius

.. math::

   R = \max_a \|\mathbf{r}_a - \mathbf{r}_{\mathrm{c}}\| + \mathrm{vacuum},

i.e. the distance from the centre to the outermost atom plus a user-specified vacuum gap.
Insertions outside this radius are automatically rejected, so the sampling stays close to
the particle surface without wasting moves in pure vacuum.

.. code-block:: python

   from mcpy.cell import SphericalCell

   spherical_cell = SphericalCell(
       atoms,
       vacuum=3.0,
       species_radii={"Ag": 2.947, "O": 0.0},
       mc_sample_points=100_000,
   )
   spherical_cell.calculate_volume(atoms)


`DomeCell` -- supported-nanoparticle dome
-----------------------------------------

`DomeCell` is the supported-particle counterpart of `SphericalCell`. It defines a
hemispherical ("dome") active region: the ball of radius

.. math::

   R = \max_{a\,\in\,\mathrm{particle}} \|\mathbf{r}_a - \mathbf{r}_{\mathrm{c}}\|
       + \mathrm{vacuum},

around the particle centroid :math:`\mathbf{r}_{\mathrm{c}}`, truncated at the support
surface (:math:`z \geq` `bottom_z`). Unlike `SphericalCell` it does **not** translate the
atoms, and it centres on a chosen *particle species* rather than the whole-system centre of
mass -- so the dome sits on the nanoparticle instead of somewhere inside the support slab.

Use this for a nanoparticle on a substrate when insertions should stay on and around the
particle (and its metal-support contact rim) rather than spreading across the bare support,
which is what a slab-spanning `CustomCell` would otherwise allow.

Key parameters:

- `particle_species` -- symbol (or list of symbols) identifying the nanoparticle, e.g.
  ``'Ag'``,
- `bottom_z` -- :math:`z`-coordinate of the support surface; the dome is clipped below it,
- `vacuum` -- padding added to the particle bounding radius,
- `species_radii` -- per-element exclusion radii,
- `mc_sample_points` -- free-volume sampling resolution.

.. code-block:: python

   from mcpy.cell import DomeCell

   dome_cell = DomeCell(
       atoms,
       particle_species="Ag",
       bottom_z=surface_z,
       vacuum=3.0,
       species_radii={"Ag": 2.068, "O": 0.0, "Al": 3.0},
       mc_sample_points=100_000,
   )
   dome_cell.calculate_volume(atoms)

A worked end-to-end run is in :doc:`examples/gcmc_dome_supported`.


`NullCell` -- placeholder
-------------------------

`NullCell` is a no-op cell. Its volume is zero and it provides no insertion points. Use it
as a placeholder where the API requires a cell-like object but no insertion region is
needed (e.g. a pure-displacement canonical run).

.. code-block:: python

   from mcpy.cell import NullCell

   null_cell = NullCell()
   V = null_cell.get_volume()  # 0


Choosing the right cell
-----------------------

- `Cell` -- bulk periodic systems where the whole box is active.
- `CustomCell` -- slab/surface windows inside a periodic box.
- `SphericalCell` -- isolated clusters and nanoparticles.
- `DomeCell` -- nanoparticles supported on a substrate (keeps insertions on the particle).
- `NullCell` -- placeholder for runs that need no insertion region.
