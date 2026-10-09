import numpy as np

from .cell import Cell


class CustomCell(Cell):
    """
    A class representing a custom cell with additional functionality.
    """

    def __init__(self, atoms, custom_height=None, bottom_z=None, species_radii=None,
                 mc_sample_points: int = 100_000, seed=None):
        """
        Initialize the CustomCell object.

        :param atoms: ASE Atoms object containing the cell information.
        :param custom_height: Optional. The height of the custom cell.
        :param bottom_z: Optional. The z-coordinate of the bottom of the custom cell.
        :param species_radii: Optional. A dictionary mapping chemical species to radii.
        :param mc_sample_points: Number of random points used to estimate the free volume.
        :param seed: Optional seed for the cell-local numpy RNG.
        """
        super().__init__(atoms, species_radii=species_radii, seed=seed,
                         mc_sample_points=mc_sample_points)
        self.dimensions, self.offset = self.get_custom_height_cell(custom_height, bottom_z)
        self.cell_volume = float(abs(np.linalg.det(self.dimensions)))
        self._dim_inv = np.linalg.inv(self.dimensions)
        self.volume = self.cell_volume

    def get_custom_height_cell(self, custom_height, bottom_z):
        """
        Calculate the dimensions and offset for a custom-height cell.
        """
        if custom_height is None or bottom_z is None:
            raise ValueError(
                "CustomCell requires explicit custom_height and bottom_z "
                f"(got custom_height={custom_height}, bottom_z={bottom_z})"
            )
        if custom_height > self.original_dimensions[2][2]:
            raise ValueError("Custom height cannot be greater than the original cell height.")
        if bottom_z < 0 or bottom_z + custom_height > self.original_dimensions[2][2]:
            raise ValueError("Custom cell exceeds the bounds of the original cell.")

        new_dimensions = np.copy(self.original_dimensions)
        new_dimensions[2][2] = custom_height

        offset = np.zeros(3)
        offset[2] = bottom_z

        return new_dimensions, offset

    def _inside_mask(self, atoms):
        """Exchangeable with the reservoir.

        In xy an atom must lie within the cell footprint. In z the upper bound
        is intentionally dropped: atoms at or above the cell floor count as
        exchangeable, including ones that desorbed and drifted above the cell
        top (e.g. a recombined H2 floating off the surface). Without this they
        would never be selected for deletion and would accumulate forever.
        Atoms below the floor (absorbed into the subsurface layers) stay
        excluded so they are kept.
        """
        frac = (atoms.positions - self.offset) @ self._dim_inv
        in_xy = np.all((frac[:, :2] >= 0.0) & (frac[:, :2] < 1.0), axis=1)
        return in_xy & (frac[:, 2] >= 0.0)

    def is_point_inside(self, point):
        """
        Check if a given point is inside the custom cell.
        """
        frac = (point - self.offset) @ self._dim_inv
        return bool(np.all((frac >= 0.0) & (frac < 1.0)))

    def is_point_exchangeable(self, point):
        """Same asymmetry as ``get_atoms_specie_inside_cell``: inside the
        xy footprint and at or above the cell floor, with the z upper bound
        dropped.

        Without this a molecule that desorbs above the cell top would stop
        being a deletion candidate *and* would drop out of the per-species
        count in the acceptance factor ``V/((N+1)Λ³)`` -- the runaway
        insertion mode described in docs/gcmc_acceptance_convention.rst.
        Molecules below the floor stay excluded, like atoms.
        """
        frac = (point - self.offset) @ self._dim_inv
        return bool(np.all(frac[:2] >= 0.0) and np.all(frac[:2] < 1.0)
                    and frac[2] >= 0.0)
