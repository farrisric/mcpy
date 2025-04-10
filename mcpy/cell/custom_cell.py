import numpy as np

from .cell import Cell


class CustomCell(Cell):
    """
    A class representing a custom cell with additional functionality.
    """

    def __init__(self, atoms, custom_height=None, bottom_z=None, species_radii=None,
                 mc_sample_points: int = 100_000):
        """
        Initialize the CustomCell object.

        :param atoms: ASE Atoms object containing the cell information.
        :param custom_height: Optional. The height of the custom cell.
        :param bottom_z: Optional. The z-coordinate of the bottom of the custom cell.
        :param species_radii: Optional. A dictionary where keys are chemical species and values are
        their radii.
        """
        super().__init__(atoms)
        self.dimensions, self.offset = self.get_custom_height_cell(custom_height, bottom_z)
        self.species_radii = species_radii if species_radii else {}
        self.cell_volume = abs(np.linalg.det(self.dimensions))
        self.mc_sample_points = mc_sample_points

    def get_custom_height_cell(self, custom_height, bottom_z):
        """
        Calculate the dimensions and offset for a custom height cell.

        :param custom_height: The height of the custom cell.
        :param bottom_z: The z-coordinate of the bottom of the custom cell.
        :return: A tuple containing the new dimensions and offset.
        """
        if custom_height > self.original_dimensions[2][2]:
            raise ValueError("Custom height cannot be greater than the original cell height.")
        if bottom_z < 0 or bottom_z + custom_height > self.original_dimensions[2][2]:
            raise ValueError("Custom cell exceeds the bounds of the original cell.")

        new_dimensions = np.copy(self.original_dimensions)
        new_dimensions[2][2] = custom_height

        offset = np.zeros(3)
        offset[2] = bottom_z

        return new_dimensions, offset

    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = np.random.rand(3)
        cartesian_point = frac_coords @ self.dimensions + self.offset
        return cartesian_point

    def calculate_volume(self, atoms) -> float:
        """
        Calculate the free volume of the custom cell using a Monte Carlo method.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param n_points: Number of random points to sample.
        :return: Estimated free volume of the cell.
        """
        frac_coords = np.random.rand(self.mc_sample_points, 3)
        cart_coords = frac_coords @ self.dimensions

        positions = atoms.get_positions() - self.offset
        radii = np.array([self.species_radii[atom.symbol] for atom in atoms])
        r_sq = radii**2

        deltas = cart_coords[:, None, :] - positions[None, :, :]
        dists_sq = np.einsum('ijk,ijk->ij', deltas, deltas)

        is_inside_any = np.any(dists_sq <= r_sq[None, :], axis=1)
        count_inside = np.count_nonzero(is_inside_any)

        occupied_fraction = count_inside / self.mc_sample_points
        occupied_volume = self.cell_volume * occupied_fraction
        free_volume = self.cell_volume - occupied_volume

        return free_volume
