import numpy as np

from .base_cell import BaseCell


class Cell(BaseCell):
    def __init__(self, atoms):
        """
        Initialize the Cell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        """
        super().__init__()
        self.original_dimensions = np.array(atoms.cell)
        self.dimensions = self.original_dimensions

    def calculate_volume(self):
        """
        Calculate the volume of the cell.

        :return: Volume of the cell.
        """
        self.volume = abs(np.linalg.det(self.dimensions))

    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = np.random.rand(3)
        cartesian_point = frac_coords @ self.dimensions
        return cartesian_point

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return self.volume
