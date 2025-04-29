from .base_cell import BaseCell


class NullCell(BaseCell):
    def __init__(self):
        """
        Initialize the Cell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        """
        super().__init__()

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return 0

    def calculate_volume(self):
        """
        Calculate the volume of the cell.

        :return: Volume of the cell.
        """
        return 0

    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        return 0
