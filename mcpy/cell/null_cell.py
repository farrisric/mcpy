from .base_cell import BaseCell


class NullCell(BaseCell):
    def __init__(self):
        """
        Initialize the NullCell object: a placeholder cell for moves that do
        not sample positions from a spatial region (displacement, shake, ...).
        """
        super().__init__()

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return 0

    def calculate_volume(self, atoms=None):
        """
        Calculate the volume of the cell. Accepts the ``atoms`` argument every
        other cell takes so a NullCell can sit in an ensemble's cells list.

        :return: Volume of the cell.
        """
        return 0

    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        return 0
