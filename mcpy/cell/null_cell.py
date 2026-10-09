from .base_cell import BaseCell


class NullCell(BaseCell):
    def __init__(self):
        """
        Initialize the NullCell object: a placeholder cell for moves that do
        not sample positions from a spatial region (displacement, shake, ...).
        It has no region, so any region query raises a ``TypeError`` that
        names the misuse instead of returning a meaningless answer.
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

    def _no_region(self, *args, **kwargs):
        raise TypeError(
            'NullCell has no region: it is a placeholder for moves that sample '
            'no positions. Insertion, deletion and molecule moves need a real '
            'cell (Cell, CustomCell, SphericalCell or DomeCell).')

    get_random_point = _no_region
    get_random_free_point = _no_region
    free_volume = _no_region
    is_point_inside = _no_region
    get_atoms_specie_inside_cell = _no_region
