from abc import ABC, abstractmethod


class BaseCell(ABC):
    """
    Abstract base class for different types of cells used in Monte Carlo simulations.
    This class provides a common interface for calculating volumes and generating random points
    within the cell.
    """
    def __init__(self):
        """
        Initialize the BaseCell object.
        """
        pass

    @abstractmethod
    def calculate_volume(self):
        """
        Calculate the volume of the cell.

        :return: Volume of the cell.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def is_point_exchangeable(self, point):
        """Whether a molecule whose center of mass sits at ``point`` may be
        exchanged with the reservoir (counted, deleted, displaced).

        The point counterpart of ``get_atoms_specie_inside_cell``, and the
        predicate the molecule moves use. It is separate from
        ``is_point_inside`` because a region may deliberately accept
        molecules it would never *propose* -- :class:`CustomCell` drops the z
        upper bound so a desorbed molecule stays deletable instead of
        accumulating forever. Defined here on the ABC so cells written
        against the original single-predicate contract (only
        ``is_point_inside``) keep working; cells whose two regions coincide
        (the box, the sphere, the dome) inherit it unchanged.
        """
        return self.is_point_inside(point)
