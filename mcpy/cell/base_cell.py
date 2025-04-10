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
