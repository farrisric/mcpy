import numpy as np

from .base_cell import BaseCell


class Cell(BaseCell):
    def __init__(self, atoms, species_radii=None, seed=None):
        """
        Initialize the Cell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param species_radii: Optional dict mapping species to radii.
        :param seed: Optional seed for the cell-local numpy RNG used by
                     :meth:`get_random_point`. ``None`` falls back to the
                     numpy global generator.
        """
        super().__init__()
        self.original_dimensions = np.array(atoms.cell)
        self.dimensions = self.original_dimensions
        self.species_radii = species_radii if species_radii else {}
        # Dimensions are fixed at construction; cache the box volume so
        # ``calculate_volume`` is a single attribute assignment.
        self._box_volume = float(abs(np.linalg.det(self.dimensions)))
        self.volume = self._box_volume
        self._rng = np.random.default_rng(seed)

    def calculate_volume(self, atoms):
        """
        Set the cell volume. ``dimensions`` is fixed at construction so the
        determinant is taken once in ``__init__``.
        """
        self.volume = self._box_volume

    def get_random_point(self):
        """
        Get a random point inside the cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = self._rng.random(3)
        return frac_coords @ self.dimensions

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return self.volume

    def get_atoms_specie_inside_cell(self, atoms, species):
        """
        Get the indices of atoms of a specific species inside the cell.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param species: List of species to filter.
        :return: Indices of atoms of the specified species inside the cell.
        """
        return np.where(np.isin(atoms.get_chemical_symbols(), species))[0]

    def get_species(self):
        """
        Get the species present in the custom cell.

        :return: A list of species present in the custom cell.
        """
        return list(self.species_radii.keys())
