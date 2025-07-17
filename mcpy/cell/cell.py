import numpy as np

from .base_cell import BaseCell


class Cell(BaseCell):
    def __init__(self, atoms, species_radii=None):
        """
        Initialize the Cell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        """
        super().__init__()
        self.original_dimensions = np.array(atoms.cell)
        self.dimensions = self.original_dimensions
        self.species_radii = species_radii if species_radii else {}

    def calculate_volume(self, atoms):
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
