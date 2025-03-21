import numpy as np
from typing import List
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase import Atoms
from .utils import total_volume_with_overlap, get_volume

class SetUnits:
    """Class for setting units based on a given string."""

    def __init__(self,
                 unit_type: str,
                 temperature: float,
                 volume: float,
                 species: List) -> None:
        """
        Initialize the SetUnits class with a specific unit type.

        Parameters:
        unit_type (str): The type of units to set. Possible values are "LJ" or "metal".
        species (List): List of species for which to set units.
        """
        self.unit_type = unit_type
        self.species = species
        self.temperature = temperature
        self.volume = volume
        self._volume = volume

        if unit_type == "LJ":
            self._set_lj_units()
        elif unit_type == "metal":
            self._set_metal_units()
        else:
            raise ValueError("Invalid unit type. Choose 'LJ' or 'metal'.")

    def update_volume(self, atoms: Atoms) -> None:
        """Update the volume of the system."""
        self.volume = self._volume
        for specie in self.species:
            n_specie = len([atom for atom in atoms if atom.symbol == specie])
            radius = covalent_radii[atomic_numbers[specie]]
            self.volume -= 4/3 * np.pi * (radius**3) * n_specie

    def update_volume_insertion(self, atoms, z_shift, box, species_bias) -> None:
        """Update the volume of the system after an insertion move."""
        volume = get_volume(box)
        atoms_bias = [atom for atom in atoms if atom.symbol in species_bias.keys()]
        atoms_in_box = [a.positions for a in atoms_bias if a.position[2] > z_shift and a.position[2] < z_shift + box[2][2]]
        radii = [species_bias[atom.symbol] for atom in atoms_bias]

        volume_with_overlap = total_volume_with_overlap(radii, atoms_in_box)
        self.volume = volume - volume_with_overlap

    def update_volume_deletion(self, atoms, z_shift, box, species_bias) -> None:
        """Update the volume of the system after a deletion move."""
        volume = get_volume(box)
        atoms_bias = [atom for atom in atoms if atom.symbol in species_bias.keys()]
        atoms_in_box = [a.positions for a in atoms_bias if a.position[2] > z_shift and a.position[2] < z_shift + box[2][2]]
        radii = [species_bias[atom.symbol] for atom in atoms_bias]

        volume_with_overlap = total_volume_with_overlap(radii, atoms_in_box)
        self.volume = volume - volume_with_overlap

    def _set_lj_units(self) -> None:
        """Set units for Lennard-Jones (LJ) potential."""
        self.BOLTZMANN_CONSTANT = 1.0
        self.PLANCK_CONSTANT = 1.0
        self.masses = {specie: 1 for specie in self.species}
        self.lambda_dbs = {specie: 1 for specie in self.species}

    def _set_metal_units(self) -> None:
        """Set units for metal potential."""
        self.PLANCK_CONSTANT = 4.13567e-15  # eV/s
        self.BOLTZMANN_CONSTANT = 8.617333e-5  # eV/K
        self.mass_conversion_factor = 1.66053906660e-27  # amu to kg
        self.lambda_conversion_factor = np.sqrt(1.60218e-19) * 1e10
        self.beta = 1/(self.temperature*self.BOLTZMANN_CONSTANT)

        self.masses = {specie: atomic_masses[atomic_numbers[specie]] for specie in self.species}
        self.lambda_dbs = {
            specie: (
                self.PLANCK_CONSTANT / np.sqrt(
                    2 * np.pi * self.masses[specie] *
                    self.mass_conversion_factor * (1 / self.beta)
                    )
                ) * self.lambda_conversion_factor
            for specie in self.species
        }

    def de_broglie_insertion(self, n_atoms, specie: str) -> float:
        """Calculate the de Broglie wavelength for insertion."""
        return (self.volume / ((n_atoms+1)*self.lambda_dbs[specie]**3))

    def de_broglie_deletion(self, n_atoms, specie: str) -> float:
        """Calculate the de Broglie wavelength for deletion."""
        return (self.lambda_dbs[specie]**3*n_atoms / self.volume)
