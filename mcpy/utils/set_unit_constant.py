import numpy as np
from typing import List
from ase.data import atomic_masses, atomic_numbers


class SetUnits:
    """Class for setting units based on a given string."""

    def __init__(self, unit_type: str, temperature: float, species: List) -> None:
        """
        Initialize the SetUnits class with a specific unit type.

        Parameters:
        unit_type (str): The type of units to set. Possible values are "LJ" or "metal".
        species (List): List of species for which to set units.
        """
        self.unit_type = unit_type
        self.species = species
        self.temperature = temperature

        if unit_type == "LJ":
            self._set_lj_units()
        elif unit_type == "metal":
            self._set_metal_units()
        else:
            raise ValueError("Invalid unit type. Choose 'LJ' or 'metal'.")

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
