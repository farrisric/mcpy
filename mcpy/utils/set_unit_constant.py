import numpy as np
from typing import List
from ase import units
from ase.data import atomic_masses, atomic_numbers


class SetUnits:
    """Class for setting units based on a given string."""

    def __init__(self,
                 unit_type: str,
                 temperature: float,
                 species: List,
                 molecules: dict = None) -> None:
        """
        Initialize the SetUnits class with a specific unit type.

        Args:
            unit_type: The type of units to set, ``"LJ"`` or ``"metal"``.
            temperature: Temperature in K.
            species: List of atomic species for which to set units.
            molecules: Mapping of molecular species name to an ASE Atoms
                template. The name is the key used in ``mu`` and
                ``lambda_dbs``; the mass is the sum of the template's atomic
                masses. Molecular species are identified by composition, so two
                templates with the same composition cannot coexist.
        """
        self.unit_type = unit_type
        self.species = species
        self.temperature = temperature
        self.molecules = molecules or {}

        compositions = {}
        for name, template in self.molecules.items():
            if name in self.species:
                raise ValueError(
                    f"'{name}' is both an atomic species and a molecular "
                    'species; the molecular mass would silently overwrite '
                    'the atomic one. Use a distinct molecular name.'
                )
            key = tuple(sorted(template.get_chemical_symbols()))
            if key in compositions:
                raise ValueError(
                    f"molecular species '{name}' and '{compositions[key]}' share "
                    f'composition {key}; molecules are identified by composition '
                    'so they cannot coexist'
                )
            compositions[key] = name

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
        self.beta = 1 / (self.temperature * self.BOLTZMANN_CONSTANT)
        names = list(self.species) + list(self.molecules)
        self.masses = {specie: 1 for specie in names}
        self.lambda_dbs = {specie: 1 for specie in names}

    def _set_metal_units(self) -> None:
        """Set units for metal potential.

        The constants come from ``ase.units`` rather than being written out
        here: ase is already a hard dependency, and its CODATA values keep the
        de Broglie wavelengths consistent with every other energy in the run.
        """
        self.PLANCK_CONSTANT = units._hplanck / units._e   # eV s
        self.BOLTZMANN_CONSTANT = units.kB                 # eV/K
        self.mass_conversion_factor = units._amu           # amu to kg
        # sqrt(J -> eV) folded with m -> Angstrom, so lambda comes out in A.
        self.lambda_conversion_factor = np.sqrt(units._e) * 1e10
        self.beta = 1/(self.temperature*self.BOLTZMANN_CONSTANT)

        self.masses = {specie: atomic_masses[atomic_numbers[specie]] for specie in self.species}
        for name, template in self.molecules.items():
            self.masses[name] = float(template.get_masses().sum())
        self.lambda_dbs = {
            specie: (
                self.PLANCK_CONSTANT / np.sqrt(
                    2 * np.pi * self.masses[specie] *
                    self.mass_conversion_factor * (1 / self.beta)
                    )
                ) * self.lambda_conversion_factor
            for specie in self.masses
        }

    def de_broglie_insertion(self, volume, n_atoms, specie: str) -> float:
        """Calculate the de Broglie wavelength for insertion."""
        if n_atoms < 0:
            raise ValueError(
                f"n_atoms={n_atoms} < 0 for insertion of '{specie}'; the "
                "pre-move species count cannot be negative"
            )
        return (volume / ((n_atoms+1)*self.lambda_dbs[specie]**3))

    def de_broglie_deletion(self, volume, n_atoms, specie: str) -> float:
        """Calculate the de Broglie wavelength for deletion."""
        return (self.lambda_dbs[specie]**3*n_atoms / volume)
