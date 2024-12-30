import numpy as np


class SetUnits:
    """Class for setting units based on a given string."""

    def __init__(self, unit_type: str) -> None:
        """
        Initialize the SetUnits class with a specific unit type.

        Parameters:
        unit_type (str): The type of units to set. Possible values are "LJ" or "metal".
        """
        if unit_type == "LJ":
            self.BOLTZMANN_CONSTANT = 1.0
            self.PLANCK_CONSTANT = 1.0
        elif unit_type == "metal":
            self.PLANCK_CONSTANT = 4.13567e-15 # eV/s
            self.BOLTZMANN_CONSTANT = 8.617333e-5  # eV/K
            self.mass_conversion_factor = 1.66053906660e-27 # amu to kg
            self.lambda_conversion_factor = np.sqrt(1.60218e-19)*1e10
        else:
            raise ValueError("Invalid unit type. Choose 'LJ' or 'metal'.")