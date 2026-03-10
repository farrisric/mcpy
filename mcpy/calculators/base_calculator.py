from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator


class BaseCalculator:
    """Adapter for using any ASE calculator in mcpy ensembles."""

    def __init__(self, calculator: ASECalculator) -> None:
        self.calculator = calculator

    def get_potential_energy(self, atoms: Atoms) -> float:
        atoms.calc = self.calculator
        return float(atoms.get_potential_energy())
