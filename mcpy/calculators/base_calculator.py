from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.optimize import LBFGS

class BaseCalculator:
    """Adapter for using any ASE calculator in mcpy ensembles."""

    def __init__(self, calculator: ASECalculator, steps: int, fmax: float) -> None:
        self.calculator = calculator
        self.steps = steps
        self.fmax = fmax

    def get_potential_energy(self, atoms: Atoms) -> float:
        atoms.calc = self.calculator
        return float(atoms.get_potential_energy())
    
    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Calculate the potential energy of the given atoms.

        :param atoms: ASE Atoms object.
        :return: Potential energy as a float.
        """
        atoms.calc = self.calculator
        opt = LBFGS(atoms, logfile=None)
        opt.run(steps=self.steps, fmax=self.fmax)
        return atoms.get_potential_energy()
