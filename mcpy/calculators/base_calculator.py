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
        """
        Relax with LBFGS up to ``self.steps`` / ``self.fmax`` then return the
        potential energy.
        """
        atoms.calc = self.calculator
        opt = LBFGS(atoms, logfile=None)
        opt.run(steps=self.steps, fmax=self.fmax)
        return atoms.get_potential_energy()
