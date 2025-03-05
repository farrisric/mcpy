from mace.calculators import MACECalculator
from ase import Atoms
from ase.optimize import LBFGS


class MACE_F_Calculator:
    def __init__(self, model_paths: str, steps: int, fmax: float, device: str = 'cuda') -> None:
        """
        Initialize the Calculator with the given model path and device.

        :param model_paths: Path to the model file.
        :param device: Device to load the model on ('cpu' or 'cuda').
        """
        self.fmax = fmax
        self.steps = steps

        self.calculator = MACECalculator(model_paths=model_paths, device=device, enable_cueq=True)

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
