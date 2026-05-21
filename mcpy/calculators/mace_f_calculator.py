from mace.calculators import MACECalculator
from ase import Atoms
from ase.optimize import LBFGS, FIRE
from typing import Union


_OPTIMIZERS = {'lbfgs': LBFGS, 'fire': FIRE}


class MACE_F_Calculator:
    def __init__(self, model_paths: Union[str, MACECalculator],
                 steps: int, fmax: float,
                 device: str = 'cpu', cueq: bool = False,
                 optimizer: str = 'lbfgs') -> None:
        """
        Initialize the Calculator with the given model path and device.

        :param model_paths: Path to the model file, or an initialized
                            MACECalculator instance to reuse directly.
        :param device: Device to load the model on ('cpu' or 'cuda').
        :param optimizer: ASE optimizer to use: 'lbfgs' (default) or 'fire'.
                          Use 'fire' for apples-to-apples comparison with
                          AlchemiFCalculator (which also uses FIRE).
        """
        self.fmax = fmax
        self.steps = steps
        self.last_relax_steps = 0
        self.total_relax_steps = 0

        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {list(_OPTIMIZERS)}, got {optimizer!r}")
        self._optimizer_cls = _OPTIMIZERS[optimizer]
        self.optimizer_name = optimizer

        if isinstance(model_paths, MACECalculator):
            self.calculator = model_paths
        else:
            self.calculator = MACECalculator(
                model_paths=model_paths,
                device=device,
                enable_cueq=cueq,
            )

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Calculate the potential energy of the given atoms.

        :param atoms: ASE Atoms object.
        :return: Potential energy as a float.
        """
        atoms.calc = self.calculator
        opt = self._optimizer_cls(atoms, logfile=None)
        opt.run(steps=self.steps, fmax=self.fmax)
        self.last_relax_steps = int(opt.nsteps)
        self.total_relax_steps += self.last_relax_steps
        return atoms.get_potential_energy()
