import logging

from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
from ase.optimize import LBFGS

logger = logging.getLogger(__name__)


class BaseCalculator:
    """Adapter for using any ASE calculator in mcpy ensembles.

    ``last_relax_steps`` holds the optimizer steps of the last call and
    ``total_relax_steps`` their running sum. ``steps=0`` is a pure single
    point: no optimizer step and no step-cap warning.
    """

    _optimizer_cls = LBFGS

    def __init__(self, calculator: ASECalculator, steps: int, fmax: float) -> None:
        self.calculator = calculator
        self.steps = steps
        self.fmax = fmax
        self.last_relax_steps = 0
        self.total_relax_steps = 0

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Relax up to ``self.steps`` / ``self.fmax`` (in place) then return the
        potential energy. Logs a warning when the step cap stops the
        relaxation before it converges.
        """
        atoms.calc = self.calculator
        if self.steps > 0:
            opt = self._optimizer_cls(atoms, logfile=None)
            converged = opt.run(steps=self.steps, fmax=self.fmax)
            self.last_relax_steps = int(opt.nsteps)
            if not converged:
                logger.warning('%s relaxation reached the %d-step cap '
                               '(fmax=%.3g eV/A); the energy may be '
                               'unconverged', type(opt).__name__, self.steps,
                               self.fmax)
        else:
            self.last_relax_steps = 0
        self.total_relax_steps += self.last_relax_steps
        return atoms.get_potential_energy()
