from .moves import BaseMove
from ase import Atoms
# from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


class BrownianMove(BaseMove):
    def __init__(self, temperature: float, calculator , steps: int, d_t: float) -> None:
        """
        Initializes the Shake move with the given maximum displacement distance and RNG.
        """
        self.temperature = temperature
        self.calculator = calculator
        self.steps = steps
        self.d_t = d_t
        self.name = 'Brownian'

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        """
        Performs the shake move by randomly displacing each atom within a sphere of radius r_max.
        """
        atoms_new = atoms.copy()
        MaxwellBoltzmannDistribution(atoms_new, temperature_K=self.temperature)
        atoms_new.calc = self.calculator
        # dyn = Langevin(new_atoms, 5 * units.fs, self.temperature * units.kB, 0.002, logfile=None)
        dyn = VelocityVerlet(atoms_new, self.d_t * units.fs, logfile=None)
        dyn.run(steps=self.steps)
        return atoms_new, 0, 'X'
