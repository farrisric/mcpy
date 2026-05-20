from .base_move import BaseMove
from ase import Atoms
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ..cell import NullCell


class BrownianMove(BaseMove):
    def __init__(self, temperature: float, calculator, steps: int, d_t: float,
                 seed: int) -> None:
        """
        Initialize the Brownian-style MD move. Mutates ``atoms`` in place by
        running ``steps`` of Velocity-Verlet at the given temperature; the
        ensemble snapshots arrays beforehand to allow rollback on rejection.
        """
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.temperature = temperature
        self.calculator = calculator
        self.steps = steps
        self.d_t = d_t
        self.name = 'Brownian'

    def do_trial_move(self, atoms: Atoms):
        """
        Run a short MD trajectory in place from a Maxwell-Boltzmann velocity
        sample at ``self.temperature``.
        """
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        atoms.calc = self.calculator
        dyn = VelocityVerlet(atoms, self.d_t * units.fs, logfile=None)
        dyn.run(steps=self.steps)
        return atoms, 0, 'X'
