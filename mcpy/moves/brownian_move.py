import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from .base_move import BaseMove
from ..cell import NullCell


class BrownianMove(BaseMove):
    """Short MD trajectory as the trial move.

    .. warning::
       Not exact detailed balance. The ensemble accepts on the potential
       energy only: there is no kinetic-energy term, so the MD proposal is
       not a symmetric (hybrid MC) move. Use it as basin-hopping-style
       exploration with a relaxing calculator. Do not use it with an
       energy-only calculator when you need equilibrium averages.
    """

    def __init__(self, temperature: float, calculator, steps: int, d_t: float,
                 seed: int) -> None:
        """
        Initialize the Brownian-style MD move. Mutates ``atoms`` in place by
        running ``steps`` of Velocity-Verlet at the given temperature; the
        ensemble snapshots arrays beforehand to allow rollback on rejection.
        ``seed`` seeds the Maxwell-Boltzmann velocity draw.
        """
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.temperature = temperature
        self.calculator = calculator
        self.steps = steps
        self.d_t = d_t
        self.name = 'Brownian'
        self._np_rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms):
        """
        Run a short MD trajectory in place from a Maxwell-Boltzmann velocity
        sample at ``self.temperature``. The atoms keep their own calculator
        and get no ``momenta`` array from the move.
        """
        previous_calc = atoms.calc
        previous_momenta = atoms.arrays.get('momenta')
        try:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature,
                                         rng=self._np_rng)
            atoms.calc = self.calculator
            dyn = VelocityVerlet(atoms, self.d_t * units.fs, logfile=None)
            dyn.run(steps=self.steps)
        finally:
            atoms.calc = previous_calc
            if previous_momenta is None:
                atoms.arrays.pop('momenta', None)
            else:
                atoms.set_momenta(previous_momenta, apply_constraint=False)
        return atoms, 0, 'X'
