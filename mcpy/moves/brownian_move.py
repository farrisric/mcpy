import numpy as np
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from .base_move import BaseMove, fixed_mask
from ..cell import NullCell
from ..utils.md_trajectory import write_md_frame


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
                 seed: int, trajectory_file: str | None = None) -> None:
        """
        Initialize the Brownian-style MD move. Mutates ``atoms`` in place by
        running ``steps`` of Velocity-Verlet at the given temperature; the
        ensemble snapshots arrays beforehand to allow rollback on rejection.
        ``seed`` seeds the Maxwell-Boltzmann velocity draw.

        ``calculator`` must be a plain ASE calculator (e.g. ``EMT()`` or a
        MACE ASE calculator), because the MD needs forces. An mcpy wrapper
        such as ``MACE_F_Calculator`` has no ``get_forces`` and fails here.

        ``trajectory_file`` is a debug option. When set, every MD step of
        every call (accepted or rejected later) is appended to it as an
        extended-XYZ frame with velocities, forces, temperature and energies.
        ``md_run`` in the info line counts the calls of this move.
        """
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.temperature = temperature
        self.calculator = calculator
        self.steps = steps
        self.d_t = d_t
        self.name = 'Brownian'
        self._np_rng = np.random.default_rng(seed)
        self.trajectory_file = trajectory_file
        self._md_run = 0

    def do_trial_move(self, atoms: Atoms):
        """
        Run a short MD trajectory in place from a Maxwell-Boltzmann velocity
        sample at ``self.temperature``. The atoms keep their own calculator
        and get no ``momenta`` array from the move. A failed proposal when
        no atom is free to move.
        """
        fixed = fixed_mask(atoms)
        if fixed.all():
            return False, 0, 'X'
        previous_calc = atoms.calc
        previous_momenta = atoms.arrays.get('momenta')
        try:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature,
                                         rng=self._np_rng)
            # Zero the net momentum: otherwise each accepted move shifts the
            # whole particle a little, and the shifts add up as a random walk.
            # Not with FixAtoms (the fixed atoms anchor the system, as in
            # run_md), and without rescaling: with one free atom that would
            # divide by zero kinetic energy.
            if not fixed.any():
                Stationary(atoms, preserve_temperature=False)
            atoms.calc = self.calculator
            dyn = VelocityVerlet(atoms, self.d_t * units.fs, logfile=None)
            if self.trajectory_file is not None:
                self._md_run += 1
                dyn.attach(self._write_frame, interval=1, atoms=atoms, dyn=dyn)
            dyn.run(steps=self.steps)
        finally:
            atoms.calc = previous_calc
            if previous_momenta is None:
                atoms.arrays.pop('momenta', None)
            else:
                atoms.set_momenta(previous_momenta, apply_constraint=False)
        return atoms, 0, 'X'

    def _write_frame(self, atoms: Atoms, dyn: VelocityVerlet) -> None:
        write_md_frame(self.trajectory_file, atoms, atoms.positions,
                       atoms.get_velocities(), atoms.get_potential_energy(),
                       atoms.get_forces(), md_run=self._md_run,
                       md_step=dyn.nsteps, dt=self.d_t,
                       temperature=self.temperature)
