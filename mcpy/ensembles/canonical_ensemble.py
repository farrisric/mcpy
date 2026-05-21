import logging
import random

import numpy as np
from ase import Atoms
from ase.units import kB as boltzmann_constant

from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)


class CanonicalEnsemble(BaseEnsemble):
    """Canonical (NVT) ensemble Monte Carlo with relaxation-based moves.

    The configuration is mutated by an operator from ``op_list`` (an ASE
    GA-style operator list), then relaxed with ``optimizer``. Acceptance is
    Metropolis at ``temperature`` on the relaxed potential energies.
    """

    def __init__(self,
                 atoms,
                 calculator,
                 cells=None,
                 units_type: str = 'metal',
                 random_seed=None,
                 optimizer=None,
                 fmax=0.1,
                 temperature=300,
                 op_list=None,
                 constraints=None,
                 p=1,
                 traj_file: str = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 outfile: str = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: int = 10,
                 trajectory_write_interval: int = 1) -> None:
        super().__init__(atoms=atoms,
                         cells=cells if cells is not None else [],
                         units_type=units_type,
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         traj_mode=traj_mode,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_mode=outfile_mode,
                         outfile_write_interval=outfile_write_interval)

        if random_seed is not None:
            random.seed(random_seed)

        self.lowest_energy = float('inf')
        self.atoms = atoms
        self.constraints = constraints
        self._temperature = temperature
        self._optimizer = optimizer
        self._fmax = fmax
        self._op_list = op_list
        self.p = p

        self._step = 0
        self._accepted_trials = 0

    def _acceptance_condition(self, potential_diff: float) -> bool:
        if potential_diff <= 0:
            return True
        if self._temperature <= 1e-16:
            return False
        p = np.exp(-potential_diff / (boltzmann_constant * self._temperature))
        return p > random.random()

    def relax(self, atoms) -> Atoms:
        atoms.info['key_value_pairs'] = {}
        atoms.calc = self._calculator
        opt = self._optimizer(atoms, logfile=None)
        opt.run(fmax=self._fmax)

        Epot = atoms.get_potential_energy()
        atoms.info['key_value_pairs']['potential_energy'] = Epot
        return atoms

    def do_mutation(self):
        new_atoms = self.atoms.copy()
        new_atoms.info['data'] = {'tag': None}
        new_atoms.info['confid'] = 1
        operation = self._op_list.get_operator()
        new_atoms, _ = operation.get_new_individual([new_atoms])
        if self.constraints:
            new_atoms.set_constraint(self.constraints)
        return new_atoms

    def trial_step(self):
        num_mutations = np.random.geometric(self.p)
        new_atoms = self.atoms.copy()
        for _ in range(num_mutations):
            new_atoms = self.do_mutation()

        new_atoms = self.relax(new_atoms)

        potential_i = self.atoms.info['key_value_pairs']['potential_energy']
        potential_f = new_atoms.info['key_value_pairs']['potential_energy']

        potential_diff = potential_f - potential_i

        if self._acceptance_condition(potential_diff):
            if new_atoms.info['key_value_pairs']['potential_energy'] < self.lowest_energy:
                self.lowest_energy = new_atoms.info['key_value_pairs']['potential_energy']
            self.atoms = new_atoms
            self.write_coordinates(self.atoms, self.lowest_energy)
            return 1
        return 0

    def initialize_run(self) -> None:
        if self._initialized:
            return
        super().initialize_run()
        self.initialize_outfile()
        self.logger.info("Canonical Ensemble Monte Carlo starting "
                         "(T=%s K, outfile=%s)", self._temperature, self._outfile)

        self.relax(self.atoms)
        self.lowest_energy = self.atoms.get_potential_energy()
        self.write_coordinates(self.atoms, self.lowest_energy)
        self.write_outfile(self._step, self.lowest_energy)

    def _run(self) -> None:
        accepted = self.trial_step()
        self._step += 1
        self._accepted_trials += accepted

        if self._step % self._outfile_write_interval == 0:
            self.write_outfile(self._step, self.lowest_energy)
            self.logger.debug("step=%d lowest_E=%s accepted=%d",
                              self._step, self.lowest_energy, self._accepted_trials)
