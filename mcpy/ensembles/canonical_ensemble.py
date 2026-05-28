import logging
import random

import numpy as np
from ase import Atoms
from ase.units import kB as boltzmann_constant

from .base_ensemble import BaseEnsemble

logger = logging.getLogger(__name__)


class CanonicalEnsemble(BaseEnsemble):
    """Canonical (NVT) ensemble Monte Carlo with relaxation-based moves.

    Each step copies the current configuration, applies one move from an mcpy
    ``MoveSelector`` (e.g. ``PermutationMove``), relaxes it with ``optimizer``,
    and accepts via the Metropolis rule at ``temperature``. Exposes
    ``get_state``/``set_state`` so it can be driven by ``ReplicaExchange`` as a
    temperature ladder.
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
                 move_selector=None,
                 constraints=None,
                 traj_file: str = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 outfile: str = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: int = 10,
                 trajectory_write_interval: int = 1,
                 minima_file: str = None,
                 minima_mode: str = 'a') -> None:
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
                         outfile_write_interval=outfile_write_interval,
                         minima_file=minima_file,
                         minima_mode=minima_mode)

        if random_seed is not None:
            random.seed(random_seed)

        self.lowest_energy = float('inf')
        self._current_energy = None
        self.atoms = atoms
        self.constraints = constraints
        self._temperature = temperature
        self._optimizer = optimizer
        self._fmax = fmax
        self.move_selector = move_selector
        self._beta = 1.0 / (boltzmann_constant * temperature)
        self.exchange_attempts = 0
        self.exchange_successes = 0

        self._step = 0
        self._accepted_trials = 0

    def get_state(self):
        return {
            "atoms": self.atoms,
            "energy": self._current_energy,
            "temperature": self._temperature,
            "beta": self._beta,
            "step": self._step,
            "exchange_attempts": self.exchange_attempts,
            "exchange_successes": self.exchange_successes,
        }

    def set_state(self, state):
        # Swap the configuration only. Each replica keeps its own temperature
        # (and thus _beta) across an exchange; restoring them would collapse
        # the temperature ladder.
        self.atoms = state["atoms"]
        self._current_energy = state["energy"]
        # Guarantee the next trial_step reads a valid energy baseline after a
        # config swap, without relying on the pickled .info surviving MPI.
        self.atoms.info.setdefault("key_value_pairs", {})
        self.atoms.info["key_value_pairs"]["potential_energy"] = state["energy"]
        if "step" in state:
            self._step = state["step"]
        if "exchange_attempts" in state:
            self.exchange_attempts = state["exchange_attempts"]
        if "exchange_successes" in state:
            self.exchange_successes = state["exchange_successes"]

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
        result = self.move_selector.do_trial_move(new_atoms)
        mutated = result[0] if isinstance(result, tuple) else result
        if not mutated:
            return None
        if self.constraints:
            mutated.set_constraint(self.constraints)
        return mutated

    def trial_step(self):
        new_atoms = self.do_mutation()
        if new_atoms is None:
            return 0

        new_atoms = self.relax(new_atoms)

        potential_i = self.atoms.info['key_value_pairs']['potential_energy']
        potential_f = new_atoms.info['key_value_pairs']['potential_energy']
        potential_diff = potential_f - potential_i

        if self._acceptance_condition(potential_diff):
            if potential_f < self.lowest_energy:
                self.lowest_energy = potential_f
            self.atoms = new_atoms
            self._current_energy = potential_f
            self.move_selector.acceptance_counter()
            # Log the accepted configuration's energy, not the running minimum.
            self.write_coordinates(self.atoms, self._current_energy)
            self._record_minimum(self.atoms, self._current_energy)
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
        self._current_energy = self.atoms.get_potential_energy()
        self.lowest_energy = self._current_energy
        self.write_coordinates(self.atoms, self._current_energy)
        self.write_outfile(self._step, self._current_energy)
        self._record_minimum(self.atoms, self._current_energy)

    def _run(self) -> None:
        accepted = self.trial_step()
        self._step += 1
        self._accepted_trials += accepted

        if self._step % self._outfile_write_interval == 0:
            self.write_outfile(self._step, self._current_energy)
            self.logger.debug("step=%d E=%s lowest_E=%s accepted=%d",
                              self._step, self._current_energy, self.lowest_energy,
                              self._accepted_trials)
