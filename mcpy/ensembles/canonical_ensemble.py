import logging
from ase.units import kB as boltzmann_constant
from ase import Atoms
import numpy as np
import random
from .base_ensemble import BaseEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanonicalEnsemble(BaseEnsemble):
    """
    Represents a canonical ensemble for Monte Carlo simulations.

    Args:
        atoms (Atoms): The initial atomic configuration.
        calculator (Calculator): The calculator used to compute energies and forces.
        random_seed (int, optional): The random seed for the PRNG. Defaults to None.
        optimizer (Optimizer, optional): The optimizer used for relaxation. Defaults to None.
        fmax (float, optional): The maximum force tolerance for relaxation. Defaults to 0.1.
        temperature (float, optional): The temperature of the ensemble in Kelvin. Defaults to 300.
        steps (int, optional): The number of Monte Carlo steps to perform. Defaults to 100.
        op_list (OperatorList, optional): The list of operators for mutations. Defaults to None.
        constraints (Constraints, optional): The constraints applied to the system. Defaults to
        None.
        traj_file (str, optional): The trajectory file name. Defaults to 'traj_test.traj'.
        outfile (str, optional): The output file name. Defaults to 'outfile.out'.
        outfile_write_interval (int, optional): The interval at which to write to the output file.
        Defaults to 10.

    Attributes:
        lowest_energy (float): The lowest potential energy found during the simulation.
        atoms (Atoms): The current atomic configuration.
        constraints (Constraints): The constraints applied to the system.
        _temperature (float): The temperature of the ensemble in Kelvin.
        _calculator (Calculator): The calculator used to compute energies and forces.
        _optimizer (Optimizer): The optimizer used for relaxation.
        _fmax (float): The maximum force tolerance for relaxation.
        _steps (int): The number of Monte Carlo steps to perform.
        _op_list (OperatorList): The list of operators for mutations.
        _step (int): The current step number.
        _accepted_trials (int): The number of accepted trials.

    """

    def __init__(self,
                 atoms,
                 calculator,
                 random_seed=None,
                 optimizer=None,
                 fmax=0.1,
                 temperature=300,
                 op_list=None,
                 constraints=None,
                 p=1,
                 traj_file: str = 'traj_test.traj',
                 outfile: str = 'outfile.out',
                 outfile_write_interval: int = 10) -> None:
        """
        Initializes a CanonicalEnsemble object.

        Args:
            atoms (Atoms): The initial atomic configuration.
            calculator (Calculator): The calculator used to compute energies and forces.
            random_seed (int, optional): The random seed for the PRNG. Defaults to None.
            optimizer (Optimizer, optional): The optimizer used for relaxation. Defaults to None.
            fmax (float, optional): The maximum force tolerance for relaxation. Defaults to 0.1.
            temperature (float, optional): The temperature of the ensemble in Kelvin.
            Defaults to 300.
            steps (int, optional): The number of Monte Carlo steps to perform. Defaults to 100.
            op_list (OperatorList, optional): The list of operators for mutations. Defaults to None.
            constraints (Constraints, optional): The constraints applied to the system. Defaults to
            None.
            traj_file (str, optional): The trajectory file name. Defaults to 'traj_test.traj'.
            outfile (str, optional): The output file name. Defaults to 'outfile.out'.
            outfile_write_interval (int, optional): The interval at which to write to the output
            file. Defaults to 10.

        Returns:
            None
        """

        super().__init__(structure=atoms,
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         outfile=outfile,
                         outfile_write_interval=outfile_write_interval)

        if random_seed is not None:
            random.seed(random_seed)

        self.lowest_energy = float('inf')  # Initialize to positive infinity
        self.atoms = atoms
        self.constraints = constraints
        self._temperature = temperature
        self._calculator = calculator
        self._optimizer = optimizer
        self._fmax = fmax
        self._op_list = op_list
        self.p = p

        self._step = 0
        self._accepted_trials = 0

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """
        Determines whether to accept a trial move based on the potential energy difference and
        temperature.

        Args:
            potential_diff (float): The potential energy difference between the current and new
            configurations.

        Returns:
            bool: True if the trial move is accepted, False otherwise.
        """

        if potential_diff <= 0:
            return True
        elif self._temperature <= 1e-16:
            return False
        else:
            p = np.exp(-potential_diff / (boltzmann_constant * self._temperature))
            return p > self._next_random_number()

    def _next_random_number(self) -> float:
        """
        Returns the next random number from the PRNG.

        Returns:
            float: The next random number.
        """

        return random.random()

    def relax(self, atoms) -> Atoms:
        """
        Relaxes the atomic configuration using the specified optimizer.

        Args:
            atoms (Atoms): The atomic configuration to relax.

        Returns:
            Atoms: The relaxed atomic configuration.
        """

        atoms.info['key_value_pairs'] = {}
        atoms.calc = self._calculator
        opt = self._optimizer(atoms, logfile=None)
        opt.run(fmax=self._fmax)

        Epot = atoms.get_potential_energy()
        atoms.info['key_value_pairs']['potential_energy'] = Epot
        return atoms

    def do_mutation(self):
        """
        Performs mutations on the current atomic configuration.

        Returns:
            Atoms: The mutated atomic configuration.
        """

        new_atoms = self.atoms.copy()
        new_atoms.info['data'] = {'tag': None}
        new_atoms.info['confid'] = 1
        operation = self._op_list.get_operator()
        new_atoms, _ = operation.get_new_individual([new_atoms])
        if self.constraints:
            new_atoms.set_constraint(self.constraints)
        return new_atoms

    def trial_step(self):
        """
        Performs a trial step in the Monte Carlo simulation.

        Returns:
            int: 1 if the trial move is accepted, 0 otherwise.
        """

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
            self.write_traj_file(self.atoms)
            return 1
        return 0

    def run(self, steps: int = 100):
        """
        Runs the Monte Carlo simulation.

        Returns:
            None
        """
        logger.info('+---------------------------------+')
        logger.info('| Canonical Ensemble Monte Carlo  |')
        logger.info('+---------------------------------+')
        logger.info('Starting simulation...')
        logger.info('Temperature: {}'.format(self._temperature))
        logger.info('Number of steps: {}'.format(steps))

        self.relax(self.atoms)
        self.write_traj_file(self.atoms)
        self.write_outfile(self._step, self.lowest_energy)
        self.lowest_energy = self.atoms.get_potential_energy()

        for _ in range(steps):
            accepted = self.trial_step()
            self._step += 1
            self._accepted_trials += accepted

            if self._step % self._outfile_write_interval == 0:
                self.write_outfile(self._step, self.lowest_energy)
                logger.info('Step: {}'.format(self._step))
                logger.info('Lowest energy: {}'.format(self.lowest_energy))
                logger.info('Accepted trials: {}'.format(self._accepted_trials))
