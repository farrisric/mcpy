from ..moves import InsertionMove, DeletionMove, DisplacementMove
from .canonical_ensemble import BaseEnsemble
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from typing import Optional, List
import logging


class GrandCanonicalEnsemble(BaseEnsemble):
    def __init__(self,
                 atoms: Atoms,
                 units_type: str,
                 calculator: Calculator,
                 mu : dict,
                 species : list,
                 temperature : float,
                 moves: dict,
                 max_displacement: float,
                 min_max_insert: List[float],
                 volume: float = None,
                 operating_box: List[List[float]] = None,
                 z_shift: float = None,
                 surface_indices: List[float] = None,
                 random_seed: Optional[int] = None,
                 traj_file: str = 'traj_test.traj',
                 trajectory_write_interval: Optional[int] = None,
                 outfile: str = 'outfile.out',
                 outfile_write_interval: int = 10) -> None:

        super().__init__(atoms=atoms,
                         units_type='metal',
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_write_interval=outfile_write_interval)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)  # Set default level, adjust as needed

        self.E_old = self.compute_energy(self.atoms)

        #  SET UNITS AND DEFINE QUANTITIES ###
        # units type and define physical constants
        self.units_type = units_type
        self.units = SetUnits(self.units_type, temperature=temperature, species=species)

        # set cell and get volume
        if operating_box:
            from ase.cell import Cell
            self.operating_box = operating_box
            self.volume = Cell(operating_box).volume  # volume in angstrom^3
            self.z_shift = z_shift
        else:
            self.operating_box = atoms.get_cell()
            self.volume = volume or atoms.get_volume()  # volume in angstrom^3
            self.z_shift = None

        # get number of atoms
        self.initial_atoms = len(self.atoms)
        self.n_atoms = len(self.atoms)
        # get species
        self.species = species
        # define temperature and beta
        self._temperature = temperature
        # define chem. pot.
        self._mu = mu

        ########################################

        #  SET MC PARAMETERS #

        self.surface_indices = surface_indices or None
        self.n_ins_del = moves[0]
        self.n_displ = moves[1]
        self.n_moves = self.n_ins_del + self.n_displ
        self.max_displacement = max_displacement
        self.min_distance, self.max_distance = min_max_insert

        self.initialize_outfile()

        self.frac_ins_del = self.n_ins_del/self.n_moves
        self.rng_move_choice = RandomNumberGenerator(seed=self._random_seed+1)
        self.rng_acceptance = RandomNumberGenerator(seed=self._random_seed+2)
        # Initialize GCMC moves
        self.insert_move = InsertionMove(species=self.species,
                                         operating_box=self.operating_box,
                                         z_shift=self.z_shift,
                                         seed=self._random_seed+3)
        self.deletion_move = DeletionMove(species=self.species,
                                          operating_box=self.operating_box,
                                          z_shift=self.z_shift,
                                          seed=self._random_seed+4)
        self.displace_move = DisplacementMove(species=self.species,
                                              seed=self._random_seed+5,
                                              constraints=atoms.constraints,
                                              max_displacement=self.max_displacement)

        self._step = 0
        self.exchange_attempts = 0
        self.exchange_successes = 0
        # COUNTERS
        self.count_moves = {'Displacements' : 0, 'Insertions' : 0, 'Deletions' : 0}
        self.count_acceptance = {'Displacements' : 0, 'Insertions' : 0, 'Deletions' : 0}

    def get_state(self):
        return {
            "atoms" : self.atoms,
            "n_atoms" : self.n_atoms,
            "energy" : self.E_old,
            "mu": self._mu,
            "temperature": self._temperature,
            "beta": self.units.beta,
            "step": self._step,
            "exchange_attempts": self.exchange_attempts,
            "exchange_successes": self.exchange_successes,
        }

    def set_state(self, state):
        self.atoms = state["atoms"]
        self.E_old = state["energy"]
        self.n_atoms = state["n_atoms"]

    def initialize_outfile(self) -> None:
        """
        Initializes the output file by overwriting any existing content and writing a header.
        """
        try:
            with open(self._outfile, 'w') as outfile:
                # Write the header with proper formatting
                outfile.write("+-------------------------------------------------+\n")
                outfile.write("| Grand Canonical Ensemble Monte Carlo Simulation |\n")
                outfile.write("+-------------------------------------------------+\n\n")

                # Write simulation parameters
                outfile.write("Simulation Parameters:\n")
                outfile.write(f"  Units type: {self.units_type}\n")
                outfile.write(f"  Temperature (K): {self._temperature}\n")
                outfile.write(f"  Volume (Å³): {self.volume:.3f}\n")
                outfile.write(f"  Chemical potentials: {self._mu}\n")
                outfile.write(f"  Number of Insertion-Deletion moves: {self.n_ins_del}\n")
                outfile.write(f"  Interval instance to accept an Insertion Move: "
                              f"{self.min_distance}-{self.max_distance} Å³\n")
                outfile.write(f"  Number of Displacement moves: {self.n_displ}\n")
                outfile.write(f"  Maximum Displacement distance: {self.max_displacement}\n\n")

                # Simulation start message
                outfile.write("Starting simulation...\n")
                outfile.write("-" * 60 + "\n")

                # Write table header
                outfile.write("{:<10} {:<10} {:<15} {:<20}\n".format(
                              "Step", "N_atoms", "Energy (eV)",
                              "Acceptance Ratios (Displ, Ins, Del)"))
        except IOError as e:
            self.logger.error(f"Failed to initialize output file '{self._outfile}': {e}")
            raise

    def write_outfile(self, step: int) -> None:
        """
        Write the step and energy to the output file.

        Args:
            step (int): The current step.
        """
        acceptance_ratios = np.array(list(self.count_acceptance.values())) / np.array(
                    list(self.count_moves.values())
                )
        try:
            with open(self._outfile, 'a') as outfile:
                outfile.write("{:<10} {:<10} {:<15.6f} {:<20}\n".format(
                    self._step,
                    self.n_atoms,
                    self.E_old,
                    ", ".join(f"{ratio*100:.1f}%" if not np.isnan(ratio)
                              else "N/A" for ratio in acceptance_ratios)
                ))
        except IOError as e:
            self.logger.error(f"Error writing to file {self._outfile}: {e}")

    def _acceptance_condition(self,
                              atoms_new: Atoms,
                              potential_diff: float,
                              delta_particles: int,
                              species: str) -> bool:
        """
        Determines whether to accept a trial move based on the potential energy difference and
        temperature.

        Args:
            potential_diff (float): The potential energy difference between the current and new
            configurations.

        Returns:
            bool: True if the trial move is accepted, False otherwise.
        """
        if delta_particles == 0:
            if potential_diff <= 0:
                return True
            else:
                p = np.exp(-potential_diff * self.units.beta)
                return p > self.rng_acceptance.get_uniform()

        if delta_particles == 1:  # Insertion move
            min_distance_surf = min(atoms_new.get_distances(-1, self.surface_indices, mic=True))
            if min_distance_surf > self.max_distance:
                return False
            added_atoms_indices = range(len(atoms_new)-1)
            min_distace_new = min(atoms_new.get_distances(-1, added_atoms_indices, mic=True))
            if min_distace_new < self.min_distance:
                return False
            db_term = (self.volume / ((self.n_atoms+1)*self.units.lambda_dbs[species]**3))
            exp_term = np.exp(-self.units.beta * (potential_diff - self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                f"Lambda_db: {self.units.lambda_dbs[species]:.3e}, p: {p:.3e}, "
                f"Beta: {self.units.beta:.3e}, "
                f"Exp: {exp_term:.3e}, Exp Arg {potential_diff - self._mu[species]}, "
                f"Potential diff: {potential_diff:.3e}, "
                f"Delta_particles: {delta_particles}")

        elif delta_particles == -1:  # Deletion move
            db_term = (self.units.lambda_dbs[species]**3*self.n_atoms / self.volume)
            exp_term = np.exp(-self.units.beta * (potential_diff + self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                f"Lambda_db: {self.units.lambda_dbs[species]:.3e}, p: {p:.3e}, "
                f"Beta: {self.units.beta:.3e}, "
                f"Exp: {exp_term:.3e}, Exp Arg {potential_diff - self._mu[species]}, "
                f"Potential diff: {potential_diff:.3e}, "
                f"Delta_particles: {delta_particles}")
        if p > 1:
            return True
        else:
            return p > self.rng_acceptance.get_uniform()

    def do_displ_move(self):
        self.count_moves['Displacements'] += 1
        return self.displace_move.do_trial_move(self.atoms)

    def do_ins_del_move(self):
        r2 = self.rng_move_choice.get_uniform()
        if r2 <= 0.5:
            self.count_moves['Insertions'] += 1
            return self.insert_move.do_trial_move(self.atoms)
        else:
            self.count_moves['Deletions'] += 1
            return self.deletion_move.do_trial_move(self.atoms)

    def do_trial_step(self, ):
        for move in range(self.n_moves):
            r1 = self.rng_move_choice.get_uniform()
            if r1 <= self.frac_ins_del:
                atoms_new, delta_particles, species = self.do_ins_del_move()
            else:
                atoms_new = self.do_displ_move()
                delta_particles = 0
                species = 'X'

            if not atoms_new:  # NOTE: be carful here
                continue

            E_new = self.compute_energy(atoms_new)
            delta_E = E_new - self.E_old
            if self._acceptance_condition(atoms_new, delta_E, delta_particles, species):
                self.atoms = atoms_new
                self.n_atoms = len(self.atoms)
                self.E_old = E_new
                if delta_particles == 0:
                    self.count_acceptance['Displacements'] += 1
                if delta_particles == 1:
                    self.count_acceptance['Insertions'] += 1
                if delta_particles == -1:
                    self.count_acceptance['Deletions'] += 1

    def compute_energy(self, atoms):
        return self._calculator.get_potential_energy(atoms)

    def initialize_run(self):
        """
        Initializes the Grand Canonical Monte Carlo simulation.
        Prepares logging and computes the initial state.
        """
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("| Grand Canonical Ensemble Monte Carlo Simulation |")
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("Simulation Parameters:")
        self.logger.info(f"Temperature (K): {self._temperature}")
        self.logger.info(f"Volume (Å³): {self.volume:.3f}")
        self.logger.info(f"Chemical potentials: {self._mu}")
        self.logger.info(f"Number of Insertion-Deletion moves: {self.n_ins_del}")
        self.logger.info(
            f"Interval instance to accept an Insertion Move: "
            f"{self.min_distance}-{self.max_distance} Å³")
        self.logger.info(f"Number of Displacement moves: {self.n_displ}")
        self.logger.info(f"Maximum Displacement distance: {self.max_displacement}")
        self.logger.info("Starting simulation...\n")
        self.logger.info("{:<10} {:<10} {:<15} {:<20}".format(
            "Step", "N_atoms", "Energy (eV)", "Acceptance Ratios (Displ, Ins, Del)"
        ))
        self.logger.info("-" * 60)
        self._initialized = True

    def run(self, steps):
        """
        Runs the main loop of the Grand Canonical Monte Carlo simulation.

        Args:
            steps (int): The number of Monte Carlo steps to run.
        """
        if not getattr(self, '_initialized', False):
            self.initialize_run()

        for step in range(1, steps + 1):
            self._run(step)

    def finalize_run(self):
        """
        Finalizes the Grand Canonical Monte Carlo simulation.
        Logs the summary statistics.
        """
        self.logger.info("\nSimulation Complete.")
        self.logger.info("Final Statistics:")
        self.logger.info(f"Total Moves Attempted: {self.n_moves}")
        self.logger.info(f"Acceptance Ratios: {self.count_acceptance}")
        self.logger.info(f"Final Energy (eV): {self.E_old:.6f}")
        self._initialized = False  # Reset the initialization flag

    def _run(self, step):
        """
        Performs a single Monte Carlo step and handles logging and writing.
        """
        self.do_trial_step()

        if self._step % self._outfile_write_interval == 0:
            acceptance_ratios = np.array(list(self.count_acceptance.values())) / np.array(
                list(self.count_moves.values())
            )
            self.logger.info("{:<10} {:<10} {:<15.6f} {:<20}".format(
                self._step,
                self.n_atoms,
                self.E_old,
                ", ".join(f"{ratio*100:.1f}%" if not np.isnan(ratio)
                             else "N/A" for ratio in acceptance_ratios)
            ))
            self.write_outfile(step)
            # reset counters
            self.count_moves = {'Displacements' : 0, 'Insertions' : 0, 'Deletions' : 0}
            self.count_acceptance = {'Displacements' : 0, 'Insertions' : 0, 'Deletions' : 0}

        if self._step % self._trajectory_write_interval == 0:
            self.write_coordinates(self.atoms)
        self._step += 1
