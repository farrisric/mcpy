from typing import Optional, List, Dict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .canonical_ensemble import BaseEnsemble
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
from ..moves.move_selector import MoveSelector
from ..cell import Cell


class GrandCanonicalEnsemble(BaseEnsemble):
    def __init__(self,
                 atoms: Atoms,
                 cells: List[Cell],
                 units_type: str,
                 calculator: Calculator,
                 mu: Dict[str, float],
                 species: List[str],
                 temperature: float,
                 move_selector: MoveSelector,
                 random_seed: Optional[int] = None,
                 traj_file: str = 'trajectory.xyz',
                 trajectory_write_interval: Optional[int] = 1,
                 outfile: str = 'outfile.out',
                 outfile_write_interval: Optional[int] = 1) -> None:

        super().__init__(atoms=atoms,
                         cells=cells,
                         units_type='metal',
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_write_interval=outfile_write_interval)

        self.E_old = self.compute_energy(self.atoms)

        self.units = SetUnits(units_type,
                              temperature=temperature,
                              species=species)

        self.initial_atoms = len(self.atoms)
        self.n_atoms = len(self.atoms)
        self._temperature = temperature
        self._mu = mu

        self.move_selector = move_selector
        self.calculate_cells_volume(self.atoms)
        self._step = 0
        self.exchange_attempts = 0
        self.exchange_successes = 0
        self.rng_acceptance = RandomNumberGenerator(seed=self._random_seed + 2)

        self.initialize_outfile()

    def get_state(self) -> Dict[str, any]:
        return {
            "atoms": self.atoms,
            "n_atoms": self.n_atoms,
            "energy": self.E_old,
            "mu": self._mu,
            "temperature": self._temperature,
            "beta": self.units.beta,
            "step": self._step,
            "exchange_attempts": self.exchange_attempts,
            "exchange_successes": self.exchange_successes,
        }

    def set_state(self, state: Dict[str, any]) -> None:
        self.atoms = state["atoms"]
        self.E_old = state["energy"]
        self.n_atoms = state["n_atoms"]

    def get_outfile_header(self) -> str:
        return (
            "+-------------------------------------------------+\n"
            "| Grand Canonical Ensemble Monte Carlo Simulation |\n"
            "+-------------------------------------------------+\n\n"
        )

    def get_outfile_metadata(self) -> str:
        return (
            "Simulation Parameters:\n"
            f"  Units type: {self.units.unit_type}\n"
            f"  Temperature (K): {self._temperature}\n"
            f"  Chemical potentials: {self._mu}\n"
            "Starting simulation...\n"
            + "{:<10} {:<10} {:<15} {:<20}".format(
                "Step", "N_atoms", "Energy (eV)",
                f"Acceptance Ratios ({', '.join(self.move_selector.move_list_names)})"
            )
            + "\n" + "-" * 60 + "\n"
        )

    def format_step_output(self) -> str:
        ratios = self.count_acceptance  # or however you calculate it
        return f"{self._step:<10} {len(self._atoms):<10} {self.E_old:<15.6f} {ratios}\n"

    def write_outfile(self) -> None:
        """
        Write the step and energy to the output file.

        Args:
            step (int): The current step.
        """
        acceptance_ratios = self.move_selector.get_acceptance_ration()
        self.move_selector.reset_counters()
        try:
            with open(self._outfile, 'a') as outfile:
                outfile.write("{:<10} {:<10} {:<15.6f} {:<20}\n".format(
                    self._step,
                    self.n_atoms,
                    self.E_old,
                    ", ".join(f"{ratio * 100:.1f}%" if not np.isnan(ratio)
                              else "N/A" for ratio in acceptance_ratios)
                ))
        except IOError as e:
            self.logger.error(f"Error writing to file {self._outfile}: {e}")

    def _acceptance_condition(self,
                              potential_diff: float,
                              delta_particles: int,
                              volume: float,
                              species: str) -> bool:
        """
        Determines whether to accept a trial move based on the potential energy difference and
        temperature.

        Args:
            atoms_new (Atoms): The new configuration of atoms.
            potential_diff (float): The potential energy difference between the current and new
            configurations.
            delta_particles (int): The change in the number of particles.
            species (str): The species of the particle.

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
            db_term = self.units.de_broglie_insertion(volume, self.n_atoms, species)
            exp_term = np.exp(-self.units.beta * (potential_diff - self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                f"Lambda_db: {self.units.lambda_dbs[species]:.3e}, p: {p:.3e}, "
                f"Beta: {self.units.beta:.3e}, "
                f"Exp: {exp_term:.3e}, Exp Arg {potential_diff - self._mu[species]}, "
                f"Potential diff: {potential_diff:.3e}, "
                f"Delta_particles: {delta_particles}")

        if delta_particles == -1:  # Deletion move
            db_term = self.units.de_broglie_deletion(volume, self.n_atoms, species)
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
        return p > self.rng_acceptance.get_uniform()

    def do_gcmc_step(self) -> None:
        """
        Performs a single Grand Canonical Monte Carlo step.
        """
        for _ in range(self.move_selector.n_moves):
            atoms_new, delta_particles, species = self.move_selector.do_trial_move(self.atoms)

            if not atoms_new:  # NOTE: be careful here
                continue

            E_new = self.compute_energy(atoms_new)
            atoms_new.wrap()
            delta_E = E_new - self.E_old
            volume = self.move_selector.get_volume()
            if self._acceptance_condition(delta_E, delta_particles, volume, species):
                self.atoms = atoms_new
                self.n_atoms = len(self.atoms)
                self.E_old = E_new
                self.move_selector.acceptance_counter()
                self.calculate_cells_volume(self.atoms)

                self.logger.debug(f"Volume: {volume:.3f}, "
                                  f"Delta_particles: {delta_particles}, "
                                  f"Species: {species}")

    def initialize_run(self) -> None:
        """
        Initializes the Grand Canonical Monte Carlo simulation.
        Prepares logging and computes the initial state.
        """
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("| Grand Canonical Ensemble Monte Carlo Simulation |")
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("Simulation Parameters:")
        self.logger.info(f"Temperature (K): {self._temperature}")
        self.logger.info(f"Chemical potentials: {self._mu}")
        self.logger.info("Starting simulation...\n")
        self.logger.info("{:<10} {:<10} {:<15} {:<20}".format(
            "Step", "N_atoms", "Energy (eV)",
            f"Acceptance Ratios ({', '.join(self.move_selector.move_list_names)})"
        ))
        self.logger.info("-" * 60)
        self._initialized = True

    def run(self, steps: int) -> None:
        """
        Runs the main loop of the Grand Canonical Monte Carlo simulation.

        Args:
            steps (int): The number of Monte Carlo steps to run.
        """
        if not getattr(self, '_initialized', False):
            self.initialize_run()

        self.write_coordinates(self.atoms, self.E_old)

        for step in range(1, steps + 1):
            self._run()

    def finalize_run(self) -> None:
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

    def _run(self) -> None:
        """
        Performs a single Monte Carlo step and handles logging and writing.

        Args:
            step (int): The current step.
        """
        self.do_gcmc_step()

        if self._step % self._outfile_write_interval == 0:
            acceptance_ratios = self.move_selector.get_acceptance_ration()
            self.logger.info("{:<10} {:<10} {:<15.6f} {:<20}".format(
                self._step,
                self.n_atoms,
                self.E_old,
                ", ".join(f"{ratio * 100:.1f}%" if not np.isnan(ratio)
                          else "N/A" for ratio in acceptance_ratios)
            ))
            self.write_outfile()

        if self._step % self._trajectory_write_interval == 0:
            self.write_coordinates(self.atoms, self.E_old)
        self._step += 1
