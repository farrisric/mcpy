from mpi4py import MPI
import numpy as np
import logging
from ..utils import RandomNumberGenerator
from collections import Counter
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class ReplicaExchange:
    def __init__(self,
                 gcmc_factory,
                 temperatures=None,
                 mus=None,
                 gcmc_steps=100,
                 exchange_interval=10,
                 outfile="replica_exchange.log",
                 write_out_interval=20,
                 seed=31):
        """
        ReplicaExchange for GCMC.

        Parameters:
        - gcmc_factory (function): Function to create a GCMC instance for a given temperature.
        - temperatures (list): List of gcmc_properties for each replica.
        - n_steps (int): Total number of GCMC steps.
        - exchange_interval (int): Steps between exchange attempts.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if temperatures:
            assert len(temperatures) == self.size, "Number of temperatures must match MPI ranks."
            self.temperatures = temperatures
            self.gcmc = gcmc_factory(T=temperatures[self.rank], rank=self.rank)

        if mus:
            assert len(mus) == self.size, "Number of mus must match MPI ranks."
            self.mus = mus
            self.gcmc = gcmc_factory(mu=mus[self.rank], rank=self.rank)

        self.gcmc_steps = gcmc_steps
        self.exchange_interval = exchange_interval

        self._re_step = 0

        gcmc_logger = logging.getLogger(self.gcmc.__class__.__name__)
        gcmc_logger.setLevel(logging.WARNING)
        self.rng = RandomNumberGenerator(seed=seed)

        logging.basicConfig(
            level=logging.DEBUG,
            format=f"%(asctime)s [Rank {self.rank}] %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(f"replica_exchange_rank_{self.rank}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

        self._outfile = outfile
        self.write_out_interval = write_out_interval

    def get_partner_rank(self, global_random):
        if global_random > 0.5:
            if self.rank == 0 or self.rank == self.size - 1:
                return None  # No valid partner for edge ranks in this case
            return self.rank - 1 if self.rank % 2 == 0 else self.rank + 1
        else:
            return self.rank + 1 if self.rank % 2 == 0 else self.rank - 1

    def _acceptance_condition_T(self, state1, state2):
        """
        Determines whether to accept a replica exchange between two replicas.

        Args:
            state1 (dict): State of the first replica, including its energy.
            temp1 (float): Temperature of the first replica.
            state2 (dict): State of the second replica, including its energy.
            temp2 (float): Temperature of the second replica.

        Returns:
            bool: True if the exchange is accepted, False otherwise.
        """
        energy1 = state1['energy']
        beta1 = state1['beta']

        energy2 = state2['energy']
        beta2 = state2['beta']

        delta = (beta2 - beta1) * (energy2 - energy1)
        exchange_prob = min(1.0, np.exp(delta))
        self.logger.debug(
            f"beta1: {beta1:.3f}, beta2: {beta2:.3f}, E1: {energy1:.3f}, "
            f"E2: {energy2:.3f}, Delta {delta:.3f}, "
            f"Rank: {self.rank}")
        return self.rng.get_uniform() < exchange_prob

    def _acceptance_condition_mu(self, state1, state2):
        state1['n_atoms_species'] = dict()
        state2['n_atoms_species'] = dict()
        exponential_arg = 0
        for specie in self.gcmc.species:
            for i, state in enumerate([state1, state2]):
                state['n_atoms_species'][specie] = state['atoms'].symbols.count(specie)

            delta_specie = state2['n_atoms_species'][specie] - state1['n_atoms_species'][specie]
            exponential_arg += (
                state2['beta'] * state2['mu'][specie] * (-delta_specie)
                ) + (
                    state1['beta'] * state1['mu'][specie] * delta_specie
                    )

        exponential = np.exp(exponential_arg)
        exchange_prob = min(1.0, exponential)

        self.logger.info(
            f"Energy1: {state1['energy']:.3f}, Energy2: {state2['energy']:.3f}, "
            f"Exponential Arg: {exponential_arg:.3f}, "
            f"Exponential: {exponential:.3f}, "
            f"Exchange Prob: {exchange_prob:.3f}, "
            f"Delta N: {delta_specie}, N1: {state1['n_atoms']}, N2: {state2['n_atoms']}, "
            f"Rank: {self.rank}"
        )
        return self.rng.get_uniform() < exchange_prob

    def do_exchange(self):
        """Attempt an exchange with a neighboring replica."""
        global_random = self.comm.bcast(self.rng.get_uniform(), root=0)
        partner_rank = self.get_partner_rank(global_random)

        if partner_rank is None:
            self.logger.debug(f"No valid partner for rank {self.rank} at step {self._re_step} "
                              f"(Random: {global_random:.3f})")
            return False

        self.logger.debug(f"Step {self._re_step}: Attempting exchange "
                          f"with partner rank {partner_rank}. "
                          f"Random: {global_random:.3f}")

        rank_state = self.gcmc.get_state()

        try:
            partner_state = self.comm.sendrecv(
                sendobj=rank_state,
                dest=partner_rank,
                source=partner_rank,
            )
        except Exception as e:
            self.logger.error(f"Communication error with rank {partner_rank}: {e}")
            return

        if self._acceptance_condition_mu(rank_state, partner_state):
            self.logger.debug(f"Accepted exchange with rank {partner_rank}")
            self.gcmc.set_state(partner_state)
            return True
        else:
            self.logger.debug(f"Rejected exchange with rank {partner_rank}")
            return False

    def run(self):
        """Run the Parallel Tempering GCMC simulation."""
        if self.rank == 0:
            self.initialize_outfile()
            self.initialize_run()

        for step in range(self.gcmc_steps):
            self.gcmc._run(1)

            if step > 0 and step % self.exchange_interval == 0:
                self.gcmc.exchange_attempts += 1
                if self.do_exchange():
                    self.gcmc.exchange_successes += 1

            if step % self.write_out_interval == 0:
                # self.summarize_states(step)
                self.write_outfile(step)

            self._re_step += 1

        self.gcmc.finalize_run()

        final_states = self.comm.gather(self.gcmc.get_state(), root=0)
        if self.rank == 0:
            self.logger.info("Final states gathered from all ranks:")
            for i, state in enumerate(final_states):
                self.logger.info(f"Rank {i}: Temperature: {state['temperature']}, "
                                 f"Energy: {state['energy']:.3f}, "
                                 f"Mu: {state['mu']}")

    def initialize_run(self):
        """
        Initializes the Replica Exchange Monte Carlo simulation.
        Prepares logging and prints the simulation parameters.
        """
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("| Replica Exchange Monte Carlo Simulation         |")
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("Simulation Parameters:")
        self.logger.info(f"Total GCMC steps: {self.gcmc_steps}")
        self.logger.info(f"Exchange interval (steps): {self.exchange_interval}")

        # Log atom count if available in GCMC state
        atom_count = len(self.gcmc.get_state()['atoms'])
        self.logger.info(f"Number of atoms in initial configuration: {atom_count}")

        if hasattr(self, 'temperatures') and self.temperatures is not None:
            self.logger.info(f"Temperatures (K): {self.temperatures}")
        else:
            self.logger.info("Temperatures: Not specified (default)")
        if hasattr(self, 'mus') and self.mus is not None:
            for i, mus in enumerate(self.mus):
                self.logger.info(f"Chemical potentials: Rank {i} - {mus}")
        else:
            self.logger.info("Chemical potentials: Not specified (default)")
        self.logger.info(f"Number of MPI ranks: {self.size}")
        self.logger.info("{:<5} {:<10} {:<25} {:<15} {:<35} {:<20} {:<25}".format(
            "Rank", "Step", "Atom Count (by species)", "Energy (eV)", "Chemical Potentials (eV)",
            "Temperature (K)", "Accepted Exchange (%)"
        ))
        self.logger.info("-" * 140)

    def summarize_states(self, step):
        """
        Gathers and summarizes the state of all GCMC instances across ranks.

        Args:
            step (int): Current simulation step.

        Returns:
            dict: A dictionary containing summarized state information for all ranks.
        """
        states = self.comm.gather(self.gcmc.get_state(), root=0)
        summary = {}

        if self.rank == 0:
            for i, state in enumerate(states):
                gcmc_step = state.get('step', 'N/A')
                energy = state.get('energy', 'N/A')
                temperature = state.get('temperature', 'N/A')
                atoms = state.get('atoms', None)  # Atoms object from ase
                exchange_attempts = state.get('exchange_attempts', 0)
                exchange_successes = state.get('exchange_successes', 0)
                chemical_potential = state.get('mu', None)

                if atoms is not None:
                    symbols = atoms.get_chemical_symbols()
                    atom_count = Counter(symbols)
                    atom_count_by_species = ', '.join(
                        [f"{species}: {count}" for species, count in atom_count.items()]
                    )
                else:
                    atom_count_by_species = "N/A"

                if exchange_attempts > 0:
                    accepted_percentage = (exchange_successes / exchange_attempts) * 100
                else:
                    accepted_percentage = 0.0

                # Format chemical potential
                if chemical_potential is not None:
                    chemical_potential_str = ', '.join(
                        [f"{species}: {mu:.3f}" for species, mu in chemical_potential.items()]
                    )
                else:
                    chemical_potential_str = "Not specified"

                # Store the summarized state information in the dictionary
                summary[i] = {
                    "step": gcmc_step,
                    "atom_count_by_species": atom_count_by_species,
                    "energy": energy,
                    "chemical_potential": chemical_potential_str,
                    "temperature": temperature,
                    "accepted_percentage": accepted_percentage
                }
                self.logger.info("{:<5} {:<10} {:<25} {:<15.6f} {:<35} {:<20} {:<25.2f}".format(
                    i, gcmc_step, atom_count_by_species, energy, chemical_potential_str,
                    temperature, accepted_percentage
                ))
        return summary
    
    def initialize_outfile(self):
        """
        Initialize the output file for writing the summarized states.
        """
        try:
            with open(self._outfile, 'w') as outfile:
                outfile.write("+-------------------------------------------------+\n")
                outfile.write("| Replica Exchange Monte Carlo Simulation         |\n")
                outfile.write("+-------------------------------------------------+\n")
                outfile.write("Simulation Parameters:\n")
                outfile.write(f"Total GCMC steps: {self.gcmc_steps}\n")
                outfile.write(f"Exchange interval (steps): {self.exchange_interval}\n")

                # Log atom count if available in GCMC state
                atom_count = len(self.gcmc.get_state()['atoms'])
                outfile.write(f"Number of atoms in initial configuration: {atom_count}\n")

                if hasattr(self, 'temperatures') and self.temperatures is not None:
                    outfile.write(f"Temperatures (K): {self.temperatures}\n")
                else:
                    outfile.write("Temperatures: Not specified (default)\n")
                for i, mus in enumerate(self.mus):
                    outfile.write(f"Chemical potentials: Rank {i} - {mus}\n")
                else:
                    outfile.write("Chemical potentials: Not specified (default)\n")
                outfile.write(f"Number of MPI ranks: {self.size}\n")
                outfile.write("{:<5} {:<10} {:<25} {:<15} {:<35} {:<20} {:<25}\n".format(
                    "Rank", "Step", "Atom Count (by species)", "Energy (eV)",
                    "Chemical Potentials (eV)", "Temperature (K)", "Accepted Exchange (%)"
                ))
                outfile.write("-" * 140 + "\n")
        except IOError as e:
            self.logger.error(f"Error opening output file: {e}")

    def write_outfile(self, step: int) -> None:
        """
        Write the summarized states to the output file.

        Args:
            step (int): The current step.
        """
        summary = self.summarize_states(step)
        try:
            with open(self._outfile, 'a') as outfile:
                for rank, state in summary.items():
                    outfile.write("{:<5} {:<10} {:<25} {:<15.6f} "
                                  "{:<35} {:<20} {:<25.2f}\n".format(
                                            rank, state["step"],
                                            state["atom_count_by_species"],
                                            state["energy"],
                                            state["chemical_potential"],
                                            state["temperature"],
                                            state["accepted_percentage"]
                                            ))
        except IOError as e:
            self.logger.error(f"Error writing summary to file: {e}")
