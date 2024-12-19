from mpi4py import MPI
import numpy as np
import logging
from npl.utils import RandomNumberGenerator


BOLTZMANN_CONSTANT_eV_K = 8.617333262e-5


class ReplicaExchange:
    def __init__(self,
                 gcmc_factory,
                 gcmc_properties,
                 gcmc_steps=100,
                 exchange_interval=10,
                 seed=31):
        """
        Parallel Tempering for GCMC.

        Parameters:
        - gcmc_factory (function): Function to create a GCMC instance for a given temperature.
        - gcmc_properties (list): List of gcmc_properties for each replica.
        - n_steps (int): Total number of GCMC steps.
        - exchange_interval (int): Steps between exchange attempts.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        assert len(gcmc_properties) == self.size, "Number of temperatures must match MPI ranks."
        self.gcmc_properties = gcmc_properties
        self.gcmc_steps = gcmc_steps
        self.exchange_interval = exchange_interval

        self.gcmc = gcmc_factory(gcmc_properties[self.rank])
        self.rng = RandomNumberGenerator(seed=seed)

        logging.basicConfig(level=logging.INFO, format=f"Rank {self.rank}: %(message)s")
        self.logger = logging.getLogger()

    def _acceptance_condition(self, state1, beta1, state2, beta2):
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
        energy2 = state2['energy']

        delta = (beta2 - beta1) * (energy2 - energy1)
        exchange_prob = min(1.0, np.exp(delta))
        return self.rng.get_uniform() < exchange_prob

    def _acceptance_criterion_(self, mu_i, mu_j, energy_i, energy_j, particle_counts, temperature):
        """
        Calculate the acceptance probability for exchanging replicas with the same temperature
        but different chemical potentials.

        Parameters:
            mu_i (dict): Chemical potentials for replica i (e.g., {'Ag': -4.0, 'O': -2.5}).
            mu_j (dict): Chemical potentials for replica j.
            energy_i (float): Energy of replica i.
            energy_j (float): Energy of replica j.
            particle_counts (dict): Number of particles for each species in the replica
                                    (e.g., {'Ag': 100, 'O': 25}).
            temperature (float): Temperature of the replicas (K).

        Returns:
            float: The acceptance probability.
        """
        beta = 1 / (BOLTZMANN_CONSTANT_eV_K * temperature)
        delta_energy = energy_i - energy_j
        delta_mu = 0.0
        for species, count in particle_counts.items():
            delta_mu += count * (mu_j[species] - mu_i[species])
        delta = beta * (delta_energy + delta_mu)
        return min(1.0, np.exp(-delta))

    def do_exchange(self):
        """Attempt an exchange with a neighboring replica."""
        if self.rank % 2 == 0:
            partner_rank = self.rank + 1
        else:
            partner_rank = self.rank - 1

        if partner_rank < 0 or partner_rank >= self.size:
            return

        rank_state = self.gcmc.get_state()
        rank_temperature = self.temperatures[self.rank]

        partner_state, partner_temp = self.comm.sendrecv(
            sendobj=(rank_state, rank_temperature),
            dest=partner_rank,
            source=partner_rank,
        )
        rank_beta = 1/(rank_temperature*BOLTZMANN_CONSTANT_eV_K)
        partner_beta = 1/(partner_temp*BOLTZMANN_CONSTANT_eV_K)
        if self._acceptance_condition(rank_state, rank_beta, partner_state, partner_beta):
            self.logger.info(f"Accepted exchange with rank {partner_rank}")
            self.gcmc.set_state(partner_state)
        else:
            self.logger.info(f"Rejected exchange with rank {partner_rank}")

    def run(self):
        """Run the Parallel Tempering GCMC simulation."""
        for step in range(self.gcmc_steps):
            self.gcmc.run(1)

            if step > 0 and step % self.exchange_interval == 0:
                self.do_exchange()

        self.gcmc.finalize_run()

        final_states = self.comm.gather(self.gcmc.get_state(), root=0)
        if self.rank == 0:
            self.logger.info(f"Final states: {final_states}")
