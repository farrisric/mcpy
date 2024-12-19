from mpi4py import MPI
import numpy as np
import logging
from ..utils import RandomNumberGenerator


BOLTZMANN_CONSTANT_eV_K = 8.617333262e-5


class ReplicaExchange:
    def __init__(self,
                 gcmc_factory,
                 temperatures,
                 mus=None,
                 gcmc_steps=100,
                 exchange_interval=10,
                 seed=31):
        """
        Parallel Tempering for GCMC.

        Parameters:
        - gcmc_factory (function): Function to create a GCMC instance for a given temperature.
        - temperatures (list): List of gcmc_properties for each replica.
        - n_steps (int): Total number of GCMC steps.
        - exchange_interval (int): Steps between exchange attempts.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        assert len(temperatures) == self.size, "Number of temperatures must match MPI ranks."
        self.temperatures = temperatures

        if mus:
            assert len(mus) == self.size, "Number of mus must match MPI ranks."
            self.mus = mus

        self.gcmc_steps = gcmc_steps
        self.exchange_interval = exchange_interval

        self.gcmc = gcmc_factory(temperatures[self.rank])
        self.rng = RandomNumberGenerator(seed=seed)

        logging.basicConfig(level=logging.INFO, format=f"Rank {self.rank}: %(message)s")
        self.logger = logging.getLogger()

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
        self.logger.info(
            f"beta1: {beta1:.3f}, beta2: {beta2:.3f}, E1: {energy1:.3f}, "
            f"E2: {energy2:.3f}, Delta {delta:.3f}, "
            f"Rank: {self.rank}")
        return self.rng.get_uniform() < exchange_prob

    def _acceptance_condition_mu(self, state1, state2):
        """
        Determines whether to accept a replica exchange between two replicas
        with the same temperature but different chemical potentials.

        Args:
            state1 (dict): State of the first replica, including its energy, mu, and beta.
            state2 (dict): State of the second replica, including its energy, mu, and beta.

        Returns:
            bool: True if the exchange is accepted, False otherwise.
        """
        energy1 = state1['energy']
        energy2 = state2['energy']
        beta = state1['beta'] 
        delta_energy = energy2 - energy1

        species1 = state1['atoms'].get_chemical_symbols()
        species2 = state2['atoms'].get_chemical_symbols()
        delta_mu = 0.0

        for species in state1['mu']:
            count1 = species1.count(species)  # Count occurrences of species in state1
            count2 = species2.count(species)  # Count occurrences of species in state2
            mu_diff = state2['mu'][species] - state1['mu'][species]
            delta_mu += (count2 - count1) * mu_diff

        delta = beta * (delta_energy + delta_mu)
        exchange_prob = min(1.0, np.exp(-delta))

        self.logger.info(
            f"Energy1: {energy1:.3f}, Energy2: {energy2:.3f}, Delta Energy: {delta_energy:.3f}, "
            f"Delta Mu: {delta_mu:.3f}, Delta: {delta:.3f}, Exchange Prob: {exchange_prob:.3f}, "
            f"Rank: {self.rank}"
        )

        return self.rng.get_uniform() < exchange_prob

    def do_exchange(self):
        """Attempt an exchange with a neighboring replica."""
        global_random = self.comm.bcast(self.rng.get_uniform(), root=0)
        partner_rank = self.get_partner_rank(global_random)

        if partner_rank is None:
            self.logger.info(f"No valid partner for rank {self.rank} "
                             f"with random {global_random:.3f}")
            return

        self.logger.info(f"Global random: {global_random:.3f}, "
                         f"Rank: {self.rank}, Partner rank: {partner_rank}")

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
            self.logger.info("Final states from all ranks:")
            for i, state in enumerate(final_states):
                self.logger.info(f"Rank {i}: {state}")
