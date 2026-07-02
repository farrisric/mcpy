from collections import Counter
import numpy as np
import logging

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from ..utils import RandomNumberGenerator
from .base_ensemble import write_xyz

logger = logging.getLogger(__name__)


class ReplicaExchange:
    def __init__(self,
                 gcmc_factory,
                 temperatures=None,
                 mus=None,
                 gcmc_steps=100,
                 exchange_interval=10,
                 outfile="replica_exchange.log",
                 write_out_interval=20,
                 seed=31,
                 global_minimum_file="global_minimum.xyz"):
        """ReplicaExchange for GCMC.

        ``gcmc_factory`` must produce per-rank unique output filenames (e.g.
        by including ``rank`` in the traj/outfile paths), otherwise all
        ranks race on the same files.
        """
        if temperatures is None and mus is None:
            raise ValueError("Provide either temperatures or mus (one per rank).")
        if temperatures is not None and mus is not None:
            # A joint (T, mu) ladder would need the full exchange criterion
            # with both the (beta2-beta1)(E2-E1) and the mu*N terms; neither
            # implemented rule covers it.
            raise ValueError("Pass temperatures OR mus, not both.")

        if MPI is None:
            raise ImportError("mpi4py is required for ReplicaExchange. Please install it.")

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

        # Select the exchange acceptance rule based on which ladder was given.
        # A mu-ladder shares one temperature, so the T-criterion would reduce to
        # delta=0 (always accept) and must not be used here.
        self._exchange_prob = self._exchange_prob_mu if mus else self._exchange_prob_T

        self._re_step = 0

        self.rng = RandomNumberGenerator(seed=seed)
        self.logger = logging.getLogger(f"{__name__}.rank{self.rank}")

        self._outfile = outfile
        self._outfile_handle = None
        self.write_out_interval = write_out_interval
        self._global_minimum_file = global_minimum_file

    def get_partner_rank(self, global_random):
        # Two alternating, symmetric pairings so that A->B implies B->A (else
        # one rank blocks forever on sendrecv). Any rank whose partner falls
        # outside [0, size) sits the round out — this covers both ends and any
        # odd-sized ensemble.
        if global_random > 0.5:
            # Odd pairing: (1,2), (3,4), ...
            partner = self.rank + 1 if self.rank % 2 else self.rank - 1
        else:
            # Even pairing: (0,1), (2,3), ...
            partner = self.rank - 1 if self.rank % 2 else self.rank + 1
        if partner < 0 or partner >= self.size:
            return None
        return partner

    def _exchange_prob_T(self, state1, state2):
        energy1 = state1['energy']
        beta1 = state1['beta']
        energy2 = state2['energy']
        beta2 = state2['beta']

        # Grand-canonical replicas (fixed mu, fluctuating N) must be compared
        # through the grand potential Phi = E - sum_s mu_s N_s, not bare energy.
        # mu rides on each replica's state (a shared-mu T-ladder leaves the
        # RE-level self.mus None), so read it from there, not from self.mus.
        mu1, mu2 = state1.get('mu'), state2.get('mu')
        if mu1 and mu2:
            for specie in mu1:
                energy1 -= mu1[specie] * state1['atoms'].symbols.count(specie)
                energy2 -= mu2[specie] * state2['atoms'].symbols.count(specie)

        delta = (beta2 - beta1) * (energy2 - energy1)
        exchange_prob = min(1.0, np.exp(delta))
        self.logger.debug(
            f"beta1: {beta1:.3f}, beta2: {beta2:.3f}, E1: {energy1:.3f}, "
            f"E2: {energy2:.3f}, Delta {delta:.3f}, "
            f"Rank: {self.rank}")
        return exchange_prob

    def _exchange_prob_mu(self, state1, state2):
        state1['n_atoms_species'] = dict()
        state2['n_atoms_species'] = dict()
        exponential_arg = 0
        deltas = {}
        for specie in state1['mu']:
            for state in (state1, state2):
                state['n_atoms_species'][specie] = state['atoms'].symbols.count(specie)

            delta_specie = state2['n_atoms_species'][specie] - state1['n_atoms_species'][specie]
            deltas[specie] = delta_specie
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
            f"Deltas: {deltas}, N1: {state1['n_atoms']}, N2: {state2['n_atoms']}, "
            f"Rank: {self.rank}"
        )
        return exchange_prob

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
        except Exception:
            self.logger.exception("Communication error with rank %s", partner_rank)
            return False

        # Both partners compute the same (symmetric) probability, but the
        # accept/reject random must be shared: the lower-rank partner draws it
        # and sends it across so both sides reach an identical decision.
        # Drawing independently would let the per-rank RNGs (which drift out of
        # sync) disagree, leaving one config duplicated and the other lost.
        exchange_prob = self._exchange_prob(rank_state, partner_state)
        try:
            if self.rank < partner_rank:
                u = self.rng.get_uniform()
                self.comm.send(u, dest=partner_rank)
            else:
                u = self.comm.recv(source=partner_rank)
        except Exception:
            self.logger.exception("Communication error with rank %s", partner_rank)
            return False

        if u < exchange_prob:
            self.logger.debug(f"Accepted exchange with rank {partner_rank}")
            self.gcmc.set_state(partner_state)
            return True
        else:
            self.logger.debug(f"Rejected exchange with rank {partner_rank}")
            return False

    def run(self):
        """Run the Parallel Tempering GCMC simulation."""
        # Per-rank GCMC initialization (opens its own outfile/traj).
        self.gcmc.initialize_run()
        if self.rank == 0:
            self.initialize_outfile()
        self.initialize_run()

        try:
            for step in range(self.gcmc_steps):
                self.gcmc._run()

                if step > 0 and step % self.exchange_interval == 0:
                    self.gcmc.exchange_attempts += 1
                    if self.do_exchange():
                        self.gcmc.exchange_successes += 1

                if step % self.write_out_interval == 0:
                    self.write_outfile(step)

                self._re_step += 1
        finally:
            self.gcmc.finalize_run()
            if self._outfile_handle is not None:
                try:
                    self._outfile_handle.close()
                except OSError:
                    self.logger.exception("Error closing %s", self._outfile)
                self._outfile_handle = None
            self._write_global_minimum()

        final_states = self.comm.gather(self.gcmc.get_state(), root=0)
        if self.rank == 0:
            self.logger.info("Final states gathered from all ranks:")
            for i, state in enumerate(final_states):
                self.logger.info(f"Rank {i}: Temperature: {state['temperature']}, "
                                 f"Energy: {state['energy']:.3f}, "
                                 f"Mu: {state.get('mu', 'N/A')}")

    def _write_global_minimum(self) -> None:
        """Gather each rank's running best, write the lowest one as a single
        XYZ frame on rank 0. No-op if disabled or no replica saw any minimum.
        """
        if self._global_minimum_file is None:
            return
        local = None
        if self.gcmc._best_atoms is not None:
            local = (self.gcmc._best_score, self.gcmc._best_energy,
                     self.gcmc._best_atoms)
        gathered = self.comm.gather(local, root=0)
        if self.rank != 0:
            return
        candidates = [(i, item) for i, item in enumerate(gathered) if item is not None]
        if not candidates:
            return
        rank, (score, energy, atoms) = min(candidates, key=lambda kv: kv[1][0])
        try:
            with open(self._global_minimum_file, 'w') as fh:
                write_xyz(atoms, energy, fh,
                          extra=f'rank={rank} score={score:.6f}')
        except OSError:
            self.logger.exception(
                "Error writing global minimum to %s", self._global_minimum_file
            )

    def initialize_run(self):
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("| Replica Exchange Monte Carlo Simulation         |")
        self.logger.info("+-------------------------------------------------+")
        self.logger.info("Simulation Parameters:")
        self.logger.info(f"Total GCMC steps: {self.gcmc_steps}")
        self.logger.info(f"Exchange interval (steps): {self.exchange_interval}")

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
        states = self.comm.gather(self.gcmc.get_state(), root=0)
        summary = {}

        if self.rank == 0:
            for i, state in enumerate(states):
                gcmc_step = state.get('step', 'N/A')
                energy = state.get('energy', 'N/A')
                temperature = state.get('temperature', 'N/A')
                atoms = state.get('atoms', None)
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

                if chemical_potential is not None:
                    chemical_potential_str = ', '.join(
                        [f"{species}: {mu:.3f}" for species, mu in chemical_potential.items()]
                    )
                else:
                    chemical_potential_str = "Not specified"

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
        """Open the RE outfile and keep the handle for reuse."""
        if self._outfile is None:
            return
        try:
            with open(self._outfile, 'w') as outfile:
                outfile.write("+-------------------------------------------------+\n")
                outfile.write("| Replica Exchange Monte Carlo Simulation         |\n")
                outfile.write("+-------------------------------------------------+\n")
                outfile.write("Simulation Parameters:\n")
                outfile.write(f"Total GCMC steps: {self.gcmc_steps}\n")
                outfile.write(f"Exchange interval (steps): {self.exchange_interval}\n")

                atom_count = len(self.gcmc.get_state()['atoms'])
                outfile.write(f"Number of atoms in initial configuration: {atom_count}\n")

                if hasattr(self, 'temperatures') and self.temperatures is not None:
                    outfile.write(f"Temperatures (K): {self.temperatures}\n")
                else:
                    outfile.write("Temperatures: Not specified (default)\n")
                if hasattr(self, 'mus') and self.mus is not None:
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
            self._outfile_handle = open(self._outfile, 'a')
        except OSError:
            self.logger.exception("Error opening output file %s", self._outfile)
            raise

    def write_outfile(self, step: int) -> None:
        """Append a summary block for ``step`` to the RE outfile."""
        summary = self.summarize_states(step)
        if self._outfile_handle is None:
            return
        try:
            for rank, state in summary.items():
                self._outfile_handle.write(
                    "{:<5} {:<10} {:<25} {:<15.6f} "
                    "{:<35} {:<20} {:<25.2f}\n".format(
                        rank, state["step"],
                        state["atom_count_by_species"],
                        state["energy"],
                        state["chemical_potential"],
                        state["temperature"],
                        state["accepted_percentage"]
                    )
                )
            self._outfile_handle.flush()
        except (OSError, AttributeError):
            self.logger.exception("Error writing summary to file %s", self._outfile)
