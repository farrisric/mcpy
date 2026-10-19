from collections import Counter
import numpy as np
import logging

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from ..utils import RandomNumberGenerator
from .base_ensemble import write_global_minimum

logger = logging.getLogger(__name__)

# One replica table, printed by the console log and the outfile of both
# replica-exchange drivers. The first column differs between them (MPI numbers
# ranks, the batched driver numbers replicas), so it is a parameter; everything
# after it is shared.
RE_COLUMNS = ('Step', 'Atom Count (by species)', 'Energy (eV)',
              'Chemical Potentials (eV)', 'Temperature (K)',
              'Accepted Exchange (%)')
_RE_TAIL_HEADER = '{:<10} {:<25} {:<15} {:<35} {:<20} {:<25}'
_RE_TAIL_ROW = '{:<10} {:<25} {:<15.6f} {:<35} {:<20} {:<25.2f}'
RE_RULE = '-' * 140


def re_header(label='Rank', width=5):
    return f'{label:<{width}} ' + _RE_TAIL_HEADER.format(*RE_COLUMNS)


def re_row(index, *values, width=5):
    return f'{index:<{width}} ' + _RE_TAIL_ROW.format(*values)


def re_banner(gcmc_steps, exchange_interval, atom_count, temperatures, mus,
              n_ranks):
    """The parameter block both drivers print, as a list of lines."""
    lines = [
        '+-------------------------------------------------+',
        '| Replica Exchange Monte Carlo Simulation         |',
        '+-------------------------------------------------+',
        'Simulation Parameters:',
        f'Total GCMC steps: {gcmc_steps}',
        f'Exchange interval (steps): {exchange_interval}',
        f'Number of atoms in initial configuration: {atom_count}',
    ]
    if temperatures is not None:
        lines.append(f'Temperatures (K): {temperatures}')
    else:
        lines.append('Temperatures: Not specified (default)')
    if mus is not None:
        lines += [f'Chemical potentials: Rank {i} - {mu}'
                  for i, mu in enumerate(mus)]
    else:
        lines.append('Chemical potentials: Not specified (default)')
    lines += [f'Number of MPI ranks: {n_ranks}', re_header(), RE_RULE]
    return lines


def mus_differ(mus):
    """True when the replicas' chemical potentials (one dict or None each)
    are not all equal. Their own-mu scores Omega are then not comparable, so
    the global minimum is written per rung. Shared by both drivers."""
    return len({tuple(sorted((mu or {}).items())) for mu in mus}) > 1


class ReplicaExchange:
    def __init__(self,
                 gcmc_factory,
                 temperatures=None,
                 mus=None,
                 gcmc_steps=100,
                 exchange_interval=10,
                 outfile='replica_exchange.log',
                 write_out_interval=20,
                 seed=31,
                 global_minimum_file='global_minimum.xyz'):
        """ReplicaExchange for GCMC.

        ``gcmc_factory`` must produce per-rank unique output filenames (e.g.
        by including ``rank`` in the traj/outfile paths), otherwise all
        ranks race on the same files.
        """
        if temperatures is None and mus is None:
            raise ValueError('Provide either temperatures or mus (one per rank).')
        if temperatures is not None and mus is not None:
            # The factory takes T or mu, not both. The swap rule itself
            # (_exchange_prob) is right for any mix of per-rank T and mu.
            raise ValueError('Pass temperatures OR mus, not both.')

        if MPI is None:
            raise ImportError('mpi4py is required for ReplicaExchange. Please install it.')

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if temperatures:
            assert len(temperatures) == self.size, 'Number of temperatures must match MPI ranks.'
            self.temperatures = temperatures
            self.gcmc = gcmc_factory(T=temperatures[self.rank], rank=self.rank)

        if mus:
            assert len(mus) == self.size, 'Number of mus must match MPI ranks.'
            self.mus = mus
            self.gcmc = gcmc_factory(mu=mus[self.rank], rank=self.rank)

        # getattr chain: CanonicalEnsemble has no ``units`` attribute (it
        # stores ``_beta`` directly) and is a supported temperature-ladder
        # ensemble here, so a units-less ensemble must pass the guard.
        if getattr(getattr(self.gcmc, 'units', None), 'molecules', None):
            raise NotImplementedError(
                'ReplicaExchange (MPI) does not support molecular species: '
                'its per-species swap bookkeeping (_exchange_prob) counts '
                'atoms by symbol via '
                '`atoms.symbols.count(specie)`, which is always 0 for a '
                "molecular name such as 'H2O'. Use BatchedReplicaExchange "
                'instead.'
            )

        self.gcmc_steps = gcmc_steps
        self.exchange_interval = exchange_interval

        self._re_step = 0

        # Per-rank stream: the lower rank of each pair draws the swap's u, so
        # identical seeds gave every pair in a round the same u. Rank 0's
        # stream (seed + 0) is the one broadcast for the pairing, unchanged.
        self.rng = RandomNumberGenerator(seed=seed + self.rank)
        self.logger = logging.getLogger(f'{__name__}.rank{self.rank}')

        self._outfile = outfile
        self._outfile_handle = None
        self.write_out_interval = write_out_interval
        self._global_minimum_file = global_minimum_file

    def get_partner_rank(self, global_random):
        # Two alternating, symmetric pairings so that A->B implies B->A (else
        # one rank blocks forever on sendrecv). Any rank whose partner falls
        # outside [0, size) sits the round out - this covers both ends and any
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

    def _exchange_prob(self, state1, state2):
        """Swap acceptance for configuration X (``state1``, slot 1) and Y
        (``state2``, slot 2), on a temperature or a mu ladder.

        Slot k weights a configuration Z by
        exp(-beta_k Phi_Z^(k)) prod_s Lambda_s(T_k)^(-3 N_Z,s), with
        Phi_Z^(k) = E_Z - sum_s mu_s^(k) N_Z,s scored with slot k's own mu.
        The swap changes the joint weight by

            ln(w'/w) = beta_1 Phi_X^(1) + beta_2 Phi_Y^(2)
                       - beta_1 Phi_Y^(1) - beta_2 Phi_X^(2)
                       + 3 sum_s (N_X,s - N_Y,s) ln(Lambda_1,s / Lambda_2,s)
                     = (beta_2 - beta_1)(E_Y - E_X)
                       + sum_s (beta_2 mu_2,s - beta_1 mu_1,s)(N_X,s - N_Y,s)
                       + the same de Broglie term.

        The code uses the second form. Only count differences enter, so
        substrate atoms cancel, and on a mu ladder (beta_1 == beta_2) the
        energies drop out exactly. With a shared mu the rule is
        (beta_2 - beta_1)(Phi_Y - Phi_X) plus the de Broglie term; with a
        shared T it is beta sum_s (mu_1,s - mu_2,s)(N_Y,s - N_X,s). The cross
        terms keep it exact when the factory sets a T-dependent mu per rank.
        A state without mu (CanonicalEnsemble) scores bare E. Same rule as
        ``BatchedReplicaExchange._accept_swap``.
        """
        beta1, beta2 = state1['beta'], state2['beta']
        mu1, mu2 = state1.get('mu') or {}, state2.get('mu') or {}
        delta = (beta2 - beta1) * (state2['energy'] - state1['energy'])
        for specie in {**mu1, **mu2}:
            dn = (state1['atoms'].symbols.count(specie)
                  - state2['atoms'].symbols.count(specie))
            delta += (beta2 * mu2.get(specie, 0.0)
                      - beta1 * mu1.get(specie, 0.0)) * dn
            # Lambda depends on T only, so the de Broglie term is zero on a
            # mu ladder (and in LJ units, where Lambda = 1).
            if dn and beta1 != beta2 and specie in mu1:
                delta += 3.0 * dn * np.log(state1['lambda_dbs'][specie]
                                           / state2['lambda_dbs'][specie])
        if not np.isfinite(delta):
            # min(1, exp(nan)) is 1: a diverged energy would always swap.
            self.logger.warning(
                'non-finite swap exponent (delta=%r); rejecting. A NaN/inf '
                'replica energy usually means a diverged relaxation.', delta)
            return 0.0
        # exp() overflows to inf for a large positive delta; short-circuit.
        exchange_prob = 1.0 if delta >= 0.0 else float(np.exp(delta))
        self.logger.debug(
            'beta1: %.3f, beta2: %.3f, E1: %.3f, E2: %.3f, mu1: %s, mu2: %s, '
            'delta: %.3f, p: %.3f, Rank: %d',
            beta1, beta2, state1['energy'], state2['energy'], mu1, mu2, delta,
            exchange_prob, self.rank)
        return exchange_prob

    def do_exchange(self):
        """Attempt an exchange with a neighboring replica. Counts the attempt
        in ``gcmc.exchange_attempts`` only when this rank has a partner.

        Communication errors propagate: swallowing one after a half-done
        exchange leaves the partner blocked forever. ``run`` aborts the job.
        """
        global_random = self.comm.bcast(self.rng.get_uniform(), root=0)
        partner_rank = self.get_partner_rank(global_random)

        if partner_rank is None:
            self.logger.debug('No valid partner for rank %d at step %d (Random: %.3f)',
                              self.rank, self._re_step, global_random)
            return False

        self.logger.debug('Step %d: Attempting exchange with partner rank %d. '
                          'Random: %.3f', self._re_step, partner_rank, global_random)
        self.gcmc.exchange_attempts += 1

        rank_state = self.gcmc.get_state()
        partner_state = self.comm.sendrecv(
            sendobj=rank_state,
            dest=partner_rank,
            source=partner_rank,
        )

        # Both partners must reach an identical decision, or one config is
        # duplicated and the other lost. So both score the pair in one order
        # (lower rank first), which makes the probability bit-identical, and
        # they share one uniform: the lower rank draws it and sends it across.
        # Per-rank RNGs drift out of sync, so independent draws would disagree.
        if self.rank < partner_rank:
            exchange_prob = self._exchange_prob(rank_state, partner_state)
            u = self.rng.get_uniform()
            self.comm.send(u, dest=partner_rank)
        else:
            exchange_prob = self._exchange_prob(partner_state, rank_state)
            u = self.comm.recv(source=partner_rank)

        if u < exchange_prob:
            self.logger.debug('Accepted exchange with rank %d', partner_rank)
            # Only the configuration travels on a swap. step and the exchange
            # tallies describe the slot, not the config; set_state restores
            # them when present (the restart round-trip), so strip them here.
            config_state = {
                k: v for k, v in partner_state.items()
                if k not in ('step', 'exchange_attempts', 'exchange_successes')
            }
            self.gcmc.set_state(config_state)
            return True
        else:
            self.logger.debug('Rejected exchange with rank %d', partner_rank)
            return False

    def run(self):
        """Run the Parallel Tempering GCMC simulation.

        An exception on any rank aborts the whole MPI job (``comm.Abort``)
        after closing this rank's files: the other ranks are blocked in a
        collective or a sendrecv with it and would otherwise hang forever.
        No collective runs on that error path.
        """
        try:
            # Per-rank GCMC initialization (opens its own outfile/traj).
            self.gcmc.initialize_run()
            if self.rank == 0:
                self.initialize_outfile()
            self.initialize_run()

            for step in range(self.gcmc_steps):
                self.gcmc._run()

                if step > 0 and step % self.exchange_interval == 0:
                    if self.do_exchange():
                        self.gcmc.exchange_successes += 1

                if step % self.write_out_interval == 0:
                    self.write_outfile(step)

                self._re_step += 1
        except BaseException:
            self.logger.exception('rank %d failed; aborting all ranks', self.rank)
            self._close_files()
            self.comm.Abort(1)
            raise
        self._close_files()
        self._write_global_minimum()

        final_states = self.comm.gather(self.gcmc.get_state(), root=0)
        if self.rank == 0:
            self.logger.info('Final states gathered from all ranks:')
            for i, state in enumerate(final_states):
                self.logger.info(f"Rank {i}: Temperature: {state['temperature']}, "
                                 f"Energy: {state['energy']:.3f}, "
                                 f"Mu: {state.get('mu', 'N/A')}")

    def _close_files(self) -> None:
        """Close this rank's files. Local only: safe on the error path."""
        self.gcmc.finalize_run()
        if self._outfile_handle is not None:
            try:
                self._outfile_handle.close()
            except OSError:
                self.logger.exception('Error closing %s', self._outfile)
            self._outfile_handle = None

    def _write_global_minimum(self) -> None:
        """Gather each rank's running best and write them on rank 0: the
        lowest-score one when all ranks share mu, one frame per rank when
        their mu differ (a mu ladder, or a factory that sets mu(T) per
        rung; see ``write_global_minimum``). Collective. No-op if disabled
        or no replica saw any minimum.
        """
        if self._global_minimum_file is None:
            return
        local = None
        if self.gcmc._best_atoms is not None:
            local = (self.gcmc._best_score, self.gcmc._best_energy,
                     self.gcmc._best_atoms)
        gathered = self.comm.gather(local, root=0)
        mus = self.comm.gather(getattr(self.gcmc, '_mu', None), root=0)
        if self.rank != 0:
            return
        per_rung = bool(getattr(self, 'mus', None)) or mus_differ(mus)
        write_global_minimum(self._global_minimum_file, gathered, per_rung,
                             self.logger)

    def initialize_run(self):
        for line in self._banner():
            self.logger.info(line)

    def _banner(self):
        return re_banner(
            self.gcmc_steps, self.exchange_interval,
            len(self.gcmc.get_state()['atoms']),
            getattr(self, 'temperatures', None), getattr(self, 'mus', None),
            self.size,
        )

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
                        [f'{species}: {count}' for species, count in atom_count.items()]
                    )
                else:
                    atom_count_by_species = 'N/A'

                if exchange_attempts > 0:
                    accepted_percentage = (exchange_successes / exchange_attempts) * 100
                else:
                    accepted_percentage = 0.0

                if chemical_potential is not None:
                    chemical_potential_str = ', '.join(
                        [f'{species}: {mu:.3f}' for species, mu in chemical_potential.items()]
                    )
                else:
                    chemical_potential_str = 'Not specified'

                summary[i] = {
                    'step': gcmc_step,
                    'atom_count_by_species': atom_count_by_species,
                    'energy': energy,
                    'chemical_potential': chemical_potential_str,
                    'temperature': temperature,
                    'accepted_percentage': accepted_percentage
                }
                self.logger.info(re_row(
                    i, gcmc_step, atom_count_by_species, energy,
                    chemical_potential_str, temperature, accepted_percentage))
        return summary

    def initialize_outfile(self):
        """Open the RE outfile and keep the handle for reuse."""
        if self._outfile is None:
            return
        try:
            with open(self._outfile, 'w') as outfile:
                outfile.write('\n'.join(self._banner()) + '\n')
            self._outfile_handle = open(self._outfile, 'a')
        except OSError:
            self.logger.exception('Error opening output file %s', self._outfile)
            raise

    def write_outfile(self, step: int) -> None:
        """Append a summary block for ``step`` to the RE outfile."""
        summary = self.summarize_states(step)
        if self._outfile_handle is None:
            return
        try:
            for rank, state in summary.items():
                self._outfile_handle.write(re_row(
                    rank, state['step'], state['atom_count_by_species'],
                    state['energy'], state['chemical_potential'],
                    state['temperature'], state['accepted_percentage']) + '\n')
            self._outfile_handle.flush()
        except (OSError, AttributeError):
            self.logger.exception('Error writing summary to file %s', self._outfile)
