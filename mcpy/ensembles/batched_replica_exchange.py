"""
BatchedReplicaExchange — single-process replica exchange GCMC with batched
energy evaluation across replicas.

Why this exists
---------------
The MPI ReplicaExchange assumes one process per replica, which on a single GPU
either oversubscribes or serializes through one CUDA context. This class holds
all replicas in one process and evaluates every replica's trial-move energy in
a single batched forward pass — one kernel launch per layer instead of N.

The MC loop itself stays sequential per replica (moves depend on the previous
accepted state). The batching is across replicas at the same logical step.

Factory requirements
--------------------
``gcmc_factory(T, rank)`` must return a fresh ``GrandCanonicalEnsemble`` with
its own ``cells``, ``move_selector`` and atoms object. Sharing cells or a
move_selector across replicas will corrupt per-replica state (cell volumes,
acceptance counters, RNG streams). See ``examples/re_gcmc_batched.py``.

The shared ``calculator`` must implement ``get_potential_energies(atoms_list)``
returning an ndarray of energies. ``AlchemiCalculator`` provides this.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np

from ..utils.gpu import empty_cuda_cache, should_empty_cache
from ..utils.random_number_generator import RandomNumberGenerator
from .base_ensemble import write_xyz

logger = logging.getLogger(__name__)


class BatchedReplicaExchange:
    def __init__(
        self,
        gcmc_factory: Callable,
        calculator,
        temperatures: Optional[List[float]] = None,
        mus: Optional[List[Dict[str, float]]] = None,
        gcmc_steps: int = 100,
        exchange_interval: int = 10,
        outfile: str = 'replica_exchange.log',
        write_out_interval: int = 20,
        seed: int = 31,
        global_minimum_file: Optional[str] = 'global_minimum.xyz',
        empty_cache_interval: int = 0,
    ) -> None:
        if temperatures is None and mus is None:
            raise ValueError("Provide either temperatures or mus (one per replica).")
        if temperatures is not None and mus is not None:
            raise ValueError("Pass temperatures OR mus, not both.")

        if temperatures is not None:
            self.temperatures = list(temperatures)
            self.mus = None
            self.n_replicas = len(self.temperatures)
            self.replicas = [
                gcmc_factory(T=T, rank=i) for i, T in enumerate(self.temperatures)
            ]
        else:
            self.mus = list(mus)
            self.temperatures = None
            self.n_replicas = len(self.mus)
            self.replicas = [
                gcmc_factory(mu=mu, rank=i) for i, mu in enumerate(self.mus)
            ]

        if not hasattr(calculator, 'get_potential_energies'):
            raise TypeError(
                "calculator must implement get_potential_energies(atoms_list). "
                "Use AlchemiCalculator or wrap your calculator."
            )
        self.calculator = calculator

        self.gcmc_steps = gcmc_steps
        self.exchange_interval = exchange_interval
        self.write_out_interval = write_out_interval
        # Opt-in periodic empty_cache() against allocator fragmentation on long
        # runs (0 = off); prefer PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
        self.empty_cache_interval = empty_cache_interval

        self._re_step = 0
        self.rng = RandomNumberGenerator(seed=seed)
        self.logger = logger

        self.exchange_attempts = [0] * self.n_replicas
        self.exchange_successes = [0] * self.n_replicas

        self._outfile = outfile
        self._outfile_handle = None
        self._global_minimum_file = global_minimum_file

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        for r in self.replicas:
            r.initialize_run()
        self._initialize_outfile()
        self._rebatch_initial_energies()

        try:
            for step in range(self.gcmc_steps):
                self._batched_gcmc_step()

                if step > 0 and step % self.exchange_interval == 0:
                    self._attempt_exchanges()

                if step % self.write_out_interval == 0:
                    self._write_outfile(step)

                if should_empty_cache(step, self.empty_cache_interval):
                    empty_cuda_cache()

                self._re_step += 1
        finally:
            for r in self.replicas:
                r.finalize_run()
            if self._outfile_handle is not None:
                try:
                    self._outfile_handle.close()
                except OSError:
                    self.logger.exception("Error closing %s", self._outfile)
                self._outfile_handle = None
            self._write_global_minimum()

    def _write_global_minimum(self) -> None:
        """Pick the lowest-score minimum across all replicas and write it as a
        single XYZ frame. No-op if disabled or no replica saw any minimum."""
        if self._global_minimum_file is None:
            return
        candidates = [r for r in self.replicas if r._best_atoms is not None]
        if not candidates:
            return
        best = min(candidates, key=lambda r: r._best_score)
        rank = self.replicas.index(best)
        try:
            with open(self._global_minimum_file, 'w') as fh:
                write_xyz(
                    best._best_atoms, best._best_energy, fh,
                    extra=f'rank={rank} score={best._best_score:.6f}',
                )
        except OSError:
            self.logger.exception(
                "Error writing global minimum to %s", self._global_minimum_file
            )

    def _rebatch_initial_energies(self) -> None:
        """Re-evaluate replica energies as one batch so all replicas start
        from energies computed by the shared calculator (and so the GPU sees
        a warm batched path before the loop)."""
        atoms_list = [r.atoms for r in self.replicas]
        energies = self.calculator.get_potential_energies(atoms_list)
        for r, E in zip(self.replicas, energies):
            r.E_old = float(E)

    # ----------------------------------------------------------- batched MC

    def _batched_gcmc_step(self) -> None:
        """
        One GCMC step across all replicas. Each replica performs
        ``move_selector.n_moves`` trial moves, matching the serial
        ``GrandCanonicalEnsemble.do_gcmc_step``. Moves are sequential within a
        replica (each depends on the previously accepted state), so a single
        sub-move is taken across all replicas at once and evaluated in one
        batched forward pass; this repeats ``n_moves`` times.

        Replicas may declare different ``n_moves``; a replica only participates
        in sub-move ``k`` while ``k < n_moves`` for that replica.
        """
        n_sub_moves = max(r.move_selector.n_moves for r in self.replicas)
        for k in range(n_sub_moves):
            active = [
                i for i, r in enumerate(self.replicas)
                if k < r.move_selector.n_moves
            ]
            self._batched_single_move(active)

        for r in self.replicas:
            r._step += 1
            if r._step % r._outfile_write_interval == 0:
                r.write_outfile()
            if r._step % r._trajectory_write_interval == 0:
                r.write_coordinates(r.atoms, r.E_old)

    def _batched_single_move(self, active: List[int]) -> None:
        """
        One trial move on each active replica:
          1. propose a trial move on each replica (in place, with snapshot)
          2. batched energy eval over the replicas with viable trials
          3. per-replica Metropolis accept/reject
        """
        snapshots: Dict[int, Dict] = {}
        trial_meta: Dict[int, tuple] = {}

        for i in active:
            r = self.replicas[i]
            snapshots[i] = {k: v.copy() for k, v in r.atoms.arrays.items()}
            result = r.move_selector.do_trial_move(r.atoms)
            atoms_new, delta_particles, species = (
                result if isinstance(result, tuple) else (result, 0, None)
            )
            if atoms_new:
                trial_meta[i] = (delta_particles, species)
            # else: move couldn't propose — atoms unchanged, snapshot harmless

        viable = [i for i in active if i in trial_meta]
        if not viable:
            return

        atoms_list = [self.replicas[i].atoms for i in viable]
        energies = self.calculator.get_potential_energies(atoms_list)
        for i, E_new in zip(viable, energies):
            r = self.replicas[i]
            delta_particles, species = trial_meta[i]
            delta_E = float(E_new) - r.E_old
            volume = r.move_selector.get_volume()
            # de Broglie particle count: total atom count before the move
            # (``r.n_atoms`` is updated only on acceptance). See
            # docs/gcmc_acceptance_convention.rst.
            if r._acceptance_condition(delta_E, delta_particles, volume, species,
                                       r.n_atoms):
                if r._wrap_on_accept:
                    r.atoms.wrap()
                r.n_atoms = len(r.atoms)
                r.E_old = float(E_new)
                r.move_selector.acceptance_counter()
                r.calculate_cells_volume(r.atoms)
                r._record_minimum(r.atoms, r.E_old)
            else:
                r.atoms.arrays = snapshots[i]

    # --------------------------------------------------------- exchange

    def _exchange_pairs(self, offset: int) -> List[tuple]:
        """Pairs (i, i+1) starting from ``offset`` ∈ {0, 1}."""
        return [(i, i + 1) for i in range(offset, self.n_replicas - 1, 2)]

    def _attempt_exchanges(self) -> None:
        offset = 0 if self.rng.get_uniform() < 0.5 else 1
        for i, j in self._exchange_pairs(offset):
            self.exchange_attempts[i] += 1
            self.exchange_attempts[j] += 1
            if self._accept_swap(i, j):
                self._swap_states(i, j)
                self.exchange_successes[i] += 1
                self.exchange_successes[j] += 1

    def _accept_swap(self, i: int, j: int) -> bool:
        """Temperature-RE Metropolis: P = exp((β_j - β_i)(Φ_j - Φ_i)).

        Replicas here are grand-canonical (fixed μ, fluctuating N), so configs
        at different temperatures must be compared through the grand potential
        Φ = E - Σ_s μ_s N_s rather than bare energy. With no chemical potential
        this reduces to the standard energy-only swap.
        """
        ri, rj = self.replicas[i], self.replicas[j]
        beta_i, beta_j = ri.units.beta, rj.units.beta
        phi_i, phi_j = self._grand_potential(ri), self._grand_potential(rj)
        delta = (beta_j - beta_i) * (phi_j - phi_i)
        p = min(1.0, float(np.exp(delta)))
        self.logger.debug(
            "swap %d<->%d: beta_i=%.3e beta_j=%.3e Phi_i=%.3f Phi_j=%.3f delta=%.3f p=%.3f",
            i, j, beta_i, beta_j, phi_i, phi_j, delta, p,
        )
        return self.rng.get_uniform() < p

    @staticmethod
    def _grand_potential(r) -> float:
        """Φ = E - Σ_s μ_s N_s for a grand-canonical replica; bare E without μ."""
        mu = getattr(r, '_mu', None)
        if not mu:
            return r.E_old
        counts = Counter(r.atoms.get_chemical_symbols())
        return r.E_old - sum(mu_s * counts[s] for s, mu_s in mu.items())

    def _swap_states(self, i: int, j: int) -> None:
        """Swap atoms + energy + n_atoms; temperatures (and β) stay pinned to
        the replica slot. Cell volume tracking gets refreshed for both."""
        ri, rj = self.replicas[i], self.replicas[j]
        ri_atoms, rj_atoms = ri.atoms, rj.atoms
        ri.atoms, rj.atoms = rj_atoms, ri_atoms
        ri.E_old, rj.E_old = rj.E_old, ri.E_old
        ri.n_atoms, rj.n_atoms = rj.n_atoms, ri.n_atoms
        ri.calculate_cells_volume(ri.atoms)
        rj.calculate_cells_volume(rj.atoms)

    # ----------------------------------------------------------- output

    def _initialize_outfile(self) -> None:
        if self._outfile is None:
            return
        try:
            with open(self._outfile, 'w') as fh:
                fh.write("+-------------------------------------------------+\n")
                fh.write("| Batched Replica Exchange (single-GPU) GCMC      |\n")
                fh.write("+-------------------------------------------------+\n")
                fh.write(f"Total GCMC steps: {self.gcmc_steps}\n")
                fh.write(f"Exchange interval (steps): {self.exchange_interval}\n")
                fh.write(f"Number of replicas: {self.n_replicas}\n")
                if self.temperatures is not None:
                    fh.write(f"Temperatures (K): {self.temperatures}\n")
                if self.mus is not None:
                    for k, mu in enumerate(self.mus):
                        fh.write(f"Chemical potentials replica {k}: {mu}\n")
                fh.write("{:<8} {:<10} {:<25} {:<15} {:<35} {:<20} {:<25}\n".format(
                    "Replica", "Step", "Atom Count (by species)", "Energy (eV)",
                    "Chemical Potentials (eV)", "Temperature (K)",
                    "Accepted Exchange (%)",
                ))
                fh.write("-" * 140 + "\n")
            self._outfile_handle = open(self._outfile, 'a')
        except OSError:
            self.logger.exception("Error opening output file %s", self._outfile)
            raise

    def _write_outfile(self, step: int) -> None:
        if self._outfile_handle is None:
            return
        try:
            for i, r in enumerate(self.replicas):
                symbols = r.atoms.get_chemical_symbols()
                count_str = ', '.join(f"{s}: {c}" for s, c in Counter(symbols).items())
                mu_str = ', '.join(f"{s}: {v:.3f}" for s, v in r._mu.items())
                attempts = self.exchange_attempts[i]
                pct = (self.exchange_successes[i] / attempts * 100) if attempts else 0.0
                self._outfile_handle.write(
                    "{:<8} {:<10} {:<25} {:<15.6f} {:<35} {:<20} {:<25.2f}\n".format(
                        i, r._step, count_str, r.E_old, mu_str,
                        r._temperature, pct,
                    )
                )
            self._outfile_handle.flush()
        except (OSError, AttributeError):
            self.logger.exception("Error writing to %s", self._outfile)
