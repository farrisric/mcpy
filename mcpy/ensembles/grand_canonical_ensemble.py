import logging
from typing import Optional, List, Dict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .base_ensemble import BaseEnsemble
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
from ..moves.move_selector import MoveSelector
from ..cell import Cell

logger = logging.getLogger(__name__)

_RATIO_COL_PER_MOVE = 8  # width per "100.0%, " entry, generous


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
                 traj_file: Optional[str] = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 trajectory_write_interval: Optional[int] = 1,
                 outfile: Optional[str] = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: Optional[int] = 1) -> None:

        super().__init__(atoms=atoms,
                         cells=cells,
                         units_type='metal',
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         traj_mode=traj_mode,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_mode=outfile_mode,
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
        self._last_logged_step = None
        self.exchange_attempts = 0
        self.exchange_successes = 0
        self.rng_acceptance = RandomNumberGenerator(seed=self._random_seed + 2)
        # PBC is fixed for the run; precompute so we can skip ``wrap()`` for
        # cluster systems and on rejected trials.
        self._wrap_on_accept = bool(np.any(np.asarray(self.atoms.pbc)))

        # Width of the acceptance-ratios column in the outfile, sized to the
        # number of moves so >4 moves don't shred the alignment.
        self._ratio_col_width = max(20, _RATIO_COL_PER_MOVE * len(move_selector.move_list))
        self._table_width = 10 + 1 + 10 + 1 + 15 + 1 + self._ratio_col_width

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
        # Restore optional bookkeeping if present (used by restart, not by RE).
        if "step" in state:
            self._step = state["step"]
        if "exchange_attempts" in state:
            self.exchange_attempts = state["exchange_attempts"]
        if "exchange_successes" in state:
            self.exchange_successes = state["exchange_successes"]

    def get_outfile_header(self) -> str:
        return (
            "+-------------------------------------------------+\n"
            "| Grand Canonical Ensemble Monte Carlo Simulation |\n"
            "+-------------------------------------------------+\n\n"
        )

    def get_outfile_metadata(self) -> str:
        ratio_label = f"Acceptance Ratios ({', '.join(self.move_selector.move_list_names)})"
        header_row = "{:<10} {:<10} {:<15} {:<{w}}".format(
            "Step", "N_atoms", "Energy (eV)", ratio_label,
            w=self._ratio_col_width,
        )
        return (
            "Simulation Parameters:\n"
            f"  Units type: {self.units.unit_type}\n"
            f"  Temperature (K): {self._temperature}\n"
            f"  Chemical potentials: {self._mu}\n"
            "  Acceptance ratios are per write-interval (cumulative totals "
            "in the finalize_run log).\n"
            "Starting simulation...\n"
            + header_row
            + "\n" + "-" * self._table_width + "\n"
        )

    def write_outfile(self) -> None:
        """Write one row: step, N, energy, per-interval acceptance ratios."""
        if self._outfile is None or self._outfile_handle is None:
            return
        if self._last_logged_step == self._step:
            return  # already wrote this step (e.g. finalize_run after a triggered write)
        acceptance_ratios = self.move_selector.interval_ratios()
        self.move_selector.reset_counters()
        ratio_str = ", ".join(
            f"{r * 100:.1f}%" if not np.isnan(r) else "N/A"
            for r in acceptance_ratios
        )
        try:
            self._outfile_handle.write("{:<10} {:<10} {:<15.6f} {:<{w}}\n".format(
                self._step,
                self.n_atoms,
                self.E_old,
                ratio_str,
                w=self._ratio_col_width,
            ))
            self._outfile_handle.flush()
            self._last_logged_step = self._step
        except (OSError, AttributeError):
            self.logger.exception("Error writing to file %s", self._outfile)

    def _acceptance_condition(self,
                              potential_diff: float,
                              delta_particles: int,
                              volume: float,
                              species: str) -> bool:
        """Metropolis / de-Broglie acceptance test for displacement, insertion,
        and deletion moves."""
        if delta_particles == 0:
            if potential_diff <= 0:
                return True
            p = np.exp(-potential_diff * self.units.beta)
            return p > self.rng_acceptance.get_uniform()

        if delta_particles == 1:  # insertion
            db_term = self.units.de_broglie_insertion(volume, self.n_atoms, species)
            exp_term = np.exp(-self.units.beta * (potential_diff - self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                "Lambda_db: %.3e, p: %.3e, Beta: %.3e, Exp: %.3e, "
                "Exp Arg %s, Potential diff: %.3e, Delta_particles: %d",
                self.units.lambda_dbs[species], p, self.units.beta, exp_term,
                potential_diff - self._mu[species], potential_diff, delta_particles)
        elif delta_particles == -1:  # deletion
            db_term = self.units.de_broglie_deletion(volume, self.n_atoms, species)
            exp_term = np.exp(-self.units.beta * (potential_diff + self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                "Lambda_db: %.3e, p: %.3e, Beta: %.3e, Exp: %.3e, "
                "Exp Arg %s, Potential diff: %.3e, Delta_particles: %d",
                self.units.lambda_dbs[species], p, self.units.beta, exp_term,
                potential_diff + self._mu[species], potential_diff, delta_particles)
        else:
            raise ValueError(
                f"unexpected delta_particles={delta_particles} from move "
                f"'{self.move_selector.get_name()}'"
            )

        if p > 1:
            return True
        return p > self.rng_acceptance.get_uniform()

    def do_gcmc_step(self) -> None:
        atoms = self.atoms
        for _ in range(self.move_selector.n_moves):
            # Snapshot arrays (positions, numbers, ...) before the trial move.
            # On rejection we restore from this snapshot, which also undoes
            # any in-place relaxation done by ``compute_energy``. This
            # replaces the previous ``atoms.copy()`` per trial in the moves.
            saved_arrays = {k: v.copy() for k, v in atoms.arrays.items()}

            atoms_new, delta_particles, species = self.move_selector.do_trial_move(atoms)

            if not atoms_new:
                # Move couldn't be proposed (e.g. empty cell). The move did
                # not mutate ``atoms``; MoveSelector already recorded the
                # failure so it won't depress the acceptance ratio.
                continue

            E_new = self.compute_energy(atoms)
            delta_E = E_new - self.E_old
            volume = self.move_selector.get_volume()
            if self._acceptance_condition(delta_E, delta_particles, volume, species):
                if self._wrap_on_accept:
                    atoms.wrap()
                self.n_atoms = len(atoms)
                self.E_old = E_new
                self.move_selector.acceptance_counter()
                self.calculate_cells_volume(atoms)

                self.logger.debug("Volume: %.3f, Delta_particles: %d, Species: %s",
                                  volume, delta_particles, species)
            else:
                atoms.arrays = saved_arrays

    def initialize_run(self) -> None:
        """Open files, write headers, log start, write initial state."""
        if self._initialized:
            return
        super().initialize_run()
        self.initialize_outfile()
        self.logger.info(
            "GCMC starting: T=%s K, mu=%s, outfile=%s",
            self._temperature, self._mu, self._outfile,
        )
        # Initial frame (step 0) for both outfile and trajectory so the
        # logged sequence is symmetric and the starting state is recoverable.
        self._write_initial_row()
        self.write_coordinates(self.atoms, self.E_old)

    def _write_initial_row(self) -> None:
        if self._outfile is None or self._outfile_handle is None:
            return
        try:
            placeholder = ", ".join("N/A" for _ in self.move_selector.move_list)
            self._outfile_handle.write("{:<10} {:<10} {:<15.6f} {:<{w}}\n".format(
                0, self.n_atoms, self.E_old, placeholder, w=self._ratio_col_width,
            ))
            self._outfile_handle.flush()
            self._last_logged_step = 0
        except (OSError, AttributeError):
            self.logger.exception("Error writing initial row to %s", self._outfile)

    def finalize_run(self) -> None:
        if not self._initialized:
            return
        # Write the trailing row + frame so the log/traj end on the actual
        # final state regardless of write_interval alignment.
        self.write_outfile()
        self.write_coordinates(self.atoms, self.E_old)
        self.logger.info(
            "GCMC complete: steps=%d, final_energy=%.6f eV",
            self._step, self.E_old,
        )
        super().finalize_run()

    def _run(self) -> None:
        """One GCMC step plus interval-based writes. Step counter is
        incremented BEFORE the modulo check so the logged step is the number
        of GCMC steps completed."""
        self.do_gcmc_step()
        self._step += 1

        if self._step % self._outfile_write_interval == 0:
            self.write_outfile()
            self.logger.debug("step=%d N=%d E=%.6f",
                              self._step, self.n_atoms, self.E_old)

        if self._step % self._trajectory_write_interval == 0:
            self.write_coordinates(self.atoms, self.E_old)
