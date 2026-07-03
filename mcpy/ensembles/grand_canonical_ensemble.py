import logging
import time
from typing import Optional, List, Dict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .base_ensemble import BaseEnsemble
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
from ..moves.move_selector import MoveSelector
from ..moves.molecule_utils import find_molecules
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
                 molecules: Optional[Dict[str, Atoms]] = None,
                 random_seed: Optional[int] = None,
                 traj_file: Optional[str] = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 trajectory_write_interval: Optional[int] = 1,
                 outfile: Optional[str] = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: Optional[int] = 1,
                 minima_file: Optional[str] = None,
                 minima_mode: str = 'a') -> None:

        super().__init__(atoms=atoms,
                         cells=cells,
                         units_type=units_type,
                         calculator=calculator,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         traj_mode=traj_mode,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_mode=outfile_mode,
                         outfile_write_interval=outfile_write_interval,
                         minima_file=minima_file,
                         minima_mode=minima_mode)

        self.E_old = self.compute_energy(self.atoms)

        self.units = SetUnits(units_type,
                              temperature=temperature,
                              species=species,
                              molecules=molecules)

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
        # The configuration changed under the cells: refresh their free
        # volumes now, or the next insertion/deletion acceptance would use
        # the previous configuration's volume (GCMC only recalculates on
        # accepted moves).
        self.calculate_cells_volume(self.atoms)

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

    def write_outfile(self, step: int = None, energy: float = None) -> None:
        """Write one row: step, N, energy, per-interval acceptance ratios.

        ``step``/``energy`` are accepted for compatibility with the
        :class:`BaseEnsemble` signature but ignored — GCMC always logs its
        own ``_step``/``E_old`` so the row matches the sampler state.
        """
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

    def _minimum_score(self, atoms: Atoms, energy: float) -> float:
        """Grand potential Ω = E − Σ μ_i N_i. Comparing raw E across moves
        that change N is not meaningful in the grand canonical ensemble.
        Molecular species count molecules, atomic species count atoms."""
        score = energy
        symbols = atoms.get_chemical_symbols()
        for specie, mu in self._mu.items():
            if specie in self.units.molecules:
                template = self.units.molecules[specie]
                n = len(find_molecules(
                    atoms, sorted(template.get_chemical_symbols())))
            else:
                n = symbols.count(specie)
            score -= mu * n
        return score

    def _acceptance_condition(self,
                              potential_diff: float,
                              delta_particles: int,
                              volume: float,
                              species: str,
                              n_atoms: int = None) -> bool:
        """Metropolis / de-Broglie acceptance test for displacement, insertion,
        and deletion moves. ``n_atoms`` is the particle count fed to the de
        Broglie combinatorial factor. Atomic moves pass the pre-move total
        atom count (the original convention, kept for consistency with the
        group's published runs); molecule moves pass their pre-move in-cell
        molecule count via ``MoveSelector.get_exchange_count()`` -- see
        docs/gcmc_acceptance_convention.rst."""
        if delta_particles == 0:
            if potential_diff <= 0:
                return True
            p = np.exp(-potential_diff * self.units.beta)
            return p > self.rng_acceptance.get_uniform()

        if delta_particles == 1:  # insertion
            db_term = self.units.de_broglie_insertion(volume, n_atoms, species)
            exp_term = np.exp(-self.units.beta * (potential_diff - self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                "Lambda_db: %.3e, p: %.3e, Beta: %.3e, Exp: %.3e, "
                "Exp Arg %s, Potential diff: %.3e, Delta_particles: %d",
                self.units.lambda_dbs[species], p, self.units.beta, exp_term,
                potential_diff - self._mu[species], potential_diff, delta_particles)
        elif delta_particles == -1:  # deletion
            db_term = self.units.de_broglie_deletion(volume, n_atoms, species)
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
            # Constraints too: ``del atoms[i]`` remaps FixAtoms indices in
            # place (ASE ``delete_atoms``), so a rejected deletion would
            # otherwise leave the restored configuration with shifted fixed
            # indices — wrong atoms frozen from then on.
            saved_arrays = {k: v.copy() for k, v in atoms.arrays.items()}
            saved_constraints = [c.copy() for c in atoms.constraints]

            atoms_new, delta_particles, species = self.move_selector.do_trial_move(atoms)

            if atoms_new is False or atoms_new is None:
                # Move couldn't be proposed (e.g. empty cell). The move did
                # not mutate ``atoms``; MoveSelector already recorded the
                # failure so it won't depress the acceptance ratio. Identity
                # check, not truthiness: an empty Atoms (last atom deleted)
                # is falsy but is a real proposal that must be scored.
                continue

            if atoms_new is not atoms:
                raise RuntimeError(
                    f"move '{self.move_selector.get_name()}' returned a "
                    "different Atoms object; GCMC moves must mutate the "
                    "passed atoms in place (copy-based moves are for "
                    "CanonicalEnsemble only)"
                )

            E_new = self.compute_energy(atoms)
            delta_E = E_new - self.E_old
            volume = self.move_selector.get_volume()
            # de Broglie particle count. Molecule moves report their in-cell
            # molecule count via ``get_exchange_count`` (textbook convention);
            # atomic moves return None and fall back to the total atom count
            # before the move (``self.n_atoms`` is updated only on acceptance).
            # See docs/gcmc_acceptance_convention.rst.
            n_exchange = self.move_selector.get_exchange_count()
            if n_exchange is None:
                n_exchange = self.n_atoms
            if self._acceptance_condition(delta_E, delta_particles, volume,
                                          species, n_exchange):
                if self._wrap_on_accept:
                    atoms.wrap()
                self.n_atoms = len(atoms)
                self.E_old = E_new
                self.move_selector.acceptance_counter()
                self.calculate_cells_volume(atoms)
                self._record_minimum(atoms, self.E_old)

                self.logger.debug("Volume: %.3f, Delta_particles: %d, Species: %s",
                                  volume, delta_particles, species)
            else:
                atoms.arrays = saved_arrays
                atoms.set_constraint(saved_constraints)

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
        self._record_minimum(self.atoms, self.E_old)

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
        t0 = time.perf_counter()
        self.do_gcmc_step()
        self._step += 1
        self._last_step_seconds = time.perf_counter() - t0

        if self._step % self._outfile_write_interval == 0:
            self.write_outfile()
            self.logger.info("step=%d N=%d E=%.6f t=%.2fs",
                             self._step, self.n_atoms, self.E_old,
                             self._last_step_seconds)

        if self._step % self._trajectory_write_interval == 0:
            self.write_coordinates(self.atoms, self.E_old)
