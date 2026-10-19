import time
import warnings
from typing import Any, Optional, List, Dict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.geometry import wrap_positions

from .base_ensemble import BaseEnsemble, code_version
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
from ..moves.base_move import free_atom_count
from ..moves.deletion_move import DeletionMove
from ..moves.insertion_move import InsertionMove
from ..moves.molecule_deletion_move import MoleculeDeletionMove
from ..moves.molecule_insertion_move import MoleculeInsertionMove
from ..moves.move_selector import MoveSelector
from ..moves.molecule_utils import find_molecules, molecule_groups
from ..cell import Cell, CustomCell, SphericalCell

_RATIO_COL_PER_MOVE = 8  # width per "100.0%, " entry, generous
_INSERTIONS = (InsertionMove, MoleculeInsertionMove)
_DELETIONS = (DeletionMove, MoleculeDeletionMove)


class GrandCanonicalEnsemble(BaseEnsemble):
    # (cell, atomic numbers) per ball surface exchanged atoms may not cross.
    _ball_guards = ()
    # (cell, atomic numbers) per CustomCell top exchanged atoms may not cross.
    _top_guards = ()

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
                 trajectory_write_interval: int = 1,
                 outfile: Optional[str] = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: int = 1,
                 minima_file: Optional[str] = None,
                 minima_mode: str = 'a') -> None:
        """``cells`` are the cells whose free volume is refreshed after
        every accepted move. The cell of every insertion move is added to
        them automatically: insertions read the cached volume."""

        super().__init__(atoms=atoms,
                         cells=cells,
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

        # Validate before the first (possibly expensive) energy call.
        self.units = SetUnits(units_type,
                              temperature=temperature,
                              species=species,
                              molecules=molecules)
        self._temperature = temperature
        self._mu = mu
        self.move_selector = move_selector
        self._check_species(species)
        self._check_exchange_balance()
        self._check_molecule_ids()
        if (any(getattr(m, 'min_insert', None) is not None for m in move_selector.move_list)
                and getattr(calculator, 'steps', 0) <= 0):
            warnings.warn(
                'min_insert is set but the calculator does not relax (no steps > 0). '
                'min_insert seeds relaxed trials (ADR 0004); with single-point '
                'energies it shifts mu by kT ln(V_free / V_seed).', UserWarning, stacklevel=2)

        self._cells = list(cells)
        for move in move_selector.move_list:
            if isinstance(move, _INSERTIONS) and all(move.cell is not c for c in self._cells):
                self._cells.append(move.cell)
        self._ball_guards = self._guards((InsertionMove, DeletionMove), SphericalCell)
        self._top_guards = self._guards((InsertionMove,), CustomCell)

        self.E_old = self.compute_energy(self.atoms)
        self.n_atoms = len(self.atoms)
        self.calculate_cells_volume(self.atoms)
        self._step = 0
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

    def _guards(self, move_types, cell_type):
        """``[(cell, atomic numbers)]``: the exchanged atomic species of each
        ``cell_type`` cell that a move of ``move_types`` uses."""
        guards = {}
        for move in self.move_selector.move_list:
            if isinstance(move, move_types) and isinstance(move.cell, cell_type):
                guards.setdefault(id(move.cell), (move.cell, set()))[1].update(
                    atomic_numbers[s] for s in move.species)
        return [(c, np.array(sorted(z))) for c, z in guards.values()]

    def _check_species(self, species) -> None:
        """Atomic species are element symbols; every exchanged species is
        declared (atomic in ``species``, molecular in ``molecules``) and has
        a mu."""
        not_elements = [s for s in species if s not in atomic_numbers]
        if not_elements:
            raise ValueError(
                f'species {not_elements} are not element symbols; list molecular '
                'species in molecules= instead')
        for move in self.move_selector.move_list:
            if not isinstance(move, _INSERTIONS + _DELETIONS):
                continue
            molecular = isinstance(move, (MoleculeInsertionMove, MoleculeDeletionMove))
            declared = self.units.molecules if molecular else species
            for s in move.species:
                if s not in declared:
                    raise ValueError(
                        f"{type(move).__name__} exchanges '{s}', which is not in "
                        f"{'molecules' if molecular else 'species'}=")
                if s not in self._mu:
                    raise ValueError(
                        f"{type(move).__name__} exchanges '{s}', which has no mu "
                        f'(mu keys: {sorted(self._mu)})')

    def _check_exchange_balance(self) -> None:
        """The acceptance assumes that insertion and deletion of a species in
        a cell are attempted with equal probability. Raise when the move
        weights, species lists or cells make them differ: the mismatch
        silently shifts mu by kT ln(p_ins / p_del)."""
        ms = self.move_selector
        total = float(sum(ms.probabilities))
        rates = ({}, {})
        for weight, move in zip(ms.probabilities, ms.move_list):
            if isinstance(move, _INSERTIONS + _DELETIONS):
                side = rates[0] if isinstance(move, _INSERTIONS) else rates[1]
                for s in move.species:
                    key = (s, id(move.cell))
                    side[key] = side.get(key, 0.0) + weight / total / len(move.species)
        for key in rates[0].keys() | rates[1].keys():
            p_ins, p_del = rates[0].get(key, 0.0), rates[1].get(key, 0.0)
            if np.isclose(p_ins, p_del, rtol=1e-9, atol=0.0):
                continue
            with np.errstate(divide='ignore'):
                shift = np.log(p_ins / p_del) / self.units.beta if p_del else np.inf
            raise ValueError(
                f"species '{key[0]}' is inserted with probability {p_ins:.4g} and "
                f'deleted with probability {p_del:.4g} per trial in the same cell. '
                f'The acceptance assumes equal ones: this shifts mu by '
                f'kT ln(p_ins/p_del) = {shift:+.4g}. Give its insertion and '
                'deletion moves the same weight, species list and cell.')

    def _check_molecule_ids(self) -> None:
        """Every ``molecule_id >= 0`` group must be a registered molecule.
        ASE fills a missing array with 0 when atoms are combined, which would
        hide the host atoms inside a fake molecule."""
        ids = molecule_groups(self.atoms)
        if not ids:
            return
        known = {tuple(sorted(t.get_chemical_symbols()))
                 for t in self.units.molecules.values()}
        symbols = np.asarray(self.atoms.get_chemical_symbols())
        bad = [mid for mid, members in ids.items()
               if tuple(sorted(symbols[members])) not in known]
        if bad:
            raise ValueError(
                f'molecule_id groups {bad[:10]} match no template in molecules=. '
                'Set molecule_id to -1 on the host atoms before you add molecules '
                '(ASE fills the array with 0 for atoms that lack it).')

    def get_state(self) -> Dict[str, Any]:
        # A copy: the dict is pickled across MPI ranks (the live atoms carry
        # the calculator, which may not pickle or may ship a whole model) and
        # must not alias the configuration the sampler keeps mutating.
        return {
            'atoms': self.atoms.copy(),
            'n_atoms': self.n_atoms,
            'energy': self.E_old,
            'mu': self._mu,
            'temperature': self._temperature,
            'beta': self.units.beta,
            'lambda_dbs': self.units.lambda_dbs,
            'step': self._step,
            'exchange_attempts': self.exchange_attempts,
            'exchange_successes': self.exchange_successes,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.atoms = state['atoms']
        self._last_traj_step = None  # new configuration, same step: new frame
        self.E_old = state['energy']
        self.n_atoms = state['n_atoms']
        # Step count and exchange statistics are restored only when the dict
        # carries them, so the get_state()/set_state() restart round-trip is
        # lossless. ``ReplicaExchange`` strips them before applying a swap:
        # they describe the *slot*, not the config, and trading them between
        # ranks made the per-rank "Accepted Exchange (%)" column meaningless.
        self._step = state.get('step', self._step)
        self.exchange_attempts = state.get(
            'exchange_attempts', self.exchange_attempts)
        self.exchange_successes = state.get(
            'exchange_successes', self.exchange_successes)
        # The configuration changed under the cells: refresh their free
        # volumes now, or the next insertion/deletion acceptance would use
        # the previous configuration's volume (GCMC only recalculates on
        # accepted moves).
        self.calculate_cells_volume(self.atoms)
        self._record_minimum(self.atoms, self.E_old)

    def get_outfile_header(self) -> str:
        return (
            '+-------------------------------------------------+\n'
            '| Grand Canonical Ensemble Monte Carlo Simulation |\n'
            '+-------------------------------------------------+\n\n'
        )

    def get_outfile_metadata(self) -> str:
        ratio_label = f"Acceptance Ratios ({', '.join(self.move_selector.move_list_names)})"
        # LJ runs are in reduced units: energy in epsilon, T in epsilon/k_B.
        lj = self.units.unit_type == 'LJ'
        energy_label = 'Energy (eps)' if lj else 'Energy (eV)'
        temperature_label = 'Temperature (reduced)' if lj else 'Temperature (K)'
        header_row = '{:<10} {:<10} {:<15} {:<{w}}'.format(
            'Step', 'N_atoms', energy_label, ratio_label,
            w=self._ratio_col_width,
        )
        return (
            'Simulation Parameters:\n'
            f'  mcpy version: {code_version()}\n'
            f'  Units type: {self.units.unit_type}\n'
            f'  {temperature_label}: {self._temperature}\n'
            f'  Chemical potentials: {self._mu}\n'
            '  Acceptance ratios are per write-interval (cumulative totals '
            'in the finalize_run log).\n'
            'Starting simulation...\n'
            + header_row
            + '\n' + '-' * self._table_width + '\n'
        )

    def _row(self, ratio_str: str) -> str:
        return '{:<10} {:<10} {:<15.6f} {:<{w}}'.format(
            self._step, self.n_atoms, self.E_old, ratio_str,
            w=self._ratio_col_width)

    def write_outfile(self, step: int = None, energy: float = None) -> None:
        """Write one row: step, N, energy, per-interval acceptance ratios.

        ``step``/``energy`` are accepted for compatibility with the
        :class:`BaseEnsemble` signature but ignored - GCMC always logs its
        own ``_step``/``E_old`` so the row matches the sampler state.
        """
        if self._last_logged_step == self._step:
            return  # already wrote this step (e.g. finalize_run after a triggered write)
        acceptance_ratios = self.move_selector.interval_ratios()
        # Reset before the disabled-outfile bail-out: the counters are
        # per-write-interval, so skipping the reset when there is nothing to
        # write turns interval_ratios() into total_ratios() for the rest of
        # the run.
        self.move_selector.reset_counters()
        if self._outfile is None or self._outfile_handle is None:
            self._last_logged_step = self._step
            return
        ratio_str = ', '.join(
            f'{r * 100:.1f}%' if not np.isnan(r) else 'N/A'
            for r in acceptance_ratios
        )
        try:
            self._outfile_handle.write(self._row(ratio_str) + '\n')
            self._outfile_handle.flush()
            self._last_logged_step = self._step
        except (OSError, AttributeError):
            self.logger.exception('Error writing to file %s', self._outfile)

    def _species_count(self, atoms: Atoms, specie: str) -> int:
        """Particle count for one μ species: molecules for molecular keys,
        atoms otherwise. The one counting convention shared by
        ``_minimum_score`` and the replica-exchange swap criterion."""
        if specie in self.units.molecules:
            template = self.units.molecules[specie]
            return len(find_molecules(
                atoms, sorted(template.get_chemical_symbols())))
        # An atomic key counts free atoms only: the O of a CO molecule is
        # not an O adatom (molecule_id < 0 marks atoms outside molecules).
        return free_atom_count(atoms, specie)

    def _minimum_score(self, atoms: Atoms, energy: float) -> float:
        """Grand potential Ω = E − Σ μ_i N_i. Comparing raw E across moves
        that change N is not meaningful in the grand canonical ensemble.
        Molecular species count molecules, atomic species count atoms."""
        score = energy
        for specie, mu in self._mu.items():
            score -= mu * self._species_count(atoms, specie)
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
                'Lambda_db: %.3e, p: %.3e, Beta: %.3e, Exp: %.3e, '
                'Exp Arg %s, Potential diff: %.3e, Delta_particles: %d',
                self.units.lambda_dbs[species], p, self.units.beta, exp_term,
                potential_diff - self._mu[species], potential_diff, delta_particles)
        elif delta_particles == -1:  # deletion
            db_term = self.units.de_broglie_deletion(volume, n_atoms, species)
            exp_term = np.exp(-self.units.beta * (potential_diff + self._mu[species]))
            p = db_term * exp_term
            self.logger.debug(
                'Lambda_db: %.3e, p: %.3e, Beta: %.3e, Exp: %.3e, '
                'Exp Arg %s, Potential diff: %.3e, Delta_particles: %d',
                self.units.lambda_dbs[species], p, self.units.beta, exp_term,
                potential_diff + self._mu[species], potential_diff, delta_particles)
        else:
            raise ValueError(
                f'unexpected delta_particles={delta_particles} from move '
                f"'{self.move_selector.get_name()}'"
            )

        if p > 1:
            return True
        return p > self.rng_acceptance.get_uniform()

    def _restore(self, snapshot) -> None:
        """Put the configuration back as it was before the trial move."""
        arrays, constraints = snapshot
        self.atoms.arrays = arrays
        self.atoms.set_constraint(constraints)

    def _propose(self):
        """Run one trial move in place. Returns ``(snapshot, meta)``, or
        ``None`` when the move could not propose (already rolled back).

        Split out of :meth:`do_gcmc_step` so ``BatchedReplicaExchange`` runs
        the identical proposal step: it has to propose on every replica before
        it can score any of them in one batched forward pass.

        The snapshot holds ``atoms.arrays`` (positions, numbers, ...) plus the
        constraints. Restoring it on rejection also undoes any in-place
        relaxation ``compute_energy`` did. The constraints are part of it
        because ``del atoms[i]`` remaps FixAtoms indices in place (ASE
        ``delete_atoms``), so a rejected deletion would otherwise leave the
        restored configuration with shifted fixed indices -- wrong atoms frozen
        from then on.
        """
        atoms = self.atoms
        snapshot = ({k: v.copy() for k, v in atoms.arrays.items()},
                    [c.copy() for c in atoms.constraints])

        atoms_new, delta_particles, species = self.move_selector.do_trial_move(atoms)

        if atoms_new is False or atoms_new is None:
            # Move couldn't be proposed (e.g. empty cell). MoveSelector already
            # recorded the failure so it won't depress the acceptance ratio.
            # Identity check, not truthiness: an empty Atoms (last atom
            # deleted) is falsy but is a real proposal that must be scored.
            #
            # Restore rather than trust the sentinel to mean "nothing was
            # touched": a move that mutates before bailing would otherwise
            # leave the configuration changed while ``E_old`` still describes
            # the previous one.
            self._restore(snapshot)
            return None

        if atoms_new is not atoms:
            raise RuntimeError(
                f"move '{self.move_selector.get_name()}' returned a "
                'different Atoms object; GCMC moves must mutate the '
                'passed atoms in place (copy-based moves are for '
                'CanonicalEnsemble only)'
            )
        if self._crosses_custom_top(snapshot[0]):
            # Zero acceptance by construction: a rejected attempt (already
            # counted by the selector), scored without an energy call.
            self._restore(snapshot)
            return None
        return snapshot, (delta_particles, species)

    def _crosses_custom_top(self, old_arrays) -> bool:
        """Whether a same-size proposal moved an exchanged atom across the
        top of a CustomCell that insertions use, in either direction. No
        insertion can put an atom above the top, so a move must not carry
        one there (or back). Checked before the energy call: atoms that
        relaxation pushes above the top are kept and stay deletable."""
        new_arrays = self.atoms.arrays
        if not self._top_guards or len(new_arrays['numbers']) != len(old_arrays['numbers']):
            return False
        above = []
        for arrays in (old_arrays, new_arrays):
            pos = arrays['positions']
            if self._wrap_on_accept:
                pos = wrap_positions(pos, self.atoms.cell, pbc=self.atoms.pbc)
            above.append([((pos - c.offset) @ c._dim_inv)[:, 2] >= 1.0
                          for c, _ in self._top_guards])
        free = new_arrays['molecule_id'] < 0 if 'molecule_id' in new_arrays else True
        for (_, numbers), was, now in zip(self._top_guards, *above):
            # Per slot: an exchanged atom above the top. Comparing this, not
            # only positions, also catches a permutation that keeps every
            # position but swaps an exchanged species above the top.
            was &= np.isin(old_arrays['numbers'], numbers)
            now &= np.isin(new_arrays['numbers'], numbers)
            if np.any((was != now) & free):
                return True
        return False

    def _outside_ball_counts(self, arrays):
        """Per ball guard: free exchanged atoms outside the ball."""
        pos = arrays['positions']
        if self._wrap_on_accept:
            pos = wrap_positions(pos, self.atoms.cell, pbc=self.atoms.pbc)
        free = arrays['molecule_id'] < 0 if 'molecule_id' in arrays else True
        counts = []
        for cell, numbers in self._ball_guards:
            d = pos - cell.center
            outside = np.einsum('ij,ij->i', d, d) > cell.radius ** 2
            counts.append(int(np.count_nonzero(
                outside & free & np.isin(arrays['numbers'], numbers))))
        return counts

    def _commit_or_rollback(self, E_new: float, snapshot, meta) -> bool:
        """Score the pending proposal and either keep it or restore the
        snapshot. Returns whether it was accepted.

        A trial that moves an exchanged atom across the ball surface of a
        SphericalCell or DomeCell is rejected: outside the ball it could not
        be deleted but would still count in the acceptance. Both directions
        are blocked, so the move stays reversible."""
        if self._ball_guards and (self._outside_ball_counts(snapshot[0])
                                  != self._outside_ball_counts(self.atoms.arrays)):
            self._restore(snapshot)
            return False
        delta_particles, species = meta
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
        if not self._acceptance_condition(delta_E, delta_particles, volume,
                                          species, n_exchange):
            self._restore(snapshot)
            return False

        if self._wrap_on_accept:
            self.atoms.wrap()
        self.n_atoms = len(self.atoms)
        self.E_old = E_new
        self.move_selector.acceptance_counter()
        self.calculate_cells_volume(self.atoms)
        self._record_minimum(self.atoms, self.E_old, self._step + 1)
        self.logger.debug('Volume: %.3f, Delta_particles: %d, Species: %s',
                          volume, delta_particles, species)
        return True

    def do_gcmc_step(self) -> None:
        for _ in range(self.move_selector.n_moves):
            proposal = self._propose()
            if proposal is None:
                continue
            snapshot, meta = proposal
            try:
                E_new = self.compute_energy(self.atoms)
            except BaseException:
                # Ctrl-C, CUDA OOM: never leave the unscored trial in place.
                self._restore(snapshot)
                raise
            self._commit_or_rollback(E_new, snapshot, meta)

    def initialize_run(self) -> None:
        """Open files, write headers, log start, write initial state."""
        if self._initialized:
            return
        super().initialize_run()
        self.initialize_outfile()
        self.logger.info(
            'GCMC starting: T=%s K, mu=%s, outfile=%s',
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
        if self._last_logged_step == self._step:
            return  # a resumed run: the previous run's last row is this one
        try:
            placeholder = ', '.join('N/A' for _ in self.move_selector.move_list)
            self._outfile_handle.write(self._row(placeholder) + '\n')
            self._outfile_handle.flush()
            self._last_logged_step = self._step
        except (OSError, AttributeError):
            self.logger.exception('Error writing initial row to %s', self._outfile)

    def finalize_run(self) -> None:
        if not self._initialized:
            return
        # Write the trailing row + frame so the log/traj end on the actual
        # final state regardless of write_interval alignment.
        self.write_outfile()
        self.write_coordinates(self.atoms, self.E_old)
        self.logger.info(
            'GCMC complete: steps=%d, final_energy=%.6f eV',
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
            self.logger.info('step=%d N=%d E=%.6f t=%.2fs',
                             self._step, self.n_atoms, self.E_old,
                             self._last_step_seconds)

        if self._step % self._trajectory_write_interval == 0:
            self.write_coordinates(self.atoms, self.E_old)
