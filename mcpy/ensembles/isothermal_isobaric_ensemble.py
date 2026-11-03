import warnings
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .base_ensemble import code_version
from .grand_canonical_ensemble import GrandCanonicalEnsemble, _DELETIONS, _INSERTIONS
from ..moves.move_selector import MoveSelector
from ..moves.volume_move import VolumeMove

_VOLUME_COL = 17  # 'Volume (sigma^3)' plus a space


class IsothermalIsobaricEnsemble(GrandCanonicalEnsemble):
    """Isothermal-isobaric (NPT) Monte Carlo at fixed ``pressure``.

    The grand-canonical machinery (in-place moves, snapshot and rollback,
    batched replica exchange) with no particle exchange. A ``VolumeMove``
    scales the box and is accepted with

        exp(-beta (dE + P dV) + (N + 1) ln(V'/V)),

    every other move keeps the box and uses the usual Metropolis rule. The
    configuration is scored by the enthalpy H = E + PV, so a temperature
    ladder of these replicas in ``BatchedReplicaExchange`` swaps with the
    correct NPT rule.

    ``pressure`` is in eV/A^3 for ``'metal'`` units (e.g. ``1 * ase.units.GPa``)
    and in reduced units for ``'LJ'``. The box must be periodic in all three
    directions and carry no constraints; molecules are not supported yet.
    """

    _scores_volume_moves = True

    def __init__(self,
                 atoms: Atoms,
                 units_type: str,
                 calculator: Calculator,
                 pressure: float,
                 temperature: float,
                 move_selector: MoveSelector,
                 random_seed: Optional[int] = None,
                 traj_file: Optional[str] = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 trajectory_write_interval: int = 1,
                 outfile: Optional[str] = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: int = 1,
                 minima_file: Optional[str] = None,
                 minima_mode: str = 'a') -> None:
        # Validate before the parent's first (possibly expensive) energy call.
        moves = move_selector.move_list
        if not any(isinstance(m, VolumeMove) for m in moves):
            raise ValueError('IsothermalIsobaricEnsemble needs a VolumeMove; '
                             'without one the run samples NVT')
        exchange = [type(m).__name__ for m in moves if isinstance(m, _INSERTIONS + _DELETIONS)]
        if exchange:
            raise ValueError(
                f'IsothermalIsobaricEnsemble keeps N fixed; remove {exchange}. '
                'Volume moves combined with particle exchange are not supported yet.')
        if not np.all(atoms.pbc):
            raise ValueError(f'the volume move needs a box periodic in x, y and z, '
                             f'got pbc={atoms.pbc}')
        if atoms.constraints:
            raise ValueError('the volume move scales every atom, so constraints '
                             f'are not supported; got {atoms.constraints}')
        if not np.isfinite(pressure):
            raise ValueError(f'pressure must be finite, got {pressure}')
        self._pressure = float(pressure)

        super().__init__(atoms=atoms,
                         cells=[],
                         units_type=units_type,
                         calculator=calculator,
                         mu={},
                         species=sorted(set(atoms.get_chemical_symbols())),
                         temperature=temperature,
                         move_selector=move_selector,
                         random_seed=random_seed,
                         traj_file=traj_file,
                         traj_mode=traj_mode,
                         trajectory_write_interval=trajectory_write_interval,
                         outfile=outfile,
                         outfile_mode=outfile_mode,
                         outfile_write_interval=outfile_write_interval,
                         minima_file=minima_file,
                         minima_mode=minima_mode)

        if getattr(calculator, 'steps', 0) > 0:
            warnings.warn(
                'the calculator relaxes trial energies at fixed lattice (ADR 0002), '
                'so the volume acceptance samples relaxed basins, not the exact NPT '
                'distribution. Use steps=0 for exact NPT.', UserWarning, stacklevel=2)
        self._table_width += _VOLUME_COL

    def get_state(self):
        # The MPI replica exchange reads the pressure from the state for its
        # P V swap term. It describes the slot: set_state ignores it.
        return {**super().get_state(), 'pressure': self._pressure}

    def _propose(self):
        # The parent snapshot holds the arrays and constraints but not the
        # lattice. Keep the lattice here so _restore can undo a volume move.
        self._trial_lattice = self.atoms.cell.copy()
        return super()._propose()

    def _restore(self, snapshot) -> None:
        super()._restore(snapshot)
        self.atoms.set_cell(self._trial_lattice)

    def _acceptance_condition(self,
                              potential_diff: float,
                              delta_particles: int,
                              volume: float,
                              species: str,
                              n_atoms: int = None) -> bool:
        """NPT acceptance for a volume move; the parent rule otherwise
        (Metropolis, since no move here changes N or the volume)."""
        move = self.move_selector.move_list[self.move_selector.to_use]
        if not isinstance(move, VolumeMove):
            return super()._acceptance_condition(
                potential_diff, delta_particles, volume, species, n_atoms)
        v_old, v_new = move.last_volumes
        log_p = (-self.units.beta * (potential_diff + self._pressure * (v_new - v_old))
                 + move.last_log_jacobian)
        self.logger.debug('V: %.3f -> %.3f, Potential diff: %.3e, log p: %.3e',
                          v_old, v_new, potential_diff, log_p)
        if log_p >= 0:
            return True
        return np.exp(log_p) > self.rng_acceptance.get_uniform()

    def _minimum_score(self, atoms: Atoms, energy: float) -> float:
        """Enthalpy H = E + PV (minus Σ μN, which is empty here). The
        replica-exchange swap scores configurations through this method,
        so the PV term also enters the swap rule."""
        return super()._minimum_score(atoms, energy) + self._pressure * atoms.get_volume()

    def get_outfile_header(self) -> str:
        title = '| Isothermal-Isobaric Ensemble Monte Carlo Simulation |'
        border = '+' + '-' * (len(title) - 2) + '+'
        return f'{border}\n{title}\n{border}\n\n'

    def get_outfile_metadata(self) -> str:
        ratio_label = f"Acceptance Ratios ({', '.join(self.move_selector.move_list_names)})"
        # LJ runs are in reduced units: energy in epsilon, T in epsilon/k_B.
        lj = self.units.unit_type == 'LJ'
        energy_label = 'Energy (eps)' if lj else 'Energy (eV)'
        volume_label = 'Volume (sigma^3)' if lj else 'Volume (A^3)'
        temperature_label = 'Temperature (reduced)' if lj else 'Temperature (K)'
        pressure_label = 'Pressure (reduced)' if lj else 'Pressure (eV/A^3)'
        header_row = '{:<10} {:<10} {:<15} {:<{v}} {:<{w}}'.format(
            'Step', 'N_atoms', energy_label, volume_label, ratio_label,
            v=_VOLUME_COL - 1, w=self._ratio_col_width,
        )
        return (
            'Simulation Parameters:\n'
            f'  mcpy version: {code_version()}\n'
            f'  Units type: {self.units.unit_type}\n'
            f'  {temperature_label}: {self._temperature}\n'
            f'  {pressure_label}: {self._pressure}\n'
            '  Acceptance ratios are per write-interval (cumulative totals '
            'in the finalize_run log).\n'
            'Starting simulation...\n'
            + header_row
            + '\n' + '-' * self._table_width + '\n'
        )

    def _row(self, ratio_str: str) -> str:
        return '{:<10} {:<10} {:<15.6f} {:<{v}.4f} {:<{w}}'.format(
            self._step, self.n_atoms, self.E_old, self.atoms.get_volume(), ratio_str,
            v=_VOLUME_COL - 1, w=self._ratio_col_width)
