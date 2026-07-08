import random
import logging

from abc import ABC, abstractmethod
from typing import Optional, List

from ase import Atoms
from ase.calculators.calculator import Calculator

import numpy as np

from ..cell import Cell

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    def __init__(self,
                 atoms: Atoms,
                 cells: List[Cell],
                 units_type: str,
                 calculator: Calculator,
                 user_tag: Optional[str] = None,
                 random_seed: Optional[int] = None,
                 traj_file: Optional[str] = 'trajectory.xyz',
                 traj_mode: str = 'w',
                 trajectory_write_interval: int = 1,
                 outfile: Optional[str] = 'outfile.out',
                 outfile_mode: str = 'w',
                 outfile_write_interval: int = 1,
                 minima_file: Optional[str] = None,
                 minima_mode: str = 'a') -> None:
        """
        Base class for ensembles in Monte Carlo simulations.

        Files are not opened in ``__init__``; they are opened by
        :meth:`initialize_run`, which is called automatically from
        :meth:`run`. Pass ``traj_file=None`` or ``outfile=None`` to disable
        either output channel. Use ``traj_mode='a'`` / ``outfile_mode='a'``
        to append to existing files (e.g. for restart).

        ``minima_file`` (off by default) enables basin-hopping-style output:
        a frame is written every time a new strictly-lower score is observed
        (see :meth:`_minimum_score`). ``minima_mode='a'`` appends a history of
        improving minima; ``'w'`` keeps only the current running best.
        """
        self.logger = logger

        self._accepted_trials = 0
        self._step = 0
        self._last_step_seconds = 0.0

        self._atoms = atoms
        self._cells = cells
        self._calculator = calculator
        self._user_tag = user_tag

        self._trajectory_write_interval = trajectory_write_interval
        self._outfile_write_interval = outfile_write_interval
        self._outfile = outfile
        self._outfile_mode = outfile_mode
        self._traj_file = traj_file
        self._traj_mode = traj_mode

        self._traj_handle = None
        self._outfile_handle = None
        self._initialized = False

        self._minima_file = minima_file
        self._minima_mode = minima_mode
        self._minima_handle = None
        self._best_score = float('inf')
        self._best_energy = float('inf')
        self._best_atoms = None

        # Seed source for derived generators (moves, acceptance test). We do
        # not touch the global ``random`` module: moves carry their own RNG.
        if random_seed is None:
            self._random_seed = random.randint(0, int(1e16))
        else:
            self._random_seed = random_seed

    def __enter__(self):
        self.initialize_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize_run()
        return False

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @property
    def cells(self) -> Cell:
        return self._cells

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms

    @property
    def step(self) -> int:
        return self._step

    def calculate_cells_volume(self, atoms) -> None:
        for cell in self.cells:
            cell.calculate_volume(atoms)

    def write_outfile(self, step: int, energy: float) -> None:
        """
        Default outfile writer: ``STEP: <s> ENERGY: <e>``. Subclasses
        typically override this. No-op when ``outfile`` is disabled or the
        handle was never opened.
        """
        if self._outfile is None or self._outfile_handle is None:
            return
        try:
            self._outfile_handle.write(f'STEP: {step} ENERGY: {energy}\n')
            self._outfile_handle.flush()
        except (OSError, AttributeError):
            self.logger.exception("Error writing to file %s", self._outfile)

    def write_coordinates(self, atoms: Atoms, energy: float) -> None:
        if self._traj_file is None or self._traj_handle is None:
            return
        try:
            write_xyz(atoms, energy, self._traj_handle)
        except (OSError, AttributeError):
            self.logger.exception("Error writing to trajectory file %s", self._traj_file)

    def _minimum_score(self, atoms: Atoms, energy: float) -> float:
        """Score used to compare configurations for the running minimum.

        Default is the potential energy. ``GrandCanonicalEnsemble`` overrides
        this to return the grand potential ``E - Σ μ_i N_i`` so the comparison
        is meaningful across configurations with different particle counts.
        """
        return energy

    def _record_minimum(self, atoms: Atoms, energy: float) -> None:
        """If ``atoms``/``energy`` improves the running best, snapshot it and
        write to ``minima_file`` (no-op when ``minima_file`` is disabled)."""
        score = self._minimum_score(atoms, energy)
        if score >= self._best_score:
            return
        self._best_score = score
        self._best_energy = energy
        self._best_atoms = atoms.copy()
        if self._minima_file is None or self._minima_handle is None:
            return
        try:
            if self._minima_mode == 'w':
                self._minima_handle.seek(0)
                self._minima_handle.truncate()
            extra = f'step={self._step} score={score:.6f}'
            write_xyz(atoms, energy, self._minima_handle, extra=extra)
        except (OSError, AttributeError):
            self.logger.exception("Error writing to minima file %s", self._minima_file)

    def close_files(self) -> None:
        if self._traj_handle is not None:
            try:
                self._traj_handle.close()
            except OSError:
                self.logger.exception("Error closing trajectory file %s", self._traj_file)
            self._traj_handle = None
        if self._outfile_handle is not None:
            try:
                self._outfile_handle.close()
            except OSError:
                self.logger.exception("Error closing outfile %s", self._outfile)
            self._outfile_handle = None
        if self._minima_handle is not None:
            try:
                self._minima_handle.close()
            except OSError:
                self.logger.exception("Error closing minima file %s", self._minima_file)
            self._minima_handle = None

    def __del__(self):
        # Safety net only. Deterministic cleanup goes through finalize_run /
        # the context manager.
        try:
            self.close_files()
        except Exception:
            pass

    def compute_energy(self, atoms: Atoms) -> float:
        return self._calculator.get_potential_energy(atoms)

    def initialize_outfile(self) -> None:
        """Create the outfile, write header/metadata, leave handle open for appends."""
        if self._outfile is None:
            return
        try:
            with open(self._outfile, self._outfile_mode) as outfile:
                if self._outfile_mode == 'w':
                    outfile.write(self.get_outfile_header())
                    outfile.write(self.get_outfile_metadata())
            self._outfile_handle = open(self._outfile, 'a')
        except OSError:
            self.logger.exception("Failed to initialize output file '%s'", self._outfile)
            raise

    def initialize_run(self) -> None:
        """Open file handles. Idempotent. Subclasses extend, then call super first."""
        if self._initialized:
            return
        if self._traj_file is not None and self._traj_handle is None:
            try:
                self._traj_handle = open(self._traj_file, self._traj_mode)
            except OSError:
                self.logger.exception("Failed to open trajectory file '%s'", self._traj_file)
                raise
        if self._minima_file is not None and self._minima_handle is None:
            try:
                # 'w' opens read+write so we can truncate on each improvement
                # without losing the handle (single-best mode).
                mode = 'w+' if self._minima_mode == 'w' else 'a'
                self._minima_handle = open(self._minima_file, mode)
            except OSError:
                self.logger.exception("Failed to open minima file '%s'", self._minima_file)
                raise
        self._initialized = True

    def finalize_run(self) -> None:
        """Log summary statistics and close files. Idempotent."""
        if not self._initialized:
            return
        self.logger.info("Simulation complete.")
        ms = getattr(self, 'move_selector', None)
        if ms is not None:
            if hasattr(ms, 'total_ratios'):
                self.logger.info("Total attempts per move: %s", ms.move_counter_total)
                self.logger.info("Cumulative acceptance ratios: %s", ms.total_ratios())
            else:
                self.logger.info("Attempts per move: %s", ms.move_counter)
                self.logger.info("Acceptance ratios: %s", ms.get_acceptance_ratio())
        if hasattr(self, 'E_old'):
            self.logger.info("Final energy (eV): %.6f", self.E_old)
        self.close_files()
        self._initialized = False

    def get_outfile_header(self) -> str:
        return "STEP ENERGY\n"

    def get_outfile_metadata(self) -> str:
        return ""

    @abstractmethod
    def _run(self) -> None:
        """Perform a single ensemble step. Implemented by subclasses."""
        raise NotImplementedError

    def run(self, steps: int) -> None:
        """Standard run loop: initialize_run, loop _run, finalize_run."""
        self.initialize_run()
        try:
            for _ in range(steps):
                self._run()
        finally:
            self.finalize_run()


def write_xyz(atoms, energy, file_or_path, extra: str = ''):
    """
    Write an XYZ frame to an open file handle or a path (append mode).
    Single-string build; avoids the per-frame ``np.savetxt`` overhead.

    ``extra`` is inserted into the comment line between ``energy=`` and the
    ``Lattice=`` field (e.g. ``"step=42 score=-1.234"``).
    """
    cell = np.asarray(atoms.get_cell())
    positions = atoms.positions
    symbols = atoms.get_chemical_symbols()
    num_atoms = len(atoms)
    cell_str = " ".join(f"{v:.8f}" for v in cell.ravel())

    mol_ids = atoms.arrays.get('molecule_id')

    comment = f'energy={energy:.6f}'
    if extra:
        comment += f' {extra}'
    comment += f' Lattice="{cell_str}"'
    if mol_ids is not None:
        comment += ' Properties=species:S:1:pos:R:3:molecule_id:I:1'
    comment += '\n'
    parts = [f"{num_atoms}\n", comment]

    for i, (s, p) in enumerate(zip(symbols, positions)):
        line = f"{s} {p[0]} {p[1]} {p[2]}"
        if mol_ids is not None:
            line += f" {mol_ids[i]}"
        line += "\n"
        parts.append(line)
    text = "".join(parts)

    if isinstance(file_or_path, str):
        with open(file_or_path, 'a') as xyz_file:
            xyz_file.write(text)
    else:
        file_or_path.write(text)
        file_or_path.flush()
