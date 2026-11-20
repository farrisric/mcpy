import contextlib
import functools
import logging
import os
import random
import signal
import subprocess
import threading

from abc import ABC, abstractmethod
from typing import Optional, List

import ase.io
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from ..cell import Cell

logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    def __init__(self,
                 atoms: Atoms,
                 cells: List[Cell],
                 calculator: Calculator,
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
        :meth:`~mcpy.ensembles.BaseEnsemble.run`. Pass ``traj_file=None`` or
        ``outfile=None`` to disable either output channel. Use
        ``traj_mode='a'`` / ``outfile_mode='a'`` to append to existing files.
        To continue a stopped run, use ``GrandCanonicalEnsemble.restart``.

        ``minima_file`` (off by default) enables basin-hopping-style output:
        a frame is written every time a new strictly-lower score is observed
        (see ``_minimum_score``). ``minima_mode='a'`` appends a history of
        improving minima (a fresh ensemble starts the file anew; a later
        ``run()`` continues it); ``'w'`` keeps only the current running best.
        """
        self.logger = logger

        self._accepted_trials = 0
        self._step = 0
        self._last_step_seconds = 0.0

        self._atoms = atoms
        self._cells = cells
        self._calculator = calculator

        self._trajectory_write_interval = trajectory_write_interval
        self._outfile_write_interval = outfile_write_interval
        self._outfile = outfile
        self._outfile_mode = outfile_mode
        self._traj_file = traj_file
        self._traj_mode = traj_mode

        self._traj_handle = None
        self._outfile_handle = None
        self._initialized = False
        # Set by finalize_run: a later run() continues the files it wrote
        # instead of truncating them.
        self._resume_files = False
        # Step of the last trajectory frame, so a frame is never written
        # twice for the same state (e.g. finalize_run after an aligned write).
        self._last_traj_step = None
        # Step of the last outfile row, so a row is never written twice.
        self._last_logged_step = None

        # Set by restart(): the next run(steps) counts steps from step 0.
        self._restart_step = 0

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
        self._start()
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
        Default outfile writer: ``STEP: <s> ENERGY: <e>``, at most once per
        step. Subclasses typically override this. No-op when ``outfile`` is
        disabled or the handle was never opened.
        """
        if self._last_logged_step == step:
            return
        if self._outfile is None or self._outfile_handle is None:
            return
        try:
            self._outfile_handle.write(f'STEP: {step} ENERGY: {energy}\n')
            self._outfile_handle.flush()
            self._last_logged_step = step
        except (OSError, AttributeError):
            self.logger.exception('Error writing to file %s', self._outfile)

    def write_coordinates(self, atoms: Atoms, energy: float) -> None:
        """Write one trajectory frame, at most once per step. Reset
        ``_last_traj_step`` to ``None`` when the configuration changes without
        a step (a replica-exchange swap) so the new state gets its frame."""
        if self._traj_file is None or self._traj_handle is None:
            return
        if self._last_traj_step == self._step:
            return
        try:
            write_xyz(atoms, energy, self._traj_handle, extra={'step': self._step})
            self._last_traj_step = self._step
        except (OSError, AttributeError):
            self.logger.exception('Error writing to trajectory file %s', self._traj_file)

    def _minimum_score(self, atoms: Atoms, energy: float) -> float:
        """Score used to compare configurations for the running minimum.

        Default is the potential energy. ``GrandCanonicalEnsemble`` overrides
        this to return the grand potential ``E - Σ μ_i N_i`` so the comparison
        is meaningful across configurations with different particle counts.
        """
        return energy

    def _record_minimum(self, atoms: Atoms, energy: float, step: int = None) -> None:
        """If ``atoms``/``energy`` improves the running best, snapshot it and
        write to ``minima_file`` (no-op when ``minima_file`` is disabled).
        ``step`` labels the frame (default ``self._step``); calls from inside
        a step pass ``self._step + 1``, the number the outfile gives it."""
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
            step = self._step if step is None else step
            write_xyz(atoms, energy, self._minima_handle,
                      extra={'step': step, 'score': round(score, 6)})
        except (OSError, AttributeError):
            self.logger.exception('Error writing to minima file %s', self._minima_file)

    def close_files(self) -> None:
        if self._traj_handle is not None:
            try:
                self._traj_handle.close()
            except OSError:
                self.logger.exception('Error closing trajectory file %s', self._traj_file)
            self._traj_handle = None
        if self._outfile_handle is not None:
            try:
                self._outfile_handle.close()
            except OSError:
                self.logger.exception('Error closing outfile %s', self._outfile)
            self._outfile_handle = None
        if self._minima_handle is not None:
            try:
                self._minima_handle.close()
            except OSError:
                self.logger.exception('Error closing minima file %s', self._minima_file)
            self._minima_handle = None

    def restart(self, step: Optional[int] = None) -> int:
        """Continue a stopped run from its files. Returns the step it
        continues from.

        Build the ensemble with the same code and file names as the first
        run, call ``restart()``, then ``run()`` with the same step count::

            ens = build()      # the same code as the first run
            ens.restart()
            ens.run(STEPS)     # runs STEPS minus the steps already done

        The next ``run(steps)`` after a restart counts ``steps`` from the
        start of the first run, as the replica-exchange drivers count
        ``gcmc_steps``, so the same script finishes the job.

        The cells, moves and seeds come from that code, so the sampled region
        is the same as in the first run. The configuration and its step come
        from the last complete frame of ``traj_file``; a torn last frame is
        cut off. The energy is computed again. Rows of ``outfile`` past that
        step are removed, because the run does those steps again. New rows,
        frames and minima are appended. The running minimum continues from
        the last frame of ``minima_file``. Every random stream is reseeded
        from its seed and the step, so the continuation does not replay the
        random numbers of the first run.

        ``step`` overrides the step label of the frame. Trajectories written
        by mcpy before restart() existed carry no label.
        """
        if self._initialized:
            raise RuntimeError('call restart() before run()')
        if self._traj_file is None:
            raise ValueError('restart() reads traj_file, and this ensemble has none')
        frame = read_last_frame(self._traj_file)
        if step is None:
            if 'step' not in frame.info:
                raise ValueError(
                    f'the last frame of {self._traj_file} has no step label (it was '
                    'written by an older mcpy); pass restart(step=...)')
            step = int(frame.info['step'])
        frame.info = {}
        frame.calc = self.atoms.calc
        self._load_configuration(frame)

        self._step = self._last_traj_step = self._last_logged_step = step
        self._restart_step = step
        self._resume_files = True
        drop_rows_after(self._outfile, step, self._row_step)
        if (self._minima_file is not None and os.path.exists(self._minima_file)
                and os.path.getsize(self._minima_file) > 0):
            best = read_last_frame(self._minima_file)
            energy = frame_energy(best)
            best.info, best.calc = {}, None
            self._best_score = self._minimum_score(best, energy)
            self._best_energy, self._best_atoms = energy, best
        reseed_streams(self, step)
        self.logger.info('restart from %s at step %d with %d atoms',
                         self._traj_file, step, len(self.atoms))
        return step

    def _load_configuration(self, atoms: Atoms) -> None:
        """Make ``atoms`` the current configuration, for ``restart``.
        Subclasses set their energy and caches from it."""
        raise NotImplementedError(f'{type(self).__name__} does not support restart()')

    @staticmethod
    def _row_step(line: str) -> Optional[int]:
        """The step of an outfile row (``STEP: <s> ENERGY: <e>``), or None
        for a header line. ``restart`` uses it to find stale rows."""
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == 'STEP:':
            try:
                return int(tokens[1])
            except ValueError:
                return None
        return None

    def compute_energy(self, atoms: Atoms) -> float:
        return self._calculator.get_potential_energy(atoms)

    def initialize_outfile(self) -> None:
        """Create the outfile, write header/metadata, leave handle open for appends."""
        if self._outfile is None:
            return
        mode = 'a' if self._resume_files else self._outfile_mode
        try:
            with open(self._outfile, mode) as outfile:
                if mode == 'w':
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
                mode = 'a' if self._resume_files else self._traj_mode
                self._traj_handle = open(self._traj_file, mode)
            except OSError:
                self.logger.exception("Failed to open trajectory file '%s'", self._traj_file)
                raise
        if self._minima_file is not None and self._minima_handle is None:
            try:
                # 'w' opens read+write so we can truncate on each improvement
                # without losing the handle (single-best mode). A resumed run
                # keeps the current best ('a+' still truncates on improvement).
                # History mode starts a fresh file and continues it on resume.
                if self._minima_mode != 'w':
                    mode = 'a' if self._resume_files else 'w'
                else:
                    mode = 'a+' if self._resume_files else 'w+'
                self._minima_handle = open(self._minima_file, mode)
            except OSError:
                self.logger.exception("Failed to open minima file '%s'", self._minima_file)
                raise
        self._initialized = True

    def finalize_run(self) -> None:
        """Log summary statistics and close files. Idempotent."""
        if not self._initialized:
            return
        self.logger.info('Simulation complete.')
        ms = getattr(self, 'move_selector', None)
        if ms is not None:
            self.logger.info('Total attempts per move: %s', ms.move_counter_total)
            self.logger.info('Cumulative acceptance ratios: %s', ms.total_ratios())
        if hasattr(self, 'E_old'):
            self.logger.info('Final energy (eV): %.6f', self.E_old)
        self.close_files()
        self._initialized = False
        self._resume_files = True

    def get_outfile_header(self) -> str:
        return 'STEP ENERGY\n'

    def get_outfile_metadata(self) -> str:
        return ''

    def _start(self) -> None:
        """``initialize_run``, undone on failure: the handles are closed and
        the ensemble is not marked initialized, so a retried ``run()`` sets
        up from scratch (headers included) instead of skipping the setup."""
        try:
            self.initialize_run()
        except BaseException:
            self.close_files()
            self._initialized = False
            raise

    @abstractmethod
    def _run(self) -> None:
        """Perform a single ensemble step. Implemented by subclasses."""
        raise NotImplementedError

    def run(self, steps: int) -> None:
        """Standard run loop: initialize_run, loop _run, finalize_run.

        Runs ``steps`` steps; after ``restart()``, only the ones that the
        stopped run had not done (``steps`` counts from its step 0)."""
        steps = max(0, steps - self._restart_step)
        self._restart_step = 0
        with sigterm_as_exit():
            self._start()
            try:
                for _ in range(steps):
                    self._run()
            finally:
                self.finalize_run()


def random_streams(ensemble):
    """``(role, generator)`` for every random stream of one replica: the
    acceptance test, the move selector, each move and each cell. A generator
    is a ``random.Random`` or a numpy ``Generator``."""
    for name in ('rng_acceptance', '_rng_acceptance'):
        rng = getattr(ensemble, name, None)
        if rng is not None:
            yield 'acceptance', rng.random
    selector = getattr(ensemble, 'move_selector', None)
    cells = list(getattr(ensemble, 'cells', None) or [])
    if selector is not None:
        yield 'move selector', selector.rng.random
        for k, move in enumerate(selector.move_list):
            yield f'move {k}', move.rng.random
            np_rng = getattr(move, '_np_rng', None)
            if np_rng is not None:
                yield f'move {k} numpy', np_rng
            cell = getattr(move, 'cell', None)
            if cell is not None and all(cell is not c for c in cells):
                cells.append(cell)
    for k, cell in enumerate(cells):
        rng = getattr(cell, '_rng', None)
        if rng is not None:
            yield f'cell {k}', rng


def reseed_streams(ensemble, salt: int) -> None:
    """Reseed every random stream of ``ensemble`` from its current state and
    ``salt``. A restarted run builds its streams from the seeds of the first
    run; without this it would replay that run's random numbers. The result
    depends only on the seeds and ``salt``, so a restart stays repeatable."""
    for _, rng in random_streams(ensemble):
        reseed(rng, salt)


def reseed(rng, salt: int) -> None:
    """Reseed one ``random.Random`` or numpy ``Generator`` in place."""
    if isinstance(rng, random.Random):
        rng.seed((rng.getrandbits(64) << 64) | int(salt))
    else:
        seq = np.random.SeedSequence([int(rng.integers(2 ** 63)), int(salt)])
        rng.bit_generator.state = type(rng.bit_generator)(seq).state


def read_last_frame(path: str) -> Atoms:
    """The last complete frame of an extended-XYZ file that mcpy wrote.

    A run killed during a write leaves a torn last frame. It is cut off the
    file (with a warning), so frames appended later keep the file readable.
    Raises ``ValueError`` when the file holds no complete frame."""
    end = _complete_frames_end(path)
    if end < os.path.getsize(path):
        logger.warning('%s ends in a torn frame; cutting it off and '
                       'restarting from the last complete frame', path)
        with open(path, 'r+b') as fh:
            fh.truncate(end)
    if end == 0:
        raise ValueError(f'{path} holds no complete frame')
    return ase.io.read(path, index=-1, format='extxyz')


def _complete_frames_end(path: str) -> int:
    """Byte offset just after the last complete XYZ frame in ``path``."""
    good = 0
    with open(path, 'rb') as fh:
        while True:
            line = fh.readline()
            try:
                n_atoms = int(line)
            except ValueError:
                return good
            body = [fh.readline() for _ in range(n_atoms + 1)]
            if not all(b.endswith(b'\n') for b in body):
                return good
            good = fh.tell()


def frame_energy(frame: Atoms) -> float:
    """The energy that ``write_xyz`` stored with a frame (ASE moves it from
    the comment line to a single-point calculator on read)."""
    if frame.calc is not None and 'energy' in frame.calc.results:
        return float(frame.calc.results['energy'])
    return float(frame.info['energy'])


def drop_rows_after(path: Optional[str], step: int, row_step) -> None:
    """Remove the rows of a log whose step is past ``step``. ``row_step``
    maps a line to its step, or to None for a line that is not a row (a
    header). A restart continues from ``step``, so later rows describe steps
    that the run does again. The file is replaced in one step."""
    if path is None or not os.path.exists(path):
        return
    with open(path) as fh:
        lines = fh.readlines()
    keep = [line for line in lines
            if (s := row_step(line)) is None or s <= step]
    if len(keep) == len(lines):
        return
    with open(f'{path}.tmp', 'w') as fh:
        fh.writelines(keep)
    os.replace(f'{path}.tmp', path)


@contextlib.contextmanager
def sigterm_as_exit():
    """Raise ``SystemExit`` on SIGTERM for the duration of a run.

    ``kill``, ``scancel`` and a wall-time limit send SIGTERM. Its default
    action ends Python without unwinding, so no ``finally`` block runs and
    the last row, the last frame and the global minimum are never written.
    Inside this context the run stops as it does on Ctrl-C. A handler that
    the application installed is left alone, and so is any thread but the
    main one, where handlers cannot be set.
    """
    def _exit(signum, frame):
        raise SystemExit(128 + signum)

    if (threading.current_thread() is not threading.main_thread()
            or signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL):
        yield
        return
    signal.signal(signal.SIGTERM, _exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def write_global_minimum(path, bests, per_rung, logger):
    """Write the replicas' best configurations to ``path``.

    ``bests`` holds one ``(score, energy, atoms)`` per replica, or ``None``
    for a replica that recorded no minimum. Scores are grand potentials
    Omega = E - sum mu N under each replica's own mu, so they are comparable
    only when all replicas share mu: then the single lowest-score frame is
    written. When their mu differ (``per_rung=True``: a mu ladder, or a T
    ladder whose factory sets mu per rung) no common score exists, so one
    frame per rung is written in rung order, each tagged with its ``rank``
    and own-mu score.

    Shared by ``ReplicaExchange`` (which gathers over MPI ranks) and
    ``BatchedReplicaExchange`` (which reads its in-process replicas). Both
    call it at every write interval, so the frames go to a temporary file
    that then replaces ``path``: a kill during the write keeps the old file.
    """
    entries = [(rank, best) for rank, best in enumerate(bests) if best is not None]
    if not entries:
        return
    if not per_rung:
        entries = [min(entries, key=lambda kv: kv[1][0])]
    try:
        with open(f'{path}.tmp', 'w') as fh:
            for rank, (score, energy, atoms) in entries:
                write_xyz(atoms, energy, fh, extra={'rank': rank, 'score': round(score, 6)})
        os.replace(f'{path}.tmp', path)
    except OSError:
        logger.exception('Error writing global minimum to %s', path)


def write_xyz(atoms, energy, file_or_path, extra: dict = None):
    """
    Write one extended-XYZ frame to an open file handle or a path (append
    mode), through ``ase.io.write`` so pbc, FixAtoms (``move_mask``), tags
    and per-atom arrays such as ``molecule_id`` survive a restart.

    ``extra`` adds info keys to the comment line next to ``energy``
    (e.g. ``{'step': 42, 'score': -1.234}``).

    ``parallel=False``: under MPI, ASE writes on rank 0 only by default,
    but every rank of ``ReplicaExchange`` owns its own files.
    """
    frame = atoms.copy()  # no calculator: its stale results must not be written
    frame.info = {'energy': float(energy), **(extra or {})}
    if isinstance(file_or_path, str):
        with open(file_or_path, 'a') as xyz_file:
            ase.io.write(xyz_file, frame, format='extxyz', parallel=False)
    else:
        ase.io.write(file_or_path, frame, format='extxyz', parallel=False)
        file_or_path.flush()


@functools.lru_cache(maxsize=None)
def code_version() -> str:
    """The mcpy version, plus ``git describe`` when mcpy runs from a git
    checkout (an editable install), for the outfile header."""
    from .. import __version__
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(os.path.join(root, '.git')):
        return __version__
    try:
        out = subprocess.run(['git', 'describe', '--tags', '--always', '--dirty'],
                             cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return __version__
    describe = out.stdout.strip()
    return f'{__version__} (git {describe})' if out.returncode == 0 and describe else __version__
