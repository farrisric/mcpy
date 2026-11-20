"""What a stopped run leaves behind.

SIGTERM, a kill during a write and an interrupt inside a move must not lose
the output or leave a trial that was never scored. All CPU-only and
torch-free; the two signal tests start a short child process.
"""
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

import mcpy
from mcpy.cell import Cell, NullCell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.ensembles import replica_exchange as rex
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector
from mcpy.moves.base_move import BaseMove

from helpers import _batched_pair, _gcmc, _lj_gcmc


class _Watching:
    """Zero energy; records at each call whether ``path`` exists."""

    def __init__(self, path=None):
        self.path, self.seen = path, []

    def get_potential_energy(self, atoms):
        self.seen.append(self.path is not None and os.path.exists(self.path))
        return 0.0

    def get_potential_energies(self, atoms_list):
        return np.array([self.get_potential_energy(a) for a in atoms_list])


def _box_gcmc(calc, rank=0, mu=-1.0, T=1.0, atoms=None, moves=None, **files):
    atoms = atoms if atoms is not None else Atoms(cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=10 + rank)
    if moves is None:
        moves = [InsertionMove(cell, ['H'], seed=20 + rank),
                 DeletionMove(cell, ['H'], seed=30 + rank)]
    files = {'traj_file': None, 'outfile': None, **files}
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ', calculator=calc, mu={'H': mu},
        species=['H'], temperature=T,
        move_selector=MoveSelector([1] * len(moves), moves, seed=40 + rank),
        random_seed=50 + rank, **files)


# --------------------------------------------------------------------------
# SIGTERM ends a run through its finally blocks; the global minimum is on
# disk during the run
# --------------------------------------------------------------------------

_CHILD = """
import time

from ase import Atoms

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector


class Slow:
    calls = 0

    def get_potential_energy(self, atoms):
        Slow.calls += 1
        if Slow.calls == 30:
            open('ready', 'w').close()
        time.sleep(0.002)
        return 0.0

    def get_potential_energies(self, atoms_list):
        return [self.get_potential_energy(a) for a in atoms_list]


def gcmc(mu=-1.0, rank=0):
    atoms = Atoms(cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=10 + rank)
    moves = [InsertionMove(cell, ['H'], seed=20 + rank),
             DeletionMove(cell, ['H'], seed=30 + rank)]
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ', calculator=Slow(), mu={'H': mu},
        species=['H'], temperature=1.0,
        move_selector=MoveSelector([1, 1], moves, seed=40 + rank),
        random_seed=50 + rank, traj_file=f't{rank}.xyz', outfile=f'o{rank}.out',
        trajectory_write_interval=10**9, outfile_write_interval=10**9)


"""
_CHILD_MAIN = {
    'gcmc': 'gcmc().run(10**9)\n',
    'batched': (
        'BatchedReplicaExchange(\n'
        "    lambda mu, rank: gcmc(mu['H'], rank), Slow(),\n"
        "    mus=[{'H': -1.0}, {'H': -0.9}], gcmc_steps=10**9,\n"
        "    write_out_interval=10**9, outfile='re.log',\n"
        "    global_minimum_file='gm.xyz').run()\n"),
}


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX signals')
@pytest.mark.parametrize('kind', ['gcmc', 'batched'])
def test_sigterm_ends_a_run_like_ctrl_c(tmp_path, kind):
    """kill, scancel and a wall-time limit send SIGTERM, which by default
    ends Python without running any finally block. The run must stop as it
    does on Ctrl-C: last frame written, global minimum written."""
    script = tmp_path / 'child.py'
    script.write_text(_CHILD + _CHILD_MAIN[kind])
    package_root = os.path.dirname(os.path.dirname(mcpy.__file__))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [package_root, os.environ.get('PYTHONPATH')])))
    proc = subprocess.Popen([sys.executable, str(script)], cwd=tmp_path, env=env)
    try:
        deadline = time.time() + 60
        while not (tmp_path / 'ready').exists():
            assert proc.poll() is None, 'the child ended before the signal'
            assert time.time() < deadline, 'the child never reached the run loop'
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=60) == 128 + signal.SIGTERM
    finally:
        proc.kill()
    # Frames are written at step 0 and by finalize_run only.
    assert len(read(tmp_path / 't0.xyz', index=':')) == 2
    if kind == 'batched':
        assert (tmp_path / 'gm.xyz').exists()


def test_sigterm_handler_is_scoped_to_the_run_and_spares_the_application():
    from mcpy.ensembles.base_ensemble import sigterm_as_exit
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    with sigterm_as_exit():
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

    def application_handler(signum, frame):
        pass

    signal.signal(signal.SIGTERM, application_handler)
    try:
        with sigterm_as_exit():
            assert signal.getsignal(signal.SIGTERM) is application_handler
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_batched_global_minimum_is_on_disk_during_the_run(tmp_path):
    """global_minimum.xyz is rewritten at every write interval, so SIGKILL
    or the OOM killer cannot lose the best structures of a long run."""
    gm = str(tmp_path / 'gm.xyz')
    calc = _Watching(gm)
    re = BatchedReplicaExchange(
        lambda mu, rank: _box_gcmc(calc, rank, mu=mu['H']), calc,
        mus=[{'H': -1.0}, {'H': -0.9}], gcmc_steps=6, write_out_interval=2,
        outfile=None, global_minimum_file=gm)
    re.run()
    assert calc.seen[-1]


class _OneRankComm:
    """MPI.COMM_WORLD of a one-rank job."""

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def allgather(self, obj):
        return [obj]

    def gather(self, obj, root=0):
        return [obj]

    def Abort(self, code=0):
        raise AssertionError('the run aborted')


def test_mpi_global_minimum_is_on_disk_during_the_run(tmp_path, monkeypatch):
    gm = str(tmp_path / 'gm.xyz')
    calc = _Watching(gm)
    monkeypatch.setattr(rex, 'MPI', SimpleNamespace(COMM_WORLD=_OneRankComm()))
    re = rex.ReplicaExchange(
        lambda T, rank: _box_gcmc(calc, rank, T=T), temperatures=[1.0], gcmc_steps=6,
        write_out_interval=2, outfile=None, global_minimum_file=gm)
    re.run()
    assert calc.seen[-1]


def test_global_minimum_is_replaced_in_one_step(tmp_path, monkeypatch):
    """A kill during the write must leave the previous file intact: the
    new content goes to a temporary file that then replaces the old one."""
    from mcpy.ensembles import base_ensemble
    gm = tmp_path / 'gm.xyz'
    gm.write_text('previous best\n')
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[5] * 3, pbc=True)

    def killed(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(base_ensemble, 'write_xyz', killed)
    with pytest.raises(KeyboardInterrupt):
        base_ensemble.write_global_minimum(str(gm), [(0.0, 0.0, atoms)], False, None)
    assert gm.read_text() == 'previous best\n'


# --------------------------------------------------------------------------
# an interrupt inside a move leaves no unscored trial, so the frame that
# finalize_run writes belongs to the logged energy
# --------------------------------------------------------------------------

class _HalfDone(BaseMove):
    """Mutates the atoms, then is interrupted."""

    def __init__(self):
        super().__init__(NullCell(), ['H'], seed=1)

    def do_trial_move(self, atoms):
        atoms.positions += 1.0
        raise KeyboardInterrupt


def test_interrupted_move_leaves_no_unscored_trial():
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[6] * 3, pbc=True)
    g = _box_gcmc(_Watching(), atoms=atoms, moves=[_HalfDone()])
    with pytest.raises(KeyboardInterrupt):
        g.do_gcmc_step()
    np.testing.assert_array_equal(g.atoms.positions, [[1, 1, 1]])


def test_batched_interrupted_proposal_restores_the_earlier_replicas():
    calc = _Watching()

    def factory(mu, rank):
        atoms = Atoms('H2', positions=[[1, 1, 1], [4, 4, 4]], cell=[6] * 3, pbc=True)
        return _box_gcmc(calc, rank, mu=mu['H'], atoms=atoms,
                         moves=[DisplacementMove(['H'], seed=60 + rank, max_displacement=0.5)])

    re = BatchedReplicaExchange(factory, calc, mus=[{'H': -1.0}, {'H': -0.9}],
                                outfile=None, global_minimum_file=None)
    before = re.replicas[0].atoms.positions.copy()

    def interrupted(atoms):
        raise KeyboardInterrupt

    re.replicas[1].move_selector.do_trial_move = interrupted
    with pytest.raises(KeyboardInterrupt):
        re._batched_single_move([0, 1])
    np.testing.assert_array_equal(re.replicas[0].atoms.positions, before)


class _Interrupting:
    def __init__(self):
        self.calls = 0

    def get_potential_energy(self, atoms):
        self.calls += 1
        if self.calls > 1:
            raise KeyboardInterrupt
        return 0.0


# An exception in the energy call restores the snapshot

def test_interrupted_energy_call_leaves_no_unscored_trial():
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
              calc=_Interrupting())
    with pytest.raises(KeyboardInterrupt):
        g.do_gcmc_step()
    assert len(g.atoms) == 1
    np.testing.assert_array_equal(g.atoms.positions, [[1, 1, 1]])


def _snapshot(pt):
    return [(r.atoms.get_positions().copy(), r.atoms.get_chemical_symbols())
            for r in pt.replicas]


# An energy error restores every pending replica


def test_batched_short_energy_list_raises_and_restores(tmp_path):
    pt = _batched_pair(tmp_path)
    before = _snapshot(pt)
    real = pt.calculator.get_potential_energies
    pt.calculator.get_potential_energies = lambda al: real(al)[:1]
    with pytest.raises(ValueError):
        pt._batched_single_move([0, 1])
    for (pos, sym), r in zip(before, pt.replicas):
        assert r.atoms.get_chemical_symbols() == sym
        np.testing.assert_array_equal(r.atoms.get_positions(), pos)


# --------------------------------------------------------------------------
# restart() continues a stopped run from its files
# --------------------------------------------------------------------------

def _rows(path):
    return [int(line.split()[0]) for line in open(path) if line[:1].isdigit()]


def test_restart_continues_a_killed_run(tmp_path):
    """A run killed after step 10, whose last frames (step 10) and a torn
    one never reached the trajectory: restart() goes back to the step-8
    frame, cuts the torn frame, drops the stale outfile rows, keeps the
    minimum and draws new random numbers."""
    def build():
        return _lj_gcmc(tmp_path=tmp_path, trajectory_write_interval=4,
                        minima_file=str(tmp_path / 'min.xyz'))
    first = build()
    first.run(10)
    traj = tmp_path / 'g.xyz'
    frames = read(str(traj), ':')
    assert [f.info['step'] for f in frames] == [0, 4, 8, 10]
    text = traj.read_text()
    traj.write_text(text[:text.rindex(f'{len(frames[-1])}\n')] + '3\nLattice=')
    n_minima = len(read(str(tmp_path / 'min.xyz'), ':'))

    g = build()
    fresh = g.rng_acceptance.random.getstate()
    assert g.restart() == 8
    assert g.rng_acceptance.random.getstate() != fresh
    assert g._best_score == pytest.approx(first._best_score)
    assert len(g.atoms) == len(frames[2])
    assert _rows(tmp_path / 'g.out') == list(range(9))
    g.run(12)  # the total, as in the first run's script
    assert _rows(tmp_path / 'g.out') == list(range(13))
    assert open(tmp_path / 'g.out').read().count('Grand Canonical') == 1
    assert [f.info['step'] for f in read(str(traj), ':')] == [0, 4, 8, 12]
    assert len(read(str(tmp_path / 'min.xyz'), ':')) >= n_minima


def test_batched_restart_runs_to_the_total_and_continues_the_log(tmp_path):
    _batched_pair(tmp_path, gcmc_steps=6).run()
    pt = _batched_pair(tmp_path, gcmc_steps=10)
    assert pt.restart() == 6
    pt.run()
    assert [r._step for r in pt.replicas] == [10, 10]
    log = (tmp_path / 're.log').read_text()
    assert log.count('Batched Replica Exchange') == 1
    steps = [rex.re_row_step(line) for line in log.splitlines()]
    steps = [s for s in steps if s is not None]
    # Rows of the first run end at step 5; the restart goes on from 7.
    assert steps == sorted(steps) and steps[-1] == 9 and 7 in steps


def test_restarted_run_matches_the_straight_run_up_to_the_stop(tmp_path):
    """Same seeds: a straight 120-step run, and the same run stopped at step
    60 and continued with restart(). Steps 0-60 are bit-identical. The
    restart recomputes the step-60 energy to 1e-5 (the trajectory stores
    positions to 1e-8 A, so not to the last bit). The continuation draws
    new random numbers: it does not replay the first run."""
    def build(tag):
        return _lj_gcmc(mu=-4.0, traj_file=str(tmp_path / f'{tag}.xyz'),
                        outfile=str(tmp_path / f'{tag}.out'))

    def rows(tag):
        return [line.split()[:3] for line in open(tmp_path / f'{tag}.out')
                if line[:1].isdigit()]

    build('straight').run(120)
    first = build('stopped')
    first.run(60)
    g = build('stopped')
    g.restart()
    assert g.E_old == pytest.approx(first.E_old, abs=1e-5)
    g.run(120)
    straight, restarted = rows('straight'), rows('stopped')
    assert [int(r[0]) for r in restarted] == list(range(121))
    assert restarted[:61] == straight[:61]
    assert restarted[61:] != straight[61:]
