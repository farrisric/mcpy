"""The replica-exchange drivers (MPI with a fake communicator, and batched):
seeding checks, failures, logs and swap bookkeeping.

The swap rule itself is in test_mpi_re_acceptance.py and
test_batched_re_acceptance.py.
"""
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from mcpy.ensembles import replica_exchange as rex
from mcpy.ensembles.base_ensemble import BaseEnsemble
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.ensembles.replica_exchange import ReplicaExchange

from helpers import _batched_pair, _BatchLJ, _frames, _lj_gcmc, _lj_replica


class _RecordingComm:
    """Stands in for MPI.COMM_WORLD on one simulated rank."""

    def __init__(self, rank=0, size=1, partner_state=None, bcast_value=0.0):
        self.rank, self.size = rank, size
        self.partner_state = partner_state
        self.bcast_value = bcast_value
        self.sent, self.calls = [], []
        self.aborted = None
        self.sendrecv_error = None

    def Get_rank(self):
        return self.rank

    def allgather(self, obj):
        return [obj] * self.size

    def Get_size(self):
        return self.size

    def bcast(self, obj, root=0):
        self.calls.append('bcast')
        return self.bcast_value

    def sendrecv(self, sendobj, dest, source):
        self.calls.append('sendrecv')
        if self.sendrecv_error:
            raise self.sendrecv_error
        pickle.dumps(sendobj)  # what mpi4py does with it
        return self.partner_state

    def send(self, obj, dest):
        self.sent.append(obj)

    def recv(self, source):
        return 0.0

    def gather(self, obj, root=0):
        self.calls.append('gather')
        return [obj] * self.size

    def Abort(self, code=0):
        self.aborted = code


def _mpi_re(monkeypatch, comm, gcmc, **kw):
    monkeypatch.setattr(rex, 'MPI', SimpleNamespace(COMM_WORLD=comm))
    kw.setdefault('temperatures', [1.0] * comm.size)
    return rex.ReplicaExchange(lambda **_: gcmc, **kw)


def _stub_gcmc():
    state = {'atoms': Atoms('Ag2'), 'energy': 0.0, 'beta': 1.0, 'mu': None}
    return SimpleNamespace(get_state=lambda: dict(state), set_state=lambda s: None,
                           exchange_attempts=0, exchange_successes=0)


# MPI ReplicaExchange (fake communicator)

def test_mpi_pairs_draw_independent_uniforms(monkeypatch):
    """Ranks 0 and 2 are both the lower partner under the even pairing; with
    identical per-rank seeds they sent the very same u to their partners."""
    sent = []
    for rank in (0, 2):
        comm = _RecordingComm(rank=rank, size=4, partner_state=_stub_gcmc().get_state())
        re = _mpi_re(monkeypatch, comm, _stub_gcmc())
        re.do_exchange()
        sent += comm.sent
    assert len(sent) == 2 and sent[0] != sent[1]


def test_mpi_exchange_attempt_counted_only_with_partner(monkeypatch):
    g = _stub_gcmc()
    comm = _RecordingComm(rank=0, size=2, partner_state=_stub_gcmc().get_state(),
                          bcast_value=0.9)  # odd pairing: rank 0 sits out
    re = _mpi_re(monkeypatch, comm, g)
    re.do_exchange()
    assert g.exchange_attempts == 0
    comm.bcast_value = 0.0  # even pairing: (0, 1)
    re.do_exchange()
    assert g.exchange_attempts == 1


def test_mpi_run_without_partner_counts_no_attempts(monkeypatch, tmp_path):
    g = _lj_gcmc()
    comm = _RecordingComm(rank=0, size=1)
    re = _mpi_re(monkeypatch, comm, g, gcmc_steps=6, exchange_interval=2,
                 outfile=str(tmp_path / 're.log'),
                 global_minimum_file=str(tmp_path / 'gm.xyz'))
    re.run()
    assert g.exchange_attempts == 0


def _batched(tmp_path, mus=(-4.0, -3.9, -3.8, -3.7), **kw):
    calc = _BatchLJ()

    def factory(mu, rank):
        return _lj_gcmc(mu=mu['H'], seed=10 * rank, tmp_path=tmp_path,
                        tag=f'r{rank}', calc=calc)
    kw.setdefault('gcmc_steps', 30)
    kw.setdefault('exchange_interval', 2)
    return BatchedReplicaExchange(
        factory, calc, mus=[{'H': m} for m in mus],
        outfile=str(tmp_path / 're.log'), write_out_interval=5,
        global_minimum_file=str(tmp_path / 'gm.xyz'), seed=7, **kw)


# BatchedReplicaExchange on CPU

def test_batched_run_on_cpu_swaps_and_stays_consistent(tmp_path):
    pt = _batched(tmp_path)
    swaps = []
    orig = pt._swap_states
    pt._swap_states = lambda i, j: (swaps.append((i, j)), orig(i, j))
    pt.run()
    assert swaps, 'no swap accepted'
    assert len({id(r.atoms) for r in pt.replicas}) == pt.n_replicas
    for k, r in enumerate(pt.replicas):
        assert r.n_atoms == len(r.atoms)
        assert r.E_old == pytest.approx(pt.calculator.get_potential_energy(r.atoms))
        assert r._step == 30
        assert _frames(tmp_path / f'r{k}.xyz') == 31  # steps 0..30, no dup
    # mu ladder: one global-minimum frame per rung
    assert len(read(str(tmp_path / 'gm.xyz'), index=':', format='extxyz')) == 4


def test_batched_step0_uses_batched_energy(tmp_path):
    """The step-0 outfile row must carry the batched (shared-calculator)
    energy, not the unbatched construction-time one."""
    pt = _batched(tmp_path, gcmc_steps=0)
    shift = 1000.0
    real = pt.calculator.get_potential_energies
    pt.calculator.get_potential_energies = lambda al: real(al) + shift
    pt.run()
    for k in range(pt.n_replicas):
        row0 = next(line for line in open(tmp_path / f'r{k}.out')
                    if line.split() and line.split()[0] == '0')
        assert float(row0.split()[2]) > shift / 2


def test_batched_setup_error_restores_logger_and_closes_files(tmp_path):
    pt = _batched(tmp_path)
    quiet = logging.getLogger(BaseEnsemble.__module__)
    quiet.setLevel(logging.NOTSET)

    def boom():
        raise RuntimeError('setup failed')
    pt.replicas[2].initialize_run = boom
    with pytest.raises(RuntimeError, match='setup failed'):
        pt.run()
    assert quiet.level == logging.NOTSET
    assert all(r._traj_handle is None for r in pt.replicas)
    assert pt._outfile_handle is None


def test_dead_interior_pair_is_reported(caplog):
    """Pair (1, 2) never accepts; slots 1 and 2 still accept through their
    outer pairs, so a per-slot tally cannot see it."""
    pt = BatchedReplicaExchange.__new__(BatchedReplicaExchange)
    pt.logger = logging.getLogger('mcpy.ensembles.batched_replica_exchange')
    pt.n_replicas = 4
    pt.mus = [{}] * 4
    pt.exchange_attempts = [0] * 4
    pt.exchange_successes = [0] * 4
    pt.pair_attempts = [0] * 3
    pt.pair_successes = [0] * 3
    pt._mid_attempts = [0] * 3
    pt._mid_successes = [0] * 3
    pt.rng = SimpleNamespace(get_uniform=iter([0.1, 0.9] * 50).__next__)
    pt._accept_swap = lambda i, j: (i, j) != (1, 2)
    pt._swap_states = lambda i, j: None
    for _ in range(100):
        pt._attempt_exchanges()
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    msgs = [r.getMessage() for r in caplog.records]
    assert any('1-2' in m for m in msgs), msgs
    assert not any('0-1' in m or '2-3' in m for m in msgs), msgs


# ReplicaExchange ladder validation (bug: passing both built the GCMC twice
# and silently used the equal-T mu criterion on a T ladder)

def test_replica_exchange_rejects_both_ladders():
    with pytest.raises(ValueError, match='not both'):
        ReplicaExchange(gcmc_factory=lambda **kw: None,
                        temperatures=[300.0, 400.0],
                        mus=[{'H': 0.0}, {'H': -0.1}])


# Replicas with identical seeds are refused


def test_batched_rejects_one_shared_move_seed(tmp_path):
    """Everything else differs; one move stream is common to both."""
    calc = _BatchLJ()

    def factory(mu, rank):
        g = _lj_replica(mu=mu['H'], seed=10 * rank, calc=calc)
        g.move_selector.move_list[2].rng.random.seed(99)
        return g
    with pytest.raises(ValueError, match='move 2'):
        BatchedReplicaExchange(factory, calc, mus=[{'H': -4.0}, {'H': -3.8}],
                               outfile=None, global_minimum_file=None)


class _FakeComm:
    """MPI.COMM_WORLD for one simulated rank. ``others`` are the objects the
    other ranks contribute to an allgather."""

    def __init__(self, rank=0, size=1, others=(), partner_state=None):
        self.rank, self.size = rank, size
        self.others = list(others)
        self.partner_state = partner_state
        self.aborted = None
        self.gathers_left = None  # then every gather raises

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def allgather(self, obj):
        out = list(self.others)
        out.insert(self.rank, obj)
        return out

    def gather(self, obj, root=0):
        if self.gathers_left is not None:
            if self.gathers_left == 0:
                raise RuntimeError('link down')
            self.gathers_left -= 1
        return [obj] * self.size

    def bcast(self, obj, root=0):
        return 0.0  # even pairing: (0, 1)

    def sendrecv(self, sendobj, dest, source):
        return self.partner_state

    def send(self, obj, dest):
        pass

    def recv(self, source):
        return 0.0  # u = 0: accept any swap with p > 0

    def Abort(self, code=0):
        self.aborted = code


def _mpi(monkeypatch, comm, factory, **kw):
    monkeypatch.setattr(rex, 'MPI', SimpleNamespace(COMM_WORLD=comm))
    if 'mus' not in kw:
        kw.setdefault('temperatures', [2.0] * comm.size)
    return rex.ReplicaExchange(factory, **kw)


# A failure on one rank aborts the job, never hangs it

def test_mpi_factory_error_aborts(monkeypatch):
    comm = _FakeComm()

    def factory(**_):
        raise RuntimeError('bad checkpoint')
    with pytest.raises(RuntimeError, match='bad checkpoint'):
        _mpi(monkeypatch, comm, factory)
    assert comm.aborted == 1


def test_mpi_final_collective_error_aborts(monkeypatch, tmp_path):
    comm = _FakeComm()
    re = _mpi(monkeypatch, comm, lambda **_: _lj_replica(), gcmc_steps=2,
              outfile=str(tmp_path / 're.log'),
              global_minimum_file=str(tmp_path / 'gm.xyz'))
    # The step-0 table and global minimum; the end-of-run ones fail.
    comm.gathers_left = 3
    with pytest.raises(RuntimeError, match='link down'):
        re.run()
    assert comm.aborted == 1


# MPI ladder checks


@pytest.mark.parametrize('ladder', ['temperatures', 'mus'])
@pytest.mark.parametrize('values', [[], [2.0, 2.5]])
def test_mpi_ladder_length_mismatch_raises_value_error(monkeypatch, ladder, values):
    if ladder == 'mus':
        values = [{'H': v} for v in values]
    with pytest.raises(ValueError, match='MPI ranks'):
        _mpi(monkeypatch, _FakeComm(), lambda **_: _lj_replica(), **{ladder: values})


# A swapped-in configuration is scored for the minimum

def test_batched_swap_records_minimum(tmp_path):
    pt = _batched_pair(tmp_path)
    ri, rj = pt.replicas
    ri._best_score = rj._best_score = np.inf
    pt._swap_states(0, 1)
    assert ri._best_score == pytest.approx(ri._minimum_score(ri.atoms, ri.E_old))
    assert rj._best_score == pytest.approx(rj._minimum_score(rj.atoms, rj.E_old))


def test_mpi_swap_records_minimum(monkeypatch):
    partner = _lj_replica(seed=10, n=3)
    comm = _FakeComm(rank=0, size=2, others=[rex.rng_fingerprint(partner)],
                     partner_state=partner.get_state())
    g = _lj_replica(seed=0, n=12)
    re = _mpi(monkeypatch, comm, lambda **_: g, temperatures=[2.0, 2.0])
    g._best_score = np.inf
    assert re.do_exchange()
    assert len(g.atoms) == 3
    assert g._best_score == pytest.approx(g._minimum_score(g.atoms, g.E_old))


# A second run() appends to the replica-exchange log

def test_batched_second_run_appends_log(tmp_path):
    pt = _batched_pair(tmp_path)
    pt.run()
    first = open(tmp_path / 're.log').read()
    pt.run()
    log = open(tmp_path / 're.log').read()
    assert log.startswith(first)
    assert log.count('Batched Replica Exchange') == 1
    assert len(log) > len(first)


def test_mpi_second_run_appends_log(monkeypatch, tmp_path):
    re = _mpi(monkeypatch, _FakeComm(), lambda **_: _lj_replica(), gcmc_steps=4,
              write_out_interval=2, outfile=str(tmp_path / 're.log'),
              global_minimum_file=None)
    re.run()
    first = open(tmp_path / 're.log').read()
    re.run()
    log = open(tmp_path / 're.log').read()
    assert log.startswith(first)
    assert log.count('Replica Exchange Monte Carlo') == 1
    assert len(log) > len(first)


# Startup re-relaxation refreshes volumes

def test_rebatch_refreshes_cell_volumes(tmp_path):
    pt = _batched_pair(tmp_path)
    seen = []
    for r in pt.replicas:
        r.calculate_cells_volume = (
            lambda atoms: seen.append(atoms.get_positions().copy()))
    real = pt.calculator.get_potential_energies

    def relax_in_place(atoms_list):
        for a in atoms_list:
            a.positions += 0.1
        return real(atoms_list)
    pt.calculator.get_potential_energies = relax_in_place
    pt._rebatch_initial_energies()
    assert len(seen) == 2
    for pos, r in zip(seen, pt.replicas):
        np.testing.assert_array_equal(pos, r.atoms.get_positions())


# A swap exchanges the whole configuration state

def test_swap_states_exchanges_configuration(tmp_path):
    calc = _BatchLJ()

    def factory(mu, rank):
        return _lj_replica(mu=mu['H'], seed=10 * rank, n=(12, 3)[rank], calc=calc,
                           radii={'H': 1.0})
    pt = BatchedReplicaExchange(factory, calc, mus=[{'H': -4.0}, {'H': -3.8}],
                                outfile=None, global_minimum_file=None)
    ri, rj = pt.replicas
    ri.E_old, rj.E_old = -10.0, -1.0
    atoms_i, atoms_j = ri.atoms, rj.atoms
    vol_i, vol_j = ri.cells[0].get_volume(), rj.cells[0].get_volume()
    assert vol_j - vol_i > 20.0  # 9 fewer exclusion spheres in rj
    pt._swap_states(0, 1)
    assert ri.atoms is atoms_j and rj.atoms is atoms_i
    assert (ri.E_old, rj.E_old) == (-1.0, -10.0)
    assert (ri.n_atoms, rj.n_atoms) == (3, 12)
    assert ri.cells[0].get_volume() == pytest.approx(vol_j, abs=6.0)
    assert rj.cells[0].get_volume() == pytest.approx(vol_i, abs=6.0)


def test_mpi_rejects_identically_seeded_ranks(monkeypatch):
    other = rex.rng_fingerprint(_lj_replica(seed=0))
    comm = _FakeComm(rank=0, size=2, others=[other])
    with pytest.raises(ValueError, match='random state'):
        _mpi(monkeypatch, comm, lambda **_: _lj_replica(seed=0))


def test_mpi_sendrecv_error_propagates(monkeypatch):
    comm = _RecordingComm(rank=0, size=2, partner_state=_stub_gcmc().get_state())
    comm.sendrecv_error = RuntimeError('link down')
    re = _mpi_re(monkeypatch, comm, _stub_gcmc())
    with pytest.raises(RuntimeError, match='link down'):
        re.do_exchange()
