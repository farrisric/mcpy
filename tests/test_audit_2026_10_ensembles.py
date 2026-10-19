"""Regression tests for the 2026-10 ensemble audit (ensembles + reference_mu).

All CPU-only: Lennard-Jones / EMT / stub calculators, and a fake MPI
communicator for ReplicaExchange (one object per simulated rank).
"""
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io import read

from mcpy.cell import Cell
from mcpy.cell.null_cell import NullCell
from mcpy.ensembles import CanonicalEnsemble, GrandCanonicalEnsemble
from mcpy.ensembles import replica_exchange as rex
from mcpy.ensembles.base_ensemble import BaseEnsemble
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector
from mcpy.utils import derive_mu_gas

BOX = 9.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _lj_atoms(n=12):
    grid = np.array([(x, y, z) for x in range(4) for y in range(4) for z in range(4)])
    return Atoms('H' * n, positions=1.6 * grid[:n] + 0.5, cell=[BOX] * 3, pbc=True)


class _BatchLJ:
    """LJ calculator with the batched interface BatchedReplicaExchange needs."""

    def __init__(self):
        self.lj = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0, smooth=False)

    def get_potential_energy(self, atoms):
        a = atoms.copy()
        a.calc = self.lj
        return a.get_potential_energy()

    def get_potential_energies(self, atoms_list):
        return np.array([self.get_potential_energy(a) for a in atoms_list])


def _lj_gcmc(mu=-4.0, T=2.0, seed=0, tmp_path=None, tag='g', calc=None, **kw):
    atoms = _lj_atoms()
    cell = Cell(atoms, seed=seed + 10)
    ms = MoveSelector(
        [1, 1, 1],
        [InsertionMove(cell, species=['H'], seed=seed + 1),
         DeletionMove(cell, species=['H'], seed=seed + 2),
         DisplacementMove(species=['H'], max_displacement=0.3, seed=seed + 3)],
        seed=seed + 4)
    files = dict(traj_file=None, outfile=None)
    if tmp_path is not None:
        files = dict(traj_file=str(tmp_path / f'{tag}.xyz'),
                     outfile=str(tmp_path / f'{tag}.out'))
    files.update(kw)
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=calc or _BatchLJ(), mu={'H': mu},
        species=['H'], units_type='LJ', temperature=T, move_selector=ms,
        random_seed=seed + 5, **files)


def _rows(path):
    return [int(line.split()[0]) for line in open(path)
            if line.split() and line.split()[0].isdigit()]


def _frames(path):
    return len(read(str(path), index=':', format='extxyz'))


class _FakeComm:
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


# --------------------------------------------------------------------------
# B3: get_state returns a detached, picklable copy
# --------------------------------------------------------------------------

def test_gcmc_get_state_is_detached_picklable_copy():
    g = _lj_gcmc(calc=EMT())
    g.atoms.calc = EMT()
    g.atoms.get_potential_energy()
    state = g.get_state()
    assert state['atoms'] is not g.atoms
    assert state['atoms'].calc is None
    pickle.dumps(state)


def test_canonical_get_state_is_detached_copy():
    from ase.optimize import LBFGS
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    mc = CanonicalEnsemble(atoms=atoms, calculator=EMT(), optimizer=LBFGS,
                           move_selector=MoveSelector([1], [DisplacementMove(
                               species=['Cu'], seed=1)]),
                           random_seed=1, traj_file=None, outfile=None)
    mc._current_energy = mc.relax(mc.atoms)  # attaches the calculator
    state = mc.get_state()
    assert state['atoms'] is not mc.atoms
    assert state['atoms'].calc is None


# --------------------------------------------------------------------------
# B4 / B5: run() twice appends; no duplicated final frame
# --------------------------------------------------------------------------

def test_second_run_appends_without_duplicates(tmp_path):
    g = _lj_gcmc(tmp_path=tmp_path)
    g.run(3)
    g.run(3)
    out = tmp_path / 'g.out'
    assert open(out).read().count('Grand Canonical Ensemble') == 1
    assert _rows(out) == list(range(7))
    assert _frames(tmp_path / 'g.xyz') == 7


def test_finalize_does_not_duplicate_last_frame(tmp_path):
    g = _lj_gcmc(tmp_path=tmp_path)
    g.run(3)
    assert _frames(tmp_path / 'g.xyz') == 4  # steps 0..3


def test_finalize_still_writes_unaligned_last_frame(tmp_path):
    g = _lj_gcmc(tmp_path=tmp_path, trajectory_write_interval=2)
    g.run(3)
    assert _frames(tmp_path / 'g.xyz') == 3  # steps 0, 2 and the final 3


def test_minima_single_best_survives_second_run(tmp_path):
    g = _lj_gcmc(tmp_path=tmp_path, minima_file=str(tmp_path / 'm.xyz'),
                 minima_mode='w')
    g.run(2)
    g.run(1)
    assert _frames(tmp_path / 'm.xyz') == 1


# --------------------------------------------------------------------------
# P4: atomic mu keys do not count atoms that belong to molecules
# --------------------------------------------------------------------------

def test_species_count_skips_atoms_inside_molecules():
    atoms = Atoms('COO', positions=[[0, 0, 0], [0, 0, 1.1], [3, 3, 3]],
                  cell=[10] * 3)
    atoms.new_array('molecule_id', np.array([0, 0, -1]))
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[NullCell()], units_type='metal',
        calculator=SimpleNamespace(get_potential_energy=lambda a: 0.0),
        mu={'O': -1.0}, species=['O'], temperature=300.0, molecules={'CO': Atoms('CO')},
        move_selector=MoveSelector([1], [DisplacementMove(species=['O'], seed=1)]),
        random_seed=1, traj_file=None, outfile=None)
    assert g._species_count(atoms, 'O') == 1


# --------------------------------------------------------------------------
# MPI ReplicaExchange (fake communicator)
# --------------------------------------------------------------------------

def test_mpi_pairs_draw_independent_uniforms(monkeypatch):
    """Ranks 0 and 2 are both the lower partner under the even pairing; with
    identical per-rank seeds they sent the very same u to their partners."""
    sent = []
    for rank in (0, 2):
        comm = _FakeComm(rank=rank, size=4, partner_state=_stub_gcmc().get_state())
        re = _mpi_re(monkeypatch, comm, _stub_gcmc())
        re.do_exchange()
        sent += comm.sent
    assert len(sent) == 2 and sent[0] != sent[1]


def test_mpi_exchange_attempt_counted_only_with_partner(monkeypatch):
    g = _stub_gcmc()
    comm = _FakeComm(rank=0, size=2, partner_state=_stub_gcmc().get_state(),
                     bcast_value=0.9)  # odd pairing: rank 0 sits out
    re = _mpi_re(monkeypatch, comm, g)
    re.do_exchange()
    assert g.exchange_attempts == 0
    comm.bcast_value = 0.0  # even pairing: (0, 1)
    re.do_exchange()
    assert g.exchange_attempts == 1


def test_mpi_run_without_partner_counts_no_attempts(monkeypatch, tmp_path):
    g = _lj_gcmc()
    comm = _FakeComm(rank=0, size=1)
    re = _mpi_re(monkeypatch, comm, g, gcmc_steps=6, exchange_interval=2,
                 outfile=str(tmp_path / 're.log'),
                 global_minimum_file=str(tmp_path / 'gm.xyz'))
    re.run()
    assert g.exchange_attempts == 0


def test_mpi_sendrecv_error_propagates(monkeypatch):
    comm = _FakeComm(rank=0, size=2, partner_state=_stub_gcmc().get_state())
    comm.sendrecv_error = RuntimeError('link down')
    re = _mpi_re(monkeypatch, comm, _stub_gcmc())
    with pytest.raises(RuntimeError, match='link down'):
        re.do_exchange()


def test_mpi_run_error_aborts_without_collectives(monkeypatch, tmp_path):
    g = _lj_gcmc()

    def boom():
        raise RuntimeError('diverged')
    g._run = boom
    comm = _FakeComm(rank=0, size=1)
    re = _mpi_re(monkeypatch, comm, g, outfile=str(tmp_path / 're.log'),
                 global_minimum_file=str(tmp_path / 'gm.xyz'))
    with pytest.raises(RuntimeError, match='diverged'):
        re.run()
    assert comm.aborted is not None
    assert 'gather' not in comm.calls


def _bare_mpi_re():
    re = object.__new__(rex.ReplicaExchange)
    re.logger = logging.getLogger('test')
    re.rank = 0
    return re


def test_mpi_exchange_prob_rejects_nan(caplog):
    re = _bare_mpi_re()
    s1 = {'beta': 1.0, 'energy': float('nan'), 'mu': None, 'atoms': Atoms('Ag')}
    s2 = {'beta': 0.5, 'energy': 1.0, 'mu': None, 'atoms': Atoms('Ag')}
    with caplog.at_level(logging.WARNING, logger='test'):
        assert re._exchange_prob(s1, s2) == 0.0
    assert 'non-finite' in caplog.text


def test_mpi_exchange_prob_debug_is_lazy(caplog):
    re = _bare_mpi_re()
    s = {'beta': 1.0, 'energy': 1.0, 'mu': None, 'atoms': Atoms('Ag')}
    with caplog.at_level(logging.DEBUG, logger='test'):
        re._exchange_prob(s, dict(s))
    rec = caplog.records[-1]
    assert rec.args, 'debug message was pre-formatted (f-string)'


def test_mpi_exchange_prob_de_broglie_cross_term():
    """3 (N1 - N2) ln(L1 / L2) on top of (b2 - b1)(Phi2 - Phi1)."""
    re = _bare_mpi_re()
    s1 = {'beta': 1.0, 'energy': 10.0, 'mu': {'Ag': 1.0}, 'atoms': Atoms('Ag5'),
          'lambda_dbs': {'Ag': 2.0}}
    s2 = {'beta': 0.5, 'energy': 20.0, 'mu': {'Ag': 1.0}, 'atoms': Atoms('Ag18'),
          'lambda_dbs': {'Ag': 1.0}}
    # Phi1 = 5, Phi2 = 2 -> 1.5; cross = 3 * (5 - 18) * ln 2
    expected = np.exp(1.5 + 3 * (5 - 18) * np.log(2.0))
    assert re._exchange_prob(s1, s2) == pytest.approx(expected, rel=1e-12)


def test_mpi_global_minimum_one_frame_per_rung_on_mu_ladder(tmp_path):
    re = _bare_mpi_re()
    re.mus = [{'H': -4.0}, {'H': -3.0}]
    re._global_minimum_file = str(tmp_path / 'gm.xyz')
    best = [(-10.0, -6.0, Atoms('H')), (-20.0, -14.0, Atoms('H2'))]
    re.comm = SimpleNamespace(gather=lambda obj, root=0: best)
    re.gcmc = SimpleNamespace(_best_atoms=Atoms('H'), _best_score=-10.0,
                              _best_energy=-6.0)
    re._write_global_minimum()
    frames = read(re._global_minimum_file, index=':', format='extxyz')
    assert [len(f) for f in frames] == [1, 2]


# --------------------------------------------------------------------------
# BatchedReplicaExchange on CPU
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# CanonicalEnsemble: T = 0 and the relaxation step cap
# --------------------------------------------------------------------------

def _canon(**kw):
    from ase.optimize import LBFGS
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    kw.setdefault('optimizer', LBFGS)
    return CanonicalEnsemble(atoms=atoms, calculator=EMT(),
                             move_selector=MoveSelector([1], [DisplacementMove(
                                 species=['Cu'], seed=1)]),
                             random_seed=1, traj_file=None, outfile=None, **kw)


def test_canonical_zero_temperature_is_greedy():
    mc = _canon(temperature=0)
    assert mc._acceptance_condition(-0.1) is True
    assert not mc._acceptance_condition(1e-9)


def test_canonical_relax_is_step_capped(caplog):
    calls = {}

    class _NeverConverges:
        def __init__(self, atoms, logfile=None):
            pass

        def run(self, fmax, steps=None):
            calls['steps'] = steps
            return False

    mc = _canon(optimizer=_NeverConverges, steps=7)
    with caplog.at_level(logging.WARNING, logger='mcpy'):
        mc.relax(mc.atoms)
    assert calls['steps'] == 7
    assert 'did not converge' in caplog.text


# --------------------------------------------------------------------------
# P10: derive_mu_gas only for homonuclear molecules
# --------------------------------------------------------------------------

def test_derive_mu_gas_rejects_heteronuclear():
    calc = SimpleNamespace(get_potential_energy=lambda a: -10.0)
    with pytest.raises(ValueError, match='homonuclear'):
        derive_mu_gas(calc, 'CO')
