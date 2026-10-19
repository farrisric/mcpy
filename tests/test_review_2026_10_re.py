"""Regression tests for the 2026-10 review, replica-exchange group.

CPU-only: Lennard-Jones calculators and a fake MPI communicator.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.ensembles import replica_exchange as rex
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector

BOX = 9.0


class _BatchLJ:
    def __init__(self):
        self.lj = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0, smooth=False)

    def get_potential_energy(self, atoms):
        a = atoms.copy()
        a.calc = self.lj
        return a.get_potential_energy()

    def get_potential_energies(self, atoms_list):
        return np.array([self.get_potential_energy(a) for a in atoms_list])


def _atoms(n):
    grid = np.array([(x, y, z) for x in range(4) for y in range(4) for z in range(4)])
    return Atoms('H' * n, positions=1.6 * grid[:n] + 0.5, cell=[BOX] * 3, pbc=True)


def _gcmc(mu=-4.0, seed=0, n=12, calc=None, radii=None, tmp_path=None, tag='g'):
    atoms = _atoms(n)
    cell = Cell(atoms, species_radii=radii, mc_sample_points=20_000, seed=seed + 10)
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
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=calc or _BatchLJ(), mu={'H': mu},
        species=['H'], units_type='LJ', temperature=2.0, move_selector=ms,
        random_seed=seed + 5, **files)


def _batched(tmp_path, calc=None, seed_of=lambda rank: 10 * rank, **kw):
    calc = calc or _BatchLJ()

    def factory(mu, rank):
        return _gcmc(mu=mu['H'], seed=seed_of(rank), calc=calc,
                     tmp_path=tmp_path, tag=f'r{rank}')
    kw.setdefault('gcmc_steps', 6)
    kw.setdefault('exchange_interval', 2)
    return BatchedReplicaExchange(
        factory, calc, mus=[{'H': m} for m in (-4.0, -3.8)],
        outfile=str(tmp_path / 're.log'), write_out_interval=2,
        global_minimum_file=None, seed=7, **kw)


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


# --------------------------------------------------------------------------
# replica-exchange-2: replicas with identical seeds are refused
# --------------------------------------------------------------------------

def test_batched_rejects_identically_seeded_replicas(tmp_path):
    with pytest.raises(ValueError, match='random state'):
        _batched(tmp_path, seed_of=lambda rank: 0)


def test_batched_rejects_one_shared_move_seed(tmp_path):
    """Everything else differs; one move stream is common to both."""
    calc = _BatchLJ()

    def factory(mu, rank):
        g = _gcmc(mu=mu['H'], seed=10 * rank, calc=calc)
        g.move_selector.move_list[2].rng.random.seed(99)
        return g
    with pytest.raises(ValueError, match='move 2'):
        BatchedReplicaExchange(factory, calc, mus=[{'H': -4.0}, {'H': -3.8}],
                               outfile=None, global_minimum_file=None)


def test_batched_accepts_independently_seeded_replicas(tmp_path):
    assert _batched(tmp_path).n_replicas == 2


def test_mpi_rejects_identically_seeded_ranks(monkeypatch):
    other = rex.rng_fingerprint(_gcmc(seed=0))
    comm = _FakeComm(rank=0, size=2, others=[other])
    with pytest.raises(ValueError, match='random state'):
        _mpi(monkeypatch, comm, lambda **_: _gcmc(seed=0))


def test_mpi_accepts_independently_seeded_ranks(monkeypatch):
    other = rex.rng_fingerprint(_gcmc(seed=10))
    comm = _FakeComm(rank=0, size=2, others=[other])
    _mpi(monkeypatch, comm, lambda **_: _gcmc(seed=0))


# --------------------------------------------------------------------------
# replica-exchange-3: a failure on one rank aborts the job, never hangs it
# --------------------------------------------------------------------------

def test_mpi_factory_error_aborts(monkeypatch):
    comm = _FakeComm()

    def factory(**_):
        raise RuntimeError('bad checkpoint')
    with pytest.raises(RuntimeError, match='bad checkpoint'):
        _mpi(monkeypatch, comm, factory)
    assert comm.aborted == 1


def test_mpi_final_collective_error_aborts(monkeypatch, tmp_path):
    comm = _FakeComm()
    re = _mpi(monkeypatch, comm, lambda **_: _gcmc(), gcmc_steps=2,
              outfile=str(tmp_path / 're.log'),
              global_minimum_file=str(tmp_path / 'gm.xyz'))
    comm.gathers_left = 1  # the step-0 table; the end-of-run ones fail
    with pytest.raises(RuntimeError, match='link down'):
        re.run()
    assert comm.aborted == 1


# --------------------------------------------------------------------------
# api-packaging-tests-docs-6: MPI ladder checks
# --------------------------------------------------------------------------

def test_mpi_numpy_temperature_ladder(monkeypatch):
    other = rex.rng_fingerprint(_gcmc(seed=10))
    comm = _FakeComm(rank=0, size=2, others=[other])
    re = _mpi(monkeypatch, comm, lambda **_: _gcmc(),
              temperatures=np.array([2.0, 2.5]))
    assert re.temperatures == [2.0, 2.5]


@pytest.mark.parametrize('ladder', ['temperatures', 'mus'])
@pytest.mark.parametrize('values', [[], [2.0, 2.5]])
def test_mpi_ladder_length_mismatch_raises_value_error(monkeypatch, ladder, values):
    if ladder == 'mus':
        values = [{'H': v} for v in values]
    with pytest.raises(ValueError, match='MPI ranks'):
        _mpi(monkeypatch, _FakeComm(), lambda **_: _gcmc(), **{ladder: values})


# --------------------------------------------------------------------------
# replica-exchange-5: a swapped-in configuration is scored for the minimum
# --------------------------------------------------------------------------

def test_batched_swap_records_minimum(tmp_path):
    pt = _batched(tmp_path)
    ri, rj = pt.replicas
    ri._best_score = rj._best_score = np.inf
    pt._swap_states(0, 1)
    assert ri._best_score == pytest.approx(ri._minimum_score(ri.atoms, ri.E_old))
    assert rj._best_score == pytest.approx(rj._minimum_score(rj.atoms, rj.E_old))


def test_mpi_swap_records_minimum(monkeypatch):
    partner = _gcmc(seed=10, n=3)
    comm = _FakeComm(rank=0, size=2, others=[rex.rng_fingerprint(partner)],
                     partner_state=partner.get_state())
    g = _gcmc(seed=0, n=12)
    re = _mpi(monkeypatch, comm, lambda **_: g, temperatures=[2.0, 2.0])
    g._best_score = np.inf
    assert re.do_exchange()
    assert len(g.atoms) == 3
    assert g._best_score == pytest.approx(g._minimum_score(g.atoms, g.E_old))


# --------------------------------------------------------------------------
# replica-exchange-7: a second run() appends to the replica-exchange log
# --------------------------------------------------------------------------

def test_batched_second_run_appends_log(tmp_path):
    pt = _batched(tmp_path)
    pt.run()
    first = open(tmp_path / 're.log').read()
    pt.run()
    log = open(tmp_path / 're.log').read()
    assert log.startswith(first)
    assert log.count('Batched Replica Exchange') == 1
    assert len(log) > len(first)


def test_mpi_second_run_appends_log(monkeypatch, tmp_path):
    re = _mpi(monkeypatch, _FakeComm(), lambda **_: _gcmc(), gcmc_steps=4,
              write_out_interval=2, outfile=str(tmp_path / 're.log'),
              global_minimum_file=None)
    re.run()
    first = open(tmp_path / 're.log').read()
    re.run()
    log = open(tmp_path / 're.log').read()
    assert log.startswith(first)
    assert log.count('Replica Exchange Monte Carlo') == 1
    assert len(log) > len(first)


# --------------------------------------------------------------------------
# ensembles-io-3: an energy error restores every pending replica
# --------------------------------------------------------------------------

def _snapshot(pt):
    return [(r.atoms.get_positions().copy(), r.atoms.get_chemical_symbols())
            for r in pt.replicas]


def test_batched_energy_error_restores_every_replica(tmp_path):
    pt = _batched(tmp_path)
    before = _snapshot(pt)

    def boom(atoms_list):
        raise RuntimeError('CUDA OOM')
    pt.calculator.get_potential_energies = boom
    with pytest.raises(RuntimeError, match='CUDA OOM'):
        for _ in range(10):  # 10 trials: some insert, delete or displace
            pt._batched_single_move([0, 1])
    for (pos, sym), r in zip(before, pt.replicas):
        assert r.atoms.get_chemical_symbols() == sym
        np.testing.assert_array_equal(r.atoms.get_positions(), pos)


def test_batched_short_energy_list_raises_and_restores(tmp_path):
    pt = _batched(tmp_path)
    before = _snapshot(pt)
    real = pt.calculator.get_potential_energies
    pt.calculator.get_potential_energies = lambda al: real(al)[:1]
    with pytest.raises(ValueError):
        pt._batched_single_move([0, 1])
    for (pos, sym), r in zip(before, pt.replicas):
        assert r.atoms.get_chemical_symbols() == sym
        np.testing.assert_array_equal(r.atoms.get_positions(), pos)


# --------------------------------------------------------------------------
# driver-energy-config-invariant-1: startup re-relaxation refreshes volumes
# --------------------------------------------------------------------------

def test_rebatch_refreshes_cell_volumes(tmp_path):
    pt = _batched(tmp_path)
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


# --------------------------------------------------------------------------
# api-packaging-tests-docs-2: a swap exchanges the whole configuration state
# --------------------------------------------------------------------------

def test_swap_states_exchanges_configuration(tmp_path):
    calc = _BatchLJ()

    def factory(mu, rank):
        return _gcmc(mu=mu['H'], seed=10 * rank, n=(12, 3)[rank], calc=calc,
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
