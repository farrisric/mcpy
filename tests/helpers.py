"""Stub calculators and ensemble builders that several test files share."""
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.cluster import Octahedron
from ase.geometry import find_mic
from ase.io import read
from ase.optimize import LBFGS

from mcpy.cell import Cell, NullCell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.ensembles.canonical_ensemble import CanonicalEnsemble
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector
from mcpy.moves.base_move import BaseMove
from mcpy.moves.permutation_move import PermutationMove


def _o2():
    return Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])


def _mic_dists(atoms, point):
    diff = find_mic(atoms.positions - point, atoms.cell, pbc=atoms.pbc)[0]
    return np.linalg.norm(diff, axis=1)


class _ZeroCalc:
    def get_potential_energy(self, atoms):
        return 0.0


def _canonical(tmp_path, **kwargs):
    atoms = Octahedron('Au', 2)
    half = len(atoms) // 2
    atoms.symbols = ['Au'] * half + ['Pt'] * (len(atoms) - half)
    ms = MoveSelector([1.0], [PermutationMove(species=['Au', 'Pt'], seed=5)])
    defaults = dict(
        atoms=atoms, calculator=EMT(), optimizer=LBFGS, fmax=0.2,
        temperature=600.0, move_selector=ms, random_seed=5,
        traj_file=str(tmp_path / 't.xyz'), outfile=str(tmp_path / 'o.out'),
    )
    defaults.update(kwargs)
    return CanonicalEnsemble(**defaults)


class StubCalc:
    """Energy = -1 eV per atom; ``run_md`` shifts every position by +1 A."""

    def get_potential_energy(self, atoms):
        return -1.0 * len(atoms)

    def run_md(self, atoms, **kwargs):
        atoms.positions += 1.0


def _selector_gcmc(atoms, cells, move_selector, mu, units_type='metal'):
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=cells, units_type=units_type, calculator=StubCalc(),
        mu=mu, species=list(mu), temperature=300.0,
        move_selector=move_selector, random_seed=3,
        traj_file=None, outfile=None,
    )


class _DirtyBailingMove(BaseMove):
    """Contract violator: mutates the atoms, then returns the ``False``
    'could not propose' sentinel."""

    def __init__(self, seed=1):
        super().__init__(NullCell(), ['X'], seed)

    def do_trial_move(self, atoms):
        atoms.positions += 1.0
        return False, 0, 'X'


def _h2():
    return Atoms('H2', positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                 cell=[10.0, 10.0, 10.0])


BOX = 9.0


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


def _gcmc(atoms, cells, moves, weights=None, mu=None, species=('H',), calc=None,
          units='LJ', T=1.0, n_moves=1, seed=0, **kw):
    ms = MoveSelector(weights or [1] * len(moves), moves, seed=seed + 100,
                      n_moves=n_moves)
    files = dict(traj_file=None, outfile=None)
    files.update(kw)
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=cells, units_type=units, calculator=calc or _ZeroCalc(),
        mu=mu if mu is not None else {s: -1.0 for s in species},
        species=list(species), temperature=T, move_selector=ms,
        random_seed=seed + 200, **files)


def _atoms(n):
    grid = np.array([(x, y, z) for x in range(4) for y in range(4) for z in range(4)])
    return Atoms('H' * n, positions=1.6 * grid[:n] + 0.5, cell=[BOX] * 3, pbc=True)


def _lj_replica(mu=-4.0, seed=0, n=12, calc=None, radii=None, tmp_path=None, tag='g'):
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


def _batched_pair(tmp_path, calc=None, seed_of=lambda rank: 10 * rank, **kw):
    calc = calc or _BatchLJ()

    def factory(mu, rank):
        return _lj_replica(mu=mu['H'], seed=seed_of(rank), calc=calc,
                           tmp_path=tmp_path, tag=f'r{rank}')
    kw.setdefault('gcmc_steps', 6)
    kw.setdefault('exchange_interval', 2)
    return BatchedReplicaExchange(
        factory, calc, mus=[{'H': m} for m in (-4.0, -3.8)],
        outfile=str(tmp_path / 're.log'), write_out_interval=2,
        global_minimum_file=None, seed=7, **kw)


def _frames(path):
    return len(read(str(path), index=':', format='extxyz'))
