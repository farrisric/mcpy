"""The checks that GrandCanonicalEnsemble makes when it is built: move
weights, species, temperature, min_insert and cells.
"""
import warnings

import pytest
from ase import Atoms

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector

from helpers import _gcmc, _ZeroCalc


def _balance_gcmc(weights, ins_species=('H',), del_species=('H',), same_cell=True):
    atoms = Atoms('O', positions=[[1, 1, 1]], cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    other = cell if same_cell else Cell(atoms, seed=2)
    return _gcmc(atoms, [cell], [InsertionMove(cell, list(ins_species), 1),
                                 DeletionMove(other, list(del_species), 2)],
                 weights=weights, species=('H', 'O'))


@pytest.mark.parametrize('case', [
    dict(weights=[1, 2]),
    dict(weights=[1, 1], ins_species=('H', 'O'), del_species=('H',)),
    dict(weights=[1, 1], same_cell=False),
])
def test_unequal_insertion_deletion_probability_raises(case):
    with pytest.raises(ValueError, match=r"species '(H|O)'.*kT ln"):
        _balance_gcmc(**case)


def test_weight_mismatch_message_gives_the_mu_shift():
    with pytest.raises(ValueError, match=r'= \+0\.6931'):
        _balance_gcmc([2, 1])  # T = 1 (LJ): kT ln 2


# Cell refresh, three-state ideal gas


class _Relaxing(_ZeroCalc):
    def __init__(self, steps):
        self.steps = steps


@pytest.mark.parametrize('calc,warns', [(_ZeroCalc(), True), (_Relaxing(0), True),
                                        (_Relaxing(50), False)])
def test_min_insert_warns_only_without_relaxation(calc, warns):
    atoms = Atoms('Cu', positions=[[0.5, 5, 5]], cell=[10] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'Cu': 3.0, 'H': 4.0}, mc_sample_points=4000, seed=11)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _gcmc(atoms, [cell], [DeletionMove(cell, ['H'], 12),
                              InsertionMove(cell, ['H'], 13, min_insert=0.5, max_atoms=1)],
              calc=calc)
    hits = [w for w in caught if 'min_insert' in str(w.message)]
    assert bool(hits) == warns
    assert all(w.category is UserWarning for w in hits)


class _NoEnergy:
    def get_potential_energy(self, atoms):
        raise AssertionError('validation must run before the first energy call')


# Input validation before the first energy

@pytest.mark.parametrize('T', [0.0, -5.0, float('nan'), float('inf')])
def test_gcmc_rejects_bad_temperature(T):
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    with pytest.raises(ValueError, match='temperature'):
        _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
              T=T, calc=_NoEnergy(), units='metal')


@pytest.mark.parametrize('species,mu,match', [
    (['H'], {'O': -1.0}, 'no mu'),
    (['O'], {'H': -1.0, 'O': -1.0}, 'not in species'),
    (['H2'], {'H2': -1.0}, 'not element symbols'),
])
def test_gcmc_rejects_inconsistent_species(species, mu, match):
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    ms = MoveSelector([1, 1], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)])
    with pytest.raises(ValueError, match=match):
        GrandCanonicalEnsemble(atoms=atoms, cells=[cell], units_type='LJ',
                               calculator=_NoEnergy(), mu=mu, species=species,
                               temperature=1.0, move_selector=ms, random_seed=1,
                               traj_file=None, outfile=None)
