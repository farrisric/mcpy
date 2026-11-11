"""Tests for InsertionMove.max_atoms and DeletionMove.min_atoms limits."""
import numpy as np
import pytest
from ase import Atoms

from mcpy.cell.cell import Cell
from mcpy.moves.deletion_move import DeletionMove
from mcpy.moves.insertion_move import InsertionMove


def _make_cell():
    atoms = Atoms(
        'Cu4O2',
        positions=[
            [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
            [1, 1, 1], [3, 1, 1],
        ],
        cell=[5, 5, 5],
        pbc=True,
    )
    cell = Cell(atoms, species_radii={'Cu': 1.28, 'O': 0.66, 'H': 0.53})
    cell.calculate_volume(atoms)
    return atoms, cell


@pytest.mark.parametrize('max_atoms, proposes', [(2, False), (3, True), (None, True)])
def test_insertion_respects_max_atoms(max_atoms, proposes):
    """Two O atoms are present, so max_atoms=2 blocks the insertion."""
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = InsertionMove(cell, species=['O'], seed=0, max_atoms=max_atoms)
    result, delta_n, species = move.do_trial_move(atoms)
    assert result is (atoms if proposes else False)
    assert (delta_n, species) == (1, 'O')
    assert len(atoms) == n0 + proposes


@pytest.mark.parametrize('min_atoms, proposes', [(2, False), (1, True), (None, True)])
def test_deletion_respects_min_atoms(min_atoms, proposes):
    """Two O atoms are present, so min_atoms=2 blocks the deletion."""
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = DeletionMove(cell, species=['O'], seed=0, min_atoms=min_atoms)
    result, delta_n, species = move.do_trial_move(atoms)
    assert result is (atoms if proposes else False)
    assert (delta_n, species) == (-1, 'O')
    assert len(atoms) == n0 - proposes


def test_atom_caps_count_free_atoms_only():
    """Two free O plus the O of a CO molecule: the caps see 2 O, not 3."""
    def atoms():
        a = Atoms('OOCO', positions=[[0, 0, 0], [2, 0, 0], [4, 4, 4], [4, 4, 5.1]],
                  cell=[8] * 3, pbc=True)
        a.new_array('molecule_id', np.array([-1, -1, 0, 0]))
        return a

    cell = Cell(atoms(), seed=1)
    a = atoms()
    assert InsertionMove(cell, ['O'], seed=2, max_atoms=3).do_trial_move(a)[0] is a
    assert DeletionMove(cell, ['O'], seed=3, min_atoms=2).do_trial_move(atoms())[0] is False
