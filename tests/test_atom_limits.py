"""Tests for InsertionMove.max_atoms and DeletionMove.min_atoms limits."""
import numpy as np
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


def test_insertion_skips_at_max_atoms():
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = InsertionMove(cell, species=['O'], seed=0, max_atoms=2)
    result, delta_n, species = move.do_trial_move(atoms)
    assert result is False
    assert delta_n == 1
    assert species == 'O'
    assert len(atoms) == n0


def test_insertion_allowed_below_max_atoms():
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = InsertionMove(cell, species=['O'], seed=1, max_atoms=3)
    result, delta_n, species = move.do_trial_move(atoms)
    assert result is atoms
    assert delta_n == 1
    assert species == 'O'
    assert len(atoms) == n0 + 1


def test_insertion_max_atoms_none_is_ignored():
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = InsertionMove(cell, species=['O'], seed=2, max_atoms=None)
    result, delta_n, _ = move.do_trial_move(atoms)
    assert result is atoms
    assert delta_n == 1
    assert len(atoms) == n0 + 1


def test_deletion_skips_at_min_atoms():
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = DeletionMove(cell, species=['O'], seed=0, min_atoms=2)
    result, delta_n, species = move.do_trial_move(atoms)
    assert result is False
    assert delta_n == -1
    assert species == 'O'
    assert len(atoms) == n0


def test_deletion_allowed_above_min_atoms():
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = DeletionMove(cell, species=['O'], seed=1, min_atoms=1)
    result, delta_n, species = move.do_trial_move(atoms)
    assert result is atoms
    assert delta_n == -1
    assert species == 'O'
    assert len(atoms) == n0 - 1


def test_deletion_min_atoms_none_is_ignored():
    atoms, cell = _make_cell()
    n0 = len(atoms)
    move = DeletionMove(cell, species=['O'], seed=2, min_atoms=None)
    result, delta_n, _ = move.do_trial_move(atoms)
    assert result is atoms
    assert delta_n == -1
    assert len(atoms) == n0 - 1
