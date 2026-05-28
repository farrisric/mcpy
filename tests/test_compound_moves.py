"""Tests for the ``n_swaps`` / ``n_steps`` compound-perturbation parameters
on PermutationMove and DisplacementMove. These bundle multiple base
perturbations into a single trial — the building block for basin-hopping
sampling, where one swap typically falls back into the same basin after
relaxation."""

import numpy as np
import pytest
from ase import Atoms
from ase.cluster import Octahedron

from mcpy.moves.displacement_move import DisplacementMove
from mcpy.moves.permutation_move import PermutationMove


def _balanced_alloy():
    atoms = Octahedron('Au', 3)
    half = len(atoms) // 2
    atoms.symbols = ['Au'] * half + ['Pt'] * (len(atoms) - half)
    return atoms


def test_permutation_default_swaps_one_pair():
    atoms = _balanced_alloy()
    before = atoms.get_atomic_numbers().copy()
    move = PermutationMove(species=['Au', 'Pt'], seed=1)
    move.do_trial_move(atoms)
    after = atoms.get_atomic_numbers()
    changed = int(np.sum(before != after))
    assert changed == 2  # exactly one pair


def test_permutation_n_swaps_changes_more_positions():
    atoms = _balanced_alloy()
    before = atoms.get_atomic_numbers().copy()
    move = PermutationMove(species=['Au', 'Pt'], seed=1, n_swaps=4)
    move.do_trial_move(atoms)
    after = atoms.get_atomic_numbers()
    changed = int(np.sum(before != after))
    # 4 swaps on a balanced cluster: at least one pair (2 positions), almost
    # always more than a single-swap trial.
    assert changed >= 2
    assert changed > 2  # with seed=1 we expect distinct pairs


def test_permutation_invalid_n_swaps():
    with pytest.raises(ValueError):
        PermutationMove(species=['Au', 'Pt'], seed=1, n_swaps=0)


def test_displacement_default_moves_one_atom():
    atoms = _balanced_alloy()
    before = atoms.positions.copy()
    move = DisplacementMove(species=['Au', 'Pt'], seed=1, max_displacement=0.2)
    move.do_trial_move(atoms)
    moved = int(np.sum(np.any(atoms.positions != before, axis=1)))
    assert moved == 1


def test_displacement_n_steps_moves_k_atoms():
    atoms = _balanced_alloy()
    before = atoms.positions.copy()
    move = DisplacementMove(
        species=['Au', 'Pt'], seed=1, max_displacement=0.2, n_steps=4,
    )
    move.do_trial_move(atoms)
    moved = int(np.sum(np.any(atoms.positions != before, axis=1)))
    assert moved == 4  # exactly K distinct atoms displaced


def test_displacement_invalid_n_steps():
    with pytest.raises(ValueError):
        DisplacementMove(species=['Au', 'Pt'], seed=1, n_steps=0)


def test_displacement_n_steps_exceeds_movable_raises():
    atoms = Atoms('Au2', positions=[[0, 0, 0], [1, 0, 0]])
    move = DisplacementMove(species=['Au'], seed=1, n_steps=5)
    with pytest.raises(ValueError):
        move.do_trial_move(atoms)
