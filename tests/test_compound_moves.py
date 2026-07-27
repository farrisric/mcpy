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


# --------------------------------------------------------------------------
# A trial must be all-or-nothing (bug: a multi-swap trial could apply k swaps
# and then report the "couldn't propose" sentinel, which the ensembles read as
# "the atoms were not touched" -- so the config changed while E_old did not)
# --------------------------------------------------------------------------

def test_permutation_absent_species_does_not_abort_a_multi_swap_trial():
    """Declaring a species that is not in the system must not turn a later
    iteration into a mid-trial bail-out. Species counts are swap-invariant, so
    the usable pair is resolved once, up front."""
    for seed in range(200):
        atoms = _balanced_alloy()
        before = atoms.get_atomic_numbers().copy()
        move = PermutationMove(species=['Au', 'Pt', 'Ag'], seed=seed, n_swaps=3)
        result, delta, _ = move.do_trial_move(atoms)
        assert result is atoms, f'seed={seed} bailed out mid-trial'
        assert delta == 0
        assert not np.array_equal(atoms.get_atomic_numbers(), before)


def test_permutation_without_two_present_species_is_a_clean_no_op():
    """The one case that genuinely cannot propose must leave the atoms
    byte-identical, since the ensembles skip the rollback on that path."""
    atoms = Atoms('Au4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]])
    before = atoms.get_atomic_numbers().copy()
    move = PermutationMove(species=['Au', 'Pt'], seed=1, n_swaps=3)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False
    assert (delta, name) == (0, 'X')
    np.testing.assert_array_equal(atoms.get_atomic_numbers(), before)


def test_permutation_stream_unchanged_when_every_species_is_present():
    """Filtering to the present species must be a no-op for a healthy setup,
    or previously published runs would no longer reproduce."""
    results = []
    for species in (['Au', 'Pt'], ['Au', 'Pt']):
        atoms = _balanced_alloy()
        PermutationMove(species=species, seed=11, n_swaps=5).do_trial_move(atoms)
        results.append(atoms.get_atomic_numbers().copy())
    np.testing.assert_array_equal(*results)
