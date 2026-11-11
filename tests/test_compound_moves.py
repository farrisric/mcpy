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


@pytest.mark.parametrize('make_move', [
    lambda: PermutationMove(species=['Au', 'Pt'], seed=1, n_swaps=0),
    lambda: DisplacementMove(species=['Au', 'Pt'], seed=1, n_steps=0),
])
def test_compound_count_below_one_raises(make_move):
    with pytest.raises(ValueError):
        make_move()


def test_displacement_n_steps_above_movable_moves_all():
    """Fewer movable atoms than n_steps: all of them move (GCMC changes the
    movable count every exchange, so this must not raise)."""
    atoms = Atoms('Au2', positions=[[0, 0, 0], [1, 0, 0]])
    before = atoms.positions.copy()
    move = DisplacementMove(species=['Au'], seed=1, n_steps=5)
    result, _, _ = move.do_trial_move(atoms)
    assert result is atoms
    assert np.all(np.any(atoms.positions != before, axis=1))


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


@pytest.mark.parametrize('kwargs, moved', [({}, 1), ({'n_steps': 4}, 4)])
def test_displacement_moves_n_steps_distinct_atoms(kwargs, moved):
    atoms = _balanced_alloy()
    before = atoms.positions.copy()
    move = DisplacementMove(species=['Au', 'Pt'], seed=1, max_displacement=0.2, **kwargs)
    move.do_trial_move(atoms)
    assert int(np.sum(np.any(atoms.positions != before, axis=1))) == moved
