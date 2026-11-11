"""MoveSelector: weights, trials per step and the outfile labels.
"""
import pytest

from mcpy.moves import MoveSelector, ShakeMove


# The trial count does not depend on the weight dtype (bug: float weights
# crashed range(); int weights silently set trials per step)


def test_move_labels_distinguish_the_molecule_moves():
    from mcpy.moves.move_selector import _abbreviate

    assert _abbreviate('MoleculeInsertionMove') == 'MolIns'
    assert _abbreviate('MoleculeDeletionMove') == 'MolDel'
    assert _abbreviate('MoleculeDisplacementMove') == 'MolDis'
    labels = [_abbreviate(n) for n in ('MoleculeInsertionMove',
                                       'MoleculeDeletionMove',
                                       'MoleculeDisplacementMove')]
    assert len(set(labels)) == 3


def test_move_labels_unchanged_for_single_word_moves():
    """Atomic runs keep the labels their existing outfiles were written with."""
    from mcpy.moves.move_selector import _abbreviate

    for name in ('InsertionMove', 'DeletionMove', 'DisplacementMove',
                 'PermutationMove', 'ShakeMove', 'BrownianMove'):
        assert _abbreviate(name) == name[:3]


# MoveSelector weights

@pytest.mark.parametrize('weights', [[1], [1, -1], [0, 0], [1, float('nan')],
                                     [1, float('inf')]])
def test_move_selector_rejects_bad_weights(weights):
    moves = [ShakeMove(0.1, 1), ShakeMove(0.1, 2)]
    with pytest.raises(ValueError):
        MoveSelector(weights, moves)
