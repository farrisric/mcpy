"""FixAtoms survival across GCMC trial-move rollback.

DeletionMove mutates atoms in place with ``del atoms[i]``; ASE then remaps
FixAtoms indices (shift-down / drop). The ensembles snapshot only the arrays
dict, so a REJECTED deletion must also restore the constraint or every
rejection permanently drifts the fixed set (wrong atoms frozen, real fixed
layer released). Accepted deletions must keep ASE's remap untouched.
"""
import types

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from mcpy.cell import NullCell
from mcpy.cell.cell import Cell
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import InsertionMove, MoveSelector
from mcpy.moves.deletion_move import DeletionMove

from helpers import StubCalc, _DirtyBailingMove, _h2, _selector_gcmc


FIXED = [4, 5]  # the two O atoms; every Cu deletion shifts these without a fix


def _make_atoms_and_move(seed=0):
    """Cu4O2 box, O atoms (indices 4, 5) fixed, deletion targets Cu only."""
    atoms = Atoms(
        'Cu4O2',
        positions=[
            [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
            [1, 1, 1], [3, 1, 1],
        ],
        cell=[5, 5, 5],
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=FIXED))
    cell = Cell(atoms, species_radii={'Cu': 1.28, 'O': 0.66})
    cell.calculate_volume(atoms)
    move = DeletionMove(cell, species=['Cu'], seed=seed)
    return atoms, move


def _fixed_indices(atoms):
    return sorted(int(i) for c in atoms.constraints for i in c.index)


def _gcmc(atoms, move, accept):
    e = GrandCanonicalEnsemble.__new__(GrandCanonicalEnsemble)
    e._atoms = atoms
    e.E_old = 0.0
    e.n_atoms = len(atoms)
    e.compute_energy = lambda a: 0.0
    e.move_selector = types.SimpleNamespace(
        n_moves=1,
        do_trial_move=move.do_trial_move,
        get_volume=lambda: 100.0,
        get_name=lambda: 'deletion',
        acceptance_counter=lambda: None,
        get_exchange_count=lambda: None,
    )
    e._acceptance_condition = lambda *a, **k: accept
    e._wrap_on_accept = False
    e.calculate_cells_volume = lambda a: None
    e._record_minimum = lambda *a, **k: None
    e._step = 0
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    return e


def test_rejected_deletion_restores_constraint():
    atoms, move = _make_atoms_and_move()
    positions_before = atoms.positions.copy()

    e = _gcmc(atoms, move, accept=False)
    e.do_gcmc_step()

    assert len(atoms) == 6
    np.testing.assert_allclose(atoms.positions, positions_before)
    assert _fixed_indices(atoms) == FIXED


def _single_atom_gcmc(mu_h):
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    box = Cell(atoms)
    ms = MoveSelector([1], [DeletionMove(box, species=['H'], seed=1)], seed=2)
    balanced = MoveSelector([1, 1], [DeletionMove(box, species=['H'], seed=1),
                                     InsertionMove(box, species=['H'], seed=4)], seed=2)
    g = _selector_gcmc(atoms, [box], balanced, mu={'H': mu_h})
    g.move_selector = ms  # deletion only: drive exactly one deletion
    return g, ms


# Deleting the last atom is a real trial move (bug: empty Atoms is falsy ->
# mutation kept with no accept/reject bookkeeping)

@pytest.mark.parametrize('mu_h, accepted', [(-2.0, True), (2.0, False)])
def test_deletion_of_last_atom_is_scored(mu_h, accepted):
    # At 300 K, mu = -2 eV makes the deletion overwhelmingly favorable and
    # mu = +2 eV overwhelmingly unfavorable.
    g, ms = _single_atom_gcmc(mu_h=mu_h)
    g.do_gcmc_step()
    assert len(g.atoms) == g.n_atoms == (0 if accepted else 1)
    assert g.E_old == pytest.approx(0.0 if accepted else -1.0)
    assert ms.move_failed_counter_total == [0]
    assert ms.move_acceptance_total == [int(accepted)]


def test_batched_re_rolls_back_a_move_that_mutates_then_reports_failure():
    from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange

    atoms = _h2()
    before = atoms.positions.copy()
    replica = _selector_gcmc(atoms, [NullCell()],
                             MoveSelector([1], [_DirtyBailingMove()]), {'H': 0.0})

    class _BatchCalc(StubCalc):
        def get_potential_energies(self, atoms_list):
            return np.array([self.get_potential_energy(a) for a in atoms_list])

    pt = BatchedReplicaExchange.__new__(BatchedReplicaExchange)
    pt.replicas = [replica]
    pt.calculator = _BatchCalc()
    pt._batched_single_move([0])
    np.testing.assert_allclose(replica.atoms.positions, before)


def test_accepted_deletion_keeps_ase_remap():
    atoms, move = _make_atoms_and_move()

    e = _gcmc(atoms, move, accept=True)
    e.do_gcmc_step()

    # One Cu (index < 4) removed and kept: both O indices shift down by one.
    assert len(atoms) == 5
    assert _fixed_indices(atoms) == [3, 4]
    assert [atoms[i].symbol for i in _fixed_indices(atoms)] == ['O', 'O']
