"""Atomic moves: FixAtoms, the in-place contract of GCMC moves, and the
Brownian and shake moves.
"""
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.io import read

from mcpy.cell import Cell, NullCell
from mcpy.moves import (AlchemiBrownianMove, BrownianMove, DeletionMove, DisplacementMove,
                        MoveSelector, ShakeMove)
from mcpy.moves.base_move import BaseMove

from helpers import StubCalc, _selector_gcmc


def _fixed_pair():
    atoms = Atoms('Ag2O', positions=[[0, 0, 0], [3, 0, 0], [6, 6, 6]],
                  cell=[12] * 3, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    return atoms


# Position moves honour FixAtoms

def test_b2_shake_honours_fixatoms():
    atoms = _fixed_pair()
    before = atoms.positions.copy()
    ShakeMove(r_max=0.5, seed=1).do_trial_move(atoms)
    np.testing.assert_array_equal(atoms.positions[:2], before[:2])
    assert not np.allclose(atoms.positions[2], before[2])


def test_b2_displacement_honours_fixatoms():
    move = DisplacementMove(species=['Ag', 'O'], seed=1, max_displacement=0.5)
    for _ in range(20):
        atoms = _fixed_pair()
        before = atoms.positions.copy()
        move.do_trial_move(atoms)
        np.testing.assert_array_equal(atoms.positions[:2], before[:2])
        assert not np.allclose(atoms.positions[2], before[2])


def _cu_cluster():
    atoms = Octahedron('Cu', 3, 1)
    atoms.center(vacuum=5.0)
    return atoms


# BrownianMove is seeded and leaves no residue on the atoms

def test_b6_brownian_is_seeded_and_clean():
    finals = []
    for _ in range(2):
        atoms = _cu_cluster()
        sentinel = object()
        atoms.calc = sentinel
        np.random.seed(None)  # the global stream must not matter
        np.random.random(np.random.randint(1, 50))
        BrownianMove(temperature=600.0, calculator=EMT(), steps=3, d_t=1.0,
                     seed=42).do_trial_move(atoms)
        assert atoms.calc is sentinel
        assert 'momenta' not in atoms.arrays
        finals.append(atoms.positions.copy())
    np.testing.assert_array_equal(finals[0], finals[1])


def test_brownian_does_not_drift_the_center_of_mass():
    atoms = _cu_cluster()
    com_before = atoms.get_center_of_mass()
    BrownianMove(temperature=2000.0, calculator=EMT(), steps=50, d_t=2.0,
                 seed=7).do_trial_move(atoms)
    np.testing.assert_allclose(atoms.get_center_of_mass(), com_before, atol=1e-6)


def test_brownian_writes_every_md_frame(tmp_path):
    path = tmp_path / 'md.xyz'
    atoms = _cu_cluster()
    move = BrownianMove(temperature=2000.0, calculator=EMT(), steps=4, d_t=2.0,
                        seed=7, trajectory_file=str(path))
    move.do_trial_move(atoms)
    move.do_trial_move(atoms)
    frames = read(path, index=':')
    assert len(frames) == 2 * (4 + 1)
    assert [f.info['md_run'] for f in frames] == [1] * 5 + [2] * 5
    assert [f.info['md_step'] for f in frames] == list(range(5)) * 2
    last = frames[-1]
    assert last.info['time_fs'] == pytest.approx(8.0)
    assert last.info['T_target_K'] == 2000.0
    assert last.info['T_K'] > 0
    assert last.info['etot'] == pytest.approx(last.info['epot'] + last.info['ekin'])
    np.testing.assert_allclose(last.positions, atoms.positions, atol=1e-6)
    assert last.get_forces().shape == (len(atoms), 3)
    assert 'momenta' not in atoms.arrays


# AlchemiBrownianMove follows the in-place GCMC contract (bug: proposal was
# built on a copy the ensemble never looked at -> silent no-op move)

def test_alchemi_brownian_move_mutates_in_place():
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    p0 = atoms.positions.copy()
    move = AlchemiBrownianMove(StubCalc(), temperature=300.0, seed=1)
    atoms_new, delta, species = move.do_trial_move(atoms)
    assert atoms_new is atoms
    assert not np.allclose(atoms.positions, p0)
    assert delta == 0


class _CopyReturningMove(BaseMove):
    """Deliberate contract violation: returns a copy instead of mutating."""

    def __init__(self, seed=1):
        super().__init__(NullCell(), species=['X'], seed=seed)

    def do_trial_move(self, atoms):
        return atoms.copy(), 0, 'X'


def test_gcmc_rejects_copy_returning_move():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    ms = MoveSelector([1], [_CopyReturningMove()], seed=2)
    g = _selector_gcmc(atoms, [Cell(atoms)], ms, mu={'H': 0.0})
    with pytest.raises(RuntimeError, match='in place'):
        g.do_gcmc_step()


def test_displacement_on_empty_box_is_a_failed_proposal():
    """A GCMC box starts empty; drawing the displacement move first must not
    crash the run (it raised ValueError before)."""
    from ase import Atoms

    from mcpy.moves import DisplacementMove

    result = DisplacementMove(species=['H'], seed=1).do_trial_move(
        Atoms(cell=[5.0] * 3, pbc=True))
    assert result[0] is False and result[1] == 0


class _MDCalc:
    def run_md(self, atoms, **kw):
        atoms.positions += 1.0


# All-fixed systems give a failed proposal

@pytest.mark.parametrize('make_move', [
    lambda: ShakeMove(0.5, seed=1),
    lambda: BrownianMove(300, EMT(), steps=2, d_t=1.0, seed=1),
    lambda: AlchemiBrownianMove(_MDCalc(), 300, seed=1),
])
def test_md_and_shake_moves_fail_when_every_atom_is_fixed(make_move):
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.5, 0, 0]], cell=[8] * 3, pbc=True)
    atoms.set_constraint(FixAtoms([0, 1]))
    before = atoms.positions.copy()
    result = make_move().do_trial_move(atoms)
    assert result[0] is False
    np.testing.assert_array_equal(atoms.positions, before)
    assert make_move().do_trial_move(Atoms(cell=[8] * 3))[0] is False


def test_deletion_never_picks_a_fixed_atom():
    atoms = Atoms('O2', positions=[[1, 1, 1], [3, 3, 3]], cell=[6] * 3, pbc=True)
    atoms.set_constraint(FixAtoms([0]))
    move = DeletionMove(Cell(atoms, seed=1), ['O'], seed=2)
    for _ in range(20):
        trial = atoms.copy()
        trial.set_constraint(FixAtoms([0]))
        move.do_trial_move(trial)
        assert len(trial) == 1 and np.allclose(trial.positions[0], [1, 1, 1])
    atoms.set_constraint(FixAtoms([0, 1]))
    assert move.do_trial_move(atoms)[0] is False


@pytest.mark.parametrize('constraints', [np.array([0]), np.array([0, 1]), [0], (0,)])
def test_displacement_accepts_array_constraints(constraints):
    atoms = Atoms('Au3', positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0]])
    move = DisplacementMove(['Au'], seed=1, constraints=constraints)
    before = atoms.positions.copy()
    for _ in range(20):
        move.do_trial_move(atoms)
    fixed = np.atleast_1d(np.asarray(constraints))
    np.testing.assert_array_equal(atoms.positions[fixed], before[fixed])


def test_shake_keeps_the_center_of_mass_of_a_free_cluster():
    atoms = Atoms('Cu3Au', positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [0, 0, 2.5]])
    com = atoms.get_center_of_mass()
    move = ShakeMove(0.5, seed=1)
    for _ in range(50):
        move.do_trial_move(atoms)
    np.testing.assert_allclose(atoms.get_center_of_mass(), com, atol=1e-10)


def test_brownian_single_free_atom_stays_finite():
    atoms = Atoms('Cu', positions=[[3, 3, 3]], cell=[8] * 3, pbc=True)
    move = BrownianMove(300, EMT(), steps=3, d_t=1.0, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        move.do_trial_move(atoms)
    assert np.all(np.isfinite(atoms.positions))
