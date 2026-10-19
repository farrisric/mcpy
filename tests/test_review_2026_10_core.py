"""Regression tests for the 2026-10 review, core group (GCMC ensemble,
moves, base and canonical ensembles, units, RNG).

Each test names the finding it guards. All CPU-only and torch-free; the
physics tests are short ideal-gas or exact-state runs (seconds).
"""
import io
import math
import warnings

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from mcpy.cell import Cell, CustomCell, SphericalCell
from mcpy.ensembles import CanonicalEnsemble, GrandCanonicalEnsemble
from mcpy.ensembles.base_ensemble import write_xyz
from mcpy.moves import (AlchemiBrownianMove, BrownianMove, DeletionMove,
                        DisplacementMove, InsertionMove, MoleculeDeletionMove,
                        MoleculeDisplacementMove, MoleculeInsertionMove,
                        MoveSelector, ShakeMove)
from mcpy.moves.molecule_utils import find_molecules, molecule_com
from mcpy.utils import RandomNumberGenerator
from mcpy.utils.set_unit_constant import SetUnits


class _ZeroCalc:
    def get_potential_energy(self, atoms):
        return 0.0


class _HardCore:
    """Zero energy, or 1e6 when two atoms are closer than ``r`` (mic)."""

    def __init__(self, r):
        self.r = r

    def get_potential_energy(self, atoms):
        if len(atoms) < 2:
            return 0.0
        d = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)]
        return 1e6 if np.any(d < self.r) else 0.0


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


def _h2():
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])


# --------------------------------------------------------------------------
# gcmc-physics-move-weight-balance
# --------------------------------------------------------------------------

def _balance_gcmc(weights, ins_species=('H',), del_species=('H',), same_cell=True):
    atoms = Atoms('O', positions=[[1, 1, 1]], cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    other = cell if same_cell else Cell(atoms, seed=2)
    return _gcmc(atoms, [cell], [InsertionMove(cell, list(ins_species), 1),
                                 DeletionMove(other, list(del_species), 2)],
                 weights=weights, species=('H', 'O'))


def test_balanced_insertion_and_deletion_pass():
    _balance_gcmc([2, 2])
    _balance_gcmc([1, 1], ins_species=('H', 'O'), del_species=('O', 'H'))


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


# --------------------------------------------------------------------------
# region-unaware-moves-leak-1: exchanged atoms may not cross the ball surface
# --------------------------------------------------------------------------

def test_spherical_cell_displacement_ideal_gas_matches_exact_distribution():
    """Ideal H around a fixed Cu in a SphericalCell, with large atomic
    displacements. Exact: P(N) ~ x^N / (N+1)!, x = zV (the +1 is the Cu in
    the total-atom count). Without the ball guard H atoms random-walk out
    of the sphere, cannot be deleted and pile up."""
    atoms = Atoms('Cu', positions=[[10, 10, 10]], cell=[20] * 3, pbc=False)
    atoms.set_constraint(FixAtoms([0]))
    cell = SphericalCell(atoms, vacuum=3.0, species_radii=None, seed=1)
    x = 3.0
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, max_displacement=2.0)],
              mu={'H': math.log(x / cell.get_volume())})
    counts = []
    for step in range(30_000):
        g.do_gcmc_step()
        if step >= 2_000:
            counts.append(len(g.atoms) - 1)
    weights = np.array([x ** n / math.factorial(n + 1) for n in range(40)])
    exact = (weights / weights.sum()) @ np.arange(40)
    assert np.mean(counts) == pytest.approx(exact, abs=0.12)
    d = np.linalg.norm(g.atoms.positions - cell.center, axis=1)
    assert np.all(d <= cell.radius)


def test_ball_guard_rejects_a_crossing_and_spares_the_dome_floor():
    """A trial that carries an exchanged atom out of the ball is rejected,
    one that only crosses the DomeCell support floor is not."""
    from mcpy.cell import DomeCell
    atoms = Atoms('Ag2O', positions=[[5, 5, 5], [6, 5, 5], [5, 5, 6]],
                  cell=[12] * 3, pbc=True)
    dome = DomeCell(atoms, 'Ag', bottom_z=4.5, vacuum=2.0, species_radii=None, seed=1)
    g = _gcmc(atoms, [dome], [InsertionMove(dome, ['O'], 1), DeletionMove(dome, ['O'], 2)],
              species=('O',))
    snapshot = ({k: v.copy() for k, v in g.atoms.arrays.items()}, [])
    g.atoms.positions[2] = [5.5, 5, 4.0]  # below the floor, inside the ball
    assert g._outside_ball_counts(snapshot[0]) == g._outside_ball_counts(g.atoms.arrays)
    g.atoms.positions[2] = [5.5, 5, 9.0]  # outside the ball
    assert not g._commit_or_rollback(-100.0, snapshot, (0, 'X'))
    np.testing.assert_allclose(g.atoms.positions[2], [5, 5, 6])


# --------------------------------------------------------------------------
# region-unaware-moves-leak-3: all-fixed systems give a failed proposal
# --------------------------------------------------------------------------

class _MDCalc:
    def run_md(self, atoms, **kw):
        atoms.positions += 1.0


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


# --------------------------------------------------------------------------
# api-packaging-tests-docs-1 / -2: cell refresh, three-state ideal gas
# --------------------------------------------------------------------------

@pytest.mark.parametrize('listed', [True, False])
def test_three_state_hard_core_gas(listed):
    """Hard-core H (exclusion radius 4) in a periodic 10^3 box, N <= 2.
    Exact: P(N) ~ z^N Q_N / N!, Q_0 = 1, Q_1 = V, Q_2 = V (V - V_h).
    Breaks when an insertion reads a stale volume: the cell is missing
    from cells=, or it is not refreshed after an accepted move."""
    L, r = 10.0, 4.0
    atoms = Atoms(cell=[L] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'H': r}, mc_sample_points=5000, seed=1)
    z = 1e-3
    g = _gcmc(atoms, [cell] if listed else [],
              [InsertionMove(cell, ['H'], 2, max_atoms=2), DeletionMove(cell, ['H'], 3),
               DisplacementMove(['H'], 4, max_displacement=3.0)],
              mu={'H': math.log(z)}, calc=_HardCore(r))
    hist = np.zeros(3)
    for _ in range(8000):
        g.do_gcmc_step()
        hist[len(g.atoms)] += 1
    v, v_h = L ** 3, 4.0 / 3.0 * math.pi * r ** 3
    exact = np.array([1.0, z * v, z * z * v * (v - v_h) / 2.0])
    exact /= exact.sum()
    np.testing.assert_allclose(hist / hist.sum(), exact, atol=0.025)


def test_insertion_cells_are_refreshed_without_being_listed():
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[6] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'H': 1.0}, seed=1)
    g = _gcmc(atoms, [], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)])
    assert g.cells == [cell]
    assert cell.get_volume() < 216.0


def test_custom_cell_atom_crossing_xy_stays_deletable():
    """An atom displaced across the periodic xy boundary is wrapped on
    acceptance, so it stays inside the CustomCell footprint (deletable)."""
    atoms = Atoms('O4', positions=[[0.2, 0.2, 10], [2.8, 0.5, 10], [1.5, 2.9, 10],
                                   [0.1, 1.5, 10]], cell=[3, 3, 20], pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=20.0, bottom_z=0.0, seed=1)
    g = _gcmc(atoms, [cell], [DisplacementMove(['O'], 2, max_displacement=0.8)],
              species=('O',))
    for _ in range(300):
        g.do_gcmc_step()
    assert len(cell.get_atoms_specie_inside_cell(g.atoms, 'O')) == 4


class _Floor:
    """Zero energy above ``z0``, 1e6 when any atom sinks below it (a
    repulsive support)."""

    def __init__(self, z0):
        self.z0 = z0

    def get_potential_energy(self, atoms):
        return 1e6 if np.any(atoms.positions[:, 2] < self.z0) else 0.0


def test_custom_cell_displacement_ideal_gas_stays_poisson():
    """Ideal H in a thin CustomCell slab over a repulsive floor, with many
    atomic displacements. Exact: the in-cell count is Poisson(zV). The top
    is a hard wall for displacements; without it atoms climb above the top
    (deletable but never insertable) and the in-cell density drops."""
    atoms = Atoms(cell=[6, 6, 20], pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=3.0, bottom_z=5.0, seed=1)
    zv = 3.0
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, max_displacement=1.0)],
              weights=[1, 1, 10], mu={'H': math.log(zv / cell.get_volume())},
              calc=_Floor(5.0))
    counts = []
    for step in range(20_000):
        g.do_gcmc_step()
        if step >= 1_000 and step % 4 == 0:
            counts.append(sum(cell.is_point_inside(p) for p in g.atoms.positions))
    assert np.mean(counts) == pytest.approx(zv, abs=0.2)
    assert np.all(g.atoms.positions[:, 2] < 8.0)


class _Counting(_ZeroCalc):
    calls = 0

    def get_potential_energy(self, atoms):
        self.calls += 1
        return 0.0


def test_custom_cell_top_crossing_is_rejected_before_the_energy_call():
    """An H just below the top: about half the displacements would cross.
    Those are rejected attempts with no energy call; the rest are accepted."""
    atoms = Atoms('H', positions=[[1, 1, 7.99]], cell=[6, 6, 20], pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=3.0, bottom_z=5.0, seed=1)
    calc = _Counting()
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, max_displacement=1.0)],
              weights=[0, 0, 1], calc=calc)
    for _ in range(100):
        g.atoms.positions[0, 2] = 7.99
        g.do_gcmc_step()
        assert g.atoms.positions[0, 2] < 8.0
    ms = g.move_selector
    assert ms.move_counter_total[2] == 100 and ms.move_failed_counter_total[2] == 0
    assert 20 < calc.calls - 1 == ms.move_acceptance_total[2] < 80


# --------------------------------------------------------------------------
# recent-changes-min-insert-unrelaxed-bias
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# molecules-2: molecule displacement keeps the COM inside the proposal region
# --------------------------------------------------------------------------

def test_molecule_ideal_gas_in_custom_cell_stays_poisson():
    """Ideal H2 molecules in a thin CustomCell slab with many rigid
    displacements. Exact: the in-cell count is Poisson(zV). If displacements
    may lift molecules above the cell top (exchangeable but never
    insertable), the in-cell density drops."""
    atoms = Atoms(cell=[6, 6, 20], pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=3.0, bottom_z=5.0, seed=1)
    tpl = _h2()
    zv = 3.0
    ms = MoveSelector([1, 1, 10], [MoleculeInsertionMove(cell, tpl, 'H2', 2),
                                   MoleculeDeletionMove(cell, tpl, 'H2', 3),
                                   MoleculeDisplacementMove(cell, tpl, 'H2', 4,
                                                            max_displacement=1.0)],
                      seed=5, n_moves=1)
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ', calculator=_ZeroCalc(),
        mu={'H2': math.log(zv / cell.get_volume())}, species=[], molecules={'H2': tpl},
        temperature=1.0, move_selector=ms, random_seed=6, traj_file=None, outfile=None)
    counts = []
    for step in range(20_000):
        g.do_gcmc_step()
        if step >= 1_000 and step % 4 == 0:
            counts.append(sum(cell.is_point_inside(molecule_com(g.atoms, m))
                              for m in find_molecules(g.atoms, ['H', 'H'])))
    assert np.mean(counts) == pytest.approx(zv, abs=0.25)


# --------------------------------------------------------------------------
# molecules-4, molecules-5, molecules-8
# --------------------------------------------------------------------------

def test_molecule_template_with_molecule_id_inserts():
    tpl = _h2()
    tpl.new_array('molecule_id', np.array([4, 4]))
    atoms = Atoms(cell=[8] * 3, pbc=True)
    move = MoleculeInsertionMove(Cell(atoms, seed=1), tpl, 'H2', seed=2)
    move.do_trial_move(atoms)
    move.do_trial_move(atoms)
    np.testing.assert_array_equal(atoms.arrays['molecule_id'], [0, 0, 1, 1])


def test_host_zero_filled_molecule_id_raises():
    host = Atoms('Cu2', positions=[[0, 0, 0], [2.5, 0, 0]], cell=[8] * 3, pbc=True)
    co = Atoms('CO', positions=[[4, 4, 4], [4, 4, 5.1]])
    co.new_array('molecule_id', np.array([0, 0]))
    atoms = host + co  # ASE zero-fills the host: Cu2 joins molecule 0
    cell = Cell(atoms, seed=1)
    with pytest.raises(ValueError, match=r'molecule_id groups \[0\]'):
        GrandCanonicalEnsemble(
            atoms=atoms, cells=[cell], units_type='LJ', calculator=_ZeroCalc(),
            mu={'CO': -1.0}, species=[], molecules={'CO': co}, temperature=1.0,
            move_selector=MoveSelector([1, 1], [
                MoleculeInsertionMove(cell, co, 'CO', 1),
                MoleculeDeletionMove(cell, co, 'CO', 2)]),
            random_seed=1, traj_file=None, outfile=None)


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


# --------------------------------------------------------------------------
# moves-cells-geometry-2, -3, -5, -7
# --------------------------------------------------------------------------

def test_gcmc_from_empty_box_with_compound_displacement():
    atoms = Atoms(cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, n_steps=5)], mu={'H': 2.0})
    for _ in range(50):
        g.do_gcmc_step()
    assert len(g.atoms) > 0


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


# --------------------------------------------------------------------------
# recent-changes-brownian-stationary-nan
# --------------------------------------------------------------------------

def test_brownian_single_free_atom_stays_finite():
    atoms = Atoms('Cu', positions=[[3, 3, 3]], cell=[8] * 3, pbc=True)
    move = BrownianMove(300, EMT(), steps=3, d_t=1.0, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        move.do_trial_move(atoms)
    assert np.all(np.isfinite(atoms.positions))


# --------------------------------------------------------------------------
# ensembles-io-2: the trajectory restarts the same system
# --------------------------------------------------------------------------

def test_write_xyz_round_trips_pbc_constraints_tags_and_molecule_id():
    atoms = Atoms('CuOO', positions=[[0, 0, 0], [1, 1, 1], [1, 1, 2.2]],
                  cell=[5, 5, 12], pbc=(True, True, False))
    atoms.set_tags([1, 0, 2])
    atoms.set_constraint(FixAtoms([0]))
    atoms.new_array('molecule_id', np.array([-1, 0, 0]))
    atoms.calc = EMT()  # its results must not leak into the frame
    buf = io.StringIO()
    write_xyz(atoms, -1.5, buf, extra={'step': 3, 'score': -2.0})
    back = ase.io.read(io.StringIO(buf.getvalue()), format='extxyz')
    assert list(back.pbc) == [True, True, False]
    assert [list(c.get_indices()) for c in back.constraints] == [[0]]
    np.testing.assert_array_equal(back.get_tags(), [1, 0, 2])
    np.testing.assert_array_equal(back.arrays['molecule_id'], [-1, 0, 0])
    assert back.get_potential_energy() == pytest.approx(-1.5)
    assert back.info['step'] == 3 and back.info['score'] == pytest.approx(-2.0)


# --------------------------------------------------------------------------
# ensembles-io-3: an exception in the energy call restores the snapshot
# --------------------------------------------------------------------------

class _Interrupting:
    def __init__(self):
        self.calls = 0

    def get_potential_energy(self, atoms):
        self.calls += 1
        if self.calls > 1:
            raise KeyboardInterrupt
        return 0.0


def test_interrupted_energy_call_leaves_no_unscored_trial():
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
              calc=_Interrupting())
    with pytest.raises(KeyboardInterrupt):
        g.do_gcmc_step()
    assert len(g.atoms) == 1
    np.testing.assert_array_equal(g.atoms.positions, [[1, 1, 1]])


# --------------------------------------------------------------------------
# ensembles-io-5, -6, -7: canonical constraints, failed setup, final row
# --------------------------------------------------------------------------

def _cu_cluster():
    return Atoms('Cu4', positions=[[0, 0, 0], [2.3, 0.2, 0], [0, 2.6, 0.1], [0.3, 0, 2.2]],
                 cell=[12] * 3, pbc=False)


def test_canonical_constraints_hold_during_the_initial_relaxation(tmp_path):
    atoms = _cu_cluster()
    before = atoms.positions[0].copy()
    ens = CanonicalEnsemble(atoms, EMT(), BFGS, MoveSelector([1], [ShakeMove(0.1, 1)]),
                            constraints=FixAtoms([0]), random_seed=1, steps=50,
                            traj_file=None, outfile=None)
    ens.run(2)
    np.testing.assert_allclose(ens.atoms.positions[0], before)


def test_canonical_writes_the_last_row_and_frame(tmp_path):
    out, traj = tmp_path / 'c.out', tmp_path / 'c.xyz'
    ens = CanonicalEnsemble(_cu_cluster(), EMT(), BFGS,
                            MoveSelector([1], [ShakeMove(0.1, 1)]), random_seed=1,
                            steps=20, outfile=str(out), traj_file=str(traj),
                            outfile_write_interval=10, trajectory_write_interval=10)
    ens.run(3)
    ens.run(3)
    steps = [line.split()[1] for line in open(out) if line.startswith('STEP:')]
    assert steps == ['0', '3', '6']
    assert len(ase.io.read(str(traj), index=':')) == 3


def test_failed_setup_does_not_break_a_retried_run(tmp_path):
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    out = tmp_path / 'o.out'
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
              outfile=str(out), traj_file=str(tmp_path / 't.xyz'))
    real = g.initialize_outfile
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk full')
        real()

    g.initialize_outfile = flaky
    with pytest.raises(OSError):
        g.run(2)
    g.run(2)
    text = out.read_text()
    assert 'Simulation Parameters' in text
    rows = [line.split()[0] for line in text.splitlines() if line[:1].isdigit()]
    assert rows == ['0', '1', '2']


# --------------------------------------------------------------------------
# ensembles-io-8, gcmc-physics-minima-step-offbyone
# --------------------------------------------------------------------------

def _minima_gcmc(tmp_path, **kw):
    atoms = Atoms(cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    return _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
                 mu={'H': 3.0}, minima_file=str(tmp_path / 'min.xyz'),
                 outfile=str(tmp_path / 'o.out'), traj_file=str(tmp_path / 't.xyz'), **kw)


def test_minima_history_starts_fresh_and_labels_the_outfile_step(tmp_path):
    (tmp_path / 'min.xyz').write_text('stale\n')
    g = _minima_gcmc(tmp_path)
    g.run(5)
    frames = ase.io.read(str(tmp_path / 'min.xyz'), index=':')
    traj = ase.io.read(str(tmp_path / 't.xyz'), index=':')
    # Each minimum is a frame of the trajectory with the same step label.
    n_at_step = {i: len(f) for i, f in enumerate(traj)}
    assert frames[0].info['step'] == 0
    for f in frames:
        assert len(f) == n_at_step[f.info['step']]
    assert [f.info['step'] for f in frames] == sorted({f.info['step'] for f in frames})
    assert max(f.info['step'] for f in frames) >= 1


# --------------------------------------------------------------------------
# ensembles-io-10: numpy integer seeds
# --------------------------------------------------------------------------

def test_numpy_integer_seed():
    a = RandomNumberGenerator(np.int64(5)).get_uniform()
    assert a == RandomNumberGenerator(5).get_uniform()
    DisplacementMove(['H'], seed=np.int32(3))


# --------------------------------------------------------------------------
# api-packaging-tests-docs-5: input validation before the first energy
# --------------------------------------------------------------------------

class _NoEnergy:
    def get_potential_energy(self, atoms):
        raise AssertionError('validation must run before the first energy call')


@pytest.mark.parametrize('T', [0.0, -5.0, float('nan'), float('inf')])
def test_gcmc_rejects_bad_temperature(T):
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    with pytest.raises(ValueError, match='temperature'):
        _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
              T=T, calc=_NoEnergy(), units='metal')


def test_canonical_rejects_negative_temperature():
    with pytest.raises(ValueError, match='temperature'):
        CanonicalEnsemble(None, None, None, None, temperature=-1.0,
                          traj_file=None, outfile=None)


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


# --------------------------------------------------------------------------
# api-packaging-tests-docs-8: MoveSelector weights
# --------------------------------------------------------------------------

@pytest.mark.parametrize('weights', [[1], [1, -1], [0, 0], [1, float('nan')],
                                     [1, float('inf')]])
def test_move_selector_rejects_bad_weights(weights):
    moves = [ShakeMove(0.1, 1), ShakeMove(0.1, 2)]
    with pytest.raises(ValueError):
        MoveSelector(weights, moves)


# --------------------------------------------------------------------------
# api-packaging-tests-docs-11: the outfile records the code version
# --------------------------------------------------------------------------

def test_outfile_header_records_the_mcpy_version(tmp_path):
    import mcpy
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    out = tmp_path / 'o.out'
    _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
          outfile=str(out)).run(1)
    assert f'mcpy version: {mcpy.__version__}' in out.read_text()


# --------------------------------------------------------------------------
# gcmc-physics-ideal-gas-test-gap: absolute metal-unit de Broglie wavelength
# --------------------------------------------------------------------------

def test_metal_units_de_broglie_wavelengths():
    from ase.units import kB
    u = SetUnits('metal', 300, ['Ar', 'O'])
    assert u.lambda_dbs['Ar'] == pytest.approx(0.15947, rel=1e-4)
    assert u.lambda_dbs['O'] == pytest.approx(0.25200, rel=1e-4)
    assert u.beta == pytest.approx(1.0 / (kB * 300), rel=1e-12)


# --------------------------------------------------------------------------
# performance-1 / molecules-9: vectorized find_molecules is unchanged
# --------------------------------------------------------------------------

def test_find_molecules_matches_per_molecule_com():
    """The vectorized grouping and COM pass gives the same candidates, in
    the same order, as a per-molecule loop, for molecules split across the
    periodic boundary."""
    rng = np.random.default_rng(0)
    atoms = Atoms(cell=[7, 7, 9], pbc=(True, True, False))
    ids = []
    for k in range(12):
        offsets = [[0, 0, 0], [0, 0, 1.1]] if k % 3 else [[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]
        mol = Atoms('CO' if k % 3 else 'OH2', positions=rng.uniform(0, 7, 3) + offsets)
        mol.positions[:, :2] %= 7.0  # some molecules straddle the boundary
        atoms += mol
        ids += [k] * len(mol)
    atoms += Atoms('Cu', positions=[[1, 1, 1]])
    atoms.new_array('molecule_id', np.array(ids + [-1]))
    cell = CustomCell(atoms, custom_height=5.0, bottom_z=2.0, seed=1)
    got = find_molecules(atoms, ['C', 'O'], cell)
    ref = [np.flatnonzero(atoms.arrays['molecule_id'] == k) for k in range(12) if k % 3]
    assert len(find_molecules(atoms, ['C', 'O'])) == len(ref) == 8
    ref = [m for m in ref if cell.is_point_exchangeable(molecule_com(atoms, m))]
    assert 0 < len(ref) < 8
    assert [list(g) for g in got] == [list(m) for m in ref]
