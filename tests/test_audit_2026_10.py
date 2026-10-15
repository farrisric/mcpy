"""Regression tests for the 2026-10 audit items in moves/ and cell/.

Each test names the audit item it guards (P1, B1, P3, ...). torch-free.
"""
import inspect
import math
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.geometry import find_mic

from mcpy.cell import BaseCell, Cell, CustomCell, DomeCell, NullCell, SphericalCell
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import (BrownianMove, DeletionMove, DisplacementMove,
                        InsertionMove, MoleculeDeletionMove,
                        MoleculeDisplacementMove, MoleculeInsertionMove,
                        MoveSelector, PermutationMove, ShakeMove)
from mcpy.moves.base_move import PROPOSAL_REJECTED


def _o2():
    return Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])


def _mic_dists(atoms, point):
    diff = find_mic(atoms.positions - point, atoms.cell, pbc=atoms.pbc)[0]
    return np.linalg.norm(diff, axis=1)


# --------------------------------------------------------------------------
# P1: the deletion acceptance uses the free volume of the post-deletion state
# --------------------------------------------------------------------------

def test_p1_deletion_volume_is_post_deletion_free_volume():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = CustomCell(atoms, custom_height=10.0, bottom_z=0.0,
                      species_radii={'H': 2.0}, mc_sample_points=20_000, seed=1)
    cell.calculate_volume(atoms)
    assert cell.get_volume() < 990.0  # the H sphere is excluded
    selector = MoveSelector([1], [DeletionMove(cell, species=['H'], seed=2)])
    result, delta, _ = selector.do_trial_move(atoms)
    assert result is atoms and delta == -1 and len(atoms) == 0
    # The reverse insertion starts from the empty box: V = 1000 exactly.
    assert selector.get_volume() == pytest.approx(1000.0)


def test_p1_molecule_deletion_volume_is_post_deletion_free_volume():
    atoms = Atoms('O2', positions=[[5, 5, 5], [5, 5, 6.2]],
                  cell=[10, 10, 10], pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    cell = CustomCell(atoms, custom_height=10.0, bottom_z=0.0,
                      species_radii={'O': 1.5}, mc_sample_points=20_000, seed=1)
    cell.calculate_volume(atoms)
    assert cell.get_volume() < 990.0
    selector = MoveSelector([1], [MoleculeDeletionMove(cell, _o2(), 'O2', seed=2)])
    result, delta, _ = selector.do_trial_move(atoms)
    assert result is atoms and len(atoms) == 0
    assert selector.get_volume() == pytest.approx(1000.0)


def test_p1_insertion_volume_is_pre_insertion_free_volume():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'H': 2.0}, mc_sample_points=50_000, seed=1)
    cell.calculate_volume(atoms)
    v_before = cell.get_volume()
    expected = 1000.0 - 4.0 / 3.0 * math.pi * 8.0
    assert v_before == pytest.approx(expected, rel=0.02)
    selector = MoveSelector([1], [InsertionMove(cell, species=['H'], seed=2)])
    selector.do_trial_move(atoms)
    assert len(atoms) == 2
    assert selector.get_volume() == v_before


def test_min_insert_is_a_seed_clearance_adr_0004():
    """min_insert seeds insertions anywhere outside its own clearance, inside
    the species_radii spheres too, while the acceptance volume stays the
    species_radii free volume."""
    atoms = Atoms('Cu', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'Cu': 2.0, 'H': 0.5},
                mc_sample_points=50_000, seed=1)
    cell.calculate_volume(atoms)
    v_free = cell.get_volume()
    assert v_free == pytest.approx(1000.0 - 4.0 / 3.0 * math.pi * 8.0, rel=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        move = InsertionMove(cell, species=['H'], seed=2, min_insert=0.5)
        MoleculeInsertionMove(cell, _o2(), 'O2', seed=1, min_insert=0.5)
    selector = MoveSelector([1], [move], seed=3)
    dists = []
    for _ in range(3000):
        selector.do_trial_move(atoms)
        dists.append(atoms.get_distance(0, 1, mic=True))
        del atoms[1]
    dists = np.array(dists)
    assert dists.min() >= 0.5
    assert np.any(dists < 2.0)
    assert selector.get_volume() == v_free


class _ZeroCalc:
    def get_potential_energy(self, atoms):
        return 0.0


def test_p1_two_state_gcmc_obeys_detailed_balance():
    """Ideal H (no energy) next to a fixed Cu obstacle, capped at one H.

    The stationary ratio P(1)/P(0) is V0 * exp(beta*mu) / ((N_tot+1) Lambda^3)
    with V0 the free volume of the Cu-only state (LJ units, Lambda = 1,
    N_tot = 1). mu is chosen so the ratio is 3, so P(1) = 0.75. Insertion is
    always accepted here, so P(1)/P(0) is set by the deletion acceptance:
    a deletion volume taken from the N+1 state (H sphere still excluded)
    gives P(1) ~ 0.69, and proposals that ignore the obstacle put H inside
    the Cu sphere.
    """
    r_cu, r_h, box = 3.0, 4.0, 10.0
    atoms = Atoms('Cu', positions=[[0.5, 5.0, 5.0]], cell=[box] * 3, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    v0 = box ** 3 - 4.0 / 3.0 * math.pi * r_cu ** 3
    mu = math.log(3.0 * 2.0 / v0)
    cell = Cell(atoms, species_radii={'Cu': r_cu, 'H': r_h},
                mc_sample_points=4000, seed=11)
    selector = MoveSelector(
        [1, 1], [DeletionMove(cell, species=['H'], seed=12),
                 InsertionMove(cell, species=['H'], seed=13, max_atoms=1)],
        seed=14, n_moves=1)
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ', calculator=_ZeroCalc(),
        mu={'H': mu}, species=['H'], temperature=1.0,
        move_selector=selector, random_seed=15, traj_file=None, outfile=None)

    n_trials, occupied = 12_000, 0
    for _ in range(n_trials):
        gcmc.do_gcmc_step()
        if len(gcmc.atoms) == 2:
            occupied += 1
            assert _mic_dists(gcmc.atoms[:1], gcmc.atoms.positions[1])[0] >= r_cu
    assert occupied / n_trials == pytest.approx(0.75, abs=0.025)


# --------------------------------------------------------------------------
# B1: the insertion exclusion sees every atom, including the slab below the
# region floor and periodic images
# --------------------------------------------------------------------------

def test_b1_insertion_avoids_atoms_below_region_floor():
    slab = fcc111('Ag', size=(3, 3, 3), vacuum=8.0, periodic=True)
    z_top = slab.positions[:, 2].max()
    radius = 2.0
    cell = CustomCell(slab, custom_height=4.0, bottom_z=z_top + 0.3,
                      species_radii={'Ag': radius, 'O': 0.0}, seed=3)
    move = InsertionMove(cell, species=['O'], seed=4)
    n0 = len(slab)
    for _ in range(300):
        result, _, _ = move.do_trial_move(slab)
        assert result is slab
        assert _mic_dists(slab[:n0], slab.positions[-1]).min() >= radius
        del slab[-1]


def test_b1_insertion_exclusion_uses_periodic_images():
    atoms = Atoms('H', positions=[[0.2, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'H': 1.5}, seed=5)
    move = InsertionMove(cell, species=['H'], seed=6)
    for _ in range(500):
        move.do_trial_move(atoms)
        assert _mic_dists(atoms[:1], atoms.positions[-1])[0] >= 1.5
        del atoms[-1]


def test_b1_unplaceable_insertion_fails_cleanly():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'H': 50.0}, seed=5)
    for move in (InsertionMove(cell, species=['H'], seed=6),
                 MoleculeInsertionMove(cell, _o2(), 'O2', seed=6)):
        assert move.do_trial_move(atoms)[0] is False
        assert len(atoms) == 1


# --------------------------------------------------------------------------
# P3: PermutationMove never touches molecule members
# --------------------------------------------------------------------------

def test_p3_permutation_skips_molecule_members():
    atoms = Atoms('Pt3CO', positions=[[0, 0, 0], [2.7, 0, 0], [0, 2.7, 0],
                                      [5, 5, 5], [5, 5, 6.1]],
                  cell=[12] * 3, pbc=True)
    atoms.new_array('molecule_id', np.array([-1, -1, -1, 0, 0]))
    move = PermutationMove(species=['Pt', 'C', 'O'], seed=1, n_swaps=3)
    for _ in range(20):
        result, _, _ = move.do_trial_move(atoms)
        assert result is False  # only Pt is free: nothing to swap with
    assert atoms.get_chemical_symbols() == ['Pt', 'Pt', 'Pt', 'C', 'O']

    atoms += Atoms('O', positions=[[8, 8, 8]])
    atoms.arrays['molecule_id'][-1] = -1
    for _ in range(20):
        move.do_trial_move(atoms)
        assert list(atoms.symbols[3:5]) == ['C', 'O']


# --------------------------------------------------------------------------
# B2: position moves honour FixAtoms
# --------------------------------------------------------------------------

def _fixed_pair():
    atoms = Atoms('Ag2O', positions=[[0, 0, 0], [3, 0, 0], [6, 6, 6]],
                  cell=[12] * 3, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    return atoms


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


def test_b2_displacement_species_restricts_candidates():
    atoms = Atoms('Ag2O', positions=[[0, 0, 0], [3, 0, 0], [6, 6, 6]],
                  cell=[12] * 3, pbc=True)
    move = DisplacementMove(species=['O'], seed=1, max_displacement=0.5)
    for _ in range(20):
        before = atoms.positions.copy()
        move.do_trial_move(atoms)
        np.testing.assert_array_equal(atoms.positions[:2], before[:2])


def test_b2_displacement_index_constraints_still_work():
    atoms = Atoms('Ag2O', positions=[[0, 0, 0], [3, 0, 0], [6, 6, 6]],
                  cell=[12] * 3, pbc=True)
    move = DisplacementMove(species=['Ag', 'O'], seed=1, constraints=[0, 2])
    before = atoms.positions.copy()
    move.do_trial_move(atoms)
    np.testing.assert_array_equal(atoms.positions[[0, 2]], before[[0, 2]])


def test_b2_molecule_displacement_honours_fixatoms():
    atoms = Atoms('O2', positions=[[5, 5, 5], [5, 5, 6.2]], cell=[10] * 3, pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    atoms.set_constraint(FixAtoms(indices=[0]))
    before = atoms.positions.copy()
    move = MoleculeDisplacementMove(Cell(atoms), _o2(), 'O2', seed=2)
    result, _, _ = move.do_trial_move(atoms)
    assert result is atoms
    np.testing.assert_array_equal(atoms.positions[0], before[0])


# --------------------------------------------------------------------------
# B6: BrownianMove is seeded and leaves no residue on the atoms
# --------------------------------------------------------------------------

def _cu_cluster():
    atoms = Octahedron('Cu', 3, 1)
    atoms.center(vacuum=5.0)
    return atoms


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


# --------------------------------------------------------------------------
# P8: spherical regions must fit the periodic box; the dome must be non-empty
# --------------------------------------------------------------------------

def test_p8_spherical_cell_larger_than_box_raises():
    atoms = Octahedron('Ag', 4, 1)
    atoms.set_cell([12, 12, 12])
    atoms.pbc = True
    with pytest.raises(ValueError, match='periodic'):
        SphericalCell(atoms, vacuum=5.0, species_radii={'Ag': 1.0})


def test_p8_dome_cell_larger_than_box_raises():
    slab = fcc111('Ag', size=(3, 3, 3), vacuum=10.0, periodic=True)
    z_top = slab.positions[:, 2].max()
    slab += Atoms('Ag', positions=[slab.cell.sum(axis=0) * [0.5, 0.5, 0]
                                   + [0, 0, z_top + 2.3]])
    with pytest.raises(ValueError, match='periodic'):
        DomeCell(slab, 'Ag', bottom_z=z_top, vacuum=8.0,
                 species_radii={'Ag': 1.0})


def test_p8_dome_cell_entirely_below_floor_raises():
    atoms = Atoms('Ag', positions=[[5, 5, 5]], cell=[20] * 3, pbc=True)
    with pytest.raises(ValueError, match='bottom_z'):
        DomeCell(atoms, 'Ag', bottom_z=9.0, vacuum=3.0, species_radii={'Ag': 1.0})


# --------------------------------------------------------------------------
# P9: SphericalCell centers on the box, so wrapping keeps the particle whole
# --------------------------------------------------------------------------

def test_p9_spherical_cell_survives_wrap_in_periodic_box():
    atoms = Octahedron('Ag', 5, 1)
    atoms.set_cell([30, 30, 30])
    atoms.pbc = True
    cell = SphericalCell(atoms, vacuum=3.0, species_radii={'Ag': 1.0})
    np.testing.assert_allclose(cell.center, [15, 15, 15])
    n_inside = len(cell.get_atoms_specie_inside_cell(atoms, 'Ag'))
    assert n_inside == len(atoms)
    atoms.wrap()
    assert len(cell.get_atoms_specie_inside_cell(atoms, 'Ag')) == n_inside


# --------------------------------------------------------------------------
# B11: NullCell refuses region queries with a clear message
# --------------------------------------------------------------------------

def test_b11_null_cell_region_queries_raise_clearly():
    cell = NullCell()
    atoms = Atoms('H', positions=[[0, 0, 0]])
    with pytest.raises(TypeError, match='NullCell'):
        cell.is_point_inside(np.zeros(3))
    with pytest.raises(TypeError, match='NullCell'):
        cell.get_atoms_specie_inside_cell(atoms, 'H')
    with pytest.raises(TypeError, match='NullCell'):
        InsertionMove(cell, species=['H'], seed=1).do_trial_move(atoms)
    with pytest.raises(TypeError, match='NullCell'):
        DeletionMove(cell, species=['H'], seed=1).do_trial_move(atoms)


# --------------------------------------------------------------------------
# O3: the sphere sampler fills its quota in one batch
# --------------------------------------------------------------------------

def test_o3_sphere_sampler_needs_one_batch():
    cell = SphericalCell(Octahedron('Ag', 3, 1), vacuum=3.0,
                         species_radii={'Ag': 1.0}, seed=0)
    calls = []

    class _CountingRng:
        def __init__(self, rng):
            self.rng = rng

        def random(self, *args, **kwargs):
            calls.append(args)
            return self.rng.random(*args, **kwargs)

    cell._rng = _CountingRng(cell._rng)
    for _ in range(5):
        calls.clear()
        cell._sample_sphere_points(100_000)
        assert len(calls) == 1


# --------------------------------------------------------------------------
# Molecule displacement out of the region counts as a rejection, not a failure
# --------------------------------------------------------------------------

class _ZWindowCell(Cell):
    def __init__(self, atoms, z_lo, z_hi):
        super().__init__(atoms)
        self.z_lo, self.z_hi = z_lo, z_hi

    def is_point_inside(self, point):
        return self.z_lo <= point[2] < self.z_hi


def test_molecule_displacement_out_of_region_is_a_rejection():
    atoms = Atoms('O2', positions=[[5, 5, 5], [5, 5, 6.2]], cell=[10] * 3, pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    cell = _ZWindowCell(atoms, 5.5, 5.7)  # any real shift leaves the window
    move = MoleculeDisplacementMove(cell, _o2(), 'O2', seed=1, max_displacement=2.0)
    assert move.do_trial_move(atoms)[0] is PROPOSAL_REJECTED
    selector = MoveSelector([1], [move])
    result = selector.do_trial_move(atoms)
    assert result[0] is False  # the ensembles still see "nothing to score"
    assert selector.move_counter == [1]
    assert selector.move_failed_counter == [0]
    assert selector.interval_ratios() == [0.0]


# --------------------------------------------------------------------------
# BaseCell interface
# --------------------------------------------------------------------------

def test_base_cell_interface_signatures():
    assert 'atoms' in inspect.signature(BaseCell.calculate_volume).parameters
    assert 'is_point_inside' in BaseCell.__abstractmethods__
