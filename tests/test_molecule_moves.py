"""
Tests for molecular GCMC: molecule bookkeeping helpers, rigid-molecule
insertion/deletion moves, molecular SetUnits masses, and the
exchange-count wiring in the GCMC acceptance step.

torch/mace/mpi4py-free.

Run with: python -m pytest tests/test_molecule_moves.py -v
"""
import math

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule as build_molecule
from ase.constraints import FixAtoms

from mcpy.cell import Cell, CustomCell, SphericalCell
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import (DeletionMove, InsertionMove, MoleculeDeletionMove, MoleculeDisplacementMove,
                        MoleculeInsertionMove, MoveSelector, PermutationMove)
from mcpy.moves.base_move import PROPOSAL_REJECTED
from mcpy.moves.molecule_utils import find_molecules, molecule_com, molecule_groups
from mcpy.utils.set_unit_constant import SetUnits

from helpers import _o2, _ZeroCalc


def _box_atoms():
    return Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)


def _water_box():
    """Box with one H2O (id 0), one O2 (id 1), and one lone H (id -1)."""
    atoms = Atoms('OH2O2H',
                  positions=[[5.0, 5.0, 5.0],    # O  of H2O
                             [5.8, 5.0, 5.0],    # H  of H2O
                             [5.0, 5.8, 5.0],    # H  of H2O
                             [2.0, 2.0, 2.0],    # O  of O2
                             [2.0, 2.0, 3.2],    # O  of O2
                             [8.0, 8.0, 8.0]],   # lone H
                  cell=[10, 10, 10], pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0, 0, 1, 1, -1]))
    return atoms


def test_molecule_com_mic_across_boundary():
    # O2 straddling the z boundary: atoms at z=9.8 and z=0.6 (bond 0.8 via mic).
    atoms = Atoms('O2', positions=[[5, 5, 9.8], [5, 5, 0.6]],
                  cell=[10, 10, 10], pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    com = molecule_com(atoms, np.array([0, 1]))
    # Midpoint of the unwrapped pair (z = 10.2) wrapped back into the box.
    np.testing.assert_allclose(com, [5.0, 5.0, 0.2], atol=1e-10)


def test_setunits_molecular_mass_and_lambda():
    water = build_molecule('H2O')
    u = SetUnits('metal', temperature=300.0, species=['Ag'],
                 molecules={'H2O': water})
    assert u.masses['H2O'] == pytest.approx(float(water.get_masses().sum()))
    # Lambda follows the same formula as atomic species, with molecular mass.
    expected = (
        u.PLANCK_CONSTANT / np.sqrt(
            2 * np.pi * u.masses['H2O'] * u.mass_conversion_factor / u.beta)
    ) * u.lambda_conversion_factor
    assert u.lambda_dbs['H2O'] == pytest.approx(expected)
    # Atomic species still present and unchanged in form.
    assert 'Ag' in u.lambda_dbs


def test_setunits_lj_molecular_defaults():
    u = SetUnits('LJ', temperature=1.0, species=['X'],
                 molecules={'X2': Atoms('X2', positions=[[0, 0, 0], [0, 0, 1]])})
    assert u.masses['X2'] == 1
    assert u.lambda_dbs['X2'] == 1


def test_setunits_rejects_isomer_compositions():
    a = Atoms('CO2', positions=[[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
    b = Atoms('OCO', positions=[[0, 0, 1.3], [0, 0, 0], [0, 0, -1.3]])
    with pytest.raises(ValueError, match='composition'):
        SetUnits('metal', temperature=300.0, species=[],
                 molecules={'co2_linear': a, 'co2_alt': b})


def _water_template():
    return build_molecule('H2O')


def test_molecule_insertion_adds_whole_molecule():
    atoms = _box_atoms()  # 2 H atoms, no molecule_id array yet
    move = MoleculeInsertionMove(Cell(atoms), _water_template(), 'H2O', seed=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is atoms
    assert delta == 1 and name == 'H2O'
    assert len(atoms) == 5
    # Lazy array creation: originals get -1, the new molecule a fresh id.
    ids = atoms.arrays['molecule_id']
    np.testing.assert_array_equal(ids[:2], [-1, -1])
    assert set(ids[2:]) == {0}
    # Rigid insertion: internal distances match the template.
    template = _water_template()
    d_new = atoms.get_all_distances()[2:, 2:]
    d_tmpl = template.get_all_distances()
    np.testing.assert_allclose(np.sort(d_new.ravel()), np.sort(d_tmpl.ravel()),
                               atol=1e-10)
    assert move.last_exchange_count == 0


def test_molecule_insertion_max_molecules_cap():
    atoms = _water_box()  # already contains one H2O
    move = MoleculeInsertionMove(Cell(atoms), _water_template(), 'H2O',
                                 seed=1, max_molecules=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False
    assert delta == 1 and name == 'H2O'
    assert len(atoms) == 6  # unchanged


def test_molecule_deletion_no_candidates():
    atoms = _box_atoms()  # no molecule_id array at all
    move = MoleculeDeletionMove(Cell(atoms), _water_template(), 'H2O', seed=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False
    assert delta == -1 and name == 'H2O'
    assert len(atoms) == 2


def test_molecule_deletion_min_molecules_floor():
    atoms = _water_box()
    move = MoleculeDeletionMove(Cell(atoms), _water_template(), 'H2O',
                                seed=1, min_molecules=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False
    assert len(atoms) == 6


class _NoMove:
    species, cell = ['X'], None

    def do_trial_move(self, atoms):
        return False, 0, 'X'


class _StubCalc:
    def get_potential_energy(self, atoms):
        return -1.0 * len(atoms)


class _HugeUphillCalc:
    """Forces rejection of every move (acceptance probability ~ 0)."""

    def get_potential_energy(self, atoms):
        return 1.0e6 * len(atoms)


def _mol_gcmc(atoms, move_selector, calculator, mu_h2o=0.0):
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[Cell(atoms)], units_type='LJ',
        calculator=calculator, mu={'H2O': mu_h2o}, species=[],
        temperature=1.0, move_selector=MoveSelector([1], [_NoMove()]), random_seed=3,
        traj_file=None, outfile=None,
        molecules={'H2O': _water_template(), 'O2': Atoms('O2')},
    )
    # The selector under test may hold one exchange move only, which the
    # constructor's insertion/deletion balance check refuses.
    g.move_selector = move_selector
    return g


def test_gcmc_rejected_molecule_move_rolls_back():
    atoms = _water_box()
    n_before = len(atoms)
    ids_before = atoms.arrays['molecule_id'].copy()
    ms = MoveSelector(
        [1], [MoleculeInsertionMove(Cell(atoms), _water_template(), 'H2O', seed=1)],
        seed=2)
    g = _mol_gcmc(atoms, ms, _HugeUphillCalc(), mu_h2o=-1.0e9)
    g.do_gcmc_step()
    assert len(g.atoms) == n_before
    np.testing.assert_array_equal(g.atoms.arrays['molecule_id'], ids_before)


def test_replica_exchange_rejects_molecular_species():
    """MPI ReplicaExchange's per-species swap bookkeeping counts atoms by
    symbol (docs/gcmc_acceptance_convention.rst), which is always 0 for a
    molecular name; the constructor must refuse rather than silently drop
    the mu*N correction (or accept-always on a mu-ladder)."""
    pytest.importorskip('mpi4py')
    from mcpy.ensembles.replica_exchange import ReplicaExchange

    class _FakeUnits:
        molecules = {'H2O': object()}

    class _FakeGCMC:
        units = _FakeUnits()

    def factory(mu=None, rank=None):
        return _FakeGCMC()

    with pytest.raises(NotImplementedError, match='molecular species'):
        ReplicaExchange(factory, mus=[{'H2O': 1.0}])


def test_replica_exchange_accepts_units_less_ensemble():
    """CanonicalEnsemble has no ``units`` attribute (it stores _beta
    directly) and is a supported temperature-ladder ensemble: the molecule
    guard must let units-less ensembles through, not die on AttributeError."""
    pytest.importorskip('mpi4py')
    from mcpy.ensembles.replica_exchange import ReplicaExchange

    class _FakeCanonical:
        pass  # deliberately no `units` attribute

    def factory(T=None, rank=None):
        return _FakeCanonical()

    re = ReplicaExchange(factory, temperatures=[300.0])
    assert isinstance(re.gcmc, _FakeCanonical)


def test_batched_re_grand_potential_counts_molecules():
    """BatchedReplicaExchange._grand_potential delegates to _minimum_score,
    which counts whole molecules via find_molecules; Counter(get_chemical_
    symbols()) would count 'H2O' 0 times and silently drop the mu*N term."""
    atoms = _water_box()  # one H2O, one O2, one lone H
    ms = MoveSelector(
        [1], [MoleculeDeletionMove(Cell(atoms), _water_template(), 'H2O', seed=1)],
        seed=2)
    g = _mol_gcmc(atoms, ms, _StubCalc(), mu_h2o=2.0)
    expected = g.E_old - 2.0 * 1  # one H2O molecule, not zero
    assert BatchedReplicaExchange._grand_potential(g) == pytest.approx(expected)


def test_write_xyz_molecule_id_roundtrip(tmp_path):
    import ase.io

    from mcpy.ensembles.base_ensemble import write_xyz

    atoms = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]], cell=[8, 8, 8], pbc=True)
    atoms.new_array('molecule_id', np.array([7, 7]))
    path = str(tmp_path / 'frame.xyz')
    write_xyz(atoms, -1.0, path)
    back = ase.io.read(path)
    np.testing.assert_array_equal(back.arrays['molecule_id'], [7, 7])


def test_molecule_deletion_picks_among_multiple_candidates():
    """With two O2 candidates, repeated trials (fresh atoms, different seeds)
    must be able to pick either one -- not always the first found -- and
    last_exchange_count must report both candidates every time."""
    o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])

    def _two_o2_box():
        atoms = Atoms('O4', positions=[[1, 1, 1], [1, 1, 2.2],
                                       [5, 5, 5], [5, 5, 6.2]],
                      cell=[10, 10, 10], pbc=True)
        atoms.new_array('molecule_id', np.array([0, 0, 1, 1]))
        return atoms

    deleted_ids = set()
    for seed in range(1, 21):
        atoms = _two_o2_box()
        move = MoleculeDeletionMove(Cell(atoms), o2, 'O2', seed=seed)
        result, delta, name = move.do_trial_move(atoms)
        assert result is atoms
        assert move.last_exchange_count == 2
        remaining_ids = set(atoms.arrays['molecule_id'].tolist())
        deleted_ids |= {0, 1} - remaining_ids
    assert deleted_ids == {0, 1}


def test_atomic_insertion_alongside_molecules_stays_free():
    # ASE ``extend`` zero-pads arrays missing from the fragment; without the
    # explicit -1 tag the inserted atom would join molecule id 0.
    atoms = _water_box()  # molecule ids 0 (H2O) and 1 (O2) present
    move = InsertionMove(Cell(atoms), species=['H'], seed=7)
    result, delta, species = move.do_trial_move(atoms)
    assert result is atoms and delta == 1
    assert atoms.arrays['molecule_id'][-1] == -1
    # H2O (id 0) still has exactly its own three members
    assert (atoms.arrays['molecule_id'] == 0).sum() == 3


def test_atomic_deletion_never_picks_molecule_members():
    # One free O plus O-containing molecules: atomic O deletion must always
    # take the free atom, whatever the seed.
    for seed in range(8):
        atoms = _water_box()
        atoms += Atoms('O', positions=[[8.0, 2.0, 2.0]])
        atoms.arrays['molecule_id'][-1] = -1
        move = DeletionMove(Cell(atoms), species=['O'], seed=seed)
        result, delta, species = move.do_trial_move(atoms)
        assert result is atoms and delta == -1
        ids = atoms.arrays['molecule_id']
        assert (ids == 0).sum() == 3  # H2O intact
        assert (ids == 1).sum() == 2  # O2 intact
        # And with no free O left, the move reports a failed proposal.
        again = move.do_trial_move(atoms)
        assert again[0] is False


def test_molecule_displacement_rigid_and_bounded():
    atoms = _water_box()
    d_before = atoms.get_all_distances(mic=True)[np.ix_([0, 1, 2], [0, 1, 2])]
    com_before = molecule_com(atoms, np.array([0, 1, 2]))
    move = MoleculeDisplacementMove(Cell(atoms), _water_template(), 'H2O',
                                    seed=3, max_displacement=0.4)
    result, delta, name = move.do_trial_move(atoms)
    assert result is atoms and delta == 0 and name == 'H2O'
    # Rigid: internal distances preserved; only the H2O moved (O2 untouched).
    d_after = atoms.get_all_distances(mic=True)[np.ix_([0, 1, 2], [0, 1, 2])]
    np.testing.assert_allclose(d_after, d_before, atol=1e-10)
    com_after = molecule_com(atoms, np.array([0, 1, 2]))
    assert 0 < np.linalg.norm(com_after - com_before) <= 0.4 + 1e-9
    np.testing.assert_array_equal(atoms.arrays['molecule_id'],
                                  [0, 0, 0, 1, 1, -1])


def test_molecule_displacement_no_candidates():
    atoms = _box_atoms()
    move = MoleculeDisplacementMove(Cell(atoms), _water_template(), 'H2O', seed=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False and delta == 0
    assert len(atoms) == 2


def test_molecule_displacement_capped_rotation():
    # With a small max_angle the orientation change per trial is bounded:
    # the O->H bond direction rotates by less than max_angle.
    atoms = _water_box()
    v_before = atoms.positions[1] - atoms.positions[0]
    move = MoleculeDisplacementMove(Cell(atoms), _water_template(), 'H2O',
                                    seed=9, max_displacement=0.3,
                                    max_angle=0.3)
    move.do_trial_move(atoms)
    v_after = atoms.positions[1] - atoms.positions[0]
    cosang = np.dot(v_before, v_after) / (
        np.linalg.norm(v_before) * np.linalg.norm(v_after))
    assert np.arccos(np.clip(cosang, -1, 1)) <= 0.3 + 1e-9
    # Rigidity still holds.
    np.testing.assert_allclose(np.linalg.norm(v_after),
                               np.linalg.norm(v_before), atol=1e-10)


def test_setunits_rejects_atomic_molecular_name_collision():
    with pytest.raises(ValueError, match='both an atomic species'):
        SetUnits('metal', temperature=300.0, species=['O'],
                 molecules={'O': Atoms('O2', positions=[[0, 0, 0],
                                                        [0, 0, 1.2]])})


class _ZWindowCell(Cell):
    """Bounded stub region: COM must sit in a z window."""

    def __init__(self, atoms, z_lo, z_hi):
        super().__init__(atoms)
        self.z_lo, self.z_hi = z_lo, z_hi

    def is_point_inside(self, point):
        return self.z_lo <= point[2] < self.z_hi


def test_molecule_displacement_mic_rigidity():
    # Molecule straddling a periodic boundary stays rigid under displacement.
    atoms = Atoms('O2', positions=[[5, 5, 9.8], [5, 5, 0.6]],
                  cell=[10, 10, 10], pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 0.8]])
    move = MoleculeDisplacementMove(Cell(atoms), o2, 'O2', seed=4,
                                    max_displacement=0.5)
    result, delta, name = move.do_trial_move(atoms)
    assert result is atoms
    d = atoms.get_distance(0, 1, mic=True)
    np.testing.assert_allclose(d, 0.8, atol=1e-9)


class _PushOut:
    """Zero energy. Like a relaxation that desorbs: each call moves every
    molecule 0.8 A away from ``center``."""

    def __init__(self, center):
        self.center = np.asarray(center)

    def get_potential_energy(self, atoms):
        for members in molecule_groups(atoms).values():
            away = atoms.positions[members].mean(axis=0) - self.center
            atoms.positions[members] += 0.8 * away / np.linalg.norm(away)
        return 0.0


def test_relaxation_cannot_carry_a_molecule_out_of_the_ball():
    """A trial whose energy call pushes a molecule's center of mass out of
    a SphericalCell is rejected: outside the ball the molecule could not be
    deleted or counted, so it would stay there for the rest of the run."""
    host = Atoms('Ag2', positions=[[9, 10, 10], [11, 10, 10]], cell=[20] * 3, pbc=True)
    sphere = SphericalCell(host, vacuum=3.0, species_radii=None, seed=1)
    h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    moves = [MoleculeInsertionMove(sphere, h2, 'H2', seed=2),
             MoleculeDeletionMove(sphere, h2, 'H2', seed=3)]
    g = GrandCanonicalEnsemble(
        atoms=host, cells=[sphere], units_type='LJ', calculator=_PushOut(sphere.center),
        mu={'H2': 2.0}, species=[], molecules={'H2': h2}, temperature=1.0,
        move_selector=MoveSelector([1, 1], moves, seed=4), random_seed=5,
        traj_file=None, outfile=None)
    for _ in range(60):
        g.do_gcmc_step()
    groups = find_molecules(g.atoms, ['H', 'H'])
    assert groups, 'no molecule was accepted: the guard rejects every trial'
    distances = [np.linalg.norm(molecule_com(g.atoms, members) - sphere.center)
                 for members in groups]
    assert max(distances) <= sphere.radius


# PermutationMove never touches molecule members

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


# Position moves honour FixAtoms

def test_b2_molecule_displacement_honours_fixatoms():
    atoms = Atoms('O2', positions=[[5, 5, 5], [5, 5, 6.2]], cell=[10] * 3, pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    atoms.set_constraint(FixAtoms(indices=[0]))
    before = atoms.positions.copy()
    move = MoleculeDisplacementMove(Cell(atoms), _o2(), 'O2', seed=2)
    result, _, _ = move.do_trial_move(atoms)
    assert result is atoms
    np.testing.assert_array_equal(atoms.positions[0], before[0])


# Molecule displacement out of the region counts as a rejection, not a failure

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


# Atomic mu keys do not count atoms that belong to molecules


def _h2():
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])


# Molecule displacement keeps the COM inside the proposal region

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


# Vectorized find_molecules is unchanged

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
