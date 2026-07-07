"""
Tests for molecular GCMC: molecule bookkeeping helpers, rigid-molecule
insertion/deletion moves, molecular SetUnits masses, and the
exchange-count wiring in the GCMC acceptance step.

torch/mace/mpi4py-free, matching the style of test_audit_regressions.py.

Run with: python -m pytest tests/test_molecule_moves.py -v
"""
import numpy as np
import pytest
from ase import Atoms

from mcpy.cell import Cell


def _box_atoms():
    return Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)


def test_box_cell_is_point_inside_always_true():
    cell = Cell(_box_atoms())
    assert cell.is_point_inside(np.array([5.0, 5.0, 5.0]))
    assert cell.is_point_inside(np.array([-100.0, 0.0, 1e6]))


from mcpy.moves.molecule_utils import (find_molecules, molecule_com,
                                       random_rotation_matrix)
from mcpy.utils import RandomNumberGenerator


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


def test_molecule_com_plain():
    atoms = _water_box()
    members = np.array([3, 4])
    com = molecule_com(atoms, members)
    np.testing.assert_allclose(com, [2.0, 2.0, 2.6])


def test_molecule_com_mic_across_boundary():
    # O2 straddling the z boundary: atoms at z=9.8 and z=0.6 (bond 0.8 via mic).
    atoms = Atoms('O2', positions=[[5, 5, 9.8], [5, 5, 0.6]],
                  cell=[10, 10, 10], pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    com = molecule_com(atoms, np.array([0, 1]))
    # Midpoint of the unwrapped pair (z = 10.2) wrapped back into the box.
    np.testing.assert_allclose(com, [5.0, 5.0, 0.2], atol=1e-10)


def test_molecule_com_is_mass_weighted():
    # H2O has heterogeneous masses: compare against ASE's own COM (independent
    # oracle; the molecule does not straddle a boundary) and confirm the result
    # differs from the unweighted mean.
    atoms = _water_box()
    members = np.array([0, 1, 2])
    com = molecule_com(atoms, members)
    np.testing.assert_allclose(com, atoms[members].get_center_of_mass())
    assert not np.allclose(com, atoms.positions[members].mean(axis=0))


def test_find_molecules_filters_composition():
    atoms = _water_box()
    water = find_molecules(atoms, sorted(['O', 'H', 'H']))
    assert len(water) == 1
    np.testing.assert_array_equal(water[0], [0, 1, 2])
    oxygen = find_molecules(atoms, sorted(['O', 'O']))
    assert len(oxygen) == 1
    np.testing.assert_array_equal(oxygen[0], [3, 4])


def test_find_molecules_missing_array_and_cell_filter():
    plain = _box_atoms()
    assert find_molecules(plain, ['H', 'H']) == []

    class _NowhereCell:
        def is_point_inside(self, point):
            return False

    atoms = _water_box()
    assert find_molecules(atoms, sorted(['O', 'H', 'H']), _NowhereCell()) == []


def test_random_rotation_matrix_is_rotation():
    rng = RandomNumberGenerator(seed=7)
    r1 = random_rotation_matrix(rng)
    r2 = random_rotation_matrix(rng)
    for r in (r1, r2):
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
    # Consecutive draws differ: orientation is actually random.
    assert not np.allclose(r1, r2)


from ase.build import molecule as build_molecule

from mcpy.utils.set_unit_constant import SetUnits


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


from mcpy.moves import MoleculeInsertionMove


def _water_template():
    return build_molecule('H2O')


def _empty_box():
    atoms = Atoms(cell=[10, 10, 10], pbc=True)
    return atoms


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


def test_molecule_insertion_orientations_vary():
    move = MoleculeInsertionMove(Cell(_empty_box()), _water_template(), 'H2O', seed=3)
    vecs = []
    for _ in range(2):
        atoms = _empty_box()
        move.do_trial_move(atoms)
        # O->H1 bond direction as an orientation fingerprint.
        vecs.append(atoms.positions[1] - atoms.positions[0])
    assert not np.allclose(vecs[0], vecs[1])


def test_molecule_insertion_max_molecules_cap():
    atoms = _water_box()  # already contains one H2O
    move = MoleculeInsertionMove(Cell(atoms), _water_template(), 'H2O',
                                 seed=1, max_molecules=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False
    assert delta == 1 and name == 'H2O'
    assert len(atoms) == 6  # unchanged


def test_molecule_insertion_next_id_after_existing():
    atoms = _water_box()  # ids 0 (H2O) and 1 (O2) present
    move = MoleculeInsertionMove(Cell(atoms), _water_template(), 'H2O', seed=1)
    move.do_trial_move(atoms)
    assert atoms.arrays['molecule_id'].max() == 2
    assert move.last_exchange_count == 1  # one H2O before the move


def test_molecule_insertion_min_insert_unplaceable():
    # Cell whose species list covers the existing atom; min_insert larger
    # than any possible separation in the box makes placement impossible.
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[4, 4, 4], pbc=True)
    cell = Cell(atoms, species_radii={'H': 1.0})
    move = MoleculeInsertionMove(cell, _water_template(), 'H2O',
                                 seed=1, min_insert=50.0)
    result, delta, name = move.do_trial_move(atoms)
    assert result is False
    assert len(atoms) == 1


from mcpy.moves import MoleculeDeletionMove


def test_molecule_deletion_removes_whole_molecule():
    atoms = _water_box()
    move = MoleculeDeletionMove(Cell(atoms), _water_template(), 'H2O', seed=1)
    result, delta, name = move.do_trial_move(atoms)
    assert result is atoms
    assert delta == -1 and name == 'H2O'
    # The H2O (3 atoms) is gone; the O2 and lone H remain.
    assert len(atoms) == 3
    assert sorted(atoms.get_chemical_symbols()) == ['H', 'O', 'O']
    assert move.last_exchange_count == 1


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


def test_molecule_deletion_ignores_other_compositions():
    atoms = _water_box()
    o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])
    move = MoleculeDeletionMove(Cell(atoms), o2, 'O2', seed=1)
    move.do_trial_move(atoms)
    # Only the O2 was deleted; the water is intact.
    assert len(atoms) == 4
    assert sorted(atoms.get_chemical_symbols()) == ['H', 'H', 'H', 'O']


from mcpy.moves import DeletionMove, MoveSelector


def test_move_selector_exchange_count():
    atoms = _water_box()
    cell = Cell(atoms)
    mol_move = MoleculeDeletionMove(cell, _water_template(), 'H2O', seed=1)
    atomic_move = DeletionMove(cell, species=['H'], seed=2)

    ms = MoveSelector([1, 0], [mol_move, atomic_move], seed=3)
    ms.do_trial_move(atoms)  # weight 0 on atomic: molecule move chosen
    assert ms.get_exchange_count() == 1

    ms2 = MoveSelector([0, 1], [mol_move, atomic_move], seed=3)
    ms2.do_trial_move(atoms)
    assert ms2.get_exchange_count() is None


from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble


class _StubCalc:
    def get_potential_energy(self, atoms):
        return -1.0 * len(atoms)


class _HugeUphillCalc:
    """Forces rejection of every move (acceptance probability ~ 0)."""

    def get_potential_energy(self, atoms):
        return 1.0e6 * len(atoms)


def _mol_gcmc(atoms, move_selector, calculator, mu_h2o=0.0):
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[Cell(atoms)], units_type='LJ',
        calculator=calculator, mu={'H2O': mu_h2o}, species=[],
        temperature=1.0, move_selector=move_selector, random_seed=3,
        traj_file=None, outfile=None,
        molecules={'H2O': _water_template()},
    )


def test_gcmc_acceptance_uses_molecule_count():
    atoms = _water_box()  # 6 atoms, one H2O molecule
    ms = MoveSelector(
        [1], [MoleculeDeletionMove(Cell(atoms), _water_template(), 'H2O', seed=1)],
        seed=2)
    g = _mol_gcmc(atoms, ms, _StubCalc())

    captured = {}
    original = g._acceptance_condition

    def spy(potential_diff, delta_particles, volume, species, n_atoms=None):
        captured['n'] = n_atoms
        return original(potential_diff, delta_particles, volume, species, n_atoms)

    g._acceptance_condition = spy
    g.do_gcmc_step()
    # Molecule count (1 H2O), not the 6-atom total.
    assert captured['n'] == 1


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


def test_gcmc_minimum_score_counts_molecules():
    atoms = _water_box()
    ms = MoveSelector(
        [1], [MoleculeDeletionMove(Cell(atoms), _water_template(), 'H2O', seed=1)],
        seed=2)
    g = _mol_gcmc(atoms, ms, _StubCalc(), mu_h2o=2.0)
    # Omega = E - mu * N_molecules = E - 2.0 * 1
    assert g._minimum_score(atoms, -6.0) == pytest.approx(-8.0)


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


from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange


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


def test_gcmc_molecule_smoke_lj():
    """Short LJ-units GCMC run with O2 insert/delete: no crash, bookkeeping
    stays consistent (every molecule id maps to exactly one O2)."""
    o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])
    atoms = Atoms(cell=[8, 8, 8], pbc=True)
    # species_radii non-empty so cell.get_species() is non-empty and the
    # min_insert retry path below actually runs against real neighbors
    # (an empty species_radii makes get_atoms_specie_inside_cell always
    # return no indices, silently skipping the overlap filter).
    cell = Cell(atoms, species_radii={'O': 0.0})
    ms = MoveSelector(
        [1, 1],
        [MoleculeInsertionMove(cell, o2, 'O2', seed=11, min_insert=0.8),
         MoleculeDeletionMove(cell, o2, 'O2', seed=12)],
        seed=13,
    )
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ',
        calculator=_StubCalc(), mu={'O2': 1.0}, species=[],
        temperature=1.0, move_selector=ms, random_seed=14,
        traj_file=None, outfile=None,
        molecules={'O2': o2},
    )
    seen_n = set()
    for _ in range(60):
        g.do_gcmc_step()
        seen_n.add(len(g.atoms))
        ids = g.atoms.arrays.get('molecule_id')
        if ids is not None:
            for mid in np.unique(ids):
                if mid < 0:
                    continue
                members = np.where(ids == mid)[0]
                assert sorted(np.asarray(g.atoms.get_chemical_symbols())[members]) \
                    == ['O', 'O']
    assert np.isfinite(g.E_old)
    assert len(seen_n) > 1  # N actually fluctuated


def test_write_xyz_molecule_id_roundtrip(tmp_path):
    import ase.io

    from mcpy.ensembles.base_ensemble import write_xyz

    atoms = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]], cell=[8, 8, 8], pbc=True)
    atoms.new_array('molecule_id', np.array([7, 7]))
    path = str(tmp_path / 'frame.xyz')
    write_xyz(atoms, -1.0, path)
    back = ase.io.read(path)
    np.testing.assert_array_equal(back.arrays['molecule_id'], [7, 7])


def test_find_molecules_queries_com_point():
    """The point handed to cell.is_point_inside must be the molecule's
    center of mass, not e.g. the first member's position."""
    atoms = _water_box()
    recorded = {}

    class _RecordingCell:
        def is_point_inside(self, point):
            recorded['point'] = point
            return True

    find_molecules(atoms, sorted(['O', 'H', 'H']), _RecordingCell())
    np.testing.assert_allclose(recorded['point'],
                               molecule_com(atoms, np.array([0, 1, 2])))


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


from mcpy.moves import InsertionMove  # noqa: E402


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
