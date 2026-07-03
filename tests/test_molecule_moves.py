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
