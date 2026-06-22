"""Tests for DomeCell: a hemispherical insertion region centered on a supported
nanoparticle, truncated at the support surface (z >= bottom_z).

These guard the supported-NP requirement: insertions stay on/around the particle
(and its metal-support rim), not on the bare support far away, and never below
the surface.

Run with: python -m pytest tests/test_dome_cell.py -v
"""
import numpy as np
import pytest  # noqa: F401
from ase import Atoms, Atom

from mcpy.cell import DomeCell

BOTTOM_Z = 0.0
VACUUM = 1.0
RADII = {'Al': 1.0, 'Ag': 1.0, 'O': 0.0}

# Ag particle deliberately off-center in xy (x=14) so its centroid differs from
# the whole-system center, and low/tall enough that the enclosing ball dips below
# the surface (so the z-clip is actually exercised).
AG_POS = np.array([
    [14.0, 10.0, 0.5],
    [14.0, 10.0, 5.0],
    [16.0, 10.0, 2.5],
    [12.0, 10.0, 2.5],
    [14.0, 12.0, 2.5],
    [14.0, 8.0, 2.5],
])


def make_supported():
    """Flat Al support layer at z=0 with an off-center Ag cluster above it."""
    xs = np.linspace(1.0, 19.0, 7)
    support = [(x, y, 0.0) for x in xs for y in xs]
    positions = support + [tuple(p) for p in AG_POS]
    symbols = ['Al'] * len(support) + ['Ag'] * len(AG_POS)
    return Atoms(symbols, positions=positions, cell=[20, 20, 20], pbc=True)


def make_cell():
    return DomeCell(make_supported(), particle_species='Ag', bottom_z=BOTTOM_Z,
                    vacuum=VACUUM, species_radii=RADII, seed=0)


def test_center_is_particle_centroid():
    """Dome centers on the particle, not the whole-system center of mass."""
    cell = make_cell()
    np.testing.assert_allclose(cell.center, AG_POS.mean(axis=0), atol=1e-9)


def test_radius_covers_particle_plus_vacuum():
    cell = make_cell()
    expected = np.linalg.norm(AG_POS - AG_POS.mean(axis=0), axis=1).max() + VACUUM
    assert cell.radius == pytest.approx(expected)


def test_sampled_points_are_inside():
    """Every sampled point passes is_point_inside (region self-consistency)."""
    cell = make_cell()
    for _ in range(2000):
        assert cell.is_point_inside(cell.get_random_point())


def test_sampled_points_above_surface():
    cell = make_cell()
    for _ in range(2000):
        assert cell.get_random_point()[2] >= BOTTOM_Z


def test_far_support_point_is_outside():
    """A point on the bare support far from the particle is excluded."""
    cell = make_cell()
    assert not cell.is_point_inside(np.array([1.0, 1.0, 0.1]))


def test_subsurface_point_is_outside():
    """A point below the support surface is excluded even if within the ball."""
    cell = make_cell()
    below = cell.center.copy()
    below[2] = BOTTOM_Z - 0.5
    assert not cell.is_point_inside(below)


def test_inserted_atom_is_counted():
    cell = make_cell()
    atoms = make_supported()
    for _ in range(500):
        trial = atoms.copy()
        trial.append(Atom('O', position=cell.get_random_point()))
        idx = cell.get_atoms_specie_inside_cell(trial, ['O'])
        assert len(idx) == 1


def test_subsurface_atom_is_not_counted():
    """O absorbed below the surface is kept (not exchangeable with reservoir)."""
    cell = make_cell()
    atoms = make_supported()
    atoms.append(Atom('O', position=[cell.center[0], cell.center[1], BOTTOM_Z - 0.5]))
    idx = cell.get_atoms_specie_inside_cell(atoms, ['O'])
    assert len(idx) == 0


def test_dome_volume_is_a_fraction_of_the_ball():
    """Free dome volume is positive and below the full enclosing ball volume."""
    cell = make_cell()
    atoms = make_supported()
    cell.calculate_volume(atoms)
    full_ball = (4.0 / 3.0) * np.pi * cell.radius ** 3
    assert 0.0 < cell.get_volume() < full_ball
