"""Cell behaviour shared by all regions: the insertion exclusion rule,
periodic images, the free-volume sampler and NullCell.
"""
import inspect

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.cluster import Octahedron

from mcpy.cell import BaseCell, Cell, CustomCell, DomeCell, NullCell, SphericalCell
from mcpy.moves import DeletionMove, InsertionMove, MoleculeInsertionMove

from helpers import _mic_dists, _o2


# The insertion exclusion sees every atom, including the slab below the
# region floor and periodic images

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


def test_b1_unplaceable_insertion_fails_cleanly():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'H': 50.0}, seed=5)
    for move in (InsertionMove(cell, species=['H'], seed=6),
                 MoleculeInsertionMove(cell, _o2(), 'O2', seed=6)):
        assert move.do_trial_move(atoms)[0] is False
        assert len(atoms) == 1


# Spherical regions must fit the periodic box; the dome must be non-empty

def test_p8_spherical_cell_larger_than_box_raises():
    atoms = Octahedron('Ag', 4, 1)
    atoms.set_cell([12, 12, 12])
    atoms.pbc = True
    with pytest.raises(ValueError, match='periodic'):
        SphericalCell(atoms, vacuum=5.0, species_radii={'Ag': 1.0})


# SphericalCell centers on the box, so wrapping keeps the particle whole

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


# NullCell refuses region queries with a clear message

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


# The sphere sampler fills its quota in one batch

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


# BaseCell interface

def test_base_cell_interface_signatures():
    assert 'atoms' in inspect.signature(BaseCell.calculate_volume).parameters
    assert 'is_point_inside' in BaseCell.__abstractmethods__


# Minimum-image handling (bug: min_insert distance check and CustomCell free
# volume ignored periodic images)

def test_insertion_exclusion_respects_minimum_image():
    # Existing atom at x=0.5 in a 10 A periodic box with a 1.0 A exclusion
    # radius: no insertion may land within 1.0 A of it through the boundary.
    atoms = Atoms('H', positions=[[0.5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'H': 1.0}, seed=2)
    move = InsertionMove(cell, species=['H'], seed=1)
    for _ in range(300):
        move.do_trial_move(atoms)
        assert atoms.get_distance(0, 1, mic=True) >= 1.0
        del atoms[1]


def test_missing_species_radius_raises_a_named_error():
    atoms = Atoms('Cu2O', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]],
                  cell=[10.0, 10.0, 10.0])
    cell = CustomCell(atoms, custom_height=5.0, bottom_z=0.0,
                      species_radii={'Cu': 2.4}, mc_sample_points=64, seed=0)
    with pytest.raises(ValueError, match=r"no radius for \['O'\]"):
        cell.calculate_volume(atoms)


def test_open_region_uses_fewer_sample_points():
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[20.0] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'X': 2.0}, seed=0)
    assert cell.mc_sample_points == 100_000
    cell.calculate_volume(atoms)
    assert cell.mc_sample_points < 20_000
    # The cap is untouched: a later dense configuration may use all of it.
    assert cell.max_sample_points == 100_000


def _slab_cell():
    atoms = fcc111('Ag', a=4.165, size=(3, 3, 2), periodic=True, vacuum=8.0)
    z_top = atoms.positions[:, 2].max()
    return atoms, CustomCell(atoms, custom_height=5.0, bottom_z=z_top - 2.0,
                             species_radii={'Ag': 2.0}, seed=5, mc_sample_points=20_000)


def _sphere_cell():
    atoms = Octahedron('Ag', 3)
    atoms.center(vacuum=8.0)
    return atoms, SphericalCell(atoms, vacuum=3.0, species_radii={'Ag': 2.0}, seed=5,
                                mc_sample_points=20_000)


def _dome_cell():
    atoms = Octahedron('Ag', 3)
    atoms.center(vacuum=8.0)
    bottom = atoms.positions[:, 2].min() + 1.0
    return atoms, DomeCell(atoms, particle_species=['Ag'], vacuum=3.0, bottom_z=bottom,
                           species_radii={'Ag': 2.0}, seed=5, mc_sample_points=20_000)


@pytest.mark.parametrize('build', [_slab_cell, _sphere_cell, _dome_cell])
def test_stream_independent_of_adapted_count(build):
    points = []
    for n in (6_000, 7_000):
        atoms, cell = build()
        cell._n_samples = n  # what one flipped sample point can do to the count
        cell.free_volume(atoms)
        points.append(cell.get_random_point())
    np.testing.assert_array_equal(points[0], points[1])


def test_sample_count_tracks_free_fraction_noise_target():
    """Relative noise sqrt((1-f)/(f N)) of the estimate meets the target
    whenever the cap allows it. The sphere fills 38 percent of the box, so the
    target needs about 25,000 points: well above the minimum count."""
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[10.0] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'X': 4.5}, mc_sample_points=10**7, seed=0)
    cell.calculate_volume(atoms)
    f = cell.get_volume() / cell.cell_volume
    n = cell.mc_sample_points
    assert np.sqrt((1 - f) / (f * n)) == pytest.approx(0.005, rel=0.01)
