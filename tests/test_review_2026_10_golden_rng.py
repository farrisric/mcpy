"""The adaptive free-volume sample count must not shift the seeded cell stream.

The count follows the free fraction, which follows GPU-noisy relaxed positions.
If the draw were count-sized, one sample point flipping across an exclusion
sphere would move every later insertion point (the Ag(111) golden diverged).
"""
import numpy as np
import pytest
from ase.build import fcc111

from mcpy.cell import CustomCell, DomeCell, SphericalCell
from ase.cluster import Octahedron


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
