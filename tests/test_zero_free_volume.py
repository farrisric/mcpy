"""
Tests that a fully-covered region never reports a free volume of exactly
zero. The MC estimator cannot distinguish "no free volume" from "less free
volume than one sample point resolves", and a literal 0.0 propagates into
``SetUnits.de_broglie_deletion`` as a division by zero: the deletion
prefactor becomes ``inf`` (every deletion auto-accepts) or ``inf * 0 = nan``
(every deletion silently rejects). The estimate is therefore floored at one
sample point's worth of the region volume, with a warning.

Run with: python -m pytest tests/test_zero_free_volume.py -v
"""
import logging

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from mcpy.cell import Cell, CustomCell, DomeCell


def test_custom_cell_full_region_floors_volume(caplog):
    atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8)
    # A radius this large covers every sample point in the 5.5 A region.
    cell = CustomCell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                      species_radii={'Ag': 50.0, 'O': 0},
                      mc_sample_points=1000, seed=0)
    with caplog.at_level(logging.WARNING):
        cell.calculate_volume(atoms)
    assert cell.get_volume() == pytest.approx(cell.cell_volume / 1000)
    assert cell.get_volume() > 0.0
    assert any('free volume' in r.message for r in caplog.records)


def test_dome_cell_full_region_floors_volume():
    atoms = Atoms('Ag', positions=[[0.0, 0.0, 5.0]],
                  cell=[10.0, 10.0, 10.0], pbc=False)
    cell = DomeCell(atoms, particle_species='Ag', bottom_z=4.0, vacuum=2.0,
                    species_radii={'Ag': 50.0}, mc_sample_points=1000, seed=0)
    cell.calculate_volume(atoms)
    assert cell.get_volume() == pytest.approx(cell.sphere_volume / 1000)
    assert cell.get_volume() > 0.0


def test_partial_coverage_is_not_floored():
    """A region with real free volume must keep its MC estimate."""
    atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8)
    cell = CustomCell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                      species_radii={'Ag': 2.11, 'O': 0},
                      mc_sample_points=20_000, seed=0)
    cell.calculate_volume(atoms)
    assert cell.get_volume() > 10 * cell.cell_volume / 20_000
    assert cell.get_volume() < cell.cell_volume
    assert np.isfinite(cell.get_volume())


def nearly_full_box():
    """One atom in a periodic 10 A cube whose sphere leaves only small holes
    at the cube corners: free fraction about 1e-4."""
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[10.0] * 3, pbc=True)
    return atoms, Cell(atoms, species_radii={'X': 8.4}, seed=1)


def test_nearly_full_region_still_yields_free_points():
    atoms, cell = nearly_full_box()
    cell.calculate_volume(atoms)
    f = cell.get_volume() / cell.cell_volume
    assert 1e-5 < f < 1e-3
    trees = cell._exclusion_trees(atoms)
    points = [cell.get_random_free_point(atoms) for _ in range(20)]
    assert all(p is not None for p in points)
    assert not cell._covered(np.array(points), trees).any()


def test_full_region_returns_none_and_warns(caplog):
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[10.0] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'X': 20.0}, mc_sample_points=2000, seed=1)
    with caplog.at_level(logging.WARNING):
        assert cell.get_random_free_point(atoms) is None
    assert any('insertion' in r.message for r in caplog.records)
