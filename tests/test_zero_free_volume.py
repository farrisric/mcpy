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

from mcpy.cell import CustomCell, DomeCell, SphericalCell


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


def test_spherical_cell_full_region_floors_volume():
    atoms = Atoms('Ag', positions=[[0.0, 0.0, 0.0]])
    cell = SphericalCell(atoms, vacuum=2.0, species_radii={'Ag': 50.0},
                         mc_sample_points=1000, seed=0)
    cell.calculate_volume(atoms)
    assert cell.get_volume() == pytest.approx(cell.sphere_volume / 1000)
    assert cell.get_volume() > 0.0


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
