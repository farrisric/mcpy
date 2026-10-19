"""
Regression tests for the 2026-10 review findings on the cells.

- moves-cells-geometry-1: a CustomCell in a box with a tilted c vector must
  shear its sub-cell like the box, so a wrapped inserted atom stays
  deletable.
- gcmc-physics-insertion-attempt-cap: a nearly full region must still yield
  a free insertion point after the scalar draws give up.
- performance-2: the free-volume estimate picks its sample count from the
  previous free fraction, capped by ``mc_sample_points``.

Run with: python -m pytest tests/test_review_2026_10_cells.py -v
"""
import logging

import numpy as np
import pytest
from ase import Atom, Atoms

from mcpy.cell import Cell, CustomCell


def tilted_box():
    cell = [[8.0, 0.0, 0.0], [4.0, 6.9, 0.0], [2.5, 1.5, 20.0]]
    return Atoms('Ag', positions=[[1.0, 1.0, 2.0]], cell=cell, pbc=True)


def test_tilted_c_inserted_atom_stays_deletable_after_wrap():
    atoms = tilted_box()
    cell = CustomCell(atoms, custom_height=5.0, bottom_z=8.0,
                      species_radii={'Ag': 0.0, 'O': 0.0}, seed=3)
    for _ in range(500):
        trial = atoms.copy()
        trial.append(Atom('O', position=cell.get_random_point()))
        trial.wrap()
        assert len(trial) - 1 in cell.get_atoms_specie_inside_cell(trial, ['O'])
        assert cell.is_point_exchangeable(trial.positions[-1])
        assert cell.is_point_inside(trial.positions[-1])


def test_tilted_c_sub_cell_follows_box_shear():
    atoms = tilted_box()
    cell = CustomCell(atoms, custom_height=5.0, bottom_z=8.0, seed=0)
    c = np.array(atoms.cell)[2]
    np.testing.assert_allclose(cell.dimensions[2], c * 5.0 / 20.0)
    np.testing.assert_allclose(cell.offset, c * 8.0 / 20.0)
    assert cell.cell_volume == pytest.approx(atoms.get_volume() * 5.0 / 20.0)


def test_in_plane_vector_with_z_component_raises():
    atoms = Atoms('Ag', cell=[[8.0, 0.0, 0.5], [0.0, 8.0, 0.0], [0.0, 0.0, 20.0]],
                  pbc=True)
    with pytest.raises(ValueError, match='z component'):
        CustomCell(atoms, custom_height=5.0, bottom_z=8.0)


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


def test_open_region_uses_fewer_sample_points():
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[20.0] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'X': 2.0}, seed=0)
    assert cell.mc_sample_points == 100_000
    cell.calculate_volume(atoms)
    assert cell.mc_sample_points < 20_000
    # The cap is untouched: a later dense configuration may use all of it.
    assert cell.max_sample_points == 100_000


def test_dense_region_keeps_the_cap():
    atoms, cell = nearly_full_box()
    cell.calculate_volume(atoms)
    assert cell.mc_sample_points == cell.max_sample_points == 100_000


def test_sample_count_tracks_free_fraction_noise_target():
    """Relative noise sqrt((1-f)/(f N)) of the estimate meets the target
    whenever the cap allows it."""
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[10.0] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'X': 3.0}, mc_sample_points=10**7, seed=0)
    cell.calculate_volume(atoms)
    f = cell.get_volume() / cell.cell_volume
    n = cell.mc_sample_points
    assert np.sqrt((1 - f) / (f * n)) == pytest.approx(0.005, rel=0.01)


def test_small_cap_is_respected():
    atoms = Atoms('X', positions=[[5.0, 5.0, 5.0]], cell=[20.0] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'X': 2.0}, mc_sample_points=1000, seed=0)
    cell.calculate_volume(atoms)
    assert cell.mc_sample_points <= 1000
