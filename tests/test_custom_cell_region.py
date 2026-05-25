"""
Tests that CustomCell's membership tests (``is_point_inside``,
``get_atoms_specie_inside_cell``) agree with the region actually sampled by
``get_random_point``. For non-orthogonal (e.g. hexagonal fcc111) cells the
in-plane lattice vector has an off-diagonal component, so an axis-aligned
box test mis-classifies ~25% of sampled points. That undercount made
``n_species_before`` go negative and ``de_broglie_insertion`` return inf,
causing unconditional acceptance of insertions.

Run with: python -m pytest tests/test_custom_cell_region.py -v
"""
import numpy as np
import pytest  # noqa: F401
from ase import Atom
from ase.build import fcc111

from mcpy.cell import CustomCell


def make_cell():
    atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8)
    return atoms, CustomCell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                             species_radii={'Ag': 2.11, 'O': 0}, seed=0)


def test_sampled_points_are_reported_inside():
    """Every point from get_random_point must pass is_point_inside."""
    _, cell = make_cell()
    for _ in range(2000):
        assert cell.is_point_inside(cell.get_random_point())


def test_inserted_atom_is_counted():
    """An atom placed at a sampled point must be counted inside the cell."""
    atoms, cell = make_cell()
    for _ in range(2000):
        trial = atoms.copy()
        trial.append(Atom('O', position=cell.get_random_point()))
        idx = cell.get_atoms_specie_inside_cell(trial, ['O'])
        assert len(idx) == 1
