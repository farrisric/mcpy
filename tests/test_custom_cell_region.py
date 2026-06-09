"""
Tests that CustomCell's membership tests (``is_point_inside``,
``get_atoms_specie_inside_cell``) agree with the region actually sampled by
``get_random_point``. For non-orthogonal (e.g. hexagonal fcc111) cells the
in-plane lattice vector has an off-diagonal component, so an axis-aligned
box test mis-classifies ~25% of sampled points. That misclassification
corrupts deletion-candidate selection and the ``InsertionMove`` minimum-distance
bias (and, historically, under a per-species de Broglie count, the acceptance
factor itself -- see docs/gcmc_acceptance_convention.rst).

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


def test_subsurface_oxygen_is_excluded_from_deletion_candidates():
    """O that drifts below the cell floor (``z < bottom_z``, i.e. absorbed into
    the subsurface) is deliberately excluded from ``get_atoms_specie_inside_cell``.
    DeletionMove draws its candidates from that list, so buried O can never be
    selected for deletion and accumulates irreversibly -- a caveat of the
    cell-restricted region independent of the de Broglie count convention. See
    docs/gcmc_acceptance_convention.rst.
    """
    atoms, cell = make_cell()
    # Same xy (inside the footprint); one above the floor, one 1 A below it.
    inside = np.array([0.5, 0.5, 0.5]) @ cell.dimensions + cell.offset
    above = inside.copy()
    below = inside.copy()
    below[2] = cell.offset[2] - 1.0  # below bottom_z -> subsurface

    atoms.append(Atom('O', position=above))
    atoms.append(Atom('O', position=below))

    counted = cell.get_atoms_specie_inside_cell(atoms, ['O'])
    total_o = int(np.count_nonzero(np.asarray(atoms.get_chemical_symbols()) == 'O'))

    assert total_o == 2       # both O are present in the structure
    assert len(counted) == 1  # only the above-floor O is a deletion candidate
