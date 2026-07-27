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


# --------------------------------------------------------------------------
# Exchangeable region vs proposal region (bug: molecules that desorbed above
# the cell top stopped being deletion candidates AND dropped out of the
# per-species de Broglie count, driving runaway insertion)
# --------------------------------------------------------------------------

def _point_at(cell, frac_z):
    """A point in the middle of the footprint at fractional height ``frac_z``
    (``> 1`` is above the cell top)."""
    return np.array([0.5, 0.5, frac_z]) @ cell.dimensions + cell.offset


def test_point_above_the_cell_top_is_exchangeable_but_not_inside():
    """The two predicates are deliberately different: ``is_point_inside``
    bounds the *proposal* region, ``is_point_exchangeable`` the region the
    reservoir can take molecules back from -- the same asymmetry
    ``get_atoms_specie_inside_cell`` already applies to single atoms."""
    _, cell = make_cell()
    escaped = _point_at(cell, 1.4)
    assert not cell.is_point_inside(escaped)
    assert cell.is_point_exchangeable(escaped)


def test_point_below_the_cell_floor_is_neither():
    _, cell = make_cell()
    buried = _point_at(cell, -0.2)
    assert not cell.is_point_inside(buried)
    assert not cell.is_point_exchangeable(buried)


def test_point_outside_the_xy_footprint_is_neither():
    _, cell = make_cell()
    outside = np.array([1.4, 0.5, 0.5]) @ cell.dimensions + cell.offset
    assert not cell.is_point_inside(outside)
    assert not cell.is_point_exchangeable(outside)


def test_desorbed_molecule_stays_a_deletion_candidate():
    """A CO whose center of mass drifts above the cell top must remain
    findable, exactly as its individual C and O atoms already do. Otherwise it
    can never be deleted and its absence from ``last_exchange_count`` inflates
    V/((N+1)Lambda^3) -- the runaway insertion mode documented in
    docs/gcmc_acceptance_convention.rst.
    """
    from ase import Atoms

    from mcpy.moves.molecule_utils import find_molecules

    atoms, cell = make_cell()
    atoms.new_array('molecule_id', np.full(len(atoms), -1, dtype=int))
    for mol_id, frac_z in enumerate((0.5, 1.4)):  # inside, and above the top
        frag = Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.13]])
        frag.positions += _point_at(cell, frac_z)
        frag.new_array('molecule_id', np.full(len(frag), mol_id, dtype=int))
        atoms += frag

    template = sorted(['C', 'O'])
    assert len(find_molecules(atoms, template)) == 2
    assert len(find_molecules(atoms, template, cell)) == 2
    # The atomic path has always seen all four member atoms; the molecular
    # path must not disagree with it.
    assert len(cell.get_atoms_specie_inside_cell(atoms, ['C', 'O'])) == 4


def test_buried_molecule_is_still_excluded():
    """The floor exclusion is intentional (buried species are kept), so the
    fix above must not open that side of the region too."""
    from ase import Atoms

    from mcpy.moves.molecule_utils import find_molecules

    atoms, cell = make_cell()
    atoms.new_array('molecule_id', np.full(len(atoms), -1, dtype=int))
    frag = Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.13]])
    frag.positions += _point_at(cell, -0.5)
    frag.new_array('molecule_id', np.zeros(len(frag), dtype=int))
    atoms += frag

    assert find_molecules(atoms, sorted(['C', 'O']), cell) == []
