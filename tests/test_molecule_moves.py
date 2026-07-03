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
