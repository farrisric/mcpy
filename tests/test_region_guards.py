"""Exchanged atoms may not cross the ball of a SphericalCell or DomeCell,
or the top of a CustomCell.
"""
import math

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from mcpy.cell import CustomCell, SphericalCell
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, PermutationMove

from helpers import _gcmc, _ZeroCalc


# Exchanged atoms may not cross the ball surface

def test_spherical_cell_displacement_ideal_gas_matches_exact_distribution():
    """Ideal H around a fixed Cu in a SphericalCell, with large atomic
    displacements. Exact: P(N) ~ x^N / (N+1)!, x = zV (the +1 is the Cu in
    the total-atom count). Without the ball guard H atoms random-walk out
    of the sphere, cannot be deleted and pile up."""
    atoms = Atoms('Cu', positions=[[10, 10, 10]], cell=[20] * 3, pbc=False)
    atoms.set_constraint(FixAtoms([0]))
    cell = SphericalCell(atoms, vacuum=3.0, species_radii=None, seed=1)
    x = 3.0
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, max_displacement=2.0)],
              mu={'H': math.log(x / cell.get_volume())})
    counts = []
    for step in range(30_000):
        g.do_gcmc_step()
        if step >= 2_000:
            counts.append(len(g.atoms) - 1)
    weights = np.array([x ** n / math.factorial(n + 1) for n in range(40)])
    exact = (weights / weights.sum()) @ np.arange(40)
    assert np.mean(counts) == pytest.approx(exact, abs=0.12)
    d = np.linalg.norm(g.atoms.positions - cell.center, axis=1)
    assert np.all(d <= cell.radius)


def test_ball_guard_rejects_a_crossing_and_spares_the_dome_floor():
    """A trial that carries an exchanged atom out of the ball is rejected,
    one that only crosses the DomeCell support floor is not."""
    from mcpy.cell import DomeCell
    atoms = Atoms('Ag2O', positions=[[5, 5, 5], [6, 5, 5], [5, 5, 6]],
                  cell=[12] * 3, pbc=True)
    dome = DomeCell(atoms, 'Ag', bottom_z=4.5, vacuum=2.0, species_radii=None, seed=1)
    g = _gcmc(atoms, [dome], [InsertionMove(dome, ['O'], 1), DeletionMove(dome, ['O'], 2)],
              species=('O',))
    snapshot = ({k: v.copy() for k, v in g.atoms.arrays.items()}, [])
    g.atoms.positions[2] = [5.5, 5, 4.0]  # below the floor, inside the ball
    assert g._outside_ball_counts(snapshot[0]) == g._outside_ball_counts(g.atoms.arrays)
    g.atoms.positions[2] = [5.5, 5, 9.0]  # outside the ball
    assert not g._commit_or_rollback(-100.0, snapshot, (0, 'X'))
    np.testing.assert_allclose(g.atoms.positions[2], [5, 5, 6])


# Cell refresh, three-state ideal gas


class _Floor:
    """Zero energy above ``z0``, 1e6 when any atom sinks below it (a
    repulsive support)."""

    def __init__(self, z0):
        self.z0 = z0

    def get_potential_energy(self, atoms):
        return 1e6 if np.any(atoms.positions[:, 2] < self.z0) else 0.0


def test_custom_cell_displacement_ideal_gas_stays_poisson():
    """Ideal H in a thin CustomCell slab over a repulsive floor, with many
    atomic displacements. Exact: the in-cell count is Poisson(zV). The top
    is a hard wall for displacements; without it atoms climb above the top
    (deletable but never insertable) and the in-cell density drops."""
    atoms = Atoms(cell=[6, 6, 20], pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=3.0, bottom_z=5.0, seed=1)
    zv = 3.0
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, max_displacement=1.0)],
              weights=[1, 1, 10], mu={'H': math.log(zv / cell.get_volume())},
              calc=_Floor(5.0))
    counts = []
    for step in range(20_000):
        g.do_gcmc_step()
        if step >= 1_000 and step % 4 == 0:
            counts.append(sum(cell.is_point_inside(p) for p in g.atoms.positions))
    assert np.mean(counts) == pytest.approx(zv, abs=0.2)
    assert np.all(g.atoms.positions[:, 2] < 8.0)


class _Counting(_ZeroCalc):
    calls = 0

    def get_potential_energy(self, atoms):
        self.calls += 1
        return 0.0


def test_custom_cell_top_crossing_is_rejected_before_the_energy_call():
    """An H just below the top: about half the displacements would cross.
    Those are rejected attempts with no energy call; the rest are accepted."""
    atoms = Atoms('H', positions=[[1, 1, 7.99]], cell=[6, 6, 20], pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=3.0, bottom_z=5.0, seed=1)
    calc = _Counting()
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              DisplacementMove(['H'], 4, max_displacement=1.0)],
              weights=[0, 0, 1], calc=calc)
    for _ in range(100):
        g.atoms.positions[0, 2] = 7.99
        g.do_gcmc_step()
        assert g.atoms.positions[0, 2] < 8.0
    ms = g.move_selector
    assert ms.move_counter_total[2] == 100 and ms.move_failed_counter_total[2] == 0
    assert 20 < calc.calls - 1 == ms.move_acceptance_total[2] < 80


def test_permutation_cannot_swap_an_exchanged_species_above_the_top():
    """A swap keeps every position, so only the species above the top
    changes: H below the top swapped with Au above it would put an H where
    deletion reaches it but no insertion can. Rejected with no energy call."""
    atoms = Atoms('HAu', positions=[[1, 1, 6.0], [3, 3, 9.0]], cell=[6, 6, 20],
                  pbc=(True, True, False))
    cell = CustomCell(atoms, custom_height=3.0, bottom_z=5.0, seed=1)
    calc = _Counting()
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3),
                              PermutationMove(['H', 'Au'], 4)],
              weights=[0, 0, 1], calc=calc)
    for _ in range(20):
        g.do_gcmc_step()
        assert g.atoms.get_chemical_symbols() == ['H', 'Au']
    assert calc.calls == 1
    assert g.move_selector.move_acceptance_total[2] == 0
