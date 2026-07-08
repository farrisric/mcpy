"""Regression tests for the 2026-07 full-library audit.

Each test pins the fixed behavior of one audited defect. All tests are
torch/mace/mpi4py-free (stub calculators, EMT where a real potential is
needed) so they run in the lightweight CI matrix.
"""
import math
import random
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from mcpy.cell import Cell, CustomCell, NullCell
from mcpy.ensembles.canonical_ensemble import CanonicalEnsemble
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.ensembles.replica_exchange import ReplicaExchange
from mcpy.moves import (AlchemiBrownianMove, DeletionMove, DisplacementMove,
                        InsertionMove, MoveSelector, PermutationMove)
from mcpy.moves.base_move import BaseMove
from mcpy.utils.set_unit_constant import SetUnits
from mcpy.utils.utils import overlap_volume


class StubCalc:
    """Energy = -1 eV per atom; ``run_md`` shifts every position by +1 A."""

    def get_potential_energy(self, atoms):
        return -1.0 * len(atoms)

    def run_md(self, atoms, **kwargs):
        atoms.positions += 1.0


def _gcmc(atoms, cells, move_selector, mu, units_type='metal'):
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=cells, units_type=units_type, calculator=StubCalc(),
        mu=mu, species=list(mu), temperature=300.0,
        move_selector=move_selector, random_seed=3,
        traj_file=None, outfile=None,
    )


# --------------------------------------------------------------------------
# LJ unit system defines beta (bug: AttributeError on first trial move)
# --------------------------------------------------------------------------

def test_lj_units_define_beta():
    assert SetUnits('LJ', temperature=1.0, species=['X']).beta == 1.0
    assert SetUnits('LJ', temperature=2.0, species=['X']).beta == pytest.approx(0.5)


class _UphillCalc:
    """Every evaluation costs more than the last, forcing the uphill
    Metropolis branch (which reads units.beta) on each trial."""

    def __init__(self):
        self.e = 0.0

    def get_potential_energy(self, atoms):
        self.e += 1.0
        return self.e


def test_gcmc_runs_in_lj_units():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    ms = MoveSelector([1], [DisplacementMove(species=['H'], seed=1)], seed=2)
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[Cell(atoms)], units_type='LJ',
        calculator=_UphillCalc(), mu={'H': 0.0}, species=['H'],
        temperature=1.0, move_selector=ms, random_seed=3,
        traj_file=None, outfile=None,
    )
    g.do_gcmc_step()  # uphill move -> exercises the beta-dependent branch
    assert np.isfinite(g.E_old)


# --------------------------------------------------------------------------
# overlap_volume analytic value (bug: returned exactly twice the lens volume)
# --------------------------------------------------------------------------

def test_overlap_volume_matches_analytic():
    # Equal spheres radius r at distance d: V = pi*(4r + d)*(2r - d)^2 / 12
    r, d = 1.0, 1.0
    exact = math.pi * (4 * r + d) * (2 * r - d) ** 2 / 12.0
    assert overlap_volume(r, r, d) == pytest.approx(exact)


def test_overlap_volume_limits():
    assert overlap_volume(1.0, 1.0, 2.5) == 0  # disjoint
    engulfed = overlap_volume(2.0, 1.0, 0.5)   # small sphere inside big one
    assert engulfed == pytest.approx((4 / 3) * math.pi)
    # Touching from inside: d -> r1 + r2 gives vanishing overlap.
    assert overlap_volume(1.0, 1.0, 1.999999) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# NullCell honors the Cell interface (bug: TypeError in calculate_volume)
# --------------------------------------------------------------------------

def test_nullcell_calculate_volume_accepts_atoms():
    NullCell().calculate_volume(Atoms('H', positions=[[0, 0, 0]]))


def test_gcmc_accepts_nullcell_in_cells_list():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    ms = MoveSelector([1], [DisplacementMove(species=['H'], seed=1)], seed=2)
    g = _gcmc(atoms, [NullCell()], ms, mu={'H': 0.0})
    g.do_gcmc_step()  # construction and volume refresh must not raise


# --------------------------------------------------------------------------
# Deleting the last atom is a real trial move (bug: empty Atoms is falsy ->
# mutation kept with no accept/reject bookkeeping)
# --------------------------------------------------------------------------

def _single_atom_gcmc(mu_h):
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    box = Cell(atoms)
    ms = MoveSelector([1], [DeletionMove(box, species=['H'], seed=1)], seed=2)
    return _gcmc(atoms, [box], ms, mu={'H': mu_h}), ms


def test_deletion_of_last_atom_accepted_updates_state():
    # mu = -2 eV makes the deletion overwhelmingly favorable at 300 K.
    g, ms = _single_atom_gcmc(mu_h=-2.0)
    g.do_gcmc_step()
    assert len(g.atoms) == 0
    assert g.n_atoms == 0
    assert g.E_old == pytest.approx(0.0)
    assert ms.move_failed_counter_total == [0]
    assert ms.move_acceptance_total == [1]


def test_deletion_of_last_atom_rejected_rolls_back():
    # mu = +2 eV makes the deletion overwhelmingly unfavorable at 300 K.
    g, ms = _single_atom_gcmc(mu_h=2.0)
    g.do_gcmc_step()
    assert len(g.atoms) == 1
    assert g.n_atoms == 1
    assert g.E_old == pytest.approx(-1.0)
    assert ms.move_failed_counter_total == [0]
    assert ms.move_acceptance_total == [0]


# --------------------------------------------------------------------------
# AlchemiBrownianMove follows the in-place GCMC contract (bug: proposal was
# built on a copy the ensemble never looked at -> silent no-op move)
# --------------------------------------------------------------------------

def test_alchemi_brownian_move_mutates_in_place():
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    p0 = atoms.positions.copy()
    move = AlchemiBrownianMove(StubCalc(), temperature=300.0, seed=1)
    atoms_new, delta, species = move.do_trial_move(atoms)
    assert atoms_new is atoms
    assert not np.allclose(atoms.positions, p0)
    assert delta == 0


def test_gcmc_accepted_brownian_proposal_moves_ensemble_atoms():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    ms = MoveSelector([1], [AlchemiBrownianMove(StubCalc(), temperature=300.0,
                                                seed=1)], seed=2)
    g = _gcmc(atoms, [Cell(atoms)], ms, mu={'H': 0.0})
    p0 = g.atoms.positions.copy()
    g.do_gcmc_step()
    assert ms.move_acceptance_total == [1]  # dE = 0 -> always accepted
    assert not np.allclose(g.atoms.positions, p0)


class _CopyReturningMove(BaseMove):
    """Deliberate contract violation: returns a copy instead of mutating."""

    def __init__(self, seed=1):
        super().__init__(NullCell(), species=['X'], seed=seed)

    def do_trial_move(self, atoms):
        return atoms.copy(), 0, 'X'


def test_gcmc_rejects_copy_returning_move():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    ms = MoveSelector([1], [_CopyReturningMove()], seed=2)
    g = _gcmc(atoms, [Cell(atoms)], ms, mu={'H': 0.0})
    with pytest.raises(RuntimeError, match='in place'):
        g.do_gcmc_step()


# --------------------------------------------------------------------------
# MoveSelector: trial count decoupled from weight dtype (bug: float weights
# crashed range(); int weights silently set trials per step)
# --------------------------------------------------------------------------

def test_moveselector_float_weights_give_one_trial_per_step():
    ms = MoveSelector([0.5, 0.5], [DisplacementMove(species=['H'], seed=1),
                                   DisplacementMove(species=['H'], seed=2)], seed=3)
    assert ms.n_moves == 1
    assert isinstance(ms.n_moves, int)


def test_moveselector_integer_weights_keep_legacy_trial_count():
    ms = MoveSelector([2, 3], [DisplacementMove(species=['H'], seed=1),
                               DisplacementMove(species=['H'], seed=2)], seed=3)
    assert ms.n_moves == 5


def test_moveselector_explicit_trials_per_step_overrides_weights():
    ms = MoveSelector([2, 3], [DisplacementMove(species=['H'], seed=1),
                               DisplacementMove(species=['H'], seed=2)],
                      seed=3, n_moves=2)
    assert ms.n_moves == 2


def test_moveselector_legacy_typo_alias_warns():
    ms = MoveSelector([1], [DisplacementMove(species=['H'], seed=1)], seed=2)
    with pytest.warns(DeprecationWarning):
        ms.get_acceptance_ration()


# --------------------------------------------------------------------------
# set_state refreshes cell free volumes (bug: acceptance after an accepted
# replica swap used the previous configuration's volume)
# --------------------------------------------------------------------------

class _RecordingCell:
    def __init__(self):
        self.calls = 0

    def calculate_volume(self, atoms):
        self.calls += 1

    def get_volume(self):
        return 1000.0


def test_gcmc_set_state_recalculates_cell_volumes():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    cell = _RecordingCell()
    ms = MoveSelector([1], [DisplacementMove(species=['H'], seed=1)], seed=2)
    g = _gcmc(atoms, [cell], ms, mu={'H': 0.0})
    calls_before = cell.calls
    other = Atoms('H3', positions=[[1, 1, 1], [3, 1, 1], [5, 1, 1]],
                  cell=[10, 10, 10], pbc=True)
    g.set_state({'atoms': other, 'energy': -3.0, 'n_atoms': 3})
    assert cell.calls == calls_before + 1


# --------------------------------------------------------------------------
# CustomCell argument validation (bug: None defaults -> TypeError)
# --------------------------------------------------------------------------

def test_custom_cell_requires_height_and_bottom():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    with pytest.raises(ValueError, match='custom_height'):
        CustomCell(atoms)


# --------------------------------------------------------------------------
# ReplicaExchange ladder validation (bug: passing both built the GCMC twice
# and silently used the equal-T mu criterion on a T ladder)
# --------------------------------------------------------------------------

def test_replica_exchange_rejects_both_ladders():
    with pytest.raises(ValueError, match='not both'):
        ReplicaExchange(gcmc_factory=lambda **kw: None,
                        temperatures=[300.0, 400.0],
                        mus=[{'H': 0.0}, {'H': -0.1}])


# --------------------------------------------------------------------------
# CanonicalEnsemble hygiene (bugs: seeded/drew from the global random module;
# trajectory_write_interval accepted but ignored)
# --------------------------------------------------------------------------

def _canonical(tmp_path, **kwargs):
    from ase.cluster import Octahedron
    atoms = Octahedron('Au', 2)
    half = len(atoms) // 2
    atoms.symbols = ['Au'] * half + ['Pt'] * (len(atoms) - half)
    ms = MoveSelector([1.0], [PermutationMove(species=['Au', 'Pt'], seed=5)])
    defaults = dict(
        atoms=atoms, calculator=EMT(), optimizer=LBFGS, fmax=0.2,
        temperature=600.0, move_selector=ms, random_seed=5,
        traj_file=str(tmp_path / 't.xyz'), outfile=str(tmp_path / 'o.out'),
    )
    defaults.update(kwargs)
    return CanonicalEnsemble(**defaults)


def test_canonical_does_not_touch_global_random(tmp_path):
    random.seed(123456)
    state_before = random.getstate()
    mc = _canonical(tmp_path)
    assert random.getstate() == state_before
    mc.run(steps=2)
    assert random.getstate() == state_before


def _count_xyz_frames(path):
    n_frames = 0
    with open(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            try:
                n_atoms = int(header.strip())
            except ValueError:
                continue
            n_frames += 1
            for _ in range(n_atoms + 1):
                fh.readline()
    return n_frames


def test_canonical_honors_trajectory_write_interval(tmp_path):
    mc = _canonical(tmp_path, trajectory_write_interval=2)
    mc.run(steps=4)
    # Initial frame + one frame at each interval boundary (steps 2 and 4),
    # independent of how many trials were accepted.
    assert _count_xyz_frames(tmp_path / 't.xyz') == 3


# --------------------------------------------------------------------------
# Minimum-image handling (bug: min_insert distance check and CustomCell free
# volume ignored periodic images)
# --------------------------------------------------------------------------

class _FixedPointCell:
    """Deterministic cell: always proposes the same insertion point."""

    def __init__(self, point, species):
        self.point = np.asarray(point, dtype=float)
        self.species = species

    def get_random_point(self):
        return self.point.copy()

    def get_atoms_specie_inside_cell(self, atoms, species):
        return np.arange(len(atoms))

    def get_species(self):
        return self.species


def test_min_insert_respects_minimum_image():
    # Existing atom at x=0.5 in a 10 A periodic box; proposal at x=9.9 is
    # 0.6 A away through the boundary -- closer than min_insert=1.0, so the
    # move must fail rather than insert an overlapping atom.
    atoms = Atoms('H', positions=[[0.5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = _FixedPointCell([9.9, 5.0, 5.0], species=['H'])
    move = InsertionMove(cell, species=['H'], seed=1, min_insert=1.0)
    result = move.do_trial_move(atoms)
    assert result[0] is False
    assert len(atoms) == 1


def test_custom_cell_free_volume_counts_periodic_images():
    # One atom of radius 2 A centered on the x-boundary of a 10 A cube:
    # the excluded sphere is split across the boundary, so the free volume
    # is 1000 - (4/3)*pi*8 ~= 966.5, not 1000 minus half a sphere (~983).
    atoms = Atoms('H', positions=[[0.0, 5.0, 5.0]], cell=[10, 10, 10], pbc=True)
    cell = CustomCell(atoms, custom_height=10.0, bottom_z=0.0,
                      species_radii={'H': 2.0}, mc_sample_points=200_000, seed=7)
    cell.calculate_volume(atoms)
    expected = 1000.0 - (4.0 / 3.0) * math.pi * 8.0
    assert cell.get_volume() == pytest.approx(expected, abs=3.0)


# --------------------------------------------------------------------------
# Packaging / import hygiene
# --------------------------------------------------------------------------

def test_mcpy_exposes_version():
    import mcpy
    assert isinstance(mcpy.__version__, str) and mcpy.__version__


def test_ensembles_package_exports_grand_canonical():
    from mcpy.ensembles import GrandCanonicalEnsemble as GCE
    assert GCE is GrandCanonicalEnsemble


def test_import_does_not_mutate_matplotlib_rcparams():
    mpl = pytest.importorskip('matplotlib')
    del mpl
    code = (
        "import matplotlib\n"
        "dpi = matplotlib.rcParams['figure.dpi']\n"
        "import mcpy.moves\n"
        "import mcpy.utils\n"
        "assert matplotlib.rcParams['figure.dpi'] == dpi, "
        "matplotlib.rcParams['figure.dpi']\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)
