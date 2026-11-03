"""IsothermalIsobaricEnsemble and VolumeMove.

The volume move scales the box in place, so a rejected trial must restore
the lattice as well as the arrays. The acceptance is
exp(-beta (dE + P dV) + (N + 1) ln(V'/V)); for an ideal gas it samples
p(V) ~ V^N exp(-beta P V), whose mean (N + 1) / (beta P) is exact.
"""
import logging

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms

from mcpy.cell import Cell
from mcpy.ensembles import (BatchedReplicaExchange, CanonicalEnsemble,
                            GrandCanonicalEnsemble, IsothermalIsobaricEnsemble)
from mcpy.ensembles.replica_exchange import ReplicaExchange
from mcpy.moves import DisplacementMove, InsertionMove, MoveSelector, VolumeMove


class _ZeroCalc:
    """Ideal gas: every configuration has zero energy."""

    def get_potential_energy(self, atoms):
        return 0.0


class _LJCalc:
    def __init__(self):
        self.lj = LennardJones(sigma=1.0, epsilon=1.0, rc=2.5)

    def get_potential_energy(self, atoms):
        a = atoms.copy()
        a.calc = self.lj
        return a.get_potential_energy()


class _FakeRng:
    def __init__(self, u):
        self.u = u

    def get_uniform(self):
        return self.u


def _atoms(n=4, box=5.0, pbc=True):
    grid = np.array([(x, y, z) for x in range(2) for y in range(2) for z in range(2)])
    return Atoms('Ar' * n, positions=1.2 * grid[:n] + 0.5, cell=[box] * 3, pbc=pbc)


def _npt(atoms=None, moves=None, calc=None, pressure=1.0, temperature=1.0, seed=0):
    atoms = _atoms() if atoms is None else atoms
    moves = [VolumeMove(seed=seed + 1, max_delta_ln_volume=0.2)] if moves is None else moves
    return IsothermalIsobaricEnsemble(
        atoms=atoms, units_type='LJ', calculator=calc or _ZeroCalc(),
        pressure=pressure, temperature=temperature,
        move_selector=MoveSelector([1] * len(moves), moves, seed=seed + 2),
        random_seed=seed + 3, traj_file=None, outfile=None)


def test_volume_move_scales_isotropically_and_reports_jacobian():
    atoms = _atoms()
    frac_before = atoms.get_scaled_positions()
    v_old = atoms.get_volume()
    move = VolumeMove(seed=1, max_delta_ln_volume=0.3)
    assert move.do_trial_move(atoms) == (atoms, 0, 'X')
    v_new = atoms.get_volume()
    assert move.last_volumes == (pytest.approx(v_old), pytest.approx(v_new))
    assert abs(np.log(v_new / v_old)) <= 0.3
    np.testing.assert_allclose(atoms.get_scaled_positions(), frac_before, atol=1e-12)
    # Isotropic: the box stays cubic.
    np.testing.assert_allclose(atoms.cell.lengths(), [v_new ** (1 / 3)] * 3)
    assert move.last_log_jacobian == pytest.approx((len(atoms) + 1) * np.log(v_new / v_old))


@pytest.mark.parametrize('bad', [0.0, -0.1, np.inf, np.nan])
def test_volume_move_rejects_bad_step(bad):
    with pytest.raises(ValueError, match='max_delta_ln_volume'):
        VolumeMove(seed=1, max_delta_ln_volume=bad)


def test_rejected_volume_move_restores_lattice_and_positions():
    ens = _npt(calc=_LJCalc())
    cell_before = ens.atoms.cell.array.copy()
    positions_before = ens.atoms.positions.copy()
    e_before = ens.E_old
    ens._acceptance_condition = lambda *a, **k: False
    ens.do_gcmc_step()
    np.testing.assert_array_equal(ens.atoms.cell.array, cell_before)
    np.testing.assert_array_equal(ens.atoms.positions, positions_before)
    assert ens.E_old == e_before


def test_accepted_volume_move_keeps_new_box_and_energy():
    ens = _npt(calc=_LJCalc())
    v_before = ens.atoms.get_volume()
    ens._acceptance_condition = lambda *a, **k: True
    ens.do_gcmc_step()
    assert ens.atoms.get_volume() != pytest.approx(v_before)
    assert ens.E_old == pytest.approx(_LJCalc().get_potential_energy(ens.atoms))


def test_volume_acceptance_exponent():
    ens = _npt(pressure=2.0, temperature=0.5)
    move = ens.move_selector.move_list[0]
    ens.move_selector.to_use = 0
    n, v_old, v_new, d_e = 4, 100.0, 105.0, 0.3
    move.last_volumes = (v_old, v_new)
    move.last_log_jacobian = (n + 1) * np.log(v_new / v_old)
    p = np.exp(-2.0 * (d_e + 2.0 * (v_new - v_old)) + (n + 1) * np.log(v_new / v_old))
    ens.rng_acceptance = _FakeRng(p * (1 - 1e-9))
    assert ens._acceptance_condition(d_e, 0, 0.0, 'X', n)
    ens.rng_acceptance = _FakeRng(p * (1 + 1e-9))
    assert not ens._acceptance_condition(d_e, 0, 0.0, 'X', n)


def test_ideal_gas_mean_volume_is_exact():
    """<V> = (N + 1) kT / P. A Jacobian of N instead of N + 1 gives N kT / P,
    20 % lower here."""
    n, temperature, pressure = 4, 1.0, 1.0
    ens = _npt(atoms=_atoms(n), pressure=pressure, temperature=temperature,
               moves=[VolumeMove(seed=1, max_delta_ln_volume=1.0)])
    for _ in range(1_000):
        ens.do_gcmc_step()
    volumes = []
    for _ in range(20_000):
        ens.do_gcmc_step()
        volumes.append(ens.atoms.get_volume())
    assert np.mean(volumes) == pytest.approx((n + 1) * temperature / pressure, rel=0.05)


def test_displacement_moves_keep_the_box():
    ens = _npt(calc=_LJCalc(), moves=[VolumeMove(seed=4),
                                      DisplacementMove(species=['Ar'], seed=5)])
    ens.move_selector._pick_index = lambda: 1
    cell_before = ens.atoms.cell.array.copy()
    for _ in range(20):
        ens.do_gcmc_step()
    np.testing.assert_array_equal(ens.atoms.cell.array, cell_before)


def test_enthalpy_is_the_minimum_score():
    ens = _npt(pressure=2.0)
    assert ens._minimum_score(ens.atoms, 1.5) == pytest.approx(1.5 + 2.0 * ens.atoms.get_volume())


@pytest.mark.parametrize('case, match', [
    ('no_volume_move', 'needs a VolumeMove'),
    ('insertion', 'keeps N fixed'),
    ('slab', 'periodic in x, y and z'),
    ('fixed_atoms', 'constraints are not supported'),
    ('pressure', 'pressure must be finite'),
])
def test_invalid_setups_raise(case, match):
    atoms = _atoms(pbc=[True, True, False] if case == 'slab' else True)
    moves = [VolumeMove(seed=1)]
    pressure = np.nan if case == 'pressure' else 1.0
    if case == 'no_volume_move':
        moves = [DisplacementMove(species=['Ar'], seed=1)]
    elif case == 'insertion':
        moves.append(InsertionMove(Cell(atoms, seed=2), species=['Ar'], seed=3))
    elif case == 'fixed_atoms':
        atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match=match):
        _npt(atoms=atoms, moves=moves, pressure=pressure)


def test_outfile_has_volume_column(tmp_path):
    out = tmp_path / 'npt.out'
    ens = IsothermalIsobaricEnsemble(
        atoms=_atoms(), units_type='LJ', calculator=_ZeroCalc(), pressure=1.0,
        temperature=1.0, move_selector=MoveSelector([1], [VolumeMove(seed=1)], seed=2),
        random_seed=3, traj_file=None, outfile=str(out))
    ens.run(2)
    text = out.read_text()
    assert 'Isothermal-Isobaric' in text and 'Volume (sigma^3)' in text
    last = text.strip().splitlines()[-1].split()
    assert float(last[3]) == pytest.approx(ens.atoms.get_volume(), abs=1e-4)


def test_batched_swap_adds_pv_cross_term():
    """BatchedReplicaExchange scores configurations through _minimum_score,
    so NPT replicas swap with
    delta = (beta_i - beta_j)(E_X - E_Y) + (beta_i P_i - beta_j P_j)(V_X - V_Y)."""
    ri = _npt(atoms=_atoms(box=30.0 ** (1 / 3)), pressure=1.0, temperature=1.0, seed=0)
    rj = _npt(atoms=_atoms(box=3.0), pressure=3.0, temperature=2.0, seed=10)
    ri.E_old, rj.E_old = -3.0, -1.0
    delta = (1.0 - 0.5) * (-3.0 + 1.0) + (1.0 * 1.0 - 0.5 * 3.0) * (30.0 - 27.0)
    assert delta == pytest.approx(-2.5)
    re = object.__new__(BatchedReplicaExchange)
    re.logger = logging.getLogger('test')
    re.replicas = [ri, rj]
    re.rng = _FakeRng(np.exp(delta) * (1 - 1e-9))
    assert re._accept_swap(0, 1)
    re.rng = _FakeRng(np.exp(delta) * (1 + 1e-9))
    assert not re._accept_swap(0, 1)


def test_other_ensembles_refuse_a_volume_move():
    atoms = _atoms()
    with pytest.raises(ValueError, match='cannot score a VolumeMove'):
        GrandCanonicalEnsemble(
            atoms=atoms, cells=[], units_type='LJ', calculator=_ZeroCalc(), mu={},
            species=['Ar'], temperature=1.0,
            move_selector=MoveSelector([1], [VolumeMove(seed=1)], seed=2),
            random_seed=3, traj_file=None, outfile=None)
    with pytest.raises(ValueError, match='cannot score a VolumeMove'):
        CanonicalEnsemble(atoms=atoms, calculator=LennardJones(), optimizer=None,
                          move_selector=MoveSelector([1], [VolumeMove(seed=1)], seed=2),
                          random_seed=3, traj_file=None, outfile=None)


def _mpi_re():
    re = object.__new__(ReplicaExchange)
    re.logger = logging.getLogger('test')
    re.rank = 0
    return re


def test_npt_state_carries_pressure():
    assert _npt(pressure=2.5).get_state()['pressure'] == 2.5


def test_mpi_swap_adds_pv_cross_term():
    """Same configurations and rule as the batched test above:
    delta = (beta_2 - beta_1)(E_Y - E_X) + (beta_2 P_2 - beta_1 P_1)(V_Y - V_X)."""
    ri = _npt(atoms=_atoms(box=30.0 ** (1 / 3)), pressure=1.0, temperature=1.0, seed=0)
    rj = _npt(atoms=_atoms(box=3.0), pressure=3.0, temperature=2.0, seed=10)
    ri.E_old, rj.E_old = -3.0, -1.0
    p = _mpi_re()._exchange_prob(ri.get_state(), rj.get_state())
    assert p == pytest.approx(np.exp(-2.5))
    # Without the pressures the swap would be scored on energy alone.
    s1, s2 = ri.get_state(), rj.get_state()
    del s1['pressure'], s2['pressure']
    assert _mpi_re()._exchange_prob(s1, s2) == pytest.approx(np.exp(-1.0))


def test_mpi_and_batched_swaps_agree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t1, t2, p1, p2 = rng.uniform(0.5, 3.0, 4)
        ri = _npt(atoms=_atoms(box=rng.uniform(2.5, 4.0)), pressure=p1, temperature=t1, seed=0)
        rj = _npt(atoms=_atoms(box=rng.uniform(2.5, 4.0)), pressure=p2, temperature=t2, seed=10)
        ri.E_old, rj.E_old = rng.uniform(-5, 0, 2)
        re = object.__new__(BatchedReplicaExchange)
        re.logger = logging.getLogger('test')
        re.replicas = [ri, rj]
        p_mpi = _mpi_re()._exchange_prob(ri.get_state(), rj.get_state())
        re.rng = _FakeRng(p_mpi * (1 - 1e-9))
        assert p_mpi == 1.0 or re._accept_swap(0, 1)
        re.rng = _FakeRng(p_mpi * (1 + 1e-9))
        assert p_mpi == 1.0 or not re._accept_swap(0, 1)
