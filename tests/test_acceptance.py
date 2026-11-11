"""
Acceptance-criterion tests for the Monte Carlo ensembles.

These exercise the scientific core of the sampler -- the Metropolis and
de-Broglie acceptance probabilities, and the replica-exchange swap rule --
without instantiating a calculator. The acceptance methods are pure functions
of energy/particle differences and the thermodynamic units, so the ensemble
objects are built with ``__new__`` and the few attributes each method reads are
attached directly. This keeps the suite free of ``torch`` / ``mace-torch`` so
it runs in the lightweight CI matrix.

Run with: python -m pytest tests/test_acceptance.py -v
"""
import math
import types
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from mcpy.cell import Cell, CustomCell
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.ensembles.canonical_ensemble import CanonicalEnsemble
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import (DeletionMove, DisplacementMove, InsertionMove, MoleculeDeletionMove,
                        MoleculeInsertionMove, MoveSelector)
from mcpy.utils.set_unit_constant import SetUnits

from helpers import _mic_dists, _o2, _ZeroCalc


class FixedRNG:
    """Stand-in for RandomNumberGenerator returning a constant draw."""

    def __init__(self, value: float) -> None:
        self.value = value

    def get_uniform(self, a: float = 0.0, b: float = 1.0) -> float:
        return self.value


# --------------------------------------------------------------------------
# Canonical (NVT) Metropolis
# --------------------------------------------------------------------------

def _nvt(temperature):
    # The real constructor (it derives beta, inf at T = 0); acceptance needs
    # nothing else, so the remaining collaborators are left empty.
    return CanonicalEnsemble(atoms=None, calculator=None, optimizer=None,
                             move_selector=None, temperature=temperature,
                             random_seed=0, traj_file=None, outfile=None)


def test_nvt_uphill_matches_metropolis_boundary():
    from ase.units import kB

    diff, T = 0.1, 300.0
    p = np.exp(-diff / (kB * T))

    e = _nvt(T)
    # Accept when the draw is just below p, reject when just above. The draw
    # comes from the ensemble-owned generator, never the global random module.
    e._rng_acceptance = FixedRNG(p * 0.99)
    assert e._acceptance_condition(diff)
    e._rng_acceptance = FixedRNG(p * 1.01)
    assert not (e._acceptance_condition(diff))


# --------------------------------------------------------------------------
# Grand canonical (muVT) de-Broglie acceptance
# --------------------------------------------------------------------------

def _gcmc(temperature, mu, rng_value):
    e = GrandCanonicalEnsemble.__new__(GrandCanonicalEnsemble)
    e.units = SetUnits('metal', temperature, ['H'])
    e._mu = {'H': mu}
    e.rng_acceptance = FixedRNG(rng_value)
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    e.move_selector = types.SimpleNamespace(get_name=lambda: 'dummy')
    return e


def test_gcmc_displacement_reduces_to_metropolis():
    # delta_particles == 0 -> plain Metropolis on units.beta
    e = _gcmc(300.0, mu=0.0, rng_value=0.5)
    assert e._acceptance_condition(-0.2, 0, volume=1000.0, species='H')

    diff = 0.1
    p = np.exp(-diff * e.units.beta)
    e_lo = _gcmc(300.0, mu=0.0, rng_value=p * 0.99)
    assert e_lo._acceptance_condition(diff, 0, volume=1000.0, species='H')
    e_hi = _gcmc(300.0, mu=0.0, rng_value=p * 1.01)
    assert not (e_hi._acceptance_condition(diff, 0, volume=1000.0, species='H'))


def test_gcmc_rejects_unexpected_delta_particles():
    e = _gcmc(300.0, mu=0.0, rng_value=0.5)
    with pytest.raises(ValueError):
        e._acceptance_condition(0.0, 2, volume=1000.0, species='H')


# --------------------------------------------------------------------------
# Total vs per-species de-Broglie count (documents the convention choice)
#
# The de Broglie combinatorial factor needs a particle count N. mcpy uses the
# *total* atom count (len(atoms): substrate + every species) -- the original
# convention, kept for consistency with the group's published runs. The
# physically standard / LAMMPS choice is the per-species exchangeable count in
# the insertion region. Both go through the same _acceptance_condition; only
# the n_atoms argument differs. These tests quantify the gap so the decision
# stays recorded in code. See docs/gcmc_acceptance_convention.rst.
# --------------------------------------------------------------------------

def _gcmc_sp(temperature, mu, rng_value, species):
    e = GrandCanonicalEnsemble.__new__(GrandCanonicalEnsemble)
    e.units = SetUnits('metal', temperature, [species])
    e._mu = {species: mu}
    e.rng_acceptance = FixedRNG(rng_value)
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    e.move_selector = types.SimpleNamespace(get_name=lambda: 'dummy')
    return e


def test_total_vs_perspecies_count_shifts_effective_mu():
    """For identical (V, dE, mu, T) the insertion probability scales by
    (N_total+1)/(N_species+1). Since N_total (all atoms) >> N_species (O in the
    cell), the per-species choice would insert far more readily -- an effective
    chemical potential shift of +kT*ln((N_total+1)/(N_species+1)). mcpy keeps
    the total count, so this shift is exactly what would change if we switched."""
    units = SetUnits('metal', 500.0, ['O'])
    volume = 1500.0
    n_total, n_species = 1000, 20  # total atoms vs O atoms inside the cell
    p_total = units.de_broglie_insertion(volume, n_total, 'O')
    p_species = units.de_broglie_insertion(volume, n_species, 'O')
    assert p_species > p_total
    np.testing.assert_allclose(p_species / p_total, (n_total + 1) / (n_species + 1))
    dmu = np.log((n_total + 1) / (n_species + 1)) / units.beta  # eV
    assert dmu > 0


def test_perspecies_count_inserts_where_total_rejects():
    """Same trial and same RNG draw: the per-species count would accept an
    insertion that the total-atom count (mcpy's convention) rejects."""
    volume, diff, mu, T = 1500.0, 0.0, -0.3, 500.0
    n_total, n_species = 1000, 20
    units = SetUnits('metal', T, ['O'])
    exp_term = np.exp(-units.beta * (diff - mu))
    p_total = units.de_broglie_insertion(volume, n_total, 'O') * exp_term
    p_species = units.de_broglie_insertion(volume, n_species, 'O') * exp_term
    assert p_total < 1.0  # total-count path is probabilistic, not auto-accept
    draw = 0.5 * (p_total + min(p_species, 1.0))  # between the two -> they disagree
    e_species = _gcmc_sp(T, mu, rng_value=draw, species='O')
    e_total = _gcmc_sp(T, mu, rng_value=draw, species='O')
    assert e_species._acceptance_condition(diff, 1, volume, 'O', n_atoms=n_species)
    assert not e_total._acceptance_condition(diff, 1, volume, 'O', n_atoms=n_total)


def test_do_gcmc_step_feeds_total_atom_count():
    """Restored-convention guard: do_gcmc_step passes the TOTAL atom count to
    the de Broglie factor, not the per-species in-cell count. If someone
    re-introduces the per-species count, this fails -- flip it deliberately and
    update docs/gcmc_acceptance_convention.rst, do not change it silently."""
    e = GrandCanonicalEnsemble.__new__(GrandCanonicalEnsemble)
    atoms = Atoms('Ag5', positions=[[i, 0.0, 0.0] for i in range(5)])
    e._atoms = atoms
    e.E_old = 0.0
    e.n_atoms = len(atoms)          # 5 total atoms before the move
    e.compute_energy = lambda a: 0.0

    def _insert(a):                 # mutates in place, returns (atoms, +1, 'O')
        a += Atoms('O', positions=[[0.0, 0.0, 9.0]])
        return a, 1, 'O'

    e.move_selector = types.SimpleNamespace(
        n_moves=1, do_trial_move=_insert, get_volume=lambda: 100.0,
        get_exchange_count=lambda: None,
    )

    captured = {}

    def _capture(delta_E, delta_particles, volume, species, n_atoms=None):
        captured['n_atoms'] = n_atoms
        return False                # reject -> only the rollback path runs

    e._acceptance_condition = _capture
    e.do_gcmc_step()

    # Total count (5), not the per-species in-cell O count (0 before insertion).
    assert captured['n_atoms'] == 5


# --------------------------------------------------------------------------
# Batched replica-exchange swap criterion: P = exp((beta_j - beta_i)(E_j - E_i))
# --------------------------------------------------------------------------

def _replica(temperature, energy):
    # No mu configured: _minimum_score (which _grand_potential delegates to)
    # is the bare energy, matching GrandCanonicalEnsemble._minimum_score with
    # an empty self._mu. `atoms` is unused by the lambda but still read by
    # _grand_potential before the call, so it must be present.
    return types.SimpleNamespace(
        units=types.SimpleNamespace(beta=1.0 / (8.617333e-5 * temperature)),
        E_old=energy,
        atoms=None,
        _minimum_score=lambda atoms, energy: energy,
    )


def _bre(replicas, rng_value, n_replicas=None):
    e = BatchedReplicaExchange.__new__(BatchedReplicaExchange)
    e.replicas = replicas
    e.rng = FixedRNG(rng_value)
    e.n_replicas = n_replicas if n_replicas is not None else len(replicas)
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    return e


def test_re_exchange_pairs_alternate_by_offset():
    e = _bre([_replica(300.0, 0.0)] * 4, rng_value=0.0, n_replicas=4)
    assert e._exchange_pairs(0) == [(0, 1), (2, 3)]
    assert e._exchange_pairs(1) == [(1, 2)]


# The deletion acceptance uses the free volume of the post-deletion state

def test_p1_deletion_volume_is_post_deletion_free_volume():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = CustomCell(atoms, custom_height=10.0, bottom_z=0.0,
                      species_radii={'H': 2.0}, mc_sample_points=20_000, seed=1)
    cell.calculate_volume(atoms)
    assert cell.get_volume() < 990.0  # the H sphere is excluded
    selector = MoveSelector([1], [DeletionMove(cell, species=['H'], seed=2)])
    result, delta, _ = selector.do_trial_move(atoms)
    assert result is atoms and delta == -1 and len(atoms) == 0
    # The reverse insertion starts from the empty box: V = 1000 exactly.
    assert selector.get_volume() == pytest.approx(1000.0)


def test_p1_molecule_deletion_volume_is_post_deletion_free_volume():
    atoms = Atoms('O2', positions=[[5, 5, 5], [5, 5, 6.2]],
                  cell=[10, 10, 10], pbc=True)
    atoms.new_array('molecule_id', np.array([0, 0]))
    cell = CustomCell(atoms, custom_height=10.0, bottom_z=0.0,
                      species_radii={'O': 1.5}, mc_sample_points=20_000, seed=1)
    cell.calculate_volume(atoms)
    assert cell.get_volume() < 990.0
    selector = MoveSelector([1], [MoleculeDeletionMove(cell, _o2(), 'O2', seed=2)])
    result, delta, _ = selector.do_trial_move(atoms)
    assert result is atoms and len(atoms) == 0
    assert selector.get_volume() == pytest.approx(1000.0)


def test_p1_insertion_volume_is_pre_insertion_free_volume():
    atoms = Atoms('H', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'H': 2.0}, mc_sample_points=50_000, seed=1)
    cell.calculate_volume(atoms)
    v_before = cell.get_volume()
    expected = 1000.0 - 4.0 / 3.0 * math.pi * 8.0
    assert v_before == pytest.approx(expected, rel=0.02)
    selector = MoveSelector([1], [InsertionMove(cell, species=['H'], seed=2)])
    selector.do_trial_move(atoms)
    assert len(atoms) == 2
    assert selector.get_volume() == v_before


def test_min_insert_is_a_seed_clearance_adr_0004():
    """min_insert seeds insertions anywhere outside its own clearance, inside
    the species_radii spheres too, while the acceptance volume stays the
    species_radii free volume."""
    atoms = Atoms('Cu', positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
    cell = Cell(atoms, species_radii={'Cu': 2.0, 'H': 0.5},
                mc_sample_points=50_000, seed=1)
    cell.calculate_volume(atoms)
    v_free = cell.get_volume()
    assert v_free == pytest.approx(1000.0 - 4.0 / 3.0 * math.pi * 8.0, rel=0.02)
    # Only mcpy's own warnings matter: NumPy 2.5 warns from inside ASE.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        move = InsertionMove(cell, species=['H'], seed=2, min_insert=0.5)
        MoleculeInsertionMove(cell, _o2(), 'O2', seed=1, min_insert=0.5)
    assert not [w for w in caught if 'min_insert' in str(w.message)]
    selector = MoveSelector([1], [move], seed=3)
    dists = []
    for _ in range(3000):
        selector.do_trial_move(atoms)
        dists.append(atoms.get_distance(0, 1, mic=True))
        del atoms[1]
    dists = np.array(dists)
    assert dists.min() >= 0.5
    assert np.any(dists < 2.0)
    assert selector.get_volume() == v_free


def test_p1_two_state_gcmc_obeys_detailed_balance():
    """Ideal H (no energy) next to a fixed Cu obstacle, capped at one H.

    The stationary ratio P(1)/P(0) is V0 * exp(beta*mu) / ((N_tot+1) Lambda^3)
    with V0 the free volume of the Cu-only state (LJ units, Lambda = 1,
    N_tot = 1). mu is chosen so the ratio is 3, so P(1) = 0.75. Insertion is
    always accepted here, so P(1)/P(0) is set by the deletion acceptance:
    a deletion volume taken from the N+1 state (H sphere still excluded)
    gives P(1) ~ 0.69, and proposals that ignore the obstacle put H inside
    the Cu sphere.
    """
    r_cu, r_h, box = 3.0, 4.0, 10.0
    atoms = Atoms('Cu', positions=[[0.5, 5.0, 5.0]], cell=[box] * 3, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    v0 = box ** 3 - 4.0 / 3.0 * math.pi * r_cu ** 3
    mu = math.log(3.0 * 2.0 / v0)
    cell = Cell(atoms, species_radii={'Cu': r_cu, 'H': r_h},
                mc_sample_points=4000, seed=11)
    selector = MoveSelector(
        [1, 1], [DeletionMove(cell, species=['H'], seed=12),
                 InsertionMove(cell, species=['H'], seed=13, max_atoms=1)],
        seed=14, n_moves=1)
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ', calculator=_ZeroCalc(),
        mu={'H': mu}, species=['H'], temperature=1.0,
        move_selector=selector, random_seed=15, traj_file=None, outfile=None)

    n_trials, occupied = 12_000, 0
    for _ in range(n_trials):
        gcmc.do_gcmc_step()
        if len(gcmc.atoms) == 2:
            occupied += 1
            assert _mic_dists(gcmc.atoms[:1], gcmc.atoms.positions[1])[0] >= r_cu
    assert occupied / n_trials == pytest.approx(0.75, abs=0.025)


class _HardCore:
    """Zero energy, or 1e6 when two atoms are closer than ``r`` (mic)."""

    def __init__(self, r):
        self.r = r

    def get_potential_energy(self, atoms):
        if len(atoms) < 2:
            return 0.0
        d = atoms.get_all_distances(mic=True)[np.triu_indices(len(atoms), 1)]
        return 1e6 if np.any(d < self.r) else 0.0


def _moves_gcmc(atoms, cells, moves, weights=None, mu=None, species=('H',), calc=None,
                units='LJ', T=1.0, n_moves=1, seed=0, **kw):
    ms = MoveSelector(weights or [1] * len(moves), moves, seed=seed + 100,
                      n_moves=n_moves)
    files = dict(traj_file=None, outfile=None)
    files.update(kw)
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=cells, units_type=units, calculator=calc or _ZeroCalc(),
        mu=mu if mu is not None else {s: -1.0 for s in species},
        species=list(species), temperature=T, move_selector=ms,
        random_seed=seed + 200, **files)


# Cell refresh, three-state ideal gas

@pytest.mark.parametrize('listed', [True, False])
def test_three_state_hard_core_gas(listed):
    """Hard-core H (exclusion radius 4) in a periodic 10^3 box, N <= 2.
    Exact: P(N) ~ z^N Q_N / N!, Q_0 = 1, Q_1 = V, Q_2 = V (V - V_h).
    Breaks when an insertion reads a stale volume: the cell is missing
    from cells=, or it is not refreshed after an accepted move."""
    L, r = 10.0, 4.0
    atoms = Atoms(cell=[L] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'H': r}, mc_sample_points=5000, seed=1)
    z = 1e-3
    g = _moves_gcmc(atoms, [cell] if listed else [],
                    [InsertionMove(cell, ['H'], 2, max_atoms=2), DeletionMove(cell, ['H'], 3),
                     DisplacementMove(['H'], 4, max_displacement=3.0)],
                    mu={'H': math.log(z)}, calc=_HardCore(r))
    hist = np.zeros(3)
    for _ in range(8000):
        g.do_gcmc_step()
        hist[len(g.atoms)] += 1
    v, v_h = L ** 3, 4.0 / 3.0 * math.pi * r ** 3
    exact = np.array([1.0, z * v, z * z * v * (v - v_h) / 2.0])
    exact /= exact.sum()
    np.testing.assert_allclose(hist / hist.sum(), exact, atol=0.025)
