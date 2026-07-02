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
import types

import numpy as np
import pytest
from ase import Atoms

from mcpy.ensembles.canonical_ensemble import CanonicalEnsemble
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.utils.set_unit_constant import SetUnits


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
    e = CanonicalEnsemble.__new__(CanonicalEnsemble)
    e._temperature = temperature
    return e


def test_nvt_downhill_always_accepted():
    e = _nvt(300)
    assert e._acceptance_condition(-1.0)
    assert e._acceptance_condition(0.0)


def test_nvt_zero_temperature_rejects_uphill():
    e = _nvt(0.0)
    assert not (e._acceptance_condition(0.5))


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
    e.units = SetUnits("metal", temperature, ["H"])
    e._mu = {"H": mu}
    e.rng_acceptance = FixedRNG(rng_value)
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    e.move_selector = types.SimpleNamespace(get_name=lambda: "dummy")
    return e


def test_gcmc_displacement_reduces_to_metropolis():
    # delta_particles == 0 -> plain Metropolis on units.beta
    e = _gcmc(300.0, mu=0.0, rng_value=0.5)
    assert e._acceptance_condition(-0.2, 0, volume=1000.0, species="H")

    diff = 0.1
    p = np.exp(-diff * e.units.beta)
    e_lo = _gcmc(300.0, mu=0.0, rng_value=p * 0.99)
    assert e_lo._acceptance_condition(diff, 0, volume=1000.0, species="H")
    e_hi = _gcmc(300.0, mu=0.0, rng_value=p * 1.01)
    assert not (e_hi._acceptance_condition(diff, 0, volume=1000.0, species="H"))


def test_gcmc_insertion_favoured_by_high_mu():
    # Large positive mu drives the acceptance probability above 1 -> always accept.
    e = _gcmc(300.0, mu=0.5, rng_value=0.999)
    assert e._acceptance_condition(0.0, 1, volume=1000.0, species="H",
                                   n_atoms=0)
    # Strongly negative mu suppresses insertion -> reject against a mid draw.
    e_rej = _gcmc(300.0, mu=-0.5, rng_value=0.5)
    assert not e_rej._acceptance_condition(0.0, 1, volume=1000.0, species="H",
                                           n_atoms=0)


def test_gcmc_insertion_probability_formula():
    e = _gcmc(300.0, mu=-0.3, rng_value=0.0)
    volume, n, diff = 1000.0, 4, 0.05
    expected = (e.units.de_broglie_insertion(volume, n, "H")
                * np.exp(-e.units.beta * (diff - e._mu["H"])))
    # Choose a draw straddling the expected probability.
    below = _gcmc(300.0, mu=-0.3, rng_value=expected * 0.99)
    above = _gcmc(300.0, mu=-0.3, rng_value=min(expected * 1.01, 1.0))
    assert below._acceptance_condition(diff, 1, volume, "H", n_atoms=n)
    if expected < 1.0:
        assert not (above._acceptance_condition(diff, 1, volume, "H", n_atoms=n))


def test_gcmc_deletion_uses_opposite_mu_sign():
    # Deletion exp term carries +mu, the opposite sign from insertion.
    e = _gcmc(300.0, mu=-0.3, rng_value=0.0)
    volume, n, diff = 1000.0, 6, 0.05
    expected = (e.units.de_broglie_deletion(volume, n, "H")
                * np.exp(-e.units.beta * (diff + e._mu["H"])))
    below = _gcmc(300.0, mu=-0.3, rng_value=expected * 0.99)
    assert below._acceptance_condition(diff, -1, volume, "H", n_atoms=n)


def test_gcmc_rejects_unexpected_delta_particles():
    e = _gcmc(300.0, mu=0.0, rng_value=0.5)
    with pytest.raises(ValueError):
        e._acceptance_condition(0.0, 2, volume=1000.0, species="H")


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
    e.units = SetUnits("metal", temperature, [species])
    e._mu = {species: mu}
    e.rng_acceptance = FixedRNG(rng_value)
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    e.move_selector = types.SimpleNamespace(get_name=lambda: "dummy")
    return e


def test_total_vs_perspecies_count_shifts_effective_mu():
    """For identical (V, dE, mu, T) the insertion probability scales by
    (N_total+1)/(N_species+1). Since N_total (all atoms) >> N_species (O in the
    cell), the per-species choice would insert far more readily -- an effective
    chemical potential shift of +kT*ln((N_total+1)/(N_species+1)). mcpy keeps
    the total count, so this shift is exactly what would change if we switched."""
    units = SetUnits("metal", 500.0, ["O"])
    volume = 1500.0
    n_total, n_species = 1000, 20  # total atoms vs O atoms inside the cell
    p_total = units.de_broglie_insertion(volume, n_total, "O")
    p_species = units.de_broglie_insertion(volume, n_species, "O")
    assert p_species > p_total
    np.testing.assert_allclose(p_species / p_total, (n_total + 1) / (n_species + 1))
    dmu = np.log((n_total + 1) / (n_species + 1)) / units.beta  # eV
    assert dmu > 0


def test_perspecies_count_inserts_where_total_rejects():
    """Same trial and same RNG draw: the per-species count would accept an
    insertion that the total-atom count (mcpy's convention) rejects."""
    volume, diff, mu, T = 1500.0, 0.0, -0.3, 500.0
    n_total, n_species = 1000, 20
    units = SetUnits("metal", T, ["O"])
    exp_term = np.exp(-units.beta * (diff - mu))
    p_total = units.de_broglie_insertion(volume, n_total, "O") * exp_term
    p_species = units.de_broglie_insertion(volume, n_species, "O") * exp_term
    assert p_total < 1.0  # total-count path is probabilistic, not auto-accept
    draw = 0.5 * (p_total + min(p_species, 1.0))  # between the two -> they disagree
    e_species = _gcmc_sp(T, mu, rng_value=draw, species="O")
    e_total = _gcmc_sp(T, mu, rng_value=draw, species="O")
    assert e_species._acceptance_condition(diff, 1, volume, "O", n_atoms=n_species)
    assert not e_total._acceptance_condition(diff, 1, volume, "O", n_atoms=n_total)


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
    return types.SimpleNamespace(
        units=types.SimpleNamespace(beta=1.0 / (8.617333e-5 * temperature)),
        E_old=energy,
    )


def _bre(replicas, rng_value, n_replicas=None):
    e = BatchedReplicaExchange.__new__(BatchedReplicaExchange)
    e.replicas = replicas
    e.rng = FixedRNG(rng_value)
    e.n_replicas = n_replicas if n_replicas is not None else len(replicas)
    e.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    return e


def test_re_swap_accepted_when_delta_nonnegative():
    # beta_j > beta_i and E_j > E_i -> delta > 0 -> p clamps to 1 -> always accept.
    e = _bre([_replica(1200.0, -10.0), _replica(300.0, -9.0)], rng_value=0.999)
    assert e._accept_swap(0, 1)


def test_re_swap_probabilistic_when_delta_negative():
    # beta_j > beta_i but E_j < E_i -> delta < 0 -> p < 1.
    ri, rj = _replica(1200.0, -9.0), _replica(300.0, -10.0)
    beta_i, beta_j = ri.units.beta, rj.units.beta
    p = np.exp((beta_j - beta_i) * (rj.E_old - ri.E_old))
    assert p < 1.0
    e_acc = _bre([ri, rj], rng_value=p * 0.99)
    assert e_acc._accept_swap(0, 1)
    e_rej = _bre([ri, rj], rng_value=p * 1.01)
    assert not (e_rej._accept_swap(0, 1))


def test_re_exchange_pairs_alternate_by_offset():
    e = _bre([_replica(300.0, 0.0)] * 4, rng_value=0.0, n_replicas=4)
    assert e._exchange_pairs(0) == [(0, 1), (2, 3)]
    assert e._exchange_pairs(1) == [(1, 2)]
