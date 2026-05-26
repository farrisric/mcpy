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


def test_nvt_uphill_matches_metropolis_boundary(monkeypatch):
    import mcpy.ensembles.canonical_ensemble as ce
    from ase.units import kB

    diff, T = 0.1, 300.0
    p = np.exp(-diff / (kB * T))

    e = _nvt(T)
    # Accept when the draw is just below p, reject when just above.
    monkeypatch.setattr(ce.random, "random", lambda: p * 0.99)
    assert e._acceptance_condition(diff)
    monkeypatch.setattr(ce.random, "random", lambda: p * 1.01)
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
                                   n_atoms_species=0)
    # Strongly negative mu suppresses insertion -> reject against a mid draw.
    e_rej = _gcmc(300.0, mu=-0.5, rng_value=0.5)
    assert not e_rej._acceptance_condition(0.0, 1, volume=1000.0, species="H",
                                           n_atoms_species=0)


def test_gcmc_insertion_probability_formula():
    e = _gcmc(300.0, mu=-0.3, rng_value=0.0)
    volume, n, diff = 1000.0, 4, 0.05
    expected = (e.units.de_broglie_insertion(volume, n, "H")
                * np.exp(-e.units.beta * (diff - e._mu["H"])))
    # Choose a draw straddling the expected probability.
    below = _gcmc(300.0, mu=-0.3, rng_value=expected * 0.99)
    above = _gcmc(300.0, mu=-0.3, rng_value=min(expected * 1.01, 1.0))
    assert below._acceptance_condition(diff, 1, volume, "H", n_atoms_species=n)
    if expected < 1.0:
        assert not (above._acceptance_condition(diff, 1, volume, "H", n_atoms_species=n))


def test_gcmc_deletion_uses_opposite_mu_sign():
    # Deletion exp term carries +mu, the opposite sign from insertion.
    e = _gcmc(300.0, mu=-0.3, rng_value=0.0)
    volume, n, diff = 1000.0, 6, 0.05
    expected = (e.units.de_broglie_deletion(volume, n, "H")
                * np.exp(-e.units.beta * (diff + e._mu["H"])))
    below = _gcmc(300.0, mu=-0.3, rng_value=expected * 0.99)
    assert below._acceptance_condition(diff, -1, volume, "H", n_atoms_species=n)


def test_gcmc_rejects_unexpected_delta_particles():
    e = _gcmc(300.0, mu=0.0, rng_value=0.5)
    with pytest.raises(ValueError):
        e._acceptance_condition(0.0, 2, volume=1000.0, species="H")


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
