"""Swap-acceptance correctness for BatchedReplicaExchange.

Grand-canonical replicas at different temperatures must be compared through the
grand potential Phi = E - sum_s mu_s N_s, not bare energy. These tests pin that
behaviour without needing torch/mpi4py by driving _accept_swap / _grand_potential
with lightweight replica stubs.
"""
import logging

import numpy as np
import pytest

from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange


class _FakeUnits:
    def __init__(self, beta):
        self.beta = beta
        self.molecules = {}


class _FakeAtoms:
    def __init__(self, symbols):
        self._symbols = symbols

    def get_chemical_symbols(self):
        return self._symbols


class _FakeReplica:
    def __init__(self, beta, energy, mu, symbols):
        self.units = _FakeUnits(beta)
        self.E_old = energy
        self._mu = mu
        self.atoms = _FakeAtoms(symbols)

    def _minimum_score(self, atoms, energy):
        """Mirrors GrandCanonicalEnsemble._minimum_score for atomic species
        (these fakes carry no molecular species) so _grand_potential's
        delegation to it can be exercised without a real ensemble."""
        score = energy
        symbols = atoms.get_chemical_symbols()
        for specie, mu in (self._mu or {}).items():
            score -= mu * symbols.count(specie)
        return score


class _FakeRng:
    def __init__(self, u):
        self.u = u

    def get_uniform(self):
        return self.u


def _bare_re():
    re = object.__new__(BatchedReplicaExchange)
    re.logger = logging.getLogger('test')
    return re


def test_grand_potential_subtracts_mu_times_count():
    re = _bare_re()
    r = _FakeReplica(beta=1.0, energy=10.0,
                     mu={'Ag': 1.0, 'O': 2.0},
                     symbols=['Ag', 'Ag', 'Ag', 'O', 'O'])
    # Phi = 10 - (1.0*3 + 2.0*2) = 3.0
    assert re._grand_potential(r) == 3.0


def test_grand_potential_without_mu_is_bare_energy():
    re = _bare_re()
    r = _FakeReplica(beta=1.0, energy=10.0, mu=None, symbols=['Ag', 'Ag'])
    assert re._grand_potential(r) == 10.0


def test_accept_swap_uses_grand_potential_not_bare_energy():
    """Chosen so the grand-potential swap accepts while a bare-energy swap
    would reject at the same random draw — proving Phi drives the decision."""
    re = _bare_re()
    ri = _FakeReplica(beta=1.0, energy=10.0, mu={'Ag': 1.0}, symbols=['Ag'] * 5)
    rj = _FakeReplica(beta=0.5, energy=20.0, mu={'Ag': 1.0}, symbols=['Ag'] * 18)
    re.replicas = [ri, rj]
    re.rng = _FakeRng(0.5)
    # Phi_i = 10 - 5 = 5, Phi_j = 20 - 18 = 2
    # delta = (0.5 - 1.0)(2 - 5) = 1.5  -> p = 1 -> accept at u=0.5
    # bare-E delta = (0.5 - 1.0)(20 - 10) = -5 -> p ~ 0.0067 -> would reject
    assert re._accept_swap(0, 1) is True


# --------------------------------------------------------------------------
# mu-ladder swaps (bug: the temperature-only criterion collapses to delta=0
# when both replicas share a temperature, accepting every swap unconditionally)
# --------------------------------------------------------------------------

_BETA_300K = 1.0 / (8.617333e-5 * 300.0)


def _mu_ladder(u):
    """Two same-temperature replicas differing only in mu and N."""
    re = _bare_re()
    ri = _FakeReplica(beta=_BETA_300K, energy=-100.0, mu={'O': -5.0},
                      symbols=['O'] * 10)
    rj = _FakeReplica(beta=_BETA_300K, energy=-100.0, mu={'O': -3.0},
                      symbols=['O'] * 60)
    re.replicas = [ri, rj]
    re.rng = _FakeRng(u)
    return re


def test_mu_ladder_swap_is_not_always_accepted():
    # beta_i == beta_j, so (beta_j - beta_i)(Phi_j - Phi_i) is identically 0
    # and the old rule returned p = 1 for every draw. The cross terms give
    # exp(beta (mu_i - mu_j)(N_j - N_i)) = exp(-3868) here: always reject.
    assert not _mu_ladder(u=1e-12)._accept_swap(0, 1)


def _partial_mu_ladder():
    """mu-ladder tuned to an acceptance well inside (0, 1), so that both the
    'just below p' and the 'just above p' draw are legal uniforms -- a p
    pinned at 1 would let the buggy always-accept rule pass either way.
    Energies differ too: they must cancel out of a same-temperature swap."""
    beta, mu_i, n_i, n_j = _BETA_300K, -5.0, 10, 12
    mu_j = mu_i + 0.5 / (beta * (n_j - n_i))  # -> delta = -0.5
    re = _bare_re()
    re.replicas = [
        _FakeReplica(beta=beta, energy=-100.0, mu={'O': mu_i},
                     symbols=['O'] * n_i),
        _FakeReplica(beta=beta, energy=-42.0, mu={'O': mu_j},
                     symbols=['O'] * n_j),
    ]
    p = float(np.exp(beta * (mu_i - mu_j) * (n_j - n_i)))
    assert p == pytest.approx(np.exp(-0.5))
    return re, p


def test_mu_ladder_swap_matches_analytic_probability():
    re, p = _partial_mu_ladder()
    re.rng = _FakeRng(p * 0.99)
    assert re._accept_swap(0, 1)
    re.rng = _FakeRng(p * 1.01)
    assert not re._accept_swap(0, 1)


def test_mu_ladder_swap_ignores_the_energy_difference():
    """At one temperature the two configs' energies cancel exactly, so the
    swap probability must not move when either replica's energy changes."""
    re, p = _partial_mu_ladder()
    re.rng = _FakeRng(p * 0.99)
    re.replicas[0].E_old += 37.0
    re.replicas[1].E_old -= 12.5
    assert re._accept_swap(0, 1)
    re.rng = _FakeRng(p * 1.01)
    assert not re._accept_swap(0, 1)


def test_mu_ladder_swap_toward_favoured_replica_always_accepted():
    # Reverse the population imbalance: the same expression is now positive,
    # so the swap is downhill in the joint weight and must always be taken.
    re = _bare_re()
    re.replicas = [
        _FakeReplica(beta=_BETA_300K, energy=-100.0, mu={'O': -3.0},
                     symbols=['O'] * 10),
        _FakeReplica(beta=_BETA_300K, energy=-100.0, mu={'O': -5.0},
                     symbols=['O'] * 60),
    ]
    re.rng = _FakeRng(1.0 - 1e-12)
    assert re._accept_swap(0, 1)  # p clamped to 1 without an exp() overflow


def test_swap_probability_is_symmetric_in_slot_order():
    """Both orderings must agree, or the pairing loop's (i, j) choice would
    silently bias the ladder."""
    re, p = _partial_mu_ladder()
    for u in (p * 0.99, p * 1.01):
        re.rng = _FakeRng(u)
        assert re._accept_swap(0, 1) == re._accept_swap(1, 0)


def test_temperature_ladder_swap_unchanged_by_the_mu_fix():
    """With a shared mu the general form must reduce exactly to the old
    (beta_j - beta_i)(Phi_j - Phi_i)."""
    ri = _FakeReplica(beta=1.0, energy=10.0, mu={'Ag': 1.0}, symbols=['Ag'] * 5)
    rj = _FakeReplica(beta=0.5, energy=20.0, mu={'Ag': 1.0}, symbols=['Ag'] * 18)
    re = _bare_re()
    re.replicas = [ri, rj]
    # Phi_i = 5, Phi_j = 2 -> delta = (0.5 - 1.0)(2 - 5) = 1.5 -> p = 1.
    legacy_delta = (rj.units.beta - ri.units.beta) * (
        re._grand_potential(rj) - re._grand_potential(ri))
    assert legacy_delta == pytest.approx(1.5)
    re.rng = _FakeRng(1.0 - 1e-12)
    assert re._accept_swap(0, 1)


def test_temperature_ladder_uphill_swap_matches_legacy_probability():
    ri = _FakeReplica(beta=1.0, energy=10.0, mu={'Ag': 1.0}, symbols=['Ag'] * 18)
    rj = _FakeReplica(beta=0.5, energy=20.0, mu={'Ag': 1.0}, symbols=['Ag'] * 5)
    re = _bare_re()
    re.replicas = [ri, rj]
    # Phi_i = -8, Phi_j = 15 -> delta = (0.5 - 1.0)(15 - -8) = -11.5.
    p = float(np.exp(-11.5))
    re.rng = _FakeRng(p * 0.99)
    assert re._accept_swap(0, 1)
    re.rng = _FakeRng(p * 1.01)
    assert not re._accept_swap(0, 1)


def test_consolidated_status_line(caplog):
    """One console line covers all replicas; per-replica detail stays in files."""
    import logging
    from types import SimpleNamespace

    from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange

    pt = BatchedReplicaExchange.__new__(BatchedReplicaExchange)
    pt.gcmc_steps = 100
    pt.exchange_attempts = [4, 4]
    pt.exchange_successes = [1, 1]
    pt.logger = logging.getLogger('mcpy.ensembles.batched_replica_exchange')
    pt.replicas = [
        SimpleNamespace(atoms=[None] * 42, E_old=-10.5),
        SimpleNamespace(atoms=[None] * 44, E_old=-11.25),
    ]
    with caplog.at_level(logging.INFO,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._log_status(30)
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert 'RE 30/100' in msg and '42' in msg and '44' in msg and '25%' in msg


# --------------------------------------------------------------------------
# Degenerate-ladder warning. A 100%-accept ladder sat unremarked in the
# "Accepted Exchange (%)" column of six consecutive production runs before the
# mu-ladder bug above was found; an all-accept tally must not stay silent.
# --------------------------------------------------------------------------

def _tally_re(attempts, successes, mus=None):
    pt = BatchedReplicaExchange.__new__(BatchedReplicaExchange)
    pt.logger = logging.getLogger('mcpy.ensembles.batched_replica_exchange')
    pt.exchange_attempts = attempts
    pt.exchange_successes = successes
    pt.mus = mus
    return pt


@pytest.mark.parametrize('successes, expect', [
    ([40, 40], True),    # every swap accepted
    ([0, 0], True),      # no swap ever accepted
    ([12, 12], False),   # healthy partial acceptance
])
def test_degenerate_ladder_warning(caplog, successes, expect):
    pt = _tally_re([40, 40], successes)
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    warned = any(r.levelno == logging.WARNING for r in caplog.records)
    assert warned is expect


def test_degenerate_ladder_warning_needs_enough_attempts(caplog):
    """Two all-accept swaps prove nothing; don't cry wolf."""
    pt = _tally_re([1, 1], [1, 1])
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    assert not caplog.records


def test_degenerate_ladder_warning_names_the_ladder_kind(caplog):
    pt = _tally_re([40, 40], [40, 40], mus=[{'CO': -1.0}, {'CO': -0.5}])
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    text = ' '.join(r.getMessage() for r in caplog.records)
    assert '80/80' in text and 'mu spacing' in text

    caplog.clear()
    pt = _tally_re([40, 40], [40, 40], mus=None)
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    assert 'temperature spacing' in ' '.join(
        r.getMessage() for r in caplog.records)
