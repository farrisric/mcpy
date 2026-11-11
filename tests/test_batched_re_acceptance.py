"""Swap-acceptance correctness for BatchedReplicaExchange.

Grand-canonical replicas at different temperatures must be compared through the
grand potential Phi = E - sum_s mu_s N_s, not bare energy. These tests pin that
behaviour without needing torch/mpi4py by driving _accept_swap / _grand_potential
with lightweight replica stubs.
"""
import logging
from collections import defaultdict

import pytest

from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange


class _FakeUnits:
    def __init__(self, beta, lambda_dbs=None):
        self.beta = beta
        self.molecules = {}
        # Lambda = 1 unless a test opts in: keeps the de Broglie cross-term
        # zero so the Lambda-free expectations below stay exact.
        self.lambda_dbs = (lambda_dbs if lambda_dbs is not None
                           else defaultdict(lambda: 1.0))


class _FakeAtoms:
    def __init__(self, symbols):
        self._symbols = symbols

    def get_chemical_symbols(self):
        return self._symbols


class _FakeReplica:
    def __init__(self, beta, energy, mu, symbols, lambda_dbs=None):
        self.units = _FakeUnits(beta, lambda_dbs)
        self.E_old = energy
        self._mu = mu
        self.atoms = _FakeAtoms(symbols)

    def _minimum_score(self, atoms, energy):
        """Mirrors GrandCanonicalEnsemble._minimum_score for atomic species
        (these fakes carry no molecular species) so _grand_potential's
        delegation to it can be exercised without a real ensemble."""
        score = energy
        for specie, mu in (self._mu or {}).items():
            score -= mu * self._species_count(atoms, specie)
        return score

    def _species_count(self, atoms, specie):
        """Mirrors GrandCanonicalEnsemble._species_count (atomic only)."""
        return atoms.get_chemical_symbols().count(specie)


class _FakeRng:
    def __init__(self, u):
        self.u = u

    def get_uniform(self):
        return self.u


def _bare_re():
    re = object.__new__(BatchedReplicaExchange)
    re.logger = logging.getLogger('test')
    return re


# --------------------------------------------------------------------------
# mu-ladder swaps (bug: the temperature-only criterion collapses to delta=0
# when both replicas share a temperature, accepting every swap unconditionally)
# --------------------------------------------------------------------------


def test_nan_energy_rejects_the_swap_and_warns(caplog):
    """A diverged relaxation (NaN energy) must sever the swap loudly: the
    old min(1, exp(nan)) form silently *accepted* every such swap, and the
    plain comparison form would silently reject."""
    re = _bare_re()
    re.replicas = [
        _FakeReplica(beta=1.0, energy=float('nan'), mu={'Ag': 1.0},
                     symbols=['Ag'] * 5),
        _FakeReplica(beta=0.5, energy=20.0, mu={'Ag': 1.0},
                     symbols=['Ag'] * 5),
    ]
    re.rng = _FakeRng(1e-12)
    with caplog.at_level(logging.WARNING, logger='test'):
        assert re._accept_swap(0, 1) is False
    assert 'non-finite' in ' '.join(r.getMessage() for r in caplog.records)


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
    pt._mid_attempts = None
    pt._mid_successes = None
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
    # One record (log aggregators must not strand the reason line), counting
    # swap attempts, not the doubled per-slot tallies (both slots of a pair
    # are incremented per attempt: 40 attempts, not 80).
    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert '40/40' in text and 'mu spacing' in text

    caplog.clear()
    pt = _tally_re([40, 40], [40, 40], mus=None)
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    assert 'temperature spacing' in ' '.join(
        r.getMessage() for r in caplog.records)


def test_dead_rung_warned_from_second_half_tally(caplog):
    """The documented real failure: pair 3-4 swapped freely while the replicas
    were still alike, then accepted nothing after the midpoint. Its whole-run
    tally (5/40) looks alive; only the second half (0/20) shows it is dead."""
    pt = _tally_re([40, 80, 80, 80, 40], [20, 35, 35, 25, 5],
                   mus=[{'CO': -1.0}] * 5)
    pt.pair_attempts = [40, 40, 40, 40]
    pt.pair_successes = [20, 15, 20, 5]
    pt._mid_attempts = [20, 20, 20, 20]
    pt._mid_successes = [10, 10, 12, 5]
    with caplog.at_level(logging.WARNING,
                         logger='mcpy.ensembles.batched_replica_exchange'):
        pt._warn_if_ladder_degenerate()
    text = ' '.join(r.getMessage() for r in caplog.records)
    assert 'pair 3-4' in text
    assert 'pair 0-1' not in text  # kept accepting in the second half


# --------------------------------------------------------------------------
# Replica isolation. A factory written for the MPI class (module-level shared
# atoms/moves, safe under one-process-per-rank) silently corrupts every
# replica under batching; the constructor must refuse it.
# --------------------------------------------------------------------------

class _StubBatchCalc:
    def get_potential_energies(self, atoms_list):
        return [0.0] * len(atoms_list)


@pytest.mark.parametrize('shared, match', [('atoms', 'share'), ('cells', 'cells')])
def test_shared_replica_state_is_refused(shared, match):
    from types import SimpleNamespace

    thing = object()

    def factory(T, rank):
        return SimpleNamespace(atoms=thing if shared == 'atoms' else object(),
                               move_selector=object(),
                               cells=[thing] if shared == 'cells' else [])

    with pytest.raises(ValueError, match=match):
        BatchedReplicaExchange(factory, _StubBatchCalc(),
                               temperatures=[300.0, 400.0])


def test_distinct_replica_state_is_accepted():
    from types import SimpleNamespace

    from mcpy.utils.random_number_generator import RandomNumberGenerator

    def factory(T, rank):
        selector = SimpleNamespace(rng=RandomNumberGenerator(seed=rank),
                                   move_list=[])
        return SimpleNamespace(atoms=object(), move_selector=selector,
                               cells=[object()])

    re = BatchedReplicaExchange(factory, _StubBatchCalc(),
                                temperatures=[300.0, 400.0])
    assert re.n_replicas == 2


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
