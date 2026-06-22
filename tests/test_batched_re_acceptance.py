"""Swap-acceptance correctness for BatchedReplicaExchange.

Grand-canonical replicas at different temperatures must be compared through the
grand potential Phi = E - sum_s mu_s N_s, not bare energy. These tests pin that
behaviour without needing torch/mpi4py by driving _accept_swap / _grand_potential
with lightweight replica stubs.
"""
import logging

from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange


class _FakeUnits:
    def __init__(self, beta):
        self.beta = beta


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
