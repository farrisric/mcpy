"""Swap-acceptance correctness for the MPI ReplicaExchange._exchange_prob_T.

A shared-mu temperature ladder leaves the RE-level ``self.mus`` None, so the
grand-potential correction must be driven by each replica's ``state['mu']``.
The module imports without mpi4py (MPI is guarded), and _exchange_prob_T only
needs a logger + rank, so we drive it with object.__new__ and plain states.
"""
import logging

from ase import Atoms

from mcpy.ensembles.replica_exchange import ReplicaExchange


def _bare_re():
    re = object.__new__(ReplicaExchange)
    re.logger = logging.getLogger('test')
    re.rank = 0
    return re


def _state(beta, energy, mu, formula):
    return {'beta': beta, 'energy': energy, 'mu': mu, 'atoms': Atoms(formula)}


def test_exchange_prob_T_subtracts_mu_N_for_grand_canonical():
    re = _bare_re()
    # Phi1 = 10 - 1*5 = 5, Phi2 = 20 - 1*18 = 2
    # delta = (beta2 - beta1)(Phi2 - Phi1) = (0.5 - 1.0)(2 - 5) = 1.5 -> p = 1
    s1 = _state(beta=1.0, energy=10.0, mu={'Ag': 1.0}, formula='Ag5')
    s2 = _state(beta=0.5, energy=20.0, mu={'Ag': 1.0}, formula='Ag18')
    assert re._exchange_prob_T(s1, s2) == 1.0
    # bare-E would give delta = (0.5-1.0)(20-10) = -5 -> p << 1
    assert re._exchange_prob_T(s1, s2) > 0.9


def test_exchange_prob_T_without_mu_is_bare_energy():
    re = _bare_re()
    s1 = _state(beta=1.0, energy=10.0, mu=None, formula='Ag5')
    s2 = _state(beta=0.5, energy=20.0, mu=None, formula='Ag18')
    # delta = (0.5 - 1.0)(20 - 10) = -5
    import numpy as np
    assert re._exchange_prob_T(s1, s2) == min(1.0, float(np.exp(-5.0)))


def _mu_state(beta, energy, mu, formula):
    # _exchange_prob_mu also logs N1/N2, so carry n_atoms.
    s = _state(beta, energy, mu, formula)
    s['n_atoms'] = len(s['atoms'])
    return s


def test_exchange_prob_mu_reads_species_from_mu_dict():
    """mu-ladder swap (shared beta): driven by Delta_N and Delta_mu, energies
    cancel. Reads species from state['mu'], not the nonexistent gcmc.species."""
    import numpy as np
    re = _bare_re()
    # Delta_N = 8 - 5 = 3; arg = b2*mu2*(-dN) + b1*mu1*dN = 1*0*(-3) + 1*0.5*3 = 1.5
    s1 = _mu_state(beta=1.0, energy=10.0, mu={'Ag': 0.5}, formula='Ag5')
    s2 = _mu_state(beta=1.0, energy=999.0, mu={'Ag': 0.0}, formula='Ag8')
    assert re._exchange_prob_mu(s1, s2) == min(1.0, float(np.exp(1.5)))


def test_exchange_prob_mu_clamps_and_ignores_energy():
    import numpy as np
    re = _bare_re()
    # Reverse direction: arg = 1*0.5*(-(-3)) ... dN = 5 - 8 = -3
    # arg = b2*mu2*(-dN) + b1*mu1*dN = 1*0.5*(3) + 1*0*(-3) = 1.5 -> clamps to 1
    s1 = _mu_state(beta=1.0, energy=-50.0, mu={'Ag': 0.0}, formula='Ag8')
    s2 = _mu_state(beta=1.0, energy=50.0, mu={'Ag': 0.5}, formula='Ag5')
    assert re._exchange_prob_mu(s1, s2) == min(1.0, float(np.exp(1.5)))
