"""Swap-acceptance correctness for the MPI ReplicaExchange._exchange_prob.

A shared-mu temperature ladder leaves the RE-level ``self.mus`` None, so the
grand-potential correction must be driven by each replica's ``state['mu']``.
The module imports without mpi4py (MPI is guarded), and _exchange_prob only
needs a logger + rank, so we drive it with object.__new__ and plain states.
"""
import logging
from collections import defaultdict
from types import SimpleNamespace

import ase.parallel
import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from mcpy.cell.null_cell import NullCell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.ensembles import replica_exchange as rex
from mcpy.ensembles.base_ensemble import write_xyz
from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange
from mcpy.ensembles.replica_exchange import ReplicaExchange
from mcpy.moves import DisplacementMove, MoveSelector


def _bare_re():
    re = object.__new__(ReplicaExchange)
    re.logger = logging.getLogger('test')
    re.rank = 0
    return re


# --------------------------------------------------------------------------
# Rung-dependent mu: real GrandCanonicalEnsemble states in metal units
# --------------------------------------------------------------------------

def _gcmc(temperature, mu, formula, energy):
    """A replica as a gcmc_factory(T=..., rank=...) builds it, holding
    ``formula`` at ``energy`` (no move is ever run)."""
    return GrandCanonicalEnsemble(
        atoms=Atoms(formula, cell=[20.0] * 3), cells=[NullCell()],
        units_type='metal',
        calculator=SimpleNamespace(get_potential_energy=lambda a: energy),
        mu=mu, species=sorted(mu), temperature=temperature,
        move_selector=MoveSelector([1], [DisplacementMove(species=['O'], seed=1)]),
        random_seed=1, traj_file=None, outfile=None)


def _log_weight(slot, config):
    """ln w_k(Z) = -beta_k Phi_Z^(k) - 3 sum_s N_Z,s ln Lambda_s(T_k): the
    grand-canonical weight of ``config``'s configuration in ``slot``."""
    n = {s: config.atoms.symbols.count(s) for s in slot._mu}
    phi = config.E_old - sum(slot._mu[s] * n[s] for s in n)
    return (-slot.units.beta * phi
            - 3.0 * sum(n[s] * np.log(slot.units.lambda_dbs[s]) for s in n))


def _p_from_weights(gi, gj):
    """Metropolis swap probability of X (in gi) and Y (in gj), written
    straight from the stationary weights: w_i(Y) w_j(X) / (w_i(X) w_j(Y))."""
    ln_ratio = (_log_weight(gi, gj) + _log_weight(gj, gi)
                - _log_weight(gi, gi) - _log_weight(gj, gj))
    return min(1.0, float(np.exp(ln_ratio)))


def test_t_ladder_swap_with_rung_dependent_mu():
    """A factory that sets mu_Ag(T) per rung (a 30 meV shift from 250 K to
    300 K) with 100 Ag on both sides. The old rule scored each config with
    its own rung's mu only and gave p = 4.9e-11 here; the stationary weights
    give p = 0.588."""
    gi = _gcmc(250.0, {'Ag': -2.80, 'O': -4.90}, 'Ag100O12', -300.0)
    gj = _gcmc(300.0, {'Ag': -2.83, 'O': -4.90}, 'Ag100O8', -280.19)
    p = _bare_re()._exchange_prob(gi.get_state(), gj.get_state())
    assert p == pytest.approx(_p_from_weights(gi, gj), rel=1e-9)
    assert p == pytest.approx(0.588, abs=1e-3)


def _random_pair(rng, shared_mu=False, shared_t=False):
    """Two replicas with random T, random per-rung mu and random Ag/O
    configurations, scaled so the swap exponent stays in a range where p is
    neither 0 nor clamped for most draws."""
    t_i = rng.uniform(200.0, 900.0)
    t_j = t_i if shared_t else rng.uniform(200.0, 900.0)
    mu_i = {'Ag': rng.uniform(-3.0, -2.5), 'O': rng.uniform(-5.5, -4.5)}
    mu_j = dict(mu_i) if shared_mu else {
        s: m + rng.uniform(-0.01, 0.01) for s, m in mu_i.items()}
    n_x = (int(rng.integers(95, 106)), int(rng.integers(0, 21)))
    n_y = (int(rng.integers(95, 106)), int(rng.integers(0, 21)))
    e_x = -2.9 * n_x[0] - 5.0 * n_x[1] + rng.normal(0.0, 0.5)
    e_y = -2.9 * n_y[0] - 5.0 * n_y[1] + rng.normal(0.0, 0.5)
    return (_gcmc(t_i, mu_i, f'Ag{n_x[0]}O{n_x[1]}', e_x),
            _gcmc(t_j, mu_j, f'Ag{n_y[0]}O{n_y[1]}', e_y))


def test_swap_matches_weights_and_batched_rule_on_random_states():
    """The MPI rule equals the probability from the stationary weights and
    makes the same accept/reject call as BatchedReplicaExchange._accept_swap,
    for T ladders with a different mu on each rung."""
    rng = np.random.default_rng(7)
    batched = object.__new__(BatchedReplicaExchange)
    batched.logger = logging.getLogger('test')
    checked = 0
    for _ in range(100):
        gi, gj = _random_pair(rng)
        p = _bare_re()._exchange_prob(gi.get_state(), gj.get_state())
        assert p == pytest.approx(_p_from_weights(gi, gj), rel=1e-9, abs=1e-300)
        if not 1e-200 < p < 1.0:
            continue
        checked += 1
        batched.replicas = [gi, gj]
        batched.rng = SimpleNamespace(get_uniform=lambda: p * (1.0 - 1e-9))
        assert batched._accept_swap(0, 1)
        batched.rng = SimpleNamespace(get_uniform=lambda: p * (1.0 + 1e-9))
        assert not batched._accept_swap(0, 1)
    assert checked > 20


# --------------------------------------------------------------------------
# Both partners of a do_exchange call reach the same probability
# --------------------------------------------------------------------------

class _PairComm:
    """MPI.COMM_WORLD for one rank of a two-rank job (even pairing)."""

    def __init__(self, partner_state):
        self.partner_state = partner_state

    def bcast(self, obj, root=0):
        return 0.0  # even pairing: (0, 1)

    def sendrecv(self, sendobj, dest, source):
        return self.partner_state

    def send(self, obj, dest):
        pass

    def recv(self, source):
        return 0.5


def _rank_view(rank, gcmc, partner):
    """ReplicaExchange on ``rank`` holding ``gcmc``; returns it and the list
    that records every probability it computes."""
    re = object.__new__(ReplicaExchange)
    re.logger = logging.getLogger('test')
    re.rank, re.size, re._re_step = rank, 2, 0
    re.rng = SimpleNamespace(get_uniform=lambda: 0.5)
    re.gcmc = gcmc
    re.comm = _PairComm(partner.get_state())
    seen = []

    def spy(state1, state2):
        seen.append(ReplicaExchange._exchange_prob(re, state1, state2))
        return seen[-1]
    re._exchange_prob = spy
    return re, seen


def test_both_ranks_compute_the_same_swap_probability():
    """Rank 0 and rank 1 each evaluate the swap of the same pair. The two
    values must be equal to the last bit: they share one uniform, and a
    one-ulp gap with u in between would duplicate one config and lose the
    other."""
    rng = np.random.default_rng(17)
    for _ in range(100):
        gi, gj = _random_pair(rng)
        re0, seen0 = _rank_view(0, gi, gj)
        re1, seen1 = _rank_view(1, gj, gi)
        re0.do_exchange()
        re1.do_exchange()
        assert seen0 == seen1


# --------------------------------------------------------------------------
# Global minimum: rungs with their own mu are not comparable
# --------------------------------------------------------------------------

def _write_gm(driver, mus, tmp_path):
    """Write the global minimum of two T-ladder replicas that hold the same
    configuration, through ``driver``. Returns the ranks of the frames."""
    atoms, energy = Atoms('Ag100O12'), -300.0

    def score(mu):
        return energy - sum(m * atoms.symbols.count(s) for s, m in mu.items())
    reps = [SimpleNamespace(_mu=mu, _best_score=score(mu), _best_energy=energy,
                            _best_atoms=atoms) for mu in mus]
    path = str(tmp_path / 'gm.xyz')
    if driver == 'mpi':
        re = _bare_re()
        re._global_minimum_file, re.gcmc = path, reps[0]
        gathered = iter([[(r._best_score, r._best_energy, r._best_atoms)
                          for r in reps], [r._mu for r in reps]])
        re.comm = SimpleNamespace(gather=lambda obj, root=0: next(gathered))
    else:
        re = object.__new__(BatchedReplicaExchange)
        re.logger, re.replicas, re.mus = logging.getLogger('test'), reps, None
        re._global_minimum_file = path
    re._write_global_minimum()
    return [f.info['rank'] for f in read(path, ':')]


@pytest.mark.parametrize('driver', ['mpi', 'batched'])
def test_global_minimum_per_rung_when_t_rungs_have_own_mu(driver, tmp_path):
    """mu_Ag 30 meV apart on two T rungs: one configuration scores 3 eV
    apart, so the lowest score always picked the higher-mu_Ag rung. Each
    rung must get its own frame, as on a mu ladder."""
    mus = [{'Ag': -2.80, 'O': -4.90}, {'Ag': -2.83, 'O': -4.90}]
    assert _write_gm(driver, mus, tmp_path) == [0, 1]


@pytest.mark.parametrize('driver', ['mpi', 'batched'])
def test_global_minimum_single_frame_when_t_rungs_share_mu(driver, tmp_path):
    mu = {'Ag': -2.80, 'O': -4.90}
    assert _write_gm(driver, [mu, dict(mu)], tmp_path) == [0]


def test_write_xyz_writes_on_a_non_master_mpi_rank(tmp_path, monkeypatch):
    """ase.io.write runs on MPI rank 0 only unless told otherwise. Every
    rank of ReplicaExchange owns its files, so rank 1 must write too."""
    path = tmp_path / 'rank1.xyz'
    atoms = Atoms('H', positions=[[1, 1, 1]], cell=[5] * 3, pbc=True)
    with monkeypatch.context() as m:
        m.setattr(ase.parallel, 'world', SimpleNamespace(size=2, rank=1))
        with open(path, 'w') as handle:
            write_xyz(atoms, -1.0, handle)
    assert path.stat().st_size > 0
    assert len(read(path, index=':')) == 1


def _bare_mpi_re():
    re = object.__new__(rex.ReplicaExchange)
    re.logger = logging.getLogger('test')
    re.rank = 0
    return re


# MPI ReplicaExchange (fake communicator)

def test_mpi_exchange_prob_rejects_nan(caplog):
    re = _bare_mpi_re()
    s1 = {'beta': 1.0, 'energy': float('nan'), 'mu': None, 'atoms': Atoms('Ag')}
    s2 = {'beta': 0.5, 'energy': 1.0, 'mu': None, 'atoms': Atoms('Ag')}
    with caplog.at_level(logging.WARNING, logger='test'):
        assert re._exchange_prob(s1, s2) == 0.0
    assert 'non-finite' in caplog.text


def _state(beta, energy, mu, formula):
    # Lambda = 1 (as in LJ units): the de Broglie term drops out, so these
    # hand-made states test the Phi terms alone.
    return {'beta': beta, 'energy': energy, 'mu': mu, 'atoms': Atoms(formula),
            'lambda_dbs': defaultdict(lambda: 1.0)}


def test_mu_ladder_swap_rejects_nan_energy(caplog):
    """On a mu ladder the energies cancel only for finite values: a diverged
    replica must not swap. The old mu rule never read the energy and so
    returned p = 1 here."""
    re = _bare_re()
    s1 = _state(beta=1.0, energy=float('nan'), mu={'Ag': 0.5}, formula='Ag5')
    s2 = _state(beta=1.0, energy=10.0, mu={'Ag': 0.0}, formula='Ag8')
    with caplog.at_level(logging.WARNING, logger='test'):
        assert re._exchange_prob(s1, s2) == 0.0
    assert 'non-finite' in caplog.text
