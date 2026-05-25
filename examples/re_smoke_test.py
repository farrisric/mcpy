"""MPI smoke test for ReplicaExchange swap correctness.

Run with several rank counts, including an odd one::

    mpirun -n 2 python examples/re_smoke_test.py
    mpirun -n 3 python examples/re_smoke_test.py
    mpirun -n 4 python examples/re_smoke_test.py

It stubs out the GCMC engine so no calculator/MACE is needed and checks two
things the recent bug fixes are about:

* paired ranks always reach the *same* accept/reject decision, so the set of
  per-replica configs stays a permutation of the originals (no config is
  duplicated or lost), and
* partner selection never deadlocks or targets an out-of-range rank, even for
  an odd number of ranks.
"""
from mpi4py import MPI

from mcpy.utils import RandomNumberGenerator
from mcpy.ensembles import ReplicaExchange


class StubGCMC:
    """Minimal stand-in for GrandCanonicalEnsemble used by ReplicaExchange.

    ``config`` is a unique label that travels with a swap; ``beta`` is the
    per-rank ladder value and must NOT move when a swap is accepted.
    """

    def __init__(self, rank, beta):
        self.config = rank          # unique, tracked across swaps
        self.beta = beta
        self.energy = 0.0
        self._mu = {'X': -1.0}
        self.exchange_attempts = 0
        self.exchange_successes = 0

    def get_state(self):
        return {
            'config': self.config,
            'energy': self.energy,
            'beta': self.beta,
            'mu': self._mu,
            'n_atoms': 0,
        }

    def set_state(self, state):
        # Adopt the partner's configuration + energy, keep our own beta.
        self.config = state['config']
        self.energy = state['energy']


def build_re(comm, beta):
    re = object.__new__(ReplicaExchange)
    re.comm = comm
    re.rank = comm.Get_rank()
    re.size = comm.Get_size()
    re.rng = RandomNumberGenerator(seed=31)  # identical seed on every rank
    re.logger = __import__('logging').getLogger('smoke')
    re._re_step = 0
    re.gcmc = StubGCMC(re.rank, beta)
    re._exchange_prob = re._exchange_prob_T
    return re


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Temperature-like ladder: distinct beta per rank so probs are sub-unity.
    beta = 1.0 + 0.5 * rank
    re = build_re(comm, beta)

    energy_rng = RandomNumberGenerator(seed=1000 + rank)
    rounds = 500
    for _ in range(rounds):
        # New energy each round so the swap probability actually varies.
        re.gcmc.energy = energy_rng.get_uniform() * 10.0 - 5.0
        re.do_exchange()
        re._re_step += 1

        configs = comm.allgather(re.gcmc.config)
        assert sorted(configs) == list(range(size)), (
            f"config set corrupted: {sorted(configs)} (expected permutation of "
            f"0..{size - 1}) — a swap pair disagreed"
        )

    comm.Barrier()
    if rank == 0:
        print(f"OK: {size} ranks, {rounds} exchange rounds, configs conserved.")


if __name__ == '__main__':
    main()
