"""Batched replica-exchange GCMC with memory-capping chunks (single GPU).

All replicas live in one process and share one ``AlchemiFCalculator``. Each MC
step relaxes and scores every replica's trial move in a batched FIRE pass. The
``chunk_size`` argument splits that batch into sub-batches, so peak GPU memory is
set by the largest chunk (``chunk_size`` x atoms-per-replica) instead of by the
replica count. That lets you run many replicas, or large nanoparticles, that
would otherwise exceed device memory at the cost of more (smaller) forward
passes.

Picking ``chunk_size``:
  * ``None``  -> whole batch in one pass (fastest, most memory).
  * ``k``     -> ceil(n_replicas / k) passes, peak memory ~ k replicas.
  * ``1``     -> one replica per pass (lowest memory, slowest).
Each forward carries a fixed ~17 ms overhead, so prefer the largest ``k`` whose
peak memory fits your card rather than ``1``.

(For a no-relaxation run, ``AlchemiCalculator`` also takes ``energy_only=True``,
which skips force autograd for a further ~12% memory saving.)

Requirements:
  pip install -e .[alchemi]

Run::

    python examples/re_gcmc_batched_chunked.py --temperatures 300 400 500 600 \
        --np-length 10 --chunk-size 1

No mpirun. One GPU. Each replica builds its OWN cell and move_selector
(BatchedReplicaExchange would otherwise share state across replicas).

GPU memory on long runs: the atom count changes every accepted move, which
fragments the CUDA caching allocator, so reserved GPU memory drifts up over a
long run (allocator fragmentation, not a leak). Launch with
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``, or pass
``empty_cache_interval=N`` to BatchedReplicaExchange as an in-loop fallback.
"""
import argparse
import os

import numpy as np
from ase.cluster import Octahedron

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402
from mcpy.ensembles import BatchedReplicaExchange  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[300, 400, 500, 600, 700, 800],
                   help='Replica temperatures (K); one replica each')
    p.add_argument('--np-length', type=int, default=10,
                   help='Octahedron edge length (larger = bigger nanoparticle)')
    p.add_argument('--chunk-size', type=int, default=1,
                   help='Replicas per forward pass. Caps peak GPU memory at '
                        'chunk_size x atoms-per-replica. 0/negative -> whole batch.')
    p.add_argument('--gcmc-steps', type=int, default=200)
    p.add_argument('--exchange-interval', type=int, default=10)
    p.add_argument('--write-interval', type=int, default=1)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cueq', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--fmax', type=float, default=0.05)
    p.add_argument('--relax-steps', type=int, default=500)
    p.add_argument('--mu-Ag', type=float, default=-2.99)
    p.add_argument('--mu-O', type=float, default=-4.91)
    p.add_argument('--delta-mu-O', type=float, default=-0.5)
    p.add_argument('--min-insert', type=float, default=0.5)
    p.add_argument('--vacuum', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--outdir', default='.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    chunk_size = args.chunk_size if args.chunk_size and args.chunk_size > 0 else None

    ss = np.random.SeedSequence(args.seed)
    n_replicas = len(args.temperatures)
    # 2 move seeds per replica (deletion + insertion) + 1 master seed for RE.
    seeds = ss.generate_state(2 * n_replicas + 1, dtype=np.uint32)
    move_seeds = [int(s) for s in seeds[:2 * n_replicas]]
    master_seed = int(seeds[-1])

    base_atoms = Octahedron('Ag', args.np_length, 1)
    print(f'{n_replicas} replicas x {len(base_atoms)} atoms; '
          f'chunk_size={chunk_size} -> peak memory ~ '
          f'{(chunk_size or n_replicas) * len(base_atoms)} atoms/pass')

    # ONE model + calculator, shared across replicas. chunk_size caps peak memory.
    calculator = AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        enable_cueq=not args.no_cueq,
        compile_model=not args.no_compile,
        chunk_size=chunk_size,
    )

    species = ['O']
    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    def gcmc_factory(T, rank):
        atoms = base_atoms.copy()
        scell = SphericalCell(atoms, vacuum=args.vacuum,
                              species_radii={'Ag': 2.947, 'O': 0},
                              mc_sample_points=100_000)
        s = move_seeds[2 * rank:2 * (rank + 1)]
        move_selector = MoveSelector(
            [1, 1],
            [DeletionMove(scell, species=species, seed=s[0]),
             InsertionMove(scell, species=species, min_insert=args.min_insert, seed=s[1])],
        )
        tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(args.outdir, f'gcmc_chunked_{tag}.out'),
            trajectory_write_interval=args.write_interval,
            outfile_write_interval=args.write_interval,
            traj_file=os.path.join(args.outdir, f'gcmc_chunked_{tag}.xyz'),
        )

    pt = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        temperatures=args.temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.write_interval,
        seed=master_seed,
        outfile=os.path.join(args.outdir, 'replica_exchange_chunked.log'),
    )

    if args.device == 'cuda':
        import torch
        torch.cuda.reset_peak_memory_stats()
        pt.run()
        print(f'peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB')
    else:
        pt.run()


if __name__ == '__main__':
    main()
