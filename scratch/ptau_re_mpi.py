"""MPI ReplicaExchange baseline for PtAu chemical ordering — mirror of
ptau_re_batched.py for a one-on-one wall-time comparison.

Launch:
    mpirun -n 3 python scratch/ptau_re_mpi.py --temperatures 400 800 1500 ...

One MPI rank per replica. Each rank loads its own AlchemiCalculator —
N model copies, no batched eval, serial GPU access via CUDA contexts.
"""
import argparse
import os
import time

import numpy as np
from ase.cluster import Octahedron

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import PermutationMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402
from mcpy.ensembles import ReplicaExchange  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[400, 800, 1500])
    p.add_argument('--gcmc-steps', type=int, default=200)
    p.add_argument('--exchange-interval', type=int, default=20)
    p.add_argument('--write-interval', type=int, default=5)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cueq', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--octa-length', type=int, default=5)
    p.add_argument('--octa-cutoff', type=int, default=1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--outdir', default='ptau_re_mpi_out')
    return p.parse_args()


def build_atoms(args, rng_seed):
    atoms = Octahedron('Pt', args.octa_length, args.octa_cutoff)
    n = len(atoms)
    n_pt = n // 2
    rng = np.random.default_rng(rng_seed)
    pt_idx = rng.choice(n, size=n_pt, replace=False)
    symbols = ['Au'] * n
    for i in pt_idx:
        symbols[i] = 'Pt'
    atoms.set_chemical_symbols(symbols)
    atoms.set_pbc(False)
    return atoms


def main():
    args = parse_args()

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    ss = np.random.SeedSequence(args.seed)
    n_replicas = len(args.temperatures)
    seeds = ss.generate_state(2 * n_replicas + 1, dtype=np.uint32)
    perm_seeds = [int(s) for s in seeds[:n_replicas]]
    init_seeds = [int(s) for s in seeds[n_replicas:2 * n_replicas]]
    master_seed = int(seeds[-1])

    t_setup_start = time.perf_counter()

    def gcmc_factory(T, rank):
        # Each rank gets its own calculator — separate CUDA context.
        calculator = AlchemiCalculator(
            checkpoint=args.checkpoint,
            device=args.device,
            enable_cueq=not args.no_cueq,
            compile_model=not args.no_compile,
        )
        atoms = build_atoms(args, init_seeds[rank])
        scell = SphericalCell(
            atoms, vacuum=3.0,
            species_radii={'Pt': 1.39, 'Au': 1.44},
            mc_sample_points=10_000,
        )
        move_selector = MoveSelector(
            [1],
            [PermutationMove(species=['Pt', 'Au'], seed=perm_seeds[rank])],
        )
        tag = f'PtAu{len(atoms)}_T{int(T)}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu={'Pt': 0.0, 'Au': 0.0},
            units_type='metal',
            species=['Pt', 'Au'],
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(args.outdir, f'gcmc_{tag}.out'),
            trajectory_write_interval=args.write_interval,
            outfile_write_interval=args.write_interval,
            traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
        )

    re = ReplicaExchange(
        gcmc_factory,
        temperatures=args.temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.write_interval,
        seed=master_seed,
        outfile=os.path.join(args.outdir, 'ptau_re_mpi.log'),
    )

    MPI.COMM_WORLD.Barrier()
    t_setup_end = time.perf_counter()

    t_run_start = time.perf_counter()
    re.run()
    MPI.COMM_WORLD.Barrier()
    t_run_end = time.perf_counter()

    if rank == 0:
        print(f"[MPI] setup_s={t_setup_end - t_setup_start:.2f} "
              f"run_s={t_run_end - t_run_start:.2f} "
              f"total_s={t_run_end - t_setup_start:.2f} "
              f"steps={args.gcmc_steps} n_replicas={n_replicas}")


if __name__ == '__main__':
    main()
