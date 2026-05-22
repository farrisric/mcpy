"""Chemical-ordering optimization of a Pt-Cu 201-atom truncated-octahedral
nanoparticle via batched replica-exchange MC on a single GPU.

Octahedron(7, 2) = 201 atoms — a closed-shell magic number for truncated
octahedral fcc clusters. 50/50 Pt/Cu.

Defaults aim for the RTX 5090 (32 GB VRAM): 12 replicas, geometric T ladder
300 K -> 2500 K, torch.compile ON for steady-state speed.
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
from mcpy.calculators import AlchemiCalculator, AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402
from mcpy.ensembles import BatchedReplicaExchange  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-replicas', type=int, default=12)
    p.add_argument('--t-min', type=float, default=300.0)
    p.add_argument('--t-max', type=float, default=2500.0)
    p.add_argument('--gcmc-steps', type=int, default=3000)
    p.add_argument('--exchange-interval', type=int, default=25)
    p.add_argument('--write-interval', type=int, default=100)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cueq', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--no-relax', action='store_true',
                   help='Use energy-only AlchemiCalculator instead of FIRE relax')
    p.add_argument('--fmax', type=float, default=0.05,
                   help='FIRE force convergence (eV/Å)')
    p.add_argument('--relax-steps', type=int, default=200,
                   help='Max FIRE steps per batched relax call')
    p.add_argument('--optimizer', default='fire2', choices=['fire', 'fire2'])
    p.add_argument('--octa-length', type=int, default=7)
    p.add_argument('--octa-cutoff', type=int, default=2)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--outdir', default='scratch/ptcu_re_out')
    return p.parse_args()


def geometric_ladder(t_min: float, t_max: float, n: int) -> list:
    return list(np.geomspace(t_min, t_max, n))


def build_atoms(args, rng_seed):
    atoms = Octahedron('Pt', args.octa_length, args.octa_cutoff)
    n = len(atoms)
    n_pt = n // 2
    rng = np.random.default_rng(rng_seed)
    pt_idx = rng.choice(n, size=n_pt, replace=False)
    symbols = ['Cu'] * n
    for i in pt_idx:
        symbols[i] = 'Pt'
    atoms.set_chemical_symbols(symbols)
    atoms.set_pbc(False)
    return atoms


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    temperatures = geometric_ladder(args.t_min, args.t_max, args.n_replicas)

    ss = np.random.SeedSequence(args.seed)
    n_replicas = len(temperatures)
    seeds = ss.generate_state(2 * n_replicas + 1, dtype=np.uint32)
    perm_seeds = [int(s) for s in seeds[:n_replicas]]
    init_seeds = [int(s) for s in seeds[n_replicas:2 * n_replicas]]
    master_seed = int(seeds[-1])

    t_setup_start = time.perf_counter()

    if args.no_relax:
        calculator = AlchemiCalculator(
            checkpoint=args.checkpoint,
            device=args.device,
            enable_cueq=not args.no_cueq,
            compile_model=not args.no_compile,
        )
    else:
        calculator = AlchemiFCalculator(
            checkpoint=args.checkpoint,
            steps=args.relax_steps,
            fmax=args.fmax,
            device=args.device,
            enable_cueq=not args.no_cueq,
            compile_model=not args.no_compile,
            optimizer=args.optimizer,
        )

    species = ['Pt', 'Cu']

    def gcmc_factory(T, rank):
        atoms = build_atoms(args, init_seeds[rank])
        scell = SphericalCell(
            atoms, vacuum=3.0,
            species_radii={'Pt': 1.39, 'Cu': 1.28},
            mc_sample_points=10_000,
        )
        move_selector = MoveSelector(
            [1],
            [PermutationMove(species=['Pt', 'Cu'], seed=perm_seeds[rank])],
        )
        tag = f'PtCu{len(atoms)}_T{int(T)}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu={'Pt': 0.0, 'Cu': 0.0},
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(args.outdir, f'gcmc_{tag}.out'),
            trajectory_write_interval=args.write_interval,
            outfile_write_interval=args.write_interval,
            traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
        )

    re = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        temperatures=temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.write_interval,
        seed=master_seed,
        outfile=os.path.join(args.outdir, 'ptcu_re.log'),
    )

    t_setup_end = time.perf_counter()
    print(f"[setup] {t_setup_end - t_setup_start:.2f}s | replicas={n_replicas} "
          f"T={[f'{t:.0f}' for t in temperatures]}")

    t_run_start = time.perf_counter()
    re.run()
    t_run_end = time.perf_counter()

    run_s = t_run_end - t_run_start
    per_step = run_s / args.gcmc_steps * 1000
    print(f"[Batched] setup_s={t_setup_end - t_setup_start:.2f} "
          f"run_s={run_s:.2f} "
          f"total_s={t_run_end - t_setup_start:.2f} "
          f"per_step_ms={per_step:.1f} "
          f"steps={args.gcmc_steps} n_replicas={n_replicas}")


if __name__ == '__main__':
    main()
