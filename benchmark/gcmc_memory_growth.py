"""Diagnose GPU memory growth over a long single-replica GCMC run.

Claim under test: during a long run, GPU memory climbs by more than the inserted
atoms justify. This logs, per step, the atom count against three memory figures:

  - allocated : torch.cuda.memory_allocated   -> live tensors (tracks atoms)
  - reserved  : torch.cuda.memory_reserved    -> caching-allocator pool (never
                shrinks; grows as the run visits new atom counts = fragmentation)
  - smi       : nvidia-smi memory.used        -> whole-device footprint

If `allocated` is a clean function of atom count but `reserved`/`smi` drift up
independently, the growth is allocator fragmentation, not a leak. The run then
repeats with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in the
environment before launching) to confirm the mitigation.

Run in the alchemi env:
    python benchmark/gcmc_memory_growth.py
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python benchmark/gcmc_memory_growth.py --tag _exp
"""
import argparse
import csv
import os
import subprocess

import numpy as np
from ase.cluster import Octahedron

from mcpy.utils.logging import configure as configure_logging
configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def smi_mb():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return float('nan')


def main():
    args = parse_args()
    import torch

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))
    atoms = Octahedron('Ag', args.np_length, 1)
    scell = SphericalCell(atoms, vacuum=args.vacuum, species_radii={'Ag': 2.947, 'O': 0},
                          mc_sample_points=100_000)
    calculator = AlchemiFCalculator(
        checkpoint=args.checkpoint, steps=args.relax_steps, fmax=args.fmax,
        device='cuda', enable_cueq=True, compile_model=args.compile)

    species = ['O']
    # Weight insertion > deletion so the atom count generally grows.
    move_selector = MoveSelector(
        [1, args.insert_weight],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=args.min_insert, seed=seed_ins)])
    mus = {'Ag': -2.99, 'O': -4.91 + args.delta_mu_O}

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[scell], calculator=calculator, mu=mus,
        units_type='metal', species=species, temperature=args.T,
        move_selector=move_selector,
        outfile=os.path.join(args.outdir, '_growth.out'),
        traj_file=os.path.join(args.outdir, '_growth.xyz'),
        trajectory_write_interval=10 ** 9, outfile_write_interval=10 ** 9)

    gcmc.initialize_run()
    rows = []
    for step in range(args.steps):
        gcmc._run()  # real per-step path
        torch.cuda.synchronize()
        rows.append(dict(
            step=gcmc._step,
            n_atoms=len(gcmc.atoms),
            allocated_MB=round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
            reserved_MB=round(torch.cuda.memory_reserved() / 1024 ** 2, 1),
            smi_MB=round(smi_mb(), 1),
        ))
        if step % 20 == 0:
            r = rows[-1]
            print(f"step {step:4d}  N={r['n_atoms']:4d}  "
                  f"alloc={r['allocated_MB']:8.1f}  resv={r['reserved_MB']:8.1f}  "
                  f"smi={r['smi_MB']:8.1f}", flush=True)
    gcmc.finalize_run()

    # Does empty_cache reclaim the drift? (fragmentation -> yes; leak -> no)
    resv_before = torch.cuda.memory_reserved() / 1024 ** 2
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    resv_after = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"\nempty_cache: reserved {resv_before:.1f} -> {resv_after:.1f} MB "
          f"(reclaimed {resv_before - resv_after:.1f} MB); "
          f"still-allocated {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

    path = os.path.join(OUTDIR, f'gcmc_memory_growth{args.tag}.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--np-length', type=int, default=6)      # 140 atoms
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--relax-steps', type=int, default=100)
    p.add_argument('--fmax', type=float, default=0.1)
    p.add_argument('--T', type=float, default=500.0)
    p.add_argument('--delta-mu-O', type=float, default=2.0,  # favor insertion
                   help='shift added to mu_O; higher -> more insertions')
    p.add_argument('--insert-weight', type=int, default=3)
    p.add_argument('--min-insert', type=float, default=0.5)
    p.add_argument('--vacuum', type=float, default=6.0)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--compile', action='store_true',
                   help='Enable torch.compile (recompiles as the atom count '
                        'changes; off by default)')
    p.add_argument('--outdir', default=os.path.join(OUTDIR, '_growth_tmp'))
    p.add_argument('--tag', default='')
    os.makedirs(p.parse_known_args()[0].outdir, exist_ok=True)
    return p.parse_args()


if __name__ == '__main__':
    main()
