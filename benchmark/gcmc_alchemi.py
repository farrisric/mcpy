"""GCMC benchmark — Alchemi (nvalchemi) calculator.

Run alongside gcmc_mace.py (same seeds, same system) to compare wall time.
Total time includes model loading + torch.compile warmup + GCMC run.

Usage:
  conda activate alchemi
  python benchmark/gcmc_alchemi.py
  python benchmark/gcmc_alchemi.py --steps 100 --outdir benchmark_results
"""
import time
T_START = time.perf_counter()  # capture before any imports

import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.cluster import Octahedron  # noqa: E402

from mcpy.utils.logging import configure as configure_logging  # noqa: E402
from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402

configure_logging(level=logging.WARNING)

# ── shared config (must match gcmc_mace.py) ──────────────────────────────────
MASTER_SEED = 42
STEPS = 100
T_K = 500.0
MU_AG = -2.99
MU_O = -4.91
DELTA_MU_O = -0.5
VACUUM = 3.0
SPECIES_RADII = {'Ag': 2.947, 'O': 0}
MIN_INSERT = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=STEPS)
    p.add_argument('--device', default='cuda')
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='Named checkpoint or local .pt path')
    p.add_argument('--fmax', type=float, default=0.05)
    p.add_argument('--relax-steps', type=int, default=500)
    p.add_argument('--no-cueq', action='store_true', help='Disable cuEquivariance')
    p.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    p.add_argument('--outdir', default='benchmark_results')
    return p.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    seed_del, seed_ins, seed_sel = (int(s) for s in ss.generate_state(3, dtype=np.uint32))

    print(f'Alchemi benchmark  |  steps={args.steps}  device={args.device}  seed={MASTER_SEED}')

    calculator = AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        enable_cueq=not args.no_cueq,
        compile_model=not args.no_compile,
    )

    atoms = Octahedron('Ag', 8, 3)
    scell = SphericalCell(atoms, vacuum=VACUUM, species_radii=SPECIES_RADII,
                          mc_sample_points=100_000)
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=['O'], seed=seed_del),
         InsertionMove(scell, species=['O'], min_insert=MIN_INSERT, seed=seed_ins)],
        seed=seed_sel,
    )

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[scell],
        calculator=calculator,
        mu={'Ag': MU_AG, 'O': MU_O + DELTA_MU_O},
        units_type='metal',
        species=['O'],
        temperature=T_K,
        move_selector=move_selector,
        outfile=os.path.join(args.outdir, 'alchemi.out'),
        traj_file=os.path.join(args.outdir, 'alchemi.xyz'),
        random_seed=MASTER_SEED,
    )

    gcmc.initialize_run()
    sync()
    t_gcmc = time.perf_counter()
    gcmc.run(args.steps)
    sync()
    t_gcmc_end = time.perf_counter()

    ms = gcmc.move_selector
    n_O = sum(1 for s in gcmc._atoms.get_chemical_symbols() if s == 'O')

    gcmc_s = t_gcmc_end - t_gcmc
    total_s = t_gcmc_end - T_START

    print('\n' + '=' * 50)
    print(f'  Alchemi results — {args.steps} steps')
    print('=' * 50)
    print('  Optimizer:       FIRE (nvalchemi GPU)')
    print(f'  GCMC run time:   {gcmc_s:>8.2f} s  ({gcmc_s/args.steps*1000:.1f} ms/step)')
    print(f'  Total wall time: {total_s:>8.2f} s  (incl. model loading + compile)')
    print(f'  Final N_O:       {n_O}')
    print(f'  Final energy:    {gcmc.E_old:.4f} eV')
    print(f'  Relax steps:     total={calculator.total_relax_steps}  '
          f'avg/call={calculator.total_relax_steps / max(1, args.steps):.1f}')
    print()
    print('  Move acceptance rates:')
    for name, acc, att, ratio in zip(ms.move_list_names,
                                     ms.move_acceptance_total,
                                     ms.move_counter_total,
                                     ms.total_ratios()):
        rate = f'{ratio:.3f}' if not np.isnan(ratio) else 'n/a'
        print(f'    {name}  accepted={acc}  attempted={att}  rate={rate}')
    print(f'\n  Output: {os.path.abspath(args.outdir)}/alchemi.out  /  alchemi.xyz')


if __name__ == '__main__':
    main()
