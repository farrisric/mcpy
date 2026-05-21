"""Benchmark: 100 GCMC steps on an Ag nanoparticle — MACE vs Alchemi.

Mirrors examples/gcmc_nano.py exactly (Octahedron Ag6, SphericalCell, O insertion/deletion)
but runs both calculators back-to-back with the same random seed so the trial-move sequences
are identical, making timing and acceptance-rate comparisons directly meaningful.

Two calculator modes (selected with --mode):
  energy-only  mace_mp()        vs  AlchemiCalculator     (single forward pass, no relax)
  relax        MACE_F_Calculator vs  AlchemiFCalculator    (geometry relax before energy)

Output per run (written to --outdir, default ./benchmark_results/):
  <label>.out         step-by-step GCMC log (N, energy, acceptance ratios)
  <label>.xyz         trajectory file (extended XYZ, viewable in OVITO/ASE)
  summary.txt         timing + acceptance comparison table

Usage:
  conda activate alchemi
  cd /path/to/mcpy
  python benchmark/gcmc_nano_benchmark.py
  python benchmark/gcmc_nano_benchmark.py --mode relax --steps 100
  python benchmark/gcmc_nano_benchmark.py --outdir /tmp/bench
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
from ase.cluster import Octahedron
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging
from mcpy.moves import DeletionMove, InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import AlchemiCalculator, AlchemiFCalculator, MACE_F_Calculator
from mcpy.cell import SphericalCell

configure_logging(level=logging.WARNING)  # suppress per-step INFO noise during benchmark


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=100, help='GCMC steps per run')
    p.add_argument('--mode', choices=['energy-only', 'relax'], default='energy-only',
                   help='energy-only: single forward pass; relax: geometry relax each step')
    p.add_argument('--model-path', default=None,
                   help='Local MACE .pt path. Omit to auto-download medium-mpa-0.')
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='Alchemi checkpoint name or .pt path (default: medium-mpa-0)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--fmax', type=float, default=0.05, help='Relax convergence (eV/Å)')
    p.add_argument('--relax-steps', type=int, default=500, help='Max relax steps')
    p.add_argument('--mace-optimizer', choices=['lbfgs', 'fire'], default='lbfgs',
                   help='ASE optimizer for MACE_F. Use "fire" for apples-to-apples '
                        'comparison with Alchemi FIRE.')
    p.add_argument('--alchemi-optimizer', choices=['fire', 'fire2'], default='fire',
                   help='nvalchemi optimizer: "fire" (classic) or "fire2" (improved).')
    p.add_argument('--no-cueq', action='store_true', help='Disable cuEquivariance')
    p.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--mu-Ag', type=float, default=-2.99)
    p.add_argument('--mu-O', type=float, default=-4.91)
    p.add_argument('--delta-mu-O', type=float, default=-0.5)
    p.add_argument('--seed', type=int, default=42, help='Master RNG seed (same for both runs)')
    p.add_argument('--vacuum', type=float, default=3.0)
    p.add_argument('--oct-cutoff', type=int, default=8,
                   help='Octahedron cutoff arg (8,3)=289 atoms, (10,3)=586, (12,4)=1009')
    p.add_argument('--oct-layers', type=int, default=3,
                   help='Octahedron layers arg')
    p.add_argument('--outdir', default='benchmark_results',
                   help='Directory for .out/.xyz/summary.txt files')
    return p.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_ensemble(atoms, calculator, args, seed_del, seed_ins, seed_sel, label):
    scell = SphericalCell(atoms, vacuum=args.vacuum,
                          species_radii={'Ag': 2.947, 'O': 0},
                          mc_sample_points=100_000)
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=['O'], seed=seed_del),
         InsertionMove(scell, species=['O'], min_insert=0.5, seed=seed_ins)],
        seed=seed_sel,
    )
    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}
    slug = label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
    return GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[scell],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=['O'],
        temperature=args.T,
        move_selector=move_selector,
        outfile=os.path.join(args.outdir, f'{slug}.out'),
        traj_file=os.path.join(args.outdir, f'{slug}.xyz'),
        random_seed=args.seed,
    )


def run_gcmc(label, calculator, args, seed_del, seed_ins, seed_sel):
    atoms = Octahedron('Ag', args.oct_cutoff, args.oct_layers)

    gcmc = build_ensemble(atoms, calculator, args, seed_del, seed_ins, seed_sel, label)

    # warmup: initialize (loads model cache, builds NL, etc.) without timing
    gcmc.initialize_run()
    sync()

    t0 = time.perf_counter()
    gcmc.run(args.steps)
    sync()
    elapsed = time.perf_counter() - t0

    ms = gcmc.move_selector
    names = ms.move_list_names
    ratios = ms.total_ratios()
    accepted = ms.move_acceptance_total
    attempted = ms.move_counter_total
    move_stats = list(zip(names, accepted, attempted, ratios))

    n_O = sum(1 for s in gcmc._atoms.get_chemical_symbols() if s == 'O')
    energy = gcmc.E_old

    total_relax = getattr(calculator, 'total_relax_steps', None)
    opt_name = getattr(calculator, 'optimizer_name', None)

    return {
        'label': label,
        'total_s': elapsed,
        'per_step_ms': elapsed / args.steps * 1000,
        'n_O': n_O,
        'energy_eV': energy,
        'move_stats': move_stats,
        'total_relax_steps': total_relax,
        'optimizer': opt_name,
    }


def format_results(results, steps, mode, oct_cutoff, oct_layers):
    lines = []
    lines.append(f'{"":=<70}')
    lines.append(f'  GCMC benchmark — {steps} steps — {mode} mode')
    n0 = len(Octahedron('Ag', oct_cutoff, oct_layers))
    lines.append(f'  System: Ag Octahedron({oct_cutoff},{oct_layers}) '
                 f'= {n0} atoms + O adsorption')
    lines.append(f'{"":=<70}')
    lines.append('')
    lines.append(f'{"Calculator":<20}  {"Total (s)":>10}  {"Per step (ms)":>14}  '
                 f'{"Final N_O":>10}  {"Final E (eV)":>14}')
    lines.append(f'{"-"*70}')

    for r in results:
        lines.append(f'{r["label"]:<20}  {r["total_s"]:>10.2f}  {r["per_step_ms"]:>14.1f}  '
                     f'{r["n_O"]:>10}  {r["energy_eV"]:>14.4f}')

    if any(r.get('total_relax_steps') is not None for r in results):
        lines.append('')
        lines.append('Relaxation step counts (per get_potential_energy call)')
        lines.append(f'{"Calculator":<20}  {"Optimizer":>10}  '
                     f'{"Total steps":>12}  {"Avg/call":>10}')
        lines.append(f'{"-"*60}')
        for r in results:
            total = r.get('total_relax_steps')
            opt = r.get('optimizer') or 'n/a'
            if total is None:
                continue
            avg = total / max(1, steps)
            lines.append(f'{r["label"]:<20}  {opt:>10}  {total:>12}  {avg:>10.1f}')

    if len(results) == 2:
        r0, r1 = results
        speedup = r0['total_s'] / r1['total_s']
        delta_e = abs(r1['energy_eV'] - r0['energy_eV'])
        delta_n = abs(r1['n_O'] - r0['n_O'])
        lines.append('')
        lines.append(f'Speedup (MACE/Alchemi): {speedup:.2f}x  '
                     f'(>1 means MACE faster, <1 means Alchemi faster)')
        lines.append(f'Final energy difference: {delta_e:.4f} eV')
        lines.append(f'Final N_O difference:    {delta_n}')

    lines.append('')
    lines.append('Move acceptance rates')
    lines.append(f'{"Calculator":<20}  {"Move":<6}  {"Accepted":>10}  '
                 f'{"Attempted":>10}  {"Rate":>8}')
    lines.append(f'{"-"*60}')
    for r in results:
        for name, acc, att, rate in r['move_stats']:
            rate_str = f'{rate:.3f}' if not np.isnan(rate) else '  n/a'
            lines.append(f'{r["label"]:<20}  {name:<6}  {acc:>10}  {att:>10}  {rate_str:>8}')

    lines.append('')
    return '\n'.join(lines)


def write_results(results, steps, mode, outdir, oct_cutoff, oct_layers):
    text = format_results(results, steps, mode, oct_cutoff, oct_layers)
    print('\n' + text)
    summary_path = os.path.join(outdir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(text + '\n')
    print(f'Summary written to {summary_path}')
    for r in results:
        slug = r['label'].replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
        print(f'  {slug}.out  /  {slug}.xyz  — step log and trajectory')


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins, seed_sel = (int(s) for s in ss.generate_state(3, dtype=np.uint32))

    enable_cueq = not args.no_cueq
    compile_model = not args.no_compile
    checkpoint = args.model_path or args.checkpoint

    print(f'Mode: {args.mode}  |  Steps: {args.steps}  |  Device: {args.device}')
    print(f'Seed: {args.seed}  |  Output: {os.path.abspath(args.outdir)}')

    results = []

    if args.mode == 'energy-only':
        print(f'\nLoading MACE (mace_mp model={checkpoint})...')
        if args.model_path:
            from mace.calculators import MACECalculator as ASEMACECalculator
            mace_calc = ASEMACECalculator(model_paths=args.model_path, device=args.device)
        else:
            mace_calc = mace_mp(model=checkpoint, device=args.device)

        print('Loading Alchemi (AlchemiCalculator)...')
        alchemi_calc = AlchemiCalculator(
            checkpoint=checkpoint,
            device=args.device,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )

        print(f'\nRunning MACE energy-only ({args.steps} steps)...')
        results.append(run_gcmc('MACE (mace_mp)', mace_calc, args, seed_del, seed_ins, seed_sel))

        print(f'Running Alchemi energy-only ({args.steps} steps)...')
        results.append(run_gcmc('Alchemi', alchemi_calc, args, seed_del, seed_ins, seed_sel))

    else:  # relax
        print(f'\nLoading MACE_F_Calculator (optimizer={args.mace_optimizer})...')
        mace_f = MACE_F_Calculator(
            model_paths=args.model_path or mace_mp(model=checkpoint, device=args.device),
            steps=args.relax_steps,
            fmax=args.fmax,
            device=args.device,
            optimizer=args.mace_optimizer,
        )

        print(f'Loading AlchemiFCalculator (optimizer={args.alchemi_optimizer}, '
              'torch.compile warmup ~30s if enabled)...')
        alchemi_f = AlchemiFCalculator(
            checkpoint=checkpoint,
            steps=args.relax_steps,
            fmax=args.fmax,
            device=args.device,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
            optimizer=args.alchemi_optimizer,
        )

        mace_label = f'MACE_F ({args.mace_optimizer.upper()})'
        print(f'\nRunning {mace_label} relax ({args.steps} steps)...')
        results.append(run_gcmc(mace_label, mace_f, args, seed_del, seed_ins, seed_sel))

        alch_label = f'AlchemiF ({args.alchemi_optimizer.upper()})'
        print(f'Running {alch_label} relax ({args.steps} steps)...')
        results.append(run_gcmc(alch_label, alchemi_f, args, seed_del, seed_ins, seed_sel))

    write_results(results, args.steps, args.mode, args.outdir,
                  args.oct_cutoff, args.oct_layers)


if __name__ == '__main__':
    main()
