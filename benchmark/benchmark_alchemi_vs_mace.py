"""Benchmark MACE (ASE/LBFGS) vs Alchemi (GPU-native) for GCMC energy evaluations.

Two modes are compared side-by-side:
  energy-only  — single forward pass, no geometry relaxation
  relax        — geometry relaxation then energy (LBFGS for MACE, FIRE for Alchemi)

Systems: Ag Octahedron nanoparticles at four sizes covering the crossover region
  where Alchemi starts to win over the ASE path (~72 atoms per NVALCHEMI_NOTES.md).

Usage:
  # Both calculators load medium-mpa-0 from cache automatically:
  python benchmark_alchemi_vs_mace.py

  # Supply a local .pt file (both calculators load the same weights):
  python benchmark_alchemi_vs_mace.py --model-path /path/to/model.pt

  # Fast smoke-test (1 warmup, 3 repeats, smallest 2 sizes):
  python benchmark_alchemi_vs_mace.py --warmup 1 --repeats 3 --max-size 140
"""
import argparse
import time

import numpy as np
import torch
from ase.cluster import Octahedron
from mace.calculators import MACECalculator as ASEMACECalculator
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging
from mcpy.calculators import AlchemiCalculator, AlchemiFCalculator, MACE_F_Calculator

configure_logging()

# (cutoff, layers) → atom count: 38, 140, 314, 586
_SIZES = [(4, 1), (6, 1), (8, 2), (10, 3)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-path', default=None,
                   help='Path to a MACE .pt checkpoint. If omitted both calculators '
                        'load --checkpoint (default medium-mpa-0).')
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='Named MACE-MP checkpoint shared by MACE and Alchemi sides.')
    p.add_argument('--device', default='cuda', help='Torch device (cuda or cpu)')
    p.add_argument('--warmup', type=int, default=2,
                   help='Warmup evaluations (not timed; warms torch.compile JIT)')
    p.add_argument('--repeats', type=int, default=5, help='Timed evaluations per config')
    p.add_argument('--fmax', type=float, default=0.05,
                   help='Force convergence threshold for relax mode (eV/Å)')
    p.add_argument('--relax-steps', type=int, default=500,
                   help='Max optimisation steps for relax mode')
    p.add_argument('--mace-optimizer', choices=['lbfgs', 'fire'], default='lbfgs',
                   help='ASE optimizer for MACE. Use "fire" for apples-to-apples '
                        'comparison with Alchemi (which only supports FIRE).')
    p.add_argument('--no-cueq', action='store_true', help='Disable cuEquivariance for Alchemi')
    p.add_argument('--no-compile', action='store_true', help='Disable torch.compile for Alchemi')
    p.add_argument('--max-size', type=int, default=10_000,
                   help='Skip systems larger than this atom count (for quick runs)')
    return p.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn, n_warmup, n_repeats):
    """Return (times_list, last_result). Warmups not included in times."""
    result = None
    for _ in range(n_warmup):
        result = fn()
        sync()
    times = []
    for _ in range(n_repeats):
        sync()
        t0 = time.perf_counter()
        result = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return times, result


def build_systems(max_size):
    systems = []
    for cutoff, layers in _SIZES:
        atoms = Octahedron('Ag', cutoff, layers)
        if len(atoms) <= max_size:
            systems.append(atoms)
    return systems


def run_energy_only(systems, args, mace_calc, alchemi_calc):
    """Single forward pass — no relaxation."""
    print('\n=== Energy-only (no relaxation) ===')
    header = f'{"N":>6}  {"MACE (s)":>10}  {"Alchemi (s)":>12}  {"Speedup":>8}  {"ΔE (eV)":>10}'
    print(header)
    print('-' * len(header))

    rows = []
    for atoms in systems:
        n = len(atoms)

        def mace_fn():
            a = atoms.copy()
            a.calc = mace_calc
            return a.get_potential_energy()

        def alchemi_fn():
            return alchemi_calc.get_potential_energy(atoms.copy())

        mace_times, mace_e = time_call(mace_fn, args.warmup, args.repeats)
        alch_times, alch_e = time_call(alchemi_fn, args.warmup, args.repeats)

        mace_mean = np.mean(mace_times)
        alch_mean = np.mean(alch_times)
        speedup = mace_mean / alch_mean
        delta_e = abs(alch_e - mace_e)

        print(f'{n:>6}  {mace_mean:>10.3f}  {alch_mean:>12.3f}  {speedup:>8.2f}x  {delta_e:>10.4f}')
        rows.append({'n': n, 'mace_s': mace_mean, 'alch_s': alch_mean,
                     'speedup': speedup, 'delta_e': delta_e})
    return rows


def run_relax(systems, args, mace_f_calc, alchemi_f_calc):
    """ASE optimizer (MACE) vs FIRE (Alchemi) geometry relaxation."""
    mace_opt = mace_f_calc.optimizer_name.upper()
    print(f'\n=== With relaxation (MACE {mace_opt} vs Alchemi FIRE) ===')
    header = (f'{"N":>6}  {"MACE (s)":>10}  {"Alchemi (s)":>12}  '
              f'{"Speedup":>8}  {"MACE steps":>11}  {"Alch steps":>11}  '
              f'{"E_MACE (eV)":>13}  {"E_Alchemi (eV)":>15}  {"ΔE (eV)":>10}')
    print(header)
    print('-' * len(header))

    rows = []
    for atoms in systems:
        n = len(atoms)

        def mace_fn():
            return mace_f_calc.get_potential_energy(atoms.copy())

        def alchemi_fn():
            return alchemi_f_calc.get_potential_energy(atoms.copy())

        # reset step counters so per-system averages are clean
        mace_f_calc.total_relax_steps = 0
        alchemi_f_calc.total_relax_steps = 0
        mace_times, mace_e = time_call(mace_fn, args.warmup, args.repeats)
        alch_times, alch_e = time_call(alchemi_fn, args.warmup, args.repeats)

        mace_mean = np.mean(mace_times)
        alch_mean = np.mean(alch_times)
        speedup = mace_mean / alch_mean
        delta_e = abs(alch_e - mace_e)
        n_calls = args.warmup + args.repeats
        mace_steps_avg = mace_f_calc.total_relax_steps / max(1, n_calls)
        alch_steps_avg = alchemi_f_calc.total_relax_steps / max(1, n_calls)

        print(f'{n:>6}  {mace_mean:>10.3f}  {alch_mean:>12.3f}  {speedup:>8.2f}x  '
              f'{mace_steps_avg:>11.1f}  {alch_steps_avg:>11.1f}  '
              f'{mace_e:>13.4f}  {alch_e:>15.4f}  {delta_e:>10.4f}')
        rows.append({'n': n, 'mace_s': mace_mean, 'alch_s': alch_mean,
                     'speedup': speedup, 'e_mace': mace_e, 'e_alch': alch_e,
                     'delta_e': delta_e,
                     'mace_steps_avg': mace_steps_avg, 'alch_steps_avg': alch_steps_avg})
    return rows


def main():
    args = parse_args()
    systems = build_systems(args.max_size)
    if not systems:
        print('No systems to benchmark (check --max-size).')
        return

    print(f'Systems: {[len(s) for s in systems]} atoms')
    print(f'Device: {args.device}  |  warmup={args.warmup}  repeats={args.repeats}')
    print(f'Relax: fmax={args.fmax} eV/Å  max_steps={args.relax_steps}')

    checkpoint = args.model_path or args.checkpoint
    enable_cueq = not args.no_cueq
    compile_model = not args.no_compile

    print(f'\nLoading MACE calculator (checkpoint={checkpoint})...')
    if args.model_path:
        mace_ase = ASEMACECalculator(model_paths=args.model_path, device=args.device)
    else:
        mace_ase = mace_mp(model=args.checkpoint, device=args.device)

    mace_f = MACE_F_Calculator(
        model_paths=mace_ase if args.model_path is None else args.model_path,
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        optimizer=args.mace_optimizer,
    )

    print('Loading Alchemi calculator (torch.compile warmup may take ~30s)...')
    alchemi = AlchemiCalculator(
        checkpoint=checkpoint,
        device=args.device,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
    )
    alchemi_f = AlchemiFCalculator(
        checkpoint=alchemi.model,  # reuse loaded model — no double download
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
    )

    run_energy_only(systems, args, mace_ase, alchemi)
    run_relax(systems, args, mace_f, alchemi_f)

    if args.mace_optimizer == 'lbfgs':
        print('\nNote: relax ΔE reflects LBFGS vs FIRE optimizer differences, not model error.')
        print('      For a like-for-like comparison rerun with --mace-optimizer fire.')
    else:
        print('\nNote: both calculators use FIRE — timing reflects forward-pass + dynamics speed.')


if __name__ == '__main__':
    main()
