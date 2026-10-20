"""Force-call cost of an Alchemi MACE calculator: floor, slope, crossover.

The time of one MACE force call has two regimes. For small systems it is
flat: a fixed cost per call (kernel launches, Python, synchronisation), and
extra atoms are almost free. For large systems the GPU is full and the time
grows linearly with the atom count. One "atoms per second" number, or one
straight line through both regimes, depends on the sizes measured. This
script reports three numbers that do not:

* ``floor_ms``: the lowest time per call, the overhead limit;
* ``slope_us_per_atom``: the cost per atom, fitted only where the time is
  at least twice the floor (the GPU is full);
* ``crossover_atoms`` = floor / slope: below it the overhead dominates,
  above it the model work does.

It times force calls on fixed, rattled Ag(111) slabs from 64 atoms up to the
GPU memory limit and writes one JSON record per run, so a code change or a
new model can be compared with an earlier record:

    python benchmark/force_call_scaling.py --out base.json
    python benchmark/force_call_scaling.py --out new.json --compare base.json
    python benchmark/force_call_scaling.py --recompute old1.json old2.json
    python benchmark/force_call_scaling.py --predict base.json --atoms 1250 \
        --relax-steps 58 --gcmc-steps 10000

``first_ms`` is the first call at each size: with cuEq it includes the
per-size kernel tuning. The metrics use the median of the later calls only.
``e_per_atom`` is the energy of each fixed structure, a check that the
model and the code still give the same numbers. ``--recompute`` adds the
metrics to existing records without a GPU.

GPU and the nvalchemi stack required (the ``alchemi`` conda env).
"""
import argparse
import datetime
import json
import os
import subprocess
import time

import numpy as np

# 4-layer Ag(111) slabs: nx * nx * 4 atoms, 64 to 65536. The run stops at
# the first size that does not fit in GPU memory and records it.
SIZES = (4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128)
WARMUP_NX = 5  # 100 atoms per structure, not one of SIZES


def slab(nx, rattle_seed):
    from ase.build import fcc111
    at = fcc111('Ag', a=4.165, size=(nx, nx, 4), periodic=True, vacuum=8.0)
    at.rattle(0.05, seed=rattle_seed)
    return at


def environment(args):
    import torch
    from importlib.metadata import PackageNotFoundError, version
    import mcpy
    here = os.path.dirname(os.path.abspath(mcpy.__file__))  # the mcpy that runs
    try:
        commit = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=here,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
    versions = {}
    for pkg in ('mcpy-gcmc', 'torch', 'nvalchemi-toolkit', 'mace-torch', 'cuequivariance'):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return dict(date=datetime.datetime.now().isoformat(timespec='seconds'),
                git=commit, versions=versions, gpu=torch.cuda.get_device_name(0),
                checkpoint=args.checkpoint, head=args.head, cueq=not args.no_cueq,
                compile=not args.no_compile, batch=args.batch, repeats=args.repeats,
                mcpy_path=here)


def time_calls(calc, atoms_list, repeats):
    """First-call and median later-call time [ms] of one force call."""
    import torch
    from mcpy.calculators.alchemi import _build_nl, _graph_energies
    batch = calc._batch(atoms_list)
    _build_nl(batch, calc._nl_hook())
    times, out = [], None
    for _ in range(repeats + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = calc.model(batch)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    n = sum(len(a) for a in atoms_list)
    return times[0], float(np.median(times[1:])), float(_graph_energies(out, batch).sum()) / n


def metrics(rows):
    """Floor, saturated slope and crossover from the timed rows.

    ``floor_reached`` is False when fewer than two sizes lie within 20% of
    the lowest time: the flat part was not measured, so floor and crossover
    are upper bounds. A slope that is not positive (flat or noisy sizes) is
    reported as None, and so is the crossover.
    """
    n = np.array([r['n_atoms'] for r in rows], float)
    t = np.array([r['ms'] for r in rows])
    floor = float(t.min())
    sat = t >= 2 * floor
    saturated = int(sat.sum()) >= 2
    if not saturated:  # the GPU never filled up: use the two largest sizes
        sat = n >= np.sort(n)[-2]
    slope, _ = np.polyfit(n[sat], t[sat], 1)
    slope = float(slope * 1e3) if slope > 0 else None
    return dict(floor_ms=floor, floor_at_atoms=int(n[t.argmin()]),
                floor_reached=int((t <= 1.2 * floor).sum()) >= 2,
                slope_us_per_atom=slope, slope_sizes=n[sat].astype(int).tolist(),
                saturated=saturated,
                crossover_atoms=floor / slope * 1e3 if slope else None)


def describe(m):
    notes = []
    if m['slope_us_per_atom'] is None:
        notes.append('no growth over the measured sizes: no slope, no crossover')
    elif not m['saturated']:
        notes.append('GPU not full at any size: slope is a lower bound')
    if not m['floor_reached']:
        notes.append('flat part not measured: floor and crossover are upper bounds')
    slope = ('n/a' if m['slope_us_per_atom'] is None
             else f"{m['slope_us_per_atom']:.3f} us/atom")
    cross = ('n/a' if m['crossover_atoms'] is None
             else f"{m['crossover_atoms']:.0f} atoms")
    text = f"floor {m['floor_ms']:.2f} ms, slope {slope}, crossover {cross}"
    return text + ''.join(f'\n  note: {x}' for x in notes)


def predict(m, n_atoms, relax_steps, gcmc_steps, moves):
    """Lower-bound run time from the warm force-call cost of a record.

    t_call = max(floor, slope * N); each trial costs (relax_steps + 1) calls,
    the +1 being the relaxation's first force call. Tuning at new sizes and
    the mcpy move/volume/file work are not included (about +5-30% in the
    profiles), so the result is a lower bound.
    """
    slope = m['slope_us_per_atom'] or 0.0
    t_call = max(m['floor_ms'], slope * 1e-3 * n_atoms)
    regime = 'overhead (floor)' if t_call == m['floor_ms'] else 'model work (slope)'
    total_s = gcmc_steps * moves * (relax_steps + 1) * t_call / 1e3
    return (f'force call at {n_atoms} atoms: {t_call:.2f} ms, set by the {regime}\n'
            f'{gcmc_steps} steps x {moves} moves x {relax_steps + 1:.0f} calls: '
            f'{total_s / 3600:.2f} h ({total_s / 60:.0f} min), a lower bound '
            f'(no new-size tuning, no mcpy overhead)')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='named MACE checkpoint or local .model path')
    p.add_argument('--head', default=None, help='head of a multihead model')
    p.add_argument('--no-cueq', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--batch', type=int, default=1,
                   help='structures per force call (copies with different rattles)')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--max-atoms', type=int, default=None,
                   help='skip sizes whose batch holds more atoms (GPU memory)')
    p.add_argument('--out', help='JSON record to write')
    p.add_argument('--compare', help='earlier JSON record to compare with')
    p.add_argument('--recompute', nargs='+', metavar='JSON',
                   help='add the metrics to existing records, no GPU run')
    p.add_argument('--predict', metavar='JSON',
                   help='estimate a GCMC run time from a record, no GPU run '
                        '(needs --atoms and --relax-steps)')
    p.add_argument('--atoms', type=int, help='--predict: atoms per force call '
                   '(all replicas together for a batched run)')
    p.add_argument('--relax-steps', type=float,
                   help='--predict: mean optimizer steps per relaxation, excluding the '
                        'first force call, from a pilot run (calc.last_relax_steps)')
    p.add_argument('--gcmc-steps', type=int, default=10_000, help='--predict: run length')
    p.add_argument('--moves', type=int, default=1, help='--predict: trial moves per step')
    args = p.parse_args()
    if args.predict:
        if args.atoms is None or args.relax_steps is None:
            p.error('--predict needs --atoms and --relax-steps')
        with open(args.predict) as f:
            rec = json.load(f)
        m = rec.get('metrics')
        if not m:
            if len(rec['rows']) < 2:
                p.error('--predict needs a record with metrics or at least 2 measured sizes')
            m = metrics(rec['rows'])
        print(predict(m, args.atoms, args.relax_steps, args.gcmc_steps, args.moves))
        return
    if args.recompute:
        for path in args.recompute:
            with open(path) as f:
                rec = json.load(f)
            if len(rec['rows']) >= 2:
                rec['metrics'] = metrics(rec['rows'])
                with open(path, 'w') as f:
                    json.dump(rec, f, indent=1)
                print(f"{path}: {describe(rec['metrics'])}")
            else:
                print(f'{path}: fewer than 2 rows, skipped')
        return
    if not args.out:
        p.error('--out is required unless --recompute is given')
    sizes = [nx for nx in SIZES
             if not args.max_atoms or nx * nx * 4 * args.batch <= args.max_atoms]
    if len(sizes) < 2:
        p.error(f'--max-atoms {args.max_atoms} with --batch {args.batch} leaves '
                f'{len(sizes)} size(s); the metrics need at least 2')

    from mcpy.calculators import AlchemiFCalculator
    calc = AlchemiFCalculator(args.checkpoint, head=args.head, device='cuda',
                              enable_cueq=not args.no_cueq,
                              compile_model=not args.no_compile, steps=1)
    # Compile before timing, at a size outside SIZES (its own padding bucket)
    # and with the batch's graph count, so no timed row starts warm.
    time_calls(calc, [slab(WARMUP_NX, k) for k in range(args.batch)], 3)

    import torch
    rows, oom_at = [], None
    for nx in sizes:
        n = nx * nx * 4 * args.batch
        atoms = [slab(nx, k) for k in range(args.batch)]
        try:
            first, med, epa = time_calls(calc, atoms, args.repeats)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            # TorchScript re-raises a CUDA OOM as a plain RuntimeError.
            if 'out of memory' not in str(exc):
                raise
            oom_at = n
            print(f'{n:7d} atoms  out of GPU memory: stop', flush=True)
            break
        finally:
            torch.cuda.empty_cache()
        rows.append(dict(n_atoms=n, first_ms=first, ms=med, e_per_atom=epa))
        print(f'{n:7d} atoms  {med:8.2f} ms  (first call {first:8.1f} ms)  '
              f'{med * 1e3 / n:7.2f} us/atom', flush=True)

    common = dict(environment(args), rows=rows, oom_at=oom_at,
                  peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    if len(rows) < 2:  # out of memory too early for any metric
        with open(args.out, 'w') as f:
            json.dump(dict(common, metrics=None), f, indent=1)
        raise SystemExit(f'only {len(rows)} size(s) fit in GPU memory: no metrics; '
                         f'rows written to {args.out}')
    m = metrics(rows)
    print(f'\n{describe(m)}')
    record = dict(common, metrics=m, max_fitting_atoms=int(rows[-1]['n_atoms']))
    with open(args.out, 'w') as f:
        json.dump(record, f, indent=1)

    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        print(f'\nvs {args.compare} ({ref["git"]}, {ref["date"]}):')
        old_m = ref.get('metrics') or (metrics(ref['rows']) if len(ref['rows']) >= 2 else None)
        if old_m:
            for key, unit in (('floor_ms', 'ms'), ('slope_us_per_atom', 'us/atom'),
                              ('crossover_atoms', 'atoms')):
                if old_m.get(key) is None or m[key] is None:
                    print(f'  {key:18s} not available in one of the records')
                else:
                    print(f'  {key:18s} {old_m[key]:10.3f} -> {m[key]:10.3f} {unit}')
        by_n = {r['n_atoms']: r for r in ref['rows']}
        for r in rows:
            old = by_n.get(r['n_atoms'])
            if old:
                print(f'  {r["n_atoms"]:7d} atoms: {old["ms"]:8.2f} -> {r["ms"]:8.2f} ms '
                      f'({old["ms"] / r["ms"]:.2f}x), '
                      f'dE/atom {(r["e_per_atom"] - old["e_per_atom"]) * 1e3:+.3f} meV')


if __name__ == '__main__':
    main()
