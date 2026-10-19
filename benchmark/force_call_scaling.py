"""Force-call cost of an Alchemi MACE calculator: t(N) = a + b * N.

One MACE force call costs a fixed overhead ``a`` (kernel launches, Python,
synchronisation) plus ``b`` per atom (the model work). A single "atoms per
second" number mixes the two, so it is not fair across system sizes. This
script times force calls on fixed, rattled Ag(111) slabs of several sizes,
fits the line, and writes one JSON record per run, so a code change or a new
model can be compared with an earlier record:

    python benchmark/force_call_scaling.py --out base.json
    python benchmark/force_call_scaling.py --out new.json --compare base.json

``first_ms`` is the first call at each size: with cuEq it includes the
per-size kernel tuning. The fit uses the median of the later calls only.
``e_per_atom`` is the energy of each fixed structure, a check that the
model and the code still give the same numbers.

GPU and the nvalchemi stack required (the ``alchemi`` conda env).
"""
import argparse
import datetime
import json
import os
import subprocess
import time

import numpy as np

# 4-layer Ag(111) slabs: nx * nx * 4 atoms, 64 to 4096.
SIZES = (4, 6, 8, 11, 16, 23, 32)


def slab(nx, rattle_seed):
    from ase.build import fcc111
    at = fcc111('Ag', a=4.165, size=(nx, nx, 4), periodic=True, vacuum=8.0)
    at.rattle(0.05, seed=rattle_seed)
    return at


def environment(args):
    import torch
    from importlib.metadata import PackageNotFoundError, version
    here = os.path.dirname(os.path.abspath(__file__))
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
                compile=not args.no_compile, batch=args.batch, repeats=args.repeats)


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
    p.add_argument('--out', required=True, help='JSON record to write')
    p.add_argument('--compare', help='earlier JSON record to compare with')
    args = p.parse_args()

    from mcpy.calculators import AlchemiFCalculator
    calc = AlchemiFCalculator(args.checkpoint, head=args.head, device='cuda',
                              enable_cueq=not args.no_cueq,
                              compile_model=not args.no_compile, steps=1)
    time_calls(calc, [slab(SIZES[0], 0)], 3)  # compile once before timing

    rows = []
    for nx in SIZES:
        atoms = [slab(nx, k) for k in range(args.batch)]
        first, med, epa = time_calls(calc, atoms, args.repeats)
        n = sum(len(a) for a in atoms)
        rows.append(dict(n_atoms=n, first_ms=first, ms=med, e_per_atom=epa))
        print(f'{n:7d} atoms  {med:8.2f} ms  (first call {first:8.1f} ms)  '
              f'{med * 1e3 / n:7.2f} us/atom', flush=True)

    n = np.array([r['n_atoms'] for r in rows], float)
    t = np.array([r['ms'] for r in rows])
    b, a = np.polyfit(n, t, 1)
    fit = dict(a_ms=float(a), b_us_per_atom=float(b * 1e3),
               max_rel_residual=float(np.max(np.abs(a + b * n - t) / t)))
    print(f'\nfit t(N) = a + b N:  a = {a:.2f} ms,  b = {b * 1e3:.3f} us/atom  '
          f'(max residual {fit["max_rel_residual"]:.0%})')
    record = dict(environment(args), rows=rows, fit=fit)
    with open(args.out, 'w') as f:
        json.dump(record, f, indent=1)

    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        print(f'\nvs {args.compare} ({ref["git"]}, {ref["date"]}):')
        print(f'  a  {ref["fit"]["a_ms"]:7.2f} -> {a:7.2f} ms')
        print(f'  b  {ref["fit"]["b_us_per_atom"]:7.3f} -> {b * 1e3:7.3f} us/atom')
        by_n = {r['n_atoms']: r for r in ref['rows']}
        for r in rows:
            old = by_n.get(r['n_atoms'])
            if old:
                print(f'  {r["n_atoms"]:7d} atoms: {old["ms"]:8.2f} -> {r["ms"]:8.2f} ms '
                      f'({old["ms"] / r["ms"]:.2f}x), '
                      f'dE/atom {(r["e_per_atom"] - old["e_per_atom"]) * 1e3:+.3f} meV')


if __name__ == '__main__':
    main()
