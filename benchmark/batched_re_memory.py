"""Map the peak-GPU-memory surface of batched RE energy evaluation.

Sweeps (atoms_per_replica, n_replicas) x lever-setting through
AlchemiCalculator.get_potential_energies, recording peak memory and per-replica
energy deviation from a full-batch baseline. Levers (all aimed at bit-exact /
within-noise energies):
  - chunk      : split the replica batch into sub-batches of <= chunk_size
  - maxnbr     : cap the neighbour list (min-safe value isolated at n=1)
  - energy_only: drop 'forces' from active_outputs (no autograd graph)

Writes a CSV, a markdown summary, and a heatmap into the repo's benchmark/ dir
(the sandbox cannot see /tmp).

Run in the `alchemi` conda env on the GPU box:
    python benchmark/batched_re_memory.py
"""
import csv
import os
import time

import numpy as np
import torch
from ase.build import bulk

from mcpy.calculators import AlchemiCalculator

OUTDIR = os.path.dirname(os.path.abspath(__file__))
SIZES = [256, 1024, 2048, 3584]            # atoms per replica (nearest carve)
REPLICAS = [1, 2, 4, 8]
MAXNBR_SWEEP = [None, 64, 48, 32, 24, 16]  # descend until energies break
NOISE_REPEATS = 5


def make_np(n_target, a=4.16, symbol='Ag'):
    reps = int(np.ceil((n_target / 4) ** (1 / 3))) + 3
    base = bulk(symbol, 'fcc', a=a, cubic=True).repeat((reps, reps, reps))
    center = base.get_positions().mean(axis=0)
    d = np.linalg.norm(base.get_positions() - center, axis=1)
    keep = np.argsort(d)[:n_target]
    atoms = base[keep]
    atoms.center(vacuum=10.0)
    atoms.set_pbc(False)
    return atoms


def measure(calc, atoms_list, chunk_size):
    """Return (energies, peak_MB, wall_ms) or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        e = calc.get_potential_energies(atoms_list, chunk_size=chunk_size)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    wall = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return e, peak, wall


def set_energy_only(calc, on):
    """Toggle force computation by editing active_outputs in place."""
    ao = calc.model.model_config.active_outputs
    if on:
        ao.discard('forces')
    else:
        ao.add('forces')


def _row(size, n, lever, setting, out, base_e, eps_noise):
    if out is None:
        return dict(size=size, n=n, lever=lever, setting=setting, peak_MB=None,
                    max_dE=None, within_noise=False, wall_ms=None, oom=True)
    de = float(np.max(np.abs(out[0] - base_e))) if base_e is not None else None
    return dict(size=size, n=n, lever=lever, setting=setting, peak_MB=out[1],
                max_dE=de, within_noise=(de is not None and de <= eps_noise),
                wall_ms=out[2], oom=False)


def main():
    calc = AlchemiCalculator(device='cuda', compile_model=False)
    rows = []

    # --- per-size single-NP reference energy + run-to-run floor. All replicas
    # are identical NPs, so every replica's energy must equal this reference up
    # to fp32 jitter; this is the fallback comparator for configs that OOM at
    # full batch (chunking lets them run). ---
    eps_noise, e_ref = {}, {}
    for size in SIZES:
        probe = [make_np(size)]
        runs = np.stack([measure(calc, probe, None)[0] for _ in range(NOISE_REPEATS)])
        eps_noise[size] = float(np.max(runs.max(axis=0) - runs.min(axis=0)))
        e_ref[size] = float(np.median(runs))
        print(f'noise floor[{size}] = {eps_noise[size]:.3e} eV')

    for size in SIZES:
        for n in REPLICAS:
            atoms_list = [make_np(size) for _ in range(n)]
            # Two whole-batch runs: b0 is the reference energies, b1 gives a
            # config-matched (same size, same n) run-to-run floor that scales
            # with both energy magnitude and the max-over-n-replicas extremes.
            b0 = measure(calc, atoms_list, None)
            b1 = measure(calc, atoms_list, None)
            if b0 and b1:
                floor = max(float(np.max(np.abs(b1[0] - b0[0]))), eps_noise[size])
                base_e = b0[0]
            else:
                floor = eps_noise[size]
                base_e = np.full(n, e_ref[size]) if not b0 else b0[0]
            rows.append(_row(size, n, 'baseline', 'whole', b0, base_e, floor)
                        if b0 else
                        dict(size=size, n=n, lever='baseline', setting='whole',
                             peak_MB=None, max_dE=None, within_noise=False,
                             wall_ms=None, oom=True))

            # chunking (only meaningful for n > 1)
            for cs in [c for c in (1, 2, 4) if c < n]:
                out = measure(calc, atoms_list, cs)
                rows.append(_row(size, n, 'chunk', cs, out, base_e, floor))

            # energy_only (forces off), whole batch
            set_energy_only(calc, True)
            out = measure(calc, atoms_list, None)
            set_energy_only(calc, False)
            rows.append(_row(size, n, 'energy_only', 'on', out, base_e, floor))

            # max_neighbors sweep (single replica isolates the NL effect)
            if n == 1:
                for mn in MAXNBR_SWEEP:
                    calc.max_neighbors = mn
                    out = measure(calc, atoms_list, None)
                    calc.max_neighbors = None
                    rows.append(_row(size, n, 'maxnbr', mn, out, base_e, floor))
            print(f'size={size} n={n} done '
                  f'(baseline peak={b0[1]:.0f} MB)' if b0 else
                  f'size={size} n={n} OOM at baseline')

    _write_csv(rows, eps_noise)
    _write_summary(rows, eps_noise)
    _write_heatmap(rows)


def _write_csv(rows, eps_noise):
    path = os.path.join(OUTDIR, 'batched_re_memory.csv')
    floors = ' '.join(f'{s}:{e:.3e}' for s, e in sorted(eps_noise.items()))
    with open(path, 'w', newline='') as fh:
        fh.write(f'# eps_noise(eV) per size: {floors}\n')
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')


def _write_summary(rows, eps_noise):
    path = os.path.join(OUTDIR, 'batched_re_memory_summary.md')
    lines = ['# Batched RE memory sweep\n\n', '## Noise floor per size (eV)\n']
    for s, e in sorted(eps_noise.items()):
        lines.append(f'- {s} atoms: {e:.3e}\n')

    lines.append('\n## Min-safe max_neighbors per size\n')
    for size in sorted({r['size'] for r in rows}):
        safe = [r['setting'] for r in rows
                if r['lever'] == 'maxnbr' and r['size'] == size
                and r['within_noise'] and r['setting'] is not None]
        lines.append(f'- {size} atoms: min-safe = '
                     f'{min(safe) if safe else "none below baseline"}\n')

    lines.append('\n## energy_only saving (within noise?)\n')
    for size in sorted({r['size'] for r in rows}):
        eo = [r for r in rows if r['lever'] == 'energy_only' and r['size'] == size
              and not r['oom']]
        base = {(r['size'], r['n']): r for r in rows
                if r['lever'] == 'baseline' and not r['oom']}
        for r in eo:
            b = base.get((r['size'], r['n']))
            if b and b['peak_MB']:
                frac = r['peak_MB'] / b['peak_MB']
                lines.append(f'- {size} atoms x{r["n"]}: {frac:.1%} of baseline, '
                             f'max|dE|={r["max_dE"]:.2e} eV, '
                             f'within_noise={r["within_noise"]}\n')

    with open(path, 'w') as fh:
        fh.writelines(lines)
    print(f'wrote {path}')


def _write_heatmap(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sizes = sorted({r['size'] for r in rows})
    reps = sorted({r['n'] for r in rows})
    grid = np.full((len(sizes), len(reps)), np.nan)
    for r in rows:
        if r['lever'] == 'baseline' and not r['oom']:
            grid[sizes.index(r['size']), reps.index(r['n'])] = r['peak_MB']
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin='lower', aspect='auto')
    ax.set_xticks(range(len(reps)), reps)
    ax.set_yticks(range(len(sizes)), sizes)
    ax.set_xlabel('n_replicas')
    ax.set_ylabel('atoms/replica')
    ax.set_title('Baseline peak GPU memory (MB); blank = OOM')
    fig.colorbar(im, label='MB')
    path = os.path.join(OUTDIR, 'batched_re_memory_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
