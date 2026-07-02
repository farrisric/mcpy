"""Scaling benchmark: batched replica-exchange GCMC with FIRE relaxation.

Measures, over a grid of (nanoparticle size x n_replicas), the GPU cost of a real
``BatchedReplicaExchange`` run driven by ``AlchemiFCalculator`` (batched FIRE
relaxation on every trial move -- the supported-NP GCMC path):

  - peak GPU memory : ``torch.cuda.max_memory_allocated`` (steady state, measured
                      after a warmup so first-step allocation/compile is excluded)
                      and peak nvidia-smi ``memory.used`` (whole-device footprint,
                      includes the CUDA context and resident model)
  - GPU utilization : nvidia-smi ``utilization.gpu``, mean and max over the timed
                      window
  - throughput      : wall-clock seconds per GCMC step (post-warmup mean)

All replicas run the SAME nanoparticle; size and replica count are the two sweep
axes. ``chunk_size`` is left OFF (whole batch) so the memory curve shows the raw
n_replicas scaling; configs that exceed the card are recorded as OOM (that is the
ceiling, itself a scaling result).

Run in the ``alchemi`` conda env on the GPU box::

    python benchmark/re_batched_scaling.py            # full small/fast grid
    python benchmark/re_batched_scaling.py --smoke    # quick harness check

Outputs (CSV + summary + heatmaps) land in benchmark/ (the sandbox cannot see
/tmp).
"""
import argparse
import csv
import os
import subprocess
import threading
import time

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

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Octahedron edge lengths -> ~140 / 338 / 664 atoms (magic numbers).
SIZE_LENGTHS = [6, 8, 10]
REPLICAS = [1, 2, 4, 8]

# GCMC / RE settings (kept fixed across the grid; only size and n_replicas vary).
MU = {'Ag': -2.99, 'O': -4.91 - 0.5}
SPECIES = ['O']
MIN_INSERT = 0.5
VACUUM = 3.0
SPECIES_RADII = {'Ag': 2.947, 'O': 0}
MC_SAMPLE_POINTS = 100_000


class GpuSampler(threading.Thread):
    """Poll nvidia-smi for whole-device memory.used (MB) and utilization (%)."""

    def __init__(self, interval=0.15):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.samples = []  # (perf_counter, mem_used_MB, util_pct)

    def run(self):
        cmd = ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
               '--format=csv,noheader,nounits']
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=5).stdout.strip()
                mem, util = (x.strip() for x in out.splitlines()[0].split(','))
                self.samples.append((time.perf_counter(), float(mem), float(util)))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)

    def window(self, t_start):
        if t_start is None:
            return list(self.samples)
        return [s for s in self.samples if s[0] >= t_start]


def make_factory(base_atoms, seeds, outdir, calculator):
    """Return a ``gcmc_factory(T, rank)`` mirroring the batched-chunked example.

    Each replica copies the template NP and builds its OWN cell + move_selector
    (BatchedReplicaExchange corrupts state if replicas share them). File writes
    are effectively disabled (huge interval) so disk I/O does not skew timing.
    """
    def gcmc_factory(T, rank):
        atoms = base_atoms.copy()
        scell = SphericalCell(atoms, vacuum=VACUUM, species_radii=SPECIES_RADII,
                              mc_sample_points=MC_SAMPLE_POINTS)
        s = seeds[2 * rank:2 * (rank + 1)]
        move_selector = MoveSelector(
            [1, 1],
            [DeletionMove(scell, species=SPECIES, seed=int(s[0])),
             InsertionMove(scell, species=SPECIES, min_insert=MIN_INSERT,
                           seed=int(s[1]))],
        )
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,  # shared; RE also re-scores via batched path
            mu=MU,
            units_type='metal',
            species=SPECIES,
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(outdir, f'_scan_rank{rank}.out'),
            trajectory_write_interval=10 ** 9,
            outfile_write_interval=10 ** 9,
            traj_file=os.path.join(outdir, f'_scan_rank{rank}.xyz'),
        )
    return gcmc_factory


def run_config(calculator, np_length, n_replicas, args):
    """Run one (size, n_replicas) point; return a result-row dict."""
    import torch

    base_atoms = Octahedron('Ag', np_length, 1)
    n_atoms = len(base_atoms)
    temperatures = (list(np.linspace(300.0, 900.0, n_replicas))
                    if n_replicas > 1 else [300.0])

    ss = np.random.SeedSequence(1234 + np_length * 100 + n_replicas)
    all_seeds = ss.generate_state(2 * n_replicas + 1, dtype=np.uint32)
    move_seeds = all_seeds[:2 * n_replicas]
    master_seed = int(all_seeds[-1])

    factory = make_factory(base_atoms, move_seeds, args.outdir, calculator)
    pt = BatchedReplicaExchange(
        factory,
        calculator=calculator,
        temperatures=temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=10 ** 9,
        seed=master_seed,
        global_minimum_file=None,
    )

    step_times = []
    warm_done_t = [None]
    orig_step = pt._batched_gcmc_step

    def timed_step():
        t0 = time.perf_counter()
        orig_step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if len(step_times) == args.warmup:
            torch.cuda.reset_peak_memory_stats()
            warm_done_t[0] = time.perf_counter()

    pt._batched_gcmc_step = timed_step

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()

    status = 'ok'
    try:
        pt.run()
    except torch.cuda.OutOfMemoryError:
        status = 'oom'
    except RuntimeError as exc:
        status = 'oom' if 'out of memory' in str(exc).lower() else 'error'
        if status == 'error':
            print(f'    RuntimeError: {exc}')
    finally:
        sampler.stop()

    peak_torch = torch.cuda.max_memory_allocated() / 1024 ** 2
    torch.cuda.empty_cache()

    win = sampler.window(warm_done_t[0])
    peak_smi = max((s[1] for s in win), default=None)
    utils = [s[2] for s in win]
    mean_util = float(np.mean(utils)) if utils else None
    max_util = float(np.max(utils)) if utils else None
    measured = step_times[args.warmup:]
    s_per_step = float(np.mean(measured)) if measured else None

    return dict(
        np_length=np_length,
        n_atoms=n_atoms,
        n_replicas=n_replicas,
        total_atoms=n_atoms * n_replicas,
        status=status,
        peak_torch_MB=round(peak_torch, 1),
        peak_smi_MB=round(peak_smi, 1) if peak_smi is not None else None,
        mean_util_pct=round(mean_util, 1) if mean_util is not None else None,
        max_util_pct=round(max_util, 1) if max_util is not None else None,
        s_per_step=round(s_per_step, 3) if s_per_step is not None else None,
        s_per_step_per_replica=(round(s_per_step / n_replicas, 3)
                                if s_per_step is not None else None),
        n_measured_steps=len(measured),
    )


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    import torch

    calculator = AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device='cuda',
        enable_cueq=not args.no_cueq,
        compile_model=not args.no_compile,
        chunk_size=None,  # whole batch: expose the raw n_replicas memory scaling
    )

    # Warm the model kernels + record resident (model + context) footprint.
    warm = Octahedron('Ag', 4, 1)
    calculator.get_potential_energies([warm])
    torch.cuda.synchronize()
    resident_MB = round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
    print(f'resident (model) memory: {resident_MB} MB\n')

    rows = []
    for length in args.sizes:
        for n in args.replicas:
            print(f'-> np_length={length} (~{len(Octahedron("Ag", length, 1))} '
                  f'atoms) x {n} replicas ...', flush=True)
            row = run_config(calculator, length, n, args)
            row['resident_MB'] = resident_MB
            rows.append(row)
            print(f'   status={row["status"]} peak_torch={row["peak_torch_MB"]} MB '
                  f'peak_smi={row["peak_smi_MB"]} MB util~{row["mean_util_pct"]}% '
                  f's/step={row["s_per_step"]}', flush=True)

    _write_csv(rows, args)
    _write_summary(rows, args)
    _write_plots(rows, args)
    print('\ndone.')


def _tagged(name, args):
    stem, ext = os.path.splitext(name)
    return os.path.join(OUTDIR, f'{stem}{args.tag}{ext}')


def _write_csv(rows, args):
    path = _tagged('re_batched_scaling.csv', args)
    with open(path, 'w', newline='') as fh:
        fh.write(f'# AlchemiFCalculator FIRE relax, chunk_size=None (whole batch), '
                 f'fmax={args.fmax}, relax_steps={args.relax_steps}, '
                 f'gcmc_steps={args.gcmc_steps} (warmup={args.warmup})\n')
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')


def _write_summary(rows, args):
    path = _tagged('re_batched_scaling_summary.md', args)
    sizes = sorted({r['n_atoms'] for r in rows})
    reps = sorted({r['n_replicas'] for r in rows})

    def cell(r, key, unit=''):
        if r is None or r['status'] != 'ok' or r[key] is None:
            return 'OOM' if (r and r['status'] == 'oom') else '-'
        return f'{r[key]:g}{unit}'

    by = {(r['n_atoms'], r['n_replicas']): r for r in rows}
    lines = ['# Batched RE-GCMC scaling (AlchemiFCalculator, FIRE relax)\n\n']
    lines.append(f'Whole batch (chunk_size=None), fmax={args.fmax}, '
                 f'relax_steps={args.relax_steps}, '
                 f'gcmc_steps={args.gcmc_steps} (warmup {args.warmup}). '
                 f'Card: 32 GB RTX 5090.\n\n')

    def table(title, key, unit=''):
        out = [f'## {title}\n\n',
               '| atoms \\\\ replicas | ' + ' | '.join(str(n) for n in reps) + ' |\n',
               '|' + '---|' * (len(reps) + 1) + '\n']
        for s in sizes:
            cells = [cell(by.get((s, n)), key, unit) for n in reps]
            out.append(f'| {s} | ' + ' | '.join(cells) + ' |\n')
        out.append('\n')
        return out

    lines += table('Peak GPU memory, nvidia-smi (MB)', 'peak_smi_MB')
    lines += table('Peak torch allocated (MB)', 'peak_torch_MB')
    lines += table('Mean GPU utilization (%)', 'mean_util_pct')
    lines += table('Throughput (s / GCMC step)', 's_per_step')

    with open(path, 'w') as fh:
        fh.writelines(lines)
    print(f'wrote {path}')


def _write_plots(rows, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sizes = sorted({r['n_atoms'] for r in rows})
    reps = sorted({r['n_replicas'] for r in rows})
    by = {(r['n_atoms'], r['n_replicas']): r for r in rows}

    def grid(key):
        g = np.full((len(sizes), len(reps)), np.nan)
        for i, s in enumerate(sizes):
            for j, n in enumerate(reps):
                r = by.get((s, n))
                if r and r['status'] == 'ok' and r[key] is not None:
                    g[i, j] = r[key]
        return g

    panels = [('peak_smi_MB', 'Peak GPU memory (MB)'),
              ('mean_util_pct', 'Mean GPU utilization (%)'),
              ('s_per_step', 'Throughput (s / GCMC step)')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, panels):
        g = grid(key)
        im = ax.imshow(g, origin='lower', aspect='auto')
        ax.set_xticks(range(len(reps)), reps)
        ax.set_yticks(range(len(sizes)), sizes)
        ax.set_xlabel('n_replicas')
        ax.set_ylabel('atoms / replica')
        ax.set_title(title + '\n(blank = OOM)')
        for i in range(len(sizes)):
            for j in range(len(reps)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f'{g[i, j]:g}', ha='center', va='center',
                            color='w', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p1 = _tagged('re_batched_scaling_heatmaps.png', args)
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    print(f'wrote {p1}')

    # Memory vs replicas, one line per size (shows the linear whole-batch trend).
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for s in sizes:
        xs, ys = [], []
        for n in reps:
            r = by.get((s, n))
            if r and r['status'] == 'ok' and r['peak_smi_MB'] is not None:
                xs.append(n)
                ys.append(r['peak_smi_MB'])
        if xs:
            ax2.plot(xs, ys, 'o-', label=f'{s} atoms')
    ax2.axhline(32607, ls='--', color='k', lw=0.8, label='32 GB card')
    ax2.set_xlabel('n_replicas')
    ax2.set_ylabel('peak GPU memory (MB)')
    ax2.set_title('Whole-batch memory vs replica count')
    ax2.legend()
    fig2.tight_layout()
    p2 = _tagged('re_batched_scaling_mem_vs_replicas.png', args)
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    print(f'wrote {p2}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', type=int, nargs='+', default=SIZE_LENGTHS,
                   help='Octahedron edge lengths to sweep')
    p.add_argument('--replicas', type=int, nargs='+', default=REPLICAS)
    p.add_argument('--gcmc-steps', type=int, default=10,
                   help='GCMC steps per config (incl. warmup)')
    p.add_argument('--warmup', type=int, default=2,
                   help='Leading steps excluded from timing / peak memory')
    p.add_argument('--exchange-interval', type=int, default=4)
    p.add_argument('--fmax', type=float, default=0.05)
    p.add_argument('--relax-steps', type=int, default=500)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--no-cueq', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--outdir', default=os.path.join(OUTDIR, '_scan_tmp'))
    p.add_argument('--tag', default='',
                   help='Suffix for output filenames (avoids clobbering runs)')
    p.add_argument('--smoke', action='store_true',
                   help='Tiny grid to validate the harness end to end')
    args = p.parse_args()
    if args.smoke:
        args.sizes = [6]
        args.replicas = [1, 2]
        args.gcmc_steps = 4
        args.warmup = 1
        args.exchange_interval = 2
    return args


if __name__ == '__main__':
    main()
