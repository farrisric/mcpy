"""A/B: GCMC wall-clock with vs without torch.compile on the local-model path.

Runs the same O-adsorption GCMC on an Ag201 octahedron (Octahedron('Ag', 7, 2))
twice — compile_model=False then compile_model=True — with identical seeds and
reports per-step timing, per-FIRE-step timing (the fair kernel-level number:
trajectories can diverge after ~meV energy differences flip an acceptance),
and totals. The checkpoint is resolved to its cached local *path* so the run
exercises the local-file branch of ``_load_model`` (the one fine-tuned models
take), not the alias branch.

Run in the ``alchemi`` conda env on the GPU box::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python benchmark/gcmc_compile_ab.py --steps 2000
"""
import argparse
import json
import os
import time

import numpy as np
from ase.cluster import Octahedron

from mcpy.utils.logging import configure as configure_logging

configure_logging()

import torch  # noqa: E402

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default=None,
                   help='Local model path; default downloads medium-mpa-0 and '
                        'uses its cached path')
    p.add_argument('--steps', type=int, default=2000, help='GCMC steps per mode')
    p.add_argument('--T', type=float, default=500.0)
    p.add_argument('--mu-Ag', type=float, default=-2.99)
    p.add_argument('--mu-O', type=float, default=-4.91)
    p.add_argument('--delta-mu-O', type=float, default=-0.5)
    p.add_argument('--fmax', type=float, default=0.05)
    p.add_argument('--relax-steps', type=int, default=500)
    p.add_argument('--min-insert', type=float, default=0.5)
    p.add_argument('--vacuum', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--write-interval', type=int, default=50)
    p.add_argument('--outdir', default='benchmark/benchmark_results/compile_ab')
    p.add_argument('--modes', default='nocompile,compile',
                   help='Comma-separated subset of {nocompile,compile} to run')
    return p.parse_args()


class TimedGCMC(GrandCanonicalEnsemble):
    """GCMC that records per-step wall time and per-step relax-step counts."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step_seconds = []
        self.step_relax_steps = []

    def _run(self):
        relax_before = self._calculator.total_relax_steps
        super()._run()
        self.step_seconds.append(self._last_step_seconds)
        self.step_relax_steps.append(
            self._calculator.total_relax_steps - relax_before)


def run_mode(compile_model, checkpoint, args):
    tag = 'compile' if compile_model else 'nocompile'
    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    atoms = Octahedron('Ag', 7, 2)
    scell = SphericalCell(atoms, vacuum=args.vacuum,
                          species_radii={'Ag': 2.947, 'O': 0},
                          mc_sample_points=100_000)
    calculator = AlchemiFCalculator(
        checkpoint=checkpoint, steps=args.relax_steps, fmax=args.fmax,
        device='cuda', compile_model=compile_model,
    )
    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=args.min_insert,
                       seed=seed_ins)],
    )
    gcmc = TimedGCMC(
        atoms=atoms,
        cells=[scell],
        calculator=calculator,
        mu={'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O},
        units_type='metal',
        species=species,
        temperature=args.T,
        move_selector=move_selector,
        random_seed=args.seed,
        outfile=os.path.join(args.outdir, f'gcmc_ab_{tag}.out'),
        traj_file=os.path.join(args.outdir, f'gcmc_ab_{tag}.xyz'),
        trajectory_write_interval=args.write_interval,
        outfile_write_interval=args.write_interval,
    )

    t0 = time.perf_counter()
    gcmc.run(args.steps)
    wall = time.perf_counter() - t0

    result = {
        'mode': tag,
        'steps': args.steps,
        'wall_s': wall,
        'total_relax_steps': calculator.total_relax_steps,
        'final_n_atoms': gcmc.n_atoms,
        'final_energy': gcmc.E_old,
        'step_seconds': gcmc.step_seconds,
        'step_relax_steps': gcmc.step_relax_steps,
    }
    del gcmc, calculator
    torch.cuda.empty_cache()
    return result


def summarize(r):
    t = np.array(r['step_seconds'])
    rs = np.array(r['step_relax_steps'])
    # Steady state: skip the first 5 steps (compile warmup / allocator ramp).
    skip = 5 if len(t) > 20 else 0
    t_ss, rs_ss = t[skip:], rs[skip:]
    return {
        'wall_min': r['wall_s'] / 60,
        'mean_s_per_step': float(t_ss.mean()),
        'median_s_per_step': float(np.median(t_ss)),
        'ms_per_fire_step': float(t_ss.sum() / max(rs_ss.sum(), 1) * 1e3),
        'total_fire_steps': int(rs.sum()),
        'warmup_first5_s': float(t[:skip].sum()),
        'final_n_atoms': r['final_n_atoms'],
        'final_energy': r['final_energy'],
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    checkpoint = args.checkpoint
    if checkpoint is None:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
        checkpoint = download_mace_mp_checkpoint('medium-mpa-0')
    print(f'checkpoint (local path): {checkpoint}')
    print(f'Ag201 octahedron, {args.steps} GCMC steps x 2 moves, seed {args.seed}')

    modes = [m.strip() for m in args.modes.split(',')]
    results = {}
    for compile_model in (False, True):
        if ('compile' if compile_model else 'nocompile') not in modes:
            continue
        r = run_mode(compile_model, checkpoint, args)
        results[r['mode']] = r
        s = summarize(r)
        print(f"\n[{r['mode']}] wall {s['wall_min']:.1f} min, "
              f"mean {s['mean_s_per_step']:.3f} s/step, "
              f"median {s['median_s_per_step']:.3f} s/step, "
              f"{s['ms_per_fire_step']:.1f} ms/FIRE-step "
              f"({s['total_fire_steps']} FIRE steps), "
              f"warmup(first 5) {s['warmup_first5_s']:.0f} s, "
              f"final N={s['final_n_atoms']} E={s['final_energy']:.3f} eV")

    with open(os.path.join(args.outdir, f'compile_ab_{"_".join(sorted(results))}.json'),
              'w') as fh:
        json.dump(results, fh)

    if not ('nocompile' in results and 'compile' in results):
        return
    a, b = summarize(results['nocompile']), summarize(results['compile'])
    print('\n=== A/B summary (steady state, first 5 steps excluded) ===')
    print(f"{'':>22} {'nocompile':>12} {'compile':>12} {'speedup':>9}")
    for key, fmt in [('mean_s_per_step', '.3f'), ('median_s_per_step', '.3f'),
                     ('ms_per_fire_step', '.1f')]:
        print(f'{key:>22} {a[key]:>12{fmt}} {b[key]:>12{fmt}} '
              f'{a[key] / b[key]:>8.2f}x')
    print(f"{'wall_min':>22} {a['wall_min']:>12.1f} {b['wall_min']:>12.1f} "
          f"{a['wall_min'] / b['wall_min']:>8.2f}x  (incl. warmup)")


if __name__ == '__main__':
    main()
