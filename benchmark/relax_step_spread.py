"""Probe: per-graph FIRE convergence-step spread inside batched RE-GCMC relaxations.

Motivation: ``AlchemiFCalculator._relax_batch`` runs ``opt.run(batch)``, which
computes the FULL batch every FIRE step until every graph is converged at the
same step (nvalchemi ``BaseDynamics.run`` breaks only when
``converged.numel() == batch.num_graphs``). A compaction loop would instead
retire each graph at its FIRST convergence, so the predicted speedup on the
relaxation forward-pass cost is

    predicted = (B * final_steps) / sum_g(first_conv_step_g)

This probe runs a realistic batched RE-GCMC workload (same as
``re_batched_scaling.py``: Ag octahedron + O insertion/deletion, one batched
relax per sub-move) with an instrumented ``_relax_batch`` that keeps the exact
``opt.run`` semantics (identical compute path, identical exit condition) while
recording, per relax call:

  - first_conv_step per graph (first step its max force fell below fmax)
  - final_steps (when the WHOLE batch was simultaneously converged / cap hit)
  - wall time of the call

Run in the ``alchemi`` conda env on the GPU box::

    python benchmark/relax_step_spread.py                # 664 atoms x 8 replicas
    python benchmark/relax_step_spread.py --smoke        # quick harness check

Outputs land in benchmark/ (the sandbox cannot see /tmp).
"""
import argparse
import csv
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
from mcpy.cell import SphericalCell  # noqa: E402
from mcpy.ensembles import BatchedReplicaExchange  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.calculators.alchemi_f_calculator import (  # noqa: E402
    ConvergenceHook,
    DynamicsStage,
    NeighborListHook,
)
from mcpy.calculators._alchemi_common import (  # noqa: E402
    _build_nl,
    _fixed_indices,
    _freeze_hook_for,
    _make_multi_batch,
    _per_graph_energies,
    _write_back_positions_batched,
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Same workload constants as re_batched_scaling.py.
MU = {'Ag': -2.99, 'O': -4.91 - 0.5}
SPECIES = ['O']
MIN_INSERT = 0.5
VACUUM = 3.0
SPECIES_RADII = {'Ag': 2.947, 'O': 0}
MC_SAMPLE_POINTS = 100_000


class ProbedCalculator(AlchemiFCalculator):
    """AlchemiFCalculator with an instrumented ``_relax_batch``.

    Identical compute path and exit condition to ``opt.run(batch)`` (full batch
    every step, stop when ALL graphs are converged at the same step or the step
    cap is reached), plus per-graph first-convergence recording.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []  # one dict per relax call

    def _relax_batch(self, atoms_list):
        n_graphs = len(atoms_list)
        t0 = time.perf_counter()
        batch = _make_multi_batch(atoms_list, self.device, self.dtype)
        batch.forces = torch.zeros_like(batch.positions)
        batch.energy = torch.zeros(n_graphs, 1, device=self.device, dtype=self.dtype)

        fixed = []
        offset = 0
        for a in atoms_list:
            fixed.extend(offset + i for i in _fixed_indices(a))
            offset += len(a)
        freeze_hooks = _freeze_hook_for(batch, fixed)

        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        opt = self._optimizer_cls(
            model=self.model,
            dt=self.dt,
            convergence_hook=ConvergenceHook.from_fmax(self.fmax),
            n_steps=self.steps,
            hooks=freeze_hooks,
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

        _build_nl(batch, nl_hook)
        opt.compute(batch)

        # opt.run(batch) equivalent, with first-convergence bookkeeping.
        first_conv = np.full(n_graphs, -1, dtype=np.int64)
        n_steps = 0
        opt._open_hooks()
        try:
            for _ in range(self.steps):
                batch, conv = opt.step(batch)
                n_steps += 1
                if conv is not None and conv.numel():
                    rows = conv.detach().cpu().numpy()
                    newly = rows[first_conv[rows] < 0]
                    first_conv[newly] = n_steps
                    if conv.numel() == n_graphs:
                        break
        finally:
            opt._close_hooks()

        wall = time.perf_counter() - t0
        self.records.append(dict(
            n_graphs=n_graphs,
            n_atoms_total=int(len(batch.positions)),
            final_steps=n_steps,
            first_conv=first_conv.copy(),
            wall_s=wall,
        ))

        self.last_relax_steps = n_steps
        self.total_relax_steps += n_steps
        _write_back_positions_batched(atoms_list, batch)
        return _per_graph_energies(batch.energy, n_graphs), n_steps


def make_factory(base_atoms, seeds, outdir, calculator):
    """Per-replica GCMC factory (own cell + move_selector per replica)."""
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
            calculator=calculator,
            mu=MU,
            units_type='metal',
            species=SPECIES,
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(outdir, f'_probe_rank{rank}.out'),
            trajectory_write_interval=10 ** 9,
            outfile_write_interval=10 ** 9,
            traj_file=os.path.join(outdir, f'_probe_rank{rank}.xyz'),
        )
    return gcmc_factory


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    calculator = ProbedCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device='cuda',
        enable_cueq=True,
        compile_model=True,
        chunk_size=None,
    )

    base_atoms = Octahedron('Ag', args.size, 1)
    print(f'NP: {len(base_atoms)} atoms x {args.replicas} replicas')

    ss = np.random.SeedSequence(4321 + args.size * 100 + args.replicas)
    all_seeds = ss.generate_state(2 * args.replicas + 1, dtype=np.uint32)
    factory = make_factory(base_atoms, all_seeds[:2 * args.replicas],
                           args.outdir, calculator)
    temperatures = list(np.linspace(300.0, 900.0, args.replicas))

    pt = BatchedReplicaExchange(
        factory,
        calculator=calculator,
        temperatures=temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=4,
        write_out_interval=10 ** 9,
        seed=int(all_seeds[-1]),
        global_minimum_file=None,
    )
    pt.run()

    # Drop warmup calls (first GCMC step: kernel autotune/compile noise), and
    # the single-graph initial-energy rebatch (n_graphs < replicas).
    n_moves = 2  # deletion + insertion per GCMC step
    records = [r for r in calculator.records[args.warmup * n_moves:]
               if r['n_graphs'] == args.replicas]
    if not records:
        raise SystemExit('no post-warmup batched relax calls recorded')

    report(records, args)


def report(records, args):
    rows = []
    speedups = []
    for k, r in enumerate(records):
        fc = r['first_conv'].astype(float)
        fc[fc < 0] = r['final_steps']  # never converged: billed to the cap
        graph_steps_full = r['n_graphs'] * r['final_steps']
        graph_steps_compact = fc.sum()
        pred = graph_steps_full / graph_steps_compact
        speedups.append(pred)
        for g, s in enumerate(r['first_conv']):
            rows.append(dict(call=k, graph=g, first_conv_step=int(s),
                             final_steps=r['final_steps'],
                             wall_s=round(r['wall_s'], 3),
                             predicted_speedup=round(pred, 3)))

    path = os.path.join(OUTDIR, f'relax_step_spread{args.tag}.csv')
    with open(path, 'w', newline='') as fh:
        fh.write(f'# size={args.size} replicas={args.replicas} fmax={args.fmax} '
                 f'relax_steps={args.relax_steps} gcmc_steps={args.gcmc_steps} '
                 f'(warmup={args.warmup}); first_conv_step=-1 means hit cap\n')
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path}')

    all_fc = np.concatenate([np.where(r['first_conv'] < 0, r['final_steps'],
                                      r['first_conv']) for r in records])
    finals = np.array([r['final_steps'] for r in records], dtype=float)
    walls = np.array([r['wall_s'] for r in records])
    total_full = sum(r['n_graphs'] * r['final_steps'] for r in records)
    total_compact = float(all_fc.sum())

    print(f'\nrelax calls analyzed : {len(records)}')
    print(f'first-conv steps     : min={all_fc.min():.0f} '
          f'p25={np.percentile(all_fc, 25):.0f} '
          f'median={np.median(all_fc):.0f} '
          f'p75={np.percentile(all_fc, 75):.0f} max={all_fc.max():.0f}')
    print(f'final steps per call : mean={finals.mean():.1f} max={finals.max():.0f}')
    print(f'wall per relax call  : mean={walls.mean():.2f} s')
    print(f'graph-steps          : full={total_full} compacted={total_compact:.0f}')
    print(f'predicted speedup    : aggregate={total_full / total_compact:.2f}x  '
          f'per-call mean={np.mean(speedups):.2f}x '
          f'(min={np.min(speedups):.2f}x max={np.max(speedups):.2f}x)')
    n_cap = int((np.concatenate([r['first_conv'] for r in records]) < 0).sum())
    if n_cap:
        print(f'WARNING: {n_cap} graph-relaxations never converged within the cap '
              f'({args.relax_steps}); they are billed at final_steps.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--size', type=int, default=10,
                   help='Octahedron edge length (10 -> ~664 atoms)')
    p.add_argument('--replicas', type=int, default=8)
    p.add_argument('--gcmc-steps', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2,
                   help='Leading GCMC steps excluded from the analysis')
    p.add_argument('--fmax', type=float, default=0.05)
    p.add_argument('--relax-steps', type=int, default=500)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--outdir', default=os.path.join(OUTDIR, '_probe_tmp'))
    p.add_argument('--tag', default='')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.size = 6
        args.replicas = 2
        args.gcmc_steps = 3
        args.warmup = 1
    return args


if __name__ == '__main__':
    main()
