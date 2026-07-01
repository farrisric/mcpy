"""Batched replica-exchange GCMC of O adsorption on an Ag octahedral nanoparticle.

Replica-exchange counterpart of ``gcmc_nano_alchemi.py``, built on the
single-GPU batched pattern of ``re_gcmc_batched.py``. All replicas live in one
Python process and share one ``AlchemiFCalculator``; every replica's trial move
is relaxed with FIRE and scored in a single batched pass per MC step (one
optimizer / one model forward per FIRE step, with converged replicas retired
from the active batch).

Requirements:
  pip install 'nvalchemi-toolkit[mace]'

Run::

    python examples/re_gcmc_nano_alchemi.py --temperatures 300 400 500 600

No mpirun. One GPU. Each replica builds its OWN cell and move_selector
(BatchedReplicaExchange would otherwise share state across replicas).

GPU memory on long runs: the atom count changes every accepted move, which
fragments the CUDA caching allocator, so reserved GPU memory drifts up over a
long run (allocator fragmentation, not a leak). Launch with
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
"""
import argparse
import os

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[300, 400, 500, 600, 700, 800],
                   help='Replica temperatures (K)')
    p.add_argument('--gcmc-steps', type=int, default=200)
    p.add_argument('--exchange-interval', type=int, default=10)
    p.add_argument('--write-interval', type=int, default=1)
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='Named checkpoint (e.g. medium-mpa-0) or path to a .pt file')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cueq', action='store_true',
                   help='Disable cuEquivariance kernel fusion')
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile (faster startup, slower inference)')
    p.add_argument('--fmax', type=float, default=0.05,
                   help='FIRE force convergence threshold (eV/Å)')
    p.add_argument('--relax-steps', type=int, default=500,
                   help='Max FIRE steps per batched energy evaluation')
    p.add_argument('--mu-Ag', type=float, default=-2.99, help='Chemical potential of Ag (eV)')
    p.add_argument('--mu-O', type=float, default=-4.91, help='Reference mu_O (eV)')
    p.add_argument('--delta-mu-O', type=float, default=-0.5, help='Shift applied to mu_O (eV)')
    p.add_argument('--min-insert', type=float, default=0.5,
                   help='Min insertion distance to existing atoms (Å)')
    p.add_argument('--vacuum', type=float, default=3.0, help='Vacuum around nanoparticle (Å)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    n_replicas = len(args.temperatures)
    # 2 move seeds per replica (deletion + insertion) + 1 master seed for RE.
    seeds = ss.generate_state(2 * n_replicas + 1, dtype=np.uint32)
    move_seeds = [int(s) for s in seeds[:2 * n_replicas]]
    master_seed = int(seeds[-1])

    # Shared template -- each factory call copies it and builds its own cell/moves.
    base_atoms = Octahedron('Ag', 6, 1)

    # ONE model, shared across replicas. Batched FIRE relax reuses this calculator.
    calculator = AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        enable_cueq=not args.no_cueq,
        compile_model=not args.no_compile,
    )

    species = ['O']
    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    def gcmc_factory(T, rank):
        atoms = base_atoms.copy()

        scell = SphericalCell(atoms, vacuum=args.vacuum, species_radii={'Ag': 2.947, 'O': 0},
                              mc_sample_points=100_000)

        s = move_seeds[2 * rank:2 * (rank + 1)]
        move_selector = MoveSelector(
            [1, 1],
            [DeletionMove(scell, species=species, seed=s[0]),
             InsertionMove(scell, species=species, min_insert=args.min_insert, seed=s[1])],
        )

        tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(args.outdir, f'gcmc_re_nano_{tag}.out'),
            trajectory_write_interval=args.write_interval,
            outfile_write_interval=args.write_interval,
            traj_file=os.path.join(args.outdir, f'gcmc_re_nano_{tag}.xyz'),
        )

    pt = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        temperatures=args.temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.write_interval,
        seed=master_seed,
        outfile=os.path.join(args.outdir, 'replica_exchange_nano.log'),
    )
    pt.run()


if __name__ == '__main__':
    main()
