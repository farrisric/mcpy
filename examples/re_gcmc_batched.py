"""Batched replica-exchange GCMC on a single GPU.

Single-process variant of ``re_gcmc.py``. All replicas live in one Python
process and share one ``AlchemiCalculator`` instance; energies for trial
moves are evaluated in a single batched forward pass per MC step.

Run::

    python examples/re_gcmc_batched.py --temperatures 250 300 350 400

No mpirun. One GPU. Each replica must build its OWN cells and move_selector
(BatchedReplicaExchange would otherwise share state across replicas).
"""
import argparse
import os

import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiCalculator  # noqa: E402
from mcpy.cell import CustomCell as Cell  # noqa: E402
from mcpy.ensembles import BatchedReplicaExchange  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[250, 300, 350, 400, 450, 500],
                   help='Replica temperatures (K)')
    p.add_argument('--gcmc-steps', type=int, default=200)
    p.add_argument('--exchange-interval', type=int, default=10)
    p.add_argument('--write-interval', type=int, default=1)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cueq', action='store_true')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--mu-Ag', type=float, default=-2.99)
    p.add_argument('--mu-O', type=float, default=-4.91)
    p.add_argument('--delta-mu-O', type=float, default=-0.5)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--outdir', default='.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    n_replicas = len(args.temperatures)
    # 4 move seeds per replica + 1 master seed for RE.
    seeds = ss.generate_state(4 * n_replicas + 1, dtype=np.uint32)
    move_seeds = [int(s) for s in seeds[:4 * n_replicas]]
    master_seed = int(seeds[-1])

    # Shared template — each factory call copies and builds its own cells/moves.
    base_atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8)
    bottom_layer = [a.index for a in base_atoms if a.tag == 3]
    base_atoms.set_constraint(FixAtoms(indices=bottom_layer))

    # ONE model, shared across replicas. Batched eval reuses this calculator.
    calculator = AlchemiCalculator(
        checkpoint=args.checkpoint,
        device=args.device,
        enable_cueq=not args.no_cueq,
        compile_model=not args.no_compile,
    )

    species = ['Ag', 'O']
    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    def gcmc_factory(T, rank):
        atoms = base_atoms.copy()
        atoms.set_constraint(FixAtoms(indices=bottom_layer))

        cell_ag_ag = Cell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                          species_radii={'Ag': 2.75, 'O': 0})
        cell_ag_o = Cell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                         species_radii={'Ag': 2.11, 'O': 0})

        s = move_seeds[4 * rank:4 * (rank + 1)]
        move_selector = MoveSelector(
            [25, 25, 25, 25],
            [DeletionMove(cell_ag_ag, species=['Ag'], seed=s[0]),
             DeletionMove(cell_ag_o, species=['O'], seed=s[1]),
             InsertionMove(cell_ag_ag, species=['Ag'], min_insert=0.5, seed=s[2]),
             InsertionMove(cell_ag_o, species=['O'], min_insert=0.5, seed=s[3])],
        )

        tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell_ag_ag, cell_ag_o],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(args.outdir, f'gcmc_batched_{tag}.out'),
            trajectory_write_interval=args.write_interval,
            outfile_write_interval=args.write_interval,
            traj_file=os.path.join(args.outdir, f'gcmc_batched_{tag}.xyz'),
        )

    pt = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        temperatures=args.temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.write_interval,
        seed=master_seed,
        outfile=os.path.join(args.outdir, 'replica_exchange_batched.log'),
    )
    pt.run()


if __name__ == '__main__':
    main()
