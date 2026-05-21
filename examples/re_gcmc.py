"""Replica-exchange GCMC of Ag and O on an fcc(111) Ag slab.

Requires mpi4py and must be launched with `mpirun -n <N> python examples/re_gcmc.py`,
with one MPI rank per temperature.
"""
import argparse
import os

import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import CustomCell as Cell  # noqa: E402
from mcpy.ensembles import ReplicaExchange  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[250, 300, 350, 400, 450, 500],
                   help='Replica temperatures (K) — one per MPI rank')
    p.add_argument('--gcmc-steps', type=int, default=200, help='GCMC steps between exchanges')
    p.add_argument('--exchange-interval', type=int, default=10, help='Steps between exchanges')
    p.add_argument('--write-interval', type=int, default=1, help='Outfile/traj write interval')
    p.add_argument('--mu-Ag', type=float, default=-2.99, help='Chemical potential of Ag (eV)')
    p.add_argument('--mu-O', type=float, default=-4.91, help='Reference mu_O (eV)')
    p.add_argument('--delta-mu-O', type=float, default=-0.5, help='Shift applied to mu_O (eV)')
    p.add_argument('--rel-max-steps', type=int, default=40, help='LBFGS relaxation steps')
    p.add_argument('--rel-fmax', type=float, default=0.1, help='LBFGS force convergence (eV/Å)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--device', default='cuda', help='Torch device for MACE')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s) for s in ss.generate_state(5, dtype=np.uint32)]
    np.savetxt(os.path.join(args.outdir, 'moves_seeds.txt'), seeds)

    atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8)
    bottom_layer = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom_layer))

    cell_ag_ag = Cell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                      species_radii={'Ag': 2.75, 'O': 0})
    cell_ag_o = Cell(atoms, custom_height=5.5, bottom_z=12.8 - 2.11,
                     species_radii={'Ag': 2.11, 'O': 0})

    calculator = mace_mp(device=args.device)
    calculator.steps = args.rel_max_steps
    calculator.fmax = args.rel_fmax

    species = ['Ag', 'O']
    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    move_selector = MoveSelector(
        [25, 25, 25, 25],
        [DeletionMove(cell_ag_ag, species=['Ag'], seed=seeds[0]),
         DeletionMove(cell_ag_o, species=['O'], seed=seeds[1]),
         InsertionMove(cell_ag_ag, species=['Ag'], min_insert=0.5, seed=seeds[2]),
         InsertionMove(cell_ag_o, species=['O'], min_insert=0.5, seed=seeds[3])],
    )

    def gcmc_factory(T, rank=0):
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
            outfile=os.path.join(args.outdir, f'gcmc_relax_{tag}.out'),
            trajectory_write_interval=args.write_interval,
            outfile_write_interval=args.write_interval,
            traj_file=os.path.join(args.outdir, f'gcmc_relax_{tag}.xyz'),
        )

    pt_gcmc = ReplicaExchange(
        gcmc_factory,
        temperatures=args.temperatures,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.write_interval,
        seed=seeds[4],
    )
    pt_gcmc.run()


if __name__ == '__main__':
    main()
