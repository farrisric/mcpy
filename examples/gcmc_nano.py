"""GCMC of O adsorption on an Ag octahedral nanoparticle in vacuum."""
import argparse
import os

import numpy as np
from ase.cluster import Octahedron
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=1_000_000, help='Number of GCMC steps')
    p.add_argument('--mu-Ag', type=float, default=-2.99, help='Chemical potential of Ag (eV)')
    p.add_argument('--mu-O', type=float, default=-4.91, help='Reference mu_O (eV)')
    p.add_argument('--delta-mu-O', type=float, default=-0.5, help='Shift applied to mu_O (eV)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--device', default='cuda', help='Torch device for MACE')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    atoms = Octahedron('Ag', 6, 1)

    scell = SphericalCell(atoms, vacuum=3, species_radii={'Ag': 2.947, 'O': 0},
                          mc_sample_points=100_000)

    calculator = mace_mp(device=args.device)

    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=0.5, seed=seed_ins)],
    )

    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}'
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[scell],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=species,
        temperature=args.T,
        move_selector=move_selector,
        outfile=os.path.join(args.outdir, f'gcmc_{tag}.out'),
        traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
    )

    gcmc.run(args.steps)


if __name__ == '__main__':
    main()
