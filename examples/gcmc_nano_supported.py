"""GCMC of O adsorption on an Ag nanoparticle supported on Al2O3."""
import argparse
import os

import numpy as np
from ase.cluster import Octahedron
from ase.io import read
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import CustomCell  # noqa: E402
from mcpy.utils.utils import get_p_at_support  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support', default='Al2O3.poscar', help='POSCAR of the support')
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

    support = read(args.support).repeat((4, 4, 1))
    support.center(vacuum=10.0, axis=2)
    z = support.positions[:, 2]
    z_half = z.min() + 0.5 * (z.max() - z.min())
    mask = z < z_half
    support.set_constraint(FixAtoms(mask=mask))

    surface_z = float(np.max(support.positions[:, 2]))
    particle = Octahedron('Ag', 5, 2)
    atoms = get_p_at_support(support, particle, contact_surface='100', gap=2.0)

    scell = CustomCell(atoms, custom_height=20, bottom_z=surface_z,
                       species_radii={'Ag': 2.068, 'O': 0, 'Al': 3})

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
