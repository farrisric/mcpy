"""GCMC of H adsorption on a Pt/Au bimetallic octahedral nanoparticle.

Also performs Pt<->Au permutation moves on the metal framework.
"""
import argparse
import os

import numpy as np
from ase.cluster import Octahedron
from ase.build import molecule
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.calculators import BaseCalculator  # noqa: E402
from mcpy.moves import DeletionMove, InsertionMove, PermutationMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=1_000_000, help='Number of GCMC steps')
    p.add_argument('--delta-mu-H', type=float, default=-0.5, help='Shift to mu_H = E(H2)/2 (eV)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--device', default='cuda', help='Torch device for MACE')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins, seed_perm = (int(s) for s in ss.generate_state(3, dtype=np.uint32))

    atoms = Octahedron('Pt', 7, 2)
    half = len(atoms) // 2
    atoms.symbols = ['Pt'] * half + ['Au'] * half + ['Pt']
    atoms.set_pbc(False)

    scell = SphericalCell(atoms, vacuum=3, species_radii={'Pt': 2, 'Au': 2.5, 'H': 0},
                          mc_sample_points=100_000)

    # BaseCalculator relaxes with LBFGS before each energy evaluation; a bare
    # mace_mp would return unrelaxed energies.
    calculator = BaseCalculator(mace_mp(device=args.device), steps=100, fmax=0.05)

    species = ['H']
    move_selector = MoveSelector(
        [1, 1, 1],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=0.5, seed=seed_ins),
         PermutationMove(species=['Au', 'Pt'], seed=seed_perm)],
    )

    e_h2 = calculator.get_potential_energy(molecule('H2'))

    mus = {'H': e_h2 / 2 + args.delta_mu_H}

    tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_H}'
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
        trajectory_write_interval=1,
        outfile_write_interval=1,
        traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
    )

    gcmc.run(args.steps)


if __name__ == '__main__':
    main()
