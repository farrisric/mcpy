"""Molecular GCMC with a MACE potential: rigid-molecule insertion/deletion of
O2 (or another ASE-buildable molecule) above an fcc(111) Ag slab.

The chemical potential is the FULL molecular chemical potential
(mu = E_MACE(relaxed molecule) + delta_mu); orientations are sampled
uniformly, so the rotational partition function is absorbed into mu
(see docs/gcmc_acceptance_convention.rst, "Molecular moves").
"""
import argparse
import os

import numpy as np
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.calculators import BaseCalculator  # noqa: E402
from mcpy.moves import MoleculeDeletionMove, MoleculeInsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import CustomCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--molecule', default='O2',
                   help='ASE g2 molecule name used as the rigid template')
    p.add_argument('--T', type=float, default=400.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=1000, help='Number of GCMC steps')
    p.add_argument('--delta-mu', type=float, default=-0.65,
                   help='Shift added to mu = E_MACE(relaxed molecule) (eV). '
                        'For O2/Ag(111) at 400 K: -0.3 saturates the region, '
                        '-0.8 strips it bare, -0.65 equilibrates with two-way '
                        'exchange at ~19 molecules')
    p.add_argument('--min-insert', type=float, default=1.5,
                   help='Min distance of inserted atoms to existing atoms (A)')
    p.add_argument('--cell-height', type=float, default=7.0,
                   help='Insertion region height above the surface (A)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--device', default='cuda', help='Torch device for MACE')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s) for s in ss.generate_state(3, dtype=np.uint32)]

    atoms = fcc111('Ag', a=4.165, size=(4, 4, 3), periodic=True, vacuum=8)
    bottom_layer = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom_layer))
    surface_z = atoms.positions[:, 2].max()

    template = molecule(args.molecule)
    name = args.molecule
    # Radii: exclude overlap volume around slab atoms; molecule species get 0
    # so the free-volume estimate only subtracts substrate atoms.
    radii = {'Ag': 2.068}
    radii.update({s: 0.0 for s in set(template.get_chemical_symbols())})
    cell = CustomCell(atoms, custom_height=args.cell_height,
                      bottom_z=surface_z, species_radii=radii)

    mace = mace_mp(device=args.device)

    # BaseCalculator relaxes before each energy evaluation; the mu reference
    # uses a stricter relaxation than the GCMC production settings below.
    ref_calculator = BaseCalculator(mace, steps=100, fmax=0.05)
    e_mol = ref_calculator.get_potential_energy(template.copy())
    mus = {name: e_mol + args.delta_mu}

    move_selector = MoveSelector(
        [1, 1],
        [MoleculeInsertionMove(cell, template, name, seed=seeds[0],
                               min_insert=args.min_insert),
         MoleculeDeletionMove(cell, template, name, seed=seeds[1])],
    )

    calculator = BaseCalculator(mace, steps=20, fmax=0.1)

    tag = f'{atoms.get_chemical_formula()}_{name}_dmu_{args.delta_mu}'
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=[],
        molecules={name: template},
        temperature=args.T,
        move_selector=move_selector,
        random_seed=seeds[2],
        outfile=os.path.join(args.outdir, f'gcmc_{tag}.out'),
        trajectory_write_interval=1,
        outfile_write_interval=1,
        traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
    )

    gcmc.run(args.steps)


if __name__ == '__main__':
    main()
