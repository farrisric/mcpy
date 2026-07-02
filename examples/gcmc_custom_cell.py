"""GCMC of Ag and O on an fcc(111) Ag slab with two CustomCell insertion regions."""
import argparse
import os

import numpy as np
from ase.build import fcc111, bulk, molecule
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.calculators import BaseCalculator  # noqa: E402
from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import CustomCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=1_000_000, help='Number of GCMC steps')
    p.add_argument('--delta-mu-O', type=float, default=-0.5, help='Shift to mu_O = E(O2)/2 (eV)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--device', default='cuda', help='Torch device for MACE')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s) for s in ss.generate_state(4, dtype=np.uint32)]

    atoms = fcc111('Ag', a=4.165, size=(4, 4, 3), periodic=True, vacuum=8)
    bottom_layer = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom_layer))

    cell_ag_ag = CustomCell(atoms, custom_height=7, bottom_z=12.8 - 2.068,
                            species_radii={'Ag': 2.947, 'O': 0})
    cell_ag_o = CustomCell(atoms, custom_height=7, bottom_z=12.8 - 2.068,
                           species_radii={'Ag': 2.068, 'O': 0})

    mace = mace_mp(device=args.device)

    species = ['Ag', 'O']
    move_selector = MoveSelector(
        [25, 25, 25, 25],
        [DeletionMove(cell_ag_ag, species=['Ag'], seed=seeds[0]),
         DeletionMove(cell_ag_o, species=['O'], seed=seeds[1]),
         InsertionMove(cell_ag_ag, species=['Ag'], min_insert=0.5, seed=seeds[2]),
         InsertionMove(cell_ag_o, species=['O'], min_insert=0.5, seed=seeds[3])],
    )

    # BaseCalculator relaxes with LBFGS before each energy evaluation; a bare
    # mace_mp would return unrelaxed energies. Reference mu values use a
    # stricter relaxation than the GCMC production settings below.
    ref_calculator = BaseCalculator(mace, steps=100, fmax=0.05)
    e_o2 = ref_calculator.get_potential_energy(molecule('O2'))
    e_ag = ref_calculator.get_potential_energy(bulk('Ag', a=4.165))

    mus = {'Ag': e_ag - 0.176, 'O': e_o2 / 2 + args.delta_mu_O}

    calculator = BaseCalculator(mace, steps=20, fmax=0.1)

    tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}'
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell_ag_ag, cell_ag_o],
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
