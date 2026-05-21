"""GCMC of O adsorption on an Ag octahedral nanoparticle using the nvalchemi GPU stack.

Mirrors gcmc_nano.py but replaces the ASE MACE path with AlchemiFCalculator
(FIRE geometry relaxation on GPU) or AlchemiCalculator (energy-only, no relax).

Requirements:
  pip install 'nvalchemi-toolkit[mace]'

Typical usage:
  # With FIRE relaxation (recommended for >=100 atoms):
  python gcmc_nano_alchemi.py --checkpoint medium-mpa-0 --device cuda

  # Energy-only (fastest; no geometry relaxation):
  python gcmc_nano_alchemi.py --checkpoint medium-mpa-0 --no-relax

  # Reuse a shared model across restart runs (pass a .pt path):
  python gcmc_nano_alchemi.py --checkpoint /path/to/model.pt
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
from mcpy.calculators import AlchemiCalculator, AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='Named checkpoint (e.g. medium-mpa-0) or path to a .pt file')
    p.add_argument('--no-relax', action='store_true',
                   help='Use AlchemiCalculator (energy-only); default uses AlchemiFCalculator '
                        '(FIRE relax)')
    p.add_argument('--device', default='cuda', help='Torch device (cuda or cpu)')
    p.add_argument('--no-cueq', action='store_true',
                   help='Disable cuEquivariance kernel fusion')
    p.add_argument('--no-compile', action='store_true',
                   help='Disable torch.compile (faster startup, slower inference)')
    p.add_argument('--fmax', type=float, default=0.05,
                   help='FIRE force convergence threshold (eV/Å) [relax mode only]')
    p.add_argument('--relax-steps', type=int, default=500,
                   help='Max FIRE steps per energy evaluation [relax mode only]')
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=1_000_000, help='Number of GCMC steps')
    p.add_argument('--mu-Ag', type=float, default=-2.99, help='Chemical potential of Ag (eV)')
    p.add_argument('--mu-O', type=float, default=-4.91, help='Reference mu_O (eV)')
    p.add_argument('--delta-mu-O', type=float, default=-0.5, help='Shift applied to mu_O (eV)')
    p.add_argument('--min-insert', type=float, default=0.5,
                   help='Min insertion distance to existing atoms (Å)')
    p.add_argument('--vacuum', type=float, default=3.0, help='Vacuum around nanoparticle (Å)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--outdir', default='.', help='Output directory')
    p.add_argument('--write-interval', type=int, default=1,
                   help='Outfile/trajectory write interval')
    return p.parse_args()


def build_calculator(args):
    enable_cueq = not args.no_cueq
    compile_model = not args.no_compile

    if args.no_relax:
        return AlchemiCalculator(
            checkpoint=args.checkpoint,
            device=args.device,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )
    return AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
    )


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    atoms = Octahedron('Ag', 6, 1)

    scell = SphericalCell(atoms, vacuum=args.vacuum, species_radii={'Ag': 2.947, 'O': 0},
                          mc_sample_points=100_000)

    calculator = build_calculator(args)

    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=args.min_insert, seed=seed_ins)],
    )

    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    mode = 'nrelax' if args.no_relax else f'fire_fmax{args.fmax}'
    tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}_{mode}'
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
        trajectory_write_interval=args.write_interval,
        outfile_write_interval=args.write_interval,
    )

    gcmc.run(args.steps)


if __name__ == '__main__':
    main()
