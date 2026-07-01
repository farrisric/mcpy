"""GCMC of O adsorption on a large supported Ag nanoparticle (nvalchemi GPU stack).

Larger-system companion to gcmc_nano_supported.py:
  - Ag711 octahedron (Octahedron('Ag', 11, 4)) on a 10x10 Al2O3 slab (~3.7k atoms)
  - DomeCell hemispherical insertion region over the supported particle
  - AlchemiFCalculator: FIRE geometry relaxation on GPU at every energy evaluation

A GCMC "step" performs ``move_selector.n_moves`` trial moves (here 2: one deletion,
one insertion), and each trial move triggers one relaxed energy evaluation.

Requirements:
  pip install 'nvalchemi-toolkit[mace]'

Typical usage:
  python gcmc_big_alchemi.py --support Al2O3.poscar --delta-mu-O -0.3 \
      --steps 300 --seed 7 --outdir out --device cuda

GPU memory on long runs: the atom count changes every accepted move, which
fragments the CUDA caching allocator, so reserved GPU memory drifts up over a
long run (allocator fragmentation, not a leak). Launch with
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``, or pass
``empty_cache_interval=N`` to the ensemble as an in-loop fallback.
"""
import argparse
import logging
import os

import numpy as np
from ase.cluster import Octahedron
from ase.io import read
from ase.constraints import FixAtoms

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.cell import DomeCell  # noqa: E402
from mcpy.utils.utils import get_p_at_support  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--support', default='Al2O3.poscar', help='POSCAR of the support')
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=300, help='Number of GCMC steps')
    p.add_argument('--mu-Ag', type=float, default=-2.99, help='Chemical potential of Ag (eV)')
    p.add_argument('--mu-O', type=float, default=-4.91, help='Reference mu_O (eV)')
    p.add_argument('--delta-mu-O', type=float, default=-0.5, help='Shift applied to mu_O (eV)')
    p.add_argument('--relax-steps', type=int, default=100, help='Max FIRE steps per energy eval')
    p.add_argument('--fmax', type=float, default=0.05, help='FIRE force threshold (eV/A)')
    p.add_argument('--optimizer', default='fire', choices=['fire', 'fire2'],
                   help='FIRE variant used for relaxation')
    p.add_argument('--max-neighbors', type=int, default=128, help='Neighbor-list cap (avoids OOM)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--device', default='cuda', help='Torch device')
    p.add_argument('--outdir', default='.', help='Output directory')
    p.add_argument('--debug', action='store_true',
                   help='DEBUG logging (per-relaxation FIRE step counts)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        configure_logging(level=logging.DEBUG)
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    support = read(args.support).repeat((10, 10, 1))
    support.center(vacuum=10.0, axis=2)
    z = support.positions[:, 2]
    z_half = z.min() + 0.5 * (z.max() - z.min())
    mask = z < z_half
    support.set_constraint(FixAtoms(mask=mask))

    surface_z = float(np.max(support.positions[:, 2]))
    particle = Octahedron('Ag', 11, 4)
    atoms = get_p_at_support(support, particle, contact_surface='100', gap=2.0)

    scell = DomeCell(atoms, particle_species='Ag', bottom_z=surface_z, vacuum=3.0,
                     species_radii={'Ag': 2.068, 'O': 0, 'Al': 3})

    calculator = AlchemiFCalculator(
        checkpoint='medium-mpa-0',
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        compile_model=False,
        max_neighbors=args.max_neighbors,
        optimizer=args.optimizer,
    )

    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=0.5, seed=seed_ins)],
    )

    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    tag = f'{atoms.get_chemical_formula()}_dmu_{args.delta_mu_O}_{args.optimizer}'
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
