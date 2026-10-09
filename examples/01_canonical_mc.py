"""Canonical (NVT) Monte Carlo: chemical ordering in a CuAu nanoparticle.

Runs on a laptop CPU in about ten seconds with ASE's built-in EMT potential,
which is a genuine (if simple) potential for Cu and Au.

The particle count never changes here. Two permutation moves explore the
chemical ordering instead: a single Cu-Au swap refines the order, and a
compound move swaps four pairs at once to jump between orderings that one
swap cannot connect. Every trial is relaxed with LBFGS before the Metropolis
test, so this doubles as a basin-hopping search: ``minima_file`` collects
every configuration that beats the previous best.

The run starts from a random Cu-Au mix. Au segregates to the surface, and the
script prints how many of the 42 surface sites Au holds at the end.

    python examples/01_canonical_mc.py

Writes ``canonical.out``, ``canonical.xyz`` and ``canonical_minima.xyz``.
"""
import numpy as np
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
from ase.optimize import LBFGS

from mcpy.ensembles import CanonicalEnsemble
from mcpy.moves import MoveSelector, PermutationMove
from mcpy.utils.logging import configure as configure_logging

TEMPERATURE = 500
STEPS = 300

# One seed per stochastic component. Leaving any of them at None makes the
# run unrepeatable.
SEED_STRUCTURE, SEED_SWAP, SEED_JUMP, SEED_SEL, SEED_ENS = 0, 1, 2, 3, 4


def build_particle():
    """A 55-atom icosahedron with half its atoms turned into Au at random."""
    atoms = Icosahedron('Cu', noshells=3)
    rng = np.random.default_rng(SEED_STRUCTURE)
    au = rng.choice(len(atoms), size=len(atoms) // 2, replace=False)
    symbols = np.array(atoms.get_chemical_symbols())
    symbols[au] = 'Au'
    atoms.set_chemical_symbols(symbols.tolist())
    return atoms


def main():
    configure_logging()

    atoms = build_particle()

    # Single swaps are cheap to accept and do most of the work; the rarer
    # four-pair jump is accepted less often but escapes orderings that single
    # swaps would have to climb out of one uphill step at a time.
    move_selector = MoveSelector(
        [3, 1],
        [PermutationMove(species=['Cu', 'Au'], seed=SEED_SWAP),
         PermutationMove(species=['Cu', 'Au'], n_swaps=4, seed=SEED_JUMP)],
        seed=SEED_SEL,
    )

    # CanonicalEnsemble relaxes each trial itself, so it takes a plain ASE
    # calculator plus an ASE optimizer class (not an instance).
    mc = CanonicalEnsemble(
        atoms=atoms,
        calculator=EMT(),
        optimizer=LBFGS,
        fmax=0.1,
        move_selector=move_selector,
        temperature=TEMPERATURE,
        random_seed=SEED_ENS,
        outfile='canonical.out',
        traj_file='canonical.xyz',
        minima_file='canonical_minima.xyz',
        outfile_write_interval=10,
        trajectory_write_interval=10,
    )
    mc.run(STEPS)

    # The icosahedron's 13 innermost atoms are the core; the other 42 are the
    # surface. EMT puts Au, the larger atom with the lower surface energy,
    # on the surface.
    final = mc.atoms
    dist = np.linalg.norm(final.positions - final.positions.mean(axis=0), axis=1)
    surface = np.argsort(dist)[13:]
    n_au = sum(final[i].symbol == 'Au' for i in surface)
    print(f'Au on the surface: {n_au} of {len(surface)} sites '
          f'({final.get_chemical_symbols().count("Au")} Au in total)')


if __name__ == '__main__':
    main()
