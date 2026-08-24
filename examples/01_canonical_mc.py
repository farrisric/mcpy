"""Canonical (NVT) Monte Carlo: chemical ordering in a CuAu nanoparticle.

Runs on a laptop CPU in about a minute with ASE's built-in EMT potential,
which is a genuine (if simple) potential for Cu and Au.

The particle count never changes here. Two moves explore the configuration
space instead: ``PermutationMove`` swaps a Cu and an Au atom, and
``DisplacementMove`` nudges atoms. Every trial is relaxed with LBFGS before
the Metropolis test, so this doubles as a basin-hopping search --
``minima_file`` collects every configuration that beats the previous best.

    python examples/01_canonical_mc.py

Writes ``canonical.out``, ``canonical.xyz`` and ``canonical_minima.xyz``.
"""
import numpy as np
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
from ase.optimize import LBFGS

from mcpy.ensembles import CanonicalEnsemble
from mcpy.moves import DisplacementMove, MoveSelector, PermutationMove
from mcpy.utils.logging import configure as configure_logging

TEMPERATURE = 500
STEPS = 300

# One seed per stochastic component. Leaving any of them at None makes the
# run unrepeatable.
SEED_STRUCTURE, SEED_PERM, SEED_DIS, SEED_SEL, SEED_ENS = 0, 1, 2, 3, 4


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

    # PermutationMove changes the chemical order; DisplacementMove relaxes
    # the geometry around it. The permutation is sampled three times as often
    # because ordering is what this run is about.
    move_selector = MoveSelector(
        [3, 1],
        [PermutationMove(species=['Cu', 'Au'], seed=SEED_PERM),
         DisplacementMove(species=['Cu', 'Au'], max_displacement=0.1, seed=SEED_DIS)],
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


if __name__ == '__main__':
    main()
