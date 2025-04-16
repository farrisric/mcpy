from ase.cluster import Octahedron

from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import SphericalCell


atoms = Octahedron('Ag', 6, 1)

scell = SphericalCell(atoms, vacuum=3, species_radii={'Ag': 2.947, 'O' : 0},
                      mc_sample_points=100_000)

calculator = MACE_F_Calculator(
                model_paths='/home/riccardo/Downloads/mace-small-density-agnesi-stress.model',
                steps=20,
                fmax=0.1,
                cueq=False,
                device='cpu',
                )

species = ['O']

move_list = [[1, 1],
             [DeletionMove(scell,
                           species=['O'],
                           seed=43215423143),
              InsertionMove(scell,
                            species=['O'],
                            min_insert=0.5,
                            seed=3675437856)]]

move_selector = MoveSelector(*move_list)

mus = {'Ag': -2.99, 'O': -4.91}
delta_mu_O = -0.5
mus['O'] += delta_mu_O
T = 500

gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector)

gcmc.run(1000000)
