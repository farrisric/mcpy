from ase.cluster import Octahedron
from ase.io import read 
from ase.constraints import FixAtoms

from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell
from mcpy.utils.utils import get_p_at_support

import numpy as np 


support = read('Al2O3.poscar').repeat((4, 4, 1))
support.center(vacuum=10.0, axis=2)
z = support.positions[:, 2]
z_half = z.min() + 0.5 * (z.max() - z.min())
mask = z < z_half                           # True for atoms to be fixed
support.set_constraint(FixAtoms(mask=mask))

surface_z = float(np.max(support.positions[:, 2]))
particle = Octahedron('Ag', 5, 2)
atoms = get_p_at_support(support, particle, contact_surface='100', gap=2.0)

scell = CustomCell(atoms, custom_height=20, bottom_z=surface_z,
                       species_radii={'Ag': 2.068, 'O' : 0, 'Al': 3})

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
