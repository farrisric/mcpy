from ase.build import fcc111
from ase.constraints import FixAtoms
from mace.calculators import mace_mp
import numpy as np

from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell as Cell
from mcpy.ensembles import ReplicaExchange


atoms = fcc111('Ag', a=4.1592, size=(4, 4, 3), periodic=True, vacuum=8)
bottom_layer = [a.index for a in atoms if a.tag == 3]
constraint = FixAtoms(indices=bottom_layer)
atoms.set_constraint(constraint)

cell_ag_ag = Cell(atoms, custom_height=5.5, bottom_z=12.8-2.11,
                  species_radii={'Ag': 2.75, 'O':0})

cell_ag_o = Cell(atoms, custom_height=5.5, bottom_z=12.8-2.11,
                 species_radii={'Ag': 2.11, 'O':0})

calculator = mace_mp(device='cuda')

species = ['Ag', 'O']

seed1=np.random.randint(100_000_000, 1_000_000_000)
seed2=np.random.randint(100_000_000, 1_000_000_000)
seed3=np.random.randint(100_000_000, 1_000_000_000)
seed4=np.random.randint(100_000_000, 1_000_000_000)

move_list = [[25, 25, 25, 25],
             [DeletionMove(cell_ag_ag,
                           species=['Ag'],
                           seed=seed1),
              DeletionMove(cell_ag_o,
                           species=['O'],
                           seed=seed2),
              InsertionMove(cell_ag_ag,
                            species=['Ag'],
                            min_insert=0.5,
                            seed=seed3),
              InsertionMove(cell_ag_o,
                            species=['O'],
                            min_insert=0.5,
                            seed=seed4)]]

move_selector = MoveSelector(*move_list)

mus = {'Ag': -2.99, 'O': -4.91}
delta_mu_O = -0.5
mus['O'] += delta_mu_O

calculator.steps = 40
calculator.fmax = 0.1

def gcmc_factory(T, rank=0):
    Temp_re = T
    gcmc = GrandCanonicalEnsemble(
                atoms=atoms,
                cells=[cell_ag_ag, cell_ag_o],
                calculator=calculator,
                mu=mus,
                units_type='metal',
                species=species,
                temperature=Temp_re,
                move_selector=move_selector,
                outfile=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.out',
                trajectory_write_interval=1,
                outfile_write_interval=1,
                traj_file=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.xyz')
    return gcmc

temperatures = [250,300,350,400,450,500]
seed5=np.random.randint(100_000_000, 1_000_000_000)
np.savetxt('moves_seeds.txt',[seed1,seed2,seed3,seed4,seed5])

pt_gcmc = ReplicaExchange(
        gcmc_factory,
        temperatures=temperatures,
        gcmc_steps=200,
        exchange_interval=10,
        write_out_interval=1,
        seed=seed5)

pt_gcmc.run()