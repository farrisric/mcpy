from ase.build import fcc110,fcc111,fcc211,fcc100, bulk, molecule
from ase.constraints import FixAtoms
from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell as Cell
from mcpy.ensembles import ReplicaExchange
import numpy as np
import sys


#### initial system parameters
metal_species = 'Au'
metal_lattice_param = 4.1755
gas_species = 'O'
species = [metal_species, gas_species]

### surface generation parametrs
surface_type = '111' # implemented surfaces: 111, 100, 110, 211 all are created with last layer fixed, only 110 has the last two layers fixed --> better to create it with at least 4-5 layers 
surface_size = (4,4,3)
vacuum = 8

### insertion parameters
metal_insertion_radii = {metal_species: 2.75, gas_species: 0.0}
gas_insertion_radii = {metal_species: 2.11, gas_species: 0.0}
min_insert = 0.5 # Angstrom, threshold below which insertion will be rejected and tried again
same_cell = True # True if insertion cells are the same for both elements

if same_cell:
    cell_height = 5
    cell_bottom_pos = 10.411
    metal_cell_height = gas_cell_height = cell_height
    metal_cell_bottom_pos = gas_cell_bottom_pos = cell_bottom_pos
else:
    metal_cell_height = 4
    metal_cell_bottom_pos = 10.411
    gas_cell_height = 5
    gas_cell_bottom_pos = 10.411


### MACE model parameters

model_path = sys.argv[1] # path to mace model
device = sys.argv[3]
rel_max_steps = 40
rel_f_max = 0.1

### Simulation parameters

mu_metal = -3.26999 # bulk metal DFT energy
delta_mu_metal = -0.16 # correction term due to the temperature not equal to 0 K
mu_gas = -4.91 # DFT energy of gas molecule, for biatomic molecule 1/2 DFT energy
delta_mu_gas = float(sys.argv[2]) # delta_mu_O

temperatures = [250,300,350,400,450,500] # replica exchange temperatures

num_metal_del_moves = num_metal_ins_moves = 25 # number for both insertion and delition moves of the metal atoms
num_gas_del_moves = num_gas_ins_moves = 25 # number of both insertion and delition moves of gas atoms
moves_num_list = [num_metal_del_moves, num_gas_del_moves, num_metal_ins_moves, num_gas_ins_moves] # the sum of all the moves number will define a gcmc step  
gcmc_steps = 200 # number of gcmc steps to perform, every steps perform the number of moves set in move_num_lis, then total number of moves = gcmc_steps*sum(moves_num_list)
exchange_interval = 10
write_out_interval = 1

### Create starting configuration

if surface_type == '111':
    atoms = fcc111(metal_species, a= metal_lattice_param, size=surface_size, periodic=True, vacuum=vacuum)
    bottom_layer = [a.index for a in atoms if a.tag == surface_size[-1]]
    constraint = FixAtoms(indices=bottom_layer)
    atoms.set_constraint(constraint)
elif surface_type == '100':
    atoms = fcc100(metal_species, a= metal_lattice_param, size=surface_size, periodic=True, vacuum=vacuum)
    bottom_layer = [a.index for a in atoms if a.tag == surface_size[-1]]
    constraint = FixAtoms(indices=bottom_layer)
    atoms.set_constraint(constraint)
elif surface_type == '110':
    atoms = fcc100(metal_species, a= metal_lattice_param, size=surface_size, periodic=True, vacuum=vacuum)
    bottom_layer = [a.index for a in atoms if a.tag == surface_size[-1] or a.tag == surface_size[-1]-1]
    constraint = FixAtoms(indices=bottom_layer)
    atoms.set_constraint(constraint)
elif surface_type == '211':
    atoms = fcc100(metal_species, a= metal_lattice_param, size=surface_size, periodic=True, vacuum=vacuum)
    layer_size = surface_size[0]*surface_size[1]
    bottom_layer = list(range(len(atoms)))[-layer_size:]
    constraint = FixAtoms(indices=bottom_layer)
    atoms.set_constraint(constraint)
else:
    raise RuntimeError("Invalid surface type selected")


### Initialize Cell object for the insertion regions
cell_metal_gcmc = Cell(atoms, custom_height=metal_cell_height, bottom_z=metal_cell_bottom_pos,
                  species_radii=metal_insertion_radii) 

cell_gas_gcmc = Cell(atoms, custom_height=gas_cell_height, bottom_z=gas_cell_bottom_pos,
                 species_radii=gas_insertion_radii) 


### Initialize MACE calculator
calculator = MACE_F_Calculator(
                model_paths=model_path,
                steps=rel_max_steps,
                fmax=rel_f_max,
                cueq=False,
                device=device,
                )

### Generate seeds for MC moves
seed1=np.random.randint(100_000_000, 1_000_000_000)
seed2=np.random.randint(100_000_000, 1_000_000_000)
seed3=np.random.randint(100_000_000, 1_000_000_000)
seed4=np.random.randint(100_000_000, 1_000_000_000)

### Define the moves of the gcmc simulations
move_list = [moves_num_list, # probabilities of every move type, sum is total number of moves per mc step 
             [DeletionMove(cell_metal_gcmc,
                           species=[metal_species],
                           seed=seed1),
              DeletionMove(cell_gas_gcmc,
                           species=[metal_species],
                           seed=seed2),
              InsertionMove(cell_metal_gcmc,
                            species=[metal_species],
                            min_insert=min_insert,
                            seed=seed3),
              InsertionMove(cell_gas_gcmc,
                            species=[gas_species],
                            min_insert=min_insert,
                            seed=seed4)]]

move_selector = MoveSelector(*move_list)

### Set chemical potentials and relaxation paramters
mus = {metal_species: mu_metal+delta_mu_metal, gas_species: mu_gas+delta_mu_gas}

calculator.steps = rel_max_steps
calculator.fmax = rel_f_max

### Define factory for Replica Exchange 
def gcmc_factory(T, rank=0):
    Temp_re = T
    gcmc = GrandCanonicalEnsemble(
                atoms=atoms,
                cells=[cell_metal_gcmc, cell_gas_gcmc],
                calculator=calculator,
                mu=mus,
                units_type='metal',
                species=species,
                temperature=Temp_re,
                move_selector=move_selector,
                outfile=f're_gcmc_relax_{atoms.get_chemical_formula()}_{gas_species}_dmu_{delta_mu_gas}.out',
                trajectory_write_interval=write_out_interval,
                outfile_write_interval=write_out_interval,
                traj_file=f're_gcmc_relax_{atoms.get_chemical_formula()}_{gas_species}_dmu_{delta_mu_gas}.xyz')
    return gcmc


### Seed for RE move and output all the seeds
seed5=np.random.randint(100_000_000, 1_000_000_000)
np.savetxt('moves_seeds.txt',[seed1,seed2,seed3,seed4,seed5])

### Create replica exchange object
pt_gcmc = ReplicaExchange(
        gcmc_factory,
        temperatures=temperatures,
        gcmc_steps=gcmc_steps, # every step is 1000 mc moves
        exchange_interval=exchange_interval, # frequency of replica exchanges in gcmc_steps 
        write_out_interval=write_out_interval, # write data for every write_out_interval gmcc_steps 
        seed=seed5)

### Run simulation
pt_gcmc.run()
