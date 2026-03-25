import logging
logging.basicConfig(
    filename='insertion.log',
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="a"
)
from ase.build import fcc111
from ase.constraints import FixAtoms
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors


def create_surface111(species, a, size=(2,2,3),vacuum=8,constraint_tag=3):
    atoms = fcc111(species, a=a, size=size, periodic=True, vacuum=vacuum)
    bottom_layer = [a.index for a in atoms if a.tag == constraint_tag]
    constraint = FixAtoms(indices=bottom_layer)
    atoms.set_constraint(constraint)
    atoms.set_pbc(True)
    return atoms

def create_insertion_move(atoms, insert_species, metal_species, cell_bottom, cell_height):
    cell = CustomCell(atoms, custom_height=cell_height, bottom_z=cell_bottom,
            species_radii={metal_species: 0, insert_species: 0})
    seed = np.random.randint(100_000_000, 1_000_000_000) 
    insertion = InsertionMove(cell,
                            species=[insert_species],
                            min_insert=0.5,
                            seed=seed)
    return insertion

# model definition and relaxation parameters
model_path = sys.argv[1]
device = 'cuda'
if device == 'cuda':
    cueq = False   # cueq always off. Change this to activate it
else:
    cueq = False
relax_max_steps = 40 # maximum relaxation steps
relax_fmax = 0.1     # max force threshold for convergence of the relaxation

# metal parameters and gas species
metal_species = 'Cu'   # speices of the surface atoms
lattice_param = 3.63 # angstroms
gas_species = 'O'      # species of the adsorbate
insert_both = True     # change to insert only the gas or to check also the distance for metal instertion

if insert_both:
    species_to_insert = [metal_species, gas_species]
else:
    species_to_insert = [gas_species]

# cell parameters
cell_bottom = 10.09 # z coord. of the cell bottom
cell_height = 5 # angstroms

# number of trials and logging
n_trials = 10000
log_file = 'insertion.log'
log_freq = 500 # log every these steps


calculator = MACE_F_Calculator(
                model_paths=model_path,
                steps=relax_max_steps,
                fmax=relax_fmax,
                cueq=cueq,
                device=device,
                )



with open(log_file, "w") as f:
    f.write("### Insertion log ###\n")
    f.write("\n---------------------------------------\n")
    f.write(f"System: {metal_species}-{gas_species}\n")
    f.write(f"{metal_species} lattice parameter: {lattice_param:.4f}\n")
    f.write(f"MACE model used: {model_path}\n")
    f.write(f"MACE model device: {device}\n")
    f.write(f"Species to insert: {species_to_insert}\n")
    f.write(f"Num. trials per inserted species = {n_trials}\n")
    f.write("---------------------------------------\n")
    f.write("###### Starting insertions ######\n")

for insert_species in species_to_insert:
    relaxed_distances = []
    insertion_distances = []
    with open(log_file, "a") as f:
        f.write("---------------------------------------\n")
        f.write(f"Starting insertion of {insert_species}\n")
        f.write("---------------------------------------\n")
    for i in range(n_trials):
        if i%log_freq == 0:
             message = f"Insertion {i} / {n_trials}"
             logging.info(message)
        atoms = create_surface111(metal_species, lattice_param)
        # create cell and define insertion move
        insertion = create_insertion_move(atoms, insert_species, metal_species, cell_bottom, cell_height)
        # perform metal insertion 
        atoms_new, _, _ = insertion.do_trial_move(atoms) 
        # compute initial distance
        last_atom_index = len(atoms_new) - 1
        dist = atoms_new.get_distances(last_atom_index,range(len(atoms_new)-1),mic=True)
        insertion_distances.append(np.sort(dist)[0])
        # relax and compute final distance
        calculator.get_potential_energy(atoms_new)
        dist = atoms_new.get_distances(last_atom_index,range(len(atoms_new)-1),mic=True)
        relaxed_distances.append(np.sort(dist)[0])
    np.save(f'{insert_species}_distances.npy',np.vstack((insertion_distances,relaxed_distances)).T)

logging.shutdown()

with open(log_file, "a") as f:
    f.write("---------------------------------------\n")
    f.write("###### Insertions completed ######")

# plot results
plt.rcParams.update({
"figure.dpi" : 200,
"font.family": "sans-serif",
"font.sans-serif": "DejaVu Sans",
"font.size" : 14,
'mathtext.default':  'regular'
    })


dist_metal = np.load(f'{metal_species}_distances.npy')
dist_gas = np.load(f'{gas_species}_distances.npy')

fig, ax = plt.subplots(figsize=(10, 6))

for insert_species in species_to_insert:
    Z = atomic_numbers[insert_species]
    color = jmol_colors[Z]
    alpha = 0.5
    light_color = [c + (1 - c) * alpha for c in color]

    distances = np.load(f'{insert_species}_distances.npy')
    # --- Histograms (density normalized) ---
    bins = ax.hist(
        distances[:, 1],
        bins=100,
        alpha=0.5,
        label=f'{metal_species}-{insert_species} insertion dist.',
        color=light_color,
        density=True
    )

# --- KDEs ---
    x = np.linspace(distances[:, 1].min(), distances[:, 1].max(), 500)
    kde = gaussian_kde(distances[:, 1])

    ax.plot(x, kde(x), lw=2, c=color)
    
    relax_dist = bins[1][np.argmax(bins[0])] + (bins[1][2] - bins[1][1]) / 2

    ax.axvline(
        relax_dist,
        c=color,
        linestyle='--',
        label=f'{metal_species}-{insert_species} relaxation dist. = {relax_dist:.2f} $\\AA$'
    )
# --- Labels & formatting ---
ax.set_xlabel('Insertion distance after relaxation [$\\AA$]', fontsize=22)
ax.set_ylabel('Density', fontsize=22)
ax.tick_params(direction='in', labelsize=15)

# --- Relaxation distances from histogram peaks ---

ax.legend(fontsize=18)
plt.savefig('dist_hist.png', bbox_inches='tight', dpi=300)
