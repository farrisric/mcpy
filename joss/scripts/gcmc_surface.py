"""
GCMC simulation for Pd-Au alloy surfaces with H adsorption.

Usage:
    python gcmc_surface.py <surface_type> <delta_mu_H> <model_path> <device>

    surface_type : 111 or 100
    delta_mu_H   : float, e.g. -0.5
    model_path   : path to MACE model file
    device       : cpu or cuda
"""
import sys
import numpy as np
from pathlib import Path

from ase.build import fcc111, fcc100, molecule, bulk
from ase.constraints import FixAtoms

from mace.calculators import MACECalculator

from mcpy.moves import DeletionMove, InsertionMove, PermutationMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell as Cell

# ── CLI arguments ──────────────────────────────────────────────────────────────
surface_type = sys.argv[1]          # "111" or "100"
delta_mu_H = float(sys.argv[2])    # chemical potential offset for H
model_path = sys.argv[3]           # MACE model path
device = sys.argv[4]               # "cpu" or "cuda"

# ── System parameters ──────────────────────────────────────────────────────────
metal_a = "Pd"
metal_b = "Au"
lattice_param = 4.00               # Pd-Au average lattice parameter (Angstrom)
surface_size = (3, 3, 4)
vacuum = 8                         # Angstrom

# ── Build alloy surface ───────────────────────────────────────────────────────
if surface_type == "111":
    atoms = fcc111(metal_a, a=lattice_param, size=surface_size,
                   periodic=True, vacuum=vacuum)
elif surface_type == "100":
    atoms = fcc100(metal_a, a=lattice_param, size=surface_size,
                   periodic=True, vacuum=vacuum)
else:
    raise ValueError(f"Unsupported surface type: {surface_type}")

bottom_layer = [a.index for a in atoms if a.tag == surface_size[-1]]
constraint = FixAtoms(indices=bottom_layer)
atoms.set_constraint(constraint)

# Randomise half the non-fixed atoms to Au to create a Pd50Au50 alloy
rng = np.random.default_rng(42)
free_indices = [i for i in range(len(atoms)) if i not in bottom_layer]
n_swap = len(free_indices) // 2
swap_indices = rng.choice(free_indices, size=n_swap, replace=False)
symbols = list(atoms.symbols)
for idx in swap_indices:
    symbols[idx] = metal_b
atoms.symbols = symbols

# ── Insertion cells ────────────────────────────────────────────────────────────
top_z = max(atoms.positions[:, 2])
cell_bottom = top_z - 2.0
cell_height = 5.0

insertion_radii = {metal_a: 2.0, metal_b: 2.0, "H": 0.0}

cell_h = Cell(atoms, custom_height=cell_height, bottom_z=cell_bottom,
              species_radii=insertion_radii)

# ── Calculator ─────────────────────────────────────────────────────────────────
rel_steps = 40
rel_fmax = 0.1

ase_calc = MACECalculator(
    model_paths=model_path,
    device=device,
    default_dtype="float64",
    head="omat_pbe",
)

calculator = MACE_F_Calculator(
    model_paths=ase_calc,
    steps=rel_steps,
    fmax=rel_fmax,
)

# ── Reference energies ─────────────────────────────────────────────────────────
h2 = molecule("H2")
calculator.steps = 100
calculator.fmax = 0.05
e_h2 = calculator.get_potential_energy(h2)
mu_H_ref = e_h2 / 2.0

calculator.steps = rel_steps
calculator.fmax = rel_fmax

# ── Moves ──────────────────────────────────────────────────────────────────────
species = ["H"]

seed1 = np.random.randint(100_000_000, 1_000_000_000)
seed2 = np.random.randint(100_000_000, 1_000_000_000)
seed3 = np.random.randint(100_000_000, 1_000_000_000)

move_list = [
    [1, 1, 1],
    [
        DeletionMove(cell_h, species=species, seed=seed1),
        InsertionMove(cell_h, species=species, min_insert=0.5, seed=seed2),
        PermutationMove(species=[metal_a, metal_b], seed=seed3),
    ],
]

move_selector = MoveSelector(*move_list)

# ── Chemical potentials ────────────────────────────────────────────────────────
mus = {"H": mu_H_ref + delta_mu_H}

# ── Simulation parameters ─────────────────────────────────────────────────────
temperature = 500  # K
gcmc_steps = 10

# ── Output ─────────────────────────────────────────────────────────────────────
results_dir = Path("../results") / f"gcmc_surface_{surface_type}"
results_dir.mkdir(parents=True, exist_ok=True)

tag = f"PdAu_{surface_type}_dmu_{delta_mu_H}"
outfile = str(results_dir / f"{tag}.out")
traj_file = str(results_dir / f"{tag}.xyz")

np.savetxt(str(results_dir / f"{tag}_seeds.txt"), [seed1, seed2, seed3])

# ── Run ────────────────────────────────────────────────────────────────────────
gcmc = GrandCanonicalEnsemble(
    atoms=atoms,
    cells=[cell_h],
    calculator=calculator,
    mu=mus,
    units_type="metal",
    species=species,
    temperature=temperature,
    move_selector=move_selector,
    outfile=outfile,
    trajectory_write_interval=1,
    outfile_write_interval=1,
    traj_file=traj_file,
)

gcmc.run(gcmc_steps)
