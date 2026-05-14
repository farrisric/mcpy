"""
GCMC simulation for Pd-Au alloy nanoparticles with H adsorption.

Usage:
    python gcmc_nano.py <nano_size> <delta_mu_H> <model_path> <device>

    nano_size  : "small" (140 atoms, length=5 cutoff=1) or "large" (405 atoms, length=7 cutoff=2)
    delta_mu_H : float, e.g. -0.5
    model_path : path to MACE model file
    device     : cpu or cuda
"""
import sys
import numpy as np
from pathlib import Path

from ase.cluster import Octahedron
from ase.build import molecule

from mace.calculators import MACECalculator

from mcpy.moves import DeletionMove, InsertionMove, PermutationMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import SphericalCell

# ── CLI arguments ──────────────────────────────────────────────────────────────
nano_size = sys.argv[1]             # "small" or "large"
delta_mu_H = float(sys.argv[2])
model_path = sys.argv[3]
device = sys.argv[4]

# ── System parameters ──────────────────────────────────────────────────────────
metal_a = "Pd"
metal_b = "Au"

if nano_size == "small":
    length, cutoff = 5, 1           # ~140 atoms truncated octahedron
elif nano_size == "large":
    length, cutoff = 7, 2           # ~405 atoms truncated octahedron
else:
    raise ValueError(f"Unsupported nano_size: {nano_size}. Use 'small' or 'large'.")

# ── Build Pd nanoparticle, randomise half to Au ────────────────────────────────
atoms = Octahedron(metal_a, length=length, cutoff=cutoff)
atoms.set_pbc(False)

rng = np.random.default_rng(42)
n_atoms = len(atoms)
n_swap = n_atoms // 2
swap_indices = rng.choice(n_atoms, size=n_swap, replace=False)
symbols = list(atoms.symbols)
for idx in swap_indices:
    symbols[idx] = metal_b
atoms.symbols = symbols

print(f"Nanoparticle: {n_atoms} atoms ({nano_size}), "
      f"{symbols.count(metal_a)} {metal_a} + {symbols.count(metal_b)} {metal_b}")

# ── Spherical insertion cell ───────────────────────────────────────────────────
species_radii = {metal_a: 2.0, metal_b: 2.5, "H": 0.0}

scell = SphericalCell(atoms, vacuum=3.0, species_radii=species_radii,
                      mc_sample_points=100_000)

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
        DeletionMove(scell, species=species, seed=seed1),
        InsertionMove(scell, species=species, min_insert=0.5, seed=seed2),
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
results_dir = Path("../results") / f"gcmc_nano_{nano_size}"
results_dir.mkdir(parents=True, exist_ok=True)

tag = f"PdAu_nano_{nano_size}_dmu_{delta_mu_H}"
outfile = str(results_dir / f"{tag}.out")
traj_file = str(results_dir / f"{tag}.xyz")

np.savetxt(str(results_dir / f"{tag}_seeds.txt"), [seed1, seed2, seed3])

# ── Run ────────────────────────────────────────────────────────────────────────
gcmc = GrandCanonicalEnsemble(
    atoms=atoms,
    cells=[scell],
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
