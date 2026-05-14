"""
Replica-exchange GCMC for Pd-Au alloy nanoparticles with H adsorption.

Usage (run with mpirun):
    mpirun -np <n_replicas> python regcmc_nano.py <nano_size> <delta_mu_H> <model_path> <device>

    n_replicas : number of MPI ranks (must match temperature ladder length)
    nano_size  : "small" (140 atoms) or "large" (405 atoms)
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
from mcpy.ensembles import ReplicaExchange

# ── CLI arguments ──────────────────────────────────────────────────────────────
nano_size = sys.argv[1]
delta_mu_H = float(sys.argv[2])
model_path = sys.argv[3]
device = sys.argv[4]

# ── System parameters ──────────────────────────────────────────────────────────
metal_a = "Pd"
metal_b = "Au"

if nano_size == "small":
    length, cutoff = 5, 1
elif nano_size == "large":
    length, cutoff = 7, 2
else:
    raise ValueError(f"Unsupported nano_size: {nano_size}. Use 'small' or 'large'.")

# ── Build Pd-Au nanoparticle ───────────────────────────────────────────────────
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

# ── RE-GCMC parameters ────────────────────────────────────────────────────────
temperatures = [300, 600]
gcmc_steps_per_exchange = 10
exchange_interval = 5

# ── Output ─────────────────────────────────────────────────────────────────────
results_dir = Path("../results") / f"regcmc_nano_{nano_size}"
results_dir.mkdir(parents=True, exist_ok=True)

tag = f"PdAu_nano_{nano_size}_dmu_{delta_mu_H}"

seed_re = np.random.randint(100_000_000, 1_000_000_000)
np.savetxt(str(results_dir / f"{tag}_seeds.txt"), [seed1, seed2, seed3, seed_re])

# ── GCMC factory ───────────────────────────────────────────────────────────────
def gcmc_factory(T, rank=0):
    return GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[scell],
        calculator=calculator,
        mu=mus,
        units_type="metal",
        species=species,
        temperature=T,
        move_selector=move_selector,
        outfile=str(results_dir / f"{tag}_rank_{rank}.out"),
        trajectory_write_interval=1,
        outfile_write_interval=1,
        traj_file=str(results_dir / f"{tag}_rank_{rank}.xyz"),
    )

# ── Run ────────────────────────────────────────────────────────────────────────
pt_gcmc = ReplicaExchange(
    gcmc_factory,
    temperatures=temperatures,
    gcmc_steps=gcmc_steps_per_exchange,
    exchange_interval=exchange_interval,
    write_out_interval=1,
    outfile=str(results_dir / f"{tag}_re.log"),
    seed=seed_re,
)

pt_gcmc.run()
