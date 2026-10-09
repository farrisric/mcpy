"""GCMC on an Ag(111) surface with a machine-learned potential (GPU).

A production run, in contrast to examples 00-02: a real potential, a real
surface, and both species exchanged with their reservoirs.

  * O is inserted and deleted in a 7 A region that starts one O-Ag distance
    *below* the top layer, so subsurface sites are reachable.
  * Ag is inserted and deleted through its own cell, identical to the O one
    except for the exclusion radius an insertion sees. Exchanging Ag is what
    lets the surface reconstruct into an oxide instead of merely decorating
    the lattice it started with.
  * Reference chemical potentials are derived from the running potential, so
    swapping the checkpoint keeps the setup self-consistent. Only DELTA_MU_O
    is a physical knob, and its balance point differs per checkpoint.

Every setting is a constant below; edit and re-run. Set EXCHANGE_AG to False
for the O-only setup. ``tests/test_ag111_gcmc_golden.py`` pins that O-only
setup with a different checkpoint and that checkpoint's own calibration.

Requirements: pip install 'nvalchemi-toolkit[mace]', and a CUDA GPU.

    python examples/03_gcmc_surface_mace.py
"""
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms

from mcpy.calculators import AlchemiFCalculator
from mcpy.cell import CustomCell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector
from mcpy.utils import derive_mu_bulk, derive_mu_gas
from mcpy.utils.logging import configure as configure_logging

# --- potential -------------------------------------------------------------
# A named checkpoint (e.g. 'medium-mpa-0') or a path to a local .pt file.
CHECKPOINT = 'medium-mpa-0'
DEVICE = 'cuda'
ENABLE_CUEQ = True    # cuEquivariance kernel fusion; verify per checkpoint
COMPILE_MODEL = True  # torch.compile: ~2x faster forward pass
RELAX_STEPS = 300     # max FIRE steps per energy evaluation
FMAX = 0.05           # FIRE force threshold (eV/A)
OPTIMIZER = 'fire2'   # 'fire2' usually converges in fewer steps than 'fire'

# --- calibration -----------------------------------------------------------
# Measured with CHECKPOINT = 'medium-mpa-0'. Shared with
# examples/04_replica_exchange_batched.py: keep the two copies identical, and
# re-measure all of them when you change the potential (docs/species_radii.rst).
AG_LATTICE = 4.165   # fcc Ag lattice constant (A): the minimum of E(a), 4.166 A
# Relaxed O-Ag bond of O in the fcc hollow of Ag(111) (2.136 A). Used twice:
# as the exclusion radius an incoming O sees around Ag, and as how far below
# the top layer the exchange region starts, so subsurface sites are reachable.
R_O_AG = 2.14
# An incoming Ag needs a larger radius: 2.75 A sits just below the relaxed Ag
# adatom-Ag bond on Ag(111) (2.78 A). This is why each species gets its own
# cell over the same region.
R_AG_AG = 2.75
# Exclusion radius around an existing O (A). A minimal clearance that stops an
# insertion on top of an O atom, not a bond length.
R_O = 0.5
# Shift (eV) on the bulk Ag energy per atom. Carried over from the group's
# earlier Ag-oxidation setup; set 0.0 for Ag in equilibrium with bulk Ag.
MU_AG_CORRECTION = -0.176

# --- system ----------------------------------------------------------------
SLAB_SIZE = (16, 16, 4)     # fcc111 supercell (Nx, Ny, Nlayers) = 1024 atoms
VACUUM = 10.0               # vacuum above and below the slab (A)
FIX_LAYERS = 2              # bottom layers frozen
CELL_HEIGHT = 7.0           # height of the exchange region (A)

# --- thermodynamics --------------------------------------------------------
TEMPERATURE = 500.0
DELTA_MU_O = -0.3           # shift applied to mu_O = E(O2)/2
EXCHANGE_AG = True          # False = O only

# --- run -------------------------------------------------------------------
# A step takes a few seconds on one GPU (3.7 s measured on a 3.7k-atom
# system), so this is several hours: a first look, not a converged sweep.
STEPS = 10_000
WRITE_INTERVAL = 10
SEED = 42                   # one master seed; every RNG below derives from it


def build_slab():
    nx, ny, nlayers = SLAB_SIZE
    atoms = fcc111('Ag', a=AG_LATTICE, size=(nx, ny, nlayers),
                   periodic=True, vacuum=VACUUM)
    # tag=1 is the top layer, tag=nlayers the bottom one (ASE convention).
    fix_from = nlayers - FIX_LAYERS + 1
    atoms.set_constraint(
        FixAtoms(indices=[a.index for a in atoms if a.tag >= fix_from]))
    return atoms


def main():
    configure_logging()

    # One master seed, one derived stream per stochastic component. Any of
    # them left at None would make the run unrepeatable.
    (seed_ins_o, seed_del_o, seed_cell_o,
     seed_ins_ag, seed_del_ag, seed_cell_ag,
     seed_sel, seed_ens) = (
        int(s) for s in np.random.SeedSequence(SEED).generate_state(8, dtype=np.uint32)
    )

    atoms = build_slab()

    # Both cells cover the same region and differ only in the radii, i.e. in
    # how much of that region each species may actually be inserted into.
    z_top = float(atoms.positions[atoms.get_tags() == 1, 2].max())
    bottom_z = z_top - R_O_AG
    cell_o = CustomCell(atoms, custom_height=CELL_HEIGHT, bottom_z=bottom_z,
                        species_radii={'Ag': R_O_AG, 'O': R_O}, seed=seed_cell_o)
    cell_ag = CustomCell(atoms, custom_height=CELL_HEIGHT, bottom_z=bottom_z,
                         species_radii={'Ag': R_AG_AG, 'O': R_O}, seed=seed_cell_ag)

    # AlchemiFCalculator relaxes each trial structure with GPU-native FIRE,
    # honouring the FixAtoms constraint. Swap in AlchemiCalculator for
    # energy-only evaluation (faster, but no relaxation of the trial).
    calculator = AlchemiFCalculator(
        checkpoint=CHECKPOINT,
        steps=RELAX_STEPS,
        fmax=FMAX,
        device=DEVICE,
        enable_cueq=ENABLE_CUEQ,
        compile_model=COMPILE_MODEL,
        optimizer=OPTIMIZER,
    )

    # Hardcoded reference energies silently change meaning when the
    # checkpoint changes, so they are measured with the running potential.
    mu_ag = derive_mu_bulk(calculator, 'Ag', a=AG_LATTICE) + MU_AG_CORRECTION
    mu_o = derive_mu_gas(calculator, 'O2') + DELTA_MU_O

    moves = [InsertionMove(cell_o, species=['O'], seed=seed_ins_o),
             DeletionMove(cell_o, species=['O'], seed=seed_del_o)]
    cells = [cell_o]
    species = ['O']
    if EXCHANGE_AG:
        moves += [InsertionMove(cell_ag, species=['Ag'],
                                seed=seed_ins_ag),
                  DeletionMove(cell_ag, species=['Ag'], seed=seed_del_ag)]
        cells.append(cell_ag)
        species.insert(0, 'Ag')
    move_selector = MoveSelector([1] * len(moves), moves, seed=seed_sel)

    nx, ny, nlayers = SLAB_SIZE
    tag = f'Ag{nx * ny * nlayers}_111_dmuO_{DELTA_MU_O}_{OPTIMIZER}_fmax{FMAX}'

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=cells,
        calculator=calculator,
        mu={'Ag': mu_ag, 'O': mu_o},
        species=species,
        units_type='metal',
        temperature=TEMPERATURE,
        move_selector=move_selector,
        random_seed=seed_ens,
        outfile=f'gcmc_{tag}.out',
        traj_file=f'gcmc_{tag}.xyz',
        outfile_write_interval=WRITE_INTERVAL,
        trajectory_write_interval=WRITE_INTERVAL,
    )
    gcmc.run(STEPS)


if __name__ == '__main__':
    main()
