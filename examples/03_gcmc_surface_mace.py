"""GCMC on an Ag(111) surface with a machine-learned potential.

A real potential on a real surface, in contrast to examples 00-02. The
settings are those of the Methods of the Ag oxidation study (Telari, Farris,
Neyman and Bruix), for one replica of its replica-exchange runs:

  * MACE foundation model mace-small-density-agnesi-stress (small AGNESI).
  * Every trial is relaxed with at most 40 optimizer steps. The study uses
    FMAX = 0.1 eV/A; the looser 0.5 here keeps a CPU run short.
  * O and Ag are both inserted and deleted, with equal probability, in a
    region near the top layer. The region starts one O-Ag distance *below*
    the top layer, so subsurface sites are reachable.
  * Fixed chemical potentials from DFT references: mu_Ag = -2.99 eV and
    mu_O = E(O2)/2 + DELTA_MU_O, with E(O2)/2 = -4.91 eV.

The study runs six such replicas (250-500 K) per DELTA_MU_O in
(-0.40, -0.60, -0.90) with replica exchange; see
``04_replica_exchange_batched.py`` for that setup.

Every setting is a constant below; edit and re-run. The potential runs
through ``MACE_F_Calculator`` (mace-torch with an ASE optimizer), so it works
on a CPU. Set DEVICE = 'cuda' to use a GPU.

Requirements: pip install -e '.[mace]'

    python examples/03_gcmc_surface_mace.py
"""
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms

from mace.calculators import mace_mp

from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector
from mcpy.utils.logging import configure as configure_logging

# --- potential -------------------------------------------------------------
# A named MACE-MP checkpoint or a path to a local model file. 'small-0b2' is
# mace-small-density-agnesi-stress (small AGNESI). mace_mp downloads a named
# checkpoint on the first run and caches it in ~/.cache/mace.
CHECKPOINT = 'small-0b2'
DEVICE = 'cpu'        # or 'cuda'
RELAX_STEPS = 40      # max optimizer steps per energy evaluation
FMAX = 0.5            # force threshold (eV/A)
OPTIMIZER = 'lbfgs'   # 'lbfgs' or 'fire'

# --- chemical potentials ---------------------------------------------------
# DFT (PBE) references, so they do not change with the checkpoint.
# mu_Ag = E(Ag, bulk) + dG_Ag(T) = -2.82 - 0.163 eV, from thermochemical
# tables, and kept the same at every temperature.
MU_AG = -2.99
# E(O2)/2 from E(Ag2O) - 2 E(Ag) + dHf(Ag2O), which avoids the PBE O2 error.
MU_O_REF = -4.91

# --- insertion radii -------------------------------------------------------
# O-Ag radius (A), from the first maximum of the O-Ag radial distribution.
# Used twice: as the exclusion radius an incoming O sees around Ag, and as how
# far below the top layer the exchange region starts.
R_O_AG = 2.11
# An incoming Ag needs a larger radius. This is why each species gets its own
# cell over the same region.
R_AG_AG = 2.75
# Exclusion radius around an existing O (A). A minimal clearance that stops an
# insertion on top of an O atom, not a bond length.
R_O = 0.5

# --- system ----------------------------------------------------------------
AG_LATTICE = 4.159          # PBE fcc Ag lattice constant (A)
SLAB_SIZE = (4, 4, 3)       # fcc111 supercell (Nx, Ny, Nlayers) = 48 atoms
VACUUM = 8.0                # vacuum above and below the slab (A)
FIX_LAYERS = 1              # bottom layers frozen
CELL_HEIGHT = 7.0           # height of the exchange region (A)

# --- thermodynamics --------------------------------------------------------
TEMPERATURE = 500.0
DELTA_MU_O = -0.40          # the study uses -0.40, -0.60 and -0.90
EXCHANGE_AG = True          # False = O only

# --- run -------------------------------------------------------------------
# The study runs 20,000 moves per replica and samples every 100 moves.
STEPS = 20_000
WRITE_INTERVAL = 1
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

    # MACE_F_Calculator relaxes each trial structure with an ASE optimizer,
    # honouring the FixAtoms constraint.
    calculator = MACE_F_Calculator(
        mace_mp(model=CHECKPOINT, device=DEVICE),
        steps=RELAX_STEPS,
        fmax=FMAX,
        optimizer=OPTIMIZER,
    )

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
        mu={'Ag': MU_AG, 'O': MU_O_REF + DELTA_MU_O},
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
