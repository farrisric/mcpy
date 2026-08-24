"""Replica-exchange GCMC on a single GPU, with batched energy evaluation.

Parallel tempering without MPI. All replicas live in one Python process and
share one ``AlchemiFCalculator``, so every replica's trial energy for a step
is computed in a single batched forward pass. This is the supported
replica-exchange path on one GPU; the MPI ``ReplicaExchange`` needs one rank
per replica and cannot exchange molecules.

Each replica must build its OWN cells and move selector: sharing them would
share their RNG streams and acceptance counters across the ladder. That is
what ``gcmc_factory`` below is for -- it is called once per replica.

Ladder spacing is the parameter that decides whether the run works at all.
See ``docs/replica_exchange_ladder_spacing.rst``; a ladder whose neighbours
never swap samples no better than independent runs.

Every setting is a constant below; edit and re-run.

Requirements: pip install 'nvalchemi-toolkit[mace]', and a CUDA GPU.

    python examples/04_replica_exchange_batched.py
"""
import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms

from mcpy.calculators import AlchemiFCalculator
from mcpy.cell import CustomCell
from mcpy.ensembles import BatchedReplicaExchange, GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector
from mcpy.utils import derive_mu_bulk, derive_mu_gas
from mcpy.utils.logging import configure as configure_logging

# --- the ladder ------------------------------------------------------------
# A temperature ladder OR a mu ladder, never both. Neighbouring rungs must
# overlap in energy or no swap is ever accepted.
TEMPERATURES = [300, 400, 500, 600, 700, 800]
EXCHANGE_INTERVAL = 10   # GCMC steps between swap attempts
GCMC_STEPS = 200         # GCMC steps per replica, in total

# --- potential -------------------------------------------------------------
CHECKPOINT = 'mace-small-density-agnesi-stress.model'
DEVICE = 'cuda'
ENABLE_CUEQ = True       # cuEquivariance kernel fusion; verify per checkpoint
COMPILE_MODEL = True     # torch.compile: ~2x faster forward pass

# --- system ----------------------------------------------------------------
AG_LATTICE = 4.1592
SLAB_SIZE = (4, 4, 3)
VACUUM = 8.0
R_O_AG = 2.11            # exclusion radius an incoming O sees
R_AG_AG = 2.75           # exclusion radius an incoming Ag sees
CELL_HEIGHT = 7.0
MIN_INSERT = 0.5

# --- thermodynamics --------------------------------------------------------
DELTA_MU_O = -0.3        # shift applied to mu_O = E(O2)/2

# --- run -------------------------------------------------------------------
WRITE_INTERVAL = 1
SEED = 42                # one master seed; every RNG below derives from it

# Per replica: 2 cells + 4 moves + 1 selector + 1 ensemble = 8 streams.
SEEDS_PER_REPLICA = 8


def main():
    configure_logging()

    n_replicas = len(TEMPERATURES)
    seeds = np.random.SeedSequence(SEED).generate_state(
        SEEDS_PER_REPLICA * n_replicas + 1, dtype=np.uint32)
    master_seed = int(seeds[-1])

    # Shared template. Every factory call copies it and builds its own cells.
    base_atoms = fcc111('Ag', a=AG_LATTICE, size=SLAB_SIZE,
                        periodic=True, vacuum=VACUUM)
    bottom_layer = [a.index for a in base_atoms if a.tag == SLAB_SIZE[2]]
    base_atoms.set_constraint(FixAtoms(indices=bottom_layer))
    bottom_z = float(base_atoms.positions[base_atoms.get_tags() == 1, 2].max()) - R_O_AG

    # ONE model, shared across replicas: this is what makes the batched
    # forward pass possible in the first place.
    calculator = AlchemiFCalculator(
        checkpoint=CHECKPOINT,
        device=DEVICE,
        enable_cueq=ENABLE_CUEQ,
        compile_model=COMPILE_MODEL,
    )

    # Measured with the running potential rather than hardcoded, so swapping
    # the checkpoint keeps the setup self-consistent.
    mus = {'Ag': derive_mu_bulk(calculator, 'Ag', a=AG_LATTICE),
           'O': derive_mu_gas(calculator, 'O2') + DELTA_MU_O}

    def gcmc_factory(T, rank):
        """Build replica ``rank``. Called once per rung of the ladder.

        BatchedReplicaExchange calls this as ``gcmc_factory(T=..., rank=...)``
        for a temperature ladder and ``gcmc_factory(mu=..., rank=...)`` for a
        mu ladder, so the parameter name is part of the contract.
        """
        s = [int(x) for x in
             seeds[SEEDS_PER_REPLICA * rank:SEEDS_PER_REPLICA * (rank + 1)]]

        atoms = base_atoms.copy()
        atoms.set_constraint(FixAtoms(indices=bottom_layer))

        # Same region for both species, differing only in the radii, i.e. in
        # how much of that region each species may be inserted into.
        cell_ag = CustomCell(atoms, custom_height=CELL_HEIGHT, bottom_z=bottom_z,
                             species_radii={'Ag': R_AG_AG, 'O': 0.0}, seed=s[0])
        cell_o = CustomCell(atoms, custom_height=CELL_HEIGHT, bottom_z=bottom_z,
                            species_radii={'Ag': R_O_AG, 'O': 0.0}, seed=s[1])

        move_selector = MoveSelector(
            [1, 1, 1, 1],
            [InsertionMove(cell_ag, species=['Ag'], min_insert=MIN_INSERT, seed=s[2]),
             DeletionMove(cell_ag, species=['Ag'], seed=s[3]),
             InsertionMove(cell_o, species=['O'], min_insert=MIN_INSERT, seed=s[4]),
             DeletionMove(cell_o, species=['O'], seed=s[5])],
            seed=s[6],
        )

        tag = f'{atoms.get_chemical_formula()}_dmu_{DELTA_MU_O}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell_ag, cell_o],
            calculator=calculator,
            mu=mus,
            species=['Ag', 'O'],
            units_type='metal',
            temperature=T,
            move_selector=move_selector,
            random_seed=s[7],
            outfile=f'gcmc_batched_{tag}.out',
            traj_file=f'gcmc_batched_{tag}.xyz',
            outfile_write_interval=WRITE_INTERVAL,
            trajectory_write_interval=WRITE_INTERVAL,
        )

    replica_exchange = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        temperatures=TEMPERATURES,
        gcmc_steps=GCMC_STEPS,
        exchange_interval=EXCHANGE_INTERVAL,
        write_out_interval=WRITE_INTERVAL,
        seed=master_seed,
        outfile='replica_exchange_batched.log',
    )
    replica_exchange.run()


if __name__ == '__main__':
    main()
