"""H on a Pt nanoparticle versus electrode potential, with replica exchange.

Grand-canonical H adsorption on a Pt nanoparticle, run as a potential ladder
on one GPU. Each rung holds the particle at a different electrode potential
U_SHE. The computational hydrogen electrode (CHE) turns (pH, U_SHE) into the
H chemical potential:

    mu_H = E(H2)/2 - e*U_SHE - kT*ln(10)*pH

so a lower potential means a higher mu_H and more adsorbed H. One run gives
the H coverage at every potential of the ladder.

The potential is MACE mh-1 with cuEquivariance kernels. mh-1 is a multihead
model, so the head must be pinned, and ``head=`` only works with a local
``.model`` path. mcpy repairs the mh-1 cuEq conversion bug on load
(``use_edge_irreps_first``); older mcpy gives load-varying energies here.

Ladder spacing decides whether replicas swap at all. Swaps need
beta*dmu*sigma_N ~ 1, and sigma_N grows where the isotherm is steep, so a
uniform ladder that works on a bare particle dies once it fills up. Run a
short isotherm first, then respace the ladder with
``docs/replica_exchange_ladder_spacing.rst``. The run warns at the end about
any neighbour pair that stopped swapping.

Every setting is a constant below; edit and re-run.

Requirements: pip install -e '.[alchemi]', nvalchemi from GitHub main (for
LBFGS), and a CUDA GPU.

    python examples/05_pt_nanoparticle_h_che.py
"""
import os

import numpy as np
from ase.cluster import Octahedron

from mcpy.calculators import AlchemiFCalculator
from mcpy.cell import SphericalCell
from mcpy.ensembles import BatchedReplicaExchange, GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector
from mcpy.utils import mu_h_che
from mcpy.utils.logging import configure as configure_logging

# --- the ladder ------------------------------------------------------------
# One rung per electrode potential (V vs SHE), all at the same pH and T.
# A mu ladder only: the swap rule is exact for mu ladders.
U_SHE_LADDER = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]
PH = 0.0
TEMPERATURE = 300.0
EXCHANGE_INTERVAL = 10   # GCMC steps between swap attempts
GCMC_STEPS = 500         # GCMC steps per replica, in total

# --- potential -------------------------------------------------------------
# A local .model path: head= does not work with a download alias.
CHECKPOINT = os.path.expanduser('~/.cache/mace/macemh1model')
# Heads: matpes_r2scan, mp_pbe_refit_add, spice_wB97M, oc20_usemppbe, omol,
# omat_pbe. E(H2) and the Pt-H energies must come from the same head.
HEAD = 'omat_pbe'
DEVICE = 'cuda'
ENABLE_CUEQ = True
COMPILE_MODEL = True
# LBFGS converges in ~3x fewer steps than FIRE. It needs nvalchemi newer
# than 0.2.0 (GitHub main); set 'fire' on nvalchemi 0.2.0.
OPTIMIZER = 'lbfgs'

# --- thermodynamics --------------------------------------------------------
# Shift (eV) on mu_H. The CHE helper uses bare energies; the usual CHE
# correction for H* (ZPE and entropy, +0.24 eV per adsorbed H, Norskov et al.
# 2005) is the same as lowering mu_H by 0.24 eV. Set 0.0 for bare energies.
MU_H_CORRECTION = -0.24

# --- system ----------------------------------------------------------------
# Cuboctahedron, 55 Pt atoms.
NP_LENGTH = 5
NP_CUTOFF = 2
PT_LATTICE = 3.92        # A; relaxation corrects the small misfit to mh-1
BOX_VACUUM = 12.0        # A of empty box around the particle

# --- insertion region ------------------------------------------------------
# The sphere reaches this far (A) beyond the outermost Pt.
SHELL = 2.5
# Exclusion radius an incoming H sees around Pt (A). Just below the Pt-H
# on-top bond (~1.55 A), so on-top, bridge and hollow sites stay reachable.
R_H_PT = 1.5
# Exclusion radius around an existing H (A). Stops an insertion on top of an
# H atom; it is a clearance, not a bond length.
R_H = 0.5

# --- run -------------------------------------------------------------------
WRITE_INTERVAL = 1
SEED = 42                # one master seed; every RNG below derives from it

# Per replica: 1 cell + 2 moves + 1 selector + 1 ensemble = 5 streams.
SEEDS_PER_REPLICA = 5


def main():
    configure_logging()

    n_replicas = len(U_SHE_LADDER)
    seeds = np.random.SeedSequence(SEED).generate_state(
        SEEDS_PER_REPLICA * n_replicas + 1, dtype=np.uint32)
    master_seed = int(seeds[-1])

    base_atoms = Octahedron('Pt', NP_LENGTH, cutoff=NP_CUTOFF,
                            latticeconstant=PT_LATTICE)
    base_atoms.center(vacuum=BOX_VACUUM)
    base_atoms.set_pbc(True)

    # ONE model, shared across replicas, so all trial energies of a step go
    # through one batched forward pass.
    calculator = AlchemiFCalculator(
        checkpoint=CHECKPOINT,
        head=HEAD,
        device=DEVICE,
        enable_cueq=ENABLE_CUEQ,
        compile_model=COMPILE_MODEL,
        optimizer=OPTIMIZER,
    )

    # E(H2) comes from the running calculator, so the reference always
    # matches the head in use.
    mus = [{'H': mu_h_che(calculator, ph=PH, u_she=u, temperature=TEMPERATURE)
            + MU_H_CORRECTION}
           for u in U_SHE_LADDER]

    def gcmc_factory(mu, rank):
        """Build replica ``rank``. Called once per rung of the ladder.

        BatchedReplicaExchange calls this as ``gcmc_factory(mu=..., rank=...)``
        on a mu ladder, so the parameter name is part of the contract.
        """
        s = [int(x) for x in
             seeds[SEEDS_PER_REPLICA * rank:SEEDS_PER_REPLICA * (rank + 1)]]

        atoms = base_atoms.copy()
        cell = SphericalCell(atoms, vacuum=SHELL,
                             species_radii={'Pt': R_H_PT, 'H': R_H}, seed=s[0])

        move_selector = MoveSelector(
            [1, 1],
            [InsertionMove(cell, species=['H'], seed=s[1]),
             DeletionMove(cell, species=['H'], seed=s[2])],
            seed=s[3],
        )

        tag = f'Pt{len(atoms)}_pH{PH}_U{U_SHE_LADDER[rank]:+.2f}_rank{rank}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell],
            calculator=calculator,
            mu=mu,
            species=['H'],
            units_type='metal',
            temperature=TEMPERATURE,
            move_selector=move_selector,
            random_seed=s[4],
            outfile=f'gcmc_{tag}.out',
            traj_file=f'gcmc_{tag}.xyz',
            outfile_write_interval=WRITE_INTERVAL,
            trajectory_write_interval=WRITE_INTERVAL,
        )

    replica_exchange = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        mus=mus,
        gcmc_steps=GCMC_STEPS,
        exchange_interval=EXCHANGE_INTERVAL,
        write_out_interval=WRITE_INTERVAL,
        seed=master_seed,
        outfile='replica_exchange_pt_h.log',
    )
    replica_exchange.run()


if __name__ == '__main__':
    main()
