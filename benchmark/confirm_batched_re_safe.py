"""Confirmation: safe settings (energy_only + chunk_size) make a batched RE that
OOMs at whole-batch instead fit and run end-to-end.

12 replicas of a 1150-atom Ag octahedral NP (~13.8k atoms total) — whole-batch
energy eval OOMs on the 32 GB card; chunk_size=1 caps peak at one replica. Runs
a few real GCMC steps (O insertion/deletion around the NP) and logs peak memory.

Run in the `alchemi` conda env on the GPU box:
    python benchmark/confirm_batched_re_safe.py
"""
import os

import numpy as np
import torch
from ase.cluster import Octahedron

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402
from mcpy.ensembles import BatchedReplicaExchange  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confirm_run')
N_REPLICAS = 16
TEMPERATURES = list(np.linspace(300, 800, N_REPLICAS))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    base_atoms = Octahedron('Ag', 12, 1)
    print(f'{N_REPLICAS} replicas x {len(base_atoms)} atoms '
          f'= {N_REPLICAS * len(base_atoms)} atoms total')

    # Energy-only + chunk_size=1 are the bit-exact memory levers from the sweep.
    calc = AlchemiCalculator(device='cuda', compile_model=False,
                             energy_only=True, chunk_size=1)

    mus = {'Ag': -2.99, 'O': -4.91 - 0.5}
    ss = np.random.SeedSequence(0)
    move_seeds = [int(s) for s in ss.generate_state(2 * N_REPLICAS, dtype=np.uint32)]

    def gcmc_factory(T, rank):
        atoms = base_atoms.copy()
        scell = SphericalCell(atoms, vacuum=3.0, species_radii={'Ag': 2.947, 'O': 0},
                              mc_sample_points=5_000)
        s = move_seeds[2 * rank:2 * (rank + 1)]
        move_selector = MoveSelector(
            [1, 1],
            [DeletionMove(scell, species=['O'], seed=s[0]),
             InsertionMove(scell, species=['O'], min_insert=0.5, seed=s[1])],
        )
        return GrandCanonicalEnsemble(
            atoms=atoms, cells=[scell], calculator=calc, mu=mus,
            units_type='metal', species=['O'], temperature=T,
            move_selector=move_selector,
            outfile=os.path.join(OUTDIR, f'rank{rank}.out'),
            trajectory_write_interval=100, outfile_write_interval=100,
            traj_file=os.path.join(OUTDIR, f'rank{rank}.xyz'),
        )

    pt = BatchedReplicaExchange(
        gcmc_factory, calculator=calc, temperatures=TEMPERATURES,
        gcmc_steps=10, exchange_interval=5, write_out_interval=10, seed=1,
        outfile=os.path.join(OUTDIR, 'replica_exchange.log'),
    )

    # 1. Show whole-batch (no chunking) OOMs at this size. Passing chunk_size=None
    # inherits the instance default (1), so force whole-batch via the instance attr.
    atoms_list = [r.atoms for r in pt.replicas]
    calc.chunk_size = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        calc.get_potential_energies(atoms_list)
        torch.cuda.synchronize()
        whole = torch.cuda.max_memory_allocated() / 1024**2
        print(f'whole-batch eval fit at {whole:.0f} MB (did not OOM)')
    except torch.cuda.OutOfMemoryError:
        print('whole-batch eval OOMs (as expected)')
        torch.cuda.empty_cache()
    calc.chunk_size = 1

    # 2. Run the real RE with chunk_size=1; log peak memory.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pt.run()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    energies = np.array([r.E_old for r in pt.replicas])
    counts = [sum(1 for s in r.atoms.get_chemical_symbols() if s == 'O')
              for r in pt.replicas]
    print(f'RE (chunk_size=1, energy_only) peak = {peak:.0f} MB')
    print(f'final energies finite: {bool(np.all(np.isfinite(energies)))}')
    print(f'O counts per replica: {counts}')


if __name__ == '__main__':
    main()
