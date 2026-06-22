"""Verify AlchemiFCalculator chunking: chunked batched FIRE must give the same
relaxed energies as whole-batch (FIRE dynamics are per-graph), at lower peak
memory. Run in the `alchemi` conda env on the GPU box.
"""
import numpy as np
import torch
from ase.cluster import Octahedron

from mcpy.calculators import AlchemiFCalculator


def rattled_nps(n, seed=0):
    out = []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = Octahedron('Ag', 9, 1)            # 483 atoms
        a.center(vacuum=5.0)
        a.set_pbc(False)
        a.positions += rng.normal(0, 0.12, a.positions.shape)  # off-equilibrium
        out.append(a)
    return out


def relax(calc, atoms_list, chunk_size):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    e = calc.get_potential_energies(atoms_list, chunk_size=chunk_size)
    torch.cuda.synchronize()
    return e, torch.cuda.max_memory_allocated() / 1024**2


def main():
    calc = AlchemiFCalculator(device='cuda', compile_model=False,
                              steps=200, fmax=0.05)
    n = 4

    # Two whole-batch relaxations from identical fresh inputs -> noise floor.
    e_w0, mem_w = relax(calc, rattled_nps(n), None)
    e_w1, _ = relax(calc, rattled_nps(n), None)
    floor = float(np.max(np.abs(e_w1 - e_w0)))

    # Chunked (one NP per forward) from identical fresh inputs.
    e_c, mem_c = relax(calc, rattled_nps(n), 1)
    max_de = float(np.max(np.abs(e_c - e_w0)))

    print(f'whole-batch peak = {mem_w:.0f} MB')
    print(f'chunk=1     peak = {mem_c:.0f} MB ({mem_c / mem_w:.1%} of whole)')
    print(f'relaxation noise floor (whole vs whole) = {floor:.3e} eV')
    print(f'chunk vs whole  max|dE|                 = {max_de:.3e} eV')
    print('VERDICT: chunked relaxation matches whole-batch'
          if max_de <= max(floor, 1e-2) else
          'VERDICT: chunked relaxation DIFFERS — investigate graph coupling')


if __name__ == '__main__':
    main()
