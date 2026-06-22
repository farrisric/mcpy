"""Largest single structure (one forward) that fits on the GPU, energy-only vs
forces-on. Run in the `alchemi` conda env.
"""
import numpy as np
import torch
from ase.build import bulk

from mcpy.calculators import AlchemiCalculator


def make_np(n, a=4.16, symbol='Ag'):
    reps = int(np.ceil((n / 4) ** (1 / 3))) + 3
    base = bulk(symbol, 'fcc', a=a, cubic=True).repeat((reps, reps, reps))
    c = base.get_positions().mean(0)
    keep = np.argsort(np.linalg.norm(base.get_positions() - c, axis=1))[:n]
    at = base[keep]
    at.center(vacuum=10.0)
    at.set_pbc(False)
    return at


def probe(calc, label):
    print(f'--- {label} ---')
    for n in [4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]:
        atoms = make_np(n)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            calc.get_potential_energies([atoms])
            torch.cuda.synchronize()
            mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f'  {n:6d} atoms : {mb:8.0f} MB')
        except torch.cuda.OutOfMemoryError:
            print(f'  {n:6d} atoms : OOM')
            torch.cuda.empty_cache()
            break


def main():
    calc = AlchemiCalculator(device='cuda', compile_model=False, energy_only=True)
    probe(calc, 'energy_only=True')
    calc.model.model_config.active_outputs.add('forces')
    probe(calc, 'forces on')


if __name__ == '__main__':
    main()
