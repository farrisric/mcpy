"""Parity check: compacted batched relaxation vs serial.

Validates the compacted batched path of ``AlchemiFCalculator`` (retire each
graph from the batch at its first convergence — the only batched path)
against:

  1. the serial single-graph path (``get_potential_energy`` per structure) —
     same first-convergence stopping rule, so agreement should be tight,
  2. the physical contract: after write-back, a fresh forward on each relaxed
     structure has max per-atom force <= fmax (excluding FixAtoms rows),
  3. FixAtoms rows are bit-identical to the input positions.

Run in the ``alchemi`` conda env on the GPU box::

    python benchmark/verify_compact_parity.py
"""
import numpy as np
from ase.cluster import Octahedron
from ase.constraints import FixAtoms

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.calculators.alchemi_f_calculator import NeighborListHook  # noqa: E402
from mcpy.calculators._alchemi_common import (  # noqa: E402
    _build_nl,
    _fixed_indices,
    _make_multi_batch,
)

FMAX = 0.05
STEPS = 500


def make_structures():
    """Six perturbed NPs: rattled, some with an O adatom, one with FixAtoms."""
    rng = np.random.default_rng(7)
    structs = []
    for k in range(6):
        atoms = Octahedron('Ag', 8, 1)
        atoms.rattle(stdev=0.05, seed=100 + k)
        if k in (3, 4):  # O adatom just outside the surface, random direction
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            center = atoms.positions.mean(axis=0)
            r = np.max(np.linalg.norm(atoms.positions - center, axis=1))
            atoms += type(atoms)('O', positions=[center + d * (r + 1.8)])[0:1]
        if k == 5:  # freeze the innermost 30 atoms
            center = atoms.positions.mean(axis=0)
            order = np.argsort(np.linalg.norm(atoms.positions - center, axis=1))
            atoms.set_constraint(FixAtoms(indices=order[:30].tolist()))
        structs.append(atoms)
    return structs


def fresh_max_forces(calc, atoms_list):
    """Fresh forward on each structure; per-graph max |F| excluding FixAtoms."""
    batch = _make_multi_batch(atoms_list, calc.device, calc.dtype)
    nl_hook = NeighborListHook(calc._nl_config, max_neighbors=calc.max_neighbors)
    _build_nl(batch, nl_hook)
    out = calc.model(batch)
    forces = out['forces'].detach().cpu().numpy()
    batch_idx = batch.batch_idx.detach().cpu().numpy()
    fmaxes = []
    offset = 0
    for i, atoms in enumerate(atoms_list):
        f = forces[batch_idx == i]
        free = np.setdiff1d(np.arange(len(atoms)), _fixed_indices(atoms))
        fmaxes.append(float(np.linalg.norm(f[free], axis=1).max()))
        offset += len(atoms)
    return np.array(fmaxes)


def main():
    calc = AlchemiFCalculator(steps=STEPS, fmax=FMAX, device='cuda',
                              chunk_size=None)
    structs = make_structures()
    print(f'{len(structs)} structures: '
          + ', '.join(f'{len(a)}at' + ('+Fix' if a.constraints else '')
                      for a in structs))

    a_comp = [a.copy() for a in structs]
    e_comp = calc.get_potential_energies(a_comp)
    steps_comp = calc.last_relax_steps

    e_serial = np.array([calc.get_potential_energy(a.copy()) for a in structs])

    print(f'\nbatch steps: compact={steps_comp}')
    print(f'{"graph":>5} {"E_compact":>12} {"E_serial":>12} {"comp-serial":>12}')
    for i in range(len(structs)):
        print(f'{i:>5} {e_comp[i]:>12.4f} {e_serial[i]:>12.4f} '
              f'{e_comp[i] - e_serial[i]:>12.4f}')

    failures = []

    # 2. force contract on the compacted write-back geometries
    fm = fresh_max_forces(calc, a_comp)
    print('\nfresh max|F| on compacted geometries: '
          + ' '.join(f'{x:.3f}' for x in fm))
    if steps_comp < STEPS and (fm > FMAX * 1.10).any():
        failures.append(f'force contract violated: max|F|={fm.max():.3f} '
                        f'> {FMAX} (+10% slack)')

    # 3. FixAtoms rows unchanged
    fixed = _fixed_indices(structs[5])
    if not np.allclose(a_comp[5].positions[fixed],
                       structs[5].positions[fixed], atol=0.0):
        failures.append('FixAtoms rows moved in compacted path')

    # 1. compact vs serial: same stopping rule -> tight agreement
    d_serial = np.abs(e_comp - e_serial)
    if (d_serial > 0.02).any():
        failures.append(f'compact vs serial max delta {d_serial.max():.4f} eV '
                        f'> 0.02 eV')

    print(f'\ncompact-vs-serial |dE|: max={d_serial.max():.4f} eV')

    if failures:
        print('\nFAIL')
        for f in failures:
            print(' -', f)
        raise SystemExit(1)
    print('\nPASS')


if __name__ == '__main__':
    main()
