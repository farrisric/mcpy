"""Parity check: torch.compile on the local-checkpoint path of ``_load_model``.

``_load_model`` historically skipped ``torch.compile`` when loading a local
``.model`` file (the branch fine-tuned checkpoints take), so production GCMC
ran uncompiled — a measured ~2x on the forward pass. This validates the
compiled local path against the uncompiled one:

  1. the private nvalchemi helper the compile step relies on is importable
     (pins ``_patch_e3nn_irrep_len_for_compile`` so an upgrade that renames
     it fails loudly here, not mid-run),
  2. single-point energy/forces parity on identical geometries (same batch,
     same neighbor list; only the compiled forward differs),
  3. relaxed-energy parity through ``get_potential_energy`` on perturbed
     structures, including one with FixAtoms,
  4. compiled relaxation handles a changing atom count (the GCMC access
     pattern) without recompiling to a wrong result.

Run in the ``alchemi`` conda env on the GPU box::

    python benchmark/verify_compile_parity.py [path/to/checkpoint.model]

Without an argument, the MACE-MP medium-mpa-0 checkpoint is downloaded to the
local cache and its cached *path* is used, which exercises the same local-file
branch as a fine-tuned model.
"""
import sys

import numpy as np
from ase.cluster import Octahedron
from ase.constraints import FixAtoms

from mcpy.utils.logging import configure as configure_logging

configure_logging()

import torch  # noqa: E402

from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.calculators.alchemi_f_calculator import NeighborListHook  # noqa: E402
from mcpy.calculators._alchemi_common import (  # noqa: E402
    _build_nl,
    _make_multi_batch,
)

FMAX = 0.05
STEPS = 500
SP_ENERGY_TOL = 5e-3    # eV, single-point |dE| on ~10^3-10^4 eV totals (fp32)
SP_FORCE_TOL = 5e-3     # eV/A, single-point max |dF| component
RELAX_ENERGY_TOL = 0.02  # eV, same tolerance as verify_compact_parity.py


def resolve_checkpoint() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    from mace.calculators.foundations_models import download_mace_mp_checkpoint
    return download_mace_mp_checkpoint('medium-mpa-0')


def make_structures():
    """Three perturbed NPs: rattled, one with an O adatom, one with FixAtoms."""
    rng = np.random.default_rng(7)
    structs = []
    for k in range(3):
        atoms = Octahedron('Ag', 8, 1)
        atoms.rattle(stdev=0.05, seed=100 + k)
        if k == 1:  # O adatom just outside the surface, random direction
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            center = atoms.positions.mean(axis=0)
            r = np.max(np.linalg.norm(atoms.positions - center, axis=1))
            atoms += type(atoms)('O', positions=[center + d * (r + 1.8)])[0:1]
        if k == 2:  # freeze the innermost 30 atoms
            center = atoms.positions.mean(axis=0)
            order = np.argsort(np.linalg.norm(atoms.positions - center, axis=1))
            atoms.set_constraint(FixAtoms(indices=order[:30].tolist()))
        structs.append(atoms)
    return structs


def single_point(calc, atoms_list):
    """One forward per structure list; per-graph energies and stacked forces."""
    batch = _make_multi_batch(atoms_list, calc.device, calc.dtype)
    nl_hook = NeighborListHook(calc._nl_config, max_neighbors=calc.max_neighbors)
    _build_nl(batch, nl_hook)
    out = calc.model(batch)
    e = out['energy'].detach().cpu().numpy().reshape(-1)
    f = out['forces'].detach().cpu().numpy()
    return e, f


def main():
    checkpoint = resolve_checkpoint()
    print(f'checkpoint: {checkpoint}')

    # 1. pin the private helper the compile branch imports
    from nvalchemi.models.mace import _patch_e3nn_irrep_len_for_compile  # noqa: F401
    print('pinned: nvalchemi.models.mace._patch_e3nn_irrep_len_for_compile')

    calc_ref = AlchemiFCalculator(checkpoint, steps=STEPS, fmax=FMAX,
                                  device='cuda', compile_model=False)
    calc_cmp = AlchemiFCalculator(checkpoint, steps=STEPS, fmax=FMAX,
                                  device='cuda', compile_model=True)
    structs = make_structures()
    print(f'{len(structs)} structures: '
          + ', '.join(f'{len(a)}at' + ('+Fix' if a.constraints else '')
                      for a in structs))

    failures = []

    # 2. single-point parity on identical geometries
    e_ref, f_ref = single_point(calc_ref, structs)
    e_cmp, f_cmp = single_point(calc_cmp, structs)
    d_e = np.abs(e_cmp - e_ref)
    d_f = np.abs(f_cmp - f_ref).max()
    print(f'\nsingle-point |dE|: {" ".join(f"{x * 1e3:.3f}" for x in d_e)} meV, '
          f'max|dF|: {d_f * 1e3:.3f} meV/A')
    if (d_e > SP_ENERGY_TOL).any():
        failures.append(f'single-point |dE| max {d_e.max() * 1e3:.2f} meV '
                        f'> {SP_ENERGY_TOL * 1e3:.0f} meV')
    if d_f > SP_FORCE_TOL:
        failures.append(f'single-point max|dF| {d_f * 1e3:.2f} meV/A '
                        f'> {SP_FORCE_TOL * 1e3:.0f} meV/A')

    # 3. relaxed-energy parity (serial path, FixAtoms included)
    e_relax_ref = np.array([calc_ref.get_potential_energy(a.copy())
                            for a in structs])
    e_relax_cmp = np.array([calc_cmp.get_potential_energy(a.copy())
                            for a in structs])
    d_relax = np.abs(e_relax_cmp - e_relax_ref)
    print(f'{"graph":>5} {"E_uncompiled":>14} {"E_compiled":>14} {"|dE|":>9}')
    for i in range(len(structs)):
        print(f'{i:>5} {e_relax_ref[i]:>14.4f} {e_relax_cmp[i]:>14.4f} '
              f'{d_relax[i]:>9.4f}')
    if (d_relax > RELAX_ENERGY_TOL).any():
        failures.append(f'relaxed |dE| max {d_relax.max():.4f} eV '
                        f'> {RELAX_ENERGY_TOL} eV')

    # 4. changing atom count through the compiled model (GCMC access pattern)
    grown = structs[0].copy()
    grown.rattle(stdev=0.05, seed=1)
    center = grown.positions.mean(axis=0)
    r = np.max(np.linalg.norm(grown.positions - center, axis=1))
    grown += type(grown)('Ag', positions=[center + [0, 0, r + 2.5]])[0:1]
    e_grown_cmp = calc_cmp.get_potential_energy(grown.copy())
    e_grown_ref = calc_ref.get_potential_energy(grown.copy())
    d_grown = abs(e_grown_cmp - e_grown_ref)
    print(f'\nN+1 structure ({len(grown)}at): uncompiled {e_grown_ref:.4f} '
          f'compiled {e_grown_cmp:.4f} |dE|={d_grown:.4f} eV')
    if d_grown > RELAX_ENERGY_TOL:
        failures.append(f'N+1 relaxed |dE| {d_grown:.4f} eV '
                        f'> {RELAX_ENERGY_TOL} eV')

    # sanity: the compiled path actually wrapped the model
    inner = calc_cmp.model.model
    if not isinstance(inner, torch._dynamo.eval_frame.OptimizedModule):
        failures.append(f'compiled calculator holds {type(inner).__name__}, '
                        'expected torch OptimizedModule — compile branch '
                        'not taken')

    if failures:
        print('\nFAIL')
        for f in failures:
            print(' -', f)
        raise SystemExit(1)
    print('\nPASS')


if __name__ == '__main__':
    main()
