"""MACE parity benchmark: mcpy molecular GCMC vs LAMMPS fix gcmc + pair_style mace.

Requires the patched ACEsuit LAMMPS build (pair_style mace, MC, MOLECULE) and a
libtorch-exported MACE model (``mace_create_lammps_model``); pass both paths.

Stages (gated):
  gate  single-point energy parity on an Ag2O probe configuration
  I     ideal-gas rigid-O2 GCMC in metal units vs the analytic Poisson result
        (validates the physical de Broglie wavelength on both sides)
  II    rigid O2 above a frozen Ag(111) slab, exchange-only, no relaxation:
        <N_mol> and <PE> at two chemical potentials

Both sides use the same float32 MACE-MPA-0 model, T = 400 K, and identical
insertion regions (orthogonal slab so LAMMPS `region block` sampling matches
CustomCell exactly). No relaxation anywhere: LAMMPS fix gcmc accepts on raw
energy differences, so the mcpy side evaluates single-point energies only.

Usage:
    python benchmark/mace_gcmc_parity.py --lmp <lmp> --model <model.pt> \
        --python-model <mace model> --stage all
"""
import argparse
import math
import os

import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule

from _parity_common import ZeroCalc, block_stats, compare, run_lammps
from mcpy.cell import Cell, CustomCell
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import (MoleculeDeletionMove, MoleculeInsertionMove,
                        MoveSelector)

T = 400.0
KB = 8.617333e-5
BETA = 1.0 / (KB * T)

# Directory containing the libtorch shared libraries the patched LAMMPS
# links against; override for a different environment.
TORCH_LIB = os.environ.get(
    'MACE_PARITY_TORCH_LIB',
    os.path.expanduser(
        '~/miniconda3/envs/mcpy/lib/python3.14/site-packages/torch/lib'))


def _libtorch_env():
    """LAMMPS pair_style mace needs libtorch on the loader path."""
    return {'LD_LIBRARY_PATH': TORCH_LIB + ':' + os.environ.get('LD_LIBRARY_PATH', '')}


# --------------------------------------------------------------------------
# mcpy side
# --------------------------------------------------------------------------

class MaceSinglePoint:
    """Bare single-point MACE energy (no relaxation), float32 on GPU."""

    def __init__(self, model_path, device='cuda'):
        from mace.calculators import MACECalculator
        self.calc = MACECalculator(model_paths=model_path, device=device,
                                   default_dtype='float32')

    def get_potential_energy(self, atoms):
        atoms.calc = self.calc
        return float(atoms.get_potential_energy())


def run_mcpy(atoms, cells, calc, moves, mu, molecules, steps, seed):
    ms = MoveSelector([1] * len(moves), moves, seed=seed + 1, n_moves=1)
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=cells, units_type='metal', calculator=calc,
        mu=mu, species=[], temperature=T, move_selector=ms,
        molecules=molecules, random_seed=seed, traj_file=None, outfile=None)
    n_series = np.empty(steps)
    e_series = np.empty(steps)
    for i in range(steps):
        g.do_gcmc_step()
        n_series[i] = len(g.atoms)
        e_series[i] = g.E_old
    return n_series, e_series


# --------------------------------------------------------------------------
# geometry shared by both sides
# --------------------------------------------------------------------------

def o2_template():
    return molecule('O2')  # d = 1.21 A, mass 2 x 15.999


def slab_atoms():
    atoms = fcc111('Ag', a=4.165, size=(4, 4, 3), orthogonal=True,
                   periodic=True, vacuum=10.0)
    return atoms


def write_slab_data(atoms, path):
    """Minimal LAMMPS data file, atom_style molecular, types: 1=Ag, 2=O."""
    # Round box bounds to the same precision used for the gcmc region string
    # so the region hi bounds compare as inside the box.
    cell = np.round(atoms.cell.lengths(), 6)
    with open(path, 'w') as f:
        f.write('# Ag(111) slab, types 1=Ag 2=O\n\n')
        f.write(f'{len(atoms)} atoms\n2 atom types\n\n')
        f.write(f'0.0 {cell[0]:.6f} xlo xhi\n')
        f.write(f'0.0 {cell[1]:.6f} ylo yhi\n')
        f.write(f'0.0 {cell[2]:.6f} zlo zhi\n\n')
        f.write('Masses\n\n1 107.8682\n2 15.999\n\n')
        f.write('Atoms # molecular\n\n')
        for i, p in enumerate(atoms.positions):
            f.write(f'{i + 1} 0 1 {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n')


O2_MOL = """# rigid O2 (types: 2=O in the slab type map; no bonds)

2 atoms

Coords

1 0.0 0.0 0.0
2 0.0 0.0 {d:.6f}

Types

1 {t}
2 {t}

Masses

1 15.999
2 15.999
"""


# --------------------------------------------------------------------------
# gate: single-point parity
# --------------------------------------------------------------------------

def gate(outdir, lmp, model, py_calc):
    print('== Gate: single-point energy parity (Ag2O probe, float32) ==')
    a = Atoms('Ag2O', positions=[[5, 5, 5], [7, 5, 5], [6, 6.4, 5]],
              cell=[10, 10, 10], pbc=True)
    e_py = py_calc.get_potential_energy(a.copy())
    deck = f"""units metal
atom_style atomic
boundary p p p
region box block 0 10 0 10 0 10
create_box 2 box
mass 1 107.8682
mass 2 15.999
pair_style mace
pair_coeff * * {model} Ag O
create_atoms 1 single 5.0 5.0 5.0
create_atoms 1 single 7.0 5.0 5.0
create_atoms 2 single 6.0 6.4 5.0
thermo_style custom step atoms pe
run 0
"""
    th = run_lammps(deck, outdir, 'gate', lmp, extra_env=_libtorch_env())
    e_l = th['pe'][-1]
    diff = abs(e_py - e_l)
    ok = diff < 5e-4
    print(f'  python {e_py:.6f} eV  lammps {e_l:.6f} eV  |d|={diff:.2e}  '
          f'{"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------
# stage I: ideal-gas rigid O2, metal units, vs analytic
# --------------------------------------------------------------------------

def stage1(outdir, lmp, steps_mcpy, lmp_steps):
    print('== Stage I: ideal rigid-O2 gas (metal units) vs analytic ==')
    mu = -0.34
    box = 15.0
    mass = 2 * 15.999
    lam = (4.13567e-15 / math.sqrt(
        2 * math.pi * mass * 1.66053906660e-27 * KB * T
    )) * math.sqrt(1.60218e-19) * 1e10
    n_exact = box ** 3 * math.exp(BETA * mu) / lam ** 3
    print(f'  lambda(O2, {T:.0f} K) = {lam:.6f} A, analytic <N_mol> = {n_exact:.2f}')

    template = o2_template()
    atoms = Atoms(cell=[box] * 3, pbc=True)
    cell = Cell(atoms)
    moves = [MoleculeInsertionMove(cell, template, 'O2', seed=41),
             MoleculeDeletionMove(cell, template, 'O2', seed=42)]
    n_ser, _ = run_mcpy(atoms, [cell], ZeroCalc(), moves, {'O2': mu},
                        {'O2': template}, steps_mcpy, seed=43)
    m, s = block_stats(n_ser / 2.0)
    ok_mcpy = abs(m - n_exact) < 2 * s
    print(f'  mcpy   <N_mol> = {m:.3f} ± {s:.3f}  '
          f'({abs(m - n_exact) / s:.2f} sigma)  {"PASS" if ok_mcpy else "FAIL"}')

    with open(os.path.join(outdir, 'o2_ideal.mol'), 'w') as f:
        f.write(O2_MOL.format(d=template.get_distance(0, 1), t=1))
    deck = f"""units metal
atom_style molecular
boundary p p p
region box block 0 {box} 0 {box} 0 {box}
create_box 1 box
mass 1 15.999
pair_style zero 3.0
pair_coeff * *
molecule o2mol o2_ideal.mol
fix g all gcmc 1 20 0 0 4242 {T} {mu} 0.0 mol o2mol
thermo 5
thermo_style custom step atoms pe
run {lmp_steps}
"""
    th = run_lammps(deck, outdir, 'stage1', lmp, extra_env=_libtorch_env())
    ml, sl = block_stats(th['atoms'] / 2.0)
    ok_lmp = abs(ml - n_exact) < 2 * sl
    print(f'  lammps <N_mol> = {ml:.3f} ± {sl:.3f}  '
          f'({abs(ml - n_exact) / sl:.2f} sigma)  {"PASS" if ok_lmp else "FAIL"}')
    ok = ok_mcpy and ok_lmp
    print(f'  Stage I: {"PASS" if ok else "FAIL"}')
    return ok


# --------------------------------------------------------------------------
# stage II: rigid O2 above frozen Ag(111), exchange-only, no relaxation
# --------------------------------------------------------------------------

def stage2(outdir, lmp, model, py_calc, steps_mcpy, lmp_attempts, deltas):
    print('== Stage II: rigid O2 above frozen Ag(111), no relaxation ==')
    template = o2_template()
    e_rigid = py_calc.get_potential_energy(
        Atoms('O2', positions=template.positions, cell=[20, 20, 20], pbc=True))
    print(f'  E_rigid(O2, isolated) = {e_rigid:.4f} eV')

    slab = slab_atoms()
    ztop = slab.positions[:, 2].max()
    z0, height = ztop + 1.5, 6.0
    lx, ly = slab.cell.lengths()[:2]
    print(f'  region: z in [{z0:.3f}, {z0 + height:.3f}], '
          f'V = {lx * ly * height:.1f} A^3, slab atoms: {len(slab)}')

    write_slab_data(slab, os.path.join(outdir, 'slab.data'))
    with open(os.path.join(outdir, 'o2.mol'), 'w') as f:
        f.write(O2_MOL.format(d=template.get_distance(0, 1), t=2))

    all_ok = True
    for delta in deltas:
        mu = e_rigid + delta
        print(f'  delta_mu={delta}  (mu={mu:.4f} eV)')
        atoms = slab.copy()
        cell = CustomCell(atoms, custom_height=height, bottom_z=z0,
                          species_radii={'Ag': 0.0, 'O': 0.0})
        moves = [MoleculeInsertionMove(cell, template, 'O2', seed=51),
                 MoleculeDeletionMove(cell, template, 'O2', seed=52)]
        n_ser, e_ser = run_mcpy(atoms, [cell], py_calc, moves, {'O2': mu},
                                {'O2': template}, steps_mcpy, seed=53)
        n_slab = len(slab)
        mn, sn = block_stats((n_ser - n_slab) / 2.0)
        me, se = block_stats(e_ser)

        nsteps = lmp_attempts // 10
        deck = f"""units metal
atom_style molecular
boundary p p p
read_data slab.data
pair_style mace
pair_coeff * * {model} Ag O
molecule o2mol o2.mol
region gcmc_reg block 0 {round(lx, 6):.6f} 0 {round(ly, 6):.6f} {z0:.6f} {z0 + height:.6f}
group exch empty
fix g exch gcmc 1 10 0 0 888 {T} {mu} 0.0 mol o2mol region gcmc_reg full_energy
thermo 2
thermo_style custom step atoms pe
run {nsteps}
"""
        th = run_lammps(deck, outdir, f'stage2_d{delta}', lmp, extra_env=_libtorch_env())
        mln, sln = block_stats((th['atoms'] - n_slab) / 2.0)
        mle, sle = block_stats(th['pe'])

        okn, _ = compare('<Nmol>', mn, sn, mln, sln, 2, width=12)
        oke, _ = compare('<PE>', me, se, mle, sle, 3, width=12)
        all_ok &= okn and oke
    print(f'  Stage II: {"PASS" if all_ok else "FAIL"}')
    return all_ok


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lmp', required=True, help='patched lmp binary (pair mace)')
    p.add_argument('--model', required=True, help='libtorch-exported MACE .pt')
    p.add_argument('--python-model', required=True,
                   help='original MACE model file for the python side')
    p.add_argument('--stage', default='all', choices=['all', 'gate', '1', '2'])
    p.add_argument('--mcpy-steps', type=int, default=30_000)
    p.add_argument('--lmp-steps', type=int, default=20_000,
                   help='LAMMPS timesteps for stage I (20 exchanges each)')
    p.add_argument('--lmp-attempts', type=int, default=30_000,
                   help='LAMMPS exchange attempts for stage II')
    p.add_argument('--deltas', type=float, nargs='+', default=[-0.36, -0.30])
    p.add_argument('--outdir', default='benchmark/mace_parity')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    py_calc = MaceSinglePoint(args.python_model)

    if args.stage in ('all', 'gate'):
        if not gate(args.outdir, args.lmp, os.path.abspath(args.model), py_calc):
            raise SystemExit('Gate FAILED: potentials differ, aborting.')
        if args.stage == 'gate':
            return

    if args.stage in ('all', '1'):
        if not stage1(args.outdir, args.lmp, args.mcpy_steps, args.lmp_steps):
            raise SystemExit('Stage I FAILED: lambda/mu conventions differ.')
        if args.stage == '1':
            return

    if args.stage in ('all', '2'):
        stage2(args.outdir, args.lmp, os.path.abspath(args.model), py_calc,
               args.mcpy_steps, args.lmp_attempts, args.deltas)


if __name__ == '__main__':
    main()
