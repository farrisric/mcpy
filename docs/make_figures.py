"""Generate the documentation figures in docs/_static.

Usage (from the repository root)::

    python docs/make_figures.py

Two kinds of figures are produced:

1. Drawn from scratch (geometry only, no MLIP): the cell-geometry overview
   and the free-volume schematic.
2. Measured from a real run: a 500-step EMT GCMC of O on Ag(111) (the same
   setup as ``docs/examples/gcmc_simulations.rst``) provides the
   run-evolution curves and the snapshot strip.

Real MACE figures (phase diagrams, coverage traces) are extracted from the
executed notebooks by ``extract_notebook_figures()``.
"""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / 'docs' / '_static'

# Okabe-Ito, colorblind-safe
BLUE = '#0072B2'
SKY = '#56B4E9'
ORANGE = '#E69F00'
VERMILION = '#D55E00'
GREEN = '#009E73'
PURPLE = '#CC79A7'

ATOM_COLORS = {'Ag': '#BFBFBF', 'O': VERMILION, 'Al': '#8FA8C8', 'Cu': '#C87B4B', 'Pd': '#2E7E85'}
ATOM_RADII = {'Ag': 1.45, 'O': 0.66, 'Al': 1.21, 'Cu': 1.32, 'Pd': 1.39}

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def draw_atoms_2d(ax, atoms, plane='xz', alpha=1.0):
    """Draw atoms as circles projected on a Cartesian plane."""
    i, j = {'xz': (0, 2), 'xy': (0, 1), 'yz': (1, 2)}[plane]
    depth = ({0, 1, 2} - {i, j}).pop()
    order = np.argsort(atoms.positions[:, depth])
    for a in order:
        atom = atoms[a]
        ax.add_patch(Circle(
            (atom.position[i], atom.position[j]), ATOM_RADII[atom.symbol],
            facecolor=ATOM_COLORS[atom.symbol], edgecolor='black',
            linewidth=0.6, alpha=alpha, zorder=2))
    ax.set_aspect('equal')


def shade(ax, patch):
    patch.set_facecolor(SKY)
    patch.set_alpha(0.30)
    patch.set_edgecolor(BLUE)
    patch.set_linewidth(1.4)
    patch.set_linestyle('--')
    patch.set_zorder(3)
    ax.add_patch(patch)


def _equalize_panels(axes):
    """Give every panel the same data-window size so the boxes align."""
    boxes = []
    for ax in axes:
        ax.autoscale_view()
        bb = ax.dataLim
        boxes.append((bb.x0, bb.x1, bb.y0, bb.y1))
    w = max(x1 - x0 for x0, x1, _, _ in boxes) * 1.08
    h = max(y1 - y0 for _, _, y0, y1 in boxes) * 1.08
    for ax, (x0, x1, y0, y1) in zip(axes, boxes):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(cx - w / 2, cx + w / 2)
        ax.set_ylim(cy - h / 2, cy + h / 2)


def fig_cells():
    """Four cell geometries, side view, active region shaded."""
    from ase.build import bulk, fcc111
    from ase.cluster import Octahedron

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))

    # A: Cell -- whole periodic box
    ax = axes[0]
    box = bulk('Ag', 'fcc', a=4.085, cubic=True).repeat((4, 4, 4))
    draw_atoms_2d(ax, box)
    shade(ax, Rectangle((0, 0), box.cell[0, 0], box.cell[2, 2]))
    ax.set_title('Cell\nwhole periodic box')

    # B: CustomCell -- slab window
    ax = axes[1]
    slab = fcc111('Ag', a=4.085, size=(4, 4, 3), vacuum=8.0, periodic=True)
    draw_atoms_2d(ax, slab)
    bottom_z = slab.positions[:, 2].max() + 0.5
    x0, x1 = slab.positions[:, 0].min() - 2, slab.positions[:, 0].max() + 2
    shade(ax, Rectangle((x0, bottom_z), x1 - x0, 5.0))
    ax.set_title('CustomCell\nslab window above the surface')

    # C: SphericalCell -- nanoparticle
    ax = axes[2]
    nano = Octahedron('Ag', 5, 1)
    nano.positions -= nano.get_center_of_mass()
    draw_atoms_2d(ax, nano)
    R = np.linalg.norm(nano.positions, axis=1).max() + 3.0
    shade(ax, Circle((0, 0), R))
    ax.set_title('SphericalCell\nparticle plus vacuum margin')

    # D: DomeCell -- supported particle
    ax = axes[3]
    support = fcc111('Al', a=4.05, size=(8, 8, 2), vacuum=2.0, periodic=True)
    part = Octahedron('Ag', 4, 1)
    part.positions -= part.get_center_of_mass()
    surface_z = support.positions[:, 2].max()
    center_xy = support.cell.sum(axis=0)[:2] / 2
    lift = surface_z + ATOM_RADII['Al'] + ATOM_RADII['Ag'] - part.positions[:, 2].min()
    part.positions += (*center_xy, lift)
    combined = support + part
    draw_atoms_2d(ax, combined)
    c = part.get_center_of_mass()
    R = np.linalg.norm(part.positions - c, axis=1).max() + 3.0
    theta = np.linspace(0, 2 * np.pi, 200)
    arc = np.c_[c[0] + R * np.cos(theta), c[2] + R * np.sin(theta)]
    arc = arc[arc[:, 1] >= surface_z]
    arc = arc[np.argsort(np.arctan2(arc[:, 1] - c[2], arc[:, 0] - c[0]))]
    shade(ax, Polygon(arc, closed=True))
    ax.set_title('DomeCell\ndome clipped at the support')

    _equalize_panels(axes)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.savefig(STATIC / 'fig_cells.png')
    plt.close(fig)


def fig_free_volume():
    """Free-volume estimator schematic: exclusion disks and sample points."""
    from ase.build import fcc111

    rng = np.random.default_rng(7)
    slab = fcc111('Ag', a=4.085, size=(4, 1, 3), vacuum=8.0, periodic=False)
    r_ag = 2.75
    bottom_z = slab.positions[:, 2].max() + 0.5
    height = 5.0
    x0, x1 = -1.0, slab.positions[:, 0].max() + 2.0

    pts = np.c_[rng.uniform(x0, x1, 500), rng.uniform(bottom_z, bottom_z + height, 500)]
    xz = slab.positions[:, [0, 2]]
    occ = (np.linalg.norm(pts[:, None] - xz[None], axis=2) <= r_ag).any(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    draw_atoms_2d(ax, slab)
    for p in xz[xz[:, 1] > slab.positions[:, 2].max() - 1]:
        ax.add_patch(Circle(p, r_ag, facecolor='none', edgecolor='gray',
                            linestyle=':', linewidth=1.0, zorder=1))
    shade(ax, Rectangle((x0, bottom_z), x1 - x0, height))
    ax.scatter(*pts[~occ].T, s=8, color=GREEN, zorder=4, label='free')
    ax.scatter(*pts[occ].T, s=8, color=VERMILION, zorder=4, label='occupied')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xlim(x0 - 3.5, x1 + 3.5)
    ax.set_ylim(slab.positions[:, 2].min() - 2, bottom_z + height + 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Random points in the cell window, classified by the '
                 f'exclusion radius ($r_\\mathrm{{Ag}}$ = {r_ag} Å)')
    fig.savefig(STATIC / 'fig_free_volume.png')
    plt.close(fig)


def run_emt_gcmc(workdir):
    """The EMT smoke-test run from docs/examples/gcmc_simulations.rst."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms

    from mcpy.calculators import BaseCalculator
    from mcpy.cell import CustomCell
    from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
    from mcpy.moves import InsertionMove, DeletionMove
    from mcpy.moves.move_selector import MoveSelector

    atoms = fcc111('Ag', a=4.085, size=(3, 3, 3), vacuum=8.0, periodic=True)
    atoms.set_constraint(FixAtoms(indices=[a.index for a in atoms if a.tag == 3]))
    cell = CustomCell(atoms, custom_height=5.0,
                      bottom_z=atoms.positions[:, 2].max() + 0.5,
                      species_radii={'Ag': 2.75, 'O': 0.0})
    calculator = BaseCalculator(calculator=EMT(), steps=20, fmax=0.1)
    ss = np.random.SeedSequence(0)
    s1, s2 = (int(x) for x in ss.generate_state(2, dtype=np.uint32))
    moves = MoveSelector(
        [1, 1],
        [InsertionMove(cell, species=['O'], min_insert=0.5, seed=s1),
         DeletionMove(cell, species=['O'], seed=s2)])
    # EMT's O-Ag binding is shallow (about -0.05 eV per adsorbed O), so the
    # chemical potential must sit near that scale for insertions to accept.
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=calculator,
        mu={'O': -0.2}, units_type='metal', species=['O'], temperature=500.0,
        move_selector=moves,
        outfile=str(workdir / 'gcmc_demo.out'),
        traj_file=str(workdir / 'gcmc_demo.xyz'))
    gcmc.run(steps=1000)
    return workdir / 'gcmc_demo.xyz'


def fig_run(traj_path):
    """Run-evolution curves and snapshot strip from the EMT trajectory."""
    from ase.io import read

    traj = read(traj_path, index=':')
    n_o = [sum(a.symbol == 'O' for a in f) for f in traj]
    energy = [f.info.get('energy', f.get_potential_energy()) for f in traj]
    steps = np.arange(len(traj))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(steps, n_o, color=BLUE, linewidth=1.4)
    a1.set_ylabel(r'$N_\mathrm{O}$ on slab')
    a2.plot(steps, energy, color=ORANGE, linewidth=1.4)
    a2.set_ylabel('Energy (eV)')
    a2.set_xlabel('GCMC step')
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(STATIC / 'fig_run_evolution.png')
    plt.close(fig)

    picks = [0, len(traj) // 4, len(traj) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, k in zip(axes, picks):
        frame = traj[k]
        draw_atoms_2d(ax, frame, plane='xy')
        ax.margins(0.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        n = sum(a.symbol == 'O' for a in frame)
        ax.set_title(f'step {k}   $N_\\mathrm{{O}}$ = {n}')
    fig.savefig(STATIC / 'fig_run_snapshots.png')
    plt.close(fig)


def extract_notebook_figures():
    """Copy the real MACE figures out of the executed notebooks."""
    wanted = {
        # (notebook, embedded-PNG index): target file
        ('phase_diagram.ipynb', 0): 'fig_phase_diagram_ag_o.png',
        ('co_on_cupd_replica_exchange.ipynb', 1): 'fig_phase_diagram_co_cupd.png',
        ('co_on_cupd_replica_exchange.ipynb', 0): 'fig_co_coverage_raw.png',
        ('gcmc_basics_lj.ipynb', 1): 'fig_lj_equilibration.png',
    }
    for (nb_name, want_idx), target in wanted.items():
        data = json.load(open(ROOT / 'notebooks' / nb_name))
        idx = 0
        for cell in data['cells']:
            for out in cell.get('outputs', []):
                png = out.get('data', {}).get('image/png')
                if png is None:
                    continue
                if idx == want_idx:
                    (STATIC / target).write_bytes(base64.b64decode(png))
                idx += 1

    # The CO coverage figure pairs the per-replica trace with a rough demo
    # isotherm; only the trace belongs in the docs. Crop the left panel.
    from PIL import Image
    im = Image.open(STATIC / 'fig_co_coverage_raw.png')
    im.crop((0, 0, int(im.width * 0.48), im.height)).save(STATIC / 'fig_co_coverage.png')
    (STATIC / 'fig_co_coverage_raw.png').unlink()


if __name__ == '__main__':
    import tempfile

    STATIC.mkdir(exist_ok=True)
    fig_cells()
    fig_free_volume()
    extract_notebook_figures()
    with tempfile.TemporaryDirectory() as td:
        traj = run_emt_gcmc(Path(td))
        fig_run(traj)
    print(f'Figures written to {STATIC}')
