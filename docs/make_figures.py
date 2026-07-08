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

# EMT demo chemical potential. EMT's toy O-Ag energetics are bistable
# (near-empty below ~ -0.25 eV, multilayer condensation above ~ -0.2 eV),
# so the demo drives at -0.2 eV and caps the O population at one monolayer
# with max_atoms; deletions still fire, so the plateau fluctuates.
MU_O_DEMO = -0.2
MAX_O_DEMO = 9  # one O per top-layer Ag on the 3x3 slab

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
    """Free-volume figure built with the tutorial's own CustomCell: points
    come from cell.get_random_point() and the free volume from the cell's
    estimator."""
    from ase.build import fcc111
    from scipy.spatial import cKDTree

    from mcpy.cell import CustomCell

    r_ag = 2.75
    height = 5.0
    slab = fcc111('Ag', a=4.085, size=(4, 4, 3), vacuum=8.0, periodic=True)
    bottom_z = slab.positions[:, 2].max() + 0.5
    cell = CustomCell(slab, custom_height=height, bottom_z=bottom_z,
                      species_radii={'Ag': r_ag, 'O': 0.0},
                      mc_sample_points=100_000, seed=7)
    cell.calculate_volume(slab)
    v_free, v_cell = cell.get_volume(), cell.cell_volume

    pts = np.array([cell.get_random_point() for _ in range(600)])
    # classify the plotted points exactly as the cell's estimator does
    tree = cKDTree(cell._periodic_images(slab))
    dists, _ = tree.query(pts - cell.offset, k=1)
    occ = dists <= r_ag

    fig, ax = plt.subplots(figsize=(7, 4.4))
    draw_atoms_2d(ax, slab)
    top = slab.positions[slab.positions[:, 2] > bottom_z - 1.2]
    for p in top[:, [0, 2]]:
        ax.add_patch(Circle(p, r_ag, facecolor='none', edgecolor='gray',
                            linestyle=':', linewidth=1.0, zorder=1))
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    shade(ax, Rectangle((x0, bottom_z), x1 - x0, height))
    ax.scatter(pts[~occ][:, 0], pts[~occ][:, 2], s=8, color=GREEN,
               zorder=4, label='free')
    ax.scatter(pts[occ][:, 0], pts[occ][:, 2], s=8, color=VERMILION,
               zorder=4, label='occupied')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xlim(x0 - 3.5, x1 + 3.5)
    ax.set_ylim(slab.positions[:, 2].min() - 2, bottom_z + height + 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('CustomCell sampling (side view): '
                 f'$V_\\mathrm{{free}}$ = {v_free:.0f} of {v_cell:.0f} '
                 f'$\\mathrm{{\\AA^3}}$ '
                 f'({100 * (1 - v_free / v_cell):.0f}% occupied)')
    fig.savefig(STATIC / 'fig_free_volume.png')
    plt.close(fig)


def _run_one_emt_gcmc(outdir, mu, seed=0, steps=1000):
    """One capped EMT GCMC run of the tutorial setup; returns the trajectory."""
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
    ss = np.random.SeedSequence(seed)
    s1, s2 = (int(x) for x in ss.generate_state(2, dtype=np.uint32))
    moves = MoveSelector(
        [1, 1],
        [InsertionMove(cell, species=['O'], min_insert=0.5, seed=s1,
                       max_atoms=MAX_O_DEMO),
         DeletionMove(cell, species=['O'], seed=s2)])
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=calculator,
        mu={'O': mu}, units_type='metal', species=['O'], temperature=500.0,
        move_selector=moves,
        outfile=str(outdir / 'gcmc.out'),
        traj_file=str(outdir / 'gcmc.xyz'))
    gcmc.run(steps=steps)
    return outdir / 'gcmc.xyz'


def run_emt_gcmc(workdir):
    """The single-condition EMT demo run behind the run-evolution figures."""
    return _run_one_emt_gcmc(Path(workdir), mu=MU_O_DEMO, seed=0)


def fig_run(traj_path, mu_o=-0.25):
    """Run-evolution curves and snapshot strip from the EMT trajectory."""
    from ase.io import read

    traj = read(traj_path, index=':')
    n_o = np.array([sum(a.symbol == 'O' for a in f) for f in traj])
    energy = np.array([f.info.get('energy', f.get_potential_energy()) for f in traj])
    grand = energy - mu_o * n_o
    steps = np.arange(len(traj))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(steps, n_o, color=BLUE, linewidth=1.4)
    a1.set_ylabel(r'$N_\mathrm{O}$ on slab')
    a2.plot(steps, grand, color=ORANGE, linewidth=1.4)
    a2.set_ylabel(r'$\Omega = E - \mu_\mathrm{O} N_\mathrm{O}$ (eV)')
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


def fig_radii(n_trials=150):
    """Calibration figure for species_radii: relaxed nearest-neighbour
    distances of trial O insertions on Ag(111) (EMT stand-in for the
    production calculator)."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    rng = np.random.default_rng(11)
    slab0 = fcc111('Ag', a=4.085, size=(3, 3, 3), vacuum=8.0, periodic=True)
    bottom = [a.index for a in slab0 if a.tag == 3]
    z_top = slab0.positions[:, 2].max()

    distances = []
    for _ in range(n_trials):
        slab = slab0.copy()
        slab.set_constraint(FixAtoms(indices=bottom))
        pos = [rng.uniform(0, slab.cell[0, 0]), rng.uniform(0, slab.cell[1, 1]),
               z_top + rng.uniform(1.0, 3.5)]
        slab.append('O')
        slab.positions[-1] = pos
        slab.calc = EMT()
        try:
            LBFGS(slab, logfile=None).run(fmax=0.1, steps=50)
        except Exception:
            continue
        d = slab.get_distances(len(slab) - 1, range(len(slab) - 1), mic=True)
        distances.append(d.min())

    distances = np.array(distances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(distances, bins=30, color=SKY, edgecolor='white')
    r = np.percentile(distances, 5)
    ax.axvline(r, color=VERMILION, linewidth=1.6, linestyle='--',
               label=f'first peak edge $r$ = {r:.2f} Å')
    ax.set_xlabel('nearest O-Ag distance after relaxation (Å)')
    ax.set_ylabel('count')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(STATIC / 'fig_species_radii.png')
    plt.close(fig)


# On EMT's energy scale (e_o2/2 is +0.46 eV) adsorption turns favourable
# around an absolute mu_O of -0.2 eV, i.e. dmu around -0.65.
SWEEP_DMUS = (-0.75, -0.70, -0.65, -0.60, -0.55)


def run_emt_sweep(workdir):
    """The tutorial's mu sweep, EMT edition: one capped GCMC run per dmu."""
    from ase.build import molecule
    from ase.calculators.emt import EMT

    workdir = Path(workdir)
    o2 = molecule('O2', cell=[12.0] * 3)
    o2.calc = EMT()
    e_o2 = o2.get_potential_energy()
    for dmu in SWEEP_DMUS:
        outdir = workdir / f'dmu_{dmu:+.2f}'
        outdir.mkdir(parents=True, exist_ok=True)
        _run_one_emt_gcmc(outdir, mu=e_o2 / 2 + dmu, seed=int(abs(dmu) * 100))
    return e_o2


def fig_sweep_phase_diagram(sweep_dir, e_o2):
    """The tutorial's own phase-diagram call, run on the EMT demo sweep."""
    from ase.io import read

    from mcpy.utils.phase_diagram import plot_phase_diagram

    sweep_dir = Path(sweep_dir)
    clean = read(sweep_dir / f'dmu_{SWEEP_DMUS[0]:+.2f}/gcmc.xyz', index='0')
    frames = [clean] + [
        read(sweep_dir / f'dmu_{d:+.2f}/gcmc.xyz', index='500:')
        for d in SWEEP_DMUS]
    plot_phase_diagram(
        frames, adsorbate='O', metal_symbols=('Ag',), mu_ref=e_o2 / 2,
        kind='surface', T=500.0, dmu_range=(-0.7, -0.3),
        outfile=str(STATIC / 'fig_phase_diagram_emt_sweep.png'))


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
    fig_radii()
    extract_notebook_figures()
    with tempfile.TemporaryDirectory() as td:
        traj = run_emt_gcmc(Path(td))
        fig_run(traj, mu_o=MU_O_DEMO)
        e_o2 = run_emt_sweep(Path(td) / 'sweep')
        fig_sweep_phase_diagram(Path(td) / 'sweep', e_o2)
    print(f'Figures written to {STATIC}')
