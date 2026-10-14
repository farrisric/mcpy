"""Generate the documentation figures in docs/_static.

Usage (from the repository root)::

    python docs/make_figures.py

Three kinds of figures are produced:

1. Drawn from the cells' own geometry (no energies): the cell overview and
   the free-volume sampling of the tutorial's ``CustomCell``.
2. Measured from real CPU runs:

   * ``examples/00_hello_gcmc.py`` exactly as shipped (a Lennard-Jones fluid
     with a LAMMPS reference value) gives the run-evolution curves and the
     snapshot strip;
   * the tutorial's chemical-potential sweep with EMT in place of MACE gives
     the sweep phase diagram, through the tutorial's own
     ``plot_phase_diagram`` call;
   * relaxed EMT insertions give the ``species_radii`` histogram.

3. Copied from an executed notebook by ``extract_notebook_figures()``: the
   LJ equilibration trace.

The three MACE figures (``fig_co_coverage.png``,
``fig_phase_diagram_co_cupd.png`` and ``fig_phase_diagram_ag_o.png``) need a
GPU run of several hours. The notebooks ``co_on_cupd_replica_exchange.ipynb``
and ``phase_diagram.ipynb`` write them directly, the phase diagrams with
``plot_phase_diagram``; this script does not touch them.

The full script takes about five minutes on a laptop CPU.
"""

import base64
import json
import os
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / 'docs' / '_static'

# Okabe-Ito, colorblind-safe
BLUE = '#0072B2'
SKY = '#56B4E9'
ORANGE = '#E69F00'
VERMILION = '#D55E00'
GREY = '#7F7F7F'

# The tutorial's O-Ag exclusion radius (getting_started.rst, examples 03/04).
R_O_AG = 2.14

# The EMT edition of the tutorial sweep. EMT has no real O-Ag chemistry: its
# O coverage is bistable (near-empty below an absolute mu_O of about -0.25 eV,
# runaway multilayer growth above about -0.2 eV), so every run caps the O
# population at one O per top-layer Ag with ``max_atoms``.
SWEEP_DMUS = (-0.8, -0.6, -0.4, -0.2, 0.0)   # the tutorial's values
MAX_O_EMT = 9                                # one monolayer on the 3x3 slab

ATOM_RADII = {'Ag': 1.45, 'O': 0.66, 'Al': 1.21, 'H': 0.5}

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def atom_color(symbol):
    """ASE's jmol colour, the one plot_phase_diagram's thumbnails use, so a
    species has one colour in every figure."""
    return jmol_colors[atomic_numbers[symbol]]


def draw_atoms_2d(ax, atoms, plane='xz', alpha=1.0, color=None):
    """Draw atoms as circles projected on a Cartesian plane."""
    i, j = {'xz': (0, 2), 'xy': (0, 1), 'yz': (1, 2)}[plane]
    depth = ({0, 1, 2} - {i, j}).pop()
    order = np.argsort(atoms.positions[:, depth])
    for a in order:
        atom = atoms[a]
        ax.add_patch(Circle(
            (atom.position[i], atom.position[j]), ATOM_RADII[atom.symbol],
            facecolor=color or atom_color(atom.symbol), edgecolor='black',
            linewidth=0.6, alpha=alpha, zorder=2))
    ax.set_aspect('equal')


def shade(ax, patch):
    patch.set_facecolor(SKY)
    patch.set_alpha(0.35)
    patch.set_edgecolor(BLUE)
    patch.set_linewidth(1.4)
    patch.set_linestyle('--')
    patch.set_zorder(3)
    ax.add_patch(patch)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


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
    """The five cell types, side view, insertion region shaded."""
    from ase.build import bulk, fcc111
    from ase.cluster import Octahedron
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
    axes = axes.ravel()

    # Cell: the whole periodic box
    ax = axes[0]
    box = bulk('Ag', 'fcc', a=4.085, cubic=True).repeat((3, 3, 3))
    box.positions += 4.085 / 4   # any rigid shift is the same crystal; this one centres it
    draw_atoms_2d(ax, box)
    shade(ax, Rectangle((0, 0), box.cell[0, 0], box.cell[2, 2]))
    ax.set_title('Cell\nthe whole periodic box')

    # CustomCell: a window over the slab
    ax = axes[1]
    slab = fcc111('Ag', a=4.085, size=(4, 4, 3), vacuum=8.0, periodic=True)
    draw_atoms_2d(ax, slab)
    bottom_z = slab.positions[:, 2].max() + 0.5
    x0, x1 = slab.positions[:, 0].min() - 2, slab.positions[:, 0].max() + 2
    shade(ax, Rectangle((x0, bottom_z), x1 - x0, 5.0))
    ax.set_title('CustomCell\na window over the slab')

    # SphericalCell: particle plus margin
    ax = axes[2]
    nano = Octahedron('Ag', 5, 1)
    nano.positions -= nano.get_center_of_mass()
    draw_atoms_2d(ax, nano)
    R = np.linalg.norm(nano.positions, axis=1).max() + 3.0
    shade(ax, Circle((0, 0), R))
    ax.set_title('SphericalCell\nparticle plus vacuum margin')

    # DomeCell: sphere on the particle, clipped at the support surface
    ax = axes[3]
    support = fcc111('Al', a=4.05, size=(6, 6, 2), vacuum=2.0, periodic=True)
    part = Octahedron('Ag', 4, 1)
    part.positions -= part.get_center_of_mass()
    surface_z = support.positions[:, 2].max()
    center_xy = support.cell.sum(axis=0)[:2] / 2
    lift = surface_z + ATOM_RADII['Al'] + ATOM_RADII['Ag'] - part.positions[:, 2].min()
    part.positions += (*center_xy, lift)
    draw_atoms_2d(ax, support + part)
    c = part.get_center_of_mass()
    R = np.linalg.norm(part.positions - c, axis=1).max() + 3.0
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.c_[c[0] + R * np.cos(theta), c[2] + R * np.sin(theta)]
    # the part of the sphere the clip removes, drawn faintly
    ax.plot(*circle[circle[:, 1] < surface_z].T, color=BLUE, lw=1.0, ls=':', zorder=3)
    dome = circle[circle[:, 1] >= surface_z]
    dome = dome[np.argsort(np.arctan2(dome[:, 1] - c[2], dome[:, 0] - c[0]))]
    shade(ax, Polygon(dome, closed=True))
    ax.axhline(surface_z, color=VERMILION, lw=1.4, zorder=4)
    ax.set_title('DomeCell\nsphere clipped at bottom_z')

    # NullCell: no region at all
    ax = axes[4]
    draw_atoms_2d(ax, nano)
    ax.set_title('NullCell\nno region (displacement, permutation, ...)')

    _equalize_panels(axes[:5])
    for ax in axes:
        bare(ax)

    axes[5].legend(
        handles=[Patch(facecolor=SKY, alpha=0.35, edgecolor=BLUE, linestyle='--',
                       label='insertion region'),
                 Line2D([], [], color=BLUE, ls=':', label='part removed by the clip'),
                 Line2D([], [], color=VERMILION, lw=1.4, label='support surface (bottom_z)'),
                 Patch(facecolor=atom_color('Ag'), edgecolor='black', label='Ag'),
                 Patch(facecolor=atom_color('Al'), edgecolor='black', label='Al (support)')],
        loc='center', frameon=False)
    fig.tight_layout()
    fig.savefig(STATIC / 'fig_cells.png')
    plt.close(fig)


def fig_free_volume():
    """Top view of the tutorial's CustomCell: points come from
    cell.get_random_point() and are classified with the cell's own exclusion
    rule; V_free in the title is cell.get_volume()."""
    from ase.build import fcc111
    from matplotlib.lines import Line2D

    from mcpy.cell import CustomCell

    height = 5.0
    slab = fcc111('Ag', a=4.165, size=(4, 4, 3), vacuum=8.0, periodic=True)
    bottom_z = slab.positions[:, 2].max() + 0.5
    cell = CustomCell(slab, custom_height=height, bottom_z=bottom_z,
                      species_radii={'Ag': R_O_AG, 'O': 0.5},
                      mc_sample_points=100_000, seed=7)
    cell.calculate_volume(slab)
    v_free, v_cell = cell.get_volume(), cell.cell_volume

    # Only a thin slice, 1.5-1.8 A above the top layer: seen from above, the
    # whole window would hide the pattern the exclusion spheres cut into it.
    z_top = slab.positions[:, 2].max()
    pts = np.array([cell.get_random_point() for _ in range(15000)])
    pts = pts[(pts[:, 2] > z_top + 1.5) & (pts[:, 2] < z_top + 1.8)]
    occ = cell._covered(pts, cell._exclusion_trees(slab))

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    a, b = slab.cell[0, :2], slab.cell[1, :2]
    outline = Polygon(np.array([[0, 0], a, a + b, b]), closed=True, fill=False,
                      edgecolor='black', linewidth=1.4, zorder=5)
    ax.add_patch(outline)
    top = slab[slab.get_tags() == 1]
    tiled = top.repeat((3, 3, 1))
    tiled.positions -= slab.cell[0] + slab.cell[1]
    for atom in tiled:
        ax.add_patch(Circle(atom.position[:2], ATOM_RADII['Ag'],
                            facecolor=atom_color('Ag'), edgecolor='black',
                            linewidth=0.6, zorder=2, clip_path=outline))
    ax.scatter(pts[~occ][:, 0], pts[~occ][:, 1], s=7, color=BLUE, marker='o',
               zorder=4)
    ax.scatter(pts[occ][:, 0], pts[occ][:, 1], s=9, color=ORANGE, marker='x',
               linewidths=0.9, zorder=4)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker='o', ls='', label='free'),
        Line2D([], [], color=ORANGE, marker='x', ls='',
               label=f'occupied (within {R_O_AG} Å of an Ag)'),
        Line2D([], [], color='black', lw=1.4, label='periodic cell (top view)')],
        loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
        fontsize=9)
    ax.set_aspect('equal')
    corners = np.array([[0, 0], a, a + b, b])
    ax.set_xlim(corners[:, 0].min() - 0.5, corners[:, 0].max() + 0.5)
    ax.set_ylim(corners[:, 1].min() - 0.5, corners[:, 1].max() + 0.5)
    bare(ax)
    ax.set_title('CustomCell over Ag(111), top view of the points 1.5-1.8 Å\n'
                 f'above the top layer. Whole window: $V_\\mathrm{{free}}$ = '
                 f'{v_free:.0f} of {v_cell:.0f} $\\mathrm{{\\AA^3}}$ '
                 f'({100 * (1 - v_free / v_cell):.0f}% occupied)')
    fig.savefig(STATIC / 'fig_free_volume.png')
    plt.close(fig)


def run_hello_gcmc(workdir):
    """Run examples/00_hello_gcmc.py unchanged inside ``workdir``."""
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        runpy.run_path(str(ROOT / 'examples' / '00_hello_gcmc.py'), run_name='__main__')
    finally:
        os.chdir(cwd)
    return Path(workdir)


def fig_run(workdir):
    """Run-evolution curves and snapshot strip from example 00's output."""
    from ase.io import read

    ex = runpy.run_path(str(ROOT / 'examples' / '00_hello_gcmc.py'))
    rows = [line.split() for line in open(workdir / 'hello_gcmc.out')]
    rows = np.array([r[:3] for r in rows if len(r) > 3 and r[0].isdigit()], dtype=float)
    steps, n, energy = rows.T
    half = ex['STEPS'] / 2

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    for ax in (a1, a2):
        ax.axvspan(0, half, color=GREY, alpha=0.12, lw=0)
        ax.grid(alpha=0.3)
    a1.axhspan(ex['LAMMPS_N'] - 2 * ex['LAMMPS_N_ERR'], ex['LAMMPS_N'] + 2 * ex['LAMMPS_N_ERR'],
               color=ORANGE, alpha=0.35, lw=0)
    a1.axhline(ex['LAMMPS_N'], color=ORANGE, lw=1.4, label=r'LAMMPS fix gcmc $\langle N\rangle$')
    a1.plot(steps, n, color=BLUE, lw=1.4, marker='o', ms=2.5, label='mcpy')
    a1.set_ylabel('N (particles in the box)')
    a1.legend(loc='lower right', frameon=False)
    a1.text(half / 2, 20, 'equilibration\n(not averaged)', ha='center', va='bottom',
            color=GREY)
    a2.plot(steps, energy, color=BLUE, lw=1.4, marker='o', ms=2.5)
    a2.set_ylabel(r'potential energy ($\varepsilon$)')
    a2.set_xlabel('GCMC step')
    a2.set_xlim(0, ex['STEPS'])
    fig.suptitle(rf'examples/00_hello_gcmc.py: LJ fluid, $T^*$ = {ex["TEMPERATURE"]}, '
                 rf'$\mu^*$ = {ex["MU"]}, box {ex["BOX"]:g}$\sigma$')
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(STATIC / 'fig_run_evolution.png')
    plt.close(fig)

    # Frame 0 is the starting state; after it a frame is written every
    # trajectory_write_interval steps, so frame k is step k * interval.
    traj = read(workdir / 'hello_gcmc.xyz', index=':')
    interval = ex['STEPS'] // (len(traj) - 1)
    picks = [200, 1000, ex['STEPS']]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, step in zip(axes, picks):
        frame = traj[step // interval]
        frame.wrap()
        draw_atoms_2d(ax, frame, plane='xy', alpha=0.45, color=SKY)
        L = ex['BOX']
        ax.add_patch(Rectangle((0, 0), L, L, fill=False, edgecolor='black', lw=1.4, zorder=5))
        ax.set_xlim(-0.8, L + 0.8)
        ax.set_ylim(-0.8, L + 0.8)
        bare(ax)
        ax.set_title(f'step {step}:  N = {len(frame)}')
    fig.suptitle('Top view of the periodic box (particles drawn at radius $\\sigma/2$)')
    fig.tight_layout()
    fig.savefig(STATIC / 'fig_run_snapshots.png')
    plt.close(fig)


def fig_radii(n_trials=150):
    """Calibration figure for species_radii: relaxed nearest-neighbour
    distances of trial O insertions on Ag(111) (EMT stand-in for the
    production calculator). Same loop as docs/species_radii.rst."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    rng = np.random.default_rng(11)
    slab0 = fcc111('Ag', a=4.085, size=(3, 3, 3), vacuum=8.0, periodic=True)
    z_top = slab0.positions[:, 2].max()

    distances = []
    for _ in range(n_trials):
        slab = slab0.copy()
        slab.set_constraint(FixAtoms(indices=[a.index for a in slab if a.tag == 3]))
        slab.append('O')
        slab.positions[-1] = [rng.uniform(0, slab.cell[0, 0]),
                              rng.uniform(0, slab.cell[1, 1]),
                              z_top + rng.uniform(1.0, 3.5)]
        slab.calc = EMT()
        LBFGS(slab, logfile=None).run(fmax=0.1, steps=50)
        d = slab.get_distances(len(slab) - 1, range(len(slab) - 1), mic=True)
        distances.append(d.min())

    distances = np.array(distances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(distances, bins=30, color=SKY, edgecolor='white')
    r = np.percentile(distances, 5)
    ax.axvline(r, color=VERMILION, linewidth=1.6, linestyle='--',
               label=f'first-peak edge $r$ = {r:.2f} Å (EMT)')
    ax.set_xlabel('nearest O-Ag distance after relaxation (Å)')
    ax.set_ylabel('count')
    ax.set_title(f'{n_trials} relaxed O insertions on Ag(111), EMT')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(STATIC / 'fig_species_radii.png')
    plt.close(fig)


def _run_one_emt_gcmc(outdir, mu, seed):
    """The tutorial's sections 2-5 at one mu, with EMT in place of MACE and
    the insertion capped at one monolayer."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms

    from mcpy.calculators import BaseCalculator
    from mcpy.cell import CustomCell
    from mcpy.ensembles import GrandCanonicalEnsemble
    from mcpy.moves import DeletionMove, InsertionMove, MoveSelector

    atoms = fcc111('Ag', a=4.085, size=(3, 3, 3), vacuum=8.0, periodic=True)
    atoms.set_constraint(FixAtoms(indices=[a.index for a in atoms if a.tag == 3]))
    s_cell, s_ins, s_del, s_sel, s_ens = (
        int(x) for x in np.random.SeedSequence(seed).generate_state(5, dtype=np.uint32))
    cell = CustomCell(atoms, custom_height=5.0,
                      bottom_z=atoms.positions[:, 2].max() + 0.5,
                      species_radii={'Ag': R_O_AG, 'O': 0.5}, seed=s_cell)
    calculator = BaseCalculator(calculator=EMT(), steps=20, fmax=0.1)
    moves = MoveSelector(
        [1, 1],
        [InsertionMove(cell, species=['O'], seed=s_ins, max_atoms=MAX_O_EMT),
         DeletionMove(cell, species=['O'], seed=s_del)],
        seed=s_sel)
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=calculator,
        mu={'O': mu}, units_type='metal', species=['O'], temperature=500.0,
        move_selector=moves, random_seed=s_ens,
        outfile=str(outdir / 'gcmc.out'), traj_file=str(outdir / 'gcmc.xyz'),
        outfile_write_interval=10, trajectory_write_interval=1)
    gcmc.run(steps=1000)


def run_emt_sweep(workdir):
    """The tutorial's mu sweep, EMT edition: one capped run per dmu, written
    to the tutorial's own paths (sweep/dmu_{dmu:+.2f}/gcmc.xyz)."""
    from ase.calculators.emt import EMT

    from mcpy.utils import derive_mu_gas

    mu_o_ref = derive_mu_gas(EMT(), 'O2')
    for i, dmu in enumerate(SWEEP_DMUS):
        outdir = Path(workdir) / f'sweep/dmu_{dmu:+.2f}'
        outdir.mkdir(parents=True, exist_ok=True)
        _run_one_emt_gcmc(outdir, mu=mu_o_ref + dmu, seed=42 + i)
    return mu_o_ref


def fig_sweep_phase_diagram(workdir, mu_o_ref):
    """The tutorial's section 7 code, verbatim apart from the output path."""
    from ase.io import read

    from mcpy.utils import plot_phase_diagram

    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        # --- verbatim from docs/getting_started.rst, section 7 ---
        clean = read('sweep/dmu_-0.80/gcmc.xyz', index='0')

        frames = [clean] + [
            read(f'sweep/dmu_{dmu:+.2f}/gcmc.xyz', index='500:')   # drop equilibration
            for dmu in (-0.8, -0.6, -0.4, -0.2, 0.0)
        ]

        results = plot_phase_diagram(
            frames,
            adsorbate='O',
            metal_symbols=('Ag',),
            mu_ref=mu_o_ref,
            kind='surface',
            T=500.0,
            dmu_range=(-1.0, 0.0),
            # the tutorial writes 'phase_diagram.png'
            outfile=str(STATIC / 'fig_phase_diagram_emt_sweep.png'),
        )
        # --- end verbatim ---
    finally:
        os.chdir(cwd)
    print('EMT sweep transitions:', results['transitions'])


def extract_notebook_figures(targets=None):
    """Copy figures out of the executed notebooks. ``targets`` limits the
    copy to those output file names (default: all)."""
    wanted = {
        # (notebook, embedded-PNG index): target file
        ('gcmc_basics_lj.ipynb', 1): 'fig_lj_equilibration.png',
    }
    for (nb_name, want_idx), target in wanted.items():
        if targets is not None and target not in targets:
            continue
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


if __name__ == '__main__':
    import tempfile

    STATIC.mkdir(exist_ok=True)
    fig_cells()
    fig_free_volume()
    fig_radii()
    extract_notebook_figures()
    with tempfile.TemporaryDirectory() as td:
        fig_run(run_hello_gcmc(td))
        fig_sweep_phase_diagram(td, run_emt_sweep(td))
    print(f'Figures written to {STATIC}')
