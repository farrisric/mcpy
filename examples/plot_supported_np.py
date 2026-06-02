"""Render a supported-nanoparticle GCMC trajectory frame as a showcase figure.

A single 3/4 view with depth shading (atoms farther from the viewer are
darkened) so the particle reads as sitting on the support instead of pasted
onto it. Adsorbed atoms are identified as those appended after the initial
structure: the first frame of the trajectory is the clean (pre-adsorption)
system, so any atom whose index is >= ``len(frame0)`` was inserted by GCMC.
This is robust to deletions because insertions are appended at the end and the
support is never deleted.

Needs matplotlib and ase (e.g. ``conda run -n base python ...``).

Usage::

    python examples/plot_supported_np.py traj.xyz --out figure.png
    python examples/plot_supported_np.py traj.xyz --particle-species Ag --frame -1
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from ase.io import read  # noqa: E402
from ase.io.utils import rotate  # noqa: E402
from ase.visualize.plot import plot_atoms  # noqa: E402

# role -> (base rgb, draw radius)
AG = (0.74, 0.77, 0.83)
SUPPORT_METAL = (0.82, 0.73, 0.55)
SUPPORT_O = (0.93, 0.80, 0.80)
ADSORBED = (0.90, 0.07, 0.07)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('trajectory', help='Extended-XYZ GCMC trajectory')
    p.add_argument('--out', default='supported_np.png', help='Output PNG path')
    p.add_argument('--particle-species', default='Ag',
                   help='Symbol of the nanoparticle metal (coloured silver)')
    p.add_argument('--frame', type=int, default=-1, help='Frame index to render')
    p.add_argument('--rotation', default='-80x,-15y,0z', help='ASE view rotation')
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


def atom_styles(atoms, n_initial, particle_species):
    """Per-atom base colours and draw radii, keyed on role."""
    symbols = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    colors = np.zeros((n, 3))
    radii = np.zeros(n)
    for i in range(n):
        if i >= n_initial:                       # appended by GCMC = adsorbed
            colors[i], radii[i] = ADSORBED, 0.66
        elif symbols[i] == particle_species:
            colors[i], radii[i] = AG, 1.44
        elif symbols[i] == 'O':
            colors[i], radii[i] = SUPPORT_O, 0.40
        else:
            colors[i], radii[i] = SUPPORT_METAL, 0.66
    return colors, radii


def depth_shade(colors, atoms, rotation):
    """Darken atoms by their depth along the view direction (fake volume)."""
    depth = atoms.positions @ rotate(rotation)
    z = depth[:, 2]
    norm = (z - z.min()) / (np.ptp(z) + 1e-9)    # 0 = far, 1 = near
    return colors * (0.40 + 0.60 * norm)[:, None], np.argsort(z)


def main():
    args = parse_args()
    n_initial = len(read(args.trajectory, index=0))
    atoms = read(args.trajectory, index=args.frame)
    n_ads = len(atoms) - n_initial

    colors, radii = atom_styles(atoms, n_initial, args.particle_species)
    colors, order = depth_shade(colors, atoms, args.rotation)   # far drawn first

    fig, ax = plt.subplots(figsize=(8.5, 8))
    plot_atoms(atoms[order], ax, colors=colors[order], radii=radii[order],
               rotation=args.rotation, show_unit_cell=0)
    for patch in ax.patches:                     # soften the cartoon outlines
        fc = patch.get_facecolor()
        patch.set_edgecolor((fc[0] * 0.55, fc[1] * 0.55, fc[2] * 0.55))
        patch.set_linewidth(0.25)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_title(f'O on Al$_2$O$_3$-supported {args.particle_species} nanoparticle  '
                 f'|  {n_ads} adsorbed', fontsize=13)

    handles = [
        mpatches.Patch(color=AG, label=f'{args.particle_species} (nanoparticle)'),
        mpatches.Patch(color=SUPPORT_METAL, label='support metal'),
        mpatches.Patch(color=SUPPORT_O, label='support O'),
        mpatches.Patch(color=ADSORBED, label='adsorbed'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, facecolor='white')
    print(f'saved {args.out}  ({n_ads} adsorbed atoms, frame {args.frame})')


if __name__ == '__main__':
    main()
