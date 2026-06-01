import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

plt.rcParams.update(
    {
        "figure.dpi": 200,
        # "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.size": 14,
        "mathtext.default": "regular",
    }
)


def free_en(en_dft, n_host, n_oxygen, e_host, e_o2, delta_mu_o, delta_mu_host, area):
    free = en_dft - (n_host * e_host) - (n_oxygen * e_o2 * 0.5)
    return (free - (n_host * delta_mu_host) - (n_oxygen * delta_mu_o)) / area


def from_mu_to_press(mu, T=500, mu_ref=-0.5, p_0=1, k_b=0.00008617, U_dft=-9.8612 / 2):
    return np.exp(((mu - mu_ref - U_dft) * 2) / (k_b * T)) * p_0


def analyze_phase_diagram_results(
    trajectory_path="relaxed_structures.xyz",
    host_symbol="Ag",
    oxygen_symbol="O",
    idx_ref=2400,
    e_host=-2.82894684,
    delta_mu_host=0.0,
    e_host_oxide=-3.6312459783333333 * 3,
    h_f_host_oxide=0.323,
    n_bins=300,
    delta_mu_o_min=-1.0,
    delta_mu_o_max=0.0,
    z_threshold=13.4,
    T=500,
    xlim=(-1.0, -0.2),
    oxide_ratio_vmax=0.67,
    save_plot=True,
    output_plot_path="lines_phases_mace.png",
    show_plot=False,
):
    """Analyze relaxed structures and build a surface phase diagram.

    Returns a dictionary with arrays and metadata useful for post-processing.
    """
    traj = read(trajectory_path, ":")
    if len(traj) == 0:
        raise ValueError(f"No structures found in trajectory: {trajectory_path}")
    if idx_ref >= len(traj):
        raise ValueError(
            f"idx_ref={idx_ref} is out of range for trajectory with {len(traj)} frames."
        )

    e_o2 = (e_host_oxide - (2 * (e_host + delta_mu_host)) + h_f_host_oxide) * 2
    delta_mu_o = np.linspace(delta_mu_o_min, delta_mu_o_max, n_bins)

    area = np.linalg.norm(traj[-1].get_cell()[0]) * np.linalg.norm(traj[-1].get_cell()[1])

    conf_ref = traj[idx_ref]
    free_ref = free_en(
        conf_ref.get_potential_energy(),
        conf_ref.get_chemical_symbols().count(host_symbol),
        conf_ref.get_chemical_symbols().count(oxygen_symbol),
        e_host,
        e_o2,
        delta_mu_o,
        delta_mu_host,
        area,
    )

    free_matrix = []
    for conf in traj:
        free = free_en(
            conf.get_potential_energy(),
            conf.get_chemical_symbols().count(host_symbol),
            conf.get_chemical_symbols().count(oxygen_symbol),
            e_host,
            e_o2,
            delta_mu_o,
            delta_mu_host,
            area,
        )
        free_matrix.append((free - free_ref) * 1000.0)
    free_matrix = np.vstack(free_matrix)

    stable_conf_idx = np.argmin(free_matrix, axis=0)
    min_energy = free_matrix[stable_conf_idx, np.arange(n_bins)]

    phase_order = [stable_conf_idx[0]]
    transitions = []
    for i in range(1, n_bins):
        if stable_conf_idx[i] != stable_conf_idx[i - 1]:
            transitions.append(float(delta_mu_o[i]))
            phase_order.append(stable_conf_idx[i])

    cmap = LinearSegmentedColormap.from_list("silver_red", [(0.75, 0.75, 0.75), (1, 0, 0)])
    norm = Normalize(vmin=0, vmax=oxide_ratio_vmax)
    phase_colors = []
    phase_oxide_ratios = []

    for idx in phase_order:
        conf = traj[int(idx)]
        oxygen_count = 0
        host_surface_count = 0
        for atom in conf:
            if atom.symbol == oxygen_symbol:
                oxygen_count += 1
            elif atom.symbol == host_symbol and atom.position[2] > z_threshold:
                host_surface_count += 1

        if host_surface_count == 0:
            ratio = 0.0
        else:
            ratio = oxygen_count / host_surface_count
        phase_oxide_ratios.append(float(ratio))
        phase_colors.append(cmap(norm(ratio)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for row in free_matrix:
        ax.plot(delta_mu_o, row, c="slategray", zorder=9, lw=0.5, alpha=0.5)

    # Tint the plot background using stable-segment boundaries.
    region_bounds = [delta_mu_o_min] + transitions + [delta_mu_o_max]
    ymin = float(np.min(free_matrix))
    ymax = float(np.max(free_matrix))
    for i in range(len(region_bounds) - 1):
        color = phase_colors[min(i, len(phase_colors) - 1)]
        ax.axvspan(
            region_bounds[i],
            region_bounds[i + 1],
            facecolor=color,
            alpha=0.25,
            zorder=0,
        )

    # Plot the stable envelope with a different color for each phase segment.
    segment_start = 0
    phase_segment_idx = 0
    for i in range(1, n_bins):
        if stable_conf_idx[i] != stable_conf_idx[i - 1]:
            color = phase_colors[min(phase_segment_idx, len(phase_colors) - 1)]
            ax.plot(
                delta_mu_o[segment_start:i],
                min_energy[segment_start:i],
                color=color,
                lw=2.2,
                zorder=12,
            )
            segment_start = i
            phase_segment_idx += 1
    color = phase_colors[min(phase_segment_idx, len(phase_colors) - 1)] if phase_colors else "black"
    ax.plot(
        delta_mu_o[segment_start:],
        min_energy[segment_start:],
        color=color,
        lw=2.2,
        zorder=12,
    )

    # Draw transition lines below bulk-oxide limit.
    for x in transitions:
        if x <= -h_f_host_oxide:
            ax.axvline(x, c="black", lw=1, zorder=10)

    linew = 0.8
    mpl.rcParams["hatch.linewidth"] = linew
    ax.axvspan(
        -h_f_host_oxide,
        0,
        facecolor="darkred",
        hatch="//",
        linestyle="dashed",
        lw=linew,
        edgecolor="black",
        zorder=8,
    )

    ax2 = ax.twiny()
    ax2.plot(np.log10(from_mu_to_press(delta_mu_o + (e_o2 / 2), T=T)), min_energy, lw=0)
    ax2.set_xlabel(f"log$_{{10}}$ $(p_{{O_2}}/p_0)$, T = {T} K", fontsize=18, labelpad=10)
    ax2.tick_params(direction="in", labelsize=15)

    ax.set_xlabel("$\\Delta \\mu_{O}(T,p)$ [eV]", fontsize=18)
    ax.set_ylabel("$\\gamma$ [meV/$\\AA^2$]", fontsize=18)
    ax.tick_params(direction="in", labelsize=15)
    ax.set_xlim(*xlim)
    ax2.set_xlim(
        np.log10(from_mu_to_press(xlim[0] + (e_o2 / 2), T=T)),
        np.log10(from_mu_to_press(xlim[1] + (e_o2 / 2), T=T)),
    )
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if save_plot:
        fig.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "delta_mu_o": delta_mu_o,
        "stable_conf_idx": stable_conf_idx,
        "min_energy_mev_per_a2": min_energy,
        "transitions_delta_mu_o": np.array(transitions),
        "phase_order_conf_idx": np.array(phase_order),
        "phase_oxide_ratios": np.array(phase_oxide_ratios),
        "plot_path": output_plot_path if save_plot else None,
    }


def _frame_energy(atoms):
    """Best-effort potential-energy lookup for a frame (info, calc, or getter)."""
    energy = atoms.info.get("energy")
    if energy is None and atoms.calc is not None:
        energy = atoms.calc.results.get("energy")
    if energy is None:
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            energy = None
    return energy


def _flatten_frames(frames):
    """Accept a flat list of Atoms or a list of trajectories (list of lists)."""
    flat = []
    for item in frames:
        if isinstance(item, Atoms):
            flat.append(item)
        else:
            flat.extend(item)
    return flat


def plot_phase_diagram(
    frames,
    adsorbate="H",
    metal_symbols=("Pd", "Ag"),
    mu_ref=-3.05,
    kind="surface",
    T=300.0,
    dmu_range=(-1.0, 0.0),
    n_bins=400,
    min_phase_width=0.03,
    rotation="0x,0y,0z",
    show_structures=True,
    system_label=None,
    outfile=None,
    show_plot=False,
    k_b=8.617e-5,
):
    """Build a grand-canonical adsorbate phase diagram for one configuration.

    All ``frames`` are assumed to belong to a single system configuration (e.g.
    one surface facet or one nanoparticle size); they typically sweep a range of
    chemical potentials. The lowest-energy adsorbate-free frame is the reference.

    Parameters
    ----------
    frames : list[ase.Atoms] | list[list[ase.Atoms]]
        Frames for one configuration. A list of trajectories is flattened.
    adsorbate : str
        Symbol of the species whose count defines the stoichiometry (e.g. "H").
    metal_symbols : tuple[str, ...]
        Host species, used for per-atom normalization and structure labels.
    mu_ref : float
        Reference chemical potential of the adsorbate (e.g. ½E(H2)) [eV].
    kind : {"surface", "nano"}
        ``"surface"`` normalizes gamma by the in-plane cell area (meV/Å²);
        ``"nano"`` normalizes by the number of metal atoms (meV/atom).
    T : float
        Temperature [K] for the twin log10(p/p0) axis.
    dmu_range : tuple[float, float]
        (min, max) of Δμ_adsorbate [eV].
    n_bins : int
        Number of Δμ grid points.
    min_phase_width : float
        Minimum Δμ width [eV] for a phase to be kept; narrower phases merge.
    rotation : str
        ASE ``plot_atoms`` rotation string for thumbnails.
    show_structures : bool
        Render the per-phase structure thumbnails (requires ase.visualize.plot).
    system_label : str | None
        Text label drawn inside the plot.
    outfile : str | None
        If given, save the figure to this path.
    show_plot : bool
        Show interactively instead of closing the figure.
    k_b : float
        Boltzmann constant [eV/K].

    Returns
    -------
    dict
        Arrays and metadata describing the diagram.
    """
    if kind not in ("surface", "nano"):
        raise ValueError(f"kind must be 'surface' or 'nano', got {kind!r}")

    flat = _flatten_frames(frames)
    data = []
    for at in flat:
        energy = _frame_energy(at)
        if energy is None:
            continue
        n_ads = sum(1 for a in at if a.symbol == adsorbate)
        data.append((at, float(energy), n_ads))
    if not data:
        raise ValueError("No frames with a retrievable energy were provided.")

    clean = sorted((d for d in data if d[2] == 0), key=lambda d: d[1])
    if not clean:
        raise ValueError(f"No adsorbate-free ({adsorbate} count == 0) reference frame found.")
    ref_atoms, ref_E = clean[0][0], clean[0][1]

    n_metal = sum(1 for a in ref_atoms if a.symbol in metal_symbols)
    if kind == "surface":
        c = ref_atoms.get_cell()
        norm_factor = float(np.linalg.norm(np.cross(c[0], c[1])))
        unit = r"meV/Å²"
        symbol = r"\gamma"
    else:
        norm_factor = n_metal
        unit = r"meV/atom"
        symbol = r"\Delta G"

    best = {}
    for at, energy, n_ads in data:
        if n_ads not in best or energy < best[n_ads][1]:
            best[n_ads] = (at, energy)
    stoich = sorted(best)

    dmu_grid = np.linspace(dmu_range[0], dmu_range[1], n_bins)
    free = np.array(
        [((best[n][1] - n * (mu_ref + dmu_grid)) - ref_E) / norm_factor * 1000.0 for n in stoich]
    )
    s_idx = np.argmin(free, axis=0)
    min_g = free[s_idx, np.arange(n_bins)]

    raw_order = [int(s_idx[0])]
    raw_trans = []
    for i in range(1, n_bins):
        if s_idx[i] != s_idx[i - 1]:
            raw_trans.append(float(dmu_grid[i]))
            raw_order.append(int(s_idx[i]))

    bounds = [dmu_grid[0]] + raw_trans + [dmu_grid[-1]]
    widths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    phase_order = []
    transitions = []
    for i, p in enumerate(raw_order):
        if widths[i] >= min_phase_width:
            phase_order.append(p)
            if i + 1 < len(bounds) - 1:
                transitions.append(bounds[i + 1])
    if not phase_order:
        phase_order = raw_order
        transitions = raw_trans
    transitions = transitions[: max(0, len(phase_order) - 1)]

    ratios = [stoich[i] / n_metal for i in phase_order] if n_metal else [0.0] * len(phase_order)
    cmap = LinearSegmentedColormap.from_list(
        "navy_red",
        [
            (0.10, 0.15, 0.50),
            (0.95, 0.78, 0.85),
            (0.85, 0.30, 0.25),
            (0.45, 0.05, 0.05),
        ],
    )
    cnorm = Normalize(vmin=0.0, vmax=max(max(ratios), 0.5))
    colors = [cmap(cnorm(r)) for r in ratios]

    def _press(dmu):
        return 10.0 ** (2.0 * dmu / (k_b * T * np.log(10.0)))

    n_p = len(phase_order)
    if show_structures:
        fig = plt.figure(figsize=(max(10, 2.2 * n_p), 8.5))
        gs = GridSpec(
            2, n_p, figure=fig, height_ratios=[2.6, 1.0],
            hspace=0.45, wspace=0.10, top=0.86, bottom=0.06, left=0.10, right=0.97,
        )
        ax = fig.add_subplot(gs[0, :])
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    for row in free:
        ax.plot(dmu_grid, row, c="slategray", lw=0.5, alpha=0.55, zorder=9)

    region_bounds = [dmu_grid[0]] + transitions + [dmu_grid[-1]]
    for i in range(len(region_bounds) - 1):
        ax.axvspan(
            region_bounds[i], region_bounds[i + 1],
            facecolor=colors[min(i, len(colors) - 1)], alpha=1.0, zorder=0,
        )

    for x in transitions:
        ax.axvline(x, c="black", lw=1.0, zorder=10)

    seg = 0
    for i in range(1, n_bins):
        if s_idx[i] != s_idx[i - 1]:
            ax.plot(dmu_grid[seg:i], min_g[seg:i], color="black", lw=2.0, zorder=12)
            seg = i
    ax.plot(dmu_grid[seg:], min_g[seg:], color="black", lw=2.0, zorder=12)

    span = min_g.max() - min_g.min()
    ax.set_xlim(dmu_grid[0], dmu_grid[-1])
    ax.set_ylim(min_g.min() - 0.25 * max(span, 50.0), min_g.max() + 1.2 * max(span, 50.0))
    ax.set_xlabel(r"$\Delta\mu_{%s}(T,p)$ [eV]" % adsorbate, fontsize=18)
    ax.set_ylabel(rf"${symbol}$ [{unit}]", fontsize=18)
    ax.tick_params(direction="in", labelsize=14)

    ax2 = ax.twiny()
    ax2.plot(np.log10(_press(dmu_grid)), min_g, lw=0)
    ax2.set_xlabel(
        rf"log$_{{10}}$ $(p_{{{adsorbate}_2}}/p_0)$, T = {int(T)} K", fontsize=16, labelpad=10
    )
    ax2.set_xlim(np.log10(_press(dmu_grid[0])), np.log10(_press(dmu_grid[-1])))
    ax2.tick_params(direction="in", labelsize=13)

    if show_structures:
        from ase.visualize.plot import plot_atoms

        for i, pidx in enumerate(phase_order):
            axs = fig.add_subplot(gs[1, i])
            n_ads = stoich[pidx]
            atoms = best[n_ads][0]
            plot_atoms(
                atoms, ax=axs, rotation=rotation, radii=1.0,
                show_unit_cell=2 if kind == "surface" else 0,
            )
            axs.set_xticks([])
            axs.set_yticks([])
            for sp in axs.spines.values():
                sp.set_edgecolor(colors[i])
                sp.set_linewidth(3.5)
            label = "".join(
                f"{sym}$_{{{sum(1 for a in atoms if a.symbol == sym)}}}$" for sym in metal_symbols
            )
            if n_ads > 0:
                label += f"{adsorbate}$_{{{n_ads}}}$"
            axs.set_title(
                label, fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=colors[i],
                          linewidth=2.0, boxstyle="round,pad=0.3"),
            )

    if system_label:
        ax.text(
            0.012, 0.97, system_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "dmu_grid": dmu_grid,
        "free": free,
        "stoich": stoich,
        "stable_idx": s_idx,
        "min_gamma": min_g,
        "transitions": np.array(transitions),
        "phase_order": np.array(phase_order),
        "phase_ratios": np.array(ratios),
        "unit": unit,
        "plot_path": outfile,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Analyze relaxed structures and plot a phase diagram."
    )
    parser.add_argument("trajectory_path", nargs="?", default="relaxed_structures.xyz")
    parser.add_argument("--output", dest="output_plot_path", default="lines_phases_mace.png")
    parser.add_argument("--idx-ref", type=int, default=2400)
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    analyze_phase_diagram_results(
        trajectory_path=args.trajectory_path,
        idx_ref=args.idx_ref,
        output_plot_path=args.output_plot_path,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
