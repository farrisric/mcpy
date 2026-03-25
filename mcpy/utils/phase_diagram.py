import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
    analyze_phase_diagram_results(trajectory_path="/home/energystorage/projects/mcpy/notebooks/phase_diagram_demo/relaxed_structures.xyz", idx_ref=0, output_plot_path="lines_phases_mace.png", show_plot=True)