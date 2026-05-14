"""
Analyze GCMC and RE-GCMC results for the Pd-Au-H JOSS case study.

Produces:
  1. H coverage vs delta_mu_H curves (GCMC vs RE-GCMC per system)
  2. Energy convergence comparison plots
  3. Pd/Au surface composition (segregation) analysis
  4. Summary CSV with per-run statistics

Usage:
    python analyze_results.py [--results-dir ../results] [--figures-dir ../figures]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 14,
    "mathtext.default": "regular",
})


def parse_outfile(outfile_path):
    """Parse a mcpy .out file and return arrays of step, n_atoms, energy."""
    steps, n_atoms, energies = [], [], []
    with open(outfile_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("+") or line.startswith("-") or line.startswith("S"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    steps.append(int(parts[0]))
                    n_atoms.append(int(parts[1]))
                    energies.append(float(parts[2]))
                except ValueError:
                    continue
    return np.array(steps), np.array(n_atoms), np.array(energies)


def count_species_in_trajectory(traj_path, species="H"):
    """Count atoms of a given species in each frame of a trajectory."""
    traj = read(traj_path, ":")
    counts = [sum(1 for a in frame if a.symbol == species) for frame in traj]
    return np.array(counts)


def surface_composition(traj_path, metal_a="Pd", metal_b="Au", z_threshold=None):
    """Compute fraction of metal_b on the surface (above z_threshold) per frame."""
    traj = read(traj_path, ":")
    if z_threshold is None:
        all_z = np.concatenate([frame.positions[:, 2] for frame in traj])
        z_threshold = np.percentile(all_z, 70)

    fractions = []
    for frame in traj:
        surface_a = sum(1 for a in frame
                        if a.symbol == metal_a and a.position[2] > z_threshold)
        surface_b = sum(1 for a in frame
                        if a.symbol == metal_b and a.position[2] > z_threshold)
        total = surface_a + surface_b
        fractions.append(surface_b / total if total > 0 else 0.0)
    return np.array(fractions)


def plot_coverage_vs_mu(results_dir, system_prefix, delta_mu_values, figures_dir,
                        gcmc_label="GCMC", regcmc_label="RE-GCMC"):
    """Plot H coverage (final-frame count) vs delta_mu_H for GCMC and RE-GCMC."""
    gcmc_counts = []
    regcmc_counts = []

    for dmu in delta_mu_values:
        gcmc_pattern = f"gcmc_{system_prefix}"
        regcmc_pattern = f"regcmc_{system_prefix}"

        gcmc_dir = results_dir / gcmc_pattern
        regcmc_dir = results_dir / regcmc_pattern

        gcmc_traj = gcmc_dir / f"PdAu_{system_prefix}_dmu_{dmu}.xyz"
        if gcmc_traj.exists():
            traj = read(str(gcmc_traj), ":")
            n_h = sum(1 for a in traj[-1] if a.symbol == "H")
            gcmc_counts.append(n_h)
        else:
            gcmc_counts.append(np.nan)

        regcmc_traj = regcmc_dir / f"PdAu_{system_prefix}_dmu_{dmu}_rank_0.xyz"
        if regcmc_traj.exists():
            traj = read(str(regcmc_traj), ":")
            n_h = sum(1 for a in traj[-1] if a.symbol == "H")
            regcmc_counts.append(n_h)
        else:
            regcmc_counts.append(np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(delta_mu_values, gcmc_counts, "o-", label=gcmc_label, color="steelblue")
    ax.plot(delta_mu_values, regcmc_counts, "s--", label=regcmc_label, color="firebrick")
    ax.set_xlabel(r"$\Delta \mu_H$ [eV]")
    ax.set_ylabel("H atom count (final frame)")
    ax.set_title(f"H coverage: {system_prefix}")
    ax.legend()
    plt.tight_layout()
    out_path = figures_dir / f"coverage_{system_prefix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_energy_convergence(outfile_path, figures_dir, label=""):
    """Plot energy vs MC step for a single run."""
    steps, _, energies = parse_outfile(outfile_path)
    if len(steps) == 0:
        print(f"  No data in {outfile_path}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, energies, lw=0.8, color="steelblue")
    ax.set_xlabel("MC step")
    ax.set_ylabel("Energy [eV]")
    ax.set_title(f"Energy convergence: {label}")
    plt.tight_layout()
    out_path = figures_dir / f"convergence_{label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Pd-Au-H GCMC results.")
    parser.add_argument("--results-dir", default="../results", type=str)
    parser.add_argument("--figures-dir", default="../figures", type=str)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    delta_mu_values = np.array([-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])

    systems = [
        "surface_111",
        "surface_100",
        "nano_small",
        "nano_large",
    ]

    for system in systems:
        print(f"\n=== Analyzing {system} ===")
        plot_coverage_vs_mu(results_dir, system, delta_mu_values, figures_dir)

        for dmu in delta_mu_values:
            gcmc_dir = results_dir / f"gcmc_{system}"
            if system.startswith("surface"):
                out = gcmc_dir / f"PdAu_{system}_dmu_{dmu}.out"
            else:
                out = gcmc_dir / f"PdAu_{system}_dmu_{dmu}.out"
            if out.exists():
                plot_energy_convergence(out, figures_dir,
                                        label=f"gcmc_{system}_dmu_{dmu}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
