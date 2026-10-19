import argparse
import functools
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.units import kJ, mol
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Applied per-call via _with_mpl_style, never at import time: importing mcpy
# must not rewrite the host application's global rcParams.
_MPL_STYLE = {
    'figure.dpi': 200,
    'font.sans-serif': 'Arial',
    'font.size': 14,
    'mathtext.default': 'regular',
}

logger = logging.getLogger(__name__)

# NIST-JANAF Thermochemical Tables (4th ed., Chase 1998), janaf.nist.gov tables
# O-029 (O2) and H-050 (H2), p0 = 1 bar. Per gas: H(0) - H(Tr) [kJ/mol], then
# the rows T [K], S [J/mol/K], H - H(Tr) [kJ/mol] from 100 to 1000 K.
_JANAF = {
    'O2': (-8.683, np.array([
        (100.0, 173.307, -5.779), (200.0, 193.485, -2.868), (250.0, 199.990, -1.410),
        (298.15, 205.147, 0.0), (300.0, 205.329, 0.054), (350.0, 209.880, 1.531),
        (400.0, 213.871, 3.025), (450.0, 217.445, 4.543), (500.0, 220.693, 6.084),
        (600.0, 226.451, 9.244), (700.0, 231.466, 12.499), (800.0, 235.921, 15.835),
        (900.0, 239.931, 19.241), (1000.0, 243.578, 22.703)])),
    'H2': (-8.467, np.array([
        (100.0, 100.727, -5.468), (200.0, 119.412, -2.774), (250.0, 125.640, -1.378),
        (298.15, 130.680, 0.0), (300.0, 130.858, 0.053), (350.0, 135.325, 1.502),
        (400.0, 139.216, 2.959), (450.0, 142.656, 4.420), (500.0, 145.737, 5.882),
        (600.0, 151.077, 8.811), (700.0, 155.606, 11.749), (800.0, 159.548, 14.702),
        (900.0, 163.051, 17.676), (1000.0, 166.216, 20.680)])),
}


def _janaf_dmu0(gas, T, n_atoms):
    """Delta mu^0(T) per adsorbed unit [eV] of a ``_JANAF`` gas at p0.

    Delta mu^0(T) = [(H(T) - H(0)) - T S(T)] / n_atoms per gas molecule: the
    convention of Reuter and Scheffler, PRB 65, 035406 (2001), Table I. The
    reference is the total energy of the molecule, without zero-point energy.
    Linear interpolation between the table rows.
    """
    h0, rows = _JANAF[gas]
    t, s, dh = rows.T
    if not t[0] <= T <= t[-1]:
        raise ValueError(
            f'T = {T} K is outside the NIST-JANAF {gas} table ({t[0]:g} to {t[-1]:g} K). '
            'Give Delta mu^0(T) yourself (from_mu_to_press: mu_ref).')
    dmu0 = (dh - h0 - t * s / 1000.0) * (kJ / mol) / n_atoms
    return float(np.interp(T, t, dmu0))


def _log10_p_over_p0(dmu, dmu0, T, n_atoms, k_b=8.617e-5):
    """log10(p/p0) of the reservoir gas, from p/p0 = exp(n (dmu - dmu0(T)) / kT)."""
    return n_atoms * (np.asarray(dmu, float) - dmu0) / (k_b * T * np.log(10.0))


def _with_mpl_style(style):
    """Run the decorated function inside an rc_context so ``style`` (and any
    rcParams the function sets while plotting) is reverted on return."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with mpl.rc_context(style):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def free_en(en_dft, n_host, n_oxygen, e_host, e_o2, delta_mu_o, delta_mu_host, area):
    free = en_dft - (n_host * e_host) - (n_oxygen * e_o2 * 0.5)
    return (free - (n_host * delta_mu_host) - (n_oxygen * delta_mu_o)) / area


def from_mu_to_press(mu, T=500, mu_ref=None, p_0=1, k_b=0.00008617, U_dft=-9.8612 / 2):
    """O2 pressure in equilibrium with the O chemical potential ``mu`` [eV].

    p = p_0 exp(2 (mu - U_dft - mu_ref) / kT). ``U_dft`` is E(O2)/2 on the
    energy scale of ``mu``. ``mu_ref`` is Delta mu_O^0(T) at ``p_0``; ``None``
    takes it at ``T`` from the NIST-JANAF O2 table (-0.495 eV at 500 K).
    """
    if mu_ref is None:
        mu_ref = _janaf_dmu0('O2', T, 2)
    return np.exp(((mu - mu_ref - U_dft) * 2) / (k_b * T)) * p_0


@_with_mpl_style(_MPL_STYLE)
def analyze_phase_diagram_results(
    trajectory_path='relaxed_structures.xyz',
    host_symbol='Ag',
    oxygen_symbol='O',
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
    output_plot_path='lines_phases_mace.png',
    show_plot=False,
):
    """Analyze relaxed structures and build a surface phase diagram.

    Returns a dictionary with arrays and metadata useful for post-processing.
    """
    traj = read(trajectory_path, ':')
    if len(traj) == 0:
        raise ValueError(f'No structures found in trajectory: {trajectory_path}')
    if idx_ref >= len(traj):
        raise ValueError(
            f'idx_ref={idx_ref} is out of range for trajectory with {len(traj)} frames.'
        )

    # E(Ag2O) = 2 E(Ag) + E(O2)/2 - h_f fixes the O2 reference; it does not
    # depend on mu_Ag. Bulk Ag2O is stable where 2 mu_Ag + mu_O >= E(Ag2O),
    # with mu_Ag = e_host + delta_mu_host, so at delta_mu_o >= dmu_o_oxide.
    e_o2 = (e_host_oxide - (2 * e_host) + h_f_host_oxide) * 2
    dmu_o_oxide = -h_f_host_oxide - 2 * delta_mu_host
    delta_mu_o = np.linspace(delta_mu_o_min, delta_mu_o_max, n_bins)

    cell = traj[-1].get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))

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

    cmap = LinearSegmentedColormap.from_list('silver_red', [(0.75, 0.75, 0.75), (1, 0, 0)])
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
        ax.plot(delta_mu_o, row, c='slategray', zorder=9, lw=0.5, alpha=0.5)

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
    color = phase_colors[min(phase_segment_idx, len(phase_colors) - 1)] if phase_colors else 'black'
    ax.plot(
        delta_mu_o[segment_start:],
        min_energy[segment_start:],
        color=color,
        lw=2.2,
        zorder=12,
    )

    # Draw transition lines below bulk-oxide limit.
    for x in transitions:
        if x <= dmu_o_oxide:
            ax.axvline(x, c='black', lw=1, zorder=10)

    linew = 0.8
    mpl.rcParams['hatch.linewidth'] = linew
    if dmu_o_oxide < 0:
        ax.axvspan(
            dmu_o_oxide,
            0,
            facecolor='darkred',
            hatch='//',
            linestyle='dashed',
            lw=linew,
            edgecolor='black',
            zorder=8,
        )

    try:
        dmu0 = _janaf_dmu0('O2', T, 2)
    except ValueError:
        logger.warning('T = %g K is outside the NIST-JANAF O2 table (100 to 1000 K): the '
                       'pressure axis is omitted.', T)
    else:
        ax2 = ax.twiny()
        ax2.plot(_log10_p_over_p0(delta_mu_o, dmu0, T, 2), min_energy, lw=0)
        ax2.set_xlabel(f'log$_{{10}}$ $(p_{{O_2}}/p_0)$, T = {T} K', fontsize=18, labelpad=10)
        ax2.tick_params(direction='in', labelsize=15)
        ax2.set_xlim(*_log10_p_over_p0(np.array(xlim), dmu0, T, 2))

    ax.set_xlabel('$\\Delta \\mu_{O}(T,p)$ [eV]', fontsize=18)
    ax.set_ylabel('$\\gamma$ [meV/$\\AA^2$]', fontsize=18)
    ax.tick_params(direction='in', labelsize=15)
    ax.set_xlim(*xlim)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    if save_plot:
        fig.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        'delta_mu_o': delta_mu_o,
        'stable_conf_idx': stable_conf_idx,
        'min_energy_mev_per_a2': min_energy,
        'transitions_delta_mu_o': np.array(transitions),
        'phase_order_conf_idx': np.array(phase_order),
        'phase_oxide_ratios': np.array(phase_oxide_ratios),
        'plot_path': output_plot_path if save_plot else None,
    }


def _frame_energy(atoms):
    """Best-effort potential-energy lookup for a frame (info, calc, or getter)."""
    energy = atoms.info.get('energy')
    if energy is None and atoms.calc is not None:
        energy = atoms.calc.results.get('energy')
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


def _lower_envelope(intercepts, slopes, x0, x1):
    """Exact lower envelope of the lines ``a + b * x`` on ``[x0, x1]``.

    Returns the winning line indices from left to right and the x of each
    switch (the analytic crossing of two neighbouring lines). Slopes must be
    pairwise different, which holds here: the slope is set by the adsorbate
    count and each count has one line.
    """
    a = np.asarray(intercepts, float)
    b = np.asarray(slopes, float)
    # At x0 the winner is the lowest line; on a tie, the steepest one, since it
    # is the lowest just right of x0.
    y0 = a + b * x0
    tied = np.flatnonzero(np.isclose(y0, y0.min(), rtol=0.0, atol=1e-12))
    cur = int(tied[np.argmin(b[tied])])
    order, crossings = [cur], []
    x = x0
    while True:
        steeper = np.flatnonzero(b < b[cur])
        if steeper.size == 0:
            break
        xs = (a[steeper] - a[cur]) / (b[cur] - b[steeper])
        keep = xs >= x - 1e-12
        if not keep.any():
            break
        x_next = xs[keep].min()
        if x_next >= x1:
            break
        cand = steeper[keep][np.isclose(xs[keep], x_next, rtol=0.0, atol=1e-12)]
        cur = int(cand[np.argmin(b[cand])])
        x = max(float(x_next), x)
        order.append(cur)
        crossings.append(x)
    return order, crossings


# Paul Tol's "light" scheme: colourblind-safe and light enough to sit behind
# black lines. Phases cycle through it, so neighbouring phases always differ.
PHASE_COLORS = ('#77AADD', '#EE8866', '#44BB99', '#EEDD88', '#99DDFF',
                '#FFAABB', '#BBCC33', '#AAAA00', '#DDDDDD')

_PD_STYLE = {
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'mathtext.default': 'regular',
    'hatch.linewidth': 0.7,
}

_THUMBS_PER_ROW = 6


@_with_mpl_style(_PD_STYLE)
def plot_phase_diagram(
    frames,
    adsorbate='H',
    metal_symbols=('Pd', 'Ag'),
    mu_ref=-3.05,
    kind='surface',
    T=300.0,
    dmu_range=(-1.0, 0.0),
    n_bins=400,
    min_phase_width=0.03,
    rotation='0x,0y,0z',
    show_structures=True,
    system_label=None,
    outfile=None,
    show_plot=False,
    k_b=8.617e-5,
    gamma_in_ev=False,
    adsorbate_count_fn=None,
    adsorbate_label=None,
    atoms_per_reservoir_molecule=2,
    hatch_from=None,
    hatch_label=None,
    dpi=250,
    mu_metal=None,
    dmu_standard=None,
):
    """Build a grand-canonical adsorbate phase diagram for one configuration.

    All ``frames`` are assumed to belong to a single system configuration (e.g.
    one surface facet or one nanoparticle size); they typically sweep a range of
    chemical potentials. Each frame scores E - sum_m mu_m (N_m - N_m,ref) over
    the ``metal_symbols``; this is E when all frames share one metal
    composition. The adsorbate-free frame with the lowest score is the
    reference, and the lowest score wins per adsorbate count.

    Every phase on the stable envelope is kept. The phase boundaries are the
    exact crossings of the linear free-energy lines, so the dashed transition
    lines sit on the kinks of the envelope. The phases are numbered from left
    to right; the same number is on the plot and on the thumbnail title, and
    neighbouring phases never share a colour.

    Parameters
    ----------
    frames : list[ase.Atoms] | list[list[ase.Atoms]]
        Frames for one configuration. A list of trajectories is flattened.
    adsorbate : str
        Symbol of the species whose count defines the stoichiometry (e.g. "H").
    metal_symbols : tuple[str, ...]
        Host species, used for per-atom normalization, structure labels and
        the ``mu_metal`` term.
    mu_ref : float
        Reference chemical potential of the adsorbate (e.g. ½E(H2)) [eV].
    kind : str
        ``"surface"`` normalizes gamma by the in-plane cell area (meV/Å²);
        ``"nano"`` normalizes by the number of metal atoms (meV/atom).
    T : float
        Temperature [K] for the twin log10(p/p0) axis.
    dmu_range : tuple[float, float]
        (min, max) of Δμ_adsorbate [eV].
    n_bins : int
        Number of Δμ grid points of the returned ``dmu_grid``/``free`` arrays.
        The phases and transitions do not depend on it.
    min_phase_width : float
        Phases narrower than this [eV] are still drawn, numbered and shown as a
        thumbnail; only their number moves to a second row on the plot, so it
        does not collide with the numbers of its neighbours.
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
    gamma_in_ev : bool
        Report the unnormalized formation energy in eV instead of the default
        per-atom (nano) or per-area (surface) value in meV.
    adsorbate_label : str | None
        Species name used in the axis labels. Defaults to ``adsorbate`` -
        pass e.g. ``"CO"`` when the adsorbate is counted via a proxy atom.
    atoms_per_reservoir_molecule : int
        Adsorbed units per reservoir gas molecule, used in the pressure
        conversion of the twin axis. The default 2 is the dissociative
        diatomic convention (atomic H or O from an H2/O2 reservoir,
        mu_atom = mu_gas/2). Use 1 for molecular adsorbates (CO, H2O, ...)
        whose reservoir molecule is the adsorbed unit itself.
    adsorbate_count_fn : callable | None
        Optional ``atoms -> int`` returning the adsorbate count for a frame.
        Use when the adsorbate symbol also occurs in an inert sublattice (e.g.
        oxygen adsorbed on a particle that sits on an oxide support): the count
        defines the stoichiometry while the full frame is still rendered. When
        ``None`` (default) the count is the number of atoms whose symbol equals
        ``adsorbate``.
    hatch_from : float | None
        Hatch the range Δμ > ``hatch_from`` [eV], for example where the bulk
        oxide is more stable than any surface phase.
    hatch_label : str | None
        Legend text of the hatched range.
    dpi : int
        Resolution of the saved figure.
    mu_metal : dict[str, float] | None
        Chemical potential [eV] of each host species whose count differs
        between frames, e.g. ``{'Ag': mu_ag}`` from a run that exchanges Ag.
        If the frames hold more than one count of a species in
        ``metal_symbols`` and ``mu_metal`` does not give it, a ValueError is
        raised.
    dmu_standard : float | None
        Delta mu^0(T) of the adsorbate at p0 [eV per adsorbed unit]. The twin
        axis is log10(p/p0) = n (dmu - dmu_standard) / (kT ln 10), with n =
        ``atoms_per_reservoir_molecule``. ``None`` takes it from the built-in
        NIST-JANAF tables for O (O2 reservoir) and H (H2 reservoir), 100 to
        1000 K. For any other adsorbate or T the twin axis is omitted, with a
        warning. The built-in values leave out the zero-point energy (ZPE):
        with a bare ``mu_ref``, the true p = p0 is at Delta mu^0(T) + ZPE / n.

    Returns
    -------
    dict
        Arrays and metadata describing the diagram.
    """
    if kind not in ('surface', 'nano'):
        raise ValueError(f"kind must be 'surface' or 'nano', got {kind!r}")

    flat = _flatten_frames(frames)
    data = []
    for at in flat:
        energy = _frame_energy(at)
        if energy is None:
            continue
        if adsorbate_count_fn is not None:
            n_ads = int(adsorbate_count_fn(at))
        else:
            n_ads = sum(1 for a in at if a.symbol == adsorbate)
        syms = at.get_chemical_symbols()
        n_m = tuple(syms.count(m) for m in metal_symbols)
        data.append((at, float(energy), n_ads, n_m))
    if not data:
        raise ValueError('No frames with a retrievable energy were provided.')

    mu_m = dict(mu_metal or {})
    unknown = sorted(set(mu_m) - set(metal_symbols))
    if unknown:
        raise ValueError(f'mu_metal gives {unknown}, which are not in metal_symbols '
                         f'{tuple(metal_symbols)}.')
    for i, m in enumerate(metal_symbols):
        counts = sorted({d[3][i] for d in data})
        if len(counts) > 1 and m not in mu_m:
            raise ValueError(
                f'The frames hold {counts} {m} atoms. Pass mu_metal={{{m!r}: mu_{m}}} '
                f'(eV) to compare frames with different {m} counts.')

    def score(energy, n_m, n_ref):
        # E - sum_m mu_m (N_m - N_m,ref); exactly E when the metal counts match.
        return energy - sum(mu_m[m] * (n - r)
                            for m, n, r in zip(metal_symbols, n_m, n_ref) if n != r)

    clean = [d for d in data if d[2] == 0]
    if not clean:
        raise ValueError(f'No adsorbate-free ({adsorbate} count == 0) reference frame found.')
    ref = min(clean, key=lambda d: score(d[1], d[3], clean[0][3]))
    ref_atoms, ref_E = ref[0], ref[1]

    n_metal = sum(1 for a in ref_atoms if a.symbol in metal_symbols)
    gamma_scale = 1.0 if gamma_in_ev else 1000.0
    if kind == 'surface':
        c = ref_atoms.get_cell()
        area = float(np.linalg.norm(np.cross(c[0], c[1])))
        norm_factor = 1.0 if gamma_in_ev else area
        unit = r'eV' if gamma_in_ev else r'meV/Å²'
        unit_tex = r'eV' if gamma_in_ev else r'meV/Å$^2$'
        symbol = r'\gamma'
    else:
        norm_factor = 1.0 if gamma_in_ev else n_metal
        unit = r'eV' if gamma_in_ev else r'meV/atom'
        unit_tex = unit
        symbol = r'\Delta G'

    best = {}
    for at, energy, n_ads, n_m in data:
        omega = score(energy, n_m, ref[3])
        if n_ads not in best or omega < best[n_ads][1]:
            best[n_ads] = (at, omega, n_m)
    stoich = sorted(best)

    # Each stoichiometry is a straight line in dmu: free = icpt + slope * dmu.
    icpt = np.array([(best[n][1] - n * mu_ref - ref_E) / norm_factor * gamma_scale
                     for n in stoich])
    slope = np.array([-n / norm_factor * gamma_scale for n in stoich])
    x0, x1 = float(dmu_range[0]), float(dmu_range[1])
    phase_order, transitions = _lower_envelope(icpt, slope, x0, x1)

    dmu_grid = np.linspace(x0, x1, n_bins)
    free = icpt[:, None] + slope[:, None] * dmu_grid[None, :]
    s_idx = np.argmin(free, axis=0)
    min_g = free[s_idx, np.arange(n_bins)]

    n_m_phase = [sum(best[stoich[i]][2]) for i in phase_order]
    ratios = [stoich[i] / n if n else 0.0 for i, n in zip(phase_order, n_m_phase)]
    n_p = len(phase_order)
    colors = [PHASE_COLORS[i % len(PHASE_COLORS)] for i in range(n_p)]
    bounds = [x0] + transitions + [x1]
    env_y = [icpt[phase_order[0]] + slope[phase_order[0]] * x0] + [
        icpt[p] + slope[p] * x for p, x in zip(phase_order, bounds[1:])]

    label = adsorbate_label if adsorbate_label is not None else adsorbate
    n_res = float(atoms_per_reservoir_molecule)
    gas_label = (label if atoms_per_reservoir_molecule == 1
                 else f'{label}_{atoms_per_reservoir_molecule}')

    rows = -(-n_p // _THUMBS_PER_ROW) if show_structures else 0
    cols = min(n_p, _THUMBS_PER_ROW)
    fig = plt.figure(figsize=(8.0, 5.2 + 1.75 * rows))
    gs = GridSpec(
        1 + rows, max(cols, 1), figure=fig, height_ratios=[3.4] + [1.25] * rows,
        hspace=0.55, wspace=0.08, top=0.88, bottom=0.03 if rows else 0.2,
        left=0.11, right=0.97,
    )
    ax = fig.add_subplot(gs[0, :])

    for row in free:
        ax.plot(dmu_grid, row, c='0.55', lw=0.6, alpha=0.7, zorder=3)

    span = float(np.ptp(env_y)) or 50.0 * gamma_scale / 1000.0
    lo, hi = min(env_y) - 0.12 * span, max(env_y) + 0.45 * span
    x_pad = 0.015 * (x1 - x0)
    for i in range(n_p):
        ax.axvspan(bounds[i], bounds[i + 1], facecolor=colors[i], alpha=0.75, lw=0, zorder=0)
        narrow = bounds[i + 1] - bounds[i] < min_phase_width
        x_txt = min(max(0.5 * (bounds[i] + bounds[i + 1]), x0 + x_pad), x1 - x_pad)
        ax.text(x_txt, 0.955 - (0.09 if narrow else 0.0), str(i + 1),
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=10, fontweight='bold', zorder=7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85,
                          boxstyle='round,pad=0.15'))
    for x in transitions:
        ax.axvline(x, c='k', lw=0.8, ls='--', zorder=4)
    ax.plot(bounds, env_y, c='k', lw=2.2, zorder=5)

    handles = [Line2D([], [], c='k', lw=2.2, label='stable envelope'),
               Line2D([], [], c='0.55', lw=0.8, label='other stoichiometries'),
               Line2D([], [], c='k', lw=0.8, ls='--', label='phase transition')]
    if hatch_from is not None and hatch_from < x1:
        ax.axvspan(max(hatch_from, x0), x1, facecolor='none', edgecolor='0.2',
                   hatch='///', lw=0, zorder=1)
        if hatch_from > x0:
            ax.axvline(hatch_from, c='0.2', lw=1.2, zorder=4)
        handles.append(Patch(facecolor='none', edgecolor='0.2', hatch='///',
                             label=hatch_label or 'hatched range'))

    ax.set_xlim(x0, x1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(rf'$\Delta\mu_{{{label}}}$ (eV)')
    ax.set_ylabel(rf'${symbol}$ ({unit_tex})')
    ax.tick_params(direction='in')
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=2 if len(handles) == 4 else 3, frameon=False, fontsize=9.5,
              handlelength=2.2, columnspacing=1.2)

    # p = p0 at dmu = dmu_standard, the reservoir's Delta mu^0(T).
    gas = label if atoms_per_reservoir_molecule == 1 else f'{label}{atoms_per_reservoir_molecule}'
    if dmu_standard is None and gas in _JANAF:
        try:
            dmu_standard = _janaf_dmu0(gas, T, n_res)
        except ValueError:
            pass  # T outside the table: same as no table
    if dmu_standard is None:
        logger.warning('No built-in Delta mu^0(T) for a %s reservoir at T = %g K: the pressure '
                       'axis is omitted. Pass dmu_standard to draw it.', gas, T)
    else:
        ax2 = ax.twiny()
        ax2.set_xlim(*_log10_p_over_p0(np.array([x0, x1]), dmu_standard, T, n_res, k_b))
        ax2.set_xlabel(rf'$\log_{{10}}(p_{{{gas_label}}}/p^0)$ at T = {T:g} K')
        ax2.tick_params(direction='in')

    if show_structures:
        from ase.visualize.plot import plot_atoms

        for i, pidx in enumerate(phase_order):
            r, col = divmod(i, _THUMBS_PER_ROW)
            axs = fig.add_subplot(gs[1 + r, col])
            n_ads = stoich[pidx]
            atoms = best[n_ads][0]
            plot_atoms(atoms, ax=axs, rotation=rotation, radii=0.9,
                       show_unit_cell=2 if kind == 'surface' else 0)
            # Pad the data limits to the grid cell's shape, so every thumbnail
            # fills an equal box and the titles line up.
            box = axs.get_position(original=True)
            cell_ratio = (box.height * fig.get_figheight()) / (box.width * fig.get_figwidth())
            (xa, xb), (ya, yb) = axs.get_xlim(), axs.get_ylim()
            if (yb - ya) < cell_ratio * (xb - xa):
                pad = 0.5 * (cell_ratio * (xb - xa) - (yb - ya))
                axs.set_ylim(ya - pad, yb + pad)
            else:
                pad = 0.5 * ((yb - ya) / cell_ratio - (xb - xa))
                axs.set_xlim(xa - pad, xb + pad)
            axs.set_xticks([])
            axs.set_yticks([])
            for sp in axs.spines.values():
                sp.set_edgecolor(colors[i])
                sp.set_linewidth(3.0)
            title = ''.join(
                f'{sym}$_{{{sum(1 for a in atoms if a.symbol == sym)}}}$' for sym in metal_symbols
            )
            if n_ads > 0:
                # Molecule-aware: with a proxy-counted adsorbate (e.g. CO via
                # its C atom) the label names the molecule, parenthesized so
                # (CO)_3 reads as three molecules rather than C-O3.
                ads = label if len(label) == 1 else f'({label})'
                title += f'{ads}$_{{{n_ads}}}$'
            axs.set_title(f'{i + 1}: {title}', fontsize=9, pad=3)

    if system_label:
        ax.text(0.012, 0.03, system_label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', zorder=7,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    if outfile:
        fig.savefig(outfile, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        'dmu_grid': dmu_grid,
        'free': free,
        'stoich': stoich,
        'stable_idx': s_idx,
        'min_gamma': min_g,
        'transitions': np.array(transitions),
        'phase_order': np.array(phase_order),
        'phase_ratios': np.array(ratios),
        'phase_metal_counts': [dict(zip(metal_symbols, best[stoich[i]][2]))
                               for i in phase_order],
        'unit': unit,
        'plot_path': outfile,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Analyze relaxed structures and plot a phase diagram.'
    )
    parser.add_argument('trajectory_path', nargs='?', default='relaxed_structures.xyz')
    parser.add_argument('--output', dest='output_plot_path', default='lines_phases_mace.png')
    parser.add_argument('--idx-ref', type=int, default=2400)
    parser.add_argument('--show', action='store_true')
    return parser


def main():
    args = _build_arg_parser().parse_args()
    analyze_phase_diagram_results(
        trajectory_path=args.trajectory_path,
        idx_ref=args.idx_ref,
        output_plot_path=args.output_plot_path,
        show_plot=args.show,
    )


if __name__ == '__main__':
    main()
