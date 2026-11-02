---
name: analyze-results
description: Use when reading finished GCMC or replica-exchange output - judging acceptance ratios and convergence from outfile.out, inspecting trajectory.xyz, or building coverage isotherms and phase diagrams from run frames.
---

# Analyze a finished run

## Steps

1. `outfile.out`: step, N_atoms, energy, per-move acceptance per write interval
   (cumulative totals only in the `finalize_run` log block).
   Healthy ranges: displacement 20-60 %; insertion/deletion often 1-10 % on dense
   surfaces and that is fine; 0 % everywhere means mu or radii are wrong
   (`troubleshoot-run` skill).
2. Convergence: split the N(step) trace in half; tail-half drift of mean N must be small
   against its fluctuation. Judge every replica rung separately.
3. Replica-exchange swap health: second-half acceptance per neighbour pair,
   `second_half = 2*cum_end - cum_mid`, never the cumulative column
   (details in `replica-exchange-setup`).
4. `trajectory.xyz`: extended XYZ with `energy=` and `Lattice=` in the comment line,
   `molecule_id` column when molecule moves ran. Read with
   `ase.io.read('trajectory.xyz', index=':')`.
5. Phase diagrams: use `mcpy.utils.phase_diagram.plot_phase_diagram(frames, adsorbate=...,
   metal_symbols=..., mu_ref=..., kind='surface'|'nanoparticle', dmu_range=...)`.
   Do not hand-roll coverage curves; the library routine does the envelope, transition
   lines, and structure thumbnails. Guide: `docs/phase_diagrams.rst`.
   (`analyze_phase_diagram_results` is the older oxidation-specific entry point.)
6. Basin-hopping output, if enabled: `minima_file` holds strictly improving frames scored
   by grand potential Omega = E - sum(mu_i N_i), not bare energy; replica exchange also
   writes `global_minimum.xyz` (one frame per rung when rungs' mu differ).

Done when: every reported number comes from the converged tail, not the whole trace.

## Traps

- Averaging N over the whole run includes the equilibration ramp; cut it.
- Low insertion acceptance alone is not failure; zero NET growth at a mu above the known
  transition is.
