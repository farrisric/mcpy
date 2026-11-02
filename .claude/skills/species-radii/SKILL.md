---
name: species-radii
description: Use when setting or calibrating species_radii for a cell - computing exclusion radii for free-volume estimation, after changing calculator or relaxation settings, or when insertions all land inside atoms or free volume looks wrong.
---

# Calibrate species_radii

Method: `docs/species_radii.rst`. Tool: `scripts/compute_radii.py`.
Radii set the free volume, and free volume enters the insertion/deletion acceptance
directly - this is a physics input, not a tuning knob.

## Steps

1. Run `python scripts/compute_radii.py <model_path>` (GPU, `run-gpu` skill).
   Edit the constants at the top first: `metal_species`, `lattice_param`, `gas_species`,
   `cell_bottom`, `cell_height`. It does N random insertions above the surface, relaxes
   each with the production calculator, and histograms the nearest adsorbate-host distance.
2. The radius `r_relax` is the centre of the tallest histogram bin (the binding peak).
   The script draws it as the dashed line in `dist_hist.png` and logs to `insertion.log`.
3. Set it per host element: `species_radii={'Cu': r_relax, 'O': r_O}` on the cell.
   The adsorbate's own radius matters once coverage grows (adsorbate-adsorbate exclusion);
   calibrate it the same way against an adsorbate-covered patch.
4. Re-calibrate whenever the calculator, checkpoint, or relaxation settings
   (fmax, step cap) change. The value is tied to that specific setup.

Done when: each exchanged species and each host species has a measured radius, and the
cell's reported free volume is a plausible fraction of the window volume.

## Traps

- With a relaxing calculator, smaller-than-calibrated radii can be more efficient
  (fewer FIRE calls per accepted insertion: measured 898 vs 1230 for 0.5 vs 2.14) because
  relaxation rescues close trials. But changing radii changes the free volume and shifts
  the effective mu (~0.036 eV in that test) - never change radii mid-campaign.
- `species_radii={..: 0}` (as in the calibration script itself) means no exclusion;
  fine for calibration, wrong for production.
- Relation to the deprecated `min_insert` (still an `InsertionMove` parameter): ADR 0003/0004 -
  the cell's radii rule now owns both proposal region and acceptance volume; do not
  reintroduce a separate insertion cutoff.
