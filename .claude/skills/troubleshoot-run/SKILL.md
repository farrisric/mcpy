---
name: troubleshoot-run
description: Use when a GCMC or replica-exchange run misbehaves - zero insertions, GPU OOM or memory climbing, dead replica ladder, unrepeatable results, frozen atoms moving, or energies that differ between runs.
---

# Troubleshoot a run

Match the symptom; each row is a known cause, not a guess.

| Symptom | First check | Fix |
|---|---|---|
| Zero insertions, everything else fine | Adsorption energy vs mu_ref | Often physics: weak binding + negative delta_mu. Verify with `choose-mu` step 3 before touching code. |
| Insertions proposed but all rejected with huge dE | `species_radii` | Radii too small: trials spawn inside atoms. `species-radii` skill. |
| `nvidia-smi` memory climbs over hours, false OOM | Live torch allocated vs reserved | Allocator fragmentation, not a leak. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`run-gpu` skill). |
| Real OOM at start | Atom count vs ceiling | Memory rules in `size-gpu-run`. |
| Replica ladder: isotherm flat, cumulative swap acceptance looks fine | Second-half acceptance per pair | Dead ladder; respace rungs (`replica-exchange-setup`). |
| Same script, different results | Any seed left `None` | Seed cells, moves, `MoveSelector`, ensemble. |
| Fixed/support atoms moved | mcpy version | Position write-back and `FixAtoms` bugs fixed on main (2026-05/06); `git pull` + reinstall editable. |
| Energies differ run-to-run, same structure | Checkpoint + cuEq + mcpy version | mh-1 with cuEq needs the load-time repair (on main since 2026-10-13); update mcpy. On an older mcpy, set `enable_cueq=False`. |
| Constraints corrupted after rejected deletions | mcpy version | Rollback bug fixed df95920 (2026-05-20); update. |
| torch SIGBUS on import | conda env | Wrong env; use `alchemi`, not `base`. |
| Desorbed atoms float in vacuum and N drifts up | Cell's `is_point_exchangeable` vs where they sit | An atom outside the exchangeable region is invisible to deletion and to the acceptance N, so it accumulates. `CustomCell` drops the z upper bound on purpose so desorbed species stay deletable; for `SphericalCell`/`DomeCell`, enlarge the region or lower mu so desorption stops. |
| Inserted atoms pair into gas molecules (e.g. O + O -> O2 above the surface) | mu vs mu_ref | Physics: mu at or above the gas reference condenses the reservoir phase. Lower delta_mu below 0 vs `derive_mu_gas`; shrink the cell height so trials land near the surface; if the stable adsorbate IS the molecule, switch to molecule moves (`molecule-gcmc` skill). |
| Step time grows with torch.compile | `compile_model` | Atom count changes each move; set `compile_model=False`. |

Done when: the symptom maps to a row and the fix is verified by re-running the failing case.

## If no row matches

Reproduce small and end-to-end first (a few hundred steps, CPU if possible), then bisect:
swap the calculator for EMT/LJ to split "MC logic" from "potential/GPU" causes.
