---
name: run-gpu
description: Launch any mcpy run that imports torch, mace or nvalchemi (MACE or Alchemi calculators, GPU GCMC, batched replica exchange). Use before running such a script or sizing a system for the GPU.
---

# Run mcpy on the GPU

## Steps

1. Run the script with the conda env that has torch, mace and nvalchemi.
   On the reference workstation it is `alchemi`: `conda run -n alchemi python <script>`.
   The `base` env torch crashes on import (SIGBUS); keep `base` for torch-free work (pytest, flake8, ASE-only builds, plotting).
2. Launch every run longer than a few steps with the allocator setting:
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
   Reason: GCMC changes the atom count each move, so the CUDA cache fragments.
   `nvidia-smi` memory then climbs to the card limit and gives false OOMs, while live memory stays flat.
   This is not a leak; the env var is the fix.
3. Check the system fits: memory rules, ceilings, and `chunk_size` choice live in the
   `size-gpu-run` skill (single source for the measured numbers).
4. Derive reference chemical potentials at runtime with `mcpy.utils.reference_mu` (`derive_mu_gas`, `derive_mu_bulk`).
   Each checkpoint gives a different E(O2)/2, so a hardcoded mu from another model is wrong.

Done when: the run starts in the torch env, with the env var set, and the atom count is inside the ceiling.

## Model gotchas

- `compile_model=False` for GCMC and replica exchange: the atom count changes every move, so torch.compile recompiles each new size and never pays back.
- mh-1 checkpoint: keep cuEq on (`enable_cueq=True`); it is ~3x faster and ~3x lighter than e3nn.
  mcpy repairs the checkpoint on load (upstream ACEsuit/mace#1298) and turns `compile_model` off for it: cuEq + torch.compile gives wrong mh-1 forces.
- Under ~200 atoms per force call (`chunk_size` x atoms per replica): set `enable_cueq=False`.
  cuEq has the higher fixed cost per call, so e3nn wins until the GPU fills.
  `python benchmark/force_call_scaling.py` measures the floor and the crossover for a model.
- `max_neighbors` does not change memory in the nvalchemi path; look elsewhere for memory savings.
