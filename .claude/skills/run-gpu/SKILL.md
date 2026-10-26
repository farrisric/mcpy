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
3. Check the system fits (reference: RTX 5090, 32 GB, medium-mpa-0, fp32, cuEq):
   - FIRE relax: about 0.1 GiB + 1.3 MiB per atom (supported oxide); about 0.9 MiB per atom (bare metal NP).
   - Relax ceiling: about 32k atoms bare NP, about 23k atoms supported oxide.
   - Batched replica exchange: set `chunk_size` on the calculator so peak memory follows the chunk, not the replica count.
     Pick the largest chunk that fits; `chunk_size=1` is many times slower.
4. Derive reference chemical potentials at runtime with `mcpy.utils.reference_mu` (`derive_mu_gas`, `derive_mu_bulk`).
   Each checkpoint gives a different E(O2)/2, so a hardcoded mu from another model is wrong.

Done when: the run starts in the torch env, with the env var set, and the atom count is inside the ceiling.

## Model gotchas

- `compile_model=False` for GCMC and replica exchange: the atom count changes every move, so torch.compile recompiles each new size and never pays back.
- mh-1 checkpoint: run with cuEq off (`enable_cueq=False`).
  With cuEq on, its energies are wrong and change from load to load (upstream ACEsuit/mace#1298).
- `max_neighbors` does not change memory in the nvalchemi path; look elsewhere for memory savings.
