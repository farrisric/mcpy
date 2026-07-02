# nvalchemi-toolkit — Learned API Notes

These notes are for the next Claude session working with `nvalchemi-toolkit` (v0.1.0).
Written after hands-on benchmarking + building an mcpy calculator prototype.

---

## Environment

```
conda activate alchemi   # Python 3.12
nvalchemi-toolkit==0.1.0
MACE==1.1.2 / mace-torch==0.3.15
torch==2.12.0+cu130
cuequivariance-torch==0.10.0
GPU: NVIDIA GeForce RTX 5090 (33.7 GB)
```

---

## Core data types

### `AtomicData`
Single-system graph. Created from ASE:
```python
from nvalchemi.data import AtomicData
data = AtomicData.from_atoms(atoms, device='cuda', dtype=torch.float32)
# fields: positions, atomic_numbers, cell, pbc, (optional) energy, forces, stress
```

### `Batch`
Multi-system graph batch. Created from a list of AtomicData:
```python
from nvalchemi.data.batch import Batch
batch = Batch.from_data_list([data1, data2, ...], device='cuda')
# fields accessible as batch.positions, batch.forces, etc.
# batch.num_graphs = number of systems
# batch.num_nodes  = total atoms
```

**Critical**: `batch.__setattr__` stores tensors in an internal `MultiLevelStorage`.
`compute()` writes outputs via `copy_()` — target tensors must exist first:
```python
batch.forces = torch.zeros_like(batch.positions)        # [N_atoms, 3]
batch.energy = torch.zeros(batch.num_graphs, 1, ...)    # [B, 1]
```
If you skip this, `compute()` silently does nothing and `batch.forces` raises `AttributeError`.

---

## Model: MACEWrapper

```python
from nvalchemi.models.mace import MACEWrapper

model = MACEWrapper.from_checkpoint(
    'medium-mpa-0',           # or local .pt path; downloads to ~/.cache/mace/
    device=torch.device('cuda'),
    dtype=torch.float32,
    enable_cueq=True,         # cuEquivariance — big speedup, requires cuequivariance pkg
    compile_model=True,       # torch.compile; ~30s warmup, faster after
)
```

`model.model_config` is a `ModelConfig` pydantic object:
- `model.model_config.neighbor_config` → `NeighborConfig(cutoff=6.0, format=COO, half_list=False, skin=0.0)`
- `model.model_config.active_outputs` → `{'forces', 'energy'}` by default

The model requires `batch.neighbor_list` (shape `[E, 2]`, int32) to be present before forward.
Build it with the NL hook (see below) — never call `model(batch)` without it.

---

## Neighbor list

`NeighborListHook` builds/updates `batch.neighbor_list` (COO format) before each compute.

```python
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.dynamics.base import DynamicsStage

nl_config = model.model_config.neighbor_config   # reuse from model
nl_hook   = NeighborListHook(nl_config)

# Standalone (outside dynamics loop):
ctx = HookContext(batch=batch, step_count=0)
nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)   # writes batch.neighbor_list

# Inside dynamics (register on optimizer):
opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)
# hook fires automatically each step
```

---

## FIRE optimizer — the bootstrap pattern

**Problem**: `BaseDynamics.step()` order is `pre_update → compute → post_update`.
FIRE's `pre_update` reads `batch.forces` *before* `compute` runs.
On step 1 forces don't exist yet → `AttributeError`.

**Solution**: bootstrap before calling `opt.run()`:
```python
from nvalchemi.dynamics import FIRE as AlchemiFIRE, ConvergenceHook
from nvalchemi.dynamics.base import DynamicsStage

batch.forces = torch.zeros_like(batch.positions)
batch.energy = torch.zeros(batch.num_graphs, 1, device='cuda', dtype=torch.float32)

nl_hook = NeighborListHook(nl_config)
opt = AlchemiFIRE(
    model=model,
    dt=1.0,          # NOT 0.1 — see "FIRE tuning" below
    convergence_hook=ConvergenceHook.from_fmax(0.05),
    n_steps=500,
)
opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

# Bootstrap: manually run NL + compute before the loop
ctx = HookContext(batch=batch, step_count=0)
nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)
opt.compute(batch)          # populates batch.forces / batch.energy via copy_()

opt.run(batch)              # now safe
print(opt.step_count)       # steps taken
```

`opt.run()` exits early if convergence_hook satisfied; otherwise runs n_steps.

---

## Convergence hook

```python
from nvalchemi.dynamics import ConvergenceHook

# fmax-style (max force norm, per atom)
conv = ConvergenceHook.from_fmax(threshold=0.05)

# multi-criterion
conv = ConvergenceHook(criteria=[
    {'key': 'forces', 'threshold': 0.05, 'reduce_op': 'norm', 'reduce_dims': -1},
    {'key': 'energy_change', 'threshold': 1e-6},
])
```

---

## Available dynamics

```
nvalchemi.dynamics:
  FIRE, FIRE2                          # geometry optimizers
  FIRE2VariableCell, FIREVariableCell  # cell + positions
  NVE, NVT (Langevin, NoseHoover)
  NPT, NPH
```

All share the same `BaseDynamics` API: `__init__(model, ..., hooks, convergence_hook, n_steps)` → `.run(batch)`.

---

## Hook registration

```python
from nvalchemi.dynamics.base import DynamicsStage

# Stages in order each step:
# BEFORE_STEP → BEFORE_PRE_UPDATE → pre_update → AFTER_PRE_UPDATE
# → BEFORE_COMPUTE → compute → AFTER_COMPUTE
# → BEFORE_POST_UPDATE → post_update → AFTER_POST_UPDATE
# → AFTER_STEP → (convergence check) → ON_CONVERGE

opt.register_hook(my_hook, stage=DynamicsStage.BEFORE_COMPUTE)
# hook must have .frequency (int) and .stage attributes
```

---

## Performance data (RTX 5090, MACE medium-mpa-0, fmax=0.05)

### vs ASE FIRE on CPU (historical — not relevant for GPU workflows)

| System size | ASE FIRE (CPU) | Alchemi FIRE | Speedup |
|------------:|---------------:|--------------:|--------:|
| 36 atoms    | 0.87s          | 0.90s         | 0.9x    |
| 72 atoms    | 2.5s           | 2.3s          | 1.1x    |
| 144 atoms   | 18s            | 1.1s          | **17x** |
| 288 atoms   | 171s           | 3.0s          | **58x** |

This table compares to **ASE FIRE on CPU**, which is unrealistic for production. If MACE is already on CUDA, the real comparison is below.

### vs MACE-on-CUDA via ASE FIRE (realistic, 2026-05 benchmark)

Single-shot FIRE relaxation, RTX 5090, MACE-MP-0 medium, dt tuned per side:

| N atoms | MACE fwd (ms) | Alchemi fwd (ms) | Fwd ratio | MACE FIRE iters | Alchemi FIRE iters | Net speedup |
|--------:|--------------:|------------------:|----------:|----------------:|-------------------:|------------:|
| 289     | 16            | 14                | 1.12x     | 89              | 161                | **0.61x** (loses)  |
| 586     | **38**        | **14**            | **2.65x** | 67              | 90                 | **1.96x**          |
| 976     | **64**        | **15**            | **4.17x** | 53              | 177                | **1.25x**          |

### vs MACE-on-CUDA via ASE LBFGS (production setup, 20-step GCMC)

20 GCMC steps on 586-atom Ag octahedron, MACE-MP-0 medium, RTX 5090,
`compile_model=False`, `dt=1.0`. This is the most realistic head-to-head
(LBFGS is mcpy's default ASE optimizer, FIRE is Alchemi's only relaxation):

| Calculator       | Wall (20 steps) | Per relax iter | Avg iters/call | **Net** |
|------------------|----------------:|---------------:|---------------:|--------:|
| MACE_F (LBFGS)   | 168.2 s         | 38.4 ms        | 218.8          | 1.0x    |
| AlchemiF (FIRE)  | **54.7 s**      | **14.3 ms**    | 190.9          | **3.08x** |

Final N_O / energy diverge between runs (LBFGS and FIRE trace different
minima → different acceptance paths in GCMC). This is not a model error.

**Alchemi forward time is ~flat in N** (kernel fusion + GPU-resident NL).
**MACE-via-ASE forward scales linearly in N** (per-atom kernel launch overhead).
**Crossover ~400 atoms** when MACE is on CUDA.

Caveats:
- Alchemi FIRE typically needs **~1.5-2x more iters** than ASE FIRE for the same fmax (Warp kernel uses different integration tuning). This drags the net speedup down.
- 5-step GCMC runs have high variance — one bad relax dominates. Longer runs settle higher speedups.
- Smaller systems (<300 atoms) on GPU: just use `MACE_F_Calculator(optimizer='lbfgs')` — Alchemi has no win.

---

## mcpy integration

Calculator at:
`/home/energystorage/projects/mcpy/mcpy/calculators/alchemi_calculator.py`

Two classes:
- `AlchemiCalculator` — energy only, single forward pass
- `AlchemiFCalculator` — FIRE/FIRE2 relax then return energy (drop-in for `MACE_F_Calculator`)

Recommended GCMC config (proven by benchmarks):
```python
from mcpy.calculators import AlchemiFCalculator

calc = AlchemiFCalculator(
    checkpoint='medium-mpa-0',
    steps=500,
    fmax=0.05,
    device='cuda',
    enable_cueq=True,
    compile_model=True,    # ~30-60s warmup + one recompile on first N-change, then faster
    dt=1.0,                # NOT 0.1 — see "FIRE tuning" below
    optimizer='fire',      # 'fire2' has conservative defaults that converge slower
)
energy = calc.get_potential_energy(atoms)
```

Share a pre-loaded model to avoid reloading per-ensemble:
```python
from nvalchemi.models.mace import MACEWrapper
model = MACEWrapper.from_checkpoint('medium-mpa-0', enable_cueq=True, compile_model=True, ...)
calc  = AlchemiFCalculator(checkpoint=model, steps=500, fmax=0.05, dt=1.0)
```

### FIRE tuning (2026-05 findings)

| dt   | Avg FIRE steps (single relax, 289 atoms) |
|------|------------------------------------------:|
| 0.01 | 311 |
| 0.05 | 99  |
| **0.10 (old default)** | **85** |
| 0.50 | 55  |
| **1.00 (new default)** | **47** |

Why: ASE FIRE starts at `dt=0.1 fs` and grows to `dtmax=1.0 fs` quickly, so most of its run is at dt=1.0. nvalchemi FIRE has `dt_max = dt * 10.0` by default — starting at `dt=1.0` lands near ASE's plateau. Starting at `dt=0.1` wastes steps ramping up.

FIRE2 defaults (`delaystep=60`, conservative growth) converge slower than tuned FIRE in our tests. Stick with `'fire'`.

### `torch.compile` in GCMC — yes (guidance reversed 2026-07-02)

An earlier version of these notes said compile recompiles on every atom-count change and measured energy-only GCMC 7x slower.
That is obsolete on the current torch/nvalchemi stack: automatic dynamic shapes kick in after the first N-change, so a variable-N run pays one initial compile plus one recompile, then every atom count runs on the cached dynamic graph.
Measured on the RTX 5090 (medium-mpa-0, cuEq, fp32, GCMC-like N walk):

| Path | uncompiled | compiled | gain |
|------|-----------:|---------:|-----:|
| FIRE relax, per FIRE step (~200-290 at) | 15.7 ms | 12.0 ms | 1.30x |
| FIRE relax, Ag201+O GCMC wall (1600 steps) | 2.89 s/step | 2.02 s/step | 1.43x |
| Energy-only, per eval (~200-220 at, varying N) | 15.6 ms median | 12.1 ms median | 1.3x (2x on mean) |
| Forward at 3.7k atoms | 67.7 ms | 34.0 ms | 2.0x |

Warmup cost: ~10-60 s total, once per process.
`benchmark/gcmc_compile_ab.py` reproduces the A/B; `benchmark/verify_compile_parity.py` checks compiled-vs-uncompiled energies (<= 0.7 meV observed) and must pass after nvalchemi upgrades.

For fixed-N MD and variable-N GCMC alike: **`compile_model=True`** (the default).
Set `compile_model=False` only for short smoke tests where the warmup dominates.

---

## Common pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `AttributeError: 'Batch' has no attribute 'forces'` | No bootstrap before `opt.run()`, or `compute()` target missing | Pre-alloc `batch.forces`/`batch.energy`, build NL, call `opt.compute(batch)` |
| `AttributeError: 'Batch' has no attribute 'neighbor_list'` | Model called before NL built | Build NL via hook before any `model(batch)` call |
| `compute()` silently no-ops | Target tensors absent; `copy_()` skips None targets | Pre-alloc as above |
| Slow on small systems | Alchemi overhead > GPU benefit below ~400 atoms when MACE is also on CUDA | Use `MACE_F_Calculator(optimizer='lbfgs')` for <400-atom GCMC; Alchemi for ≥500 atoms |
| Long pauses early in a GCMC run | torch.compile warmup + one recompile on the first atom-count change | Expected, once per process (~10-60 s total); only disable compile for short smoke tests |
| FIRE needs hundreds of steps to converge | Default `dt=0.1` is too small for nvalchemi (its `dt_max=dt*10`) | Use `dt=1.0` — halves the step count |
| Final energy differs between MACE and Alchemi by ≫0.01 eV | Models silently misaligned (`mace_mp('medium')` ≠ `'medium-mpa-0'`) | Pass identical `--checkpoint` to both sides — they resolve to different cached files |
| `OptionalDependencyError: mace not installed` | Old error — MACE is installed | Was a stale notebook cell; `pip install 'nvalchemi-toolkit[mace]'` if fresh env |
