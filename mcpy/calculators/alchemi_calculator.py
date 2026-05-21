"""
Alchemi (nvalchemi) calculator for mcpy.

Two classes:
  AlchemiCalculator      — energy-only (no relaxation), fast single forward pass
  AlchemiFCalculator     — FIRE geometry relaxation then energy, mirrors MACE_F_Calculator

Both accept an ASE Atoms object and return a float (eV).

Requirements:
  pip install 'nvalchemi-toolkit[mace]'

Notes on the bootstrap pattern
-------------------------------
nvalchemi's FIRE optimizer runs pre_update BEFORE compute on every step.
On step 1 the batch has no forces yet, so we must:
  1. Pre-allocate batch.forces and batch.energy (compute() writes via copy_()).
  2. Build the neighbor list once via NeighborListHook before calling compute().
  3. Call opt.compute(batch) to populate initial forces.
  4. Then call opt.run(batch).
"""

from __future__ import annotations

from typing import Union

import torch
from ase import Atoms

from nvalchemi.data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.dynamics import FIRE as AlchemiFIRE, FIRE2 as AlchemiFIRE2, ConvergenceHook
from nvalchemi.dynamics.base import DynamicsStage


_ALCHEMI_OPTIMIZERS = {'fire': AlchemiFIRE, 'fire2': AlchemiFIRE2}


def _load_model(
    checkpoint: Union[str, MACEWrapper],
    device: str,
    dtype: torch.dtype,
    enable_cueq: bool,
    compile_model: bool,
) -> MACEWrapper:
    if isinstance(checkpoint, MACEWrapper):
        return checkpoint
    return MACEWrapper.from_checkpoint(
        checkpoint,
        device=torch.device(device),
        dtype=dtype,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
    )


def _make_batch(atoms: Atoms, device: str, dtype: torch.dtype) -> Batch:
    data = AtomicData.from_atoms(atoms, device=device, dtype=dtype)
    return Batch.from_data_list([data], device=device)


def _build_nl(batch: Batch, nl_hook: NeighborListHook) -> None:
    ctx = HookContext(batch=batch, step_count=0)
    nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)


class AlchemiCalculator:
    """
    Energy-only Alchemi calculator (no geometry relaxation).

    Equivalent to MACECalculator but uses the nvalchemi GPU-native stack.
    Use this when mcpy handles its own MC moves and only needs E(atoms).

    Parameters
    ----------
    checkpoint : str | MACEWrapper
        Named checkpoint (e.g. 'medium-mpa-0'), local .pt path,
        or a pre-loaded MACEWrapper to share across calculators.
    device : str
        'cuda' or 'cpu'.
    dtype : torch.dtype
        Model and data dtype. float32 recommended for speed.
    enable_cueq : bool
        cuEquivariance kernel fusion — significant speedup on GPU.
    compile_model : bool
        torch.compile the model. Adds ~30s warmup, then faster.
    """

    def __init__(
        self,
        checkpoint: Union[str, MACEWrapper] = 'medium-mpa-0',
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        enable_cueq: bool = True,
        compile_model: bool = True,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model = _load_model(checkpoint, device, dtype, enable_cueq, compile_model)
        self._nl_config = self.model.model_config.neighbor_config

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Single forward pass — no geometry relaxation.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to evaluate. Not mutated.

        Returns
        -------
        float
            Potential energy in eV.
        """
        batch = _make_batch(atoms, self.device, self.dtype)
        # positions must have grad enabled — MACEWrapper always computes forces via autograd
        batch.positions.requires_grad_(True)
        nl_hook = NeighborListHook(self._nl_config)
        _build_nl(batch, nl_hook)
        out = self.model(batch)
        return float(out['energy'].sum().item())


class AlchemiFCalculator:
    """
    Alchemi calculator with FIRE geometry relaxation.

    Mirrors MACE_F_Calculator: relax atoms with FIRE then return energy.
    Uses the fully GPU-resident nvalchemi dynamics stack —
    significantly faster than ASE FIRE above ~100 atoms.

    Parameters
    ----------
    checkpoint : str | MACEWrapper
        Named checkpoint, local .pt path, or pre-loaded MACEWrapper.
    steps : int
        Maximum FIRE steps before returning.
    fmax : float
        Force convergence threshold in eV/Å.
    device : str
        'cuda' or 'cpu'.
    dtype : torch.dtype
        float32 recommended; float64 for higher accuracy.
    enable_cueq : bool
        cuEquivariance kernel fusion.
    compile_model : bool
        torch.compile. Best to pre-warm in __init__ if reusing.
    dt : float
        FIRE initial timestep (default 1.0). Matches ASE FIRE's dtmax;
        benchmarks show dt=1.0 converges in ~half the steps vs dt=0.1.
    optimizer : str
        'fire' (default, classic FIRE) or 'fire2' (Guénolé et al variant —
        typically converges in fewer steps).
    """

    def __init__(
        self,
        checkpoint: Union[str, MACEWrapper] = 'medium-mpa-0',
        steps: int = 500,
        fmax: float = 0.05,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        enable_cueq: bool = True,
        compile_model: bool = True,
        dt: float = 1.0,
        optimizer: str = 'fire',
    ) -> None:
        self.steps = steps
        self.fmax = fmax
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.last_relax_steps = 0
        self.total_relax_steps = 0
        if optimizer not in _ALCHEMI_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {list(_ALCHEMI_OPTIMIZERS)}, got {optimizer!r}"
            )
        self._optimizer_cls = _ALCHEMI_OPTIMIZERS[optimizer]
        self.optimizer_name = optimizer
        self.model = _load_model(checkpoint, device, dtype, enable_cueq, compile_model)
        self._nl_config = self.model.model_config.neighbor_config

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Relax with Alchemi FIRE, then return the relaxed potential energy.

        Parameters
        ----------
        atoms : ase.Atoms
            Starting geometry. Not mutated (a copy is used internally).

        Returns
        -------
        float
            Relaxed potential energy in eV.
        """
        batch = _make_batch(atoms, self.device, self.dtype)

        # compute() writes via copy_() — pre-allocate target tensors
        batch.forces = torch.zeros_like(batch.positions)
        batch.energy = torch.zeros(1, 1, device=self.device, dtype=self.dtype)

        nl_hook = NeighborListHook(self._nl_config)
        opt = self._optimizer_cls(
            model=self.model,
            dt=self.dt,
            convergence_hook=ConvergenceHook.from_fmax(self.fmax),
            n_steps=self.steps,
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

        # Bootstrap: build NL and compute initial forces before the FIRE loop
        _build_nl(batch, nl_hook)
        opt.compute(batch)

        opt.run(batch)
        self.last_relax_steps = int(opt.step_count)
        self.total_relax_steps += self.last_relax_steps
        return float(batch.energy.sum().item())
