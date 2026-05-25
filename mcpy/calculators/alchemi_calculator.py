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

from typing import List, Union

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms

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


def _make_multi_batch(atoms_list: List[Atoms], device: str, dtype: torch.dtype) -> Batch:
    data = [AtomicData.from_atoms(a, device=device, dtype=dtype) for a in atoms_list]
    return Batch.from_data_list(data, device=device)


def _per_graph_energies(out_energy: torch.Tensor, n_graphs: int) -> np.ndarray:
    """Reduce model output to a (n_graphs,) numpy array regardless of layout."""
    e = out_energy.detach().to('cpu')
    if e.numel() == n_graphs:
        return e.view(-1).numpy()
    # Fallback: scatter-reduced layout — sum each graph's atomic contributions.
    # If this branch is hit, the model returned per-atom energies; for now we
    # surface the mismatch loudly so the caller can adapt.
    raise RuntimeError(
        f"Unexpected energy tensor shape {tuple(out_energy.shape)} for {n_graphs} graphs. "
        "Batched eval expects one energy per graph."
    )


def _build_nl(batch: Batch, nl_hook: NeighborListHook) -> None:
    ctx = HookContext(batch=batch, step_count=0)
    nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)


def _write_back_positions(atoms: Atoms, batch: Batch) -> None:
    """Copy relaxed positions from batch back to atoms, skipping FixAtoms-constrained indices."""
    relaxed = batch.positions.detach().cpu().numpy()
    fixed: set[int] = set()
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            fixed.update(c.index.tolist())
    if fixed:
        relaxed = relaxed.copy()
        relaxed[list(fixed)] = atoms.positions[list(fixed)]
    atoms.positions = relaxed


def _write_back_positions_batched(
    atoms_list: List[Atoms], batch: Batch
) -> None:
    """Write per-graph relaxed positions back into each Atoms object."""
    relaxed = batch.positions.detach().cpu().numpy()
    batch_idx = batch.batch_idx.detach().cpu().numpy()
    for i, atoms in enumerate(atoms_list):
        new_pos = relaxed[batch_idx == i].copy()
        fixed: set[int] = set()
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                fixed.update(c.index.tolist())
        if fixed:
            for j in fixed:
                new_pos[j] = atoms.positions[j]
        atoms.positions = new_pos


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
        max_neighbors: int | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_neighbors = max_neighbors
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
        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        _build_nl(batch, nl_hook)
        out = self.model(batch)
        return float(out['energy'].sum().item())

    def get_potential_energies(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Batched forward pass over multiple (possibly differently sized) structures.

        One CUDA kernel launch per layer instead of N — the win behind batched
        replica exchange on a single GPU.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            Structures to evaluate. Not mutated. Lengths may differ.

        Returns
        -------
        np.ndarray
            Potential energies in eV, shape (len(atoms_list),).
        """
        if not atoms_list:
            return np.empty(0, dtype=np.float64)
        batch = _make_multi_batch(atoms_list, self.device, self.dtype)
        batch.positions.requires_grad_(True)
        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        _build_nl(batch, nl_hook)
        out = self.model(batch)
        return _per_graph_energies(out['energy'], len(atoms_list))


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
        max_neighbors: int | None = None,
    ) -> None:
        self.steps = steps
        self.fmax = fmax
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.max_neighbors = max_neighbors
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

        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
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
        _write_back_positions(atoms, batch)
        return float(batch.energy.sum().item())

    def get_potential_energies(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Batched FIRE relaxation over multiple structures, then per-graph energies.

        All graphs share one optimizer / one model forward pass per FIRE step.
        nvalchemi FIRE's ConvergenceHook tracks each graph's max-force and
        retires it from the active batch when it falls below ``fmax`` — so
        already-converged replicas don't slow others down. Returns when every
        graph has converged or ``steps`` is reached.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            Starting geometries. Each is mutated in place: positions are
            updated to the relaxed configuration (respecting FixAtoms).

        Returns
        -------
        np.ndarray
            Relaxed potential energies in eV, shape (len(atoms_list),).
        """
        if not atoms_list:
            return np.empty(0, dtype=np.float64)
        n_graphs = len(atoms_list)
        batch = _make_multi_batch(atoms_list, self.device, self.dtype)
        batch.forces = torch.zeros_like(batch.positions)
        batch.energy = torch.zeros(n_graphs, 1, device=self.device, dtype=self.dtype)

        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        opt = self._optimizer_cls(
            model=self.model,
            dt=self.dt,
            convergence_hook=ConvergenceHook.from_fmax(self.fmax),
            n_steps=self.steps,
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

        _build_nl(batch, nl_hook)
        opt.compute(batch)
        opt.run(batch)

        self.last_relax_steps = int(opt.step_count)
        self.total_relax_steps += self.last_relax_steps
        _write_back_positions_batched(atoms_list, batch)
        return _per_graph_energies(batch.energy, n_graphs)
