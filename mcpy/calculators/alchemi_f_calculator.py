"""
Alchemi (nvalchemi) calculator with FIRE geometry relaxation.

``AlchemiFCalculator`` relaxes atoms with nvalchemi's GPU-resident FIRE before
returning the energy, mirroring MACE_F_Calculator. For an energy-only single
forward pass, see :class:`AlchemiCalculator` in ``alchemi_calculator.py``. Shared
internals (model loading, batching, neighbor-list bootstrap, FixAtoms handling,
MD) live in ``_alchemi_common.py``.

Requirements:
  pip install 'nvalchemi-toolkit[mace]'
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np
import torch
from ase import Atoms

from nvalchemi.models.mace import MACEWrapper
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics import FIRE as AlchemiFIRE, FIRE2 as AlchemiFIRE2, ConvergenceHook
from nvalchemi.dynamics.base import DynamicsStage

from ..utils.chunking import chunk_ranges
from ._alchemi_common import (
    _build_nl,
    _fixed_indices,
    _freeze_hook_for,
    _load_model,
    _make_batch,
    _make_multi_batch,
    _per_graph_energies,
    _run_langevin_md,
    _write_back_positions,
    _write_back_positions_batched,
)


logger = logging.getLogger(__name__)

_ALCHEMI_OPTIMIZERS = {'fire': AlchemiFIRE, 'fire2': AlchemiFIRE2}


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
    chunk_size : int | None
        Default sub-batch size for batched relaxation (see
        ``get_potential_energies``). ``None`` relaxes the whole batch at once.
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
        chunk_size: int | None = None,
        head: Union[str, int, None] = None,
    ) -> None:
        self.steps = steps
        self.fmax = fmax
        self.device = device
        self.dtype = dtype
        self.dt = dt
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size
        self.last_relax_steps = 0
        self.total_relax_steps = 0
        if optimizer not in _ALCHEMI_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {list(_ALCHEMI_OPTIMIZERS)}, got {optimizer!r}"
            )
        self._optimizer_cls = _ALCHEMI_OPTIMIZERS[optimizer]
        self.optimizer_name = optimizer
        self.model = _load_model(checkpoint, device, dtype, enable_cueq, compile_model, head)
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

        freeze_hooks = _freeze_hook_for(batch, _fixed_indices(atoms))

        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        opt = self._optimizer_cls(
            model=self.model,
            dt=self.dt,
            convergence_hook=ConvergenceHook.from_fmax(self.fmax),
            n_steps=self.steps,
            hooks=freeze_hooks,
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

        # Bootstrap: build NL and compute initial forces before the FIRE loop
        _build_nl(batch, nl_hook)
        opt.compute(batch)

        opt.run(batch)
        self.last_relax_steps = int(opt.step_count)
        self.total_relax_steps += self.last_relax_steps
        logger.debug("FIRE relaxation: %d/%d steps (fmax=%.3g eV/A)",
                     self.last_relax_steps, self.steps, self.fmax)
        _write_back_positions(atoms, batch)
        return float(batch.energy.sum().item())

    def _relax_batch(self, atoms_list: List[Atoms]) -> tuple:
        """Batched FIRE relaxation of one (sub-)batch. Mutates positions in
        place; returns ``(per_graph_energies, step_count)``."""
        n_graphs = len(atoms_list)
        batch = _make_multi_batch(atoms_list, self.device, self.dtype)
        batch.forces = torch.zeros_like(batch.positions)
        batch.energy = torch.zeros(n_graphs, 1, device=self.device, dtype=self.dtype)

        # Map each graph's FixAtoms indices into the concatenated (sub-)batch.
        fixed: List[int] = []
        offset = 0
        for a in atoms_list:
            fixed.extend(offset + i for i in _fixed_indices(a))
            offset += len(a)
        freeze_hooks = _freeze_hook_for(batch, fixed)

        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        opt = self._optimizer_cls(
            model=self.model,
            dt=self.dt,
            convergence_hook=ConvergenceHook.from_fmax(self.fmax),
            n_steps=self.steps,
            hooks=freeze_hooks,
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

        _build_nl(batch, nl_hook)
        opt.compute(batch)
        opt.run(batch)

        _write_back_positions_batched(atoms_list, batch)
        return _per_graph_energies(batch.energy, n_graphs), int(opt.step_count)

    def get_potential_energies(
        self, atoms_list: List[Atoms], chunk_size: int | None = None
    ) -> np.ndarray:
        """
        Batched FIRE relaxation over multiple structures, then per-graph energies.

        All graphs in a (sub-)batch share one optimizer / one model forward pass
        per FIRE step. nvalchemi FIRE's ConvergenceHook tracks each graph's
        max-force and retires it from the active batch when it falls below
        ``fmax`` — so already-converged replicas don't slow others down. Returns
        when every graph has converged or ``steps`` is reached.

        ``chunk_size`` splits ``atoms_list`` into consecutive sub-batches of at
        most that many structures, each relaxed independently, capping peak GPU
        memory at the largest chunk. ``None`` (the default) falls back to the
        instance ``chunk_size``; if that is also ``None`` the whole list is
        relaxed at once. FIRE dynamics are per-graph (forces and the convergence
        criterion are per-graph), so a graph's relaxed energy is independent of
        which others share its sub-batch, up to GPU run-to-run noise.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            Starting geometries. Each is mutated in place: positions are
            updated to the relaxed configuration (respecting FixAtoms).
        chunk_size : int | None
            Per-call override of the instance ``chunk_size``.

        Returns
        -------
        np.ndarray
            Relaxed potential energies in eV, shape (len(atoms_list),).
        """
        if not atoms_list:
            return np.empty(0, dtype=np.float64)
        cs = self.chunk_size if chunk_size is None else chunk_size
        out: List[np.ndarray] = []
        steps: List[int] = []
        for start, stop in chunk_ranges(len(atoms_list), cs):
            energies, step_count = self._relax_batch(atoms_list[start:stop])
            out.append(energies)
            steps.append(step_count)
        # last = deepest chunk relaxation; total accumulates work across chunks.
        self.last_relax_steps = max(steps)
        self.total_relax_steps += sum(steps)
        logger.debug("FIRE relaxation (batched, %d graphs, %d chunk(s)): "
                     "max %d/%d steps (fmax=%.3g eV/A)",
                     len(atoms_list), len(steps), self.last_relax_steps,
                     self.steps, self.fmax)
        return np.concatenate(out)

    def run_md(
        self,
        atoms: Atoms,
        *,
        temperature: float,
        friction: float = 0.01,
        dt: float = 2.0,
        steps: int = 100,
        seed: int = 42,
    ) -> None:
        """Run NVT Langevin MD in place on ``atoms`` (Maxwell-Boltzmann IC at T).

        Reuses this calculator's model and neighbor-list config, so no second
        model is loaded. ``friction`` is in 1/fs, ``dt`` in fs, ``temperature``
        in K. ``FixAtoms`` constraints are honored.
        """
        _run_langevin_md(
            self.model, self._nl_config, atoms,
            temperature=temperature, friction=friction, dt=dt, steps=steps,
            seed=seed, device=self.device, dtype=self.dtype,
            max_neighbors=self.max_neighbors,
        )
