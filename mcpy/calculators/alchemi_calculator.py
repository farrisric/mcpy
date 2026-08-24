from __future__ import annotations

from typing import List, Union

import numpy as np
import torch
from ase import Atoms

from nvalchemi.models.mace import MACEWrapper
from nvalchemi.hooks.neighbor_list import NeighborListHook

from ..utils.chunking import chunk_ranges
from ._alchemi_common import (
    _build_nl,
    _load_model,
    _make_batch,
    _per_graph_energies,
    _MDMixin,
)


class AlchemiCalculator(_MDMixin):
    """
    Energy-only Alchemi calculator (no geometry relaxation).

    Single-point MACE evaluation on the nvalchemi GPU-native stack.
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
        chunk_size: int | None = None,
        energy_only: bool = False,
        head: Union[str, int, None] = None,
    ) -> None:
        if energy_only and isinstance(checkpoint, MACEWrapper):
            # energy_only mutates the wrapper's own model_config (below), and a
            # pre-loaded wrapper is shared by every calculator built from it --
            # including an AlchemiFCalculator whose FIRE relaxation needs the
            # forces this would switch off. There is no way to scope the change
            # to one calculator, so refuse instead of breaking the other one
            # silently and order-dependently.
            raise ValueError(
                "energy_only=True cannot be combined with a pre-loaded "
                "MACEWrapper: dropping 'forces' from its active outputs would "
                "also disable forces for every other calculator sharing that "
                "wrapper. Pass the checkpoint path instead, so this "
                "calculator loads its own model."
            )
        self.device = device
        self.dtype = dtype
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size
        self.energy_only = energy_only
        self.model = _load_model(checkpoint, device, dtype, enable_cueq, compile_model, head)
        self._nl_config = self.model.model_config.neighbor_config
        if energy_only:
            # MC energy evaluation never uses forces. Dropping 'forces' from
            # active_outputs sets compute_force=False in the MACE forward, so no
            # autograd graph is built — lower peak memory, energy unchanged up to
            # fp32 rounding. See docs/superpowers/specs/2026-06-09-...-design.md.
            self.model.model_config.active_outputs.discard('forces')

    def _forward(self, batch):
        """Model forward. The wrapper manages positions grad itself when forces
        are active; with energy_only nothing needs grad, and no_grad keeps the
        forward graph-free even when model parameters still require grad
        (compile_model=False local loads)."""
        if self.energy_only:
            with torch.no_grad():
                return self.model(batch)
        return self.model(batch)

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
        nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
        _build_nl(batch, nl_hook)
        out = self._forward(batch)
        return float(out['energy'].sum().item())

    def get_potential_energies(
        self, atoms_list: List[Atoms], chunk_size: int | None = None
    ) -> np.ndarray:
        """
        Batched forward pass over multiple (possibly differently sized) structures.

        One CUDA kernel launch per layer instead of N — the win behind batched
        replica exchange on a single GPU.

        ``chunk_size`` splits ``atoms_list`` into consecutive sub-batches of at
        most that many structures, evaluated one forward pass each, capping peak
        GPU memory at the largest chunk instead of the whole batch. ``None`` (the
        default) falls back to the instance ``chunk_size``; if that is also
        ``None`` the whole list is evaluated in a single pass. Per-graph energies
        are unaffected by chunking (MACE message passing is per-graph), up to
        GPU run-to-run noise.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            Structures to evaluate. Not mutated. Lengths may differ.
        chunk_size : int | None
            Per-call override of the instance ``chunk_size``.

        Returns
        -------
        np.ndarray
            Potential energies in eV, shape (len(atoms_list),).
        """
        if not atoms_list:
            return np.empty(0, dtype=np.float64)
        cs = self.chunk_size if chunk_size is None else chunk_size
        out: List[np.ndarray] = []
        for start, stop in chunk_ranges(len(atoms_list), cs):
            chunk = atoms_list[start:stop]
            batch = _make_batch(chunk, self.device, self.dtype)
            nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
            _build_nl(batch, nl_hook)
            model_out = self._forward(batch)
            out.append(_per_graph_energies(model_out['energy'], len(chunk)))
        return np.concatenate(out)
