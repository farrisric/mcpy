from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from nvalchemi._typing import AtomCategory
from nvalchemi.data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.dynamics import NVTLangevin, initialize_velocities
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook


class _HeadMACEWrapper(MACEWrapper):
    """MACEWrapper that pins a multihead MACE model to one head.

    nvalchemi's batch carries no ``head`` field, so the inner MACE model falls
    back to head 0. Fine-tuned models often keep the pretrain head at index 0
    and the fine-tuned head elsewhere, so head 0 is the wrong potential. This
    injects a fixed head index into the MACE input.
    """

    def __init__(self, model: torch.nn.Module, head_index: int) -> None:
        super().__init__(model)
        self._head_index = head_index

    def adapt_input(self, data, **kwargs):
        d = super().adapt_input(data, **kwargs)
        d['head'] = torch.full((data.num_graphs,), self._head_index,
                               dtype=torch.long, device=data.positions.device)
        return d


def _load_model(
    checkpoint: Union[str, MACEWrapper],
    device: str,
    dtype: torch.dtype,
    enable_cueq: bool,
    compile_model: bool,
    head: Union[str, int, None] = None,
) -> MACEWrapper:
    if isinstance(checkpoint, MACEWrapper):
        return checkpoint
    # Local .model file: load directly. MACEWrapper.from_checkpoint treats the
    # string as a download alias, so it cannot open local paths. ``head``
    # selects a multihead model's head by name or index.
    if isinstance(checkpoint, (str, os.PathLike)) and os.path.exists(checkpoint):
        raw = torch.load(checkpoint, map_location=device, weights_only=False).to(dtype)
        # Resolve the head index before any cuEq conversion, while ``raw.heads``
        # is guaranteed present.
        idx = None if head is None else (
            head if isinstance(head, int) else list(raw.heads).index(head)
        )
        if enable_cueq:
            try:
                import cuequivariance  # noqa: F401
                from mace.cli.convert_e3nn_cueq import run as _to_cueq
            except ImportError as exc:
                raise ImportError(
                    "enable_cueq=True requires the 'cuequivariance' package; "
                    "install with: pip install 'nvalchemi-toolkit[mace]'"
                ) from exc
            raw = _to_cueq(raw, return_model=True, device=device)
        raw = raw.to(device)
        if compile_model:
            # Mirror MACEWrapper.from_checkpoint step 3: the model is
            # inference-only after this. The e3nn patch is a private nvalchemi
            # helper; benchmark/verify_compile_parity.py pins it so an upgrade
            # that renames it fails loudly.
            from nvalchemi.models.mace import _patch_e3nn_irrep_len_for_compile
            _patch_e3nn_irrep_len_for_compile()
            raw.eval()
            for param in raw.parameters():
                param.requires_grad = False
            raw = torch.compile(raw)
        return MACEWrapper(raw) if idx is None else _HeadMACEWrapper(raw, idx)
    if head is not None:
        raise ValueError(
            "head= is only supported when loading a local .model path; "
            f"got alias checkpoint {checkpoint!r}"
        )
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


def _fixed_indices(atoms: Atoms) -> List[int]:
    """Indices held by ASE ``FixAtoms`` constraints, sorted and de-duplicated."""
    idx: set[int] = set()
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            idx.update(int(i) for i in c.index)
    return sorted(idx)


def _freeze_hook_for(batch: Batch, fixed: List[int]) -> List[FreezeAtomsHook]:
    """Tag ``fixed`` batch rows as SPECIAL and return the hook that holds them.

    nvalchemi's FIRE has no notion of ASE ``FixAtoms``; without this the
    "fixed" atoms relax freely and are only snapped back afterward, leaving the
    returned energy inconsistent with the stored geometry. Marking them
    ``AtomCategory.SPECIAL`` and registering :class:`FreezeAtomsHook` zeros their
    forces/velocities and restores positions every FIRE step.
    """
    if not fixed:
        return None
    batch.atom_categories[fixed] = AtomCategory.SPECIAL.value
    return [FreezeAtomsHook()]


def _run_langevin_md(
    model: MACEWrapper,
    nl_config,
    atoms: Atoms,
    *,
    temperature: float,
    friction: float,
    dt: float,
    steps: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    max_neighbors: int | None,
) -> None:
    """Run NVT Langevin (BAOAB) MD in place on ``atoms`` using ``model``.

    Mirrors the FIRE bootstrap in :meth:`AlchemiFCalculator.get_potential_energy`:
    pre-allocate the tensors ``compute()`` writes into, seed velocities from a
    Maxwell-Boltzmann draw at ``temperature``, build the neighbor list once, then
    run ``steps`` of Langevin dynamics. ``FixAtoms``-constrained atoms are held
    every step by :class:`FreezeAtomsHook` and restored on write-back.

    ``dt`` is in fs, ``friction`` in 1/fs, ``temperature`` in K (the integrator
    converts to internal units). Mutates ``atoms`` in place; returns nothing.
    """
    batch = _make_batch(atoms, device, dtype)

    # compute() writes forces/energy via copy_() — pre-allocate the targets.
    batch.forces = torch.zeros_like(batch.positions)
    batch.energy = torch.zeros(1, 1, device=device, dtype=dtype)
    batch.velocities = torch.zeros_like(batch.positions)
    temp = torch.full((batch.num_graphs,), float(temperature), device=device, dtype=dtype)
    initialize_velocities(
        batch.velocities, batch.atomic_masses, temp, batch.batch_idx.int(),
        random_seed=seed,
    )

    freeze_hooks = _freeze_hook_for(batch, _fixed_indices(atoms))
    nl_hook = NeighborListHook(nl_config, max_neighbors=max_neighbors)
    opt = NVTLangevin(
        model=model,
        dt=dt,
        temperature=temperature,
        friction=friction,
        random_seed=seed,
        n_steps=steps,
        hooks=freeze_hooks,
    )
    opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

    # Bootstrap: build NL and compute initial forces before the MD loop.
    _build_nl(batch, nl_hook)
    opt.compute(batch)

    opt.run(batch)
    _write_back_positions(atoms, batch)
