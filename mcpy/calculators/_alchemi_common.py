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
from nvalchemi.hooks import DynamicsContext
from nvalchemi.dynamics import NVTLangevin, initialize_velocities
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook, NaNDetectorHook


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
        if head is not None:
            raise ValueError(
                'head= cannot be applied to a pre-loaded MACEWrapper: the '
                'head is baked in at wrap time, so it would be silently '
                'ignored and a multihead model would evaluate on head 0 '
                '(usually the pretrain head). Load from the .model path '
                'with head=, or wrap the raw model in _HeadMACEWrapper '
                'yourself before sharing it.'
            )
        return checkpoint
    # The donated_buffer=False workaround (b333c8e) is gone: the crash it
    # papered over came from the wrapper running in train mode
    # (create_graph=True made the FIRE force backward need the buffers the
    # compiled backward donates); with wrapper.eval() below, donation is safe
    # (verified compile+cuEq, alias and local paths, varying-N GCMC).
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
        wrapper = MACEWrapper(raw) if idx is None else _HeadMACEWrapper(raw, idx)
        # MACEWrapper.forward passes training=self.training down to MACE, where
        # it becomes create_graph/retain_graph on the force autograd. nn.Module
        # defaults to train mode, and from_checkpoint's wrapper.eval() (alias
        # path below) never runs here — without this every force pass builds a
        # retained second-order graph.
        wrapper.eval()
        return wrapper
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


def _make_batch(atoms_list, device: str, dtype: torch.dtype) -> Batch:
    """Batch one ``Atoms`` or a list of them; a single structure is a batch of one."""
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    data = [AtomicData.from_atoms(a, device=device, dtype=dtype) for a in atoms_list]
    return Batch.from_data_list(data, device=device)


def _per_graph_energies(out_energy: torch.Tensor, n_graphs: int) -> np.ndarray:
    """Reduce model output to a (n_graphs,) numpy array regardless of layout."""
    e = out_energy.detach().to('cpu')
    if e.numel() == n_graphs:
        return e.view(-1).numpy()
    raise RuntimeError(
        f"Unexpected energy tensor shape {tuple(out_energy.shape)} for {n_graphs} graphs. "
        "Batched eval expects one energy per graph."
    )


def _build_nl(batch: Batch, nl_hook: NeighborListHook) -> None:
    ctx = DynamicsContext(batch=batch, step_count=0)
    nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)


def _prepare_batch(batch: Batch, n_graphs: int, device: str,
                   dtype: torch.dtype) -> None:
    """Pre-allocate the tensors ``compute()`` writes into via ``copy_()``."""
    batch.forces = torch.zeros_like(batch.positions)
    batch.energy = torch.zeros(n_graphs, 1, device=device, dtype=dtype)


def _bootstrap(opt, batch: Batch, nl_hook: NeighborListHook,
               fixed: List[int]) -> None:
    """Build the neighbor list and the initial forces before the loop starts.

    ``FreezeAtomsHook`` zeros frozen forces only at ``AFTER_POST_UPDATE``, and
    this ``compute()`` runs outside the hook loop, so step 0 would otherwise
    displace the frozen rows once.
    """
    _build_nl(batch, nl_hook)
    opt.compute(batch)
    if fixed:
        batch.forces[fixed] = 0.0


def _write_back_positions(atoms: Atoms, batch: Batch) -> None:
    """Copy relaxed positions from batch back to atoms, skipping FixAtoms-constrained indices."""
    # float64: the batch is float32, and assigning the float64 frozen rows
    # into a float32 array would truncate them, leaving the "restored" atoms
    # about 5e-7 A off their pinned positions at slab-sized coordinates.
    relaxed = batch.positions.detach().cpu().numpy().astype(np.float64)
    fixed = _fixed_indices(atoms)
    if fixed:
        relaxed[fixed] = atoms.positions[fixed]
    atoms.positions = relaxed


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
    _prepare_batch(batch, 1, device, dtype)
    batch.velocities = torch.zeros_like(batch.positions)
    temp = torch.full((batch.num_graphs,), float(temperature), device=device, dtype=dtype)
    initialize_velocities(
        batch.velocities, batch.atomic_masses, temp, batch.batch_idx.int(),
        random_seed=seed,
    )

    fixed = _fixed_indices(atoms)
    freeze_hooks = _freeze_hook_for(batch, fixed)
    nl_hook = NeighborListHook(nl_config, max_neighbors=max_neighbors)
    opt = NVTLangevin(
        model=model,
        dt=dt,
        temperature=temperature,
        friction=friction,
        random_seed=seed,
        n_steps=steps,
        # NaN forces/energy would otherwise propagate silently into the
        # proposal the ensemble scores; the hook raises at the offending step.
        hooks=(freeze_hooks or []) + [NaNDetectorHook()],
    )
    opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)

    _bootstrap(opt, batch, nl_hook, fixed)
    opt.run(batch)
    _write_back_positions(atoms, batch)


class _MDMixin:
    """``run_md`` for both Alchemi calculators.

    Identical in each, docstring included, so it lives here. The energy_only
    guard applies only to :class:`AlchemiCalculator`; the F variant has no such
    attribute and rejects a forces-less model in its constructor instead.
    """

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
        if getattr(self, 'energy_only', False):
            # The integrator would otherwise die steps deep inside nvalchemi
            # ("NVTLangevin requires forces...") without naming the cause.
            raise ValueError(
                'run_md needs forces, but this calculator was built with '
                'energy_only=True (forces are stripped from the model '
                'outputs). Build a separate calculator without energy_only '
                'for MD.'
            )
        _run_langevin_md(
            self.model, self._nl_config, atoms,
            temperature=temperature, friction=friction, dt=dt, steps=steps,
            seed=seed, device=self.device, dtype=self.dtype,
            max_neighbors=self.max_neighbors,
        )
