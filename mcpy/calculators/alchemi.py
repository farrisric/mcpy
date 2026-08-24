"""GPU-native MACE calculators on the nvalchemi stack.

Two calculators over one base: :class:`AlchemiCalculator` evaluates energies
with a single forward pass, :class:`AlchemiFCalculator` relaxes with FIRE first.
Everything they share -- loading and configuring the model, batching, the
neighbour-list hook, the chunked batched-evaluation template, Langevin MD --
lives on :class:`_AlchemiBase`, so a subclass only says what is different about
its per-chunk evaluation.

Both are optional: importing this module needs ``nvalchemi-toolkit[mace]``.
"""
from __future__ import annotations

import logging
import os
from typing import List, Union

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from nvalchemi._typing import AtomCategory
from nvalchemi.data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.dynamics import (FIRE as AlchemiFIRE, FIRE2 as AlchemiFIRE2,
                                ConvergenceHook, NVTLangevin,
                                initialize_velocities)
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook, NaNDetectorHook
from nvalchemi.hooks import DynamicsContext
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

from ..utils.chunking import chunk_ranges

logger = logging.getLogger(__name__)

_ALCHEMI_OPTIMIZERS = {'fire': AlchemiFIRE, 'fire2': AlchemiFIRE2}


# --------------------------------------------------------------- model loading

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


# ------------------------------------------------------------ batch/atoms glue

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
    """Run the neighbour-list hook once, outside any dynamics loop."""
    ctx = DynamicsContext(batch=batch, step_count=0)
    nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)


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
    forces/velocities and restores positions every step.
    """
    if not fixed:
        return None
    batch.atom_categories[fixed] = AtomCategory.SPECIAL.value
    return [FreezeAtomsHook()]


def _write_back_positions(atoms: Atoms, batch: Batch) -> None:
    """Copy positions from batch back to atoms, restoring FixAtoms rows."""
    # float64: the batch is float32, and assigning the float64 frozen rows
    # into a float32 array would truncate them, leaving the "restored" atoms
    # about 5e-7 A off their pinned positions at slab-sized coordinates.
    relaxed = batch.positions.detach().cpu().numpy().astype(np.float64)
    fixed = _fixed_indices(atoms)
    if fixed:
        relaxed[fixed] = atoms.positions[fixed]
    atoms.positions = relaxed


# ------------------------------------------------------------------- base class

class _AlchemiBase:
    """Shared machinery for the Alchemi calculators.

    Holds the model and the device/dtype/chunking state, the batching and
    neighbour-list helpers, the chunked ``get_potential_energies`` template, and
    Langevin MD. A subclass implements :meth:`_evaluate_chunk`, which is the
    only thing that differs between evaluating and relaxing.

    Parameters
    ----------
    checkpoint : str | MACEWrapper
        Named checkpoint (e.g. 'medium-mpa-0'), local .pt path, or a pre-loaded
        MACEWrapper to share across calculators.
    device : str
        'cuda' or 'cpu'.
    dtype : torch.dtype
        Model and data dtype. float32 recommended for speed.
    enable_cueq : bool
        cuEquivariance kernel fusion — significant speedup on GPU.
    compile_model : bool
        torch.compile the model. Adds ~30s warmup, then faster.
    max_neighbors : int | None
        Neighbour-list cap handed to ``NeighborListHook``.
    chunk_size : int | None
        Default sub-batch size for :meth:`get_potential_energies`. ``None``
        evaluates the whole list at once.
    head : str | int | None
        Head of a multihead model, by name or index. Local paths only.
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
        head: Union[str, int, None] = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size
        self.model = _load_model(checkpoint, device, dtype, enable_cueq,
                                 compile_model, head)
        self._nl_config = self.model.model_config.neighbor_config

    # -- helpers a subclass builds its evaluation from

    def _batch(self, atoms_or_list) -> Batch:
        return _make_batch(atoms_or_list, self.device, self.dtype)

    def _nl_hook(self) -> NeighborListHook:
        return NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)

    def _prepare(self, batch: Batch, n_graphs: int) -> None:
        """Pre-allocate the tensors ``compute()`` writes into via ``copy_()``."""
        batch.forces = torch.zeros_like(batch.positions)
        batch.energy = torch.zeros(n_graphs, 1, device=self.device, dtype=self.dtype)

    @staticmethod
    def _bootstrap(opt, batch: Batch, nl_hook: NeighborListHook,
                   fixed: List[int]) -> None:
        """Build the neighbour list and initial forces before the loop starts.

        ``FreezeAtomsHook`` zeros frozen forces only at ``AFTER_POST_UPDATE``,
        and this ``compute()`` runs outside the hook loop, so step 0 would
        otherwise displace the frozen rows once.
        """
        _build_nl(batch, nl_hook)
        opt.compute(batch)
        if fixed:
            batch.forces[fixed] = 0.0

    # -- the public evaluation surface

    def _evaluate_chunk(self, chunk: List[Atoms]) -> np.ndarray:
        """Per-graph energies for one sub-batch. Implemented by subclasses."""
        raise NotImplementedError

    def get_potential_energy(self, atoms: Atoms) -> float:
        """Energy of one structure, as a batch of one.

        Delegating keeps the single-structure and batched paths from drifting
        into different strategies, which is what happened before: the relaxing
        calculator used to step the whole batch to full convergence here and
        retire graphs individually in the batched path.
        """
        return float(self.get_potential_energies([atoms])[0])

    def get_potential_energies(
        self, atoms_list: List[Atoms], chunk_size: int | None = None
    ) -> np.ndarray:
        """
        Per-graph energies over multiple (possibly differently sized) structures.

        One CUDA kernel launch per layer instead of N — the win behind batched
        replica exchange on a single GPU.

        ``chunk_size`` splits ``atoms_list`` into consecutive sub-batches of at
        most that many structures, evaluated one at a time, capping peak GPU
        memory at the largest chunk instead of the whole batch. ``None`` (the
        default) falls back to the instance ``chunk_size``; if that is also
        ``None`` the whole list goes in one pass. Results are unaffected by
        chunking (MACE message passing and FIRE convergence are both per-graph),
        up to GPU run-to-run noise.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
            Structures to evaluate. Lengths may differ. Whether they are
            mutated depends on the subclass: relaxation writes positions back.
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
        out = [self._evaluate_chunk(atoms_list[start:stop])
               for start, stop in chunk_ranges(len(atoms_list), cs)]
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
        """Run NVT Langevin (BAOAB) MD in place on ``atoms``.

        Velocities start from a Maxwell-Boltzmann draw at ``temperature``.
        Reuses this calculator's model and neighbour-list config, so no second
        model is loaded. ``friction`` is in 1/fs, ``dt`` in fs, ``temperature``
        in K (the integrator converts to internal units). ``FixAtoms``
        constraints are held every step and restored on write-back.
        """
        batch = self._batch(atoms)
        self._prepare(batch, 1)
        batch.velocities = torch.zeros_like(batch.positions)
        temp = torch.full((batch.num_graphs,), float(temperature),
                          device=self.device, dtype=self.dtype)
        initialize_velocities(
            batch.velocities, batch.atomic_masses, temp, batch.batch_idx.int(),
            random_seed=seed,
        )

        fixed = _fixed_indices(atoms)
        freeze_hooks = _freeze_hook_for(batch, fixed)
        nl_hook = self._nl_hook()
        opt = NVTLangevin(
            model=self.model,
            dt=dt,
            temperature=temperature,
            friction=friction,
            random_seed=seed,
            n_steps=steps,
            # NaN forces/energy would otherwise propagate silently into the
            # proposal the ensemble scores; the hook raises at the offending
            # step.
            hooks=(freeze_hooks or []) + [NaNDetectorHook()],
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)
        self._bootstrap(opt, batch, nl_hook, fixed)
        opt.run(batch)
        _write_back_positions(atoms, batch)


# ------------------------------------------------------------------ energy only

class AlchemiCalculator(_AlchemiBase):
    """
    Energy-only Alchemi calculator (no geometry relaxation).

    Single-point MACE evaluation on the nvalchemi GPU-native stack. Use this
    when mcpy handles its own MC moves and only needs E(atoms).

    Takes the shared parameters of :class:`_AlchemiBase`, plus:

    Parameters
    ----------
    energy_only : bool
        Drop ``forces`` from the model's active outputs. MC energy evaluation
        never uses them, and without them the MACE forward builds no autograd
        graph: lower peak memory, energy unchanged up to fp32 rounding.
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
        super().__init__(checkpoint, device, dtype, enable_cueq, compile_model,
                         max_neighbors, chunk_size, head)
        self.energy_only = energy_only
        if energy_only:
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

    def _evaluate_chunk(self, chunk: List[Atoms]) -> np.ndarray:
        """One forward pass; ``chunk`` is not mutated."""
        batch = self._batch(chunk)
        nl_hook = self._nl_hook()
        _build_nl(batch, nl_hook)
        out = self._forward(batch)
        return _per_graph_energies(out['energy'], len(chunk))

    def run_md(self, atoms: Atoms, **kwargs) -> None:
        if self.energy_only:
            # The integrator would otherwise die steps deep inside nvalchemi
            # ("NVTLangevin requires forces...") without naming the cause.
            raise ValueError(
                'run_md needs forces, but this calculator was built with '
                'energy_only=True (forces are stripped from the model '
                'outputs). Build a separate calculator without energy_only '
                'for MD.'
            )
        return super().run_md(atoms, **kwargs)


# -------------------------------------------------------------- FIRE relaxation

class AlchemiFCalculator(_AlchemiBase):
    """
    Alchemi calculator with FIRE geometry relaxation.

    Mirrors MACE_F_Calculator: relax atoms with FIRE then return energy. Uses
    the fully GPU-resident nvalchemi dynamics stack — significantly faster than
    ASE FIRE above ~100 atoms.

    Takes the shared parameters of :class:`_AlchemiBase`, plus:

    Parameters
    ----------
    steps : int
        Maximum FIRE steps before returning.
    fmax : float
        Force convergence threshold in eV/Å.
    dt : float
        FIRE initial timestep in fs (default 1.0, the value of ASE FIRE's
        dtmax; benchmarks show dt=1.0 converges in ~half the steps vs 0.1).
        Note nvalchemi grows the adaptive timestep up to dt_max = 10*dt,
        unlike ASE's 1 fs cap; per-step displacement stays clamped by
        maxstep=0.2 A either way.
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
        chunk_size: int | None = None,
        head: Union[str, int, None] = None,
    ) -> None:
        self.steps = steps
        self.fmax = fmax
        self.dt = dt
        self.last_relax_steps = 0
        self.total_relax_steps = 0
        if optimizer not in _ALCHEMI_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {list(_ALCHEMI_OPTIMIZERS)}, got {optimizer!r}"
            )
        self._optimizer_cls = _ALCHEMI_OPTIMIZERS[optimizer]
        self.optimizer_name = optimizer
        super().__init__(checkpoint, device, dtype, enable_cueq, compile_model,
                         max_neighbors, chunk_size, head)
        if 'forces' not in self.model.model_config.active_outputs:
            # A model reused from an energy_only AlchemiCalculator (its
            # ``.model`` attribute) has forces stripped; FIRE would only fail
            # steps deep inside nvalchemi without naming the cause.
            raise ValueError(
                "the provided model has 'forces' disabled in its active "
                'outputs (an energy_only AlchemiCalculator strips them), but '
                'FIRE relaxation needs forces. Load a fresh wrapper for this '
                'calculator instead of sharing an energy-only one.'
            )

    def get_potential_energies(
        self, atoms_list: List[Atoms], chunk_size: int | None = None
    ) -> np.ndarray:
        """Relax every structure with batched FIRE, then return the energies.

        Each structure in ``atoms_list`` is mutated in place: positions are
        updated to the relaxed configuration, respecting FixAtoms. Adds the
        relax-step accounting to the inherited chunking template.
        """
        self._chunk_steps: List[int] = []
        energies = super().get_potential_energies(atoms_list, chunk_size)
        if self._chunk_steps:
            # last = deepest chunk relaxation; total accumulates across chunks.
            self.last_relax_steps = max(self._chunk_steps)
            self.total_relax_steps += sum(self._chunk_steps)
            logger.debug("FIRE relaxation (batched, %d graphs, %d chunk(s)): "
                         "max %d/%d steps (fmax=%.3g eV/A)",
                         len(atoms_list), len(self._chunk_steps),
                         self.last_relax_steps, self.steps, self.fmax)
        return energies

    def _evaluate_chunk(self, chunk: List[Atoms]) -> np.ndarray:
        """Batched FIRE relaxation of one sub-batch, mutating positions."""
        batch = self._batch(chunk)
        self._prepare(batch, len(chunk))

        # Map each graph's FixAtoms indices into the concatenated sub-batch.
        fixed: List[int] = []
        offset = 0
        for a in chunk:
            fixed.extend(offset + i for i in _fixed_indices(a))
            offset += len(a)
        freeze_hooks = _freeze_hook_for(batch, fixed)

        nl_hook = self._nl_hook()
        opt = self._optimizer_cls(
            model=self.model,
            dt=self.dt,
            convergence_hook=ConvergenceHook.from_fmax(self.fmax),
            n_steps=self.steps,
            # NaN forces/energy would otherwise flow silently into the GCMC
            # acceptance decision; the hook raises at the offending step.
            hooks=(freeze_hooks or []) + [NaNDetectorHook()],
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)
        self._bootstrap(opt, batch, nl_hook, fixed)
        energies, n_steps = self._run_compacted(opt, batch, chunk)
        self._chunk_steps.append(n_steps)
        return energies

    def _run_compacted(self, opt, batch, atoms_list: List[Atoms]) -> tuple:
        """Step FIRE manually, retiring each graph from the batch at its first
        convergence so subsequent steps only compute the still-active graphs.

        ``opt.run`` computes the FULL batch every step and stops only when all
        graphs are converged at the same step; with mixed trial moves the fast
        graphs pay for the slowest one.

        Batch rows are removed with ``Batch.index_select`` and the optimizer's
        per-graph FIRE state is shrunk in lockstep with
        ``_sync_state_to_batch`` — the same primitives nvalchemi's inflight
        refill machinery uses. Positions and energy are harvested per graph at
        retirement; returns ``(per_graph_energies, step_count)``.
        """
        alive = np.arange(len(atoms_list))  # batch row -> atoms_list index
        energies = np.zeros(len(atoms_list))
        n_steps = 0
        opt._open_hooks()
        try:
            while alive.size and n_steps < self.steps:
                batch, conv = opt.step(batch)
                n_steps += 1
                if conv is None or conv.numel() == 0:
                    continue
                rows = np.unique(conv.detach().cpu().numpy())
                self._harvest(batch, rows, alive, energies, atoms_list)
                keep = np.setdiff1d(np.arange(alive.size), rows)
                alive = alive[keep]
                if alive.size:
                    keep_t = torch.as_tensor(keep, device=batch.device)
                    opt._sync_state_to_batch(keep_t, 0, batch)
                    batch = batch.index_select(keep_t)
                    # _last_converged indexes the pre-shrink batch; clear it
                    # or the next hook context builds an out-of-bounds mask.
                    opt._last_converged = None
            if alive.size:  # step cap hit: harvest the stragglers as-is
                logger.warning('%d/%d graphs reached the %d-step FIRE cap '
                               '(fmax=%.3g eV/A); their energies may be '
                               'unconverged', alive.size, len(atoms_list),
                               self.steps, self.fmax)
                self._harvest(batch, np.arange(alive.size), alive, energies,
                              atoms_list)
        finally:
            opt._close_hooks()
        return energies, n_steps

    @staticmethod
    def _harvest(batch, rows, alive, energies, atoms_list) -> None:
        """Write energy + relaxed positions of ``rows`` (batch row indices)
        back to their originating Atoms objects, restoring FixAtoms rows."""
        e = _per_graph_energies(batch.energy, int(batch.num_graphs))
        # float64 before restoring frozen rows: see _write_back_positions.
        pos = batch.positions.detach().cpu().numpy().astype(np.float64)
        batch_idx = batch.batch_idx.detach().cpu().numpy()
        for row in rows:
            g = int(alive[row])
            atoms = atoms_list[g]
            energies[g] = e[row]
            new_pos = pos[batch_idx == row].copy()
            for j in _fixed_indices(atoms):
                new_pos[j] = atoms.positions[j]
            atoms.positions = new_pos
