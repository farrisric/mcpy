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

import logging
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
from nvalchemi.dynamics import FIRE as AlchemiFIRE, FIRE2 as AlchemiFIRE2, ConvergenceHook
from nvalchemi.dynamics import NVTLangevin, initialize_velocities
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook

from ..utils.chunking import chunk_ranges


logger = logging.getLogger(__name__)

_ALCHEMI_OPTIMIZERS = {'fire': AlchemiFIRE, 'fire2': AlchemiFIRE2}


def _is_local_model_file(checkpoint: Union[str, Path]) -> bool:
    """True if checkpoint points to an existing local .model/.pt/.pth file."""
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        return False
    return path.suffix.lower() in {".model", ".pt", ".pth"}
def _load_local_mace_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype | None,
    enable_cueq: bool,
    compile_model: bool,
    **compile_kwargs,
) -> MACEWrapper:
    """Load a local MACE checkpoint without triggering foundation-model download."""
    model = torch.load(model_path, map_location=device, weights_only=False)
    if dtype is not None:
        model = model.to(dtype=dtype)
    if enable_cueq:
        try:
            import cuequivariance # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "cuequivariance is required for enable_cueq=True. "
                "Install with: pip install 'nvalchemi-toolkit[mace]'"
            ) from exc
        from mace.cli.convert_e3nn_cueq import run as convert_mace_weights
        model = convert_mace_weights(model, return_model=True, device=device)
    model = model.to(device)
    if compile_model:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model = torch.compile(model, **compile_kwargs)
    return MACEWrapper(model)
def _load_model(
    checkpoint: Union[str, Path, MACEWrapper],
    device: str,
    dtype: torch.dtype,
    enable_cueq: bool,
    compile_model: bool,
    **compile_kwargs,
) -> MACEWrapper:
    if isinstance(checkpoint, MACEWrapper):
        return checkpoint
    dev = torch.device(device)
    # Local file -> torch.load + MACEWrapper
    if _is_local_model_file(checkpoint):
        return _load_local_mace_model(
            Path(checkpoint).expanduser().resolve(),
            device=dev,
            dtype=dtype,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
            **compile_kwargs,
        )
    # Foundation name ("medium-mpa-0") or URL -> from_checkpoint
    return MACEWrapper.from_checkpoint(
        str(checkpoint),
        device=dev,
        dtype=dtype,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
        **compile_kwargs,
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
        chunk_size: int | None = None,
        energy_only: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.max_neighbors = max_neighbors
        self.chunk_size = chunk_size
        self.energy_only = energy_only
        self.model = _load_model(checkpoint, device, dtype, enable_cueq, compile_model)
        self._nl_config = self.model.model_config.neighbor_config
        if energy_only:
            # MC energy evaluation never uses forces. Dropping 'forces' from
            # active_outputs sets compute_force=False in the MACE forward, so no
            # autograd graph is built — lower peak memory, energy unchanged up to
            # fp32 rounding. See docs/superpowers/specs/2026-06-09-...-design.md.
            self.model.model_config.active_outputs.discard('forces')

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
            batch = _make_multi_batch(chunk, self.device, self.dtype)
            batch.positions.requires_grad_(True)
            nl_hook = NeighborListHook(self._nl_config, max_neighbors=self.max_neighbors)
            _build_nl(batch, nl_hook)
            model_out = self.model(batch)
            out.append(_per_graph_energies(model_out['energy'], len(chunk)))
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
