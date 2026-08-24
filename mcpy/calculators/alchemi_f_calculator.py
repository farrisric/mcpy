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
from nvalchemi.dynamics.hooks import NaNDetectorHook

from ..utils.chunking import chunk_ranges
from ._alchemi_common import (
    _bootstrap,
    _fixed_indices,
    _freeze_hook_for,
    _load_model,
    _make_batch,
    _per_graph_energies,
    _prepare_batch,
    _MDMixin,
)


logger = logging.getLogger(__name__)

_ALCHEMI_OPTIMIZERS = {'fire': AlchemiFIRE, 'fire2': AlchemiFIRE2}


class AlchemiFCalculator(_MDMixin):
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
        FIRE initial timestep in fs (default 1.0, the value of ASE FIRE's
        dtmax; benchmarks show dt=1.0 converges in ~half the steps vs 0.1).
        Note nvalchemi grows the adaptive timestep up to dt_max = 10*dt,
        unlike ASE's 1 fs cap; per-step displacement stays clamped by
        maxstep=0.2 A either way.
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
        self._nl_config = self.model.model_config.neighbor_config

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Relax with Alchemi FIRE, then return the relaxed potential energy.

        One structure is a batch of one: this delegates to the same batched
        relaxation the multi-structure path uses, so the two cannot use
        different FIRE strategies. (They did: this path called ``opt.run``,
        which steps the whole batch until every graph converges, while the
        batched path retires each graph at its own first convergence.)

        Parameters
        ----------
        atoms : ase.Atoms
            Starting geometry. Mutated in place: positions are updated to
            the relaxed configuration (FixAtoms rows restored). The GCMC
            ensembles rely on this write-back.

        Returns
        -------
        float
            Relaxed potential energy in eV.
        """
        return float(self.get_potential_energies([atoms])[0])

    def _relax_batch(self, atoms_list: List[Atoms]) -> tuple:
        """Batched FIRE relaxation of one (sub-)batch. Mutates positions in
        place; returns ``(per_graph_energies, step_count)``."""
        batch = _make_batch(atoms_list, self.device, self.dtype)
        _prepare_batch(batch, len(atoms_list), self.device, self.dtype)

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
            hooks=(freeze_hooks or []) + [NaNDetectorHook()],
        )
        opt.register_hook(nl_hook, stage=DynamicsStage.BEFORE_COMPUTE)
        _bootstrap(opt, batch, nl_hook, fixed)
        return self._run_compacted(opt, batch, atoms_list)

    def _run_compacted(self, opt, batch, atoms_list: List[Atoms]) -> tuple:
        """Step FIRE manually, retiring each graph from the batch at its first
        convergence so subsequent steps only compute the still-active graphs.

        ``opt.run`` computes the FULL batch every step and stops only when all
        graphs are converged at the same step; with mixed trial moves the fast
        graphs pay for the slowest one. Retiring at first convergence also
        matches the single-graph ``get_potential_energy`` semantics (a lone
        graph stops at its own first convergence).

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

    def get_potential_energies(
        self, atoms_list: List[Atoms], chunk_size: int | None = None
    ) -> np.ndarray:
        """
        Batched FIRE relaxation over multiple structures, then per-graph energies.

        All graphs in a (sub-)batch share one optimizer / one model forward pass
        per FIRE step. Each graph is removed from the batch at its first
        convergence, so already-converged graphs stop paying for the forward
        pass. Returns when every graph has converged or ``steps`` is reached.

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
