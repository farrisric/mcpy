import numpy as np

from .base_move import BaseMove
from ..cell import NullCell


class VolumeMove(BaseMove):
    """Isotropic volume change: a random walk in ln V.

    Scales the lattice and every atom position by the same factor, in place.
    ``IsothermalIsobaricEnsemble`` rolls back the lattice and the arrays if
    the move is rejected; no other ensemble scores it correctly.

    The new volume is ``V exp(u * max_delta_ln_volume)`` with ``u`` uniform in
    [-1, 1]. The proposal is symmetric in ln V, so the acceptance carries the
    Jacobian ``(N + 1) ln(V'/V)`` (Frenkel & Smit, Sec. 5.4). The move keeps
    it in ``last_log_jacobian`` and the volumes in ``last_volumes``, valid for
    the last proposal, for the ensemble to read.
    """

    def __init__(self, seed: int, max_delta_ln_volume: float = 0.01) -> None:
        if not (np.isfinite(max_delta_ln_volume) and max_delta_ln_volume > 0):
            raise ValueError(
                f'max_delta_ln_volume must be finite and > 0, got {max_delta_ln_volume}')
        super().__init__(NullCell(), [], seed)
        self.max_delta_ln_volume = float(max_delta_ln_volume)
        self.last_volumes = None
        self.last_log_jacobian = None

    def do_trial_move(self, atoms):
        v_old = atoms.get_volume()
        ln_ratio = (2.0 * self.rng.get_uniform() - 1.0) * self.max_delta_ln_volume
        atoms.set_cell(atoms.cell.array * np.exp(ln_ratio / 3.0), scale_atoms=True)
        self.last_volumes = (v_old, atoms.get_volume())
        self.last_log_jacobian = (len(atoms) + 1) * ln_ratio
        return atoms, 0, 'X'
