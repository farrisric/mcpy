from ase import Atoms
from .base_move import BaseMove
from ..cell import NullCell


class AlchemiBrownianMove(BaseMove):
    """Brownian (NVT Langevin) move driven by the nvalchemi GPU-native MD stack.

    Proposes a trial configuration by running a short Langevin MD trajectory on
    the GPU, reusing the model held by the supplied Alchemi calculator (no
    second model is loaded). The ensemble then scores the proposal with its own
    calculator, exactly like any other move. ``FixAtoms`` constraints are
    honored.

    .. warning::
       Not exact detailed balance. The ensemble accepts on the potential
       energy only: there is no kinetic-energy term, and a Langevin
       trajectory is not a symmetric proposal. Use it as basin-hopping-style
       exploration with a relaxing calculator (``AlchemiFCalculator``). Do
       not use it with an energy-only calculator when you need equilibrium
       averages.

    This move is pure Python; the nvalchemi dependency enters only through
    ``calculator.run_md`` at call time.

    Positions are not wrapped back into the box after the MD trajectory.
    This is safe: the periodic ``Cell`` selects deletion candidates by
    species only (no position test), MACE handles PBC through neighbor-list
    shifts, and region cells (DomeCell, CustomCell) treat out-of-region
    drift exactly as they do for any displacement move.

    Parameters
    ----------
    calculator : AlchemiFCalculator | AlchemiCalculator
        Calculator exposing ``run_md``; its model is reused for the MD.
        The MD needs forces: an ``AlchemiCalculator`` that loads its own
        model is energy-only by default, so build it with
        ``energy_only=False`` (``run_md`` raises ``ValueError`` otherwise).
    temperature : float
        Langevin target temperature in Kelvin.
    friction : float
        Langevin friction coefficient in 1/fs (default 0.01).
    steps : int
        Number of MD steps per move (default 100).
    dt : float
        MD timestep in fs (default 2.0).
    seed : int
        Base RNG seed; a fresh per-call seed is derived for the stochastic
        Langevin step so repeated calls explore different trajectories.
    trajectory_file : str | None
        Debug option. When set, every MD step of every call (accepted or
        rejected later) is appended to it as an extended-XYZ frame with
        velocities, forces, temperature and energies. ``md_run`` in the info
        line counts the calls of this move. Slows the MD down.
    """

    def __init__(self, calculator, temperature: float, friction: float = 0.01,
                 steps: int = 100, dt: float = 2.0, seed: int = 0,
                 trajectory_file: str | None = None) -> None:
        super().__init__(NullCell(), species=['X'], seed=seed)
        self.calculator = calculator
        self.temperature = temperature
        self.friction = friction
        self.steps = steps
        self.dt = dt
        self.name = 'AlchemiBrownian'
        self.trajectory_file = trajectory_file
        self._md_run = 0

    def do_trial_move(self, atoms: Atoms):
        # Mutates ``atoms`` in place, like every GCMC move: the ensemble
        # scores the object it passed in and rolls back from its arrays
        # snapshot on rejection. Running the MD on a copy would leave the
        # ensemble's atoms untouched and turn the move into a silent no-op.
        md_seed = self.rng.random.randrange(2 ** 31)
        traj_kw = {}
        if self.trajectory_file is not None:
            self._md_run += 1
            traj_kw = dict(trajectory_file=self.trajectory_file, md_run=self._md_run)
        self.calculator.run_md(
            atoms,
            temperature=self.temperature,
            friction=self.friction,
            dt=self.dt,
            steps=self.steps,
            seed=md_seed,
            **traj_kw,
        )
        return atoms, 0, 'X'
