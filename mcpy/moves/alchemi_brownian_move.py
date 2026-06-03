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

    This move is pure Python; the nvalchemi dependency enters only through
    ``calculator.run_md`` at call time.

    Parameters
    ----------
    calculator : AlchemiCalculator | AlchemiFCalculator
        Calculator exposing ``run_md``; its model is reused for the MD.
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
    """

    def __init__(self, calculator, temperature: float, friction: float = 0.01,
                 steps: int = 100, dt: float = 2.0, seed: int = 0) -> None:
        super().__init__(NullCell(), species=['X'], seed=seed)
        self.calculator = calculator
        self.temperature = temperature
        self.friction = friction
        self.steps = steps
        self.dt = dt
        self.name = 'AlchemiBrownian'

    def do_trial_move(self, atoms: Atoms):
        atoms_new = atoms.copy()
        md_seed = self.rng.random.randrange(2 ** 31)
        self.calculator.run_md(
            atoms_new,
            temperature=self.temperature,
            friction=self.friction,
            dt=self.dt,
            steps=self.steps,
            seed=md_seed,
        )
        return atoms_new, 0, 'X'
