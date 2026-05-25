import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from ase.units import fs
from .base_move import BaseMove
from ..cell import NullCell


def get_surface_atoms(atoms, cutoff=3.2, threshold=8):
    indices = neighbor_list('i', atoms, cutoff)
    counts = np.bincount(indices, minlength=len(atoms))
    return np.where(counts < threshold)[0]


class PermutationMove(BaseMove):
    """
    Swap the chemical symbols of two atoms of different species.
    Useful for optimizing chemical order in nanoalloys.
    """
    def __init__(self, species: list[str], seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species, seed)

    def do_trial_move(self, atoms) -> Atoms:
        atoms_new = atoms.copy()
        species_pair = self.rng.random.sample(self.species, 2)
        indices_symbol_a = [atom.index for atom in atoms_new if atom.symbol == species_pair[0]]
        indices_symbol_b = [atom.index for atom in atoms_new if atom.symbol == species_pair[1]]
        if len(indices_symbol_a) == 0 or len(indices_symbol_b) == 0:
            return False, 0, 'X'
        i = self.rng.random.choice(indices_symbol_a)
        j = self.rng.random.choice(indices_symbol_b)
        atoms_new[i].symbol, atoms_new[j].symbol = atoms_new[j].symbol, atoms_new[i].symbol
        return atoms_new, 0, 'X'


class ShakeMove(BaseMove):
    """
    Randomly displace all atoms within a sphere of radius r_max.
    Very efficient for shape optimization.
    """
    def __init__(self, r_max: float, seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.r_max = r_max
        self.name = 'Shake'
        self.rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        atoms_new = atoms.copy()
        n = len(atoms)
        # Uniform directions on S^2 (Gaussian normalized; a cube-uniform
        # vector would bias toward diagonals).
        displacements = self.rng.standard_normal(size=(n, 3))
        directions = displacements / np.linalg.norm(displacements, axis=1, keepdims=True)
        # Uniform-in-ball radius: r_max * u^(1/3), u ~ U(0,1).
        radii = self.r_max * np.cbrt(self.rng.random(size=(n, 1)))
        new_positions = atoms_new.get_positions() + directions * radii
        atoms_new.set_positions(new_positions)
        return atoms_new, 0, 'X'


class BrownianMove(BaseMove):
    """
    Apply short high-temperature molecular dynamics to simulate Brownian motion.
    Useful for large cluster relaxation.
    """
    def __init__(self, temperature: float, calculator, steps: int, d_t: float, seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.temperature = temperature
        self.calculator = calculator
        self.steps = steps
        self.d_t = d_t
        self.name = 'Brownian'

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        atoms_new = atoms.copy()
        MaxwellBoltzmannDistribution(atoms_new, temperature_K=self.temperature)
        atoms_new.calc = self.calculator
        dyn = VelocityVerlet(atoms_new, self.d_t * fs, logfile=None)
        dyn.run(steps=self.steps)
        return atoms_new, 0, 'X'


class BallMove(BaseMove):
    """
    Randomly displace a single atom within the cluster volume.
    """
    def __init__(self, radius: float, seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.radius = radius
        self.name = 'Ball'
        self.rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        atoms_new = atoms.copy()
        i = self.rng.integers(0, len(atoms))
        # Uniform inside a ball of radius `self.radius`: uniform direction
        # times r = radius * u^(1/3).
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        r = self.radius * np.cbrt(self.rng.random())
        atoms_new[i].position += direction * r
        return atoms_new, 0, 'X'


class ShellMove(BaseMove):
    """
    Move a surface atom within a spherical shell, refining surface arrangement.
    """
    def __init__(self, r_shell: float, seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.r_shell = r_shell
        self.name = 'Shell'
        self.rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        atoms_new = atoms.copy()
        surface_indices = get_surface_atoms(atoms_new)
        if len(surface_indices) == 0:
            return False, 0, 'X'
        i = self.rng.choice(surface_indices)
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        atoms_new[i].position += direction * self.r_shell
        return atoms_new, 0, 'X'


class BondMove(BaseMove):
    """
    Move a low-coordination (weakly bonded) surface atom.
    """
    def __init__(self, r_max: float, seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.r_max = r_max
        self.name = 'Bond'
        self.rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        atoms_new = atoms.copy()
        surface_indices = get_surface_atoms(atoms_new)
        if len(surface_indices) == 0:
            return False, 0, 'X'
        i = self.rng.choice(surface_indices)
        displacement = self.rng.normal(size=3)
        displacement *= self.r_max / np.linalg.norm(displacement)
        atoms_new[i].position += displacement
        return atoms_new, 0, 'X'


class HighEnergyAtomsMove(BaseMove):
    """
    Displace atoms depending on local energy:
    - atoms below threshold move slightly (shake)
    - atoms above threshold move more (bond-like)
    """
    def __init__(self, calculator, energy_threshold: float, r_max: float, seed: int) -> None:
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.energy_threshold = energy_threshold
        self.calculator = calculator
        self.r_max = r_max
        self.rng = np.random.default_rng(seed)
        self.name = 'HighEnergy'

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        atoms_new = atoms.copy()
        try:
            local_energies = self.calculator.calculator.results['node_energy']
        except Exception:
            return False, 0, 'X'

        if local_energies is None:
            return False, 0, 'X'

        for i, e in enumerate(local_energies):
            if e < self.energy_threshold:
                atoms_new[i].position += self.rng.normal(size=3) * self.r_max * 0.2
            else:
                atoms_new[i].position += self.rng.normal(size=3) * self.r_max
        return atoms_new, 0, 'X'
