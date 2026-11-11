"""
Tests for the SphericalCell free-volume sampler and get_random_point.
Run with: python -m pytest tests/test_optimizations.py -v
"""
import numpy as np
from ase import Atoms

from mcpy.cell.spherical_cell import SphericalCell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_nanoparticle(n=13):
    """Small FCC Cu nanoparticle-like cluster at origin."""
    rng = np.random.default_rng(42)
    positions = rng.uniform(-3, 3, size=(n, 3))
    return Atoms('Cu' * n, positions=positions)


SPECIES_RADII = {'Cu': 1.28, 'Au': 1.44, 'H': 0.53}


# ---------------------------------------------------------------------------
# SphericalCell free-volume sampler (kdtree path)
# ---------------------------------------------------------------------------

class TestSphericalCellVolume:
    def setup_method(self):
        self.atoms = make_nanoparticle(13)
        self.cell = SphericalCell(self.atoms, vacuum=3.0,
                                  species_radii=SPECIES_RADII,
                                  mc_sample_points=5_000, seed=0)

    def test_volume_drops_when_adding_atoms(self):
        self.cell.calculate_volume(self.atoms)
        v_before = self.cell.volume
        atoms_plus = self.atoms.copy()
        # Add many large atoms — free volume must drop meaningfully.
        for k in range(8):
            atoms_plus += Atoms('Au', positions=[[k * 0.4, 0.0, 0.0]])
        self.cell.calculate_volume(atoms_plus)
        assert self.cell.volume < v_before


# ---------------------------------------------------------------------------
# SphericalCell.get_random_point — Marsaglia method
# ---------------------------------------------------------------------------

class TestSphericalCellGetRandomPoint:
    def setup_method(self):
        self.atoms = make_nanoparticle(13)
        self.cell = SphericalCell(self.atoms, vacuum=3.0,
                                  species_radii=SPECIES_RADII,
                                  mc_sample_points=1_000, seed=0)

    def test_distribution_is_uniform(self):
        n = 10_000
        points = np.array([self.cell.get_random_point() for _ in range(n)])
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    mask = (
                        (np.sign(points[:, 0] - self.cell.center[0]) == sx) &
                        (np.sign(points[:, 1] - self.cell.center[1]) == sy) &
                        (np.sign(points[:, 2] - self.cell.center[2]) == sz)
                    )
                    fraction = mask.sum() / n
                    assert abs(fraction - 0.125) < 0.03, \
                        f'Octant ({sx},{sy},{sz}) fraction={fraction:.3f}, expected ~0.125'
