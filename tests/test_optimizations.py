"""
Tests for SphericalCell free-volume sampler, in-place InsertionMove,
SphericalCell.get_random_point, and write_xyz.
Run with: python -m pytest tests/test_optimizations.py -v
"""
import numpy as np
import pytest  # noqa: F401
from ase import Atoms

from mcpy.cell.spherical_cell import SphericalCell
from mcpy.moves.insertion_move import InsertionMove


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

    def test_volume_in_bounds(self):
        self.cell.calculate_volume(self.atoms)
        assert 0.0 < self.cell.volume <= self.cell.sphere_volume

    def test_volume_drops_when_adding_atoms(self):
        self.cell.calculate_volume(self.atoms)
        v_before = self.cell.volume
        atoms_plus = self.atoms.copy()
        # Add many large atoms — free volume must drop meaningfully.
        for k in range(8):
            atoms_plus += Atoms('Au', positions=[[k * 0.4, 0.0, 0.0]])
        self.cell.calculate_volume(atoms_plus)
        assert self.cell.volume < v_before

    def test_geometry_change_reflected(self):
        """A pure displacement (same N) must still be reflected — no n_atoms
        cache that masks geometry changes."""
        self.cell.calculate_volume(self.atoms)
        v1 = self.cell.volume

        atoms2 = self.atoms.copy()
        # Push everything far out of the sphere → free volume should rise to ~sphere
        atoms2.positions += 1000.0
        self.cell.calculate_volume(atoms2)
        v2 = self.cell.volume

        assert v2 > v1, 'Moving atoms outside the sphere should free volume'

    def test_empty_atoms_full_sphere(self):
        empty = Atoms()
        self.cell.calculate_volume(empty)
        assert self.cell.volume == pytest.approx(self.cell.sphere_volume)


# ---------------------------------------------------------------------------
# InsertionMove (in-place semantics)
# ---------------------------------------------------------------------------

class TestInsertionMove:
    def setup_method(self):
        from mcpy.cell.cell import Cell
        self.atoms = Atoms(
            'Cu8',
            positions=np.array([
                [0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [0, 0, 1.8],
                [1.8, 1.8, 0], [1.8, 0, 1.8], [0, 1.8, 1.8], [1.8, 1.8, 1.8],
            ]),
            cell=[5, 5, 5],
            pbc=True,
        )
        self.cell = Cell(self.atoms, species_radii={'Cu': 1.28, 'H': 0.53})
        self.cell.calculate_volume(self.atoms)

    def test_basic_insertion_inplace(self):
        n0 = len(self.atoms)
        move = InsertionMove(self.cell, species=['H'], seed=0, min_insert=None)
        atoms_new, delta, species = move.do_trial_move(self.atoms)
        assert delta == 1
        assert species == 'H'
        # In-place: the move returns the same object, now N+1.
        assert atoms_new is self.atoms
        assert len(self.atoms) == n0 + 1

    def test_species_radii_exclusion_respected(self):
        """Inserted atom stays outside every prior atom's exclusion sphere."""
        radii = self.cell.species_radii
        move = InsertionMove(self.cell, species=['H'], seed=1)
        for _ in range(20):
            prior = self.atoms.copy()
            move.do_trial_move(self.atoms)
            dists = self.atoms.get_distances(len(prior), range(len(prior)), mic=True)
            limits = np.array([radii[s] for s in prior.get_chemical_symbols()])
            assert np.all(dists >= limits)

    def test_insertion_without_min_insert(self):
        n0 = len(self.atoms)
        move = InsertionMove(self.cell, species=['H'], seed=2, min_insert=None)
        move.do_trial_move(self.atoms)
        assert len(self.atoms) == n0 + 1

    def test_full_cell_insertion_fails_cleanly(self):
        n0 = len(self.atoms)
        self.cell.species_radii = {'Cu': 1e9, 'H': 1e9}
        move = InsertionMove(self.cell, species=['H'], seed=3)
        result, delta_n, _ = move.do_trial_move(self.atoms)
        # No free point: the move reports failure and leaves atoms untouched
        # rather than inserting a guaranteed overlap.
        assert result is False
        assert len(self.atoms) == n0


# ---------------------------------------------------------------------------
# SphericalCell.get_random_point — Marsaglia method
# ---------------------------------------------------------------------------

class TestSphericalCellGetRandomPoint:
    def setup_method(self):
        self.atoms = make_nanoparticle(13)
        self.cell = SphericalCell(self.atoms, vacuum=3.0,
                                  species_radii=SPECIES_RADII,
                                  mc_sample_points=1_000, seed=0)

    def test_point_inside_sphere(self):
        for _ in range(200):
            pt = self.cell.get_random_point()
            assert np.linalg.norm(pt - self.cell.center) <= self.cell.radius + 1e-12

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


# ---------------------------------------------------------------------------
# write_xyz
# ---------------------------------------------------------------------------

class TestWriteXyz:
    def test_write_xyz_to_buffer(self):
        from mcpy.ensembles.base_ensemble import write_xyz
        import io
        atoms = Atoms('Cu4', positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0], [0, 0, 1.8]],
                      cell=[5, 5, 5], pbc=True)
        buf = io.StringIO()
        write_xyz(atoms, -1.234, buf)
        content = buf.getvalue()
        assert str(len(atoms)) in content
        assert 'energy=-1.234000' in content
        assert 'Lattice=' in content

    def test_write_xyz_path_fallback(self, tmp_path):
        from mcpy.ensembles.base_ensemble import write_xyz
        atoms = Atoms('Cu2', positions=[[0, 0, 0], [1.8, 0, 0]], cell=[5, 5, 5], pbc=True)
        path = str(tmp_path / 'traj.xyz')
        write_xyz(atoms, -2.5, path)
        content = open(path).read()
        assert 'energy=-2.500000' in content
        assert str(len(atoms)) in content
