import numpy as np

from .cell import Cell


class SphericalCell(Cell):
    """
    A class representing a spherical cell for nanoparticles.

    .. warning::
       Construction TRANSLATES the passed atoms in place so their center of
       mass sits at the origin (the cell center). Use :class:`DomeCell` for
       supported systems where the atoms must not be moved.
    """

    def __init__(self, atoms, vacuum, species_radii, mc_sample_points=100_000, seed=None):
        """
        Initialize the SphericalCell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param vacuum: Padding added to the bounding radius.
        :param species_radii: Dict mapping atom symbols to atomic radii.
        :param mc_sample_points: Number of random points used to estimate
                                 the free volume.
        :param seed: Optional seed for the cell-local numpy RNG used by
                     ``get_random_point`` and the volume sampler.
        """
        super().__init__(atoms, species_radii=species_radii, seed=seed)
        self.center = np.zeros(3)
        self.move_atoms_to_center(atoms)
        self.radius = float(
            np.linalg.norm(atoms.positions - self.center, axis=1).max() + vacuum
        )
        self.mc_sample_points = int(mc_sample_points)
        self.sphere_volume = (4.0 / 3.0) * np.pi * (self.radius ** 3)
        self.volume = self.sphere_volume

    def move_atoms_to_center(self, atoms):
        """
        Translate the atoms so that their center of mass coincides with the
        center of the spherical cell. Modifies the atoms in place.
        """
        current_com = atoms.get_center_of_mass()
        atoms.translate(self.center - current_com)

    def is_point_inside(self, point):
        """
        Check if a given point is inside the spherical cell.
        """
        diff = point - self.center
        return float(np.dot(diff, diff)) <= self.radius ** 2

    def get_random_point(self):
        """
        Uniform sample inside the sphere via the cube-root method. Uses the
        cell-local RNG so runs are reproducible when a seed is provided.
        """
        direction = self._rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        r = self.radius * (self._rng.random() ** (1.0 / 3.0))
        return self.center + r * direction

    def _inside_mask(self, atoms):
        """Inside the sphere."""
        diff = atoms.positions - self.center
        return np.einsum('ij,ij->i', diff, diff) <= self.radius ** 2

    def _sample_sphere_points(self, n):
        """
        Uniform random points inside the sphere. Rejection sample from the
        bounding cube; oversample to keep loops short.
        """
        out = np.empty((n, 3), dtype=float)
        filled = 0
        radius_sq = self.radius ** 2
        # 2 * (6/pi) ≈ ~3.82; 1.5x is enough most of the time, but loop to
        # cover the tail.
        oversample = int(1.5 * n) + 1
        while filled < n:
            pts = (self._rng.random((oversample, 3)) - 0.5) * 2.0 * self.radius
            d2 = np.einsum('ij,ij->i', pts, pts)
            keep = pts[d2 <= radius_sq]
            take = min(len(keep), n - filled)
            out[filled:filled + take] = keep[:take]
            filled += take
        return out + self.center

    def calculate_volume(self, atoms):
        """
        Estimate the free volume of the spherical cell.

        Approach: sample ``mc_sample_points`` points uniformly inside the
        sphere, then for each unique atomic radius query the nearest atom of
        that radius via a cKDTree. A point is "free" iff no nearest atom is
        within that atom's radius. Avoids the O(N_points * N_atoms) broadcast.
        """
        n_atoms = len(atoms)
        if n_atoms == 0:
            self.volume = self.sphere_volume
            return

        pts = self._sample_sphere_points(self.mc_sample_points)

        covered = self._covered_mask(pts, atoms)
        free_fraction = float(np.count_nonzero(~covered)) / self.mc_sample_points
        self.volume = self._clamp_free_volume(
            free_fraction * self.sphere_volume, self.sphere_volume)
