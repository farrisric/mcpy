import numpy as np

from .cell import Cell


class SphericalCell(Cell):
    """
    A class representing a spherical cell for nanoparticles.

    .. warning::
       Construction TRANSLATES the passed atoms in place so their center of
       mass sits at the cell center: the center of the periodic box, or the
       origin when the atoms have no box. Use :class:`DomeCell` for
       supported systems where the atoms must not be moved.
    """

    def __init__(self, atoms, vacuum, species_radii, mc_sample_points=100_000, seed=None):
        """
        Initialize the SphericalCell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param vacuum: Padding added to the bounding radius.
        :param species_radii: Dict mapping atom symbols to exclusion radii
                              (docs/species_radii.rst).
        :param mc_sample_points: Number of random points used to estimate
                                 the free volume.
        :param seed: Optional seed for the cell-local numpy RNG used by
                     ``get_random_point`` and the volume sampler.
        :raises ValueError: if the sphere does not fit inside the periodic box.
        """
        super().__init__(atoms, species_radii=species_radii, seed=seed,
                         mc_sample_points=mc_sample_points)
        # Box center, not the origin: with pbc the ensemble wraps the atoms
        # into the box after every accepted move, which would scatter a
        # particle centered on a box corner across the box.
        self.center = 0.5 * np.array(atoms.cell).sum(axis=0)
        self.move_atoms_to_center(atoms)
        self.radius = float(
            np.linalg.norm(atoms.positions - self.center, axis=1).max() + vacuum
        )
        self._check_fits_box(atoms)
        self.sphere_volume = (4.0 / 3.0) * np.pi * (self.radius ** 3)
        self.volume = self.sphere_volume

    def _check_fits_box(self, atoms, lower_z=None):
        """Raise if the region crosses a face of the periodic box.

        Along each periodic direction the distance from ``center`` to both
        faces must be at least ``radius``; otherwise the region overlaps its
        own periodic image. ``lower_z`` replaces the bottom of the sphere
        along the third axis when the region is clipped there (dome).
        """
        cell = np.array(atoms.cell)
        for axis in np.flatnonzero(atoms.pbc):
            normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
            normal /= np.linalg.norm(normal)
            if normal @ cell[axis] < 0:
                normal = -normal
            height = float(normal @ cell[axis])
            pos = float(normal @ self.center)
            below = self.radius
            if lower_z is not None and axis == 2 and np.allclose(normal, [0, 0, 1]):
                below = self.center[2] - lower_z
            if pos - below < -1e-9 or pos + self.radius > height + 1e-9:
                raise ValueError(
                    f'{type(self).__name__} of radius {self.radius:.3f} A does '
                    f'not fit the periodic box along axis {axis} (box height '
                    f'{height:.3f} A, center at {pos:.3f} A). The region would '
                    f'overlap its periodic image: enlarge the box or reduce '
                    f'vacuum.')

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
        # The ball fills pi/6 ~ 0.524 of its bounding cube, so 2n cube draws
        # keep ~1.05n points: one batch almost always; the loop covers the
        # rare short batch.
        oversample = 2 * n + 1
        while filled < n:
            pts = (self._rng.random((oversample, 3)) - 0.5) * 2.0 * self.radius
            d2 = np.einsum('ij,ij->i', pts, pts)
            keep = pts[d2 <= radius_sq]
            take = min(len(keep), n - filled)
            out[filled:filled + take] = keep[:take]
            filled += take
        return out + self.center

    def _region_volume(self):
        return self.sphere_volume

    def _volume_samples(self):
        return self._sample_sphere_points(self.mc_sample_points), self.sphere_volume

    def _bounds(self):
        return self.center - self.radius, self.center + self.radius
