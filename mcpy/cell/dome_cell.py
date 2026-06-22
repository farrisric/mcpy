import numpy as np
from scipy.spatial import cKDTree

from .cell import Cell
from .spherical_cell import SphericalCell


class DomeCell(SphericalCell):
    """A hemispherical ('dome') insertion region centered on a supported particle.

    The region is the ball of radius ``R`` around the particle centroid,
    truncated at the support surface (``z >= bottom_z``). It differs from
    :class:`SphericalCell` in two ways that matter for a supported nanoparticle:

    - it centers on a chosen *particle* species (e.g. the metal) rather than on
      the whole-system center of mass, so the dome sits on the nanoparticle and
      not on the support slab;
    - it does not translate the atoms, and it clips the region at the support
      surface.

    The result is that trial insertions land on and around the particle (and its
    metal-support contact rim) instead of spreading across the bare support.
    """

    def __init__(self, atoms, particle_species, bottom_z, vacuum,
                 species_radii, mc_sample_points: int = 100_000, seed=None):
        """
        :param atoms: ASE Atoms object (support + particle).
        :param particle_species: Symbol or list of symbols identifying the
                                 nanoparticle (e.g. ``'Ag'``). Used to locate the
                                 dome center and radius.
        :param bottom_z: z-coordinate of the support surface; the region is
                         clipped to ``z >= bottom_z``.
        :param vacuum: Padding added to the particle bounding radius.
        :param species_radii: Dict mapping atom symbols to radii (free-volume
                              exclusion).
        :param mc_sample_points: Number of random points used to estimate the
                                 free volume.
        :param seed: Optional seed for the cell-local numpy RNG.
        """
        # Initialise the base Cell directly: SphericalCell.__init__ would move the
        # atoms to their center of mass, which is wrong for a supported slab.
        Cell.__init__(self, atoms, species_radii=species_radii, seed=seed)

        if isinstance(particle_species, str):
            particle_species = [particle_species]
        symbols = np.asarray(atoms.get_chemical_symbols())
        particle_mask = np.isin(symbols, particle_species)
        if not particle_mask.any():
            raise ValueError(
                f'No atoms of particle_species={particle_species} found in the cell.'
            )

        particle_pos = atoms.positions[particle_mask]
        self.center = particle_pos.mean(axis=0)
        self.bottom_z = float(bottom_z)
        self.radius = float(
            np.linalg.norm(particle_pos - self.center, axis=1).max() + vacuum
        )
        self.species_radii = species_radii
        self.mc_sample_points = int(mc_sample_points)
        self.sphere_volume = (4.0 / 3.0) * np.pi * (self.radius ** 3)
        self.volume = self.sphere_volume

    def is_point_inside(self, point):
        """Inside the ball and at or above the support surface."""
        if point[2] < self.bottom_z:
            return False
        diff = point - self.center
        return float(np.dot(diff, diff)) <= self.radius ** 2

    def get_random_point(self):
        """
        Uniform sample inside the dome (upper part of the ball). Samples the full
        ball via the cube-root method and rejects points below the surface, so
        the result is uniform over the clipped region.
        """
        while True:
            direction = self._rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            r = self.radius * (self._rng.random() ** (1.0 / 3.0))
            point = self.center + r * direction
            if point[2] >= self.bottom_z:
                return point

    def get_atoms_specie_inside_cell(self, atoms, specie):
        """
        Vectorized: indices of atoms whose symbol is in ``specie`` and whose
        position is inside the dome (within the ball and above the surface).
        Atoms that drifted below the surface (absorbed into the support) are
        excluded, so they are never selected for deletion.
        """
        if len(atoms) == 0:
            return np.empty(0, dtype=int)
        symbols = np.asarray(atoms.get_chemical_symbols())
        if isinstance(specie, str):
            species_list = [specie]
        else:
            species_list = list(specie)
        species_mask = np.isin(symbols, species_list)
        diff = atoms.positions - self.center
        within = np.einsum('ij,ij->i', diff, diff) <= self.radius ** 2
        above = atoms.positions[:, 2] >= self.bottom_z
        return np.where(species_mask & within & above)[0]

    def calculate_volume(self, atoms):
        """
        Estimate the free volume of the dome: sample the full ball, keep points
        above the surface, then exclude points covered by an atom (per-radius
        cKDTree nearest-atom query). The free fraction is normalised by the full
        ball sample count, so ``volume`` is the free dome volume in absolute
        units.
        """
        pts = self._sample_sphere_points(self.mc_sample_points)
        above = pts[:, 2] >= self.bottom_z
        pts = pts[above]

        n_atoms = len(atoms)
        if n_atoms == 0:
            dome_fraction = float(np.count_nonzero(above)) / self.mc_sample_points
            self.volume = dome_fraction * self.sphere_volume
            return

        symbols = atoms.get_chemical_symbols()
        radii = np.fromiter(
            (self.species_radii[s] for s in symbols),
            dtype=float, count=n_atoms,
        )
        positions = atoms.positions

        covered = np.zeros(len(pts), dtype=bool)
        for r in np.unique(radii):
            if r <= 0.0:
                continue
            mask = radii == r
            tree = cKDTree(positions[mask])
            dists, _ = tree.query(pts, k=1, distance_upper_bound=float(r))
            covered |= np.isfinite(dists)

        free_fraction = float(np.count_nonzero(~covered)) / self.mc_sample_points
        self.volume = free_fraction * self.sphere_volume
