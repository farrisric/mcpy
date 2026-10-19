import logging

import numpy as np
from scipy.spatial import cKDTree

from .base_cell import BaseCell

logger = logging.getLogger(__name__)

_MAX_INSERT_ATTEMPTS = 1000
# Adaptive free-volume sampling: draw enough points that the relative noise
# sqrt((1 - f) / (f N)) of the estimate stays near this target, given the
# free fraction f of the previous estimate.
_TARGET_REL_NOISE = 0.005
_MIN_SAMPLE_POINTS = 5_000


class Cell(BaseCell):
    """The whole periodic box as the sampling volume.

    Also the base of the region cells, and the one place that defines the
    insertion exclusion rule: a point is excluded when it lies within
    ``species_radii[symbol]`` of any atom, periodic images included. The
    insertion proposal (:meth:`get_random_free_point`) and the free-volume
    estimate (:meth:`free_volume`) both test that rule, so the proposal
    density is exactly ``1 / V`` for the ``V`` the acceptance uses (see
    docs/gcmc_acceptance_convention.rst). Without ``species_radii`` nothing
    is excluded and the volume is the exact region volume.
    """

    def __init__(self, atoms, species_radii=None, seed=None,
                 mc_sample_points: int = 100_000):
        """
        Initialize the Cell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param species_radii: Optional dict mapping species to exclusion
                              radii. When given, the volume is the free
                              volume and insertions avoid the excluded
                              spheres.
        :param seed: Optional seed for the cell-local numpy RNG used by
                     :meth:`get_random_point` and the free-volume sampler.
                     ``None`` seeds that generator from OS entropy, so the
                     run is not repeatable.
        :param mc_sample_points: Maximum number of random points used to
                                 estimate the free volume when
                                 ``species_radii`` is set. The first estimate
                                 draws this many; later ones draw only enough
                                 for about 0.5% relative noise at the previous
                                 free fraction (see :attr:`mc_sample_points`).
        """
        super().__init__()
        self.original_dimensions = np.array(atoms.cell)
        self.dimensions = self.original_dimensions
        self.offset = np.zeros(3)
        self.species_radii = species_radii if species_radii else {}
        self.mc_sample_points = int(mc_sample_points)
        self.cell_volume = float(abs(np.linalg.det(self.dimensions)))
        self.volume = self.cell_volume
        self._rng = np.random.default_rng(seed)

    @property
    def mc_sample_points(self):
        """Points the next free-volume estimate draws. Adaptive, at most
        ``max_sample_points``."""
        return self._n_samples

    @mc_sample_points.setter
    def mc_sample_points(self, n):
        """Set the cap; the next estimate draws the full cap."""
        self.max_sample_points = self._n_samples = int(n)

    def _adapt_sample_points(self, free_fraction):
        """Pick the next sample count from the free fraction of the domain
        just measured: N = (1 - f) / (f * target^2), clamped to the cap."""
        cap = self.max_sample_points
        if free_fraction <= 0.0:
            self._n_samples = cap
            return
        n = (1.0 - free_fraction) / (free_fraction * _TARGET_REL_NOISE ** 2)
        self._n_samples = int(np.clip(np.ceil(n), min(_MIN_SAMPLE_POINTS, cap), cap))

    def calculate_volume(self, atoms):
        """Store the free volume for ``atoms`` (see :meth:`free_volume`)."""
        self.volume = self.free_volume(atoms)

    def free_volume(self, atoms):
        """Free volume of the region for ``atoms``, without storing it.

        The region volume minus the exclusion spheres of every atom
        (periodic images included), estimated from ``mc_sample_points``
        uniform points, then adapts that count for the next call. Exact
        region volume when nothing can be excluded.
        """
        trees = self._exclusion_trees(atoms)
        if not trees:
            return self._region_volume()
        n_drawn = self.mc_sample_points
        points, domain_volume = self._volume_samples()
        n_free = np.count_nonzero(~self._covered(points, trees))
        volume = self._clamp_free_volume(domain_volume * n_free / n_drawn, domain_volume)
        self._adapt_sample_points(n_free / n_drawn)
        return volume

    def get_random_free_point(self, atoms, clearance=None):
        """A point uniform over the free volume of ``atoms``.

        Redraws :meth:`get_random_point` until the point is outside every
        exclusion sphere, the same test :meth:`free_volume` counts with.
        After ``_MAX_INSERT_ATTEMPTS`` misses it takes the first free point
        of one batch of free-volume samples (still uniform over the free
        volume). Returns ``None``, with a warning, when that batch misses too.
        Giving up silently would thin insertions but not deletions in a
        nearly full region.

        With ``clearance`` the point only keeps ``clearance`` from every
        atom instead, a seed for a relaxed trial whose acceptance still uses
        the ``species_radii`` free volume (ADR 0004).
        """
        trees = self._exclusion_trees(atoms, clearance)
        for _ in range(_MAX_INSERT_ATTEMPTS):
            point = self.get_random_point()
            if not trees or not self._covered(point[None, :], trees)[0]:
                return point
        points, _ = self._volume_samples()
        free = np.flatnonzero(~self._covered(points, trees))
        if len(free):
            return points[free[0]]
        logger.warning(
            'No free insertion point in %d + %d draws: the region is '
            'effectively full and this insertion is lost.',
            _MAX_INSERT_ATTEMPTS, len(points))
        return None

    def get_random_point(self):
        """
        Get a random point inside the cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = self._rng.random(3)
        return frac_coords @ self.dimensions + self.offset

    def _region_volume(self):
        """Exact volume of the proposal region."""
        return self.cell_volume

    def _volume_samples(self):
        """``mc_sample_points`` uniform draws over a domain that contains the
        region, keeping only the ones inside the region, and the domain
        volume. The free volume is ``domain_volume * n_free / n_drawn``.

        Draws the full cap and keeps the first ``mc_sample_points``: the
        adaptive count follows GPU-noisy positions, and a count-sized draw
        would shift the seeded stream that later insertion points use."""
        frac = self._rng.random((self.max_sample_points, 3))[:self.mc_sample_points]
        return frac @ self.dimensions + self.offset, self.cell_volume

    def _bounds(self):
        """Axis-aligned bounding box (lo, hi) of the proposal region."""
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1)
                            for k in (0, 1)], dtype=float)
        corners = corners @ self.dimensions + self.offset
        return corners.min(axis=0), corners.max(axis=0)

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return self.volume

    def get_atoms_specie_inside_cell(self, atoms, specie):
        """Indices of atoms whose symbol is in ``specie`` and that lie inside
        the exchangeable region.

        ``specie`` may be a single symbol or a list. The region test is
        ``_inside_mask``, which the box cell answers with "everywhere";
        the region cells override it.
        """
        if len(atoms) == 0:
            return np.empty(0, dtype=int)
        symbols = np.asarray(atoms.get_chemical_symbols())
        species_list = [specie] if isinstance(specie, str) else list(specie)
        species_mask = np.isin(symbols, species_list)
        return np.where(species_mask & self._inside_mask(atoms))[0]

    def _inside_mask(self, atoms):
        """Per-atom boolean: is this atom in the exchangeable region?

        The box cell spans the whole periodic cell, so every atom qualifies.
        """
        return np.ones(len(atoms), dtype=bool)

    def get_species(self):
        """
        Get the species present in the custom cell.

        :return: A list of species present in the custom cell.
        """
        return list(self.species_radii.keys())

    def _radii_for(self, atoms):
        """Per-atom exclusion radii for the free-volume samplers.

        Raises a message that names the missing symbols instead of the bare
        ``KeyError`` a dict lookup would throw from inside the sampler. The
        check lives here rather than in ``__init__`` because GCMC inserts
        species that are absent when the cell is built.

        :return: ndarray of radii, one per atom, in ``atoms`` order.
        """
        symbols = atoms.get_chemical_symbols()
        missing = sorted(set(symbols) - set(self.species_radii))
        if missing:
            raise ValueError(
                f'{type(self).__name__}.species_radii has no radius for '
                f'{missing}. Every species the system can contain needs one, '
                f'including species inserted during the run; got '
                f'{sorted(self.species_radii)}.'
            )
        return np.fromiter((self.species_radii[s] for s in symbols),
                           dtype=float, count=len(symbols))

    def _periodic_images(self, atoms):
        """Positions of the atoms plus their -1/0/+1 periodic images along
        each periodic direction, and the number of copies."""
        pbc = np.asarray(atoms.pbc, dtype=bool)
        shifts = [(-1, 0, 1) if p else (0,) for p in pbc]
        vectors = np.array([[i, j, k] for i in shifts[0] for j in shifts[1]
                            for k in shifts[2]], dtype=float) @ np.array(atoms.cell)
        images = (atoms.positions[None, :, :] + vectors[:, None, :]).reshape(-1, 3)
        return images, len(vectors)

    def _exclusion_trees(self, atoms, clearance=None):
        """One cKDTree per distinct positive radius over every atom and
        periodic image whose sphere can reach the region. Empty when nothing
        is excluded. Shared by the free-volume estimate and the insertion
        proposal so both apply the same rule. ``clearance`` replaces the
        ``species_radii`` with one radius for every atom."""
        if len(atoms) == 0 or (clearance is None and not self.species_radii):
            return []
        positions, n_copies = self._periodic_images(atoms)
        if clearance is None:
            radii = np.tile(self._radii_for(atoms), n_copies)
        else:
            radii = np.full(len(positions), float(clearance))
        lo, hi = self._bounds()
        reach = radii[:, None]
        near = np.all((positions > lo - reach) & (positions < hi + reach), axis=1)
        return [(cKDTree(positions[near & (radii == r)]), float(r))
                for r in np.unique(radii[near]) if r > 0.0]

    @staticmethod
    def _covered(points, trees):
        """Which of ``points`` lie inside some exclusion sphere."""
        covered = np.zeros(len(points), dtype=bool)
        for tree, r in trees:
            # distance_upper_bound returns inf above r; that is "not covered".
            dists, _ = tree.query(points, k=1, distance_upper_bound=r)
            covered |= np.isfinite(dists)
        return covered

    def is_point_inside(self, point):
        """The box cell spans the whole periodic cell: every point is inside.

        Exists so molecule moves can call ``is_point_inside`` uniformly on
        any cell type (the region cells implement a real test).
        """
        return True
