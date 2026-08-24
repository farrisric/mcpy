import numpy as np
from scipy.spatial import cKDTree

from .base_cell import BaseCell


class Cell(BaseCell):
    def __init__(self, atoms, species_radii=None, seed=None):
        """
        Initialize the Cell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param species_radii: Optional dict mapping species to radii.
        :param seed: Optional seed for the cell-local numpy RNG used by
                     :meth:`get_random_point`. ``None`` falls back to the
                     numpy global generator.
        """
        super().__init__()
        self.original_dimensions = np.array(atoms.cell)
        self.dimensions = self.original_dimensions
        self.species_radii = species_radii if species_radii else {}
        # Dimensions are fixed at construction; cache the box volume so
        # ``calculate_volume`` is a single attribute assignment.
        self._box_volume = float(abs(np.linalg.det(self.dimensions)))
        self.volume = self._box_volume
        self._rng = np.random.default_rng(seed)

    def calculate_volume(self, atoms):
        """
        Set the cell volume. ``dimensions`` is fixed at construction so the
        determinant is taken once in ``__init__``.
        """
        self.volume = self._box_volume

    def get_random_point(self):
        """
        Get a random point inside the cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = self._rng.random(3)
        return frac_coords @ self.dimensions

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
        :meth:`_inside_mask`, which the box cell answers with "everywhere";
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

    def _covered_mask(self, points, atoms, positions=None, radii=None):
        """Which of ``points`` fall inside some atom's exclusion sphere.

        One cKDTree per distinct radius, querying the nearest atom of that
        radius: a point is covered when that nearest atom is within it. Avoids
        the O(N_points * N_atoms) broadcast. Shared by every cell that
        estimates a free volume; the cells differ in how they *sample* the
        points, not in how they test them.

        ``positions`` and ``radii`` override the ones taken from ``atoms``, for
        the periodic-image expansion :class:`CustomCell` needs.
        """
        if radii is None:
            radii = self._radii_for(atoms)
        if positions is None:
            positions = atoms.positions
        covered = np.zeros(len(points), dtype=bool)
        for r in np.unique(radii):
            if r <= 0.0:
                continue
            tree = cKDTree(positions[radii == r])
            # distance_upper_bound returns inf above r; that is the rejection.
            dists, _ = tree.query(points, k=1, distance_upper_bound=float(r))
            covered |= np.isfinite(dists)
        return covered

    def is_point_inside(self, point):
        """The box cell spans the whole periodic cell: every point is inside.

        Exists so molecule moves can call ``is_point_inside`` uniformly on
        any cell type (the region cells implement a real test).
        """
        return True
