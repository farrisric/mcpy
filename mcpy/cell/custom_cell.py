import numpy as np
from scipy.spatial import cKDTree

from .cell import Cell


class CustomCell(Cell):
    """
    A class representing a custom cell with additional functionality.
    """

    def __init__(self, atoms, custom_height=None, bottom_z=None, species_radii=None,
                 mc_sample_points: int = 100_000, seed=None):
        """
        Initialize the CustomCell object.

        :param atoms: ASE Atoms object containing the cell information.
        :param custom_height: Optional. The height of the custom cell.
        :param bottom_z: Optional. The z-coordinate of the bottom of the custom cell.
        :param species_radii: Optional. A dictionary mapping chemical species to radii.
        :param mc_sample_points: Number of random points used to estimate the free volume.
        :param seed: Optional seed for the cell-local numpy RNG.
        """
        super().__init__(atoms, species_radii=species_radii, seed=seed)
        self.dimensions, self.offset = self.get_custom_height_cell(custom_height, bottom_z)
        self.species_radii = species_radii if species_radii else {}
        self.cell_volume = float(abs(np.linalg.det(self.dimensions)))
        self.mc_sample_points = int(mc_sample_points)
        self._dim_inv = np.linalg.inv(self.dimensions)
        self.volume = self.cell_volume

    def get_custom_height_cell(self, custom_height, bottom_z):
        """
        Calculate the dimensions and offset for a custom-height cell.
        """
        if custom_height > self.original_dimensions[2][2]:
            raise ValueError("Custom height cannot be greater than the original cell height.")
        if bottom_z < 0 or bottom_z + custom_height > self.original_dimensions[2][2]:
            raise ValueError("Custom cell exceeds the bounds of the original cell.")

        new_dimensions = np.copy(self.original_dimensions)
        new_dimensions[2][2] = custom_height

        offset = np.zeros(3)
        offset[2] = bottom_z

        return new_dimensions, offset

    def get_random_point(self):
        """
        Get a random point inside the custom cell.
        """
        frac_coords = self._rng.random(3)
        return frac_coords @ self.dimensions + self.offset

    def calculate_volume(self, atoms) -> float:
        """
        Estimate the free volume of the custom cell using kdtree nearest-atom
        queries per unique radius. Avoids the O(N_points * N_atoms) broadcast.
        """
        n_atoms = len(atoms)
        if n_atoms == 0:
            self.volume = self.cell_volume
            return

        frac_coords = self._rng.random((self.mc_sample_points, 3))
        cart_coords = frac_coords @ self.dimensions  # in cell frame

        positions = atoms.positions - self.offset
        symbols = atoms.get_chemical_symbols()
        radii = np.fromiter(
            (self.species_radii[s] for s in symbols),
            dtype=float, count=n_atoms,
        )

        covered = np.zeros(self.mc_sample_points, dtype=bool)
        for r in np.unique(radii):
            if r <= 0.0:
                continue
            mask = radii == r
            tree = cKDTree(positions[mask])
            dists, _ = tree.query(cart_coords, k=1, distance_upper_bound=float(r))
            covered |= np.isfinite(dists)

        occupied_fraction = float(np.count_nonzero(covered)) / self.mc_sample_points
        self.volume = self.cell_volume * (1.0 - occupied_fraction)

    def get_atoms_specie_inside_cell(self, atoms, specie):
        """
        Vectorized: indices of atoms with symbol in ``specie`` that are
        exchangeable with the reservoir.

        In xy an atom must lie within the cell footprint. In z the upper bound
        is intentionally dropped: atoms at or above the cell floor count as
        exchangeable, including ones that desorbed and drifted above the cell
        top (e.g. a recombined H2 floating off the surface). Without this they
        would never be selected for deletion and would accumulate forever.
        Atoms below the floor (absorbed into the subsurface layers) stay
        excluded so they are kept.
        """
        if len(atoms) == 0:
            return np.empty(0, dtype=int)
        symbols = np.asarray(atoms.get_chemical_symbols())
        if isinstance(specie, str):
            species_list = [specie]
        else:
            species_list = list(specie)
        species_mask = np.isin(symbols, species_list)
        frac = (atoms.positions - self.offset) @ self._dim_inv
        in_xy = np.all((frac[:, :2] >= 0.0) & (frac[:, :2] < 1.0), axis=1)
        above_floor = frac[:, 2] >= 0.0
        inside = in_xy & above_floor
        return np.where(species_mask & inside)[0]

    def is_point_inside(self, point):
        """
        Check if a given point is inside the custom cell.
        """
        frac = (point - self.offset) @ self._dim_inv
        return bool(np.all((frac >= 0.0) & (frac < 1.0)))

    def get_species(self):
        """
        Get the species present in the custom cell.
        """
        return list(self.species_radii.keys())
