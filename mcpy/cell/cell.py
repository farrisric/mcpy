import numpy as np
import math


class BaseCell:
    def __init__(self, atoms):
        """
        Initialize the BaseCell object.

        :param atoms: An object containing the cell information (e.g., atoms.cell provides the cell
        dimensions).
        :param custom_height: Optional. The height of the custom cell (must be less than or equal
        to the original height).
        :param bottom_z: Optional. The z-coordinate of the bottom of the custom cell.
        """
        self.original_dimensions = np.array(atoms.cell)
        self.dimensions = self.original_dimensions
        self.offset = 0

    def calculate_volume(self):
        """
        Calculate the volume of the cell.

        :return: Volume of the cell.
        """
        self.volume = abs(np.linalg.det(self.dimensions))

    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = np.random.rand(3)
        cartesian_point = frac_coords @ self.dimensions
        return cartesian_point

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return self.volume


class Cell(BaseCell):
    def __init__(self, atoms):
        pass


class CustomCell(BaseCell):
    """
    A class representing a custom cell with additional functionality.
    """

    def __init__(self, atoms, custom_height=None, bottom_z=None, species_radii=None,
                 mc_sample_points: int = 100_000):
        """
        Initialize the CustomCell object.

        :param atoms: ASE Atoms object containing the cell information.
        :param custom_height: Optional. The height of the custom cell.
        :param bottom_z: Optional. The z-coordinate of the bottom of the custom cell.
        :param species_radii: Optional. A dictionary where keys are chemical species and values are
        their radii.
        """
        super().__init__(atoms)
        self.dimensions, self.offset = self.get_custom_height_cell(custom_height, bottom_z)
        self.species_radii = species_radii if species_radii else {}
        self.cell_volume = abs(np.linalg.det(self.dimensions))
        self.mc_sample_points = mc_sample_points

    def get_custom_height_cell(self, custom_height, bottom_z):
        """
        Calculate the dimensions and offset for a custom height cell.

        :param custom_height: The height of the custom cell.
        :param bottom_z: The z-coordinate of the bottom of the custom cell.
        :return: A tuple containing the new dimensions and offset.
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
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        frac_coords = np.random.rand(3)
        cartesian_point = frac_coords @ self.dimensions + self.offset
        return cartesian_point

    def calculate_volume(self, atoms) -> float:
        """
        Calculate the free volume of the custom cell using a Monte Carlo method.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param n_points: Number of random points to sample.
        :return: Estimated free volume of the cell.
        """
        frac_coords = np.random.rand(self.mc_sample_points, 3)
        cart_coords = frac_coords @ self.dimensions

        positions = atoms.get_positions() - self.offset
        radii = np.array([self.species_radii[atom.symbol] for atom in atoms])
        r_sq = radii**2

        deltas = cart_coords[:, None, :] - positions[None, :, :]
        dists_sq = np.einsum('ijk,ijk->ij', deltas, deltas)

        is_inside_any = np.any(dists_sq <= r_sq[None, :], axis=1)
        count_inside = np.count_nonzero(is_inside_any)

        occupied_fraction = count_inside / self.mc_sample_points
        occupied_volume = self.cell_volume * occupied_fraction
        free_volume = self.cell_volume - occupied_volume

        return free_volume
    
    def get_atoms_inside_cell(self, atoms):
        """
        Get the atoms that are inside the small cell within the supercell.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :return: ASE Atoms object containing only the atoms inside the small cell.
        """
        inside_indices = []

        atoms_species = [a for a in atoms if a.symbol in self.species_radii.keys()]
        for a in atoms_species:
            position = a.position
            if self.offset[2] <= position[2] <= self.offset[2] + self.dimensions[2][2]:
                inside_indices.append(a.index)

        return atoms[inside_indices]

    def get_atoms_specie_inside_cell(self, atoms, specie):
        """
        Get the atoms that are inside the small cell within the supercell.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :return: ASE Atoms object containing only the atoms inside the small cell.
        """
        inside_indices = []

        atoms_species = [a for a in atoms if a.symbol in specie]
        for a in atoms_species:
            position = a.position
            if self.offset[2] <= position[2] <= self.offset[2] + self.dimensions[2][2]:
                inside_indices.append(a.index)

        return inside_indices

    def get_custom_dimensions(self):
        """
        Get the dimensions and offset of the custom cell.

        :return: A tuple containing the dimensions and offset.
        """
        return self.dimensions, self.offset


class SphericalCell(BaseCell):
    """
    A class representing a spherical cell for nanoparticles.
    """

    def __init__(self, atoms, vacuum, species_radii, mc_sample_points=10000):
        """
        Initialize the SphericalCell object.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :param center: Optional. The center of the spherical cell (default is the geometric center
                       of the atoms).
        :param radius: Optional. The radius of the spherical cell.
        """
        super().__init__(atoms)
        self.center = atoms.get_center_of_mass()
        self.radius = np.linalg.norm(atoms.positions - self.center, axis=1).max() + vacuum
        self.species_radii = species_radii
        self.mc_sample_points = mc_sample_points
        self.sphere_volume = (4 / 3) * np.pi * (self.radius ** 3)

    def calculate_volume(self, atoms):
        """
        Calculate the volume of the spherical cell.

        :return: Volume of the spherical cell.
        """
        self.volume = self.estimate_free_volume(atoms)

    def is_point_inside(self, point):
        """
        Check if a given point is inside the spherical cell.

        :param point: A numpy array representing the point (x, y, z).
        :return: True if the point is inside the spherical cell, False otherwise.
        """
        distance = np.linalg.norm(point - self.center)
        return distance <= self.radius

    def get_random_point(self):
        """
        Get a random point inside the spherical cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        while True:
            random_point = self.center + (np.random.rand(3) - 0.5) * 2 * self.radius
            if self.is_point_inside(random_point):
                return random_point

    def get_atoms_specie_inside_cell(self, atoms, specie):
        return [a.index for a in atoms if a.symbol in specie and self.is_point_inside(a.position)]

    def estimate_free_volume(self, atoms):
        """
        Estimate the free volume of the spherical cell using Monte Carlo sampling.

        :param atoms: ASE Atoms object.
        :param species_radii: Dict mapping atom symbols to atomic radii.
        :param mc_sample_points: Number of random points.
        :return: Estimated free volume.
        """
        # Generate random points inside the spherical cell
        random_points = []
        batch_size = int(1.2 * self.mc_sample_points)  # slight oversampling
        while len(random_points) < self.mc_sample_points:
            points = self.center + (np.random.rand(batch_size, 3) - 0.5) * 2 * self.radius
            dists = np.linalg.norm(points - self.center, axis=1)
            inside = points[dists <= self.radius]
            random_points.append(inside)
            if len(np.concatenate(random_points)) >= self.mc_sample_points:
                break

        cart_coords = np.concatenate(random_points)[:self.mc_sample_points]  # (N, 3)

        # Atom positions and radii
        positions = atoms.get_positions()
        radii = np.array([self.species_radii[atom.symbol] for atom in atoms])
        r_sq = radii**2  # (n_atoms,)

        # Vectorized distance computation (N, M, 3)
        deltas = cart_coords[:, None, :] - positions[None, :, :]  # (n_points, n_atoms, 3)
        dists_sq = np.einsum('ijk,ijk->ij', deltas, deltas)        # (n_points, n_atoms)

        is_inside_any_atom = np.any(dists_sq <= r_sq[None, :], axis=1)
        count_free = np.count_nonzero(~is_inside_any_atom)

        free_fraction = count_free / self.mc_sample_points
        return free_fraction * self.sphere_volume
