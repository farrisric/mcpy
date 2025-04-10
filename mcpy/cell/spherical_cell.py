import numpy as np

from .cell import Cell


class SphericalCell(Cell):
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
        :param species_radii: Dict mapping atom symbols to atomic radii.
        :param mc_sample_points: Number of random points.
        """
        super().__init__()
        self.center = atoms.get_center_of_mass()
        self.move_atoms_to_center(atoms)
        self.radius = np.linalg.norm(atoms.positions - self.center, axis=1).max() + vacuum
        self.species_radii = species_radii
        self.mc_sample_points = mc_sample_points
        self.sphere_volume = (4 / 3) * np.pi * (self.radius ** 3)

    def center(self, atoms):
        """
        Translate the atoms so that their center of mass coincides with the center of the spherical
        cell. This modifies the atoms in place.

        :param atoms: ASE Atoms object to be moved.
        """
        current_com = atoms.get_center_of_mass()
        shift = self.center - current_com
        atoms.translate(shift)

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
        :return: Estimated free volume.
        """
        random_points = []
        batch_size = int(1.2 * self.mc_sample_points)
        while len(random_points) < self.mc_sample_points:
            points = self.center + (np.random.rand(batch_size, 3) - 0.5) * 2 * self.radius
            dists = np.linalg.norm(points - self.center, axis=1)
            inside = points[dists <= self.radius]
            random_points.append(inside)
            if len(np.concatenate(random_points)) >= self.mc_sample_points:
                break

        cart_coords = np.concatenate(random_points)[:self.mc_sample_points]

        positions = atoms.get_positions()
        radii = np.array([self.species_radii[atom.symbol] for atom in atoms])
        r_sq = radii**2

        deltas = cart_coords[:, None, :] - positions[None, :, :]
        dists_sq = np.einsum('ijk,ijk->ij', deltas, deltas)

        is_inside_any_atom = np.any(dists_sq <= r_sq[None, :], axis=1)
        count_free = np.count_nonzero(~is_inside_any_atom)

        free_fraction = count_free / self.mc_sample_points
        return free_fraction * self.sphere_volume
