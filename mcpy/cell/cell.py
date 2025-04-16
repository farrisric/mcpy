import numpy as np
import math


class Cell:
    def __init__(self, atoms, custom_height=None, center_factor=None, species_radii=None):
        """
        Initialize the Cell object.

        :param atoms: An object containing the cell information (e.g., atoms.cell provides the cell dimensions).
        :param custom_height: Optional. The height of the custom cell (must be less than or equal to the original height).
        :param center_factor: Optional. A float between 0 and 1 indicating where to center the custom cell along the z-axis.
                              0 means bottom-aligned, 1 means top-aligned, and 0.5 means centered.
        :param species_radii: Optional. A dictionary where keys are chemical species and values are their radii.
        """
        self.original_dimensions = np.array(atoms.cell)
        if custom_height is not None and center_factor is not None:
            self.dimensions = self.get_custom_height_cell(custom_height, center_factor)
        else:
            self.dimensions = self.original_dimensions
        self.volume = self.calculate_volume()
        self.species_radii = species_radii if species_radii else {}

    def calculate_volume(self):
        """
        Calculate the volume of the cell using the determinant of the dimensions matrix.

        :return: Volume of the cell.
        """
        return abs(np.linalg.det(self.dimensions))

    def get_reduced_simulation_box(self, reduction_factor):
        """
        Get a reduced simulation box by scaling the dimensions.

        :param reduction_factor: A float representing the scaling factor (0 < reduction_factor <= 1).
        :return: A new numpy array representing the reduced simulation box.
        """
        if not (0 < reduction_factor <= 1):
            raise ValueError("Reduction factor must be between 0 and 1.")
        return self.dimensions * reduction_factor

    def update_volume_with_new_atom(self, atom_volume):
        """
        Update the volume of the cell when a new atom is added.

        :param atom_volume: The volume of the new atom to be added.
        """
        self.volume += atom_volume

    @staticmethod
    def sphere_volume(radius):
        """
        Calculate the volume of a sphere given its radius.

        :param radius: Radius of the sphere.
        :return: Volume of the sphere.
        """
        return (4 / 3) * math.pi * radius**3

    @staticmethod
    def overlap_volume(r1, r2, d):
        """
        Calculate the volume of overlap between two spheres of radius r1, r2 and distance d.

        :param r1: Radius of the first sphere.
        :param r2: Radius of the second sphere.
        :param d: Distance between the centers of the two spheres.
        :return: Overlap volume.
        """
        if d >= r1 + r2:
            return 0  # No overlap
        if d <= abs(r1 - r2):
            return (4 / 3) * math.pi * min(r1, r2)**3

        part1 = (math.pi * (r1 + r2 - d)**2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2)**2)) / (12 * d)
        return part1

    @staticmethod
    def total_volume_with_overlap(spheres, positions):
        """
        Calculate the total volume of spheres considering overlaps.

        :param spheres: List of sphere radii.
        :param positions: List of sphere center positions as (x, y, z) tuples.
        :return: Total volume considering overlaps.
        """
        total_vol = sum(Cell.sphere_volume(radius) for radius in spheres)
        num_spheres = len(spheres)

        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                d = math.dist(positions[i], positions[j])  # Distance between sphere centers
                total_vol -= Cell.overlap_volume(spheres[i], spheres[j], d)

        return total_vol

    def get_custom_height_cell(self, height, center_factor):
        """
        Get a smaller cell with the same x and y plane but with a custom height and centering.

        :param height: The height of the new cell (must be less than or equal to the original height).
        :param center_factor: A float between 0 and 1 indicating where to center the new cell along the z-axis.
                              0 means bottom-aligned, 1 means top-aligned, and 0.5 means centered.
        :return: A numpy array representing the custom cell dimensions.
        """
        if height > self.original_dimensions[2, 2]:
            raise ValueError("Height must be less than or equal to the original cell height.")
        if not (0 <= center_factor <= 1):
            raise ValueError("Center factor must be between 0 and 1.")

        new_dimensions = self.original_dimensions.copy()
        z_offset = (self.original_dimensions[2, 2] - height) * center_factor
        new_dimensions[2, 2] = height
        new_dimensions[2, :] += z_offset * self.original_dimensions[2, :] / self.original_dimensions[2, 2]
        return new_dimensions

    def get_random_point(self):
        """
        Get a random point inside the cell or the custom cell.

        :return: A numpy array representing the random point (x, y, z).
        """
        dimensions = self.dimensions
        random_point = np.dot(dimensions, np.random.rand(3))
        return random_point

    def calculate_occupied_volume(self, species_positions):
        """
        Calculate the occupied volume of the chemical species inside the cell, accounting for overlaps
        and considering only atoms that are inside the cell.

        :param species_positions: A dictionary where keys are chemical species and values are lists of positions
                                  (x, y, z) of the atoms of that species.
        :return: The total occupied volume inside the cell.
        """
        if not self.species_radii:
            raise ValueError("No species radii provided to calculate occupied volume.")

        total_volume = 0
        for species, positions in species_positions.items():
            if species not in self.species_radii:
                raise ValueError(f"Radius for species '{species}' not provided.")
            radius = self.species_radii[species]

            # Filter positions to include only those inside the cell
            filtered_positions = [
                pos for pos in positions
                if np.all(np.dot(np.linalg.inv(self.dimensions), pos) >= 0) and
                   np.all(np.dot(np.linalg.inv(self.dimensions), pos) <= 1)
            ]

            total_volume += sum(self.sphere_volume(radius) for _ in filtered_positions)

            # Subtract overlapping volumes
            num_positions = len(filtered_positions)
            for i in range(num_positions):
                for j in range(i + 1, num_positions):
                    d = math.dist(filtered_positions[i], filtered_positions[j])  # Distance between atom centers
                    total_volume -= self.overlap_volume(radius, radius, d)

        return total_volume


1