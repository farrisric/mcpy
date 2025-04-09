import numpy as np
import math


class Cell:
    def __init__(self, atoms, custom_height=None, bottom_z=None, species_radii=None):
        """
        Initialize the Cell object.

        :param atoms: An object containing the cell information (e.g., atoms.cell provides the cell
        dimensions).
        :param custom_height: Optional. The height of the custom cell (must be less than or equal
        to the original height).
        :param bottom_z: Optional. The z-coordinate of the bottom of the custom cell.
        :param species_radii: Optional. A dictionary where keys are chemical species and values are
        their radii.
        """
        self.original_dimensions = np.array(atoms.cell)
        if custom_height is not None and bottom_z is not None:
            self.dimensions, self.offset = self.get_custom_height_cell(custom_height, bottom_z)
        else:
            self.dimensions = self.original_dimensions
            self.offset = 0
        self.species_radii = species_radii if species_radii else {}
        self.cell_volume = abs(np.linalg.det(self.dimensions))

    def calculate_volume(self, atoms):
        """
        Calculate the volume of the cell using the determinant of the dimensions matrix.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :return: Volume of the cell.
        """
        self.volume = self.estimate_free_volume(atoms)

    def get_volume(self):
        """
        Get the volume of the cell.

        :return: Volume of the cell.
        """
        return self.volume

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

        part1 = (
            math.pi * (r1 + r2 - d)**2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2)**2)
            ) / (12 * d)
        return part1

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

    def calculate_occupied_volume(self, atoms):
        """
        Calculate the occupied volume of the chemical species inside the cell, accounting for
        overlaps and considering only atoms that are inside the cell.

        :param atoms: ASE Atoms object containing the atomic configuration.
        :return: The total occupied volume inside the cell.
        """
        if not self.species_radii:
            raise ValueError("No species radii provided to calculate occupied volume.")

        atoms_inside_cell = self.get_atoms_inside_cell(atoms)

        total_volume = 0
        for atom in atoms_inside_cell:
            species = atom.symbol
            if species not in self.species_radii:
                raise ValueError(f"Radius for species '{species}' not provided.")
            radius = self.species_radii[species]
            total_volume += self.sphere_volume(radius)

        # Subtract overlapping volumes
        num_atoms = len(atoms_inside_cell)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                d = math.dist(atoms_inside_cell[i].position, atoms_inside_cell[j].position)
                total_volume -= self.overlap_volume(
                    self.species_radii[atoms_inside_cell[i].symbol],
                    self.species_radii[atoms_inside_cell[j].symbol],
                    d
                )

        return total_volume

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

    def estimate_free_volume(self, atoms, n_points: int = 100_000) -> float:
        frac_coords = np.random.rand(n_points, 3)
        cart_coords = frac_coords @ self.dimensions

        positions = atoms.get_positions() - self.offset
        radii = np.array([self.species_radii[atom.symbol] for atom in atoms])
        r_sq = radii**2  # (n_atoms,)

        # Compute all squared distances: (n_points, n_atoms)
        deltas = cart_coords[:, None, :] - positions[None, :, :]
        dists_sq = np.einsum('ijk,ijk->ij', deltas, deltas)

        # Compare each distance to the squared radius of the corresponding atom
        is_inside_any = np.any(dists_sq <= r_sq[None, :], axis=1)
        count_inside = np.count_nonzero(is_inside_any)

        occupied_fraction = count_inside / n_points
        occupied_volume = self.cell_volume * occupied_fraction
        free_volume = self.cell_volume - occupied_volume

        return free_volume
