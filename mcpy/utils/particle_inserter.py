import numpy as np
from . import RandomNumberGenerator

class PositionGenerator:
    def __init__(self, 
                 seed : int,
                 insertion_type : str = 'box',
                 radius : float = None,
                 center : list = None,
                 operating_box : list[list] = None,
                 z_shift : float = None):
        """
        Initialize the ParticleInserter class.

        Parameters:
        - center: List or array-like, the center of the sphere or box.
        - radius_or_dimensions: If insertion_type is 'spherical', this is the radius of the sphere.
                               If insertion_type is 'box', this is the dimensions of the box [dx, dy, dz].
        - insertion_type: String, either 'spherical' or 'box' to determine the insertion method.
        """
        self.insertion_type = insertion_type
        self.rng = RandomNumberGenerator(seed=seed)

        if self.insertion_type == 'spherical':
            self.radius = radius
            self.center = np.array(center)
            if self.radius == None:
                raise ValueError("if insertion_type is 'spherical', radius cannot be None")
            if self.center.all() == None:
                raise ValueError("if insertion_type is 'spherical', center cannot be None")
            self.gen = self.spherical_insertion
        elif self.insertion_type == 'box':
            self.box = operating_box
            if self.box == None:
                raise ValueError("if insertion_type is 'box', you need to set an operating_box")
            self.z_shift = z_shift
            self.gen = self.box_insertion
        else:
            raise ValueError("insertion_type must be either 'spherical' or 'box'")

    def spherical_insertion(self):
        """Generate a random point within a sphere."""
        rad = self.radius*self.rng.get_uniform()
        theta = np.arccos(2*self.rng.get_uniform() -1)
        phi = 2*np.pi*self.rng.get_uniform()    
        position = np.array([
                    rad*np.sin(theta)*np.cos(phi),
                    rad*np.sin(theta)*np.sin(phi),
                    rad*np.cos(theta)
                ])
        return position + self.center

    def box_insertion(self):
        """Generate a random point within a box."""
        position = np.array([
            self.box[i]*self.rng.get_uniform() for i in range(3)
            ]).sum(axis=0)
        if self.z_shift:
            position[2] += self.z_shift

        return position