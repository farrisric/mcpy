from random import Random


class RandomNumberGenerator:
    """Class for generating random numbers with optional seeding."""

    def __init__(self, seed=None, warm_up=1000000) -> None:
        """
        Initialize the random number generator.

        Parameters:
        seed (int, optional): Seed for reproducibility. Default is None.
        """
        self.random = Random(seed)

        if warm_up:
            for _ in range(warm_up):
                self.get_uniform()
                self.get_gaussian()

    def get_uniform(self, a=0.0, b=1.0) -> float:
        """
        Generate a random number uniformly distributed between a and b.

        Parameters:
        a (float): Lower bound.
        b (float): Upper bound.

        Returns:
        float: Random number.
        """
        return self.random.uniform(a, b)

    def get_gaussian(self, mu=0.0, sigma=1.0) -> float:
        """
        Generate a random number with a Gaussian distribution.

        Parameters:
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.

        Returns:
        float: Random number.
        """
        return self.random.gauss(mu, sigma)
