import torch
from torch.quasirandom import SobolEngine

from ConfigSpace import ConfigurationSpace, Configuration
from .base_sampler import BaseSampler

class SobolSampler(BaseSampler):
    """
    Sobol sequence sampler.
    """

    def __init__(self, config_space: ConfigurationSpace, seed:int=0):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.

        lower_bounds : lower bounds in [0, 1] for continuous dimensions (optional)

        upper_bounds : upper bounds in [0, 1] for continuous dimensions (optional)

        seed : int (optional)
            Seed number for sobol sequence.
        """
        super().__init__(config_space, seed)

    def _generate(self, size:int):
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=self.seed)
        X = sobol.draw(n=size).numpy()
        return X

