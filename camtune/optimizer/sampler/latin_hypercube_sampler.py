from skopt.sampler import Lhs

from ConfigSpace import ConfigurationSpace
from .base_sampler import BaseSampler


class LatinHypercubeSampler(BaseSampler):
    """
    Latin hypercube sampler.
    """

    def __init__(
      self, 
      config_space: ConfigurationSpace, 
      criterion:str = 'maximin',
      iterations:int = 10000,
      seed: int = 0,
    ):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.

        lower_bounds : lower bounds in [0, 1] for continuous dimensions (optional)
        upper_bounds : upper bounds in [0, 1] for continuous dimensions (optional)

        criterion : str or None, default='maximin'
            When set to None, the latin hypercube is not optimized

            - 'correlation' : optimized latin hypercube by minimizing the correlation
            - 'maximin' : optimized latin hypercube by maximizing the minimal pdist
            - 'ratio' : optimized latin hypercube by minimizing the ratio
              `max(pdist) / min(pdist)`

        iterations : int
            Define the number of iterations for optimizing latin hypercube.
        """
        super().__init__(config_space, seed)
        self.criterion = criterion
        self.iterations = iterations

    def _generate(self, size):
        lhs = Lhs(criterion=self.criterion, iterations=self.iterations)

        dims = [(0, 1)] * self.dims
        X = lhs.generate(dims, size, random_state=self.seed)
        return X
