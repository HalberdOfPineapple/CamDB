import torch

from botorch.utils.transforms import unnormalize
from pyDOE import lhs

from .base_sampler import BaseSampler
from camtune.optimizer.optim_utils import generate_random_discrete


class LHSSampler(BaseSampler):
    def __init__(self, bounds: torch.Tensor, seed:int = 0, discrete_dims: list = None):
        super().__init__(bounds=bounds, seed=seed, discrete_dims=discrete_dims)
    
    def generate(self, n_init: int) -> torch.Tensor:
        """
        Generate initial design using Latin Hypercube Sampling.

        Returns:
            X_init: (n_init, dim)
        """
        # https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube-lhs
        lhs_arr = lhs(self.dimension, samples=n_init) # (n_init, dim)
        X_init = torch.tensor(lhs_arr, dtype=self.dtype, device=self.device)
        
        X_init[:, self.continuous_dims] = unnormalize(
            X_init[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
        X_init[:, self.discrete_dims] = torch.round(
            unnormalize(X_init[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
        )

        return X_init