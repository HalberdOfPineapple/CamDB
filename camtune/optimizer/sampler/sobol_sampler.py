import torch

from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

from .base_sampler import BaseSampler
from camtune.optimizer.optim_utils import generate_random_discrete

class SobolSampler(BaseSampler):
    def __init__(self, bounds: torch.Tensor, seed:int = 0, discrete_dims: list = None):
        super().__init__(bounds=bounds, seed=seed, discrete_dims=discrete_dims)
                 
    def generate(self, n_init: int) -> torch.Tensor:
        """
        TODO: add support for discrete dimensions
        
        Returns:
            X_init: (n_init, dim)
        """
        sobol = SobolEngine(dimension=self.bounds.shape[1], scramble=True, seed=self.seed)
        X_init = sobol.draw(n=n_init).to(dtype=self.dtype, device=self.device)
        
        return X_init
