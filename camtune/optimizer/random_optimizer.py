import torch
from typing import Callable, List, Dict, Any, Optional, Union, Tuple
from botorch.utils.transforms import unnormalize

from .base_optimizer import BaseOptimizer
from camtune.optimizer.optim_utils import generate_random_discrete

class RandomOptimizer(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        seed: int = 0,
        discrete_dims: List[int] = None,
        optimizer_params: Dict[str, Any] = None,
    ):
        super().__init__(bounds, obj_func, seed, discrete_dims, optimizer_params)

        if self.optimizer_params is not None and self.optimizer_params['method'] == 'LHS':
            self.use_lhs = True
            from camtune.optimizer.sampler import LHSSampler
            self.sampler = LHSSampler(bounds, seed, discrete_dims)
        else:
            self.use_lhs = False

    def optimize(self, num_evals: int, X_init: torch.Tensor, Y_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_evals: number of evaluations
            X_init: (num_init, dim)
            Y_init: (num_init, 1)
        """ 
        # X_sampled = torch.randn(num_evals, self.dimension, device=self.device, dtype=self.dtype)
        torch.manual_seed(self.seed)

        if self.use_lhs:
            X_sampled = self.sampler.generate(num_evals)
            Y_sampled = torch.tensor(
                [self.obj_func(x) for x in X_sampled], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)
        else:
            X_sampled = torch.empty(num_evals, self.dimension, device=self.device, dtype=self.dtype)

            X_sampled[:, self.continuous_dims] = torch.rand(num_evals, len(self.continuous_dims), device=self.device, dtype=self.dtype)
            X_sampled[:, self.continuous_dims] = unnormalize(X_sampled[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
            X_sampled[:, self.discrete_dims] = generate_random_discrete(num_evals, self.bounds, self.discrete_dims)

            Y_sampled = torch.tensor(
                [self.obj_func(x) for x in X_sampled], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)

        X = torch.cat([X_init, X_sampled], dim=0)
        Y = torch.cat([Y_init, Y_sampled], dim=0)
        return X, Y