import torch
import numpy as np
from typing import List

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement

ACQ_FUNC_MAP = {
    'ei': ExpectedImprovement,
    'qei': qExpectedImprovement,
}

def generate_random_discrete(num_evals: int, bounds: torch.Tensor, discrete_dims: List[int]) -> torch.Tensor:
    discrete_dim = len(discrete_dims)
    lower_bounds = bounds[0, discrete_dims]
    upper_bounds = bounds[1, discrete_dims]

    # Generate random samples within the unit hypercube [0,1]^D and then scale them to the bounds
    device, dtype = bounds.device, bounds.dtype
    random_samples = torch.rand(num_evals, discrete_dim, device=device, dtype=dtype)
    scaled_samples = lower_bounds + random_samples * (upper_bounds - lower_bounds)

    # Round the samples to the nearest integer
    rounded_samples = torch.round(scaled_samples)
    return rounded_samples