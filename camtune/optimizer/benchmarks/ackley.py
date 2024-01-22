import torch
from .base_benchmark import Benchmark

from botorch.test_functions.synthetic import Ackley as BoTAckley
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from typing import Optional, List, Tuple

DEFAULT_LB = -32.768
DEFAULT_UB = 32.768

class Ackley(Benchmark):
    def __init__(self, 
      dim:int=2, 
      noise_std:bool=None, 
      negate:bool=False, 
      bounds:Optional[List[Tuple[float, float]]]=None,
    ):
        self.dim = dim
        self._func = BoTAckley(
            dim=dim, noise_std=noise_std, negate=negate, bounds=bounds)

        config_space = ConfigurationSpace()
        for i in range(dim):  # 2-dimensional Ackley function
            if bounds:
                lb, ub = bounds[i][0], bounds[i][1]
            else:
                lb, ub = DEFAULT_LB, DEFAULT_UB

            config_space.add_hyperparameter(UniformFloatHyperparameter(f"x{i}", lb, ub))
        self._configspace = config_space
    
    def eval(self, config: Configuration):
        self._configspace.check_configuration(config)
        x = torch.tensor([config[f"x{i}"] for i in range(self.dim)], dtype=torch.float)

        return self.func(x).item()
    
    def eval_tensor(self, cands: torch.Tensor) -> torch.Tensor:
        return self.func(cands)