import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional, Union, Tuple

class BaseOptimizer(ABC):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
    ):
        self.seed = seed
        self.obj_func = obj_func
        self.batch_size = batch_size

        self.bounds = bounds
        self.dtype, self.device = bounds.dtype, bounds.device
        self.dimension = self.bounds.shape[1]
        
        self.discrete_dims = discrete_dims
        self.continuous_dims = [i for i in range(self.dimension) if i not in discrete_dims]

        self.optimizer_params = optimizer_params if optimizer_params is not None else {}

    @abstractmethod
    def optimize(self, num_evals: int, X_init: torch.Tensor, Y_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def initial_sampling(self, num_init: int):
        raise NotImplementedError