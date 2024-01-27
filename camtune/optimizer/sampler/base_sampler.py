import torch
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    def __init__(self, bounds: torch.Tensor, seed:int = 0, discrete_dims: list = None):
        self.bounds = bounds
        self.dimension = self.bounds.shape[1]
        
        self.dtype = bounds.dtype
        self.device = bounds.device

        self.seed = seed
        self.discrete_dims = discrete_dims
        self.continuous_dims = [i for i in range(self.bounds.shape[1]) if i not in self.discrete_dims]

    @abstractmethod
    def generate(self, n_init: int) -> torch.Tensor:
        raise NotImplementedError