import torch
from abc import ABC, abstractmethod
from botorch.test_functions.base import BaseTestProblem
from ConfigSpace import Configuration, ConfigurationSpace

class Benchmark(ABC):
    @property
    def func(self):
        return self._func

    @property
    def configspace(self) -> ConfigurationSpace:
        return self._configspace

    @abstractmethod
    def eval(self, config: Configuration):
        raise NotImplementedError
    
    
    def eval_tensor(self, cands: torch.Tensor) -> torch.Tensor:
        return self.func(cands)