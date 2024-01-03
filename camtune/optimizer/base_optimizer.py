# TODO
from abc import ABC, abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace
from botorch.acquisition.acquisition import AcquisitionFunction
from typing import Callable

from camtune.run_history import RunHistory

class BaseOptimizer(ABC):
    def __init__(
      self, 
      configspace: ConfigurationSpace, 
      obj_func: Callable,
      **kwargs,
    ):
        self.configspace = configspace
        self.obj_func = obj_func
    
    @abstractmethod
    def optimize(self, num_evals: int, run_history: RunHistory):
        raise NotImplemented