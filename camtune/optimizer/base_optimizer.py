import torch
import numpy as np

from abc import ABC, abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace
from botorch.acquisition.acquisition import AcquisitionFunction
from typing import Callable

from .benchmarks import Benchmark
from .optim_utils import get_bounds_from_configspace
from camtune.run_history import RunHistory


class BaseOptimizer(ABC):
    def __init__(
      self, 
      configspace: ConfigurationSpace, 
      benchmark: Benchmark,
      param_logger=None,
      **kwargs,
    ):
        self.configspace = configspace
        self.benchmark = benchmark
        self.logger = param_logger

        # self.bounds have shape 2 x D
        bounds = get_bounds_from_configspace(configspace)
        self.bounds = \
          torch.tensor(np.array([[bound[0], bound[1]] for bound in bounds]), dtype=torch.float32).T
    
    @abstractmethod
    def get_init_samples(self, num_init: int):
        raise NotImplemented

    @abstractmethod
    def optimize(self, num_evals: int, num_init, run_history: RunHistory):
        raise NotImplemented