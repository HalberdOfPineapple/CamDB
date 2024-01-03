from typing import Callable
from ConfigSpace import ConfigurationSpace

from camtune.run_history import RunHistory
from camtune.search_space import SearchSpace
from .base_optimizer import BaseOptimizer

class RandomOptimizer(BaseOptimizer):
    def __init__(
      self, 
      configspace: ConfigurationSpace, 
      obj_func: Callable,
      **kwargs,
    ):
        super().__init__(configspace, obj_func, **kwargs)
    
    def optimize(self, num_evals: int, run_history: RunHistory):
        sample_configs = self.configspace.sample_configuration(size=num_evals)
        sample_config_perfs = [
            self.obj_func(sample_config) for sample_config in sample_configs
        ]

        run_history.add_records(sample_configs, sample_config_perfs)