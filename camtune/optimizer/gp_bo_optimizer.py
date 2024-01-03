from typing import Callable
from ConfigSpace import ConfigurationSpace
from botorch.acquisition import AcquisitionFunction

from camtune.run_history import RunHistory
from camtune.search_space import SearchSpace

from .base_optimizer import BaseOptimizer
from .optim_utils import ACQ_FUNC_MAP

class GPBOOptimizer(BaseOptimizer):
    def __init__(
      self, 
      configspace: ConfigurationSpace, 
      obj_func: Callable,
      acq_func: str, 
      **kwargs,
    ):
        super(GPBOOptimizer).__init__(configspace, obj_func, **kwargs)
        self.acq_func = ACQ_FUNC_MAP[acq_func]
    
    def optimize(self, num_evals: int, run_history: RunHistory):
        pass