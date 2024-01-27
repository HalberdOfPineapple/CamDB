import torch
from abc import ABC, abstractmethod
from typing import Callable
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

dtype = torch.double
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

PROPERTY_SET = {
    'seed', 'Cp', 'leaf_size', 'node_selection_type', 'initial_sampling_method',
    'bounds', 'num_init', 'obj_func', 'optimizer_type', 'optimizer_params',
    'classifier_type', 'classifier_params',
}
class BaseBenchmark(ABC):
    @property
    def obj_func(self) -> Callable:
        return self._obj_func