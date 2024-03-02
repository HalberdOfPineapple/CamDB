import os
import math
import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List

import botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")
ACQFS = {"ts", "ei"}

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 5 # float("nan")  # to be post-initialized
    success_counter: int = 0
    success_tolerance: int = 3 # 10  # paper's version: 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    # def __post_init__(self):
    #     self.failure_tolerance = math.ceil(
    #         max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
    #     )

def update_state(state: TurboState, Y_next: torch.Tensor):
    
    # Note that `tensor(bool)`` can directly be used for condition eval
    if max(Y_next) > state.best_value: 
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    
    state.best_value = max(state.best_value, max(Y_next).item())

    # "Whenever L falls below a given minimum threshold L_min, we discard 
    #  the respective TR and initialize a new one with side length L_init"
    if state.length < state.length_min:
        state.restart_triggered = True

    return state
