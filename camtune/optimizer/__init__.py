import torch
from typing import Callable, List, Dict, Any

from ConfigSpace import ConfigurationSpace
from .base_optimizer import BaseOptimizer
from .random_optimizer import RandomOptimizer
from .gp_bo_optimizer import GPBOOptimizer
from .turbo_optimizer import TuRBO
from .sampler import BaseSampler, SobolSampler, LHSSampler

INIT_DESIGN_MAP = {
    "LHS": LHSSampler,
    "SOBOL": SobolSampler,
}

OPTIMIZER_MAP = {
    "random": RandomOptimizer,
    "gp-bo": GPBOOptimizer,
    "turbo": TuRBO,
}

def build_init_design(
    init_design: str, bounds: torch.Tensor, seed:int=0, discrete_dims: List[int]=None
) -> BaseSampler:
    init_design = init_design.upper()
    if init_design in INIT_DESIGN_MAP:
        return INIT_DESIGN_MAP[init_design](bounds=bounds, seed=seed, discrete_dims=discrete_dims)
    else:
        raise ValueError(f"Undefined initial design: {init_design}")

def build_optimizer(
    optimizer_type: str,
    bounds: torch.Tensor,
    batch_size: int, 
    obj_func: Callable,
    seed:int=0, 
    discrete_dims: List[int]=[],
    optimizer_params: Dict[str, Any] = None,
) -> BaseSampler:
    optimizer_type = optimizer_type.lower()
    if optimizer_type in OPTIMIZER_MAP:
        return OPTIMIZER_MAP[optimizer_type](
            bounds=bounds, 
            obj_func=obj_func,
            batch_size=batch_size,
            seed=seed, 
            discrete_dims=discrete_dims,
            optimizer_params=optimizer_params,
        )
    else:
        raise ValueError(f"Undefined optimizer type: {optimizer_type}")