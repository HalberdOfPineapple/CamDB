from ConfigSpace import ConfigurationSpace
from .base_optimizer import BaseOptimizer
from .random_optimizer import RandomOptimizer
from .gp_bo_optimizer import GPBOOptimizer
from .sampler import BaseSampler, LatinHypercubeSampler, SobolSampler

def build_init_design(init_design: str, configspace: ConfigurationSpace, seed:int=0) -> BaseSampler:
    if init_design.upper() == "LHS":
        return LatinHypercubeSampler(configspace, seed=seed)
    elif init_design.upper() == "SOBOL":
        return SobolSampler(configspace, seed=seed)
    else:
        raise ValueError(f"Undefined initial design: {init_design}")

def build_optimizer(args: dict, **kwargs) -> BaseSampler:
    strategy = args['strategy']
    if strategy == "random":
        return RandomOptimizer(**kwargs)
    elif strategy == "gp-bo":
        return GPBOOptimizer(acq_func=args['acquisition'],**kwargs)
    else:
        raise ValueError(f"Undefined optimization strategy: {strategy}")