import torch
from typing import Callable
from ConfigSpace import ConfigurationSpace
from botorch.utils.transforms import unnormalize
from typing import List, Callable, Optional, Union, Tuple


from camtune.utils.logger import print_log
from camtune.optimizer import build_optimizer, build_init_design, BaseOptimizer

# tuner_params:
#   seed: 1024
#   num_evals: 500
#   batch_size: 1
  
#   init_design: sobol
#   init_by_dim: True
#   num_init: 20

#   optimizer: gp-bo
#   optimizer_params:
#     acquisition: 'ei'

class Tuner:
    def __init__(self, 
      expr_name: str, 
      args: dict, 
      obj_func: Callable,
      bounds: torch.Tensor, # (2, D)
      discrete_dims: Optional[List[int]] = [],
    ):
        """
        Args:
            expr_name: experiment name
            args: dict of tuner parameters
            obj_func: objective function to be optimized. Note that the function takes tensor as the input 
            bounds: (2, D)
            discrete_dims: list of indices of discrete dimensions
        """
        self.expr_name = expr_name
        self.args = args

        # Directly assume obj_func is to be maximized. 
        # If for minimization, negate it before passing it in.
        self.obj_func = obj_func

        self.seed = self.args['seed']
        self.bounds = bounds
        self.dtype, self.device = bounds.dtype, bounds.device
        self.dimension = self.bounds.shape[1]

        self.num_evals= self.args['num_evals']
        self.num_init = self.args['num_init'] if not self.args['init_by_dim'] else self.dimension * 2

        self.discrete_dims = discrete_dims

        if 'perf_name' in self.args:
            self.perf_name, self.perf_unit = self.args['perf_name'], self.args['perf_unit']
        else:
            self.perf_name, self.perf_unit = 'func_val', 'null'

        self.init_sampler = build_init_design(
            self.args['init_design'], 
            self.bounds,
            seed=self.seed,
            discrete_dims=discrete_dims,
        )

        optimizer_params = None if 'optimizer_params' not in self.args else self.args['optimizer_params']
        self.optimizer: BaseOptimizer = build_optimizer(
            self.args['optimizer'],
            bounds=self.bounds,
            obj_func=self.obj_func,
            seed=self.seed,
            discrete_dims=discrete_dims,
            optimizer_params=optimizer_params,
        )

    def tune(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            result_X: (num_evals, dim)
            result_Y: (num_evals, 1)
        """
        if self.num_evals < self.num_init:
            raise ValueError(
                f'num_evals ({self.num_evals}) must be greater than num_init ({self.num_init})')

        if self.num_init != 0 and self.args['optimizer'] != 'MCTS':
            print_log(f'[Tuner] Start initial profiling by {self.args["init_design"]}', print_msg=True)
            X_init: torch.Tensor = self.init_sampler.generate(self.num_init)
            Y_init: torch.Tensor = torch.tensor(
                    [self.obj_func(x) for x in X_init], dtype=self.dtype, device=self.device,
                ).unsqueeze(-1)
        else:
            X_init: torch.Tensor = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
            Y_init: torch.Tensor = torch.empty((0, 1), dtype=self.dtype, device=self.device)

        num_evals = self.num_evals - self.num_init
        print_log(f'[Tuner] Start optimization using {self.args["optimizer"]} for {self.num_evals} iterations', print_msg=True)
        result_X, result_Y = self.optimizer.optimize(num_evals, X_init, Y_init)

        return result_X, result_Y
    


        