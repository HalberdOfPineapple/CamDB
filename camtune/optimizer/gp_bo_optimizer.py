import torch 
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Union, Tuple

from botorch.utils.transforms import unnormalize, standardize, normalize

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.fit import fit_gpytorch_model, fit_gpytorch_mll

from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .base_optimizer import BaseOptimizer
from .optim_utils import ACQ_FUNC_MAP
from camtune.utils import print_log

GP_ATTRS = {
    'batch_size': 1,
    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
}

class GPBOOptimizer(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        seed: int = 0,
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
    ):
        super().__init__(bounds, obj_func, seed, discrete_dims, optimizer_params)

        self.acq_func_cls = qExpectedImprovement if 'acquisition' not in self.optimizer_params \
                            else ACQ_FUNC_MAP[self.optimizer_params['acquisition']]
        self.num_calls = 0

        for k, v in GP_ATTRS.items():
            if k not in self.optimizer_params:
                setattr(self, k, v)
            else:
                setattr(self, k, self.optimizer_params[k])


    
    def optimize(self, num_evals: int, X_init: torch.Tensor, Y_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.seed)

        X_sampled: torch.Tensor = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        Y_sampled: torch.Tensor = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        while X_sampled.shape[0] < num_evals:
            batch_size = min(self.batch_size, num_evals - X_sampled.shape[0])

            train_X = torch.cat([X_sampled, X_init], dim=0)
            train_X[:, self.continuous_dims] = normalize(train_X[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
            train_Y = standardize(torch.cat([Y_sampled, Y_init], dim=0))

            if len(self.discrete_dims) == 0:
                model = SingleTaskGP(train_X, train_Y)
            else:
                model = MixedSingleTaskGP(train_X, train_Y, cat_dims=self.discrete_dims) 
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)


            # currently assume qEI is used
            acqf = self.acq_func_cls(model, best_f=train_Y.max())
            if len(self.discrete_dims) == 0:    
                X_next, acq_values = optimize_acqf(
                    acq_function=acqf,
                    bounds=torch.stack([
                        torch.zeros(self.dimension, dtype=self.dtype, device=self.device),
                        torch.ones(self.dimension, dtype=self.dtype, device=self.device),
                    ]),
                    q=batch_size,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
            else:
                bounds = torch.stack([
                    torch.zeros(self.dimension, dtype=self.dtype, device=self.device),
                    torch.ones(self.dimension, dtype=self.dtype, device=self.device),
                ])
                bounds[:, self.discrete_dims] = self.bounds[:, self.discrete_dims]
                X_next, acq_values = optimize_acqf_mixed(
                    acq_function=acqf,
                    bounds=bounds,
                    q=batch_size,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
            
            
            X_next[:, self.continuous_dims] = unnormalize(X_next[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
            Y_next = torch.tensor(
                [self.obj_func(x) for x in X_next], dtype=self.dtype, device=self.device,
            ).unsqueeze(-1)

            X_sampled = torch.cat([X_sampled, X_next], dim=0)
            Y_sampled = torch.cat([Y_sampled, Y_next], dim=0)
            self.num_calls += batch_size

            log_msg = (
                f"Sample {len(X_sampled) + len(X_init)} | "
                f"Best value: {Y_sampled.max().item():.2f} |"
            )
            print_log(log_msg)
        
        result_X = torch.cat([X_init, X_sampled], dim=0)
        result_Y = torch.cat([Y_init, Y_sampled], dim=0)
        return result_X, result_Y
