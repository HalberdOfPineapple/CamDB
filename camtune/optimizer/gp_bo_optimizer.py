import torch 
import numpy as np
from typing import Callable, List, Dict, Any, Optional, Union, Tuple

from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, standardize, normalize
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.fit import fit_gpytorch_model, fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .base_optimizer import BaseOptimizer
from .optim_utils import ACQ_FUNC_MAP, generate_random_discrete, round_by_bounds
from camtune.utils import print_log, DTYPE, DEVICE

GP_ATTRS = {
    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
}
MIX_BETA = True

class GPBOOptimizer(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor,
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = [],
        in_mcts: bool = False,
        optimizer_params: Dict[str, Any] = None,
    ):
        super().__init__(bounds, obj_func,
                         batch_size=batch_size,
                         seed=seed, 
                         discrete_dims=discrete_dims, 
                         optimizer_params=optimizer_params)

        self.acq_func_cls = qExpectedImprovement if 'acquisition' not in self.optimizer_params \
                            else ACQ_FUNC_MAP[self.optimizer_params['acquisition']]
        self.num_calls = 0
        self.in_mcts = in_mcts

        for k, v in GP_ATTRS.items():
            if k not in self.optimizer_params:
                setattr(self, k, v)
            else:
                setattr(self, k, self.optimizer_params[k])
            
    def initial_sampling(self, num_init: int):
        sobol = SobolEngine(dimension=self.dimension, scramble=True, seed=self.seed)
        X_init = sobol.draw(n=num_init).to(dtype=DTYPE, device=DEVICE)

        X_init = unnormalize(X_init, self.bounds)
        X_init[:, self.discrete_dims] = round_by_bounds(X_init[:, self.discrete_dims], self.bounds[:, self.discrete_dims])

        Y_init = torch.tensor(
            [self.obj_func(x) for x in X_init], dtype=DTYPE, device=DEVICE,
        ).unsqueeze(-1)

        self.num_calls += len(X_init)
        return X_init, Y_init

    def generate_batch_continuous(self, batch_size: int, acqf: Callable) -> torch.Tensor:
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

        return X_next, acq_values

    def generate_batch_mixed(self, batch_size: int, acqf: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        # (raw_samples, discrete_dim)
        discrete_random_samples: torch.Tensor = generate_random_discrete(
            self.raw_samples, self.bounds, self.discrete_dims
        ) 

        fixed_feature_list = []
        for discrete_val_tensor in discrete_random_samples:
            fixed_feature = {
                feat_idx: discrete_val_tensor[i].item() for i, feat_idx in enumerate(self.discrete_dims)
            }
            fixed_feature_list.append(fixed_feature)

        X_next, acq_values = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=torch.stack([
                torch.zeros(self.dimension, dtype=self.dtype, device=self.device),
                torch.ones(self.dimension, dtype=self.dtype, device=self.device)
            ]),
            q=batch_size,
            fixed_features_list=fixed_feature_list,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return X_next, acq_values
    
    def normalize_model_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize inputs to [0, 1] for GP model. The discrete dimensions will also be normalized 
        in the beta phase when mixed space has not been supported.
        """
        if MIX_BETA:
            return normalize(X, self.bounds)
        else:
            X[:, self.continuous_dims] = normalize(X[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
            return X
    
    def unnormalize_model_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """Unnormalize inputs to original space for GP model. The discrete dimensions will also be 
        unnormalized in the beta phase when mixed space has not been supported.
        """
        if MIX_BETA:
            X_unnorm = unnormalize(X, self.bounds)
            X_unnorm[:, self.discrete_dims] = torch.round(X_unnorm[:, self.discrete_dims])
            return X_unnorm
        else:
            X[:, self.continuous_dims] = unnormalize(X[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
            return X
    
    def mcts_optimize(
            self,
            num_evals: int,
            mcts_params: Dict[str, Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mcts_params = {
        #     'num_init': self.local_num_init,
        #     'path': path,
        #     'X_in_region': leaf_node.sample_bag[0],
        #     'Y_in_region': leaf_node.sample_bag[1],
        # }
        from .mcts_utils import Node
        path: List[Node] = mcts_params['path']
        num_init: int = mcts_params['num_init']

        # Currently directly initialize from the leaf node
        X_sampled: torch.Tensor = mcts_params['X_in_region']
        Y_sampled: torch.Tensor = mcts_params['Y_in_region']
        
        

        raise NotImplementedError

    def optimize(
            self, 
            num_evals: int, 
            X_init: torch.Tensor, 
            Y_init: torch.Tensor,
            mcts_params: Dict[str, Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.seed)
        if self.in_mcts:
            return self.mcts_optimize(num_evals, mcts_params)

        X_sampled: torch.Tensor = torch.empty((0, self.dimension), dtype=self.dtype, device=self.device)
        Y_sampled: torch.Tensor = torch.empty((0, 1), dtype=self.dtype, device=self.device)
        while X_sampled.shape[0] < num_evals:
            batch_size = min(self.batch_size, num_evals - X_sampled.shape[0])

            train_X = torch.cat([X_sampled, X_init], dim=0)
            train_X = self.normalize_model_inputs(train_X)
            train_Y = standardize(torch.cat([Y_sampled, Y_init], dim=0))

            if len(self.discrete_dims) == 0 or MIX_BETA:
                model = SingleTaskGP(train_X, train_Y)
            else:
                model = MixedSingleTaskGP(train_X, train_Y, cat_dims=self.discrete_dims) 
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # currently assume qEI is used
            acqf = self.acq_func_cls(model, best_f=train_Y.max())
            if len(self.discrete_dims) == 0 or MIX_BETA:    
                X_next, acq_values = self.generate_batch_continuous(batch_size, acqf)
            else:
                X_next, acq_values = self.generate_batch_mixed(batch_size, acqf)
            
            X_next = self.unnormalize_model_inputs(X_next)
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
            print_log(log_msg, print_msg=True)
        
        result_X = torch.cat([X_init, X_sampled], dim=0)
        result_Y = torch.cat([Y_init, Y_sampled], dim=0)
        return result_X, result_Y
