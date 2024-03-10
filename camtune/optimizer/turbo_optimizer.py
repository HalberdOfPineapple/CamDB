import os
import math
import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union, List, Dict, Any

import botorch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize, standardize, normalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from camtune.utils import print_log, DTYPE, DEVICE
from .base_optimizer import BaseOptimizer
from .optim_utils import generate_random_discrete, round_by_bounds
from .turbo_utils import *
from .mcts_utils import Node


TURBO_ATTRS = {
    'num_restarts': 10,
    'raw_samples': 512,
    'n_candidates': 5000,
    'max_cholesky_size': float("inf"),
    'acqf': "ts",
    "init_bounding_box_length": 0.0005,
}
MIXED_BETA = True

class TuRBO(BaseOptimizer):
    def __init__(
        self, 
        bounds: torch.Tensor,
        obj_func: Callable, 
        batch_size: int = 1,
        seed: int=0, 
        discrete_dims: List[int] = [],
        in_mcts: bool = False,
        optimizer_params: Dict[str, Any] = None,
    ):
        super().__init__(bounds, obj_func, 
                         batch_size=batch_size,
                         seed=seed, 
                         discrete_dims=discrete_dims, 
                         optimizer_params=optimizer_params)
        self.seed = seed
        self.in_mcts: bool = in_mcts
        self.obj_func = obj_func

        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.batch_size = batch_size

        self.num_calls = 0
        self.n_init: int = 2 * self.dimension if 'n_init' not in optimizer_params else optimizer_params['n_init']
        self.X = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
        self.Y = torch.empty((0, 1), dtype=DTYPE, device=DEVICE)
        
        for k, v in TURBO_ATTRS.items():
            if k not in optimizer_params:
                setattr(self, k, v)
            else:
                setattr(self, k, optimizer_params[k])

        if self.acqf not in ACQFS:
            raise ValueError(f"Acquisition function {self.acqf} not supported")
    
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
    
    def generate_samples_in_region(
            self, 
            num_samples: int, 
            path: List[Node], 
            region_center: torch.Tensor # input is unnormalized
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        region_center_norm: torch.Tensor = normalize(region_center, self.bounds)

        X_init: torch.Tensor = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
        bounding_box_length = self.init_bounding_box_length
        weights = torch.ones(self.dimension, dtype=DTYPE, device=DEVICE)
    
        # sobol_samples - (sobol_num_samples, dim)
        sobol_num_samples = 2 * num_samples
        sobol = SobolEngine(self.dimension, scramble=True, seed=self.seed)
        sobol_samples = sobol.draw(sobol_num_samples).to(dtype=DTYPE, device=DEVICE)

        while X_init.shape[0] < num_samples and bounding_box_length < 1.0:
            # bounding_box_lbs, bounding_box_ubs - (1, dim)
            bouning_box_lbs = torch.clamp(region_center_norm - bounding_box_length / 2 * weights, 0.0, 1.0)
            bouning_box_ubs = torch.clamp(region_center_norm + bounding_box_length / 2 * weights, 0.0, 1.0)
            sobol_cands = sobol_samples * (bouning_box_ubs - bouning_box_lbs) + bouning_box_lbs

            sobol_cands = unnormalize(sobol_cands, self.bounds)
            sobol_cands[:, self.discrete_dims] = round_by_bounds(sobol_cands[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
            in_region = Node.path_filter(path, sobol_cands) # (num_in_region_samples, )

            X_init = torch.cat((X_init, sobol_cands[in_region]), dim=0)
            if X_init.shape[0] < num_samples:
                bounding_box_length *= 2
    
        if X_init.shape[0] > num_samples:
            X_init = X_init[:num_samples]
        elif X_init.shape[0] < num_samples:
            # if not enough samples are generated within the path-defined region, generate the rest randomly
            num_rand_samples = num_samples - X_init.shape[0]

            rand_samples = sobol.draw(num_rand_samples).to(dtype=DTYPE, device=DEVICE)
            rand_samples = unnormalize(rand_samples, self.bounds)
            rand_samples[:, self.discrete_dims] = round_by_bounds(rand_samples[:, self.discrete_dims], self.bounds[:, self.discrete_dims])

            X_init = torch.cat((X_init, rand_samples), dim=0)
        
        Y_init = torch.tensor(
            [self.obj_func(x) for x in X_init], dtype=DTYPE, device=DEVICE,
        ).unsqueeze(-1)

        self.num_calls += len(X_init)
        return X_init, Y_init

    def generate_batch(self, 
        state: TurboState,
        model: botorch.models.model.Model, 
        X: torch.Tensor, # train_X - normalized, unit scale
        Y: torch.Tensor, # train_Y - standardized (unit scale)
        batch_size: int,
    ) -> torch.Tensor:
        assert X[:, self.continuous_dims].min() >= 0.0 and X[:, self.continuous_dims].max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        dim: int = self.dimension
        x_center: torch.Tensor = X[Y.argmax(), :].clone()

        # Length scales for all dimensions
        # weights - (dim, )
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Clamps all elements in input into the range [min, max]
        # tr_lbs, tr_ubs - (1, dim)
        tr_lbs = torch.clamp(x_center - state.length / 2 * weights, 0.0, 1.0)
        tr_ubs = torch.clamp(x_center + state.length / 2 * weights, 0.0, 1.0)

        if self.acqf == "ts":
            sobol = SobolEngine(dim, scramble=True, seed=self.seed)

            # pert - (n_candidates, dim)
            pert = sobol.draw(self.n_candidates).to(dtype=DTYPE, device=DEVICE)
            pert = tr_lbs + (tr_ubs - tr_lbs) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = torch.rand(self.n_candidates, dim, dtype=DTYPE, device=DEVICE) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=DEVICE)] = 1

            # Create candidate points from the perturbations and the mask
            X_cands = x_center.expand(self.n_candidates, dim).clone()
            X_cands[mask] = pert[mask]

            if not MIXED_BETA:
                X_cands[:, self.discrete_dims] = unnormalize(X_cands[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
                X_cands = torch.round(X_cands[:, self.discrete_dims])

            # Sample on the candidate set 
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cands, num_samples=batch_size)
        elif self.acqf == "ei":
            ei = qExpectedImprovement(model=model, best_f=Y.max())

            if len(self.discrete_dims) == 0 or MIXED_BETA:
                X_next, acq_value = optimize_acqf(
                    ei,
                    bounds=torch.stack([tr_lbs, tr_ubs]),
                    q=batch_size,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )
            else:
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

                X_next, acq_value = optimize_acqf_mixed(
                    ei,
                    bounds=torch.stack([tr_lbs, tr_ubs]),
                    q=batch_size,
                    fixed_features_list=fixed_feature_list,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )

        return X_next

    def optimize(self, num_evals: int, X_init: torch.Tensor, Y_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TuRBO is does not support external initiaization")

    def optimize(self, num_evals: int, mcts_params: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.seed)

        restart_counter = 0
        while self.num_calls < num_evals:
            print_log('-' * 80, print_msg=True)
            print_log(f"[TuRBO] Restart {restart_counter}:", print_msg=True)

            num_init = min(self.n_init, num_evals - self.num_calls)
            if not self.in_mcts:
                X_sampled, Y_sampled = self.initial_sampling(num_init)
            else:
                # Note X_in_region and Y_in_region are in original scale (unnormalized)
                X_in_region: torch.Tensor = mcts_params['X_in_region']
                Y_in_region: torch.Tensor = mcts_params['Y_in_region']
                path: List[int] = mcts_params['path']

                region_center: torch.Tensor = X_in_region[torch.argmax(Y_in_region), :].clone()
                X_sampled, Y_sampled = self.generate_samples_in_region(
                    num_samples=num_init,
                    path=path,
                    region_center=region_center,
                )
             
            if self.num_calls >= num_evals: 
                self.X = torch.cat((self.X, X_sampled), dim=0)
                self.Y = torch.cat((self.Y, Y_sampled), dim=0)
                break

            print_log(f"[TuRBO] {'(MCTS) ' if self.in_mcts else ''}Start local modeling with {num_init} data points", print_msg=True)
            state = TurboState(dim=self.dimension, batch_size=self.batch_size)

            while not state.restart_triggered and self.num_calls < num_evals: # Run until TuRBO converges
                train_X = torch.cat([X_sampled, self.X], dim=0)

                if MIXED_BETA:
                    # Normalize all dimensions in BETA phase
                    train_X = normalize(train_X, self.bounds)
                else:
                    train_X[:, self.continuous_dims] = normalize(
                        train_X[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
                train_Y = standardize(torch.cat([Y_sampled, self.Y], dim=0))
    
                # Define the model (Posterior)
                if len(self.discrete_dims) == 0 or MIXED_BETA:
                    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                    covar_module = ScaleKernel(
                        MaternKernel(
                            nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=Interval(0.005, 4.0),
                        ),
                    )
                    model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, likelihood=likelihood)
                else:
                    model = MixedSingleTaskGP(train_X, train_Y, cat_dims=self.discrete_dims) 
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    fit_gpytorch_mll(mll)
                    batch_size = min(self.batch_size, num_evals - self.num_calls)
                    X_next = self.generate_batch(
                        state=state, model=model,
                        X=train_X, Y=train_Y,
                        batch_size=batch_size,
                    )
                
                if MIXED_BETA:
                    X_next = unnormalize(X_next, self.bounds)
                    X_next[:, self.discrete_dims] = round_by_bounds(X_next[:, self.discrete_dims], self.bounds[:, self.discrete_dims])
                else:
                    X_next[:, self.continuous_dims] = unnormalize(X_next[:, self.continuous_dims], self.bounds[:, self.continuous_dims])
                Y_next = torch.tensor(
                    [self.obj_func(x) for x in X_next], dtype=DTYPE, device=DEVICE,
                ).unsqueeze(-1)

                state = update_state(state, Y_next)
                X_sampled = torch.cat((X_sampled, X_next), dim=0)
                Y_sampled = torch.cat((Y_sampled, Y_next), dim=0)
                self.num_calls += len(X_next)

                print_log(
                    f"[TuRBO] [Restart {restart_counter}] {len(self.X) + len(X_sampled)}) "
                    f"Best value: {state.best_value:.2e} | TR length: {state.length:.2e} | "
                    f"num. restarts: {state.failure_counter}/{state.failure_tolerance} | "
                    f"num. successes: {state.success_counter}/{state.success_tolerance}", 
                    print_msg=True)

            self.X = torch.cat((self.X, X_sampled), dim=0)
            self.Y = torch.cat((self.Y, Y_sampled), dim=0)
            restart_counter += 1

            # If TuRBO is used in MCTS, only one restart is allowed
            if self.in_mcts: break 

        return self.X, self.Y
    