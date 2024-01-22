import torch 
import numpy as np
from typing import Callable

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import InputStandardize, Normalize, ChainedInputTransform
from botorch.models.transforms.outcome import Standardize
from ConfigSpace import ConfigurationSpace

from camtune.run_history import RunHistory
from .base_optimizer import BaseOptimizer
from .optim_utils import ACQ_FUNC_MAP, convert_configurations_to_array, convert_to_valid_configs
from .benchmarks import Benchmark

class GPBOOptimizer(BaseOptimizer):
    def __init__(
      self, 
      configspace: ConfigurationSpace, 
      benchmark: Benchmark,
      acq_func: str, 
      is_minimizing: bool=True,
      batch_size: int = 1,
      param_logger=None,
      **kwargs,
    ):
        super().__init__(configspace, benchmark, param_logger=param_logger, **kwargs)
        # self.acq_func = ACQ_FUNC_MAP[acq_func]
        self.acq_func = UpperConfidenceBound
        self.is_minimizing = is_minimizing
        self.batch_size = batch_size


        self.input_transform = ChainedInputTransform(
            standardize=InputStandardize(d=self.bounds.shape[1]),
            normalize=Normalize(d=self.bounds.shape[1]),
        )
        self.output_transform = Standardize(m=1)
    
    def get_init_samples(self, num_init: int):
        train_x = torch.rand(num_init, len(self.configspace.get_hyperparameters()), dtype=torch.double)
        train_y = self.benchmark.eval_tensor(train_x).unsqueeze(-1)

        best_observed_value = train_y.max().item()
        return train_x, train_y, best_observed_value
    
    def optimize(self, num_evals: int, num_init: int, run_history: RunHistory):
        # hist_configs = [config for (config, _) in run_history.records]
        # hist_perfs = np.array([perf for (_, perf) in run_history.records])
        # num_init = len(hist_configs)

        # # X_hist: N x D, Y_hist: N x 1
        # X_hist = torch.from_numpy(convert_configurations_to_array(hist_configs)).double()
        # Y_hist = torch.from_numpy(hist_perfs).double().unsqueeze(-1)
        # TODO
        num_iters = 0
        while num_iters < num_evals:
            batch_size = min(self.batch_size, num_evals - num_iters)

            # Fit the GP model
            GP = SingleTaskGP(X_hist, Y_hist, 
                              input_transform=self.input_transform,
                              outcome_transform=self.output_transform)
            mll = ExactMarginalLogLikelihood(GP.likelihood, GP)
            fit_gpytorch_model(mll)

            # Optimize the acquisition function
            # cands: (batch_size, D)
            # acq_func = self.acq_func(
            #     model=GP, 
            #     best_f=run_history.best_perf, 
            #     maximize=not self.is_minimizing)
            acq_func = self.acq_func(
                model=GP, 
                beta=0.2,
                maximize=not self.is_minimizing)
            cands, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=batch_size,
                num_restarts=5,
                raw_samples=20,  # Number of samples for initialization (TODO)
            )
            perfs = self.benchmark.eval_tensor(cands).unsqueeze(-1) # (batch_size, 1)

            torch.cat([X_hist, cands], dim=0)
            torch.cat([Y_hist, perfs], dim=0)

            num_iters += batch_size

            configs = convert_to_valid_configs(self.configspace, cands.numpy())
            perfs = np.array(perfs)
            run_history.add_records(configs, perfs)

            if self.logger:
                self.logger.info(f"Sample {num_init + num_iters}: best perf = {np.max(perfs)}")

        
