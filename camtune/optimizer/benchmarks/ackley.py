
import torch
from botorch.test_functions import Ackley

from .base_benchmark import BaseBenchmark, dtype, device
from camtune.utils.logger import print_log

class AckleyBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print_log(f"[ackley] Using Ackley {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -5.)
        self.ub: float = kwargs.get('ub', 10.)
        self.bounds: torch.tensor = torch.tensor([[self.lb] * self.dim, [self.ub] * self.dim]).to(dtype=dtype, device=device)
        self._obj_func = Ackley(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=dtype, device=device)
        
        self.discrete_dims: list = None

class ExtendedAckleyBenchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        # Function settings
        self.dim: int = kwargs.get('dim', 20)
        print_log(f"[ackley] Using Ackley {self.dim}D")

        self.negate: bool = kwargs.get('negate', True)
        self.lb: float = kwargs.get('lb', -5.)
        self.ub: float = kwargs.get('ub', 10.)
        self.discrete_lb: int = kwargs.get('discrete_lb', 0)  # Lower bound for discrete variables
        self.discrete_ub: int = kwargs.get('discrete_ub', 5)  # Upper bound for discrete variables
        self.discrete_dims: list = [self.dim + i for i in range(4)]  #

        self.bounds: torch.tensor = torch.tensor(
            [[self.lb] * self.dim + [self.discrete_lb] * 4,
             [self.ub] * self.dim + [self.discrete_ub] * 4]
        ).to(dtype=dtype, device=device)

        self.ackley = Ackley(
            dim=self.dim, negate=self.negate,
            bounds=[(self.lb, self.ub) for _ in range(self.dim)]).to(dtype=dtype, device=device)
        
        def ext_obj_func(x: torch.Tensor) -> torch.Tensor:
            # Separate continuous and discrete variables
            if x.ndim == 1:
                x = torch.unsqueeze(x, 0)
            X_continuous = x[:, :self.dim]
            X_discrete = x[:, self.dim:]

            # Evaluate Ackley function for continuous variables
            ackley_val = self.ackley(X_continuous)
            discrete_product = (-1 if self.negate else 1) * torch.sum(X_discrete, dim=1, keepdim=True)

            return ackley_val + discrete_product

        self._obj_func = ext_obj_func