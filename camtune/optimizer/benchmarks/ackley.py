
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