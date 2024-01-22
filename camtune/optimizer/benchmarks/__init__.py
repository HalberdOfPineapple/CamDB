from .base_benchmark import Benchmark
from .ackley import Ackley

def build_benchmark(benchmark_name: str, benchmark_params: dict) -> Benchmark:
    if benchmark_name.upper() == 'ACKLEY':
        return Ackley(**benchmark_params)
    else:
        raise ValueError(f"Undefined benchmark: {benchmark_name}")