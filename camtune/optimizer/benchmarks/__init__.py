from .base_benchmark import BaseBenchmark
from .ackley import AckleyBenchmark

BENCHMARK_MAP = {
    'ackley': AckleyBenchmark,
}