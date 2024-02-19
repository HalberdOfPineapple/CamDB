from .base_benchmark import BaseBenchmark
from .ackley import AckleyBenchmark, ExtendedAckleyBenchmark

BENCHMARK_MAP = {
    'ackley': AckleyBenchmark,
    'ackley_ext': ExtendedAckleyBenchmark,
}