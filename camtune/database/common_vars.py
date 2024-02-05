import os

BENCHMARK_DIR = '/home/viktor/Experiments/CamDB/camtune/benchmarks'
BENCHMARK = 'TPCH'
QUERY_PATH_MAP = {
    'TPCH': os.path.join(BENCHMARK_DIR, 'tpch'),
}