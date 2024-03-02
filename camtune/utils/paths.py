import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(BASE_DIR, 'config')

KNOB_DIR = os.path.join(BASE_DIR, 'knobs')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

OPTIM_CONFIG_DIR = os.path.join(BASE_DIR, 'optimizer', 'config')
OPTIM_LOG_DIR = os.path.join(BASE_DIR, 'optimizer', 'logs')

BENCHMARK_SET = {
    'TPCH',
    'SYSBENCH',
}
BENCHMARK_DIR = os.path.join(BASE_DIR, 'benchmarks')
QUERY_PATH_MAP = {
    'TPCH': os.path.join(BENCHMARK_DIR, 'tpch'),
}

paths = [CONFIG_DIR, LOG_DIR, OPTIM_CONFIG_DIR, OPTIM_LOG_DIR, BENCHMARK_DIR]
for path in paths:
    if not os.path.exists(path):
        # recursively create the path
        os.makedirs(path)
