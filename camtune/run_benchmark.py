import os
import argparse
import yaml

from optimizer import build_benchmark, Benchmark
from tuner import Tuner
from camtune.utils.logger import load_logger

opt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimizer')
CONFIG_DIR = os.path.join(opt_dir, 'opt_configs')
RES_DIR = os.path.join(opt_dir, 'results')
if not os.path.exists(RES_DIR): os.mkdir(RES_DIR)

def main(config):
    benchmark_name: str = config['benchmark']
    benchmark_params: dict = config['benchmark_params']
    benchmark: Benchmark = build_benchmark(benchmark_name, benchmark_params)

    res_dir = os.path.join(RES_DIR, config['expr_name'])
    if not os.path.exists(res_dir): os.mkdir(res_dir)

    logger = load_logger(log_dir=res_dir, logger_name=config['expr_name'])
    tuner = Tuner(
        config['expr_name'], config, 
        benchmark=benchmark, 
        res_dir=res_dir, param_logger=logger)
    tuner.tune()

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config_name', '-c', default='random_ackley_20')
    args = args.parse_args()

    config_file_name = os.path.join(CONFIG_DIR, f'{args.config_name}.yaml')
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)