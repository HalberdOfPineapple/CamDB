import os
import argparse
import torch
import yaml
from time import perf_counter
from typing import Callable

from tuner import Tuner
from camtune.optimizer.benchmarks import BaseBenchmark, BENCHMARK_MAP
from camtune.utils import (init_logger, get_logger, print_log, 
                           OPTIM_CONFIG_DIR, OPTIM_LOG_DIR)



def main(config):
    expr_name = config['expr_name']
    init_logger(expr_name, log_dir=OPTIM_LOG_DIR)
    logger = get_logger()

    # --------------------------------------
    # Print config
    print_log("-" * 50)
    for k, v in config.items():
        if not isinstance(v, dict):
            print_log(f"{k}: {v}", print_msg=True)
        else:
            print_log(f"{k}:", print_msg=True)
            for k2, v2 in v.items():
                print_log(f"    {k2}: {v2}", print_msg=True)
    print_log("-" * 50)

    # --------------------------------------
    # Setup benchmark
    benchmark_cls = BENCHMARK_MAP[config['benchmark']]
    benchmark: BaseBenchmark = benchmark_cls(**config['benchmark_params'])
    obj_func: Callable = benchmark.obj_func
    bounds: torch.Tensor = benchmark.bounds
    discrete_dims: list = benchmark.discrete_dims if benchmark.discrete_dims is not None else []

    # --------------------------------------
    # Setup tuner
    tuner = Tuner(
        expr_name=expr_name,
        args=config['tuner_params'],
        obj_func=obj_func,
        bounds=bounds,
        discrete_dims=discrete_dims,
    )

    #  --------------------------------------
    # Optimization
    start_time = perf_counter()

    result_X, result_Y = tuner.tune()

    elapsed_time = perf_counter() - start_time
    # --------------------------------------

    # --------------------------------------
    best_X = result_X[result_Y.argmax()].tolist()
    best_Y = result_Y.max().item()
    result_X, result_Y = result_X.detach().cpu().numpy(), result_Y.detach().cpu().numpy()
    
    if benchmark.negate:
        best_Y = -best_Y
        result_Y = -result_Y

    print_log("-" * 50, print_msg=True)
    print_log(f"Best X: {best_X}", print_msg=False)
    print_log(f"Best Y: {best_Y}", print_msg=True)
    print_log(f"Elapsed time: {elapsed_time:.2f} seconds", print_msg=True)

    data_file_name = f"{expr_name}_data.log"
    with open(os.path.join(OPTIM_LOG_DIR, data_file_name), 'w') as f:
        for x, fx in zip(result_X, result_Y):
            x = str(list(x))
            f.write(f"{fx}, {x}\n")

    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--expr_name', '-e', default='random_ackley_20')
    args = args.parse_args()

    config_file_name = os.path.join(OPTIM_CONFIG_DIR, f'{args.expr_name}.yaml')
    with open(config_file_name, 'r') as f:
        config = yaml.safe_load(f)
    config['expr_name'] = args.expr_name

    main(config)