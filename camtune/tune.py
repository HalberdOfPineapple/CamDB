import os
import yaml
import json
import torch
import argparse
import numpy as np
from time import perf_counter
from typing import List, Optional
from ConfigSpace import ConfigurationSpace, Configuration

from camtune.search_space import SearchSpace
from camtune.tuner import Tuner
from camtune.database import PostgresqlDB
from camtune.utils import (init_logger, get_logger, print_log, \
                           LOG_DIR, CONFIG_DIR, KNOB_DIR)

DEFAULT_EXPR_NAME = 'postgre_tpch_remote'
eval_counter = 0

def test_func(expr_name: str, expr_config: dict, db: PostgresqlDB, search_space: SearchSpace):
    perf_name: str = expr_config['database']['perf_name']
    perf_unit: str = expr_config['database']['perf_unit']

    configs, perfs = [], []
    for i in range(0, expr_config['tune']['num_evals']):
        print_log('-' * 50, print_msg=True)
        print_log(f'[Test Tune] Iteration {i} with configuration:', print_msg=True)
        configuration = search_space.input_space.sample_configuration()
        print_log(configuration, print_msg=True)

        eval_result = db.step(configuration)
        configs.append(configuration)
        perfs.append(eval_result[perf_name])
        
        print_log(f'[Test Tune] Iteration {i}:\t{eval_result[perf_name]:.3f} ({perf_unit})', print_msg=True)

    data_file_name = f"{expr_name}_data.log"
    with open(os.path.join(LOG_DIR, data_file_name), 'w') as f:
        result_dicts = {}
        for i, (config, perf) in enumerate(zip(configs, perfs)):
            result_dict = {perf_name: perf}
            result_dict['config'] = dict(config)
            result_dicts[i] = result_dict
        json.dump(result_dicts, f)
    
def tensor_to_config(
        sample_tensor: torch.Tensor, 
        search_space: SearchSpace
    ) -> Configuration:
        tensor_vals = sample_tensor.cpu().numpy()

        
        configspace: ConfigurationSpace = search_space.input_space
        hps: List = search_space.input_variables
        discrete_dims: List[int] = search_space.discrete_dims
        continuous_dims: List[int] = search_space.continuous_dims

        valid_config = {}
        for knob_idx in discrete_dims:
            num_val = int(tensor_vals[knob_idx])
            config_val = search_space.discrete_idx_to_value(knob_idx, num_val)
            valid_config[hps[knob_idx].name] = config_val
        
        for knob_idx in continuous_dims:
            config_val = float(tensor_vals[knob_idx])
            valid_config[hps[knob_idx].name] = config_val

        return Configuration(configspace, values=valid_config)

def main(expr_name: str, expr_config: dict, test_tuning: bool=False):
    # Logging configurations
    print_log("=" * 50, print_msg=True)
    print_log(f"Start Experiment: {expr_name}", print_msg=True)
    for k, v in expr_config.items():
        if 'pwd' in k or 'passwd' in k:
            continue

        if isinstance(v, dict):
            print_log(f"{k}:", print_msg=True)
            for kk, vv in v.items():
                print_log(f"\t{kk}: {vv}", print_msg=True)
        else:
            print_log(f"{k}: {v}", print_msg=True)
    print_log("=" * 50, print_msg=True)

    # -----------------------------------
    # Setup logger
    init_logger(expr_name=expr_name, log_dir=LOG_DIR)

    # -----------------------------------
    # Build Database Controller
    db = PostgresqlDB(expr_config)
    perf_name: str = expr_config['database']['perf_name']
    perf_unit: str = expr_config['database']['perf_unit']
    negate: bool = expr_config['database']['negate']

    # -----------------------------------
    # Setup search space from knob definition file
    knob_definition_path: str = os.path.join(
        KNOB_DIR, expr_config['database']['knob_definitions'])
    search_space = SearchSpace(
        knob_definition_path, 
        is_kv_config=True,
        seed=expr_config['tune']['seed'],
    )
    bounds: torch.Tensor = search_space.bounds
    discrete_dims: List[int] = search_space.discrete_dims
    # print(f"Bounds: {bounds}")
    # print(f"Discrete dims: {discrete_dims}")


    # -----------------------------------
    # Test tuning
    if test_tuning:
        test_func(expr_name, expr_config, db, search_space)
        return

    # -----------------------------------
    # Setup objective function
    # db.step() takes a Configuration object as input
    # while the Tuner will suggest tensors as outputs to be evaluated
    # => need to convert tensors to Configuration objects
    
    def obj_func(sample_tensor: torch.Tensor) -> float:
        global eval_counter

        config: Configuration = tensor_to_config(sample_tensor, search_space)

        print_log('=' * 80, print_msg=True)
        print_log(f'[Tune] [Iteration {eval_counter}]: using configuration: ', print_msg=True)
        for k, v in dict(config).items():
            print_log(f"\t{k}: {v}", print_msg=True)

        eval_result: dict = db.step(config)
        if eval_result['knob_applied'] == False:
            print_log(f"[Tune] [Iteration {eval_counter}]: Knob application failed.")
            return -np.inf if negate else np.inf

        perf = eval_result[perf_name]
        print_log(
            f'[Tune] [Iteration {eval_counter}]: {perf_name}: {perf:.3f} ({perf_unit})', print_msg=True)

        eval_counter += 1
        return -perf if negate else perf

    # -----------------------------------
    # Initialize tuner and start tuning
    # TODO: Interface to be decided later (now the Tuner requires a Benchmark instead of ConfigSpace)
    tuner = Tuner(
        expr_name = expr_name, 
        args = expr_config['tune'], 
        obj_func = obj_func,
        bounds = bounds, # (2, D)
        discrete_dims = discrete_dims,
    )

    #  --------------------------------------
    # Optimization
    start_time = perf_counter()

    result_X, result_Y = tuner.tune()

    elapsed_time = perf_counter() - start_time
    # --------------------------------------

    # Logging results
    best_config: Configuration = tensor_to_config(result_X[result_Y.argmax()], search_space)
    best_Y = result_Y.max().item()

    result_X: torch.Tensor = result_X.detach().cpu()
    result_Y: np.array = result_Y.detach().cpu().numpy()
    if negate:
        best_Y = -best_Y
        result_Y = -result_Y

    print_log("=" * 80, print_msg=True)
    print_log(f"Best {perf_name}: {best_Y}", print_msg=True)
    print_log(f"Best config:", print_msg=True)
    for k, v in dict(best_config).items():
        print_log(f"\t{k}: {v}", print_msg=True)
    print_log(f"Elapsed time: {elapsed_time:.2f} seconds", print_msg=True)

    data_file_name = f"{expr_name}_data.json"
    metric = f"{perf_name} ({expr_config['database']['perf_unit']})"
    with open(os.path.join(LOG_DIR, data_file_name), 'w') as f:
        result_dicts = {}
        for i, fx in enumerate(result_Y):
            x: torch.Tensor = result_X[i] # (D, )
            result_dict = {metric: float(fx[0])}
            result_dict['config'] = dict(tensor_to_config(x, search_space))
            result_dicts[i] = result_dict
        json.dump(result_dicts, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr_name', '-e', type=str, default=DEFAULT_EXPR_NAME, help='Experiment configuration file name')
    parser.add_argument('--test', '-t', action='store_true', help='Test tuning')
    args = parser.parse_args()

    expr_name: str = args.expr_name
    config_file_path = os.path.join(CONFIG_DIR, f'{expr_name}.yaml')
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.test:
        expr_name = 'test_tuning'
    main(expr_name, config, args.test)
