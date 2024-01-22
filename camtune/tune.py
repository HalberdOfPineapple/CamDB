import os
import argparse
import yaml

from camtune.utils.logger import init_logger, get_logger, print_log
from camtune.search_space import SearchSpace
from camtune.tuner import Tuner
from camtune.database import PostgresqlDB

current_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(current_dir, 'config')
LOG_DIR = os.path.join(current_dir, 'logs')
RES_DIR = os.path.join(current_dir, 'results')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
if not os.path.exists(RES_DIR): os.mkdir(RES_DIR)

DEFAULT_CONFIG_NAME = 'postgre_tpch_remote'

def main(expr_config: dict):
    # -----------------------------------
    # Setup logger
    init_logger(log_dir=LOG_DIR, logger_name=expr_config['expr_name'])
    # -----------------------------------
    # Build Database Controller
    db = PostgresqlDB(expr_config)

    # -----------------------------------
    # Setup search space from knob definition file
    search_space = SearchSpace(
        expr_config['tune']['knob_definitions'], 
        is_kv_config=True,
        seed=expr_config['tune']['seed'],
    )

    # -----------------------------------
    # Initialize tuner and start tuning
    # TODO: Interface to be decided later (now the Tuner requires a Benchmark instead of ConfigSpace)
    # tuner = Tuner(
    #     expr_name=expr_config['expr_name'], 
    #     args=expr_config['tune'], 
    #     obj_func=db.step, 
    #     configspace=search_space.input_space, 
    #     res_dir=RES_DIR, 
    #     param_logger=logger)
    # tuner.tune()

    for i in range(0, expr_config['tune']['computation_budget']):
        print_log('-' * 50, print_msg=True)
        print_log(f'[Test Tune] Iteration {i} with configuration:', print_msg=True)

        configuration = search_space.input_space.sample_configuration()
        print_log(configuration, print_msg=True)

        eval_result = db.step(configuration)

        print_log(f'[Test Tune] Iteration {i}:\t{eval_result:.3f}', print_msg=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', type=str, default=DEFAULT_CONFIG_NAME, help='Experiment configuration file name')
    args = parser.parse_args()

    config_file_path = os.path.join(CONFIG_DIR, f'{args.config_name}.yaml')
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
