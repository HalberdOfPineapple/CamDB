import os
import argparse
import yaml

from camtune.utils.logger import load_logger
from camtune.search_space import SearchSpace
from camtune.tuner import Tuner
from camtune.database import PostgresqlDB

current_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(current_dir, 'config')
LOG_DIR = os.path.join(current_dir, 'logs')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

DEFAULT_CONFIG_NAME = 'postgre_tpch_remote'

def main(expr_config: dict):
    # -----------------------------------
    # Setup logger
    logger = load_logger(log_dir=LOG_DIR, logger_name=expr_config['expr_name'])

    # -----------------------------------
    # Build Database Controller
    db = PostgresqlDB(expr_config, logger)

    # -----------------------------------
    # Setup search space from knob definition file
    search_space = SearchSpace(
        expr_config['tune']['knob_definitions'], 
        is_kv_config=True,
        seed=expr_config['tune']['seed'],
    )

    # -----------------------------------
    # Initialize tuner and start tuning
    tuner = Tuner(expr_config, db.step, search_space.input_space)
    tuner.tune()

    # for i in range(0, expr_config['tune']['computation_budget']):
        # configuration = search_space.input_space.sample_configuration()
        # print(configuration)
        # eval_result = db.step(configuration)

        # print('-' * 50)
        # print(f'[Test Tune] Iteration {i}:')
        # print(eval_result)

        # logger.info(f'[Test Tune] Iteration {i}:\t{eval_result}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-c', type=str, default=DEFAULT_CONFIG_NAME, help='Experiment configuration file name')
    args = parser.parse_args()

    config_file_path = os.path.join(CONFIG_DIR, f'{args.config_name}.yaml')
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    main(config)
