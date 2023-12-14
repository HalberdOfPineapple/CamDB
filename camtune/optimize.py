
import argparse
from configparser import ConfigParser

from camtune.config_space import ConfigurationSpace
from camtune.sampler import RandomSampler
from camtune.database import PostgresqlDB

DEFAULT_CONFIG_FILE = '/home/viktor/Experiments/CamDB/camtune/config/postgre_tpch_remote.ini'

def main(expr_config: ConfigParser):
    # -----------------------------------
    # Read configurations for experiments
    expr_config.read(args.config)

    # -----------------------------------
    # Build Database Controller
    db = PostgresqlDB(config)

    # -----------------------------------
    # Build Configuration Sampler
    config_space = ConfigurationSpace(expr_config['tune']['knob_definitions'], is_kv_config=True)
    sampler = RandomSampler(config_space)

    for i in range(0, int(expr_config['tune']['computation_budget'])):
        configuration = sampler.sample_config()
        eval_result = db.step(configuration)
        sampler.update_model(configuration, eval_result)

        print('-' * 50)
        print(f'Iteration {i}:')
        print(eval_result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE, help='config file')
    args = parser.parse_args()

    config = ConfigParser()
    main(config)
