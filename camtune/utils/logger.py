import time
import logging
import os
import pathlib

from .paths import LOG_DIR

LOGGER: logging.Logger = None
EXPR_NAME: str = None

def init_logger(expr_name: str, log_dir: str=LOG_DIR):
    # Setup logging
    print(f"[logger.py] Initialize logger for experiment: {expr_name}")

    log_filename = os.path.join(log_dir, f'{expr_name}.log')
    with open(log_filename, 'w') as f:
        f.write('')
    logging.basicConfig(filename=log_filename, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the logger
    global LOGGER
    LOGGER = logging.getLogger(expr_name)
    LOGGER.info('\n')


def get_logger():
    global LOGGER
    return LOGGER

def print_log(msg: str, end:str='\n', level: str='info', print_msg: bool=False):
    if print_msg:
        print(msg, end=end)

    global LOGGER
    if LOGGER is not None:
        if level == 'debug':
            LOGGER.debug(msg)
        elif level == 'info':
            LOGGER.info(msg)
        elif level == 'warning':
            LOGGER.warning(msg)
        elif level == 'error':
            LOGGER.error(msg)
        elif level == 'critical':
            LOGGER.critical(msg)
        else:
            raise ValueError(f'Unknown log level: {level}')