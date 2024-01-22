import time
import logging
import os
import pathlib

DEFAULT_LOGGER_NAME = 'test_pgsql'
LOG_LEVEL_DICT = {
    "logging.debug": logging.DEBUG,
    "logging.info": logging.INFO,
    "logging.warning": logging.WARNING,
    "logging.error": logging.ERROR,
    "logging.critical": logging.CRITICAL
}

DEFAULT_LOG_FORMAT = os.environ.get('DEFAULT_LOG_FORMAT',
                                    '[%(asctime)s:%(filename)s#L%(lineno)d:%(levelname)s]: %(message)s')
DEFAULT_LOG_LEVEL = LOG_LEVEL_DICT[os.environ.get('DEFAULT_LOG_LEVEL', 'logging.INFO').lower()]

logger = None

def create_logger(logger_name, log_file_path, log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT):
    pathlib.Path(log_file_path).parents[0].mkdir(exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(handler)
    return logger

def init_logger(log_dir:str= './logs', logger_name: str=DEFAULT_LOGGER_NAME):
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger_path = os.path.join(log_dir, f'{time_str}.log')

    global logger
    logger = create_logger(logger_name, logger_path)
    logger.info(f'[Logger] Logger created at {logger_path}')

def get_logger():
    global logger
    return logger

def print_log(msg: str, level: str='info', print_msg: bool=False):
    if print_msg:
        print(msg)

    global logger
    if logger is not None:
        if level == 'debug':
            logger.debug(msg)
        elif level == 'info':
            logger.info(msg)
        elif level == 'warning':
            logger.warning(msg)
        elif level == 'error':
            logger.error(msg)
        elif level == 'critical':
            logger.critical(msg)
        else:
            raise ValueError(f'Unknown log level: {level}')