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


def get_logger(logger_name, log_file_path, log_level=DEFAULT_LOG_LEVEL, log_format=DEFAULT_LOG_FORMAT):
    pathlib.Path(log_file_path).parents[0].mkdir(exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    handler = logging.FileHandler(log_file_path)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(handler)
    return logger

def load_logger(log_dir:str= './logs', logger_name: str=DEFAULT_LOGGER_NAME):
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger_path = os.path.join(log_dir, f'{time_str}.log')
    logger = get_logger(logger_name, logger_path)
    return logger