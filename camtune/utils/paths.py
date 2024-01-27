import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(BASE_DIR, 'config')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RES_DIR = os.path.join(BASE_DIR, 'results')

OPTIM_CONFIG_DIR = os.path.join(BASE_DIR, 'optimizer', 'config')
OPTIM_LOG_DIR = os.path.join(BASE_DIR, 'optimizer', 'logs')

paths = [CONFIG_DIR, LOG_DIR, RES_DIR, OPTIM_CONFIG_DIR, OPTIM_LOG_DIR]
for path in paths:
    if not os.path.exists(path):
        # recursively create the path
        os.makedirs(path)
