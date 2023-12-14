from ConfigSpace.configuration_space import Configuration
from camtune.config_space import ConfigurationSpace
from .base_sampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, config_space: ConfigurationSpace, seed: int=0):
        self.seed = 0
        self.config_space = config_space
    
    def sample_config(self, size=None) -> Configuration:
        return self.config_space.input_space.sample_configuration(size)
    
    def update_model(self, configuration: Configuration, exec_res: dict):
        pass