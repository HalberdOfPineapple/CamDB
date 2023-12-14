# TODO
from abc import ABC, abstractmethod
from ConfigSpace.configuration_space import Configuration


class BaseSampler(ABC):
    @abstractmethod
    def sample_config(self, size) -> Configuration:
        raise NotImplementedError
    
    @abstractmethod
    def update_model(self, configuration: Configuration, exec_res: dict):
        raise NotImplementedError