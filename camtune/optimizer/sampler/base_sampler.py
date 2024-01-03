import numpy as np

from abc import ABC, abstractmethod
from ConfigSpace import ConfigurationSpace, Configuration
from camtune.optimizer.optim_utils import get_bounds_from_config_space, convert_to_valid_config

class BaseSampler(ABC):
    """
    Generate samples within the specified domain (which defaults to the whole config space).

    Users should call generate() which auto-scales the samples to the domain.

    To implement new design methodologies, subclasses should implement _generate().
    """

    def __init__(self, configspace: ConfigurationSpace, seed: int = 0):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        dims (int): Number of dimensions

        lower_bounds : lower bounds in [0, 1] for continuous dimensions (optional)
        upper_bounds : upper bounds in [0, 1] for continuous dimensions (optional)
        """
        self.configspace: ConfigurationSpace = configspace
        self.dims: int = len(list(configspace.values()))

        bounds = get_bounds_from_config_space(configspace)

        self.lower_bounds = np.array([bound[0] for bound in bounds])
        self.upper_bounds = np.array([bound[1] for bound in bounds])

        self.seed = seed

    @abstractmethod
    def _generate(self, size:int):
        """
        Create samples in the domain specified during construction.

        Returns
        -------
        configs : list
            List of N sampled configurations within domain. (return_config is True)

        X : array, shape (N, D)
            Design matrix X in the specified domain. (return_config is False)
        """
        raise NotImplementedError

    def generate(self, size:int):
        """
        Create samples in the domain specified during construction.

        Returns
        -------
        configs : list
            List of N sampled configurations within domain. (return_config is True)

        X : array, shape (N, D)
            Design matrix X in the specified domain. (return_config is False)
        """
        X = self._generate(size)
        X = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * X

        configs = []
        for config_vec in X:
            valid_config = convert_to_valid_config(self.configspace, config_vec)
            configs.append(Configuration(self.configspace, values=valid_config))

        return configs