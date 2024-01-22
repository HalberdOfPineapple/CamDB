import torch
import numpy as np
from typing import List

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from botorch.acquisition.analytic import ExpectedImprovement

ACQ_FUNC_MAP = {
    'ei': ExpectedImprovement,
}

def get_bounds_from_configspace(configspace):
    """
    Obtain the list of lower and upper bounds of all hyperparameters in a ConfigurationSpace object.
    
    Args:
    - configspace: A ConfigurationSpace object.

    Returns:
    - bounds: A list of tuples, where each tuple contains the lower and upper bound of a hyperparameter.
    """
    bounds = []
    for hp in configspace.get_hyperparameters():
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            bounds.append((hp.lower, hp.upper))
        elif isinstance(hp, CategoricalHyperparameter):
            # For categorical variables, bounds are not defined in the same way.
            # You might represent them as indices, e.g., (0, len(choices) - 1)
            bounds.append((0, len(hp.choices) - 1))
        else:
            raise TypeError(f"Unsupported hyperparameter type: {type(hp)}")
    return bounds

def convert_to_valid_config(configspace: ConfigurationSpace, continuous_config: np.array) -> Configuration:
    """ Convert a continuous configuration to a valid discrete/categorical configuration """
    valid_config = {}
    for i, hp in enumerate(configspace.get_hyperparameters()):
        if isinstance(hp, CategoricalHyperparameter):
            # Map to nearest category
            choice = hp.choices[np.argmin(np.abs([c - continuous_config[i] for c in range(len(hp.choices))]))]
            valid_config[hp.name] = choice
        elif isinstance(hp, UniformIntegerHyperparameter):
            # Round to nearest integer
            valid_config[hp.name] = int(round(continuous_config[i]))
        else:
            valid_config[hp.name] = max(min(np.float64(continuous_config[i]), hp.upper), hp.lower)
    return Configuration(configspace, values=valid_config)

def convert_to_valid_configs(configspace: ConfigurationSpace, continuous_configs: np.array) -> List[Configuration]:
    """
    Args:
        continuous_configs (np.array): with shape being (batch_size, D)

    """
    valid_configs: List[Configuration] = []
    for config in continuous_configs:
        valid_configs.append(convert_to_valid_config(configspace, config))
    return valid_configs

def convert_configurations_to_array(configurations: List[Configuration]):
    """
    Convert a list of Configuration objects to a numpy array.

    Args:
    - configurations (list of Configuration): List of Configuration objects.

    Returns:
    - numpy.ndarray: Numeric representation of configurations (shape: NxD)
    """
    if not configurations:
        return np.array([])

    def convert_hyperparameter(hp, value):
        if isinstance(hp, CategoricalHyperparameter):
            # Convert categorical values to integers based on their order in the choices list
            return hp.choices.index(value)
        elif isinstance(hp, (UniformIntegerHyperparameter, UniformFloatHyperparameter)):
            # Use integer and float values directly
            return value
        else:
            raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
    
    def convert_config(config: Configuration):
        dim = len(config.config_space.get_hyperparameters())
        res = np.zeros((dim,))
        for i, hp in enumerate(config.config_space.get_hyperparameters()):
            res[i] = convert_hyperparameter(hp, config[hp.name])
        return res

    # Convert each configuration in the list
    config_arrays = np.array([convert_config(config) for config in configurations])
    return config_arrays

