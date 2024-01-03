import torch
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter
from botorch.acquisition.analytic import ExpectedImprovement

ACQ_FUNC_MAP = {
    'ei': ExpectedImprovement,
}

def get_bounds_from_config_space(config_space):
    """
    Obtain the list of lower and upper bounds of all hyperparameters in a ConfigurationSpace object.
    
    Args:
    - config_space: A ConfigurationSpace object.

    Returns:
    - bounds: A list of tuples, where each tuple contains the lower and upper bound of a hyperparameter.
    """
    bounds = []
    for hp in config_space.get_hyperparameters():
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            bounds.append((hp.lower, hp.upper))
        elif isinstance(hp, CategoricalHyperparameter):
            # For categorical variables, bounds are not defined in the same way.
            # You might represent them as indices, e.g., (0, len(choices) - 1)
            bounds.append((0, len(hp.choices) - 1))
        else:
            raise TypeError(f"Unsupported hyperparameter type: {type(hp)}")
    return bounds

def convert_to_valid_config(configspace: ConfigurationSpace, continuous_config: np.array) -> dict:
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
            valid_config[hp.name] = continuous_config[i]
    return valid_config


