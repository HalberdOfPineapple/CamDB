import json
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from camtune.utils import print_log, DTYPE, DEVICE

SEED = 1
KNOB_TYPES = ['enum', 'integer', 'real', 'float']
VALID_ADAPTER_ALIASES = ['none', 'tree', 'rembo', 'hesbo']

PGSQL_KNOB_JSON_13 = "/home/viktor/Experiments/TPC-H/knob_definitions/postgres-13.json"
PGSQL_JOB_SHAP = "/home/viktor/Experiments/TPC-H/knob_definitions/JOB_SHAP_PostgreSQL.json"


class SearchSpace:
    def __init__(
          self, 
          knob_definition_path: str,
          is_kv_config: bool,
          include=None, 
          ignore=[],
          seed=1,
    ) -> None:
        self.include = None
        self.ignore = []
        self.seed = seed

        with open(knob_definition_path, 'r') as f:
            definitions = json.load(f)
        
        self.is_kv_config = is_kv_config
        if is_kv_config:
            self.all_knobs = set(definitions.keys())
            self.include_knobs = include if include is not None else self.all_knobs - set(ignore)

            self.knobs = [{'name':name, **info} for name, info in definitions.items() if name in self.include_knobs]
            self.knobs_dict = {d['name']: d for d in self.knobs}
        else:
            self.all_knobs = set([d['name'] for d in definitions])
            self.include_knobs = include if include is not None else self.all_knobs - set(ignore) 

            self.knobs = [info for info in definitions if info['name'] in self.include_knobs]
            self.knobs_dict = { d['name']: d for d in self.knobs}

        self._bounds = None
        self.init_input_space()
    
    def init_input_space(self):
        self.input_variables = []
        self.lbs, self.ubs = [], []
        self.discrete_dims, self.continuous_dims = [], []
        self.idx_maps = {}

        for info in self.knobs:
            name, knob_type = info['name'], info['type']
            
            if knob_type not in KNOB_TYPES:
                raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')

            # Categorical variables
            if knob_type == 'enum':
                variable = CSH.CategoricalHyperparameter(
                                name=name,
                                choices=info["enum_values"] if self.is_kv_config else info['choices'],
                                default_value=info['default'])
                
                self.lbs.append(0)
                self.ubs.append(len(info["enum_values"]) - 1)

                var_idx = len(self.input_variables)
                self.discrete_dims.append(var_idx)
                self.idx_maps[var_idx] = {i: v for i, v in enumerate(info["enum_values"])}

            # Discrete numerical variables
            elif knob_type == 'integer':
                variable = CSH.UniformIntegerHyperparameter(
                                name=name,
                                lower=info['min'],
                                upper=info['max'],
                                default_value=info['default'])
                # When inputting the value of knobs, we do not need to care about the unit
                # The knob value, shown in PostgreSQL, will automatically be divided by the unit and we only need to
                # care about checking whether the value is correctly applied.
                self.lbs.append(info['min'])
                self.ubs.append(info['max'])

                var_idx = len(self.input_variables)
                self.discrete_dims.append(var_idx)
    
            # Continuous numerical variables
            elif knob_type == 'real' or knob_type == 'float':
                variable = CSH.UniformFloatHyperparameter(
                                name=name,
                                lower=info['min'],
                                upper=info['max'],
                                default_value=info['default'])
                self.lbs.append(info['min'])
                self.ubs.append(info['max'])

                var_idx = len(self.input_variables)
                self.continuous_dims.append(var_idx)
            else:
                raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')
    
            self.input_variables.append(variable)
        self.input_space = CS.ConfigurationSpace(name='input', seed=self.seed)
        self.input_space.add_hyperparameters(self.input_variables)
    
    def discrete_idx_to_value(self, knob_idx: int, num_val: int) -> int:
        knob_info: dict = self.knobs[knob_idx]
        if knob_info['type'] == 'enum':
            return self.idx_maps[knob_idx][num_val]
        elif knob_info['type'] == 'integer':
            return num_val
        else:
            raise NotImplementedError(f'Knob type of "{knob_info["type"]}" does not to be mapped.')
    
    @property
    def bounds(self) -> torch.Tensor:
        if self._bounds is None:
            self._bounds = torch.tensor([self.lbs, self.ubs], device=DEVICE, dtype=DTYPE) # (2, D)
        return self._bounds
