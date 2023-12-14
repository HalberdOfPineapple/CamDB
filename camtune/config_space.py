import json

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

SEED = 1
KNOB_TYPES = ['enum', 'integer', 'real', 'float']
VALID_ADAPTER_ALIASES = ['none', 'tree', 'rembo', 'hesbo']

PGSQL_KNOB_JSON_13 = "/home/viktor/Experiments/TPC-H/knob_definitions/postgres-13.json"
PGSQL_JOB_SHAP = "/home/viktor/Experiments/TPC-H/knob_definitions/JOB_SHAP_PostgreSQL.json"


class ConfigurationSpace:
    def __init__(
          self, 
          knob_definition_path,
          is_kv_config,
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

        self.input_space = self.init_input_space()
    
    def init_input_space(self):
        input_dimensions = [ ]
        for info in self.knobs:
            name, knob_type = info['name'], info['type']
            
            if knob_type not in KNOB_TYPES:
                raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')

            ## Categorical
            if knob_type == 'enum':
                dim = CSH.CategoricalHyperparameter(
                                name=name,
                                choices=info["enum_values"] if self.is_kv_config else info['choices'],
                                default_value=info['default'])
            ## Numerical
            elif knob_type == 'integer':
                dim = CSH.UniformIntegerHyperparameter(
                                name=name,
                                lower=info['min'],
                                upper=info['max'],
                                default_value=info['default'])
            elif knob_type == 'real' or knob_type == 'float':
                dim = CSH.UniformFloatHyperparameter(
                                name=name,
                                lower=info['min'],
                                upper=info['max'],
                                default_value=info['default'])
            
            input_dimensions.append(dim)

        input_space = CS.ConfigurationSpace(name='input', seed=self.seed)
        input_space.add_hyperparameters(input_dimensions)

        return input_space