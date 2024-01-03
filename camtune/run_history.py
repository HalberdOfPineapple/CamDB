import os
from typing import List, Optional, Tuple, Any
from ConfigSpace import Configuration

current_dir = os.path.dirname(os.path.abspath(__file__))
RES_DIR =  os.path.join(current_dir, 'results') 
if not os.path.exists(RES_DIR): os.mkdir(RES_DIR)

class RunHistory:
    def __init__(
      self, 
      expr_name: str, 
      perf_name: str='latency',
      perf_unit: str='ms',
      records: Optional[List[Tuple[Configuration, Any]]]=None,
      is_minimizing: bool=True,
    ):
        self.expr_name = expr_name
        self.perf_name = perf_name
        self.perf_unit = perf_unit

        self.records = records if records else []
        self.is_minimizing = is_minimizing

        if len(self.records) > 0:
          best_pair = sorted(self.records, 
                            key=lambda record: record[1] if is_minimizing else -record[1])[0]
          self.best_config = best_pair[0]
          self.best_perf = best_pair[1]
        else:
          self.best_config = None
          self.best_perf = None
    
    def add_records(self, configs: List[Configuration], perf_metrics: list[Any]):
        for config, perf_metric in zip(configs, perf_metrics):
          self.records.append((config, perf_metric))
          if (self.best_config is None or self.best_perf is None) or \
            (self.is_minimizing and perf_metric < self.best_perf) or \
            (not self.is_minimizing and perf_metric > self.best_perf):
              self.best_config = config
              self.best_perf = perf_metric
    
    def save_to_file(self):
        res_dir = os.path.join(RES_DIR, self.expr_name)
        if not os.path.exists(res_dir): os.mkdir(res_dir)

        import json
        res_dict = {}
        records_file = os.path.join(res_dir, 'records.csv')
        for i, (config, perf) in enumerate(self.records):
            res_dict[str(i)] = {
                "perf_name": self.perf_name,
                "perf_unit": self.perf_unit,
                "perf_metric": perf,
                "configuration": config.get_dictionary(),
            }
        with open(records_file, 'w') as f:
          json.dump(res_dict, f)

        best_record_file = os.path.join(res_dir, 'best.log')
        with open(best_record_file, 'w') as f:
            f.write(f"Experiment name: {self.expr_name}\n")
            f.write(f"Best Performance: {self.best_perf}\n")
            f.write("Best Configuration:\n")
            for knob, val in dict(self.best_config).items():
                f.write(f"\t{knob}:\t{val}\n")
