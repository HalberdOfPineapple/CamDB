from typing import Callable
from ConfigSpace import ConfigurationSpace
from typing import List

from camtune.run_history import RunHistory
from camtune.utils.logger import print_log

from camtune.optimizer import build_optimizer, build_init_design, BaseOptimizer
from camtune.optimizer.sampler import BaseSampler
from camtune.optimizer.benchmarks import Benchmark


class Tuner:
    def __init__(self, 
      expr_name: str, 
      args: dict, 
      benchmark: Benchmark, 
      res_dir: str,
      param_logger=None,
    ):
        if param_logger:
            global logger
            logger = param_logger

        self.expr_name = expr_name
        self.args = args
        self.seed = self.args['seed']

        self.is_minimizing: bool = self.args["is_minimizing"]
        if 'perf_name' in self.args:
            perf_name, perf_unit = self.args['perf_name'], self.args['perf_unit']
        else:
            perf_name, perf_unit = 'func_val', 'null'
        self.run_history: RunHistory = RunHistory(
            self.expr_name, res_dir, perf_name, perf_unit, 
            is_minimizing=self.is_minimizing,
        )

        self.num_evals: int = self.args['computation_budget']
        self.num_init: int = self.args["num_init"]
        if self.num_init > self.num_evals:
            raise ValueError(f"Number of required initial samples ({self.num_init}) are greater than computational budget ({self.num_evals})")

        self.benchmark = benchmark
        self.configspace: ConfigurationSpace = benchmark.configspace

        self.init_sampler: BaseSampler = build_init_design(
            self.args["init_design"],
            configspace=self.configspace,
            seed=self.seed,
        )
        self.optimizer: BaseOptimizer = build_optimizer(
            self.args,
            configspace=self.configspace,
            benchmark=self.benchmark,
            is_minimizing=self.is_minimizing,
            param_logger=logger,
        )

    def tune(self):
        # if self.num_init != 0:
        #     print_log(f'[Tuner] Start initial profiling by {self.args["init_design"]}')
        #     init_samples = self.init_sampler.generate(self.num_init)
        #     init_sample_perf = [self.benchmark.eval(init_sample) for init_sample in init_samples]
        #     self.run_history.add_records(init_samples, init_sample_perf)

        num_evals = self.num_evals - self.num_init
        print_log(f'[Tuner] Start optimization by strategy: {self.args["strategy"]} for {num_evals} iterations')
        self.optimizer.optimize(num_evals, self.num_init, self.run_history)
        print_log(f'[Tuner] Main optimization loop terminated')

        print_log(f'[Tuner] Start saving to log files...')
        self.run_history.save_to_file()
        print_log(f'[Tuner] Logging completed.')
        