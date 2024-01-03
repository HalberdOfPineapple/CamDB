from typing import Callable
from ConfigSpace import ConfigurationSpace
from typing import List

from camtune.run_history import RunHistory
from camtune.utils.logger import get_logger

from camtune.optimizer import build_optimizer, build_init_design, BaseOptimizer
from camtune.optimizer.sampler import BaseSampler


logger = get_logger('test_pgsql', './logs/output.log')


class Tuner:
    def __init__(self, args: dict, obj_func: Callable, configspace: ConfigurationSpace, param_logger=None):
        self.expr_name = args['expr_name']
        self.args = args['tune']
        self.seed = self.args['seed']

        self.is_minimizing: bool = self.args["is_minimizing"]
        self.run_history: RunHistory = RunHistory(self.expr_name)

        self.num_evals: int = self.args['computation_budget']
        self.num_init: int = self.args["num_init"]
        if self.num_init > self.num_evals:
            raise ValueError(f"Number of required initial samples ({self.num_init}) are greater than computational budget ({self.num_evals})")
        
        self.configspace: ConfigurationSpace = configspace
        self.obj_func: Callable = obj_func

        self.init_sampler: BaseSampler = build_init_design(
            self.args["init_design"],
            configspace=self.configspace,
            seed=self.seed,
        )
        self.optimizer: BaseOptimizer = build_optimizer(
            self.args,
            configspace=self.configspace,
            obj_func=self.obj_func,
        )

        if param_logger:
            global logger
            logger = param_logger
        
        
    def tune(self):
        logger.info(f'[Tuner] Start initial profiling by {self.args["init_design"]}')
        init_samples = self.init_sampler.generate(self.num_init)
        init_sample_perf = [self.obj_func(init_sample) for init_sample in init_samples]
        self.run_history.add_records(init_samples, init_sample_perf)

        num_evals = self.num_evals - self.num_init
        self.optimizer.optimize(num_evals, self.run_history)
        self.run_history.save_to_file()
        