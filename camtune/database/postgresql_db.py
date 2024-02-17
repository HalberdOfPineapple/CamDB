
from ConfigSpace.configuration_space import Configuration 

from camtune.database.utils import initialize_knobs
from camtune.database.postgresql import (
    init_global_vars,  start_pg_default, recover_default_config,
    PostgreExecutor, PostgreKnobApplier
)
from camtune.database.postgresql.variables import *

from camtune.utils.logger import get_logger, print_log
LOGGER = None


class PostgresqlDB:
    def __init__(self, args: dict):
        global LOGGER
        LOGGER = get_logger()

        args_db, args_ssh = args['database'], args['ssh']

        # ---------------- Connection & Server Settings --------------
        init_global_vars(args_db, args_ssh)

        # ------------------ Mode Settings -----------------------
        self.remote_mode: bool = args_db['remote_mode']
        self.online_mode: bool = args_db['online_mode']

        # ------------------ Workload Settings -----------------------
        # Note that query can be saved locally
        self.executor = PostgreExecutor(
            benchmark=args_db['benchmark'],
            benchmark_fast=args_db['benchmark_fast'],
            exec_mode=args_db['exec_mode'],
            remote_mode=args_db['remote_mode'],
        )
        
        # ------------------ Knob Settings -----------------------
        self.knob_details = \
            initialize_knobs(args_db['knob_definitions'], args_db['knob_num'])
        self.knob_applier = PostgreKnobApplier(
            remote_mode=self.remote_mode,
            knob_details=self.knob_details
        )
        

    def step(self, config: Configuration) :
        recover_default_config(self.remote_mode)
        knobs = dict(config).copy()

        print_log('-' * 35 + ' Applying Knobs ' + '-' * 35)
        self.knob_applier.apply_knobs(knobs, self.online_mode)
        print_log('-' * 80)

        res, failed = self.executor.run_benchmark()
        res['knob_applied'] = True

        recover_default_config(self.remote_mode)
        start_pg_default(POSTGRE_PWD)

        return res