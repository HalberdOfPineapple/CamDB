import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))

from postgresql_db import PostgresqlDB, BENCHMARK
from config_space import ConfigurationSpace, PGSQL_KNOB_JSON_13, PGSQL_JOB_SHAP

IS_KV_CONFIG = {
    PGSQL_JOB_SHAP: True,
    PGSQL_KNOB_JSON_13: False,
}
DB_PWD = "741286" 
REMOTE_MODE = True

DB_CONN_PARAMS = {
    "host": "localhost",
    "port": "5432",
    "user_name": "viktor",
    "passwd": DB_PWD,
    "db_name": "tpch" # database name
}

if __name__ == '__main__':
    db = PostgresqlDB(
        knob_definition_path=PGSQL_JOB_SHAP,
        remote_mode=REMOTE_MODE,
        benchmark=BENCHMARK,
        **DB_CONN_PARAMS
    )

    db.exec_queries()

    