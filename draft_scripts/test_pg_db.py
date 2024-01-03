from camtune.database import PostgresqlDB
from camtune.search_space import SearchSpace, PGSQL_KNOB_JSON_13, PGSQL_JOB_SHAP

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
        **DB_CONN_PARAMS
    )

    config_space = SearchSpace(
        knob_definition_path=PGSQL_JOB_SHAP, 
        is_kv_config=IS_KV_CONFIG[PGSQL_JOB_SHAP]
    )
    sample_config = config_space.sample_config()

    db.apply_knobs_offline(dict(sample_config))