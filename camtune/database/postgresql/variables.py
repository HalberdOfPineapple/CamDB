#   db_host: localhost
#   db_port: '5432'
#   db_user_name: viktor
#   db_passwd: '741286'
#   db_name: tpch

#   pg_ctl: /usr/lib/postgresql/16/bin/pg_ctl
#   pg_data: /var/lib/postgresql/16/main
#   pg_server: /usr/lib/postgresql/16/bin/postgres
#   pg_conf: /var/lib/postgresql/experiment/conf/tune_cnf.conf
#   pg_sock: /var/run/postgresql/.s.PGSQL.543

DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "viktor"
DB_PWD = "741286"
DB_NAME = "tpch"

# PG_CTL = "/usr/lib/postgresql/16/bin/pg_ctl"
# PG_DATA = "/var/lib/postgresql/16/main"
# PG_SERVER = "/usr/lib/postgresql/16/bin/postgres"
# PG_CONF = "/var/lib/postgresql/experiment/conf/tune_cnf.conf"
# PG_DEFAULT_CONF = "/var/lib/postgresql/experiment/conf/default_cnf.conf"
# PG_SOCK = "/var/run/postgresql/.s.PGSQL.543"

PG_CTL = "/usr/lib/postgresql/14/bin/pg_ctl"
PG_DATA = "/var/lib/postgresql/14/main"
PG_SERVER = "/usr/lib/postgresql/14/bin/postgres"
PG_CONF = "/var/lib/postgresql/14/experiment/conf/tune_conf.conf"
PG_DEFAULT_CONF = "/var/lib/postgresql/14/experiment/conf/default_conf.conf"
PG_SOCK = "/var/run/postgresql/.s.PGSQL.543"


POSTGRE_USER = "postgres"
POSTGRE_PWD = "741286"

RESTART_WAIT_TIME = 0 # 20
TIMEOUT_CLOSE = 5 # 60

def init_global_vars(args_db: dict, args_user: dict):
    global PG_CTL, PG_DATA, PG_SERVER, PG_CONF, PG_DEFAULT_CONF, PG_SOCK, \
        DB_HOST, DB_PORT, DB_USER, DB_PWD, DB_NAME
    PG_CTL = PG_CTL if 'pg_ctl' not in args_db else args_db['pg_ctl']
    PG_DATA = PG_DATA if 'pg_data' not in args_db else args_db['pg_data']
    PG_SERVER = PG_SERVER if 'pg_server' not in args_db else args_db['pg_server']
    PG_CONF = PG_CONF if 'pg_conf' not in args_db else args_db['pg_conf']
    PG_SOCK = PG_SOCK if 'pg_sock' not in args_db else args_db['pg_sock']

    DB_HOST = DB_HOST if 'db_host' not in args_db else args_db['db_host']
    DB_PORT = DB_PORT if 'db_port' not in args_db else args_db['db_port']
    DB_USER = DB_USER if 'db_user_name' not in args_db else args_db['db_user_name']
    DB_PWD = DB_PWD if 'db_passwd' not in args_db else args_db['db_passwd']
    DB_NAME = DB_NAME if 'db_name' not in args_db else args_db['db_name']

    global POSTGRE_USER, POSTGRE_PWD
    POSTGRE_USER = POSTGRE_USER if 'pg_user_name' not in args_user else args_user['pg_user_name']
    POSTGRE_PWD = POSTGRE_PWD if 'pg_passwd' not in args_user else args_user['pg_passwd']
