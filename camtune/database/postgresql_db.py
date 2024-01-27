import os
import json
import time
import subprocess
import paramiko
from typing import List, Tuple
import configparser

from camtune.utils.logger import get_logger, print_log
from camtune.utils.parser import ConfigParser, parse_pgbench_output

from ConfigSpace.configuration_space import Configuration 
from .dbconnector import PostgresqlConnector
from camtune.utils import get_logger, print_log
LOGGER = None

PG_CTL = "/usr/lib/postgresql/16/bin/pg_ctl"
PG_DATA = "/var/lib/postgresql/16/main"
PG_SERVER = "/usr/lib/postgresql/16/bin/postgres"
PG_CONF = "/var/lib/postgresql/experiment/conf/tune_cnf.conf"
PG_SOCK = "/var/run/postgresql/.s.PGSQL.543"

dst_data_path = os.environ.get("DATADST")
src_data_path = os.environ.get("DATASRC")
RESTART_WAIT_TIME = 0 # 20
TIMEOUT_CLOSE = 60

REMOTE_MODE = True
ISOLATION_MODE = False

# SSH user is set to postgres for allowing acces for pg_ctl and postgres server
# while the PostgresqlDB.user refers to the user of the database accessing data
SSH_USER = 'postgres'
SSH_PWD = '741286'

BENCHMARK_DIR = '/home/viktor/Experiments/CamDB/camtune/benchmarks'
BENCHMARK = 'TPCH'
QUERY_PATH_MAP = {
    'TPCH': os.path.join(BENCHMARK_DIR, 'tpch'),
}


def initialize_knobs(knobs_config, num) -> dict:
    if num == -1:
        with open(knobs_config, 'r') as f:
            knob_details = json.load(f)
    else:
        with open(knobs_config, 'r') as f:
            knob_tmp = json.load(f)
            i = 0

            knob_details = {}
            knob_names = list(knob_tmp.keys())
            while i < num:
                key = knob_names[i]
                knob_details[key] = knob_tmp[key]
                i = i + 1

    return knob_details

class PostgresqlDB:
    def __init__(self, args: dict):
        global LOGGER
        LOGGER = get_logger()

        args_db, args_ssh, args_tune = args['database'], args['ssh'], args['tune']

        # ---------------- Connection & Server Settings --------------
        self.host = args_db['db_host']
        self.port = args_db['db_port']
        self.user_name = args_db['db_user_name']
        self.db_name = args_db['db_name']
        self.passwd = args_db['db_passwd']

        self.pg_ctl =  args_db['pg_ctl']
        self.pg_data =  args_db['pg_data']
        self.pg_server = args_db['pg_server']
        self.pg_cnf = args_db['pg_conf']
        self.pg_sock = args_db['pg_sock']

        # ------------------ Workload Settings -----------------------
        # Note that query can be saved locally
        self.benchmark: str = args_db['benchmark'] # e.g. 'TPCH'
        self.benchmark_fast: bool = args_db['benchmark_fast']
        if self.benchmark.upper() not in QUERY_PATH_MAP:
            raise ValueError(f"[PostgresqlDB] Undefined Benchmark {self.benchmark}")
        self.query_dir = QUERY_PATH_MAP[self.benchmark]

        self.use_pgbench: bool = args_db['use_pgbench']
        if self.use_pgbench:
            self.query_dir = self.query_dir + '_pgbench'

        # ------------------ Mode Settings -----------------------
        self.remote_mode: bool = args_db['remote_mode']
        if self.remote_mode:
            self.ssh_user = args_ssh['ssh_user']
            self.ssh_pwd = args_ssh['ssh_pwd']
        
        self.online_mode: bool = args_db['online_mode']
        
        # ------------------ Knob Settings -----------------------
        self.knob_details = \
            initialize_knobs(args_tune['knob_definitions'], args_tune['knob_num'])

    def step(self, config: Configuration) :
        knobs = dict(config).copy()

        print_log('-' * 35 + ' Applying Knobs ' + '-' * 35)
        if self.online_mode:
            self.apply_knobs_online(knobs)
        else:
            self.apply_knobs_offline(knobs)
        print_log('-' * 80)

        res, failed = self.run_benchmark()
        # res = {}
        return res

    def run_benchmark(self) -> Tuple[dict, list]:
        benchmark_name = self.benchmark + ('_fast' if self.benchmark_fast else '')
        res = {'benchmark': benchmark_name}

        total_exec_time = 0
        query_file_names = []

        if self.benchmark.upper() == 'TPCH':
            if not self.benchmark_fast:
                query_list_file = 'tpch_query_list.txt'
            else:
                query_list_file = 'tpch_query_fast_list.txt'
                # query_list_file = 'tpch_query_copy.txt'
            print_log(f"[PostgresqlDB] Exeucting queries listed in {query_list_file}")

            lines = open(os.path.join(BENCHMARK_DIR, query_list_file), 'r').readlines()
            for line in lines:
                query_file_names.append(line.rstrip())
            query_file_names = [os.path.join(self.query_dir, query_file_name) for query_file_name in query_file_names]
            
            if self.use_pgbench:
                results, failed = self.exec_queries_pgbench(query_file_names)
                for query_file_name in query_file_names:
                    if query_file_name in results:
                        # latency measured in ms
                        avg_latency = float(results[query_file_name]['latency_average'])
                        total_exec_time += avg_latency
            else:
                results, failed = self.exec_queries(query_file_names)

                for _, exec_res in results.items():
                    exec_time = float(exec_res[0]['execution_time'])
                    total_exec_time += exec_time
            res['total_exec_time'] = total_exec_time

        return res, failed

    def exec_queries_pgbench(self, query_file_names: List[str]):
        res: dict = {}
        failed = []
        if self.remote_mode:
            remote_tmp_sql = '/tmp/tmp.sql'

            # Build SSH connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
            
            for query_file_name in query_file_names:
                sftp = ssh.open_sftp()
                try:
                    # Put local query file into a temporary file in remote server
                    sftp.put(query_file_name, remote_tmp_sql)
                except IOError:
                    print_log(f'[PostgresqlDB] Remote SFTP put SQL query {query_file_name} failed')
                    if sftp: sftp.close()
                    continue
                if sftp: sftp.close()

                # Execute pgbench command
                command = f"pgbench -f {remote_tmp_sql} {self.db_name} -n"
                _, stdout, stderr  = ssh.exec_command(command)
                            
                retcode = stdout.channel.recv_exit_status()
                if retcode != 0:
                    print_log(f'[PostgresqlDB] Remote executing SQL query {query_file_name} using pgbench failed, with following information:')
                    print_log(f'[PostgresqlDB] STDOUT: {stdout.read().decode("utf-8").strip()}')
                    LOGGER.error(f'[PostgresqlDB] STDERR: {stderr.read().decode("utf-8").strip()}')
                    failed.append(query_file_name)
                    continue

                output = stdout.read().decode('utf-8')
                res[query_file_name] = parse_pgbench_output(output)
                print_log(f'[PostgresqlDB] Remote executing SQL query {query_file_name} using pgbench successfully')
                            
            if ssh: ssh.close()
        else:
            for query_file_name in query_file_names:
                command = f"pgbench -f {query_file_name} {self.db_name} -n"
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print_log(f'[PostgresqlDB] Local execution of SQL query {query_file_name} using pgbench failed')
                res[query_file_name] = parse_pgbench_output(result.stdout)

        return res, failed

    # Assume the query files are stored locally (to be executed remotely)
    def exec_queries(self, query_file_names: List[str], json=True):
        queries = []
        for query_file in query_file_names:
            with open(query_file, 'r') as f:
                query_lines = f.readlines()
            query = ' '.join(query_lines)
            queries.append((query_file, query))

        results = {}
        failed = []
        db_conn: PostgresqlConnector = None
        try:
            # Build connection to PostgreSQL server
            db_conn = PostgresqlConnector(host=self.host,
                                          port=self.port,
                                          user=self.user_name,
                                          passwd=self.passwd,
                                          name=self.db_name)
            if db_conn.conn.closed == 0:
                print_log('[PostgresqlDB] Connected to PostgreSQL server for query execution:')

            # Executing queries and fetch execution results
            for query_file, query in queries:
                print_log(f'[PostgresqlDB] Executing {query_file}')

                try:
                    result = db_conn.fetch_results(query, json=json)
                    results[query_file] = result
                except Exception as e:
                    print_log(f'[PostgresqlDB] Query execution failed when executing {query_file}')
                    print_log(f"[PostgresqlDB] Error information: {e}")
                    print_log(f"[PostgresqlDB] Error information: {e}")
                    failed.append(query_file)

            # Close connection
            db_conn.close_db()
        except:
            print_log(f'[PostgresqlDB] Query execution failed.')
            if db_conn: db_conn.close_db()
        
        return results, failed
    
    def apply_knobs_offline(self, knobs: dict):
        self.kill_postgres()

        if 'min_wal_size' in knobs.keys():
            if 'wal_segment_size' in knobs.keys():
                wal_segment_size = knobs['wal_segment_size']
            else:
                wal_segment_size = 16
            if knobs['min_wal_size'] < 2 * wal_segment_size:
                knobs['min_wal_size'] = 2 * wal_segment_size
                print_log('[PostgresqlDB] Knob "min_wal_size" must be at least twice "wal_segment_size"')

        # --------------------------------------------------------------------
        # Adjust knobs values by modifying the configuration file offline
        knobs_not_in_cnf = self.modify_config_file(knobs)

        # --------------------------------------------------------------------
        # If PostgreSQL server cannot start normally, terminate the program
        success = self.start_postgres()
        if not success:
            raise RuntimeError("[PostgresqlDB] PostgreSQL failed to start after applying knobs offline.")

        print_log('[PostgresqlDB] Sleeping for {} seconds after restarting postgres'.format(RESTART_WAIT_TIME))
        time.sleep(RESTART_WAIT_TIME)

        # --------------------------------------------------------------------
        # Apply knobs that have not been written in configuration file online
        if len(knobs_not_in_cnf) > 0:
            tmp_rds = {}
            for knob_rds in knobs_not_in_cnf:
                tmp_rds[knob_rds] = knobs[knob_rds]
            self.apply_knobs_online(tmp_rds)
        else:
            print_log("[PostgresqlDB] No knobs need to be applied online")

        self.check_knobs_applied(knobs, online=False)

    def apply_knobs_online(self, knobs: dict):
        # apply knobs remotely
        print_log(f"[PostgresqlDB] Knobs to be applied online: {list(knobs.keys())}")
        db_conn: PostgresqlConnector = None
        try:
            db_conn = PostgresqlConnector(host=self.host,
                                            port=self.port,
                                            user=self.user_name,
                                            passwd=self.passwd,
                                            name=self.db_name)

            for key in knobs.keys():
                db_conn.set_knob_value(key, knobs[key])
            db_conn.close_db()
        except Exception as e:
            print_log(f"[PostgresqlDB] Online knob setting failed with information: {e}")

        self.check_knobs_applied(knobs, online=True)

    def check_knobs_applied(self, knobs: dict, online: bool) -> int:
        num_not_applied = 0
        db_conn: PostgresqlConnector = None
        try:
            db_conn = PostgresqlConnector(host=self.host,
                                            port=self.port,
                                            user=self.user_name,
                                            passwd=self.passwd,
                                            name=self.db_name)
            for k, v in knobs.items():
                if k in self.knob_details and 'unit' in self.knob_details[k]:
                    unit = self.knob_details[k]['unit']
                else:
                    unit = None
                applied, actual_val = db_conn.check_knob_apply(k, v, unit=unit)
                if not applied:
                    num_not_applied += 1
                    print_log(f"[PostgresqlDB] Knob {k} is not successfully set to {v} (actual value: {actual_val})")
            db_conn.close_db()
        except Exception as e:
            print_log(f"[PostgresqlDB] Knobs checking failed with exception information: {e}")
            if db_conn: 
                db_conn.close_db()
            return -1

        check_mode = "online" if online else "offline"
        if num_not_applied > 0:
            print_log(f"[PostgresqlDB] {num_not_applied} / {len(knobs)} knobs not successfully applied {check_mode}.")
        elif num_not_applied == 0:
            print_log(f"[PostgresqlDB] Knobs successfully applied {check_mode}.")
        return num_not_applied

    
    def modify_config_file(self, knobs: dict):
        if self.remote_mode:
            cnf = '/tmp/pglocal.cnf'
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
    
            # Fetch PG configuration file to local through SFTP
            sftp = ssh.open_sftp()
            try:
                sftp.get(self.pg_cnf, cnf) 
            except IOError:
                print_log('[PostgresqlDB] Remote SFTP get failed: PostgreSQL configuration file does not exist.')

            if sftp: sftp.close()
            if ssh: ssh.close()
        else:
            cnf = self.pg_cnf

        # Update configuration file (locally)
        cnf_parser = ConfigParser(cnf)
        knobs_not_in_cnf = []
        for key in knobs.keys():
            if key not in self.knob_details.keys():
                knobs_not_in_cnf.append(key)
                continue
            cnf_parser.set(key, knobs[key])
        cnf_parser.replace()

        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            sftp = None
            try:
                # Put newly generated configuration to SSH server
                sftp = ssh.open_sftp()

                # Note cnf is the local temporary file while `self.pg_cnf` is the remote file
                sftp.put(cnf, self.pg_cnf)
            except IOError as e:
                print_log(f'[PostgresqlDB] Remote SFTP put failed: {e}.')

            if sftp: sftp.close()
            if ssh: ssh.close()

        print_log('[PostgresqlDB] config file modification done.')
        return knobs_not_in_cnf

    def kill_postgres(self):
        kill_cmd = '{} stop -D {}'.format(self.pg_ctl, self.pg_data)
        force_kill_cmd1 = "ps aux|grep '" + self.pg_sock + "'|awk '{print $2}'|xargs kill -9"
        force_kill_cmd2 = "ps aux|grep '" + self.pg_cnf + "'|awk '{print $2}'|xargs kill -9"

        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(kill_cmd)
            ret_code = ssh_stdout.channel.recv_exit_status()
            if ret_code == 0:
                print_log("[PostgresqlDB] Remote PostgreSQL server shut down successfully")
                print_log('\n')
            else:
                print_log("[PostgresqlDB] Remote shut down attempt failed: force the server to shut down")
                ssh.exec_command(force_kill_cmd1)
                ssh.exec_command(force_kill_cmd2)
                print_log('[PostgresqlDB] Remote PostgreSql server shut down by forcing')

            ssh.close()
        else:
            p_close = subprocess.Popen(kill_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
            try:
                outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
                ret_code = p_close.poll()
                if ret_code == 0:
                    print_log("[PostgresqlDB] Local PostgreSQL server shut down successfully")
            except subprocess.TimeoutExpired:
                print_log("[PostgresqlDB] Local shut down attempt failed: force the server to shut down")
                os.system(force_kill_cmd1)
                os.system(force_kill_cmd2)
                print_log('[PostgresqlDB] Local PostgreSql server shut down by forcing')
    
    def start_postgres(self):
        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            print_log('[PostgresqlDB] Remotely starting PostgreSQL server...')
            start_cmd = '{} --config_file={} -D {}'.format(self.pg_server, self.pg_cnf, self.pg_data)
            wrapped_cmd = 'echo $$; exec ' + start_cmd
            _, start_stdout, _ = ssh.exec_command(wrapped_cmd)
            self.pid = int(start_stdout.readline())

            if ISOLATION_MODE:
                cgroup_cmd = 'sudo -S cgclassify -g memory,cpuset:server ' + str(self.pid)
                ssh_stdin, ssh_stdout, _ = ssh.exec_command(cgroup_cmd)
                ssh_stdin.write(self.ssh_pwd + '\n')
                ssh_stdin.flush()

                ret_code = ssh_stdout.channel.recv_exit_status()
                ssh.close()
                if not ret_code:
                    print_log('[PostgresqlDB] Add {} to memory,cpuset:server'.format(self.pid))
                else:
                    print_log('[PostgresqlDB] Failed to add {} to memory,cpuset:server'.format(self.pid))

        else:
            proc = subprocess.Popen([self.pg_server, '--config_file={}'.format(self.pg_cnf), '-D',  self.pg_data])
            self.pid = proc.pid
            if ISOLATION_MODE:
                command = 'sudo cgclassify -g memory,cpuset:server ' + str(self.pid)
                p = os.system(command)
                if not p:
                    print_log('[PostgresqlDB] add {} to memory,cpuset:server'.format(self.pid))
                else:
                    print_log('[PostgresqlDB] Failed: add {} to memory,cpuset:server'.format(self.pid))

        count = 0
        start_success = True
        print_log('[PostgresqlDB] Wait for connection to the started server...')
        while True:
            try:
                db_conn = PostgresqlConnector(host=self.host,
                                          port=self.port,
                                          user=self.user_name,
                                          passwd=self.passwd,
                                          name=self.db_name)
                db_conn = db_conn.conn
                if db_conn.closed == 0:
                    print_log('[PostgresqlDB] Successfully connected to the started PostgreSQL Server')
                    db_conn.close()
                    break
            except:
                pass

            time.sleep(1)
            count = count + 1
            if count > 30: # 600
                start_success = False
                print_log("[PostgresqlDB] PG server start failed: can not connect to DB")
                clear_cmd = """ps -ef|grep postgres|grep -v grep|cut -c 9-15|xargs kill -9"""
                subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                 close_fds=True)
                print_log("kill all postgres process")
                break

        print_log('[PostgresqlDB] {} seconds waiting for connection'.format(count))
        print_log('[PostgresqlDB] start command: postgres --config_file={}'.format(self.pg_cnf))
        print_log(f'[PostgresqlDB] PostgresSQL is {"" if start_success else "not"} up')
    
        return start_success