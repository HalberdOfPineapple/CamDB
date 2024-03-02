import os
import paramiko
from typing import List, Tuple
from subprocess import CompletedProcess

from camtune.utils import print_log, get_logger, QUERY_PATH_MAP, BENCHMARK_DIR, BENCHMARK_SET
from camtune.database.workloads import SYSBENCH_SCRIPTS_DIR, SYSBENCH_WORKLOADS, parse_sysbench_output
from camtune.database.utils import run_as_user

from .utils import parse_pgbench_output, run_as_postgre
from .connector import PostgresqlConnector
from .variables import *

LOGGER = None
EXEC_MODES = ['raw', 'pgbench', 'explain']

SYSBENCH_CMD_TMP = "sysbench {} --db-driver=pgsql --pgsql-db={} --pgsql-user={} --pgsql-password='{}' --pgsql-host={} --tables={} --table-size={} "
SYSBENCH_CMD_PREPARE = SYSBENCH_CMD_TMP + "prepare"
SYSBENCH_CMD_RUN = SYSBENCH_CMD_TMP + "--time={} run"
SYSBENCH_CMD_CLEANUP = SYSBENCH_CMD_TMP + "cleanup"

def extract_sql(query_for_time: str):
    start_token = "query := '"
    end_token = "';"

    start_index = query_for_time.find(start_token) + len(start_token)
    end_index = query_for_time.find(end_token, start_index)
    extracted_query = query_for_time[start_index:end_index]

    # Correcting for escaped single quotes within the SQL query
    extracted_query_corrected = extracted_query.replace("''", "'")

    return extracted_query_corrected

def extract_total_cost(plan):
    return float(plan.split('cost=')[1].split('..')[1].split(' ')[0])


class PostgreExecutor():
    def __init__(self, 
                 benchmark: str, 
                 benchmark_fast: bool, 
                 exec_mode: str='raw', 
                 remote_mode: bool=False,
                 sysbench_mode: str=None,
    ):
        if benchmark.upper() not in BENCHMARK_SET:
            raise ValueError(f"[PGExecutor] Undefined Benchmark {benchmark}")
        self.remote_mode = remote_mode

        self.benchmark = benchmark
        self.benchmark_fast = benchmark_fast
        self.sysbench_mode = sysbench_mode
        
        self.exec_mode = exec_mode
        if self.exec_mode not in EXEC_MODES:
            raise ValueError(f"[PGExecutor] Unsupported execution mode {self.exec_mode}")
        
        # Initialize query paths
        self.init_query_paths()

        global LOGGER
        LOGGER = get_logger()
    
    def init_query_paths(self):
        if self.benchmark.lower() == 'sysbench':
            self.query_dir = ""
            self.query_file_names = []
            return

        # Get the directory to the query files
        self.query_dir = QUERY_PATH_MAP[self.benchmark.upper()]
        if self.exec_mode == 'pgbench':
            self.query_dir = self.query_dir + '_pgbench'
        elif self.exec_mode == 'explain':
            self.query_dir = self.query_dir + '_explain'
        
        # Get the list of exact query file names
        query_file_names = []
        if self.benchmark.upper() == 'TPCH':
            if not self.benchmark_fast:
                query_list_file = 'tpch_query_list.txt'
            else:
                query_list_file = 'tpch_query_fast_list.txt'
            print_log(f"[PGExecutor] Exeucting queries listed in {query_list_file}")

            lines = open(os.path.join(BENCHMARK_DIR, query_list_file), 'r').readlines()
            for line in lines:
                query_file_names.append(line.rstrip())
            query_file_names = [os.path.join(self.query_dir, query_file_name) for query_file_name in query_file_names]
        else:
            raise ValueError(f"[PGExecutor] Unsupported benchmark {self.benchmark}")
        self.query_file_names = query_file_names

    def run_benchmark(self) -> Tuple[dict, list]:
        if self.benchmark.lower() == 'sysbench':
            return self.run_sysbench()

        benchmark_name = self.benchmark + ('_fast' if self.benchmark_fast else '')
        res = {'benchmark': benchmark_name}

        if self.exec_mode == 'pgbench':   
            total_exec_time = 0

            results, failed = self.exec_queries_pgbench()
            for query_file_name in self.query_file_names:
                if query_file_name in results:
                    # latency measured in ms
                    avg_latency = float(results[query_file_name]['latency_average'])
                    total_exec_time += avg_latency

            res['total_exec_time'] = total_exec_time
        elif self.exec_mode == 'raw':
            total_exec_time = 0

            results, failed = self.exec_queries()
            for _, exec_res in results.items():
                exec_time = float(exec_res[0]['execution_time'])
                total_exec_time += exec_time

            res['total_exec_time'] = total_exec_time
        elif self.exec_mode == 'explain':
            total_cost: float = 0

            results, failed = self.exec_queries()
            for _, exec_res in results.items():
                exec_plan: str = exec_res[0]['QUERY PLAN']
                total_cost += extract_total_cost(exec_plan)

            res['total_cost'] = total_cost
        else:
            raise ValueError(f"[PGExecutor] Unsupported execution mode {self.exec_mode}")

        return res, failed


    # --------------------------------------------------------
    # Raw Queries
    # --------------------------------------------------------

    # Assume the query files are stored locally (to be executed remotely)
    def exec_queries(self, json: bool=True):
        queries = []
        for query_file in self.query_file_name:
            with open(query_file, 'r') as f:
                query_lines = f.readlines()
            query = ' '.join(query_lines)
            queries.append((query_file, query))

        results = {}
        failed = []
        db_conn: PostgresqlConnector = None
        try:
            # Build connection to PostgreSQL server
            db_conn = PostgresqlConnector()
            if db_conn.conn.closed == 0:
                print_log('[PGExecutor] Connected to PostgreSQL server for query execution:')

            # Executing queries and fetch execution results
            for query_file, query in queries:
                print_log(f'[PGExecutor] Executing {query_file}')
                try:
                    result = db_conn.fetch_results(query, json=json)
                    results[query_file] = result
                except Exception as e:
                    print_log(f'[PGExecutor] Query execution failed when executing {query_file}')
                    print_log(f"[PGExecutor] Error information: {e}")
                    failed.append(query_file)

                # Close connection
            db_conn.close_db()
        except:
            print_log(f'[PGExecutor] Query execution failed.')
            if db_conn: db_conn.close_db()

        return results, failed
    
    # --------------------------------------------------------
    # PgBench
    # --------------------------------------------------------
    def exec_queries_pgbench(self):
        if self.remote_mode:
            return self.exec_queries_pgbench_remote()
        else:
            return self.exec_queries_pgbench_local()
    
    def exec_queries_pgbench_local(self):
        res: dict = {}
        failed = []
        for query_file_name in self.query_file_name:
            command = f"pgbench -f {query_file_name} {self.db_name} -n"
            result = run_as_postgre(command, self.postgre_pwd)
            if result.returncode != 0:
                print_log(f'[PGExecutor] Local execution of SQL query {query_file_name} using pgbench failed')
            res[query_file_name] = parse_pgbench_output(result.stdout)
        return res, failed

    def exec_queries_pgbench_remote(self):
        res: dict = {}
        failed = []
        remote_tmp_sql = '/tmp/tmp.sql'

        # Build SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(DB_HOST, username=POSTGRE_USER, password=POSTGRE_PWD,
                    disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
                
        for query_file_name in self.query_file_name:
            sftp = ssh.open_sftp()
            try:
                # Put local query file into a temporary file in remote server
                sftp.put(query_file_name, remote_tmp_sql)
            except IOError:
                print_log(f'[PGExecutor] Remote SFTP put SQL query {query_file_name} failed')
                if sftp: sftp.close()
                continue
            if sftp: sftp.close()

            # Execute pgbench command
            command = f"pgbench -f {remote_tmp_sql} {self.db_name} -n"
            _, stdout, stderr  = ssh.exec_command(command)
                                
            retcode = stdout.channel.recv_exit_status()
            if retcode != 0:
                print_log(f'[PGExecutor] Remote executing SQL query {query_file_name} using pgbench failed, with following information:')
                print_log(f'[PGExecutor] STDOUT: {stdout.read().decode("utf-8").strip()}')
                LOGGER.error(f'[PGExecutor] STDERR: {stderr.read().decode("utf-8").strip()}')
                failed.append(query_file_name)
                continue

            output = stdout.read().decode('utf-8')
            res[query_file_name] = parse_pgbench_output(output)
            print_log(f'[PGExecutor] Remote executing SQL query {query_file_name} using pgbench successfully')
                                
        if ssh: ssh.close()
        return res, failed

    # --------------------------------------------------------
    # Sysbench
    # --------------------------------------------------------
    def run_sysbench(self):
        if self.remote_mode:
            return self.run_sysbench_remote()
        else:
            return self.run_sysbench_local()
    
    def run_sysbench_local(self):
        sysbench_config: dict = SYSBENCH_WORKLOADS[self.sysbench_mode]
        script_path = os.path.join(SYSBENCH_SCRIPTS_DIR, sysbench_config['script'])
        command = SYSBENCH_CMD_RUN.format(
            script_path,
            DB_NAME, DB_USER, DB_PWD, DB_HOST,
            sysbench_config['tables'], sysbench_config['table_size'], sysbench_config['time']
        )
        result: CompletedProcess = run_as_postgre(command, POSTGRE_PWD)
        result_dict = parse_sysbench_output(result)
        return result_dict, []
    
    def run_sysbench_remote(self):
        raise NotImplementedError("[PGExecutor] Remote execution of sysbench is not supported yet")

    def prepare_sysbench_data(self):
        if self.remote_mode:
            raise NotImplementedError("[PGExecutor] (prepare) Remote execution of sysbench is not supported yet")
        else:
            print_log(f"[PGExecutor] Preparing sysbench for {self.sysbench_mode} mode")
            sysbench_config: dict = SYSBENCH_WORKLOADS[self.sysbench_mode]
            script_path = os.path.join(SYSBENCH_SCRIPTS_DIR, sysbench_config['script'])
            command = SYSBENCH_CMD_PREPARE.format(
                script_path,
                DB_NAME, DB_USER, DB_PWD, DB_HOST,
                sysbench_config['tables'], sysbench_config['table_size']
            )
            run_as_postgre(command, POSTGRE_PWD)
            # run_as_user(command, DB_USER, DB_PWD)

    def cleanup_sysbench_data(self):
        if self.remote_mode:
            raise NotImplementedError("[PGExecutor] (cleanup) Remote execution of sysbench is not supported yet")
        else:
            print_log(f"[PGExecutor] Cleaning up sysbench data for {self.sysbench_mode} mode")
            sysbench_config: dict = SYSBENCH_WORKLOADS[self.sysbench_mode]
            script_path = os.path.join(SYSBENCH_SCRIPTS_DIR, sysbench_config['script'])
            command = SYSBENCH_CMD_CLEANUP.format(
                script_path,
                DB_NAME, DB_USER, DB_PWD, DB_HOST,
                sysbench_config['tables'], sysbench_config['table_size']
            )
            run_as_postgre(command, POSTGRE_PWD)