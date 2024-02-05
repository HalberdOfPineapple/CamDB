import paramiko
from typing import List, Tuple

from camtune.utils import print_log, get_logger
from camtune.database.common_vars import *

from .utils import parse_pgbench_output, run_as_postgre
from .connector import PostgresqlConnector
from .variables import *

LOGGER = None

class PostgreExecutor():
    def __init__(self, benchmark: str, benchmark_fast: bool, use_pgbench: bool, remote_mode: bool):
        
        if benchmark.upper() not in QUERY_PATH_MAP:
            raise ValueError(f"[PGExecutor] Undefined Benchmark {self.benchmark}")
        
        self.benchmark = benchmark
        self.benchmark_fast = benchmark_fast

        self.use_pgbench = use_pgbench
        self.remote_mode = remote_mode
        self.query_dir = QUERY_PATH_MAP[self.benchmark]
        if self.use_pgbench:
            self.query_dir = self.query_dir + '_pgbench'

        global LOGGER
        LOGGER = get_logger()

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
            print_log(f"[PGExecutor] Exeucting queries listed in {query_list_file}")

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
        if self.remote_mode:
            return self.exec_queries_pgbench_remote(query_file_names)
        else:
            return self.exec_queries_pgbench_local(query_file_names)
    
    def exec_queries_pgbench_local(self, query_file_names: List[str]):
        res: dict = {}
        failed = []
        for query_file_name in query_file_names:
            command = f"pgbench -f {query_file_name} {self.db_name} -n"
            result = run_as_postgre(command, self.postgre_pwd)
            if result.returncode != 0:
                print_log(f'[PGExecutor] Local execution of SQL query {query_file_name} using pgbench failed')
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

    def exec_queries_pgbench_remote(self, query_file_names: List[str]):
        res: dict = {}
        failed = []
        remote_tmp_sql = '/tmp/tmp.sql'

        # Build SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(DB_HOST, username=POSTGRE_USER, password=POSTGRE_PWD,
                    disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
                
        for query_file_name in query_file_names:
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