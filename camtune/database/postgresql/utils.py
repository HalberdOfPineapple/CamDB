import subprocess
import re
import socket
import time
import paramiko
import os

from camtune.utils import print_log
from camtune.database.utils import run_as_user

from .variables import *
from .connector import PostgresqlConnector

KILL_CMD = '{} stop -D {}'.format(PG_CTL, PG_DATA)

def run_as_postgre(command, password):
    return run_as_user(command, 'postgres', password)

def check_pg_running():
    proc = subprocess.Popen(["pgrep -u postgres -f -- -D"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    return len(out.strip().decode()) > 0

def parse_pgbench_output(output):
    parsed_data = {}
    patterns = {
        'transaction_type': r'transaction type:\s*(.+)',
        'scaling_factor': r'scaling factor:\s*(.+)',
        'query_mode': r'query mode:\s*(.+)',
        'number_of_clients': r'number of clients:\s*(\d+)',
        'number_of_threads': r'number of threads:\s*(\d+)',
        'number_of_transactions': r'number of transactions actually processed:\s*(\d+)/\d+',
        'failed_transactions': r'number of failed transactions:\s*(\d+)',
        'latency_average': r'latency average = ([\d.]+)\s*ms',
        'initial_connection_time': r'initial connection time = ([\d.]+)\s*ms',
        'tps': r'tps = ([\d.]+)\s*'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            parsed_data[key] = match.group(1)

    return parsed_data

def start_pg_default(password):
    sudo_command = f"echo {password} | sudo service postgresql start"
    result = subprocess.run(sudo_command, shell=True,
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result

def get_ssh_cli():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(DB_HOST, username=POSTGRE_USER, password=POSTGRE_PWD,
                disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
    return ssh

def recover_default_config(remote_mode: bool):
    if remote_mode :
        ssh = get_ssh_cli()
        sftp = None
        try:
             # Put newly generated configuration to SSH server
            sftp = ssh.open_sftp()

            # Note cnf is the local temporary file while `self.pg_cnf` is the remote file
            sftp.put(PG_DEFAULT_CONF, PG_CONF)
        except IOError as e:
            print_log(f'[PostgresqlDB] Remote SFTP put failed when recovering configuration file to default: {e}.')

        if sftp: sftp.close()
        if ssh: ssh.close()
    else:
        cp_cmd = 'cp {} {}'.format(PG_DEFAULT_CONF, PG_CONF)
        run_as_postgre(cp_cmd, POSTGRE_PWD)

# --------------------------------------------------------------------
def kill_postgres(remote_mode: bool):
    if remote_mode:
        return kill_postgres_remote()
    else:
        return kill_postgres_local()
            
def kill_postgres_remote():
    ssh = get_ssh_cli()
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(KILL_CMD)
    ret_code = ssh_stdout.channel.recv_exit_status()
    if ret_code == 0:
        print_log("[PGUtils] Remote PostgreSQL server shut down successfully")
        print_log('\n')
    else:
        print_log("[PGUtils] Failed to shut down PostgreSQL server")
        return False

    ssh.close()
    return True

def kill_postgres_local():
    kill_cmd = ['sudo', '-u', POSTGRE_USER] + KILL_CMD.split()
    kill_cmd = f"echo {POSTGRE_PWD} | {' '.join(kill_cmd)}"
    p_close = subprocess.Popen(kill_cmd, shell=True, 
                                       stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)

    outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
    ret_code = p_close.poll()
    if ret_code == 0:
        print_log("[PGUtils] Local PostgreSQL server shut down successfully", print_msg=True)
    else:
        print_log(f"[PGUtils] Local shut down attempt ({kill_cmd}) failed with output: {outs.decode('utf-8')}", print_msg=True)
    return ret_code == 0

# --------------------------------------------------------------------
def start_postgres(remote_mode: bool):
    if remote_mode:
        if not start_postgres_remote():
            return False
    else:
        start_postgres_local()

    # return True
    return try_connect_pg()

def start_postgres_local():
    print_log('[PGUtils] Locally starting PostgreSQL server...', print_msg=True)
    launch_cmd = f"sudo -S -u postgres {PG_SERVER} --config_file={PG_CONF} -D {PG_DATA}"
    print_log(f'[PGUtils] Launch command: {launch_cmd}')
    proc = subprocess.Popen(f"echo {POSTGRE_PWD} | {launch_cmd}", 
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def try_connect_pg():
    count = 0
    start_success = True
    print_log('[PGUtils] Wait for connection to the started server...')
    while True:
        try:
            db_conn = PostgresqlConnector()
            db_conn = db_conn.conn
            if db_conn.closed == 0:
                print_log('[PGUtils] Successfully connected to the started PostgreSQL Server')
                db_conn.close()
                break
        except:
            pass

        time.sleep(1)
        count = count + 1
        if count > 10: # 30
            start_success = False
            print_log(f"[PGUtils] Failed to connect to newly-started PG server after {count} tries")
            break

    print_log('[PGUtils] {} seconds waiting for connection'.format(count))
    print_log(f'[PGUtils] PostgresSQL is {"successfully" if start_success else "not"} launched.', print_msg=True)
    
    return start_success

def start_postgres_remote():
    print_log('[PGUtils] Remotely starting PostgreSQL server...')
    ssh = get_ssh_cli()
    start_cmd = '{} --config_file={} -D {}'.format(PG_SERVER, PG_CONF, PG_DATA)
    wrapped_cmd = 'echo $$; exec ' + start_cmd
    _, start_stdout, start_stderr = ssh.exec_command(wrapped_cmd)


    # Check if there was an error
    stderr_output = start_stderr.read().decode()
    if stderr_output:
        print_log("[PGUtils] Error occurred while starting PostgreSQL server remotely: " + stderr_output, print_msg=True)
        return False
    return True