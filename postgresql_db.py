import os
import json
import time
import glob
import threading
import subprocess
import paramiko
import logging
import numpy as np
import multiprocessing as mp
from getpass import getpass
from dbconnector import PostgresqlConnector
from utils.logger import get_logger
from utils.parser import ConfigParser

PG_CTL = "/usr/lib/postgresql/16/bin/pg_ctl"
PG_DATA = "/var/lib/postgresql/16/main"
PG_SERVER = "/usr/lib/postgresql/16/bin/postgres"
PG_CONF = "/var/lib/postgresql/experiment/conf/tune_cnf.conf"
PG_SOCK = "/var/run/postgresql/.s.PGSQL.543"

dst_data_path = os.environ.get("DATADST")
src_data_path = os.environ.get("DATASRC")
RESTART_WAIT_TIME = 0 # 20
TIMEOUT_CLOSE = 60

logger = get_logger('test_pgsql', './logs/output.log')
REMOTE_MODE = True
ISOLATION_MODE = False

# SSH user is set to postgres for allowing acces for pg_ctl and postgres server
# while the PostgresqlDB.user refers to the user of the database accessing data
SSH_USER = 'postgres'
SSH_PWD = '741286'

BENCHMARK_DIR = '/home/viktor/Experiments/CamDB/benchmarks'
BENCHMARK = 'TPCH'
QUERY_PATH_MAP = {
    'TPCH': os.path.join(BENCHMARK_DIR, 'tpch'),
}


def initialize_knobs(knobs_config, num):
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
    def __init__(self, 
                 host, port, user_name, db_name, passwd, 
                 knob_definition_path, knob_num=-1, remote_mode=REMOTE_MODE,
                 benchmark=BENCHMARK):
        self.host = host
        self.port = port
        self.user_name = user_name
        self.db_name = db_name
        self.passwd = passwd

        self.pg_ctl =  PG_CTL
        self.pgdata =  PG_DATA
        self.postgres = PG_SERVER
        self.pgcnf = PG_CONF
        self.pgsock = PG_SOCK

        # Note that query can be saved locally
        self.benchmark = BENCHMARK
        self.query_path = QUERY_PATH_MAP[self.benchmark]

        self.remote_mode = remote_mode
        if self.remote_mode:
            self.ssh_user = SSH_USER
            self.ssh_pwd = SSH_PWD
            # self.ssh_pk_file = os.path.expanduser('~/.ssh/id_rsa')
            # self.pk = paramiko.RSAKey.from_private_key_file(self.ssh_pk_file)
        
        self.knob_details = initialize_knobs(knob_definition_path, knob_num)

    def exec_queries(self):
        for query_file in glob.glob(self.query_path + '/*.sql'):
            with open(query_file, 'r') as f:
                query_lines = f.readlines()
            query = ' '.join(query_lines)
            print(query)

    
    def modify_config_file(self, knobs):
        if self.remote_mode:
            cnf = '/tmp/pglocal.cnf'
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})
    
            # Fetch PG configuration file to local through SFTP
            sftp = ssh.open_sftp()
            try:
                sftp.get(self.pgcnf, cnf) 
            except IOError:
                logger.info('PGCNF not exists!')

            if sftp: sftp.close()
            if ssh: ssh.close()
        else:
            cnf = self.pgcnf

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
            
            # Put newly generated configuration to SSH server
            sftp = ssh.open_sftp()
            try:
                sftp.put(cnf, self.pgcnf)
            except IOError:
                logger.info('PGCNF not exists!')
            if sftp: sftp.close()
            if ssh: ssh.close()

        logger.info('generated config file done')
        return knobs_not_in_cnf

    def kill_postgres(self):
        kill_cmd = '{} stop -D {}'.format(self.pg_ctl, self.pgdata)
        force_kill_cmd1 = "ps aux|grep '" + self.pgsock + "'|awk '{print $2}'|xargs kill -9"
        force_kill_cmd2 = "ps aux|grep '" + self.pgcnf + "'|awk '{print $2}'|xargs kill -9"

        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(kill_cmd)
            ret_code = ssh_stdout.channel.recv_exit_status()
            if ret_code == 0:
                logger.info("Close db successfully")
                print()
            else:
                logger.info("Force close DB!")
                ssh.exec_command(force_kill_cmd1)
                ssh.exec_command(force_kill_cmd2)

            ssh.close()
            logger.info('postgresql is shut down remotely')
        else:
            p_close = subprocess.Popen(kill_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                       close_fds=True)
            try:
                outs, errs = p_close.communicate(timeout=TIMEOUT_CLOSE)
                ret_code = p_close.poll()
                if ret_code == 0:
                    logger.info("Close db successfully")
            except subprocess.TimeoutExpired:
                logger.info("Force close!")
                os.system(force_kill_cmd1)
                os.system(force_kill_cmd2)
            logger.info('postgresql is shut down')
    
    def start_postgres(self):
        if self.remote_mode:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.host, username=self.ssh_user, password=self.ssh_pwd,
                        disabled_algorithms={'pubkeys': ['rsa-sha2-256', 'rsa-sha2-512']})

            start_cmd = '{} --config_file={} -D {}'.format(self.postgres, self.pgcnf, self.pgdata)
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
                    logger.info('add {} to memory,cpuset:server'.format(self.pid))
                else:
                    logger.info('Failed: add {} to memory,cpuset:server'.format(self.pid))

        else:
            proc = subprocess.Popen([self.postgres, '--config_file={}'.format(self.pgcnf), '-D',  self.pgdata])
            self.pid = proc.pid
            if ISOLATION_MODE:
                command = 'sudo cgclassify -g memory,cpuset:server ' + str(self.pid)
                p = os.system(command)
                if not p:
                    logger.info('add {} to memory,cpuset:server'.format(self.pid))
                else:
                    logger.info('Failed: add {} to memory,cpuset:server'.format(self.pid))

        count = 0
        start_success = True
        logger.info('wait for connection')
        while True:
            try:
                db_conn = PostgresqlConnector(host=self.host,
                                          port=self.port,
                                          user=self.user_name,
                                          passwd=self.passwd,
                                          name=self.db_name)
                db_conn = db_conn.conn
                if db_conn.closed == 0:
                    logger.info('Connected to PostgreSQL db')
                    db_conn.close()
                    break
            except:
                pass

            time.sleep(1)
            count = count + 1
            if count > 30: # 600
                start_success = False
                logger.info("can not connect to DB")
                clear_cmd = """ps -ef|grep postgres|grep -v grep|cut -c 9-15|xargs kill -9"""
                subprocess.Popen(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,
                                 close_fds=True)
                logger.info("kill all postgres process")
                break

        logger.info('finish {} seconds waiting for connection'.format(count))
        logger.info('postgres --config_file={}'.format(self.pgcnf))
        logger.info('postgres is up')
        return start_success

    def apply_knobs_online(self, knobs: dict):
        # self.restart_rds()
        # apply knobs remotely
        logger.info(f"Knobs to be applied online: {list(knobs.keys())}")
        db_conn = PostgresqlConnector(host=self.host,
                                        port=self.port,
                                        user=self.user_name,
                                        passwd=self.passwd,
                                        name=self.db_name)

        for key in knobs.keys():
            db_conn.set_knob_value(key, knobs[key])
        db_conn.close_db()

        self.check_knobs_applied(knobs)
        logger.info("[{}] Knobs applied online!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return True
    
    def apply_knobs_offline(self, knobs):
        self.kill_postgres()

        if 'min_wal_size' in knobs.keys():
            if 'wal_segment_size' in knobs.keys():
                wal_segment_size = knobs['wal_segment_size']
            else:
                wal_segment_size = 16
            if knobs['min_wal_size'] < 2 * wal_segment_size:
                knobs['min_wal_size'] = 2 * wal_segment_size
                logger.info('"min_wal_size" must be at least twice "wal_segment_size"')

        knobs_not_in_cnf = self.modify_config_file(knobs)
        success = self.start_postgres()

        try:
            logger.info('sleeping for {} seconds after restarting postgres'.format(RESTART_WAIT_TIME))
            time.sleep(RESTART_WAIT_TIME)

            if len(knobs_not_in_cnf) > 0:
                tmp_rds = {}
                for knob_rds in knobs_not_in_cnf:
                    tmp_rds[knob_rds] = knobs[knob_rds]
                self.apply_knobs_online(tmp_rds)
            else:
                logger.info("No Knobs need to be applied online")
        except:
            success = False

        self.check_knobs_applied(knobs)
        logger.info("[{}] Knobs applied offline!".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return success

    def check_knobs_applied(self, knobs: dict):
        db_conn = PostgresqlConnector(host=self.host,
                                        port=self.port,
                                        user=self.user_name,
                                        passwd=self.passwd,
                                        name=self.db_name)
        for k, v in knobs.items():
            applied, actual_val = db_conn.check_knob_apply(k, v)
            if not applied:
                logger.info(f"Knob {k} is not successfully set to {v} (actual value: {actual_val})")