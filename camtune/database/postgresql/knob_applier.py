import paramiko
import subprocess
import os
import json
import time
import socket
from typing import List, Tuple

from camtune.utils import print_log, get_logger
from camtune.database.config_parser import ConfigParser

from .connector import PostgresqlConnector
from .variables import *
from .utils import check_pg_running, kill_postgres, start_postgres, get_ssh_cli, run_as_postgre


class PostgreKnobApplier:
    def __init__(self, remote_mode: bool, knob_details: dict):
        self.remote_mode = remote_mode
        self.knob_details = knob_details
    
    def apply_knobs(self, knobs: dict, online: bool):
        if online:
            self.apply_knobs_online(knobs)
        else:
            self.apply_knobs_offline(knobs)

    def apply_knobs_offline(self, knobs: dict):
        if check_pg_running():
            success = kill_postgres(self.remote_mode)
            if not success:
                raise RuntimeError("[PGKnobApplier] PostgreSQL failed to shut down before applying knobs offline.")
        else:
            print_log("[PGKnobApplier] PostgreSQL server is not running, directly applying knobs offline")

        if 'min_wal_size' in knobs.keys():
            if 'wal_segment_size' in knobs.keys():
                wal_segment_size = knobs['wal_segment_size']
            else:
                wal_segment_size = 16
            if knobs['min_wal_size'] < 2 * wal_segment_size:
                knobs['min_wal_size'] = 2 * wal_segment_size
                print_log('[PGKnobApplier] Knob "min_wal_size" must be at least twice "wal_segment_size"')

        # --------------------------------------------------------------------
        # Adjust knobs values by modifying the configuration file offline
        knobs_not_in_cnf = self.modify_config_file(knobs)

        # --------------------------------------------------------------------
        # If PostgreSQL server cannot start normally, terminate the program
        success = start_postgres(self.remote_mode)
        if not success:
            raise RuntimeError("[PGKnobApplier] PostgreSQL failed to start after applying knobs offline.")

        print_log('[PGKnobApplier] Sleeping for {} seconds after restarting postgres'.format(RESTART_WAIT_TIME))
        time.sleep(RESTART_WAIT_TIME)

        # --------------------------------------------------------------------
        # Apply knobs that have not been written in configuration file online
        if len(knobs_not_in_cnf) > 0:
            tmp_rds = {}
            for knob_rds in knobs_not_in_cnf:
                tmp_rds[knob_rds] = knobs[knob_rds]
            self.apply_knobs_online(tmp_rds)
        else:
            print_log("[PGKnobApplier] No knobs need to be applied online")

        self.check_knobs_applied(knobs, online=False)

    def apply_knobs_online(self, knobs: dict):
        # apply knobs remotely
        print_log(f"[PGKnobApplier] Knobs to be applied online: {list(knobs.keys())}")
        db_conn: PostgresqlConnector = None
        try:
            db_conn = PostgresqlConnector()
            for key in knobs.keys():
                db_conn.set_knob_value(key, knobs[key])
            db_conn.close_db()
        except Exception as e:
            print_log(f"[PGKnobApplier] Online knob setting failed with information: {e}")

        self.check_knobs_applied(knobs, online=True)
    
    # --------------------------------------------------------------------
    # Knob checking
    def get_unit(self, knob_name: str) -> Tuple[str, str]:
        unit = None if 'unit' not in self.knob_details[knob_name] else self.knob_details[knob_name]['unit']
        byte_unit = None if 'byte_unit' not in self.knob_details[knob_name] else self.knob_details[knob_name]['byte_unit']
        return unit, byte_unit

    def check_knobs_applied(self, knobs: dict, online: bool) -> int:
        num_not_applied = 0
        db_conn: PostgresqlConnector = None
        try:
            db_conn = PostgresqlConnector()
            for knob, knob_val in knobs.items():
                if knob in self.knob_details:
                    unit, byte_unit = self.get_unit(knob)
                    applied, actual_val = db_conn.check_knob_apply(
                            knob, knob_val, unit=unit, byte_unit=byte_unit)
    
                    if not applied:
                        num_not_applied += 1
                        print_log(f"[PGKnobApplier] Knob {knob} is not successfully set to {knob_val} (actual value: {actual_val})")
            db_conn.close_db()
        except Exception as e:
            print_log(f"[PGKnobApplier] Knobs checking failed with exception information: {e}")
            if db_conn: db_conn.close_db()
            return -1

        check_mode = "online" if online else "offline"
        if num_not_applied > 0:
            print_log(f"[PGKnobApplier] {num_not_applied} / {len(knobs)} knobs not successfully applied {check_mode}.")
        elif num_not_applied == 0:
            print_log(f"[PGKnobApplier] Knobs successfully applied {check_mode}.")
        return num_not_applied
    

    # --------------------------------------------------------------------
    # Configuration file modification
    def modify_config_file(self, knobs: dict):
        if self.remote_mode: 
            return self.modify_config_file_remote(knobs)
        else:
            return self.modify_config_file_local(knobs)
        
    def write_config_file(self, cnf_path: str, knobs: dict) -> Tuple[dict, str]:
        # Update configuration file (locally)
        cnf_parser = ConfigParser(cnf_path)
        knobs_not_in_cnf = []
        for key in knobs.keys():
            if key not in self.knob_details.keys():
                knobs_not_in_cnf.append(key)
                continue
            cnf_parser.set(key, knobs[key])
        tmp_cnf_path = cnf_parser.replace()
        return knobs_not_in_cnf, tmp_cnf_path

    def modify_config_file_remote(self, knobs: dict):
        cnf = '/tmp/pglocal.cnf'
        ssh = get_ssh_cli()
        sftp = None
        try:
            sftp = ssh.open_sftp()
            sftp.get(PG_CONF, cnf) 
        except IOError:
            print_log('[PGKnobApplier] Remote SFTP get failed: PostgreSQL configuration file does not exist.')


        knobs_not_in_cnf: dict = self.write_config_file(cnf, knobs)

        try:
            # Note cnf is the local temporary file while PG_CONF is the remote file
            sftp = ssh.open_sftp()
            sftp.put(cnf, PG_CONF)
        except IOError as e:
            print_log(f'[PGKnobApplier] Remote SFTP put failed when applying knobs to config file: {e}.')

        if sftp: sftp.close()
        if ssh: ssh.close()

        print_log('[PGKnobApplier] config file modification done (remotely).', print_msg=True)
        return knobs_not_in_cnf

    
    def modify_config_file_local(self, knobs: dict):
        # Update configuration file (locally)
        knobs_not_in_cnf, tmp_cnf_path = self.write_config_file(PG_CONF, knobs)
        
        cp_cmd = 'cp {} {}'.format(tmp_cnf_path, PG_CONF)
        run_as_postgre(cp_cmd, POSTGRE_PWD)

        print_log('[PGKnobApplier] config file modification done (locally).', print_msg=True)
        return knobs_not_in_cnf

    
