import psycopg2
import re
import math
from abc import ABC, abstractmethod


class DBConnector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def connect_db(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def fetch_results(self, sql, json=True):
        pass

    @abstractmethod
    def execute(self, sql):
        pass

class PostgresqlConnector(DBConnector):
    def __init__(
      self, host='localhost', 
      port=5432, user='viktor', 
      passwd='741286', name='tpch', 
      socket=''
    ):
        super().__init__()
        self.dbhost = host
        self.dbport = port
        self.dbuser = user
        self.dbpasswd = passwd
        self.dbname = name
        self.sock = socket

        self.conn = self.connect_db()
        if self.conn:
            self.cursor = self.conn.cursor()

    def connect_db(self):
        conn = False
        if self.sock:
            conn = psycopg2.connect(host=self.dbhost,
                                    user=self.dbuser,
                                    password=self.dbpasswd,
                                    database=self.dbname,
                                    port=self.dbport,
                                    unix_socket=self.sock)
        else:
            conn = psycopg2.connect(host=self.dbhost,
                                    user=self.dbuser,
                                    password=self.dbpasswd,
                                    database=self.dbname,
                                    port=self.dbport)
        return conn

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def fetch_results(self, sql, json=True):
        results = False
        if self.conn:
            # print("Starting query execution...")
            self.cursor.execute(sql)
            # print("Query execution finished.")
            
            results = self.cursor.fetchall()
            
            if json:
                columns = [col[0] for col in self.cursor.description]
                return [dict(zip(columns, row)) for row in results]
        return results

    def execute(self, sql):
        results = False
        if self.conn:
            self.cursor.execute(sql)
    
    def check_knob_apply(self, k, v0):
        sql = 'SHOW {};'.format(k)
        r = self.fetch_results(sql)
        print(r)

        if len(r) == 0 or k not in r[0]:
            raise ValueError(
                f"[PostgresqlConnector] Knob {k} is not correctly detected on DBMS")

        # sample r: [{'backend_flush_after': '856kB'}]
        if r[0][k] == 'ON':
            vv = 1
        elif r[0][k] == 'OFF':
            vv = 0
        else:
            vv = r[0][k].strip()

        actual_val = re.findall(r"[-+]?\d*\.\d+|\d+", vv)
        if isinstance(v0, int):
            applied = int(actual_val[0]) == v0
        elif isinstance(v0, float):
            applied = math.isclose(float(actual_val[0]), v0, rel_tol=1e-5)
        else:
            applied = vv == v0

        return applied, vv

    def set_knob_value(self, k, v):
        sql = 'SHOW {};'.format(k)
        r = self.fetch_results(sql)

        # type convert
        if v == 'ON':
            v = 1
        elif v == 'OFF':
            v = 0

        if r[0][k] == 'ON':
            v0 = 1
        elif r[0][k] == 'OFF':
            v0 = 0
        else:
            try:
                v0 = eval(r[0][k])
            except:
                v0 = r[0][k].strip()
        if v0 == v:
            return True