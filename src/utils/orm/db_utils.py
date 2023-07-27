import time
import pymysql
from dbutils.pooled_db import PooledDB
import cfg
from utils.log import logger

MYSQL = cfg.MYSQL


def call_time(func):
    def inner(*args, **kwargs):
        old_time = time.time()
        result = func(*args, **kwargs)
        func_name = str(func).split(" ")[1]
        print("{} use time: {}s".format(func_name, time.time() - old_time))
        # logger.info("{} use time: {}s".format(func_name, time.time() - old_time))
        return result

    return inner


class MySQLHandler(object):
    def __init__(self):
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=20,
            blocking=True,
            host=MYSQL["host"],
            port=MYSQL["port"],
            user=MYSQL["username"],
            password=MYSQL["passwd"],
            database=MYSQL["db"],
            charset="utf8",
        )

    def get_conn(self):
        conn = self.pool.connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        return conn, cursor

    def close_conn(self, conn, cursor):
        cursor.close()
        conn.close()

    def fetch_one(self, sql, args=None):
        conn, cursor = self.get_conn()
        cursor.execute(sql, args)
        res = cursor.fetchone()
        self.close_conn(conn, cursor)
        return res

    def fetch_all(self, sql, args=None):
        conn, cursor = self.get_conn()
        cursor.execute(sql, args)
        res = cursor.fetchall()
        self.close_conn(conn, cursor)
        return res

    def insert_one(self, sql, args=None):
        conn, cursor = self.get_conn()
        res = cursor.execute(sql, args)
        conn.commit()
        self.close_conn(conn, cursor)
        return res

    def insert_batch(self, sql, args=None):
        conn, cursor = self.get_conn()
        res = cursor.executemany(sql, args)
        conn.commit()
        self.close_conn(conn, cursor)
        return res

    def update(self, sql, args=None):
        conn, cursor = self.get_conn()
        res = cursor.execute(sql, args)
        conn.commit()
        self.close_conn(conn, cursor)
        return res

    def delete(self, sql, args=None):
        conn, cursor = self.get_conn()
        res = cursor.execute(sql, args)
        conn.commit()
        self.close_conn(conn, cursor)
        return res


mysql_handler = MySQLHandler()
