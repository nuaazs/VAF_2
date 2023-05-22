import pymysql
import cfg
from utils.log.log_wraper import logger

def check():
    # check if mysql is connected
    logger.info("** -> Checking mysql connection ... ")
    try:
        conn = pymysql.connect(
            host=cfg.MYSQL["host"],
            port=cfg.MYSQL["port"],
            user=cfg.MYSQL["username"],
            password=cfg.MYSQL["passwd"],
            charset="utf8mb4",
        )
        with conn:
            with conn.cursor() as cursor:
                cursor.execute('show databases;')
                result = cursor.fetchall()
                now_dbs = tuple(x[0] for x in result)
                logger.info(f"** -> Now databases in MYSQL: {now_dbs}")
                for table_name in cfg.SQL_TABLES:
                    logger.info(f"** -> Checking {table_name} ... ")
                    if table_name not in now_dbs:
                        logger.info(f"** -> {table_name} not exist, creating ... ")
                        cursor.execute(f'create database {table_name};')
                        cursor.execute(f'use {table_name};')
                        sql_file = cfg.SQL_FILES[table_name]
                        a = 0
                        with open(sql_file) as f:
                            sql = ''
                            for i in f:
                                if i == '\n' or i[0] == '/' or i[0] == '-':
                                    pass
                                else:
                                    a = a + 1
                                    # 处理空行
                                    i = i.strip()
                                    i = i.strip('\r')
                                    i = i.strip('\n')
                                    # 构造字符串
                                    sql = sql + i
                                    # 判断此行sql语句中是否只含有 ‘；’ ，如果含有，则进行判断是否在结尾，反之，继续拼接
                                    if ';' in i:
                                        pot = i.rfind(';')
                                        if pot + 1 == len(i):
                                            cursor.execute(sql)
                                            sql = ''
        logger.info("** -> Mysql test: Pass ! ")
        return True,"Connected"
    except Exception as e:
        print(e)
        logger.error("** -> Mysql test: Error !!! ")
        logger.error(f"** -> Mysql Error Message: {e}")
        return False,e

