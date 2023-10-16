import pandas as pd
import pyodbc


import pymysql
import re
import os
import requests
import wget

# 数据库连接信息
host = '116.62.120.233'
port = 3306
user = 'changjiangsd'
password = 'changjiangsd9987'
database = 'cticdr'


# 连接数据库
conn = pymysql.connect(host=host, port=port, user=user, passwd=password, db=database)


query = "SELECT * FROM cti_record"
df = pd.read_sql(query, conn)

conn.close()

df.to_csv("cti_record.csv", index=False)