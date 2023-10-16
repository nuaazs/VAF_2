import pymysql

# 连接数据库
conn = pymysql.connect(
    host='sh-cynosdbmysql-grp-anrxxi7u.sql.tencentcdb.com',
    user='root',
    password='Nt3380518!zhaosheng123',
    database='shuadan',
    port=20264
)

# 创建游标对象
cursor = conn.cursor()

# 查询tid表格中product_ID为空的记录
query_sql = "SELECT user, wangwang, add_time FROM tid WHERE product_ID IS NULL"
cursor.execute(query_sql)
results = cursor.fetchall()

# 更新tid表格，补充product_ID

for row in results:
    print(row)
    user = row[0]
    wangwang = row[1]
    add_time = row[2] # datetime.datetime
    # get date from datetime
    add_time_date = add_time.strftime("%Y-%m-%d")
    query_sql2 = f"SELECT ID FROM ysy_log_v2 WHERE owner = '{user}' AND wangwang = '{wangwang}' AND type = 'done' AND date = '{add_time_date}'"
    cursor.execute(query_sql2)
    results2 = cursor.fetchall()
    # print(results2)
    if len(results2) == 1:
        for row2 in results2:
            print("*"*10)
            print(row2)
            product_id = row2[0]
            product_id = product_id.split("_")[0]
            update_sql = f"UPDATE tid SET product_ID = '{product_id}' WHERE user = '{user}' AND wangwang = '{wangwang}' AND add_time = '{add_time}'"
            cursor.execute(update_sql)
            conn.commit()
            print(update_sql)

    # # 如果查询结果中有多个符合条件的记录，则跳过
    # if product_id.count(',') > 0:
    #     continue

    # update_sql = f"""
    # UPDATE tid
    # SET product_ID = '{product_id}'
    # WHERE user = '{user}' AND wangwang = '{wangwang}'
    # """
    # cursor.execute(update_sql)

# 提交事务并关闭连接
conn.commit()
conn.close()
