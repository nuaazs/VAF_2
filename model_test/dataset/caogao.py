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

# 下载目录
download_path = '/datasets_hdd/cj_downloadwavs'
# 定义下载信息日志文件名
download_log_file = '/datasets_hdd/cj_downloadwavs/download_log.txt'


def download(url, calling_number, call_start_time):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return False
        file_path = f"{save_path}/{calling_number}"
        os.makedirs(file_path, exist_ok=True)
        with open(f"{file_path}/{calling_number}_{call_start_time}.wav", "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        return False
max_id = 0
try:
    # 创建游标对象
    cursor = conn.cursor()

    # 定义SQL查询以获取 begintime 和 record_file_name 列的值
    # query = f"select cti_record.begintime,cti_record.timestamp,cti_record.record_file_name,cti_cdr_call.timestamp,cti_cdr_call.caller_num,cti_record.id from cti_record,cti_cdr_call   where cti_record.timestamp = cti_cdr_call.timestamp limit 500;"
    query = f"SELECT cti_record.begintime, cti_record.customer_uuid, cti_record.record_file_name, cti_cdr_call.call_uuid, cti_cdr_call.caller_num, cti_record.id FROM cti_record INNER JOIN cti_cdr_call ON cti_record.customer_uuid = cti_cdr_call.call_uuid WHERE STR_TO_DATE(cti_record.begintime, '%Y-%m-%d %H:%i:%s') > '2023-05-10 15:30:29';"

# 执行查询
    cursor.execute(query)

    # 获取结果集
    results = cursor.fetchall()
    # 关闭游标和数据库连接
    cursor.close()
    conn.close()
    print('ok')
    for row in results:
        begintime = row[0]
        record_file_name = row[2]
        phone = row[4]
        id = row[5]

        if id > max_id:
            max_id = id

        formatted_begintime = str(begintime).replace(' ', '').replace(':', '').replace('-', '')

        # 构建保存文件的路径
        file_name = f"{formatted_begintime}{phone}.wav"

        # 构建下载链接
        download_url = f"http://116.62.120.233/{record_file_name}"  # 替换为实际的下载链接

        # 确定存储文件夹路径
        folder_path = os.path.join(download_path, phone)

        # 只有在下载链接有效时才下载文件
        if download_url.startswith("http://") or download_url.startswith("https://"):
            save_path = ''
            # 使用 wget 下载文件，但只有在下载链接有效时才下载
            try:
                r = requests.get(download_url)
                if r.status_code != 200:
                    print('404',download_url)
                    continue
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                save_path = os.path.join(folder_path, file_name)
                with open(f"{save_path}", "wb") as f:
                    f.write(r.content)
                # 记录下载信息到日志文件
                with open(download_log_file, 'a') as log_file:
                    log_file.write(f"Downloaded {download_url} to {save_path}\n")
                print(f"Downloaded {download_url} to {save_path}")
            except Exception as e:
                print(f"Failed to download {download_url}: {str(e)}")
                if os.path.exists(save_path):
                    os.remove(save_path)
        else:
            print(f"Invalid download URL: {download_url}")



    with open('max_id.txt', 'w') as max_id_file:
        max_id_file.write(str(max_id))
except Exception as e:
    print(f"An error occurred: {str(e)}")
