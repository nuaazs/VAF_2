#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   check_file.py
@Time    :   2023/06/27 17:40:21
@Author  :   Carry
@Version :   1.0
@Desc    :   将指定路径下的wav文件进行注册筛选第一步，通过的存入new_check_list表，进行第二步人工筛选
'''
from tqdm import tqdm
import cfg
import glob
import requests
import json
import os
import pymysql
import sys

sys.path.append("/VAF/src")

msg_db = cfg.MYSQL


def inster_db(phone, response):
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cur = conn.cursor()

    query_sql = f"INSERT INTO new_check_list (phone, file_url,valid_length,preprocessed_file_url,register_time) \
                        VALUES (%s,%s,%s,%s,NOW());"
    try:
        args = (phone, response["file_url"], response["after_vad_length"], response["preprocessed_file_url"])
        cur.execute(query_sql, args)
        conn.commit()
    except Exception as e:
        print(e)
        conn.rollback()
    cur.close()
    conn.close()


def test_by_file(file):
    try:
        request_file = {"wav_file": open(file, "rb")}
        wav_url = f"local://{file}"
        phone = os.path.basename(file).split(".")[0].replace("_", "AAA")
        values = {
            "spkid": str(phone),
            "show_phone": "123",
            # "wav_url": wav_url,
            # "wav_channel": "1",
            "need_check": "1",
        }
        test_file_url = "http://192.168.3.169:7777/register/file"
        response = requests.request(
            "POST", test_file_url, files=request_file, data=values
        )
        if response.ok and response.json()["status"] == "success":
            print(f"File:{file}.Response info:{response.text}")
            inster_db(phone, response.json())
        elif response.ok and response.json()["status"] == "error":
            with open("logs/error_type.txt", "a") as f:
                f.writelines(f"File:{file}.Response info:{response.text}\n")
        else:
            with open("logs/requests_error.txt", "a") as f:
                f.writelines(
                    f"Status_code:{response.status_code}.File:{file}.Response info:{response.text}\n"
                )
    except Exception as e:
        print(f"File:{file},Error info:{e}")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    files = glob.glob("/datasets_hdd/datasets/auto_test_bak/*.wav")
    # files = glob.glob("/datasets_hdd/datasets/111679197.wav")
    files = sorted(files)
    files = files[:100]
    for file in tqdm(files):
        test_by_file(file)
