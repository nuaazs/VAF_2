#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   register_to_db.py
@Time    :   2023/06/29 08:58:05
@Author  :   Carry
@Version :   1.0
@Desc    :   将人工审核后的文件注册到数据库
'''
import glob
import requests
import os
import sys

from tqdm import tqdm

sys.path.append("/VAF/src")


def test_by_file(file):
    try:
        request_file = {"wav_file": open(file, "rb")}
        phone = os.path.basename(file).split(".")[0].replace("_", "AAA")
        values = {
            "spkid": str(phone),
            "show_phone": "123",
            # "wav_channel": "1",
        }
        test_file_url = "http://192.168.3.169:7777/register/file"
        response = requests.request(
            "POST", test_file_url, files=request_file, data=values
        )
        if response.ok and response.json()["status"] == "success":
            print(f"File:{file}.Response info:{response.text}")
        elif response.ok and response.json()["status"] == "error":
            with open("logs/register_to_db_error_type.txt", "a") as f:
                f.writelines(f"File:{file}.Response info:{response.text}\n")
        else:
            with open("logs/register_to_db_requests_error.txt", "a") as f:
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
