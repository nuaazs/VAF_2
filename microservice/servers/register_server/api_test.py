#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   api_test.py
@Time    :   2023/08/14 14:03:54
@Author  :   Carry
@Version :   1.0
@Desc    :   读取23月份人工审核的excel文件，下载文件并注册
'''

import glob
import os
import random
import phone
import requests
from tqdm import tqdm
import wget

def main(record_info):
    """
    下载文件并注册
    """
    record_id = record_info[1]
    file_url=record_info[2]
    record_type = record_info[3]

    file_path = f"/opt/datasets/1_spesker/mon2/{record_id}.wav"
    wget.download(file_url, file_path)

    file_url = "http://192.168.3.169:8899/register/file"
    files = {'wav_file': open(file_path, 'rb')}
    data = {
        'spkid': record_id,
        'record_month': 2,
        "use_asr":1,
        "record_type":record_type,
    }
    response = requests.post(file_url, files=files, data=data)
    print(response.text)


if __name__ == "__main__":
    with open("./record_ids.csv", "r") as f:
        spkids = f.readlines()
    print(f"Total spkids: {len(spkids)}")
    for i in tqdm(spkids):
        main(i)
        break