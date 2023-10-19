#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   register_date.py
@Time    :   2023/08/14 14:03:54
@Author  :   Carry
@Version :   1.0
@Desc    :   读取23月份人工审核的excel文件，下载文件并注册
'''

import requests
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from loguru import logger

logger.add("log/register_date_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

def main(record_info):
    """
    下载文件并注册
    """
    record_id = record_info[1]
    file_url = record_info[2]
    record_type = record_info[3]

    url = "http://192.168.3.169:8899/search/url"
    data = {
        'spkid': record_id,
        'wav_url': file_url,
    }
    # files = {'wav_file': open(file_path, 'rb')}
    # response = requests.post(file_url, files=files, data=data)

    response = requests.post(url, data=data)
    logger.info(response.text)


if __name__ == "__main__":
    with open("./record_ids.csv", "r") as f:
        spkids = f.readlines()
    spkids = [i.strip().split(",") for i in spkids]
    logger.info(f"Total spkids: {len(spkids)}")
    # for i in tqdm(spkids):
    #     main(i)

    
    r = process_map(main, spkids, max_workers=2)