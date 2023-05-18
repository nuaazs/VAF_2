"""
@file: test.py
@time: 2021/8/31 15:00
@desc:各模块时间测试
"""
# -*- coding: utf-8 -*-
import glob
import json
import multiprocessing
import subprocess
import sys
import time
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import random
from tqdm.contrib.concurrent import process_map
import requests
from tqdm import tqdm
import config as cfg
from logs_file import Logger

dt = datetime.now().strftime('%Y-%m-%d')
LOG = Logger(f'logs/{dt}_test.log', level='debug').logger


def test_by_file(file):
    """
    通过文件上传测试
    Args:
        file:

    Returns:

    """
    request_file = {"wav_file": open(file, "rb")}
    wav_url = f"local://{file}"
    phone = os.path.basename(file).split("_")[0]
    values = {
        "spkid": str(phone),
        # 'spkid': str(random.randint(10000000000, 99999999999)),
        "wav_url": wav_url,
    }
    response = requests.request(
        "POST", cfg.TEST_FILE_URL, files=request_file, data=values)
    if response.json()['status'] == 'success':
        LOG.info(f'File:{file},Response info:{response.text}')
        mark_file(response)
    else:
        LOG.error(f'{file} request failed. Response info:{response.text}')


def mark_file(response):
    """
    标记文件
    Args:
        file:

    Returns:

    """
    count = 0
    for key, value in response.json()['used_time'].items():
        count += value
    other_total = response.elapsed.total_seconds() - count
    context = response.json()['used_time']
    context['other_total'] = other_total
    with open(mark_file_path, 'a') as f:
        f.writelines('{}\n'.format(context))


def main():
    start_time = time.time()
    files_original = glob.glob(cfg.WAV_PATH_GRAY + '/*.wav')
    if not files_original:
        LOG.error(f'No wav file in {cfg.WAV_PATH_GRAY}')
        return
    random.seed(123)
    random.shuffle(files_original)
    files = files_original[:cfg.TEST_COUNT]

    LOG.info(
        f'Remain count:{len(files)}. Call time:{time.time() - start_time}')

    process_map(test_by_file, files, max_workers=cfg.WORKERS)

    # with multiprocessing.Pool(cfg.WORKERS) as p:
    #     list((tqdm(p.imap(test_by_file, files), total=len(files), desc='监视进度')))


def parse_data():
    """
    解析数据
    Returns:

    """
    with open(mark_file_path, 'r') as f:
        li = f.readlines()
    dict_list = [json.loads(i.strip().replace("'", '"')) for i in li]
    sum_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in sum_dict:
                sum_dict[key] = value
            else:
                sum_dict[key] += value

    # 输出每个 key 的总和
    LOG.info(f'Count:{len(dict_list)}')
    for key, value in sum_dict.items():
        LOG.info(f'Mean {key}:{round(value / len(li), 2)}')


if __name__ == "__main__":
    mark_file_path = 'logs/test_report.txt'
    LOG.info(f'Start! Dir is:{cfg.WAV_PATH_GRAY}')
    t1 = time.time()
    if os.path.exists(mark_file_path):
        os.remove(mark_file_path)
    main()
    parse_data()
    LOG.info(f'Call time:{time.time() - t1}')
