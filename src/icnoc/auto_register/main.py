# -*- coding: utf-8 -*-
# 本地linux的minio名称 -> 可在linux上面的 ~/.config/rclone/rclone.conf 文件中查看

import glob
import multiprocessing
import subprocess
import time
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import random

import requests
from logs import Logger

from minio import Minio

dt = datetime.now().strftime('%Y-%m-%d')
LOG = Logger(f'{dt}_auto_test.log', level='debug').logger

# 公司
MINIO_HOST = 'http://192.168.3.202:9000'
TEST_URL = 'http://192.168.3.202:8190/test/url'  # 服务地址
BUCKETS_NAME_BLACK = "black-raw"  # 黑库桶名
BUCKETS_NAME_GRAY = "gray-raw"  # 灰库桶名
WAV_PATH_GRAY = '/home/recbak/gray/20221127'

client = Minio(
    "192.168.3.202:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)


def rclone_job(file):
    # command = f"rclone sync {file} minio:/{BUCKETS_NAME_GRAY}"
    # subprocess.call(command, shell=True)

    file_name = os.path.basename(file)
    payload = {
        # 'spkid': file_name.split('.')[0].replace('_', ''),
        'spkid': str(random.randint(10000000000, 99999999999)),
        'show_phone': '123',
        'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_GRAY}/{file_name}'
    }
    response = requests.request("POST", TEST_URL, data=payload)
    if not response.ok:
        LOG.error(f'{file} request failed. Response info:{response.text}')
    else:
        LOG.info(f'File:{file},Response info:{response.text}')
        with open('output.txt', 'a') as f:
            f.writelines('{},{}\n'.format(response.json()['used_time'], os.getpid()))  # os.getpid() 是进程编号


def test_by_file(file):
    """
    通过文件上传测试
    Args:
        file:

    Returns:

    """
    url = 'http://192.168.3.202:8190/test/file'
    request_file = {"wav_file": open(file, "rb")}
    wav_url = f"local://{file}"
    phone = os.path.basename(file).split("_")[0]
    values = {
        # "spkid": str(phone),
        'spkid': str(random.randint(10000000000, 99999999999)),
        "wav_url": wav_url,
    }
    response = requests.request("POST", url, files=request_file, data=values)
    if not response.ok:
        LOG.error(f'{file} request failed. Response info:{response.text}')
    else:
        LOG.info(f'File:{file},Response info:{response.text}')
        with open('output.txt', 'a') as f:
            f.writelines('{},{}\n'.format(response.json()['used_time'], os.getpid()))  # os.getpid() 是进程编号


def main():
    start_time = time.time()
    # 方式2
    # files = glob.glob(WAV_PATH_GRAY + '/*')
    files = ['/mnt/xuekx/workplace/voiceprint-recognition-system/src/api_test/test_data/13003661007/13003661007_1.wav'
             for i in range(10)]
    print(f'Total count:{len(files)},call time:{time.time() - start_time}')

    pool = multiprocessing.Pool(1)
    pool.map(rclone_job, files)
    # pool.map(test_by_file, files)

    # from multiprocessing.pool import ThreadPool
    # pool = ThreadPool(16)
    # pool.map(rclone_job, files)

    # with ThreadPoolExecutor(max_workers=16) as t:
    #     obj_list = []
    #     begin = time.time()
    #     for page in files:
    #         obj = t.submit(rclone_job, page)
    #         obj_list.append(obj)


if __name__ == "__main__":
    LOG.info(f'Start! Dir is:{WAV_PATH_GRAY}')
    t1 = time.time()
    main()
    LOG.info(f'Call time:{time.time() - t1}')
