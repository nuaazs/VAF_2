# -*- coding: utf-8 -*-
import glob
import multiprocessing
import subprocess
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

LOG = Logger(f'logs/{dt}_main.log', level='debug').logger


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
    response = requests.request("POST", cfg.TEST_FILE_URL, files=request_file, data=values)
    if response.json()['status'] == 'success':
        LOG.info(f'File:{file},Response info:{response.text}')
    else:
        LOG.error(f'{file} request failed. Response info:{response.text}')
    mark_file(file)


def mark_file(spkid):
    """
    标记文件
    Args:
        file:

    Returns:

    """
    with open(mark_file_path, 'a') as f:
        f.writelines(f'{spkid}\n')  # os.getpid() 是进程编号


def main():
    start_time = time.time()
    files_original = glob.glob(cfg.WAV_PATH_GRAY + '/*.wav')
    if not files_original:
        LOG.error(f'No wav file in {cfg.WAV_PATH_GRAY}')
        return
    random.seed(123)
    random.shuffle(files_original)

    with open(f'logs/files_original-{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.txt', 'w') as f:
        f.writelines(f'{files_original}')

    if os.path.exists(mark_file_path):
        with open(mark_file_path, 'r') as f:
            processed_list = [line.strip() for line in f.readlines()]
        files = list(set(files_original).difference(set(processed_list)))
        LOG.info(f'Processed count:{len(processed_list)},remain count:{len(files)}')
    else:
        files = files_original

    LOG.info(f'Remain count:{len(files)}. Call time:{time.time() - start_time}')

    process_map(test_by_file, files, max_workers=cfg.WORKERS)

    # with multiprocessing.Pool(cfg.WORKERS) as p:
    #     p.map(test_by_file, files)

    # with multiprocessing.Pool(cfg.WORKERS) as p:
    #     list((tqdm(p.imap(test_by_file, files), total=len(files), desc='监视进度')))

    os.rename(mark_file_path, f'{mark_file_path}.{dt}')
    LOG.info(f'Rename {mark_file_path} to {mark_file_path}.{dt}')


if __name__ == "__main__":
    mark_file_path = 'logs/processed_list.txt'
    LOG.info(f'Start! Dir is:{cfg.WAV_PATH_GRAY}')
    t1 = time.time()
    main()
    LOG.info(f'Call time:{time.time() - t1}')
