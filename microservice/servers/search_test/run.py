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
from loguru import logger

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time:YYYY-MM-DD_HH-mm}.log", rotation="500 MB", encoding="utf-8",
           enqueue=True, compression="zip", backtrace=True, diagnose=True)


def search_file(file_path):
    try:
        record_id = os.path.basename(file_path)
        # TODO:
        record_id = record_id.split('.')[0]
        data = {
            'spkid': record_id,
        }
        files = {'wav_file': open(file_path, 'rb')}
        response = requests.post(cfg.SEARCH_FILE_URL, files=files, data=data)
        logger.info(f'File:{file_path}.Response info:{response.json()}')
        if response.ok and response.json()['code'] == 500:
            with open('log/error.txt', 'a') as f:
                f.writelines(f'File:{file_path}.Response info:{response.json()}\n')
        mark_file(file_path)
    except Exception as e:
        logger.error(f'File:{file_path}. Error info:{e}')


def search_url(file_path):
    try:
        record_id = os.path.basename(file_path)
        # TODO:
        record_id = record_id.split('.')[0]
        data = {
            'spkid': record_id,
            'wav_url': f"http://192.168.3.199:9000/gray-raw/{os.path.basename(file_path)}"
        }
        response = requests.post(cfg.SEARCH_URL, data=data)
        logger.info(f'File:{file_path}.Response info:{response.json()}')
        if response.ok and response.json()['code'] == 500:
            with open('log/error.txt', 'a') as f:
                f.writelines(f'File:{file_path}.Response info:{response.json()}\n')
        mark_file(file_path)
    except Exception as e:
        logger.error(f'File:{file_path}. Error info:{e}')


def mark_file(spkid):
    """
    标记文件
    Args:
        file:

    Returns:

    """
    with lock:
        with open(mark_file_path, 'a') as f:
            f.writelines(f'{spkid}\n')  # os.getpid() 是进程编号


def main():
    start_time = time.time()
    files_original = glob.glob(cfg.WAV_PATH_GRAY + '/*.wav')
    files_original = sorted(files_original)
    # files_original = files_original*10
    # files_original = files_original[1:100]
    if not files_original:
        logger.error(f'No wav file in {cfg.WAV_PATH_GRAY}')
        return

    with open(f'log/files_original-{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.txt', 'w') as f:
        f.writelines(f'{files_original}')

    if os.path.exists(mark_file_path):
        with open(mark_file_path, 'r') as f:
            processed_list = [line.strip() for line in f.readlines()]
        files = list(set(files_original).difference(set(processed_list)))
        logger.info(f'Processed count:{len(processed_list)},remain count:{len(files)}')
    else:
        files = files_original

    logger.info(f'Remain count:{len(files)}. Call time:{time.time() - start_time}')

    with multiprocessing.Pool(cfg.WORKERS) as p:
        list((tqdm(p.imap(search_file, files), total=len(files), desc='监视进度')))

    # for file in files:
    #     search_url(file)

    # with multiprocessing.Pool(cfg.WORKERS) as p:
    #     p.map(test_by_file, files)

    # process_map(search_url, files, max_workers=cfg.WORKERS, chunksize=10, desc='进度tqdm:')

    # from multiprocessing.pool import ThreadPool
    # with ThreadPool(cfg.WORKERS) as p:
    #     p.map(test_by_file, files)

    time_str = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.rename(mark_file_path, f'{mark_file_path}_{time_str}')
    logger.info(f'Rename {mark_file_path} to {mark_file_path}_{time_str}')


if __name__ == "__main__":
    lock = multiprocessing.Lock()
    mark_file_path = 'log/processed_list.txt'
    logger.info(f'Start! Dir is:{cfg.WAV_PATH_GRAY}')
    t1 = time.time()
    main()
    logger.info(f'Call time:{time.time() - t1}')
