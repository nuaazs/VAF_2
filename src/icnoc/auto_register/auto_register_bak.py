import glob
import multiprocessing
import subprocess
import time
import os
from datetime import datetime
import requests
from logs import Logger
import config as cfg

dt = datetime.now().strftime('%Y-%m-%d')
LOG = Logger(f'{dt}_auto_register.log', level='debug').logger

MARK_FILE_NAME = 'register_id_list.txt'
MINIO_HOST = cfg.MINIO_HOST
REQ_API_HOST = cfg.REQ_API_HOST
REGISTER_URL = f'{REQ_API_HOST}/register/url'

BUCKETS_NAME_BLACK = cfg.BUCKETS_NAME_BLACK
WAV_PATH_BLACK = cfg.WAV_PATH_BLACK


# 本地linux的minio名称 -> 可在linux上面的 ~/.config/rclone/rclone.conf 文件中查看


def rclone_job(file):
    if file not in id_list:
        LOG.info(f'Start to process file:{file}')
        command = f"rclone sync {file} minio:/{BUCKETS_NAME_BLACK}"
        subprocess.call(command, shell=True)
        file_name = os.path.basename(file)
        payload = {
            'spkid': file_name.split('.')[0],
            'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_BLACK}/{file_name}'
        }
        response = requests.request("POST", REGISTER_URL, data=payload)
        if not response.ok:
            LOG.error(f'{file} request failed. Response info:{response.text}')
        with open(MARK_FILE_NAME, 'a+') as f1:
            f1.write(file + '\n')


def run_mul():
    files = glob.glob(WAV_PATH_BLACK + '/*')
    pool = multiprocessing.Pool(4)
    pool.map(rclone_job, files)


def run():
    files = glob.glob(WAV_PATH_BLACK + '/*')
    for file in files:
        if file not in id_list:
            LOG.info(f'Start to process file:{file}')
            command = f"rclone sync {file} minio:/{BUCKETS_NAME_BLACK}"
            subprocess.call(command, shell=True)
            file_name = os.path.basename(file)
            payload = {
                'spkid': file_name.split('.')[0],
                'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_BLACK}/{file_name}'
            }
            response = requests.request("POST", REGISTER_URL, data=payload)
            if not response.ok:
                LOG.error(f'{file} request failed. Response info:{response.text}')
            with open(MARK_FILE_NAME, 'a+') as f1:
                f1.write(file + '\n')


if __name__ == "__main__":
    LOG.info(f'Start!')
    ret = subprocess.Popen('ps aux|grep rclone', shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, encoding='utf-8').communicate()

    while 'rclone sync' in ret[0]:
        time.sleep(60)
        ret = subprocess.Popen('ps aux|grep rclone', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding='utf-8').communicate()

    if not os.path.exists(MARK_FILE_NAME):
        with open(MARK_FILE_NAME, 'w') as f: pass
    with open(MARK_FILE_NAME, 'r+') as f:
        id_list = f.readlines()
        id_list = [i.strip() for i in id_list]
    run()
