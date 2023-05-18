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
LOG = Logger(f'{dt}_auto_test.log', level='debug').logger

MARK_FILE_NAME = 'test_id_list.txt'
MINIO_HOST = cfg.MINIO_HOST
REQ_API_HOST = cfg.REQ_API_HOST
TEST_URL = f'{REQ_API_HOST}/test/url'

BUCKETS_NAME_GRAY = cfg.BUCKETS_NAME_GRAY
WAV_PATH_GRAY = cfg.WAV_PATH_GRAY


# 本地linux的minio名称 -> 可在linux上面的 ~/.config/rclone/rclone.conf 文件中查看


def rclone_job(file):
    if file not in id_list:
        LOG.info(f'Start to process file:{file}')
        command = f"rclone sync {file} minio:/{BUCKETS_NAME_GRAY}"
        subprocess.call(command, shell=True)
        file_name = os.path.basename(file)
        payload = {
            'spkid': file_name.split('.')[0],
            'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_GRAY}/{file_name}'
        }
        response = requests.request("POST", TEST_URL, data=payload)
        if not response.ok:
            LOG.error(f'{file} request failed. Response info:{response.text}')
        else:
            LOG.info(f'File:{file},Response info:{response.text}')
        with open(MARK_FILE_NAME, 'a+') as f1:
            f1.write(file + '\n')


def run_mul():
    files = glob.glob(WAV_PATH_GRAY + '/*')
    pool = multiprocessing.Pool(4)
    pool.map(rclone_job, files)


def run():
    LOG.info(f'Gray wav path:{WAV_PATH_GRAY}')
    files = glob.glob(WAV_PATH_GRAY + '/*')
    files = files[:10]
    LOG.info(f'sss:{files[:10]}')
    for file in files:
        if file not in id_list:
            LOG.info(f'Start to process file:{file}')
            command = f"rclone sync {file} minio:/{BUCKETS_NAME_GRAY}"
            subprocess.call(command, shell=True)
            file_name = os.path.basename(file)
            payload = {
                'spkid': file_name.split('.')[0],
                'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_GRAY}/{file_name}'
            }
            response = requests.request("POST", TEST_URL, data=payload)
            if not response.ok:
                LOG.error(f'{file} request failed. Response info:{response.text}')
            else:
                LOG.info(f'File:{file},Response info:{response.text}')
            with open(MARK_FILE_NAME, 'a+') as f1:
                f1.write(file + '\n')


if __name__ == "__main__":
    LOG.info(f'Start!')
    import time

    t1 = time.time()
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
    LOG.info(f'Call time:{time.time() - t1}')
