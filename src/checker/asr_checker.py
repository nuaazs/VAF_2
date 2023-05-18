# coding = utf-8
# @Time    : 2023-04-20  14:23:37
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Check ASR api.

import requests
import cfg
import time
from minio import Minio

from utils.log import logger


def check():
    # read wav urls from minio bucket <testing>
    all_pass = True
    HOST = f"{cfg.MINIO['host']}:{cfg.MINIO['port']}"
    ACCESS_KEY = cfg.MINIO["access_key"]
    SECRET_KEY = cfg.MINIO["secret_key"]
    client = Minio(HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
    # get url list by client.list_objects("testing", recursive=True)
    # get first 10 urls
    urls = [f"http://{HOST}/testing/"+obj.object_name for obj in client.list_objects("testing", recursive=True)][:10]
    logger.info(f"** Check ASR api. Total wav num:{len(urls)}")
    time_used_list = []
    for wav_url in urls:
        logger.info(f"** Check ASR api. Wav url:{wav_url}")
        spkid = wav_url.split("/")[-1].split(".")[0]
        url = cfg.ASR_SERVER
        start = time.time()
        params = {"file_url": wav_url, "spkid": spkid}
        try:
            r = requests.request("POST", url, data=params)
            if r.status_code == 200:
                text = r.json()["corrected_result"]
            else:
                all_pass = False
                logger.info(f"** Check ASR api. Wav url:{wav_url} failed. Response:{r.json()}")
                text = ""
        except Exception as e:
            all_pass = False
            logger.info(f"** Check ASR api. Wav url:{wav_url} failed. Response:{e}")
            text = ""
        end = time.time()
        time_used = end - start
        time_used_list.append(time_used)
    mean_time_used = sum(time_used_list) / len(time_used_list)
    if all_pass:
        return True, f"** Check ASR api. Mean time used:{mean_time_used}"
    else:
        return False, f"** Check ASR api. Mean time used:{mean_time_used}"
    
        
