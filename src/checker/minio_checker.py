# coding = utf-8
# @Time    : 2022-09-05  15:08:59
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Minio upload files.

from datetime import timedelta
from minio.commonconfig import GOVERNANCE
from minio.retention import Retention
from datetime import datetime
from minio import Minio
from datetime import timedelta
from utils.log.log_wraper import logger
import os
import cfg


def check(bucket_name_list=cfg.BUCKETS):
    """
    Check minio connection and bucket.
    If bucket not exist, create it.
    For <testing> bucket, check if the file in bucket is the same as the file in local.
    If the file in bucket is not the same as the file in local, upload it.
    """
    try:
        logger.info("** -> Checking minio connection ... ")
        HOST = f"{cfg.MINIO['host']}:{cfg.MINIO['port']}"
        ACCESS_KEY = cfg.MINIO["access_key"]
        SECRET_KEY = cfg.MINIO["secret_key"]
        client = Minio(HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)
        for bucket_name in bucket_name_list:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"** -> Bucket:{bucket_name} not exist, now creating ... ")
                # Change bucket from "private" to "public"
                client.set_bucket_policy(bucket_name, "", "public")
            if bucket_name == "testing":
                for filename in os.listdir(cfg.TEST_WAVS_DIR):
                    filepath = os.path.join(cfg.TEST_WAVS_DIR, filename)
                    # check if filename already exist in minio bucket <testing>,
                    # if filename not exist in <testing> bucket, upload it
                    if client.stat_object(bucket_name, filename).st_size != os.stat(filepath).st_size:
                        client.fput_object(bucket_name, filename, filepath)
                        logger.info(f"** -> File:{filename} not exist in bucket:{bucket_name}, now uploading ... ")
                    else:
                        logger.info(f"** -> File:{filename} already exist in bucket:{bucket_name}, skip ... ")
        logger.info(f"** -> Minio test: Pass ! ")
        return True, ""
    except Exception as e:
        print(e)
        logger.error(f"** -> Minio test: Error !!! ")
        logger.error(f"** -> Minio Error Message: {e}")
        return False, e