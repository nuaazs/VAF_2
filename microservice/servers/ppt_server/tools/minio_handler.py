#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   orm_handler.py
@Time    :   2023/10/14 20:04:03
@Author  :   Carry
@Version :   1.0
@Desc    :   minio操作
'''
from datetime import timedelta
from minio.commonconfig import GOVERNANCE
from minio.retention import Retention
from datetime import datetime
from minio import Minio
from datetime import timedelta
import os
import cfg

HOST = f"{cfg.MINIO['host']}:{cfg.MINIO['port']}"
ACCESS_KEY = cfg.MINIO["access_key"]
SECRET_KEY = cfg.MINIO["secret_key"]
client = Minio(HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)


def upload_file(bucket_name=None, filepath=None, filename=None, save_days=-1):
    assert bucket_name and filepath and filename, "Please check your params."
    if save_days < 0:
        result = client.fput_object(bucket_name, filename, filepath, legal_hold=True)
    else:
        date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0,) + timedelta(days=save_days)
        result = client.fput_object(bucket_name, filename, filepath, retention=Retention(GOVERNANCE, date), legal_hold=True)
    return f"http://{HOST}/{bucket_name}/{filename}"


def remove_urls_from_bucket(bucket_name="testing", urls=[]):
    for url in urls:
        filename = url.split("/")[-1]
        client.remove_object(bucket_name, filename)
    return True
