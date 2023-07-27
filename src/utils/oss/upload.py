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

import cfg

HOST = f"{cfg.MINIO['host']}:{cfg.MINIO['port']}"
ACCESS_KEY = cfg.MINIO["access_key"]
SECRET_KEY = cfg.MINIO["secret_key"]
client = Minio(HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

def upload_file(
    bucket_name="raw",
    filepath="/VAF-System/demo_flask/utils/orm.py",
    filename="orm.py",
    save_days=30,
):
    # Upload data with tags, retention and legal-hold.
    date = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0,
    ) + timedelta(days=save_days)

    if save_days < 0:
        result = client.fput_object(bucket_name, filename, filepath, legal_hold=True,)
    else:
        result = client.fput_object(
            bucket_name,
            filename,
            filepath,
            retention=Retention(GOVERNANCE, date),
            legal_hold=True,
        )
    return f"http://{HOST}/{bucket_name}/{filename}"
