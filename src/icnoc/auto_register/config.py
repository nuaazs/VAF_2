# -*- coding: utf-8 -*-
"""
@author: Carry
@file: config.py
@time: 2022/11/17 11:00 
@desc:
"""
import os 


MINIO_HOST = 'http://172.16.185.59:9901'
REQ_API_HOST = 'http://172.16.185.192:8888'  # 服务地址

BUCKETS_NAME_BLACK = "black-raw"  # 黑库桶名
BLACK_DIR_PATH = '/home/recbak/black'
WAV_PATH_BLACK = f"{BLACK_DIR_PATH}/{max(os.listdir(BLACK_DIR_PATH))}"  # 本地黑库文件夹路径/mnt/black/20221127 （取最新日期的文件夹）

BUCKETS_NAME_GRAY = "gray-raw"  # 灰库桶名
GRAY_DIR_PATH = '/home/recbak/gray'
WAV_PATH_GRAY = f"{GRAY_DIR_PATH}/{max(os.listdir(GRAY_DIR_PATH))}"  # 本地灰库文件夹路径/mnt/black/20221127 （取最新日期的文件夹）

