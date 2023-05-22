# -*- coding: utf-8 -*-
"""
@author: Carry
@file: config.py
@time: 2022/11/17 11:00 
@desc:
"""
import os

# 电信

# MINIO_HOST = 'http://172.16.185.59:9901'
#
# REQ_API_HOST = 'http://172.16.185.192:8888'  # 服务地址
#
# BUCKETS_NAME_BLACK = "black-raw"  # 黑库桶名
#
# BUCKETS_NAME_GRAY = "gray-raw"  # 灰库桶名
#
# WAV_PATH_BLACK = '/mnt/xuekx/workplace/voiceprint-recognition-system/src/api_test_bak/test_data/'
#
# WAV_PATH_GRAY = '/mnt/xuekx/workplace/voiceprint-recognition-system/src/api_test_bak/test_data/'

# 公司

MINIO_HOST = 'http://192.168.3.202:9000'
# 服务地址
TEST_URL = 'http://192.168.3.202:8190/test/url'
TEST_FILE_URL = 'http://192.168.3.202:8191/test/file'
# 黑库桶名
BUCKETS_NAME_BLACK = "black-raw"
# 灰库桶名
BUCKETS_NAME_GRAY = "gray-raw"

# 文件存放地址
WAV_PATH_GRAY = '/mnt/xuekx/test_data/13003661007'

WORKERS = 4
TEST_COUNT = 100
