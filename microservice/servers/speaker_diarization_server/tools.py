#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tools.py
@Time    :   2023/08/16 16:33:27
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

from collections import Counter
import glob
import shutil
import numpy as np
import pymysql
import torchaudio
from tqdm import tqdm
import cfg
from pydub import AudioSegment
from utils.oss.upload import upload_file
import requests
from loguru import logger


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    response = requests.request(method, url, files=files, data=data, json=json, headers=headers)
    response.raise_for_status()
    return response.json()


def extract_audio_segment(input_file, output_file, start_time, end_time):
    audio = AudioSegment.from_file(input_file)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    extracted_segment = audio[start_ms:end_ms]
    extracted_segment.export(output_file, format="wav")


def find_items_with_highest_value(dictionary):
    value_counts = Counter(dictionary.values())
    max_count = max(value_counts.values())
    for key, value in dictionary.items():
        if value_counts[value] == max_count:
            keys_with_max_value = value
    items_with_highest_value = {key: value for key, value in dictionary.items() if value_counts[value] == max_count}
    return items_with_highest_value, keys_with_max_value
