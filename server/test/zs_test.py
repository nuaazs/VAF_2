#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2023/10/11 12:46:05
@Author  :   Carry
@Version :   1.0
@Desc    :   比对测试脚本
'''


import requests
import os
from tqdm import tqdm
import uuid


def do_requests(file_path_01, file_path_02, window_length):
    url = "http://192.168.3.169:7001/compare"

    payload = {
        'uuid': 	str(uuid.uuid4()),
        'window_length': window_length,
        'need_vad': 0,
    }
    files = [
        ('wav_files', (os.path.basename(file_path_01), open(file_path_01, 'rb'), 'audio/wav')),
        ('wav_files', (os.path.basename(file_path_02), open(file_path_02, 'rb'), 'audio/wav'))
    ]
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    return response.json()


root_path = '/VAF/model_test/data/test/cjsd300/'

with open('/home/xuekaixiang/workplace/cjsd300.trials', 'r') as f:
    lines = f.readlines()
    new_files = [i.strip() for i in lines]
print(len(new_files))

for second in [3, 5, 8, 10, 12, 15]:
    for i in tqdm(new_files[:2]):
        try:
            file_path_01 = root_path+i.split(' ')[0]
            file_path_02 = root_path+i.split(' ')[1]
            response = do_requests(file_path_01, file_path_02, window_length=second)
            score = response['score']
            os.makedirs(str(second), exist_ok=True)
            with open(f'{second}/result_score.txt', 'a') as f:
                f.write(i+' '+str(score)+'\n')
        except Exception as e:
            print(e)
    break
print('done')

