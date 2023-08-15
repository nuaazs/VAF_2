#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   api_test.py
@Time    :   2023/08/14 14:03:54
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

import glob
import os
import random
import phone
import requests
from tqdm import tqdm

def main(file_path):
    file_url = "http://192.168.3.169:8899/register/file"
    spkid = os.path.basename(file_path).split(".")[0].split('-')[-1]
    files = {'wav_file': open(file_path, 'rb')}
    data = {
        'spkid': spkid,
        'record_month': "8",
    }
    response = requests.post(file_url, files=files, data=data)
    print(response.text)

   

if __name__ == "__main__":
    wav_files = glob.glob("/datasets/changzhou/*.wav")
    print(f"Total wav files: {len(wav_files)}")
    # wav_files = sorted(wav_files)
    # for i in tqdm(wav_files):
    #     main(i)
    #     break