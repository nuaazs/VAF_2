#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   test_api.py
@Time    :   2023/10/26 15:52:07
@Author  :   Carry
@Version :   1.0
@Desc    :   vaf server test api
'''
import requests

def request_search(calling_number, file_path):
    url = "http://127.0.0.1:5550/search/file"
    data = {
        'spkid': calling_number,
    }
    files = {'wav_file': open(file_path, 'rb')}
    response = requests.post(url, files=files, data=data)
    print(response.text)
    if response.json().get("code") == 200:
        if response.json()['compare_result']['is_hit']:
            top_10 = response.json()['compare_result']['top_10']
            print(f"Hit black. calling_number:{calling_number}. top_10:{top_10}")
        else:
            print(f"Not hit black. calling_number:{calling_number}")
    else:
        print(f"Search file failed. calling_number:{calling_number}")


request_search("test_spkid", "./test_audio.wav")