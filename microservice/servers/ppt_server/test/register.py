#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   register.py
@Time    :   2023/08/30 10:42:16
@Author  :   Carry
@Version :   1.0
@Desc    :   将音频文件上传ppt注册接口
'''


import glob
import os
import requests
from faker import Faker
import phone as ph
faker = Faker(locale='zh_CN')
url = "http://192.168.3.169:8989/register/file"
headers = {
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
}

files = glob.glob("/datasets/cjsd_download/*/*.wav")
phone_set = set()
for i in files:
    phone = i.split("/")[-2]
    if phone not in phone_set and phone == "18351956663":
      try:
         info = ph.Phone().find(phone)
         phone_area = info['province'] + "-" + info['city']
      except Exception as e:
         phone_area = ""
      name = faker.name()
      filepath = i
      data = {"phone": phone, "name": name}
      files = [('wav_file', (filepath, open(filepath, 'rb')))]
      response = requests.request("POST", url, headers=headers, data=data, files=files)
      print(response.text)
      phone_set.add(phone)
