#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cmd_handler.py
@Time    :   2023/10/14 16:54:59
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

from loguru import logger
import requests


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: spkid:{data['spkid']}. msg:{e}.")
        return None
