#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   vad_pipeline.py
@Time    :   2023/10/14 22:43:24
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''


import os
import shutil
from loguru import logger
import cfg
from tools.file_handler import get_audio_and_conver
from tools.vad_handler import energybase_vad


def vad_pipeline(request, filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    smooth_threshold = float(request.form.get('smooth_threshold', 0.5))
    min_duration = float(request.form.get('min_duration', 2))
    energy_thresh = float(request.form.get('energy_thresh', 1e8))

    spkid = request.form.get('spkid')
    if not spkid:
        logger.error(f"spkid is None.")
        return {"code": 500, "spkid": spkid, "message": "spkid is None."}
    spkid = str(spkid).strip()

    channel = int(request.form.get('channel', 0))
    spkid_folder = f"{tmp_folder}/{spkid}"
    if filetype == "file":
        file_data = request.files.get('wav_file')
        if not file_data:
            logger.error(f"wav_file is None.")
            return {"code": 500, "spkid": spkid, "message": "wav_file is None."}
        file_path = get_audio_and_conver(spkid, spkid_folder, file_data=file_data, channel=channel)
    elif filetype == "url":
        file_url = request.form.get('wav_url')
        if not file_url:
            logger.error(f"wav_url is None.")
            return {"code": 500, "spkid": spkid, "message": "wav_url is None."}
        file_path = get_audio_and_conver(spkid, spkid_folder, file_url=file_url, channel=channel)
    else:
        logger.error(f"filetype is not in ['file', 'url'].")
        return {"code": 500, "spkid": spkid, "message": "filetype is not in ['file', 'url']."}

    _, vad_length, time_list = energybase_vad(file_path, spkid_folder, smooth_threshold, min_duration, energy_thresh)
    shutil.rmtree(spkid_folder)
    return {"code": 200, "spkid": spkid, "message": "success", "vad_length": vad_length, "time_list": time_list}
