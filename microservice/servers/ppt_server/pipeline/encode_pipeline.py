#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   encode_pipeline.py
@Time    :   2023/10/14 22:12:44
@Author  :   Carry
@Version :   1.0
@Desc    :   获取音频特征流程
'''


import os
import shutil
from loguru import logger
import cfg
from tools.file_handler import get_audio_and_conver
from tools.vad_handler import energybase_vad
from tools.embedding_handler import encode_files


def encode_pipeline(request, filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    NEED_VAD = request.form.get('need_vad',  default=True, type=lambda x: x.lower() == 'true')  # 是否启动VAD
    channel = int(request.form.get('channel', 0))
    spkid = request.form.get('spkid')
    if not spkid:
        logger.error(f"spkid is None.")
        return {"code": 500, "spkid": spkid, "message": "spkid is None."}
    spkid = str(spkid).strip()

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

    if NEED_VAD:
        VAD_MIN_LENGTH = request.form.get('vad_min_length', default=10, type=int)  # VAD最小长度
        vad_file_path, vad_length, _ = energybase_vad(file_path, spkid_folder)
        if vad_length < VAD_MIN_LENGTH:
            logger.info(f"VAD length is less than {VAD_MIN_LENGTH}s. spkid:{spkid}. vad_length:{vad_length}")
            shutil.rmtree(spkid_folder)
            return {"code": 200, "spkid": spkid, "message": "VAD length is less than {}s. vad_length:{}".format(VAD_MIN_LENGTH, vad_length)}
        file_path = vad_file_path
    try:
        file_emb = encode_files(spkid, [file_path], need_list=True)
    except Exception as e:
        logger.error(f"Encode failed. spkid:{spkid}.response:{e}")
        return {"code": 500, "spkid": spkid, "message": "Encode failed. response:{}".format(e)}
    shutil.rmtree(spkid_folder)
    return {"code": 200, "spkid": spkid, "message": "success", "embedding": file_emb}
