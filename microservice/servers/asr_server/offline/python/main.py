#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/10/20 10:00:29
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

import shutil
from flask import Flask, request, jsonify,send_file
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
from loguru import logger
import torch
import cfg
from tools.file_handler import get_audio_and_conver

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

param_dict = dict()
param_dict['hotword'] = "hotword.txt"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict,
    device=cfg.DEVICE)


punc_inference_pipline = pipeline(
    task=Tasks.punctuation,
    model=f'damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    device=cfg.DEVICE
)


@app.route('/transcribe/<filetype>', methods=['POST'])
def transcribe(filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    spkid = request.form.get('spkid')
    if not spkid:
        logger.error(f"spkid is None.")
        return file_path({"code": 500, "spkid": spkid, "message": "spkid is None."})

    channel = int(request.form.get('channel', 0))
    NEED_PUNC = int(request.form.get('postprocess', 0))
    spkid_folder = f"{tmp_folder}/{spkid}"
    if filetype == "file":
        file_data = request.files.get('wav_file')
        if not file_data:
            logger.error(f"wav_file is None.")
            return file_path({"code": 500, "spkid": spkid, "message": "wav_file is None."})
        file_path = get_audio_and_conver(spkid, spkid_folder, file_data=file_data, channel=channel)
    elif filetype == "url":
        file_url = request.form.get('wav_url')
        if not file_url:
            logger.error(f"wav_url is None.")
            return jsonify({"code": 500, "spkid": spkid, "message": "wav_url is None."})
        file_path = get_audio_and_conver(spkid, spkid_folder, file_url=file_url, channel=channel)
    else:
        logger.error(f"filetype is not in ['file', 'url'].")
        return jsonify({"code": 500, "spkid": spkid, "message": "filetype is not in ['file', 'url']."})

    try:
        transcription = inference_pipeline(audio_in=file_path)
        if NEED_PUNC:
            transcription = punc_inference_pipline(text_in=transcription['text'])
        return jsonify({"code": 200, "spkid": spkid, "message": "success", "transcription": transcription})

    except Exception as e:
        logger.error(f"denoise error. spkid:{spkid}. error:{e}")
        return jsonify({"code": 500, "spkid": spkid, "message": f"denoise error. error:{e}"})
    finally:
        torch.cuda.empty_cache()
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)