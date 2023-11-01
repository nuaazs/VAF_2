<<<<<<< HEAD
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


ans = pipeline(
    Tasks.acoustic_noise_suppression,
    model='damo/speech_frcrn_ans_cirm_16k')

result = ans(
    '/datasets/icassp/noise_audio1.wav',
    output_path='denoise_audio1.wav')
result = ans(
    '/datasets/icassp/noise_audio2.wav',
    output_path='denoise_audio2.wav')
result = ans(
    '/datasets/icassp/noise_audio3.wav',
    output_path='denoise_audio3.wav')
result = ans(
    '/datasets/icassp/noise_audio4.wav',
    output_path='denoise_audio4.wav')
=======
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
import cfg
from tools.file_handler import get_audio_and_conver

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

ans = pipeline(Tasks.acoustic_noise_suppression, model='damo/speech_frcrn_ans_cirm_16k',device=cfg.DEVICE)

@app.route('/denoise/<filetype>', methods=['POST'])
def denoise(filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    spkid = request.form.get('spkid')
    if not spkid:
        logger.error(f"spkid is None.")
        return file_path({"code": 500, "spkid": spkid, "message": "spkid is None."})

    channel = int(request.form.get('channel', 0))
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
        output_path = f"{spkid_folder}/output.wav"
        result = ans(file_path, output_path=output_path)
        response = send_file(output_path, as_attachment=True)
        return response
    except Exception as e:
        logger.error(f"denoise error. spkid:{spkid}. error:{e}")
        return jsonify({"code": 500, "spkid": spkid, "message": f"denoise error. error:{e}"})
    finally:
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
>>>>>>> b09c63e15ee52b2e55d2d97613511b1f099a74ef
