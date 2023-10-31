#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   register.py
@Time    :   2023/10/14 22:01:35
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''

import datetime
import os
import shutil
from loguru import logger
import cfg
from tools.file_handler import get_audio_and_conver, extract_audio_segment
from tools.vad_handler import energybase_vad
from tools.embedding_handler import encode_files
from tools.orm_handler import inster_redis_db,  add_speaker, compare_handler
from tools.request_handler import send_request
from tools.minio_handler import upload_file


def register_pipeline(request, filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    NEED_LANG = request.form.get('need_lang', default=False, type=lambda x: x.lower() == 'true')  # 是否启动语言检测
    NEED_ASR = request.form.get('need_asr', default=False, type=lambda x: x.lower() == 'true')    # 是否启动语音识别
    NEED_COMPARE = request.form.get('need_compare', default=True, type=lambda x: x.lower() == 'true')    # 注册是否去重

    spkid = request.form.get('spkid')
    if not spkid:
        logger.error(f"spkid is None.")
        return {"code": 500, "spkid": spkid, "message": "spkid is None."}
    spkid = str(spkid).strip()

    record_month = str(request.form.get('record_month', str(datetime.datetime.now().month)))  # 音频文件的月份
    record_type = request.form.get('record_type', "")  # 音频文件的涉诈类型

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

    # do vad
    vad_file_path, vad_length, time_list = energybase_vad(file_path, spkid_folder)
    logger.info(f"spkid:{spkid}. After VAD vad_length:{vad_length}")
    if vad_length < cfg.VAD_MIN_LENGTH:
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)
        return {"code": 200, "spkid": spkid, "message": "VAD length is less than {}s. vad_length:{}".format(cfg.VAD_MIN_LENGTH, vad_length)}
    # cut 10s
    extract_audio_segment(vad_file_path, vad_file_path, 0, 10)

    # do lang classify
    if NEED_LANG:
        # TODO: todo test
        wav_files = ["local://" + vad_file_path]
        data = {"spkid": spkid, "filelist": ",".join(wav_files)}
        response = send_request(cfg.LANG_URL, data=data)
        if response and response['code'] == 200:
            pass_list = response['pass_list']
            url_list = response['file_url_list']
            mandarin_wavs = [i for i in url_list if pass_list[url_list.index(i)] == 1]
        else:
            logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            return {"code": 500, "spkid": spkid, "message": "Lang_classify failed. response:{}".format(response)}
    else:
        mandarin_wavs = [vad_file_path]

    # get embedding
    try:
        file_emb = encode_files(spkid, mandarin_wavs)
    except Exception as e:
        logger.error(f"Encode failed. spkid:{spkid}.response:{e}")
        return {"code": 500, "spkid": spkid, "message": "Encode failed. response:{}".format(e)}

    # compare with black base when register
    if NEED_COMPARE:
        model_type = list(cfg.BLACK_TH.keys())[0]
        black_limit = cfg.BLACK_TH[model_type]
        model_type = model_type.replace("_", "")
        emb_new = file_emb[model_type]
        compare_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=black_limit)
        logger.info(f"spkid:{spkid}. compare_result:{compare_results}")
        if compare_results['inbase']:
            logger.info(f"Speaker already exists. spkid:{spkid}. Compare result:{compare_results}")
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            return {"code": 200, "spkid": spkid, "message": "Speaker already exists.", "compare_result": compare_results}

    # do asr
    text = ""
    if NEED_ASR:
        data = {"spkid": spkid, "postprocess": "1"}
        files = [('wav_file', (file_path, open(file_path, 'rb')))]
        response = send_request(cfg.ASR_URL, files=files, data=data)
        if response and response.get('transcription') and response.get('transcription').get('text'):
            text = response['transcription']["text"]
        else:
            logger.error(f"ASR failed. spkid:{spkid}.message:{response['message']}")

    # upload to oss
    register_bucket_name = cfg.MINIO['register_bucket_name']
    raw_url = upload_file(register_bucket_name, file_path, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file(register_bucket_name, vad_file_path, f"{spkid}/vad_{spkid}.wav")

    # save to db
    db_info = {}
    db_info['spkid'] = spkid
    db_info['valid_length'] = vad_length
    db_info['raw_url'] = raw_url
    db_info['selected_url'] = selected_url
    db_info['record_month'] = record_month
    db_info['asr_text'] = text
    db_info['record_type'] = record_type
    if add_speaker(db_info):
        for model_type in cfg.ENCODE_MODEL_LIST:
            model_type = model_type.replace("_", "")
            emb_new = file_emb[model_type]
            inster_redis_db(embedding=emb_new, spkid=spkid, use_model_type=model_type)
        logger.info(f"Add speaker success. spkid:{spkid}")
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)
        return {"code": 200, "spkid": spkid, "file_url": raw_url, "preprocessed_file_url": selected_url, "message": "Add speaker success."}
    else:
        logger.error(f"Inser to mysql failed. Please check logs. spkid:{spkid}")
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)
        return {"code": 500, "spkid": spkid, "message": "Inser to mysql failed. Please check logs."}
