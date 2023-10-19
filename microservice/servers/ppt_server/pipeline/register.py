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
from tools.file_handler import get_audio_and_conver, extract_audio_segment, get_joint_wav
from tools.orm_handler import inster_redis_db, add_speaker, compare_handler, check_phone
from tools.request_handler import send_request
from tools.minio_handler import upload_file
import phone as ph


def register_pipeline(request, filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    NEED_COMPARE = request.form.get('need_compare', default=False, type=lambda x: x.lower() == 'true')    # 注册是否去重
    phone = request.form.get('phone')
    name = request.form.get('name', "佚名")
    gender = request.form.get('gender', "")
    if not phone:
        logger.error(f"phone is None.")
        return {"code": 500, "phone": phone, "message": "手机号为空"}

    spkid = phone
    if check_phone(phone):
        logger.info(f"phone already exists. phone:{spkid}")
        return {"code": 200, "phone": spkid, "message": "手机号已存在"}

    spkid = str(spkid).strip().replace("_", "")
    channel = int(request.form.get('channel', 0))
    spkid_folder = f"{tmp_folder}/{spkid}"
    if filetype == "file":
        file_data = request.files.get('wav_file')
        if not file_data:
            logger.error(f"wav_file is None.")
            return {"code": 500, "phone": spkid, "message": "wav_file is None."}
        file_path = get_audio_and_conver(spkid, spkid_folder, file_data=file_data, start=1.5, channel=channel)
    elif filetype == "url":
        file_url = request.form.get('wav_url')
        if not file_url:
            logger.error(f"wav_url is None.")
            return {"code": 500, "spkid": spkid, "message": "wav_url is None."}
        file_path = get_audio_and_conver(spkid, spkid_folder, file_url=file_url, start=1.5, channel=channel)
    else:
        logger.error(f"filetype is not in ['file', 'url'].")
        return {"code": 500, "spkid": spkid, "message": "filetype is not in ['file', 'url']."}

    # step1 VAD
    data = {"spkid": phone}
    files = [('wav_file', (file_path, open(file_path, 'rb')))]
    response = send_request(cfg.VAD_URL, files=files, data=data)
    if response and response['code'] == 200:
        vad_length = response['vad_length']
        logger.info(f"VAD success. spkid:{phone}. vad_length:{vad_length}.file_path:{file_path}")
        if vad_length < cfg.VAD_MIN_LENGTH:
            logger.error(f"VAD failed. spkid:{phone}. vad_length:{vad_length}.file_path:{file_path}")
            return {"code": 7001, "phone": phone, "message": "VAD有效长度小于{}秒，当前{}秒".format(cfg.VAD_MIN_LENGTH, vad_length)}
    else:
        logger.error(f"VAD failed. spkid:{phone}. response:{response}")
        return {"code": 7002, "phone": phone, "message": "VAD请求失败"}

    # step2 截取音频片段
    output_file_li = []
    d = {}
    valid_length = 0
    for idx, i in enumerate(response['time_list']):
        output_file = f"{spkid_folder}/{phone}_{idx}.wav"  # 截取后的音频片段保存路径
        extract_audio_segment(file_path, output_file, start_time=i[0], end_time=i[1])
        output_file_li.append(output_file)
        valid_length += (i[1]-i[0])
        d[output_file] = (i[0], i[1])
    selected_path = get_joint_wav(spkid, spkid_folder, output_file_li)

    # get embedding
    data = {"spkid": phone}
    files = {'wav_file': open(selected_path, 'rb')}
    response = send_request(cfg.ENCODE_URL, data=data, files=files)
    if response and response['code'] == 200:
        file_emb = response['embedding']
    else:
        logger.error(f"Encode failed. spkid:{phone}.response:{response}")
        return {"code": 7004, "phone": phone, "message": "编码请求失败"}

    # compare with black base when register
    if NEED_COMPARE:
        model_type = list(cfg.BLACK_TH.keys())[0]
        emb_new = file_emb[model_type]["embedding"][spkid]
        compare_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=cfg.BLACK_TH[model_type])
        logger.info(f"spkid:{spkid}. compare_result:{compare_results}")
        if compare_results['inbase']:
            logger.info(f"Speaker already exists. spkid:{spkid}. Compare result:{compare_results}")
            # if os.path.exists(spkid_folder):
            #     shutil.rmtree(spkid_folder)
            return {"code": 200, "spkid": spkid, "message": "Speaker already exists.", "compare_result": compare_results}

    # upload to oss
    register_bucket_name = cfg.MINIO['register_bucket_name']
    raw_url = upload_file(register_bucket_name, file_path, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file(register_bucket_name, selected_path, f"{spkid}/vad_{spkid}.wav")

    if cfg.SHOW_PUBLIC:
        raw_url = raw_url.replace(cfg.HOST, cfg.PUBLIC_HOST)
        selected_url = selected_url.replace(cfg.HOST, cfg.PUBLIC_HOST)

    db_info = {}
    db_info['spkid'] = phone
    db_info['name'] = name
    db_info['gender'] = gender
    db_info['raw_url'] = raw_url
    db_info['selected_url'] = selected_url
    db_info['valid_length'] = valid_length
    try:
        info = ph.Phone().find(phone)
        phone_area = info['province'] + "-" + info['city']
    except Exception as e:
        phone_area = ""
    db_info['phone_area'] = phone_area

    if add_speaker(db_info):
        for model_type in cfg.ENCODE_MODEL_LIST:
            emb_new = file_emb[model_type]["embedding"][spkid]
            inster_redis_db(embedding=emb_new, spkid=spkid, use_model_type=model_type)
        logger.info(f"Add speaker success. spkid:{spkid}")
        # if os.path.exists(spkid_folder):
        #     shutil.rmtree(spkid_folder)
        return {"code": 200, "phone": phone, "message": "注册成功"}
    else:
        logger.error(f"Inser to mysql failed. Please check logs. spkid:{spkid}")
        # if os.path.exists(spkid_folder):
        #     shutil.rmtree(spkid_folder)
        return {"code": 500, "spkid": spkid, "message": "Inser to mysql failed. Please check logs."}
