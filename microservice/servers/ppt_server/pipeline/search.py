#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   search.py
@Time    :   2023/10/15 21:06:16
@Author  :   Carry
@Version :   1.0
@Desc    :   1:N检索
'''


import os
from loguru import logger
import numpy as np
import cfg
from tools.file_handler import get_audio_and_conver, extract_audio_segment, get_joint_wav
from tools.orm_handler import get_spkid, get_embeddings_from_spkid, check_phone, get_name_from_phone, add_hit
from tools.request_handler import send_request
from tools.minio_handler import upload_file


def search_pipeline(request, filetype):
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

    spkid = str(spkid).strip()
    channel = int(request.form.get('channel', 0))
    spkid_folder = f"{tmp_folder}/{spkid}"
    if filetype == "file":
        file_data = request.files.get('wav_file')
        if not file_data:
            logger.error(f"wav_file is None.")
            return {"code": 500, "phone": spkid, "message": "wav_file is None."}
        file_path = get_audio_and_conver(spkid, spkid_folder, file_data=file_data, start=0.5, channel=channel)
    elif filetype == "url":
        file_url = request.form.get('wav_url')
        if not file_url:
            logger.error(f"wav_url is None.")
            return {"code": 500, "spkid": spkid, "message": "wav_url is None."}
        file_path = get_audio_and_conver(spkid, spkid_folder, file_url=file_url, start=0.5, channel=channel)
    else:
        logger.error(f"filetype is not in ['file', 'url'].")
        return {"code": 500, "spkid": spkid, "message": "filetype is not in ['file', 'url']."}

    # step1 VAD
    data = {"spkid": phone, "min_duration": 0.25}
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

    register_list = list(get_spkid())

    final_score_list = []
    for i in register_list:
        embeddings_db = get_embeddings_from_spkid(i)
        final_score = 0
        for model_type in cfg.ENCODE_MODEL_LIST:
            emb_db = embeddings_db[model_type]
            emb_new = file_emb[model_type]["embedding"][spkid]
            hit_score = np.dot(emb_new, emb_db) / (np.linalg.norm(emb_new) * np.linalg.norm(emb_db))
            final_score += hit_score
            logger.info(f"Model: {model_type}, score: {hit_score}")
        final_score /= len(cfg.ENCODE_MODEL_LIST)
        logger.info(f"Final score: {final_score}")
        final_score_list.append([i, final_score])

    final_score_list = sorted(final_score_list, key=lambda x: x[1], reverse=True)
    # get top 10
    final_score_list = final_score_list[:10]
    logger.info(f"top_10: {final_score_list}")

    hit_spkid = final_score_list[0][0]
    hit_score = final_score_list[0][1]
    logger.info(f"hit_spkid:{hit_spkid}, hit_score:{hit_score}")
    hit_spkname = get_name_from_phone(hit_spkid)

    compare_result = {}
    if hit_score < cfg.COMPARE_SCORE_THRESHOLD:
        compare_result["model"] = {"is_hit": False, "hit_spkid": hit_spkid, "score": hit_score, "hit_spkname": hit_spkname}
        logger.info(f"spkid:{spkid} is not in black list. score:{hit_score}")
        if cfg.SHOW_PUBLIC:
            del compare_result['model']["score"]
        return {"code": 200, "compare_result": compare_result, "message": "{}不在黑库中".format(spkid)}

    compare_result['model'] = {"is_hit": True, "hit_spkid": hit_spkid, "score": hit_score, "hit_spkname": hit_spkname}
    # OSS
    test_bucket_name = cfg.MINIO['test_bucket_name']
    raw_url = upload_file(test_bucket_name, file_path, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file(test_bucket_name, selected_path, f"{spkid}/vad_{spkid}.wav")
    if cfg.SHOW_PUBLIC:
        raw_url = raw_url.replace(cfg.HOST, cfg.PUBLIC_HOST)
        selected_url = selected_url.replace(cfg.HOST, cfg.PUBLIC_HOST)

    db_info = {}
    db_info['spkid'] = phone
    db_info['name'] = name
    db_info['gender'] = gender
    db_info['valid_length'] = valid_length
    db_info['file_url'] = raw_url
    db_info['preprocessed_file_url'] = selected_url
    db_info['message'] = str(compare_result)
    db_info['hit_score'] = hit_score
    db_info['hit_spkid'] = hit_spkid
    add_hit(db_info)
    if cfg.SHOW_PUBLIC:
        del compare_result['model']["score"]
    return {"code": 200, "message": "success", "file_url": raw_url, "compare_result": compare_result}
