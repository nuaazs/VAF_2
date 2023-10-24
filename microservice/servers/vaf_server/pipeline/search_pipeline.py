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
import shutil
import time
from loguru import logger
import numpy as np
import torch
import torchaudio
import cfg
from tools.file_handler import get_audio_and_conver, extract_audio_segment, find_items_with_highest_value
from tools.orm_handler import get_embeddings_from_db, add_hit
from tools.minio_handler import upload_file
from tools.vad_handler import energybase_vad
from tools.embedding_handler import encode_files
from tools.request_handler import send_request
from tqdm.contrib.concurrent import process_map
import concurrent.futures

t0 = time.time()
spkid_embedding_dict = get_embeddings_from_db()
t3 = time.time()
logger.info(f"The number of spkid in db: {len(spkid_embedding_dict)}. time:{t3-t0}")


def calculate_final_score(input_data):
    """
    计算最终得分
    Args:
        input_data: (i, value, file_emb, spkid)
    Returns:
        [i, final_score]
    """
    i, value, file_emb, spkid = input_data
    final_score = 0
    for model_type in cfg.ENCODE_MODEL_LIST:
        model_type = model_type.replace("_", "")
        emb_db = value[model_type]
        emb_new = file_emb[model_type]["embedding"][spkid]
        hit_score = np.dot(emb_new, emb_db) / (np.linalg.norm(emb_new) * np.linalg.norm(emb_db))
        final_score += hit_score
    final_score /= len(cfg.ENCODE_MODEL_LIST)
    return [i, final_score]


call_time_info = {}


def search_pipeline(request, filetype):
    tmp_folder = cfg.TMP_FOLDER
    os.makedirs(tmp_folder, exist_ok=True)

    spkid = request.form.get('spkid')
    if not spkid:
        logger.error(f"spkid is None.")
        return {"code": 500, "spkid": spkid, "message": "spkid is None."}
    spkid = str(spkid).strip()
    name = request.form.get('name')
    gender = request.form.get('gender')
    channel = int(request.form.get('channel', 0))

    NEED_LANG_CHECK = request.form.get('need_lang_check', default=False, type=lambda x: x.lower() == 'true')    # 是否进行语种检测
    NEED_CLUSTER = request.form.get('need_cluster', default=False, type=lambda x: x.lower() == 'true')    # 是否进行聚类

    # step1 save and convert audio file
    t1 = time.time()
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
    call_time_info['save_and_convert'] = time.time() - t1

    # step2 do lang classify
    if NEED_LANG_CHECK:
        t1 = time.time()
        data = {"spkid": spkid, "filelist": "local://" + file_path}
        response = send_request(cfg.LANG_URL, data=data)
        if response and response['code'] == 200:
            pass_list = response['pass_list']
            url_list = response['file_url_list']
            mandarin_wavs = [i for i in url_list if pass_list[url_list.index(i)] == 1]
            if not mandarin_wavs:
                if os.path.exists(spkid_folder):
                    shutil.rmtree(spkid_folder)
                return {"code": 200, "spkid": spkid, "message": "Not mandarin. response:{}".format(response)}
        else:
            logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            return {"code": 500, "spkid": spkid, "message": "Lang_classify failed. response:{}".format(response)}
        call_time_info['lang_classify'] = time.time() - t1

    # step3 do vad
    t1 = time.time()
    vad_file_path, vad_length, time_list = energybase_vad(file_path, spkid_folder)
    logger.info(f"spkid:{spkid}. After VAD vad_length:{vad_length}")
    if vad_length < cfg.VAD_MIN_LENGTH:
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)
        return {"code": 200, "spkid": spkid, "message": "VAD length is less than {}s. vad_length:{}".format(cfg.VAD_MIN_LENGTH, vad_length)}
    # cut 10s
    extract_audio_segment(vad_file_path, vad_file_path, 0, cfg.VAD_MIN_LENGTH)
    call_time_info['vad'] = time.time() - t1

    # step4 do cluster
    if NEED_CLUSTER:
        t1 = time.time()
        # 截取音频片段
        output_file_li = []
        d = {}
        for idx, i in enumerate(time_list):
            output_file = f"{spkid_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
            extract_audio_segment(file_path, output_file, start_time=i[0], end_time=i[1])
            output_file_li.append(output_file)
            d[output_file] = (i[0], i[1])

        wav_files = ["local://"+i for i in output_file_li]
        data = {"spkid": spkid, "filelist": ",".join(wav_files)}
        response = send_request(cfg.LANG_URL, data=data)
        if response and response['code'] == 200:
            pass_list = response['pass_list']
            url_list = response['file_url_list']
            mandarin_wavs = [i.replace("local://", "") for i in url_list if pass_list[url_list.index(i)] == 1]
        else:
            logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            return {"code": 500, "spkid": spkid, "message": "Cluster Lang_classify failed. response:{}".format(response)}

        # 提取特征
        file_emb = encode_files(spkid, mandarin_wavs)

        # 聚类
        file_emb = file_emb[cfg.USE_MODEL_TYPE.replace("_", "")]
        data = {
            "emb_dict": file_emb["embedding"],
            "cluster_line": 3,
            "mer_cos_th": 0.7,
            "cluster_type": "spectral",  # spectral or umap_hdbscan
            "min_cluster_size": 1,
        }
        try:
            response = send_request(cfg.CLUSTER_URL, json=data)
        except Exception as e:
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            logger.error(f"Cluster failed. spkid:{spkid}.response:{e}")
            return {"code": 500, "spkid": spkid, "message": "Cluster failed. response:{}".format(e)}
        items, keys_with_max_value = find_items_with_highest_value(response['labels'])
        max_score = response['scores'][keys_with_max_value]['max']
        min_score = response['scores'][keys_with_max_value]['min']

        if min_score < cfg.CLUSTER_MIN_SCORE_THRESHOLD:
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            logger.info(f"After cluster min_score:{min_score} < {cfg.CLUSTER_MIN_SCORE_THRESHOLD}. spkid:{spkid}.response:{response}")
            return {"code": 200, "spkid": spkid, "message": f"After cluster min_score:{min_score} < {cfg.CLUSTER_MIN_SCORE_THRESHOLD}. spkid:{spkid}.response:{response}"}
        total_duration = 0
        for i in items.keys():
            total_duration += file_emb['length'][i]
        if total_duration < cfg.VAD_MIN_LENGTH:
            if os.path.exists(spkid_folder):
                shutil.rmtree(spkid_folder)
            logger.info(f"After cluster total_duration:{total_duration} < {cfg.VAD_MIN_LENGTH}s. spkid:{spkid}.response:{response}")
            return {"code": 200, "spkid": spkid, "message": f"After cluster total_duration:{total_duration} < {cfg.VAD_MIN_LENGTH}s. spkid:{spkid}.response:{response}"}

        # Resample 16k
        selected_files = sorted(items.keys(), key=lambda x: x.split("/")[-1].replace(".wav", "").split("_")[0])
        resampled_waveform_li = []
        for file in selected_files:
            waveform, sample_rate = torchaudio.load(file.replace("local://", ""))
            if sample_rate == 8000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            resampled_waveform_li.append(waveform)
        audio_data = np.concatenate(resampled_waveform_li, axis=-1)
        vad_file_path = os.path.join(spkid_folder, f"{spkid}_selected.wav")
        torchaudio.save(vad_file_path, torch.from_numpy(audio_data), sample_rate=16000)
        # cut 10s
        extract_audio_segment(vad_file_path, vad_file_path, 0, cfg.VAD_MIN_LENGTH)
        call_time_info['cluster'] = time.time() - t1

    # step5 get embedding
    t1 = time.time()
    try:
        file_emb = encode_files(spkid, [vad_file_path])
    except Exception as e:
        logger.error(f"Encode failed. spkid:{spkid}.response:{e}")
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)
        return {"code": 500, "spkid": spkid, "message": "Encode failed. response:{}".format(e)}
    call_time_info['encode'] = time.time() - t1

    t1 = time.time()
    datas = [(i, value, file_emb, spkid) for i, value in spkid_embedding_dict.items()]

    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)
    # with executor as e:
    #     final_score_list = list(e.map(calculate_final_score, datas))

    final_score_list = process_map(calculate_final_score, datas, max_workers=16, chunksize=1000, desc='calculate_final_score----')

    final_score_list = sorted(final_score_list, key=lambda x: x[1], reverse=True)
    call_time_info['campare'] = time.time() - t1

    # get top 10
    final_score_list = final_score_list[:10]
    logger.info(f"top_10: {final_score_list}")

    hit_spkid = final_score_list[0][0]
    hit_score = final_score_list[0][1]
    logger.info(f"hit_spkid:{hit_spkid}, best hit_score:{hit_score}")

    if hit_score < cfg.HIT_SCORE_THRESHOLD:
        compare_result = {"is_hit": False, "hit_spkid": hit_spkid, "hit_score": hit_score}
        logger.info(f"spkid:{spkid} is not in black list. score:{hit_score}")
        if os.path.exists(spkid_folder):
            shutil.rmtree(spkid_folder)
        return {"code": 200, "compare_result": compare_result, "message": "{} is not in black list".format(spkid), "call_time_info": call_time_info}

    compare_result = {"is_hit": True, "hit_spkid": hit_spkid, "hit_score": hit_score, "top_10": final_score_list}
    # OSS
    t1 = time.time()
    hit_bucket_name = cfg.MINIO['hit_bucket_name']
    raw_url = upload_file(hit_bucket_name, file_path, f"{spkid}/raw_{spkid}.wav")
    selected_url = upload_file(hit_bucket_name, vad_file_path, f"{spkid}/vad_{spkid}.wav")
    call_time_info['upload_oss'] = time.time() - t1

    db_info = {}
    db_info['spkid'] = spkid
    db_info['name'] = name
    db_info['gender'] = gender
    db_info['valid_length'] = vad_length
    db_info['file_url'] = raw_url
    db_info['preprocessed_file_url'] = selected_url
    db_info['message'] = str(compare_result)
    db_info['hit_score'] = hit_score
    db_info['hit_spkid'] = hit_spkid
    add_hit(db_info)
    if os.path.exists(spkid_folder):
        shutil.rmtree(spkid_folder)
    return {"code": 200, "message": "success", "file_url": raw_url, "compare_result": compare_result, "call_time_info": call_time_info}
