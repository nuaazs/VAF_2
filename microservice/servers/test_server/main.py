#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   音频推理，检测是否再黑库中
'''
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from loguru import logger
from utils.orm.db_orm import get_embeddings, to_database
import wget
from tqdm.contrib.concurrent import process_map
import requests
import os
import cfg
from tqdm import tqdm
import torchaudio
import torch
import pymysql
import numpy as np
import time
import glob
import shutil
from flask import Flask, request, jsonify
import sys
sys.path.append("/home/xuekaixiang/workplace/vaf/microservice/servers")
from speaker_diarization_server.main import find_items_with_highest_value

from utils.oss.upload import upload_file
app = Flask(__name__)

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

host = "http://192.168.3.169"
encode_url = f"{host}:5001/encode"  # 提取特征
cluster_url = f"{host}:5011/cluster"  # cluster
asr_url = f"{host}:5000/transcribe/file"  # ASR
vad_url = f"{host}:5005/energy_vad/file"  # VAD
lang_url = f"{host}:5002/lang_classify"  # 语种识别
msg_db = cfg.MYSQL
use_model_type = "ECAPATDNN"


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(
            method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Request failed: spkid:{data['spkid']}. msg:{e}")
        return None


def extract_audio_segment(input_file, output_file, start_time, end_time):
    """
    截取音频片段
    """
    audio = AudioSegment.from_file(input_file)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    extracted_segment = audio[start_ms:end_ms]
    extracted_segment.export(output_file, format="wav")


def add_hit(spkid):
    """
    录入hit
    """
    conn = pymysql.connect(
        host=msg_db.get("host"),
        port=msg_db.get("port"),
        db=msg_db.get("db"),
        user=msg_db.get("username"),
        passwd=msg_db.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        query_sql = f"insert into speaker (phone, valid_length,file_url,preprocessed_file_url,register_time) \
                    select record_id, wav_duration, file_url, selected_url, now() \
                    from check_for_speaker_diraization where record_id = %s;"
        cursor.execute(query_sql, (spkid))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{spkid}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()

def get_similarities_result(emb_db, emb_new):
    """
    获取相似度最高的spkid
    """
    cosine_similarities = cosine_similarity(emb_db, emb_new)
    print(f"cosine_similarities: {cosine_similarities}")
    top_indices = np.argsort(cosine_similarities.ravel())[-1]
    spkid = list(emb_db_dic.keys())[top_indices]
    score = cosine_similarities[top_indices][0]
    print(f"top1_index: {top_indices}, spkid: {spkid}, score: {score}")
    print(emb_db[top_indices])
    return spkid, score


def pipeline(tmp_folder, filepath, spkid):
    # step1 VAD
    data = {"spkid": spkid, "length": 90}
    files = [('file', (filepath, open(filepath, 'rb')))]
    response = send_request(vad_url, files=files, data=data)
    if not response:
        return None

    # step2 截取音频片段
    output_file_li = []
    d = {}
    for idx, i in enumerate(response['timelist']):
        output_file = f"{tmp_folder}/{spkid}_{idx}.wav"  # 截取后的音频片段保存路径
        extract_audio_segment(filepath, output_file, start_time=i[0]/1000, end_time=i[1]/1000)
        output_file_li.append(output_file)
        d[output_file] = (i[0]/1000, i[1]/1000)

    # step3 普通话过滤
    wav_files = ["local://"+i for i in output_file_li]
    data = {"spkid": spkid, "filelist": ",".join(wav_files)}
    response = send_request(lang_url, data=data)
    if response['code'] == 200:
        pass_list = response['pass_list']
        url_list = response['file_url_list']
        mandarin_wavs = [i for i in url_list if pass_list[url_list.index(i)] == 1]
    else:
        logger.error(f"Lang_classify failed. spkid:{spkid}.response:{response}")
        return None

    # step4 提取特征
    data = {"spkid": spkid, "filelist": ",".join(mandarin_wavs)}
    response = send_request(encode_url, data=data)
    if response['code'] == 200:
        file_emb = response['file_emb']
    else:
        logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
        return None

    # step5 聚类
    # TODO:选择最优的模型
    file_emb = file_emb[use_model_type]
    data = {
        "emb_dict": file_emb["embedding"],
        "cluster_line": 3,
        "mer_cos_th": 0.7,
        "cluster_type": "spectral",  # spectral or umap_hdbscan
        "min_cluster_size": 1,
    }
    response = send_request(cluster_url, json=data)
    logger.info(f"\t * -> Cluster result: {response}")
    items, keys_with_max_value = find_items_with_highest_value(response['labels'])
    max_score = response['scores'][keys_with_max_value]['max']
    min_score = response['scores'][keys_with_max_value]['min']

    if min_score < 0.8:
        logger.info(f"After cluster min_score  < 0.8. spkid:{spkid}.response:{response['scores']}")
        return None
    total_duration = 0
    for i in items.keys():
        total_duration += file_emb['length'][i]
    if total_duration < cfg.MIN_LENGTH_REGISTER:
        logger.info(f"After cluster total_duration:{total_duration} < {cfg.MIN_LENGTH_REGISTER}s. spkid:{spkid}.response:{response}")
        return None
    selected_files = sorted(items.keys(), key=lambda x: x.split("/")[-1].replace(".wav", "").split("_")[0])
    audio_data = np.concatenate([torchaudio.load(file.replace("local://", ""))[0] for file in selected_files], axis=-1)
    _selected_path = os.path.join(tmp_folder, f"{spkid}_selected.wav")
    torchaudio.save(_selected_path, torch.from_numpy(audio_data), sample_rate=8000)

    selected_times = [d[_data.replace("local://", "")] for _data in selected_files]

    # step6 ASR
    # text = ""
    # data = {"spkid": spkid, "postprocess": "1"}
    # files = [('wav_file', (filepath, open(filepath, 'rb')))]
    # response = send_request(asr_url, files=files, data=data)
    # if response.get('transcription') and response.get('transcription').get('text'):
    #     text = response['transcription']["text"]
    #     # logger.info(f"\t * -> ASR结果: {text}")
    # else:
    #     logger.error(
    #         f"ASR failed. spkid:{spkid}.message:{response['message']}")

    # step7 NLP
    # nlp_result = classify_text(text)
    # logger.info(f"\t * -> 文本分类结果: {nlp_result}")

    # step7 话术过滤
    # a, b = check_text(text)
    # if a == "正常":
    #     # todo 查找新话术逻辑
    #     return None


    return {
        "spkid": spkid,
        "raw_file_path": filepath,
        "selected_path": _selected_path,
        "selected_times": selected_times,
        "total_duration": total_duration,
    }



def main(filepath):
    try:
        spkid = os.path.basename(filepath).split(".")[0].split('-')[-1]
        tmp_folder = f"/tmp/test/{spkid}"
        os.makedirs(tmp_folder, exist_ok=True)
        pipeline_result = pipeline(tmp_folder, filepath, spkid)
        if pipeline_result:
            # 提取特征
            data = {"spkid": spkid, "filelist": ",".join(pipeline_result['selected_path'])}
            response = send_request(encode_url, data=data)
            if response['code'] == 200:
                file_emb = response['file_emb']
            else:
                logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
                return 
            
            # 撞库
            compare_result = {}
            for i in cfg.ENCODE_MODEL_LIST:
                spkid, score = get_similarities_result( np.array(list(emb_db_dic[i].values())), file_emb[i]['embedding'])
                logger.info(f"spkid:{spkid}, score:{score}")
                if score < cfg.BLACK_TH[i]:
                    logger.info(f"spkid:{spkid} is not in black list. score:{score}")
                    return
                compare_result[i]={"spkid":spkid,"score":score}
            # ASR

            # NLP

            # OSS
            raw_url = upload_file("test", filepath, f"{spkid}/raw_{spkid}.wav")
            selected_url = upload_file("test", os.path.join(tmp_folder, f"{spkid}_selected.wav"), f"{spkid}/{spkid}_selected.wav")
            pipeline_result['raw_url']=raw_url
            pipeline_result['selected_url']=selected_url

            add_hit(pipeline_result)
    except Exception as e:
        logger.error(f"Pipeline failed. spkid:{spkid}. msg:{e}.")
    finally:
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    emb_db_dic={}
    for i in cfg.ENCODE_MODEL_LIST:
        emb_db_dic[i] = get_embeddings(use_model_type=i)

    wav_files = glob.glob("./*.wav")
    logger.info(f"Total wav files: {len(wav_files)}")
    wav_files = sorted(wav_files)
    process_map(main, wav_files, max_workers=1, desc='TQDMING---:')
