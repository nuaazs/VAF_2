#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tools.py
@Time    :   2023/08/14 10:36:37
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''
from collections import Counter
from utils.oss.upload import upload_file
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
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
from utils.preprocess.save import save_file, save_url
from loguru import logger

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: spkid:{data['spkid']}. msg:{e}")
        return None


def find_items_with_highest_value(dictionary):
    value_counts = Counter(dictionary.values())
    max_count = max(value_counts.values())
    for key, value in dictionary.items():
        if value_counts[value] == max_count:
            keys_with_max_value = value
    items_with_highest_value = {key: value for key, value in dictionary.items() if value_counts[value] == max_count}
    return items_with_highest_value, keys_with_max_value


def extract_audio_segment(input_file, output_file, start_time, end_time):
    """
    截取音频片段
    """
    audio = AudioSegment.from_file(input_file)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    extracted_segment = audio[start_ms:end_ms]
    extracted_segment.export(output_file, format="wav")


def get_similarities_result(emb_db_dic,emb_type, emb_db, emb_new):
    """
    获取相似度最高的spkid
    """
    cosine_similarities = cosine_similarity(emb_db, emb_new)
    top_indices = np.argsort(cosine_similarities.ravel())[-1]
    spkid = list(emb_db_dic[emb_type].keys())[top_indices]
    score = cosine_similarities[top_indices][0]
    print(f"top1_index: {top_indices}, spkid: {spkid}, score: {score}")
    return spkid, score


def cosine_similarity_db(input_data):
    base_item, base_embedding, embedding = input_data
    base_embedding = torch.tensor(base_embedding)
    return [similarity(base_embedding, embedding).numpy(), base_item]


def compare_handler(model_type=None, embedding=None, black_limit=0.78, top_num=10):
    """
    是否在黑库中 并返回top1-top10
    """
    emb_db = get_embeddings(use_model_type=model_type)
    embedding = torch.tensor(embedding).to('cpu')
    input_data = [(k, emb_db[k], embedding) for k in emb_db.keys()]

    t1 = time.time()
    results = process_map(cosine_similarity_db, input_data, max_workers=1, chunksize=1000, desc='Doing----')
    if not results:
        return {'best_score': 0, 'inbase': 0}
    t2 = time.time()
    logger.info(f"compare_handler time:{t2-t1}")

    return_results = {}
    results = sorted(results, key=lambda x: float(x[0]) * (-1))
    return_results["best_score"] = float(np.array(results[0][0]))

    if results[0][0] <= black_limit:
        return_results["inbase"] = 0
        return return_results
    else:
        return_results["inbase"] = 1
        # top1-top10
        if len(results) < top_num:
            top_num = len(results)
        for index in range(top_num):
            return_results[f"top_{index + 1}"] = f"{results[index][0]:.5f}"
            return_results[f"top_{index + 1}_id"] = str(results[index][1])
    return return_results


def get_joint_wav(tmp_folder,phone, wav_list):
    """
    拼接音频
    :param phone:
    :param wav_list:
    :return:
    """
    playlist = AudioSegment.empty()
    for wav in wav_list:
        playlist = playlist + AudioSegment.from_wav(wav)
    output_name = f'{tmp_folder}/{phone}/{phone}_joint.wav'
    playlist.export(output_name, format='wav')
    return output_name
