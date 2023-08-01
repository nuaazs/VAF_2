#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cluster_shengting.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   省厅注册逻辑
'''
import glob
import subprocess
import numpy as np
import pymysql
import torchaudio
from tqdm import tqdm
import cfg
import os
import torch
from utils.oss.upload import upload_file
import requests
from tqdm.contrib.concurrent import process_map
import wget
from sklearn.metrics.pairwise import cosine_similarity

from loguru import logger

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8",
           enqueue=True, compression="zip", backtrace=True, diagnose=True)

encode_url = "http://192.168.3.169:5001/encode"  # 提取特征
cluster_url = "http://192.168.3.169:5011/cluster"  # cluster
asr_url = "http://192.168.3.169:5000/transcribe/file"  # ASR
use_model_type = "ECAPATDNN"


msg_db = cfg.MYSQL

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

def update_db(spkid):
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
        sql = "update check_for_speaker_diraization set after_cluster=1 where record_id=%s"
        cursor.execute(sql, (spkid))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{spkid}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def get_selected_url_from_db():
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
        sql = "select record_id,selected_url from check_for_speaker_diraization"
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    except Exception as e:
        logger.error(f"Get selected_url from db failed. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
    return result


def main():
    # step 1 话术过滤
    # a, b = check_text(text)
    # if a == "正常":
    #     # todo 查找新话术逻辑
    #     return None

    # step 2 encode
    selected_urls = get_selected_url_from_db()
    logger.info(f"len(selected_urls):{len(selected_urls)}")

    tmp_folder = "/tmp/cluster_diraization"
    os.makedirs(tmp_folder, exist_ok=True)
    embeddings = []
    black_id_all = []
    for idx, i in enumerate(selected_urls):
        try:
            save_path = tmp_folder
            i = i['selected_url']
            file_name = os.path.join(save_path, os.path.basename(i))
            wget.download(i, file_name)

            spkid = os.path.basename(i).split(".")[0].split('_')[0]
            # step4 提取特征
            data = {"spkid": spkid, "filelist": ["local://"+file_name]}
            response = send_request(encode_url, data=data)
            if response['code'] == 200:
                black_id_all.append(spkid)
                for key in response['file_emb'][use_model_type]["embedding"].keys():
                    file_emb = response['file_emb'][use_model_type]["embedding"][key]
                    embeddings.append(file_emb)
            else:
                logger.error(
                    f"Encode failed. spkid:{spkid}.response:{response}")
        except Exception as e:
            logger.error(f"Encode failed. spkid:{spkid}.msg:{e}")
    with open("emb_map.txt", "w+") as f:
        for item in black_id_all:
            f.write(item + "\n")
    embeddings = np.array(embeddings)
    print(embeddings.shape)  # (143, 192)

    embeddings = embeddings.astype(np.float32)
    embeddings.tofile('emb.bin')
    read_data = embeddings

    # read_data = np.fromfile("emb.bin", dtype=np.float32)
    # print(read_data.shape)
    # read_data = read_data.reshape(-1, 192)
    # print(read_data.shape)
    
    
    # step 3 聚类
    threshold = 0.85
    cosine_similarities = cosine_similarity(read_data)
    np.fill_diagonal(cosine_similarities, 0)
    mask = np.zeros_like(cosine_similarities, dtype=bool)

    filtered_results = []
    for i in range(cosine_similarities.shape[0]):
        for j in range(i+1, cosine_similarities.shape[1]):
            if cosine_similarities[i, j] > threshold and not mask[i, j]:
                filtered_results.append((i, j))
                mask[i, j] = True

    print(
        f"Found {len(filtered_results)} pairs above the threshold of {threshold}")

    map_d = {}
    with open("emb_map.txt", "r") as f:
        for idx, line in enumerate(f.readlines()):
            map_d[idx] = line.strip()

    last_index = None
    result = []
    print("Filtered results:")
    for i, j in filtered_results:
        if last_index == i:
            continue
        logger.info(
            f"{map_d[i]} and {map_d[j]} have cosine similarity of {cosine_similarities[i, j]}")
        result.append(map_d[i])
        last_index = i
    print(f"len(result):{len(result)}")
    print(f"result:{result}")
    
    for i in result:
        print(i)
        update_db(str(i))


if __name__ == "__main__":
    main()
