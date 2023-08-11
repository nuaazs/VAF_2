#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   省厅音频注册
'''
import multiprocessing
import sys
import time
import numpy as np
import pymysql
import torch
from tqdm import tqdm
import cfg
import os
import requests
from tqdm.contrib.concurrent import process_map
import wget
from utils.orm.db_orm import get_embeddings, to_database
from loguru import logger
sys.path.append("../")

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

name = os.path.basename(__file__).split(".")[0]
logger.add("log/"+name+"_{time}.log", rotation="500 MB", encoding="utf-8", enqueue=True, compression="zip", backtrace=True, diagnose=True)

host = "http://192.168.3.169"
encode_url = f"{host}:5001/encode"  # 提取特征
cluster_url = f"{host}:5011/cluster"  # cluster
asr_url = f"{host}:5000/transcribe/file"  # ASR
msg_db = cfg.MYSQL


def send_request(url, method='POST', files=None, data=None, json=None, headers=None):
    try:
        response = requests.request(method, url, files=files, data=data, json=json, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: spkid:{data['spkid']}. msg:{e}")
        return None


def add_speaker(spkid):
    """
    录入黑库表
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
        query_sql = f"insert into black_speaker_info (record_id, valid_length,file_url,preprocessed_file_url,record_month,register_time) \
                    select record_id, wav_duration, file_url, selected_url,record_month, now() \
                    from check_for_speaker_diraization where record_id = %s;"
        cursor.execute(query_sql, (spkid))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{spkid}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def get_selected_url_from_db():
    """
    获取需要注册的音频url
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


def cosine_similarity(input_data):
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
    results = process_map(cosine_similarity, input_data, max_workers=4, chunksize=1000, desc='Doing----')
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


def main():
    """
    register
    """
    tmp_folder = "/tmp/register"
    os.makedirs(tmp_folder, exist_ok=True)
    selected_urls = get_selected_url_from_db()
    logger.info(f"len(selected_urls):{len(selected_urls)}")

    for file in tqdm(selected_urls):
        try:
            save_path = tmp_folder
            i = file['selected_url']
            file_name = os.path.join(save_path, os.path.basename(i))
            wget.download(i, file_name)
            spkid = os.path.basename(i).split(".")[0].split('_')[0]
            data = {"spkid": spkid, "filelist": ["local://"+file_name]}
            response = send_request(encode_url, data=data)

            compare_result = {}
            if response['code'] == 200:
                for model_type in cfg.ENCODE_MODEL_LIST:
                    emb_new = list(response['file_emb'][model_type]["embedding"].values())[0]
                    return_results = compare_handler(model_type=model_type, embedding=emb_new, black_limit=cfg.BLACK_TH[model_type])
                    compare_result[model_type] = return_results
            else:
                logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
                continue

            need_register = [True for k, v in compare_result.items() if v['inbase'] == 0 and v['best_score'] < cfg.BLACK_TH[k]]
            if need_register and all(need_register):
                logger.info(f"Need register. spkid:{spkid}. compare_result:{compare_result}")
                add_success = to_database(embedding=torch.tensor(emb_new), spkid=spkid, use_model_type=model_type)
                if add_success:
                    logger.info(f"Add speaker success. spkid:{spkid}")
                    add_speaker(spkid)
            else:
                logger.info(f"Speaker already exists. spkid:{spkid}. Compare result:{compare_result}")

        except Exception as e:
            logger.error(f"Register failed. spkid:{spkid}.msg:{e}")
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)


if __name__ == "__main__":
    main()
