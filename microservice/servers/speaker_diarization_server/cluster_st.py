#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cluster_shengting.py
@Time    :   2023/07/24 10:46:54
@Author  :   Carry
@Version :   1.0
@Desc    :   省厅新音频筛选逻辑 话术过滤+特征提取+聚类
'''
import csv
import datetime
import random
import numpy as np
import pymysql
from tqdm import tqdm
import cfg
import os
import wget
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from tools import send_request
from get_output_from_hit.rrr_filter_plus import check_text


def insert_to_db(data):
    conn = pymysql.connect(
        host=cfg.MYSQL.get("host"),
        port=cfg.MYSQL.get("port"),
        db=cfg.MYSQL.get("db"),
        user=cfg.MYSQL.get("username"),
        passwd=cfg.MYSQL.get("passwd"),
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    try:
        spkid = data['spkid']
        raw_file_path = data['raw_file_path']
        selected_url = data['selected_url']
        asr_text = data['asr_text']
        total_duration = data['total_duration']
        selected_times = str(data['selected_times'])
        record_month = data['record_month']
        vad_times = str(data['vad_times'])

        sql = "INSERT INTO check_for_speaker_diraization (`record_id`, `file_url`, `selected_url`, `asr_text`, `wav_duration`,`create_time`,`selected_times`, `record_month`,`vad_times`) VALUES (%s, %s, %s, %s, %s,now(), %s, %s, %s);"
        cursor.execute(sql, (spkid, raw_file_path, selected_url, asr_text, total_duration, selected_times, record_month, vad_times))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. spkid:{data['spkid']}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()


def cluster_handler(read_data, threshold):
    """
    聚类
    return: 需要删除的record list
    """
    cosine_similarities = cosine_similarity(read_data)
    np.fill_diagonal(cosine_similarities, 0)
    mask = np.zeros_like(cosine_similarities, dtype=bool)

    filtered_results = []
    for i in range(cosine_similarities.shape[0]):
        for j in range(i+1, cosine_similarities.shape[1]):
            if cosine_similarities[i, j] > threshold and not mask[i, j]:
                filtered_results.append((i, j))
                mask[i, j] = True

    logger.info(f"Found {len(filtered_results)} pairs above the threshold of {threshold}")

    map_d = {}
    with open("output/emb_map.txt", "r") as f:
        for idx, line in enumerate(f.readlines()):
            map_d[idx] = line.strip()

    last_index = None
    result = []
    logger.info("Filtered results:")
    for i, j in filtered_results:
        if last_index == i:
            continue
        logger.info(f"{map_d[i]} and {map_d[j]} have cosine similarity of {cosine_similarities[i, j]}")
        result.append(map_d[i])
        last_index = i
    logger.info(f"Need to delete {len(result)} records.")

    return result


def encode_handler(need_cluster_records):
    """
    get embeddings from records and save to emb.bin
    返回emb和spkid
    """
    records_lis = need_cluster_records
    tmp_folder = "/tmp/cluster_diraization"
    os.makedirs(tmp_folder, exist_ok=True)
    embeddings = []
    black_id_all = []
    for i in tqdm(records_lis):
        try:
            spkid = i['spkid']
            selected_url = i['selected_url']
            save_path = tmp_folder
            file_name = os.path.join(save_path, os.path.basename(selected_url))
            wget.download(selected_url, file_name)

            # step4 提取特征
            data = {"spkid": spkid, "filelist": ["local://"+file_name]}
            response = send_request(cfg.ENCODE_URL, data=data)
            if response['code'] == 200:
                black_id_all.append(spkid)
                for key in response['file_emb'][cfg.USE_MODEL_TYPE]["embedding"].keys():
                    file_emb = response['file_emb'][cfg.USE_MODEL_TYPE]["embedding"][key]
                    embeddings.append(file_emb)
            else:
                logger.error(f"Encode failed. spkid:{spkid}.response:{response}")
        except Exception as e:
            logger.error(f"Encode failed. spkid:{spkid}.msg:{e}")
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)
    assert len(embeddings) == len(black_id_all), "len(embeddings) != len(black_id_all)"

    with open("output/emb_map.txt", "w+") as f:
        for item in black_id_all:
            f.write(item + "\n")
    embeddings = np.array(embeddings)
    logger.info(embeddings.shape)  # (143, 192)

    embeddings = embeddings.astype(np.float32)
    embeddings.tofile('output/emb.bin')
    logger.info("Save embeddings to emb.bin")
    return embeddings, black_id_all


def check_text_handler(need_cluster_records):
    """
    话术过滤
    return: after_check_text_records 话术过滤后的records
    """
    logger.info(f"Befor check_text_handler len:{len(need_cluster_records)}")
    after_check_text_records = []
    for i in tqdm(need_cluster_records):
        spkid = i['spkid']
        asr_text = i['asr_text']
        a, b = check_text(asr_text)
        logger.info(f"Record ID:{spkid} check text result:{a},{b}")
        if b == "未命中":
            with open("output/new_text.txt", "a+") as f:
                f.write(f"{spkid}\t{asr_text}\n")
        else:
            after_check_text_records.append(i)
    logger.info(f"After check_text_handler len:{len(after_check_text_records)}")
    return after_check_text_records


def cluster_pipleline(need_cluster_records):
    """
    话术过滤+特征提取+聚类
    """
    # 话术过滤
    # need_cluster_records = check_text_handler(need_cluster_records)
    # 编码
    read_data, after_check_text_records = encode_handler(need_cluster_records)
    # 聚类
    need_del_records = cluster_handler(read_data, 0.82)
    # 剔除重复数据
    result_records = [x for x in need_cluster_records if x not in need_del_records]
    logger.info(f"Need check len:{len(result_records)}")

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_file = f"output/check_result_{date}.csv"
    with open(csv_file, 'w+', newline='') as csvfile:
        for i in result_records:
            insert_to_db(i)  # backup to db
            fieldnames = i.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(i)


def local_test():
    """
    直接读取本地的emb.bin文件测试
    """
    with open(f"output/need_cluster_records.txt", "r+") as f:
        result = f.readlines()
    need_cluster_records = []
    for i in result:
        i = eval(i)
        need_cluster_records.append(i)

    # 话术过滤
    # need_cluster_records = check_text_handler(need_cluster_records)
    # #编码
    # read_data, after_check_text_records = encode_handler(need_cluster_records)

    with open("output/emb_map.txt", "r") as f:
        after_check_text_records = f.readlines()
        after_check_text_records = [i.strip() for i in after_check_text_records]
    logger.info(f"Read after_check_text_records len:{len(after_check_text_records)}")

    logger.info("Read embeddings from emb.bin")
    read_data = np.fromfile("output/emb.bin", dtype=np.float32)
    logger.info(read_data.shape)
    read_data = read_data.reshape(-1, cfg.EMBEDDING_LEN[cfg.USE_MODEL_TYPE])
    logger.info(read_data.shape)

    # 聚类
    need_del_records = cluster_handler(read_data, 0.82)

    # 剔除重复数据
    result_records = [x for x in need_cluster_records if x not in need_del_records]
    logger.info(f"Need check len:{len(result_records)}")

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_file = f"output/check_result_{date}.csv"
    with open(csv_file, 'w+', newline='') as csvfile:
        for i in result_records:
            insert_to_db(i)  # backup to db
            fieldnames = i.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(i)


if __name__ == "__main__":
    local_test()
