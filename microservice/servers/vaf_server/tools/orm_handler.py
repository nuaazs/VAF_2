#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   orm_handler.py
@Time    :   2023/10/14 20:04:03
@Author  :   Carry
@Version :   1.0
@Desc    :   数据库操作
'''
import pickle
import struct
import time
import pymysql
import redis
import numpy as np
import torch
import cfg
from tqdm.contrib.concurrent import process_map
from loguru import logger


############################################ redis操作 ############################################
def toRedis(r, a, n):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    shape = struct.pack(">II", 256, 1)
    encoded = shape + a.tobytes()
    r.set(n, encoded)
    return


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    a = np.frombuffer(encoded, dtype=np.float32, offset=8)
    return a


def get_embedding(spkid):
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )

    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        now_id = key.split("_")[1]
        if now_id == spkid:
            embedding = fromRedis(r, key)

    return embedding


def get_embeddings(use_model_type=None):
    """
    获取redis中的所有emb
    Args:
        use_model_type: 使用的模型类型
    Returns:
        all_embedding: 所有emb
    """
    assert use_model_type, "use_model_type is None"
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    all_embedding = {}
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        embedding_type = key.replace(key.split('_')[-1], "").strip("_")
        if embedding_type != use_model_type:
            continue
        spkid = key.split("_")[-1]
        embedding_1 = fromRedis(r, key)
        all_embedding[spkid] = embedding_1
    logger.info(f"Total : {len(all_embedding.keys())} embeddings in database.Use model type:{use_model_type}")
    return all_embedding


def get_embeddings_from_spkid_bak(spkid):
    """
    Get embeddings from spkid
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    embeddings = {}
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        spkid_db = key.split("_")[-1]
        if spkid_db == spkid:
            embedding = fromRedis(r, key)
            embeddings[key.replace("_"+spkid_db, '')] = embedding

    return embeddings


def get_embeddings_from_spkid(spkid):
    """
    Get embeddings from spkid
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    embeddings = {}
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        spkid_db = key.split("_")[-1]
        if spkid_db == spkid:
            embedding = fromRedis(r, key)
            embeddings[key.replace("_"+spkid_db, '')] = embedding

    return embeddings


def get_embeddings_from_db():
    """
    Get embeddings from db
    Returns:
        embeddings = {spkid: {model_type: embedding, ...}, ...}
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    embeddings = {}
    spkid_emb_dict = {}
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        spkid_db = key.split("_")[-1]
        if spkid_db not in spkid_emb_dict:
            emb_dict = {}
            for i in cfg.ENCODE_MODEL_LIST:
                i = i.replace("_", "")
                emb_dict[i] = fromRedis(r, i+"_"+spkid_db)
            spkid_emb_dict[spkid_db] = emb_dict
    embeddings = spkid_emb_dict
    return embeddings


def get_spkid():
    """
    Get spkid
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    spkids = set()
    for key in r.keys():
        key = key.decode("utf-8")
        if "_" not in key:
            continue
        uuid_db = key.split("_")[-1]
        spkids.add(uuid_db)

    return spkids


def inster_redis_db(embedding, spkid, use_model_type, mode="register"):
    """
    将emb存入redis数据库
    Args:
        embedding: emb
        spkid: 说话人id
        use_model_type: 使用的模型类型
        mode: register or test
    Returns:
        True or False
    """
    if not use_model_type:
        logger.error(f"No use_model_type. spkid:{spkid}")
        return False

    embedding_npy = np.array(embedding, dtype=np.float32)

    if mode == "register":
        db = cfg.REDIS["register_db"]
    else:
        db = cfg.REDIS["test_db"]
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=db,
        password=cfg.REDIS["password"],
    )
    toRedis(r, embedding_npy, f"{use_model_type}_{spkid}")

    return True


def delete_by_key(spkid):
    """
    删除redis中的emb
    Args:
        spkid: 说话人id
    """
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    keys = r.keys(f'*{spkid}*')

    for key in keys:
        key = key.decode("utf-8")
        r.delete(key)
        logger.info(f"Delete key:{key} success.")
    return


def save_redis_to_pkl():
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=cfg.REDIS["register_db"],
        password=cfg.REDIS["password"],
    )
    all_embedding = {}
    for key in r.keys():
        key = key.decode("utf-8")
        spkid = key
        embedding_1 = fromRedis(r, key)
        all_embedding[spkid] = {"embedding_1": embedding_1}
    with open(cfg.BLACK_BASE, "wb") as f:
        pickle.dump(all_embedding, f, pickle.HIGHEST_PROTOCOL)


############################################ mysql操作 ############################################
msg_db = cfg.MYSQL


def add_speaker(db_info):
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
    spkid = db_info.get("spkid")
    valid_length = db_info.get("valid_length")
    raw_url = db_info.get("raw_url")
    selected_url = db_info.get("selected_url")
    record_month = db_info.get("record_month")
    asr_text = db_info.get("asr_text")
    record_type = db_info.get("record_type")

    table_name = msg_db['black_table_name']
    try:
        query_sql = f"insert into {table_name} (record_id, valid_length,file_url,preprocessed_file_url,record_month,asr_text,record_type,register_time) VALUES (%s, %s, %s,%s,%s,%s,%s,now())"
        cursor.execute(query_sql, (spkid, valid_length, raw_url, selected_url, record_month, asr_text, record_type))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{spkid}. msg:{e}.")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)


def cosine_similarity(input_data):
    """
    计算余弦相似度
    Args:
        input_data: [db_item, db_embedding, embedding]
    Returns:
        [similarity_score, db_item]
    """
    db_item, db_embedding, embedding = input_data
    db_embedding = torch.tensor(db_embedding)
    return [similarity(db_embedding, embedding).numpy(), db_item]


def compare_handler(model_type=None, embedding=None, black_limit=0.78, top_num=10):
    """
    与黑库比对 返回最高分数和top1-top10
    Args:
        model_type: 使用的模型类型
        embedding: emb
        black_limit: 黑库阈值
        top_num: top_num
    Returns:
        return_results: {'best_score': 0.9999998211860657, 'inbase': 1, 'top_1': '1.00000', 'top_1_id': '18136655705'}
    """
    emb_db = get_embeddings(use_model_type=model_type)
    embedding = torch.tensor(embedding).to('cpu')
    input_data = [(k, emb_db[k], embedding) for k in emb_db.keys()]

    t1 = time.time()
    results = process_map(cosine_similarity, input_data, max_workers=1, chunksize=1000, desc='Doing----')
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


def add_hit(db_info):
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
        phone = db_info["spkid"]
        name = db_info["name"]
        gender = db_info["gender"]
        valid_length = db_info["valid_length"]
        file_url = db_info["file_url"]
        preprocessed_file_url = db_info["preprocessed_file_url"]
        message = db_info["message"]
        hit_score = db_info["hit_score"]
        hit_spkid = db_info["hit_spkid"]

        table_name = msg_db['hit_table_name']
        query_sql = f"insert into {table_name} (name,phone,gender,valid_length,file_url,preprocessed_file_url,message,hit_score,hit_spkid,hit_time) \
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s,now());"
        cursor.execute(query_sql, (name, phone, gender, valid_length, file_url, preprocessed_file_url, message, hit_score, hit_spkid))
        conn.commit()
    except Exception as e:
        logger.error(f"Insert to db failed. record_id:{phone}. msg:{e}.")
        conn.rollback()
    cursor.close()
    conn.close()
