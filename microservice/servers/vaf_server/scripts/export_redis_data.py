#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   export_redis_data.py
@Time    :   2023/10/31 14:06:09
@Author  :   Carry
@Version :   1.0
@Desc    :   将redis中数据导出为npy文件,共输出三个文件，分别为：三个模型的特征向量文件，文件名为：模型名.npy,输出在output文件夹下
'''


import numpy as np
import redis

# 连接到 Redis 服务器
REDIS = {
    "host": "127.0.0.1",
    "port": 6379,
    "register_db": 0,
    "test_db": 2,
    "password": "",
}


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    a = np.frombuffer(encoded, dtype=np.float32, offset=8)
    return a


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
        host=REDIS["host"],
        port=REDIS["port"],
        db=REDIS["register_db"],
        password=REDIS["password"],
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
    print(f"Total : {len(all_embedding.keys())} embeddings in database.Use model type:{use_model_type}")
    return all_embedding


# import npy file to redis
def import_npy_to_redis(npy_file):
    r = redis.Redis(
        host=REDIS["host"],
        port=REDIS["port"],
        db=REDIS["register_db"],
        password=REDIS["password"],
    )
    all_embedding = np.load(npy_file, allow_pickle=True).item()
    for k, v in all_embedding.items():
        k = f"{npy_file.split('.')[0]}_{k}"
        r.set(f"{k}", v.tobytes())
        print(f"{k} {v}")
    print(f"Total : {len(all_embedding.keys())} embeddings in database.Use model type:{npy_file}")


for i in ["resnet101cjsd", "resnet221cjsdlm", "resnet293cjsdlm"]:
    all_embedding = get_embeddings(i)
    # save to npy
    np.save(f"./output/{i}.npy", all_embedding)
    # save to txt
    with open(f"./output/{i}.txt", "w") as f:
        for k, v in all_embedding.items():
            f.write(f"{k} {v}\n")
print("Done!")
