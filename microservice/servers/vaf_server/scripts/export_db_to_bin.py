#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   export_db_to_bin.py
@Time    :   2023/10/30 13:34:43
@Author  :   Carry
@Version :   1.0
@Desc    :   将redis中的特征数据导出为二进制文件，共输出四个文件，分别为：三个模型的特征向量文件，一个黑名单id文件，文件名为：模型名_vectorDB.bin，black_id_all.txt，已做升序排序
'''

import cfg
from tqdm import tqdm
import redis
import numpy as np
import argparse
import sys
sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default=cfg.REDIS["host"], help='redis host')
parser.add_argument('--port', type=int, default=cfg.REDIS["port"], help='redis port')
parser.add_argument('--db', type=int, default=cfg.REDIS["register_db"], help='redis db')
parser.add_argument('--password', type=str, default=cfg.REDIS["password"], help='redis password')

args = parser.parse_args()


def fromRedis(r, n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    a = np.frombuffer(encoded, dtype=np.float32, offset=8)
    return a


r = redis.Redis(host=args.host, port=args.port, db=args.db, password=args.password)
data = {}


for model in tqdm(cfg.ENCODE_MODEL_LIST):
    model = model.replace("_", "")
    data[model] = {}
    for key in r.keys():
        key = key.decode('utf-8')
        if model not in key:
            continue
        spkid = key.split("_")[1]
        embedding_1 = fromRedis(r, key)
        data[model][spkid] = embedding_1

# sort by spkid
for key in data.keys():
    data[key] = dict(sorted(data[key].items(), key=lambda x: x[0]))

for model in cfg.ENCODE_MODEL_LIST:
    model = model.replace("_", "")
    data_list_all = []
    black_id_all = []
    for key in data[model].keys():
        # f.write(str(data[key][key]).replace('[', '').replace(']', '').replace(' ', '') + '\n')
        data_list_all.append(data[model][key])
        black_id_all.append(key)

    # 保存为二进制dtype=np.float32
    data_list_all = np.array(data_list_all)
    print(data_list_all.shape)
    # reshape -1
    data_list_all = data_list_all.reshape(-1)
    # type to float32
    data_list_all = data_list_all.astype(np.float32)
    data_list_all.tofile(f'./output/{model}_vectorDB.bin')

with open(f'./output/black_id_all.txt', 'w') as f:
    for item in black_id_all:
        f.write(item + '\n')

print("Done!")
