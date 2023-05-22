# export redis data to a python dict and save as a npy file
import redis
import numpy as np
import os
import sys
import argparse
import json
import time
import sys
sys.path.append("../")
import cfg

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default=cfg.REDIS["host"], help='redis host')
parser.add_argument('--port', type=int, default=cfg.REDIS["port"], help='redis port')
parser.add_argument('--db', type=int, default=cfg.REDIS["register_db"], help='redis db')
parser.add_argument('--password', type=str, default=cfg.REDIS["password"], help='redis password')
parser.add_argument('--save_path', type=str, default='./vectorDB.bin', help='save path')
parser.add_argument('--save_txt_path', type=str, default='black_id_all.txt', help='save path')

args = parser.parse_args()

def fromRedis(r,n):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = r.get(n)
    a = np.frombuffer(encoded, dtype=np.float32, offset=8)
    return a


# connect to redis
r = redis.Redis(host=args.host, port=args.port, db=args.db, password=args.password)
# export db to a python dict
data = {}



for key in r.keys():
    if '_' not in key.decode('utf-8'):
        continue
    key = key.decode('utf-8')
    spkid = key.split("_")[1]
    embedding_1 = fromRedis(r,key)
    data[spkid] = {}
    data[spkid][spkid]= embedding_1
    # print(f"SPKID: {spkid}, embedding_1: {embedding_1}")
# print(data)
# 将Data保存到vectorDB.txt
# 文件有100万行，每一行是一个特征，每行有192个浮点数，用逗号分隔
data_list_all = []
black_id_all = []
# with open('vectorDB.txt', 'w') as f:
for key in data.keys():
    # f.write(str(data[key][key]).replace('[', '').replace(']', '').replace(' ', '') + '\n')
    data_list_all.append(data[key][key])
    black_id_all.append(key)
# 保存为二进制dtype=np.float32

data_list_all = np.array(data_list_all)
print(data_list_all.shape)
# reshape -1
data_list_all = data_list_all.reshape(-1)
# type to float32
data_list_all = data_list_all.astype(np.float32)
data_list_all.tofile(args.save_path)
# black_id_all 写入文件
with open(args.save_txt_path, 'w') as f:
    for item in black_id_all:
        f.write(item + '\n')