# export redis data to a python dict and save as a npy file
import redis
import numpy as np
import os
import sys
import argparse
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='192.168.3.202', help='redis host')
parser.add_argument('--port', type=int, default=6379, help='redis port')
parser.add_argument('--db', type=int, default=1, help='redis db')
parser.add_argument('--password', type=str, default=None, help='redis password')
parser.add_argument('--save_path', type=str, default='/home/zhaosheng/utils/examples/ex13_export_redis_to_npy/redis_data_43804_black.npy', help='save path')
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
# 获取一个(n, 192)的数组
data_ = np.array([data[key][key] for key in data.keys()])
# assert data_.shape == (1000000, 192)
# 将Data用二进制的形式保存到vectorDB.bin
data_.tofile('vectorDB.bin')
# 读取二进制文件
data_ = np.fromfile('vectorDB.bin', dtype=np.float32)
print(data_.shape)