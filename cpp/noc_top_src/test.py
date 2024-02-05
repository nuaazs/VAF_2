# python Np 读取verctorDB.bin 获得声纹黑库shape（80000,192）
# 读取vectorA.txt 获得声纹A shape（1,192）
# 循环计算声纹A与声纹黑库的余弦相似度

import numpy as np
import time
import os

import torch

cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def get_vectorDB():
    # 读取声纹黑库
    vectorDB = np.fromfile('./code/vectorDB.bin', dtype=np.float32)
    vectorDB = vectorDB.reshape(-1, 192)
    return vectorDB

def get_vectorA():
    # 读取声纹A
    vectorA = np.loadtxt('./vectorA.txt')
    vectorA = vectorA.reshape(192,).tolist()
    return vectorA

import ctypes
lib = ctypes.cdll.LoadLibrary('./libexample.so')

calculate = lib.test
calculate.argtypes = [ctypes.POINTER(ctypes.c_float) , ctypes.c_size_t]#
calculate.restype = ctypes.c_float

def retrieve(data):
    input_data = (ctypes.c_float * len(data))(*data)
    result = calculate(input_data, len(data))
    if result >=0:
        sign_flag = 1
    else:
        sign_flag = -1
    result = abs(result)
    score = (result - int(result))*sign_flag
    index = str(int(result))
    return score,index

    # 将结果转换成字典并返回
    # return {'id': result.id, 'similarity': result.similarity}

def retrieve_python(A):
    A = np.array(A)
    A = A.reshape(-1, 192)
    db = get_vectorDB()
    score_list = []
    for B in db:
        B = B.reshape(-1, 192)
        score = cosine_similarity(torch.from_numpy(A), torch.from_numpy(B))
        # print(score)
        score_list.append(score)
    score = max(score_list)
    index = score_list.index(max(score_list))
    return score,index

if __name__ == '__main__':
    # 加载动态链接库
    As = np.random.rand(100,192)
    for A in As:
        A = A.reshape(192,).tolist()
        score,index = retrieve(A)
        # score_python,index_python = retrieve_python(A)
        print(f"score:{score},index:{index}")
        # print(f"score_python:{score_python},index_python:{index_python}")

