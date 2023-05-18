# coding = utf-8
# @Time    : 2022-09-05  15:35:17
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Scores.
import os

import torch
import numpy as np
# import sys
# sys.path.append("/ssd2/voiceprint-recognition-system/src/")
from utils.orm.query import get_wav_url
import cfg
import multiprocessing
from utils.orm import get_embeddings
from utils.log import logger
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
database = get_embeddings(class_index=-1)

# load black_id
try:
    file_path = "cpp/black_id_all.txt"
    # read file as dict key:value -> index:spkid
    black_id = {}
    with open(file_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            black_id[index] = line.strip()
except:
    from utils.log import logger

    logger.error("black_id_all.txt not found")


# print(black_id)

def get_scores(database, new_embedding, black_limit, similarity, top_num=10):
    return_results = {}
    results = []
    top_list = ""
    # Read embeddings in database
    for base_item in database:
        base_embedding = torch.tensor(database[base_item]["embedding_1"])
        results.append([similarity(base_embedding, new_embedding), base_item])
    results = sorted(results, key=lambda x: float(x[0]) * (-1))
    return_results["best_score"] = float(np.array(results[0][0]))

    if results[0][0] <= black_limit:
        return_results["inbase"] = 0
        return return_results, top_list
    else:
        return_results["inbase"] = 1
        # top1-top10
        for index in range(top_num):
            return_results[f"top_{index + 1}"] = f"{results[index][0].numpy():.5f}"
            return_results[f"top_{index + 1}_id"] = str(results[index][1])
            top_list += f"Top {index + 1} 评分:{results[index][0].numpy():.2f} 说话人:{results[index][1]}<br/>"
    return return_results, top_list


def cosine_similarity(input_data):
    base_item, embedding = input_data
    base_embedding = torch.tensor(database[base_item]["embedding_1"])
    return [similarity(base_embedding, embedding).numpy(), base_item]


# pool = multiprocessing.Pool(processes=cfg.TEST_THREADS, initializer=print("pool start"))


import ctypes

lib = ctypes.cdll.LoadLibrary('./cpp/libexample.so')

calculate = lib.test
calculate.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]  #
calculate.restype = ctypes.c_float


def retrieve(data):
    input_data = (ctypes.c_float * len(data))(*data)
    result = calculate(input_data, len(data))
    logger.info(f"Hit result:{result}")
    if int(result) == result:
        if result>0:
            score = 1.0
            index = str(int(result) - 1)
            spkid = black_id[int(index)]
            return score, spkid
        else:
            score = -1.0
            index = str(int(abs(result)) - 1)
            spkid = black_id[int(index)]
            return score, spkid
    if result >= 0:
        sign_flag = 1
    else:
        sign_flag = -1
    result = abs(result)
    score = (result - int(result)) * sign_flag
    index = str(int(result))
    spkid = black_id[int(index)]
    return score, spkid


def test_wav(database, embedding, black_limit):
    embedding = torch.tensor(embedding).to('cpu')
    input = [(base_item, embedding) for base_item in database.keys()]
    # results = pool.map(cosine_similarity, input)
    score, index = retrieve(embedding.numpy().reshape(192, ).tolist())
    top_10 = [[score, index], [score, index], [score, index], [score, index], [score, index], [score, index],
              [score, index], [score, index], [score, index], [score, index]]
    best_score = score
    best_id = index
    # results = sorted(results, key=lambda x: float(x[0]) * (-1))
    # top_10 = [f"{_score},{_spk_id}" for _score, _spk_id in results[:10]]
    # best_score = float(np.array(results[0][0]))
    # best_id = str(",".join(map(str, np.array(results)[:10, 1])))
    top_10 = str("|".join(map(str, np.array(top_10))))
    inbase = best_score >= black_limit
    return inbase, {"best_score": best_score, "spk": best_id, "top_10": top_10}
