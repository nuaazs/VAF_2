# coding = utf-8
# @Time    : 2022-09-05  15:35:17 # @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Scores.
import ctypes
import os

import torch
import numpy as np  # import sys
# sys.path.append("/ssd2/voiceprint-recognition-system/src/")
from utils.orm.query import get_wav_url
import cfg
import multiprocessing
from utils.orm import get_embeddings
from utils.log import logger
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
# database = get_embeddings(class_index=-1)


BLACK_ID = {}
for _model in cfg.NAMELIST.keys():
    BLACK_ID[_model] = {}
    with open(cfg.NAMELIST[_model], "r") as f:
        for index, line in enumerate(f.readlines()):
            BLACK_ID[_model][index] = line.strip()
# # load black_id
# try:
#     file_path = "cpp/black_id_all.txt"
#     # read file as dict key:value -> index:spkid
#     black_id = {}
#     with open(file_path, "r") as f:
#         for index, line in enumerate(f.readlines()):
#             black_id[index] = line.strip()
# except:
#     from utils.log import logger

#     logger.error("black_id_all.txt not found")


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
    base_item, embedding, database = input_data
    base_embedding = torch.tensor(database[base_item]["embedding_1"])
    return [similarity(base_embedding, embedding).numpy(), base_item]


lib = ctypes.cdll.LoadLibrary('./cpp/lib/get_top.so')

# 修改函数参数类型
calculate = lib.test
calculate.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
calculate.restype = ctypes.c_float


def get_similarity_by_np(embedding, index, embedding_type,score):
    """
    通过np文件计算相似度
    """
    read_data = np.fromfile(f'/VAF/src/cpp/test/{embedding_type.lower()}_0103_a.bin', dtype=np.float32)
    read_data = read_data.reshape(-1, cfg.EMBEDDING_LEN[embedding_type])
    with open(f'/VAF/src/cpp/test/{embedding_type.lower()}_0103_a.txt', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if line.strip() == index:
                print(idx)
                break
    base_embedding = torch.tensor(read_data[idx])
    score2 = similarity(base_embedding, embedding)
    print(f"embedding_type: {embedding_type}, score: {score}, score2: {score2}")
    return score2


def retrieve(data, embedding_type, shmid_file=b'shmid.txt', voicenum=88275, featsize=192):
    # return 0.99,"15151832002"

    input_data = (ctypes.c_float * len(data))(*data)
    result = calculate(input_data, voicenum, featsize, shmid_file)
    try:
        # ctypes.c_free(input_data)
        # ctypes.pythonapi.PyMem_Free(input_data)
        del input_data
    except Exception as e:
        logger.info(f"del input_data error: {e}")
        print(f"del error: {e}")
    logger.info(f"Hit result:{result}")
    print("del success")
    if int(result) == result:
        if result > 0:
            score = 1.0
            index = str(int(result) - 1)
            spkid = BLACK_ID[embedding_type][int(index)]
            return score, spkid
        else:
            score = -1.0
            index = str(int(abs(result)) - 1)
            spkid = BLACK_ID[embedding_type][int(index)]
            return score, spkid
    if result >= 0:
        sign_flag = 1
    else:
        sign_flag = -1
    result = abs(result)
    score = (result - int(result)) * sign_flag
    index = str(int(result))
    spkid = BLACK_ID[embedding_type][int(index)]
    return score, spkid


def test_wav(embedding, black_limit, embedding_type):
    embedding = torch.tensor(embedding).to('cpu')
    # input = [(base_item, embedding) for base_item in database.keys()]
    # results = pool.map(cosine_similarity, input)
    shmid_txt = cfg.SHMID[embedding_type]
    # shmid_txt -> b'shmid.txt'
    shmid_txt = shmid_txt.encode('utf-8')
    featsize = cfg.EMBEDDING_LEN[embedding_type]
    id_list_txt_file = cfg.NAMELIST[embedding_type]
    # voicenum = id_list_txt_file文件的行数
    voicenum = len(open(id_list_txt_file, 'r').readlines())
    score, index = retrieve(embedding.numpy().reshape(cfg.EMBEDDING_LEN[embedding_type], ).tolist(), embedding_type, shmid_txt, voicenum, featsize)
    top_10 = [[score, index], [score, index], [score, index], [score, index], [score, index], [score, index],
              [score, index], [score, index], [score, index], [score, index]]
    best_score = score
    best_id = index
    top_10_str = ""
    for i, _data in enumerate(top_10, 1):
        # str("|".join(map(str, np.array(top_10))))
        if i < 10:
            top_10_str += f"{_data[0]}_{_data[1]}|"
        else:
            top_10_str += f"{_data[0]}_{_data[1]}"
    inbase = best_score >= black_limit
    return inbase, {"best_score": best_score, "spk": best_id, "top_10": top_10_str}


pool = multiprocessing.Pool(processes=cfg.TEST_THREADS, initializer=print("pool start"))


def get_similarity(embedding, black_limit, embedding_type):
    """
    实时读取黑库进行比对，返回最高分和对应的说话人
    """
    database = get_embeddings(class_index=-1)
    return_results = {}
    embedding = torch.tensor(embedding).to('cpu')
    input = [(base_item, embedding, database) for base_item in database.keys() if base_item.split("_")[0] == embedding_type]
    results = pool.map(cosine_similarity, input)
    results = sorted(results, key=lambda x: float(x[0]) * (-1))
    top_10 = [f"{_score},{_spk_id}" for _score, _spk_id in results[:10]]
    best_score = float(np.array(results[0][0]))
    best_id = str(",".join(map(str, np.array(results)[:10, 1])))
    top_10 = str("|".join(map(str, np.array(top_10))))
    return_results["best_score"] = best_score
    inbase = best_score >= black_limit
    return inbase, {"best_score": best_score, "spk": best_id, "top_10": top_10}
