#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   compare_with_db.py
@Time    :   2023/09/18 12:11:58
@Author  :   Carry
@Version :   1.0
@Desc    :   与存量黑库进行比对
'''


import numpy as np
import pandas
import torch
import cfg
from utils.orm.db_orm import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity

old_emb_db = get_embeddings(use_model_type=cfg.USE_MODEL_TYPE)

# 获取新的emb
read_data = np.fromfile("output/emb.bin", dtype=np.float32)
print(read_data.shape)
read_data = read_data.reshape(-1, cfg.EMBEDDING_LEN[cfg.USE_MODEL_TYPE])
print(read_data.shape)

new_embs = {}
with open("output/emb_map.txt", "r") as f:
    for idx, line in enumerate(f.readlines()):
        new_embs[line.strip()] = read_data[idx]
print(len(new_embs))


def compare(inut_csv_file, output_csv_file):
    info_li = pandas.read_csv(inut_csv_file, skiprows=1, header=None)
    info_li = info_li.values.tolist()

    result_li = []
    for i in info_li:
        spkid = i[0]
        tensor1 = torch.tensor(new_embs[str(spkid)], dtype=torch.float32)
        for j in old_emb_db:
            tensor2 = torch.tensor(old_emb_db[j], dtype=torch.float32)
            # 计算余弦相似度
            score = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=0)
            if score.item() < 0.82:
                print(spkid, j, score.item())
                result_li.append(i)
                break
            else:
                print(f"gt 0.82 {spkid} {j} {score.item()}")
    pandas.DataFrame(result_li).to_csv(output_csv_file, index=False, header=None)
    print(f"Before len:{len(info_li)},After len:{len(result_li)}")


inut_csv_file = f"output/check_result_2023-09-20.csv"
output_csv_file = f"output/check_result_2023-09-20_finally.csv"
compare(inut_csv_file, output_csv_file)
