# coding = utf-8
# @Time    : 2022-09-05  15:13:14
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Pre classify.

import torch
from utils.encoder import similarity
import cfg


def classify(embedding):
    max_class_score = 0
    max_class_index = 0
    if cfg.CLASSIFY:
        for index, i in enumerate(torch.eye(192).to(cfg.DEVICE)):
            now_class_score = similarity(embedding, i)
            if now_class_score > max_class_score:
                max_class_score = now_class_score
                max_class_index = index
    return max_class_index
