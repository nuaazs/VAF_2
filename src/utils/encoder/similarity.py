# coding = utf-8
# @Time    : 2022-09-05  15:04:36
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: models.

import torch
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
