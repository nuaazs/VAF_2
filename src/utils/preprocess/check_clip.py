# coding = utf-8
# @Time    : 2022-09-05  15:12:59
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: clip detact.

import numpy as np


def norm_numpy_array(wav):
    return wav / np.max(np.abs(wav))


def check_clip(wav, th):
    wav = np.array(wav)
    data = norm_numpy_array(wav)
    if (len(data[data > 0.99]) / data.shape[0]) > th:
        return True
    return False
