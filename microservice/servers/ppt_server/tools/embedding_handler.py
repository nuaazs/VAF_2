#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   vad_handler.py
@Time    :   2023/10/14 16:52:59
@Author  :   Carry
@Version :   1.0
@Desc    :   声纹编码
'''
import cfg
import torch
import torchaudio
import numpy as np
from dguard.interface.pretrained import load_by_name

# load model
model_info = {}
for model_type in cfg.ENCODE_MODEL_LIST:
    model_info_dict = {}
    model_name = model_type
    device = cfg.DEVICE
    model, feature_extractor, sample_rate = load_by_name(model_name, device)
    model_info_dict["model"] = model
    model_info_dict["feature_extractor"] = feature_extractor
    model_info_dict["sample_rate"] = sample_rate
    model.eval()
    model.to(device)
    model_info[model_type] = model_info_dict


def encode_files(spkid, wav_files,  start=0, end=999, need_list=False):
    """
    提取特征
    :param spkid: 说话人id
    :param wav_files: 音频文件路径
    :param start: 开始时间
    :param end: 结束时间
    :param need_list: 是否需要list
    :return: embedding
    """
    file_emb = {}
    for model_type in cfg.ENCODE_MODEL_LIST:
        model_info_dict = model_info[model_type]
        model = model_info_dict["model"]
        feature_extractor = model_info_dict["feature_extractor"]
        sample_rate = model_info_dict["sample_rate"]

        file_emb[model_type] = {}
        file_emb[model_type]["embedding"] = {}

        for _index, wav_file in enumerate(wav_files):
            _data, sr = torchaudio.load(wav_file)
            assert sr == sample_rate, f"File {wav_file} sr is {sr}, not {sample_rate}."
            _data = _data.reshape(1, -1)
            _data = _data[:, int(start*sr):int(end*sr)]
            feat = feature_extractor(_data)
            feat = feat.unsqueeze(0)
            feat = feat.to(cfg.DEVICE)
            with torch.no_grad():
                embeddings = model(feat)[-1].detach().cpu().numpy()
            embeddings = embeddings.astype(np.float32).reshape(-1)
            if need_list:
                file_emb[model_type]["embedding"][spkid] = embeddings.tolist()
            else:
                file_emb[model_type]["embedding"][spkid] = embeddings
    return file_emb
