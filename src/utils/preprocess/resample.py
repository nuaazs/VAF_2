# coding = utf-8
# @Time    : 2022-09-05  15:32:55
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Resample.

from calendar import c
import torchaudio.transforms as T
import torchaudio
import os
from pathlib import Path


# cfg
import cfg

# log
from utils.log import logger
from utils.cmd import run_cmd

def read_wav_data(
    wav_filepath,
):
    wav, sr = torchaudio.load(wav_filepath)
    print(wav.shape)
    print(wav_filepath)
    assert wav.shape[0] == 1, f"wav channel != 1"
    wav = wav.reshape(-1)
    return wav.unsqueeze(0)

def resample(
    wav,
    sr,
    target_sr,
):
    
    if sr == target_sr:
        return wav
    wav = wav.unsqueeze(0)

    assert wav.shape[0] == 1
    assert len(wav.shape) == 2
    resample = T.Resample(sr, target_sr)
    wav = resample(wav)
    wav = wav.reshape(-1)
    return wav
