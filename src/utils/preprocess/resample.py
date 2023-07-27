# coding = utf-8
# @Time    : 2022-09-05  15:32:55
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Resample.

from calendar import c
import torch
import torchaudio.transforms as T
import torchaudio
import os
from pathlib import Path


# cfg
import cfg

# log
from utils.log import logger
from utils.cmd import run_cmd

class resample_pipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        resample_freq=8000,
    ):
        super().__init__()
        self.resample = T.Resample(orig_freq=input_freq, new_freq=resample_freq)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        return resampled

r = resample_pipeline(8000,16000)
r.to(cfg.DEVICE)

def read_wav_data(
    wav_filepath,
):
    wav, sr = torchaudio.load(wav_filepath)
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
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)
    # assert wav.shape[0] == 1
    assert len(wav.shape) == 2
    # resample = T.Resample(sr, target_sr)
    wav = r(wav)
    # wav = wav.reshape(-1)
    return wav
