import torch


def remove(wav, start, end):
    wav_out = wav.clone()
    wav_out[:, start:end] = 0
    return wav_out
