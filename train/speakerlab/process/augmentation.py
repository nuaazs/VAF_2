# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
from scipy import signal
import numpy as np
import random

from speakerlab.utils.fileio import load_wav_scp


OGG = {
    "high": ["4.5k","5.5k","7.7k"],
    "mid": ["9.5k"],
    "low": ["12.5k","16.0k","32k"],
}


def addreverb(wav, rir_wav):
    # wav: [T,], rir_wav: [T,]
    wav = wav.numpy()
    rir_wav = rir_wav.numpy()
    wav_len = wav.shape[0]
    rir_wav = rir_wav / np.sqrt(np.sum(rir_wav**2))
    out_wav = signal.convolve(wav, rir_wav,
                                mode='full')[:wav_len]

    out_wav = out_wav / (np.max(np.abs(out_wav)) + 1e-6)
    return torch.from_numpy(out_wav)

def addnoise(wav, noise=None, snr_high=15, snr_low=0):
    # wav: [T,], noise: [T,]
    if noise is None:
        noise = torch.randn_like(waveform)
    noise = noise.numpy()
    wav = wav.numpy()

    wav_len = wav.shape[0]
    noise_len = noise.shape[0]
    if noise_len >= wav_len:
        start = random.randint(0, noise_len - wav_len)
        noise = noise[start:start + wav_len]
    else:
        noise = noise.repeat(wav_len // noise_len + 1)
        noise = noise[:wav_len]

    wav_db = 10 * np.log10(np.mean(wav**2) + 1e-6)
    noise_db = 10 * np.log10(np.mean(noise**2) + 1e-6)
    noise_snr = random.uniform(snr_low, snr_high)
    noise = np.sqrt(10**(
        (wav_db - noise_db - noise_snr) / 10)) * noise
    out_wav = wav + noise

    out_wav = out_wav / (np.max(np.abs(out_wav)) + 1e-6)
    return torch.from_numpy(out_wav)

class NoiseReverbCorrupter(object):
    def __init__(
        self,
        noise_prob=0.0,
        reverb_prob=0.0,
        noise_file=None,
        reverb_file=None,
        noise_snr_low=0,
        noise_snr_high=15,
    ):
        if reverb_prob > 0.0:
            if reverb_file is None:
                raise ValueError('Reverb_file not be assigned.')
            self.add_reverb = addreverb
            self.reverb_data = load_wav_scp(reverb_file)
            self.reverb_data_keys = list(self.reverb_data.keys())

        if noise_prob > 0.0:
            if noise_file is None:
                raise ValueError('Noise_file not be assigned.')

            self.add_noise = addnoise
            self.noise_data = load_wav_scp(noise_file)
            self.noise_data_keys = list(self.noise_data.keys())

        self.reverb_prob = reverb_prob
        self.noise_prob = noise_prob
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high

    def __call__(self, wav, fs=16000):
        if self.reverb_prob > random.random():
            reverb_path =  self.reverb_data[random.choice(self.reverb_data_keys)]
            reverb, fs_rir = torchaudio.load(reverb_path)
            assert fs_rir == fs
            wav = self.add_reverb(wav, reverb[0])
        if self.noise_prob > random.random():
            noise_path =  self.noise_data[random.choice(self.noise_data_keys)]
            noise, fs_noise = torchaudio.load(noise_path)
            assert fs_noise == fs
            wav = self.add_noise(
                wav, noise[0],
                snr_high=self.noise_snr_high,
                snr_low=self.noise_snr_low,)
        return wav



class NoiseReverbCorrupterCodec(object):
    def __init__(
        self,
        noise_prob=0.0,
        reverb_prob=0.0,
        noise_file=None,
        reverb_file=None,
        noise_snr_low=0,
        noise_snr_high=15,
        codec_prob=0.0,
    ):
        if reverb_prob > 0.0:
            if reverb_file is None:
                raise ValueError('Reverb_file not be assigned.')
            self.add_reverb = addreverb
            self.reverb_data = load_wav_scp(reverb_file)
            self.reverb_data_keys = list(self.reverb_data.keys())

        if noise_prob > 0.0:
            if noise_file is None:
                raise ValueError('Noise_file not be assigned.')

            self.add_noise = addnoise
            self.noise_data = load_wav_scp(noise_file)
            self.noise_data_keys = list(self.noise_data.keys())

        # if codec_prob > 0.0:
        #     self.codec_prob = codec_prob
        #     # random choose key from dict OGG
        #     self.codec_data = OGG
        #     self.codec_data_keys = list(self.codec_data.keys())
            
        self.reverb_prob = reverb_prob
        self.noise_prob = noise_prob
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high

    def __call__(self, wav, fs=16000):
        if self.reverb_prob > random.random():
            reverb_path =  self.reverb_data[random.choice(self.reverb_data_keys)]
            reverb, fs_rir = torchaudio.load(reverb_path)
            assert fs_rir == fs
            wav = self.add_reverb(wav, reverb[0])
        if self.noise_prob > random.random():
            noise_path =  self.noise_data[random.choice(self.noise_data_keys)]
            noise, fs_noise = torchaudio.load(noise_path)
            assert fs_noise == fs
            wav = self.add_noise(
                wav, noise[0],
                snr_high=self.noise_snr_high,
                snr_low=self.noise_snr_low,)
        # if random.random() < self.codec_prob:
        #     self.ogg_type_mode = random.choice(self.codec_data_keys)
        #     self.ogg_type = random.choice(self.codec_data[self.ogg_type_mode])
        #     # ffmpeg -y -i input.wav -c:a libopus -b:a 4.5k output.ogg && \
        #     # ffmpeg -y -i output.ogg -ar 8000 output.wav
        #     ffmpeg_cmd1 = f"ffmpeg -i - -acodec libvorbis -ab {self.ogg_type} -ar 8000 -ac 1"
        #     ffmpeg_cmd2 = f"ffmpeg -i {ogg_path} -ar {fs} -ac 1 -f wav -"
        #     wav = self.codec(wav)
        return wav
