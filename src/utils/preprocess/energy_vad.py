import torch
import torchaudio as ta
import torchaudio.functional as AF
import math
import logging
import numpy as np

# import matplotlib.pyplot as plt

def preemphasis(signal, preemph=0.97):
    """
    Pre-emphasis on the input signal
    :param signal: (time,)
    :param preemph:
    :return: (time,)
    """
    return torch.cat((signal[0:1], signal[1:] - preemph * signal[:-1]))

def framesig(signal, framelen, framehop, winfunc=lambda x: torch.ones((x,))):
    """
    Frame a signal into overlapping frames.
    :param signal: (time,)
    :param framelen:
    :param framehop:
    :param winfunc:
    :return: (nframes, framelen)
    """
    slen = len(signal)
    framelen = round(framelen)  # round_half_up(framelen)
    framehop = round(framehop)  # round_half_up(framehop)
    if slen <= framelen:
        nframes = 1
    else:
        nframes = 1 + int(math.ceil((1.0 * slen - framelen) / framehop))

    padlen = int((nframes - 1) * framehop + framelen)

    zeros = torch.zeros((padlen - slen,))
    padsignal = torch.cat((signal, zeros))

    indices = torch.arange(0, framelen).view((1, -1)) \
              + torch.arange(0, nframes * framehop, framehop).view((-1, 1))
    frames = padsignal[indices]
    win = winfunc(framelen).view((1, -1))
    return frames * win

def calculate_frequencies(frames, samplerate):
    """
    Calculate half frequencies of the sample rate.
    :param frame: (nframes, framelen)
    :param samplerate:
    :return: (framelen//2,) or or (framelen//2 - 1,)
    """
    # 半频
    winlen = frames.shape[1]
    t = winlen * 1.0 / samplerate
    if samplerate % 2 == 0:
        return torch.arange(1, winlen // 2 + 1) / t
    else:
        return torch.arange(1, (winlen - 1) // 2 + 1) / t


def calculate_energy(frames):
    """
    Calculate energy of each frame by rfft.
    :param frame: (nframes, framelen)
    :return: (nframes, framelen//2) or (nframes, framelen//2 - 1) 
             that equals to half frequencies
    """
    # mag = torch.norm(torch.rfft(frames, 1), dim=2)[:,1:]
    mag = torch.norm(torch.view_as_real(torch.fft.rfft(frames, dim=1)), dim=2)[:,1:]
    energy = mag ** 2
    return energy


def freq_energy(frames, samplerate):
    """
    Calculate a pair of (frequencies, energy) of each frame.
    :param frame: (nframes, framelen)
    :param samplerate:
    :return: freq (framelen//2,) or (framelen//2-1,)
            energy (nframes, framelen//2) or (nframes, framelen//2 - 1) corresponding to freq
    """
    freq = calculate_frequencies(frames, samplerate)
    energy = calculate_energy(frames)
    return freq, energy

def energy_ratio(freq, energy, thresEnergy, lowfreq, highfreq):
    """
    Calculate the ratio between energy of speech band and total energy for a frame.
    :param freq: (winlen//2)
    :param energy: (nframes, winlen//2)
    :param lowfreq:
    :param highfreq:
    :return: (nframes,)
    """
    voice_energy = torch.sum(energy[:, (freq > lowfreq) & (freq < highfreq)], dim=1)
    full_energy = torch.sum(energy, dim=1)
    full_energy[full_energy == 0] = 2.220446049250313e-16  # 极小正数替换 0
    detection = (torch.div(voice_energy, full_energy) >= thresEnergy).type(torch.float32)
    return detection

def smooth_detection(detection, winlen, speechlen):
    """
    Apply median filter with length of {speechlen} to smooth detected speech regions
    :param detect: (nframes,)
    :param winlen:
    :param speechlen:
    :return: (nframes,)
    """
    medianwin = max(int(speechlen / winlen), 1)
    if medianwin % 2 == 0:
        medianwin -= 1
    mid = (medianwin - 1) // 2
    y = torch.zeros((len(detection), medianwin), dtype=detection.dtype)
    y[:, mid] = detection
    for i in range(mid):
        j = mid - i
        y[j:, i] = detection[:-j]
        y[:j, i] = detection[0]
        y[:-j, -(i + 1)] = detection[j:]
        y[-j:, -(i + 1)] = detection[-1]
    medianEnergy, _ = torch.median(y.type(torch.float), dim=1)
    return medianEnergy


def energy_VAD(wav, sr=16000, winlen=0.02, hoplen=0.01, thresEnergy=0.6, speechlen=0.5,
              lowfreq=300, highfreq=3000, preemph=0.97):
    """
    Use signal energy to detect voice activity in PyTorch's Tensor.
    Detects speech regions based on ratio between speech band energy and total energy.
    Outputs are two tensor with the number of frames where the first output is start frame
        and the second output is to indicate voice activity.
    :param wav: (time,)
    :param samplerate:
    :param winlen:
    :param hoplen:
    :param thresEnergy:
    :param speechlen:
    :param lowfreq:
    :param highfreq:
    :param preemph:
    :return: (nframes,), (nframes,), (time)
    """
    if len(wav) < round(winlen * sr):
        return torch.tensor([0]), torch.tensor([0])
    wav = preemphasis(wav, preemph=preemph)
    # tuple to torch tensor
    # wav = np.array(wav)
    # wav = torch.from_numpy(wav)
    frames = framesig(wav, winlen * sr, hoplen * sr)
    freq, energy = freq_energy(frames, sr)
    detection = energy_ratio(freq, energy, thresEnergy, lowfreq, highfreq)
    detection = smooth_detection(detection, winlen, speechlen)
    starts = torch.arange(0, frames.shape[0]) * round(hoplen * sr)
    return detection, starts, wav

if __name__ == '__main__':
    wav_file = '/lyxx/online/src/api_test/test.wav'
    wav, sr = ta.load(wav_file)
    wav = wav[0]
    detection, starts, wav = energy_VAD(wav, sr=sr, winlen=0.02, hoplen=0.01,
                                thresEnergy=0.6, speechlen=0.5, lowfreq=300, 
                                highfreq=3000, preemph=0.97)
    plt.figure(figsize=(12, 4))
    plt.plot(wav)
    plt.plot(starts, detection*0.1)