import random
import pickle
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
from torchaudio.sox_effects import apply_effects_file

class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

if __name__ == '__main__':
    model = FBank(n_mels=80, sample_rate=16000, mean_nor=True)
    EFFECTS = [
        ["remix", "-"],
        ["channels", "1"],
        ["rate", "16000"],
        ["gain", "-1.0"],
        ["silence", "1", "0.1", "0.1%", "-1", "0.1", "0.1%"],
        ["trim", "0", "10"],
    ]
    input_values = apply_effects_file("/VAF/train/data/raw_data/voxceleb1/dev/wav/id10001/7gWzIy6yIIk/00002.wav", EFFECTS)
    input_values = torch.tensor(input_values[0].numpy())
    # input_values = input_values.unsqueeze(0)
    print(input_values.shape)
    feature = model(input_values)
    print(feature.shape)
    print(feature)