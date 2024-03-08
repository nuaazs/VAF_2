# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

import torchaudio
import torch
from dguard.models.feature_extractor.wavlm import wavlm_extractor

class preprocessor():
    def __init__(self, checkpoint="/VAF/train/pretrained_models/WavLM-Large.pt",device="cuda"):
        self.device = device
        ckpt = torch.load(checkpoint)
        self.cfg = WavLMConfig(ckpt['cfg'])
        print(f"Loading WavLM-Large cfg success")
        self.model = WavLM(self.cfg)
        print(f"Loading WavLM-Large success")
        self.model.load_state_dict(ckpt['model'])
        print(f"Loading WavLM-Large state_dict success")
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.model.eval()
        self.wav_reader = WavReader()
        self.label_encoder = SpkLabelEncoder()
        self.augmentations = NoiseReverbCorrupter()
        self.feature_extractor = wavlm_extractor(checkpoint="/VAF/train/pretrained_models/WavLM-Large.pt")
        self.sample_rate = 16000
        self.duration = 3.0
        self.speed_pertub = False
    def extract(wav_data):
        wav = torch.tensor(wav_data)
        wav = wav.unsqueeze(0)
        if self.device != "cpu":
            wav = wav.to(self.device)
        feat = self.feature_extractor(wav)
        return feat

def get_feature(wav_path, preprocessor,speed_pertub=True,sample_rate=16000):
    wav = torchaudio.load(wav_path)[0]
    if speed_pertub:
        speeds = [1.0, 0.9, 1.1]
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                wav.unsqueeze(0), sample_rate, [['speed', str(speeds[speed_idx])], ['rate', str(self.sample_rate)]])
                feat = preprocessor(wav)
        else:
            speed_idx = 0



    #         self.duration = duration
    #     self.sample_rate = sample_rate
    #     self.speed_pertub = speed_pertub

    # def __call__(self, wav_path):
    #     #TODO: add codec enhancement
    #     wav, sr = torchaudio.load(wav_path)
    #     assert sr == self.sample_rate
    #     wav = wav[0]

    #     if self.speed_pertub:
    #         speeds = [1.0, 0.9, 1.1]
    #         speed_idx = random.randint(0, 2)
    #         if speed_idx > 0:
    #             wav, _ = torchaudio.sox_effects.apply_effects_tensor(
    #                 wav.unsqueeze(0), self.sample_rate, [['speed', str(speeds[speed_idx])], ['rate', str(self.sample_rate)]])
    #     else:
    #         speed_idx = 0

    #     wav = wav.squeeze(0)
    #     data_len = wav.shape[0]

    #     chunk_len = int(self.duration * sr)
    #     if data_len >= chunk_len:
    #         start = random.randint(0, data_len - chunk_len)
    #         end = start + chunk_len
    #         wav = wav[start:end]
    #     else:
    #         wav = F.pad(wav, (0, chunk_len - data_len))

    #     return wav, speed_idx
    