# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from torch.utils.data import Dataset
from dguard.utils.fileio import load_data_csv


class BaseSVDataset(Dataset):
    def __init__(self, data_file: str,  preprocessor: dict):
        self.data_points = self.read_file(data_file)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data_points)


class WavSVDataset(BaseSVDataset):
    def __init__(self, data_file: str, preprocessor: dict, device="cpu",load_feat_direrctly=False):
        super(WavSVDataset, self).__init__(data_file, preprocessor)
        self.device = device

    def __getitem__(self, index):
        data = self.get_data(index)
        wav_path = data['path']
        spk = data['spk']
        wav, speed_index = self.preprocessor['wav_reader'](wav_path)
        spkid = self.preprocessor['label_encoder'](spk, speed_index)
        wav = self.preprocessor['augmentations'](wav)
        wav = wav.unsqueeze(0)
        if self.device != "cpu":
            wav = wav.to(self.device)
        # print(f"Wav shape: {wav.shape}")
        if "feature_extractor" in self.preprocessor:
            feat = self.preprocessor['feature_extractor'](wav)
            return feat, spkid
        else:
            return wav, spkid

    def get_data(self, index):
        if not hasattr(self, 'data_keys'):
            self.data_keys = list(self.data_points.keys())
        key = self.data_keys[index]

        return self.data_points[key]

    def read_file(self, data_file):
        return load_data_csv(data_file)
