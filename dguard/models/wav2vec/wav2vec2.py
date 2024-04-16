#!/bin/bash
# coding = utf-8
# @Time    : 2024-03-23  16:14:26
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Need transformers==4.16.2

import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

# https://github.com/TencentGameMate/chinese_speech_pretrain
# https://github.com/kehanlu/Mandarin-Wav2Vec2
# Download model:
# huggingface-cli download --token xxx --resume-download TencentGameMate/chinese-wav2vec2-large --local-dir wav2vec2-large-model
# huggingface-cli download --token xxx --resume-download TencentGameMate/chinese-wav2vec2-base --local-dir wav2vec2-base-model

model_path="/VAF/dguard/models/wav2vec/wav2vec2-base-model"
wav_path="/VAF/test/data/test/cjsd300/13011899170/20220720091622/20220720091622_0.wav"
mask_prob=0.0
mask_length=10
base_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
base_model = Wav2Vec2Model.from_pretrained(model_path)

# device = torch.device("cpu")
# base_model = base_model.to(device)
# base_model.eval()

def get_feature(wav_data,sample_rate):
    # print(f"Get feature from wav data: {wav_data.shape}")
    wav_data = wav_data.reshape(-1)#.cuda()
    input_values = base_feature_extractor(wav_data, return_tensors="pt",sampling_rate=sample_rate).input_values
    with torch.no_grad():
        outputs = base_model(input_values)
        last_hidden_state = outputs.last_hidden_state
    # print(f"Last hidden state: {last_hidden_state.shape}")
    return last_hidden_state[0]