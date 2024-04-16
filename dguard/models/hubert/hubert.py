

import torch
import torch.nn.functional as F
import soundfile as sf

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)





# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
# Download model:
# huggingface-cli download --token xxx --resume-download TencentGameMate/chinese-hubert-large --local-dir hubert-large-model
# huggingface-cli download --token xxx --resume-download TencentGameMate/chinese-hubert-base --local-dir hubert-base-model

model_path="/VAF/dguard/models/hubert/hubert-base-model"
base_model = HubertModel.from_pretrained(model_path)
base_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

def get_feature(wav_data,sample_rate):
    # print(f"Get feature from wav data: {wav_data.shape}")
    wav_data = wav_data.reshape(-1)#.cuda()
    input_values = base_feature_extractor(wav_data, return_tensors="pt",sampling_rate=sample_rate).input_values
    with torch.no_grad():
        outputs = base_model(input_values)
        last_hidden_state = outputs.last_hidden_state
    print(f"Last hidden state: {last_hidden_state.shape}")
    return last_hidden_state[0]