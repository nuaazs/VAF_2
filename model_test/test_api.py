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
if torch.cuda.is_available():
    device = torch.device('cuda', 0)
model_path="/home/duanyibo/dyb"
wav_path="/VAF/model_test/test.wav"
mask_prob=0.0
mask_length=10
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
model = model.to(device)
model = model.half()
for param in model.parameters():
    param.requires_grad = False
wav, sr = sf.read(wav_path)
input_values = feature_extractor(wav, return_tensors="pt",sampling_rate=16000).input_values
input_values = input_values.half()
input_values = input_values.to(device)
with torch.no_grad():
    outputs = model(input_values)
    last_hidden_state = outputs.projected_states
    print(last_hidden_state.size())