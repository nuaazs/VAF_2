import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import random
from dguard.interface.pretrained import load_by_name,ALL_MODELS
from sklearn.metrics.pairwise import cosine_similarity
# from encode_wav import duanyibo # 特征提取函数
import argparse
from IPython import embed

model,feature_extractor,sample_rate = load_by_name("repvgg",0) # repvgg eres2net
model.eval()
# assert 16k, if not convert
wav, sr = torchaudio.load("duanyibo_01_16k.wav")
wav2, sr2 = torchaudio.load("duanyibo_02_16k.wav")
# embed()
# wav=wav.to("cuda:0")
feat = feature_extractor(wav)
feat2 = feature_extractor(wav2)
feat = feat.unsqueeze(0)
feat2 = feat2.unsqueeze(0)
feat = feat.to(0)
feat2 = feat2.to(0)
with torch.no_grad():
    emb = model(feat)[-1].detach().cpu().numpy()
    print(emb.shape)
    emb2 = model(feat2)[-1].detach().cpu().numpy()
    print(emb2.shape)

cosine_score = cosine_similarity(emb.reshape(1, -1), emb2.reshape(1, -1))[0][0]
print(cosine_score)