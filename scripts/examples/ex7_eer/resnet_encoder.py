import sys
sys.path.append("/home/zhaosheng/wespeaker_models/src")
import os
import subprocess
import cfg
# load model to cuda:0

from speaker import Speaker
spker = Speaker('/home/zhaosheng/wespeaker_models/resnet221_LM_entire_model.pt', device='cpu')
print(f"Load model Success")
def generate_embedding(file_path,sr=16000):
    embedding = spker.extract_embedding(file_path)
    return embedding[0]
