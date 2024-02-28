# coding = utf-8
# @Time    : 2023-03-08  18:45:30
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: ResNet34 Encoder for Speaker identification.

import argparse
import os

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddleaudio
import yaml
import torch

from paddleaudio.transforms import *
from paddleaudio.utils import get_logger
import subprocess
import metrics
from dataset import get_val_loader
from models import EcapaTDNN, ResNetSE34, ResNetSE34V2
import cfg

logger = get_logger()


class Normalize:
    def __init__(self, mean_file, eps=1e-5):
        self.eps = eps
        mean = paddle.load(mean_file)['mean']
        self.mean = mean.unsqueeze((0, 2))

    def __call__(self, x):
        assert x.ndim == 3
        return x - self.mean
    
file2feature = {}

with open(cfg.config_file) as f:
    config = yaml.safe_load(f)
paddle.set_device(cfg.DEVICE)
print(f"Device: {cfg.DEVICE}")
logger.info('model:' + config['model']['name'])
print('model:' + config['model']['name'])
logger.info(f'using ' + config['model']['name'])
ModelClass = eval(config['model']['name'])
model = ModelClass(**config['model']['params'])
state_dict = paddle.load(cfg.weight)
if 'model' in state_dict.keys():
    state_dict = state_dict['model']

model.load_dict(state_dict)
print("start calc")
# melspectrogram = LogMelSpectrogram(**config['fbank'])
transforms = []
melspectrogram = LogMelSpectrogram(**config['fbank'])
transforms += [melspectrogram]
if config['normalize']:
    transforms += [Normalize(config['mean_std_file'])]
transforms = Compose(transforms)

similarity = paddle.nn.CosineSimilarity(axis=-1, eps=1e-6)
def get_feature(file, model, transforms, random_sampling=False):
    # global file2feature
    # if file in file2feature:
    #     return file2feature[file]
    # resample file to 16k
    output_filepath = file.replace('8k', '16k')
    # mkdir output_dir
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # resample
    if not os.path.exists(output_filepath):
        cmd = f"ffmpeg -i {file} -ar 16000 {output_filepath}"
        subprocess.call(cmd, shell=True)
    s0, fs =paddleaudio.load(output_filepath)
    
    #s0, _ = paddleaudio.load(file, sr=cfg.sr)
    logger.info(f"encoder: {file}. shape: {s0.shape}")
    s = paddle.to_tensor(s0[None, :])
    s = transforms(s).astype('float32')
    with paddle.no_grad():
        feature = model(s)  #.squeeze()
    feature = feature / paddle.sqrt(paddle.sum(feature**2))
    # file2feature.update({file: feature})
    return feature

def generate_embedding(filepath):
    # cmd = f"ffmpeg -y -i {filepath} -ac 1 -ar 16000 ./test.wav"
    # subprocess.call(cmd, shell=True)
    feature = get_feature(filepath, model, transforms, random_sampling=False)
    # print(type(feature))
    # print(feature.shape)
    # feature = torch.tensor(feature)
    return feature

if __name__ == '__main__':
    melspectrogram = LogMelSpectrogram(**config['fbank'])
    # filepath = "/lyxx/datasets/raw/VoxCeleb2/wav/id09202/10tNrD_DPkw/00001.m4a"
    filepath = "/lyxx/datasets/raw/VoxCeleb2/wav/id09203/aJ5xw25ipXg/00011.m4a"
    filepath2 = "/lyxx/datasets/raw/VoxCeleb2/wav/id09202/10tNrD_DPkw/00002.m4a"
    f1 = generate_embedding(filepath)
    f2 =generate_embedding(filepath2)
    print(similarity(f1,f2))
