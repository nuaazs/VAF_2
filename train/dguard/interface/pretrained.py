# coding = utf-8
# @Time    : 2023-08-02  09:00:45
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Load pretrained model by name.

import os
import re
import pathlib
import torch
import torchaudio
import wget
# import sys
# sys.path.append('/VAF/train')
from dguard.utils.builder import build
from dguard.utils.config import yaml_config_loader,Config

#TODO: upload to remote server
model_info ={
    'dfresnet_233':{
        "config": "/VAF/train/egs/voxceleb/sv-dfresnet/conf/dfresnet233.yaml",
        "ckpt": '/VAF/train/egs/voxceleb/sv-dfresnet/exp/dfresnet233/models/CKPT-EPOCH-76-00/embedding_model.ckpt',
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'eres2net':{
        "config": "/VAF/train/egs/voxceleb/sv-eres2net/conf/eres2net.yaml",
        "ckpt": '/VAF/train/egs/voxceleb/sv-eres2net/exp/eres2net/models/eres2net_voxceleb.ckpt',
        'embedding_size': '192',
        'sample_rate': '16000'
    },
    'campp':{
        "config": "/VAF/train/egs/voxceleb/sv-cam++/conf/cam++.yaml",
        "ckpt": '/VAF/train/egs/voxceleb/sv-cam++/exp/campp/models/campp_voxceleb.bin',
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'ecapatdnn_1024':{
        "config": "/VAF/train/egs/voxceleb/sv-ecapatdnn/conf/ecapatdnn1024.yaml",
        "ckpt": '/VAF/train/egs/voxceleb/sv-ecapatdnn/exp/models/ecapatdnn_voxceleb.bin',
        'embedding_size': '192',
        'sample_rate': '16000'
    },
    'mfa_conformer':{
        "config": "/VAF/train/egs/voxceleb/sv-conformer/conf/conformer.yaml",
        "ckpt": '/VAF/train/egs/voxceleb/sv-conformer/exp/conformer/models/CKPT-EPOCH-157-00/embedding_model.ckpt',
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'repvgg':{
        "config": "/VAF/train/egs/voxceleb/sv-repvgg/conf/repvgg.yaml",
        "ckpt": '/VAF/train/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-142-00/embedding_model.ckpt',
        'embedding_size': '512',
        'sample_rate': '16000'
    },

}

ALL_MODELS = list(model_info.keys())

def download_or_load(url):
    if url.startswith('http'):
        if os.path.exists(f"/tmp/dguard/{os.path.basename(url)}"):
            print(f"Find tmp file {url} in /tmp/dguard/{os.path.basename(url)}")
            ckpt_path = f"/tmp/dguard/{os.path.basename(url)}"
            return ckpt_path
        # wget to /tmp/dguard
        os.makedirs('/tmp/dguard', exist_ok=True)
        ckpt = wget.download(url, out='/tmp/dguard')
        ckpt_path = f"/tmp/dguard/{os.path.basename(url)}"
    else:
        ckpt_path = url
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt {ckpt_path} not found.")
    return ckpt_path

def load_by_name(model_name,device='cuda:0'):
    if model_name in model_info:
        ckpt = download_or_load(model_info[model_name]['ckpt'])
        config = yaml_config_loader(download_or_load(model_info[model_name]['config']))
        config = Config(config)
        embedding_model = build('embedding_model', config)
        embedding_model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        embedding_model.eval()
        feature_extractor = build('feature_extractor', config)
        sample_rate = int(model_info[model_name]['sample_rate'])
        embedding_model.to(device)
        # feature_extractor.to(device)
        print(f"Load model {model_name} successfully. Embedding size: {model_info[model_name]['embedding_size']}")
        return embedding_model,feature_extractor,sample_rate
    else:
        all_models = list(model_info.keys())
        print("All models: ", all_models)
        raise NotImplementedError(f"Model {model_name} not implemented.")

# 推理
def inference(model,feature_extractor,wav_path,sample_rate=16000):
    model.eval()
    wav, fs = torchaudio.load(wav_path)
    assert fs == sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
    # wav = wav.to(next(model.parameters()).device)
    wav = torch.tensor(wav, dtype=torch.float32)
    feat = feature_extractor(wav)
    feat = feat.unsqueeze(0)
    feat = feat.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(feat)
    return output

# useage
# from dguard.interface.pretrained import load_by_name,ALL_MODELS
# print(ALL_MODELS)
# model,feature_extractor,sample_rate = load_by_name('dfresnet_233')
