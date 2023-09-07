# coding = utf-8
# @Time    : 2023-08-02  09:00:45
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Load pretrained model by name.
DEV=True
import os
import re
import pathlib
import torch
import torchaudio
import wget
if DEV:
    import sys
    sys.path.append('/VAF/train')

from dguard.utils.builder import build
from dguard.utils.config import yaml_config_loader,Config
from dguard.process.backend import random_choose_ten_crops,calculate_cmf,calculate_cosine_distance
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
    'resnet34_lm':{
        "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet34_LM.yaml",
        "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet34_LM/voxceleb_resnet34_LM.pt',
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet152_lm':{
        "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet152_LM.yaml",
        "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet152_LM/voxceleb_resnet152_LM.pt',
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet221_lm':{
        "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet221_LM.yaml",
        "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet221_LM/voxceleb_resnet221_LM/voxceleb_resnet221_LM.pt',
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet293_lm':{
        "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet293_LM.yaml",
        "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet293_LM/voxceleb_resnet293_LM/voxceleb_resnet293_LM.pt',
        'embedding_size': '256',
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

def load_by_name(model_name,device='cuda:0',strict=True):
    if model_name in model_info:
        if "_lm" in model_name:
            strict=False
        ckpt = download_or_load(model_info[model_name]['ckpt'])
        config = yaml_config_loader(download_or_load(model_info[model_name]['config']))
        config = Config(config)
        embedding_model = build('embedding_model', config)
        embedding_model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=strict)
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
        outputs = model(feat)
        # outputs = model(x)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        output = embeds.detach().cpu().numpy()
    return output

# 推理

def get_embedding(model,feature_extractor,wav_data):
    model.eval()
    wav = torch.tensor(wav_data, dtype=torch.float32)
    feat = feature_extractor(wav)
    feat = feat.unsqueeze(0)
    feat = feat.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(feat)
        # outputs = model(x)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        output = embeds
    return output

def get_cmf(model,feature_extractor,wav_data,segment_length):
    selected_crops,selected_crops_emb = random_choose_ten_crops(wav_data,segment_length,get_embedding_func=lambda x:get_embedding(model,feature_extractor,x))
    cmf = calculate_cmf(selected_crops_emb)
    return cmf

def inference_list(model_name,wav_path_list,device='cpu',segment_length=3*16000,cmf=True,redundancy=1):
    
    model,feature_extractor,sample_rate = load_by_name(model_name,device=device)
    model.eval()
    if redundancy>1:
        print(f"Load model {model_name} successfully. Embedding size: {model_info[model_name]['embedding_size']}")
    result = []
    for wav_path in wav_path_list:
        wav, fs = torchaudio.load(wav_path) # wav shape: [1, T]
        assert fs == sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
        # wav = wav.to(next(model.parameters()).device)
        wav = torch.tensor(wav, dtype=torch.float32)
        feat = feature_extractor(wav)
        feat = feat.unsqueeze(0)
        feat = feat.to(next(model.parameters()).device)
        if cmf:
            cmf_embedding = get_cmf(model,feature_extractor,wav,segment_length=segment_length)
        else:
            cmf_embedding = None
        with torch.no_grad():
            outputs = model(feat)
            # outputs = model(x)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            output = embeds
            result.append([output,cmf_embedding])
    return result

# useage
if DEV:
    from dguard.interface.pretrained import load_by_name,ALL_MODELS
    print(ALL_MODELS)
    result = inference_list('resnet293_lm',['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'])
    print(result)
    print(len(result))
    print(result[0][0].shape)
    print(result[0][1].shape)
    cos_score = calculate_cosine_distance(result[0][0],result[1][0])
    print(f"cos score: {cos_score}")
    print(f"cmf shape: {result[0][1].shape}")
    factor = torch.dot(result[0][1].reshape(-1),result[1][1].reshape(-1))
    print(f"factor: {factor}")