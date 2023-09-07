# coding = utf-8
# @Time    : 2023-08-02  09:00:45
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Load pretrained model by name.
DEV=False
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dev', action='store_true', help='dev mode')
    args= parser.parse_args()
    if args.dev:
        DEV=True

import os
import re
import pathlib
import torch
import torchaudio
import torch.nn.functional as F
import wget
if DEV:
    import sys
    sys.path.append('/VAF/train')

from dguard.utils.builder import build
from dguard.utils.config import yaml_config_loader,Config
from dguard.process.backend import random_choose_ten_crops,calculate_cmf,calculate_cosine_distance
import warnings
warnings.filterwarnings("ignore")


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
    'resnet101_lm':{
        "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet101_LM.yaml",
        "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet101_LM/voxceleb_resnet101_LM.pt',
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
    'campp_lm':{
        "config": "/VAF/train/pretrained_models/wespeaker/voxceleb_CAM++_LM/voxceleb_CAM++_LM/config.yaml",
        "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_CAM++_LM/voxceleb_CAM++_LM/avg_model.pt',
        'embedding_size': '256',
        'sample_rate': '16000'
    },

    # 'resnet34':{
    #     "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet34.yaml",
    #     "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet34/voxceleb_resnet34.pt',
    #     'embedding_size': '256',
    #     'sample_rate': '16000'
    # },
    # 'resnet152':{
    #     "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet152.yaml",
    #     "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet152/voxceleb_resnet152.pt',
    #     'embedding_size': '256',
    #     'sample_rate': '16000'
    # },
    # 'resnet221':{
    #     "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet221.yaml",
    #     "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet221/voxceleb_resnet221/voxceleb_resnet221.pt',
    #     'embedding_size': '256',
    #     'sample_rate': '16000'
    # },
    # 'resnet293':{
    #     "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet293.yaml",
    #     "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet293/voxceleb_resnet293/voxceleb_resnet293.pt',
    #     'embedding_size': '256',
    #     'sample_rate': '16000'
    # },
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
        # print(f"\t*-> Load model {model_name} successfully. Embedding size: {model_info[model_name]['embedding_size']}")
        return embedding_model,feature_extractor,sample_rate
    else:
        all_models = list(model_info.keys())
        print("\t*-> All models: ", all_models)
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
        print(f"\t*-> Load model {model_name} successfully. Embedding size: {model_info[model_name]['embedding_size']}")
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

class PretrainedModel:
    def __init__(self,model_name,device='cpu',strict=True,mode="extract"):
        # mode: compare or extract
        self.model_name = model_name
        self.device = device
        self.strict = strict
        self.mode = mode
        self.model, self.feature_extractor, self.sample_rate = load_by_name(model_name,device=device,strict=strict)
        self.model.eval()
        print(f"*-> Load model {model_name} successfully. Embedding size: {model_info[model_name]['embedding_size']}")

    def inference(self,wav_path_list,cmf=True,segment_length=3*16000,crops_num_limit=1,segment_length_limit=2*16000):
        result = []
        for wav_path in wav_path_list:
            wav, fs = torchaudio.load(wav_path)
            print(f"* {wav_path}")
            print(f"\t*-> Raw wav time length: {wav.shape[1]/fs} seconds.")
            assert fs == self.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
            # wav = wav.to(next(model.parameters()).device)
            wav = torch.tensor(wav, dtype=torch.float32)
            feat = self.feature_extractor(wav)
            feat = feat.unsqueeze(0)
            feat = feat.to(next(self.model.parameters()).device)
            if cmf:
                cmf_embedding,crops_num = self.get_cmf(wav,segment_length=segment_length,segment_length_limit=segment_length_limit)
            else:
                cmf_embedding = None
                crops_num = 1
            with torch.no_grad():
                outputs = self.model(feat)
                # outputs = model(x)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                output = embeds
                result.append([output,cmf_embedding,crops_num])
                # print(f"output shape: {output.shape}")
                # print(f"cmf_embedding shape: {cmf_embedding.shape}")
        if self.mode=="compare":
            assert len(result)==2, "Compare model should have two inputs."
            cos_score = self.calculate_cosine_distance(result[0][0],result[1][0])
            factor = self.calculate_factor(result[0][1],result[1][1])
            return cos_score,factor
        else:
            return result

    def get_cmf(self,wav_data,segment_length,segment_length_limit):
        selected_crops,selected_crops_emb = random_choose_ten_crops(wav_data,segment_length,get_embedding_func=lambda x:get_embedding(self.model,self.feature_extractor,x),segment_length_limit=segment_length_limit)
        crops_num = selected_crops.shape[0]
        print(f"\t*-> Get #{selected_crops.shape[0]} crops")
        cmf = calculate_cmf(selected_crops_emb)
        return cmf,crops_num

    def calculate_cosine_distance(self,x, y):
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")
        """计算余弦距离"""
        # x: [batch_size, embedding_size]
        # y: [batch_size, embedding_size]
        # output: [batch_size]
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return torch.sum(x * y, dim=1)

    def calculate_factor(self,cmf1,cmf2):
        # if cmf1 and cmf2 is constant, just multiply them
        if isinstance(cmf1,float) and isinstance(cmf2,float):
            return cmf1*cmf2
        # 
        cmf1 = torch.tensor(cmf1, dtype=torch.float32)
        cmf2 = torch.tensor(cmf2, dtype=torch.float32)
        factor = torch.dot(cmf1.reshape(-1),cmf2.reshape(-1))
        return factor

# useage
if DEV:
    # infer = PretrainedModel('resnet293_lm',mode="compare")
    # cos_score,factor = infer.inference(['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'],cmf=True,segment_length=3*16000)
    # print(f"cos_score: {cos_score}, factor: {factor}")
    # print("="*50)
    infer = PretrainedModel('resnet101_lm',mode="extract")
    segment_length=-5
    result =  infer.inference(['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'],cmf=True,segment_length=segment_length,segment_length_limit=2*16000)
    # result = infer.inference(["/datasets/cjsd_download_test_vad/male_8/s2023_08_02_18_39_10_e2023_08_02_18_40_06.wav","/datasets/cjsd_download_test_vad/male_8/s2023_07_31_18_48_15_e2023_07_31_18_49_06.wav"],cmf=True,segment_length=-1,segment_length_limit=10*16000)
    print(f"result len: {len(result)}")
    print(f"result[0] len: {len(result[0])}")
    print(f"result[0][0] shape: {result[0][0].shape}")
    print(f"result[1][0] shape: {result[1][0].shape}")

    # print(f"result[0][1] shape: {result[0][1].shape}")
    # print(f"result[1][1] shape: {result[1][1].shape}")
    emb0 = result[0][0]
    emb1 = result[1][0]
    num0 = result[0][2]
    num1 = result[1][2]
    cmf0 = result[0][1]
    cmf1 = result[1][1]
    
    max_crop_num = int(abs(segment_length))+(int(abs(segment_length))-1)
    print(max_crop_num)
    print(f"num0: {num0}, factor: {1-((max_crop_num-num0)/max_crop_num)*0.3}")
    print(f"num1: {num1}, factor: {1-((max_crop_num-num1)/max_crop_num)*0.3}")
    print(f"cmf_0: {cmf0}")
    print(f"cmf_1: {cmf1}")
    # cmf_factor = infer.calculate_factor(cmf0,cmf1)
    # print(f"cmf_factor: {cmf_factor}")
    # cos_score = infer.calculate_cosine_distance(emb0,emb1)
    # print(f"cos_score: {cos_score}")
