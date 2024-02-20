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
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


#TODO: upload to remote server
model_info ={
    'dfresnet_233':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'eres2net':{
        'embedding_size': '192',
        'sample_rate': '16000'
    },
    'campp':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'ecapatdnn_1024':{
        'embedding_size': '192',
        'sample_rate': '16000'
    },
    'mfa_conformer':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'repvgg':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'resnet34_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet152_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet101_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet101_cjsd':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet221_cjsd_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet101_cjsd8000':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet101_cjsd8000_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet221_cjsd8000_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet221_cjsd8000':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet221_cjsd8000_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet221_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet293_cjsd':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet293_cjsd_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet293_lm':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'campp_lm':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'campp_cjsd':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'campp_cjsd_lm':{
        'embedding_size': '512',
        'sample_rate': '16000'
    },
    'campp_200k':{
        'embedding_size': '192',
        'sample_rate': '16000'
    },
    'eres2net_200k':{
        'embedding_size': '192',
        'sample_rate': '16000'
    },
    'resnet101_cjsd_2024':{
        'embedding_size': '256',
        'sample_rate': '16000'
    },
    'resnet101_cjsd_lm_2024':{
        'embedding_size': '256',
        'sample_rate': '16000'
    }
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
        if "lm" in model_name or "cjsd" in model_name:
            strict=False
        root_path=model_info.get("root_path","dguard/files")
        pt_url=model_info.get("pt_url",None)
        yaml_url=model_info.get("yaml_url",None)
        if pt_url:
            ckpt = download_or_load(pt_url)
        else:
            ckpt = os.path.join(root_path,"pt",model_name+".pt")
        if yaml_url:
            config = yaml_config_loader(yaml_url)
        else:
            config = os.path.join(root_path,"yaml",model_name+".yaml")
        config = config.replace("_lm.yaml",".yaml")
        config = Config(yaml_config_loader(config))
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
    def __init__(self,model_name,device='cpu',strict=False,mode="extract"):
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
            print(f"* {wav_path}")
            wav, fs = torchaudio.load(wav_path)
            
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
                if segment_length<0:
                    max_crop_num = int(abs(segment_length))+(int(abs(segment_length))-1)
                    # alpha = 1 if crops_num>=max_crop_num else 1-((max_crop_num-crops_num)/max_crop_num)#*0.3
                    alpha = crops_num/abs(segment_length)
                else:
                    alpha = 1
                result.append([output,cmf_embedding,crops_num,alpha])
                # print(f"output shape: {output.shape}")
                # print(f"cmf_embedding shape: {cmf_embedding.shape}")
        #TODO 这里逻辑要改，不然alpha相等。
        if self.mode=="compare":
            assert len(result)==2, "Compare model should have two inputs."
            cos_score = self.calculate_cosine_distance(result[0][0],result[1][0])
            if segment_length<0:
                max_crop_num = int(abs(segment_length))+(int(abs(segment_length))-1)
                # alpha = 1 if crops_num>=max_crop_num else 1-((max_crop_num-crops_num)/max_crop_num)#*0.3

                alpha = crops_num/abs(segment_length)
            else:
                alpha = 1
            if cmf == True:
                factor = self.calculate_factor(result[0][1],result[1][1])
            else:
                factor = None
            print(f"Factor: {factor}, Alpha: {alpha}")
            return cos_score,factor,alpha

        else:
            return result

    def get_cmf(self,wav_data,segment_length,segment_length_limit):
        selected_crops,selected_crops_emb = random_choose_ten_crops(wav_data,segment_length,get_embedding_func=lambda x:get_embedding(self.model,self.feature_extractor,x),segment_length_limit=segment_length_limit)
        crops_num = selected_crops.shape[0]
        # print(f"\t*-> Get #{selected_crops.shape[0]} crops")
        cmf = calculate_cmf(selected_crops_emb)
        return cmf,crops_num

    def calculate_cosine_distance(self,x, y):
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        cos_sim = F.cosine_similarity(x, y, dim=-1)
        # print(cos_sim)
        return cos_sim
        # # print(f"x shape: {x.shape}")
        # # print(f"y shape: {y.shape}")
        # """计算余弦距离"""
        # # x: [batch_size, embedding_size]
        # # y: [batch_size, embedding_size]
        # # output: [batch_size]
        # x = F.normalize(x, p=2, dim=1)
        # y = F.normalize(y, p=2, dim=1)
        # return torch.sum(x * y, dim=1)

    def calculate_factor(self,cmf1,cmf2):
        # if cmf1 and cmf2 is constant, just multiply them
        if isinstance(cmf1,float) and isinstance(cmf2,float):
            print("Get factor by multiply")
            return cmf1*cmf2
        # # 
        # print("Get factor by torch.dot")
        # cmf1 = torch.tensor(cmf1, dtype=torch.float32)
        # cmf2 = torch.tensor(cmf2, dtype=torch.float32)
        # factor = torch.dot(cmf1.reshape(-1),cmf2.reshape(-1))
        # return factor

        print("Get factor by cosine similarity")
        factor = self.calculate_cosine_distance(cmf1,cmf2)
        return factor

# useage
if DEV:
    for model_name in ["resnet101_cjsd8000","resnet101_cjsd8000_lm","resnet221_cjsd8000","resnet221_cjsd8000_lm","eres2net_200k"]: # ALL_MODELS: # ["campp","campp_cjsd","campp_200k"]:
        print(f"Model name: {model_name}")
        infer = PretrainedModel(model_name,mode="extract")
        result = infer.inference(['/VAF/model_test/data/test/cjsd300/13002931667/20230112161623/20230112161623_1.wav','/VAF/model_test/data/test/cjsd300/13002931667/20230625163725/20230625163725_1.wav'],cmf=False,segment_length=3*16000)
        print(f"result len: {len(result)}")
        print(f"result[0] len: {len(result[0])}")
        print(f"result[0][0] shape: {result[0][0].shape}")
        print(f"result[1][0] shape: {result[1][0].shape}")
        emb0 = result[0][0]
        emb1 = result[1][0]
        print(f"="*50)
        cos_score = infer.calculate_cosine_distance(emb0,emb1)
        print(f"cos_score: {cos_score}")
        print(f"="*50)