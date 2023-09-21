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
    # 'resnet101':{
    #     "config": "/VAF/train/egs/voxceleb/sv-resnet/conf/resnet101.yaml",
    #     "ckpt": '/VAF/train/pretrained_models/wespeaker/voxceleb_resnet101/voxceleb_resnet101.pt',
    #     'embedding_size': '256',
    #     'sample_rate': '16000'
    # },
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
            factor = self.calculate_factor(result[0][1],result[1][1])
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
    # infer = PretrainedModel('resnet293_lm',mode="compare")
    # cos_score,factor,alpha = infer.inference(['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'],cmf=True,segment_length=3*16000)
    # print(f"cos_score: {cos_score}, factor: {factor}")
    # print("="*50)
    infer = PretrainedModel('resnet152_lm',mode="extract")
    segment_length=-5
    # data_two=['id10284/RNYNkXzY5Hk/00008.wav','id10284/YN4cTBWM-QE/00005.wav']
    data_twos=[["/VAF/train/dguard/interface/lol_to_xieyukai_2_result_2610_16k.wav","/VAF/train/dguard/interface/20230907_172246-24.64-40.715_16k.wav"]] # "/VAF/train/dguard/interface/zhaosheng_to_xieyukai_2_result_2610_16k.wav"]]
    


    # data_twos = [['id10278/Pp-rAswo4Xg/00011.wav','id10300/pVKJjm4sKdI/00002.wav'],
    #     ['id10270/x6uYqmx31kE/00005.wav','id10303/R5JGtwc4o1M/00005.wav'],
    #     ['id10305/nJbBcMdxQU4/00010.wav','id10276/3vWez3baO60/00003.wav'],
    #     ['id10278/Pp-rAswo4Xg/00009.wav','id10300/AytdA2W0Y_M/00008.wav'],
    #     ['id10280/d7S0qeo6EP0/00006.wav','id10282/ek_uqADnhPQ/00001.wav'],
    #     ['id10306/C3AhSlYJd3w/00002.wav','id10289/8l5ZnDf-FUA/00018.wav'],
    #     ['id10283/gAKHqHbNt2g/00006.wav','id10309/tGEWD2GaiDw/00006.wav'],
    #     ['id10278/Pp-rAswo4Xg/00005.wav','id10300/fAe7DXqrZls/00020.wav'],
    #     ['id10292/KgY2xhg4Tqk/00005.wav','id10273/8cfyJEV7hP8/00016.wav'],
    #     ['id10305/nJbBcMdxQU4/00004.wav','id10276/X1Yr4eI2oRw/00001.wav'],
    #     ['id10283/r9-0pljhZqs/00012.wav','id10309/rxnN8thYzEQ/00015.wav'],
    #     ['id10278/Pp-rAswo4Xg/00028.wav','id10300/8EDKH42GZ1o/00031.wav'],
    #     ['id10300/xiFC4HwUcww/00004.wav','id10278/Pp-rAswo4Xg/00008.wav'],
    #     ['id10276/CooJzMgoxzw/00001.wav','id10305/3QrLepYlH6o/00009.wav'],
    #     ['id10303/VCEs8oAQAiM/00009.wav','id10270/x6uYqmx31kE/00005.wav'],
    #     ['id10282/zGjY8J48FoE/00002.wav','id10280/d7S0qeo6EP0/00004.wav'],
    #     ['id10282/IDA_ElNHLn4/00002.wav','id10280/9AtE4C309P8/00008.wav'],
    #     ['id10305/gbTZ7k9e_Z0/00010.wav','id10276/3vWez3baO60/00003.wav'],
    #     ['id10283/clznvDi-ybg/00015.wav','id10302/K2_D_tFdAgY/00035.wav'],
    #     ['id10283/vaK4t1-WD4M/00017.wav','id10309/rxnN8thYzEQ/00008.wav'],
    #     ['id10276/X1Yr4eI2oRw/00003.wav','id10283/r9-0pljhZqs/00001.wav'],
    #     ['id10300/1ZyvrJaiLQk/00029.wav','id10294/3BRezdwX-4I/00001.wav'],
    #     ['id10306/kQOkcEZUpJ4/00002.wav','id10289/8l5ZnDf-FUA/00016.wav'],
    #     ['id10300/Fi8lnFPYgII/00022.wav','id10278/x5dqrDlwxR4/00001.wav'],
    #     ['id10306/SBO7kM1IPaY/00001.wav','id10289/sf4uMnkYFG8/00011.wav'],
    #     ['id10305/nJbBcMdxQU4/00006.wav','id10276/S0j29b3KVWQ/00001.wav'],
    #     ['id10276/YloBWJxXzMI/00008.wav','id10283/arklnCzCq48/00001.wav'],
    #     ['id10298/6qFnVechX9o/00024.wav','id10300/Fi8lnFPYgII/00019.wav'],
    #     ['id10289/3g9CjhcNEWk/00011.wav','id10306/2SaEbN8hYz4/00002.wav'],
    #     ['id10300/Fi8lnFPYgII/00012.wav','id10284/RNYNkXzY5Hk/00013.wav'],
    #     ['id10276/S0j29b3KVWQ/00004.wav','id10305/Ih2s_PikIdI/00002.wav'],
    #     ['id10302/K2_D_tFdAgY/00041.wav','id10275/Mdk1SXywHck/00006.wav'],
    #     ['id10276/HrA2BLcLApA/00014.wav','id10305/Ih2s_PikIdI/00002.wav'],
    #     ['id10278/_a9CIdlTOr8/00004.wav','id10300/SQzWyPhRqmk/00001.wav'],
    #     ['id10270/8jEAjG6SegY/00029.wav','id10303/Zs8VK91yVMI/00013.wav'],
    #     ['id10270/8jEAjG6SegY/00029.wav','id10303/Zs8VK91yVMI/00023.wav'],
    #     ['id10282/zGjY8J48FoE/00002.wav','id10280/9AtE4C309P8/00013.wav'],
    #     ['id10283/vaK4t1-WD4M/00017.wav','id10309/rqaAm4bEsXc/00002.wav'],
    #     ['id10278/x5dqrDlwxR4/00007.wav','id10300/1ZyvrJaiLQk/00011.wav'],
    #     ['id10303/R5JGtwc4o1M/00002.wav','id10270/8jEAjG6SegY/00020.wav'],
    #     ['id10298/9o3HnyKpHLM/00001.wav','id10300/1ZyvrJaiLQk/00035.wav'],
    #     ['id10276/YloBWJxXzMI/00008.wav','id10305/ZLzkvnq0JxI/00004.wav'],
    #     ['id10309/rqaAm4bEsXc/00001.wav','id10283/h87Y8nir1o0/00003.wav'],
    #     ['id10283/GcHWzqveqyc/00004.wav','id10276/YloBWJxXzMI/00013.wav'],
    #     ['id10300/Fi8lnFPYgII/00018.wav','id10278/s_rtHBpzrQc/00008.wav'],
    #     ['id10276/HrA2BLcLApA/00005.wav','id10305/cpXXdMJuCdw/00004.wav'],
    #     ['id10300/Fi8lnFPYgII/00007.wav','id10284/EoCPhxtWUOc/00004.wav'],
    #     ['id10275/QjrBKqx_Xeo/00005.wav','id10283/clznvDi-ybg/00001.wav'],
    #     ['id10283/N69Hp2DGMLk/00001.wav','id10309/qFrRfhWombs/00005.wav']]



    # data_twos = [['id10288/A3ZvNuG8_oM/00004.wav','id10288/A3ZvNuG8_oM/00013.wav'],
    #     ['id10297/utAY0zpsv1U/00004.wav','id10297/6-VE8e8RtZE/00002.wav'],
    #     ['id10282/U3xR3MZjEVg/00011.wav','id10282/qkZNuvX1UNo/00010.wav'],
    #     ['id10307/0nH78dDh0N0/00012.wav','id10307/0nH78dDh0N0/00002.wav'],
    #     ['id10282/qkZNuvX1UNo/00007.wav','id10282/U3xR3MZjEVg/00008.wav'],
    #     ['id10280/PQBBKBYGWgU/00001.wav','id10280/YcIZDAhexy8/00001.wav'],
    #     ['id10296/_BVSKK5mGnY/00007.wav','id10296/f_k09R8r_cA/00005.wav'],
    #     ['id10307/f8Ms66atECE/00002.wav','id10307/f8Ms66atECE/00007.wav'],
    #     ['id10299/SX-117N_MoI/00004.wav','id10299/SX-117N_MoI/00002.wav'],
    #     ['id10307/f8Ms66atECE/00010.wav','id10307/f8Ms66atECE/00007.wav']]

    for data_two in data_twos:
        # data_two=[os.path.join("/VAF/train/data/raw_data/voxceleb1/test/wav",_data) for _data in data_two]
        result = infer.inference(data_two,cmf=True,segment_length=segment_length,segment_length_limit=2*16000)
        # result =  infer.inference(['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'],cmf=True,segment_length=segment_length,segment_length_limit=2*16000)
        # result = infer.inference(["/datasets/cjsd_download_test_vad/male_8/s2023_08_02_18_39_10_e2023_08_02_18_40_06.wav","/datasets/cjsd_download_test_vad/male_8/s2023_07_31_18_48_15_e2023_07_31_18_49_06.wav"],cmf=True,segment_length=-1,segment_length_limit=10*16000)
        # print(f"result len: {len(result)}")
        # print(f"result[0] len: {len(result[0])}")
        # print(f"result[0][0] shape: {result[0][0].shape}")
        # print(f"result[1][0] shape: {result[1][0].shape}")

        # print(f"result[0][1] shape: {result[0][1].shape}")
        # print(f"result[1][1] shape: {result[1][1].shape}")
        emb0 = result[0][0]
        emb1 = result[1][0]
        num0 = result[0][2]
        num1 = result[1][2]
        cmf0 = result[0][1]
        cmf1 = result[1][1]
        alpha0 = result[0][3]
        alpha1 = result[1][3]
        
        
        cmf_factor = infer.calculate_factor(cmf0,cmf1)
        print(f"="*50)
        print(f"cmf_factor: {cmf_factor}")
        cos_score = infer.calculate_cosine_distance(emb0,emb1)
        print(f"cos_score: {cos_score}")
        print(f"alpha: {alpha0} {alpha1}")
        alpha = alpha0*alpha1
        # if alpha >=2:
        #     alpha = 2
        final_score = cos_score+alpha*(cmf_factor-cos_score)
        print(f"final_score: {final_score}")
        print(f"="*50)
