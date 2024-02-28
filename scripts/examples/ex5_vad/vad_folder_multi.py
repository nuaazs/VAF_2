# coding = utf-8
# @Time    : 2023-03-14  10:26:30
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: VAD *.<ext> files in a folder, and save the results to dst folder with save folder structure.

import os
import numpy as np
import torchaudio
import torch
import argparse
import logging
import re
import random
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
# set seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='/datasets_hdd/testdata_cnceleb_16k/', help='长江时代数据集路径')
parser.add_argument('--save_path', type=str, default='/datasets_hdd/testdata_cnceleb_16k_vad/', help='保存路径')
parser.add_argument('--sr', type=int, default=16000, help='采样率')
parser.add_argument('--device', type=str, default='cuda:0', help='显卡ID')
parser.add_argument('--vad_model', type=str, default='vad_16k_en_phone_crdnns', help='VAD模型')
parser.add_argument('--process_id', type=int, default=1, help='进程编号')
parser.add_argument('--process_num', type=int, default=8, help='进程总数')
parser.add_argument('--ext', type=str, default='wav', help='音频文件后缀')
parser.add_argument('--model_save_path', type=str, default='/ssd2/online/models/VAD', help='模型保存路径')
args = parser.parse_args()

# logger
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
os.makedirs("./log",exist_ok=True)
handler = logging.FileHandler(f"./log/vad_{args.process_id}.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load VAD model from speechbrain
logger.info(f"Load VAD model from speechbrain: {args.model_save_path}/{args.vad_model}")
from speechbrain.pretrained import VAD
vad_model = VAD.from_hparams(
    source= f"{args.model_save_path}/{args.vad_model}",
    savedir=f"{args.model_save_path}/{args.vad_model}_id{args.process_id}",
    run_opts={"device": args.device},
)



# vad fuction
def vad(wav_file,
        th=0.5,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.6,
        en_deactivation_th=0.6,
        predict_speech_save_path=None):
    # print(f"Now processing {wav_file} and save to {predict_speech_save_path}")
    # make sure the save path exists
    os.makedirs(os.path.dirname(predict_speech_save_path), exist_ok=True)
    wav,sr=torchaudio.load(wav_file)
    assert sr == args.sr, f"{wav_file}: sr is not {args.sr}"
    datalength = len(wav[0])
    before_vad_length =  datalength/ sr
    boundaries = vad_model.get_speech_segments(
        audio_file=wav_file,
        large_chunk_size=30,
        small_chunk_size=10,
        overlap_small_chunk=True,
        apply_energy_VAD=True,
        double_check=True,
        close_th=0.1,
        len_th=0.1,
        activation_th=activation_th,
        deactivation_th=deactivation_th,
        en_activation_th=en_activation_th,
        en_deactivation_th=en_deactivation_th,
        speech_th=th,
    )
    upsampled_boundaries = vad_model.upsample_boundaries(boundaries, wav_file)
    predict_speech = wav[upsampled_boundaries > 0.5]
    predict_noise =  wav[upsampled_boundaries < 0.5]
    after_vad_length = len(predict_speech) / sr
    vad_data = predict_speech.reshape(1, -1)
    # save vad_data to wav
    torchaudio.save(predict_speech_save_path, vad_data, sr)
    
    
    return {
        "before_vad_length": before_vad_length,
        "after_vad_length": after_vad_length,
        "vad": vad_data,
        "mask": upsampled_boundaries
    }

def get_info(phone_path,dst_path):
    total_length = 0
    for wav_file in os.listdir(phone_path):
        try:
            wav_file = os.path.join(phone_path, wav_file)
            # save file path = wav_file.replace(args.data_path, args.save_path)
            save_file_path = wav_file.replace(args.data_path, args.save_path)
            vad_result = vad(wav_file,predict_speech_save_path=save_file_path)
            total_length += vad_result["after_vad_length"]
        except Exception as e:
            logger.error(e)
            continue
    return total_length

if __name__ == "__main__":
    # get all phone folders, with re phone number
    phone_folders = sorted([os.path.join(args.data_path, _phone) for _phone in os.listdir(args.data_path) if re.match(r'^1\d{10}$', _phone)])
    length = len(phone_folders)//args.process_num
    phone_folders = phone_folders[(args.process_id-1)*length:(args.process_id)*length]
    logger.info(f"Total Phone folders #{len(phone_folders)}")
    # print(phone_folders)
    pass_num,total_num = 0,0
    pbar = tqdm(phone_folders)
    pbar.set_description(f"WorkID {args.process_id}")
    info_list = []
    for phone_path in pbar:
        total_length = get_info(phone_path, args.save_path)
        info_list.append([phone_path,total_length])
    # save info_list to npy
    info_npy = np.array(info_list)
    np.save(f"./log/vad_{args.process_id}.npy", info_npy)