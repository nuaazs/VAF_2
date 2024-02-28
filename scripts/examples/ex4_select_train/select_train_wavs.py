# coding = utf-8
# @Time    : 2023-03-14  10:26:30
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 预处理长江时代数据，提取单通道，计算VAD后有效音频时常.
# 获取满足要求的说话人数据（音频条数>=2 , 有效音频时长>=180s

import os
import sys
import glob
import numpy as np
import torchaudio
import torch
import argparse
import logging
import re
import random
import subprocess
from tqdm import tqdm

# set seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='/lyxx/lifan/results_cjsd_download/cjsd_download_single_8k', help='长江时代数据集路径')
parser.add_argument('--save_path', type=str, default='/ssd2/cti_aftervad_train_data', help='保存路径')
parser.add_argument('--duration_th', type=int, default=180, help='有效音频时长阈值')
parser.add_argument('--num_th', type=int, default=2, help='有效音频条数阈值')
parser.add_argument('--sr', type=int, default=8000, help='采样率')
parser.add_argument('--channel', type=int, default=1, help='通道')
parser.add_argument('--start', type=int, default=7, help='音频起始位置（秒）')
parser.add_argument('--device', type=str, default='cuda:0', help='显卡ID')
parser.add_argument('--vad_model', type=str, default='vad_8k_en_phone_crdnns', help='VAD模型')
parser.add_argument('--process_id', type=int, default=1, help='进程编号')
parser.add_argument('--process_num', type=int, default=8, help='进程总数')
args = parser.parse_args()

# logger
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
os.makedirs("./log",exist_ok=True)
handler = logging.FileHandler(f"./log/{args.process_id}.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load VAD model from speechbrain
logger.info(f"Load VAD model from speechbrain: ../models/VAD/{args.vad_model}")
from speechbrain.pretrained import VAD
vad_model = VAD.from_hparams(
    source= f"../models/VAD/{args.vad_model}",
    #savedir=f"./pretrained_models/{args.vad_model}_{process_id}",
    run_opts={"device": args.device},
)

# def preprocess_wav(wav_file):
#     """select channel, and remove first 7s

#     Args:
#         wav_file (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     cmd = f"ffmpeg -i {wav_file} -ac {args.channel} -ar {args.sr} -ss {args.start} -y {wav_file}"
#     # subprocess()


# vad fuction
def vad(wav_file,
        th=0.5,
        activation_th=0.5,
        deactivation_th=0.25,
        en_activation_th=0.6,
        en_deactivation_th=0.6,
        predict_speech_save_path=None):
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
    return {
        "before_vad_length": before_vad_length,
        "after_vad_length": after_vad_length,
        "vad": predict_speech.reshape(1, -1),
        "mask": upsampled_boundaries
    }

def check_valid(phone_path,dst_path):
    """判断该手机号目录下文件是否满足要求
        如果满足要求直接复制到目标目录

    Args:
        phone_path (str): 目录地址

    Returns:
        check_result (Bool): 是否满足要求
        total_length （float）：有效音频时长(秒)
        total_number (int): 有效音频条数
    """
    # wav_file = preprocess_wav(wav_file)
    total_length = 0
    for wav_file in os.listdir(phone_path):
        try:
            wav_file = os.path.join(phone_path, wav_file)
            vad_result = vad(wav_file)
            total_length += vad_result["after_vad_length"]
        except Exception as e:
            logger.error(e)
            # rm wav_file
            cmd = f"rm {wav_file}"
            # print(cmd)
            subprocess.call(cmd, shell=True)
            continue

        # print(f"\t{wav_file} Length:{vad_result['after_vad_length']}")
    
    if total_length < args.duration_th:
        # print(total_length)
        logger.info(f"{phone_path}, {total_length}, 0")
        return False
    # cp to dst_path
    cmd = f"cp -r {phone_path} {dst_path}"
    subprocess.call(cmd, shell=True)
    # print(cmd)
    logger.info(f"{phone_path}, {total_length}, 1")
    return True

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
    for phone_path in pbar:
        check_result = check_valid(phone_path, args.save_path)
        if check_result:
            pass_num += 1
        total_num += 1
        pass_rate = pass_num / total_num
        pbar.set_postfix(PassNum=pass_num,PassRadio=pass_rate, refresh=False)   
        # print(f"Phone {phone_path} check result: {check_result}")