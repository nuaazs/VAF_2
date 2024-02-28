# coding = utf-8
# @Time    : 2023-03-17  09:03:19
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 统计目录下所有手机号子文件夹中的文件，不同手机号是否有相似度较高的音频.


import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import glob
import os
import numpy as np
import argparse
import json
import shutil
import argparse
import subprocess
from tqdm import tqdm

# 从<data_path>中每个手机号目录提取<sample_num>个音频。
# 每个手机号所有的编码 和其他手机号所有的编码 两两比对
# 如果有<num_th>组音频的相似度大于<similarity_th>，则认为可能是同一个人
# 将可能是同一个人的手机号写入<same_phone.txt>

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='/ssd2/cti_fail_data',help='')
parser.add_argument('--sample_num', type=float, default=3,help='')
parser.add_argument('--similarity_th', type=float, default=0.8,help='')
parser.add_argument('--num_th', type=int, default=2,help='')

parser.add_argument('--model_path', type=str, default="../../models/ECAPATDNN",help='')
parser.add_argument('--cache_path', type=str, default="../../cache",help='')
parser.add_argument('--sr', type=int, default=16000,help='')
args = parser.parse_args()

# log
import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
os.makedirs("./log",exist_ok=True)
handler = logging.FileHandler(f"./log/check.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
classifier = EncoderClassifier.from_hparams(source=args.model_path,
                                            run_opts={"device": "cuda:0"})


def resample_wav(wav_file):
    os.makedirs(f"{args.cache_path}/files_16k", exist_ok=True)
    # resample by ffmpeg, no output
    output_file = get_resample_wav_path(wav_file)
    cmd = f"ffmpeg -i {wav_file} -ar {args.sr} {output_file} -y > /dev/null 2>&1"
    subprocess.call(cmd, shell=True)
    return output_file

def get_resample_wav_path(wav_file):
    filename = os.path.basename(wav_file)
    phone = wav_file.split('/')[-2]
    return f"{args.cache_path}/files_16k/{phone}-{filename}"

def embedding(filepath):
    # get all wav files in filepath
    wav_list = sorted(glob.glob(filepath + "/*.wav"))
    embeddings_result = {}
    for wav in wav_list:
        # resample to args.sr
        # if f"{args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy" exists, load it
        wav_file_name = get_resample_wav_path(wav)
        if os.path.exists(f"{args.cache_path}/embeddings_npy/{os.path.basename(wav_file_name).split('.')[0]}.npy"):
            embeddings = np.load(f"{args.cache_path}/embeddings_npy/{os.path.basename(wav_file_name).split('.')[0]}.npy")
            embeddings_result[wav_file_name]=embeddings
            continue
        wav = resample_wav(wav)

        wav_data, sr = torchaudio.load(wav)
        assert sr == args.sr, f"sr: {sr} != {args.sr}"
        try:
            embeddings = classifier.encode_batch(wav_data).clone().detach().cpu()
        except:
            print(f"Error: {wav}")
            os.remove(wav)
            continue
        embeddings_result[wav]=embeddings
        # save_embeddings as npy to cache_path/embeddings_npy
        os.makedirs(f"{args.cache_path}/embeddings_npy", exist_ok=True)
        np.save(f"{args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy", embeddings)
        # print(f"\t# Save {wav} embeddings to {args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy")
        # print(f"\t# Embeddings shape: {embeddings.shape}")

        # remove resampled wav
        os.remove(wav)
    return embeddings_result

def check_similarity(embedding_list_1,embedding_list_2,num_th):
    # 两两比对, 如果有<num_th>组音频的相似度大于<similarity_th>，则认为可能是同一个人
    # embedding_list : [[embedding1,wavpath1], [embedding2,wavpath2], ...]
    # cos_th: cos similarity threshold
    # return: True or False
    length_1 = len(embedding_list_1)
    length_2 = len(embedding_list_2)
    #可疑音频对
    pair_dict = {}
    for i in range(length_1):
        for j in range(length_2):
            emb_pair = (embedding_list_1[i][0], embedding_list_2[j][0])
            pair_key = f"{embedding_list_1[i][1]}-{embedding_list_2[j][1]}"
            if pair_key in pair_dict:
                continue
            e1 = torch.from_numpy(emb_pair[0])
            e2 = torch.from_numpy(emb_pair[1])
            pair_score = similarity(e1, e2)
            score_float = pair_score.item()

            if score_float > args.similarity_th:
                num_th -= 1
                pair_dict[pair_key] = score_float
                if num_th <= 0:
                    return False,pair_dict
    return True,pair_dict
        

def get_embedding_list(embeddings_result,phone):
    output_embeddings_result = {}
    # check if embeddings from save speaker are similar enough
    wav_file_paths = sorted(embeddings_result.keys())
    # get embedding_list : [[embedding1,wavpath1], [embedding2,wavpath2], ...]
    embedding_list = []
    for wav_file_path in wav_file_paths:
        embedding_list.append([embeddings_result[wav_file_path],wav_file_path])
    return embedding_list




if __name__ == "__main__":
    args = parser.parse_args()
    # make cache dir
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)
    
    # get phone list from data_path. Use regular expression to filter the phone number.
    phone_list = [_phone for _phone in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, _phone))]
    # phone
    phone_list = sorted([phone for phone in phone_list])
    total_num, all_pass_num, pass_num = 0, 0, 0
    all_embeddings = {}
    pbar = tqdm(phone_list)
    for phone in pbar:
        phone_path = os.path.join(args.data_path, phone)
        res = embedding(phone_path)
        all_embeddings[phone] = get_embedding_list(res,phone)
    assert len(all_embeddings.keys()) == len(phone_list), f"len(all_embeddings.keys()): {len(all_embeddings.keys())} != len(phone_list): {len(phone_list)}"
    print(f"Get #{len(all_embeddings.keys())} speakers embeddings.")
    print(f"Now check similarity between speakers.")
    # check similarity between speakers
    pbar = tqdm(range(len(phone_list)))
    for i in pbar:
        for j in range(i+1,len(phone_list)):
            total_num += 1
            phone_1 = phone_list[i]
            phone_2 = phone_list[j]
            # check if embeddings from save speaker are similar enough
            embeddings_1 = all_embeddings[phone_1]
            embeddings_2 = all_embeddings[phone_2]
            pass_flag,pair_dict = check_similarity(embeddings_1,embeddings_2,args.num_th)
            if not pass_flag:
                print(f"Error: {phone_1} and {phone_2} are same person.")
                # print(f"Pair dict: {pair_dict}")
                # pass
            else:
                pass_num += 1
            # update pbar
            pbar.set_description(f"Pass Rate: {pass_num/total_num*100:.2f}%")
    