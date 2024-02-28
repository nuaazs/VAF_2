# coding = utf-8
# @Time    : 2023-03-17  09:03:19
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 统计目录下所有手机号子文件夹中的文件，是不是属于同一个人.


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

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str, default='/ssd2/cti_aftervad_train_data_vad',help='')
parser.add_argument('--num_th', type=float, default=2,help='')
parser.add_argument('--similarity_th', type=float, default=0.8,help='')
parser.add_argument('--low_th', type=float, default=0.7,help='')

parser.add_argument('--model_path', type=str, default="speechbrain/spkrec-ecapa-voxceleb",help='')
parser.add_argument('--cache_path', type=str, default="cache",help='')
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
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            run_opts={"device": "cuda:0"})


def resample_wav(wav_file):
    os.makedirs(f"{args.cache_path}/files_16k", exist_ok=True)
    # resample by ffmpeg, no output
    filename = os.path.basename(wav_file)
    phone = wav_file.split('/')[-2]
    cmd = f"ffmpeg -i {wav_file} -ar {args.sr} {args.cache_path}/files_16k/{phone}-{filename} -y > /dev/null 2>&1"
    subprocess.call(cmd, shell=True)
    return f"{args.cache_path}/files_16k/{phone}-{filename}"


def embedding(filepath):
    # get all wav files in filepath
    wav_list = sorted(glob.glob(filepath + "/*.wav"))
    embeddings_result = {}
    for wav in wav_list:
        # resample to args.sr
        wav = resample_wav(wav)
        # if f"{args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy" exists, load it
        if os.path.exists(f"{args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy"):
            embeddings = np.load(f"{args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy")
            embeddings_result[wav]=embeddings
            continue
        wav_data, sr = torchaudio.load(wav)
        assert sr == args.sr, f"sr: {sr} != {args.sr}"
        try:
            embeddings = classifier.encode_batch(wav_data).clone().detach().cpu()
        except:
            print(f"Error: {wav}")
            continue
        embeddings_result[wav]=embeddings
        # save_embeddings as npy to cache_path/embeddings_npy
        os.makedirs(f"{args.cache_path}/embeddings_npy", exist_ok=True)
        np.save(f"{args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy", embeddings)
        # print(f"\t# Save {wav} embeddings to {args.cache_path}/embeddings_npy/{os.path.basename(wav).split('.')[0]}.npy")
        # print(f"\t# Embeddings shape: {embeddings.shape}")
    return embeddings_result

def check_similarity(embedding_list,num_th):
    # embedding_list : [[embedding1,wavpath1], [embedding2,wavpath2], ...]
    # cos_th: cos similarity threshold
    # return: True or False
    length = len(embedding_list)
    if length < num_th:
        return False,[]
    mean_score_list = []
    for i in range(length):
        current_embedding = embedding_list[i][0]
        other_embeddings = []
        for j in range(0, i):
            other_embeddings.append(embedding_list[j][0])
        for j in range(i + 1, length):
            other_embeddings.append(embedding_list[j][0])
        score_list = []
        for other_embedding in other_embeddings:
            # convert to tensor
            current_embedding = torch.tensor(current_embedding)#.clone().detach()
            other_embedding = torch.tensor(other_embedding)#.clone().detach()
            cos = similarity(current_embedding, other_embedding)
            score_list.append(cos)
        # get mean, median, max, min
        mean = torch.mean(torch.stack(score_list))
        median = torch.median(torch.stack(score_list))
        max = torch.max(torch.stack(score_list))
        min = torch.min(torch.stack(score_list))
        # mean_score_list.append([mean,wavfile])
        mean_score_list.append([mean,embedding_list[i][1]])
    # calc all mean score
    mean_score_list = sorted(mean_score_list, key=lambda x: x[0], reverse=True)
    # print(f"\t# mean_score_list: {mean_score_list}")
    # print(f"\t# mean_score_list[-1][0]: {mean_score_list[-1][0]}")
    # if mean_score_list[-1][0] < args.low_th 
    all_mean_score = torch.mean(torch.stack([x[0] for x in mean_score_list]))
    # print(f"\t# all_mean_score: {all_mean_score}")
    if (mean_score_list[-1][0]< args.low_th)  or (all_mean_score < args.similarity_th):
        # remove the lowest score wav from embedding_list by wavfile
        #  and check again by recursion
        embedding_list = [x for x in embedding_list if x[1] != mean_score_list[-1][1]]
        return check_similarity(embedding_list,num_th)
    else:
        return True,embedding_list
        

def check_embeddings_result(embeddings_result,phone):
    output_embeddings_result = {}
    # check if embeddings from save speaker are similar enough
    wav_file_paths = sorted(embeddings_result.keys())
    # get embedding_list : [[embedding1,wavpath1], [embedding2,wavpath2], ...]
    embedding_list = []
    for wav_file_path in wav_file_paths:
        embedding_list.append([embeddings_result[wav_file_path],wav_file_path])
    # check similarity
    num_th = max(args.num_th,len(embedding_list) - 1)
    is_valid,embedding_list_output = check_similarity(embedding_list,num_th)
    

    return is_valid,embedding_list_output



if __name__ == "__main__":
    args = parser.parse_args()
    # make cache dir
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)
    
    # get phone list from data_path. Use regular expression to filter the phone number.
    phone_list = [_phone for _phone in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, _phone))]
    # phone
    phone_list = sorted([phone for phone in phone_list if len(phone) == 11])
    total_num, all_pass_num, pass_num = 0, 0, 0
    all_embeddings = {}
    pbar = tqdm(phone_list)
    for phone in pbar:
        phone_path = os.path.join(args.data_path, phone)
        res = embedding(phone_path)
        all_embeddings[phone] = res
        check_result,emb_file_list = check_embeddings_result(res,phone)


        file_num_raw = len(res.keys())
        file_num = len(emb_file_list)
        # logger.info(f"\t# Phone: {phone}")
        # logger.info(f"\t# Raw file number: {file_num_raw}")
        # logger.info(f"\t# Output file number: {file_num}")
        if file_num_raw == file_num:
            all_pass_num += 1
        total_num += 1
        all_pass_rate = all_pass_num / total_num
        # if pass, write file list to pass.txt
        if check_result:
            pass_num += 1
            with open(os.path.join(args.cache_path, "pass.txt"), "a", encoding='utf-8') as f:
                for item in emb_file_list:
                    # print(item[1])
                    
                    phone = item[1].split('/')[-1].split('-')[0]
                    filename = "-".join(item[1].split('/')[-1].split('-')[1:])
                    raw_path = os.path.join(args.data_path,phone,filename)

                    f.write(raw_path+"\n")
        else:
            # if not, write phone_num to fail.txt
            with open(os.path.join(args.cache_path, "fail.txt"), "a", encoding='utf-8') as f:
                f.write(phone+"\n")
        # update pbar
        pbar.set_description(f"APR: {all_pass_rate*100:.2f}% | Pass Rate: {pass_num/total_num*100:.2f}%")
    