# coding = utf-8
# @Time    : 2023-05-14  22:17:20
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 利用后端API接口，传入VAD后数据文件，获取对应文件的embedding特征，保存为numpy文件.
import requests
from tqdm import tqdm
import numpy as np
import os
import argparse
from speechbrain.pretrained import SpeakerRecognition
import torch
import torchaudio.transforms as T
import torchaudio
from CAMPP import get_embedding

parser = argparse.ArgumentParser(description='')
parser.add_argument('--fold_path', type=str, default='/datasets_hdd/testdata_cnceleb_wav_16k/male/register',help='After vad data path')
parser.add_argument('--dst_path', type=str, default="/datasets_hdd/testdata_cnceleb_embedding_16k/male/register",help='Path for output embedding npy files')
parser.add_argument('--worker_index', type=int, default=1,help='')
parser.add_argument('--total_workers', type=int, default=1,help='')
parser.add_argument('--emb_dim', type=int, default=192,help='')
parser.add_argument('--device', type=str, default="cuda:0",help='')
args = parser.parse_args()

emb = SpeakerRecognition.from_hparams(
    source="/VAF/src/nn/ECAPATDNN",
    savedir=f"./pretrained_models/ECAPATDNN_{args.worker_index}",
    run_opts={"device": args.device},
)
emb_CAMPP = get_embedding(n_mels = 80,embedding_size = 512,sample_rate = 16000,epoch = 80,
        checkpoints_dir = f"/VAF/src/nn/CAMPP",device=args.device)


def resample(wav,sr,target_sr):
    wav = wav.reshape(1,-1)
    if sr == target_sr:
        return wav
    resample = T.Resample(sr, target_sr)
    wav = resample(wav)
    wav = wav.reshape(1,-1)
    return wav


def get_embedding(file_path,savepath):
    filename = file_path.split('/')[-1].split('.')[0]
    spk_id = file_path.split('/')[-2]
    output_path = os.path.join(savepath,spk_id,f"{filename.replace('_','')}_ECAPATDNN"+".npy")
    output_path2 = os.path.join(savepath,spk_id,f"{filename.replace('_','')}_CAMPP"+".npy")
    # mkdir of output_path
    os.makedirs(os.path.join(savepath,spk_id),exist_ok=True)
    # if os.path.exists(output_path):
    if os.path.exists(output_path):
        if os.path.exists(output_path2):
            # print(f"Skip {filename}")
            return 1
    # read wav data from file_path, and resample to 16k
    wav, fs = torchaudio.load(file_path)
    # print(f"Befor reshape: {wav.shape}")
    # print(f"fs: {fs}")
    wav=wav.reshape(-1)
    # print("After reshape:",wav.shape)
    print(f"Length: {len(wav)/fs}")
    # if len(wav)/fs < 10:
    #     # print(f"Too short. Length: {len(wav)/fs}")
    #     return 0
    wav = resample(wav, fs, 16000)
    wav.to(args.device)
    try:
        embedding = emb.encode_batch(wav)[0][0]
        embedding_campp = emb_CAMPP.encode_batch(wav).reshape(-1)
        print(embedding.shape)
        print(embedding_campp.shape)
    except Exception as e:
        print(e)
        return 0
    assert embedding.shape == (args.emb_dim,)
    # save as npy
    
    emb_npy = embedding.detach().cpu().numpy()
    emb_npy=emb_npy.reshape(-1)
    emb2_npy = embedding_campp.detach().cpu().numpy()
    emb2_npy=emb2_npy.reshape(-1)
    np.save(output_path2,emb2_npy)
    np.save(output_path,emb_npy)
    return
        

if __name__ == "__main__":
    # make dst folder
    os.makedirs(args.dst_path,exist_ok=True)

    # get all wavs in args.fold_path, recursive
    all_wavs = []

    for root, dirs, files in os.walk(args.fold_path):
        for file in files:
            if file.endswith(".wav"):
                all_wavs.append(os.path.join(root, file))

    all_wavs = sorted(all_wavs)
    print(f"Total {len(all_wavs)} wavs.")
    # tiny_len = len(all_wavs)//args.total_workers
    # all_wavs = all_wavs[args.worker_index*tiny_len:(args.worker_index+1)*tiny_len]
    # multi process call get_embedding
    print(f"Index #{args.worker_index}/{args.total_workers} start. Total {len(all_wavs)} wavs.")
    for file_path in tqdm(all_wavs):
        get_embedding(file_path,args.dst_path)
