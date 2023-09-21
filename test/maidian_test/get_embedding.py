import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import random
from dguard.interface.pretrained import load_by_name,ALL_MODELS
import argparse
import glob
import logging

logging.basicConfig(level=logging.WARNING,filename="./get_embedding.log",filemode='w')

def encode_wav(wav_file_path):
    # assert 16k, if not convert
    feat=torch.tensor(np.load(wav_file_path))
    feat = feat.unsqueeze(0)
    feat = feat.to(args.index)
    emb = model(feat)[-1].detach().cpu().numpy()
    # embedding = emb.forward(wav).reshape(1,-1).detach().cpu().numpy()
    return emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wav_path', type=str, default=" ",help='')
    parser.add_argument('--index', type=int, default="0",help='GPU')
    parser.add_argument('--save_path', type=str, default=" ",help='')
    parser.add_argument('--model', type=str, default='tnet',help='')
    parser.add_argument('--length', type=float, default=10.0,help='')
    parser.add_argument('--exp', type=str, default="60w",help='')
    parser.add_argument('--overwrite', type=float, default=0,help='')
    args = parser.parse_args()

    model,feature_extractor,sample_rate = load_by_name(args.model,args.index)
    model.eval()
    npy_path=os.path.join(args.wav_path,'**/*.npy')
    npy_files = glob.glob(npy_path)

    file_lists=os.path.join(args.save_path.rsplit("/",2)[0],args.exp,args.model,"list.txt")
    os.makedirs(file_lists.rsplit("/",1)[0], exist_ok=True)
    f=open(file_lists,"w")
    for line in npy_files:
      f.write(line + "\n")
    f.close()
    ##################################################
    for npy_file in tqdm(npy_files):
        try:
            spkid= npy_file.split(" ")[0].split(".npy")[0].split("/")[-2]
            utt = npy_file.split(" ")[0].split(".npy")[0].split("/")[-1]
            new_path = os.path.join(args.save_path,f"{spkid}_{utt}.npy")
            if not args.overwrite:
                if os.path.exists(new_path):
                    continue
            with torch.no_grad():
                embedding = encode_wav(npy_file)
                np.save(new_path,embedding)
        except Exception as e:
            print("Other error occurred:", e)
            logging.error(f"{npy_file},{e}")
