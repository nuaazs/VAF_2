import os
import numpy as np
import torch
from tqdm import tqdm

# set seed
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import cfg

# args
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--total_worker', type=int, default=1,help='')
parser.add_argument('--worker_idx', type=int, default=0,help='')
# parser.add_argument('--model_name', type=str, default="ERES2NET",help='')
args = parser.parse_args()

# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save log file
fh = logging.FileHandler(f'./log/log{args.worker_idx}.txt')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def get_embedding(file_path,embeddings):
    phone = file_path.split('/')[-2]
    filename = file_path.split('/')[-1].split('.')[0]
    print(embeddings[phone].keys())
    return phone,filename,embeddings[phone][filename]

if __name__ == "__main__":
    embeddings = {}
    all_wavs = []
    
    # cache_path = f"../../cache/{cfg.NAME}"
    # models = cfg.MODEL_NAME.split(",")
    # all_embedding = {}
    # if cfg.LOAD_NPY:
    for model in cfg.MODEL_NAME.split(","):
        embeddings = np.load(f"../../cache/{cfg.NAME}/{model}/embeddings.npy",allow_pickle=True).item()
        print(f"embedding: {embeddings.keys()}")
        embeddings = embeddings[model]
        print(f"embedding: {embeddings.keys()}")
        for phone in embeddings:
            phone_file_nums = 0
            for filename in embeddings[phone]:
                all_wavs.append(os.path.join(cfg.DATA_FOLDER,phone,filename))
                phone_file_nums += 1
        print(f"Model {model}, Load npy file successfully.")
        all_wavs = sorted(all_wavs)
        print(f"All Wavs Length: {len(all_wavs)}")
        length = len(all_wavs)//args.total_worker
        all_wavs_2 = all_wavs[args.worker_idx*length:(args.worker_idx+1)*length]
        print(f"Now choosed {len(all_wavs_2)} wavs.")


        if os.path.exists(f"../../cache/{cfg.NAME}/{model}/wav_pairs.npy") and cfg.LOAD_NPY:
            wav_pairs = np.load(f"../../cache/{cfg.NAME}/{model}/wav_pairs.npy",allow_pickle=True).tolist()
        else:
            wav_pairs = []
            # get one-to-one pairs, without repeat from all_wavs
            for wav1 in tqdm(all_wavs_2):
                for wav2 in all_wavs:
                    if wav1 == wav2:
                        continue
                    wav_pairs.append([wav1,wav2])
        print("Load npy file (wav pairs) successfully.")
        print(f"Total wav pairs: {len(wav_pairs)}")
        print(f"Worker {args.worker_idx} has {len(wav_pairs)} wav pairs.")
        labels = []
        scores = []
        label_0_cout = 0
        label_1_cout = 0


        # print(f"Wav pairs length: {len(wav_pairs)}")
        # print(f"{wav_pairs}")
        for wav1, wav2 in tqdm(wav_pairs):
            # try:
            print(f"Loading {wav1}...")
            phone1, filename1, embedding1 = get_embedding(wav1,embeddings)
            print(f"Loading {wav2}...")
            phone2, filename2, embedding2 = get_embedding(wav2,embeddings)
            # except:
            #     logger.error(f"Error in {wav1} or {wav2}")
            #     continue

            #print(f"emb1 shape: {embedding1.shape}")
            score = similarity(torch.Tensor(embedding1),torch.Tensor(embedding2)).numpy()
            # print()
            print(f"Phone#1: {phone1}\tPhone#2: {phone2}\tScore:{score}")
            #print(score)
            if phone1 == phone2:
                label = 1
                label_1_cout += 1
            else:
                label = 0
                label_0_cout += 1
            if label_0_cout >= 1000 and label == 0:
                continue
            if label_1_cout >= 1000 and label == 1:
                continue
            if label_0_cout >= 1000 and label_1_cout >= 1000:
                break
            scores.append(score)
            labels.append(label)
        # labels = np.array(labels)
        # scores = np.array(scores)
        # print(f"labels: {labels.shape}")
        # print(f"scores: {scores.shape}")
        # # save scores and labels
        # np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_scores_{args.worker_idx}.npy",scores)
        # np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_labels_{args.worker_idx}.npy",labels)
        
        # load noise embeddings
        if cfg.ADD_NOISE:
            noise_info = []
            noise_npys = [os.path.join(cfg.NOISE_PATH,npy) for npy in os.listdir(cfg.NOISE_PATH) if npy.endswith('.npy')]
            for npy_file in noise_npys:
                noise_embeddings = np.load(npy_file,allow_pickle=True).item()
                for phone in noise_embeddings:
                    for filename in noise_embeddings[phone]:
                        embedding = noise_embeddings[phone][filename]
                        noise_info.append([phone,filename,embedding])
            print("Adding noise to scores and labels.")
            phone_set = set()
            for wav in tqdm(all_wavs_2):
                phone = wav.split('/')[-2]
                if phone not in phone_set:
                    phone_set.add(phone)
                else:
                    continue
                phone,filename,embedding = get_embedding(wav,embeddings)
                # compare with noise
                for noise_phone,noise_filename,noise_embedding in noise_info:
                    score = similarity(torch.Tensor(embedding),torch.Tensor(noise_embedding)).numpy()[0][0]
                    # print(f"Phone#1: {phone}\tPhone#2: {noise_phone}\tScore:{score}")
                    #print(score)
                    if phone == noise_phone:
                        label = 1
                    else:
                        label = 0

                    scores.append(score)
                    labels.append(label)    
        labels = np.array(labels)
        scores = np.array(scores)
        print(f"labels: {labels.shape}")
        print(f"scores: {scores.shape}")
        # save scores and labels
        np.save(f"../../cache/{cfg.NAME}/{model}/scores_{args.worker_idx}.npy",scores)
        np.save(f"../../cache/{cfg.NAME}/{model}/labels_{args.worker_idx}.npy",labels)
