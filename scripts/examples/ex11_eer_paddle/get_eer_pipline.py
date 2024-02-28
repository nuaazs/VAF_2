import os
import numpy as np
import paddle
import metrics

import torch
from tqdm import tqdm

# set seed
import random
random.seed(0)
# paddle.seed(0)
np.random.seed(0)
torch.manual_seed(0)





import cfg

# args
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--total_worker', type=int, default=1,help='')
parser.add_argument('--worker_idx', type=int, default=0,help='')

args = parser.parse_args()


# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save log file
fh = logging.FileHandler(f'log{args.worker_idx}.txt')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

if "paddle" in cfg.NAME:
    similarity = paddle.nn.CosineSimilarity(axis=-1, eps=1e-6)
elif "speechbrain" in cfg.NAME:
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
else:
    similarity = None

def get_embedding(file_path,embeddings):
    phone = file_path.split('/')[-2]
    filename = file_path.split('/')[-1].split('.')[0]
    return phone,filename,embeddings[phone][filename]

def get_score(features1, features2):
    score = float(paddle.dot(features1.squeeze(), features2.squeeze()))
    return score


if __name__ == "__main__":
    embeddings = {}
    all_wavs = []
    embeddings = np.load(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",allow_pickle=True).item()
    for phone in embeddings:
        phone_file_nums = 0
        for filename in embeddings[phone]:
            all_wavs.append(os.path.join(cfg.DATA_FOLDER,phone,filename))
            phone_file_nums += 1
    print("Load npy file successfully.")
    all_wavs = sorted(all_wavs)
    print(f"All Wavs Length: {len(all_wavs)}")
    length = len(all_wavs)//args.total_worker
    all_wavs_2 = all_wavs[args.worker_idx*length:(args.worker_idx+1)*length]
    print(f"Now choosed {len(all_wavs_2)} wavs.")

    if os.path.exists(f"../../cache/{cfg.NAME}/{cfg.NAME}_wav_pairs.npy") and cfg.LOAD_NPY:
        wav_pairs = np.load(f"../../cache/{cfg.NAME}/{cfg.NAME}_wav_pairs.npy",allow_pickle=True).tolist()
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
    for wav1, wav2 in tqdm(wav_pairs):
        try:
            phone1, filename1, embedding1 = get_embedding(wav1,embeddings)
            phone2, filename2, embedding2 = get_embedding(wav2,embeddings)
        except:
            logger.error(f"Error in {wav1} or {wav2}")
            continue
        score = similarity(paddle.Tensor(embedding1),paddle.Tensor(embedding2)).numpy()[0]#.detach().cpu().numpy()[0]
        # score = get_score(paddle.Tensor(embedding1),paddle.Tensor(embedding2))
        if phone1 == phone2:
            label = 1
        else:
            label = 0
        #print(f"{phone1}:{filename1}\n{phone2}:{filename2}\n\t{score} - {label}")
        scores.append(score)
        labels.append(label)
        # if label == 0 and score > 0.8:
        #     logger.error(f"{phone1}:{filename1}\n{phone2}:{filename2}\n\t{score} - {label}")
        # if label == 1 and score < 0.6:
        #     logger.error(f"{phone1}:{filename1}\n{phone2}:{filename2}\n\t{score} - {label}")
    
    labels = np.array(labels)
    scores = np.array(scores)
    print(f"labels: {labels.shape}")
    print(f"scores: {scores.shape}")

    # save scores and labels
    np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_scores_{args.worker_idx}.npy",scores)
    np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_labels_{args.worker_idx}.npy",labels)
