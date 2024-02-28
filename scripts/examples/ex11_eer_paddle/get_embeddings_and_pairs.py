import os
import numpy as np
import paddle
# import metrics

import torch
from tqdm import tqdm

# set seed
import random
random.seed(0)
# paddle.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save log file
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
import cfg

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

    # if npy exist, just load
    if os.path.exists(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy") and cfg.LOAD_NPY:
        embeddings = np.load(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",allow_pickle=True).item()
        for phone in embeddings:
            phone_file_nums = 0
            for filename in embeddings[phone]:
                all_wavs.append(os.path.join(cfg.DATA_FOLDER,phone,filename))
                phone_file_nums += 1
                # if phone_file_nums>=2:
                #     break
        print("Load npy file successfully.")
    else:
        from paddlespeech_encoder import generate_embedding as generate_embedding_paddle
        # from encoder import generate_embedding
        for phone in tqdm(os.listdir(cfg.DATA_FOLDER)):
            embeddings[phone] = {}
            phone_path = os.path.join(cfg.DATA_FOLDER, phone)
            for file in os.listdir(phone_path):
                try:
                    file_path = os.path.join(phone_path, file)
                    filename = os.path.splitext(file)[0]
                    file_size = os.path.getsize(file_path)
                    # if > 5 MB skip
                    if file_size > 20 * 1024 * 1024:
                        continue
                    # if "paddle" in cfg.NAME:
                    #     embeddings[phone][filename] = generate_embedding(file_path).detach().cpu().numpy()
                    elif "speechbrain" in cfg.NAME:
                        embeddings[phone][filename]=generate_embedding_paddle(file_path)#.detach().cpu().numpy()
                        print(f"Shape: {embeddings[phone][filename].shape}")
                    else:
                        print("Please check your config file.")
                    all_wavs.append(file_path)
                except Exception as e:
                    logger.error(f"Error in {file_path}: {e}")
                    continue
        os.makedirs(f"../../cache/{cfg.NAME}",exist_ok=True)
        np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",embeddings)
    print("Done!")