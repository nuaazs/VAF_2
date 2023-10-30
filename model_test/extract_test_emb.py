# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import torch
import torchaudio
import importlib
from kaldiio import WriteHelper
from tqdm import tqdm
import dguard.models as M
from dguard.utils.builder import build
from dguard.utils.utils import get_logger
from dguard.utils.config import build_config
from dguard.utils.fileio import load_wav_scp
from dguard.interface.pretrained import load_by_name,ALL_MODELS
import random
import numpy as np

parser = argparse.ArgumentParser(description='Extract embeddings for evaluation.')
parser.add_argument('--exp_dir', default='', type=str, help='Exp dir')
parser.add_argument('--data', default='', type=str, help='Data dir')
parser.add_argument('--model_name', default='', type=str, help='model_name')
parser.add_argument('--seed', default=123, type=int, help='seed')
parser.add_argument('--length', default=3.0, type=float, help='wav length')
parser.add_argument('--start_from', default=0.0, type=float, help='start from(s)')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')



def main():
    args = parser.parse_args(sys.argv[1:])
    seed = args.seed
    # set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    embedding_dir = os.path.join(args.exp_dir, 'embeddings')
    os.makedirs(embedding_dir, exist_ok=True)

    logger = get_logger()

    if args.use_gpu:
        if torch.cuda.is_available():
            gpu = int(args.gpu[rank % len(args.gpu)])
            device = torch.device('cuda', gpu)
        else:
            msg = 'No cuda device is detected. Using the cpu device.'
            if rank == 0:
                logger.warning(msg)
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    fusion_result = False

    if "_and_" in args.model_name:
        MODELS = []
        models = []
        feature_extractors = []
        sample_rates = []
        MODELS = args.model_name.split("_and_")
        fusion_result = True
        data = load_wav_scp(args.data)
        data_k = list(data.keys())
        local_k = data_k[rank::world_size]
        if len(local_k) == 0:
            msg = "The number of threads exceeds the number of files"
            logger.info(msg)
            sys.exit()

        for _index in range(len(MODELS)):
            now_model = MODELS[_index]
            model,feature_extractor,sample_rate = load_by_name(now_model,device)
            emb_ark = os.path.join(embedding_dir, f"fusion_{now_model}_"+'xvector_%02d.ark'%rank)
            emb_scp = os.path.join(embedding_dir, f"fusion_{now_model}_"+'xvector_%02d.scp'%rank)

            if rank == 0:
                logger.info('Start extracting embeddings.')
            with torch.no_grad():
                with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
                    for k in tqdm(local_k):
                        wav_path = data[k]
                        if args.exp_dir.split("/")[-1] == "cti_result":
                            num_samples = int(10*config.sample_rate)
                            wav, fs = torchaudio.load(wav_path, frame_offset=0, num_frames=num_samples)
                        else:
                            wav, fs = torchaudio.load(wav_path)
                        # assert fs == config.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
                        # random choose a segment len=fs*args.length from wav by random
                        if args.length*fs<wav.shape[1]:
                            random_start = random.randint(args.start_from*fs, wav.shape[1] - int(fs*args.length))
                            wav = wav[:,random_start:random_start+int(fs*args.length)]
                        feat = feature_extractor(wav)
                        feat = feat.unsqueeze(0)
                        feat = feat.to(device)
                        emb = model(feat)[-1].detach().cpu().numpy()
                        writer(k, emb)

    else:  
        MODEL = args.model_name
        model,feature_extractor,sample_rate = load_by_name(MODEL,device)
        model.eval()
        model.to(device)
        # print(args.data)
        data = load_wav_scp(args.data)
        data_k = list(data.keys())
        local_k = data_k[rank::world_size]
        if len(local_k) == 0:
            msg = "The number of threads exceeds the number of files"
            logger.info(msg)
            sys.exit()

        emb_ark = os.path.join(embedding_dir, 'xvector_%02d.ark'%rank)
        emb_scp = os.path.join(embedding_dir, 'xvector_%02d.scp'%rank)

        if rank == 0:
            logger.info('Start extracting embeddings.')
        with torch.no_grad():
            with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer:
                for k in tqdm(local_k):
                    wav_path = data[k]
                    if args.exp_dir.split("/")[-1] == "cti_result":
                        num_samples = int(10*config.sample_rate)
                        wav, fs = torchaudio.load(wav_path, frame_offset=0, num_frames=num_samples)
                    else:
                        wav, fs = torchaudio.load(wav_path)
                    # assert fs == config.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
                    # random choose a segment len=fs*args.length from wav by random
                    if args.length*fs<wav.shape[1]:
                        random_start = random.randint(args.start_from*fs, wav.shape[1] - int(fs*args.length))
                        wav = wav[:,random_start:random_start+int(fs*args.length)]
                    half_len = int(wav.shape[1]/2)
                    wav_a = wav[:,:half_len]
                    wav_b = wav[:,half_len:]
                    feat_a = feature_extractor(wav_a)
                    feat_b = feature_extractor(wav_b)
                    feat_a = feat_a.unsqueeze(0)
                    feat_b = feat_b.unsqueeze(0)
                    feat_a = feat_a.to(device)
                    feat_b = feat_b.to(device)
                    emb_a = model(feat_a)[-1].detach().cpu().numpy().reshape(1,-1)
                    emb_b = model(feat_b)[-1].detach().cpu().numpy().reshape(1,-1)


                    feat = feature_extractor(wav)
                    feat = feat.unsqueeze(0)
                    feat = feat.to(device)
                    emb = model(feat)[-1].detach().cpu().numpy().reshape(1,-1)

                    emb = np.concatenate((emb_a,emb_b,emb),axis=1) # [1,512*3]
                    # assert emb.shape == (1,512*3), "emb shape error"
                    emb = emb.reshape(-1)
                    writer(k, emb)

if __name__ == "__main__":
    main()
