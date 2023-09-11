# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import torch
import numpy as np
import torchaudio
import importlib
from kaldiio import WriteHelper
from tqdm import tqdm
import dguard.models as M
from speakerlabduanyibo.utils.builder import build
from speakerlabduanyibo.utils.utils import get_logger
from speakerlabduanyibo.utils.config import build_config
from speakerlabduanyibo.utils.fileio import load_wav_scp
from dguard.interface.pretrained import load_by_name,ALL_MODELS
from dguard.interface import PretrainedModel

parser = argparse.ArgumentParser(description='Extract embeddings for evaluation.')
parser.add_argument('--exp_dir', default='', type=str, help='Exp dir')
parser.add_argument('--data', default='', type=str, help='Data dir')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
CKPT_PATH = {
    "CAMPP_EMB_512":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/campp_voxceleb/campp_voxceleb.bin",
    "ECAPA_TDNN_1024_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/ecapatdnn1024_voxceleb/ecapatdnn1024_voxceleb.bin",
    "ERES2NET_BASE_EMB_192":"/home/zhaosheng/asr_damo_websocket/online/microservice/servers/encode_utils/damo_models/eres2netbase_voxceleb/eres2netbase_voxceleb.ckpt",
    "REPVGG_TINY_A0_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-100-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_95":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-95-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_95":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-95-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_90":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-90-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_90":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-90-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_85":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-85-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_85":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-85-00/embedding_model.ckpt",
    "REPVGG_TINY_A0_EMB_512_80":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-80-00/embedding_model.ckpt",
    "DFRESNET56_EMB_512_80":"/home/zhaosheng/3D-Speaker/egs/voxceleb/sv-dfresnet/exp/dfresnet56/models/CKPT-EPOCH-80-00/embedding_model.ckpt",
}

def main():
    args = parser.parse_args(sys.argv[1:])
    config_file = os.path.join("/home/duanyibo/dyb/test_model/speakerlabduanyibo", 'config.yaml')
    config = build_config(config_file)
    # rank = 0
    # world_size = 1

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # assert world_size == 7, f" world_size {world_size} != 7 "

    embedding_dir = os.path.join(args.exp_dir, 'embeddings')
    os.makedirs(embedding_dir, exist_ok=True)
    cmf_embedding_dir = os.path.join(args.exp_dir, 'cmf_embeddings')
    os.makedirs(cmf_embedding_dir, exist_ok=True)
    cmf_nums_dir = os.path.join(args.exp_dir, 'cmf_num')
    os.makedirs(cmf_nums_dir, exist_ok=True)
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
    MODEL = args.exp_dir.split("/")[-2]
    print(MODEL)
    infer = PretrainedModel(MODEL, device=device, strict=True, mode="extract")
    # import model
    # model = getattr(M, MODEL)()

    # load parameters, set to eval mode, move to GPU
    # device = "cuda:0"

    # model.load_state_dict(torch.load(CKPT_PATH[MODEL], map_location="cpu"),strict=True)

    # Build the embedding model
    # feature_extractor = build('feature_extractor', config)
    # model,feature_extractor,sample_rate = load_by_name(MODEL,device)
    # model.eval()
    # model.to(device)
    data = load_wav_scp(args.data)
    data_k = list(data.keys())
    local_k = data_k[rank::world_size]
    if len(local_k) == 0:
        msg = "The number of threads exceeds the number of files"
        logger.info(msg)
        sys.exit()

    emb_ark = os.path.join(embedding_dir, 'xvector_%02d.ark'%rank)
    emb_scp = os.path.join(embedding_dir, 'xvector_%02d.scp'%rank)
    cmf_emb_ark = os.path.join(cmf_embedding_dir, 'xvector_%02d.ark'%rank)
    cmf_emb_scp = os.path.join(cmf_embedding_dir, 'xvector_%02d.scp'%rank)
    cmf_num_ark = os.path.join(cmf_nums_dir, 'xvector_%02d.ark'%rank)
    cmf_num_scp = os.path.join(cmf_nums_dir, 'xvector_%02d.scp'%rank)

    if rank == 0:
        logger.info('Start extracting embeddings.')
    with torch.no_grad():
        with WriteHelper(f'ark,scp:{emb_ark},{emb_scp}') as writer ,WriteHelper(f'ark,scp:{cmf_emb_ark},{cmf_emb_scp}') as cmf_writer,WriteHelper(f'ark,scp:{cmf_num_ark},{cmf_num_scp}') as num_writer:
            for k in tqdm(local_k):
                wav_path = data[k]
                result = infer.inference([wav_path], cmf=True, segment_length=-5,crops_num_limit=1,segment_length_limit=2*16000)
                # emb = mode(feat).detach().cpu().numpy()
                emb = result[0][0].detach().cpu().numpy()
                cmf_emb = result[0][1]
                cmf_num = result[0][2]
                # cmf_emb = (1-((9-cmf_num)/9)*0.3)*cmf_emb
                
                cmf_emb_np = np.array([cmf_emb,cmf_emb,cmf_emb], dtype=np.float32)
                cmf_num_np = np.array([cmf_num,cmf_num,cmf_num], dtype=np.float32)
                
                writer(k, emb)
                cmf_writer(k,cmf_emb_np)
                num_writer(k,cmf_num_np)
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
