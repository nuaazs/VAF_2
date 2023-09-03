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
from speakerlabduanyibo.utils.builder import build
from speakerlabduanyibo.utils.utils import get_logger
from speakerlabduanyibo.utils.config import build_config
from speakerlabduanyibo.utils.fileio import load_wav_scp
from dguard.interface.pretrained import load_by_name,ALL_MODELS

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

    # import model
    # model = getattr(M, MODEL)()

    # load parameters, set to eval mode, move to GPU
    # device = "cuda:0"

    # model.load_state_dict(torch.load(CKPT_PATH[MODEL], map_location="cpu"),strict=True)

    # Build the embedding model
    # feature_extractor = build('feature_extractor', config)
    model,feature_extractor,sample_rate = load_by_name(MODEL,device)
    model.eval()
    model.to(device)
    print(args.data)
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
                assert fs == config.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
                feat = feature_extractor(wav)
                feat = feat.unsqueeze(0)
                feat = feat.to(device)
                emb = model(feat)[-1].detach().cpu().numpy()
                # emb = mode(feat).detach().cpu().numpy()
                writer(k, emb)

if __name__ == "__main__":
    main()
