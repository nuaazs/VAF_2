# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import torch
import torchaudio

from dguard.utils.builder import build
from dguard.utils.utils import get_logger
from dguard.utils.config import build_config
from dguard.utils.fileio import load_wav_scp

parser = argparse.ArgumentParser(description='Prediction for LID.')
parser.add_argument('--exp_dir', default='', type=str, help='Exp dir')
parser.add_argument('--data', default='', type=str, help='Data dir')
parser.add_argument('--use_gpu', action='store_true', help='Use gpu or not')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')


def main():
    args = parser.parse_args()
    config_file = os.path.join(args.exp_dir, 'config.yaml')
    config = build_config(config_file)

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    result_dir = os.path.join(args.exp_dir, 'results/predicts')
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, 'predict%d.txt'%rank)

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

    # Build the model
    embedding_model = build('embedding_model', config)
    label_encoder = build('label_encoder', config)
    config.num_classes = len(config.label_encoder)
    classifier = build('classifier', config)

    # Recover model params of last epoch
    config.checkpointer['args']['checkpoints_dir'] = os.path.join(args.exp_dir, 'models')
    config.checkpointer['args']['recoverables'] = {'embedding_model':embedding_model, 'classifier': classifier, 'label_encoder':label_encoder}
    checkpointer = build('checkpointer', config)
    checkpointer.recover_if_possible(epoch=config.num_epoch, device=device)

    model = torch.nn.Sequential(embedding_model, classifier)
    model.to(device)
    model.eval()
    feature_extractor = build('feature_extractor', config)

    data = load_wav_scp(args.data)
    data_k = list(data.keys())
    local_k = data_k[rank::world_size]
    if len(local_k) == 0:
        msg = "The number of threads exceeded the number of files"
        logger.info(msg)
        sys.exit()

    if rank == 0:
        logger.info('Start predicting...')
    with torch.no_grad():
        with open(result_file, 'w') as f:
            for k in local_k:
                wav_path = data[k]
                wav, fs = torchaudio.load(wav_path)
                assert fs == config.sample_rate, f"The sample rate of wav is {fs} and inconsistent with that of the pretrained model."
                feat = feature_extractor(wav)
                feat = feat.unsqueeze(0)
                feat = feat.to(device)
                output = model(feat).detach().cpu()
                predict_id = output.argmax(-1).item()
                predict_lang = label_encoder.ind2lab[predict_id]
                f.write('%s %s\n'%(k, predict_lang))

if __name__ == "__main__":
    main()