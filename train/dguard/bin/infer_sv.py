# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio. 
Please pre-install "modelscope".
Usage:
    1. extract the embedding from the wav file.
        `python infer_sv.py --model_id $model_id --wavs $wav_path `
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv.py --model_id $model_id --wavs $wav_path1 $wav_path2 `
    3. extract embeddings from the wav list.
        `python infer_sv.py --model_id $model_id --wavs $wav_list `
"""

import os
import sys
import re
import pathlib
import numpy as np
import argparse
import torch
import torchaudio

try:
    from dguard.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from dguard.process.processor import FBank

from dguard.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

parser = argparse.ArgumentParser(description='Extract speaker embeddings.')
parser.add_argument('--model_id', default='', type=str, help='Model id in modelscope')
parser.add_argument('--wavs', nargs='+', type=str, help='Wavs')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')

CAMPPLUS_VOX = {
    'obj': 'dguard.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'dguard.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'dguard.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'dguard.models.eres2net.ResNet_aug.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': CAMPPLUS_VOX, 
        'model_pt': 'campplus_voxceleb.bin', 
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0', 
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2', 
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.4', 
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
}

def main():
    args = parser.parse_args()
    assert isinstance(args.model_id, str) and \
        is_official_hub_path(args.model_id), "Invalid modelscope model id."
    assert args.model_id in supports, "Model id not currently supported."
    save_dir = os.path.join(args.local_model_dir, args.model_id.split('/')[1])
    save_dir =  pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    conf = supports[args.model_id]
    # download models from modelscope according to model_id
    cache_dir = snapshot_download(
                args.model_id,
                revision=conf['revision'],
                )
    cache_dir = pathlib.Path(cache_dir)

    embedding_dir = save_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True, parents=True)

    # link
    download_files = ['examples', conf['model_pt']]
    for src in cache_dir.glob('*'):
        if re.search('|'.join(download_files), src.name):
            dst = save_dir / src.name
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(src)

    pretrained_model = save_dir / conf['model_pt']
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    model = conf['model']
    embedding_model = dynamic_import(model['obj'])(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    def compute_embedding(wav_file, save=True):
        # load wav
        wav = load_wav(wav_file)
        # compute feat
        feat = feature_extractor(wav).unsqueeze(0)
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu().numpy()
        
        if save:
            save_path = embedding_dir / (
            '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            np.save(save_path, embedding)
            print(f'[INFO]: The extracted embedding from {wav_file} is saved to {save_path}.')
        
        return embedding

    # extract embeddings
    print(f'[INFO]: Extracting embeddings...')

    if args.wavs is None or len(args.wavs) == 2:
        if args.wavs is None:
            try:
                # use example wavs
                examples_dir = save_dir / 'examples'
                wav_path1, wav_path2 = list(examples_dir.glob('*.wav'))[0:2]
                print(f'[INFO]: No wavs input, use example wavs instead.')
            except:
                assert Exception('Invalid input wav.')
        else:
            # use input wavs
            wav_path1, wav_path2 = args.wavs

        embedding1 = compute_embedding(wav_path1)
        embedding2 = compute_embedding(wav_path2)

        # compute similarity score
        print('[INFO]: Computing the similarity score...')
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        scores = similarity(torch.from_numpy(embedding1), torch.from_numpy(embedding2)).item()
        print('[INFO]: The similarity score between two input wavs is %.4f' % scores)
    elif len(args.wavs) == 1:
        # input one wav file
        if args.wavs[0].endswith('.wav'):
            # input is wav path
            wav_path = args.wavs[0]
            embedding = compute_embedding(wav_path)
        else:
            try:
                # input is wav list
                wav_list_file = args.wavs[0]
                with open(wav_list_file,'r') as f:
                    wav_list = f.readlines()
            except:
                raise Exception('[ERROR]: Input should be wav file or wav list.')
            for wav_path in wav_list:
                wav_path = wav_path.strip()
                embedding = compute_embedding(wav_path)
    else:
        raise Exception('[ERROR]: Supports up to two input files')


if __name__ == '__main__':
    main()
