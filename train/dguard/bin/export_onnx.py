# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

# Open Neural Network Exchange (ONNX)
# ONNX is a universal model representation format that can be used across multiple deep learning frameworks and platforms.
# It allows models to be exported from one framework to ONNX format, and then loaded and executed in other ONNX-supported frameworks or engines.
# Unlike JIT, ONNX is not limited to a specific framework and focuses on cross-framework deployment and inference.

from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import sys
sys.path.append('/VAF/train')
from dguard.utils.config import build_config
from dguard.utils.builder import build
import torch.onnx as onnx

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=False, default="/VAF/train/egs/voxceleb/sv-eres2net/conf/eres2net.yaml", help='config file')
    parser.add_argument('--checkpoint', required=False, default="/VAF/train/egs/voxceleb/sv-eres2net/exp/eres2net/models/eres2net_voxceleb.ckpt",help='checkpoint model')
    parser.add_argument('--output_file', required=False, default="/VAF/train/pretrained_models/onnx/eres2net.onnx",help='output file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("checkpoint file ({}) does not exist !!!".format(args.checkpoint))
    # make sure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    config = build_config(args.config, None, False)
    model = build('embedding_model', config)

    mean_vec=False
    if mean_vec:
        mean_vec = torch.tensor(np.load(mean_vec), dtype=torch.float32)
    else:
        embed_dim = configs["embedding_size"]
        mean_vec = torch.zeros(embed_dim, dtype=torch.float32)
    class Model(nn.Module):
        def __init__(self, model, mean_vec=None):
            super(Model, self).__init__()
            self.model = model
            self.register_buffer("mean_vec", mean_vec)

        def forward(self, feats):
            outputs = self.model(feats)  # embed or (embed_a, embed_b)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            embeds = embeds - self.mean_vec
            return embeds
    model = Model(model, mean_vec)



    
    # print(model)
    checkpoint = torch.load(args.checkpoint,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint,strict=False)#checkpoint['state_dict'])
    print('Load checkpoint successfully, see {}'.format(args.checkpoint))
    # Set the model to evaluation mode
    model.eval()


    feat_dim = configs.get('fbank_dim', 80)
    num_frms = 200
    example_input = torch.ones(1, num_frms, feat_dim)
    # test py
    output=model(example_input)
    print(output.shape)
    torch.onnx.export(
        model, example_input,
        args.output_file,
        do_constant_folding=True,
        verbose=False,
        opset_version=14,
        input_names=['feats'],
        output_names=['embs'],
        dynamic_axes={'feats': {0: 'B', 1: 'T'}, 'embs': {0: 'B'}})
    
    # Create a dummy input tensor
    # example_input = torch.randn(1,800, 80)
    # Export the model to ONNX format
    # dynamic_axes = {'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}}
    # torch.onnx.export(model, example_input, args.output_file, do_constant_folding=True,verbose=True,opset_version=14,input_names=['feats'],output_names=['embs'],dynamic_axes={'feats': {0: 'B', 1: 'T'}, 'embs': {0: 'B'}})
    print('Export model successfully, see {}'.format(args.output_file))

if __name__ == '__main__':
    main()
