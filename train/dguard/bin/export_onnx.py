# coding = utf-8
# @Time    : 2023-08-10  09:12:39
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Export ONNX model.
# Open Neural Network Exchange (ONNX)
# ONNX is a universal model representation format that can be used across multiple deep learning frameworks and platforms.
# It allows models to be exported from one framework to ONNX format, and then loaded and executed in other ONNX-supported frameworks or engines.
# Unlike JIT, ONNX is not limited to a specific framework and focuses on cross-framework deployment and inference.

from __future__ import print_function

import argparse
import os

import torch
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
    print(model)
    checkpoint = torch.load(args.checkpoint,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)#checkpoint['state_dict'])
    # Set the model to evaluation mode
    model.eval()
    # Create a dummy input tensor
    example_input = torch.randn(1,800, 80)
    # Export the model to ONNX format
    # dynamic_axes = {'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(model, example_input, args.output_file, do_constant_folding=True,verbose=True,opset_version=14,input_names=['feats'],output_names=['embs'],dynamic_axes={'feats': {0: 'B', 1: 'T'}, 'embs': {0: 'B'}})
    print('Export model successfully, see {}'.format(args.output_file))

if __name__ == '__main__':
    main()
