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

import onnxruntime as rt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=False, default="/VAF/train/egs/voxceleb/sv-repvgg/conf/repvgg.yaml", help='config file')
    parser.add_argument('--checkpoint', required=False, default="/VAF/train/egs/voxceleb/sv-repvgg/exp/repvgg/models/CKPT-EPOCH-142-00/embedding_model.ckpt",help='checkpoint model')
    parser.add_argument('--output_file', required=False, default="/VAF/train/pretrained_models/onnx/repvgg_epoch142.onnx",help='output file')
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
        
    # 加载ONNX模型
    model_path = "/VAF/train/pretrained_models/onnx/repvgg_epoch142.onnx"
    sess = rt.InferenceSession(model_path)

    # 定义输入数据
    example_input = np.random.randn(1, 800, 80).astype(np.float32)

    # 进行推理
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    outputs = sess.run([output_name], {input_name: example_input})
    output_from_pytorch = model(torch.from_numpy(example_input)).detach().numpy()

    # 获取推理结果
    output = outputs[0]
    print(f"output from onnxruntime: {output}")
    print(f"output from pytorch: {output_from_pytorch}")
    # 打印输出结果
    # print(output)


if __name__ == '__main__':
    main()




