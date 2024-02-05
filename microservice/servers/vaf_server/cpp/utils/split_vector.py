# export redis data to a python dict and save as a npy file
# coding = utf-8
# @Time    : 2023-05-14  22:33:28
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 将vector bin以及id list文件分割成多份.

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_bin_path', type=str, default="./vector.bin", help='vector bin path')
parser.add_argument('--raw_txt_path', type=str, default='./vector.txt', help='vector txt path')
parser.add_argument('--number', type=int, default=1, help='number of split')
parser.add_argument('--emb_size', type=int, default=192, help='number of split')

parser.add_argument('--save_folder', type=str, default='./split_output', help='')
args = parser.parse_args()

if __name__ == '__main__':
    # make save folder
    os.makedirs(args.save_folder, exist_ok=True)
    # 读取二进制文件
    data_ = np.fromfile(args.raw_bin_path, dtype=np.float32).reshape(-1, args.emb_size)
    print(f"Raw Data shape: {data_.shape}")
    raw_len = data_.shape[0]
    # split data to number parts
    split_len = raw_len // args.number
    for i in range(args.number):
        if i == args.number - 1:
            data = data_[i * split_len:]
        else:
            data = data_[i * split_len: (i + 1) * split_len]
        data = data.reshape(-1)
        print(f"Data shape: {data.shape}")
        data.tofile(os.path.join(args.save_folder, f"vector_{i}.bin"))
        print(f"Saved data to {os.path.join(args.save_folder, f'vector_{i}.bin')}")

    # split id to number parts
    with open(args.raw_txt_path, 'r') as f:
        id_list = f.readlines()
    split_len = len(id_list) // args.number
    for i in range(args.number):
        if i == args.number - 1:
            data = id_list[i * split_len:]
        else:
            data = id_list[i * split_len: (i + 1) * split_len]
        with open(os.path.join(args.save_folder, f"id_{i}.txt"), 'w') as f:
            for id in data:
                f.write(id)
        print(f"Saved id to {os.path.join(args.save_folder, f'id_{i}.txt')}")