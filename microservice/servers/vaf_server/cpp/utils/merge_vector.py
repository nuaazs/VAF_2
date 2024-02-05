# export redis data to a python dict and save as a npy file
# coding = utf-8
# @Time    : 2023-05-14  22:33:28
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 将vector bin以及id list文件分割成多份.

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold_path', type=str, default="./vector.bin", help='vector bin path')
parser.add_argument('--output', type=str, default="vector_a_all", help='vector bin path')
args = parser.parse_args()

if __name__ == '__main__':
    # rm fold_path/output.bin and fold_path/output.txt
    if os.path.exists(os.path.join(args.fold_path, f"{args.output}.bin")):
        # alert
        # raise error
        print(f"Error: {os.path.join(args.fold_path, f'{args.output}.bin')} already exists, please remove it first.")
        # exit
        exit(1)
        # os.remove(os.path.join(args.fold_path, f"{args.output}.bin"))
        # print(f"Removed {os.path.join(args.fold_path, f'{args.output}.bin')}")
    if os.path.exists(os.path.join(args.fold_path, f"{args.output}.txt")):
        print(f"Error: {os.path.join(args.fold_path, f'{args.output}.txt')} already exists, please remove it first.")
        # exit
        exit(1)
    # 读取fold_path下的所有文件，文件名相同的txt和bin文件为同一组
    # 获得所有的组别
    all_pairs = []
    for file in os.listdir(args.fold_path):
        if file.endswith(".txt"):
            all_pairs.append(file.split('.')[0])
    all_pairs = list(set(all_pairs))
    print(f"Total {len(all_pairs)} pairs")

    # for each pair, merge bin to one file, merge txt to one file

    # merge bin
    bin_files = []
    for pair in all_pairs:
        bin_files.append(os.path.join(args.fold_path, f"{pair}.bin"))
    data = []
    for file in bin_files:
        data.append(np.fromfile(file, dtype=np.float32))
    data = np.concatenate(data, axis=0)
    data.tofile(os.path.join(args.fold_path, f"{args.output}.bin"))
    print(f"Saved data to {os.path.join(args.fold_path, f'{args.output}.bin')}")

    # merge txt
    txt_files = []
    for pair in all_pairs:
        txt_files.append(os.path.join(args.fold_path, f"{pair}.txt"))
    data = []
    for file in txt_files:
        with open(file, 'r') as f:
            data += f.readlines()
    with open(os.path.join(args.fold_path, f"{args.output}.txt"), 'w') as f:
        for id in data:
            f.write(id)
    print(f"Saved id to {os.path.join(args.fold_path, f'{args.output}.txt')}")


