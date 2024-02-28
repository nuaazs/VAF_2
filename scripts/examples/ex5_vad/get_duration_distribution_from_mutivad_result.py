# coding = utf-8
# @Time    : 2023-03-14  10:26:30
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: VAD *.<ext> files in a folder, and save the results to dst folder with save folder structure.

import os
import numpy as np
import torchaudio
import torch
import argparse
import logging
import re
import random
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
# set seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


if __name__ == "__main__":

    # load all vad_*.npy in log folder and concat them to one list
    vad_list = []
    # find all vad_*.npy in log folder
    vad_files = os.listdir("log")
    print(f"vad_files: {vad_files}")
    # re vad_*.npy
    vad_files = [os.path.join("log", vad_file) for vad_file in vad_files if re.match(r"vad_\d+\.npy", vad_file)]
    for vad_file in vad_files:
        print(f"vad_file: {vad_file}")
        vad = np.load(vad_file, allow_pickle=True)
        # to list
        vad = vad.tolist()
        vad_list+=vad
    # vad_list = np.concatenate(vad_list, axis=0)
    print(f"vad_list: {len(vad_list)}"
            f"vad_list[0]: {vad_list[0]}")
    # get duration distribution
    data_list = [int(i[1]) for i in vad_list]
    print(data_list)
    # plot length distribution by info_list
    plt.figure(figsize=(20,10))
    # plot distribution, from 0 to 600s, step 10s
    

    plt.hist(data_list, bins=np.arange(0, 800, 10))
    plt.xlabel('length (s)')
    plt.ylabel('count')
    # xticks
    plt.xticks([i for i in range(0, 800, 50)])
    plt.grid()
    plt.title(f"All data")
    plt.savefig(f"./log/vad_all.png")