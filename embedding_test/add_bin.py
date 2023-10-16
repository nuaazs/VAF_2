import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import random
from dguard.interface.pretrained import load_by_name,ALL_MODELS

import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--GR_path', type=str, default=" ",help='')
    parser.add_argument('--MD_path', type=str, default="",help='')
    parser.add_argument('--save_path', type=str, default=" ",help='')
    args = parser.parse_args()
    data1 = np.fromfile(args.GR_path,dtype=np.uint8)
    data2 = np.fromfile(args.MD_path,dtype=np.uint8)
    merge_data = np.concatenate((data1,data2))
    merge_data.tofile(args.save_path)
