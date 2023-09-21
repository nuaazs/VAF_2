import os
import numpy as np
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--npy_path', type=str, default="/home/duanyibo/dyb/test_npy",help='')
    parser.add_argument('--bin_path', type=str, default="/home/duanyibo/dyb/bin_path",help='')
    args = parser.parse_args()
    folder = args.npy_path
    out_file = os.path.join(args.bin_path,'enroll.bin')
    out_text = os.path.join(args.bin_path,'enroll.txt')

    merged = []
    paths = []
    for filename in tqdm(os.listdir(folder)):
        print(filename)
        if filename.endswith('.npy'): 
            data = np.load(os.path.join(folder, filename))
            data = data.reshape(-1)
            # print(data.shape)
            merged.append(data)
            paths.append(filename)

    merged = np.concatenate(merged)
    merged.tofile(out_file)
    with open(out_text, 'w') as f:
        for path in paths:
            f.write(path + '\n')
