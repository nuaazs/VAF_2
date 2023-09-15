import numpy as np
import argparse
import os
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, required=True, help='path to csv files, split by ,')
parser.add_argument('--labels', type=str, required=True, help='labels, split by ,')
parser.add_argument('--save_dir', type=str, required=True, help='save dir')
args = parser.parse_args()

score_files = args.files.split(",")
labels = args.labels.split(",")

for _index,score_path in enumerate(score_files):
    file_label = labels[_index]
    plt.figure()
    lines = open(score_path).readlines()
    scores={}
    scores = {'target': [], 'nontarget': []} 
    for line in lines:
        cols = line.split(',')
        if cols[0].split("_")[0]==cols[1].split("_")[0]:
            label = "target"
        else:
            label = "nontarget"
        score = float(cols[-1])
        scores[label].append(score)
    num_bins = 100
    target_count = len(scores['target'])
    nontarget_count = len(scores['nontarget'])
    plt.hist(scores['target'], bins=num_bins, range=(-1, 1), color='g', alpha=0.5, label='target', weights=np.ones(len(scores['target'])) / target_count)
    plt.hist(scores['nontarget'], bins=num_bins, range=(-1, 1), color='r', alpha=0.5, label='nontarget', weights=np.ones(len(scores['nontarget'])) / nontarget_count)
    plt.legend()
    model = score_path.split("/")[-2].split("_")[1]
    trial = score_path.split("/")[-2].split("_")[0]
    plt.title(f'{model} {trial} of Scores')
    plt.xlabel('Score')
    plt.ylabel('Percentage')
    plt.savefig(f'{args.save_dir}/{file_label}.png')
    plt.clf()

