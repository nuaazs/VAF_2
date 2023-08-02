import numpy as np
import os
import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
folder_path="result"
paths = glob.glob(f'{folder_path}/**/*.score', recursive=True)
# path = '/home/duanyibo/dyb/test_model/result/CAMPP_EMB_512/male_result/scores/male.trials.score'
for path in tqdm(paths):
    lines = open(path).readlines()
    scores={}
    scores = {'target': [], 'nontarget': []} 
    for line in lines:
        cols = line.split(' ')  
        label = cols[2]
        score = float(cols[3])
        scores[label].append(score)

    
    num_bins = 100
    target_count = len(scores['target'])
    nontarget_count = len(scores['nontarget'])
    plt.hist(scores['target'], bins=num_bins, range=(-1, 1), color='g', alpha=0.5, label='target', weights=np.ones(len(scores['target'])) / target_count)

    plt.hist(scores['nontarget'], bins=num_bins, range=(-1, 1), color='r', alpha=0.5, label='nontarget', weights=np.ones(len(scores['nontarget'])) / nontarget_count)

    plt.legend()
    model = path.split("/")[-4]
    trial = path.split("/")[-1].split(".")[0]
    plt.title(f'{model} {trial} of Scores')
    plt.xlabel('Score')
    plt.ylabel('Percentage')
    os.makedirs(f'score_plot/{model}/',exist_ok=True)
    plt.savefig(f'score_plot/{model}/{model}_{trial}.png')
    plt.clf()

