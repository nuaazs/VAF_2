# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import re
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper
from sklearn.metrics.pairwise import cosine_similarity

from dguard.utils.utils import get_logger
from dguard.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm,compute_tn_fn_tp_fp)

parser = argparse.ArgumentParser(description='Compute score and metrics')
parser.add_argument('--enrol_data', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data', default='', type=str, help='Test data dir')
parser.add_argument('--scores_dir', default='', type=str, help='Scores dir')
parser.add_argument('--scores_all', default='', type=str, help='Scores dir')
parser.add_argument('--exp_id', default='', type=str, help='Scores dir')

parser.add_argument('--trials', nargs='+', help='Trial')
parser.add_argument('--p_target', default=0.01, type=float, help='p_target in DCF')
parser.add_argument('--c_miss', default=1, type=float, help='c_miss in DCF')
parser.add_argument('--c_fa', default=1, type=float, help='c_fa in DCF')

def main():
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.scores_dir, exist_ok=True)

    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt = "%(message)s")
    # logger_all = get_logger(fpath=, fmt = "%(message)s")
    def collect(data_dir):
        data_dict = {}
        emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$',i)]
        if len(emb_arks) == 0:
            raise Exception(f'No embedding ark files found in {data_dir}')

        # load embedding data
        for ark in emb_arks:
            with ReadHelper(f'ark:{ark}') as reader:
                for key, array in reader:
                    data_dict[key] = array

        return data_dict

    enrol_dict = collect(args.enrol_data)
    test_dict = collect(args.test_data)

    for trial in args.trials:
        scores = []
        labels = []

        trial_name = os.path.basename(trial)
        score_path = os.path.join(args.scores_dir, f'{trial_name}.score')
        with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
            lines = trial_f.readlines()
            for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                pair = line.strip().split()
                enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                              test_emb.reshape(1, -1))[0][0]
                # write the score
                score_f.write(' '.join(pair)+' %.5f\n'%cosine_score)
                scores.append(cosine_score)
                if pair[2] == '1' or pair[2] == 'target':
                    labels.append(1)
                elif pair[2] == '0' or pair[2] == 'nontarget':
                    labels.append(0)
                else:
                    raise Exception(f'Unrecognized label in {line}.')

        # compute metrics
        scores = np.array(scores)
        print(f"Socre shape is {scores.shape}")
        labels = np.array(labels)
        print(f"Label shape is {labels.shape}")
        

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)
        min_dcf = compute_c_norm(fnr,
                                fpr,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        th_matrix_result = compute_tn_fn_tp_fp(scores, labels)
        
        # write the metrics
        logger.info(f"Results of {trial_name} is:")
        logger.info("\t\tEER = {0:.4f}".format(100 * eer))
        logger.info("\t\tminDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf))
        for _info in th_matrix_result:
            logger.info("\t\tTH:{0:.2f}\tTP:{1:.4f}\tFP:{2:.4f}\tTN:{3:.4f}\tFN:{4:.4f}".format(*_info))
        # logger_all.info(f"{args.exp_id},{100 * eer:.4f},{min_dcf:.4f}")
        # append to args.scores_all
        # if args.scores_all not exist, create it
        if not os.path.exists(args.scores_all):
            with open(args.scores_all, 'w') as f:
                f.write("model,trails,time,trial_name,EER,minDCF\n")
        with open(args.scores_all, 'a') as f:
            f.write(f"{args.exp_id},{trial_name},{100 * eer:.4f},{min_dcf:.4f}\n")



if __name__ == "__main__":
    main()
