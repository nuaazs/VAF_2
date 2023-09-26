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

parser.add_argument('--total', default=1, type=float, help='total')
parser.add_argument('--rank', default=0, type=float, help='rank')
parser.add_argument('--tiny_save_dir', default='', type=str, help='')

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


        # find all score_*.npy in tiny_save_dir/trial_name
        tiny_save_dir = args.tiny_save_dir
        score_files = sorted([os.path.join(tiny_save_dir,trial_name,i) for i in os.listdir(os.path.join(tiny_save_dir,trial_name)) if i.startswith('score_')])
        label_files = sorted([os.path.join(tiny_save_dir,trial_name,i) for i in os.listdir(os.path.join(tiny_save_dir,trial_name)) if i.startswith('label_')])
        # load scores and merge them
        for score_file in score_files:
            score = np.load(score_file)
            scores.append(score)
        scores = np.concatenate(scores,axis=0)
        scores=scores.reshape(-1)
        # tolist
        # scores = scores.tolist()
        print(f"Shape of scores is {scores.shape}")
        # load labels and merge them
        for label_file in label_files:
            label = np.load(label_file)
            labels.append(label)
        labels = np.concatenate(labels,axis=0)
        labels=labels.reshape(-1)
        print(f"Shape of labels is {labels.shape}")


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
            th,tp,fp,tn,fn = _info
            try:
                precision = (tp/(tp+fp)) *100
                recall = (tp/(tp+fn)) *100
                acc = ((tp+tn)/(tp+fn+tn+fp)) *100
            except:
                precision = " - "
                recall = " - "
            logger.info(f"\t\tTH:{th:.2f}\tTP:{tp:.4f}\tFP:{fp:.4f}\tTN:{tn:.4f}\tFN:{fn:.4f}\tP:{precision:.2f}\tR:{recall:.2f}\tACC:{acc:.2f}")
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
