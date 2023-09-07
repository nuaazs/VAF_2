# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper
from sklearn.metrics.pairwise import cosine_similarity

from speakerlabduanyibo.utils.utils import get_logger
from speakerlabduanyibo.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Compute score and metrics')
parser.add_argument('--enrol_data', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data', default='', type=str, help='Test data dir')
parser.add_argument('--scores_dir', default='', type=str, help='Scores dir')
parser.add_argument('--trials', nargs='+', help='Trial')
parser.add_argument('--p_target', default=0.01, type=float, help='p_target in DCF')
parser.add_argument('--c_miss', default=1, type=float, help='c_miss in DCF')
parser.add_argument('--c_fa', default=1, type=float, help='c_fa in DCF')

def main():
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.scores_dir, exist_ok=True)

    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt = "%(message)s")

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
    cmf_enrol_dict = collect(os.path.join(args.enrol_data.split("/embeddings")[0],"cmf_embeddings"))
    cmf_test_dict = collect(os.path.join(args.test_data.split("/embeddings")[0],"cmf_embeddings"))
    # trail_list = args.trials.split(",")
    for trial in args.trials:
        scores = []
        cmf_scores =[]
        labels = []

        trial_name = os.path.basename(trial)
        score_path = os.path.join(args.scores_dir, f'{trial_name}.score')
        cmf_score_path = os.path.join(args.scores_dir, f'cmf_{trial_name}.score')
        with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f,open(cmf_score_path, 'w') as cmf_score_f:
            lines = trial_f.readlines()
            for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                pair = line.strip().split()
                enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                              test_emb.reshape(1, -1))[0][0]
                cmf0,cmf1=cmf_enrol_dict[pair[0]][0], cmf_test_dict[pair[1]][0]
                print(cmf0,cmf1)
                factor =cmf0*cmf1
                # print(factor)
                cmf_score = factor*cosine_score
                # cmf_factor = infer.calculate_factor(cmf0,cmf1)
                # write the score
                score_f.write(' '.join(pair)+' %.5f\n'%cosine_score)
                cmf_score_f.write(' '.join(pair)+' %.5f\n'%cmf_score)
                cmf_scores.append(cmf_score)
                scores.append(cosine_score)
                if pair[2] == '1' or pair[2] == 'target':
                    labels.append(1)
                elif pair[2] == '0' or pair[2] == 'nontarget':
                    labels.append(0)
                else:
                    raise Exception(f'Unrecognized label in {line}.')

        # compute metrics
        scores = np.array(scores)
        cmf_scores = np.array(cmf_scores)
        labels = np.array(labels)

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)
        min_dcf = compute_c_norm(fnr,
                                fpr,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        min_dcf_noc = compute_c_norm(fnr,
                                    fpr,
                                    p_target=0.000005,
                                    c_miss=1,
                                    c_fa=5)
        # write the metrics
        logger.info("Results of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf))
        logger.info("minDCF_noc (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            0.000005, 1, 5, min_dcf_noc))

        cmf_fnr, cmf_fpr = compute_pmiss_pfa_rbst(cmf_scores, labels)
        cmf_eer, cmf_thres = compute_eer(cmf_fnr, cmf_fpr, cmf_scores)
        cmf_min_dcf = compute_c_norm(cmf_fnr,
                                cmf_fpr,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        cmf_min_dcf_noc = compute_c_norm(cmf_fnr,
                                    cmf_fpr,
                                    p_target=0.000005,
                                    c_miss=1,
                                    c_fa=5)
        # write the metrics
        logger.info("CMF_Results of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * cmf_eer))
        logger.info("CMF_minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, cmf_min_dcf))
        logger.info("CMF_NminDCF_noc (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            0.000005, 1, 5, cmf_min_dcf_noc))


if __name__ == "__main__":
    main()
    