

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
from sklearn.linear_model import LogisticRegression
from speakerlabduanyibo.utils.utils import get_logger
from speakerlabduanyibo.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Compute score and metrics')
parser.add_argument('--enrol_data', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data', default='', type=str, help='Test data dir')
parser.add_argument('--enrol_data2', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data2', default='', type=str, help='Test data dir')
parser.add_argument('--enrol_data3', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data3', default='', type=str, help='Test data dir')
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
    ##############################
    #train_enrol_dict = collect("/home/duanyibo/vaf/test/result_9_1/repvgg/cti1_result/embeddings")
    #train_test_dict = collect("/home/duanyibo/vaf/test/result_9_1/repvgg/cti1_result/embeddings")
    #train_enrol_dict2 = collect("/home/duanyibo/vaf/test/result_9_1/eres2net/cti1_result/embeddings")
    #train_test_dict2 = collect("/home/duanyibo/vaf/test/result_9_1/eres2net/cti1_result/embeddings")
    #train_enrol_dict3 = collect("/home/duanyibo/vaf/test/result_9_1/dfresnet_233/cti1_result/embeddings")
    # train_test_dict3 = collect("/home/duanyibo/vaf/test/result_9_1/dfresnet_233/cti1_result/embeddings")
    # train_trials="/home/duanyibo/vaf/test/dataset/cti_test/cti.trials"
    train_enrol_dict = collect(args.enrol_data)
    train_test_dict = collect(args.enrol_data3)
    train_enrol_dict2 = collect(args.enrol_data2)
    train_test_dict2 = collect(args.enrol_data2)
    train_enrol_dict3 = collect(args.enrol_data3)
    train_test_dict3 = collect(args.enrol_data3)
    train_trials= args.trials[0]
    train_scores = []
    train_labels = []
    train_trial_name = os.path.basename(train_trials)
    train_score_path = os.path.join(args.scores_dir, f'{train_trial_name}.score')
    with open(train_trials, 'r') as train_trial_f, open(train_score_path, 'w') as train_score_f:
        train_lines = train_trial_f.readlines()
        for line in tqdm(train_lines, desc=f'scoring trial {train_trial_name}'):
            train_pair = line.strip().split()
            train_enrol_emb, train_test_emb = train_enrol_dict[train_pair[0]], train_test_dict[train_pair[1]]
            train_cosine_score1 = cosine_similarity(train_enrol_emb.reshape(1, -1),
                                            train_test_emb.reshape(1, -1))[0][0]

            train_enrol_emb2, train_test_emb2 = train_enrol_dict2[train_pair[0]], train_test_dict2[train_pair[1]]
            train_cosine_score2 = cosine_similarity(train_enrol_emb2.reshape(1, -1),
                                            train_test_emb2.reshape(1, -1))[0][0]
                                    
            train_enrol_emb3, train_test_emb3 = train_enrol_dict3[train_pair[0]], train_test_dict3[train_pair[1]]
            train_cosine_score3 = cosine_similarity(train_enrol_emb3.reshape(1, -1),
                                            train_test_emb3.reshape(1, -1))[0][0]
            train_scores.append([train_cosine_score1, train_cosine_score2, train_cosine_score3])
            # train_cosine_score=(train_cosine_score3+train_cosine_score1+train_)/2
            # print(f"cosine_score1:{cosine_score1},cosine_score2:{cosine_score2},cosine_score1:{cosine_score2} ,cosine_score:{cosine_score}")
            # write the score
            if train_pair[2] == '1' or train_pair[2] == 'target':
                train_labels.append(1)
            elif train_pair[2] == '0' or train_pair[2] == 'nontarget':
                train_labels.append(0)
            else:
                raise Exception(f'Unrecognized label in {line}.')

    # compute metrics
    train_scores = np.array(train_scores)
    train_labels = np.array(train_labels)
    lr = LogisticRegression()
    lr.fit(train_scores, train_labels)
    ##################################
    enrol_dict = collect(args.enrol_data)
    test_dict = collect(args.test_data)
    
    enrol_dict2 = collect(args.enrol_data2)
    test_dict2 = collect(args.enrol_data2)
    
    enrol_dict3 = collect(args.enrol_data3)
    test_dict3 = collect(args.enrol_data3)
    # trail_list = args.trials.split(",")
    for trial in args.trials:
        scores = []
        labels = []
        rep_scores=[]
        eres2net_scores=[]
        dfrenet_scores=[]
        avg_scores=[]
        trial_name = os.path.basename(trial)
        score_path = os.path.join(args.scores_dir, f'{trial_name}.score')
        with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
            lines = trial_f.readlines()
            for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                pair = line.strip().split()
                enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                cosine_score1 = cosine_similarity(enrol_emb.reshape(1, -1),
                                              test_emb.reshape(1, -1))[0][0]
                rep_scores.append(cosine_score1)
                enrol_emb2, test_emb2 = enrol_dict2[pair[0]], test_dict2[pair[1]]
                cosine_score2 = cosine_similarity(enrol_emb2.reshape(1, -1),
                                              test_emb2.reshape(1, -1))[0][0]
                eres2net_scores.append(cosine_score2)                        
                enrol_emb3, test_emb3 = enrol_dict3[pair[0]], test_dict3[pair[1]]
                cosine_score3 = cosine_similarity(enrol_emb3.reshape(1, -1),
                                              test_emb3.reshape(1, -1))[0][0]
                dfrenet_scores.append(cosine_score3)
                avg_scores.append(float((cosine_score1 + cosine_score2 + cosine_score3)/3))
                x = np.array([cosine_score1, cosine_score2, cosine_score3]).reshape(1,-1)
                ensemble_score = lr.predict_proba(x)[:,1]
                # cosine_score=(cosine_score3+cosine_score1)/2
                # print(f"cosine_score1:{cosine_score1},cosine_score2:{cosine_score2},cosine_score1:{cosine_score2} ,cosine_score:{cosine_score}")
                # write the score
                score_f.write(' '.join(pair)+' %.5f\n'%ensemble_score[0])
                scores.append(ensemble_score[0]) 
                if pair[2] == '1' or pair[2] == 'target':
                    labels.append(1)
                elif pair[2] == '0' or pair[2] == 'nontarget':
                    labels.append(0)
                else:
                    raise Exception(f'Unrecognized label in {line}.')

        # compute metrics
        scores = np.array(scores)
        rep_scores=np.array(rep_scores)
        eres2net_scores = np.array(eres2net_scores)
        dfrenet_scores = np.array(dfrenet_scores)
        avg_scores = np.array(avg_scores)
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

        fnr1, fpr1 = compute_pmiss_pfa_rbst(rep_scores, labels)
        eer1, thres1 = compute_eer(fnr1, fpr1, rep_scores)
        min_dcf1 = compute_c_norm(fnr1,
                                fpr1,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        min_dcf_noc1 = compute_c_norm(fnr1,
                                    fpr1,
                                    p_target=0.000005,
                                    c_miss=1,
                                    c_fa=5)
                # write the metrics
        logger.info("Results_repvgg of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer1))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf1))
        logger.info("minDCF_noc (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            0.000005, 1, 5, min_dcf_noc1))

        fnr2, fpr2 = compute_pmiss_pfa_rbst(eres2net_scores, labels)
        eer2, thres2 = compute_eer(fnr2, fpr2, eres2net_scores)
        min_dcf2 = compute_c_norm(fnr2,
                                fpr2,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        min_dcf_noc2 = compute_c_norm(fnr2,
                                    fpr2,
                                    p_target=0.000005,
                                    c_miss=1,
                                    c_fa=5)
                # write the metrics
        logger.info("Results_eres2net of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer2))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf2))
        logger.info("minDCF_noc (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            0.000005, 1, 5, min_dcf_noc2))

        fnr3, fpr3 = compute_pmiss_pfa_rbst(dfrenet_scores, labels)
        eer3, thres3 = compute_eer(fnr3, fpr3, dfrenet_scores)
        min_dcf3 = compute_c_norm(fnr3,
                                fpr3,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        min_dcf_noc3 = compute_c_norm(fnr3,
                                    fpr3,
                                    p_target=0.000005,
                                    c_miss=1,
                                    c_fa=5)
                # write the metrics
        logger.info("Results_dfresnet of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer3))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf3))
        logger.info("minDCF_noc (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            0.000005, 1, 5, min_dcf_noc3))


        fnr4, fpr4 = compute_pmiss_pfa_rbst(avg_scores, labels)
        eer4, thres4 = compute_eer(fnr4, fpr4, avg_scores)
        min_dcf4 = compute_c_norm(fnr4,
                                fpr4,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
        min_dcf_noc4 = compute_c_norm(fnr4,
                                    fpr4,
                                    p_target=0.000005,
                                    c_miss=1,
                                    c_fa=5)
                # write the metrics
        logger.info("Results_avf_3 of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer4))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf4))
        logger.info("minDCF_noc (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            0.000005, 1, 5, min_dcf_noc4))


if __name__ == "__main__":
    main()
    