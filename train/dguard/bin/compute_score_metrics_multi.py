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
parser.add_argument('--emb_size', default=256, type=float, help='emb_size')
parser.add_argument('--rank', default=0, type=float, help='rank')
parser.add_argument('--tiny_save_dir', default='', type=str, help='')


def get_mean(list2d):
    for _index,_result in enumerate(list2d):
        # print(f"Index:{_index},Length:{len(_result)}")
        # print(_result)
        length =len(_result)
        if _index == 0:
            _result_data = [i[1] for i in _result]
            final_result=np.array(_result_data)
        else:
            _result_data = [i[1] for i in _result]
            final_result+=np.array(_result_data)
    return final_result/len(list2d)
def main():
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.scores_dir, exist_ok=True)

    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt = "%(message)s")
    # logger_all = get_logger(fpath=, fmt = "%(message)s")
    def collect(data_dir,prefix=None):
        data_dict = {}
        if prefix:
            emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$',i) and i.startswith(prefix)]
        else:
            emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$',i)]
        if len(emb_arks) == 0:
            # print(f"No embedding ark files found in {data_dir} with prefix {prefix}.")
            raise Exception(f'No embedding ark files found in {data_dir} with prefix {prefix}.')

        # load embedding data
        for ark in emb_arks:
            with ReadHelper(f'ark:{ark}') as reader:
                for key, array in reader:
                    data_dict[key] = array

        return data_dict
    # Get all models
    enrol_data_subfiles = [_file for _file in os.listdir(args.enrol_data) if ".ark" in _file]
    # Check if "fusion_<model>_*" exists in enrol_data_subfiles
    fusion_models = [i for i in enrol_data_subfiles if re.search('^fusion_.*',i)]
    # Get all fusion model names
    fusion_model_names = set([name.split("_")[1] for name in fusion_models])
    if len(fusion_model_names) == 0:
        # print(f"No fusion model found in {args.enrol_data}.")
        enrol_dict = collect(args.enrol_data)
        test_dict = collect(args.test_data)

        for trial in args.trials:
            scores = []
            labels = []

            trial_name = os.path.basename(trial)
            score_path = os.path.join(args.scores_dir, f'{args.rank}_{trial_name}.score')
            with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
                lines = trial_f.readlines()
                # lines_sorted = sorted(lines, key=lambda x: x.split()[0])
                if args.total != args.rank:
                    tiny_length = len(lines)//args.total
                    lines = lines[int(args.rank*tiny_length):int((args.rank+1)*tiny_length)]
                    # print(f"Line -> Line[{args.rank*tiny_length}:{(args.rank+1)*tiny_length}]")
                else:
                    lines = lines[int(args.rank*len(lines)//args.total):]
                    # print(f"Line -> Line[{args.rank*len(lines)//args.total}:]")

                for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                    pair = line.strip().split()
                    enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                    # assert enrol_emb.shape[0] == args.emb_size*3, f"Enrol emb shape is {enrol_emb.shape}, not {args.emb_size*3}"
                    args.emb_size = int(enrol_emb.shape[0]/3)
                    # TODO: a,b,full
                    enrol_emb_a = enrol_emb[:args.emb_size]
                    enrol_emb_b = enrol_emb[args.emb_size:args.emb_size*2]
                    enrol_emb_full = enrol_emb[args.emb_size*2:args.emb_size*3]

                    test_emb_a = test_emb[:args.emb_size]
                    test_emb_b = test_emb[args.emb_size:args.emb_size*2]
                    test_emb_full = test_emb[args.emb_size*2:args.emb_size*3]


                    cos_enrol_a_b = cosine_similarity(enrol_emb_a.reshape(1, -1),
                                                enrol_emb_b.reshape(1, -1))[0][0]
                    cos_test_a_b = cosine_similarity(test_emb_a.reshape(1, -1),
                                                test_emb_b.reshape(1, -1))[0][0]
                    mean_self_score = (cos_enrol_a_b+cos_test_a_b)/2
                    radio = 0.7/((cos_enrol_a_b + cos_enrol_a_b)/2)
                    # print(f"="*30)
                    # print(f"Radio1 :{0.7/cos_enrol_a_b} Radio2:{0.7/cos_test_a_b}")
                    # if radio > 3:
                    #     radio = 3
                    # if radio < 0.33:
                    #     radio = 0.33
                    
                    raw_cosine_score = cosine_similarity(enrol_emb_full.reshape(1, -1),
                                                test_emb_full.reshape(1, -1))[0][0]

                    # with cmf
                    cosine_score = raw_cosine_score # *radio
                    print(f"Raw Score: {raw_cosine_score}, New Score: {cosine_score}")

                    # without cmf
                    # cosine_score = raw_cosine_score


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
            labels = np.array(labels)
            # save scores and labels
            os.makedirs(os.path.join(args.tiny_save_dir,trial_name),exist_ok=True)
            score_save_path = os.path.join(args.tiny_save_dir,trial_name,f'score_{args.rank}.npy')
            label_save_path = os.path.join(args.tiny_save_dir,trial_name,f'label_{args.rank}.npy')
            np.save(score_save_path,scores)
            np.save(label_save_path,labels)

    else:
        # print(f"All fusion models found in {args.enrol_data}.")
        # print(fusion_model_names)
        scores_all_info = {}
        labels_all_info = {}
        for model_name in fusion_model_names:
            scores_all_info[model_name] = {}
            labels_all_info[model_name] = {}
            enrol_dict = collect(args.enrol_data,prefix=f"fusion_{model_name}_")
            test_dict = collect(args.test_data,prefix=f"fusion_{model_name}_")
            for trial in args.trials:
                
                scores = []
                labels = []

                trial_name = os.path.basename(trial)
                scores_all_info[model_name][trial_name] = {}
                labels_all_info[model_name][trial_name] = {}
                score_path = os.path.join(args.scores_dir, f'{args.rank}_{trial_name}.score')
                with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
                    lines = trial_f.readlines()
                    # lines_sorted = sorted(lines, key=lambda x: x.split()[0])
                    if args.total != args.rank:
                        tiny_length = len(lines)//args.total
                        lines = lines[int(args.rank*tiny_length):int((args.rank+1)*tiny_length)]
                        # print(f"Line -> Line[{args.rank*tiny_length}:{(args.rank+1)*tiny_length}]")
                    else:
                        lines = lines[int(args.rank*len(lines)//args.total):]
                        # print(f"Line -> Line[{args.rank*len(lines)//args.total}:]")

                    for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                        pair = line.strip().split()
                        if pair[0] == pair[1]:
                            continue
                        enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                        cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                                    test_emb.reshape(1, -1))[0][0]
                        # write the score
                        score_f.write(' '.join(pair)+' %.5f\n'%cosine_score)
                        scores.append([pair[0]+"_"+pair[1],cosine_score])
                        if pair[2] == '1' or pair[2] == 'target':
                            labels.append([pair[0]+"_"+pair[1],1])
                        elif pair[2] == '0' or pair[2] == 'nontarget':
                            labels.append([pair[0]+"_"+pair[1],0])
                        else:
                            raise Exception(f'Unrecognized label in {line}.')
                scores_sorted = sorted(scores, key=lambda x: x[0])
                labels_sorted = sorted(labels, key=lambda x: x[0])
                scores_all_info[model_name][trial_name]["scores_sorted"] = scores_sorted
                labels_all_info[model_name][trial_name]["labels_sorted"] = labels_sorted
        
        for trial_name in args.trials:
            trial_name = os.path.basename(trial_name)
            scores_mean = []
            label_mean = []
            for model_name in scores_all_info.keys():
                scores = scores_all_info[model_name][trial_name]["scores_sorted"]
                labels = labels_all_info[model_name][trial_name]["labels_sorted"]
                scores_mean.append(scores)
                label_mean.append(labels)
            

            scores_mean = get_mean(scores_mean)
            label_mean = get_mean(label_mean)
            # assert label_mean == labels

            # compute metrics
            scores = np.array(scores_mean)
            labels = np.array(label_mean)
            # print(labels)
            # save scores and labels
            os.makedirs(os.path.join(args.tiny_save_dir,trial_name),exist_ok=True)
            score_save_path = os.path.join(args.tiny_save_dir,trial_name,f'score_{args.rank}.npy')
            label_save_path = os.path.join(args.tiny_save_dir,trial_name,f'label_{args.rank}.npy')
            np.save(score_save_path,scores)
            np.save(label_save_path,labels)

if __name__ == "__main__":
    main()
