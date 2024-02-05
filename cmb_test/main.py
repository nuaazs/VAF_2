# coding = utf-8
# @Time    : 2023-10-24  14:56:40
# @Author  : zhaosheng@lyxxkj.com.cn
# @Describe: SV-Task test script.

import os
import yaml
import argparse
import numpy as np
from tqdm import tqdm

# Utils
from utils.trails import read_trails
from utils.scores import calc_result
from utils.logger_wrapper import init_logger
from utils.files import find_wav_files
from utils.compare import compare_res,compare_trial,collision
from utils.encode import get_speaker_embedding,get_speaker_embedding_list
from utils.encode import register,search
# Init LOG
import logging

parser = argparse.ArgumentParser(description='')

# Config
parser.add_argument('--yaml', type=str, default='./conf/test_1_1.yaml',help='')

# DCF
parser.add_argument('--p_target', default=0.01, type=float, help='p_target in DCF')
parser.add_argument('--c_miss', default=1, type=float, help='c_miss in DCF')
parser.add_argument('--c_fa', default=1, type=float, help='c_fa in DCF')
parser.add_argument('--min_recall', default=-1, type=float, help='min recall to find TH')
parser.add_argument('--min_precision', default=-1, type=float, help='min precision to find TH')

# load npy, store true
parser.add_argument('--load_npy', action='store_true', help='load data from npy file')

args = parser.parse_args()

if __name__ == '__main__':

    # Read config
    conf = yaml.load(open(args.yaml,'r'),Loader=yaml.FullLoader)
    count = 0
    test_type = conf['test_type']
    spkid_location = conf['spkid_location']
    exp_name = conf['exp_name']
    logger = init_logger(log_path=os.path.join("./log",f"{exp_name}.log"))

    if test_type=="one2one":
        # 1:1 MODE
        trails = read_trails(conf['trails_path'],spkid_location=spkid_location)
        all_labels,all_scores = [],[]
        for i in tqdm(range(len(trails))):
            tiny_data = trails.iloc[i]
            cmf = conf['cmf']
            if cmf == 0 or cmf == "False" or cmf == "0" or cmf == "false":
                cmf = False
            if cmf:
                logger.info("Using CMF")
            count += 1
            try:
                res = compare_trial(trial=tiny_data,url=conf['encode_url'],wav_length=conf['wav_length'],cmf=cmf)
            except Exception as e:
                logger.error(f"Error in {i}th trial: {e}")
                continue
            label = tiny_data['label']
            all_scores.append(res)
            all_labels.append(label)
            if i >= 100:
                break
        calc_result(all_scores,all_labels,p_target=args.p_target,c_miss=args.c_miss,c_fa=args.c_fa,min_recall=args.min_recall,min_precision=args.min_precision)
        logger.info(f"* Total #{count} trials.")
        logger.info(f"* Success radio: {len(all_scores)/count:.2f}")


    elif test_type=="one2N":
        # 1:N
        if conf['mode'] == "api":
            register_npy_save_path = os.path.join("cache",f"{exp_name}_1_N_register_api.npy")
            test_npy_save_path = os.path.join("cache",f"{exp_name}_1_N_test_api.npy")

            if args.load_npy and os.path.exists(register_npy_save_path):
                logger.info(f"Load data from {register_npy_save_path}")
                registered_ids = np.load(register_npy_save_path,allow_pickle=True)
            if args.load_npy and os.path.exists(test_npy_save_path):
                logger.info(f"Load data from {test_npy_save_path}")
                search_result_dict = np.load(test_npy_save_path,allow_pickle=True)
            else:
                register_wavs = find_wav_files(root=conf['register_wavs_path'],pattern=conf['register_wav_pattern'])
                test_wavs = find_wav_files(root=conf['test_wavs_path'],pattern=conf['test_wav_pattern'])
                success_count, registered_ids, registered_spks = register(audio_list=register_wavs,url=conf['register_url'],wav_length=conf['wav_length'],spkid_location=spkid_location)
                success_count, searched_ids, searched_spks, search_result_dict = search(audio_list=test_wavs,url=conf['search_url'],wav_length=conf['wav_length'],spkid_location=spkid_location)
                # save search_result
                np.save(test_npy_save_path,search_result_dict)
                # save registered_ids
                np.save(register_npy_save_path,registered_ids)

            TN,TP,FN,FP = 0,0,0,0
            th_start,th_end,th_step = conf['threshold'].split(',')
            th_start,th_end,th_step = float(th_start),float(th_end),float(th_step)
            for th in np.arange(th_start,th_end,th_step):
                logger.info(f"TH: {th}")

                for now_spkid in search_result_dict.keys():
                    real_spkid = now_spkid.split("-")[0]
                    now_result = search_result_dict[now_spkid]["top_10"]
                    top1_id = now_result[0][0].split("-")[0]
                    top1_score = now_result[0][1]
                    if top1_score >= th:
                        if top1_id == real_spkid:
                            TP += 1
                        else:
                            FP += 1
                            logger.error(f"* FP: {real_spkid} vs {top1_id}")
                            logger.error(f"\t {now_result}")
                    else:
                        if top1_id == real_spkid:
                            FN += 1
                            logger.error(f"* FN: {real_spkid} vs {top1_id}")
                            logger.error(f"\t {now_result}")
                        else:
                            if top1_id in registered_ids:
                                FN += 1
                                logger.error(f"* FN: {real_spkid} vs {top1_id}")
                                logger.error(f"\t {now_result}")
                            else:
                                TN += 1
                precision = TP/(TP+FP+1e-5)
                recall = TP/(TP+FN+1e-5)
                accuracy = (TP+TN)/(TP+TN+FP+FN+1e-5)
                logger.info(f"Collision precision is {precision}")
                logger.info(f"Collision recall is {recall}")
                logger.info(f"Collision accuracy is {accuracy}")

            
        else:
            register_npy_save_path = os.path.join("cache",f"{exp_name}_1_N_register.npy")
            test_npy_save_path = os.path.join("cache",f"{exp_name}_1_N_test.npy")
            if args.load_npy and os.path.exists(register_npy_save_path):
                logger.info(f"Load data from {register_npy_save_path}")
                all_register_data = np.load(register_npy_save_path,allow_pickle=True)
            else:
                register_wavs = find_wav_files(root=conf['register_wavs_path'],pattern=conf['register_wav_pattern'])
                all_register_data = get_speaker_embedding_list(audio_list=register_wavs,url=conf['encode_url'],wav_length=conf['wav_length'],spkid_location=spkid_location)
                np.save(register_npy_save_path,all_register_data)
            if args.load_npy and os.path.exists(test_npy_save_path):
                logger.info(f"Load data from {test_npy_save_path}")
                all_test_data = np.load(test_npy_save_path,allow_pickle=True)
            else:
                test_wavs = find_wav_files(root=conf['test_wavs_path'],pattern=conf['test_wav_pattern'])
                all_test_data = get_speaker_embedding_list(audio_list=test_wavs,url=conf['encode_url'],wav_length=conf['wav_length'],spkid_location=spkid_location)
                np.save(test_npy_save_path,all_test_data)
            cmf = conf['cmf']
            if cmf == 0 or cmf == "False" or cmf == "0" or cmf == "false":
                cmf = False
            if cmf:
                logger.info("Using CMF")
            th_start,th_end,th_step = conf['threshold'].split(',')
            th_start,th_end,th_step = float(th_start),float(th_end),float(th_step)
            for th in np.arange(th_start,th_end,th_step):
                logger.info(f"TH: {th}")
                collision_result = collision(register_data=all_register_data,test_data=all_test_data,topk=conf['topk'],th=th,test_num=300,cmf=cmf,logger=logger)
                logger.info(f"Collision: {collision_result}")
    else:
        raise ValueError(f"test_type {test_type} is not supported!")