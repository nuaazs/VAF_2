# coding = utf-8
# @Time    : 2023-10-27  14:56:40
# @Author  : zhaosheng@lyxxkj.com.cn
# @Describe: SV-Task data-4 test script.

import argparse
import numpy as np
from tqdm import tqdm
import logging

# PUT ID MAP HERE
id_map = {}

parser = argparse.ArgumentParser(description='')
parser.add_argument('--th_start', default=0.01, type=float, help='TH Start')
parser.add_argument('--th_end', default=1.00, type=float, help='TH End')
parser.add_argument('--th_step', default=0.01, type=float, help='TH Step')
# load npy, store true
parser.add_argument('--register_npy_path', default="./cache/3s.npy", type=str, help='npy path')
parser.add_argument('--test_npy_path', default="./cache/test.npy", type=str, help='npy path')
parser.add_argument('--result', default="./3s.txt", type=str, help='Result save path')
args = parser.parse_args()

# Init LOG
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("./test.log",mode='a')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

result_logger = logging.getLogger("result")
result_logger.setLevel(level = logging.INFO)
result_handler = logging.FileHandler(args.result,mode='a')
result_handler.setLevel(logging.INFO)
result_formatter = logging.Formatter('%(message)s')
result_handler.setFormatter(result_formatter)
result_logger.addHandler(result_handler)


if __name__ == '__main__':
    register_npy_save_path = args.register_npy_path
    test_npy_save_path = args.test_npy_path
    registered_ids = np.load(register_npy_save_path,allow_pickle=True)
    search_result_dict = np.load(test_npy_save_path,allow_pickle=True)

    TN,TP,FN,FP = 0,0,0,0
    register_id_count = len(registered_ids)
    test_id_count = len(search_result_dict.keys())

    for th in np.arange(args.th_start,args.th_end,args.th_step):
        logger.info(f"TH: {th}")
        for now_spkid in search_result_dict.keys():
            if now_spkid in id_map.keys():
                now_spkid = id_map[now_spkid]
            now_result = search_result_dict[now_spkid]["top_10"]
            top1_id = now_result[0][0]
            if top1_id in id_map.keys():
                top1_id = id_map[top1_id]
            top1_score = now_result[0][1]

            if top1_score >= th:
                if top1_id == now_spkid:
                    TP += 1
                else:
                    FP += 1
                    logger.error(f"* FP: {now_spkid} -- {top1_id}")
                    logger.error(f"\t {now_result}")
            else:
                if top1_id == now_spkid:
                    FN += 1
                    logger.error(f"* FN: {now_spkid} -- {top1_id}")
                    logger.error(f"\t {now_result}")
                else:
                    if top1_id in registered_ids:
                        FN += 1
                        logger.error(f"* FN: {now_spkid} -- {top1_id}")
                        logger.error(f"\t {now_result}")
                    else:
                        TN += 1
        precision = TP/(TP+FP+1e-5)
        recall = TP/(TP+FN+1e-5)
        accuracy = (TP+TN)/(TP+TN+FP+FN+1e-5)
        far = FP/(FP+TN+1e-5)
        frr = FN/(FN+TP+1e-5)
        result_logger.info("register,test,th,precision,recall,accuracy,far,frr")
        result_logger.info(f"{register_id_count},{test_id_count},{th},{precision},{recall},{accuracy},{far},{frr}")