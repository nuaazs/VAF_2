# coding = utf-8
# @Time    : 2023-04-03  07:09:47
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 多进程给每个命中的记录添加content.

import os
import sys
import pandas as pd
import numpy as np
import wget
import argparse
from rrr import get_rrr_content
from rrr_filter_plus import check_text
parser = argparse.ArgumentParser(description='')
parser.add_argument('--total_workers', type=int, default=1,help='')
parser.add_argument('--worker_index', type=int, default=0,help='start from 0')
args = parser.parse_args()

# df = pd.DataFrame(all_data,columns=["phone","file_url","blackbase_phone","valid_length","hit_score","mean_score"])
# df.to_csv("hit_data.csv",index=False)

if __name__ == '__main__':
    # Read hit_data.csv
    df = pd.read_csv("hit_data.csv")
    # sort by hit_score
    df = df.sort_values(by="hit_score",ascending=False)
    # choosed worker_index * each_worker_num
    each_worker_num = len(df) // args.total_workers
    choosed_range = range(args.worker_index * each_worker_num, (args.worker_index + 1) * each_worker_num)
    return_data = []
    for i in choosed_range:
        data = df.iloc[i]
        file_url = data["file_url"]
        phone = data["phone"]
        blackbase_phone = data["blackbase_phone"]
        hit_score = data["hit_score"]
        valid_length = data["valid_length"]
        mean_score = data["mean_score"]
        # get text
        # text,is_hit_key,keys_text = get_rrr_content(file_url,spkid="zhaosheng")
        text = data["content_text"]
        hit_time = "" #data["hit_time"]

        level_info,level,all_keys_text = check_text(text)
        if hit_score>0.78 and ((level_info == "高" and level=="high") or (hit_score>0.80 and level=="mid")):# 
            return_data.append([phone,file_url,blackbase_phone,valid_length,hit_score,mean_score,text,level,level_info,all_keys_text,hit_time])
    return_df = pd.DataFrame(return_data,columns=["phone","file_url","blackbase_phone","valid_length","hit_score","mean_score","content","level","level_info","all_keys_text","hit_time"])
    os.makedirs("./csvs",exist_ok=True)
    return_df.to_csv("./csvs/hit_data_{}.csv".format(args.worker_index),index=False)
    print("worker_index: {} done!".format(args.worker_index))
