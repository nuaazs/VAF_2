# -*- coding: utf-8 -*-
'''
@author: Carry
@contact: xkx94317@gmail.com
@file: main.py
@time: 2019/5/17 11:56 
@desc: 
'''
import glob
import os.path

from tqdm import tqdm
import numpy as np
import os

# with open('/home/xz/zhaosheng/embeddings/cjsd/all.txt', 'r') as f:
#     phone_id = [i.strip().split('_')[0] for i in f.readlines()]
#
# print(len(phone_id))
# print(len(set(phone_id)))

li_1 = ['TP', 'TN', 'FP', 'FN']
output_path = "data/input_a/vector_a_all_split_data/all_results"
os.makedirs(output_path,exist_ok=True)
for i in tqdm(li_1):
    for j in np.arange(0.40, 1.01, 0.01):
        j = "{:.2f}".format(j)
        print(f"{i}_{j}")
        all_ = glob.glob(f"data/input_a/vector_a_all_split_data/*/{i}_{j}.txt")
        with open(f'{output_path}/{i}_{j}.txt', 'w') as f:
            for k in all_:
                with open(f'{k}', 'r') as f2:
                    for line in f2.readlines():
                        f.write(line)
