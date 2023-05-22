from sql import MysqlWrapper

from asr import get_asr_content
# from nlp import get_data

import pandas as pd
import numpy as np
import wget
import os
# import matplotlib.pyplot as plt

mysql = MysqlWrapper(host="192.168.3.201", port=3306, user="root", passwd="longyuan", db="si")

def get_hit_score_distribution(df):
    # get hit_score distribution
    hit_score = df["hit_score"].values
    hit_score = np.array(hit_score)
    print(hit_score)
    hit_score = hit_score.astype(np.float)
    print("hit_score distribution:")
    print("max:",hit_score.max())
    print("min:",hit_score.min())
    print("mean:",hit_score.mean())
    print("std:",hit_score.std())
    print("median:",np.median(hit_score))
    print("var:",np.var(hit_score))
    print("percentile 90:",np.percentile(hit_score,90))
    # print hit_score distribution
    for i in range(10):
        print(f"percentile {i*10}:",np.percentile(hit_score,i*10))

    # plot distribution
    # x scores
    # # y num
    # plt.hist(hit_score,bins=100)
    # plt.savefig("hit_score_distribution.png")

def dowanload_data(df,score_th=0.5):
    # download df["file_url"] and name it as f"{blackbase_phone}_{phone}_{hit_score}.wav" in the folder "download_data/{blackbase_phone}"
    for i in range(len(df)):
        data = df.iloc[i]
        phone = data["phone"]
        file_url = data["file_url"]
        blackbase_phone = data["blackbase_phone"]
        blackbase_phone_top = blackbase_phone.split(",")[0]
        hit_score = float(data["hit_score"])
        # print(hit_score)
        if hit_score < score_th:
            continue
        # download file_url and name it as f"{blackbase_phone}_{phone}_{hit_score}.wav" in the folder "download_data/{blackbase_phone}"
        os.makedirs(f"download_data/{blackbase_phone_top}",exist_ok=True)
        wget.download(file_url,f"download_data/{blackbase_phone_top}/{blackbase_phone_top}_{phone}_{hit_score}.wav")

if __name__ == '__main__':
    os.makedirs("download_data",exist_ok=True)
    df = pd.read_csv("hit_data_all.csv")
    get_hit_score_distribution(df)
    # df with hit_keys_num>th
    df = df[df["hit_keys_num"]>0]
    dowanload_data(df)