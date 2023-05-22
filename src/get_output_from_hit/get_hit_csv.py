from sql import MysqlWrapper

from asr import get_asr_content
# from nlp import get_data

import pandas as pd
import numpy as np
import wget
import os
# import matplotlib.pyplot as plt

mysql = MysqlWrapper(host="192.168.3.201", port=3306, user="root", passwd="longyuan", db="si")

def get_hit_data():
    all_data = []
    sql = "select phone,file_url,blackbase_phone,blackbase_id,top_10,valid_length,hit_score,content_text,hit_time from hit;"    
    for data in mysql.query(sql):
        gray_phone,file_url,blackbase_phone,blackbase_id,top_10,valid_length,hit_score,content_text,hit_time = data
        top_10_data = top_10.split("|")
        scores = []
        phones = []
        for _data in top_10_data:
            score,phone = _data.split(",")
            score = float(score.replace("[","").replace("]",""))
            scores.append(score)
            phones.append(phone)
        mean_score = sum(scores)/len(scores)
        blackbase_phone= blackbase_phone.split(",")[0]
        _data = [gray_phone,file_url,blackbase_phone,valid_length,hit_score,mean_score,content_text,hit_time]
        all_data.append(_data)
    # convert to dataframe
    df = pd.DataFrame(all_data,columns=["gray_phone","file_url","blackbase_phone","valid_length","hit_score","mean_score","content_text","hit_time"])
    df.to_csv("hit_data.csv",index=False)
    return df

if __name__ == '__main__':
    df = get_hit_data()
    print(df.head())