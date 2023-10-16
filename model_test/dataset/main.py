import os
import re
import cfg
import pickle
import pymysql
import re
import os
import requests
import wget
import pandas as pd
import glob
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('main.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)


df = pd.read_csv('cti_record.csv')
map_data = {}
for index, row in df.iterrows():
    map_data[row['record_file_name'].split("/")[-1]] = row['begintime']

def add_key_or_append(data_dict, phone,fid, value):
    if phone not in data_dict:
        data_dict[phone] = {}
    if fid not in data_dict[phone]:
        data_dict[phone][fid] = []
    data_dict[phone][fid].append(value)
    return data_dict

with open(cfg.train_wav_scp, 'r') as f:
    wav_files = f.readlines()
    
if __name__ == '__main__':
    
    data_dict = {}
    num = 0
    for wav_file in tqdm(wav_files):
        try:
            num = num + 1
            wav_filename = wav_file.split("/")[-1].strip()
            root =  wav_file.replace(wav_filename, "").strip()
            # print(f"#{num} wav_filename: {wav_filename}, root: {root}")
            if wav_filename.startswith("20"):
                fid = wav_filename[:-15]
                phone = root.split("/")[-2]
            if wav_filename.startswith("cti_"):
                fid=map_data[wav_filename]
                phone = root.split("/")[-2]
                fid = str(fid).replace(' ', '').replace(':', '').replace('-', '')
            # assert len(fid) == 14, "fid length is not 14"
            data_dict = add_key_or_append(data_dict, phone,fid, os.path.join(root, wav_filename))
        except Exception as e:
            logger.error(e)
            print(e)
            continue

    print(f"Data Dict Len #{len(data_dict)}")
   
    # save data_dict to file
    with open('data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    # load data_dict from file
    with open('data_dict.pkl', 'rb') as f:
        data_dict_load = pickle.load(f)
    print(f"Data Dict Len #{len(data_dict_load)}")
    # print(data_dict_load)