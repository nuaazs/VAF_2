# coding = utf-8
# @Time    : 2023-10-24  13:10:34
# @Author  : zhaosheng@lyxxkj.com.cn
# @Describe: Read trails from csv file.

import pandas as pd
import os

def read_trails(trails_path,spkid_location=-2):
    """
    Read trails from a csv file.
    csv: speakerid1, wav_path1, speakerid2, wav_path2, label
    Args:
        path: path of trails
    Returns:
        dataframe: trails
    """
    df = pd.read_csv(trails_path, sep=' ', header=None, names=['wav_path1','wav_path2','label'])
    df['wav_path1'] = df['wav_path1'].apply(lambda x:os.path.join('/VAF/model_test/data/test/cti_v2',x))
    df['wav_path2'] = df['wav_path2'].apply(lambda x:os.path.join('/VAF/model_test/data/test/cti_v2',x))
    # 'speakerid1',  'speakerid2', 'label'
    # ADD speakerid1 from wav_path1
    df['speakerid1'] = df['wav_path1'].apply(lambda x:x.split('/')[spkid_location])
    # ADD speakerid2 from wav_path2
    df['speakerid2'] = df['wav_path2'].apply(lambda x:x.split('/')[spkid_location])
    # ADD label = (speakerid1==speakerid2)
    df['label'] = df['speakerid1'] == df['speakerid2']
    # remove other columns
    df = df[['speakerid1','wav_path1','speakerid2','wav_path2','label']]

    return df