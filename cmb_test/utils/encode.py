# coding = utf-8
# @Time    : 2023-10-24  13:05:43
# @Author  : zhaosheng@lyxxkj.com.cn
# @Describe: Get speaker embedding.

import os
import requests
import json
import numpy as np
from tqdm import tqdm

def get_speaker_embedding(audio_path,url,wav_length):
    '''
    Get speaker embedding.
    Args:
        audio_path: audio path
        url: url
    Returns:
        emb: speaker embedding
    '''
    spkid = audio_path.split('/')[-1].split('.')[0]
    data = {'spkid':spkid,'window_size':wav_length}
    files = [
        ('wav_files',(audio_path,open(audio_path,'rb'),'audio/wav'))
    ]
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }
    response = requests.request("POST",url,files=files,data=data,headers=headers)
    # print(response.json())
    # if 'embedding' not in response.json().keys():
    #     print(f"Error in {audio_path}")
    #     return None
    emb = response.json()['embedding']
    result = {}
    for model in emb.keys():
        model_emb = emb[model]['embedding']
        model_emb_key = model_emb.keys()
        for key in model_emb_key:
            model_emb_result = model_emb[key]
        result[model] = np.array(model_emb_result)
    return result

def get_speaker_embedding_list(audio_list,url,wav_length,spkid_location=-2):
    '''
    Get speaker embedding list.
    Args:
        audio_list: audio list
        url: url
    Returns:
        emb_list: speaker embedding list
    '''
    emb_list = []
    for audio_path in tqdm(audio_list):
        try:
            emb = get_speaker_embedding(audio_path,url,wav_length)
            spkid = audio_path.split('/')[spkid_location]
            emb_list.append({'spkid':spkid,'emb':emb})
        except:
            print(f"Error in {audio_path}")
            continue
    return emb_list

def register(audio_list,url,wav_length,spkid_location=-2):
    '''
    Register speaker.
    Args:
        audio_list: audio list
        url: url
    Returns:
        response: response
    '''
    success_count = 0
    registered_ids = []
    registered_spks = []
    for audio_path in tqdm(audio_list):
        spkid = audio_path.split('/')[spkid_location]
        filename = audio_path.split('/')[-1]
        new_id = f"{spkid}-{filename.replace('.wav','').replace('_','')}"
        
        with open(audio_path, "rb") as wav_file:
            response = requests.post(url, files={"wav_files": wav_file}, data={"spkid": spkid, "window_size": wav_length})
        if response.status_code == 200:
            success_count += 1
            registered_ids.append(new_id)
            registered_spks.append(spkid)
        else:
            print(f"Error in {audio_path}")
            print(response.json())
    return success_count, registered_ids, registered_spks

def search(audio_list,url,wav_length,spkid_location=-2):
    '''
    Search speaker.
    Args:
        audio_list: audio list
        url: url
    Returns:
        response: response
    '''
    success_count = 0
    searched_ids = []
    searched_spks = []
    search_result_dict = {}
    for audio_path in tqdm(audio_list):
        spkid = audio_path.split('/')[spkid_location]
        filename = audio_path.split('/')[-1]
        new_id = f"{spkid}-{filename.replace('.wav','').replace('_','')}"
        
        with open(audio_path, "rb") as wav_file:
            response = requests.post(url, files={"wav_files": wav_file}, data={"spkid": spkid, "window_size": wav_length})
        if response.status_code == 200:
            success_count += 1
            searched_ids.append(new_id)
            searched_spks.append(spkid)
            search_result_dict[new_id] = response.json()
        else:
            print(f"Error in {audio_path}")
            print(response.json())
    return success_count, searched_ids, searched_spks, search_result_dict
