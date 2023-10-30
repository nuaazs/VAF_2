# coding = utf-8
# @Time    : 2023-10-24  12:57:18
# @Author  : zhaosheng@lyxxkj.com.cn
# @Describe: Compare.

import numpy as np
from utils.encode import get_speaker_embedding
from tqdm import tqdm

# save extracted embeddings, to avoid repeat extraction
global_emb_dict = {}

def get_emb(emb):
    '''
    Get embedding.
    Args:
        emb: embedding
    Returns:
        emb_a: embedding_a
        emb_b: embedding_b
        emb_full: embedding_full
    '''
    emb_len = emb.reshape(-1).shape[0]
    tiny_emb_size = int(emb_len/3)
    emb_a = emb[:tiny_emb_size]
    emb_b = emb[tiny_emb_size:2*tiny_emb_size]
    emb_full = emb[2*tiny_emb_size:]
    return emb_a, emb_b, emb_full

def get_cosine_score(emb_a, emb_b):
    '''
    Get cosine score.
    Args:
        emb_a: embedding_a
        emb_b: embedding_b
    Returns:
        score: cosine score
    '''
    score = emb_a.dot(emb_b)/(np.linalg.norm(emb_a)*np.linalg.norm(emb_b))
    return score

def compare(emb1,emb2,cmf=False):
    '''
    Compare two embeddings.
    Args:
        emb1: embedding1
        emb2: embedding2
    Returns:
        similarity: similarity
    '''
    if cmf:
        emb1_a,emb1_b,emb1_full = get_emb(emb1)
        emb2_a,emb2_b,emb2_full = get_emb(emb2)
        similarity_full = get_cosine_score(emb1_full,emb2_full)
        # High score
        similarity1_ab = get_cosine_score(emb1_a,emb1_b)
        similarity2_ab = get_cosine_score(emb2_a,emb2_b)
        high_median = np.median([similarity1_ab,similarity2_ab])
        high_max = np.max([similarity1_ab,similarity2_ab])
        high_mean = np.mean([similarity1_ab,similarity2_ab])

        # Low score
        similarity12_aa = get_cosine_score(emb1_a,emb2_a)
        similarity12_bb = get_cosine_score(emb1_b,emb2_b)
        similarity12_ab = get_cosine_score(emb1_a,emb2_b)
        similarity12_ba = get_cosine_score(emb1_b,emb2_a)
        low_mean = np.mean([similarity12_aa,similarity12_bb,similarity12_ab,similarity12_ba])
        low_median = np.median([similarity12_aa,similarity12_bb,similarity12_ab,similarity12_ba])

        if low_mean >= high_max:
            result = 1
            return result,similarity_full
        
        if abs(high_max - low_mean)/high_max >= 0.5:
            result =  0.0
        elif abs(high_max - low_mean)/high_max < 0.3:
            result =  1
        else:
            if (similarity_full) >= high_max:
                result =  1
            else:
                result =  0.0
        raw_score = similarity_full
        print(f"high_median: {high_median}",f"\tlow_median: {low_median}",f"\traw:{similarity_full}",f"\tresult: {result}")
        return result,raw_score
        
    else:
        emb1_a,emb1_b,emb1_full = get_emb(emb1)
        emb2_a,emb2_b,emb2_full = get_emb(emb2)
        similarity_full = get_cosine_score(emb1_full,emb2_full)
        return similarity_full,similarity_full

def compare_res(res1,res2,cmf):
    '''
    Compare two results.
    Args:
        res1: result1
        res2: result2
    Returns:
        similarity: similarity
    '''
    key1 = sorted([i for i in res1.keys()])
    key2 = sorted([i for i in res2.keys()])
    assert key1 == key2
    similarity = {}
    raw_score = {}
    for k in key1:
        similarity[k],raw_score[k] = compare(res1[k],res2[k],cmf=cmf)
    if cmf:
        yes_count = 0
        no_count = 0
        for i in similarity.values():
            if i < 0.5:
                no_count += 1
            else:
                yes_count += 1
        if yes_count >= no_count:
            return 0.9+0.1*np.mean([i for i in raw_score.values()])
        else:
            return 0.1*np.mean([i for i in raw_score.values()])
    else:
        mean_similarity = np.mean([i for i in raw_score.values()])
        return mean_similarity

def compare_trial(trial,url,wav_length,cmf):
    '''
    Compare trial.
    Args:
        trial: trial
        url: url
        wav_length: wav_length
    Returns:
        similarity: similarity
    '''
    spkid1 = trial['speakerid1']
    spkid2 = trial['speakerid2']
    wav1 = trial['wav_path1']
    wav2 = trial['wav_path2']
    label = trial['label']
    print(f"spkid1: {spkid1}",f"\tspkid2: {spkid2}",f"\tlabel: {label}")
    
    if wav1 not in global_emb_dict.keys():
        emb1_dict = get_speaker_embedding(wav1,url=url,wav_length=wav_length)
        global_emb_dict[wav1] = emb1_dict
    else:
        emb1_dict = global_emb_dict[wav1]

    if wav2 not in global_emb_dict.keys():
        emb2_dict = get_speaker_embedding(wav2,url=url,wav_length=wav_length)
        global_emb_dict[wav2] = emb2_dict
    else:
        emb2_dict = global_emb_dict[wav2]

    similarity = compare_res(emb1_dict,emb2_dict,cmf)
    return similarity

def collision(register_data,test_data,topk,th,test_num=None,cmf=False,logger=None):
    '''
    Collision.
    Args:
        register_info: register info
        test_info: test info
    Returns:
        collision: collision
    '''
    if test_num is not None:
        test_data = test_data[:test_num]
        register_data = register_data[:test_num]
    all_results = []
    registered_ids = []
    for register_tiny_data in tqdm(register_data):
        registered_ids.append(register_tiny_data['spkid'])
    registered_ids = list(set(registered_ids))
    TN,TP,FN,FP = 0,0,0,0
    for test_tiny_data in tqdm(test_data):
        tiny_results = []
        for register_tiny_data in register_data:
            test_tiny_data_emb = test_tiny_data['emb']
            register_tiny_data_emb = register_tiny_data['emb']
            similarity = compare_res(test_tiny_data_emb,register_tiny_data_emb,cmf)
            tiny_results.append([register_tiny_data['spkid'],similarity])
            # get top 5, sorted by similarity
            tiny_results = sorted(tiny_results,key=lambda x:x[-1],reverse=True)[:topk]
        # print(tiny_results)
        # top K results
        # if score of top 1 is greater than TH and test spkid is same as top 1 spkid, then TP+1
        # if score of top 1 is greater than TH and test spkid is not same as top 1 spkid, then FP+1
        # if score of top 1 is less than TH and test spkid is same as top 1 spkid, then FN+1
        # if score of top 1 is less than TH and test spkid is not same as top 1 spkid and test id is not in registered_ids, then TN+1
        # if score of top 1 is less than TH and test spkid is not same as top 1 spkid and test id is in registered_ids, then FN+1
        if tiny_results[0][-1] >= th:
            if test_tiny_data['spkid'] == tiny_results[0][0]:
                TP += 1
            else:
                FP += 1
                logger.error(f"* FP: {test_tiny_data['spkid']} vs {tiny_results[0][0]}")
                logger.error(f"\t {tiny_results}")
        else:
            if test_tiny_data['spkid'] == tiny_results[0][0]:
                FN += 1
                logger.error(f"* FN: {test_tiny_data['spkid']} vs {tiny_results[0][0]}")
                logger.error(f"\t {tiny_results}")
            else:
                if test_tiny_data['spkid'] in registered_ids:
                    FN += 1
                    logger.error(f"* FN: {test_tiny_data['spkid']} vs {tiny_results[0][0]}")
                    logger.error(f"\t {tiny_results}")
                else:
                    TN += 1
    precision = TP/(TP+FP+1e-5)
    recall = TP/(TP+FN+1e-5)
    accuracy = (TP+TN)/(TP+TN+FP+FN+1e-5)
    logger.info(f"Collision precision is {precision}")
    logger.info(f"Collision recall is {recall}")
    logger.info(f"Collision accuracy is {accuracy}")
    return
