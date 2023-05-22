import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
# from speechbrain.pretrained import VAD

import torchaudio
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
import subprocess

# cfg
import cfg

# utils
from utils.oss import upload_file

from utils.preprocess.vad import vad
from utils.preprocess.energy_vad import energy_VAD
from utils.encoder.encode import encode

from tqdm import tqdm
# cos similarity
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def calc_f1(embeddings, score_th=0.78):
    print(f"\n\n# TH: {score_th}")
    # calc TN TP FN FP
    TP,TN,FN,FP = 0,0,0,0
    register_info = []
    test_info = []
    for phone in embeddings.keys():
        speech_num = len(embeddings[phone])
        if speech_num < 2:
            continue
        # register
        register_embeddings = embeddings[phone][:int(speech_num/2)]
        register_info+=register_embeddings # [embedding,spk_id]
        # test
        test_embeddings = embeddings[phone][int(speech_num/2):]
        test_info+=test_embeddings # [embedding,spk_id]
    print(f"\tregister wav num: {len(register_info)}, test wav num: {len(test_info)}")
    for test_data in test_info:
        scores = []
        test_embedding = test_data[0]
        test_spk_id = test_data[1]
        for register_data in register_info:
            register_embedding = register_data[0]
            register_spk_id = register_data[1]
            scores.append([similarity(register_embedding, test_embedding),test_spk_id])

        # max score is the same spk
        scores = sorted(scores,key=lambda x:x[0],reverse=True)
        score, test_spk_id= scores[0]
        if score>score_th:
            print(f"{test_spk_id} hit {register_spk_id} with score {score}")
            if test_spk_id.split("_")[0] == register_spk_id.split("_")[0]:
                TP += 1
            else:
                FP += 1
        else:
            print(f"{test_spk_id} hit nothing, max score is {score} with {register_spk_id}")
            FN += 1
    # calc precision recall F1
    precision = TP/(TP+FP+1e-6)
    recall = TP/(TP+FN+1e-6)
    F1 = 2*precision*recall/(precision+recall+1e-6)
    print(f"TP:{TP}; TN:{TN}; FN:{FN}; FP:{FP}")
    print(f"precision:{precision}; recall:{recall}; F1:{F1}")

    # calc frr and far
    frr = FN/(FN+TP+1e-6)
    far = FP/(FP+TN+1e-6)
    print(f"frr:{frr}; far:{far}")
    return precision, recall, F1, frr, far


if __name__ == "__main__":
    score_th = 0.78
    fold_path = "/lyxx/datasets/preprocessed/cjsd_8k_vad"
    embeddings = {}
    useful_file_num = 0
    useful_spk_num = 0
    with open("scores.csv","w") as f:
        pbar = tqdm(os.listdir(fold_path))
        for phone_num in pbar:
            phone_path = os.path.join(fold_path,phone_num)
            for scene_path in os.listdir(phone_path):
                scene_id = scene_path
                scene_path = os.path.join(phone_path,scene_path)

                for filename in [_file for _file in os.listdir(scene_path) if "vad.wav" not in _file]:
                    spk_id = phone_num+"_"+scene_id+filename.split(".")[0]
                    file_path = os.path.join(scene_path,filename)
                    wav, sr = torchaudio.load(file_path)
                    wav = wav[0,:].reshape(1,-1)
                    wav_length = wav.shape[1]/sr
                    # spk_id = file_path.split("/")[-2]+"_"+file_path.split("/")[-1].split(".")[0]
                    try:
                        vad_response = vad(wav,spk_id,action_type="test",device="cuda:0")
                    except Exception as e:
                        print(e)
                        continue
                    if vad_response["after_length"]<10:
                        print(f"file:{file_path} after vad length is too short, skip")
                        continue
                    
                    vad_result = vad_response["wav_torch"]
                    # print(vad_result.shape)
                    # save vad result file
                    torchaudio.save(file_path.replace(".wav","_vad.wav"),vad_result.reshape(1,-1),sr)

                    assert len(vad_result.shape) == 1
                    length = vad_result.shape[0]
                    vad_result_a = vad_result[:int(length/2)]
                    vad_result_b = vad_result[int(length/2):]
                    response= encode(vad_result,action_type="test")
                    # print(response)
                    result = response["tensor"]
                    response_a= encode(vad_result_a,action_type="test")
                    result_a = response_a["tensor"]
                    response_b= encode(vad_result_b,action_type="test")
                    result_b = response_b["tensor"]
                    score_a_raw  = similarity(result_a, result)
                    score_b_raw  = similarity(result_b, result)
                    score_a_b = similarity(result_a, result_b)

                    if score_a_b<0.7:
                        print(f"file:{file_path} self test score_a_b:{score_a_b} is too low, skip")
                        continue
                    useful_file_num += 1
                    if phone_num not in embeddings.keys():
                        embeddings[phone_num] = []
                        
                    embeddings[phone_num].append([result,spk_id])
                    if len(embeddings[phone_num]) >1:
                        useful_spk_num += 1
                    
                    # update tqdm prefix
                    pbar.set_postfix({'useful_file_num': useful_file_num, 'useful_spk_num': useful_spk_num})
                    print(f"score_a_raw:{score_a_raw}; score_b_raw:{score_b_raw}; score_a_b:{score_a_b}")
                    f.write(f"{spk_id},{score_a_raw},{score_b_raw},{score_a_b}\n")
                    
            
            # print(result)
            # print(result_a)
            # print(result_b)
    # save embeddings
    np.save("embeddings.npy",embeddings)

    print(f"Embeddings extracted, start calc eer and min_dcf...\n\n")

    # calc eer and min_dcf
    frr_list = []
    far_list = []
    for score_th in np.arange(0.5,1,0.01):
        # calc precision recall F1
        precision, recall, F1, frr, far = calc_f1(embeddings,score_th)
        frr_list.append([frr,score_th])
        far_list.append([far,score_th])

    # calc eer and min_dcf from different score_th
    frr_list = sorted(frr_list,key=lambda x:x[0])
    far_list = sorted(far_list,key=lambda x:x[0])
    eer = (frr_list[0][0]+far_list[0][0])/2
    min_dcf = min(frr_list[0][0]+1*far_list[0][0],frr_list[0][0]+10*far_list[0][0])
    print(f"eer:{eer}; min_dcf:{min_dcf}")


