import requests
import glob
import json
import os
import torch
import numpy as np
import torchaudio
from tqdm import tqdm
from dguard.interface.pretrained import load_by_name,ALL_MODELS


url = "http://192.168.3.169:5005/energy_vad/file"



news_path = "/datasets/cjsd_download_test"
save_path = "/home/duanyibo/dyb/vad/feat"
os.makedirs(news_path,exist_ok=True)
wav_files = glob.glob('/datasets/cjsd_download_test/**/*.wav')
for file in tqdm(wav_files):
    try:
        # print(file)
        payload={"spkid":file.split('/')[-2]}
        files=[
        ('file',(file,open(file,'rb'),'application/octet-stream'))
        ]
        headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
        }

        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        waveform, sample_rate = torchaudio.load(file) 
        # print(waveform.shape)
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        # waveform = waveform.reshape
        # print(waveform.shape)
        file_time = response.json()["timelist"]
        # print(file_time)
        waveforms = torch.empty(0)
        for times in file_time:
            start = int(times[0])*16
            end = int(times[1])*16
            # print(start,end)
            snippet = waveform[:,start:end]
            # print(snippet.shape)
            waveforms = torch.cat((waveforms,snippet),dim=1)
        # 使用torch.cat拼接音频片段
        model,feature_extractor,sample_rate = load_by_name("repvgg",0)
        feat = feature_extractor(waveforms)
        new_path = os.path.join(save_path,file.rsplit(".",1)[0].split(news_path)[-1].split("/",1)[-1])
        # print(new_path)
        os.makedirs(new_path.rsplit("/",1)[0], exist_ok=True)
        np.save(new_path,feat)
        # # 写入wav文件  
        # file_name = os.path.join(news_path,file.split("/datasets/cjsd_download_test/")[-1])
        # os.makedirs(file_name.rsplit("/",1)[0],exist_ok=True)
        # # print(waveforms.shape)
        # torchaudio.save(file_name, waveform, sample_rate=16000)
    except Exception as e:
        print("Other error occurred:", e)
