import requests
from tqdm import tqdm
import numpy as np

def get_embedding(file_path,vad=0):
    url = "http://127.0.0.1:7777/get_embedding/file"
    payload={"spkid":"zhaosheng","vad":vad}
    files=[
    ('wav_file',(file_path.split('/')[-1],open(file_path,'rb'),'application/octet-stream'))
    ]
    headers = {
    'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
    }
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return np.array(response.json()["ECAPATDNN"]),np.array(response.json()["CAMPP"])

    

if __name__ == "__main__":
    e1,c1 = get_embedding("/datasets_hdd/datasets/cjsd_download_8k/15871694607/cti_record_11005_1654497268960680_1-f5e9079d-772a-49e5-b8f5-983f86c9bf73.wav")
    e2,c2 = get_embedding("/datasets_hdd/datasets/cjsd_download_8k/15871694607/cti_record_11005_1654497268960680_1-f5e9079d-772a-49e5-b8f5-983f86c9bf73.wav",vad=1)
    print(e1.shape,e2.shape)
    print(c1.shape,c2.shape)
    # get sim of two embeddings:e1 e2
    from  torch.nn import CosineSimilarity
    import torch
    similarity = CosineSimilarity(dim=-1, eps=1e-6)
    print(similarity(torch.from_numpy(e1),torch.from_numpy(e2)))
    print(similarity(torch.from_numpy(c1),torch.from_numpy(c2)))