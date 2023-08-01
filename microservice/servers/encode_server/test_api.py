import requests
import numpy as np
import torch
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# 定义接口 URL
url = "http://localhost:5001/encode"

# 准备要发送的数据
data = {
    "spkid": "151",
    "channel": 0,
    "filelist": "local:///datasets/test/cti_test_dataset_16k_envad_bak/18659111928/18659111928_001_Distance00_Dialect00.wav,local:///datasets/test/cti_test_dataset_16k_envad_bak/18659111928/18659111928_007_Distance00_Dialect00.wav,local:///datasets/test/cti_test_dataset_16k_envad_bak/19513386018/19513386018_001_Distance00_Dialect00.wav",
    "save_oss": "False",
    "score_threshold": 0.7
}

# 发送 POST 请求
response = requests.post(url, data=data)

# 解析响应结果
result = response.json()

# 打印结果
print(result)
print(result.keys())
print(result["file_emb"].keys())
print(result["file_emb"]["CAMPP_200k_F3dspeaker"]["embedding"].keys())
a = np.array(result["file_emb"]["CAMPP_200k_F3dspeaker"]["embedding"]["local:///datasets/test/cti_test_dataset_16k_envad_bak/18659111928/18659111928_001_Distance00_Dialect00.wav"]).reshape(1,-1)
print(a.shape)

b = np.array(result["file_emb"]["CAMPP_200k_F3dspeaker"]["embedding"]["local:///datasets/test/cti_test_dataset_16k_envad_bak/18659111928/18659111928_007_Distance00_Dialect00.wav"]).reshape(1,-1)
print(b.shape)
c = np.array(result["file_emb"]["CAMPP_200k_F3dspeaker"]["embedding"]["local:///datasets/test/cti_test_dataset_16k_envad_bak/19513386018/19513386018_001_Distance00_Dialect00.wav"]).reshape(1,-1)
print(similarity(torch.tensor(a),torch.tensor(b)))
print(similarity(torch.tensor(a),torch.tensor(c)))
print(similarity(torch.tensor(b),torch.tensor(c)))
print(a)
print(b)