import requests
import numpy as np
import torch
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# 定义接口 URL
url = "http://localhost:5001/encode"

file1_path = "/home/xuekaixiang/workplace/vaf/microservice/servers/encode_server/18136655705_selected.wav"
file2_path = "/home/xuekaixiang/workplace/vaf/microservice/servers/encode_server/18136655705_selected.wav"

# 准备要发送的数据
data = {
    "spkid": "151",
    "channel": 0,
    "filelist": f"local://{file1_path},local://{file2_path}",
    "save_oss": "False",
}

# 发送 POST 请求
response = requests.post(url, data=data)

# 解析响应结果
result = response.json()

# 打印结果
print(result["file_emb"].keys())
a = np.array(result["file_emb"]["eres2net"]["embedding"][f"local://{file1_path}"]).reshape(1, -1)
print(a.shape)
b = np.array(result["file_emb"]["eres2net"]["embedding"][f"local://{file2_path}"]).reshape(1, -1)
print(b.shape)
print(similarity(torch.tensor(a), torch.tensor(b)))
