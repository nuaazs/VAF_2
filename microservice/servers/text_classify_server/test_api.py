import requests

# 定义接口 URL
url = "http://localhost:5003/text_classify"

# 准备要发送的数据
data = {
    "text": "基于涉嫌诈骗的单通道文本(约2w条，全量核对) + 正常单通道通话文本（约1w条）进行二分类.损失函数: Cross Entropy Loss"
}

# 发送 POST 请求
response = requests.post(url, json=data)

# 解析响应结果
result = response.json()

# 打印分类结果
print(result)
