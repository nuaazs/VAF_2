import requests

# 定义接口 URL
url = "http://localhost:5003/text_classify"

# 准备要发送的数据
data = {
    "text": "这是一条测试文本"
}

# 发送 POST 请求
response = requests.post(url, json=data)

# 解析响应结果
result = response.json()

# 打印分类结果
print(result)
