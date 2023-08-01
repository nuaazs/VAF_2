import requests
import os

# 定义接口 URL
url = "http://localhost:5009/gender_classify"

# 准备要发送的数据
data = {
    "spkid": "123456",
    "channel": 0,
    "filelist": "local:///home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav,local:///home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时21分12秒-15261175941-1672554056.6345503000.wav",
    "save_oss": "False"
}
# 发送 POST 请求
response = requests.post(url, data=data)
# 解析响应结果
result = response.json()

# 打印结果
print(result)
