import requests

# 请求的URL
url = "http://localhost:5000/transcribe/url"

# 上传的音频文件路径
audio_path = "/home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav"

data = {"channel": 0, "spkid": "zhaosheng","save_oss":False,"url":f"local://{audio_path}","postprocess":0}
# 构造multipart/form-data格式的请求体
files=[
   ('wav_file',(audio_path.split("/")[-1],open(audio_path,'rb'),'application/octet-stream'))
]
# 发送POST请求
response = requests.post(url, files=files,data=data)

# 解析响应数据
transcription = response.json().get("transcription")

# 打印转录结果
print(transcription)
