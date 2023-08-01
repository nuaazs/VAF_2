import requests

# 上传本地音频文件
file_url = 'http://localhost:5004/nn_vad/file'
file_path = '/home/zhaosheng/asr_damo_websocket/online/speaker-diraization/data/speaker_diarisation_test_data/1c99701339235ed853362d5a448a94ed-江苏-常州-2023年01月01日14时20分25秒-13357881270-1672554009.6297503000.wav'
spkid = 'speaker_id'
channel = 0
smooth_threshold = 0.5
min_duration = 1
save_oss = 1

files = {'file': open(file_path, 'rb')}
data = {
    'spkid': spkid,
    'channel': channel,
    'smooth_threshold': smooth_threshold,
    'min_duration': min_duration,
    'save_oss': save_oss,
}

response = requests.post(file_url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    print(result)
    file_list = result['file_list']
    url_list = result['url_list']
    for file in file_list:
        print(file)
    for url in url_list:
        print(url)
else:
    print('Error:', response.text)

print("=====================================")
# 传递音频文件URL
url_url = 'http://localhost:5004/nn_vad/url'
audio_url = "http://127.0.0.1:9000/testing/0008/cti_record_11006_1654829635356708_1-3795ad39-5142-4597-8a5a-7ff73f5d080e.wav"
spkid = 'speaker_id'
channel = 0
smooth_threshold = 0.5
min_duration = 1
save_oss = 1

data = {
    'spkid': spkid,
    'channel': channel,
    'url': audio_url,
    'smooth_threshold': smooth_threshold,
    'min_duration': min_duration,
    'save_oss': save_oss,
}

response = requests.post(url_url, data=data)

if response.status_code == 200:
    result = response.json()
    print(result)
    file_list = result['file_list']
    url_list = result['url_list']
    for file in file_list:
        print(file)
    for url in url_list:
        print(url)
else:
    print('Error:', response.text)
