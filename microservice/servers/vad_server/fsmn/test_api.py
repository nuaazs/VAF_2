import requests

# 上传本地音频文件
file_url = 'http://localhost:5004/fsmn_vad/file'
file_path = '/datasets/test/cti_test_dataset_16k/18968580260/cti_record_11002_1640309224773385_1-886b36a0-cbf3-4a04-b2fd-6c70b5b31e00.wav'
spkid = '18968580260'
channel = 0
smooth_threshold = 0.5
min_duration = 1
save_oss = False

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
# # 传递音频文件URL
# url_url = 'http://localhost:5004/fsmn_vad/url'
# audio_url = "http://127.0.0.1:9000/testing/0008/cti_record_11006_1654829635356708_1-3795ad39-5142-4597-8a5a-7ff73f5d080e.wav"
# spkid = '1654829635356708'
# channel = 0
# smooth_threshold = 0.5
# min_duration = 1
# save_oss = False

# data = {
#     'spkid': spkid,
#     'channel': channel,
#     'url': audio_url,
#     'smooth_threshold': smooth_threshold,
#     'min_duration': min_duration,
#     'save_oss': save_oss,
# }

# response = requests.post(url_url, data=data)

# if response.status_code == 200:
#     result = response.json()
#     print(result)
#     file_list = result['file_list']
#     url_list = result['url_list']
#     for file in file_list:
#         print(file)
#     for url in url_list:
#         print(url)
# else:
#     print('Error:', response.text)
