import os
import uuid
import requests


def do_requests(file_path_01, file_path_02, window_size):
    url = "http://192.168.3.199:5001/compare"

    payload = {
        'spkid': str(uuid.uuid4()).replace('-', ''),
        'window_size': window_size
    }
    files = [
        ('wav_files', (os.path.basename(file_path_01), open(file_path_01, 'rb'), 'audio/wav')),
        ('wav_files', (os.path.basename(file_path_02), open(file_path_02, 'rb'), 'audio/wav'))
    ]
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    return response.json()


file_path_01 = './raw_18136655705.wav'
file_path_02 = './raw_18136655705_copy.wav'
window_size = 5
res = do_requests(file_path_01, file_path_02, window_size)
print(res)
