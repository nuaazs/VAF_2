import os
import uuid
import requests


def do_requests(file_path_01, window_size):
    url = "http://192.168.3.199:5001/search"

    payload = {
        'spkid': str(uuid.uuid4()).replace('-', ''),
        'window_size': window_size
    }
    files = [
        ('wav_files', (os.path.basename(file_path_01), open(file_path_01, 'rb'), 'audio/wav')),
    ]
    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    # print(response.text)
    return response.json()


file_path = './raw_18136655705.wav'
window_size = 5
res = do_requests(file_path, window_size)
print(res)
