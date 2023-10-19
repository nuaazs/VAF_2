import requests


def request_search(calling_number, file_path):
    url = "http://192.168.3.199:8989/search/file"
    data = {
        'spkid': calling_number,
    }
    files = {'wav_file': open(file_path, 'rb')}
    response = requests.post(url, files=files, data=data)
    print(response.text)
    if response.json().get("code") == 200:
        if response.json()['compare_result']['model']['is_hit']:
            print(f"Hit black. calling_number:{calling_number}")
        else:
            print(f"Not hit black. calling_number:{calling_number}")


request_search("12321", "/datasets/cjsd_upload/16795540025/16795540025_20230905143818.wav")