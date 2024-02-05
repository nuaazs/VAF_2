import requests
import torchaudio
url = "http://192.168.3.169:5007/denoise/file"

payload = {}
files = [
    (
        "wav_file",
        ("test_wav.wav", open("/datasets/test/sv_test/cjsd300/13002931667/20230112161623/20230112161623_0.wav", "rb"), "audio/wav"),
    )
]
headers = {"User-Agent": "Apifox/1.0.0 (https://apifox.com)"}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

# save wav
with open("test.wav", "wb") as f:
    f.write(response.content)
a=torchaudio.load("test.wav")
