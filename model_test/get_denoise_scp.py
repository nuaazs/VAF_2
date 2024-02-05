import os
import torchaudio
import pandas as pd
import requests
from tqdm import tqdm



# 读取txt文件
with open('/VAF/model_test/scp/cjsd300.scp', 'r') as f:
	lines = f.readlines()
addresses = [line.strip().split(' ')[1] for line in lines]

url = "http://192.168.3.169:5007/denoise/file"
# 逐个读取wav文件
for address in tqdm(addresses):
	paths = os.path.join('/VAF/model_test', address)
	save_path = paths.replace('cjsd300', 'cjsd300denoised')
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	print(f"paths:{paths},save_path:{save_path}")
	payload={}
	files=[
		(
		'wav_file',
		("test_wav.wav", open(paths,'rb'),'audio/wav')
		)
	]
	headers = {'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'}

	response = requests.request("POST", url, headers=headers, data=payload, files=files)
	# save wav
	with open(save_path, "wb") as f:
		f.write(response.content)

	waveform=torchaudio.load(save_path)
