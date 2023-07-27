import requests
import argparse
import datetime
import os
import random
import numpy as np
import json
from multiprocessing.dummy import Pool as ThreadPool

# ramdom seed
np.random.seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--ip", type=str, default="192.168.3.202", help="server ip")
parser.add_argument("--port", type=int, default=8190, help="port number")
parser.add_argument("--folder", type=str, default="../si_test_wavs", help="")
parser.add_argument("--mode", type=str, default="file", help="url|file")
parser.add_argument("--register_radio", type=float, default=0.5, help="")
args = parser.parse_args()

register_url = f"http://{args.ip}:{args.port}/register/{args.mode}"
test_url = f"http://{args.ip}:{args.port}/test/{args.mode}"
headers = {"Content-Type": "multipart/form-data"}
endtime = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
begintime = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")

register_info = {}
test_info = {}
register_num,test_num = 0,0
register_wavs = []
test_wavs = []

# get register and test wav file
spk_id_list = os.listdir(args.folder)
for spk_id in spk_id_list:
    wav_files = os.listdir(os.path.join(args.folder, spk_id))
    # shuffle
    random.shuffle(wav_files)
    # split
    register_num = max(1,int(len(wav_files) * args.register_radio))
    register_wav_files = wav_files[:register_num]
    test_wav_files = wav_files[register_num:]
    register_info[spk_id] = [os.path.join(args.folder, spk_id, wav_file) for wav_file in register_wav_files]
    register_num += len(register_info[spk_id])
    register_wavs += register_info[spk_id]
    test_info[spk_id] = [os.path.join(args.folder, spk_id, wav_file) for wav_file in test_wav_files]
    test_num += len(test_info[spk_id])
    test_wavs += test_info[spk_id]

print(f"Total #{len(spk_id_list)} speakers, #{register_num} register files, #{test_num} test files.")
print(register_info)
print(test_info)

start_time = datetime.datetime.now()
item_number = 0
download_total = 0
vad_total = 0
self_test_total = 0
classify_total = 0
other_total = 0
total_total = 0
to_database_total = 0
test_total = 0
test_embedding_total = 0

def test(item):
    print(item)
    request_file = {"wav_file": open(item, "rb")}
    wav_url = f"local://{item}"
    phone = item.split("/")[-2]
    values = {
        "spkid": str(phone),
        "show_phone": "15151832002",
        "call_begintime": begintime,
        "call_endtime": endtime,
        "wav_url": wav_url,
    }
    try:
        resp = requests.request("POST", test_url, files=request_file, data=values)
        print(resp.json())
    except Exception as e:
        print(e)
        return
    
def register(item):
    request_file = {"wav_file": open(item, "rb")}
    wav_url = f"local://{item}"
    # phone = random.randint(11111111111, 99999999999)
    phone = item.split("/")[-2]
    values = {
        "spkid": str(phone),
        "show_phone": "15151832002",
        "call_begintime": begintime,
        "call_endtime": endtime,
        "wav_url": wav_url,
    }
    try:
        resp = requests.request("POST", register_url, files=request_file, data=values)
        print(resp.json())
        return resp.json()
    except Exception as e:
        print(e)
        return


pool = ThreadPool(4)
start = datetime.datetime.now()
pool.map(register, register_wavs)
pool.close()
pool.join()
time_used = datetime.datetime.now() - start
total = time_used.total_seconds()
mean_time = total / len(register_wavs)
print(f"\n\n\t-> Register Mean Used Time:{mean_time}")

# pool = ThreadPool(4)
# start = datetime.datetime.now()
# pool.map(test, test_wavs)
# pool.close()
# pool.join()
# time_used = datetime.datetime.now() - start
# total = time_used.total_seconds()
# mean_time = total / len(test_wavs)
# print(f"\n\n\t-> Test Mean Used Time:{mean_time}")
TN,TP,FN,FP = 0,0,0,0
print(test_wavs)
for wav_file in test_wavs:
    result = str(test(wav_file))
    print(result)
    # convert json to dict
    result = eval(result)
    # print(result)
    print(result.keys())
    hit = result["inbase"]
    if hit:
        print(f"{wav_file} hit")
        hit_phone = result["spkid"]
        if hit_phone == wav_file.split("/")[-2]:
            print("hit phone right")
            TP += 1
        else:
            print("hit phone wrong")
            FP += 1
    else:
        print(f"{wav_file} not hit")
        FN += 1

print(f"TP:{TP},FP:{FP},FN:{FN}")
acc = TP / (TP + FP)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * acc * recall / (acc + recall)
print(f"acc:{acc},recall:{recall},precision:{precision},f1:{f1}")

