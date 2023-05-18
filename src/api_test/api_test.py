# coding = utf-8
# @Time    : 2022-09-11  18:41:22
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: API test script, test the running speed of CPU and GPU version and the proportion of each part.
import glob

import requests
import argparse
import datetime
import os
import random
import numpy as np
import json
from multiprocessing.dummy import Pool as ThreadPool

parser = argparse.ArgumentParser(description="")
parser.add_argument("--ip", type=str, default="192.168.3.202", help="server ip")
parser.add_argument("--port", type=int, default=8191, help="port number")  # old 8199 new 8190
parser.add_argument("--path", type=str, default="test", help="test|register")
parser.add_argument("--wav_file", type=str, default="./1p1c8k.wav", help="")
parser.add_argument(
    "--wav_path",
    type=str,
    default="/lyxx/cti_aftervad_train_data/19963599993",
    help="The directory address of the wav file for testing.",
)
parser.add_argument("--mode", type=str, default="file", help="url|file")
parser.add_argument(
    "--test_num",
    type=int,
    default=10,
    help="The total number of files you want to test, if not enough to test the same files repeatedly.",
)
args = parser.parse_args()

url = f"http://{args.ip}:{args.port}/{args.path}/{args.mode}"
headers = {"Content-Type": "multipart/form-data"}
endtime = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
begintime = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
wavs = sorted(
    [
        os.path.join(args.wav_path, _file)
        for _file in os.listdir(args.wav_path)
        if ".wav" in _file
    ]
)
# wavs = [args.wav_file] * args.test_num
print(f"* Start testing:\n\t total {len(wavs)} files in fold.")
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
encode_total = 0

for wav in wavs:
    for i in range(1):
        if item_number>=args.test_num:
            break

        if args.mode == 'file':
            request_file = {'wav_file':open(wav, 'rb')}
            phone = random.randint(11111111111, 99999999999)
            values = {"spkid": str(phone),"show_phone": "15151832002","call_begintime":begintime,"call_endtime":endtime}
            start = datetime.datetime.now()
            resp = requests.request("POST",url, files=request_file, data=values)
            print(resp.json())
            print(f"\n\n#{item_number}")
            print(json.dumps(resp.json(), sort_keys=False, indent=4))
            time_used =datetime.datetime.now() - start
            total = time_used.total_seconds()
            
            if resp.json()["status"]=="success":
                item_number += 1
                total_total += total
            dowanload_time = resp.json()["used_time"]["download_used_time"]
            vad_time = resp.json()["used_time"]["vad_used_time"]
            encode_time = resp.json()["used_time"]["encode_time"]
            to_database_time = resp.json()["used_time"].get("to_database_used_time",0)
            test_time = resp.json()["used_time"].get("test_used_time",0)
            test_embedding_time = resp.json()["used_time"].get("embedding_used_time",0)
            download_total += dowanload_time
            vad_total += vad_time
            encode_total += encode_time
            to_database_total += to_database_time
            test_total += test_time
            test_embedding_total += test_embedding_time
            other_total += total - dowanload_time - vad_time - encode_time - to_database_time - test_time - test_embedding_time
        else:

            wav_url = f"local://{wav}"
            wav_url = "http://192.168.3.202:9000/raw/raw_020229975851662540251_1_1734894.wav"
            phone = random.randint(11111111111, 99999999999)
            values = {"spkid": str(phone),"show_phone": "15151832002","wav_url":wav_url,"call_begintime":begintime,"call_endtime":endtime}
            start = datetime.datetime.now()
            resp = requests.request("POST",url, data=values)
            print(f"\n\n#{item_number}")
            print(json.dumps(resp.json(), sort_keys=False, indent=4))
            time_used =datetime.datetime.now() - start
            total = time_used.total_seconds()

            if resp.json()["status"]=="success":
                item_number += 1
                total_total += total
            dowanload_time = resp.json()["used_time"]["download_used_time"]
            vad_time = resp.json()["used_time"]["vad_used_time"]
            encode_time = resp.json()["used_time"]["encode_time"]
            to_database_time = resp.json()["used_time"].get("to_database_used_time",0)
            test_time = resp.json()["used_time"].get("test_used_time",0)
            test_embedding_time = resp.json()["used_time"].get("embedding_used_time",0)
            download_total += dowanload_time
            vad_total += vad_time
            encode_total += encode_time
            to_database_total += to_database_time
            test_total += test_time
            test_embedding_total += test_embedding_time
            other_total += total - dowanload_time - vad_time - encode_time - to_database_time - test_time - test_embedding_time


def register(item):
    request_file = {"wav_file": open(item, "rb")}
    wav_url = f"local://{item}"
    phone = random.randint(11111111111, 99999999999)
    values = {
        "spkid": str(phone),
        "show_phone": "15151832002",
        "call_begintime": begintime,
        "call_endtime": endtime,
        "wav_url": wav_url,
    }
    try:
        resp = requests.request("POST", url, files=request_file, data=values)
        print(resp.json())
    except Exception as e:
        print(e)
        return

print(f"\t\t-> Download Used Time:{download_total*100/total_total:.2f}%")
print(f"\t\t-> Vad Used Time:{vad_total*100/total_total:.2f}%")
print(f"\t\t-> Self test Used Time:{self_test_total*100/total_total:.2f}%")
print(f"\t\t-> Classify Used Time:{classify_total*100/total_total:.2f}%")
print(f"\t\t-> To DataBase Used Time:{to_database_total*100/total_total:.2f}%")
print(f"\t\t-> Test Used Time:{test_total*100/total_total:.2f}%")
print(f"\t\t-> GetEmbedding Used Time:{test_embedding_total*100/total_total:.2f}%")
print(f"\t\t-> Other Used Time:{other_total*100/total_total:.2f}%")
print(f"="*50)
# print Mean Used Time for each module
print(f"\t\t-> Mean Used Time (s):{total_total/item_number}")
print(f"\t\t-> Mean VAD Time:{vad_total/item_number}")
print(f"\t\t-> Mean Download Time:{download_total/item_number}")
print(f"\t\t-> Mean Encode Time:{encode_total/item_number}")
print(f"\t\t-> Mean To DataBase Time:{to_database_total/item_number}")
print(f"\t\t-> Mean Test Time:{test_total/item_number}")
print(f"\t\t-> Mean GetEmbedding Time:{test_embedding_total/item_number}")
print(f"\t\t-> Mean Other Time:{other_total/item_number}")