import argparse
import datetime
import glob
import multiprocessing.pool
import os.path
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import numpy
import numpy as np
import redis
import requests

import cfg

parser = argparse.ArgumentParser(description="")
parser.add_argument("--ip", type=str, default="192.168.3.202", help="server ip")
parser.add_argument("--port", type=int, default=8190, help="port number")  # old 8199 new 8190
parser.add_argument("--path", type=str, default="test", help="test|register")
parser.add_argument(
    "--wav_path",
    type=str,
    default="/mnt/dataset/gray",
    help="The directory address of the wav file for testing.",
)
parser.add_argument("--mode", type=str, default="file", help="url|file")
parser.add_argument(
    "--test_num",
    type=int,
    default=1,
    help="The total number of files you want to test, if not enough to test the same files repeatedly.",
)
args = parser.parse_args()

headers = {"Content-Type": "multipart/form-data"}
endtime = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
begintime = (datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


def test(item):
    """
    测试接口
    Args:
        item:

    Returns:

    """
    url = f"http://{args.ip}:{args.port}/test/{args.mode}"
    request_file = {"wav_file": open(item, "rb")}
    wav_url = f"local://{item}"
    phone = os.path.basename(item).split("_")[0]
    values = {
        "spkid": str(phone),
        "wav_url": wav_url,
    }
    try:
        resp = requests.request("POST", url, files=request_file, data=values)
        print(resp.json())
    except Exception as e:
        print(e)
        return


def register(item):
    """
    测试接口
    Args:
        item:

    Returns:

    """
    url = f"http://{args.ip}:{args.port}/register/{args.mode}"
    request_file = {"wav_file": open(item, "rb")}
    wav_url = f"local://{item}"
    phone = os.path.basename(item).split("_")[0]
    values = {
        "spkid": str(phone),
        "wav_url": wav_url,
        "register_date": '20230406',
    }
    try:
        resp = requests.request("POST", url, files=request_file, data=values)
        print(resp.json())
    except Exception as e:
        print(e)
        return


def get_feature():
    r = redis.Redis(
        host=cfg.REDIS["host"],
        port=cfg.REDIS["port"],
        db=2,
        password=cfg.REDIS["password"],
    )
    for i in r.keys():
        v = r.get(i)
        image_array = np.frombuffer(v, dtype=np.float32, offset=8)
        np.save(f"{i}_all.npy", image_array)


def load_npy(file_name):
    s = numpy.load(file_name)
    print(s.shape)
    print(s)
    return torch.Tensor(s)


def vad_req(item, mode='file'):
    """
    VAD api
    Args:
        item:
        mode:

    Returns:

    """
    if mode == 'url':
        spkid = '2'
        minio_url = f'http://192.168.3.202:9000/black-raw/{spkid}.wav'
        values = {
            "spkid": spkid,
            "wav_url": minio_url,
        }
        url = f"http://{args.ip}:{args.port}/vad/{mode}"
        resp = requests.request("POST", url, data=values)
        print(resp.json())
    else:
        phone = os.path.basename(item).split("_")[0]
        values = {
            "spkid": str(phone),
        }
        url = f"http://{args.ip}:{args.port}/vad/{mode}"
        request_file = {"wav_file": open(item, "rb")}
        resp = requests.request("POST", url, files=request_file, data=values)
        print(resp.json())


if __name__ == "__main__":
    # get_feature()
    # d1 = load_npy("b'999_13115961957'.npy")
    # d2 = load_npy("b'999_13115961957'_all.npy")
    # from utils.encoder import similarity
    #
    # score = similarity(d1, d2)
    # print(score)
    file_list = glob.glob("./test_data/*/*_1.wav")
    file_list = [file_list[0] for i in range(100)]
    # for file in file_list:
    #     # vad_req(file, mode='url')
    #     test(file)
    #     # register(file)
    #     break
    # 多线程运行
    pool = multiprocessing.pool.ThreadPool(processes=10)
    pool.map(test, file_list)
    pool.close()
    pool.join()
