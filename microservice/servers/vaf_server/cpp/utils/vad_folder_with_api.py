# coding = utf-8
# @Time    : 2023-05-14  22:32:05
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 基于后端API接口，获取某个目录下所有wav文件，并保存vad后音频至另一目录.

import requests
from tqdm import tqdm
import os
import wget

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fold_path', type=str, default='/datasets_hdd/datasets/cjsd_download',help='Folder of raw wav files')
parser.add_argument('--dst_path', type=str, default="/datasets_hdd/datasets/cjsd_vad_0.1_0.1",help='Folder to save vad wav files')
parser.add_argument('--thread', type=int, default=4,help='Thread number, same as the number of API server')
parser.add_argument('--url', type=str, default="http://127.0.0.1:8888/test/file",help='API server url')
args = parser.parse_args()

def get_file(file_path,fold_path=args.fold_path,savepath=args.dst_path):
    """对该音频文件进行VAD，保存为wav文件
    Args:
        file_path (str): 文件路径
    Returns:
        None
    """
    # ID format, 视情况修改
    filename = file_path.split('/')[-1]
    # save path
    # 保存路径保持与原始文件相同的格式，方便后续处理
    # 比如原始文件为：/datasets_hdd/datasets/cjsd_download/0001/0001_0001_0001_0001.wav ，
    # fold_path为/datasets_hdd/datasets/cjsd_download ，
    # savepath为/datasets_hdd/datasets/cjsd_vad_0.1_0.1
    # 则保存为：/datasets_hdd/datasets/cjsd_vad_0.1_0.1/0001/0001_0001_0001_0001.wav
    rel_path = os.path.relpath(file_path, fold_path)
    print(rel_path)
    save_dir = os.path.join(savepath, os.path.dirname(rel_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    wav_path = os.path.join(savepath, rel_path)
    print(wav_path)

    # print(file_path)
    payload={"spkid":str(filename),"only_vad":1}
    files=[
    ('wav_file',(file_path,open(file_path,'rb'),'application/octet-stream'))
    ]
    if os.path.exists(os.path.join(args.dst_path,filename+".wav")):
        return 1
    try:
    # print(payload)
        response = requests.request("POST", args.url, data=payload, files=files)
        print(response)
        if "output_vad_file_path" not in response.json():
            print("!!!!!!!!Error!!!!!!!!"*2)
            print(response.json())
            print("!!!!!!!!Error!!!!!!!!"*2)
            return 0
        else:
            url = response.json()["output_vad_file_path"]
            # downalod url wav to dst_path
            if savepath:
                wget.download(url, out=wav_path)
    except Exception as e:
        print("!!!!!!!!Error!!!!!!!!"*2)
        print(e)
        print("!!!!!!!!Error!!!!!!!!"*2)
        return 0
    return 1

if __name__ == "__main__":
    # make dst folder
    os.makedirs(args.dst_path,exist_ok=True)
    # get all wavs in args.fold_path, recursive
    all_wavs = []
    for phone in os.listdir(args.fold_path):
        phone_path = os.path.join(args.fold_path,phone)
        for file in os.listdir(phone_path):
            if file.endswith(".wav"):
                all_wavs.append(os.path.join(phone_path, file))
    # multi process call get_embedding
    from multiprocessing import Pool
    pool = Pool(processes=args.thread)
    url_list = list(tqdm(pool.imap(get_file, all_wavs), total=len(all_wavs)))
    pool.close()
    pool.join()
print("Done!")
print(f"* Total accecpt #{len(all_wavs)} files, success #{len([_r for _r in url_list if _r>0])}")
