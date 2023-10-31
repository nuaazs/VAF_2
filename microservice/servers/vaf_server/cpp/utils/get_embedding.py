# coding = utf-8
# @Time    : 2023-05-14  22:17:20
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: 利用后端API接口，传入VAD后数据文件，获取对应文件的embedding特征，保存为numpy文件.
import requests
from tqdm import tqdm
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fold_path', type=str, default='/datasets_hdd/testdata_cnceleb_wav_phone/female/register',help='After vad data path')
parser.add_argument('--dst_path', type=str, default="/datasets_hdd/testdata_cnceleb_embedding/female/register",help='Path for output embedding npy files')
parser.add_argument('--thread', type=int, default=1,help='Thread number, same as the number of API server')
parser.add_argument('--emb_type', type=str, default="ECAPATDNN,CAMPP",help='')
parser.add_argument('--url', type=str, default="http://127.0.0.1:8888/get_embedding/file",help='API server url')
args = parser.parse_args()
emb_type_list = args.emb_type.split(',')
def get_embedding(file_path,savepath=args.dst_path):
    """获取该文件的embedding特征
    Args:
        file_path (str): 文件路径
    Returns:
        None
    """
    filename = file_path.split('/')[-1].split('.')[0]
    payload={"spkid":str(filename)}
    files=[
    ('wav_file',(file_path.split('/')[-1],open(file_path,'rb'),'application/octet-stream'))
    ]
    # if os.path.exists(os.path.join(args.dst_path,filename+"CAMPP"+".npy")):
    #     if os.path.exists(os.path.join(args.dst_path,filename+"ECAPATDNN"+".npy")):
    #         # print(f"Skip {filename}")
    #         return 1
    # try:
    response = requests.request("POST", args.url,data=payload, files=files)
    # print(response.status_code)
    # if response.status_code != 200:
    #     return 0
    for emb_type in emb_type_list:
        print(response.json())
        if emb_type not in response.json():
            return 0
        else:
            emb = np.array(response.json()[emb_type]) # shape (len_of_emb,)
            if savepath:
                output_path = os.path.join(savepath,filename+emb_type)
                np.save(output_path,emb)
                print(f"Saved {output_path}")
    # except Exception as e:
    #     return 0
    # return 1

if __name__ == "__main__":
    # make dst folder
    os.makedirs(args.dst_path,exist_ok=True)

    # get all wavs in args.fold_path, recursive
    all_wavs = []
    # find all wavs，recursive
    for root, dirs, files in os.walk(args.fold_path):
        for file in files:
            if file.endswith(".wav"):
                all_wavs.append(os.path.join(root, file))
    print(f"Total {len(all_wavs)} wav files.")
    print(all_wavs)
    

    # multi process call get_embedding
    from multiprocessing import Pool
    pool = Pool(processes=args.thread)
    embeddings_list = list(tqdm(pool.imap(get_embedding, all_wavs), total=len(all_wavs)))
    pool.close()
    pool.join()
print("Done!")
print(f"* Total accecpt #{len(all_wavs)} files, success #{len([_r for _r in embeddings_list if _r>0])}")
