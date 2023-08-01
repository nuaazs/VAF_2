# coding = utf-8
# @Time    : 2022-09-05  15:33:45
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Save file.

import os
import time

import wget
import shutil

import cfg
from utils.log import logger
from utils.oss import upload_file
from utils.cmd import run_cmd

def save_file(file, spk, channel=0,start=0,length=999,sr=16000, upload=False,uuid=None,server_name="all"):
    """save wav file from post request.

    Args:
        file (request.file): wav file.
        spk (string): speack id
        receive_path (string): save path

    Returns:
        string: file path
        string: file url in minio
    """
    end = start + length
    receive_path = cfg.TEMP_PATH
    spk_dir = os.path.join(receive_path, server_name,str(spk))
    os.makedirs(spk_dir, exist_ok=True)
    spk_filelist = os.listdir(spk_dir)
    speech_number = len(spk_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.webm

    ext = file.filename.split('.')[-1]
    save_name = f"{speech_number}.{ext}"
    save_path = os.path.join(spk_dir, save_name)
    if uuid:
        save_path_wav = os.path.join(spk_dir, f"raw_{uuid}.wav")
    else:
        save_path_wav = os.path.join(spk_dir, f"raw_{speech_number}.wav")
    logger.info(f"\t\tSave file path: {save_path}")
    file.save(save_path)
    # conver to wav
    logger.info("\t\tConver to wav.")
    cmd = f"ffmpeg -i {save_path} -y  -ss {start} -to {end} -ar {sr}  -ac 1 -vn -map_channel 0.0.{channel} -y  {save_path_wav}"
    # print(cmd)
    run_cmd(cmd,util_exist=save_path_wav)

    if upload:
        url = upload_file(
            bucket_name="raw",
            filepath=save_path_wav,
            filename=f"{spk}_{speech_number}.wav",
            save_days=cfg.MINIO["test_save_days"],
        )
    else:
        url = save_path_wav
    return save_path_wav, url


def save_url(url, spk, channel,start=0,length=999,sr=16000, upload=False,uuid=None,server_name="all"):
    """save wav file from post request.

    Args:
        file (request.file): wav file.
        spk (string): speack id
        receive_path (string): save path

    Returns:
        string: file path
    """
    end = start + length
    receive_path = cfg.TEMP_PATH
    spk_dir = os.path.join(receive_path,server_name, str(spk))
    os.makedirs(spk_dir, exist_ok=True)
    spk_filelist = os.listdir(spk_dir)
    speech_number = len(spk_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.webm
    ext = url.split('.')[-1]
    save_name = f"{speech_number}.{ext}"

    if url.startswith("local://"):
        previous_path = url.replace("local://", "")
        save_path = os.path.join(spk_dir, save_name)
        shutil.copy(previous_path, save_path)
    else:
        save_path = os.path.join(spk_dir, save_name)
        t1 = time.time()
        wget.download(url, save_path)
        t2 = time.time()
        # print(f"wget time: {t2 - t1}")
    if uuid:
        save_path_wav = os.path.join(spk_dir, f"raw_{uuid}.wav")
    else:
        save_path_wav = os.path.join(spk_dir, f"raw_{speech_number}.wav")
    # conver to wav
    # print(f"ffmpeg -i {save_path} -y  -ss {start} -to {end} -ar {sr}  -ac 1 -vn -map_channel 0.0.{channel} -y  {save_path_wav} > /dev/null 2>&1")
    cmd = f"ffmpeg -i {save_path} -y  -ss {start} -to {end} -ar {sr}  -ac 1 -vn -map_channel 0.0.{channel} -y  {save_path_wav}"
    # run_cmd(f"rm {save_path}")
    run_cmd(cmd,util_exist=save_path_wav)

    if upload and url.startswith("local://"): 
        url_uploaded = upload_file(
            bucket_name="raw",
            filepath=save_path_wav,
            filename=f"{spk}_{speech_number}.wav",
            save_days=cfg.MINIO["test_save_days"],
        )
    else:
        url_uploaded = url
    return save_path_wav, url_uploaded

