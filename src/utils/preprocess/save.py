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


def save_file(file, spk, channel, upload=False):
    """save wav file from post request.

    Args:
        file (request.file): wav file.
        spk (string): speack id
        receive_path (string): save path

    Returns:
        string: file path
        string: file url in minio
    """
    receive_path = cfg.TEMP_PATH
    spk_dir = os.path.join(receive_path, str(spk))
    os.makedirs(spk_dir, exist_ok=True)
    spk_filelist = os.listdir(spk_dir)
    speech_number = len(spk_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.webm
    pid = os.getpid()
    ext = file.filename.split('.')[-1]
    save_name = f"raw_{speech_number}_{pid}.{ext}"
    save_path = os.path.join(spk_dir, save_name)
    save_path_wav = os.path.join(spk_dir, f"raw_{speech_number}_{pid}_preprocessed.wav")
    logger.info("\t\tSave file path: {save_path}")
    file.save(save_path)
    # conver to wav
    logger.info("\t\tConver to wav.")
    end = cfg.WAV_START + cfg.WAV_LENGTH
    run_cmd(f"ffmpeg -i {save_path} -y -ss {cfg.WAV_START} -to {end} -ar {cfg.SR} -ac 1 -vn -map_channel 0.0.{channel} {save_path_wav}")
    if upload:
        url = upload_file(
            bucket_name="raw",
            filepath=save_path_wav,
            filename=f"raw_{spk}_{speech_number}_{pid}.wav",
            save_days=cfg.MINIO["test_save_days"],
        )
    else:
        url = save_path_wav
    return save_path_wav, url


def save_url(url, spk, channel, upload=False):
    """save wav file from post request.

    Args:
        file (request.file): wav file.
        spk (string): speack id
        receive_path (string): save path

    Returns:
        string: file path
    """
    receive_path = cfg.TEMP_PATH
    spk_dir = os.path.join(receive_path, str(spk))
    os.makedirs(spk_dir, exist_ok=True)
    spk_filelist = os.listdir(spk_dir)
    speech_number = len(spk_filelist) + 1
    # receive wav file and save it to  ->  <receive_path>/<spk_id>/raw_?.webm
    pid = os.getpid()
    ext = url.split('.')[-1]
    save_name = f"raw_{speech_number}_{pid}.{ext}"

    if url.startswith("local://"):
        previous_path = url.replace("local://", "")
        save_path = os.path.join(spk_dir, save_name)
        shutil.copy(previous_path, save_path)
    else:
        save_path = os.path.join(spk_dir, save_name)
        t1 = time.time()
        wget.download(url, save_path)
        t2 = time.time()
        print(f"wget time: {t2 - t1}")
    save_path_wav = os.path.join(spk_dir, f"raw_{speech_number}_{pid}_preprocessed.wav")
    # conver to wav
    end = cfg.WAV_START + cfg.WAV_LENGTH
    run_cmd(f"ffmpeg -i {save_path} -y -ss {cfg.WAV_START} -to {end} -ar {cfg.SR} -ac 1 -vn -map_channel 0.0.{channel} {save_path_wav}")
    # run_cmd(f"rm {save_path}")

    if upload:
        url_uploaded = upload_file(
            bucket_name="raw",
            filepath=save_path_wav,
            filename=f"raw_{spk}_{speech_number}_{pid}.wav",
            save_days=cfg.MINIO["test_save_days"],
        )
    else:
        url_uploaded = url
    return save_path_wav, url_uploaded
