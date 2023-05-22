import multiprocessing
import re
import subprocess
import time
import os
from datetime import datetime
import requests
from logs import Logger
import config as cfg
import glob
from pydub import AudioSegment

SIMILAR_SCORE = 0.78

dt = datetime.now().strftime('%Y-%m-%d')
LOG = Logger(f'{dt}_auto_register.log', level='debug').logger

MINIO_HOST = cfg.MINIO_HOST
REQ_API_HOST = cfg.REQ_API_HOST
REGISTER_URL = f'{REQ_API_HOST}/register/url'

BUCKETS_NAME_BLACK = cfg.BUCKETS_NAME_BLACK
WAV_PATH_BLACK = '/home/recbak/file/20221202'


# 本地linux的minio名称 -> 可在linux上面的 ~/.config/rclone/rclone.conf 文件中查看


def rclone_job(file):
    """
    rclone sync and register
    Args:
        file:

    Returns:

    """
    LOG.info(f'Start to process file:{file}')
    command = f"rclone sync {file} minio:/{BUCKETS_NAME_BLACK}"
    subprocess.call(command, shell=True)
    file_name = os.path.basename(file)
    payload = {
        'spkid': file_name.split('.')[0].replace('_', ''),
        'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_BLACK}/{file_name}',
        "register_date": '10000',
    }
    response = requests.request("POST", REGISTER_URL, data=payload)
    if not response.ok:
        LOG.error(f'{file} request failed. Response info:{response.text}')
    else:
        LOG.info(f'File:{file},Response info:{response.text}')


def get_similarity(file_01, file_02):
    """
    获取相似度得分
    :param wav_01_url:
    :param wav_02_url:
    :return:
    """
    # return 1
    url = "http://106.14.148.126:8185/similarity/file"
    payload = {
        'spkid_1': 'spkid_1',
        'spkid_2': 'spkid_2'
    }
    wav_request_file = {
        "wav_file_1": open(file_01, 'rb'),
        "wav_file_2": open(file_02, 'rb')
    }
    resp = requests.request("POST", url=url, data=payload, files=wav_request_file)
    ret = resp.json()
    return ret['similarity']


def get_joint_wav(wav_list):
    """
     拼接多个音频
    :param wav_list:
    :return: 拼接后音频路径
    """
    playlist = AudioSegment.empty()
    for wav in wav_list:
        playlist += AudioSegment.from_wav(wav)
    output_name = wav_list[0].replace('.wav', '_joint.wav')
    playlist.export(output_name, format="wav")
    return output_name


def register_voiceprint(file):
    """
    注册音频接口
    :param joint_wav_url:
    :return:
    """

    LOG.info(f'Register file:{file}')
    command = f"rclone sync {file} minio:/{BUCKETS_NAME_BLACK}"
    subprocess.call(command, shell=True)
    file_name = os.path.basename(file)
    payload = {
        'spkid': file_name.split('.')[0].replace('_', ''),
        'wav_url': f'{MINIO_HOST}/{BUCKETS_NAME_BLACK}/{file_name}'
    }
    response = requests.request("POST", REGISTER_URL, data=payload)
    if not response.ok:
        LOG.error(f'{file} request failed. Response info:{response.text}')
    else:
        LOG.info(f'File:{file},Response info:{response.text}')


def main_local():
    wav_list = glob.glob('../wavs_dir' + '/*')
    for i in wav_list:
        i = '/s/s/c_j_20221111111111_18136655705_o.wav'
        phone_num = re.findall('1\d{10}', os.path.basename(i))[0]
        phone_num = '8041'
        slice_wav_list = glob.glob('../wavs_dir' + f'/*{phone_num}*.wav')
        if len(slice_wav_list) > 1:
            for wav in slice_wav_list:
                similarity_wav_list = [wav]
                wav_base = wav
                check_wav_list = slice_wav_list
                check_wav_list.remove(wav_base)
                for j in check_wav_list:
                    score = get_similarity(wav_base, j)
                    if score > SIMILAR_SCORE:
                        similarity_wav_list.append(j)
                if len(similarity_wav_list) > 1:
                    joint_wav = get_joint_wav(similarity_wav_list)
                    register_voiceprint(joint_wav)
                else:
                    register_voiceprint(wav_base)

                [slice_wav_list.remove(k) for k in similarity_wav_list]
        else:
            register_voiceprint(i)


def main_bak():
    """
    主函数
    Returns:

    """
    wav_list = glob.glob(WAV_PATH_BLACK + '/*')
    for i in wav_list:
        phone_num = re.findall('1\d{10}', os.path.basename(i))[0]
        slice_wav_list = glob.glob(WAV_PATH_BLACK + f'/*{phone_num}*.wav')
        if len(slice_wav_list) > 1:
            for wav in slice_wav_list:
                similarity_wav_list = [wav]
                wav_base = wav
                check_wav_list = slice_wav_list
                check_wav_list.remove(wav_base)
                for j in check_wav_list:
                    score = get_similarity(wav_base, j)
                    if score > SIMILAR_SCORE:
                        similarity_wav_list.append(j)
                if len(similarity_wav_list) > 1:
                    joint_wav = get_joint_wav(similarity_wav_list)
                    register_voiceprint(joint_wav)
                else:
                    register_voiceprint(wav_base)

                [slice_wav_list.remove(k) for k in similarity_wav_list]
        else:
            register_voiceprint(i)


def main():
    wav_list = glob.glob(WAV_PATH_BLACK + '/*.wav')
    wav_list = [
        '/mnt/xuekx/workplace/voiceprint-recognition-system/src/api_test/test_data/13003661007/13003661007_1.wav']
    for i in wav_list:
        rclone_job(i)

    # pool = multiprocessing.Pool(4)
    # pool.map(rclone_job, wav_list)


if __name__ == "__main__":
    LOG.info(f'Start! Dir is:{WAV_PATH_BLACK}')
    import time

    t1 = time.time()
    main()
    LOG.info(f'Call time:{time.time() - t1}')
