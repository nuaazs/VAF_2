#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   file_processor.py
@Time    :   2023/10/14 14:50:32
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''
import os
import wget
from loguru import logger
from pydub import AudioSegment
from tools.cmd_handler import run_cmd


def get_audio_and_conver(spkid, receive_path, file_data=None, file_url=None, start=0, length=999, sr=16000, channel=0):
    """
    get audio and conver to wav
    Args:
        file_data (request.file): wav file.
        file_url (string): wav file url.
        spkid (string): speack id
        receive_path (string): save path
        start (int): start time
        length (int): length
        sr (int): sample rate
        channel (int): channel
    Returns:
        string: file path
    """
    os.makedirs(receive_path, exist_ok=True)
    save_path = os.path.join(receive_path, f"raw_{spkid}.wav")
    logger.info(f"Save file path: {save_path}")
    if file_data:
        file_data.save(save_path)
    elif file_url:
        wget.download(file_url, save_path)
    converted_save_path = os.path.join(receive_path, f"converted_{spkid}.wav")
    end = start + length
    logger.info(f"Conver to wav by ffmpeg...")
    cmd = f"ffmpeg -i {save_path} -y  -ss {start} -to {end} -ar {sr}  -ac 1 -vn -map_channel 0.0.{channel} -y  {converted_save_path}"
    run_cmd(cmd)
    return converted_save_path


def extract_audio_segment(input_file, output_file, start_time, end_time):
    """
    extract audio segment
    Args:
        input_file (string): input file
        output_file (string): output file
        start_time (int): start time (s)
        end_time (int): end time (s)
    Returns:
        string: file path
    """
    audio = AudioSegment.from_file(input_file)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    extracted_segment = audio[start_ms:end_ms]
    extracted_segment.export(output_file, format="wav")


def get_joint_wav(spkid, receive_path, wav_list):
    """
    get joint wav
    Args:
        spkid (string): speack id
        receive_path (string): save path
        wav_list (list): wav list
    Returns:
        string: file path
    """
    os.makedirs(receive_path, exist_ok=True)
    playlist = AudioSegment.empty()
    for wav in wav_list:
        playlist = playlist + AudioSegment.from_wav(wav)
    output_name = f'{receive_path}/{spkid}_joint.wav'
    playlist.export(output_name, format='wav')
    return output_name
