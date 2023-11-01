#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   vad_handler.py
@Time    :   2023/10/14 16:52:59
@Author  :   Carry
@Version :   1.0
@Desc    :   None
'''
import os
from loguru import logger
from tools.cmd_handler import run_cmd


def energybase_vad(filepath, receive_path, smooth_threshold=0.5, min_duration=2, energy_thresh=1e8):
    """
    energybase vad
    Args:
        filepath (string): wav file path.
        receive_path (string): after vad save path
        smooth_threshold (float): smooth threshold
        min_duration (float): min duration
        energy_thresh (float): energy threshold
    Returns:
        string: file path
        float: after vad voice length
        list: voice time list (seconds)

     ffmpeg: 这是 FFmpeg 命令行工具的执行命令。它用于处理音视频文件。
        -i {filepath}: 这个选项后跟着输入文件的路径。-i 表示输入，{filepath} 应该被替换为实际的音频文件的路径。这是转码操作的源文件。
        -f s16le: 这个选项指定了输出文件的音频格式。在这里，s16le 表示使用 16位有符号的小端格式。
        -acodec pcm_s16le: 这个选项指定了音频编解码器，以确保音频以 16位有符号的小端格式编码。pcm_s16le 是无损音频编码。
        -ar 16000: 这个选项指定了输出音频文件的采样率，即每秒的采样点数。在这里，音频将被设置为 16,000 Hz（16 kHz）的采样率。
        -map_metadata -1: 这个选项用于删除输出文件的元数据信息，即不复制源文件的元数据到输出文件中。-1 表示删除所有元数据。
        -y: 这个选项用于强制覆盖目标文件，如果目标文件已经存在的话。不加这个选项时，FFmpeg 会询问是否覆盖。
    """
    os.makedirs(receive_path, exist_ok=True)
    bin_path = filepath.replace(os.path.basename(filepath).split('.')[-1], "bin")
    bin_path = os.path.join(receive_path, bin_path)
    run_cmd(f"ffmpeg -i {filepath} -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y {bin_path}")

    vad_output_path = f"{receive_path}/vad_{os.path.basename(filepath).split('.')[0]}.wav"
    if os.path.exists(f"{receive_path}/output.txt"):
        os.remove(f"{receive_path}/output.txt")
    run_cmd(f'./vad_wav --wav-bin={bin_path} --energy-thresh={energy_thresh} --text-out={receive_path}/output.txt --smooth-threshold={smooth_threshold} --min-duration={min_duration} --wav-out={vad_output_path}')
    voice_length = 0
    time_list = []
    if not os.path.exists(f"{receive_path}/output.txt"):
        logger.error(f"{receive_path}/output.txt file not exist.")
        return vad_output_path, voice_length, time_list
    
    with open(f"{receive_path}/output.txt", "r") as f:
        for line in f.readlines():
            start, end = line.strip().split(",")
            voice_length += float(end) - float(start)
            time_list.append([float(start), float(end)])
    return vad_output_path, voice_length, time_list
