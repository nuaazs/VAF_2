# coding = utf-8
# @Time    : 2023-03-16  13:38:57
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Check all wav file in folder list, and calc useful speakers then move them in another folder.
# @Version : 1.0

# Check all wav file in folder list, and calc useful speakers then move them in another folder.
# Usage: python check_wav_files.sh <wav_folder_list_name> <useful_wav_folder> <process_num>
# Example: python check_wav_files.sh wav_folder_list.txt useful_wav_folder 4


import os
import shutil
import argparse
import tqdm
import numpy as np
from pathlib import Path
import torchaudio
import subprocess
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--src_folder', type=str, default='',help='')
parser.add_argument('--dst_folder', type=str, default='',help='')
parser.add_argument('--cache_folder', type=str, default='./wav_file_cache',help='')
parser.add_argument('--wav_num_th', type=int, default=2,help='')
parser.add_argument('--wav_duration_th', type=int, default=180,help='seconds')
parser.add_argument('--process_num', type=int, default=2,help='')
parser.add_argument('--process_id', type=int, default=1,help='')

args = parser.parse_args()

def get_all_files(input_folder, extension):
    # find all files in input_folder with extension
    # even in subfolders
    # use glob.glob() to find all files
    # print('input_folder: ', input_folder)
    files = []
    for root, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
        for dir in dirs:
            files += get_all_files(os.path.join(root,dir), extension)
    return files

def get_wav_duration(wav_file):
    # get wav duration, fast
    # use sox to get duration
    # print('wav_file: ', wav_file)
    # cmd = 'sox ' + wav_file + ' -n stat 2>&1 | grep "Length (seconds)" | awk \'{print $3}\''
    # duration = subprocess.check_output(cmd, shell=True)
    # duration = float(duration)
    # use torchaudio to get duration
    # print('wav_file: ', wav_file)
    waveform, sample_rate = torchaudio.load(wav_file)
    # print('waveform.shape: ', waveform.shape)
    duration = waveform.shape[1] / sample_rate
    return duration

def get_folder_duration(wav_files,extension='.wav'):
    # get all wavs in folder, and sum the duration
    duration = 0
    for wav_file in wav_files:
        duration += get_wav_duration(wav_file)
    return duration

if __name__ == '__main__':
    # merget all dicts
    # args.useful_wav_folder + '/speaker_dict_' + str(args.process_id) + '.npy'
    # find all npys
    npy_list = sorted([_file for _file in os.listdir(args.cache_folder) if "speaker_dict_" in _file and ".npy" in _file])
    # read all npys
    all_speaker_dict = {}
    dict_list = []
    for npy_file in npy_list:
        dict_list.append(np.load(args.cache_folder + '/' + npy_file, allow_pickle=True).item())
    # merge all dicts
    for dict in dict_list:
        for key in dict.keys():
            if key in all_speaker_dict.keys():
                all_speaker_dict[key] += dict[key]
            else:
                all_speaker_dict[key] = dict[key]

    


    if args.dst_folder != '':
        # move useful wav files to dst_folder
        if not os.path.exists(args.dst_folder):
            os.makedirs(args.dst_folder)
        id_list = sorted(all_speaker_dict.keys())

        # split id_list to process_num parts
        id_list = id_list[args.process_id::args.process_num]
        

        duration_cache = []

        total_wav_num, useful_wav_num = 0, 0
        pbar=tqdm(id_list)
        for id in pbar:
            file_list = all_speaker_dict[id]
            duration = get_folder_duration(file_list)
            file_num = len(file_list)

            # cache duration,id
            duration_cache.append([duration,id])
            if file_num >= args.wav_num_th and duration >= args.wav_duration_th:
                for file in file_list:
                    savepath = file.replace(args.src_folder, args.dst_folder)
                    # get savepath's father dir
                    savepath_father_dir = os.path.dirname(savepath)
                    os.makedirs(savepath_father_dir, exist_ok=True)
                    shutil.copy(file, savepath)
                
                    
                    # update pbar info
                useful_wav_num += 1
                total_wav_num += 1
                pass_rate = useful_wav_num / total_wav_num
                pbar.set_description(f"#P:{args.process_id} Rate:{pass_rate:.2f}")
                    # print('copy ' + file + ' to ' + args.dst_folder + '/' + file.split('/')[-1])
            else:
                total_wav_num += 1
        # np save duration_cache
        np.save(args.cache_folder + '/duration_cache_' + str(args.process_id) + '.npy', duration_cache)