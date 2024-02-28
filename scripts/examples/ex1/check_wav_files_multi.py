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


import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--wav_folder_list_file', type=str, default='./wav_folder_list.txt',help='')
parser.add_argument('--cache_folder', type=str, default='./wav_file_cache',help='')
parser.add_argument('--useful_wav_folder', type=str, default='./useful_wav_folder',help='')
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


# get all wav files in folder list
def get_all_wav_files(wav_folder_list_file):
    wav_files = []
    with open(wav_folder_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            wav_files += get_all_files(line, '.wav')
    return wav_files

if __name__ == '__main__':
    # mkdir useful_wav_folder
    if not os.path.exists(args.useful_wav_folder):
        os.makedirs(args.useful_wav_folder)
    # get all wav file list in folder list, and save them to npy file (wav_file_cache).
    # if npy file exists, load it.
    wav_folder_list_file = os.path.join(args.cache_folder,"wav_file_cache.npy")
    wav_files = get_all_wav_files(args.wav_folder_list_file)
    wav_files = sorted(wav_files)
    total_length = len(wav_files)
    tiny_length = total_length // args.process_num
    wav_files = wav_files[(args.process_id-1) * tiny_length : args.process_id * tiny_length]

    # check all wav files
    # generate dict {speaker_id: [wav_file1, wav_file2, ...]}
    # and save dict to npy file
    speaker_dict = {}
    for wav_file in tqdm.tqdm(wav_files):
        # get speaker id
        speaker_id = wav_file.split('/')[-2]
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = []
        speaker_dict[speaker_id].append(wav_file)
    
    # print how many speakers, and how many wav files
    # for speaker_id in speaker_dict, print the average length of wav files
    print('Process_id: ', args.process_id)
    print('Total speakers: ', len(speaker_dict))
    total_wav_files = 0
    for speaker_id in speaker_dict:
        total_wav_files += len(speaker_dict[speaker_id])
    print('Total wav files: ', total_wav_files)

    # save speaker_dict to npy file
    np.save(args.cache_folder + '/speaker_dict_' + str(args.process_id) + '.npy', speaker_dict)