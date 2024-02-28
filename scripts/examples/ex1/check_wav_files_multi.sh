#!/bin/bash
# @Time    : 2023-03-16  15:32:05
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: check all .<ext> files in folders.

# Usage: check_wav_files_multi.sh <wav_folder_list_txt> <cache_folder> <file extension> <process_num>
# Example: check_wav_files_multi.sh wav_folder_list.txt cache wav 20

# Help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$1" == "--h" ] || [ "$1" == "-help" ]; then
    echo "Usage: check_wav_files_multi.sh <wav_folder_list_txt> <cache_folder> <file extension> <process_num>"
    echo "Example: check_wav_files_multi.sh wav_folder_list.txt cache wav 20"
    exit 0
fi

# Check if the number of arguments is correct
if [ $# -ne 4 ]; then
    echo "Usage: check_wav_files_multi.sh <wav_folder_list_txt> <cache_folder> <file extension> <process_num>"
    exit 1
fi

# Check if the wav_folder_list_txt exists
if [ ! -f "$1" ]; then
    echo "Error: wav_folder_list_txt does not exist"
    exit 1
fi

# Check if the cache_folder exists
# if not exists, create it
if [ ! -d "$2" ]; then
    mkdir -p "$2"
fi

# Check if the file extension is valid
# file extension must be wav, flac, m4a or mp3
if [ "$3" != "wav" ] && [ "$3" != "flac" ] && [ "$3" != "m4a" ] && [ "$3" != "mp3" ]; then
    echo "Error: file extension must be wav, flac, m4a or mp3"
    exit 1
fi

# Check if the process_num is valid
# process_num must be a positive integer
if [ $4 -lt 1 ]; then
    echo "Error: process_num must be a positive integer"
    exit 1
fi


# echo "Start processing..."
process_num=$4
# Echo all the arguments
echo "==========================="
echo "wav_folder_list_txt: $1"
echo "cache_folder: $2"
echo "file extension: $3"
echo "process_num: $4"
echo "==========================="

for process_idx in $(seq 1 $process_num)
do
    python check_wav_files_multi.py --process_id=${process_idx} --process_num $process_num --wav_folder_list_file $1 --cache_folder $2 &
done