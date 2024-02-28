#!/bin/bash
# @Time    : 2023-03-16  15:32:05
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: .

# Usage: merge_check_wav_files_multi.sh <dst_path> <cache_folder> <file extension> <process_num>
# Example: merge_check_wav_files_multi.sh ./output cache wav 20

# Help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$1" == "--h" ] || [ "$1" == "-help" ]; then
    echo "Usage: merge_check_wav_files_multi.sh <src_folder> <dst_path> <cache_folder> <process_num>"
    echo "Example: merge_check_wav_files_multi.sh /ssd2/cti_aftervad_train_data ./output cache 20"
    echo "If dst_path="", means no output"
    exit 0
fi

# Check if the number of arguments is correct
if [ $# -ne 4 ]; then
    echo "Usage: check_wav_files_multi.sh <src_folder> <dst_path> <cache_folder> <process_num>"
    exit 1
fi
# Check if src path exists
if [ ! -d "$1" ]; then
    echo "Error: src path does not exist"
    exit 1
fi


# Check if dst path exists
if [ ! -d "$2" ]; then
    echo "Error: dst path does not exist"
    exit 1
fi

# Check if the cache_folder exists
# if not exists, create it
if [ ! -d "$3" ]; then
    # return error
    echo "Error: cache_folder does not exist"
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
echo "src folder: $1"
echo "dst folder: $2"
echo "cache folder: $3"
echo "process_num: $4"
echo "==========================="

for process_idx in $(seq 1 $process_num)
do
    python merge_check_wav_files_multi.py --process_id ${process_idx} --process_num $process_num --cache_folder $3 --dst_folder $2 --src_folder $1 &
done