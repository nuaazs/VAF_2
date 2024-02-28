#!/bin/bash
for i in {1..8}
do
    # calc cuda_num from i, cuda_num = i%4
    cuda_num=$((i%2))
    echo "process_id=${i}, cuda_num=${cuda_num}"
    CUDA_VISIBLE_DEVICES=${cuda_num} python vad_folder_multi.py --process_id=${i} --process_num 8 &
done

# CUDA_VISIBLE_DEVICES=0 python vad_folder_multi.py --process_id=1 --process_num 8