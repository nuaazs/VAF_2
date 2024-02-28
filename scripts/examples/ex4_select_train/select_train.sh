#!/bin/bash
for i in {1..8}
do
    # calc cuda_num from i, cuda_num = i%4
    cuda_num=$((i%2))
    echo "process_id=${i}, cuda_num=${cuda_num}"
    CUDA_VISIBLE_DEVICES=${cuda_num} python select_train_wavs.py --process_id=${i} --process_num 8 &
done