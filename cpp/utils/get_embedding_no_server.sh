#!/bin/bash

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--fold_path', type=str, default='/datasets_hdd/datasets/cjsd_vad_0.1_0.1/',help='After vad data path')
# parser.add_argument('--dst_path', type=str, default="/datasets_hdd/datasets/cjsd_0101_embeddings_ecapatdnn_16k",help='Path for output embedding npy files')
# parser.add_argument('--worker_index', type=int, default=1,help='')
# parser.add_argument('--total_workers', type=int, default=80,help='')
# parser.add_argument('--emb_dim', type=int, default=192,help='')
# parser.add_argument('--device', type=str, default="cuda:3",help='')
# args = parser.parse_args()

# 调用 python get_embedding_no_server.py
# 一共起24个进程
# python get_embedding_no_server.py --worker_index {i} --total_workers 24 --device "cuda:{j}"
# 其中i为0~23，j为0~7
for i in {0..47}
do
    cuda_index=$((i%8))
    CUDA_VISIBLE_DEVICES=${cuda_index} python get_embedding_no_server.py --worker_index ${i} --total_workers 48 &
    # echo "python get_embedding_no_server.py --worker_index ${i} --total_workers 24 --device cuda:${cuda_index}"
done
