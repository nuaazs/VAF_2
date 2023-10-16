#!/bin/bash
wav_path=/datasets/fbank
gpus="0"
#save_path
exp="resnet152_0912"
# ============================================================================================================
save_path=/result/$exp/embedding
mkdir -p $save_path
bin_path=/result/$exp/bin
mkdir -p $bin_path

# models="campp_voxcelebphone_longyuan ecapatdnn1024_voxcelebphone_longyuan eres2netbase_voxceleb"
models="resnet152_lm"

for model in $models; do
    mkdir -p $save_path/$model
    for gpu in $gpus; do
        echo "$gpu"
        CUDA_VISIBLE_DEVICES=$gpu python get_embedding.py --wav_path $wav_path --index $gpu --save_path $save_path/$model --model $model --exp $exp 
    done
    mkdir -p $bin_path/$model
    python npy2bin.py --npy_path $save_path/$model --bin_path $bin_path/$model
done
