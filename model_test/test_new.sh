#!/bin/bash

# Usage: ./test.sh -c config.yaml
######################################################################################
# Step 0: Init
# Author: zhaosheng@nuaa.edu.cn
# Date: 2021/09/25
# Description: This script is used to test the model on the test set.
######################################################################################
set -e
. ./path.sh || exit 1
. utils/parse_options.sh || exit 1

# Parse arguments from command line
while getopts ":c:" opt; do
  case $opt in
    c) config_file=$OPTARG;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# Check if config file is provided
if [ -z "$config_file" ]; then
  echo "Please provide a config file using the -c option."
  exit 1
fi

# Load all parameters in <config_file>
source $config_file

echo "Read config file $config_file"
echo "EXP Name: $exp_name"

# if result/$exp_name already exists, ask if continue,input y/n/Y/N/yes/no/YES/NO
if [ -d result/$exp_name ]; then
    read -p "result/$exp_name already exists, do you want to continue? [y/n/r]" ynr
    case $ynr in
        [Yy]* ) echo "continue";;
        [Rr]* ) echo "Remove old folder" && rm -rf result/$exp_name/*;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
fi
# copy now script to result/$exp_name/run_shell_<timestr>.sh
timestr=$(date "+%Y%m%d_%H%M%S")
mkdir -p result/$exp_name
cp $0 result/$exp_name/run_shell_$timestr.sh
cp $config_file result/$exp_name/config.sh


######################################################################################
# Function to extract embeddings
######################################################################################
extract_embeddings() {
    local seed=$1
    local model=$2
    local dataset=$3
    local length=$4
    
    echo "Model: $model Datasets: $dataset"
    
    trial_path=$(ls ./trials/$dataset*.trials)
    # remove \r \t \n in trial_path
    trial_path=$(echo $trial_path | tr -d '\r\t\n')    
    mkdir -p result/$exp_name/$model/$dataset/$length
    scp_path=./scp/$dataset.scp
    echo "============================="
    echo "* Trial Paths: $trial_path"
    echo "-----------------------------"
    echo "* Scp Path: $scp_path"
    echo "-----------------------------"
    echo "* Seed: $seed"
    echo "-----------------------------"
    echo "* Length: $length"
    echo "-----------------------------"
    if [ ! -d result/$exp_name/$model/$trial/embeddings ] || $overwrite; then
        torchrun --nproc_per_node=$nj --master_port=$master_port extract_test_emb.py --seed $seed --length $length --model_name $model --exp_dir result/$exp_name/$model/$dataset/$length \
            --start_from=$start_from --data $scp_path --use_gpu --gpu $gpus || echo echo "Model:$model Trials:$dataset torchrun error"
    fi
    wait
    
    tiny_save_dir=result/$exp_name/$model/$trial/scores/tiny
    mkdir -p $tiny_save_dir
    for now_index in $(seq 0 $[$cpu_nj-1]); do
        # echo "Now index: $now_index"
        python ${DGUARD_ROOT}/bin/compute_score_metrics_multi.py --total=$cpu_nj --tiny_save_dir=$tiny_save_dir --rank=$now_index --enrol_data result/$exp_name/$model/$dataset/$length/embeddings --test_data result/$exp_name/$model/$dataset/$length/embeddings \
            --scores_all="result/$exp_name/seed_${seed}.csv" --exp_id="${model},${dataset},${length}" --scores_dir result/$exp_name/$model/$dataset/$length/scores --trials $trial_path || echo echo "Model:$model Trials:$dataset compute_score_metrics error" &
    done
    wait
    python ${DGUARD_ROOT}/bin/compute_score_metrics_merge.py --total=1 --tiny_save_dir=$tiny_save_dir --rank=0 --enrol_data result/$exp_name/$model/$dataset/$length/embeddings --test_data result/$exp_name/$model/$dataset/$length/embeddings \
        --scores_all="result/$exp_name/seed_${seed}.csv" --exp_id="${model},${dataset},${length}" --scores_dir result/$exp_name/$model/$dataset/$length/scores --trials $trial_path || echo echo "Model:$model Trials:$dataset compute_score_metrics error" &

    echo echo "Model:$model Trials:$trial Lenght:$length done!!"
    echo "============================="
}

######################################################################################
# Step 1: Extract embeddings
######################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for seed in $seeds; do
        for model in $models; do
            for dataset in $datasets; do
                for length in $lengths; do
                    extract_embeddings $seed $model $dataset $length
                done
            done
        done
    done
fi

######################################################################################
# Step 2: Get mean csv
######################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python mean_csv.py --csv_folder result/$exp_name --prefix "seed_" --output ./result/$exp_name/mean.csv
fi

######################################################################################
# Step 3: Plot
######################################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python plot.py --csv_path ./result/$exp_name/mean.csv --png_path ./result/$exp_name/mean_png/mean
fi