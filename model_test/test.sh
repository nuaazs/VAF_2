#!/bin/bash
set -e
. ./path.sh || exit 1
. utils/parse_options.sh || exit 1

models="resnet152_lm"
datasets="cti_v2_20s cti_v2"
overwrite=true
gpus="0" 
nj=1
overwrite=false
master_port=45642

for model in $models; do
    echo echo "$model start"
    echo "Model:$model Datasets:$datasets"
    for dataset in $datasets; do
        trial_path=$(ls ./trials/$dataset*.trials)
        # remove \r \t \n in trial_path
        trial_path=$(echo $trial_path | tr -d '\r\t\n')    
        mkdir -p result/$model/$dataset
        scp_path=./scp/$dataset.scp
        echo "============================="
        echo "* Trial Paths: $trial_path"
        echo "-----------------------------"
        echo "* Scp Path: $scp_path"
        echo "-----------------------------"
        if [ ! -d result/$model/$trial/embeddings ] || $overwrite; then
            torchrun --nproc_per_node=$nj  --master_port=$master_port extract_test_emb.py --exp_dir result/$model/$trial \
                                            --data $scp_path --use_gpu --gpu $gpus || echo echo "Model:$model Trials:$trial torchrun error"
        fi
        python dguard/bin/compute_score_metrics.py --enrol_data result/$model/$trial/embeddings --test_data result/$model/$trial/embeddings \
                                                        --scores_dir result/$model/$trial/scores --trials $trial_path || echo echo "Model:$model Trials:$trial compute_score_metrics error"
        echo echo "Model:$model Trials:$trial done"
        echo "============================="
    done
done