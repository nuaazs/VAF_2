#!/bin/bash
set -e
. ./path.sh || exit 1

stage=4
stop_stage=5

exp=/home/zhaosheng/3D-Speaker/egs/3dspeaker/sv-cam++/exp
exp_name=cam++
gpus="1"
model_stage=True
data_scp=/datasets/test/dataset_info/test_wav.scp
trial=/datasets/test/dataset_info/trail_cross
. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

echo "Extracting speaker embeddings..."
nj=8
python speakerlab/bin/extract.py --exp_dir $exp_dir \
           --data $data_scp --use_gpu --gpu $gpus

echo "Stage5: Computing score metrics..."
trials=$trial
python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_dir/embeddings --test_data $exp_dir/embeddings \
                                                 --scores_dir $exp_dir/scores --trials $trials
