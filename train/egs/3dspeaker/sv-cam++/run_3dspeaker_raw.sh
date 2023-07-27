#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
. ./path.sh || exit 1

stage=2
stop_stage=5

data=3dspeaker_data_raw
exp=exp
exp_name=cam++_3dspeaker_data_raw
gpus="0 1 2 3 4 5 6 7"

. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we prepare the raw datasets, including Voxceleb1 and Voxceleb2.
  echo "Stage1: Preparing 3D Speaker dataset..."
  ./local/prepare_data.sh --stage 3 --stop_stage 3 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # In this stage we prepare the data index files for training.
  echo "Stage2: Preparing training data index files..."
  python local/prepare_data_csv.py --data_dir $data/3dspeaker/train
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Train the speaker embedding model.
  echo "Stage3: Training the speaker model..."
  num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  torchrun --master_port=25644 --nproc_per_node=$num_gpu speakerlab/bin/train.py --config conf/cam++.yaml --gpu $gpus \
           --data $data/3dspeaker/train/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Extract embeddings of test datasets.
  echo "Stage4: Extracting speaker embeddings..."
  torchrun --nproc_per_node=8 speakerlab/bin/extract.py --exp_dir $exp_dir \
           --data $data/3dspeaker/test/wav.scp --use_gpu --gpu $gpus
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Output score metrics.
  echo "Stage5: Computing score metrics..."
  trials="$data/3dspeaker/trials/trials_cross_device $data/3dspeaker/trials/trials_cross_distance $data/3dspeaker/trials/trials_cross_dialect"
  python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_dir/embeddings --test_data $exp_dir/embeddings \
                                                 --scores_dir $exp_dir/scores --trials $trials
fi
