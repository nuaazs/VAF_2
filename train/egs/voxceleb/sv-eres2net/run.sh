# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

#!/bin/bash
# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Modified by: zhaosheng@nuaa.edu.cn 2024-03-08

set -e
. ./path.sh || exit 1

stage=3
stop_stage=8

data=data
exp=exp
exp_name=vox12_eres2net34_aug
data_type=shard
num_avg=10
gpus=[0,1,2,3,4,5,6,7]
checkpoint=
config=/VAF/train/egs/voxceleb/sv-eres2net/conf/eres2net34_aug_wenet.yaml
lm_config=/VAF/train/egs/voxceleb/sv-eres2net/conf/eres2net34_aug_wenet_lm.yaml
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300
. utils/parse_options.sh || exit 1

exp_dir=$exp/$exp_name

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # In this stage we prepare the raw datasets, including Voxceleb1 and Voxceleb2.
  echo "Stage 1: Preparing Voxceleb dataset ..."
  ./local/prepare_data.sh --stage 1 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # In this stage we convert the raw data to shard or raw list.
  echo "Stage 2: Covert train and test data to ${data_type} ..."
  for dset in vox2_dev vox1; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 32 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # In this stage we train the speaker model.
  echo "Stage 3: Training the speaker model ..."
  # 3D-Speaker Trainer
  # num_gpu=$(echo $gpus | awk -F ' ' '{print NF}')
  # torchrun --nproc_per_node=$num_gpu dguard/bin/train.py --config conf/cam++.yaml --gpus $gpus \
  #          --data $data/vox2_dev/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir

  # Wenet Trainer
  echo "Start training (wespeaker-mode)..."
  # add dguard/wespeaker to PYTHONPATH
  # export PYTHONPATH=$PYTHONPATH:$(pwd)/dguard
  export PYTHONPATH=../../../dguard:$PYTHONPATH
  echo "GPUS: $gpus"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    dguard/bin/wenet_trainer.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # In this stage we average the model and extract embeddings.
  echo "Stage 4: Do model average ..."
  python dguard/bin/average.py \
    --dst_model $exp_dir/models/avg_model.pt \
    --src_path $exp_dir/models \
    --num ${num_avg}

  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $exp_dir/models/avg_model.pt \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # In this stage we score the model.
  echo "Stage 5: Scoring ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # In this stage we normalize the scores.
  echo "Stage 6: Score normalization ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # In this stage we export the final model.
  echo "Stage 7: Exporting the final model ..."
  python dguard/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # In this stage we fine-tune the model with large margin.
  echo "Stage 8: Fine-tuning the model ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
