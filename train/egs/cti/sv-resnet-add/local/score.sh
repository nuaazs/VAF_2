#!/bin/bash
# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

# Copyright (c) 2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data

stage=-1
stop_stage=-1

. tools/parse_options.sh
. path.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "apply cosine scoring ..."
  mkdir -p ${exp_dir}/scores
  trials_dir=${data}/cjsd1000/trials
  for x in $trials; do
    echo $x
    python wespeaker/bin/score.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${exp_dir}/embeddings/cjsd1000/xvector.scp \
      --cal_mean True \
      --cal_mean_dir ${exp_dir}/embeddings/cjsd8000 \
      ${trials_dir}/${x}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "compute metrics (EER/minDCF) ..."
  scores_dir=${exp_dir}/scores
  for x in $trials; do
    python wespeaker/bin/compute_metrics.py \
        --p_target 0.01 \
        --c_fa 1 \
        --c_miss 1 \
        ${scores_dir}/${x}.score \
        2>&1 | tee -a ${scores_dir}/cjsd1000_cos_result

    echo "compute DET curve ..."
    python wespeaker/bin/compute_det.py \
        ${scores_dir}/${x}.score
  done
fi
