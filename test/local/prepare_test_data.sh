#!/bin/bash

# Copyright (c) 2023 Yafeng Chen (chenyafeng.cyf@alibaba-inc.com)
#               2023 Luyao Cheng (shuli.cly@alibaba-inc.com)
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

stage=4
stop_stage=4

. utils/parse_options.sh || exit 1

three_speaker_dir=/home/duanyibo/dyb/test_model/3D-speaker
voxceleb_dir=/datasets/voxceleb1

#准备voxceleb1 scp utt2spk spk2utt

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # test
  test_base_path=${three_speaker_dir}/files #/home/duanyibo/dyb/test_model/3D-speaker
  test_rawdata_path=${three_speaker_dir}/test_files #/home/duanyibo/dyb/test_model/3D-speaker/test_files
  mkdir -p $test_base_path
  if [ ! -d ${three_speaker_dir}/3D_test_files ]; then
    awk -v base_path="${three_speaker_dir}/" '{print $1" "base_path $2}' ${three_speaker_dir}/files/test_wav.scp > ${test_base_path}/all_wav.scp
    cp ${test_base_path}/test_utt2info.csv ${test_base_path}/utt2info.csv
    cp ${test_base_path}/test_utt2spk ${test_base_path}/all_utt2spk
    grep -v "Device09" ${test_base_path}/all_wav.scp > ${test_base_path}/wav.scp
    grep -v "Device09" ${test_base_path}/all_utt2spk > ${test_base_path}/utt2spk
    ./utils/utt2spk_to_spk2utt.pl ${test_base_path}/utt2spk >${test_base_path}/spk2utt
  fi
  ## trials
  if [ ! -d ${test_base_path}/trials ]; then
    mkdir -p ${test_base_path}/trials
    cp ${three_speaker_dir}/files/trials_cross_device ${test_base_path}/trials/trials_cross_device
    cp ${three_speaker_dir}/files/trials_cross_distance ${test_base_path}/trials/trials_cross_distance
    cp ${three_speaker_dir}/files/trials_cross_dialect ${test_base_path}/trials/trials_cross_dialect
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then  
  if [ ! -d ${voxceleb_dir}/wav.scp ]; then
    find ${voxceleb_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${voxceleb_dir}/wav.scp
    awk '{print $1}' ${voxceleb_dir}/wav.scp | awk -F "/" '{print $0,$1}' >${voxceleb_dir}/utt2spk
    ./utils/utt2spk_to_spk2utt.pl ${voxceleb_dir}/utt2spk >${voxceleb_dir}/spk2utt
  fi
  if [ ! -d ${voxceleb_dir}/trials ]; then
    echo "Download trials for vox1 ..."
    mkdir -p ${voxceleb_dir}/trials
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt -O ${voxceleb_dir}/trials/vox1-O.txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard.txt -O ${voxceleb_dir}/trials/vox1-H.txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all.txt -O ${voxceleb_dir}/trials/vox1-E.txt
    wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -O ${voxceleb_dir}/trials/vox1-O\(cleaned\).txt
    wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt -O ${voxceleb_dir}/trials/vox1-H\(cleaned\).txt
    wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt -O ${voxceleb_dir}/trials/vox1-E\(cleaned\).txt
    # transform them into kaldi trial format
    awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${voxceleb_dir}/trials/vox1-O\(cleaned\).txt >${voxceleb_dir}/trials/vox1_O_cleaned.trial
    awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${voxceleb_dir}/trials/vox1-H\(cleaned\).txt >${voxceleb_dir}/trials/vox1_H_cleaned.trial
    awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${voxceleb_dir}/trials/vox1-E\(cleaned\).txt >${voxceleb_dir}/trials/vox1_E_cleaned.trial
  fi
  
  echo "Data Preparation Success !!!"
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then  
  rawdata_dir=/home/duanyibo/dyb/test_model/cnceleb
  combine_short_audio=1
  if [ ${combine_short_audio} -eq 1 ];then
    echo "combine short audios and convert flac to wav ..."
    bash local/comb_cn1_cn2.sh --cnceleb1_audio_dir ${rawdata_dir}/CN-Celeb_flac/data/ \
                                --cnceleb2_audio_dir ${rawdata_dir}/CN-Celeb2_flac/data/ \
                                --min_duration 5 \
                                --get_dur_nj 60 \
                                --statistics_dir ${rawdata_dir}/statistics \
                                --store_data_dir ${rawdata_dir}
    echo "convert success"
  else
    echo "convert flac to wav ..."
    python local/flac2wav.py \
        --dataset_dir ${rawdata_dir}/CN-Celeb_flac \
        --nj 16

    # python local/flac2wav.py \
    #     --dataset_dir ${rawdata_dir}/CN-Celeb2_flac \
    #     --nj 16
    echo "convert success"
  fi
  echo "Prepare wav.scp for each dataset ..."
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

  export LC_ALL=C # kaldi config
  cnceleb_dir=/home/duanyibo/dyb/test_model/cnceleb
  cnceleb_trials=/home/duanyibo/dyb/test_model/cnceleb_files
  mkdir -p $cnceleb_trials
  echo "Prepare data for testing ..."
  find ${cnceleb_dir}/CN-Celeb_wav/eval -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${cnceleb_trials}/eval/wav.scp
  awk '{print $1}' ${cnceleb_trials}/eval/wav.scp | awk -F "[/-]" '{print $0,$2}' >${cnceleb_trials}/eval/utt2spk

  echo "Prepare data for enroll ..."
  awk '{print $0}' ${cnceleb_dir}/CN-Celeb_flac/eval/lists/enroll.map | \
    awk -v p=${cnceleb_dir}/CN-Celeb_wav/data '{for(i=2;i<=NF;i++){print $i, p"/"$i}}' >${cnceleb_trials}/eval/enroll.scp
  cat ${cnceleb_trials}/eval/enroll.scp >>${cnceleb_trials}/eval/wav.scp
  awk '{print $1}' ${cnceleb_trials}/eval/enroll.scp | awk -F "/" '{print $0,$1"-enroll"}' >>${cnceleb_trials}/eval/utt2spk
  cp ${cnceleb_dir}/CN-Celeb_flac/eval/lists/enroll.map ${cnceleb_trials}/eval/enroll.map

  echo "Prepare evalution trials ..."
  mkdir -p ${cnceleb_trials}/eval/trials
  # CNC-Eval-Avg.lst
  awk '{if($3==0)label="nontarget";else{label="target"}; print $1,$2,label}' ${cnceleb_dir}/CN-Celeb_flac/eval/lists/trials.lst >${cnceleb_trials}/eval/trials/CNC-Eval-Avg.lst
  # CNC-Eval-Concat.lst
  python local/format_trials_cnceleb.py \
    --cnceleb_root ${cnceleb_dir}/CN-Celeb_flac \
    --dst_trl_path ${cnceleb_trials}/eval/trials/CNC-Eval-Concat.lst

  echo "Success !!! Now data preparation is done !!!"
fi
