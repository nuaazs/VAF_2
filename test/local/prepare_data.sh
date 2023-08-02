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

stage=3
stop_stage=3
data=/datasets_hdd/datasets/3dspeaker_16k_phone/

. utils/parse_options.sh || exit 1

download_dir=${data}/download_data
rawdata_dir=${data}

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   echo "Download musan.tar.gz, rirs_noises.zip, train.tar.gz test.tar.gz 3dspeaker_files.tar.gz"
#   echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

#   ./local/download_data.sh --download_dir ${download_dir}
# fi

# if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#   echo "Decompress all archives ..."
#   echo "This could take some time ..."

#   for archive in musan.tar.gz rirs_noises.zip train.tar.gz test.tar.gz 3dspeaker_files.tar.gz; do
#     [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
#   done
#   [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  # if [ ! -d ${rawdata_dir}/musan ]; then
  #   tar -xzvf ${download_dir}/musan.tar.gz 
  #   # -C ${rawdata_dir}
  # fi

#   if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
#     unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
#   fi

#   if [ ! -d ${rawdata_dir}/3dspeaker ]; then
#     mkdir -p ${rawdata_dir}/3dspeaker
#     mkdir -p ${rawdata_dir}/3dspeaker/test ${rawdata_dir}/3dspeaker/train ${rawdata_dir}/3dspeaker/files
#     tar -zxvf ${download_dir}/train.tar.gz -C ${rawdata_dir}/3dspeaker/
#     tar -xzvf ${download_dir}/test.tar.gz -C ${rawdata_dir}/3dspeaker/
#     tar -xzvf ${download_dir}/3dspeaker_files.tar.gz -C ${rawdata_dir}/3dspeaker/
#   fi

#   echo "Decompress success !!!"
# fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for 3dspeaker datasets"
  export LC_ALL=C # kaldi config

  mkdir -p ${download_dir}/musanfile ${download_dir}/rirsfile 
  # musan
  find /datasets/ADD2023/musan/noise/free-sound -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${download_dir}/musanfile/wav.scp
  # rirs
  awk '{print $5}' /datasets/ADD2023/RIRS_NOISES/real_rirs_isotropic_noises/rir_list | xargs -I {} echo {} ${download_dir}/{} > ${download_dir}/rirsfile/wav.scp
  # # 3dspeaker
  # base_path=${data}/3dspeaker/
  ## train
  train_base_path=/datasets_hdd/datasets/3dspeaker_16k_phone/train
  train_rawdata_path=/datasets_hdd/datasets
  # mkdir -p $train_base_path
  awk -v base_path="/datasets_hdd/datasets/3dspeaker_16k_phone/" '{print $1" "base_path $2}' ${train_rawdata_path}/3dspeaker_16k_phone/files/train_wav.scp > ${train_base_path}/all_wav.scp
  cp ${train_rawdata_path}/3dspeaker_16k_phone/files/train_utt2info.csv ${train_base_path}/utt2info.csv
  cp ${train_rawdata_path}/3dspeaker_16k_phone/files/train_utt2spk ${train_base_path}/all_utt2spk
  grep -v "Device09" ${train_base_path}/all_wav.scp > ${train_base_path}/wav.scp
  grep -v "Device09" ${train_base_path}/all_utt2spk > ${train_base_path}/utt2spk
  ./utils/utt2spk_to_spk2utt.pl ${train_base_path}/utt2spk > ${train_base_path}/spk2utt

  ## test
  # test_base_path=${base_path}/test
  # test_rawdata_path=${rawdata_dir}/3dspeaker/
  # mkdir -p $test_base_path
  # awk -v base_path="${test_rawdata_path}" '{print $1" "base_path $2}' ${rawdata_dir}/3dspeaker/files/test_wav.scp > ${test_base_path}/all_wav.scp
  # cp ${rawdata_dir}/3dspeaker/files/test_utt2info.csv ${test_base_path}/utt2info.csv
  # cp ${rawdata_dir}/3dspeaker/files/test_utt2spk ${test_base_path}/all_utt2spk
  # grep -v "Device09" ${test_base_path}/all_wav.scp > ${test_base_path}/wav.scp
  # grep -v "Device09" ${test_base_path}/all_utt2spk > ${test_base_path}/utt2spk
  # ./utils/utt2spk_to_spk2utt.pl ${test_base_path}/utt2spk >${test_base_path}/spk2utt

  # ## trials
  # mkdir -p ${base_path}/trials
  # cp ${rawdata_dir}/3dspeaker/files/trials_cross_device ${base_path}/trials/trials_cross_device
  # cp ${rawdata_dir}/3dspeaker/files/trials_cross_distance ${base_path}/trials/trials_cross_distance
  # cp ${rawdata_dir}/3dspeaker/files/trials_cross_dialect ${base_path}/trials/trials_cross_dialect
  
  echo "Data Preparation Success !!!"
fi
