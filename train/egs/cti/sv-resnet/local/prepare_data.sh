#!/bin/bash
# This code incorporates a significant amount of code adapted from the following open-source projects: 
# alibaba-damo-academy/3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)  
# and wenet-e2e/wespeaker (https://github.com/wenet-e2e/wespeaker).
# We have extensively utilized the outstanding work from these repositories to enhance the capabilities of our project.
# For specific copyright and licensing information, please refer to the original project links provided.
# We express our gratitude to the authors and contributors of these projects for their 
# invaluable work, which has contributed to the advancement of this project.

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

stage=-1
stop_stage=-1
data=data

. tools/parse_options.sh || exit 1

data=`realpath ${data}`
download_dir=${data}/download_data
rawdata_dir=${data}/raw_data

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip."
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi
  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  mkdir -p ${data}/musan ${data}/rirs ${data}/cjsd8000 ${data}/cjsd1000
  # musan
  echo "Prepare musan wav.scp ..."
  echo "Looking for musan wav files in ${rawdata_dir}/musan ..."
  real_musan_path=`realpath ${rawdata_dir}/musan`
  find ${real_musan_path} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/musan/wav.scp
  # rirs
  echo "Prepare rirs wav.scp ..."
  echo "Looking for rirs wav files in ${rawdata_dir}/RIRS_NOISES/simulated_rirs ..."
  real_RIRS_NOISES_path=`realpath ${rawdata_dir}/RIRS_NOISES`
  find ${real_RIRS_NOISES_path} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >${data}/rirs/wav.scp
  
  # cjsd1000 # test data
  echo "Prepare cjsd1000 wav.scp, utt2spk, spk2utt, and trials ..."
  echo "Looking for cjsd1000 wav files in ${rawdata_dir}/cjsd1000 ..."
  real_cjsd1000_path=`realpath ${rawdata_dir}/cjsd1000`
  find ${real_cjsd1000_path} -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/cjsd1000/wav.scp
  awk '{print $1}' ${data}/cjsd1000/wav.scp | awk -F "/" '{print $0,$1}' >${data}/cjsd1000/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/cjsd1000/utt2spk >${data}/cjsd1000/spk2utt
  if [ ! -d ${data}/cjsd1000/trials ]; then
    echo "Download trials for cjsd1000 ..."
    mkdir -p ${data}/cjsd1000/trials
    wget --no-check-certificate https://shengbucket.oss-cn-hangzhou.aliyuncs.com/files/cjsd1000_normal.kaldi -O ${data}/cjsd1000/trials/cjsd1000_normal.kaldi
  fi
  # vox2
  echo "Prepare cjsd8000 wav.scp, utt2spk, and spk2utt ..."
  echo "Looking for cjsd8000 wav files in ${rawdata_dir}/cjsd8000 ..."
  real_cjsd8000_wav_path=`realpath ${rawdata_dir}/cjsd8000`
  find ${real_cjsd8000_wav_path} -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >${data}/cjsd8000/wav.scp
  awk '{print $1}' ${data}/cjsd8000/wav.scp | awk -F "/" '{print $0,$1}' >${data}/cjsd8000/utt2spk
  ./tools/utt2spk_to_spk2utt.pl ${data}/cjsd8000/utt2spk >${data}/cjsd8000/spk2utt

  echo "Success !!!"
fi
