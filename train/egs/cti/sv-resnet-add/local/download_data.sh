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

download_dir=data/download_data

. tools/parse_options.sh || exit 1

[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

if [ ! -f ${download_dir}/musan.tar.gz ]; then
  echo "Downloading musan.tar.gz ..."
  wget --no-check-certificate https://openslr.elda.org/resources/17/musan.tar.gz -P ${download_dir}
  md5=$(md5sum ${download_dir}/musan.tar.gz | awk '{print $1}')
  [ $md5 != "0c472d4fc0c5141eca47ad1ffeb2a7df" ] && echo "Wrong md5sum of musan.tar.gz" && exit 1
fi

if [ ! -f ${download_dir}/rirs_noises.zip ]; then
  echo "Downloading rirs_noises.zip ..."
  wget --no-check-certificate https://us.openslr.org/resources/28/rirs_noises.zip -P ${download_dir}
  md5=$(md5sum ${download_dir}/rirs_noises.zip | awk '{print $1}')
  [ $md5 != "e6f48e257286e05de56413b4779d8ffb" ] && echo "Wrong md5sum of rirs_noises.zip" && exit 1
fi

echo "Download success !!!"
