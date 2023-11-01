#!/bin/bash

start=$1
step=$2

dir="/datasets/cjsd_upload_test"

files=()

for file in "$dir"/*; do
  if [ -f "$file" ]; then
    files+=("$file")
  fi
done
#echo ${files}

length=${#files[@]}
echo "数组长度：$length"

for entry in "${files[@]:$start:$step}"; do
  echo "$entry"
  rclone sync $entry minio:gray-raw/
done




