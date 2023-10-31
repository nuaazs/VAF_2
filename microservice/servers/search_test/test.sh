#!/bin/bash

# 指定目录路径
directory="/datasets/cjsd_upload/"

# 指定API接口URL
api_url="http://192.168.3.199:5550/search/file"


# 使用find命令递归查找所有.wav文件，然后使用xargs并行处理
find "$directory" -type f -name "*.wav" | xargs -I {} -P 4 sh -c 'wav_file="{}"; spkid=$(basename "$wav_file" .wav); curl -X POST -F "wav_file=@$wav_file" -F "spkid=$spkid" $api_url'


# 参数说明：
# -P 4 表示并行处理的线程数，可以根据需要调整


