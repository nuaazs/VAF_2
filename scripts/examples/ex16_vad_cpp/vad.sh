#!/bin/bash
# ROOT_PATH="/home/zhaosheng/VAF_UTILS/utils/examples/ex16_vad_cpp"
if [ $# -ne 3 ];then
    echo "Usage: $0 <input.wav> <output.vad.wav> <output.txt>"
    exit 1
fi

# get file name from $1 and add with random number
FILE_NAME=$(basename $1)_$(date +%s%N).bin

# extract pcm data from wav file
ffmpeg -i $1 -f s16le -acodec pcm_s16le -ar 8000 -map_metadata -1 ${FILE_NAME} > /dev/null 2>&1

# apply voice activity detection
# echo "* Doing ->    ./apply-vad ${FILE_NAME} $2 $3"
apply-vad ${FILE_NAME} $2 $3
# echo "* Doing ->    rm -f ${FILE_NAME}"
# remove intermediate binary file
rm -f ${FILE_NAME}
echo "Done"

# 用途： 通过ffmpeg和apply-vad实现vad
# 通过命令行参数接收 1. wav文件地址<input.vad> 2. 保存vad后的wav文件地址<output.vad.wav>
# 实现过程
# 1. 通过ffmpeg 生成bin ： ffmpeg -i <input.vad.wav> -f s16le -acodec pcm_s16le -ar 44100 -map_metadata -1 output.bin
# 2. bin传给apply-wav : apply-vad <output.bin> <output.vad.wav>
# 3. 删除中间变量bin文件