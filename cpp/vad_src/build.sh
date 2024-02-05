#!/bin/bash
# coding = utf-8
# @Time    : 2023-11-01  09:08:19
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: VAD UTILS.

# Build
echo "Building ../bin/vad_wav"
g++ vad_output_file.cpp -std=c++11 -o ../bin/vad_wav
echo "    *-> Done."

# Generate bin
echo "Generating test_vad.bin"
ffmpeg -i xyx.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 -map_metadata -1 -y  test_vad.bin


# Test
echo "Testing ../bin/vad_wav"
../bin/vad_wav --wav-bin='test_vad.bin' --energy-thresh=5e7 --text-out='./test_out1.txt' --min-duration=0.1 --smooth-threshold=0.1 --wav-out='./output1.wav'
../bin/vad_wav --wav-bin='test_vad.bin' --energy-thresh=5e7 --text-out='./test_out2.txt' --min-duration=0.1 --smooth-threshold=1.0 --wav-out='./output2.wav'
