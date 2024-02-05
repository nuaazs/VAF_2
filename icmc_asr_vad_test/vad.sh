#!/bin/bash
# coding = utf-8
# get wav file path from command line
wav_path=$1
# Generate bin
echo "Generating test_vad.bin"
ffmpeg -i $wav_path -f s16le -acodec pcm_s16le -ar 16000 -ac 1 -map_metadata -1 -y  test_vad.bin


# Test
echo "Testing ../bin/vad_wav"
./vad_wav --wav-bin='test_vad.bin' --energy-thresh=5e7 --text-out='./test_out.txt' --min-duration=0.0 --smooth-threshold=0.1 --wav-out='./output.wav'
./vad --wav-bin='test_vad.bin' --energy-thresh=5e7 --text-out='./test_out.txt' --min-duration=0.0 --smooth-threshold=0.1
