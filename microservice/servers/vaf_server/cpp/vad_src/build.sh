#!/bin/bash

# Build
echo "Building ../bin/vad_wav"
g++ vad_output_file.cpp -std=c++11 -o ../bin/vad_wav
echo "    *-> Done."

# Generate bin
echo "Generating test_vad.bin"
ffmpeg -i input.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 -map_metadata -1 -y  test_vad.bin


# Test
echo "Testing ../bin/vad_wav"
../bin/vad_wav --wav-bin='test_vad.bin' --energy-thresh=1e8 --text-out='./test_out.txt' --min-duration=2.0 --smooth-threshold=0.5 --wav-out='./output.wav'

# rm
rm -rf test_vad.bin