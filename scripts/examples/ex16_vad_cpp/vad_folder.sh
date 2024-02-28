#!/bin/bash
# Usage: vad_folder.sh <audio-dir> <vad-dir> <txt-dir> <num-thread>

if [ $# -ne 4 ];then
    echo "Usage: $0 <audio-dir> <vad-dir> <txt-dir> <num-thread>"
    exit 1
fi

AUDIO_DIR=$1
echo "AUDIO_DIR: ${AUDIO_DIR}"
VAD_DIR=$2
TXT_DIR=$3
NUM_THREAD=$4

if [ ! -d "${VAD_DIR}" ]; then
    mkdir -p ${VAD_DIR}
fi

if [ ! -d "${TXT_DIR}" ]; then
    mkdir -p ${TXT_DIR}
fi

function vad_file() {
    local audio_file="$1"
    # read AUDIO_DIR from global variable
    local AUDIO_DIR="$2"
    local VAD_DIR="$3"
    local TXT_DIR="$4"
    echo "AUDIO_DIR: ${AUDIO_DIR}"
    local relative_path=$(realpath --relative-to=${AUDIO_DIR} ${audio_file})
    local vad_file="${VAD_DIR}/${relative_path}"
    local txt_file="${TXT_DIR}/${relative_path%.wav}.txt"

    (
        # get file name from $1 and add with random number
        FILE_NAME=$(basename ${audio_file})_$(date +%s%N).bin

        # extract pcm data from wav file
        ffmpeg -i ${audio_file} -f s16le -acodec pcm_s16le -ar 8000 -map_metadata -1 ${FILE_NAME} > /dev/null 2>&1

        # apply voice activity detection
        echo "apply-vad ${FILE_NAME} ${vad_file} ${txt_file}"
        # make father dir of vad_file
        mkdir -p $(dirname ${vad_file})
        # make father dir of txt_file
        mkdir -p $(dirname ${txt_file})
        apply-vad ${FILE_NAME} ${vad_file} ${txt_file}

        # remove intermediate binary file
        rm -rf ${FILE_NAME}
        # rm -rf $(dirname ${FILE_NAME})

    )
}

export -f vad_file

find "${AUDIO_DIR}" -type f -name "*.wav" | parallel -j${NUM_THREAD} vad_file {} $AUDIO_DIR $VAD_DIR $TXT_DIR

echo "Done"