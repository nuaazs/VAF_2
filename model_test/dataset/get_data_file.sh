#!/bin/bash

# data_info.txt
# 13961976343,20220412084131,/datasets_hdd/datasets/cjsd_download/13961976343/cti_record_11008_1649724091811971_1-abfe679d-d170-4878-8074-d70b588369fb.wav
# 13961976343,20220412163847,/datasets_hdd/datasets/cjsd_download/13961976343/cti_record_11006_1649752727132048_1-e99b6af1-50b9-47b2-8f67-fbf65f5b965a.wav
# 13961976343,20220319124045,/datasets_hdd/datasets/cjsd_download/13961976343/cti_record_11003_1647664845487956_1-56173d89-7f30-4d53-b7d3-4f992945680f.wav

# read data_info.txt ,and get wav file path(3rd column)
# then use ffmpeg to get wav channel 1 and sample rate 16000
# then save the wav file path to  <phone>/<fid>.wav
# phone is 1st column, fid is 2nd column

for line in `cat data_info.txt`
do
    phone=`echo $line | awk -F ',' '{print $1}'`
    fid=`echo $line | awk -F ',' '{print $2}'`
    wav_path=`echo $line | awk -F ',' '{print $3}'`
    mkdir -p /datasets_hdd/cjsd_zhaosheng_train_data/data/$phone
    mkdir -p data_vad/$phone
    mkdir -p /datasets_hdd/cjsd_zhaosheng_train_data/data_bin/$phone
    mkdir -p data_info/$phone
    ffmpeg -i $wav_path -ss 7 -map_channel 0.0.1 /datasets_hdd/cjsd_zhaosheng_train_data/data/$phone/$fid.wav -ar 16000 -y > /dev/null 2>&1
    ffmpeg -i /datasets_hdd/cjsd_zhaosheng_train_data/data/$phone/$fid.wav -f s16le -acodec pcm_s16le -ar 16000 -map_metadata -1 -y  /datasets_hdd/cjsd_zhaosheng_train_data/data_bin/$phone/$fid.bin > /dev/null 2>&1
    ./vad_wav --wav-bin=/datasets_hdd/cjsd_zhaosheng_train_data/data_bin/$phone/$fid.bin --text-out=data_info/$phone/$fid.txt --energy-thresh=10e7 --wav-out=data_vad/$phone/$fid.wav > /dev/null 2>&1
    # if data/$phone/$fid.wav filesize < 1000, then delete it
    if [ `ls -l data_vad/$phone/$fid.wav | awk '{print $5}'` -lt 1000 ];then
        rm /datasets_hdd/cjsd_zhaosheng_train_data/data/$phone/$fid.wav
        rm /datasets_hdd/cjsd_zhaosheng_train_data/data_bin/$phone/$fid.bin
        rm data_vad/$phone/$fid.wav
        rm data_info/$phone/$fid.txt
    fi
done