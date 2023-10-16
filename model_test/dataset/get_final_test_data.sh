#!/bin/bash
# Read /home/zhaosheng/Documents/cjsd_train_data/choosed_5000.csv
# each line is <phone>,<size>,<size> . get 5000 phone numbers
# and cp -r ./data_vad_split/<phone> to ./data_vad_split_5000/<phone>

for line in `cat /home/zhaosheng/Documents/cjsd_train_data/choosed_300.csv`
do
    phone=`echo $line | awk -F ',' '{print $1}'`
    # if len(phone) == 10 ,add "0"
    if [ ${#phone} -eq 10 ];then
        phone="0$phone"
        mkdir -p ./data_vad_split_test_300/$phone
        cp -r ./data_vad_split/$phone ./data_vad_split_test_300/
    fi
    mkdir -p ./data_vad_split_test_300/$phone
    cp -r ./data_vad_split/$phone ./data_vad_split_test_300/
done