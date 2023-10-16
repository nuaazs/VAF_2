#!/bin/bash
now_path=$(pwd)
now_datetime=$(date "+%Y%m%d%H%M%S")
mkdir -p $now_datetime
cd .. && zip -r test.zip model_test -x "model_test/result/*" -x "model_test/dguard/*" && cd $now_path
cd .. && zip -r deploy.zip model_deploy && cd $now_path
cd .. && zip -r cpp.zip cpp && cd $now_path
cd ../train && zip -r dguard.zip dguard -x "dguard/files/onnx/*" && cd $now_path
mv ../test.zip $now_datetime
mv ../train/deploy.zip $now_datetime
mv ../dguard.zip $now_datetime
mv ../cpp.zip $now_datetime
