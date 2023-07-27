#!/bin/bash
# for i in 1 to 8
# cp cfg.py to cfg_{i}.py
# use sed to change the cfg_{i}.py, change PORT = 7777 -> PORT = 819{i}
# then run the following command
# docker rm -f vaf_{i}
# docker run -it --name vaf_{i} --gpus="device={i-1}" -v ${PWD}/cfg_{i}.py:/VAF/src/cfg.py -v ${PWD}/log_{i}:/VAF/src/log --net host zhaosheng/vaf:v0.5.0



for i in {1..8}
do
    docker rm -f vaf_${i}
done
echo "all remove"

for i in {1..8}
do
    cp cfg.py cfg_${i}.py
    cp -r ${PWD}/cpp ${PWD}/cpp_${i}
    sed -i "s/PORT = 7777/PORT = 819${i}/g" cfg_${i}.py
    # cp -r pretrained_models pretrained_models_${i}
    # run docker and > /dev/null 2>&1 to hide the output, make it run in background
    # cuda_num = i - 1, calculate the cuda_nun
    cuda_num=`expr ${i} - 1`
    echo "docker run -d --name vaf_${i} --gpus="device=${cuda_num}" -v ${PWD}/cpp_${i}:/VAF/src/cpp -v ${PWD}/cfg_${i}.py:/VAF/src/cfg.py -v ${PWD}/log_${i}:/VAF/src/log -v ${PWD}/pretrained_models_${i}:/VAF/src/pretrained_models --net host zhaosheng/vaf:v1.0 bash > /dev/null 2>&1 &"
    docker run --name vaf_${i} --gpus "device=${cuda_num}" -v ${PWD}/start.sh:/VAF/src/start.sh -v ${PWD}/cpp_${i}:/VAF/src/cpp -v ${PWD}/cfg_${i}.py:/VAF/src/cfg.py -v ${PWD}/log_${i}:/VAF/src/log -v ${PWD}/pretrained_models_${i}:/VAF/src/pretrained_models --net host zhaosheng/vaf:v2.0 > ${i}_docker.log 2>&1 &
done
