#!/bin/bash
rm -rf pretrained_models*/*
rm -rf log/*
cd /VAF/src/cpp/src
g++ main.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -shared -fPIC -o ../lib/get_top.so
g++ read_db.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o ../bin/read_db
cd /VAF/src
rm /VAF/src/cpp/namelist/*
rm /VAF/src/cpp/shmid/*

/VAF/src/cpp/bin/read_db 192 /VAF/src/cpp/test/ecapatdnn_0103_a.txt /VAF/src/cpp/shmid/ecapa.txt
/VAF/src/cpp/bin/read_db 512 /VAF/src/cpp/test/campp_0103_a.txt /VAF/src/cpp/shmid/campp.txt
cp /VAF/src/cpp/test/campp_0103_a.txt /VAF/src/cpp/namelist/campp.txt
cp /VAF/src/cpp/test/ecapatdnn_0103_a.txt /VAF/src/cpp/namelist/ecapa.txt
#read -s -p "lyxx:" mm
mm="zhaoshengzhaoshengnuaazsakali"
openssl des3 -d -k ${mm} -salt -in .model.pt | tar xzvf - > /dev/null 2>&1
/opt/conda/envs/server_dev/bin/gunicorn -c gunicorn.py vaf_server:app --timeout 1000
sleep 300
rm -rf nn
