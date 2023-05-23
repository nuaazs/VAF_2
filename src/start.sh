#!/bin/bash
rm -rf pretrained_models*/*
rm -rf log/*
/VAF/src/cpp/bin/read_db 192 /VAF/src/cpp/test/ecapatdnn_0103_a.txt /VAF/src/cpp/shmid/ecapa.txt
/VAF/src/cpp/bin/read_db 512 /VAF/src/cpp/test/campp_0103_a.txt /VAF/src/cpp/shmid/campp.txt
/opt/conda/envs/server_dev/bin/gunicorn -c gunicorn.py vaf_server:app --timeout 1000