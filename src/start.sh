#!/bin/bash
rm -rf pretrained_models/*
rm -rf log/*
cd cpp
# /opt/conda/envs/server_dev/bin/python export_db.py
./read_db
./test
cd ..
/opt/conda/envs/server_dev/bin/gunicorn -c gunicorn.py vaf_server:app --timeout 1000