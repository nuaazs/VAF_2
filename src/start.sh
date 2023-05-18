#!/bin/bash
rm -rf pretrained_models*/*
rm -rf log/*
/opt/conda/envs/server_dev/bin/gunicorn -c gunicorn.py vaf_server:app --timeout 1000