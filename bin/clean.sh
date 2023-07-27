#!/bin/bash
# remote pretrained_models __pycache__ and *.pyc

find . -name "__pycache__" | xargs rm -rf
find . -name "*.pyc" | xargs rm -rf
find . -name "pretrained_models" | xargs rm -rf
find . -name "*.log" | xargs rm
rm -rf ./src/pretrained_models*
rm -rf ./src/log_*
rm -rf ./src/nn
rm -rf ./src/cfg_*
rm -rf ./src/log/*
rm -rf ./src/cpp_*
rm -rf docker/vaf/src.tar.gz
tar -cvf docker/build/vaf/src.tar.gz ./src
