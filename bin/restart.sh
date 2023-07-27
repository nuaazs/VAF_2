#!/bin/bash

pwd=$PWD
bash clean.sh
cd docker/build/vaf
docker build -t zhaosheng/vaf:v2.0 .
cd $pwd/docker/runtime/vaf
bash start.sh
