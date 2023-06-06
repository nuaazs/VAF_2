#!/bin/bash
pwd=$PWD
cd docker/vaf
docker build -t zhaosheng/vaf:v2.0 .
cd $pwd
cd src/
./start_multi.sh
