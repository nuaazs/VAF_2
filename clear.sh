#!/bin/bash
docker rm -f vaf_1
docker rm -f vaf_2
docker rm -f html_1
docker rm -f html_2
docker rmi -f zhaosheng/vaf:v2.0
docker rmi -f zhaosheng/html:v2.0
rm -rf runtime/vaf/log_*
rm -rf runtime/vaf/cpp_*
rm -rf runtime/vaf/cfg_*
rm -rf runtime/vaf/*docker.log
rm -rf runtime/vaf/pretrained_models_*

rm -rf runtime/html/log_*
rm -rf runtime/html/cfg_*
rm -rf runtime/vaf/pretrained_models_*
