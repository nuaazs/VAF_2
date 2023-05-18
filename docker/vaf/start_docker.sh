#!/bin/bash
docker run -it --gpus all --net host --name gpuasr3 -v ${PWD}/start.sh:/VAF/asr/start.sh -v ${PWD}/cfg.py:/VAF/asr/cfg.py zhaosheng/asr:gpu_v1.0.2