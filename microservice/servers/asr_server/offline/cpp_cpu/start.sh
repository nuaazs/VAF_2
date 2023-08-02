#!/bin/bash
MODEL_PATH="./models/damo/"
./build/bin/funasr-wss-server --model-dir ${MODEL_PATH}speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx --vad-dir ${MODEL_PATH}speech_fsmn_vad_zh-cn-16k-common-onnx --punc-dir ${MODEL_PATH}/punc_ct-transformer_zh-cn-common-vocab272727-onnx