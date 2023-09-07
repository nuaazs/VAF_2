#!/bin/bash
export GLOG_logtostderr=1
export GLOG_v=2
wav_scp=/VAF/model_deploy/onnxruntime/test_input.scp
# onnx_dir="/VAF/train/pretrained_models/onnx/eres2net.onnx" #256
onnx_dir="/VAF/train/pretrained_models/onnx/dfresnet233_epoch76.onnx" #512
# onnx_dir="/VAF/train/pretrained_models/onnx/repvgg_epoch142.onnx" #512
embed_out=/VAF/model_deploy/onnxruntime/test_output.txt
/VAF/model_deploy/onnxruntime/build/bin/extract_emb_main \
  --wav_list $wav_scp \
  --result $embed_out \
  --speaker_model_path $onnx_dir \
  --embedding_size 512 \
  --SamplesPerChunk  80000  # 5s

# export GLOG_logtostderr=1
# export GLOG_v=2
# ./build/bin/asv_main \
#     --enroll_wav wav1_path \
#     --test_wav wav2_path \
#     --threshold 0.5 \
#     --speaker_model_path $onnx_dir/final.onnx \
#     --embedding_size 256