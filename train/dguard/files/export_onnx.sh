#!/bin/bash


# model_name is all file name in /VAF/train/dguard/files/pt
# model_name_without_lm is model_name repalce "lm" to ""
# python export_onnx.py --config ../files/yaml/<model_name_without_lm>.yaml --checkpoint ../files/pt/<model_name>.pt --output_file ../files/onnx/<model_name>.onnx

for model_name in $(ls pt)
do
    # rm .pt
    model_name=${model_name//.pt/}
    model_name_without_lm=${model_name//_lm/}
    python ../bin/export_onnx.py --config yaml/${model_name_without_lm}.yaml --checkpoint pt/${model_name}.pt --output_file onnx/${model_name}.onnx
done