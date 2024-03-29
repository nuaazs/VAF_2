#!/bin/bash

# 解析命令行参数
model_name=""
wav_scp=""
output_txt=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      model_name="$2"
      shift
      shift
      ;;
    --wav_scp)
      wav_scp="$2"
      shift
      shift
      ;;
    --output_txt)
      output_txt="$2"
      shift
      shift
      ;;
    --embedding_size)
      embedding_size="$2"
      shift
      shift
      ;;
    *)
      echo "未知的参数: $1"
      exit 1
      ;;
  esac
done

# 检查必要的参数是否存在
if [[ -z $model_name ]]; then
  echo "未指定模型名称，请使用 --model 参数指定 dfresnet233、repvgg 或 eres2net。"
  exit 1
fi

if [[ -z $wav_scp ]]; then
  echo "未指定 wav_scp 文件地址，请使用 --wav_scp 参数指定地址。"
  exit 1
fi

if [[ -z $output_txt ]]; then
  echo "未指定保存的 output.txt 地址，请使用 --output_txt 参数指定地址。"
  exit 1
fi

if [[ -z $embedding_size ]]; then
  echo "未指定 embedding_size，请使用 --embedding_size 参数指定地址。"
  exit 1
fi

# model name
# onnx_dir=f"../dguard/files/onnx/{model_name}.onnx"
onnx_dir="${DGUARD_ROOT}/files/onnx/${model_name}.onnx"


# 执行命令
${DGUARD_DEPLOY_ROOT}/onnxruntime/build/bin/extract_emb_main \
  --wav_list "$wav_scp" \
  --result "$output_txt" \
  --speaker_model_path "$onnx_dir" \
  --embedding_size "$embedding_size" \
  --SamplesPerChunk 80000

# 检查执行结果
if [ $? -eq 0 ]; then
  echo "Encode Success\n"
  # print =*50
  echo "==============="
  echo "Output:"
  cat "$output_txt"
else
  echo "Error Encode Failed\n"
fi
