## 脚本用法

```bash
bash ${DGUARD_DEPLOY_ROOT}/onnxruntime/dguard_encode.sh --model <model_name> --wav_scp <wav_scp_file> --output_txt <output_txt_file> --embedding_size 256

alias encode="${DGUARD_DEPLOY_ROOT}/onnxruntime/dguard_encode.sh"

encode --model <model_name> --wav_scp <wav_scp_file> --output_txt <output_txt_file> --embedding_size 256
```

## 参数说明

- `--model <model_name>`: 指定要使用的模型名称。可选的模型名称为 dfresnet233、repvgg 和 eres2net。
- `--wav_scp <wav_scp_file>`: 指定包含音频文件列表的 wav_scp 文件的地址。每行格式为 `<ID> <FILE_PATH>`，其中 `<ID>` 是音频文件的唯一标识符，`<FILE_PATH>` 是音频文件的路径。
- `--output_txt <output_txt_file>`: 指定保存输出结果的文本文件的地址。
- `--embedding_size <embedding_size>`: 指定输出特征的维度。默认值为 256。

## wav_scp 文件内容格式说明

wav_scp 文件是一个包含音频文件列表的文本文件。每行表示一个音频文件，由唯一标识符和文件路径组成，之间用空格分隔。具体格式如下：

```
<ID> <FILE_PATH>
```

其中：
- `<ID>` 是音频文件的唯一标识符，可以是任意字符串，用于区分不同的音频文件。
- `<FILE_PATH>` 是音频文件的路径，是一个指向实际音频文件的有效路径。

请确保 wav_scp 文件中的音频文件存在，并且文件路径是正确的。

## 使用示例

以下是一个使用示例：

```
encode --model dfresnet_233 --wav_scp ./test/test_input.scp --output_txt ./test/test_output.txt --embedding_size 512

encode --model mfa_conformer --wav_scp ./test/test_input.scp --output_txt ./test/test_output.txt --embedding_size 512

encode --model resnet101_cjsd --wav_scp ./test/test_input.scp --output_txt ./test/test_output.txt --embedding_size 256
```

本示例中，使用 `dfresnet233` 模型对 `./test/test_input.scp` 文件中列出的音频文件进行特征提取，并将输出结果保存到 `./test/test_output.txt` 文件中。