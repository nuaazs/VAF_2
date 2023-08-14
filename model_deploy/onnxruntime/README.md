## 脚本用法

```
# bash /VAF/model_deploy/onnxruntime/dguard_encode.sh --model <model_name> --wav_scp <wav_scp_file> --output_txt <output_txt_file>
encode --model <model_name> --wav_scp <wav_scp_file> --output_txt <output_txt_file>

```

## 参数说明

- `--model <model_name>`: 指定要使用的模型名称。可选的模型名称为 dfresnet233、repvgg 和 eres2net。
- `--wav_scp <wav_scp_file>`: 指定包含音频文件列表的 wav_scp 文件的地址。每行格式为 `<ID> <FILE_PATH>`，其中 `<ID>` 是音频文件的唯一标识符，`<FILE_PATH>` 是音频文件的路径。
- `--output_txt <output_txt_file>`: 指定保存输出结果的文本文件的地址。

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
bash my_script.sh --model dfresnet233 --wav_scp /path/to/wav.scp --output_txt /path/to/output.txt
```

本示例中，使用 `dfresnet233` 模型对 `/path/to/wav.scp` 文件中列出的音频文件进行特征提取，并将输出结果保存到 `/path/to/output.txt` 文件中。