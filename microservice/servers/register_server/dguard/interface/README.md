# PretrainedModel 使用说明

PretrainedModel 是一个预训练模型类，用于从音频中提取特征向量，并可进行特征向量的比较和提取。此文档将详细介绍 PretrainedModel 类的使用方法和示例结果。

## 导入 PretrainedModel

```python
from dguard.interface import PretrainedModel
```

## 初始化 PretrainedModel 对象

```python
infer = PretrainedModel(model_name, device='cpu', strict=True, mode="extract")
```

- `model_name`：模型名称，指定加载的预训练模型的名称。
- `device`：设备类型，默认为 'cpu'，可选择 'cuda' 或 'cuda:0' 等指定 GPU 设备。
- `strict`：是否严格加载模型参数，默认为 True。
- `mode`：模式选择，可选值为 "compare" 或 "extract"，默认为 "extract"。如果选择 "compare"，则用于比较两个音频的相似度；如果选择 "extract"，则只提取特征向量而不进行比较。

## 方法：inference

```python
result = infer.inference(wav_path_list, cmf=True, segment_length=3*16000, crops_num_limit=1)
```

- `wav_path_list`：待处理的音频文件路径列表。
- `cmf`：是否计算 CMF（Crops Merging Feature），默认为 True。如果为 True，则会计算 CMF 并返回。
- `segment_length`：分段长度，默认为 3 秒乘以采样率。
- `crops_num_limit`：提取的分段数量限制，默认为 1。

返回结果：
- 如果选择 "compare" 模式，则返回一个元组 `(cos_score, factor)`，其中 `cos_score` 是两个音频特征向量余弦相似度的评分，`factor` 是 CMF 的相关系数。
- 如果选择 "extract" 模式，则返回一个列表 `result`，其中每个元素是一个列表 `[output, cmf_embedding, crops_num]`，`output` 是提取的特征向量，`cmf_embedding` 是计算得到的 CMF，`crops_num` 是提取的分段数量。

## 示例

### Compare 模式

```python
infer = PretrainedModel('resnet293_lm', mode="compare")
cos_score, factor = infer.inference(['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'], cmf=True, segment_length=3*16000)
print(f"cos_score: {cos_score}, factor: {factor}")
```

输出结果示例：
```
cos_score: 0.8765, factor: 0.9876
```

### Extract 模式

```python
infer = PretrainedModel('resnet293_lm', mode="extract")
result = infer.inference(['/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00001.wav','/VAF/train/data/raw_data/voxceleb1/test/wav/id10270/5sJomL_D0_g/00002.wav'], cmf=True, segment_length=3*16000)
print(f"result len: {len(result)}")
print(f"result[0][0] shape: {result[0][0].shape}")
print(f"result[1][0] shape: {result[1][0].shape}")
print(f"result[0][1] shape: {result[0][1].shape}")
print(f"result[1][1] shape: {result[1][1].shape}")
```

输出结果示例：
```
result len: 2
result[0][0] shape: (512,)
result[1][0] shape: (512,)
result[0][1] shape: (256,)
result[1][1] shape: (256,)
```

以上是 PretrainedModel 类的使用说明和示例。根据需要选择合适的模式和参数进行特征提取和比较，以满足音频处理需求。