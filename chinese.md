## 盯小语声纹识别系统

![pic](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)![pic](https://img.shields.io/badge/Pytorch-1.10.1-green.svg)![pic](https://img.shields.io/badge/Python-3.9-blue.svg)![pic](https://img.shields.io/badge/ECAPATDNN-pink.svg)![pic](https://img.shields.io/badge/CAM++-red.svg)![pic](https://img.shields.io/badge/ResNet-yellow.svg)![pic](https://img.shields.io/badge/MINIO-blue.svg)![pic](https://img.shields.io/badge/Long-Yuan-green.svg)



 [文档](./docs/main.md) |  [预训练模型](./docs/models.md) | [电信测试pipeline](./docs/pipeline.md) | [测试结果](./docs/test_result.md)

 [部署文档](./docs/icnoc_deploy_README.md) 



## 🔥 更新

2023-05-31: 上线音频性别分类模块。

2023-05-10: 添加[语种识别](https://arxiv.org/abs/2005.07143)模型，对音频进行质量筛选，剔除外语、方言及质量不合格的中文普通话音频。

2023-05-11: 新增[CAM++](https://arxiv.org/abs/2303.00332)模型，对结果进行融合。



## 🚩 待办

- [x] 自监督

- [x] 语音增强模块

- [ ] 音频降噪模块

- [x] 话者分离模块

- [x] 音频性别检测模块

- [ ] 前端管理后台接口

- [ ] wenet-asr加速

- [x] docker文档

- [-] Runtime



## ⭐ Highlights

Provide a full-stack production solution for voiceprint recognition.

**Accurate:** Achieves SOTA results on many public speech datasets.

**Lightweight:** easy to install, easy to use, well documented.



## 🎲 模块划分

### 1. 预处理
音频预处理模块主要负责对音频进行预处理，包括音频格式转换、音频分割（通道、时长选取）、音频降噪（可选）、VAD模块。音频预处理模块的输入为音频文件，输出为预处理后的音频文件。
依赖：[ffmpeg](https://ffmpeg.org/)
VAD模型：[CRDNN_8k_phone_LY](https://aclanthology.org/C16-1229.pdf)

### 2. 质量检测

语种分类模型：[ECAPA-TDNN](https://arxiv.org/abs/2005.07143)

### 3. 对象存储
对象存储模块主要负责对预处理后的音频文件进行存储，包括音频文件的上传、下载、删除、查询等操作。对象存储模块的输入为预处理后的音频文件，输出为音频文件的存储地址。
依赖：[minio](https://min.io/)

### 4. 声纹黑库
声纹黑库模块主要负责对声纹黑库进行管理，包括声纹黑库的创建、删除、查询等操作。声纹黑库模块的输入为声纹黑库的名称，输出为声纹黑库的ID。
声纹模型：`LYNet_8k_phone`
存储方式：float32二进制存储（`<blackbase_name>.bin` + `<blackbase_name>.txt`）

### 5. 声纹编码
声纹编码模块主要负责对音频进行声纹编码，包括音频的特征提取、声纹编码、声纹特征的存储等操作。声纹编码模块的输入为预处理后的音频文件，输出为声纹特征。
现支持模型：[ECAPA-TDNN](https://arxiv.org/abs/2005.07143)、[CAM++](https://arxiv.org/abs/2303.00332)、[ResNet](https://arxiv.org/abs/1512.03385)-family

### 6. 声纹比对
声纹比对模块主要负责对声纹特征进行比对，包括声纹特征的查询、比对、相似度计算等操作。声纹比对模块的输入为声纹特征，输出为比对结果。
比对方式：余弦相似度（based on cpp）

### 7. 语音识别
语音识别模块主要负责对音频进行语音识别，包括音频的识别、识别结果的存储等操作。语音识别模块的输入为预处理后的音频文件，输出为识别结果。
ASR模型：[Conformer_phone_16k](https://arxiv.org/abs/2005.08100)

### 8. 语义判别
语义判别模块主要负责对识别结果进行语义判别，包括语义判别、语义判别结果的存储等操作。语义判别模块的输入为识别结果，输出为语义判别结果。


### 9. 可视化
结果导出与可视化模块主要负责对结果进行导出与可视化，包括结果的导出、可视化等操作。结果导出与可视化模块的输入为各模块的输出结果，输出为结果的导出与可视化。

