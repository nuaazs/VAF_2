# speaker model result

The document contains the EER and minDCF results of ECAPA-TDNN, CAMplus models on different test sets. 

# 模型

1.ECAPA-TDNN
路径：/home/zhaosheng/VAF/src/nn/ECAPATDNN/
2.CAMplus
路径：/home/duanyibo/dyb/3D-Speaker/egs/sv-cam++/voxceleb/exp/cam++_16k/models/CKPT-EPOCH-80-00

# Data
## Train

VoxCeleb 1+2 先降采样到8K，再重采样到16K的数据，包含7205个说话人。
地址：/datasets/voxceleb12_phone_16k

## Test

VoxCeleb1-O: 40-speaker 37611-pair
数据地址：/home/duanyibo/dyb/data/wespeaker/examples/voxceleb/v2/data/raw_data/voxceleb1/test
testpair-地址：https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt


Changjiang-test: 1000-speaker 3000-pair
数据地址：/datasets/test/cti_mini_test_16k
testpair-地址：/datasets/test_pairs.txt

# Result

* Scoring: cosine 
* Metric: EER(%) and minDCF(%)

| Model          | Params | vox1-O-clean  | Changjiang-test |
|:--------------:|:------:|:-------------:|:---------------:|
| ECAPA_TDNN_192 | 14.65M | 1.611/0.2240  |   5.770/0.4634  | 
| CAMplus        |  7.18M | 2.552/0.3046  |   5.686/0.4745  |
| ResNet-19      |  7.18M | 21.214/0.9457 |  20.233/0.7891  |