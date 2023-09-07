# ERes2Net

## Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.00005, 0.2], batch_size 512, 8 gpu(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF(p-target=0.01)

## Voxceleb Results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| ERes2Net-Base | 4.6M | 0.97  |  0.090 |

## pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- Voxceleb: [speech_eres2net_sv_en_voxceleb_16k](https://modelscope.cn/models/damo/speech_eres2net_sv_en_voxceleb_16k/summary)
- 200k labeled speakers: [speech_eres2net_sv_zh-cn_16k-common](https://modelscope.cn/models/damo/speech_eres2net_sv_zh-cn_16k-common/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# CAM++ trained on VoxCeleb
model_id=damo/speech_eres2net_sv_en_voxceleb_16k
# Run inference
python dguard/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

## Citations
If you are using ERes2Net model in your research, please cite: 
```BibTeX
@article{eres2net,
  title={An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification},
  author={Yafeng Chen, Siqi Zheng, Hui Wang, Luyao Cheng, Qian Chen, Jiajun Qi},
  booktitle={Interspeech 2023},
  year={2023},
  organization={IEEE}
}
```
