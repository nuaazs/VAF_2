# CAM++

## Training config
- Feature: 80-dim fbank, mean normalization, speed perturb
- Training: lr [0.0001, 0.1], batch_size 256, 4 gpu(Tesla V100), additive angular margin
- Metrics: EER(%), MinDCF

## Voxceleb Results
- Train set: Voxceleb2-dev, 5994 speakers
- Test set: Voxceleb-O

| Model | Params | EER(%) | MinDCF |
|:-----:|:------:|:------:|:------:|
| CAM++ | 7.18M  | 0.73 | 0.0911 |

## pretrained model
Pretrained models are accessible on [ModelScope](https://www.modelscope.cn/models?page=1&tasks=speaker-verification&type=audio).

- Voxceleb: [speech_campplus_sv_en_voxceleb_16k](https://modelscope.cn/models/damo/speech_campplus_sv_en_voxceleb_16k/summary)
- 200k labeled speakers: [speech_campplus_sv_zh-cn_16k-common](https://www.modelscope.cn/models/damo/speech_campplus_sv_zh-cn_16k-common/summary)

Here is a simple example for directly extracting embeddings. It downloads the pretrained model from [ModelScope](https://www.modelscope.cn/models) and extracts embeddings.
``` sh
# Install modelscope
pip install modelscope
# CAM++ trained on VoxCeleb
model_id=damo/speech_campplus_sv_en_voxceleb_16k
# CAM++ trained on 200k labeled speakers
model_id=damo/speech_campplus_sv_zh-cn_16k-common
# Run inference
python dguard/bin/infer_sv.py --model_id $model_id --wavs $wav_path
```

## Citations
If you are using CAM++ model in your research, please cite: 
```BibTeX
@article{cam++,
  title={CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking},
  author={Hui Wang and Siqi Zheng and Yafeng Chen and Luyao Cheng and Qian Chen},
  booktitle={Interspeech 2023},
  year={2023},
  organization={IEEE}
}
```
