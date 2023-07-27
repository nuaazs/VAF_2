# coding = utf-8
# @Time    : 2022-09-05  15:04:36
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: models.

from speechbrain.pretrained import SpeakerRecognition
import torch
import cfg

random_seed = torch.randint(1000, 9999, (1,)).item()

emb = SpeakerRecognition.from_hparams(
    source="./nn/ECAPATDNN",
    savedir=f"./pretrained_models/ECAPATDNN_{random_seed}",
    run_opts={"device": cfg.DEVICE},
)