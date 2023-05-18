# coding = utf-8
# @Time    : 2022-09-05  15:04:36
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: models.

from speechbrain.pretrained import SpeakerRecognition
import torch
import cfg

similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
# get current random seed from 1000~9999
random_seed = torch.randint(1000, 9999, (1,)).item()
spkreg = SpeakerRecognition.from_hparams(
    source="./nn/ECAPATDNN-16k-phone_1",
    savedir=f"./pretrained_models/ECAPATDNN-16k-phone_1_{random_seed}",
    run_opts={"device": cfg.DEVICE},
)