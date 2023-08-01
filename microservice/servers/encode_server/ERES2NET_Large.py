
config = {
    "framework": "pytorch",
    "task": "speaker-verification",
    "model": {
        "type": "eres2net-aug-sv",
        "model_config": {
            "sample_rate": 16000
        },
        "pretrained_model": "pretrained_eres2net_aug.ckpt",
        "yesOrno_thr": 0.43
    },
    "pipeline": {
        "type": "speaker-verification-eres2net"
    }
}

from speechbrain.lobes.features import Fbank

# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Alibaba, Inc. and its affiliates.
""" Res2Net implementation is adapted from https://github.com/wenet-e2e/wespeaker.
    ERes2Net_aug incorporates both local and global feature fusion techniques
    to improve the performance. The training code is located on the following
    GitHub repository: https://github.com/alibaba-damo-academy/3D-Speaker.
"""
import math
import os
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
import torchaudio
import modelscope.models.audio.sv.pooling_layers as pooling_layers
from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.sv.fusion import AFF
from modelscope.utils.constant import Tasks
import cfg

class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv1x1(in_planes, out_planes, stride=1):
    '1x1 convolution without padding'
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    '3x3 convolution with padding'
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlockERes2Net(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, baseWidth=24, scale=3):
        super(BasicBlockERes2Net, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2Net_diff_AFF(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, baseWidth=24, scale=3):
        super(BasicBlockERes2Net_diff_AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = conv1x1(in_planes, width * scale, stride)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = scale

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(conv3x3(width, width))
            bns.append(nn.BatchNorm2d(width))
        # Add different fuse_model parameters
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = conv1x1(width * scale, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2Net_aug(nn.Module):

    def __init__(self,
                 block=BasicBlockERes2Net,
                 block_fuse=BasicBlockERes2Net_diff_AFF,
                 num_blocks=[3, 4, 6, 3],
                 m_channels=64,
                 feat_dim=80,
                 embedding_size=192,
                 pooling_func='TSTP',
                 two_emb_layer=False):
        super(ERes2Net_aug, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block_fuse, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(
            block_fuse, m_channels * 8, num_blocks[3], stride=2)

        self.layer1_downsample = nn.Conv2d(
            m_channels * 4,
            m_channels * 8,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False)
        self.layer2_downsample = nn.Conv2d(
            m_channels * 8,
            m_channels * 16,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False)
        self.layer3_downsample = nn.Conv2d(
            m_channels * 16,
            m_channels * 32,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False)

        self.fuse_mode12 = AFF(channels=m_channels * 8)
        self.fuse_mode123 = AFF(channels=m_channels * 16)
        self.fuse_mode1234 = AFF(channels=m_channels * 32)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == 'TSDP' else 2
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out1_downsample = self.layer1_downsample(out1)
        fuse_out12 = self.fuse_mode12(out2, out1_downsample)
        out3 = self.layer3(out2)
        fuse_out12_downsample = self.layer2_downsample(fuse_out12)
        fuse_out123 = self.fuse_mode123(out3, fuse_out12_downsample)
        out4 = self.layer4(out3)
        fuse_out123_downsample = self.layer3_downsample(fuse_out123)
        fuse_out1234 = self.fuse_mode1234(out4, fuse_out123_downsample)
        stats = self.pool(fuse_out1234)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a


# @MODELS.register_module(
#     Tasks.speaker_verification, module_name=Models.eres2net_aug_sv)
class SpeakerVerificationERes2Net(TorchModel):
    r"""Enhanced Res2Net_aug architecture with local and global feature fusion.
    ERes2Net_aug is an upgraded version of ERes2Net that uses a larger number of
    parameters to achieve better recognition performance.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, model_dir, model_config: Dict[str, Any], *args,
                 **kwargs):
        super().__init__(model_dir, model_config, *args, **kwargs)
        self.model_config = model_config
        self.other_config = kwargs
        self.feature_dim = 80
        self.device = cfg.ENCODE_ERES2NET_DEVICE
        self.embedding_model = ERes2Net_aug()
        # eval
        self.embedding_model = self.embedding_model.eval()
        self.embedding_model = self.embedding_model.to(self.device)
        pretrained_model_name = kwargs['pretrained_model']
        self.__load_check_point(pretrained_model_name)

        self.embedding_model.eval()

    def encode_batch(self, audio):
        assert len(audio.shape) == 2 and audio.shape[
            0] == 1, 'modelscope error: the shape of input audio to model needs to be [1, T]'
        # audio shape: [1, T]
        audio = audio.to(self.device)
        feature = self.__extract_feature(audio)
        with torch.no_grad():
            embedding = self.embedding_model(feature)

        return embedding

    def __extract_feature(self, audio):
        feature = Kaldi.fbank(audio, num_mel_bins=self.feature_dim)
        feature = feature - feature.mean(dim=0, keepdim=True)
        feature = feature.unsqueeze(0)
        return feature

    def __load_check_point(self, pretrained_model_name, device=None):
        if not device:
            device = torch.device('cpu')
        self.embedding_model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir, pretrained_model_name),
                map_location=device),
            strict=True)
            
import io
from typing import Any, Dict, List, Union

import soundfile as sf
import torch

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.speaker_verification,
    module_name=Pipelines.speaker_verification_eres2net)
class ERes2Net_Pipeline(Pipeline):
    """Speaker Verification Inference Pipeline
    use `model` to create a Speaker Verification pipeline.

    Args:
        model (SpeakerVerificationPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the pipeline's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> p = pipeline(
    >>>    task=Tasks.speaker_verification, model='damo/speech_ecapa-tdnn_sv_en_voxceleb_16k')
    >>> print(p([audio_1, audio_2]))

    """

    def __init__(self, model: InputModel, **kwargs):
        """use `model` to create a speaker verification pipeline for prediction
        Args:
            model (str): a valid offical model id
        """
        super().__init__(model=model, **kwargs)
        self.model_config = self.model.model_config
        self.config = self.model.other_config
        self.thr = self.config['yesOrno_thr']

    def __call__(self,
                 in_audios: List[str],
                 thr: float = None) -> Dict[str, Any]:
        if thr is not None:
            self.thr = thr
        if self.thr < -1 or self.thr > 1:
            raise ValueError(
                'modelscope error: the thr value should be in [-1, 1], but found to be %f.'
                % self.thr)
        outputs = self.preprocess(in_audios)
        outputs = self.forward(outputs)
        outputs = self.postprocess(outputs)

        return outputs

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        emb1 = self.model(inputs['data1'])
        emb2 = self.model(inputs['data2'])

        return {'emb1': emb1, 'emb2': emb2}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        score = self.compute_cos_similarity(inputs['emb1'], inputs['emb2'])
        score = round(score, 5)
        if score >= self.thr:
            ans = 'yes'
        else:
            ans = 'no'

        return {OutputKeys.SCORE: score, OutputKeys.TEXT: ans}

    def preprocess(self, inputs: List[str],
                   **preprocess_params) -> Dict[str, Any]:
        if len(inputs) != 2:
            raise ValueError(
                'modelscope error: Two input audio files are required.')
        output = {}
        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                file_bytes = File.read(inputs[i])
                data, fs = sf.read(io.BytesIO(file_bytes), dtype='float32')
                if len(data.shape) == 2:
                    data = data[:, 0]
                if fs != self.model_config['sample_rate']:
                    raise ValueError(
                        'modelscope error: Only support %d sample rate files'
                        % self.model_cfg['sample_rate'])
                output['data%d' %
                       (i + 1)] = torch.from_numpy(data).unsqueeze(0)
            else:
                raise ValueError(
                    'modelscope error: The input type is temporarily restricted to audio file address'
                    % i)
        return output

    def compute_cos_similarity(self, emb1: torch.Tensor,
                               emb2: torch.Tensor) -> float:
        assert len(emb1.shape) == 2 and len(emb2.shape) == 2
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine = cos(emb1, emb2)
        return cosine.item()


emb = SpeakerVerificationERes2Net(model_dir=f"models/speech_eres2net_sv_zh-cn_16k-common",model_config=config,pretrained_model=config['model']['pretrained_model'])
# import cfg
# emb = emb.to(cfg.ENCODE_ERES2NET_DEVICE)
if __name__ == "__main__":
    wav_data,sr = torchaudio.load("/home/zhaosheng/asr_damo_websocket/online/microservice/models/speech_eres2net_sv_zh-cn_16k-common/examples/speaker1_b_en_16k.wav")
    
    
    # wav_data = wav_data.reshape(-1)
    a = emb.encode_batch(wav_data)
    print(a)
    print(a.shape)
