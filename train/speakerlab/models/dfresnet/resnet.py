# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import speakerlab.models.resnet.pooling_layers as pooling_layers

class Depthwise(nn.Module):
    # conv2d → depthwise conv2d: In the original bottleneck
    # block of ResNet, the standard 3x3 2-dimensional convolution
    # operator is adopted. In order to reduce the parameter number, we attempt to substitute the standard 3x3 convolution with
    # depthwise convolution [25] (Figure 2 (B)), which is a special
    # case of grouped convolution where the number of groups equals
    # the number of channels. From Table 2, we can see that this
    # change reduces the parameter number to 7.18M and FLOPs to
    # 1.75G, resulting in further performance degradation to 1.96.
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(Depthwise, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_planes, bias=bias)
        # self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        hidden_planes = self.expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes,hidden_planes, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(hidden_planes)

        # Deepwise conv
        self.conv2 = Depthwise(hidden_planes, hidden_planes, stride=stride,bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_planes)

        self.conv3 = nn.Conv2d(hidden_planes,
                               out_planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(planes,
                          out_planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("\t\tAfter conv1: {}".format(out.shape))
        out = F.relu(self.bn2(self.conv2(out)))
        # print("\t\tAfter conv2: {}".format(out.shape))
        out = self.bn3(self.conv3(out))
        # print("\t\tAfter conv3: {}".format(out.shape))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DFResNet(nn.Module):
    def __init__(self,
                 num_blocks=[3,3,9,3],
                 block=Bottleneck,
                 m_channels=32,
                 feat_dim=80,
                 embed_dim=512,
                 pooling_func='GSP'):
        super(DFResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        # self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=1)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=1)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=1)
        self.pool = getattr(pooling_layers, pooling_func)(
            in_dim=self.stats_dim * block.expansion)
        # self.pool = self.pool.to('cuda:0')
        self.pool_out_dim = self.pool.get_out_dim()
        self.seg_1 = nn.Linear(5120, #self.pool_out_dim
                               embed_dim)
        #if self.two_emb_layer:
        #    self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
        #    self.seg_2 = nn.Linear(embed_dim, embed_dim)
        self.seg_bn_1 = nn.Identity()
        # self.seg_2 = nn.Identity()
        # downsample by a separate downsampling layer which consists of a 3×3 convolution layer with stride 2 and padding 1 followed by a batchnorm
        # [10, 32, 80, 200] -> [10, 64, 40, 100]
        self.downsample_1 = nn.Sequential(
            nn.Conv2d(m_channels, m_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels*2)
        )
        self.downsample_2 = nn.Sequential(
            nn.Conv2d(m_channels * 2, m_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels * 4)
        )
        self.downsample_3 = nn.Sequential(
            nn.Conv2d(m_channels * 4, m_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(m_channels * 8)
        )
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T) 10,200,80 -> 10,80,200
        x = x.unsqueeze_(1) # 10,80,200 -> 10,1,80,200
        # print(f"Input: {x.shape}")
        out = F.relu(self.bn1(self.conv1(x)))
        # print(f"Conv1: {out.shape}") # 10,32,80,200
        out = self.layer1(out)
        # print(f"Layer 1: {out.shape}") # 10,32,80,200
        out = self.downsample_1(out) 
        # print(f"Layer 1 downsample: {out.shape}") # 10,64,40,100
        out = self.layer2(out)
        # print(f"Layer 2: {out.shape}") # 10,64,40,100
        out = self.downsample_2(out)
        # print(f"Layer 2 downsample: {out.shape}")
        out = self.layer3(out)
        # print(f"Layer 3: {out.shape}") # 10,128,20,50
        out= self.downsample_3(out)
        # print(f"Layer 3 downsample: {out.shape}")
        out = self.layer4(out)
        # print(f"Layer 4: {out.shape}") # 10,256,10,25


        stats = self.pool(out)
        # print(f"Stats: {stats.shape}")

        embed_a = self.seg_1(stats)
        # print(f"Embed A: {embed_a.shape}")
        return embed_a




def DFResNet56(feat_dim, embed_dim, pooling_func='GSP', two_emb_layer=False):
    return ResNet(Bottleneck, [3,3,9,3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)




if __name__ == '__main__':
    x = torch.zeros(10, 200, 80)
    x=x.to('cuda:0')
    model = DFResNet56(feat_dim=80,
                     embed_dim=256,
                     pooling_func='GSP') #MQMHASTP
    # to cuda
    model = model.to('cuda:0')
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    x_np = torch.randn(1, 200, 80)
    flops, params = profile(model, inputs=(x_np, ))
    print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
