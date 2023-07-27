# coding = utf-8
# @Time    : 2023-07-21  16:10:25
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Pooling Layers.

# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
# Copyright (c) 2023 Sheng Zhao (zhaosheng@nuaa.edu.cn)
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

"""
Pooling functions to aggregate frame-level deep features
into segment-level speaker embeddings

High-order statistics are surprisingly effective, TSDP acts similarly as TSTP,
even though we remove the mean statistic, on Voxceleb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """
    # x (batch_size, channels, temporal_dim) --mean(dim=-1)--> (batch_size, channels) --flatten(start_dim=1)--> (batch_size, channels * 1)
    def __init__(self, in_dim=0, **kwargs):
        super(TAP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_dim=1) # (batch_size, num_channels, spatial_dim) -> (batch_size, num_channels * spatial_dim)
        return pooling_mean

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """
    # x (batch_size, channels, temporal_dim) --std(dim=-1)--> (batch_size, channels) --flatten(start_dim=1)--> (batch_size, channels * 1

    def __init__(self, in_dim=0, **kwargs):
        super(TSDP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_std = pooling_std.flatten(start_dim=1) # (batch_size, num_channels, spatial_dim) -> (batch_size, num_channels * spatial_dim)
        return pooling_std

    def get_out_dim(self):
        self.out_dim = self.in_dim
        return self.out_dim


class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    # x (batch_size, channels, temporal_dim) --mean(dim=-1)--> (batch_size, channels) --flatten(start_dim=1)--> (batch_size, channels * 1)
    # x (batch_size, channels, temporal_dim) --std(dim=-1)--> (batch_size, channels) --flatten(start_dim=1)--> (batch_size, channels * 1)
    # --cat--> (batch_size, channels * 2)

    def __init__(self, in_dim=0, **kwargs):
        super(TSTP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1) # (batch_size, num_channels, spatial_dim) -> (batch_size, num_channels * spatial_dim)
        pooling_std = pooling_std.flatten(start_dim=1) # (batch_size, num_channels, spatial_dim) -> (batch_size, num_channels * spatial_dim)
        stats = torch.cat((pooling_mean, pooling_std), 1) # -> (batch_size, 2 * num_channels * spatial_dim)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim


class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """
    # x (batch_size, channels, temporal_dim)
    # if self.global_context_att:
    #     -- context_mean --> 计算x在时间维度上的均值，得到context_mean，形状为(batch_size, channels, 1)，然后利用.expand_as(x)将其扩展成与x相同的形状 --> (batch_size, channels, temporal_dim)
    #     -- context_std --> 计算x在时间维度上的标准差，得到context_std，形状为(batch_size, channels, 1)，然后利用.expand_as(x)将其扩展成与x相同的形状 --> (batch_size, channels, temporal_dim)
    #     -- cat --> (batch_size, channels * 3, temporal_dim)
    # else:
    #     -- x --> (batch_size, channels, temporal_dim)
    # -- linear1 --> (batch_size, bottleneck_dim, temporal_dim) 这里使用的卷积核大小为1，所以只在通道维度上进行卷积操作
    # -- linear2 --> alpha (batch_size, channels, temporal_dim) 
    # -- softmax(dim=2)（时间维度） --> (batch_size, channels, temporal_dim) 得到归一化的注意力权重
    # 将alpha与x按维度2相乘，然后在维度2上求和，得到mean -- mean -->(batch_size, bottleneck_dim) 得到加权平均值
    # 将alpha与x的平方按维度2相乘，然后在维度2上求和，再减去mean的平方，最后取平方根得到std -- var --> (batch_size, bottleneck_dim) 得到加权方差
    # -- cat --> (batch_size, bottleneck_dim * 2) 最终得到的是加权平均值和加权方差的拼接
    # Attentive体现在以下几个方面：
    # 在forward方法中，通过对输入x进行一系列的计算和操作，得到了注意力权重alpha。这里的alpha是通过两层线性变换得到的，并经过softmax函数归一化得到的。这个过程即为注意力机制的体现。
    # 注意力权重alpha与输入x按维度2相乘，并在维度2上求和，得到了加权平均值mean和加权方差var。这里使用注意力权重作为权重对输入进行加权求和操作，体现了对不同时间维度的关注程度不同。
    # 最后将加权平均值mean和加权方差std拼接在一起作为输出，表示对输入的统计汇总信息。这种统计汇总信息是根据注意力权重以及输入的特征计算得到的，体现了对不同时间维度的重要性加权聚合的思想。

    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False, **kwargs):
        super(ASTP, self).__init__()
        self.in_dim = in_dim
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(
                in_dim * 3, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                in_dim, bottleneck_dim,
                kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim,
                                 kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
            or a 4-dimensional tensor in resnet architecture (B,C,F,T)
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MHASTP(torch.nn.Module):
    """ Multi head attentive statistics pooling
    Reference:
        Self Multi-Head Attention for Speaker Recognition
        https://arxiv.org/pdf/1906.09890.pdf
    """
    # x (batch_size, channels, temporal_dim)
    r"""
    Multi head attentive statistics pooling（MHASTP）：

    MHASTP在ASP的基础上进行了改进，引入多头机制来增加模型的表达能力。
    MHASTP将输入序列按照头数进行分割，并对每个头单独计算注意力权重和统计特征。
    每个头都有自己的卷积层和注意力机制，以捕捉不同的局部特征和注意力分布。
    最后，将所有头的特征进行拼接得到最终的特征表示。
    优点：MHASTP通过多头机制提高了模型的表达能力，能够更好地捕捉输入序列的局部特征和注意力分布。
    缺点：MHASTP的计算量较大，且注意力权重的计算是独立的，没有考虑不同头之间的关系。"""

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 head_num=2,
                 d_s=1,
                 bottleneck_dim=64,
                 **kwargs):
        super(MHASTP, self).__init__()
        assert (in_dim % head_num
                ) == 0  # make sure that head num can be divided by input_dim
        self.in_dim = in_dim
        self.head_num = head_num
        d_model = int(in_dim / head_num)
        channel_dims = [bottleneck_dim for i in range(layer_num + 1)]
        if d_s > 1:
            d_s = d_model
        else:
            d_s = 1
        self.d_s = d_s
        channel_dims[0], channel_dims[-1] = d_model, d_s
        heads_att_trans = []
        for i in range(self.head_num):
            att_trans = nn.Sequential()
            for i in range(layer_num - 1):
                att_trans.add_module(
                    'att_' + str(i),
                    nn.Conv1d(channel_dims[i], channel_dims[i + 1], 1, 1))
                att_trans.add_module('tanh' + str(i), nn.Tanh())
            att_trans.add_module(
                'att_' + str(layer_num - 1),
                nn.Conv1d(channel_dims[layer_num - 1], channel_dims[layer_num],
                          1, 1))
            heads_att_trans.append(att_trans)
        self.heads_att_trans = nn.ModuleList(heads_att_trans)

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        bs, f_dim, t_dim = input.shape
        chunks = torch.chunk(input, self.head_num, 1)
        # split
        chunks_out = []
        # for i in range(self.head_num):
        #     att_score = self.heads_att_trans[i](chunks[i])
        for i, layer in enumerate(self.heads_att_trans):
            att_score = layer(chunks[i])
            alpha = F.softmax(att_score, dim=-1)
            mean = torch.sum(alpha * chunks[i], dim=2)
            var = torch.sum(alpha * chunks[i]**2, dim=2) - mean**2
            std = torch.sqrt(var.clamp(min=1e-7))
            chunks_out.append(torch.cat((mean, std), dim=1))
        out = torch.cat(chunks_out, dim=1)
        return out

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim


class MQMHASTP(torch.nn.Module):
    """ An attentive pooling
    Reference:
        multi query multi head attentive statistics pooling
        https://arxiv.org/pdf/2110.05042.pdf
    Args:
        in_dim: the feature dimension of input
        layer_num: the number of layer in the pooling layer
        query_num: the number of querys
        head_num: the number of heads
        bottleneck_dim: the bottleneck dimension

    SA (H = 1, Q = 1, n = 2, d_s = 1) ref:
        https://www.danielpovey.com/files/2018_interspeech_xvector_attention.pdf
    MHA (H > 1, Q = 1, n = 1, d_s = 1) ref:
        https://arxiv.org/pdf/1906.09890.pdf
    AS (H = 1, Q > 1, n = 2, d_s = 1) ref:
        https://arxiv.org/pdf/1803.10963.pdf
    VSA (H = 1, Q > 1, n = 2, d_s = d_h) ref:
        http://www.interspeech2020.org/uploadfile/pdf/Mon-2-10-5.pdf
    """
    r"""
    Multi query Multi head attentive statistics pooling (MQMHASTP)是一种多查询多头的注意力统计池化方法，
    相比于单一查询的Attentive statistics pooling (ASP)，以及多头的Multi head attentive statistics pooling (MHASTP)，
    具有以下区别和优点：

    多查询（Multi query）：MQMHASTP引入了多查询的概念，即在特征池化过程中，使用多个查询来获取更丰富的信息。
    每个查询可以关注不同的特征子空间，从而提高模型对输入序列的表达能力。

    多头（Multi head）：MQMHASTP借鉴了MHASTP的思想，在每个查询中同样引入多头机制，
    以捕获输入序列中的局部特征和注意力分布。每个头都有自己的卷积层和注意力机制，可以独立学习不同位置的重要性和特征表示，
    进一步提高模型的表达能力。

    统计池化（Attentive statistics pooling）：MQMHASTP和MHASTP都采用了统计池化的方式，
    即对不同位置的特征进行加权求和，并计算均值和标准差作为最终的统计特征。这样可以将序列编码成固定长度的向量表示，
    并保留一定的统计信息。

    增强表达能力：相比于ASP和MHASTP，MQMHASTP具有更高的表达能力。
    多查询和多头机制使得模型能够更好地捕捉输入序列中的不同特征子空间和重要性分布。
    通过引入多个查询和多个注意力头，MQMHASTP可以提取更加丰富和区分性的特征，适用于需要更高级别特征的任务。
"""

    def __init__(self,
                 in_dim,
                 layer_num=2,
                 query_num=2,
                 head_num=8,
                 d_s=2,
                 bottleneck_dim=64,
                 **kwargs):
        super(MQMHASTP, self).__init__()
        self.n_query = nn.ModuleList([
            MHASTP(in_dim,
                   layer_num=layer_num,
                   head_num=head_num,
                   d_s=d_s,
                   bottleneck_dim=bottleneck_dim) for i in range(query_num)
        ])
        self.query_num = query_num
        self.in_dim = in_dim

    def forward(self, input):
        """
        input: a 3-dimensional tensor in xvector architecture
            or a 4-dimensional tensor in resnet architecture
            0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(input.shape) == 4:  # B x F x T
            input = input.reshape(input.shape[0],
                                  input.shape[1] * input.shape[2],
                                  input.shape[3])
        assert len(input.shape) == 3
        res = []
        for i, layer in enumerate(self.n_query):
            res.append(layer(input))
        out = torch.cat(res, dim=-1)
        return out

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2 * self.query_num
        return self.out_dim

class GSP(nn.Module):
    """
    Global statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    # 和 TSTP 一样

    def __init__(self, in_dim=0, **kwargs):
        super(GSP, self).__init__()
        self.in_dim = in_dim

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats

    def get_out_dim(self):
        self.out_dim = self.in_dim * 2
        return self.out_dim
        
if __name__ == '__main__':
    data = torch.randn(16, 512, 10, 35)
    # model = StatisticsPooling()
    model = MQMHASTP(512 * 10)
    model = MHASTP(512 * 10)
    model = MQMHASTP(512 * 10, context=False)
    print(model)

    out = model(data)
    print(out.shape)
    print(model.get_out_dim())
