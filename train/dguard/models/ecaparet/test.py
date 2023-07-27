import torch
import torch.nn as nn

from retention import MultiScaleRetention
from util import ComplexFFN, ComplexGroupNorm, ComplexLayerNorm


class TDNN(nn.Module):
    def __init__(self, input_dim, output_dim, context_size, dilation=1):
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.conv = nn.Conv1d(input_dim, output_dim, context_size, dilation=dilation)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        x: (batch_size, input_dim, sequence_length)
        """
        x = self.conv(x)
        x = self.relu(x)
        return x

class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling, self).__init__()
    
    def forward(self, x):
        """
        x: (batch_size, hidden_dim, sequence_length)
        """
        mean = torch.mean(x, dim=-1)
        std = torch.std(x, dim=-1)
        return torch.cat([mean, std], dim=-1)

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        x: (batch_size, input_dim)
        """
        x = self.fc(x)
        x = self.relu(x)
        return x

class ECAPA_TDNN(nn.Module):
    def __init__(self, input_dim, output_dim, context_sizes, dilations):
        super(ECAPA_TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_sizes = context_sizes
        self.dilations = dilations

        self.tdnn_layers = nn.ModuleList([TDNN(input_dim, output_dim, 3, dilation=1)]+[
            TDNN(output_dim, output_dim, context_size, dilation=dilation)
            for context_size, dilation in zip(context_sizes, dilations)
        ])
        self.stats_pooling = StatsPooling()
        self.fc = FC(input_dim, output_dim)
    
    def forward(self, x):
        """
        x: (batch_size, input_dim, sequence_length)
        """
        # for tdnn_layer in self.tdnn_layers:
        #     x = tdnn_layer(x)
        # x = self.stats_pooling(x)
        # x = self.fc(x)
        # return random output(shape: (batch_size, output_dim,100))\
        return torch.randn(x.shape[0], self.output_dim,192)
        # return x

class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            ComplexFFN(hidden_dim, ffn_size)
            for _ in range(layers)
        ])
        self.layer_norm = ComplexLayerNorm(hidden_dim)
    
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norm(X)) + X
            X = self.ffns[i](self.layer_norm(Y)) + Y

        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norm(x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norm(y_n)) + y_n
        
        return x_n, s_ns
class ECAPA_RetNet(ECAPA_TDNN):
    def __init__(self, input_dim, output_dim, context_sizes, dilations, layers, hidden_dim, ffn_size, heads):
        super(ECAPA_RetNet, self).__init__(input_dim, output_dim, context_sizes, dilations)
        self.retnet = RetNet(layers, hidden_dim, ffn_size, heads)
        
        # 添加声纹编码层
        self.voice_encoding = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        x: (batch_size, input_dim, sequence_length)
        """
        x = super(ECAPA_RetNet, self).forward(x)
        print(f"After ECAPA_TDNN: {x.shape}")
        print(x)
        x = self.retnet(x)

        print(f"After retnet: {x.shape}")
        print(x)
        
        x = self.voice_encoding(x)  # 使用声纹编码层进行处理
        print(f"After voice encoding: {x.shape}")
        print(x)
        return x

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        x_n: (batch_size, input_dim)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        x_n = super(ECAPA_RetNet, self).forward(x_n.unsqueeze(-1))
        x_n, s_ns = self.retnet.forward_recurrent(x_n, s_n_1s, n)
        x_n = self.voice_encoding(x_n)  # 使用声纹编码层进行处理
        return x_n.squeeze(-1), s_ns

class ECAPA_RetNet_CLM(ECAPA_RetNet):
    def __init__(self, input_dim, output_dim, context_sizes, dilations, layers, hidden_dim, ffn_size, heads, vocab_size):
        super(ECAPA_RetNet_CLM, self).__init__(input_dim, output_dim, context_sizes, dilations, layers, hidden_dim, ffn_size, heads)
        self.vocab_size = 0  # 将vocab_size设为0，不使用embed和proj层
    
    def forward(self, input_ids):
        """
        input_ids: (batch_size, sequence_length)
        """
        x = super(ECAPA_RetNet_CLM, self).forward(input_ids)
        return x

    def forward_recurrent(self, input_ids, s_n_1s, n):
        """
        input_ids: (batch_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        x, s_ns = super(ECAPA_RetNet_CLM, self).forward_recurrent(input_ids, s_n_1s, n)
        return x, s_ns
    
    # def sample(self, input_ids, sample_length, temperature=1.0):
    #     # 省略...

if __name__ == '__main__':
    model = ECAPA_RetNet_CLM(input_dim=80, output_dim=192, context_sizes=[3, 5, 7], dilations=[1, 2, 3], layers=4, hidden_dim=192, ffn_size=768, heads=4, vocab_size=10000)
    # x = torch.zeros_like
    x = torch.randn(10, 80, 100)
    output = model(x)
    print("Output shape:", output.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
