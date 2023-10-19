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

        self.tdnn_layers = nn.ModuleList([
            TDNN(input_dim, output_dim, context_size, dilation=dilation)
            for context_size, dilation in zip(context_sizes, dilations)
        ])
        self.stats_pooling = StatsPooling()
        self.fc = FC(output_dim * 2 * len(context_sizes), output_dim)
    
    def forward(self, x):
        """
        x: (batch_size, input_dim, sequence_length)
        """
        for tdnn_layer in self.tdnn_layers:
            x = tdnn_layer(x)
        x = self.stats_pooling(x)
        x = self.fc(x)
        return x

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
    
    def forward(self, x):
        """
        x: (batch_size, input_dim, sequence_length)
        """
        x = super(ECAPA_RetNet, self).forward(x)
        x = self.retnet(x)
        return x

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        x_n: (batch_size, input_dim)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        x_n = super(ECAPA_RetNet, self).forward(x_n.unsqueeze(-1))
        x_n, s_ns = self.retnet.forward_recurrent(x_n, s_n_1s, n)
        return x_n.squeeze(-1), s_ns

class ECAPA_RetNet_CLM(ECAPA_RetNet):
    def __init__(self, input_dim, output_dim, context_sizes, dilations, layers, hidden_dim, ffn_size, heads, vocab_size):
        super(ECAPA_RetNet_CLM, self).__init__(input_dim, output_dim, context_sizes, dilations, layers, hidden_dim, ffn_size, heads)
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, output_dim)
        self.proj = nn.Parameter(torch.randn(output_dim, vocab_size, dtype=torch.float32) / output_dim)
    
    def forward(self, input_ids):
        """
        input_ids: (batch_size, sequence_length)
        """
        x = self.embed(input_ids)
        x = super(ECAPA_RetNet_CLM, self).forward(x.permute(0, 2, 1))
        x = x @ self.proj.to(x.dtype)

        return x.real
    
    def forward_recurrent(self, input_ids, s_n_1s, n):
        """
        input_ids: (batch_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        x = self.embed(input_ids)
        x, s_ns = super(ECAPA_RetNet_CLM, self).forward_recurrent(x.unsqueeze(-1).permute(0, 2, 1), s_n_1s, n)
        x = x @ self.proj.to(x.dtype)

        return x.squeeze(-1).real, s_ns
    
    def sample(self, input_ids, sample_length, temperature=1.0):
        """
        input_ids: (batch_size, sequence_length)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
        """
        s_n_1s = [
            [
                torch.zeros(self.output_dim // self.heads, self.output_dim // self.heads, dtype=torch.complex64).unsqueeze(0).repeat(input_ids.shape[0], 1, 1)
                for _ in range(self.heads)
            ] for _ in range(self.layers)
        ]
        for i in range(input_ids.shape[1]):
            x, s_n_1s = self.forward_recurrent(input_ids[:, i], s_n_1s, i+1)
        
        # get softmax of x (real part only)
        x = x.real / temperature
        x = torch.softmax(x, dim=-1)
        x = torch.multinomial(x, num_samples=1)
        next_char = x[:, -1]
        output_ids = []
        # now start sampling!
        for i in range(sample_length):
            x, s_n_1s = self.forward_recurrent(next_char, s_n_1s, i+1)
            x = x.real / temperature
            x = torch.softmax(x, dim=-1)
            x = torch.multinomial(x, num_samples=1)
            next_char = x[:, -1]
            output_ids.append(next_char)

        output_ids = torch.stack(output_ids, dim=1)

        return output_id


if __name__ == '__main__':
    model = ECAPA_RetNet(input_dim=80, output_dim=192, context_sizes=[3, 5, 7], dilations=[1, 2, 3], layers=4, hidden_dim=192, ffn_size=768, heads=4)
    x = torch.zeros(10, 200, 80)
    output = model(x)
    print("Output shape:", output.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
   