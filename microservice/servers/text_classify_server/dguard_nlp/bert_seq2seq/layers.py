import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, ent_type_size, inner_dim, RoPE=True, trill_mask=True):
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.trill_mask = trill_mask
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def rope(self, batch_size, seq_len, dim, qw, kw):
        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, dim)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None,::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[...,::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[...,::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, last_hidden_state, padding_mask):
        self.device = last_hidden_state.device
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[...,:self.inner_dim], outputs[...,self.inner_dim:] # TODO:修改为Linear获取？

        if self.RoPE:
            qw, kw = self.rope(batch_size, seq_len, self.inner_dim, qw, kw)

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = padding_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits*pad_mask - (1-pad_mask)*1e12

        # 排除下三角
        if self.trill_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits/self.inner_dim**0.5

    def compute_loss(self, logits, labels):
        # logits:
        # labels:
        pass

        bh = logits.shape[0] * logits.shape[1]
        labels = torch.reshape(labels, shape=(bh, -1))
        logits = torch.reshape(logits, shape=(bh, -1))
        return multilabel_crossentropy(logits, labels)

    def compute_loss_sparse(self, logits, labels, mask_zero=False):
        return sparse_multilabel_categorical_crossentropy(y_pred=logits, y_true=labels, mask_zero=mask_zero)


def multilabel_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()

def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    shape = y_pred.shape
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss

class CRFLayer(nn.Module):
    """
    """
    def __init__(self, output_dim):
        super(CRFLayer, self).__init__()

        self.output_dim = output_dim
        self.trans = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.trans.data.uniform_(-0.1, 0.1)

    def compute_loss(self, y_pred, y_true, mask):
        """
        计算CRF损失
        """
        y_pred = y_pred * mask
        y_true = y_true * mask
        target_score = self.target_score(y_pred, y_true)
        log_norm = self.log_norm_step(y_pred, mask)
        log_norm = self.logsumexp(log_norm, dim=1)# 计算标量
        return log_norm - target_score

    def forward(self, y_pred, y_true, mask):
        """
        y_true: [[1, 2, 3], [2, 3, 0] ]
        mask: [[1, 1, 1], [1, 1, 0]]
        """
        if y_pred.shape[0] != mask.shape[0] or y_pred.shape[1] != mask.shape[1]:
            raise Exception("mask shape is not match to y_pred shape")
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1))
        mask = mask.float()
        y_true = y_true.reshape(y_pred.shape[:-1])
        y_true = y_true.long()
        y_true_onehot = F.one_hot(y_true, self.output_dim)
        y_true_onehot = y_true_onehot.float()

        return self.compute_loss(y_pred, y_true_onehot, mask)

    def target_score(self, y_pred, y_true):
        """
        计算状态标签得分 + 转移标签得分
        y_true: (batch, seq_len, out_dim)
        y_pred: (batch, seq_len, out_dim)
        """
        # print(y_pred.shape)
        # print(y_true.shape)
        point_score = torch.einsum("bni,bni->b", y_pred, y_true)
        trans_score = torch.einsum("bni,ij,bnj->b", y_true[:, :-1], self.trans, y_true[:, 1: ])

        return point_score + trans_score

    def log_norm_step(self, y_pred, mask):
        """
        计算归一化因子Z(X)
        """
        state = y_pred[:, 0] # 初始Z(X)
        y_pred = y_pred[:, 1: ].contiguous()
        mask = mask[:, 1:].contiguous()
        batch, seq_len, out_dim = y_pred.shape
        for t in range(seq_len):
            cur_mask = mask[:, t]
            state = torch.unsqueeze(state, 2) # (batch, out_dim, 1)
            g = torch.unsqueeze(self.trans, 0) # (1, out_dim, out_dim)
            outputs = self.logsumexp(state + g, dim=1) # batch, out_dim
            outputs = outputs + y_pred[:, t]
            outputs = cur_mask * outputs + (1 - cur_mask) * state.squeeze(-1)
            state = outputs

        return outputs

    def logsumexp(self, x, dim=None, keepdim=False):
        """
        避免溢出
        """
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        out = xm + torch.log(torch.sum(torch.exp(x - xm), dim=dim, keepdim=True))
        return out if keepdim else out.squeeze(dim)
