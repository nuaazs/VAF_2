import json
import os
from typing import List
import torch
import numpy as np
import torch.nn.functional as F
import time

join = os.path.join
def load_config(config_path):
    with open(config_path) as f :
        j = json.load(f)

    return j

class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` enforcing an exponential penalty on repeated sequences.
    Args:
        repetition_penalty (:obj:`float`):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:


        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores

class TemperatureLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).
    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class TopPLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.
    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to top_p or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        # print(sorted_logits.softmax(dim=-1))
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TopKLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.
    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = top_k
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(max(self.top_k, self.min_tokens_to_keep), scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class ListProcessor(LogitsProcessor):
    def __init__(self, list_processor: List[LogitsProcessor]) -> None:
        super().__init__()
        self.list_processor = list_processor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for processor in self.list_processor:

            scores = processor(input_ids, scores)

        return scores


def viterbi_decode(nodes, trans):
    """
    维特比算法 解码
    nodes: (seq_len, target_size)
    trans: (target_size, target_size)
    """
    scores = nodes[0]
    scores[1:] -= 100000 # 刚开始标签肯定是"O"
    target_size = nodes.shape[1]
    seq_len = nodes.shape[0]
    labels = torch.arange(0, target_size).view(1, -1)
    path = labels
    for l in range(1, seq_len):
        scores = scores.view(-1, 1)
        M = scores + trans + nodes[l].view(1, -1)
        scores, ids = M.max(0)
        path = torch.cat((path[:, ids], labels), dim=0)

    return path[:, scores.argmax()]

def decode_labels(labels, target):
    entities = []
    starting = False
    for i, label in enumerate(labels):
        if label > 0:
            label_name = target[label]

            if label_name[0] == "B":
                starting = True
                entities.append([[i], label_name[2:]])
            elif starting:
                entities[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return entities

def bert_predict_generate(model, input_ids, token_type_ids):
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        score = model(**{"input_ids": input_ids, "token_type_ids": token_type_ids})["logits"]
    return score

def gpt_predict_generate(model, input_ids):
    with torch.no_grad():
        device = next(model.parameters()).device
        input_ids = torch.tensor(input_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
        score = model(**{"input_ids": input_ids})["logits"]
    return score

def bert_beam_search(model, token_ids, word2ix, token_type_ids=None,
                      beam_size=1, out_max_length=50):
    """
    beam-search操作
    """
    sep_id = word2ix["[SEP]"]
    if token_type_ids is None:
        token_type_ids = np.zeros_like(token_ids).astype(np.long)

    output_ids = None
    # 用来保存累计得分
    with torch.no_grad():
        output_scores = np.zeros([1])
        for step in range(out_max_length):
            if step == 0:
                scores = bert_predict_generate(model, token_ids, token_type_ids)
                # 重复beam-size次 输入ids
                token_ids = np.tile(token_ids.reshape([1, -1]), [beam_size, 1])
                token_type_ids = np.tile(token_type_ids.reshape([1, -1]), [beam_size, 1])
            else:
                scores = bert_predict_generate(model, new_input_ids,
                                      new_token_type_ids)

            logit_score = F.log_softmax(scores[:, -1], dim=-1).cpu().numpy()

            logit_score = output_scores.reshape([-1, 1]) + logit_score # 累计得分
            ## 取topk的时候我们是展平了然后再去调用topk函数
            # 展平
            logit_score = logit_score.reshape([-1])
            hype_pos = np.argpartition(logit_score, -beam_size, axis=-1)[-beam_size:]
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1]) # 行索引
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int).reshape([-1, 1]) # 列索引

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else :
                output_ids = np.concatenate([output_ids[indice1], indice2], axis=1).astype(np.int)

            new_input_ids = np.concatenate([token_ids, output_ids], axis=1)
            new_token_type_ids = np.concatenate([token_type_ids, np.ones_like(output_ids)], axis=1)

            end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                # 说明出现终止了～
                return output_ids[best_one][:-1]
            else :
                # 保留未完成部分
                flag = (end_counts < 1)  # 标记未完成序列
                if not flag.all():  # 如果有已完成的
                    token_ids = token_ids[flag]
                    token_type_ids = token_type_ids[flag]
                    new_input_ids = new_input_ids[flag]
                    new_token_type_ids = new_token_type_ids[flag]
                    output_ids = output_ids[flag]  # 扔掉已完成序列
                    output_scores = output_scores[flag]  # 扔掉已完成序列
                    beam_size = flag.sum()  # topk相应变化

        return output_ids[output_scores.argmax()]

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def t5_random_sample(model, tokenizer, text, input_max_length, out_max_length,
                     top_k, top_p, repetition_penalty, temperature, device):

    token_ids = tokenizer.encode_plus(text, max_length=input_max_length)["input_ids"]
    token_ids = torch.tensor(token_ids, device=device, dtype=torch.long).view(1, -1)
    output_ids = []
    input_decoder_ids = torch.tensor(tokenizer.token_start_id, device=device, dtype=torch.long).view(1, -1)
    lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
          TemperatureLogitsProcessor(temperature=temperature),
          TopKLogitsProcessor(top_k=top_k),
          TopPLogitsProcessor(top_p=top_p),
          ]
    list_processor = ListProcessor(lp)
    with torch.no_grad():
        for step in range(out_max_length):
            if step == 0:
                model_out = model(**{"input_ids":token_ids, "decoder_input_ids":input_decoder_ids})
                scores = model_out["logits"]
                encoder_last_hidden_state = model_out["encoder_last_hidden_state"]
            else :
                model_out = model(**{"encoder_last_hidden_state":encoder_last_hidden_state, "decoder_input_ids":input_decoder_ids})
                scores = model_out["logits"]

            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.token_unk_id] = -float('Inf')
            # filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
            filtered_logits = list_processor(input_decoder_ids, logit_score)

            filterd_logits_prob = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filterd_logits_prob, num_samples=1)
            if tokenizer.token_end_id == next_token.item():
                break
            output_ids.append(next_token.item())
            input_decoder_ids = torch.cat((input_decoder_ids, next_token.long()), dim=1)
    return tokenizer.decode(output_ids)

def gpt_random_sample(model, tokenizer, text, input_max_length, out_max_length,
                      top_k, top_p, repetition_penalty, temperature, device, add_sep=False):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    if add_sep:
        token_ids = tokenizer_out["input_ids"]
    else :
        token_ids = tokenizer_out["input_ids"][:-1]

    lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
          TemperatureLogitsProcessor(temperature=temperature),
          TopKLogitsProcessor(top_k=top_k),
          TopPLogitsProcessor(top_p=top_p),
          ]
    list_processor = ListProcessor(lp)

    token_ids = torch.tensor(token_ids, device=device, dtype=torch.long).view(1, -1)
    output_ids = []
    sep_id = tokenizer.token_end_id
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(**{"input_ids": token_ids})["logits"]
            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.token_unk_id] = -float('Inf')

            filtered_logits = list_processor(token_ids, logit_score)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if sep_id == next_token.item():
                break
            output_ids.append(next_token.item())
            token_ids = torch.cat((token_ids, next_token.long()), dim=1)

    return tokenizer.decode(output_ids)

def gpt_random_sample_from_ids(model, tokenizer, input_ids, out_max_length,
                      top_k, top_p, repetition_penalty, temperature, device):

    lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
          TemperatureLogitsProcessor(temperature=temperature),
          TopKLogitsProcessor(top_k=top_k),
          TopPLogitsProcessor(top_p=top_p),
          ]
    list_processor = ListProcessor(lp)

    token_ids = torch.tensor(input_ids, device=device, dtype=torch.long).view(1, -1)
    output_ids = []
    sep_id = tokenizer.token_end_id
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(**{"input_ids": token_ids})["logits"]
            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.token_unk_id] = -float('Inf')

            filtered_logits = list_processor(token_ids, logit_score)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if sep_id == next_token.item():
                break
            output_ids.append(next_token.item())
            token_ids = torch.cat((token_ids, next_token.long()), dim=1)

    return tokenizer.decode(output_ids)

def glm_random_sample(model, tokenizer, text, input_max_length, out_max_length, top_k, top_p,
                      repetition_penalty, temperature, device, prefix_flag="", post_flag=""):

    return glm_generate_sample(model, tokenizer, text, seq_length=input_max_length,
                                    out_seq_length=out_max_length, top_k=top_k,
                                    temperature=temperature, repetition_penalty=repetition_penalty)


# def glm_generate_sample(model, tokenizer,
#                         text, top_k=40,
#                         seq_length=512,
#                         out_seq_length=512,
#                         eod_token=50000,
#                         temperature=0.9,
#                         prefix_flag=""):
#     device = next(model.parameters()).device
#     model.eval()
#
#     # generation_mask = '[gMASK]'
#     # if 'MASK]' not in text:
#     #     text += ' ' + generation_mask
#     # context_tokens = tokenizer.EncodeAsIds(text, maxlen=seq_length)
#     # context_tokens = [tokenizer.get_command('ENC').Id] + context_tokens
#     # if not text.endswith('[gMASK]'):
#     #     context_tokens = context_tokens + [tokenizer.get_command('eos').Id]
#     # context_length = len(context_tokens)
#     # context_length_tensor = torch.cuda.LongTensor([context_length])
#     # context_length = context_length_tensor[0].item()
#     # context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
#     # text = tokenizer.DecodeIds(context_tokens_tensor.tolist())
#
#     tokenizer_out = tokenizer.encode_plus(text, max_length=seq_length, prefix_flag=prefix_flag)
#     tokens = torch.from_numpy(tokenizer_out["input_ids"])
#     # position_ids = torch.from_numpy(tokenizer_out["position_ids"])
#     # attention_mask = torch.from_numpy(tokenizer_out["attention_mask"])
#
#     start_time = time.time()
#     mems = []
#     # tokens = context_tokens_tensor
#     tokens = tokens.view(1, -1).contiguous()
#     tokens = tokens.to(device)
#     attention_mask = torch.tensor([tokens.size(1)], device=device, dtype=torch.long)
#     position_ids = torch.arange(tokens.size(1), device=device, dtype=torch.long)
#     block_position_ids = torch.zeros(tokens.size(1), device=device, dtype=torch.long)
#     position_ids = torch.stack((position_ids, block_position_ids), dim=0)
#     position_ids = position_ids.unsqueeze(0)
#     mask_tokens = ['MASK', 'sMASK', 'gMASK']
#     mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
#     end_tokens = [tokenizer.get_command('eop').Id, eod_token]
#     mask_positions = []
#     for token in mask_tokens:
#         mask_positions += (tokens == token).nonzero(as_tuple=True)[0].tolist()
#     mask_positions.sort()
#     output_ = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask, return_memory=True)
#     mems=output_['hidden_states']
#     for mask_position in mask_positions:
#         position = mask_position
#         tokens, mems = glm_sample_sequence(model, tokenizer, tokens, position,
#                                            mems=mems, end_tokens=end_tokens,
#                                            out_seq_length=out_seq_length,temperature=temperature, top_k=top_k)
#     output_tokens_list = tokens.view(-1).contiguous()
#
#     decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
#
#     return decode_tokens

def glm_generate_sample(model,  tokenizer,text, top_k=40,seq_length=512,out_seq_length=512,
                        eod_token=50000,temperature=0.9, repetition_penalty=1.0):
    # device=torch.cuda.current_device()
    device = next(model.parameters()).device
    model.eval()

    generation_mask = '[gMASK]'
    if 'MASK]' not in text:
        text += ' ' + generation_mask
    context_tokens = tokenizer.EncodeAsIds(text)
    context_tokens = [tokenizer.get_command('ENC').Id] + context_tokens
    context_tokens = context_tokens[:seq_length]

    if not text.endswith('[gMASK]'):
        context_tokens = context_tokens + [tokenizer.get_command('eos').Id]
    context_length = len(context_tokens)
    context_length_tensor = torch.cuda.LongTensor([context_length])
    context_length = context_length_tensor[0].item()
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    text = tokenizer.DecodeIds(context_tokens_tensor.tolist())

    start_time = time.time()
    mems = []
    tokens = context_tokens_tensor
    tokens = tokens.view(1, -1).contiguous()
    tokens = tokens.to(device)
    attention_mask = torch.tensor([tokens.size(1)], device=device, dtype=torch.long)
    position_ids = torch.arange(tokens.size(1), device=device, dtype=torch.long)
    block_position_ids = torch.zeros(tokens.size(1), device=device, dtype=torch.long)
    position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    position_ids = position_ids.unsqueeze(0)
    mask_tokens = ['MASK', 'sMASK', 'gMASK']
    mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
    end_tokens = [tokenizer.get_command('eop').Id, eod_token]
    mask_positions = []
    for token in mask_tokens:
        mask_positions += (context_tokens_tensor == token).nonzero(as_tuple=True)[0].tolist()
    mask_positions.sort()
    output_ = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask, return_memory=True)
    mems=output_['hidden_states']
    for mask_position in mask_positions:
        position = mask_position
        tokens, mems = glm_sample_sequence(model, tokenizer, tokens, position,
                                           mems=mems, end_tokens=end_tokens,
                                           out_seq_length=out_seq_length,
                                           temperature=temperature,
                                           top_k=top_k,
                                           repetition_penalty=repetition_penalty)
    output_tokens_list = tokens.view(-1).contiguous()

    decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())


    return decode_tokens

def glm_sample_sequence(model,  tokenizer, context_tokens, context_length,
                        mems=None, end_tokens=None,out_seq_length=512,temperature=0.9,
                        top_k=40, repetition_penalty=1.0):
    tokens = context_tokens.new_full((1, 1), tokenizer.get_command('sop').Id)
    counter = 0
    if mems is None:
        mems = []

    last_beam_num = 1
    # rp = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
    while counter < out_seq_length:
        position_ids = context_tokens.new_ones(last_beam_num, 2, 1)
        position_ids[:, 0] = context_length
        position_ids[:, 1] = counter + 1
        attention_mask = context_tokens.new_zeros([1], device=context_tokens.device, dtype=torch.long)
        last_token = tokens[:, -1:]
        with torch.no_grad():
            output_ = model(input_ids=last_token, position_ids=position_ids, attention_mask=attention_mask,mems=mems, return_memory=True)
        mems = output_['hidden_states']
        next_token_logits = output_['logits']
        next_token_logits = next_token_logits[:, -1]
        # next_token_logits = rp(input_ids=tokens, scores=next_token_logits)
        next_token_logits /= temperature
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = -float('Inf')
        log_probs = F.softmax(next_token_logits, dim=-1)
        prev = torch.multinomial(log_probs, num_samples=1)[0]
        is_end = prev.item() in end_tokens
        if is_end:
            break
        prev = prev.view(1, 1)
        tokens = prev if tokens is None else torch.cat((tokens, prev), dim=1)
        counter += 1
        # output_tokens_list = tokens.view(-1).contiguous()
        #
        # decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
        # print(f"decoder is {decode_tokens}")
    return torch.cat((context_tokens, tokens), dim=1), mems

def bert_random_sample(model, tokenizer, text, input_max_length, out_max_length,
                       top_k, top_p, repetition_penalty, temperature, device):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    token_ids = tokenizer_out["input_ids"]
    token_type_ids = tokenizer_out["token_type_ids"]
    token_ids = torch.tensor(token_ids, device=device, dtype=torch.long).view(1, -1)
    token_type_ids = torch.tensor(token_type_ids, device=device, dtype=torch.long).view(1, -1)

    output_ids = []
    lp = [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty),
          TemperatureLogitsProcessor(temperature=temperature),
          TopKLogitsProcessor(top_k=top_k),
          TopPLogitsProcessor(top_p=top_p),
          ]
    list_processor = ListProcessor(lp)
    with torch.no_grad():
        for step in range(out_max_length):
            scores = model(**{"input_ids":token_ids, "token_type_ids":token_type_ids})["logits"]
            logit_score = torch.log_softmax(scores[:, -1], dim=-1)
            logit_score[:, tokenizer.token_unk_id] = -float('Inf')
            filtered_logits = list_processor(token_ids, logit_score)

            filterd_logits_prob = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(filterd_logits_prob, num_samples=1)
            if tokenizer.token_end_id == next_token.item():
                break
            output_ids.append(next_token.item())
            token_ids = torch.cat((token_ids, next_token.long()), dim=1)
            token_type_ids = torch.cat((token_type_ids, torch.tensor([[1]], device=device, dtype=torch.long)), dim=1)
    return tokenizer.decode(output_ids)

def bert_beamsearch(model, tokenizer, text, input_max_length, out_max_length, beam_size):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length)
    vocab = tokenizer.vocab
    token_ids = tokenizer_out["input_ids"]
    token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = bert_beam_search(model, token_ids, word2ix=vocab, beam_size=beam_size, out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output

def t5_beamsearch(model, tokenizer, text, input_max_length, out_max_length, beam_size):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length, )
    token_ids = tokenizer_out["input_ids"]
    token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = t5_beam_search(model, token_ids, tokenizer, beam_size=beam_size, out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output

def gpt_beamsearch(model, tokenizer, text, input_max_length, out_max_length, beam_size):
    tokenizer_out = tokenizer.encode_plus(text, max_length=input_max_length, )
    token_ids = tokenizer_out["input_ids"]
    token_ids = np.array(token_ids).reshape(1, -1)
    out_puts_ids = gpt_beam_search(model, token_ids, tokenizer, beam_size=beam_size, out_max_length=out_max_length)
    output = tokenizer.decode(out_puts_ids)
    return output

def t5_predict_generate(model, input_ids=None, encoder_hidden_state=None, decoder_input_ids=None):

    with torch.no_grad():
        device = next(model.parameters()).device
        decoder_input_ids = torch.tensor(decoder_input_ids, device=device)
        if input_ids is not None:
            input_ids = torch.tensor(input_ids, device=device)
            if input_ids.ndim == 1:
                input_ids = input_ids.view(1, -1)

            scores = model(**{"input_ids":input_ids, "decoder_input_ids":decoder_input_ids})
        else :
            encoder_hidden_state = torch.from_numpy(encoder_hidden_state).to(device)
            scores = model(**{"encoder_last_hidden_state": encoder_hidden_state, "decoder_input_ids":decoder_input_ids})

    return scores

def t5_beam_search(model, token_ids, tokenizer,
                     beam_size=1, out_max_length=50):
    """
    beam-search操作
    """
    sep_id = tokenizer.token_end_id
    decoder_input_ids = np.array(tokenizer.token_start_id, dtype=np.int).reshape(1, -1)

    output_ids = None
    # 用来保存累计得分
    with torch.no_grad():
        output_scores = np.zeros([1])
        for step in range(out_max_length):
            if step == 0:
                pred_out = t5_predict_generate(model, input_ids=token_ids, decoder_input_ids=decoder_input_ids)
                # 重复beam-size次 输入ids
                encoder_hidden_state = pred_out["encoder_last_hidden_state"].cpu().numpy()
                scores = pred_out["logits"]
                decoder_input_ids = np.tile(decoder_input_ids.reshape([1, -1]), [beam_size, 1])
                encoder_hidden_state = np.tile(encoder_hidden_state, [beam_size, 1, 1])
            else:
                scores = t5_predict_generate(model, encoder_hidden_state=encoder_hidden_state,
                                               decoder_input_ids=decoder_input_ids)["logits"]

            logit_score = F.log_softmax(scores[:, -1], dim=-1).cpu().numpy()

            logit_score = output_scores.reshape([-1, 1]) + logit_score # 累计得分
            ## 取topk的时候我们是展平了然后再去调用topk函数
            # 展平
            logit_score = logit_score.reshape([-1])
            hype_pos = np.argpartition(logit_score, -beam_size, axis=-1)[-beam_size:]
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1]) # 行索引
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int).reshape([-1, 1]) # 列索引

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else :
                output_ids = np.concatenate([output_ids[indice1], indice2], axis=1).astype(np.int)

            decoder_input_ids = np.concatenate([decoder_input_ids, indice2], axis=1)

            end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                # 说明出现终止了～
                return output_ids[best_one][:-1]
            else :
                # 保留未完成部分
                flag = (end_counts < 1)  # 标记未完成序列
                if not flag.all():  # 如果有已完成的
                    decoder_input_ids = decoder_input_ids[flag]
                    output_ids = output_ids[flag]  # 扔掉已完成序列
                    output_scores = output_scores[flag]  # 扔掉已完成序列
                    beam_size = flag.sum()  # topk相应变化
                    encoder_hidden_state = encoder_hidden_state[flag]

        return output_ids[output_scores.argmax()]

def gpt_beam_search(model, token_ids, tokenizer,
                   beam_size=1, out_max_length=50):
    """
    beam-search操作
    """
    sep_id = tokenizer.token_end_id

    output_ids = None
    # 用来保存累计得分
    with torch.no_grad():
        output_scores = np.zeros([1])
        for step in range(out_max_length):
            if step == 0:
                scores = gpt_predict_generate(model, input_ids=token_ids)
                token_ids = np.tile(token_ids, [beam_size, 1])
            else:
                scores = gpt_predict_generate(model, input_ids=token_ids)

            logit_score = F.log_softmax(scores[:, -1], dim=-1).cpu().numpy()
            logit_score = output_scores.reshape([-1, 1]) + logit_score # 累计得分
            ## 取topk的时候我们是展平了然后再去调用topk函数
            # 展平
            logit_score = logit_score.reshape([-1])
            hype_pos = np.argpartition(logit_score, -beam_size, axis=-1)[-beam_size:]
            hype_score = logit_score[hype_pos]
            indice1 = (hype_pos // scores.shape[-1]).reshape([-1]) # 行索引
            indice2 = (hype_pos % scores.shape[-1]).astype(np.int).reshape([-1, 1]) # 列索引

            output_scores = hype_score
            if output_ids is None:
                output_ids = indice2.reshape([beam_size, 1])
            else :
                output_ids = np.concatenate([output_ids[indice1], indice2], axis=1).astype(np.int)

            token_ids = np.concatenate([token_ids, indice2], axis=1)

            end_counts = (output_ids == sep_id).sum(1)  # 统计出现的end标记
            best_one = output_scores.argmax()
            if end_counts[best_one] == 1:
                # 说明出现终止了～
                return output_ids[best_one][:-1]
            else :
                # 保留未完成部分
                flag = (end_counts < 1)  # 标记未完成序列
                if not flag.all():  # 如果有已完成的
                    output_ids = output_ids[flag]  # 扔掉已完成序列
                    output_scores = output_scores[flag]  # 扔掉已完成序列
                    beam_size = flag.sum()  # topk相应变化
                    token_ids = token_ids[flag]

        return output_ids[output_scores.argmax()]
