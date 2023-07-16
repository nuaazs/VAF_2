import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np

def padding(indice, max_length, pad_idx=0):
    """
    pad 函数
    """

    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
       
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

def gpt_collate_fn(batch):

    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])

    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = token_ids_padded.clone()
    target_ids_padded[target_ids_padded == 0] = -100

    return {
        "input_ids": token_ids_padded,
        "labels": target_ids_padded
    }

def t5_seq2seq_collate_fn(batch):

    token_ids_src = [data["input_ids"] for data in batch]
    max_length_src = max([len(t) for t in token_ids_src])
    token_ids_tgt = [data["target_ids"] for data in batch]
    max_length_tgt = max([len(t) for t in token_ids_tgt])

    token_ids_padded = padding(token_ids_src, max_length_src)
    target_ids_padded = padding(token_ids_tgt, max_length_tgt)
    labels_ids = target_ids_padded.clone()
    labels_ids[labels_ids == 0] = -100
    target_ids_padded = target_ids_padded[:, :-1].contiguous()
    labels_ids = labels_ids[:, 1:].contiguous()

    return {
        "input_ids": token_ids_padded,
        "decoder_input_ids": target_ids_padded,
        "labels": labels_ids
    }

def bert_seq2seq_collate_fn(batch):

    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": token_type_ids_padded,
        "labels": target_ids_padded
    }

def bert_cls_collate_fn(batch):

    token_ids = [data["input_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]
    target_ids = [data["labels"] for data in batch]
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": token_type_ids_padded,
        "labels": target_ids
    }

def bert_sequence_label_collate_fn(batch):

    token_ids = [data["input_ids"] for data in batch]
    
    max_length = max([len(t) for t in token_ids])
    target_ids = [data["labels"] for data in batch]
  
    token_ids_padded = padding(token_ids, max_length)
    target_ids_padded = padding(target_ids, max_length)

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": None,
        "labels": target_ids_padded
    }

def bert_sequence_label_gp_collate_fn(batch):

    token_ids = [data["input_ids"] for data in batch]
    labels = [data["labels"] for data in batch]
    token_ids_padded = sequence_padding(token_ids)
    labels_padded = sequence_padding(labels, seq_dims=3)
    token_ids_padded = torch.from_numpy(token_ids_padded)
    labels_padded = torch.from_numpy(labels_padded)

    return {
        "input_ids": token_ids_padded,
        "token_type_ids": None,
        "labels": labels_padded
    }

def bert_gplinker_collate_fn(batch):
    input_ids = [data["input_ids"] for data in batch]
    token_type_ids = [data["token_type_ids"] for data in batch]
    entity_labels = [data["entity_labels"] for data in batch]
    head_labels = [data["head_labels"] for data in batch]
    tail_labels = [data["tail_labels"] for data in batch]

    input_ids = sequence_padding(input_ids)
    token_type_ids = sequence_padding(token_type_ids)
    entity_labels = sequence_padding(entity_labels, seq_dims=2)
    head_labels = sequence_padding(head_labels, seq_dims=2)
    tail_labels = sequence_padding(tail_labels, seq_dims=2)

    input_ids = torch.from_numpy(input_ids).long()
    token_type_ids = torch.from_numpy(token_type_ids).long()
    entity_labels = torch.from_numpy(entity_labels).long()
    head_labels = torch.from_numpy(head_labels).long()
    tail_labels = torch.from_numpy(tail_labels).long()
    
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "entity_labels": entity_labels,
        "head_labels": head_labels,
        "tail_labels": tail_labels
    }

def pad_token(tokens, max_length):
    pad_len = max_length - len(tokens)
    # pad id is 50000
    tokens += [50000] * pad_len
    return tokens

def pad_position_ids(position_ids, max_length):
    pad_len = max_length - len(position_ids[0])
    position_ids[0] += [len(position_ids[0]) + x for x in range(pad_len)]
    position_ids[1] += [1] * pad_len
    return position_ids

def pad_loss_mask(loss_mask, max_length):
    pad_len = max_length - len(loss_mask)
    loss_mask += [0] * pad_len
    return loss_mask

def glm_generation_collate_fn(batch):  #padding process in each batch

    input_ids = [data["input_ids"] for data in batch]
    position_ids = [data["position_ids"] for data in batch]
    attention_mask = [data['attention_mask'] for data in batch]
    loss_mask = [data['loss_mask'] for data in batch]
    labels = [data['labels'] for data in batch]

    max_length = max([len(t) for t in input_ids])
    for i in range(len(input_ids)):
        input_ids[i] = pad_token(input_ids[i], max_length)
        labels[i] = pad_token(labels[i], max_length)
        position_ids[i] = pad_position_ids(position_ids[i],
                                                max_length)
        loss_mask[i] = pad_loss_mask(loss_mask[i], max_length)
    return {
        'input_ids': torch.LongTensor(input_ids),
        'position_ids': torch.LongTensor(position_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'loss_mask': torch.LongTensor(loss_mask),
        'labels': torch.LongTensor(labels),
    }
