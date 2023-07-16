import torch 
import torch.nn as nn
from bert_seq2seq.basic_bert import BasicBert

class BertSeq2SeqModel(BasicBert):
    """
    """
    def __init__(self, vocab,
                 model_name="roberta",
                 size="base",
                 **kwargs):
        super(BertSeq2SeqModel, self).__init__(word2ix=vocab, model_name=model_name, size=size)
            
        self.hidden_dim = self.config.hidden_size
        self.vocab_size = len(vocab)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() ## 通过mask 取消 pad 和句子a部分预测的影响
    
    def forward(self, **data):
        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        labels = data.get("labels", None)
        device = input_ids.device

        input_shape = input_ids.shape
        seq_len = input_shape[1]
        ## 构建特殊的mask
        ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=device)
        a_mask = ones.tril()
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(2).float()
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3).float()
        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask
            
        enc_layers, _ = self.bert(input_ids, position_ids=None, token_type_ids=token_type_ids, attention_mask=a_mask,
                                    output_all_encoded_layers=True)
        squence_out = enc_layers[-1] ## 取出来最后一层输出 (batch, seq_len, 768)

        tokens_hidden_state, predictions = self.cls(squence_out)
        result_data = {"logits": predictions, "hidden_states": tokens_hidden_state}

        if labels is not None:

            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_ids[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            result_data["loss"] = loss

        return result_data


