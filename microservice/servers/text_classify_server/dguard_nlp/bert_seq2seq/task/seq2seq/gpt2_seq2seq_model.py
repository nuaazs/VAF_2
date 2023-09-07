import torch
from bert_seq2seq.model.gpt2_model import GPT2LMHeadModel, GPT2Config
from bert_seq2seq.basic_bert import BasicGPT

class GPT2(BasicGPT):
    def __init__(self, vocab,
                 model_name="gpt2",
                 **kwargs
                 ):
        super().__init__()
        self.word2ix = vocab
        if model_name == "gpt2":
            self.config = GPT2Config(len(vocab))
        else :
            self.config = None
        self.model = GPT2LMHeadModel(self.config)
        print(f"model is {model_name}")

    def _make_causal_mask(self, input_ids):
        device = input_ids.device
        bsz, tgt_len = input_ids.shape
        mask = torch.full((tgt_len, tgt_len), 0.0).to(device)
        mask_cond = torch.arange(mask.size(-1)).to(device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1.0)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    def forward(self, **data):
        input_ids = data["input_ids"]
        labels = data.get("labels", None)
        extend_mask = (input_ids > 0).float()

        return_data = {}
        attention_mask = self._make_causal_mask(input_ids)
        extend_mask = extend_mask.unsqueeze(1).unsqueeze(1) * attention_mask
        if labels is not None:
            loss, lm_logits = self.model(input_ids, labels=labels, attention_mask=extend_mask)
            return_data["loss"] = loss

        else :
            lm_logits = self.model(input_ids, attention_mask=attention_mask)
        return_data["logits"] = lm_logits

        return return_data