import torch.nn as nn
from bert_seq2seq.basic_bert import BasicBert
from bert_seq2seq.layers import GlobalPointer, CRFLayer

class BertSequenceLabling(BasicBert):
    """
    """
    def __init__(self, vocab,
                 target_size,
                 model_name="roberta",
                 size="base",
                 **kwargs):
        super(BertSequenceLabling, self).__init__(word2ix=vocab, model_name=model_name, size=size)
        self.cls = None
        self.layer_norm_cond = None
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, target_size)

    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.target_size)
        labels = labels.view(-1)
        loss = nn.CrossEntropyLoss(reduction="mean")
        return loss(predictions, labels)
    
    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data.get("token_type_ids", None)
        labels = data.get("labels", None)

        all_layers, pooled_out = self.bert(input_ids, token_type_ids=token_type_ids,
                                    output_all_encoded_layers=True)

        sequence_out = all_layers[-1]
        predictions = self.final_dense(sequence_out)

        return_data = {"logits": predictions, }

        if labels is not None:
            ## 计算loss
            loss = self.compute_loss(predictions, labels)
            return_data["loss"] = loss

        return return_data

class BertNERGP(BasicBert):
    """
    """
    def __init__(self, vocab, ent_type_size, inner_dim=64, size="base", model_name="roberta", **kwargs):
        super(BertNERGP, self).__init__(word2ix=vocab, model_name=model_name, size=size)
        self.gp = GlobalPointer(self.config.hidden_size, ent_type_size, inner_dim, RoPE=True)
        self.layer_norm_cond = None
        self.cls = None
    def compute_loss(self, logits, labels):
        pass

    def forward(self, **data):
        input_ids = data["input_ids"]
        token_type_ids = data.get("token_type_ids", None)
        padding_mask = (input_ids > 0).float()
        labels = data.get("labels", None)

        all_layers, _ = self.bert(input_ids, token_type_ids=token_type_ids,
                                  output_all_encoded_layers=True)
        sequence_out = all_layers[-1]

        gp_out = self.gp(sequence_out, padding_mask)
        return_data = {"logits": gp_out, }

        if labels is not None:
            return_data["loss"] = self.gp.compute_loss(gp_out, labels)
        return return_data

class BertNERCRF(BasicBert):
    """
    """
    def __init__(self, vocab, target_size=-1, size="base", model_name="roberta", **kwargs):
        super(BertNERCRF, self).__init__(word2ix=vocab, model_name=model_name, size=size,)
        self.layer_norm_cond = None
        self.cls = None
        self.final_dense = nn.Linear(self.config.hidden_size, target_size)
        self.crf_layer = CRFLayer(target_size)

    def compute_loss(self, logits, labels, target_mask):
        loss = self.crf_layer(logits, labels, target_mask)

        return loss.mean()

    def forward(self, **data):
        input_ids = data["input_ids"]
        token_type_ids = data.get("token_type_ids", None)
        padding_mask = (input_ids > 0).float()
        labels = data.get("labels", None)

        all_layers, _ = self.bert(input_ids, token_type_ids=token_type_ids,
                                  output_all_encoded_layers=True)
        sequence_out = all_layers[-1]

        predictions = self.final_dense(sequence_out)

        return_data = {"logits": predictions, }

        if labels is not None:
            ## 计算loss
            return_data["loss"] = self.compute_loss(predictions, labels, padding_mask)

        return return_data