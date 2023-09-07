import torch.nn as nn
from bert_seq2seq.basic_bert import BasicBert

class BertClsClassifier(BasicBert):
    """
    """
    def __init__(self, vocab,
                 target_size,
                 model_name="roberta",
                 **kwargs):
        super(BertClsClassifier, self).__init__(word2ix=vocab, model_name=model_name)
        self.target_size = target_size
        self.final_dense = nn.Linear(self.config.hidden_size, self.target_size)
        self.cls = None
        self.layer_norm_cond = None

    def compute_loss(self, predictions, labels):
        """
        计算loss
        predictions: (batch_size, 1)
        """
        predictions = predictions.view(-1, self.target_size)
        labels = labels.view(-1)
        loss = nn.CrossEntropyLoss(reduction="mean")
        return loss(predictions, labels)

    def compute_loss_sigmoid(self, predictions, labels):
        predictions = predictions.view(-1)
        labels = labels.view(-1).float()

        loss_sigmoid = nn.BCEWithLogitsLoss()
        return loss_sigmoid(predictions, labels)
    
    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data["token_type_ids"]
        labels = data.get("labels", None)

        all_layers, pooled_out = self.bert(input_ids, token_type_ids=token_type_ids,
                                    output_all_encoded_layers=True)

        predictions = self.final_dense(pooled_out)
        return_data = {"logits": predictions, }
        if labels is not None:
            ## 计算loss
            if self.target_size == 1:
                loss = self.compute_loss_sigmoid(predictions, labels)
            else :
                loss = self.compute_loss(predictions, labels)
            return_data["loss"] = loss

        return return_data