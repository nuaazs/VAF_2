import torch.nn as nn
from bert_seq2seq.basic_bert import BasicBert

class BertEmbedding(BasicBert):
    """
    """
    def __init__(self, vocab,
                 model_name="roberta",
                 size="base",
                 **kwargs):
        super(BertEmbedding, self).__init__(word2ix=vocab, model_name=model_name, size=size)
        self.layer_norm_cond = None
        self.cls.predictions.decoder = None
    
    def forward(self, **data):

        input_ids = data["input_ids"]
        token_type_ids = data.get("token_type_ids", None)

        all_layers, _ = self.bert(input_ids, token_type_ids=token_type_ids,
                                    output_all_encoded_layers=True)
        sequence_out = all_layers[-1]
        tokens_hidden_state = self.cls.predictions.transform(sequence_out)

        return_data = {"logits": tokens_hidden_state, }

        return return_data
