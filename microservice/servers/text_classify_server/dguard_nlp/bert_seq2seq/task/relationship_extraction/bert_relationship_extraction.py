
import torch.nn as nn
from bert_seq2seq.basic_bert import BasicBert
from bert_seq2seq.layers import GlobalPointer

class BertRelationshipExtraction(BasicBert):
    """
    """
    def __init__(self, vocab,
                 target_size,
                 inner_dim=64,
                 size="base",
                 model_name="roberta",
                 **kwargs):
        super(BertRelationshipExtraction, self).__init__(word2ix=vocab, model_name=model_name, size=size)
        self.entity_output = GlobalPointer(self.config.hidden_size, 2, 
                                            inner_dim, RoPE=True, trill_mask=True)
        self.head_output = GlobalPointer(self.config.hidden_size, target_size, 
                                            inner_dim, RoPE=False, trill_mask=False)
        self.tail_output = GlobalPointer(self.config.hidden_size, target_size, 
                                            inner_dim, RoPE=False, trill_mask=False)
        self.layer_norm_cond = None
        self.cls = None

    def forward(self, **data):
        input_ids = data["input_ids"]
        token_type_ids = data.get("token_type_ids", None)
        head_labels = data.get("head_labels", None)
        tail_labels = data.get("tail_labels", None)
        entity_labels = data.get("entity_labels", None)

        padding_mask = (input_ids > 0).float()

        all_layers, _ = self.bert(input_ids, token_type_ids=token_type_ids,
                                  output_all_encoded_layers=True)
        sequence_out = all_layers[-1]

        entity_output = self.entity_output(sequence_out, padding_mask)
        head_output = self.head_output(sequence_out, padding_mask)
        tail_output = self.tail_output(sequence_out, padding_mask)

        return_data = {"entity_output": entity_output, "head_output": head_output, "tail_output": tail_output}
        if entity_labels is not None:
            loss_entity = self.entity_output.compute_loss_sparse(entity_output, entity_labels, mask_zero=True)
            loss_head = self.head_output.compute_loss_sparse(head_output, head_labels, mask_zero=True)
            loss_tail = self.tail_output.compute_loss_sparse(tail_output, tail_labels, mask_zero=True)

            return_data["loss"] = (loss_entity + loss_head + loss_tail) / 3
        return return_data