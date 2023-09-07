
import torch
from bert_seq2seq.model.t5_model import T5ForConditionalGeneration, T5Config, T5SmallConfig
from bert_seq2seq.basic_bert import BasicT5
import torch.nn.functional as F

class T5Model(BasicT5):

    def __init__(self, vocab, 
                        model_name="t5",
                        size="base", 
                        **kwargs):
        super().__init__()
        if size == "base":
            config = T5Config(vocab_size=len(vocab))
        elif size == "small":
            config = T5SmallConfig(vocab_size=len(vocab))
        else:
            raise Exception("not support this model type")
        self.model = T5ForConditionalGeneration(config)
        print(f"model is {model_name}")

    def forward(self, **data):
        input_ids = data.get("input_ids", None)
        decoder_input_ids = data["decoder_input_ids"]
        encoder_last_hidden_state = data.get("encoder_last_hidden_state", None)
        if encoder_last_hidden_state is not None:
            encoder_last_hidden_state = [encoder_last_hidden_state]
        labels = data.get("labels", None)
        t5_out = self.model(input_ids=input_ids, encoder_outputs=encoder_last_hidden_state, decoder_input_ids=decoder_input_ids, labels=labels)
        if labels is not None:
            return {"logits": t5_out[1], "loss": t5_out[0], "encoder_last_hidden_state": t5_out[2]}

        return {"logits": t5_out[0], "encoder_last_hidden_state": t5_out[1]}