import numpy as np
from typing import List 
import torch
import os 
import math 
from bert_seq2seq.predictor.utils import viterbi_decode, decode_labels, \
                                        bert_beamsearch, t5_random_sample, gpt_random_sample, \
                                        t5_beamsearch, gpt_beamsearch, bert_random_sample, \
                                        gpt_random_sample_from_ids, glm_random_sample
class Predictor:

    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.class_name = type(model).__name__

    def predict_embedding(self, text, maxlen=256, pred_type="cls"):
        device = next(self.model.parameters()).device
        tokenizer_out = self.tokenizer.encode_plus(text, max_length=maxlen, truncation=True)

        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        with torch.no_grad():
            if pred_type == "cls":
                score = self.model(**{"input_ids": input_ids, "token_type_ids": token_type_ids})["logits"].cpu()[0, 0]
            elif pred_type == "mean":
                score = self.model(**{"input_ids": input_ids, "token_type_ids": token_type_ids})["logits"].cpu().mean(dim=1)[0]

        return score

    def predict_cls_classifier(self, text, max_len=512):
        ## text is text or text-pair
        device = next(self.model.parameters()).device
        if type(text) is str:
            tokenizer_out = self.tokenizer.encode_plus(text, max_length=max_len, truncation=True)
        else :
            assert len(text) == 2
            tokenizer_out = self.tokenizer.encode_plus(text[0], text[1], max_length=max_len, truncation=True)

        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        with torch.no_grad():
            score = self.model(**{"input_ids": input_ids, "token_type_ids": token_type_ids})["logits"].cpu()[0]
        return score

    def predict_masklm(self, text, max_len=512):
        device = next(self.model.parameters()).device
        tokenizer_out = self.tokenizer.encode_plus(text, max_length=max_len, truncation=True)

        input_ids = tokenizer_out["input_ids"]
        token_type_ids = tokenizer_out["token_type_ids"]
        input_ids = torch.tensor(input_ids, device=device)
        token_type_ids = torch.tensor(token_type_ids, device=device)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
            token_type_ids = token_type_ids.view(1, -1)
        with torch.no_grad():
            score = self.model(**{"input_ids": input_ids, "token_type_ids": token_type_ids})["logits"].cpu()
        score = score.argmax(dim=-1).numpy()[0]
        return self.tokenizer.decode(score)

    def predict_ner(self, text, target, maxlen=256):
        model = self.model
        model.eval()
        device = next(model.parameters()).device
        tokenizer = self.tokenizer
        tokens = tokenizer.tokenize(text, maxlen=maxlen, add_spatial_tokens=True)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        trans = model.state_dict().get("crf_layer.trans", None)
        if trans is not None:
            ## crf
            trans = trans.cpu()
            with torch.no_grad():
                out = model(**{"input_ids": token_ids})["logits"][0].cpu()
            labels = viterbi_decode(out, trans)
            entities = decode_labels(labels, target)
            return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities if mapping[w[0]] and mapping[w[-1]]]

        elif getattr(model, "gp", None) is not None :
            entities = []
            with torch.no_grad():
                scores = model(**{"input_ids": token_ids})["logits"].cpu().numpy()[0]
            ## global pointer
            scores[:, [0, -1]] -= np.inf
            scores[:, :, [0, -1]] -= np.inf
            for l, start, end in zip(*np.where(scores > 0)):
                if mapping[start] and mapping[end]:
                    entities.append(
                        (mapping[start][0], mapping[end][-1], target[l])
                    )
            return entities

        else :
            with torch.no_grad():
                scores = model(**{"input_ids": token_ids})["logits"].cpu()[0]
            labels = scores.argmax(dim=-1)
            entities = decode_labels(labels, target)
            return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities if mapping[w[0]] and mapping[w[-1]]]

    def predict_generate_beamsearch(self, text, input_max_length=256, out_max_length=100, beam_size=1, ):
        self.model.eval()
        if "bert" in self.class_name.lower():
            assert "seq2seq" in self.class_name.lower(), "this function only support seq2seq task"
            return bert_beamsearch(self.model, self.tokenizer, text, input_max_length=input_max_length,
                                   out_max_length=out_max_length, beam_size=beam_size)
        elif "t5" in self.class_name.lower():
            return t5_beamsearch(self.model, self.tokenizer, text, input_max_length=input_max_length,
                                   out_max_length=out_max_length, beam_size=beam_size)

        elif "gpt" in self.class_name.lower():
            return gpt_beamsearch(self.model, self.tokenizer, text, input_max_length=input_max_length,
                                 out_max_length=out_max_length, beam_size=beam_size)

        else :
            print("暂不支持的解码方式")
            import os
            os._exit(0)

    def predict_generate_randomsample(self, text, input_max_length=256,
                                      out_max_length=200, top_k=30, top_p=1.0,
                                      repetition_penalty=1.0, temperature=1.0, add_sep=False,
                                      ):
        device = next(self.model.parameters()).device
        if "t5" in self.class_name.lower():
            return t5_random_sample(self.model, self.tokenizer, text, input_max_length,
                                    out_max_length, top_k, top_p, repetition_penalty, temperature, device)

        elif "gpt" in self.class_name.lower():
            return gpt_random_sample(self.model, self.tokenizer, text, input_max_length,
                                     out_max_length, top_k, top_p, repetition_penalty, temperature, device, add_sep=add_sep)

        elif "bert" in self.class_name.lower():
            return bert_random_sample(self.model, self.tokenizer, text, input_max_length,
                                     out_max_length, top_k, top_p, repetition_penalty, temperature, device)

        elif "glm" in self.class_name.lower():
            return glm_random_sample(self.model, self.tokenizer, text, input_max_length,
                                     out_max_length, top_k, top_p, repetition_penalty,
                                     temperature, device)

        else:
            print("暂不支持的解码方式")
            import os
            os._exit(0)
    
    def predict_multi_response(self, sentences: List[str], top_k, top_p, 
                                repetition_penalty, temperature, input_max_length=1024,
                               out_max_length=100):
        pass 
        
        length = sum([len(text) for text in sentences])
        if length > input_max_length:
            print(f"对话过长: {length}")
            os._exit(0)
        device = next(self.model.parameters()).device
        input_ids = [self.tokenizer.token_start_id]
        for index, text in enumerate(sentences):
            if (index + 1) % 2 == 1:
                input_ids += self.tokenizer.encode_plus("A:" + text, max_length=input_max_length)["input_ids"][1:]
            else :
                input_ids +=  self.tokenizer.encode_plus("B:" + text, max_length=input_max_length)["input_ids"][1:]

        if "gpt" in self.class_name.lower():
            return gpt_random_sample_from_ids(self.model, self.tokenizer, input_ids,
                                                out_max_length, top_k, top_p, repetition_penalty,
                                                temperature, device)
        
        else :
            print(f"暂不支持的解码方式: {self.class_name}")
            os._exit(0)

        


