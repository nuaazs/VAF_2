import torch
from bert_seq2seq.task.seq2seq.bert_seq2seq_model import BertSeq2SeqModel
import os
from bert_seq2seq.task.embedding.bert_embedding import BertEmbedding
from bert_seq2seq.task.classification.bert_cls_classifier import BertClsClassifier
from bert_seq2seq.task.sequence_labeling.bert_sequence_labeling import BertNERGP, BertNERCRF, BertSequenceLabling
from bert_seq2seq.task.seq2seq.gpt2_seq2seq_model import GPT2
from bert_seq2seq.task.seq2seq.t5_seq2seq_model import T5Model
from bert_seq2seq.task.relationship_extraction.bert_relationship_extraction import BertRelationshipExtraction
# from bert_seq2seq.GLM.model.modeling_glm import GLMModel
# from GLM.model.modeling_glm import GLMModel
from bert_seq2seq.task.seq2seq.GLM_seq2seq_model import GLMSeq2SeqModel

ALL_TASK = {
    "bert_seq2seq": BertSeq2SeqModel,
    "roberta_seq2seq": BertSeq2SeqModel,
    "roberta-large_seq2seq": BertSeq2SeqModel,
    "bert_classification": BertClsClassifier,
    "roberta_classification": BertClsClassifier,
    "roberta-large_classification": BertClsClassifier,
    "bert_sequence_labeling_gp": BertNERGP,
    "roberta_sequence_labeling_gp": BertNERGP,
    "roberta-large_sequence_labeling_gp": BertNERGP,
    "bert_sequence_labeling_crf": BertNERCRF,
    "roberta_sequence_labeling_crf": BertNERCRF,
    "roberta-large_sequence_labeling_crf": BertNERCRF,
    "bert_sequence_labeling": BertSequenceLabling,
    "roberta_sequence_labeling": BertSequenceLabling,
    "roberta-large_sequence_labeling": BertSequenceLabling,
    "bert_embedding": BertEmbedding,
    "roberta_embedding": BertEmbedding,
    "roberta-large_embedding": BertEmbedding,
    "gpt2_seq2seq": GPT2,
    "t5_seq2seq": T5Model,
    "bert_relationship_extraction":BertRelationshipExtraction,
    "roberta_relationship_extraction":BertRelationshipExtraction,
    "nezha_relationship_extraction":BertRelationshipExtraction,
    "glm": GLMSeq2SeqModel,
    "glm_seq2seq": GLMSeq2SeqModel,
    "glm_lm": GLMSeq2SeqModel,

}

def load_model(vocab=None,
               model_name="roberta",
               task_name="seq2seq",
               target_size=0,
               ner_inner_dim=-1,
               size="base"):
    if model_name != "glm":
        assert vocab is not None, "vocab 字典不能为空"
    task_model = ALL_TASK.get(f"{model_name}_{task_name}", None)
    if task_model is None :
        print("no this task")
        os._exit(0)

    return task_model(vocab=vocab,
                      model_name=model_name,
                      size=size,
                      target_size=target_size,
                      ent_type_size=target_size,
                      inner_dim=ner_inner_dim)

