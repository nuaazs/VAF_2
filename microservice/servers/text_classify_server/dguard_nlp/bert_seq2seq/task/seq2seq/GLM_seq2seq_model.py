
from bert_seq2seq.basic_bert import BasicGLM
from bert_seq2seq.model.glm_model import GLMModel
import os

large_ch_config = {
    "num_layers": 24,
    "vocab_size": 50048,
    "hidden_size": 1024,
    "num_attention_heads":16,
    "embedding_dropout_prob":0.1,
    "attention_dropout_prob":0.1,
    "output_dropout_prob":0.1,
    "max_sequence_length":1024,
    "max_memory_length":511,
    "checkpoint_activations": False ,
    "checkpoint_num_layers":1 ,
    "parallel_output": True,
    "relative_encoding": False,
    "block_position_encoding": True,
    "output_predict": True,
    "spell_length": None,
    "spell_func": "lstm",
    "attention_scale":1.0
}
class GLMLargeChConfig:
    def __init__(self):
        config = large_ch_config
        self.num_layers = config["num_layers"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.embedding_dropout_prob = config["embedding_dropout_prob"]
        self.attention_dropout_prob = config["attention_dropout_prob"]
        self.output_dropout_prob = config["output_dropout_prob"]
        self.max_sequence_length = config["max_sequence_length"]
        self.max_memory_length = config["max_memory_length"]
        self.checkpoint_activations = config["checkpoint_activations"]
        self.checkpoint_num_layers = config["checkpoint_num_layers"]
        self.parallel_output = config["parallel_output"]
        self.relative_encoding = config["relative_encoding"]
        self.block_position_encoding = config["block_position_encoding"]
        self.output_predict = config["output_predict"]
        self.spell_length = config["spell_length"]
        self.spell_func = config["spell_func"]
        self.attention_scale = config["attention_scale"]

class GLMSeq2SeqModel(BasicGLM):
    """
    """
    def __init__(self,
                 size="base", **kwargs):
        super(GLMSeq2SeqModel, self).__init__()
        if size == "base":
            pass
            print("不支持GLM base模型")
            os._exit(0)
        elif size == "large":
            config = GLMLargeChConfig()

        else :
            print("不支持的size")
            os._exit(0)

        self.config = config
        self.model = GLMModel(num_layers=config.num_layers,
                              vocab_size=config.vocab_size,
                              hidden_size=config.hidden_size,
                              num_attention_heads=config.num_attention_heads,
                              embedding_dropout_prob=config.embedding_dropout_prob,
                              attention_dropout_prob=config.attention_dropout_prob,
                              output_dropout_prob=config.output_dropout_prob,
                              max_sequence_length=config.max_sequence_length,
                              max_memory_length=config.max_memory_length,
                              checkpoint_activations=config.checkpoint_activations,
                              checkpoint_num_layers=config.checkpoint_num_layers,
                              output_predict=config.output_predict,
                              parallel_output=config.parallel_output,
                              relative_encoding=config.relative_encoding,
                              block_position_encoding=config.block_position_encoding,
                              spell_length=config.spell_length,
                              spell_func=config.spell_func,
                              attention_scale=config.attention_scale)

        self.hidden_dim = self.config.hidden_size
        self.vocab_size = self.config.vocab_size

    def forward(self, **data):
        input_ids = data["input_ids"]
        labels = data.get("labels", None)
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        return_memory = data.get("return_memory", False)
        mems = data.get("mems", None)

        return self.model(input_ids=input_ids, position_ids=position_ids,
                        attention_mask=attention_mask, labels=labels,
                        return_memory=return_memory, mems=mems)

    def load_weights(self, checkpoints_path):
        self.model.load_weights_glm(checkpoints_path)


