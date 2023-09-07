
import torch
import torch.nn as nn
import os
import math

def check_params(model, load_params,):
    index = 0
    for name, _ in model.named_parameters():
        # print(name)
        if name not in list(load_params.keys()):
            index += 1

    if index > 12:
        print(f"警告：构建模型与预训练参数相差较大，可能存在参数不匹配的风险。")

def get_model(model_name, word2ix, size="base"):
    if model_name == "roberta":
        from bert_seq2seq.model.roberta_model import BertModel, BertConfig, RobertaLargeConfig, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead, CLS
        if size == "base":
            config = BertConfig(vocab_size=len(word2ix))
        elif size == "large":
            config = RobertaLargeConfig(vocab_size=len(word2ix))
        else:
            config = None
            print("不支持的model size，请输入base 或 large")
            os._exit(0)
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)

    elif model_name == "bert":
        from bert_seq2seq.model.bert_model import BertConfig, BertLargeConfig, BertModel, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead,CLS
        if size == "base":
            config = BertConfig(vocab_size=len(word2ix))
        elif size == "large":
            config = BertLargeConfig(vocab_size=len(word2ix))
        else:
            config = None
            print("不支持的model size，请输入base 或 large")
            os._exit(0)
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)

    elif model_name == "nezha":
        from bert_seq2seq.model.nezha_model import BertConfig, NezhaLargeConfig, BertModel, BertLayerNorm, BertPredictionHeadTransform,BertLMPredictionHead,CLS
        if size == "base":
            config = BertConfig(vocab_size=len(word2ix))
        elif size == "large":
            config = NezhaLargeConfig(vocab_size=len(word2ix))
        else:
            config = None
            print("不支持的model size，请输入base 或 large")
            os._exit(0)
        bert = BertModel(config)
        layer_norm_cond = BertLayerNorm(config.hidden_size, conditional=True)
        CLS = CLS(config)
        
    else:
        raise Exception("model_name_err")

    return config, bert, layer_norm_cond, CLS

class BasicBert(nn.Module):
    def __init__(self, word2ix, model_name="roberta", size="base",):
        super().__init__()
        self.config = ""
        self.word2ix = word2ix
        self.model_name = model_name
        self.config, self.bert, self.layer_norm_cond, self.cls = get_model(model_name, word2ix, size=size)
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path, strict=False):

        checkpoint = torch.load(pretrain_model_path, map_location=self.device)

        checkpoint = {k: v for k, v in checkpoint.items()}

        check_params(self, checkpoint)

        self.load_state_dict(checkpoint, strict=strict)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):

        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_load = {}
        for k, v in checkpoint.items():
            if k[:7] == "module.":
                checkpoint_load[k[7:]] = v 
            else :
                checkpoint_load[k] = v
        self.load_state_dict(checkpoint_load, strict=True)
        print(str(model_path) + " loaded!")

    def forward(self, input_text):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
        if self.model_name == "nezha":
            # 不要保存相对位置编码权重
            checkpoints = {k: v for k, v in self.state_dict().items()
                                        if "relative" not in k}
            torch.save(checkpoints, save_path)
            return
        torch.save(self.state_dict(), save_path)

class BasicGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        
        checkpoint = torch.load(model_path, map_location=device)

        checkpoint_load = {}
        for k, v in checkpoint.items():
            if k[:7] == "module.":
                checkpoint_load[k[7:]] = v 
            else :
                checkpoint_load[k] = v

        self.load_state_dict(checkpoint_load, strict=True)
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)
        
    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)


class BasicT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_load = {}
        for k, v in checkpoint.items():
            if k[:7] == "module.":
                checkpoint_load[k[7:]] = v 
            else :
                checkpoint_load[k] = v
        self.load_state_dict(checkpoint_load, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

class BasicBart(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

class BasicT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=self.device)
        checkpoint = {"model." + k: v for k, v in checkpoint.items()}

        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print("{} loaded!".format(pretrain_model_path))

    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_load = {}
        for k, v in checkpoint.items():
            if k[:7] == "module.":
                checkpoint_load[k[7:]] = v
            else :
                checkpoint_load[k] = v
        self.load_state_dict(checkpoint_load, strict=False)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

class BasicGLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def load_pretrain_params(self, pretrain_model_path):

        if os.getenv("ENV_TYPE") == "deepspeed+mpu":
            model_dir = os.path.dirname(pretrain_model_path)
            from bert_seq2seq.mpu.mp_tools import change_pytorch_model_mp_from_1_to_n, check_pytorch_model_mp_size
            model_parallel_size = int(os.getenv("MODEL_PARALLEL_SIZE"))
            assert model_parallel_size is not None
            if torch.distributed.is_initialized(
            ) and torch.distributed.get_rank() == 0:
                # change the mp_size in rank 0
                print(
                    "preparing the model weights for model parallel size = {:02d}"
                        .format(model_parallel_size))
                if model_parallel_size > 1 and not check_pytorch_model_mp_size(
                        model_dir, model_parallel_size):
                    change_pytorch_model_mp_from_1_to_n("glm",
                                                        model_dir, model_parallel_size)

            from bert_seq2seq import mpu
            torch.distributed.barrier(group=mpu.get_model_parallel_group())

            if model_parallel_size > 1:
                from bert_seq2seq.mpu import get_model_parallel_rank
                model_parallel_rank = get_model_parallel_rank()
                checkpoint_path = os.path.join(
                    model_dir,
                    "pytorch_model_{:02d}.bin".format(model_parallel_rank))
                if os.path.exists(checkpoint_path):
                    self.load_weights(checkpoint_path)
                    print(f"model loaded successed {checkpoint_path}")

        else :
            self.load_weights(pretrain_model_path)


    def load_all_params(self, model_path, device="cuda"):
        checkpoint = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint, strict=True)
        torch.cuda.empty_cache()
        print(str(model_path) + " loaded!")

    def forward(self, x):
        raise NotImplemented

    def set_device(self, device):
        self.device = torch.device(device)
        self.to(device)

    def save_all_params(self, save_path):
        torch.save(self.state_dict(), save_path)

