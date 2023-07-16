from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import AdamW
from tqdm import tqdm
import torch


class BertClassificationModel(nn.Module):
    def __init__(self,num_class=2,hidden_size=768,device="cuda:0"): # bert默认最后输出维度为768
        super(BertClassificationModel, self).__init__()
        self.num_class=num_class
        self.device=device
        self.hidden_size=hidden_size
        model_name = 'bert-base-chinese'
        # 读取分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        # 读取预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        for p in self.bert.parameters(): # 冻结bert参数
            p.requires_grad = False
        self.bert.to(self.device)
    def forward(self, batch_sentences):
        # output: [batch_size, hidden_size] -> [Batch, 768]
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=512,
                                             add_special_tokens=True)
        input_ids=torch.tensor(sentences_tokenizer['input_ids']).to(self.device)
        attention_mask=torch.tensor(sentences_tokenizer['attention_mask']).to(self.device)
        bert_out=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state =bert_out[0].to(self.device) # [batch_size, sequence_length, hidden_size]
        bert_cls_hidden_state=last_hidden_state[:,0,:] # [batch_size, hidden_size]
        # fc_out=self.fc(bert_cls_hidden_state) # [batch_size, num_class]
        return bert_cls_hidden_state