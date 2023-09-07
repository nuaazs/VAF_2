# coding = utf-8
# @Time    : 2023-08-02  09:00:45
# @Author  : zhaosheng@nuaa.edu.cn
# @Describe: Load pretrained model by name.

import os
import re
import pathlib
import torch
import torchaudio
import wget
# import sys
# sys.path.append('/home/zhaosheng/bert_fraud_classify')
# from IPython import embed
from dguard_nlp.utils.builder import build
from dguard_nlp.utils.config import yaml_config_loader,Config

#TODO: upload to remote server
model_info ={
    'bert_cos_b1_entropyloss':{
        "config": "/home/zhaosheng/bert_fraud_classify/text_classification/egs/easy_tc_fraud/conf/config.yaml",
        "embedding_model": '/home/zhaosheng/bert_fraud_classify/text_classification/egs/easy_tc_fraud/bert_entropyloss_simple/models/CKPT-EPOCH-17-00/embedding_model.ckpt',
        'classifier': '/home/zhaosheng/bert_fraud_classify/text_classification/egs/easy_tc_fraud/bert_entropyloss_simple/models/CKPT-EPOCH-17-00/classifier.ckpt',
    },
}

ALL_MODELS = list(model_info.keys())

def download_or_load(url):
    if url.startswith('http'):
        if os.path.exists(f"/tmp/dguard_nlp/{os.path.basename(url)}"):
            print(f"Find tmp file {url} in /tmp/dguard_nlp/{os.path.basename(url)}")
            ckpt_path = f"/tmp/dguard_nlp/{os.path.basename(url)}"
            return ckpt_path
        # wget to /tmp/dguard_nlp
        os.makedirs('/tmp/dguard_nlp', exist_ok=True)
        ckpt = wget.download(url, out='/tmp/dguard_nlp')
        ckpt_path = f"/tmp/dguard_nlp/{os.path.basename(url)}"
    else:
        ckpt_path = url
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt {ckpt_path} not found.")
    return ckpt_path

def load_by_name(model_name,device='cuda:0',model_dict=None):
    if model_dict is not None:
        model_info = model_dict
    if model_name in model_info:
        embedding_model_ckpt = download_or_load(model_info[model_name]['embedding_model'])
        classifier_ckpt = download_or_load(model_info[model_name]['classifier'])
        config = yaml_config_loader(download_or_load(model_info[model_name]['config']))
        config = Config(config)
        embedding_model = build('embedding_model', config)
        embedding_model.load_state_dict(torch.load(embedding_model_ckpt, map_location='cpu'), strict=True)
        classifier = build('classifier', config)
        classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'), strict=True)
        embedding_model.eval()
        classifier.eval()

       
        embedding_model.to(device)
        classifier.to(device)
        print(f"Load model {model_name} successfully.")
        return embedding_model,classifier
    else:
        all_models = list(model_info.keys())
        print("All models: ", all_models)
        raise NotImplementedError(f"Model {model_name} not implemented.")

# 推理
def inference(embedding_model,classifier,text,print_result=True):
    with torch.no_grad():
        # change text_list to text set
        feat = embedding_model(text)
        feat = feat.to(next(classifier.parameters()).device)
        outputs = classifier(feat)
        # use softmax to chang
        # tensor([[ 0.9692, -0.9692],
        # [-0.9610,  0.9610]], device='cuda:0', grad_fn=<MmBackward0>)
        # to [0,1],[Confidence,Confidence...]
        outputs = torch.softmax(outputs,dim=1)
        outputs = outputs.detach().cpu().numpy().tolist()
    
    results = []
    for i,output in enumerate(outputs):
        score_0 = output[0]
        score_1 = output[1]
        if score_0 > score_1:
            if print_result:
                print(f'第{i}条样本：{text[i][:20]}...,非涉诈，置信度为{score_0}')
            results.append([0,score_0])
        else:
            if print_result:
                print(f'第{i}条样本：{text[i][:20]}...,涉诈，置信度为{score_1}')
            results.append([1,score_1])
    return results

if __name__ == '__main__':
    embedding_model,classifier = load_by_name('bert_cos_b1_entropyloss')
    embed()
    text = ['我是一个测试样本',
            '喂你好请问是李胡菊李女士吗对啊啊我这里是淘宝商城客服中心的女士打扰一下今天主要是通知您呢您向我们淘宝商城申请的八八会员业务审核已经通过了那三号银联中心会在您名下的账户扣款今年的年费八百八十八元我这己通知您经什么了就是您在四月八号向我们提交申请的八八VIP业务今天呢给您审核通过的然后银联中心上那边呢会在两个小时内扣您的费用八百八十八元清楚吗女士士女士您不要话当时您为什么又申请进过呢您在四月八号向我们提交申请的您没有印象了吗我不知道了不知道是不是小孩弄的我不知道因为您这个情况是不是会在淘宝网站或者是看我们这个淘宝广告申接说申请点到开通的因为我们这个淘宝广告链接呢在各大平台都会有的比如快手、抖音、天猫、爱奇艺都会有的女士打扰一下就是我们是八会员呢因为已经审核通过了如果您不需要的话是要比个本来是一项扣费性的业务嘛如果你不需要的话是用快手抖音',
            ]
    results = inference(embedding_model,classifier,text,print_result=True)
    print(results)

